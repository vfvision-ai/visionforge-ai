"""
Celery training tasks.

Each task:
1. Marks the DB job as RUNNING
2. Runs the appropriate training pipeline (PyTorch / TensorFlow / Sklearn)
3. Saves results to the DB
4. Stores the model artefact path

Tasks are designed to be idempotent and resumable.
"""

from __future__ import annotations

import logging
import os
import traceback
from typing import Any, Dict

from celery import Task
from celery.utils.log import get_task_logger

from workers.celery_app import celery_app
from db.database import db_session
from db import crud

logger = get_task_logger(__name__)
# Known built-in dataset metadata so DatasetInfo has correct num_classes/image_size
_BUILTIN_META: Dict[str, Dict] = {
    "mnist":         {"num_classes": 10,  "image_size": (28, 28),  "channels": 1, "num_samples": 70000},
    "fashion_mnist": {"num_classes": 10,  "image_size": (28, 28),  "channels": 1, "num_samples": 70000},
    "fashion-mnist": {"num_classes": 10,  "image_size": (28, 28),  "channels": 1, "num_samples": 70000},
    "cifar10":       {"num_classes": 10,  "image_size": (32, 32),  "channels": 3, "num_samples": 60000},
    "cifar-10":      {"num_classes": 10,  "image_size": (32, 32),  "channels": 3, "num_samples": 60000},
    "cifar100":      {"num_classes": 100, "image_size": (32, 32),  "channels": 3, "num_samples": 60000},
    "cifar-100":     {"num_classes": 100, "image_size": (32, 32),  "channels": 3, "num_samples": 60000},
}
# Fields that belong to DatasetInfo dataclass â€” anything else is stripped
_DATASET_INFO_FIELDS = {  # noqa: E501
    "task_type", "num_classes", "num_samples", "class_names", "class_distribution",
    "image_size", "image_stats", "has_annotations", "annotation_format",
    "recommended_batch_size", "estimated_training_time", "dataset_path", "channels",
    "is_hf_dataset", "hf_dataset_name", "hf_subset", "hf_features", "hf_description",
    "is_builtin", "builtin_dataset_name", "builtin_tf_name",
}


def _build_config(
    ds_info: Any,
    model_config_dict: Dict[str, Any],
    hyperparams: Dict[str, Any],
    output_dir: str,
) -> Any:
    """Build a Config object the trainers accept, with safe defaults."""
    from pathlib import Path
    from core.model_selector import ModelConfig
    from utils.config import Config

    mc = ModelConfig(
        architecture=model_config_dict.get("architecture", "adaptive_cnn"),
        backbone=model_config_dict.get("backbone", "none"),
        num_parameters=model_config_dict.get("num_parameters", 0),
        input_size=tuple(model_config_dict.get("input_size", [224, 224])),
        pretrained=model_config_dict.get("pretrained", False),
        config_params=model_config_dict.get("config_params", {}),
        estimated_flops=model_config_dict.get("estimated_flops", 0),
        memory_requirements=model_config_dict.get("memory_requirements", 0.5),
        framework=model_config_dict.get("framework", "pytorch"),
    )

    patience = hyperparams.get("patience", 10)
    use_early_stop = hyperparams.get("early_stopping", False)

    return Config(
        dataset_info=ds_info,
        model_config=mc,
        dataset_path=Path(ds_info.dataset_path or output_dir),
        output_dir=Path(output_dir),
        max_epochs=hyperparams.get("epochs", 50),
        learning_rate=hyperparams.get("lr", 0.001),
        batch_size=hyperparams.get("batch_size", 32),
        early_stopping_patience=patience if use_early_stop else 0,
    )


def _safe_dataset_info(dataset_name: str, task_type: str, dataset_config: Dict[str, Any]):
    """Build a DatasetInfo from the dataset name, ignoring training-control keys."""
    from core.dataset_analyzer import DatasetInfo

    # Only pass keys that DatasetInfo actually accepts
    safe = {k: v for k, v in dataset_config.items() if k in _DATASET_INFO_FIELDS}

    # If dataset_config has no real DatasetInfo fields, build from known builtin metadata
    if not safe or "task_type" not in safe:
        meta = _BUILTIN_META.get(dataset_name.lower(), {})
        safe = {
            "task_type":              task_type,
            "num_classes":            meta.get("num_classes", 10),
            "num_samples":            meta.get("num_samples", 0),
            "class_names":            [],
            "class_distribution":     {},
            "image_size":             meta.get("image_size", (224, 224)),
            "image_stats":            {},
            "has_annotations":        False,
            "annotation_format":      None,
            "recommended_batch_size": 32,
            "estimated_training_time":0.0,
            "dataset_path":           dataset_name,
            "is_builtin":             True,
            "builtin_dataset_name":   dataset_name,
        }

    return DatasetInfo(**safe)


class MLTask(Task):
    """Base task class with common error handling and DB update logic."""

    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        job_id = args[0] if args else kwargs.get("job_id")
        if job_id:
            with db_session() as db:
                crud.fail_job(db, job_id, error=str(exc))
        logger.error("Task %s failed for job %s: %s", task_id, job_id, exc)

    def on_success(self, retval, task_id, args, kwargs):
        logger.info("Task %s completed for job %s", task_id, args[0] if args else "unknown")


# â”€â”€ PyTorch training â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@celery_app.task(bind=True, base=MLTask, name="workers.training_tasks.train_pytorch")
def train_pytorch(
    self: Task,
    job_id: str,
    dataset_config: Dict[str, Any],
    model_config: Dict[str, Any],
    hyperparams: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    _mark_running(job_id, self.request.id)
    logger.info("[job=%s] Starting PyTorch training. hyperparams=%s", job_id, hyperparams)

    try:
        from core.trainer import AutoTrainer
        from utils.callbacks import DBProgressCallback

        os.makedirs(output_dir, exist_ok=True)

        dataset_name = model_config.get("dataset_name", dataset_config.get("dataset_path", "MNIST"))
        task_type    = model_config.get("task_type", dataset_config.get("task_type", "classification"))
        ds_info = _safe_dataset_info(dataset_name, task_type, dataset_config)
        config  = _build_config(ds_info, model_config, hyperparams, output_dir)

        trainer = AutoTrainer(config=config)
        trainer.callback_manager.add_callback(DBProgressCallback(job_id))
        result_obj = trainer.train()  # AutoTrainer.train() takes no args

        # TrainingResults is a dataclass — normalise to plain dict
        from dataclasses import asdict
        results = asdict(result_obj)
        results["model_path"] = str(results.get("model_path", ""))
        results["log_path"]   = str(results.get("log_path", ""))

        model_path = results.get("model_path", "")
        _mark_complete(job_id, results, model_path)
        return results

    except Exception as exc:
        logger.error("[job=%s] PyTorch training failed: %s\n%s", job_id, exc, traceback.format_exc())
        with db_session() as db:
            crud.fail_job(db, job_id, error=str(exc))
        raise


# â”€â”€ TensorFlow training â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@celery_app.task(bind=True, base=MLTask, name="workers.training_tasks.train_tensorflow")
def train_tensorflow(
    self: Task,
    job_id: str,
    dataset_config: Dict[str, Any],
    model_config: Dict[str, Any],
    hyperparams: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    _mark_running(job_id, self.request.id)
    logger.info("[job=%s] Starting TensorFlow training. hyperparams=%s", job_id, hyperparams)

    try:
        from core.tensorflow_trainer import TensorFlowTrainer
        from utils.callbacks import DBProgressCallback

        os.makedirs(output_dir, exist_ok=True)

        dataset_name = model_config.get("dataset_name", dataset_config.get("dataset_path", "MNIST"))
        task_type    = model_config.get("task_type", dataset_config.get("task_type", "classification"))
        ds_info = _safe_dataset_info(dataset_name, task_type, dataset_config)
        config  = _build_config(ds_info, model_config, hyperparams, output_dir)

        trainer = TensorFlowTrainer(config=config)
        if hasattr(trainer, 'callback_manager'):
            trainer.callback_manager.add_callback(DBProgressCallback(job_id))

        data_info = trainer.prepare_data(ds_info, batch_size=hyperparams.get("batch_size", 32))
        trainer.build_model(model_config, data_info)
        results = trainer.train(
            epochs=hyperparams.get("epochs", 50),
            model_save_dir=output_dir,
        )

        model_path = results.get("model_path", "")
        _mark_complete(job_id, results, model_path)
        return results

    except Exception as exc:
        logger.error("[job=%s] TensorFlow training failed: %s\n%s", job_id, exc, traceback.format_exc())
        with db_session() as db:
            crud.fail_job(db, job_id, error=str(exc))
        raise


# â”€â”€ Scikit-learn training â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@celery_app.task(bind=True, base=MLTask, name="workers.training_tasks.train_sklearn")
def train_sklearn(
    self: Task,
    job_id: str,
    dataset_config: Dict[str, Any],
    model_config: Dict[str, Any],
    hyperparams: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    _mark_running(job_id, self.request.id)
    logger.info("[job=%s] Starting Sklearn training. hyperparams=%s", job_id, hyperparams)

    try:
        from core.sklearn_trainer import SklearnTrainer

        os.makedirs(output_dir, exist_ok=True)

        dataset_name = model_config.get("dataset_name", dataset_config.get("dataset_path", "MNIST"))
        task_type    = model_config.get("task_type", dataset_config.get("task_type", "classification"))
        ds_info = _safe_dataset_info(dataset_name, task_type, dataset_config)
        config  = _build_config(ds_info, model_config, hyperparams, output_dir)

        trainer = SklearnTrainer(config=config)
        data_info = trainer.prepare_data(ds_info)
        trainer.build_model(model_config, data_info)
        results = trainer.train(model_save_dir=output_dir)

        model_path = results.get("model_path", "")
        _mark_complete(job_id, results, model_path)
        return results

    except Exception as exc:
        logger.error("[job=%s] Sklearn training failed: %s\n%s", job_id, exc, traceback.format_exc())
        with db_session() as db:
            crud.fail_job(db, job_id, error=str(exc))
        raise


# â”€â”€ private helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _mark_running(job_id: str, celery_task_id: str):
    with db_session() as db:
        crud.start_job(db, job_id, celery_task_id)


def _mark_complete(job_id: str, results: Dict, model_path: str):
    with db_session() as db:
        crud.complete_job(
            db,
            job_id,
            results=results,
            model_path=model_path,
            training_history=results.get("training_history", []),
        )

