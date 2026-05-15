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
from datetime import datetime
from typing import Any, Dict

from celery import Task
from celery.utils.log import get_task_logger

from workers.celery_app import celery_app
from db.database import db_session
from db import crud

logger = get_task_logger(__name__)


class MLTask(Task):
    """Base task class with common error handling and DB update logic."""

    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when the task raises an exception."""
        job_id = args[0] if args else kwargs.get("job_id")
        if job_id:
            with db_session() as db:
                crud.fail_job(db, job_id, error=str(exc))
        logger.error("Task %s failed for job %s: %s", task_id, job_id, exc)

    def on_success(self, retval, task_id, args, kwargs):
        logger.info("Task %s completed for job %s", task_id, args[0] if args else "unknown")


# ── PyTorch training ──────────────────────────────────────────────────────────
@celery_app.task(bind=True, base=MLTask, name="workers.training_tasks.train_pytorch")
def train_pytorch(
    self: Task,
    job_id: str,
    dataset_config: Dict[str, Any],
    model_config: Dict[str, Any],
    hyperparams: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    """
    Asynchronously train a PyTorch model.

    Args:
        job_id:         DB TrainingJob primary key
        dataset_config: Serialised DatasetInfo dict
        model_config:   Serialised ModelConfig dict
        hyperparams:    {"epochs": 50, "lr": 0.001, "batch_size": 32, ...}
        output_dir:     Directory to save checkpoints / final model

    Returns:
        dict with training results (metrics, model_path, etc.)
    """
    _mark_running(job_id, self.request.id)

    logger.info("[job=%s] Starting PyTorch training. hyperparams=%s", job_id, hyperparams)

    try:
        from core.trainer import AutoTrainer
        from utils.config import Config

        os.makedirs(output_dir, exist_ok=True)
        config = Config(
            max_epochs=hyperparams.get("epochs", 50),
            learning_rate=hyperparams.get("lr", 0.001),
            batch_size=hyperparams.get("batch_size", 32),
            output_dir=output_dir,
            optimize_hyperparams=hyperparams.get("optimize_hyperparams", False),
        )

        # Rebuild DatasetInfo from dict
        from core.dataset_analyzer import DatasetInfo
        ds_info = DatasetInfo(**dataset_config)

        trainer = AutoTrainer(config=config)
        results = trainer.train(ds_info, framework="pytorch")

        model_path = results.get("model_path", "")
        _mark_complete(job_id, results, model_path)
        return results

    except Exception as exc:
        logger.error("[job=%s] PyTorch training failed: %s\n%s", job_id, exc, traceback.format_exc())
        with db_session() as db:
            crud.fail_job(db, job_id, error=str(exc))
        raise


# ── TensorFlow training ───────────────────────────────────────────────────────
@celery_app.task(bind=True, base=MLTask, name="workers.training_tasks.train_tensorflow")
def train_tensorflow(
    self: Task,
    job_id: str,
    dataset_config: Dict[str, Any],
    model_config: Dict[str, Any],
    hyperparams: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    """Asynchronously train a TensorFlow/Keras model."""
    _mark_running(job_id, self.request.id)

    logger.info("[job=%s] Starting TensorFlow training. hyperparams=%s", job_id, hyperparams)

    try:
        from core.tensorflow_trainer import TensorFlowTrainer
        from core.dataset_analyzer import DatasetInfo

        os.makedirs(output_dir, exist_ok=True)
        ds_info = DatasetInfo(**dataset_config)

        trainer = TensorFlowTrainer(
            epochs=hyperparams.get("epochs", 50),
            batch_size=hyperparams.get("batch_size", 32),
            learning_rate=hyperparams.get("lr", 0.001),
            output_dir=output_dir,
        )
        results = trainer.train(ds_info, model_config=model_config)

        model_path = results.get("model_path", "")
        _mark_complete(job_id, results, model_path)
        return results

    except Exception as exc:
        logger.error("[job=%s] TensorFlow training failed: %s\n%s", job_id, exc, traceback.format_exc())
        with db_session() as db:
            crud.fail_job(db, job_id, error=str(exc))
        raise


# ── Scikit-learn training ─────────────────────────────────────────────────────
@celery_app.task(bind=True, base=MLTask, name="workers.training_tasks.train_sklearn")
def train_sklearn(
    self: Task,
    job_id: str,
    dataset_config: Dict[str, Any],
    model_config: Dict[str, Any],
    hyperparams: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    """Asynchronously train a Scikit-learn model."""
    _mark_running(job_id, self.request.id)

    logger.info("[job=%s] Starting Sklearn training. hyperparams=%s", job_id, hyperparams)

    try:
        from core.sklearn_trainer import SklearnTrainer
        from core.dataset_analyzer import DatasetInfo

        os.makedirs(output_dir, exist_ok=True)
        ds_info = DatasetInfo(**dataset_config)

        trainer = SklearnTrainer(output_dir=output_dir)
        results = trainer.train(ds_info, model_config=model_config, hyperparams=hyperparams)

        model_path = results.get("model_path", "")
        _mark_complete(job_id, results, model_path)
        return results

    except Exception as exc:
        logger.error("[job=%s] Sklearn training failed: %s\n%s", job_id, exc, traceback.format_exc())
        with db_session() as db:
            crud.fail_job(db, job_id, error=str(exc))
        raise


# ── private helpers ───────────────────────────────────────────────────────────
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
