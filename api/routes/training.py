"""Training job endpoints — submit, status, cancel, list."""

from __future__ import annotations

import io
import csv
import os
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from api.schemas import TrainingSubmit, JobResponse, JobListResponse
from api.dependencies import get_db, get_current_user
from db import crud
from db.models import JobStatus, UserRole, User

# Map framework name → importable module to check availability at request time
_FRAMEWORK_MODULE = {
    "pytorch":     "torch",
    "tensorflow":  "tensorflow",
    "sklearn":     "sklearn",
}


def _check_framework_available(framework: str):
    """Raise HTTPException 400 if the requested framework is not installed."""
    module = _FRAMEWORK_MODULE.get(framework)
    if module is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown framework {framework!r}. Choose pytorch, tensorflow, or sklearn.",
        )
    import importlib
    if importlib.util.find_spec(module) is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Framework {framework!r} is not installed in this environment. "
                f"Install it (e.g. pip install {module}) or choose a different framework."
            ),
        )

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a training job",
)
def submit_training_job(
    payload: TrainingSubmit,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Submit a new asynchronous training job.

    The job is persisted immediately with status=PENDING, then handed off to a
    Celery worker. Poll `/api/v1/training/{job_id}` for status updates.
    """
    _check_framework_available(payload.framework)

    output_dir = os.path.join("/app/experiments", f"job_{payload.dataset_name}_{payload.architecture}")

    job = crud.create_job(
        db=db,
        task_type=payload.task_type,
        framework=payload.framework,
        dataset_name=payload.dataset_name,
        architecture=payload.architecture,
        hyperparams={
            "epochs": payload.epochs,
            "lr": payload.learning_rate,
            "batch_size": payload.batch_size,
            "optimize_hyperparams": payload.optimize_hyperparams,
            "n_trials": payload.n_trials,
            "experiment_name": payload.experiment_name,
            "early_stopping": payload.early_stopping,
            "patience": payload.patience,
            "min_delta": payload.min_delta,
        },
        dataset_config=payload.dataset_config,
        experiment_id=payload.experiment_id,
        output_dir=output_dir,
        user_id=current_user.id,
    )
    db.commit()

    # Dispatch to the appropriate Celery task
    try:
        task = _dispatch_task(job.id, payload, output_dir)
        crud.start_job(db, job.id, celery_task_id=task.id)
        db.commit()
        logger.info("Dispatched Celery task %s for job %s", task.id, job.id)
    except Exception as exc:
        logger.warning("Celery unavailable — job %s queued without worker: %s", job.id, exc)
        # Job stays PENDING; operator should ensure workers are running

    db.refresh(job)
    return job


@router.get("/", response_model=JobListResponse, summary="List training jobs")
def list_jobs(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=1000),
    status_filter: Optional[str] = Query(None, alias="status"),
    framework: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Admins see jobs for their own domain users only; regular users see only their own
    if current_user.role == UserRole.ADMIN:
        domain = current_user.email.split('@')[-1]
        from db.models import User as UserModel
        domain_user_ids = [
            str(u.id) for u in
            db.query(UserModel).filter(UserModel.email.like(f'%@{domain}')).all()
        ]
        jobs = crud.list_jobs(db, skip=skip, limit=limit, status=status_filter, framework=framework, user_ids=domain_user_ids)
        total = crud.count_jobs(db, status=status_filter, framework=framework, user_ids=domain_user_ids)
    else:
        jobs = crud.list_jobs(db, skip=skip, limit=limit, status=status_filter, framework=framework, user_id=str(current_user.id))
        total = crud.count_jobs(db, status=status_filter, framework=framework, user_id=str(current_user.id))
    return JobListResponse(total=total, jobs=jobs)


@router.get("/{job_id}", response_model=JobResponse, summary="Get job status")
def get_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    if current_user.role != UserRole.ADMIN and job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    return job


@router.delete(
    "/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel a pending/running job",
)
def cancel_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    if current_user.role != UserRole.ADMIN and job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")

    if job.status not in (JobStatus.PENDING, JobStatus.RUNNING):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status={job.status!r}.",
        )

    # Revoke Celery task if running
    if job.celery_task_id:
        try:
            from workers.celery_app import celery_app
            celery_app.control.revoke(job.celery_task_id, terminate=True, signal="SIGTERM")
        except Exception as exc:
            logger.warning("Could not revoke Celery task %s: %s", job.celery_task_id, exc)

    crud.cancel_job(db, job_id)
    db.commit()


@router.get("/{job_id}/download", summary="Download the trained model file")
def download_model_file(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    if current_user.role != UserRole.ADMIN and job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    if not job.model_path or not os.path.isfile(job.model_path):
        raise HTTPException(status_code=404, detail="Model file not found on disk.")
    return FileResponse(
        job.model_path,
        filename=os.path.basename(job.model_path),
        media_type="application/octet-stream",
    )


@router.get("/{job_id}/history.csv", summary="Download training history as CSV")
def download_history_csv(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    if current_user.role != UserRole.ADMIN and job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")

    raw = job.training_history

    # Normalise dict-keyed history {'train_accuracy': [...]} → list of per-epoch dicts
    if isinstance(raw, dict):
        metric_lists = {k: v for k, v in raw.items() if isinstance(v, list)}
        if metric_lists:
            n = max(len(v) for v in metric_lists.values())
            history = [
                {"epoch": i + 1, **{k: v[i] for k, v in metric_lists.items() if i < len(v)}}
                for i in range(n)
            ]
        else:
            history = []
    else:
        history = raw or []

    # If no per-epoch list, fall back to flattening job.results
    if not history and job.results:
        history = [{"epoch": 1, **{k: v for k, v in job.results.items() if not isinstance(v, (dict, list))}}]

    if not history:
        raise HTTPException(status_code=404, detail="No training history available.")

    # Collect all column keys
    all_keys: list[str] = ["epoch"]
    for row in history:
        for k in row.keys():
            if k != "epoch" and k not in all_keys:
                all_keys.append(k)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=all_keys, extrasaction="ignore")
    writer.writeheader()
    for i, row in enumerate(history):
        writer.writerow({"epoch": row.get("epoch", i + 1), **row})

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=history_{job_id[:8]}.csv"},
    )


@router.post("/{job_id}/test-samples", summary="Generate test samples and download as ZIP")
def generate_test_samples(
    job_id: str,
    num_samples: int = Query(default=50, ge=1, le=500),
    image_format: str = Query(default="png"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Extract `num_samples` labelled images from the job's dataset and return them
    as a downloadable ZIP containing the images and a labels.csv manifest.
    """
    import io as _io
    import zipfile
    import tempfile
    import types

    job = crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    if current_user.role != UserRole.ADMIN and job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")

    if job.status not in (JobStatus.COMPLETED, "completed"):
        raise HTTPException(
            status_code=400,
            detail="Job must be completed before generating test samples.",
        )

    cfg = job.dataset_config or {}
    source = cfg.get("source", "")
    dataset_name = job.dataset_name or ""
    fmt = image_format.lower() if image_format.lower() in ("png", "jpg") else "png"

    BUILTIN_NAMES = {
        "MNIST", "Fashion-MNIST", "CIFAR-10", "CIFAR-100",
        "VOC2012", "Oxford-IIIT-Pet", "COCO-Detection", "VOC2012-Det",
    }

    info = types.SimpleNamespace(
        task_type=job.task_type,
        num_classes=0,
        num_samples=0,
        class_names=_class_names_for(dataset_name),
        dataset_path=None,
        is_hf_dataset=False,
        hf_dataset_name=None,
        hf_subset=None,
        is_builtin=False,
        builtin_dataset_name=None,
        builtin_tf_name=None,
    )

    if source in ("pytorch", "tensorflow") or dataset_name in BUILTIN_NAMES:
        info.is_builtin = True
        # Normalise to the key _save_builtin_test_samples recognises
        norm = dataset_name.lower().replace("-", "").replace(" ", "").replace("_", "")
        info.builtin_dataset_name = norm
        info.builtin_tf_name = norm
    elif source == "huggingface":
        info.is_hf_dataset = True
        info.hf_dataset_name = dataset_name
        info.hf_subset = cfg.get("hf_subset") or None
    else:
        info.dataset_path = dataset_name

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            from ui.shared import save_test_samples_for_evaluation  # noqa: PLC0415
            save_test_samples_for_evaluation(info, tmpdir, num_samples, fmt)
        except Exception as exc:
            logger.error("Test sample generation failed: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate samples: {exc}",
            )

        test_dir = os.path.join(tmpdir, "test_samples")
        if not os.path.isdir(test_dir):
            raise HTTPException(status_code=500, detail="Sample generation produced no output.")

        buf = _io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in sorted(os.listdir(test_dir)):
                zf.write(os.path.join(test_dir, fname), fname)
        buf.seek(0)
        zip_bytes = buf.getvalue()

    return StreamingResponse(
        iter([zip_bytes]),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=test_samples_{job_id[:8]}.zip"},
    )


def _class_names_for(dataset_name: str) -> list:
    """Return well-known class names for common datasets."""
    _MAP: dict = {
        "MNIST": [str(i) for i in range(10)],
        "Fashion-MNIST": [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ],
        "CIFAR-10": [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ],
        "CIFAR-100": [f"class_{i}" for i in range(100)],
    }
    return _MAP.get(dataset_name, [])


@router.get("/{job_id}/results.json", summary="Download full results as JSON")
def download_results_json(
    job_id: str,
    db: Session = Depends(get_db),
):
    import json as _json
    job = crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    payload = {
        "job_id":           job.id,
        "dataset_name":     job.dataset_name,
        "architecture":     job.architecture,
        "framework":        job.framework,
        "task_type":        job.task_type,
        "hyperparams":      job.hyperparams,
        "status":           job.status,
        "results":          job.results,
        "training_history": job.training_history,
        "created_at":       job.created_at.isoformat() if job.created_at else None,
        "started_at":       job.started_at.isoformat()  if job.started_at  else None,
        "completed_at":     job.completed_at.isoformat() if job.completed_at else None,
    }
    return StreamingResponse(
        iter([_json.dumps(payload, indent=2)]),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=results_{job_id[:8]}.json"},
    )


# ── private helpers ───────────────────────────────────────────────────────────
def _dispatch_task(job_id: str, payload: TrainingSubmit, output_dir: str):
    """Route to the correct Celery task based on framework."""
    from workers.training_tasks import train_pytorch, train_tensorflow, train_sklearn

    _task_map = {
        "pytorch":     train_pytorch,
        "tensorflow":  train_tensorflow,
        "sklearn":     train_sklearn,
    }
    task_fn = _task_map.get(payload.framework)
    if not task_fn:
        raise ValueError(f"Unknown framework: {payload.framework}")

    model_config_dict = {
        "architecture": payload.architecture,
        "framework": payload.framework,
    }

    return task_fn.apply_async(
        args=[
            job_id,
            payload.dataset_config,
            model_config_dict,
            {
                "epochs": payload.epochs,
                "lr": payload.learning_rate,
                "batch_size": payload.batch_size,
                "optimize_hyperparams": payload.optimize_hyperparams,
                "n_trials": payload.n_trials,
                "experiment_name": payload.experiment_name,
                "early_stopping": payload.early_stopping,
                "patience": payload.patience,
                "min_delta": payload.min_delta,
            },
            output_dir,
        ],
        queue="training",
    )
