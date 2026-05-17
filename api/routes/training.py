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
from api.dependencies import get_db, require_api_key
from db import crud
from db.models import JobStatus

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
    _: str = Depends(require_api_key),
):
    """
    Submit a new asynchronous training job.

    The job is persisted immediately with status=PENDING, then handed off to a
    Celery worker. Poll `/api/v1/training/{job_id}` for status updates.
    """
    output_dir = os.path.join("experiments", f"job_{payload.dataset_name}_{payload.architecture}")

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
        },
        dataset_config=payload.dataset_config,
        experiment_id=payload.experiment_id,
        output_dir=output_dir,
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
    limit: int = Query(50, ge=1, le=200),
    status_filter: Optional[str] = Query(None, alias="status"),
    framework: Optional[str] = None,
    db: Session = Depends(get_db),
):
    jobs = crud.list_jobs(db, skip=skip, limit=limit, status=status_filter, framework=framework)
    return JobListResponse(total=len(jobs), jobs=jobs)


@router.get("/{job_id}", response_model=JobResponse, summary="Get job status")
def get_job(
    job_id: str,
    db: Session = Depends(get_db),
):
    job = crud.get_job(db, job_id)
    if not job:
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
    _: str = Depends(require_api_key),
):
    job = crud.get_job(db, job_id)
    if not job:
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
):
    job = crud.get_job(db, job_id)
    if not job:
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
):
    job = crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    history = job.training_history or []
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

    return task_fn.delay(
        job_id,
        payload.dataset_config,
        model_config_dict,
        {
            "epochs": payload.epochs,
            "lr": payload.learning_rate,
            "batch_size": payload.batch_size,
            "optimize_hyperparams": payload.optimize_hyperparams,
        },
        output_dir,
    )
