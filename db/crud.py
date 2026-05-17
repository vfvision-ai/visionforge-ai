"""CRUD helpers for the VisionForge database."""

from __future__ import annotations
from datetime import datetime
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session

from db.models import Experiment, TrainingJob, ModelVersion, JobStatus


# ── Experiments ───────────────────────────────────────────────────────────────
def create_experiment(db: Session, name: str, description: str = "", tags: Optional[List] = None) -> Experiment:
    exp = Experiment(name=name, description=description, tags=tags or [])
    db.add(exp)
    db.flush()
    return exp


def get_experiment(db: Session, exp_id: str) -> Optional[Experiment]:
    return db.query(Experiment).filter(Experiment.id == exp_id).first()


def list_experiments(db: Session, skip: int = 0, limit: int = 100) -> List[Experiment]:
    return db.query(Experiment).order_by(Experiment.created_at.desc()).offset(skip).limit(limit).all()


# ── Training jobs ─────────────────────────────────────────────────────────────
def create_job(
    db: Session,
    task_type: str,
    framework: str,
    dataset_name: str,
    architecture: str,
    hyperparams: Dict[str, Any],
    dataset_config: Dict[str, Any],
    experiment_id: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> TrainingJob:
    job = TrainingJob(
        task_type=task_type,
        framework=framework,
        dataset_name=dataset_name,
        architecture=architecture,
        hyperparams=hyperparams,
        dataset_config=dataset_config,
        experiment_id=experiment_id,
        output_dir=output_dir,
    )
    db.add(job)
    db.flush()
    return job


def get_job(db: Session, job_id: str) -> Optional[TrainingJob]:
    return db.query(TrainingJob).filter(TrainingJob.id == job_id).first()


def list_jobs(
    db: Session,
    skip: int = 0,
    limit: int = 50,
    status: Optional[str] = None,
    framework: Optional[str] = None,
) -> List[TrainingJob]:
    q = db.query(TrainingJob)
    if status:
        q = q.filter(TrainingJob.status == status)
    if framework:
        q = q.filter(TrainingJob.framework == framework)
    return q.order_by(TrainingJob.created_at.desc()).offset(skip).limit(limit).all()


def start_job(db: Session, job_id: str, celery_task_id: str) -> Optional[TrainingJob]:
    job = get_job(db, job_id)
    if job:
        job.status = JobStatus.RUNNING
        job.celery_task_id = celery_task_id
        job.started_at = datetime.utcnow()
    return job


def complete_job(
    db: Session,
    job_id: str,
    results: Dict[str, Any],
    model_path: str,
    training_history: Optional[List] = None,
) -> Optional[TrainingJob]:
    job = get_job(db, job_id)
    if job:
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        job.results = results
        job.model_path = model_path
        job.training_history = training_history or []
    return job


def fail_job(db: Session, job_id: str, error: str) -> Optional[TrainingJob]:
    job = get_job(db, job_id)
    if job:
        job.status = JobStatus.FAILED
        job.completed_at = datetime.utcnow()
        job.error_message = error
    return job


def cancel_job(db: Session, job_id: str) -> Optional[TrainingJob]:
    job = get_job(db, job_id)
    if job and job.status in (JobStatus.PENDING, JobStatus.RUNNING):
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.utcnow()
    return job


# ── Model versions ────────────────────────────────────────────────────────────
def create_model_version(
    db: Session,
    job_id: str,
    name: str,
    architecture: str,
    framework: str,
    task_type: str,
    model_path: str,
    num_classes: Optional[int] = None,
    val_accuracy: Optional[float] = None,
    val_loss: Optional[float] = None,
    test_accuracy: Optional[float] = None,
    extra_metrics: Optional[Dict] = None,
) -> ModelVersion:
    mv = ModelVersion(
        job_id=job_id,
        name=name,
        architecture=architecture,
        framework=framework,
        task_type=task_type,
        model_path=model_path,
        num_classes=num_classes,
        val_accuracy=val_accuracy,
        val_loss=val_loss,
        test_accuracy=test_accuracy,
        extra_metrics=extra_metrics or {},
    )
    db.add(mv)
    db.flush()
    return mv


def list_models(db: Session, skip: int = 0, limit: int = 50) -> List[ModelVersion]:
    return db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).offset(skip).limit(limit).all()


def promote_model(db: Session, model_id: str) -> Optional[ModelVersion]:
    """Promote one model to production (demotes all others)."""
    db.query(ModelVersion).update({ModelVersion.is_production: False})
    mv = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if mv:
        mv.is_production = True
    return mv
