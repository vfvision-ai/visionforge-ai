"""
SQLAlchemy ORM models for VisionForge.

Tables:
  - experiments   : top-level containers grouping related training runs
  - training_jobs : individual asynchronous training runs
  - model_versions: saved model artefacts linked to a completed job
"""

import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Boolean,
    DateTime, Text, ForeignKey, JSON, Enum as SAEnum,
)
from sqlalchemy.orm import relationship
import enum

from db.database import Base


def _uuid():
    return str(uuid.uuid4())


# ── Enums ─────────────────────────────────────────────────────────────────────
class JobStatus(str, enum.Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    CANCELLED = "cancelled"


class Framework(str, enum.Enum):
    PYTORCH      = "pytorch"
    TENSORFLOW   = "tensorflow"
    SKLEARN      = "sklearn"


class TaskType(str, enum.Enum):
    CLASSIFICATION = "classification"
    DETECTION      = "detection"
    SEGMENTATION   = "segmentation"


# ── Tables ────────────────────────────────────────────────────────────────────
class Experiment(Base):
    """Top-level container for a series of related training runs."""
    __tablename__ = "experiments"

    id          = Column(String, primary_key=True, default=_uuid)
    name        = Column(String(255), nullable=False)
    description = Column(Text, default="")
    tags        = Column(JSON, default=list)          # ["cv", "cifar-10"]
    created_at  = Column(DateTime, default=datetime.utcnow)
    updated_at  = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    jobs = relationship("TrainingJob", back_populates="experiment", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Experiment id={self.id!r} name={self.name!r}>"


class TrainingJob(Base):
    """One training run, created when the user submits a training request."""
    __tablename__ = "training_jobs"

    id              = Column(String, primary_key=True, default=_uuid)
    experiment_id   = Column(String, ForeignKey("experiments.id"), nullable=True)

    # What is being trained
    task_type       = Column(SAEnum(TaskType),  nullable=False)
    framework       = Column(SAEnum(Framework), nullable=False)
    dataset_name    = Column(String(255), nullable=False)
    architecture    = Column(String(255), nullable=False)

    # Hyperparameters (stored as JSON for flexibility)
    hyperparams     = Column(JSON, default=dict)   # {"lr": 0.001, "epochs": 50, ...}
    dataset_config  = Column(JSON, default=dict)   # dataset_info serialised

    # Status tracking
    status          = Column(SAEnum(JobStatus), default=JobStatus.PENDING, nullable=False)
    celery_task_id  = Column(String(255), nullable=True)   # Celery async result ID
    error_message   = Column(Text, nullable=True)

    # Timing
    created_at      = Column(DateTime, default=datetime.utcnow)
    started_at      = Column(DateTime, nullable=True)
    completed_at    = Column(DateTime, nullable=True)

    # Artefact paths
    output_dir      = Column(String(512), nullable=True)   # experiments/<job_id>/
    model_path      = Column(String(512), nullable=True)   # path to .pt / .keras

    # Training results (populated on completion)
    results         = Column(JSON, default=dict)   # {"accuracy": 0.97, "loss": 0.05, ...}
    training_history= Column(JSON, default=list)   # per-epoch metrics list

    experiment = relationship("Experiment", back_populates="jobs")
    model      = relationship("ModelVersion", back_populates="job", uselist=False)

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def __repr__(self):
        return f"<TrainingJob id={self.id!r} status={self.status!r}>"


class ModelVersion(Base):
    """A saved, deployable model artefact produced by a training job."""
    __tablename__ = "model_versions"

    id              = Column(String, primary_key=True, default=_uuid)
    job_id          = Column(String, ForeignKey("training_jobs.id"), nullable=False, unique=True)

    # Model metadata
    name            = Column(String(255), nullable=False)
    architecture    = Column(String(255), nullable=False)
    framework       = Column(SAEnum(Framework), nullable=False)
    task_type       = Column(SAEnum(TaskType), nullable=False)
    num_classes     = Column(Integer, nullable=True)

    # Paths
    model_path      = Column(String(512), nullable=False)   # relative path inside output_dir
    onnx_path       = Column(String(512), nullable=True)    # optional ONNX export

    # Performance summary
    val_accuracy    = Column(Float, nullable=True)
    val_loss        = Column(Float, nullable=True)
    test_accuracy   = Column(Float, nullable=True)
    extra_metrics   = Column(JSON, default=dict)

    # Lineage
    created_at      = Column(DateTime, default=datetime.utcnow)
    is_production   = Column(Boolean, default=False)   # promoted to production?

    job = relationship("TrainingJob", back_populates="model")

    def __repr__(self):
        return f"<ModelVersion id={self.id!r} arch={self.architecture!r} acc={self.val_accuracy}>"
