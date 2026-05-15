"""Pydantic schemas for the API request/response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
import uuid


# ── Experiments ───────────────────────────────────────────────────────────────
class ExperimentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(default="", max_length=2000)
    tags: List[str] = Field(default_factory=list)


class ExperimentResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    name: str
    description: str
    tags: List[str]
    created_at: datetime
    updated_at: datetime


# ── Training jobs ─────────────────────────────────────────────────────────────
class TrainingSubmit(BaseModel):
    """Payload to submit a new training job."""

    task_type: str = Field(..., description="classification | detection | segmentation")
    framework: str = Field(..., description="pytorch | tensorflow | sklearn")
    dataset_name: str = Field(..., description="Dataset name or path")
    architecture: str = Field(..., description="Model architecture (e.g. resnet50)")

    # Dataset config — mirrors DatasetInfo fields
    dataset_config: Dict[str, Any] = Field(default_factory=dict)

    # Hyperparameters
    epochs: int             = Field(default=50,    ge=1, le=1000)
    learning_rate: float    = Field(default=0.001, gt=0, lt=1.0)
    batch_size: int         = Field(default=32,    ge=1, le=512)
    optimize_hyperparams: bool = Field(default=False)

    experiment_id: Optional[str] = Field(default=None)

    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, v: str) -> str:
        allowed = {"classification", "detection", "segmentation"}
        if v.lower() not in allowed:
            raise ValueError(f"task_type must be one of {allowed}")
        return v.lower()

    @field_validator("framework")
    @classmethod
    def validate_framework(cls, v: str) -> str:
        allowed = {"pytorch", "tensorflow", "sklearn"}
        if v.lower() not in allowed:
            raise ValueError(f"framework must be one of {allowed}")
        return v.lower()


class JobResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    experiment_id: Optional[str]
    task_type: str
    framework: str
    dataset_name: str
    architecture: str
    status: str
    celery_task_id: Optional[str]
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    output_dir: Optional[str]
    model_path: Optional[str]
    results: Optional[Dict[str, Any]]
    duration_seconds: Optional[float]


class JobListResponse(BaseModel):
    total: int
    jobs: List[JobResponse]


# ── Model versions ────────────────────────────────────────────────────────────
class ModelResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    job_id: str
    name: str
    architecture: str
    framework: str
    task_type: str
    num_classes: Optional[int]
    model_path: str
    onnx_path: Optional[str]
    val_accuracy: Optional[float]
    val_loss: Optional[float]
    test_accuracy: Optional[float]
    extra_metrics: Optional[Dict[str, Any]]
    is_production: bool
    created_at: datetime


class ModelListResponse(BaseModel):
    total: int
    models: List[ModelResponse]


# ── Health ────────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    broker: str
    timestamp: datetime
