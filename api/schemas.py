"""Pydantic schemas for the API request/response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, EmailStr, Field, field_validator
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
    n_trials: int           = Field(default=20,    ge=1, le=200)
    experiment_name: str    = Field(default="")

    # Early stopping
    early_stopping: bool    = Field(default=False)
    patience: int           = Field(default=10,    ge=1, le=100)
    min_delta: float        = Field(default=0.001, ge=0.0)

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
    hyperparams: Optional[Dict[str, Any]]
    status: str
    celery_task_id: Optional[str]
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    output_dir: Optional[str]
    model_path: Optional[str]
    results: Optional[Dict[str, Any]]
    training_history: Optional[Any]   # list of per-epoch dicts OR keyed-by-metric dict
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


class ExperimentListResponse(BaseModel):
    total: int
    experiments: List[ExperimentResponse]


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


# ── Auth ──────────────────────────────────────────────────────────────────────
class UserRegister(BaseModel):
    email: EmailStr
    full_name: str = Field(..., min_length=1, max_length=255)
    password: str  = Field(..., min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def validate_email_domain(cls, v: str) -> str:
        """Reject obviously invalid / malformed domains (e.g. no TLD)."""
        parts = v.split("@")
        if len(parts) != 2:
            raise ValueError("Invalid email address.")
        domain = parts[1]
        if "." not in domain:
            raise ValueError("Email domain must contain a valid TLD (e.g. example.com).")
        tld = domain.rsplit(".", 1)[-1]
        if len(tld) < 2:
            raise ValueError("Email TLD must be at least 2 characters.")
        return v.lower()

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        errors = []
        if len(v) < 8:
            errors.append("at least 8 characters")
        if not any(c.isdigit() for c in v):
            errors.append("at least one digit (0-9)")
        if not any(c.islower() for c in v):
            errors.append("at least one lowercase letter")
        if not any(c.isupper() for c in v):
            errors.append("at least one uppercase letter")
        if not any(c in r"!@#$%^&*()_+-=[]{}|;':\",./<>?" for c in v):
            errors.append("at least one special character (!@#$%^&* etc.)")
        if errors:
            raise ValueError("Password must contain: " + ", ".join(errors) + ".")
        return v


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: datetime
    last_login_at: Optional[datetime]

    @field_validator('role', mode='before')
    @classmethod
    def normalise_role(cls, v: object) -> str:
        """Ensure role is always a plain lowercase string regardless of enum type."""
        if hasattr(v, 'value'):
            return str(v.value)
        return str(v).split('.')[-1].lower()


class UserListResponse(BaseModel):
    total: int
    users: List[UserResponse]


class UserPatch(BaseModel):
    """Admin-only: fields that can be updated on a user."""
    is_active: Optional[bool] = None
    role: Optional[str] = None  # "admin" | "user"


class BootstrapAdminRequest(BaseModel):
    email: str
    secret: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"
    user: UserResponse
