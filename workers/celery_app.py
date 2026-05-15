"""
Celery application factory for VisionForge.

Broker: Redis  (default: redis://localhost:6379/0)
Backend: Redis (default: redis://localhost:6379/1)

Both broker and backend URLs can be overridden with environment variables:
  CELERY_BROKER_URL
  CELERY_RESULT_BACKEND
"""

import os
from celery import Celery

BROKER_URL  = os.getenv("CELERY_BROKER_URL",  "redis://localhost:6379/0")
BACKEND_URL = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

celery_app = Celery(
    "ml_platform",
    broker=BROKER_URL,
    backend=BACKEND_URL,
    include=["workers.training_tasks"],
)

celery_app.conf.update(
    # Serialisation
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Timezones
    timezone="UTC",
    enable_utc=True,

    # Task behaviour
    task_track_started=True,
    task_acks_late=True,          # Re-queue on worker crash
    worker_prefetch_multiplier=1, # One task at a time per worker (ML is CPU/GPU heavy)

    # Result expiry (7 days)
    result_expires=604800,

    # Soft time limit (8 hours) / hard limit (8.5 hours) for long training jobs
    task_soft_time_limit=28800,
    task_time_limit=30600,

    # Retry policy for failed tasks (don't auto-retry ML jobs — surface errors instead)
    task_max_retries=0,
)
