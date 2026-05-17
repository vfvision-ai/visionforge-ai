"""Health-check, readiness, and system-info endpoints."""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.schemas import HealthResponse

router = APIRouter()

_VERSION = "1.0.0"


@router.get("/health", response_model=HealthResponse, summary="Liveness check")
async def health():
    """Returns 200 if the API process is alive."""
    db_status = _check_db()
    broker_status = _check_broker()

    overall = "ok" if db_status == "ok" and broker_status in ("ok", "unavailable") else "degraded"

    return HealthResponse(
        status=overall,
        version=_VERSION,
        database=db_status,
        broker=broker_status,
        timestamp=datetime.now(tz=timezone.utc),
    )


@router.get("/ready", summary="Readiness check")
async def ready():
    """Returns 200 when the API is ready to serve traffic (DB connected)."""
    if _check_db() != "ok":
        return JSONResponse(status_code=503, content={"status": "not ready", "reason": "database unavailable"})
    return {"status": "ready"}


@router.get("/system-info", summary="System hardware and library info")
async def system_info():
    """Returns hardware specs and ML library availability for the dashboard status panel."""
    info: dict = {
        "os": platform.system(),
        "python": sys.version.split()[0],
        "platform": platform.machine(),
    }

    # PyTorch
    try:
        import torch
        info["pytorch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
        else:
            info["cuda_available"] = False
    except ImportError:
        info["pytorch"] = None
        info["cuda_available"] = False

    # TensorFlow
    try:
        import tensorflow as tf  # noqa: F401
        info["tensorflow"] = tf.__version__
    except ImportError:
        info["tensorflow"] = None

    # Scikit-learn
    try:
        import sklearn
        info["sklearn"] = sklearn.__version__
    except ImportError:
        info["sklearn"] = None

    # OpenCV
    try:
        import cv2
        info["opencv"] = cv2.__version__
    except ImportError:
        info["opencv"] = None

    # Optuna
    try:
        import optuna
        info["optuna"] = optuna.__version__
    except ImportError:
        info["optuna"] = None

    # RAM
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 1024**3, 1)
        info["ram_used_gb"]  = round(vm.used  / 1024**3, 1)
    except ImportError:
        pass

    return info


# ── private ───────────────────────────────────────────────────────────────────
def _check_db() -> str:
    try:
        from sqlalchemy import text
        from db.database import engine
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return "ok"
    except Exception as exc:
        return f"error: {exc}"


def _check_broker() -> str:
    try:
        from workers.celery_app import celery_app
        celery_app.control.inspect(timeout=1).ping()
        return "ok"
    except Exception:
        return "unavailable"  # Not fatal — broker optional in dev
