"""Health-check and readiness endpoints."""

from __future__ import annotations

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


# ── private ───────────────────────────────────────────────────────────────────
def _check_db() -> str:
    try:
        from db.database import engine
        with engine.connect() as conn:
            conn.execute(engine.dialect.has_table(conn, "training_jobs") and "SELECT 1" or "SELECT 1")
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
