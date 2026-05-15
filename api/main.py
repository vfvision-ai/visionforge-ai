"""
FastAPI application for the VisionForge REST API.

Start with:
    uvicorn api.main:app --reload --port 8000

OpenAPI docs:
    http://localhost:8000/docs
    http://localhost:8000/redoc
"""

from __future__ import annotations

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from db.database import init_db
from api.routes import training, models, experiments, health

logger = logging.getLogger(__name__)

# ── lifespan (replaces deprecated on_event) ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initialising database …")
    init_db()
    logger.info("API ready.")
    yield
    logger.info("API shutting down.")

# ── app factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="VisionForge API",
    description=(
        "REST API for VisionForge — Train Vision Models Effortlessly. "
        "Programmatically submit training jobs, track experiments, "
        "and integrate with your CI/CD or MLOps workflows."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
_allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501,http://localhost:3000")
_allowed_origins = [s.strip() for s in _allowed_origins_raw.split(",") if s.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
)

# ── startup / shutdown handled via lifespan above ────────────────────────────


# ── global error handler ──────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check server logs for details."},
    )


# ── routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router,       tags=["Health"])
app.include_router(experiments.router,  prefix="/api/v1/experiments",  tags=["Experiments"])
app.include_router(training.router,     prefix="/api/v1/training",      tags=["Training"])
app.include_router(models.router,       prefix="/api/v1/models",        tags=["Models"])
