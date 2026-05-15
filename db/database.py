"""
Database layer for experiment tracking and job management.
Supports SQLite (default / local) and PostgreSQL (production).

Usage:
    from db.database import get_db, engine
    from db.models import TrainingJob, Experiment
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from contextlib import contextmanager

# ── connection URL ─────────────────────────────────────────────────────────────
# Default: SQLite for zero-config local dev / single-node Docker
# Override with DATABASE_URL env var for PostgreSQL in production
_DEFAULT_DB = "sqlite:///./ml_platform.db"
DATABASE_URL: str = os.getenv("DATABASE_URL", _DEFAULT_DB)

# SQLite connection args (not valid for PostgreSQL)
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    # Connection pool settings (ignored by SQLite)
    pool_pre_ping=True,
    pool_recycle=3600,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── FastAPI / general dependency helper ───────────────────────────────────────
def get_db():
    """Yield a DB session; close on exit. Use as FastAPI dependency or context manager."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session():
    """Synchronous context manager for use outside FastAPI."""
    db: Session = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """Create all tables (idempotent)."""
    # Import models so SQLAlchemy sees them
    from db import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
