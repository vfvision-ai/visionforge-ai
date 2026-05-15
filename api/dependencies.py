"""Shared API dependencies: DB session, API-key auth."""

from __future__ import annotations

import os
import secrets
from typing import Generator

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from db.database import SessionLocal

# ── Database session ──────────────────────────────────────────────────────────
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── API-key authentication ────────────────────────────────────────────────────
_API_KEY_NAME = "X-API-Key"
_api_key_header = APIKeyHeader(name=_API_KEY_NAME, auto_error=False)

# Load from environment.  In dev, set API_KEY=dev (or leave unset to skip auth).
_API_KEY: str | None = os.getenv("API_KEY")
_AUTH_ENABLED: bool  = bool(_API_KEY)


def require_api_key(api_key: str | None = Security(_api_key_header)) -> str:
    """
    FastAPI dependency that validates the X-API-Key header.
    Auth is disabled when API_KEY env var is not set (dev / local mode).
    """
    if not _AUTH_ENABLED:
        return "dev"

    if api_key is None or not secrets.compare_digest(api_key, _API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Set the X-API-Key header.",
        )
    return api_key
