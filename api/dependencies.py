"""Shared API dependencies: DB session, API-key auth, JWT auth."""

from __future__ import annotations

import os
import secrets
from typing import Generator, Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
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
_API_KEY: Optional[str] = os.getenv("API_KEY")
_AUTH_ENABLED: bool  = bool(_API_KEY)


def require_api_key(api_key: Optional[str] = Security(_api_key_header)) -> str:
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


# ── JWT / Bearer authentication ───────────────────────────────────────────────
_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


def get_current_access_token(token: Optional[str] = Depends(_oauth2_scheme)) -> str:
    """FastAPI dependency that returns the raw bearer token string (for logout)."""
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


def get_current_user(
    token: Optional[str] = Depends(_oauth2_scheme),
    db: Session = Depends(get_db),
):
    """
    FastAPI dependency that validates a Bearer JWT and returns the User ORM object.
    Raises HTTP 401 if the token is missing, invalid, or the user no longer exists.
    """
    from api.auth import decode_access_token
    from db.auth_crud import get_user_by_id

    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id: Optional[str] = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_user_by_id(db, user_id)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user
