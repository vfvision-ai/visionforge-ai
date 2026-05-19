"""
JWT + bcrypt authentication utilities for VisionForge.

Dependencies (add to requirements.txt):
  python-jose[cryptography]>=3.3.0
  passlib[bcrypt]>=1.7.4
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from utils.settings import settings as _settings

# ── Config ────────────────────────────────────────────────────────────────────
SECRET_KEY: str       = _settings.JWT_SECRET_KEY
ALGORITHM: str        = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int  = _settings.JWT_ACCESS_EXPIRE_MINUTES
REFRESH_TOKEN_EXPIRE_DAYS: int    = _settings.JWT_REFRESH_EXPIRE_DAYS

# ── Password hashing ──────────────────────────────────────────────────────────
_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    return _pwd_ctx.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_ctx.verify(plain, hashed)


# ── JWT helpers ───────────────────────────────────────────────────────────────
def _create_token(data: dict, expires_delta: timedelta) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(tz=timezone.utc) + expires_delta
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_access_token(user_id: str, email: str, role: str) -> str:
    return _create_token(
        {"sub": user_id, "email": email, "role": role, "type": "access"},
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )


def create_refresh_token(user_id: str) -> str:
    return _create_token(
        {"sub": user_id, "type": "refresh"},
        timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
    )


def decode_access_token(token: str) -> Optional[dict]:
    """Returns the payload dict or None if invalid / expired."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "access":
            return None
        return payload
    except JWTError:
        return None


def decode_refresh_token(token: str) -> Optional[str]:
    """Returns user_id (sub) from a valid refresh token, or None."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            return None
        return payload.get("sub")
    except JWTError:
        return None
