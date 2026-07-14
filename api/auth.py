"""
JWT + bcrypt authentication utilities for VisionForge.

Dependencies:
  python-jose[cryptography]>=3.3.0
  bcrypt>=4.0.0
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from jose import JWTError, jwt

from utils import token_store
from utils.settings import settings as _settings

# ── Config ────────────────────────────────────────────────────────────────────
SECRET_KEY: str       = _settings.JWT_SECRET_KEY
ALGORITHM: str        = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int  = _settings.JWT_ACCESS_EXPIRE_MINUTES
REFRESH_TOKEN_EXPIRE_DAYS: int    = _settings.JWT_REFRESH_EXPIRE_DAYS

# ── Password hashing ──────────────────────────────────────────────────────────
def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ── JWT helpers ───────────────────────────────────────────────────────────────
def _create_token(data: dict, expires_delta: timedelta) -> str:
    now = datetime.now(tz=timezone.utc)
    payload = data.copy()
    payload["iat"] = now
    payload["exp"] = now + expires_delta
    payload["jti"] = str(uuid.uuid4())
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
    """Returns the payload dict, or None if invalid / expired / revoked."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "access":
            return None
        if token_store.is_revoked(payload.get("jti", "")):
            return None
        return payload
    except JWTError:
        return None


def decode_refresh_token(token: str) -> Optional[dict]:
    """Returns the payload dict, or None if invalid / expired / revoked."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            return None
        if token_store.is_revoked(payload.get("jti", "")):
            return None
        return payload
    except JWTError:
        return None


def revoke_token(payload: dict) -> None:
    """Add a decoded token's jti to the Redis denylist until its natural expiry."""
    jti = payload.get("jti")
    exp = payload.get("exp")
    if not jti or not exp:
        return
    ttl_seconds = int(exp - datetime.now(tz=timezone.utc).timestamp())
    token_store.revoke(jti, ttl_seconds)
