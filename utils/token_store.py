"""Redis-backed JWT denylist.

Used to revoke individual access/refresh tokens (logout, refresh rotation)
without needing a stateful session store — only revoked jti's are kept,
each expiring automatically at the token's own exp time.
"""
from __future__ import annotations

import logging
from typing import Optional

import redis

from utils.settings import settings

logger = logging.getLogger(__name__)

_REVOKED_PREFIX = "jwt:revoked:"

_client: Optional["redis.Redis"] = None


def _get_client() -> "redis.Redis":
    global _client
    if _client is None:
        _client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
    return _client


def revoke(jti: str, ttl_seconds: int) -> None:
    """Mark a token's jti as revoked until it would have naturally expired."""
    if ttl_seconds <= 0:
        return
    try:
        _get_client().set(f"{_REVOKED_PREFIX}{jti}", "1", ex=ttl_seconds)
    except redis.RedisError:
        logger.warning("Redis unavailable — could not revoke token jti=%s", jti)


def is_revoked(jti: str) -> bool:
    try:
        return bool(_get_client().exists(f"{_REVOKED_PREFIX}{jti}"))
    except redis.RedisError:
        # Fail open: a Redis outage must not take down all authentication.
        logger.warning("Redis unavailable — skipping revocation check (failing open)")
        return False
