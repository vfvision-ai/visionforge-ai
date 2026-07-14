"""Redis-backed fixed-window rate limiter for sensitive endpoints (login, register).

Account lockout (db/auth_crud.py) already throttles repeated attempts against a
single *known* email. This complements it with a per-IP limit so an attacker
can't sidestep lockout by spraying many different email addresses.
"""
from __future__ import annotations

import logging

import redis
from fastapi import HTTPException, Request, status

from utils.settings import settings

logger = logging.getLogger(__name__)

_client: "redis.Redis | None" = None


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


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce(request: Request, scope: str, limit: int, window_seconds: int) -> None:
    """Raise 429 once more than `limit` requests hit `scope` from this IP within
    `window_seconds`. Fails open (does not block) if Redis is unreachable —
    an outage of the rate limiter must not take down login/register entirely.
    """
    key = f"ratelimit:{scope}:{_client_ip(request)}"
    try:
        client = _get_client()
        count = client.incr(key)
        if count == 1:
            client.expire(key, window_seconds)
        if count > limit:
            ttl = client.ttl(key)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Too many requests. Try again in {max(ttl, 1)}s.",
            )
    except redis.RedisError:
        logger.warning("Redis unavailable — skipping rate limit for scope=%s (failing open)", scope)
