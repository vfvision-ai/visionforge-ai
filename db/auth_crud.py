"""CRUD helpers for User authentication."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from db.models import User, UserRole

# Brute-force protection constants
MAX_FAILED_ATTEMPTS: int = 5          # lock after this many consecutive failures
LOCKOUT_DURATION_MINUTES: int = 15    # how long the account stays locked


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email.lower()).first()


def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


def create_user(
    db: Session,
    email: str,
    full_name: str,
    hashed_password: str,
    role: UserRole = UserRole.USER,
) -> User:
    user = User(
        email=email.lower(),
        full_name=full_name,
        hashed_password=hashed_password,
        role=role,
        is_active=True,
        failed_login_attempts=0,
        locked_until=None,
    )
    db.add(user)
    db.flush()
    return user


def update_last_login(db: Session, user: User) -> None:
    user.last_login_at = datetime.now(timezone.utc)


def is_account_locked(user: User) -> bool:
    """Return True if the user is currently locked out."""
    if user.locked_until is None:
        return False
    locked_until = user.locked_until
    # Normalise to UTC-aware for comparison
    if locked_until.tzinfo is None:
        locked_until = locked_until.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) < locked_until


def lockout_remaining_seconds(user: User) -> int:
    """Seconds remaining in lockout (0 if not locked)."""
    if not is_account_locked(user):
        return 0
    locked_until = user.locked_until
    if locked_until.tzinfo is None:
        locked_until = locked_until.replace(tzinfo=timezone.utc)
    delta = locked_until - datetime.now(timezone.utc)
    return max(0, int(delta.total_seconds()))


def increment_failed_logins(db: Session, user: User) -> None:
    """Increment failed attempts and lock the account if threshold is reached."""
    user.failed_login_attempts = (user.failed_login_attempts or 0) + 1
    if user.failed_login_attempts >= MAX_FAILED_ATTEMPTS:
        user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=LOCKOUT_DURATION_MINUTES)


def reset_failed_logins(db: Session, user: User) -> None:
    """Clear failed attempts and any lockout after a successful login."""
    user.failed_login_attempts = 0
    user.locked_until = None


def list_users(db: Session, skip: int = 0, limit: int = 100, domain: Optional[str] = None):
    q = db.query(User)
    if domain:
        q = q.filter(User.email.like(f'%@{domain}'))
    return q.order_by(User.created_at.desc()).offset(skip).limit(limit).all()


def count_users(db: Session, domain: Optional[str] = None) -> int:
    q = db.query(User)
    if domain:
        q = q.filter(User.email.like(f'%@{domain}'))
    return q.count()


def deactivate_user(db: Session, user_id: str) -> Optional[User]:
    user = get_user_by_id(db, user_id)
    if user:
        user.is_active = False
    return user


def promote_to_admin(db: Session, user_id: str) -> Optional[User]:
    user = get_user_by_id(db, user_id)
    if user:
        user.role = UserRole.ADMIN
    return user
