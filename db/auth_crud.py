"""CRUD helpers for User authentication."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from db.models import User, UserRole


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
    )
    db.add(user)
    db.flush()
    return user


def update_last_login(db: Session, user: User) -> None:
    user.last_login_at = datetime.now(timezone.utc)


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
