"""Authentication endpoints — register, login, refresh, me, logout."""
from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api import auth as auth_utils
from api.dependencies import get_db, get_current_user
from api.schemas import (
    UserRegister, UserLogin, RefreshRequest, TokenResponse, UserResponse, UserListResponse,
)
from db import auth_crud
from db.models import User, UserRole

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Register ──────────────────────────────────────────────────────────────────
@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new account",
)
def register(payload: UserRegister, db: Session = Depends(get_db)):
    if auth_crud.get_user_by_email(db, payload.email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )
    hashed = auth_utils.hash_password(payload.password)
    # First ever user becomes admin automatically
    role = UserRole.ADMIN if auth_crud.count_users(db) == 0 else UserRole.USER
    user = auth_crud.create_user(
        db, email=payload.email, full_name=payload.full_name,
        hashed_password=hashed, role=role,
    )
    db.commit()
    db.refresh(user)
    logger.info("New user registered: %s (role=%s)", user.email, user.role)
    return user


# ── Login ─────────────────────────────────────────────────────────────────────
@router.post("/login", response_model=TokenResponse, summary="Obtain JWT tokens")
def login(payload: UserLogin, db: Session = Depends(get_db)):
    user = auth_crud.get_user_by_email(db, payload.email)
    if not user or not auth_utils.verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled. Contact an administrator.",
        )
    auth_crud.update_last_login(db, user)
    db.commit()
    access_token  = auth_utils.create_access_token(user.id, user.email, user.role.value)
    refresh_token = auth_utils.create_refresh_token(user.id)
    logger.info("User logged in: %s", user.email)
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        user=UserResponse.model_validate(user),
    )


# ── Refresh ───────────────────────────────────────────────────────────────────
@router.post("/refresh", response_model=TokenResponse, summary="Refresh access token")
def refresh(payload: RefreshRequest, db: Session = Depends(get_db)):
    user_id = auth_utils.decode_refresh_token(payload.refresh_token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token.",
        )
    user = auth_crud.get_user_by_id(db, user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found.")
    new_access  = auth_utils.create_access_token(str(user.id), user.email, user.role.value)
    new_refresh = auth_utils.create_refresh_token(str(user.id))
    return TokenResponse(
        access_token=new_access,
        refresh_token=new_refresh,
        token_type="bearer",
        user=UserResponse.model_validate(user),
    )


# ── Me ────────────────────────────────────────────────────────────────────────
@router.get("/me", response_model=UserResponse, summary="Get current user profile")
def me(current_user: User = Depends(get_current_user)):
    return current_user


# ── Admin: list users ─────────────────────────────────────────────────────────
@router.get("/users", response_model=UserListResponse, summary="[Admin] List all users")
def list_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required.")
    users = auth_crud.list_users(db, skip=skip, limit=limit)
    total = auth_crud.count_users(db)
    return UserListResponse(total=total, users=users)
