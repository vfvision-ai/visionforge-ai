"""Experiment management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.schemas import ExperimentCreate, ExperimentResponse, ExperimentListResponse
from api.dependencies import get_db, get_current_user
from db import crud
from db.models import User, UserRole

router = APIRouter()


@router.post("/", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
def create_experiment(
    payload: ExperimentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    exp = crud.create_experiment(db, name=payload.name, description=payload.description, tags=payload.tags, user_id=current_user.id)
    db.commit()
    db.refresh(exp)
    return exp


@router.get("/", response_model=ExperimentListResponse)
def list_experiments(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role == UserRole.ADMIN:
        from db.models import User as UserModel
        domain = current_user.email.split('@')[-1]
        domain_user_ids = [
            str(u.id) for u in
            db.query(UserModel).filter(UserModel.email.like(f'%@{domain}')).all()
        ]
        experiments = crud.list_experiments(db, skip=skip, limit=limit, user_ids=domain_user_ids)
        total = crud.count_experiments(db, user_ids=domain_user_ids)
    else:
        experiments = crud.list_experiments(db, skip=skip, limit=limit, user_id=str(current_user.id))
        total = crud.count_experiments(db, user_id=str(current_user.id))
    return ExperimentListResponse(total=total, experiments=experiments)


@router.get("/{exp_id}", response_model=ExperimentResponse)
def get_experiment(
    exp_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    exp = crud.get_experiment(db, exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment {exp_id!r} not found.")
    if current_user.role != UserRole.ADMIN and exp.user_id != current_user.id:
        raise HTTPException(status_code=404, detail=f"Experiment {exp_id!r} not found.")
    return exp
