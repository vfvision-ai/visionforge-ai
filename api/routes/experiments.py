"""Experiment management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.schemas import ExperimentCreate, ExperimentResponse
from api.dependencies import get_db, require_api_key
from db import crud

router = APIRouter()


@router.post("/", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
def create_experiment(
    payload: ExperimentCreate,
    db: Session = Depends(get_db),
    _: str = Depends(require_api_key),
):
    exp = crud.create_experiment(db, name=payload.name, description=payload.description, tags=payload.tags)
    db.commit()
    db.refresh(exp)
    return exp


@router.get("/", response_model=list[ExperimentResponse])
def list_experiments(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return crud.list_experiments(db, skip=skip, limit=limit)


@router.get("/{exp_id}", response_model=ExperimentResponse)
def get_experiment(exp_id: str, db: Session = Depends(get_db)):
    exp = crud.get_experiment(db, exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment {exp_id!r} not found.")
    return exp
