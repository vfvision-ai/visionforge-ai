"""Model version endpoints — list, get, promote."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.schemas import ModelResponse, ModelListResponse
from api.dependencies import get_db, require_api_key
from db import crud

router = APIRouter()


@router.get("/", response_model=ModelListResponse, summary="List saved model versions")
def list_models(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    models = crud.list_models(db, skip=skip, limit=limit)
    return ModelListResponse(total=len(models), models=models)


@router.post(
    "/{model_id}/promote",
    response_model=ModelResponse,
    summary="Promote a model to production",
)
def promote_model(
    model_id: str,
    db: Session = Depends(get_db),
    _: str = Depends(require_api_key),
):
    """Marks *this* model as the production model and demotes all others."""
    mv = crud.promote_model(db, model_id)
    if not mv:
        raise HTTPException(status_code=404, detail=f"Model {model_id!r} not found.")
    db.commit()
    db.refresh(mv)
    return mv
