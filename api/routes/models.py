"""Model version endpoints — list, get, promote, delete, download."""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from api.schemas import ModelResponse, ModelListResponse
from api.dependencies import get_db, require_api_key
from db import crud

router = APIRouter()


@router.get("/", response_model=ModelListResponse, summary="List saved model versions")
def list_models(
    skip: int = 0,
    limit: int = 50,
    framework: str | None = None,
    task_type: str | None = None,
    db: Session = Depends(get_db),
):
    models = crud.list_models(db, skip=skip, limit=limit)
    if framework:
        models = [m for m in models if m.framework == framework]
    if task_type:
        models = [m for m in models if m.task_type == task_type]
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


@router.delete(
    "/{model_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a saved model version",
)
def delete_model(
    model_id: str,
    db: Session = Depends(get_db),
    _: str = Depends(require_api_key),
):
    mv = db.query(__import__('db.models', fromlist=['ModelVersion']).ModelVersion).filter_by(id=model_id).first()
    if not mv:
        raise HTTPException(status_code=404, detail=f"Model {model_id!r} not found.")
    db.delete(mv)
    db.commit()


@router.get("/{model_id}/download", summary="Download the model file")
def download_model(
    model_id: str,
    db: Session = Depends(get_db),
):
    mv = db.query(__import__('db.models', fromlist=['ModelVersion']).ModelVersion).filter_by(id=model_id).first()
    if not mv:
        raise HTTPException(status_code=404, detail=f"Model {model_id!r} not found.")
    path = mv.model_path
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Model file not found on disk.")
    return FileResponse(path, filename=os.path.basename(path), media_type="application/octet-stream")
