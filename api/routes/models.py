"""Model version endpoints — list, get, promote, delete, download, export."""

from __future__ import annotations

import os
import tempfile
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.schemas import ModelResponse, ModelListResponse
from api.dependencies import get_db, require_api_key
from db import crud

router = APIRouter()


@router.get("/", response_model=ModelListResponse, summary="List saved model versions")
def list_models(
    skip: int = 0,
    limit: int = 50,
    framework: Optional[str] = None,
    task_type: Optional[str] = None,
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


# ---------------------------------------------------------------------------
# Export schema
# ---------------------------------------------------------------------------

class ExportRequest(BaseModel):
    format: str = "onnx"      # "onnx" | "torchscript"
    input_size: int = 224


@router.post("/{model_id}/export", summary="Export a PyTorch model to ONNX or TorchScript")
def export_model(
    model_id: str,
    req: ExportRequest,
    db: Session = Depends(get_db),
):
    """Loads the saved checkpoint, rebuilds the model, and streams back the
    exported file.  Only supports PyTorch (*.pt / *.pth) checkpoints."""
    from db.models import ModelVersion as MV  # local import to keep module lightweight

    mv = db.query(MV).filter_by(id=model_id).first()
    if not mv:
        raise HTTPException(status_code=404, detail=f"Model {model_id!r} not found.")
    if mv.framework != "pytorch":
        raise HTTPException(status_code=400, detail="Export only supported for PyTorch models.")

    path = mv.model_path
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Model file not found on disk.")

    fmt = req.format.lower()
    if fmt not in ("onnx", "torchscript"):
        raise HTTPException(status_code=400, detail="format must be 'onnx' or 'torchscript'.")

    try:
        import torch
        import torch.nn as nn
        import torchvision.models as tvm

        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(path, map_location=device)

        arch = mv.architecture or "resnet18"
        num_classes = mv.num_classes or 10

        if isinstance(checkpoint, dict):
            arch = checkpoint.get("architecture", checkpoint.get("model_name", arch))
            num_classes = checkpoint.get("num_classes", num_classes)
            state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        else:
            state = None

        constructor = getattr(tvm, arch.lower(), tvm.resnet18)
        model = constructor(weights=None)
        # Replace final classifier layer
        if hasattr(model, "fc"):
            model.fc = nn.Linear(model.fc.in_features, int(num_classes))
        elif hasattr(model, "classifier"):
            last = model.classifier[-1]
            model.classifier[-1] = nn.Linear(last.in_features, int(num_classes))

        if state:
            try:
                model.load_state_dict(state, strict=False)
            except Exception:
                pass  # allow partial load

        model.eval().to(device)
        dummy = torch.randn(1, 3, req.input_size, req.input_size).to(device)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".onnx" if fmt == "onnx" else ".pt")
        tmp.close()

        if fmt == "onnx":
            torch.onnx.export(
                model, dummy, tmp.name,
                input_names=["input"], output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                opset_version=17,
            )
        else:  # torchscript
            with torch.no_grad():
                scripted = torch.jit.trace(model, dummy)
            scripted.save(tmp.name)

        base = f"{mv.name.replace(' ', '_')}_{fmt}"
        ext  = "onnx" if fmt == "onnx" else "pt"
        return FileResponse(tmp.name, filename=f"{base}.{ext}", media_type="application/octet-stream")

    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Required library not installed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
