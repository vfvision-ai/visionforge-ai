"""Inference endpoint — run prediction on an uploaded image using a saved model."""

from __future__ import annotations

import io
import os
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from api.dependencies import get_db

router = APIRouter()


# ── helpers ────────────────────────────────────────────────────────────────────

def _class_names(num_classes: int, stored: Optional[List[str]] = None) -> List[str]:
    if stored and len(stored) == num_classes:
        return stored
    return [f"Class {i}" for i in range(num_classes)]


def _open_image(data: bytes):
    try:
        from PIL import Image
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}") from exc


def _infer_pytorch(path: str, img, arch: str, num_classes: int, top_k: int, class_names: List[str]):
    try:
        import torch
        import torch.nn.functional as F
        import torchvision.models as tvm
        import torch.nn as nn
        from torchvision import transforms

        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(path, map_location=device)

        if isinstance(checkpoint, dict):
            arch = checkpoint.get("architecture", checkpoint.get("model_name", arch))
            state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
            num_classes = checkpoint.get("num_classes", num_classes)
        else:
            # checkpoint is already a model
            model = checkpoint
            state = None

        if isinstance(checkpoint, dict):
            constructor = getattr(tvm, arch.lower(), None) or tvm.resnet18
            model = constructor(weights=None)
            if hasattr(model, "fc"):
                model.fc = nn.Linear(model.fc.in_features, int(num_classes))
            elif hasattr(model, "classifier"):
                last = model.classifier[-1]
                model.classifier[-1] = nn.Linear(last.in_features, int(num_classes))
            if state and isinstance(state, dict):
                model.load_state_dict(state, strict=False)

        model.eval().to(device)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs  = F.softmax(logits, dim=1).cpu().numpy()[0]

        k = min(top_k, len(probs))
        import numpy as np
        top_idx = np.argsort(probs)[::-1][:k]
        names = _class_names(len(probs), class_names)
        predictions = [
            {"class_name": names[i] if i < len(names) else f"Class {i}",
             "confidence": float(probs[i]), "class_index": int(i)}
            for i in top_idx
        ]
        return {
            "predictions": predictions,
            "top_class": predictions[0]["class_name"],
            "top_confidence": predictions[0]["confidence"],
            "framework": "pytorch",
        }
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"PyTorch not installed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PyTorch inference failed: {exc}") from exc


def _infer_tensorflow(path: str, img, num_classes: int, top_k: int, class_names: List[str]):
    try:
        import numpy as np
        import tensorflow as tf

        model = tf.keras.models.load_model(path)

        img_arr = img.resize((224, 224))
        import numpy as np
        arr = np.array(img_arr).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)

        preds = model.predict(arr, verbose=0)[0]
        k = min(top_k, len(preds))
        top_idx = np.argsort(preds)[::-1][:k]
        names = _class_names(len(preds), class_names)
        predictions = [
            {"class_name": names[i] if i < len(names) else f"Class {i}",
             "confidence": float(preds[i]), "class_index": int(i)}
            for i in top_idx
        ]
        return {
            "predictions": predictions,
            "top_class": predictions[0]["class_name"],
            "top_confidence": predictions[0]["confidence"],
            "framework": "tensorflow",
        }
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"TensorFlow not installed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TensorFlow inference failed: {exc}") from exc


def _infer_sklearn(path: str, img, num_classes: int, top_k: int, class_names: List[str]):
    try:
        import pickle
        import numpy as np

        with open(path, "rb") as f:
            model = pickle.load(f)

        # Flatten greyscale 28×28 as sklearn models are usually trained on MNIST-style features
        img_small = img.convert("L").resize((28, 28))
        arr = np.array(img_small).flatten().reshape(1, -1).astype("float32") / 255.0

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(arr)[0]
            k = min(top_k, len(probs))
            top_idx = np.argsort(probs)[::-1][:k]
            names = _class_names(len(probs), class_names)
            predictions = [
                {"class_name": names[i] if i < len(names) else f"Class {i}",
                 "confidence": float(probs[i]), "class_index": int(i)}
                for i in top_idx
            ]
        else:
            pred = int(model.predict(arr)[0])
            names = _class_names(num_classes, class_names)
            label = names[pred] if pred < len(names) else f"Class {pred}"
            predictions = [{"class_name": label, "confidence": 1.0, "class_index": pred}]

        return {
            "predictions": predictions,
            "top_class": predictions[0]["class_name"],
            "top_confidence": predictions[0]["confidence"],
            "framework": "sklearn",
        }
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"scikit-learn not installed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Sklearn inference failed: {exc}") from exc


# ── endpoint ───────────────────────────────────────────────────────────────────

@router.post("/", summary="Run inference on an uploaded image")
async def run_inference(
    file: UploadFile = File(..., description="Image file (jpg/png/webp/gif)"),
    model_id: str    = Form(..., description="ModelVersion ID from /api/v1/models/"),
    top_k: int       = Form(5,  description="Number of top predictions to return"),
    db: Session      = Depends(get_db),
):
    """
    Upload an image and receive the top-k class predictions from the chosen model.

    Supported frameworks: **pytorch**, **tensorflow**, **sklearn**.
    The model file must exist on disk at the path stored in the ModelVersion record.
    """
    from db.models import ModelVersion as MV
    from db.models import TrainingJob as TJ

    mv = db.query(MV).filter_by(id=model_id).first()
    if not mv:
        raise HTTPException(status_code=404, detail=f"Model {model_id!r} not found.")

    path = mv.model_path
    if not path or not os.path.isfile(path):
        # Try resolving relative paths against the job's output_dir
        if mv.job_id:
            job = db.query(TJ).filter_by(id=mv.job_id).first()
            if job and job.output_dir and path:
                candidate = os.path.join(job.output_dir, os.path.basename(path))
                if os.path.isfile(candidate):
                    path = candidate
        if not path or not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="Model file not found on disk.")

    # Read class names from dataset_config in the parent job
    stored_class_names: List[str] = []
    if mv.job_id:
        job = db.query(TJ).filter_by(id=mv.job_id).first()
        if job and job.dataset_config:
            stored_class_names = job.dataset_config.get("class_names", []) or []

    num_classes = mv.num_classes or len(stored_class_names) or 10
    top_k = max(1, min(top_k, num_classes))

    img_data = await file.read()
    img = _open_image(img_data)

    framework = str(mv.framework).lower()
    arch = mv.architecture or "resnet18"

    if framework == "pytorch":
        return _infer_pytorch(path, img, arch, num_classes, top_k, stored_class_names)
    elif framework == "tensorflow":
        return _infer_tensorflow(path, img, num_classes, top_k, stored_class_names)
    elif framework == "sklearn":
        return _infer_sklearn(path, img, num_classes, top_k, stored_class_names)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported framework: {framework!r}")
