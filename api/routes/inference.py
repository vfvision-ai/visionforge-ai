"""Inference endpoint — run prediction on an uploaded image using a saved model."""

from __future__ import annotations

import csv
import io
import os
import zipfile
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from api.dependencies import get_db

router = APIRouter()

# Image extensions we'll try to infer inside a ZIP
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}


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
    """
    mv, path, framework, arch, num_classes, stored_class_names = _resolve_model(model_id, db)
    top_k = max(1, min(top_k, num_classes))
    img_data = await file.read()
    img = _open_image(img_data)
    return _run_one(path, framework, arch, num_classes, top_k, stored_class_names, img)


# Extensions to try when searching for a model file on disk
_MODEL_EXTS = (".pt", ".pth", ".keras", ".h5", ".joblib", ".pkl", ".model")
# Root search directories (inside the container)
_SEARCH_ROOTS = [
    "/app/experiments",
    "/app/models",
    "experiments",
    "models",
]


def _find_model_file(stored_path: Optional[str], job) -> Optional[str]:
    """
    Try a cascade of path strategies to locate the model file on disk.
    Returns the first resolved path that exists, or None.
    """
    candidates: List[str] = []

    # 1. Stored path as-is
    if stored_path:
        candidates.append(stored_path)
        # 2. Strip leading '/app' in case we're running outside Docker
        stripped = stored_path.lstrip("/")
        candidates.append(stripped)
        candidates.append(os.path.join("/app", stripped))
        # 3. Basename in job output_dir
        if job and job.output_dir:
            candidates.append(os.path.join(job.output_dir, os.path.basename(stored_path)))

    # 4. Basename in every search root (shallow)
    basename = os.path.basename(stored_path) if stored_path else ""
    if basename:
        for root in _SEARCH_ROOTS:
            candidates.append(os.path.join(root, basename))

    # 5. job output_dir itself — any model file
    if job and job.output_dir and os.path.isdir(job.output_dir):
        for fname in sorted(os.listdir(job.output_dir)):
            if any(fname.endswith(ext) for ext in _MODEL_EXTS):
                candidates.append(os.path.join(job.output_dir, fname))

    # 6. Recursive search in roots (expensive — only if nothing found yet)
    matched = next((c for c in candidates if c and os.path.isfile(c)), None)
    if matched:
        return matched

    if basename:
        for root in _SEARCH_ROOTS:
            if not os.path.isdir(root):
                continue
            for dirpath, _, files in os.walk(root):
                if basename in files:
                    return os.path.join(dirpath, basename)
                # Also try same base with different extension
                base_no_ext = os.path.splitext(basename)[0]
                for f in files:
                    if os.path.splitext(f)[0] == base_no_ext and any(f.endswith(e) for e in _MODEL_EXTS):
                        return os.path.join(dirpath, f)

    return None


def _resolve_model(model_id: str, db: Session):
    """Return (mv, path, framework, arch, num_classes, class_names) or raise 404."""
    from db.models import ModelVersion as MV, TrainingJob as TJ

    mv = db.query(MV).filter_by(id=model_id).first()
    if not mv:
        raise HTTPException(status_code=404, detail=f"Model {model_id!r} not found.")

    job = db.query(TJ).filter_by(id=mv.job_id).first() if mv.job_id else None

    path = _find_model_file(mv.model_path, job)
    if not path:
        searched = [mv.model_path or "(empty)"] + [
            os.path.join(r, os.path.basename(mv.model_path or ""))
            for r in _SEARCH_ROOTS if mv.model_path
        ]
        raise HTTPException(
            status_code=404,
            detail=(
                f"Model file not found on disk. Stored path: {mv.model_path!r}. "
                f"Also searched: {searched}. "
                "If you just completed training, try the 'Repair Models' button on the Models page."
            ),
        )

    stored_class_names: List[str] = []
    if job and job.dataset_config:
        stored_class_names = job.dataset_config.get("class_names", []) or []

    num_classes = mv.num_classes or len(stored_class_names) or 10
    return mv, path, str(mv.framework).lower(), mv.architecture or "resnet18", num_classes, stored_class_names


def _run_one(path: str, framework: str, arch: str, num_classes: int, top_k: int, class_names: List[str], img) -> Dict[str, Any]:
    if framework == "pytorch":
        return _infer_pytorch(path, img, arch, num_classes, top_k, class_names)
    elif framework == "tensorflow":
        return _infer_tensorflow(path, img, num_classes, top_k, class_names)
    elif framework == "sklearn":
        return _infer_sklearn(path, img, num_classes, top_k, class_names)
    raise HTTPException(status_code=400, detail=f"Unsupported framework: {framework!r}")


@router.post(
    "/zip",
    summary="Run batch inference on a ZIP of images (test-samples format)",
    response_class=StreamingResponse,
)
async def run_inference_zip(
    file: UploadFile = File(..., description=(
        "ZIP file matching the test-samples download format: "
        "images + optional labels.csv with columns image_name,label,label_name"
    )),
    model_id: str = Form(..., description="ModelVersion ID"),
    top_k: int    = Form(5,  description="Top-K predictions per image"),
    db: Session   = Depends(get_db),
):
    """
    Upload the same ZIP that **Download Test Samples** produces and receive a CSV
    with columns:

    ``image_name, true_label, true_label_name, predicted_class, confidence_%, correct``

    If ``labels.csv`` is absent from the ZIP the ``true_label`` / ``true_label_name``
    columns will be empty and ``correct`` will be blank.
    """
    mv, path, framework, arch, num_classes, class_names = _resolve_model(model_id, db)
    effective_top_k = max(1, min(top_k, num_classes))

    zip_data = await file.read()
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_data))
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid ZIP archive.") from exc

    names_in_zip = zf.namelist()

    # Parse labels.csv if present (any depth)
    labels_map: Dict[str, Dict[str, str]] = {}
    csv_entry = next((n for n in names_in_zip if os.path.basename(n) == "labels.csv"), None)
    if csv_entry:
        raw = zf.read(csv_entry).decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(raw))
        for row in reader:
            img_name = row.get("image_name", "").strip()
            if img_name:
                labels_map[img_name] = {
                    "label":      row.get("label", ""),
                    "label_name": row.get("label_name", ""),
                }

    # Collect image entries
    img_entries = [
        n for n in names_in_zip
        if os.path.splitext(n.lower())[1] in _IMG_EXTS and not os.path.basename(n).startswith(".")
    ]

    if not img_entries:
        raise HTTPException(status_code=400, detail="ZIP contains no supported image files (jpg/png/bmp/webp).")

    # Run inference per image, build result rows
    rows: List[List[str]] = []
    for entry in img_entries:
        base = os.path.basename(entry)
        img_data = zf.read(entry)
        try:
            from PIL import Image as PILImage
            img = PILImage.open(io.BytesIO(img_data)).convert("RGB")
        except Exception:
            rows.append([base, "", "", "ERROR: cannot decode image", "", ""])
            continue

        try:
            result = _run_one(path, framework, arch, num_classes, effective_top_k, class_names, img)
        except HTTPException as exc:
            rows.append([base, "", "", f"ERROR: {exc.detail}", "", ""])
            continue

        preds: List[Dict] = result.get("predictions") or []
        top = preds[0] if preds else {}
        pred_class = top.get("class_name", result.get("top_class", ""))
        confidence = top.get("confidence", result.get("top_confidence", 0.0))
        conf_pct   = f"{float(confidence) * 100:.1f}"

        meta = labels_map.get(base, {})
        true_label      = meta.get("label", "")
        true_label_name = meta.get("label_name", "")
        c_idx = top.get("class_index")
        correct = ""
        if true_label_name:
            # Direct name comparison
            match = str(true_label_name).strip().lower() == str(pred_class).strip().lower()
            # Fallback: compare class index against numeric label (handles e.g. "7" vs "Class 7")
            if not match and c_idx is not None and true_label:
                match = str(c_idx) == str(true_label).strip()
            correct = "1" if match else "0"
        elif true_label:
            # Only numeric index comparison available
            correct = "1" if c_idx is not None and str(c_idx) == str(true_label).strip() else "0"

        rows.append([base, true_label, true_label_name, str(pred_class), conf_pct, correct])

    # Build output CSV in-memory
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["image_name", "true_label", "true_label_name", "predicted_class", "confidence_%", "correct"])
    writer.writerows(rows)
    csv_bytes = out.getvalue().encode("utf-8")

    # Compute summary stats
    labelled = [r for r in rows if r[5] in ("0", "1")]
    correct_n = sum(1 for r in labelled if r[5] == "1")
    # Do NOT include '%' in the header value — HTTP proxies (e.g. Next.js rewrites)
    # may strip or corrupt headers containing literal '%' characters.
    acc_str = f"{correct_n / len(labelled) * 100:.1f}" if labelled else "N/A"

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=inference_results.csv",
            "X-Total-Images":  str(len(rows)),
            "X-Correct":       str(correct_n),
            "X-Accuracy":      acc_str,
        },
    )
