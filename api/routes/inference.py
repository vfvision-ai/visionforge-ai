"""Inference endpoint — run prediction on an uploaded image using a saved model."""

from __future__ import annotations

import base64
import csv
import io
import os
import zipfile
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from api.dependencies import get_db

router = APIRouter()

# Image extensions we'll try to infer inside a ZIP
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}

# ── Per-dataset default preprocessing ─────────────────────────────────────────
# mean / std = None  →  Standard (0-1): only ToTensor, no Normalize()
# This matches the Celery API trainer default (normalization_type='Standard (0-1)')
_BUILTIN_PREPROCESS: Dict[str, Dict] = {
    "mnist":        {"size": (28, 28),   "channels": 1, "mean": None, "std": None},
    "fashionmnist": {"size": (28, 28),   "channels": 1, "mean": None, "std": None},
    "cifar10":      {"size": (32, 32),   "channels": 3, "mean": None, "std": None},
    "cifar100":     {"size": (32, 32),   "channels": 3, "mean": None, "std": None},
    "stl10":        {"size": (96, 96),   "channels": 3, "mean": None, "std": None},
    "imagenet":     {"size": (224, 224), "channels": 3,
                     "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
}

# Architecture families whose standard torchvision builds use 224×224 RGB input.
# NOTE: normalization is still None by default because the Celery trainer applies
# 'Standard (0-1)' unless the user explicitly chose a different norm type.
_LARGE_ARCH_PREFIXES = (
    "resnet", "vgg", "efficientnet", "mobilenet", "densenet",
    "inception", "convnext", "regnet", "swin", "vit", "alexnet",
)


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


# ── Smart preprocessing configuration ─────────────────────────────────────────

def _detect_in_channels(state: Dict) -> int:
    """
    Infer the model's expected input channels by inspecting the first
    convolutional weight in the state dict.  Shape: (out_ch, in_ch, kH, kW).
    Falls back to 3 (RGB) if nothing conclusive is found.
    """
    try:
        import torch
        for key, val in state.items():
            if not isinstance(val, torch.Tensor):
                continue
            if val.ndim == 4 and "weight" in key.lower():
                k_lower = key.lower()
                if any(t in k_lower for t in ("conv", "features", "patch_embed", "stem")):
                    return int(val.shape[1])
        # fallback: first 4-D weight found
        for key, val in state.items():
            if isinstance(val, torch.Tensor) and val.ndim == 4 and "weight" in key.lower():
                return int(val.shape[1])
    except Exception:
        pass
    return 3


def _detect_num_classes(state: Dict) -> Optional[int]:
    """Read the output class count from the last linear layer in state_dict."""
    last = None
    try:
        import torch
        for key, val in state.items():
            if not isinstance(val, torch.Tensor):
                continue
            if val.ndim == 2 and "weight" in key.lower():
                k_lower = key.lower()
                if any(t in k_lower for t in ("fc", "classifier", "head", "linear")):
                    last = int(val.shape[0])
    except Exception:
        pass
    return last


def _normalise_ds_key(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def _get_preprocess_config(job, mv, checkpoint: Optional[Dict] = None) -> Dict:
    """
    4-tier cascade to determine the correct preprocessing parameters.

    Returns:
        {
          "size":     (H, W)          — resize target,
          "channels": int             — 1 = grayscale, 3 = RGB,
          "mean":     list | None     — Normalize mean  (None → skip Normalize),
          "std":      list | None     — Normalize std   (None → skip Normalize),
        }

    Priority:
      1. Checkpoint metadata      – saved by trainer at training time (most reliable)
      2. state_dict first-conv    – ground-truth input channels, combined with
                                    job.dataset_config image_size
      3. job.dataset_config       – image_size + channels stored when job was created
      4. Builtin dataset lookup   – known dimensions for MNIST, CIFAR-10, etc.
      5. Architecture heuristic   – large torchvision archs → 224×224
      6. Safe fallback            – 224×224, RGB, no normalise
    """
    state = checkpoint.get("model_state_dict") if checkpoint else None
    detected_ch = _detect_in_channels(state) if state else None

    # ── P1: checkpoint metadata ───────────────────────────────────────────────
    if checkpoint:
        ck_size = checkpoint.get("image_size")
        ck_ch   = checkpoint.get("channels")
        ck_mean = checkpoint.get("normalization_mean")
        ck_std  = checkpoint.get("normalization_std")
        if ck_size and ck_ch:
            return {
                "size":     tuple(int(x) for x in ck_size),
                "channels": int(ck_ch),
                "mean":     ck_mean,
                "std":      ck_std,
            }

    # use state-dict detected channels as the best source for "channels"
    cfg = (job.dataset_config or {}) if job else {}

    # ── P2 + P3: dataset_config image_size + detected channels ───────────────
    ds_size = cfg.get("image_size")
    ds_ch   = cfg.get("channels")
    if ds_size and len(ds_size) == 2:
        h, w = int(ds_size[0]), int(ds_size[1])
        ch   = detected_ch or (int(ds_ch) if ds_ch else 3)
        return {"size": (h, w), "channels": ch, "mean": None, "std": None}

    # ── P4: builtin dataset name lookup ──────────────────────────────────────
    raw_name = (cfg.get("builtin_dataset_name") or
                (job.dataset_name if job else "") or "")
    norm_key = _normalise_ds_key(raw_name)
    for canon, pre in _BUILTIN_PREPROCESS.items():
        if norm_key == _normalise_ds_key(canon):
            result = dict(pre)
            if detected_ch:
                result["channels"] = detected_ch
            return result

    # ── P5: architecture heuristic ────────────────────────────────────────────
    arch = (mv.architecture or "").lower() if mv else ""
    ch   = detected_ch or 3
    if any(arch.startswith(p) for p in _LARGE_ARCH_PREFIXES):
        return {"size": (224, 224), "channels": ch, "mean": None, "std": None}

    # ── P6: safe fallback ─────────────────────────────────────────────────────
    return {"size": (224, 224), "channels": ch or 3, "mean": None, "std": None}


# ── Per-framework preprocessing helpers ───────────────────────────────────────

def _pt_transform(img, pre: Dict):
    """Build PIL image → (1, C, H, W) float tensor using the resolved pre config."""
    from torchvision import transforms as T
    steps = [T.Resize(pre["size"])]
    if pre["channels"] == 1:
        steps.append(T.Grayscale(num_output_channels=1))
    # ToTensor: converts PIL [0,255] to float [0.0, 1.0] and transposes dimensions
    steps.append(T.ToTensor())
    if pre.get("mean") is not None and pre.get("std") is not None:
        steps.append(T.Normalize(mean=pre["mean"], std=pre["std"]))
    return T.Compose(steps)(img).unsqueeze(0)


def _build_pt_model(arch: str, num_classes: int, in_channels: int):
    """Reconstruct a PyTorch model skeleton matching the training architecture."""
    import torch.nn as nn
    from utils.model_factory import _CLASSIFICATION_ARCH_MAP, ModelFactory

    arch_key = arch.lower().replace("-", "_").replace(" ", "_")
    builder  = _CLASSIFICATION_ARCH_MAP.get(arch_key)

    if builder:
        model = builder(None)  # weights=None — we load from checkpoint below
    else:
        # Adaptive / simple CNN — rebuild using the same factory logic
        mf    = ModelFactory.__new__(ModelFactory)   # skip __init__
        model = mf._build_adaptive_cnn(num_classes, in_channels)
        return model  # caller will load state_dict

    # Patch first conv for non-RGB inputs
    if in_channels != 3:
        if hasattr(model, "conv1") and isinstance(model.conv1, nn.Conv2d):
            old = model.conv1
            model.conv1 = nn.Conv2d(in_channels, old.out_channels, old.kernel_size,
                                    old.stride, old.padding, bias=old.bias is not None)
        elif hasattr(model, "features"):
            f = model.features
            first = f[0] if isinstance(f[0], nn.Conv2d) else (
                f[0][0] if hasattr(f[0], "__getitem__") and isinstance(f[0][0], nn.Conv2d) else None)
            if first is not None:
                new_c = nn.Conv2d(in_channels, first.out_channels, first.kernel_size,
                                  first.stride, first.padding, bias=first.bias is not None)
                if isinstance(f[0], nn.Conv2d):
                    f[0] = new_c
                else:
                    f[0][0] = new_c

    # Patch classifier head
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        head = model.classifier
        if isinstance(head, nn.Linear):
            model.classifier = nn.Linear(head.in_features, num_classes)
        elif isinstance(head, nn.Sequential):
            for i in reversed(range(len(head))):
                if isinstance(head[i], nn.Linear):
                    head[i] = nn.Linear(head[i].in_features, num_classes)
                    break
    elif hasattr(model, "heads") and hasattr(model.heads, "head"):
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    return model


# ── Framework inference functions ──────────────────────────────────────────────

def _infer_pytorch(path: str, img, arch: str, num_classes: int,
                   top_k: int, class_names: List[str], pre: Dict) -> Dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as F
        import numpy as np

        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict):
            # Extract fields from checkpoint (may be enriched by newer trainer)
            arch        = checkpoint.get("architecture", checkpoint.get("model_name", arch))
            num_classes = int(checkpoint.get("num_classes", num_classes) or num_classes)
            state       = checkpoint.get("model_state_dict",
                          checkpoint.get("state_dict", None))
            # Retrieve class names from checkpoint if stored
            ck_names = checkpoint.get("class_names") or []
            if ck_names and len(ck_names) == num_classes:
                class_names = ck_names

            # Detect actual input channels from state dict weights
            if state and isinstance(state, dict):
                in_ch = _detect_in_channels(state)
                # Also let state_dict fix num_classes if checkpoint value was wrong
                det_nc = _detect_num_classes(state)
                if det_nc:
                    num_classes = det_nc
            else:
                in_ch = pre["channels"]

            # Rebuild the model skeleton with correct in_channels / num_classes
            model = _build_pt_model(arch, num_classes, in_ch)
            if state and isinstance(state, dict):
                model.load_state_dict(state, strict=False)
        else:
            # Checkpoint is already a full model object
            model  = checkpoint
            in_ch  = pre["channels"]

        model.eval().to(device)

        # Build a preprocessing transform tuned to this model's actual input
        effective_pre = dict(pre)
        effective_pre["channels"] = in_ch  # use state-dict-detected channels
        tensor = _pt_transform(img, effective_pre).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs  = F.softmax(logits, dim=1).cpu().numpy()[0]

        k       = min(top_k, len(probs))
        top_idx = np.argsort(probs)[::-1][:k]
        names   = _class_names(len(probs), class_names)
        predictions = [
            {"class_name": names[i] if i < len(names) else f"Class {i}",
             "confidence": float(probs[i]), "class_index": int(i)}
            for i in top_idx
        ]
        return {
            "predictions":    predictions,
            "top_class":      predictions[0]["class_name"],
            "top_confidence": predictions[0]["confidence"],
            "framework":      "pytorch",
        }
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"PyTorch not installed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PyTorch inference failed: {exc}") from exc


def _infer_tensorflow(path: str, img, num_classes: int,
                      top_k: int, class_names: List[str]) -> Dict[str, Any]:
    try:
        import numpy as np
        import tensorflow as tf

        model = tf.keras.models.load_model(path)

        # Read the model's exact expected input shape — ground truth from the graph
        inp_shape = model.input_shape  # e.g. (None, 28, 28, 1) or (None, 224, 224, 3)
        _, h, w, c = inp_shape[0], inp_shape[1], inp_shape[2], inp_shape[3]
        h, w, c = int(h), int(w), int(c)

        pil_img = img.convert("L") if c == 1 else img.convert("RGB")
        arr = np.array(pil_img.resize((w, h))).astype("float32") / 255.0
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]   # (H, W) → (H, W, 1)
        arr = np.expand_dims(arr, axis=0)  # (1, H, W, C)

        preds = model.predict(arr, verbose=0)[0]
        k        = min(top_k, len(preds))
        top_idx  = np.argsort(preds)[::-1][:k]
        names    = _class_names(len(preds), class_names)
        predictions = [
            {"class_name": names[i] if i < len(names) else f"Class {i}",
             "confidence": float(preds[i]), "class_index": int(i)}
            for i in top_idx
        ]
        return {
            "predictions":    predictions,
            "top_class":      predictions[0]["class_name"],
            "top_confidence": predictions[0]["confidence"],
            "framework":      "tensorflow",
        }
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"TensorFlow not installed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TensorFlow inference failed: {exc}") from exc


def _infer_sklearn(path: str, img, num_classes: int,
                   top_k: int, class_names: List[str], pre: Dict) -> Dict[str, Any]:
    try:
        import pickle
        import numpy as np

        with open(path, "rb") as f:
            model = pickle.load(f)

        h, w    = pre["size"]
        channels = pre["channels"]
        pil_img = img.convert("L") if channels == 1 else img.convert("RGB")
        arr = np.array(pil_img.resize((w, h))).astype("float32") / 255.0
        feat = arr.flatten().reshape(1, -1)

        if hasattr(model, "predict_proba"):
            probs   = model.predict_proba(feat)[0]
            k       = min(top_k, len(probs))
            top_idx = np.argsort(probs)[::-1][:k]
            names   = _class_names(len(probs), class_names)
            predictions = [
                {"class_name": names[i] if i < len(names) else f"Class {i}",
                 "confidence": float(probs[i]), "class_index": int(i)}
                for i in top_idx
            ]
        else:
            pred = int(model.predict(feat)[0])
            names = _class_names(num_classes, class_names)
            label = names[pred] if pred < len(names) else f"Class {pred}"
            predictions = [{"class_name": label, "confidence": 1.0, "class_index": pred}]

        return {
            "predictions":    predictions,
            "top_class":      predictions[0]["class_name"],
            "top_confidence": predictions[0]["confidence"],
            "framework":      "sklearn",
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
    mv, path, framework, arch, num_classes, stored_class_names, job = _resolve_model(model_id, db)
    top_k = max(1, min(top_k, num_classes))
    img_data = await file.read()
    img = _open_image(img_data)
    return _run_one(path, framework, arch, num_classes, top_k, stored_class_names, img, job, mv)


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
    """
    Return (mv, path, framework, arch, num_classes, class_names, job) or raise 404.
    job is the TrainingJob record (may be None) — needed for preprocessing cascade.
    """
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

    # mv.framework is a Framework enum — use .value to get the plain string
    # (otherwise str() gives "Framework.tensorflow" instead of "tensorflow")
    framework = mv.framework.value if hasattr(mv.framework, "value") else str(mv.framework).lower()
    return mv, path, framework, mv.architecture or "resnet18", num_classes, stored_class_names, job


def _run_one(path: str, framework: str, arch: str, num_classes: int,
             top_k: int, class_names: List[str], img,
             job=None, mv=None) -> Dict[str, Any]:
    """
    Dispatch inference to the correct framework backend.

    Preprocessing (resize, channel conversion, normalization) is determined
    automatically from a 4-tier priority cascade:
      1. Checkpoint metadata (image_size / channels / normalization saved at training time)
      2. TrainingJob.dataset_config   (image_size / channels stored when job was created)
      3. Builtin dataset name lookup  (MNIST → 28×28×1, CIFAR-10 → 32×32×3, …)
      4. Architecture heuristic       (large ImageNet archs → 224×224 RGB)
    The actual first-conv weight shape in the PyTorch state_dict is used as a
    ground-truth override for the channel count.
    """
    # Load the checkpoint once (PyTorch only) so the cascade can read metadata
    checkpoint: Optional[Dict] = None
    if framework == "pytorch" and path and os.path.isfile(path):
        try:
            import torch
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(checkpoint, dict):
                checkpoint = None
        except Exception:
            pass

    pre = _get_preprocess_config(job, mv, checkpoint)

    if framework == "pytorch":
        return _infer_pytorch(path, img, arch, num_classes, top_k, class_names, pre)
    elif framework == "tensorflow":
        return _infer_tensorflow(path, img, num_classes, top_k, class_names)
    elif framework == "sklearn":
        return _infer_sklearn(path, img, num_classes, top_k, class_names, pre)
    raise HTTPException(status_code=400, detail=f"Unsupported framework: {framework!r}")


@router.post(
    "/zip",
    summary="Run batch inference on a ZIP of images (test-samples format)",
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
    mv, path, framework, arch, num_classes, class_names, job = _resolve_model(model_id, db)
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
            result = _run_one(path, framework, arch, num_classes, effective_top_k, class_names, img, job, mv)
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
    csv_text = out.getvalue()

    # Compute summary stats
    labelled = [r for r in rows if r[5] in ("0", "1")]
    correct_n = sum(1 for r in labelled if r[5] == "1")
    acc_str = f"{correct_n / len(labelled) * 100:.1f}%" if labelled else "n/a"

    # Return JSON so the frontend receives stats as plain fields — no HTTP header
    # stripping from proxies, no client-side CSV parsing needed.
    # The CSV is base64-encoded so the browser can reconstruct a download link.
    return {
        "total":      len(rows),
        "labelled":   len(labelled),
        "correct":    correct_n,
        "accuracy":   acc_str,
        "has_labels": len(labelled) > 0,
        "csv_b64":    base64.b64encode(csv_text.encode("utf-8")).decode("ascii"),
        "filename":   "inference_results.csv",
    }
