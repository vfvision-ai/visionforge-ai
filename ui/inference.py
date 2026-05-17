"""
VisionForge — Model Inference page. Upload images and get predictions from a trained model.
"""

import streamlit as st
import os
import sys
import json
import logging
import io
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))
from ui.shared import st_rerun, get_dataset_class_names
from ui.theme import apply_theme, hero, section, card, stat_row, metric_card

logger = logging.getLogger(__name__)


# ── helpers ─────────────────────────────────────────────────────────────────

def _find_recent_model() -> Optional[Path]:
    """Return the most recently modified model file under experiments/ or models/."""
    search_dirs = ["experiments", "models", "checkpoints"]
    candidates = []
    for d in search_dirs:
        p = Path(d)
        if p.exists():
            for ext in ("*.pt", "*.pth", "*.keras", "*.h5", "*.pkl", "*.joblib"):
                candidates.extend(p.rglob(ext))
    if not candidates:
        return None
    return max(candidates, key=lambda f: f.stat().st_mtime)


def _load_pytorch_model(model_path: Path, num_classes: int, device: str):
    import torch
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        # Try to reconstruct from checkpoint metadata
        arch = checkpoint.get("architecture", checkpoint.get("model_name", "resnet18"))
        try:
            import torchvision.models as tvm
            constructor = getattr(tvm, arch.lower(), None)
            if constructor:
                model = constructor(weights=None)
                # Replace final layer
                import torch.nn as nn
                if hasattr(model, "fc"):
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
                elif hasattr(model, "classifier"):
                    last = model.classifier[-1]
                    model.classifier[-1] = nn.Linear(last.in_features, num_classes)
                state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
                model.load_state_dict(state, strict=False)
            else:
                model = checkpoint  # raw model object in checkpoint
        except Exception:
            model = checkpoint  # fall back – caller decides
    else:
        model = checkpoint  # already a model object
    if hasattr(model, "eval"):
        model.eval()
    return model


def _preprocess_image_pil(img, input_size: int = 224, grayscale: bool = False):
    """PIL Image → normalised torch tensor (1, C, H, W)."""
    import torch
    from torchvision import transforms

    if grayscale:
        transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    else:
        # Convert to RGB before the transform pipeline (transforms.RGB() doesn't exist)
        img = img.convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transform(img).unsqueeze(0)


def _preprocess_image_pil_safe(img, input_size: int = 224):
    """Robust PIL Image → torch tensor, handles 1/3/4 channel images."""
    import torch
    from torchvision import transforms

    if img.mode == "RGBA":
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)


def _run_pytorch_inference(model, img_tensor, class_names: List[str], device: str):
    import torch
    import torch.nn.functional as F
    model = model.to(device)
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    top_k = min(5, len(class_names))
    top_idx = np.argsort(probs)[::-1][:top_k]
    return [
        {"class": class_names[i] if i < len(class_names) else f"Class {i}",
         "confidence": float(probs[i])}
        for i in top_idx
    ]


def _run_sklearn_inference(model, img_pil, class_names: List[str]):
    import numpy as np
    from PIL import Image

    arr = np.array(img_pil.convert("L").resize((28, 28))).flatten().reshape(1, -1) / 255.0
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(arr)[0]
        top_k = min(5, len(probs))
        top_idx = np.argsort(probs)[::-1][:top_k]
        return [
            {"class": class_names[i] if i < len(class_names) else f"Class {i}",
             "confidence": float(probs[i])}
            for i in top_idx
        ]
    else:
        pred = int(model.predict(arr)[0])
        label = class_names[pred] if pred < len(class_names) else f"Class {pred}"
        return [{"class": label, "confidence": 1.0}]


# ── GradCAM ──────────────────────────────────────────────────────────────────

def _compute_gradcam(model, img_tensor, target_class: Optional[int], device: str):
    """Compute GradCAM heatmap for the given image and class index."""
    import torch
    import torch.nn.functional as F

    model = model.to(device)
    img_tensor = img_tensor.to(device).requires_grad_(False)

    # Find the last Conv2d layer
    target_layer = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    if target_layer is None:
        return None

    gradients = []
    activations = []

    def forward_hook(module, inp, out):
        activations.append(out.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_backward_hook(backward_hook)

    try:
        model.zero_grad()
        output = model(img_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[0, target_class]
        score.backward()

        if not gradients or not activations:
            return None

        grads = gradients[0]          # (1, C, H, W)
        acts = activations[0]         # (1, C, H, W)
        weights = grads.mean(dim=[2, 3], keepdim=True)   # (1, C, 1, 1)
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
    finally:
        fh.remove()
        bh.remove()


def _overlay_heatmap(img_pil, cam: np.ndarray):
    """Return a PIL Image with the GradCAM heatmap overlaid."""
    import cv2
    from PIL import Image

    w, h = img_pil.size
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    orig = np.array(img_pil.convert("RGB"))
    overlay = (0.5 * orig + 0.5 * heatmap_rgb).astype(np.uint8)
    return Image.fromarray(overlay)


# ── main page ─────────────────────────────────────────────────────────────────

def show_inference():
    """Model Inference & Prediction page."""
    apply_theme()

    hero(
        "🔍 Model Inference",
        "Upload images and run predictions using any trained model — with GradCAM visualisations.",
        badges=["🔭 VisionForge", "🤖 PyTorch", "🧠 TensorFlow", "📊 Scikit-learn"],
    )

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Inference Settings")
        input_size = st.slider("Input Size", 28, 512, 224, step=4,
                               help="Resize images to this square size before inference")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.1, step=0.05,
                                         help="Only show predictions above this confidence")
        show_gradcam = st.checkbox("Show GradCAM Heatmap", value=False,
                                   help="Visualise which regions influenced the prediction (PyTorch only)")
        top_k = st.slider("Top-K predictions", 1, 10, 5)

    # ── Model selection ───────────────────────────────────────────────────────
    section("🔍", "1 — Select Model")

    model_source = st.radio(
        "Model Source",
        ["Use Latest Trained Model", "Browse for Model File", "Enter Path Manually"],
        horizontal=True,
    )

    model_path: Optional[Path] = None

    if model_source == "Use Latest Trained Model":
        # Check session state first
        if st.session_state.get("training_results") and st.session_state.training_results.get("model_path"):
            model_path = Path(st.session_state.training_results["model_path"])
            st.success(f"✅ Session model: `{model_path.name}`")
        else:
            auto = _find_recent_model()
            if auto:
                model_path = auto
                st.success(f"✅ Found: `{model_path}` ({model_path.stat().st_size / 1e6:.2f} MB)")
            else:
                st.warning("⚠️ No trained model found. Train a model first or specify a path.")

    elif model_source == "Browse for Model File":
        uploaded_model = st.file_uploader(
            "Upload model file",
            type=["pt", "pth", "keras", "h5", "pkl", "joblib"],
            help="Upload a saved model file",
        )
        if uploaded_model:
            tmp_dir = Path("uploads") / "models"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            model_path = tmp_dir / uploaded_model.name
            model_path.write_bytes(uploaded_model.read())
            st.success(f"✅ Uploaded: `{model_path.name}`")

    else:  # manual path
        path_input = st.text_input("Model file path", placeholder="/path/to/model.pt")
        if path_input and Path(path_input).exists():
            model_path = Path(path_input)
            st.success(f"✅ Found: `{model_path.name}`")
        elif path_input:
            st.error("File not found.")

    # ── Determine framework & class names ─────────────────────────────────────
    framework = st.session_state.get("selected_framework", "PyTorch")
    dataset_info = st.session_state.get("dataset_info")

    # Infer class names
    class_names: List[str] = []
    if dataset_info:
        ds_name = getattr(dataset_info, "dataset_name", None) or getattr(dataset_info, "name", None)
        known = get_dataset_class_names(ds_name) if ds_name else None
        if known:
            class_names = known
        elif hasattr(dataset_info, "class_names") and dataset_info.class_names:
            class_names = dataset_info.class_names
        elif hasattr(dataset_info, "num_classes") and dataset_info.num_classes:
            class_names = [f"Class {i}" for i in range(dataset_info.num_classes)]

    # Allow manual class-name override
    with st.expander("🏷️ Class Names (optional override)", expanded=not class_names):
        classes_input = st.text_area(
            "Enter class names, one per line",
            value="\n".join(class_names),
            height=100,
        )
        if classes_input.strip():
            class_names = [c.strip() for c in classes_input.splitlines() if c.strip()]

    if not class_names:
        num_cls = st.number_input("Number of classes (if unknown)", min_value=2, max_value=1000, value=10)
        class_names = [f"Class {i}" for i in range(int(num_cls))]

    # ── Image upload ──────────────────────────────────────────────────────────
    st.markdown("---")
    section("📷", "2 — Upload Images")

    upload_mode = st.radio("Upload Mode", ["Single Image", "Batch (multiple images)"], horizontal=True)

    uploaded_files = st.file_uploader(
        "Choose image(s)",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        accept_multiple_files=(upload_mode == "Batch (multiple images)"),
    )

    if not uploaded_files:
        st.info("👆 Upload one or more images to start inference.")
        return

    if isinstance(uploaded_files, list):
        files = uploaded_files
    else:
        files = [uploaded_files]

    # ── Run inference ─────────────────────────────────────────────────────────
    st.markdown("---")
    section("🎯", "3 — Predictions")

    if not model_path or not model_path.exists():
        st.error("❌ Please select a valid model file above before running inference.")
        return

    run_btn = st.button("▶️ Run Inference", type="primary", use_container_width=True)
    if not run_btn:
        return

    # Load model (cached per path)
    @st.cache_resource(show_spinner="Loading model…")
    def _load_model(path_str: str, n_classes: int, fw: str):
        p = Path(path_str)
        suffix = p.suffix.lower()
        if suffix in (".pt", ".pth"):
            import torch
            device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            return _load_pytorch_model(p, n_classes, device), "pytorch", device
        elif suffix in (".pkl", ".joblib"):
            import joblib
            return joblib.load(p), "sklearn", "cpu"
        elif suffix in (".keras", ".h5"):
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(str(p))
                return model, "tensorflow", "cpu"
            except Exception as e:
                raise RuntimeError(f"Cannot load Keras model: {e}")
        else:
            raise ValueError(f"Unsupported model format: {suffix}")

    try:
        model, detected_fw, device = _load_model(str(model_path), len(class_names), framework)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return

    st.success(f"✅ Model loaded ({detected_fw.upper()}, device: {device})")

    # Process each image
    results_list = []
    cols = st.columns(min(len(files), 4))

    for idx, file in enumerate(files):
        from PIL import Image as PILImage

        img = PILImage.open(file)
        col = cols[idx % len(cols)]

        with col:
            st.image(img, caption=file.name, use_container_width=True)

            with st.spinner("Running…"):
                try:
                    if detected_fw == "pytorch":
                        import torch
                        tensor = _preprocess_image_pil_safe(img, input_size)
                        preds = _run_pytorch_inference(model, tensor, class_names, device)

                        # GradCAM
                        if show_gradcam:
                            try:
                                cam = _compute_gradcam(model, tensor, None, device)
                                if cam is not None:
                                    overlay = _overlay_heatmap(img, cam)
                                    st.image(overlay, caption="GradCAM", use_container_width=True)
                            except Exception as ge:
                                st.caption(f"GradCAM unavailable: {ge}")

                    elif detected_fw == "sklearn":
                        preds = _run_sklearn_inference(model, img, class_names)

                    elif detected_fw == "tensorflow":
                        import numpy as np
                        arr = np.array(img.convert("RGB").resize((input_size, input_size))) / 255.0
                        arr = arr[np.newaxis, ...]
                        raw = model.predict(arr, verbose=0)
                        import tensorflow as tf
                        probs = tf.nn.softmax(raw).numpy()[0] if raw.shape[-1] > 1 else raw[0]
                        top_k_local = min(top_k, len(class_names))
                        top_idx = np.argsort(probs)[::-1][:top_k_local]
                        preds = [
                            {"class": class_names[i] if i < len(class_names) else f"Class {i}",
                             "confidence": float(probs[i])}
                            for i in top_idx
                        ]
                    else:
                        st.error("Unknown framework.")
                        continue

                    # Filter by threshold
                    preds = [p for p in preds if p["confidence"] >= confidence_threshold]

                    if preds:
                        best = preds[0]
                        st.metric("Top Prediction", best["class"], f"{best['confidence']*100:.1f}%")

                        # Confidence bar chart
                        df = pd.DataFrame(preds[:top_k])
                        fig = px.bar(
                            df, x="confidence", y="class", orientation="h",
                            text=df["confidence"].apply(lambda v: f"{v*100:.1f}%"),
                            color="confidence",
                            color_continuous_scale="Blues",
                            range_x=[0, 1],
                        )
                        fig.update_layout(
                            height=max(150, 40 * len(df)),
                            margin=dict(l=0, r=0, t=10, b=0),
                            coloraxis_showscale=False,
                            yaxis_title=None,
                            xaxis_title="Confidence",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        results_list.append({
                            "filename": file.name,
                            "top_class": best["class"],
                            "top_confidence": f"{best['confidence']*100:.1f}%",
                            **{f"class_{i+1}": p["class"] for i, p in enumerate(preds[:top_k])},
                            **{f"conf_{i+1}": f"{p['confidence']*100:.1f}%" for i, p in enumerate(preds[:top_k])},
                        })
                    else:
                        st.warning(f"No predictions above {confidence_threshold:.0%} threshold.")

                except Exception as e:
                    st.error(f"Inference failed: {e}")
                    logger.exception("Inference error on %s", file.name)

    # ── Results table & download ───────────────────────────────────────────────
    if results_list:
        st.markdown("---")
        section("📊", "Summary Table")
        df_results = pd.DataFrame(results_list)
        st.dataframe(df_results, use_container_width=True)

        csv_bytes = df_results.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Results CSV",
            data=csv_bytes,
            file_name="inference_results.csv",
            mime="text/csv",
        )

