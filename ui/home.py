"""
Home page – VisionForge (Train Vision Models Effortlessly).
"""
import streamlit as st
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.shared import st_rerun
from ui.theme import (hero, section, card, stat_row, info_card,
                      task_card_html, gradient_text, apply_theme, step_tracker)
from ui.helpers import check_system_status

logger = logging.getLogger(__name__)


# ── Task definitions ──────────────────────────────────────────────────────────
_TASKS = {
    "Classification": {
        "icon": "🏷️", "color": "#6c63ff",
        "tagline": "Assign labels to entire images",
        "desc": "Auto-categorise images into classes with state-of-the-art deep learning and transfer learning.",
        "pills": ["CIFAR-10/100", "ImageNet", "MNIST", "Custom folders"],
        "features": [
            ("🔍", "Auto class detection", "purple"),
            ("⚡", "Transfer learning", "teal"),
            ("📊", "Accuracy / F1 tracking", "purple"),
            ("🚀", "Smart augmentation", "teal"),
        ],
    },
    "Segmentation": {
        "icon": "🎨", "color": "#00d4aa",
        "tagline": "Label every pixel in images",
        "desc": "Pixel-wise semantic segmentation for medical imaging, autonomous driving, and satellite imagery.",
        "pills": ["COCO", "Pascal VOC", "Cityscapes", "Custom masks"],
        "features": [
            ("🔍", "Mask auto-analysis", "teal"),
            ("🎯", "U-Net / DeepLab", "purple"),
            ("📊", "IoU / Dice score", "teal"),
            ("🚀", "Paired augmentation", "purple"),
        ],
    },
    "Object Detection": {
        "icon": "📦", "color": "#ffd93d",
        "tagline": "Find and locate objects in images",
        "desc": "Detect and localise multiple objects per image with modern YOLO, DETR, and Faster R-CNN architectures.",
        "pills": ["COCO", "Pascal VOC", "YOLO format", "Custom annotations"],
        "features": [
            ("🔍", "COCO / YOLO / VOC", "yellow"),
            ("🎯", "YOLO / DETR / FCOS", "purple"),
            ("📊", "mAP @ IoU 0.5", "yellow"),
            ("🚀", "Anchor optimisation", "teal"),
        ],
    },
}

_FW_INFO = {
    "PyTorch": {
        "icon": "🔥",
        "desc": "Dynamic computation graphs. Best for research & custom architectures.",
    },
    "TensorFlow/Keras": {
        "icon": "🧠",
        "desc": "Production-ready static graphs. Best for deployment & scalability.",
    },
    "Scikit-learn": {
        "icon": "📊",
        "desc": "Classical ML algorithms. Best for tabular / structured data.",
    },
}


def show_home() -> None:
    """Render the Home page."""
    apply_theme()

    if not st.session_state.get("task_type_confirmed", False):
        _show_task_selection()
        return
    _show_dashboard()


# ─────────────────────────────────────────────────────────────────────────────

def _show_task_selection():
    hero(
        "Choose Your Vision Task",
        "Select the computer-vision problem you want to solve — VisionForge adapts automatically.",
        badges=["🔭 VisionForge", "Zero Config", "GPU Ready"],
    )

    cols = st.columns(3, gap="large")
    for col, key in zip(cols, _TASKS):
        t = _TASKS[key]
        with col:
            st.markdown(task_card_html(t["icon"], key, t["tagline"], t["pills"]),
                        unsafe_allow_html=True)
            st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
            if st.button(f"Select {key}", key=f"task_{key}", use_container_width=True):
                st.session_state.selected_task_type = key
                st.session_state.task_type_confirmed = True
                st_rerun()

    # Feature highlights row
    st.markdown("<br>", unsafe_allow_html=True)
    feats = [
        ("🧠", "AutoModel",     "Smart model selection"),
        ("⚡", "HyperOptimizer","Bayesian HPO"),
        ("📊", "LiveMetrics",   "Real-time monitoring"),
        ("📤", "Export",        "ONNX / TorchScript"),
        ("🔍", "Inference",     "GradCAM + batch predict"),
    ]
    boxes = "".join(
        f'<div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);'
        f'border-radius:12px;padding:.9rem 1.4rem;min-width:160px;text-align:center;flex:1">'
        f'<div style="font-size:1.5rem">{ic}</div>'
        f'<div style="font-weight:700;margin:.3rem 0 .1rem;font-size:.92rem">{h}</div>'
        f'<div style="font-size:.78rem;color:#a0a0b8">{d}</div></div>'
        for ic, h, d in feats
    )
    st.markdown(f'<div style="display:flex;gap:.8rem;flex-wrap:wrap;margin-top:.5rem">{boxes}</div>',
                unsafe_allow_html=True)


def _show_dashboard():
    task = st.session_state.selected_task_type
    t    = _TASKS.get(task, _TASKS["Classification"])

    hero(
        f"{t['icon']} {task} Pipeline",
        t["desc"],
        badges=["🔭 VisionForge", f"🎯 {task}", "v2.0"],
    )
    step_tracker(["Home", "Dataset", "Model", "Training", "Results"],
                 st.session_state.get("current_step", 0))

    left, right = st.columns([3, 2], gap="large")

    with left:
        section("✨", "Key Features")
        feat_cols = st.columns(2)
        for idx, (icon, title, color) in enumerate(t["features"]):
            with feat_cols[idx % 2]:
                info_card(icon, title, "", color)

        st.markdown("<br>", unsafe_allow_html=True)
        section("🚀", "Quick Start")
        steps_html = "".join(
            f'<div style="display:flex;align-items:flex-start;gap:.8rem;margin-bottom:.8rem">'
            f'<div style="background:rgba(108,99,255,.2);border:1px solid rgba(108,99,255,.4);'
            f'border-radius:50%;width:28px;height:28px;display:flex;align-items:center;'
            f'justify-content:center;font-weight:700;font-size:.85rem;flex-shrink:0;color:#a89cff">{n}</div>'
            f'<div style="padding-top:.15rem;font-size:.9rem;color:#c0c0d8">{txt}</div></div>'
            for n, txt in [
                (1, "Select or upload your dataset on the <b>Dataset</b> page"),
                (2, "Review the auto-selected model architecture on the <b>Model</b> page"),
                (3, "Configure training parameters and click <b>Start Training</b>"),
                (4, "Explore metrics, download model, run <b>Inference</b> on new images"),
            ]
        )
        card(steps_html)

    with right:
        section("⚙️", "Session Configuration")

        # Active task card
        st.markdown(f"""
        <div class="cv-card" style="border-left:3px solid {t['color']};">
          <div style="font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;
               color:var(--text-secondary);font-weight:600;margin-bottom:.3rem">Active Task</div>
          <div style="font-size:1.5rem">{t['icon']}</div>
          <div style="font-weight:800;font-size:1.05rem;margin-top:.2rem">{task}</div>
          <div style="font-size:.82rem;color:var(--text-secondary);margin-top:.2rem">{t['tagline']}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔄 Change Task", key="change_task", use_container_width=True):
            st.session_state.task_type_confirmed = False
            st.session_state.selected_task_type  = None
            st.session_state.project_started     = False
            st.session_state.dataset_info        = None
            st.session_state.model_config        = None
            st.session_state.training_completed  = False
            st_rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        section("🛠️", "Framework")

        fw_opts = list(_FW_INFO.keys())
        cur_fw  = st.session_state.get("selected_framework", "PyTorch")
        try:
            fw_idx = fw_opts.index(cur_fw)
        except ValueError:
            fw_idx = 0

        fw = st.selectbox("ML Framework", fw_opts, index=fw_idx,
                          key="home_fw_select", label_visibility="collapsed")
        st.session_state.selected_framework = fw
        fw_meta = _FW_INFO[fw]
        st.markdown(
            f'<div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);'
            f'border-radius:10px;padding:.8rem 1rem;margin-top:.4rem"><span style="font-size:1.2rem">'
            f'{fw_meta["icon"]}</span>'
            f'<span style="font-size:.83rem;color:#a0a0b8;margin-left:.5rem">{fw_meta["desc"]}</span></div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        section("💻", "System Status")
        for item, ok in check_system_status().items():
            colour = "#00d4aa" if ok else "#ff6b6b"
            sym    = "✓" if ok else "✗"
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:.5rem;padding:.28rem 0;font-size:.87rem">'
                f'<span style="color:{colour};font-weight:700">{sym}</span>'
                f'<span style="color:var(--text-secondary)">{item}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        section("⚡", "Quick Actions")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📊 Dataset", use_container_width=True):
                st.session_state.project_started = True
                st.session_state.current_step = 1
                st.session_state["current_tool"] = None
                st_rerun()
        with c2:
            if st.button("🔍 Inference", key="home_inference", use_container_width=True):
                st.session_state["current_tool"] = "inference"
                st_rerun()

        if st.session_state.get("training_results"):
            res = st.session_state.training_results
            acc = res.get("best_accuracy", 0)
            if acc <= 1:
                acc *= 100
            st.markdown("<br>", unsafe_allow_html=True)
            section("🏆", "Latest Run")
            stat_row([
                (f"{acc:.1f}%", "Accuracy"),
                (f"{res.get('best_loss', 0):.3f}", "Loss"),
                (f"{res.get('training_time', 0):.0f}s", "Time"),
            ])


