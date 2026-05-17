"""
Training page – VisionForge (Train Vision Models Effortlessly).
"""

import streamlit as st
import os
import sys
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.dataset_analyzer import DatasetAnalyzer, DatasetInfo
from core.model_selector import ModelSelector, ModelConfig
from utils.config import Config

logger = logging.getLogger(__name__)

from ui.shared import st_rerun, get_dataset_class_names, save_test_samples_for_evaluation
from ui.theme import apply_theme, hero, section, card, stat_row, info_card, step_tracker

from ui.helpers import (
    check_system_status,
    get_user_specified_input_size,
    get_framework_models,
    select_tensorflow_model,
    select_sklearn_model,
    start_training,
    start_training_with_config,
)

def show_training():
    """Display training interface."""
    apply_theme()

    # Guard: task type must be selected
    if not st.session_state.get('task_type_confirmed', False) or st.session_state.get('selected_task_type') is None:
        hero("🔥 Model Training",
             "Please select a task type on the Home page first.",
             badges=["Step 4"])
        if st.button("← Go to Home", type="primary"):
            st.session_state.current_step = 0
            st_rerun()
        return

    task = st.session_state.get('selected_task_type', '')
    training_status = st.session_state.get('training_status')

    hero(
        "🔥 Model Training",
        f"Configure and launch your {task} training run — monitor metrics in real time.",
        badges=[f"🎯 {task}", "Step 4 of 5"],
    )
    step_tracker(["Home","Dataset","Model","Training","Results"],
                 st.session_state.get("current_step", 3))

    # ── COMPLETED STATE — full-width results view ─────────────────────────────
    if training_status == 'completed':
        results = st.session_state.get('training_results', {})
        acc = results.get('best_accuracy', 0)
        if acc <= 1:
            acc *= 100

        st.markdown("""
        <div style="background:rgba(0,212,170,.08);border:1px solid rgba(0,212,170,.3);
             border-radius:12px;padding:1rem 1.6rem;margin-bottom:1.2rem">
          <div style="font-weight:700;font-size:1.1rem;color:#00d4aa">✅ Training Complete</div>
        </div>
        """, unsafe_allow_html=True)

        stat_row([
            (f"{acc:.2f}%",                          "Best Accuracy"),
            (f"{results.get('best_loss', 0):.4f}",   "Best Loss"),
            (f"{results.get('training_time', 0):.0f}s", "Training Time"),
            (str(results.get('total_epochs', '—')),  "Epochs Run"),
        ])

        # ── Training curves from log_history ────────────────────────────────
        log_hist = st.session_state.get("training_log_history", [])
        training_history = results.get("training_history", {})

        # Build epoch lists either from log_history dicts or from training_history lists
        epochs_x = None
        t_loss_vals = val_loss_vals = t_acc_vals = val_acc_vals = None

        if log_hist and isinstance(log_hist[0], dict) and "epoch" in log_hist[0]:
            df_log = pd.DataFrame(log_hist)
            epochs_x     = df_log["epoch"].tolist()
            t_loss_vals  = df_log.get("loss",         pd.Series(dtype=float)).tolist()
            val_loss_vals = df_log.get("val_loss",    pd.Series(dtype=float)).tolist()
            t_acc_vals   = (df_log.get("accuracy",    pd.Series(dtype=float)) * 100).tolist()
            val_acc_vals = (df_log.get("val_accuracy", pd.Series(dtype=float)) * 100).tolist()
        elif training_history:
            t_loss_vals   = training_history.get("train_loss", [])
            val_loss_vals = training_history.get("val_loss", [])
            t_acc_vals    = [v * 100 for v in training_history.get("train_accuracy", [])]
            val_acc_vals  = [v * 100 for v in training_history.get("val_accuracy", [])]
            if t_loss_vals:
                epochs_x = list(range(1, len(t_loss_vals) + 1))

        if epochs_x:
            section("📈", "Training Curves")
            ch1, ch2 = st.columns(2)
            with ch1:
                fig_l = go.Figure()
                if t_loss_vals:
                    fig_l.add_trace(go.Scatter(x=epochs_x, y=t_loss_vals, mode="lines",
                        name="Train Loss", line=dict(color="#6c63ff", width=2)))
                if val_loss_vals:
                    fig_l.add_trace(go.Scatter(x=epochs_x, y=val_loss_vals, mode="lines",
                        name="Val Loss", line=dict(color="#ff6b6b", width=2, dash="dash")))
                fig_l.update_layout(
                    title="Loss", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", height=260,
                    margin=dict(l=0, r=0, t=30, b=0),
                    legend=dict(orientation="h", y=-0.2),
                    xaxis=dict(title="Epoch", gridcolor="rgba(255,255,255,.06)"),
                    yaxis=dict(title="Loss",  gridcolor="rgba(255,255,255,.06)"),
                )
                st.plotly_chart(fig_l, use_container_width=True, config={"displayModeBar": False})
            with ch2:
                fig_a = go.Figure()
                if t_acc_vals:
                    fig_a.add_trace(go.Scatter(x=epochs_x, y=t_acc_vals, mode="lines",
                        name="Train Acc", line=dict(color="#00d4aa", width=2)))
                if val_acc_vals:
                    fig_a.add_trace(go.Scatter(x=epochs_x, y=val_acc_vals, mode="lines",
                        name="Val Acc", line=dict(color="#ffd93d", width=2, dash="dash")))
                fig_a.update_layout(
                    title="Accuracy", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", height=260,
                    margin=dict(l=0, r=0, t=30, b=0),
                    legend=dict(orientation="h", y=-0.2),
                    xaxis=dict(title="Epoch",     gridcolor="rgba(255,255,255,.06)"),
                    yaxis=dict(title="Accuracy %", gridcolor="rgba(255,255,255,.06)"),
                )
                st.plotly_chart(fig_a, use_container_width=True, config={"displayModeBar": False})

        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if st.button("➡️ View Detailed Results", type="primary", use_container_width=True):
                st.session_state.current_step = 4
                st.session_state["current_tool"] = None
                st_rerun()
        with btn_c2:
            if st.button("🔄 Train Again", use_container_width=True):
                st.session_state.pop('training_status', None)
                st.session_state.pop('training_results', None)
                st.session_state.pop('training_log_history', None)
                st_rerun()
        return

    # ── FAILED STATE ──────────────────────────────────────────────────────────
    if training_status == 'failed':
        st.error("❌ Training failed. See error details above. Adjust your configuration and try again.")
        if st.button("🔄 Reset & Try Again", type="primary"):
            st.session_state.pop('training_status', None)
            st_rerun()
        st.markdown("---")

    if not st.session_state.get('dataset_info') or not st.session_state.get('model_config'):
        st.markdown("""
        <div style="background:rgba(255,107,107,.08);border:1px solid rgba(255,107,107,.3);
             border-radius:12px;padding:1.2rem 1.6rem;margin-bottom:1rem">
          <div style="font-weight:700;font-size:1rem;margin-bottom:.4rem">⚠️ Prerequisites missing</div>
          <div style="font-size:.88rem;color:#c0c0d8">
            Please complete <b>Dataset Analysis</b> (Step 2) and <b>Model Selection</b> (Step 3) before training.
          </div>
        </div>
        """, unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📊 Go to Dataset", use_container_width=True):
                st.session_state.current_step = 1
                st_rerun()
        with c2:
            if st.button("🧠 Go to Model Selection", use_container_width=True):
                st.session_state.current_step = 2
                st_rerun()
        return

    # ── Training Presets ─────────────────────────────────────────────────────
    section("⚡", "Quick Presets")
    st.markdown('<div style="font-size:.84rem;color:var(--text-secondary);margin-bottom:.6rem">One-click configuration — adjust individual settings below if needed.</div>', unsafe_allow_html=True)

    _PRESETS = {
        "🐢 Debug":    {"epochs": 3,   "lr": 0.001,  "batch": 32, "es": False, "hpo": False, "patience": 5,  "desc": "3 epochs · fast sanity check"},
        "🚀 Quick":    {"epochs": 10,  "lr": 0.001,  "batch": 64, "es": True,  "hpo": False, "patience": 5,  "desc": "10 epochs · no HPO"},
        "⭐ Standard": {"epochs": 50,  "lr": 0.0005, "batch": 64, "es": True,  "hpo": False, "patience": 10, "desc": "50 epochs · early stopping"},
        "💪 Full":     {"epochs": 150, "lr": 0.0001, "batch": 32, "es": True,  "hpo": True,  "patience": 15, "desc": "150 epochs · HPO enabled"},
    }

    if "preset_vals" not in st.session_state:
        st.session_state.preset_vals = _PRESETS["⭐ Standard"].copy()

    pcols = st.columns(4)
    for i, (pname, pvals) in enumerate(_PRESETS.items()):
        with pcols[i]:
            is_active = st.session_state.preset_vals.get("_name") == pname
            border = "border:2px solid #6c63ff;" if is_active else "border:1px solid rgba(255,255,255,.1);"
            st.markdown(
                f'<div style="background:rgba(255,255,255,.04);{border}border-radius:10px;'
                f'padding:.6rem .8rem;text-align:center;margin-bottom:.4rem">'
                f'<div style="font-weight:700;font-size:.95rem">{pname}</div>'
                f'<div style="font-size:.72rem;color:#a0a0b8;margin-top:.2rem">{pvals["desc"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("Apply", key=f"preset_{pname}", use_container_width=True):
                st.session_state.preset_vals = {**pvals, "_name": pname}
                st_rerun()

    pv = st.session_state.preset_vals
    st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,.06);margin:.8rem 0'>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # _pending_config is set inside the column (button click) and consumed
    # OUTSIDE the column so all training UI (live charts, progress) renders
    # at full-width page level, not cramped inside a column.
    # ─────────────────────────────────────────────────────────────────────────
    _pending_config = None

    col1, col2 = st.columns([1, 1])

    with col1:
        section("⚙️", "Training Configuration")

        import datetime as _dt
        default_exp_name = f"exp_{_dt.datetime.now().strftime('%Y%m%d_%H%M')}"
        exp_name = st.text_input(
            "Experiment Name",
            value=st.session_state.get("experiment_name", default_exp_name),
            help="Used as the output folder name. Keep it descriptive.",
        )
        st.session_state.experiment_name = exp_name

        max_epochs = st.slider("Max Epochs", 1, 200, int(pv.get("epochs", 50)))

        lr_options = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001]
        preset_lr = pv.get("lr", 0.001)
        lr_idx = min(range(len(lr_options)), key=lambda i: abs(lr_options[i] - preset_lr))
        learning_rate = st.select_slider(
            "Learning Rate",
            options=lr_options,
            value=lr_options[lr_idx],
            format_func=lambda x: f"{x:.5f}".rstrip("0"),
            help="Initial learning rate for the optimizer.",
        )
        st.session_state.preset_vals["lr"] = learning_rate

        batch_size = st.select_slider(
            "Batch Size",
            options=[8, 16, 32, 64, 128, 256],
            value=int(pv.get("batch", 64)),
            help="Larger values train faster but need more GPU memory.",
        )
        st.session_state.preset_vals["batch"] = batch_size

        st.markdown("#### ⏹️ Early Stopping")
        enable_early_stopping = st.checkbox(
            "Enable Early Stopping",
            value=bool(pv.get("es", True)),
            help="Automatically stop training when validation performance stops improving",
        )
        if enable_early_stopping:
            col_es1, col_es2 = st.columns(2)
            with col_es1:
                early_stopping_patience = st.slider("Patience", 3, 30, int(pv.get("patience", 10)),
                                                    help="Epochs to wait before stopping")
            with col_es2:
                early_stopping_min_delta = st.number_input("Min Improvement", 0.0001, 0.01, 0.001,
                                                           format="%.4f")
        else:
            early_stopping_patience = None
            early_stopping_min_delta = None

        optimize_hyperparams = st.checkbox(
            "Enable Hyperparameter Optimization",
            value=bool(pv.get("hpo", False)),
            help="Use Bayesian optimization to find best hyperparameters",
        )
        if optimize_hyperparams:
            try:
                import optuna  # noqa: F401
                st.info("✅ Optuna available — Bayesian HPO will run before the main training loop.")
            except ImportError:
                st.warning(
                    "⚠️ **Optuna is not installed.** Hyperparameter optimization will be skipped "
                    "and default parameters will be used.  \n"
                    "Install it with: `pip install optuna==3.3.0`"
                )
                optimize_hyperparams = False  # disable silently-broken HPO
        n_trials = st.slider("Optimization Trials", 5, 50, 20) if optimize_hyperparams else 20

        st.markdown("#### 📁 Output Directory")
        output_method = st.radio(
            "Choose output location:",
            ["📝 Default Location", "✏️ Custom Path"],
            horizontal=True,
            key="output_method",
        )
        import os as _os
        if output_method == "📝 Default Location":
            output_dir = f"./experiments/{exp_name}"
            st.info(f"📁 Using: `{output_dir}`")
        else:
            output_dir = st.text_input("Custom Output Directory", value="./experiments",
                                       help="Enter full path to output directory")

        # Build and display config summary
        training_config = {
            'experiment_name': exp_name,
            'max_epochs': max_epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'optimize_hyperparams': optimize_hyperparams,
            'n_trials': n_trials,
            'output_dir': output_dir,
            'enable_early_stopping': enable_early_stopping,
            'early_stopping_patience': early_stopping_patience,
            'early_stopping_min_delta': early_stopping_min_delta,
        }

        st.markdown("#### 📋 Summary")
        sum_c1, sum_c2 = st.columns(2)
        with sum_c1:
            st.markdown(f"🗂️ **Dataset:** {st.session_state.dataset_info.task_type.title()}")
            st.markdown(f"🧠 **Model:** {st.session_state.model_config.architecture}")
            st.markdown(f"📦 **Batch:** {batch_size}")
        with sum_c2:
            st.markdown(f"🔄 **Epochs:** {max_epochs}")
            st.markdown(f"📈 **LR:** {learning_rate}")
            st.markdown(f"🔧 **HPO:** {'✅ Yes' if optimize_hyperparams else '❌ No'}")

        import json as _json
        with st.expander("📤 Export Config (JSON)"):
            st.code(_json.dumps(training_config, indent=2), language="json")
            st.download_button("⬇️ Download config.json",
                               data=_json.dumps(training_config, indent=2),
                               file_name=f"{exp_name}_config.json",
                               mime="application/json")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            st.session_state.pop('auto_navigate_results', None)
            # Do NOT call training here — set the pending flag instead.
            # Training will run OUTSIDE the column context so live charts
            # are rendered at full-page width.
            _pending_config = training_config

    with col2:
        section("📦", "Experiment Overview")
        ds = st.session_state.dataset_info
        mc = st.session_state.model_config
        fw = st.session_state.get("selected_framework", "PyTorch")
        fw_icon = {"PyTorch": "🔥", "TensorFlow/Keras": "🧠", "Scikit-learn": "📊"}.get(fw, "🛠️")

        est_time = getattr(ds, 'estimated_training_time', None)
        est_str = ""
        if est_time:
            est_total = est_time * max_epochs / max(getattr(ds, 'num_samples', 1000) / 1000, 1)
            est_str = f"⏱️ Estimated: <b>~{est_total:.0f}s</b>"

        st.markdown(f"""
        <div class="cv-card" style="border-left:3px solid var(--accent);margin-bottom:.8rem">
          <div style="font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;
               color:var(--text-secondary);margin-bottom:.4rem">Dataset</div>
          <div style="font-weight:700">{getattr(ds,'dataset_name',getattr(ds,'task_type','—')).title()}</div>
          <div style="font-size:.82rem;color:var(--text-secondary);margin-top:.3rem">
            🏷️ {getattr(ds,'num_classes',0)} classes &nbsp;·&nbsp;
            🖼️ {getattr(ds,'num_samples',0):,} samples
          </div>
        </div>
        <div class="cv-card" style="border-left:3px solid var(--accent2);margin-bottom:.8rem">
          <div style="font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;
               color:var(--text-secondary);margin-bottom:.4rem">Model</div>
          <div style="font-weight:700">{mc.architecture}</div>
          <div style="font-size:.82rem;color:var(--text-secondary);margin-top:.3rem">
            {fw_icon} {fw} &nbsp;·&nbsp; 📥 {str(getattr(mc,'input_size','—'))}
          </div>
        </div>
        """, unsafe_allow_html=True)

        if est_str:
            st.markdown(
                f'<div style="background:rgba(108,99,255,.08);border:1px solid rgba(108,99,255,.2);'
                f'border-radius:8px;padding:.7rem 1rem;font-size:.85rem">{est_str}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("""
        <div style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);
             border-radius:10px;padding:1.2rem;text-align:center;
             color:var(--text-secondary);font-size:.9rem;margin-top:1rem">
          👈 Configure settings and click <b>🚀 Start Training</b><br>
          <span style="font-size:.78rem">Live loss &amp; accuracy curves will appear below once training starts.</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Training runs HERE — at full-page level, outside any column ───────────
    # This ensures the progress bars, live charts, and log expander are
    # rendered full-width and clearly visible to the user.
    if _pending_config is not None:
        st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,.1);margin:1rem 0'>",
                    unsafe_allow_html=True)
        start_training_with_config(_pending_config)

