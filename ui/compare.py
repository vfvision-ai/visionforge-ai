"""
VisionForge — Experiment Comparison page. Compare training runs side by side.
"""

import streamlit as st
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))
from ui.shared import st_rerun
from ui.theme import apply_theme, hero, section, card, stat_row

logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────

_SEARCH_DIRS = ["experiments", "models", "checkpoints", "results", "outputs"]


def _collect_result_files() -> List[Path]:
    """Return all *_results.json files sorted by modification time."""
    found = []
    for d in _SEARCH_DIRS:
        p = Path(d)
        if p.exists():
            found.extend(p.rglob("*_results.json"))
            found.extend(p.rglob("results.json"))
    return sorted(set(found), key=lambda f: f.stat().st_mtime, reverse=True)


def _load_result(path: Path) -> Optional[Dict]:
    try:
        with open(path) as fh:
            data = json.load(fh)
        data["_file"] = str(path)
        data["_name"] = path.stem.replace("_results", "")
        data["_mtime"] = path.stat().st_mtime
        return data
    except Exception as e:
        logger.warning("Cannot read %s: %s", path, e)
        return None


def _normalise(result: Dict) -> Dict:
    """Extract comparable metrics from a results dict, handling different schemas."""
    acc = result.get("best_accuracy", result.get("accuracy", result.get("val_accuracy", None)))
    if acc is not None and acc <= 1.0:
        acc = acc * 100.0

    loss = result.get("best_loss", result.get("loss", result.get("val_loss", None)))

    precision = result.get("precision", result.get("best_precision", None))
    recall    = result.get("recall",    result.get("best_recall",    None))
    f1        = result.get("f1",        result.get("f1_score",       result.get("best_f1", None)))

    if precision is not None and precision <= 1.0:
        precision = precision * 100
    if recall is not None and recall <= 1.0:
        recall = recall * 100
    if f1 is not None and f1 <= 1.0:
        f1 = f1 * 100

    epochs_trained = result.get("epochs_trained", result.get("epoch", result.get("total_epochs", None)))
    training_time  = result.get("training_time", result.get("total_time", None))
    framework      = result.get("framework", "Unknown")
    architecture   = result.get("architecture", result.get("model_name", result.get("model", "Unknown")))
    dataset        = result.get("dataset_name", result.get("dataset", "Unknown"))
    num_params     = result.get("num_params", result.get("parameters", None))

    return {
        "Name":          result.get("_name", "Unknown"),
        "Framework":     framework,
        "Architecture":  architecture,
        "Dataset":       dataset,
        "Accuracy (%)":  round(acc, 2) if acc is not None else None,
        "Loss":          round(float(loss), 4) if loss is not None else None,
        "Precision (%)": round(precision, 2) if precision is not None else None,
        "Recall (%)":    round(recall, 2) if recall is not None else None,
        "F1 (%)":        round(f1, 2) if f1 is not None else None,
        "Epochs":        epochs_trained,
        "Train Time (s)":round(float(training_time), 1) if training_time is not None else None,
        "Params":        num_params,
        "_file":         result["_file"],
        "_raw":          result,
    }


def _epoch_curves(result: Dict) -> Optional[pd.DataFrame]:
    """Extract per-epoch metrics into a DataFrame if available."""
    history = (
        result.get("history") or
        result.get("training_history") or
        result.get("log_history") or
        result.get("epoch_history")
    )
    if not history:
        return None

    if isinstance(history, dict):
        # Keras-style: {"val_accuracy": [...], "loss": [...], ...}
        max_len = max(len(v) for v in history.values() if isinstance(v, list))
        df = pd.DataFrame({k: v for k, v in history.items() if isinstance(v, list) and len(v) == max_len})
        df.index.name = "Epoch"
        df = df.reset_index()
    elif isinstance(history, list) and history and isinstance(history[0], dict):
        # List of epoch dicts
        df = pd.DataFrame(history)
        if "epoch" not in df.columns:
            df.insert(0, "epoch", range(len(df)))
    else:
        return None

    return df


# ── main page ─────────────────────────────────────────────────────────────────

def show_compare():
    """Experiment Comparison page."""
    apply_theme()

    hero(
        "📊 Experiment Comparison",
        "Compare multiple training runs side by side — metrics, training curves, and raw details.",
        badges=["� VisionForge", "�📈 Analytics", "Multi-run"],
    )

    # ── Discover experiments ──────────────────────────────────────────────────
    col_scan, col_manual = st.columns([1, 1])

    with col_scan:
        if st.button("🔍 Scan for Experiments", type="primary", use_container_width=True):
            st.session_state["_compare_files"] = _collect_result_files()

    with col_manual:
        uploaded = st.file_uploader(
            "Or upload results JSON file(s)",
            type=["json"],
            accept_multiple_files=True,
        )
        if uploaded:
            tmp_dir = Path("uploads") / "compare"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            for uf in uploaded:
                dest = tmp_dir / uf.name
                dest.write_bytes(uf.read())
            all_files = list(tmp_dir.glob("*.json"))
            existing = st.session_state.get("_compare_files", [])
            merged = list({str(f): f for f in existing + all_files}.values())
            st.session_state["_compare_files"] = merged

    result_files: List[Path] = st.session_state.get("_compare_files", [])

    if not result_files:
        st.info(
            "Click **Scan for Experiments** to auto-discover `*_results.json` files "
            "in your experiments/models directories, or upload them manually above."
        )
        # Show hint about current session
        if st.session_state.get("training_results"):
            st.success("✅ You have a current session – train more experiments and come back to compare them!")
        return

    # ── Load & normalise ──────────────────────────────────────────────────────
    raw_results = [_load_result(f) for f in result_files]
    raw_results = [r for r in raw_results if r is not None]

    if not raw_results:
        st.warning("No readable result files found.")
        return

    normed = [_normalise(r) for r in raw_results]

    # ── Selection ─────────────────────────────────────────────────────────────
    section("📋", "1 — Select Experiments")

    names = [n["Name"] for n in normed]
    selected_names = st.multiselect(
        "Choose experiments",
        options=names,
        default=names[:min(6, len(names))],
        help="Select 2 or more experiments to compare",
    )

    if len(selected_names) < 1:
        st.info("Select at least one experiment above.")
        return

    selected = [n for n in normed if n["Name"] in selected_names]
    df_display = pd.DataFrame([
        {k: v for k, v in s.items() if not k.startswith("_")}
        for s in selected
    ])

    # ── Summary table ─────────────────────────────────────────────────────────
    section("📊", "2 — Summary Table")

    # Highlight best values
    numeric_cols = ["Accuracy (%)", "Loss", "Precision (%)", "Recall (%)", "F1 (%)", "Train Time (s)"]
    present_numeric = [c for c in numeric_cols if c in df_display.columns and df_display[c].notna().any()]

    def _highlight_best(s):
        styles = [""] * len(s)
        try:
            vals = pd.to_numeric(s, errors="coerce")
            if s.name == "Loss" or s.name == "Train Time (s)":
                best_idx = vals.idxmin()
            else:
                best_idx = vals.idxmax()
            styles[df_display.index.get_loc(best_idx)] = "background-color: #d4edda; font-weight: bold"
        except Exception:
            pass
        return styles

    styled_df = df_display[
        [c for c in ["Name", "Framework", "Architecture", "Dataset",
                      "Accuracy (%)", "Loss", "F1 (%)", "Precision (%)", "Recall (%)",
                      "Epochs", "Train Time (s)", "Params"]
         if c in df_display.columns]
    ]
    try:
        st.dataframe(styled_df.style.apply(_highlight_best, subset=present_numeric),
                     use_container_width=True)
    except Exception:
        st.dataframe(styled_df, use_container_width=True)

    # CSV download
    st.download_button(
        "⬇️ Download Comparison CSV",
        data=styled_df.to_csv(index=False).encode(),
        file_name="experiment_comparison.csv",
        mime="text/csv",
    )

    # ── Bar chart – key metrics ────────────────────────────────────────────────
    section("📈", "3 — Metric Charts")

    chart_metric = st.selectbox(
        "Primary metric to compare",
        [m for m in ["Accuracy (%)", "F1 (%)", "Precision (%)", "Recall (%)", "Loss", "Train Time (s)"]
         if m in df_display.columns and df_display[m].notna().any()],
    )

    if chart_metric in df_display.columns:
        fig_bar = px.bar(
            df_display.dropna(subset=[chart_metric]),
            x="Name", y=chart_metric,
            color="Framework",
            text=df_display[chart_metric].apply(
                lambda v: f"{v:.2f}" if pd.notna(v) else ""),
            title=f"{chart_metric} by Experiment",
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Multi-metric radar chart
    radar_metrics = [m for m in ["Accuracy (%)", "F1 (%)", "Precision (%)", "Recall (%)"]
                     if m in df_display.columns and df_display[m].notna().any()]
    if len(radar_metrics) >= 3 and len(selected) >= 1:
        st.markdown("##### 🕸️ Multi-Metric Radar")
        fig_radar = go.Figure()
        for _, row in df_display.iterrows():
            values = [float(row.get(m, 0) or 0) for m in radar_metrics]
            values += [values[0]]
            cats = radar_metrics + [radar_metrics[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=values, theta=cats, fill="toself", name=row["Name"],
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True, height=400,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Training curves ───────────────────────────────────────────────────────
    section("📉", "4 — Training Curves")

    # Find experiments that have curve data
    curve_data = {}
    for s in selected:
        raw = s["_raw"]
        df_curve = _epoch_curves(raw)
        if df_curve is not None:
            curve_data[s["Name"]] = df_curve

    if not curve_data:
        st.info("No per-epoch history found in selected experiments.")
    else:
        curve_metric = st.selectbox(
            "Metric to plot",
            ["val_accuracy", "accuracy", "val_loss", "loss",
             "train_accuracy", "train_loss"],
        )

        fig_curve = go.Figure()
        for exp_name, df_curve in curve_data.items():
            # Try exact column then partial match
            col_match = None
            if curve_metric in df_curve.columns:
                col_match = curve_metric
            else:
                for c in df_curve.columns:
                    if curve_metric.split("_")[-1] in c.lower():
                        col_match = c
                        break

            if col_match:
                epoch_col = "epoch" if "epoch" in df_curve.columns else df_curve.columns[0]
                fig_curve.add_trace(go.Scatter(
                    x=df_curve[epoch_col],
                    y=df_curve[col_match],
                    mode="lines+markers",
                    name=f"{exp_name} – {col_match}",
                ))

        if fig_curve.data:
            fig_curve.update_layout(
                title=f"{curve_metric} over Epochs",
                xaxis_title="Epoch",
                yaxis_title=curve_metric,
                height=400,
            )
            st.plotly_chart(fig_curve, use_container_width=True)
        else:
            st.info(f"Column `{curve_metric}` not found in selected experiments.")

    # ── Raw JSON viewer ───────────────────────────────────────────────────────
    section("📄", "5 — Raw Details")
    for s in selected:
        with st.expander(f"📄 {s['Name']}", expanded=False):
            st.json({k: v for k, v in s["_raw"].items() if not k.startswith("_")})
            st.caption(f"File: `{s['_file']}`")

