#!/usr/bin/env python3
"""
Computer Vision Model Training Pipeline — Web Interface
Entry point: delegates to ui/ page modules.
"""

import streamlit as st

# MUST be the first Streamlit command
try:
    st.set_page_config(
        page_title="VisionForge",
        page_icon="🔭",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": "**VisionForge** · v2.0 · Train Vision Models Effortlessly",
        },
    )
except st.errors.StreamlitAPIException:
    pass  # already set (e.g. during hot-reload)

import sys
import os
import logging
import atexit
import multiprocessing as mp
from pathlib import Path

# ── project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── platform helpers (Windows DLL paths, etc.) ───────────────────────────────
try:
    from utils.platform_utils import setup_dll_directories
    setup_dll_directories()
except Exception:
    pass

# ── logging ───────────────────────────────────────────────────────────────────
try:
    from utils.logging_config import setup_logging
    setup_logging()
except Exception:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ── suppress noisy 3rd-party warnings ────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", message=".*torch.classes.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Tried to instantiate class.*", category=UserWarning)

# ── multiprocessing (spawn is safer than fork on all platforms) ───────────────
try:
    mp.set_start_method("spawn", force=False)
except RuntimeError:
    pass

# ── resource cleanup on exit ─────────────────────────────────────────────────
def _cleanup():
    import gc
    gc.collect()
    try:
        from datasets import disable_caching
        disable_caching()
    except Exception:
        pass

atexit.register(_cleanup)

# ── core modules ──────────────────────────────────────────────────────────────
try:
    from core.dataset_analyzer import DatasetAnalyzer
    from core.model_selector import ModelSelector
    from core.trainer import AutoTrainer
    from core.optimizer import HyperparameterOptimizer
    from utils.config import Config
    _CORE_LOADED = True
except Exception as exc:
    st.error(f"⚠️ Core module import error: {exc}")
    _CORE_LOADED = False

# ── page imports ──────────────────────────────────────────────────────────────
try:
    from ui.home import show_home
    from ui.dataset import show_dataset_analysis
    from ui.models import show_model_selection
    from ui.training import show_training
    from ui.results import show_results
    from ui.settings import show_settings
    from ui.inference import show_inference
    from ui.compare import show_compare
    from ui.theme import apply_theme
    _UI_LOADED = True
except Exception as exc:
    st.error(f"⚠️ UI module import error: {exc}")
    _UI_LOADED = False

# ── global theme (injected immediately on load) ───────────────────────────────
try:
    from ui.theme import apply_theme as _apply_theme_early
    _apply_theme_early()
except Exception:
    pass


# ── compatibility shim ────────────────────────────────────────────────────────
def st_rerun():
    """Smooth st.rerun() shim across Streamlit versions."""
    try:
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
    except Exception:
        pass


# ── session state defaults ────────────────────────────────────────────────────
_SESSION_DEFAULTS = {
    "current_path": os.path.expanduser("~"),
    "browse_mode": True,
    "selected_files": [],
    "dataset_info": None,
    "uploaded_dataset_path": None,
    "current_step": 0,
    "current_tool": None,          # None = workflow; "settings"/"inference"/"compare" = tool override
    "project_started": False,
    "model_config": None,
    "training_completed": False,
    "training_status": None,
    "training_results": None,
    "training_log_history": [],
    "early_stopping_config": {},
    "selected_framework": "PyTorch",
    "manual_config_in_progress": False,
    "selected_task_type": None,
    "task_type_confirmed": False,
}

def _init_session_state():
    for key, value in _SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ── navigation helpers ────────────────────────────────────────────────────────
_WORKFLOW = [
    ("🏠 Home",              "Get Started"),
    ("📊 Dataset Analysis",  "Analyze Your Data"),
    ("🧠 Model Selection",   "Choose Best Model"),
    ("🔥 Training",          "Train Your Model"),
    ("📈 Results",           "View Results"),
]
_STEP_ICONS = ["🏠", "📊", "🧠", "🔥", "📈"]
_STEP_NAMES = ["Home", "Dataset", "Model", "Training", "Results"]


def _max_unlocked_step() -> int:
    """Return the highest workflow step the user has unlocked.
    Only used to *clamp forward* jumps — does NOT force the step upward.
    """
    s = st.session_state
    if not s.get("project_started"):
        return 0
    if not s.get("dataset_info"):
        return 1
    if not s.get("model_config") or s.get("manual_config_in_progress", False):
        return 2
    if not s.get("training_completed", False):
        return 3
    return 4


def _render_sidebar():
    """Render sidebar. Navigation is driven by current_step + current_tool in session state."""
    step         = st.session_state.current_step
    max_step     = _max_unlocked_step()
    current_tool = st.session_state.get("current_tool")
    fw           = st.session_state.get("selected_framework", "PyTorch")

    # ── Brand ─────────────────────────────────────────────────────────────────
    st.sidebar.markdown("""
    <div style="padding:1.2rem 0.8rem 0.8rem;text-align:center">
      <div style="font-size:2rem;margin-bottom:.2rem">🔭</div>
      <div style="font-weight:800;font-size:1.12rem;background:linear-gradient(135deg,#6c63ff,#00d4aa);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">
        VisionForge
      </div>
      <div style="font-size:0.69rem;color:#505068;margin-top:.15rem">v2.0 · Train Vision Models Effortlessly</div>
    </div>
    <hr style="border:none;border-top:1px solid rgba(255,255,255,.06);margin:.4rem 0 .6rem">
    """, unsafe_allow_html=True)

    # ── Workflow step progress ─────────────────────────────────────────────────
    st.sidebar.markdown(
        '<p style="font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;'
        'color:#505068;font-weight:700;padding:0 .8rem;margin:0 0 .25rem">Workflow</p>',
        unsafe_allow_html=True,
    )

    for i in range(5):
        is_active   = (i == step) and (current_tool is None)
        is_done     = i < step
        is_unlocked = i <= max_step

        if is_active:
            fg = "#a89cff"; bg = "rgba(108,99,255,.14)"; brd = "3px solid #6c63ff"; fw_w = "700"
            indicator = _STEP_ICONS[i]
        elif is_done:
            fg = "#00c49a"; bg = "transparent"; brd = "3px solid transparent"; fw_w = "500"
            indicator = "✓"
        elif is_unlocked:
            fg = "#b0b0c8"; bg = "transparent"; brd = "3px solid transparent"; fw_w = "500"
            indicator = _STEP_ICONS[i]
        else:
            fg = "#3a3a52"; bg = "transparent"; brd = "3px solid transparent"; fw_w = "400"
            indicator = _STEP_ICONS[i]

        st.sidebar.markdown(
            f'<div style="display:flex;align-items:center;gap:.55rem;padding:.38rem .8rem;'
            f'background:{bg};border-left:{brd};border-radius:0 8px 8px 0;margin-bottom:.05rem">'
            f'<span style="font-size:.88rem;min-width:1.1rem;text-align:center">{indicator}</span>'
            f'<span style="font-size:.86rem;font-weight:{fw_w};color:{fg}">{_STEP_NAMES[i]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Clickable button for every unlocked step that is not the current one
        if is_unlocked and not is_active:
            if st.sidebar.button(
                f"OPEN {_STEP_NAMES[i].upper()}",
                key=f"_nav_step_{i}",
                use_container_width=True,
                help=f"Navigate to {_STEP_NAMES[i]}",
            ):
                st.session_state.current_step = i
                st.session_state.current_tool = None
                st_rerun()

    # ── Framework badge ────────────────────────────────────────────────────────
    st.sidebar.markdown(
        '<hr style="border:none;border-top:1px solid rgba(255,255,255,.05);margin:.6rem 0 .45rem">',
        unsafe_allow_html=True,
    )
    fw_icons = {"PyTorch": "🔥", "TensorFlow/Keras": "🧠", "Scikit-learn": "📊"}
    st.sidebar.markdown(
        f'<div style="display:flex;align-items:center;gap:.5rem;padding:.32rem .8rem;'
        f'background:rgba(0,212,170,.06);border:1px solid rgba(0,212,170,.14);'
        f'border-radius:8px;margin:0 .4rem .45rem">'
        f'<span>{fw_icons.get(fw,"🛠️")}</span>'
        f'<span style="font-size:.79rem;font-weight:600;color:#00b88e">{fw}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Tools ─────────────────────────────────────────────────────────────────
    st.sidebar.markdown(
        '<p style="font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;'
        'color:#505068;font-weight:700;padding:0 .8rem;margin:.1rem 0 .25rem">Tools</p>',
        unsafe_allow_html=True,
    )
    _TOOLS = [
        ("⚙️ Settings",          "settings"),
        ("🔍 Run Inference",      "inference"),
        ("📊 Compare Experiments","compare"),
    ]
    for tlabel, tkey in _TOOLS:
        is_tool_active = (current_tool == tkey)
        fg  = "#a89cff" if is_tool_active else "#b0b0c8"
        bg  = "rgba(108,99,255,.14)" if is_tool_active else "transparent"
        brd = "3px solid #6c63ff"   if is_tool_active else "3px solid transparent"
        st.sidebar.markdown(
            f'<div style="display:flex;align-items:center;gap:.55rem;padding:.36rem .8rem;'
            f'background:{bg};border-left:{brd};border-radius:0 8px 8px 0;margin-bottom:.04rem">'
            f'<span style="font-size:.85rem;color:{fg}">{tlabel}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if not is_tool_active:
            if st.sidebar.button(
                f"OPEN {tlabel.split(' ', 1)[-1].upper()}",
                key=f"_nav_tool_{tkey}",
                use_container_width=True,
            ):
                st.session_state.current_tool = tkey
                st_rerun()


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    _init_session_state()

    # Clamp forward only: prevent skipping past unlocked steps.
    # NEVER force-advance — that breaks back-navigation.
    max_step = _max_unlocked_step()
    if st.session_state.current_step > max_step:
        st.session_state.current_step = max_step

    _render_sidebar()

    if not _UI_LOADED:
        st.error("⚠️ UI modules could not be loaded. Check terminal logs for import errors.")
        return

    tool = st.session_state.get("current_tool")
    if tool:
        _TOOL_DISPATCH = {
            "settings":  show_settings,
            "inference": show_inference,
            "compare":   show_compare,
        }
        fn = _TOOL_DISPATCH.get(tool)
        if fn:
            fn()
        else:
            # Unknown tool — clear and show home
            st.session_state.current_tool = None
            st_rerun()
    else:
        _STEP_DISPATCH = [
            show_home,
            show_dataset_analysis,
            show_model_selection,
            show_training,
            show_results,
        ]
        _STEP_DISPATCH[min(st.session_state.current_step, 4)]()


if __name__ == "__main__":
    main()
