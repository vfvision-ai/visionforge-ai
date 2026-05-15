"""
Settings page – VisionForge (Train Vision Models Effortlessly).
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
from ui.theme import apply_theme, hero, section, card, stat_row, info_card
from ui.helpers import get_system_info


def show_settings():
    """Display system settings."""
    apply_theme()

    hero(
        "⚙️ Settings & Tools",
        "Manage hardware, logging, model files, and export options.",
        badges=["🛠️ System", "📤 Export"],
    )

    # Tabbed layout for cleaner organisation
    tab_hw, tab_models, tab_export = st.tabs(
        ["💻 Hardware & Logging", "📦 Model Management", "📤 Export & Deploy"]
    )

    with tab_hw:
        col1, col2 = st.columns([1, 1])

        with col1:
            section("💻", "Hardware")

            # GPU selection
            gpu_id = st.selectbox("GPU Device", [0, 1, 2, 3], help="Select GPU device ID")

            # CPU workers
            num_workers = st.slider("CPU Workers", 0, 16, 4, help="Number of CPU workers for data loading")

            section("🔧", "Training Defaults")

            # Mixed precision
            use_mixed_precision = st.checkbox("Enable Mixed Precision", value=True)

            # Early stopping
            early_stopping_patience = st.slider("Early Stopping Patience", 5, 50, 10)

        with col2:
            section("📝", "Logging")

            # Log level
            log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"])

            # Log interval
            log_interval = st.slider("Log Interval", 1, 100, 10)

            section("🖥️", "System Information")

            # Display system info
            system_info = get_system_info()
            for key, value in system_info.items():
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;padding:.3rem 0;'
                    f'border-bottom:1px solid rgba(255,255,255,.05);font-size:.875rem">'
                    f'<span style="color:var(--text-secondary)">{key}</span>'
                    f'<span style="font-weight:600">{value}</span></div>',
                    unsafe_allow_html=True,
                )

    with tab_models:
        section("📦", "Model Management & Downloads")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔍 Find All Models")
        
        # Search for models across common directories
        search_dirs = [
            "./experiments",
            "./models", 
            "./checkpoints",
            "./results",
            "./outputs",
            os.path.expanduser("~/models")
        ]
        
        if st.button("🔍 Scan for Models", type="primary"):
            all_models = []
            
            for search_dir in search_dirs:
                if os.path.exists(search_dir):
                    search_path = Path(search_dir)
                    
                    # Find all model files
                    model_patterns = ['*.pt', '*.pth', '*.keras', '*.h5', '*.pkl', '*.joblib', '*.onnx']
                    
                    for pattern in model_patterns:
                        for model_file in search_path.rglob(pattern):
                            if model_file.is_file():
                                size_mb = model_file.stat().st_size / (1024 * 1024)
                                all_models.append({
                                    'file': str(model_file),
                                    'name': model_file.name,
                                    'dir': str(model_file.parent),
                                    'size_mb': size_mb,
                                    'modified': model_file.stat().st_mtime
                                })
            
            if all_models:
                # Sort by modification time (newest first)
                all_models.sort(key=lambda x: x['modified'], reverse=True)
                
                st.success(f"📦 Found {len(all_models)} model file(s)")
                
                # Display models in expandable format
                for i, model in enumerate(all_models[:10]):  # Show first 10
                    with st.expander(f"📄 {model['name']} ({model['size_mb']:.1f} MB)"):
                        st.write(f"**Directory**: `{model['dir']}`")
                        st.write(f"**Full Path**: `{model['file']}`")
                        st.write(f"**Size**: {model['size_mb']:.2f} MB")
                        
                        # Download button
                        try:
                            with open(model['file'], 'rb') as f:
                                model_data = f.read()
                            
                            st.download_button(
                                label="⬇️ Download",
                                data=model_data,
                                file_name=model['name'],
                                mime='application/octet-stream',
                                key=f"settings_download_{i}"
                            )
                        except Exception as e:
                            st.error(f"Cannot read file: {e}")
                
                if len(all_models) > 10:
                    st.info(f"... and {len(all_models) - 10} more models. Check the Results page for complete model management.")
            
            else:
                st.warning("🤔 No model files found in common directories")
                st.info("💡 Train a model first or check the Results page for specific experiment models")

        with col2:
            st.markdown("### 📊 Model Statistics")
        
        # Quick stats about available models
        if st.button("📈 Show Model Statistics"):
            stats = {
                'pytorch_models': 0,
                'tensorflow_models': 0,
                'sklearn_models': 0,
                'other_models': 0,
                'total_size_mb': 0
            }
            
            for search_dir in search_dirs:
                if os.path.exists(search_dir):
                    search_path = Path(search_dir)
                    
                    # Count different model types
                    pytorch_files = list(search_path.rglob('*.pt')) + list(search_path.rglob('*.pth'))
                    tf_files = list(search_path.rglob('*.keras')) + list(search_path.rglob('*.h5'))
                    sklearn_files = list(search_path.rglob('*.pkl')) + list(search_path.rglob('*.joblib'))
                    other_files = list(search_path.rglob('*.onnx'))
                    
                    stats['pytorch_models'] += len(pytorch_files)
                    stats['tensorflow_models'] += len(tf_files)
                    stats['sklearn_models'] += len(sklearn_files)
                    stats['other_models'] += len(other_files)
                    
                    # Calculate total size
                    all_files = pytorch_files + tf_files + sklearn_files + other_files
                    for file_path in all_files:
                        if file_path.is_file():
                            stats['total_size_mb'] += file_path.stat().st_size / (1024 * 1024)
            
            # Display statistics
            stat_col1, stat_col2 = st.columns(2)
            
            with stat_col1:
                st.metric("🔥 PyTorch Models", stats['pytorch_models'])
                st.metric("🧠 TensorFlow Models", stats['tensorflow_models'])
            
            with stat_col2:
                st.metric("📊 Scikit-learn Models", stats['sklearn_models'])
                st.metric("🔧 Other Models", stats['other_models'])
            
            st.metric("💾 Total Storage", f"{stats['total_size_mb']:.1f} MB")
            
            total_models = sum([stats['pytorch_models'], stats['tensorflow_models'], 
                              stats['sklearn_models'], stats['other_models']])
            
            if total_models > 0:
                st.success(f"📊 **Summary**: {total_models} models using {stats['total_size_mb']:.1f} MB")
            else:
                st.info("ℹ️ No models found. Train some models to see statistics!")
        
        st.markdown("### 🗂️ Quick Actions")
        
        # Link to results page
        if st.button("📈 Go to Results & Downloads", use_container_width=True):
            st.session_state.current_step = 4  # Results page
            st_rerun()
        
        # Clear cache button
        if st.button("🧹 Clear Model Cache", help="Clear any cached model data"):
            # Clear any model-related session state
            cache_keys = ['model_config', 'training_results', 'cached_models']
            for key in cache_keys:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("✅ Model cache cleared!")

    with tab_export:
        section("📤", "Model Export & Deployment")
        st.markdown("Export trained models to portable formats for deployment.")

        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            st.markdown("#### 🔍 Select Model to Export")

            # Auto-find latest model
        def _find_model_files():
            search_dirs = ["experiments", "models", "checkpoints", "results"]
            found = []
            for d in search_dirs:
                p = Path(d)
                if p.exists():
                    for ext in ("*.pt", "*.pth"):
                        found.extend(p.rglob(ext))
            return sorted(found, key=lambda f: f.stat().st_mtime, reverse=True)

        pt_models = _find_model_files()

        if pt_models:
            model_options = {str(m): m for m in pt_models[:20]}
            selected_model_str = st.selectbox(
                "Available PyTorch models",
                list(model_options.keys()),
                format_func=lambda x: Path(x).name,
            )
            export_model_path = model_options[selected_model_str]
        else:
            st.info("💡 No PyTorch models found. Train a model first.")
            export_model_path = None
            # Allow manual entry
            manual_path = st.text_input("Or enter model path manually")
            if manual_path and Path(manual_path).exists():
                export_model_path = Path(manual_path)

        export_format = st.selectbox(
            "Export Format",
            ["ONNX", "TorchScript (CPU)", "TorchScript (GPU)"],
            help="ONNX is broadly supported; TorchScript is PyTorch-native.",
        )

        export_input_size = st.number_input("Input image size", min_value=16, max_value=1024, value=224, step=8)
        export_num_classes = st.number_input(
            "Number of classes",
            min_value=2, max_value=10000,
            value=int(getattr(getattr(st.session_state.get("dataset_info"), "num_classes", None) or 10, "__int__", lambda: 10)()),
        )

        with exp_col2:
            st.markdown("#### ⚡ Export")

        if st.button("🚀 Export Model", type="primary", use_container_width=True,
                     disabled=(export_model_path is None)):
            if export_model_path and export_model_path.exists():
                with st.spinner(f"Exporting to {export_format}…"):
                    try:
                        import torch
                        device = "cuda" if torch.cuda.is_available() else "cpu"

                        # Load model
                        checkpoint = torch.load(export_model_path, map_location=device)
                        arch = "resnet18"
                        if isinstance(checkpoint, dict):
                            arch = checkpoint.get("architecture",
                                   checkpoint.get("model_name", "resnet18"))
                            state = checkpoint.get("model_state_dict",
                                    checkpoint.get("state_dict", checkpoint))
                        else:
                            state = None

                        import torchvision.models as tvm
                        import torch.nn as nn

                        constructor = getattr(tvm, arch.lower(), tvm.resnet18)
                        model = constructor(weights=None)
                        if hasattr(model, "fc"):
                            model.fc = nn.Linear(model.fc.in_features, int(export_num_classes))
                        elif hasattr(model, "classifier"):
                            last = model.classifier[-1]
                            model.classifier[-1] = nn.Linear(last.in_features, int(export_num_classes))

                        if state:
                            model.load_state_dict(state, strict=False)

                        model.eval().to(device)
                        dummy = torch.randn(1, 3, int(export_input_size), int(export_input_size)).to(device)

                        export_dir = Path("exports")
                        export_dir.mkdir(exist_ok=True)
                        stem = export_model_path.stem

                        if export_format == "ONNX":
                            out_path = export_dir / f"{stem}.onnx"
                            torch.onnx.export(
                                model, dummy, str(out_path),
                                input_names=["input"],
                                output_names=["output"],
                                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                                opset_version=17,
                            )
                        else:  # TorchScript
                            out_path = export_dir / f"{stem}_torchscript.pt"
                            scripted = torch.jit.trace(model, dummy)
                            scripted.save(str(out_path))

                        size_mb = out_path.stat().st_size / 1e6
                        st.success(f"✅ Exported to `{out_path}` ({size_mb:.2f} MB)")

                        # Download button
                        with open(out_path, "rb") as fh:
                            st.download_button(
                                "⬇️ Download Exported Model",
                                data=fh.read(),
                                file_name=out_path.name,
                                mime="application/octet-stream",
                            )

                    except ImportError as ie:
                        st.error(f"Missing dependency: {ie}")
                    except Exception as e:
                        st.error(f"Export failed: {e}")
                        logger.exception("Model export error")
            else:
                st.error("❌ Model file not found.")

        st.markdown("---")
        st.markdown("##### Export Format Notes")
        st.markdown("""
| Format | Use Case | Runtime |
|--------|----------|---------|
| **ONNX** | Broad compatibility, C++/Java/JS | ONNX Runtime |
| **TorchScript** | PyTorch-native deployment | `torch.jit.load()` |
""")
        with st.expander("📖 How to use exported models"):
            st.code("""
# ONNX Runtime inference
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession("model.onnx")
result = sess.run(None, {"input": np.random.randn(1,3,224,224).astype("float32")})

# TorchScript inference
import torch
model = torch.jit.load("model_torchscript.pt")
model.eval()
with torch.no_grad():
    out = model(torch.randn(1, 3, 224, 224))
""", language="python")


