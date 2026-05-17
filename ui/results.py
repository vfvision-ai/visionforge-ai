"""
Results page – VisionForge (Train Vision Models Effortlessly).
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
from ui.theme import apply_theme, hero, section, card, stat_row, metric_card, step_tracker


def show_results():
    """Display training results and analysis."""
    apply_theme()

    hero(
        "📈 Training Results",
        "Explore model performance, download trained weights, and generate test samples.",
        badges=["Step 5 of 5", "🏆 Final"],
    )
    step_tracker(["Home","Dataset","Model","Training","Results"],
                 st.session_state.get("current_step", 4))

    # Check for recent training completion in session state
    if st.session_state.get('training_completed') and st.session_state.get('training_results'):
        st.markdown("""
        <div style="background:rgba(0,212,170,.12);border:1px solid rgba(0,212,170,.35);
             border-radius:12px;padding:1rem 1.4rem;margin-bottom:1.2rem;display:flex;align-items:center;gap:.8rem">
          <span style="font-size:1.5rem">🎉</span>
          <div>
            <div style="font-weight:700;font-size:1rem">Training completed successfully!</div>
            <div style="font-size:.85rem;color:#a0a0b8;margin-top:.15rem">Latest results shown below — scroll down to download your model.</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Display session state results immediately
        results = st.session_state.training_results

        accuracy = results.get('best_accuracy', 0)
        if accuracy <= 1:
            accuracy *= 100
        loss      = results.get('best_loss', 0)
        train_t   = results.get('training_time', 0)
        framework = results.get('framework', 'Unknown')

        # Detect task type for right metric label
        task_type = 'classification'
        if st.session_state.get('dataset_info'):
            task_type = getattr(st.session_state.dataset_info, 'task_type', 'classification')

        is_segmentation = task_type == 'segmentation'
        is_detection    = task_type == 'detection'
        if is_detection:
            acc_label = 'Best mAP@50'
            acc_icon  = '🎯'
        elif is_segmentation:
            acc_label = 'Best mIoU'
            acc_icon  = '🗺️'
        else:
            acc_label = 'Final Accuracy'
            acc_icon  = '🎯'

        # Also pull mIoU/Dice / mAP@50 if present in results
        miou_val  = results.get('best_miou', results.get('val_miou', None))
        dice_val  = results.get('best_dice', results.get('val_dice', None))
        map50_val = results.get('best_map50', results.get('val_map50', None))

        # Beautiful metric row
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            metric_card(acc_label, f"{accuracy:.2f}%", icon=acc_icon, color="purple")
        with mc2:
            if is_segmentation and dice_val is not None:
                d = dice_val * 100 if dice_val <= 1 else dice_val
                metric_card("Best Dice", f"{d:.2f}%", icon="🎲", color="teal")
            elif is_detection and map50_val is not None:
                p = map50_val * 100 if map50_val <= 1 else map50_val
                metric_card("mAP@50", f"{p:.2f}%", icon="📦", color="teal")
            else:
                metric_card("Final Loss", f"{loss:.4f}", icon="📉", color="teal")
        with mc3:
            metric_card("Training Time", f"{train_t:.1f}s", icon="⏱️", color="yellow")
        with mc4:
            metric_card("Framework", framework, icon="🛠️", color="purple")

        # ── Training Curves ────────────────────────────────────────────────
        log_hist = st.session_state.get("training_log_history", [])
        # Also try training_history inside results dict
        if not log_hist and "training_history" in results:
            log_hist = results["training_history"]

        if log_hist:
            st.markdown("<br>", unsafe_allow_html=True)
            section("📊", "Training Curves")
            try:
                df_log = pd.DataFrame(log_hist)
                has_epoch = "epoch" in df_log.columns
                x_col = "epoch" if has_epoch else df_log.index

                curve_c1, curve_c2 = st.columns(2)
                with curve_c1:
                    fig_loss = go.Figure()
                    if "loss" in df_log.columns:
                        fig_loss.add_trace(go.Scatter(
                            x=df_log[x_col] if has_epoch else df_log.index,
                            y=df_log["loss"], mode="lines+markers",
                            name="Train Loss", line=dict(color="#6c63ff", width=2),
                            marker=dict(size=4),
                        ))
                    if "val_loss" in df_log.columns:
                        fig_loss.add_trace(go.Scatter(
                            x=df_log[x_col] if has_epoch else df_log.index,
                            y=df_log["val_loss"], mode="lines+markers",
                            name="Val Loss", line=dict(color="#ff6b6b", width=2, dash="dash"),
                            marker=dict(size=4),
                        ))
                    fig_loss.update_layout(
                        title="Loss over Epochs", template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        height=280, margin=dict(l=0, r=0, t=40, b=0),
                        legend=dict(orientation="h", y=-0.2),
                        xaxis=dict(title="Epoch", gridcolor="rgba(255,255,255,.06)"),
                        yaxis=dict(title="Loss", gridcolor="rgba(255,255,255,.06)"),
                    )
                    st.plotly_chart(fig_loss, use_container_width=True, config={"displayModeBar": False})

                with curve_c2:
                    fig_acc = go.Figure()
                    # Segmentation: plot mIoU / Dice curves
                    if is_segmentation:
                        for col_name, label, color in [
                            ("train_miou", "Train mIoU", "#00d4aa"),
                            ("val_miou",   "Val mIoU",   "#ffd93d"),
                            ("train_dice", "Train Dice", "#6c63ff"),
                            ("val_dice",   "Val Dice",   "#ff6b6b"),
                        ]:
                            if col_name in df_log.columns:
                                vals = df_log[col_name]
                                if vals.max() <= 1.0:
                                    vals = vals * 100
                                fig_acc.add_trace(go.Scatter(
                                    x=df_log[x_col] if has_epoch else df_log.index,
                                    y=vals, mode="lines+markers",
                                    name=label,
                                    line=dict(color=color, width=2,
                                              dash="dash" if "val" in col_name else "solid"),
                                    marker=dict(size=4),
                                ))
                        fig_acc.update_layout(
                            title="mIoU & Dice over Epochs", template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            height=280, margin=dict(l=0, r=0, t=40, b=0),
                            legend=dict(orientation="h", y=-0.2),
                            xaxis=dict(title="Epoch", gridcolor="rgba(255,255,255,.06)"),
                            yaxis=dict(title="Score (%)", gridcolor="rgba(255,255,255,.06)"),
                        )
                    else:
                        if is_detection:
                            # Detection: plot mAP@50 curve
                            for col_name, label, color in [
                                ("val_map50", "Val mAP@50", "#ffd93d"),
                            ]:
                                if col_name in df_log.columns:
                                    vals = df_log[col_name]
                                    if vals.max() <= 1.0:
                                        vals = vals * 100
                                    fig_acc.add_trace(go.Scatter(
                                        x=df_log[x_col] if has_epoch else df_log.index,
                                        y=vals, mode="lines+markers",
                                        name=label, line=dict(color=color, width=2),
                                        marker=dict(size=4),
                                    ))
                            fig_acc.update_layout(
                                title="mAP@50 over Epochs", template="plotly_dark",
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                height=280, margin=dict(l=0, r=0, t=40, b=0),
                                legend=dict(orientation="h", y=-0.2),
                                xaxis=dict(title="Epoch", gridcolor="rgba(255,255,255,.06)"),
                                yaxis=dict(title="mAP@50 (%)", gridcolor="rgba(255,255,255,.06)"),
                            )
                        else:
                            for col_name, label, color in [
                                ("accuracy", "Train Acc", "#00d4aa"),
                                ("val_accuracy", "Val Acc", "#ffd93d"),
                                ("acc", "Train Acc", "#00d4aa"),
                                ("val_acc", "Val Acc", "#ffd93d"),
                            ]:
                                if col_name in df_log.columns:
                                    vals = df_log[col_name]
                                    if vals.max() <= 1.0:
                                        vals = vals * 100
                                    fig_acc.add_trace(go.Scatter(
                                        x=df_log[x_col] if has_epoch else df_log.index,
                                        y=vals, mode="lines+markers",
                                        name=label, line=dict(color=color, width=2,
                                        dash="dash" if "val" in col_name else "solid"),
                                        marker=dict(size=4),
                                    ))
                            fig_acc.update_layout(
                                title="Accuracy over Epochs", template="plotly_dark",
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                height=280, margin=dict(l=0, r=0, t=40, b=0),
                                legend=dict(orientation="h", y=-0.2),
                                xaxis=dict(title="Epoch", gridcolor="rgba(255,255,255,.06)"),
                                yaxis=dict(title="Accuracy (%)", gridcolor="rgba(255,255,255,.06)"),
                            )
                    st.plotly_chart(fig_acc, use_container_width=True, config={"displayModeBar": False})

                # Download CSV of training history
                csv_data = df_log.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Training History (CSV)",
                    data=csv_data,
                    file_name="training_history.csv",
                    mime="text/csv",
                )
            except Exception as _e:
                logger.warning("Could not render training curves: %s", _e)

        # Quick model download for recently trained model
        if 'model_path' in results and results['model_path']:
            st.markdown("<br>", unsafe_allow_html=True)
            section("⬇️", "Quick Download – Just Trained Model")

            model_path = Path(results['model_path'])
            if model_path.exists():
                file_size_mb = model_path.stat().st_size / (1024 * 1024)

                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"""
                    <div class="cv-card" style="border-left:3px solid var(--accent)">
                      <div style="font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;color:var(--text-secondary);margin-bottom:.5rem">Model File</div>
                      <div style="font-weight:700;margin-bottom:.3rem">📄 {model_path.name}</div>
                      <div style="font-size:.85rem;color:var(--text-secondary)">📊 {file_size_mb:.2f} MB &nbsp;·&nbsp; 🎯 {framework}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.code(str(model_path), language="text")

                with col3:
                    try:
                        with open(model_path, 'rb') as f:
                            model_data = f.read()

                        st.download_button(
                            label="⬇️ Download Model",
                            data=model_data,
                            file_name=model_path.name,
                            mime='application/octet-stream',
                            help=f"Download the just-trained {framework} model"
                        )
                    except Exception as e:
                        st.error(f"❌ Cannot read model file: {e}")
            else:
                st.warning(f"⚠️ Model file not found: `{results['model_path']}`")
            
        # Test samples generation and download section
        st.markdown("---")
        st.subheader("🔬 Test Samples for Evaluation")
            
        # Check if test samples already exist
        test_samples_dir = Path("test_samples")
        existing_samples = list(test_samples_dir.glob("*.png")) + list(test_samples_dir.glob("*.jpg")) if test_samples_dir.exists() else []
        labels_csv = test_samples_dir / "labels.csv" if test_samples_dir.exists() else None
            
        if existing_samples and labels_csv and labels_csv.exists():
            st.success(f"✅ Found {len(existing_samples)} existing test samples!")
                
            # Display existing samples info
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"📁 **Location**: `{test_samples_dir}`")
                st.write(f"🖼️ **Images**: {len(existing_samples)} samples")
                st.write(f"📊 **Labels**: `labels.csv`")
                
            with col2:
                # Download existing samples as ZIP
                if st.button("📦 Download Existing Samples"):
                    try:
                        import zipfile
                        import io
                            
                        # Create ZIP in memory
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            # Add all image files
                            for img_path in existing_samples:
                                zip_file.write(img_path, img_path.name)
                            # Add labels CSV
                            if labels_csv.exists():
                                zip_file.write(labels_csv, labels_csv.name)
                            
                        zip_buffer.seek(0)
                        st.download_button(
                            label="⬇️ Download ZIP",
                            data=zip_buffer.getvalue(),
                            file_name="test_samples.zip",
                            mime="application/zip",
                            help="Download all test samples and labels as ZIP file"
                        )
                    except Exception as e:
                        st.error(f"❌ Failed to create ZIP: {e}")
                
            with col3:
                # Option to regenerate samples
                if st.button("🔄 Regenerate Samples"):
                    st.session_state.regenerate_samples = True
                    st_rerun()
            
        # Test samples configuration
        st.markdown("**🎛️ Configure Test Samples:**")
            
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            num_samples = st.number_input(
                "Number of Samples",
                min_value=1,
                max_value=500,
                value=50,
                step=10,
                help="How many test samples to extract"
            )
            
        with col2:
            image_format = st.selectbox(
                "Image Format",
                options=["PNG", "JPG"],
                index=0,
                help="Output image format"
            )
            
        with col3:
            generate_samples = st.button(
                "🔬 Generate Test Samples",
                help="Extract test samples from the dataset for evaluation",
                type="primary"
            )
            
        # Generate samples if requested
        if generate_samples or st.session_state.get('regenerate_samples', False):
            if st.session_state.get('regenerate_samples', False):
                st.session_state.regenerate_samples = False
                
            if st.session_state.get('dataset_info'):
                dataset_info = st.session_state.dataset_info
                    
                # Check if it's a dict with training results keys (wrong type)
                if isinstance(dataset_info, dict) and 'best_accuracy' in dataset_info:
                    st.error("❌ Dataset information not available. The system has training results but no dataset info.")
                    st.info("💡 This usually happens after training. Please select and analyze a dataset first.")
                else:
                    with st.spinner(f"🔬 Generating {num_samples} test samples..."):
                        try:
                            output_dir = "."  # Use current directory as base
                            save_test_samples_for_evaluation(
                                dataset_info, 
                                output_dir, 
                                num_samples=num_samples,
                                image_format=image_format.lower()
                            )
                            st.success(f"✅ Generated {num_samples} test samples!")
                            st.balloons()
                            st_rerun()  # Refresh to show download options
                                
                        except Exception as e:
                            st.error(f"❌ Failed to generate test samples: {str(e)}")
                            logger.error(f"Test sample generation failed: {str(e)}")
            else:
                st.error("❌ No dataset information found. Please train a model first.")
            
        # Show model configuration if available
        if st.session_state.get('model_config'):
            st.subheader("🧠 Model Configuration")
            config = st.session_state.model_config
            try:
                import json as _json
                # Convert to dict safely, handling non-serializable fields
                raw = config.__dict__ if hasattr(config, '__dict__') else config
                safe = _json.loads(_json.dumps(raw, default=str))
                st.json(safe)
            except Exception:
                st.write(config)
            
        # Show dataset information if available
        if st.session_state.get('dataset_info'):
            dataset_info = st.session_state.dataset_info
                
            info_col1, info_col2, info_col3 = st.columns(3)
            with info_col1:
                st.metric("Task Type", dataset_info.task_type.title())
                st.metric("Number of Classes", dataset_info.num_classes)
            with info_col2:
                st.metric("Number of Samples", dataset_info.num_samples)
                    
                # Handle different dataset types
                if hasattr(dataset_info, 'image_size') and dataset_info.image_size:
                    if isinstance(dataset_info.image_size, (list, tuple)) and len(dataset_info.image_size) >= 2:
                        st.metric("Image Size", f"{dataset_info.image_size[0]}×{dataset_info.image_size[1]}")
                    else:
                        st.metric("Image Size", str(dataset_info.image_size))
                elif hasattr(dataset_info, 'is_text_dataset') and dataset_info.is_text_dataset:
                    st.metric("Data Type", "Text Dataset")
                else:
                    st.metric("Image Size", "N/A")
            with info_col3:
                st.metric("Batch Size", dataset_info.recommended_batch_size)
                st.metric("Estimated Time", f"{dataset_info.estimated_training_time:.1f}s")
            
            st.markdown("---")
        else:
            st.warning("🤔 Training completed but no results found in session state.")
    else:
        st.info(
            "ℹ️ No training results in this session yet. "
            "Complete the Training step (Step 4) to see results here, "
            "or load a historical experiment from the directory browser below."
        )
    
    # Dynamic results directory detection and selection
    st.subheader("📁 Historical Results")
    
    # Detect available result directories
    
    def find_result_directories():
        """Find all directories containing training results"""
        potential_dirs = []
        current_dir = Path(".")
        
        # Common result directory patterns
        patterns = [
            "experiments",
            "experiment_*",
            "results",
            "output",
            "training_results",
            "*_results"
        ]
        
        for pattern in patterns:
            for path in current_dir.glob(pattern):
                if path.is_dir():
                    # Check if directory contains result files
                    has_results = any([
                        list(path.glob("*.json")),      # JSON result files
                        list(path.glob("*.pt")),        # PyTorch model files
                        list(path.glob("*.keras")),     # TensorFlow model files
                        list(path.glob("*.h5")),        # Keras model files
                        list(path.glob("logs/*")),      # Log directories
                        list(path.glob("checkpoints/*")) # Checkpoint directories
                    ])
                    
                    if has_results:
                        # Get directory modification time for sorting
                        mod_time = path.stat().st_mtime
                        potential_dirs.append((str(path), mod_time))
        
        # Sort by modification time (newest first)
        potential_dirs.sort(key=lambda x: x[1], reverse=True)
        return [dir_path for dir_path, _ in potential_dirs]
    
    available_dirs = find_result_directories()
    
    if not available_dirs:
        st.warning("🤔 No result directories found. Train a model first to generate results.")
        st.info("💡 **Tip**: Result directories should contain .json, .pt, .keras, or .h5 files")
        return
    
    # Directory selection with dropdown
    default_dir = available_dirs[0] if available_dirs else "./experiments"
    
    results_dir = st.selectbox(
        "Select Results Directory",
        options=available_dirs,
        index=0,
        help="Choose which experiment results to display"
    )
    
    # Add manual directory input as backup
    with st.expander("🔧 Manual Directory Input (Advanced)"):
        manual_dir = st.text_input(
            "Custom Results Directory",
            value=results_dir,
            help="Enter a custom path to results directory"
        )
        if manual_dir != results_dir:
            results_dir = manual_dir
    
    if not Path(results_dir).exists():
        st.error(f"❌ Results directory not found: `{results_dir}`")
        st.info("💡 Please select a valid directory from the dropdown or check your manual input.")
        return
    
    st.success(f"✅ **Selected Results Directory**: `{results_dir}`")
    
    # Load and display results dynamically
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Training Metrics & Accuracy")
        
        # Dynamic result file detection
        result_files = []
        results_path = Path(results_dir)
        
        # Look for different types of result files
        for pattern in ["*.json", "*_results.json", "*results*.json"]:
            for file_path in results_path.glob(pattern):
                result_files.append(file_path)
        
        
        if not result_files:
            st.warning("🤔 No result files found in the selected directory")
            st.info("💡 **Expected files**: *.json, *_results.json, logs/training_metrics.json")
        
        # Load the most recent or most comprehensive result file
        results_data = None
        selected_file = None
        
        # Priority order: specific result files > logs > any json
        for pattern_priority in ["*results*.json", "logs/training_metrics.json", "*.json"]:
            matching_files = list(results_path.glob(pattern_priority))
            if matching_files:
                # Sort by modification time, get most recent
                selected_file = max(matching_files, key=lambda x: x.stat().st_mtime)
                break
        
        if selected_file and selected_file.exists():
            try:
                with open(selected_file) as f:
                    results_data = json.load(f)
            except Exception as e:
                st.error(f"❌ Failed to load result file: {e}")
        
        # Handle data based on structure (DataFrame for logs vs direct results)
        df = None
        if results_data:
            if isinstance(results_data, list):
                # Log-style data (list of records)
                df = pd.DataFrame(results_data)
            elif isinstance(results_data, dict) and 'training_history' in results_data:
                # PyTorch/TensorFlow style with training_history
                # Convert training history to DataFrame for plotting
                if results_data['training_history']:
                    df = pd.DataFrame(results_data['training_history'])
            else:
                pass
            
            # Display key accuracy metrics - handle both DataFrame and direct results
            latest_metrics = None
            
            if df is not None and not df.empty:
                # DataFrame-style metrics (from logs)
                latest_metrics = df.iloc[-1]
            elif results_data and isinstance(results_data, dict):
                # Direct result data (from result files)  
                latest_metrics = results_data
            
            if latest_metrics is not None:
                # Create metrics summary
                st.subheader("🎯 Final Model Performance")
                
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    # Validation Accuracy - multiple possible keys
                    val_acc = None
                    for key in ['val_accuracy', 'val_acc', 'best_accuracy', 'final_val_accuracy']:
                        if key in latest_metrics:
                            val_acc = latest_metrics[key]
                            break
                    
                    if val_acc is not None:
                        if val_acc > 1:  # Already percentage
                            st.metric("Validation Accuracy", f"{val_acc:.2f}%")
                        else:  # Convert to percentage
                            st.metric("Validation Accuracy", f"{val_acc:.4f}", f"{val_acc*100:.2f}%")
                    else:
                        st.metric("Validation Accuracy", "N/A")
                
                with metric_cols[1]:
                    # Training Accuracy - multiple possible keys
                    train_acc = None
                    for key in ['train_accuracy', 'train_acc', 'final_train_accuracy']:
                        if key in latest_metrics:
                            train_acc = latest_metrics[key]
                            break
                    
                    if train_acc is not None:
                        if train_acc > 1:  # Already percentage
                            st.metric("Training Accuracy", f"{train_acc:.2f}%")
                        else:  # Convert to percentage
                            st.metric("Training Accuracy", f"{train_acc:.4f}", f"{train_acc*100:.2f}%")
                    else:
                        st.metric("Training Accuracy", "N/A")
                
                with metric_cols[2]:
                    # Loss - multiple possible keys
                    loss = None
                    for key in ['val_loss', 'best_loss', 'final_loss']:
                        if key in latest_metrics:
                            loss = latest_metrics[key]
                            break
                    
                    if loss is not None:
                        st.metric("Final Loss", f"{loss:.6f}")
                    else:
                        st.metric("Final Loss", "N/A")
                
                with metric_cols[3]:
                    # Framework info
                    framework = latest_metrics.get('framework', 'Unknown')
                    st.metric("Framework", framework)
                
                
                # Accuracy details table
                st.subheader("📈 Detailed Performance Metrics")
                
                performance_data = []
                if 'val_accuracy' in latest_metrics or 'val_acc' in latest_metrics:
                    val_acc_key = 'val_accuracy' if 'val_accuracy' in latest_metrics else 'val_acc'
                    val_acc = latest_metrics[val_acc_key]
                    performance_data.append({"Metric": "Validation Accuracy", "Value": f"{val_acc:.6f}", "Percentage": f"{val_acc*100:.4f}%"})
                
                if 'train_accuracy' in latest_metrics or 'train_acc' in latest_metrics:
                    train_acc_key = 'train_accuracy' if 'train_accuracy' in latest_metrics else 'train_acc'
                    train_acc = latest_metrics[train_acc_key]
                    performance_data.append({"Metric": "Training Accuracy", "Value": f"{train_acc:.6f}", "Percentage": f"{train_acc*100:.4f}%"})
                
                if 'val_loss' in latest_metrics:
                    performance_data.append({"Metric": "Validation Loss", "Value": f"{latest_metrics['val_loss']:.6f}", "Percentage": "N/A"})
                
                if 'train_loss' in latest_metrics:
                    performance_data.append({"Metric": "Training Loss", "Value": f"{latest_metrics['train_loss']:.6f}", "Percentage": "N/A"})
                
                # Additional metrics if available
                for metric in ['precision', 'recall', 'f1_score', 'auc', 'top5_accuracy']:
                    if metric in latest_metrics:
                        value = latest_metrics[metric]
                        percentage = f"{value*100:.4f}%" if metric != 'auc' else f"{value:.6f}"
                        performance_data.append({"Metric": metric.replace('_', ' ').title(), "Value": f"{value:.6f}", "Percentage": percentage})
                
                if performance_data:
                    perf_df = pd.DataFrame(performance_data)
                    st.dataframe(perf_df, use_container_width=True)
                
                # Best accuracy achieved
                st.subheader("🏆 Best Performance Achieved")
                best_metrics = {}
                
                for acc_col in ['val_accuracy', 'val_acc', 'train_accuracy', 'train_acc']:
                    if acc_col in df.columns:
                        best_val = df[acc_col].max()
                        best_epoch = df[df[acc_col] == best_val]['epoch'].iloc[0] if 'epoch' in df.columns else "Unknown"
                        best_metrics[acc_col] = {"value": best_val, "epoch": best_epoch}
                
                if best_metrics:
                    best_cols = st.columns(2)
                    
                    with best_cols[0]:
                        for key in ['val_accuracy', 'val_acc']:
                            if key in best_metrics:
                                value = best_metrics[key]['value']
                                epoch = best_metrics[key]['epoch']
                                st.info(f"🎯 **Best Validation Accuracy:** {value:.6f} ({value*100:.4f}%) at epoch {epoch}")
                                break
                    
                    with best_cols[1]:
                        for key in ['train_accuracy', 'train_acc']:
                            if key in best_metrics:
                                value = best_metrics[key]['value']
                                epoch = best_metrics[key]['epoch']
                                st.info(f"🚀 **Best Training Accuracy:** {value:.6f} ({value*100:.4f}%) at epoch {epoch}")
                                break
            
            # Plot training curves with accuracy
            st.subheader("📊 Training Curves")
            
            # Create subplot for loss and accuracy
            from plotly.subplots import make_subplots
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Loss Curves', 'Accuracy Curves'),
                vertical_spacing=0.1
            )
            
            # Create epoch index if not present
            if 'epoch' not in df.columns:
                df = df.copy()
                df['epoch'] = range(1, len(df) + 1)
            
            # Loss curves
            if 'train_loss' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['epoch'], y=df['train_loss'],
                    mode='lines', name='Training Loss',
                    line=dict(color='red')
                ), row=1, col=1)
            
            if 'val_loss' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['epoch'], y=df['val_loss'],
                    mode='lines', name='Validation Loss',
                    line=dict(color='orange')
                ), row=1, col=1)
            
            # Accuracy curves
            for acc_col, color, name in [
                ('train_accuracy', 'blue', 'Training Accuracy'),
                ('train_acc', 'blue', 'Training Accuracy'),
                ('val_accuracy', 'green', 'Validation Accuracy'),
                ('val_acc', 'green', 'Validation Accuracy')
            ]:
                if acc_col in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df['epoch'], y=df[acc_col],
                        mode='lines', name=name,
                        line=dict(color=color)
                    ), row=2, col=1)
                    break  # Only add one of each type
            
            fig.update_layout(
                height=600,
                showlegend=True,
                title_text="Training Progress Overview"
            )
            
            fig.update_xaxes(title_text="Epoch", row=2, col=1)
            fig.update_yaxes(title_text="Loss", row=1, col=1)
            fig.update_yaxes(title_text="Accuracy", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional metrics visualization
            if any(col in df.columns for col in ['precision', 'recall', 'f1_score']):
                st.subheader("📋 Additional Metrics")
                
                fig2 = go.Figure()
                
                for metric, color in [('precision', 'purple'), ('recall', 'brown'), ('f1_score', 'pink')]:
                    if metric in df.columns:
                        fig2.add_trace(go.Scatter(
                            x=df['epoch'], y=df[metric],
                            mode='lines+markers', name=metric.replace('_', ' ').title(),
                            line=dict(color=color)
                        ))
                
                fig2.update_layout(
                    title="Precision, Recall & F1-Score",
                    xaxis_title="Epoch",
                    yaxis_title="Score",
                    yaxis=dict(range=[0, 1])
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            
        else:
            st.info("📊 No training metrics found. Start a training session to see detailed accuracy metrics here.")
    
    with col2:
        st.subheader("📥 Model Download & Export")
        
        # Find all model files in the results directory
        results_path = Path(results_dir)
        model_files = []
        
        # Common model file extensions for different frameworks
        model_extensions = {
            '*.pt': 'PyTorch Model',
            '*.pth': 'PyTorch Model', 
            '*.keras': 'TensorFlow/Keras Model',
            '*.h5': 'Keras Model',
            '*.pkl': 'Pickle Model (Scikit-learn)',
            '*.joblib': 'Joblib Model (Scikit-learn)',
            '*.onnx': 'ONNX Model',
            '*.pb': 'TensorFlow SavedModel',
            '*.tflite': 'TensorFlow Lite Model'
        }
        
        # Scan for model files
        for pattern, description in model_extensions.items():
            for model_file in results_path.glob(pattern):
                file_size = model_file.stat().st_size
                file_size_mb = file_size / (1024 * 1024)
                model_files.append({
                    'file': model_file,
                    'name': model_file.name,
                    'type': description,
                    'size': file_size,
                    'size_mb': file_size_mb,
                    'path': str(model_file)
                })
        
        # Also look in subdirectories for models
        for subdir in ['checkpoints', 'models', 'saved_models']:
            subdir_path = results_path / subdir
            if subdir_path.exists():
                for pattern, description in model_extensions.items():
                    for model_file in subdir_path.glob(pattern):
                        file_size = model_file.stat().st_size
                        file_size_mb = file_size / (1024 * 1024)
                        model_files.append({
                            'file': model_file,
                            'name': f"{subdir}/{model_file.name}",
                            'type': description,
                            'size': file_size,
                            'size_mb': file_size_mb,
                            'path': str(model_file)
                        })
        
        if model_files:
            st.success(f"📦 Found {len(model_files)} model file(s) available for download")
            
            # Sort by file size (largest first, typically the main model)
            model_files.sort(key=lambda x: x['size'], reverse=True)
            
            # Display available models
            for i, model_info in enumerate(model_files):
                with st.expander(f"📄 {model_info['name']} ({model_info['type']}) - {model_info['size_mb']:.1f} MB"):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**File**: `{model_info['name']}`")
                        st.write(f"**Type**: {model_info['type']}")
                        st.write(f"**Size**: {model_info['size_mb']:.2f} MB ({model_info['size']:,} bytes)")
                        st.write(f"**Path**: `{model_info['path']}`")
                    
                    with col2:
                        # Copy path button
                        if st.button(f"📋 Copy Path", key=f"copy_path_{i}"):
                            # Use JavaScript to copy to clipboard (basic approach)
                            st.code(model_info['path'])
                            st.info("💡 Path displayed above - copy manually")
                    
                    with col3:
                        # Download button
                        try:
                            with open(model_info['file'], 'rb') as model_file:
                                model_data = model_file.read()
                                
                            st.download_button(
                                label=f"⬇️ Download",
                                data=model_data,
                                file_name=model_info['name'],
                                mime='application/octet-stream',
                                key=f"download_{i}",
                                help=f"Download {model_info['type']} ({model_info['size_mb']:.1f} MB)"
                            )
                        except Exception as e:
                            st.error(f"❌ Cannot read file: {e}")
            
            # Bulk download option for multiple models
            if len(model_files) > 1:
                st.markdown("---")
                st.subheader("📦 Bulk Download Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create ZIP archive of all models
                    if st.button("🗜️ Download All Models (ZIP)", type="primary"):
                        import zipfile
                        import io
                        
                        # Create ZIP in memory
                        zip_buffer = io.BytesIO()
                        
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            for model_info in model_files:
                                try:
                                    # Add each model to ZIP
                                    zip_file.write(model_info['file'], model_info['name'])
                                except Exception as e:
                                    st.warning(f"⚠️ Could not add {model_info['name']} to ZIP: {e}")
                        
                        zip_buffer.seek(0)
                        
                        # Calculate total size
                        total_size_mb = sum(m['size_mb'] for m in model_files)
                        
                        st.download_button(
                            label=f"⬇️ Download ZIP ({total_size_mb:.1f} MB)",
                            data=zip_buffer.getvalue(),
                            file_name=f"models_{Path(results_dir).name}.zip",
                            mime='application/zip',
                            key="download_all_zip"
                        )
                
                with col2:
                    # Create model info summary
                    if st.button("📄 Download Model Summary (JSON)"):
                        # Create summary of all models
                        model_summary = {
                            'results_directory': results_dir,
                            'total_models': len(model_files),
                            'total_size_mb': sum(m['size_mb'] for m in model_files),
                            'models': [
                                {
                                    'name': m['name'],
                                    'type': m['type'],
                                    'size_mb': m['size_mb'],
                                    'path': m['path']
                                }
                                for m in model_files
                            ],
                            'generated_at': pd.Timestamp.now().isoformat()
                        }
                        
                        summary_json = json.dumps(model_summary, indent=2)
                        
                        st.download_button(
                            label="⬇️ Download Summary",
                            data=summary_json,
                            file_name=f"model_summary_{Path(results_dir).name}.json",
                            mime='application/json',
                            key="download_summary"
                        )
            
            # Model export options
            st.markdown("---")
            st.subheader("🔄 Model Export & Conversion")
            
            st.info("💡 **Convert your trained models to different formats for deployment**")
            
            # Get PyTorch and TensorFlow models separately
            pytorch_models = [m for m in model_files if m['type'] in ['PyTorch Model']]
            tensorflow_models = [m for m in model_files if m['type'] in ['TensorFlow/Keras Model', 'Keras Model']]
            
            if pytorch_models or tensorflow_models:
                # Create tabs for different conversion types
                conv_tab1, conv_tab2, conv_tab3 = st.tabs(["🔄 ONNX Export", "📱 TensorFlow Lite", "🍎 CoreML"])
                
                # ONNX Export Tab
                with conv_tab1:
                    st.markdown("#### Export to ONNX Format")
                    st.write("ONNX (Open Neural Network Exchange) allows cross-platform model deployment.")
                    
                    # Select model to convert
                    available_models = pytorch_models + tensorflow_models
                    if available_models:
                        model_names = [m['name'] for m in available_models]
                        selected_model_idx = st.selectbox(
                            "Select model to convert:",
                            range(len(model_names)),
                            format_func=lambda i: f"{model_names[i]} ({available_models[i]['type']})",
                            key="onnx_model_select"
                        )
                        
                        selected_model = available_models[selected_model_idx]
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**Model**: `{selected_model['name']}`")
                            st.write(f"**Size**: {selected_model['size_mb']:.2f} MB")
                        
                        with col2:
                            if st.button("🔄 Convert to ONNX", key="convert_onnx", type="primary"):
                                with st.spinner("Converting model to ONNX format..."):
                                    try:
                                        # Determine model type and convert
                                        if selected_model['type'] == 'PyTorch Model':
                                            # PyTorch to ONNX conversion
                                            try:
                                                import torch
                                                import torch.onnx
                                                
                                                # Load PyTorch model
                                                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                                                model = torch.load(selected_model['file'], map_location=device)
                                                model.eval()
                                                
                                                # Create dummy input (adjust dimensions as needed)
                                                batch_size = 1
                                                dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
                                                
                                                # Output path
                                                onnx_path = selected_model['file'].with_suffix('.onnx')
                                                
                                                # Export
                                                torch.onnx.export(
                                                    model,
                                                    dummy_input,
                                                    str(onnx_path),
                                                    export_params=True,
                                                    opset_version=14,
                                                    do_constant_folding=True,
                                                    input_names=['input'],
                                                    output_names=['output'],
                                                    dynamic_axes={
                                                        'input': {0: 'batch_size'},
                                                        'output': {0: 'batch_size'}
                                                    }
                                                )
                                                
                                                st.success(f"✅ Successfully converted to ONNX!")
                                                st.write(f"📄 Saved to: `{onnx_path.name}`")
                                                
                                                # Offer download
                                                with open(onnx_path, 'rb') as f:
                                                    st.download_button(
                                                        label="⬇️ Download ONNX Model",
                                                        data=f.read(),
                                                        file_name=onnx_path.name,
                                                        mime='application/octet-stream',
                                                        key="download_onnx"
                                                    )
                                            
                                            except ImportError:
                                                st.error("❌ PyTorch not installed. Install with: `pip install torch`")
                                            except Exception as e:
                                                st.error(f"❌ Conversion failed: {str(e)}")
                                                st.info("💡 Tip: Ensure model architecture matches expected input dimensions")
                                        
                                        elif selected_model['type'] in ['TensorFlow/Keras Model', 'Keras Model']:
                                            # TensorFlow to ONNX conversion
                                            try:
                                                import tensorflow as tf
                                                import tf2onnx
                                                
                                                # Load Keras model
                                                model = tf.keras.models.load_model(selected_model['file'])
                                                
                                                # Output path
                                                onnx_path = selected_model['file'].with_suffix('.onnx')
                                                
                                                # Convert using tf2onnx
                                                spec = (tf.TensorSpec(model.inputs[0].shape, tf.float32, name="input"),)
                                                model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=14)
                                                
                                                # Save ONNX model
                                                with open(onnx_path, "wb") as f:
                                                    f.write(model_proto.SerializeToString())
                                                
                                                st.success(f"✅ Successfully converted to ONNX!")
                                                st.write(f"📄 Saved to: `{onnx_path.name}`")
                                                
                                                # Offer download
                                                with open(onnx_path, 'rb') as f:
                                                    st.download_button(
                                                        label="⬇️ Download ONNX Model",
                                                        data=f.read(),
                                                        file_name=onnx_path.name,
                                                        mime='application/octet-stream',
                                                        key="download_onnx_tf"
                                                    )
                                            
                                            except ImportError as ie:
                                                st.error(f"❌ Required library not installed: {str(ie)}")
                                                st.info("💡 Install with: `pip install tf2onnx onnx`")
                                            except Exception as e:
                                                st.error(f"❌ Conversion failed: {str(e)}")
                                                st.info("💡 Tip: Ensure TensorFlow model is properly saved")
                                    
                                    except Exception as e:
                                        st.error(f"❌ Error during conversion: {str(e)}")
                    else:
                        st.warning("⚠️ No compatible models found for ONNX conversion")
                
                # TensorFlow Lite Tab
                with conv_tab2:
                    st.markdown("#### Export to TensorFlow Lite")
                    st.write("TensorFlow Lite format optimized for mobile and edge devices.")
                    
                    if tensorflow_models:
                        model_names = [m['name'] for m in tensorflow_models]
                        selected_model_idx = st.selectbox(
                            "Select TensorFlow/Keras model:",
                            range(len(model_names)),
                            format_func=lambda i: model_names[i],
                            key="tflite_model_select"
                        )
                        
                        selected_model = tensorflow_models[selected_model_idx]
                        
                        # Optimization options
                        st.write("**Optimization Options:**")
                        optimize = st.checkbox("Enable optimizations (smaller file size)", value=True, key="tflite_optimize")
                        quantize = st.checkbox("Apply quantization (INT8)", value=False, key="tflite_quantize")
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**Model**: `{selected_model['name']}`")
                        
                        with col2:
                            if st.button("📱 Convert to TFLite", key="convert_tflite", type="primary"):
                                with st.spinner("Converting to TensorFlow Lite..."):
                                    try:
                                        import tensorflow as tf
                                        
                                        # Load Keras model
                                        model = tf.keras.models.load_model(selected_model['file'])
                                        
                                        # Create converter
                                        converter = tf.lite.TFLiteConverter.from_keras_model(model)
                                        
                                        # Apply optimizations
                                        if optimize:
                                            converter.optimizations = [tf.lite.Optimize.DEFAULT]
                                        
                                        if quantize:
                                            converter.target_spec.supported_types = [tf.float16]
                                        
                                        # Convert
                                        tflite_model = converter.convert()
                                        
                                        # Save
                                        tflite_path = selected_model['file'].with_suffix('.tflite')
                                        with open(tflite_path, 'wb') as f:
                                            f.write(tflite_model)
                                        
                                        # Show size comparison
                                        original_size = selected_model['size_mb']
                                        tflite_size = len(tflite_model) / (1024 * 1024)
                                        reduction = ((original_size - tflite_size) / original_size) * 100
                                        
                                        st.success(f"✅ Successfully converted to TensorFlow Lite!")
                                        st.write(f"📄 Saved to: `{tflite_path.name}`")
                                        st.write(f"📊 Original: {original_size:.2f} MB → TFLite: {tflite_size:.2f} MB")
                                        st.write(f"🎯 Size reduction: {reduction:.1f}%")
                                        
                                        # Download button
                                        st.download_button(
                                            label="⬇️ Download TFLite Model",
                                            data=tflite_model,
                                            file_name=tflite_path.name,
                                            mime='application/octet-stream',
                                            key="download_tflite"
                                        )
                                    
                                    except ImportError:
                                        st.error("❌ TensorFlow not installed. Install with: `pip install tensorflow`")
                                    except Exception as e:
                                        st.error(f"❌ Conversion failed: {str(e)}")
                    else:
                        st.warning("⚠️ No TensorFlow/Keras models found")
                        st.info("💡 TFLite conversion requires TensorFlow/Keras models (.keras, .h5)")
                
                # CoreML Tab
                with conv_tab3:
                    st.markdown("#### Export to Core ML")
                    st.write("Core ML format for iOS and macOS deployment.")
                    
                    if tensorflow_models or pytorch_models:
                        available_models = tensorflow_models + pytorch_models
                        model_names = [m['name'] for m in available_models]
                        selected_model_idx = st.selectbox(
                            "Select model:",
                            range(len(model_names)),
                            format_func=lambda i: f"{model_names[i]} ({available_models[i]['type']})",
                            key="coreml_model_select"
                        )
                        
                        selected_model = available_models[selected_model_idx]
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**Model**: `{selected_model['name']}`")
                        
                        with col2:
                            if st.button("🍎 Convert to CoreML", key="convert_coreml", type="primary"):
                                with st.spinner("Converting to Core ML format..."):
                                    try:
                                        import coremltools as ct
                                        
                                        if selected_model['type'] == 'PyTorch Model':
                                            # PyTorch to CoreML
                                            import torch
                                            model = torch.load(selected_model['file'], map_location='cpu')
                                            model.eval()
                                            
                                            # Trace model
                                            example_input = torch.rand(1, 3, 224, 224)
                                            traced_model = torch.jit.trace(model, example_input)
                                            
                                            # Convert to CoreML
                                            coreml_model = ct.convert(
                                                traced_model,
                                                inputs=[ct.ImageType(name="input", shape=(1, 3, 224, 224))]
                                            )
                                            
                                        elif selected_model['type'] in ['TensorFlow/Keras Model', 'Keras Model']:
                                            # TensorFlow to CoreML
                                            import tensorflow as tf
                                            model = tf.keras.models.load_model(selected_model['file'])
                                            
                                            # Convert to CoreML
                                            coreml_model = ct.convert(model)
                                        
                                        # Save CoreML model
                                        coreml_path = selected_model['file'].with_suffix('.mlmodel')
                                        coreml_model.save(str(coreml_path))
                                        
                                        st.success(f"✅ Successfully converted to Core ML!")
                                        st.write(f"📄 Saved to: `{coreml_path.name}`")
                                        
                                        # Download button
                                        with open(coreml_path, 'rb') as f:
                                            st.download_button(
                                                label="⬇️ Download CoreML Model",
                                                data=f.read(),
                                                file_name=coreml_path.name,
                                                mime='application/octet-stream',
                                                key="download_coreml"
                                            )
                                    
                                    except ImportError as ie:
                                        st.error(f"❌ Required library not installed: {str(ie)}")
                                        st.info("💡 Install coremltools: `pip install coremltools`")
                                    except Exception as e:
                                        st.error(f"❌ Conversion failed: {str(e)}")
                    else:
                        st.warning("⚠️ No compatible models found")
            
            else:
                st.info("ℹ️ No PyTorch or TensorFlow models found for conversion")
                st.write("Conversion tools support:")
                st.write("• PyTorch models (.pt, .pth)")
                st.write("• TensorFlow/Keras models (.keras, .h5)")
            
            # Installation guide
            with st.expander("📦 Installation Guide for Conversion Tools"):
                st.markdown("""
                ### Required Packages for Model Conversion
                
                **ONNX Export:**
                ```bash
                pip install torch onnx  # For PyTorch models
                pip install tf2onnx onnx  # For TensorFlow models
                ```
                
                **TensorFlow Lite:**
                ```bash
                pip install tensorflow
                ```
                
                **Core ML:**
                ```bash
                pip install coremltools
                ```
                
                **All conversion tools:**
                ```bash
                pip install torch onnx tf2onnx tensorflow coremltools
                ```
                """)
        
        else:
            st.warning("🤔 No model files found in the selected results directory")
            st.info("💡 **Expected locations**: *.pt, *.keras, *.h5, *.pkl, *.joblib, checkpoints/, models/")
            
            # Show what files are actually present
            all_files = list(results_path.glob("*"))
            if all_files:
                st.write("📁 **Files found in directory**:")
                for file_path in all_files[:10]:  # Show first 10 files
                    if file_path.is_file():
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        st.write(f"  • `{file_path.name}` ({size_mb:.2f} MB)")
                
                if len(all_files) > 10:
                    st.write(f"  ... and {len(all_files) - 10} more files")
        
        st.markdown("---")
        st.subheader("Model Information")
        
        # Load model config
        config_file = Path(results_dir) / "config.yaml"
        if config_file.exists():
            try:
                import yaml
                
                # Custom loader to safely handle Python objects in YAML
                class SafePathLoader(yaml.SafeLoader):
                    """Custom YAML loader that safely handles Path objects and tuples."""
                    pass
                
                def path_constructor(loader, node):
                    """Convert pathlib paths to strings."""
                    return str(loader.construct_scalar(node))
                
                def tuple_constructor(loader, node):
                    """Convert tuples to lists for JSON compatibility."""
                    return list(loader.construct_sequence(node))
                
                # Register constructors for Python objects
                SafePathLoader.add_constructor('tag:yaml.org,2002:python/object/apply:pathlib.PosixPath', path_constructor)
                SafePathLoader.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)
                SafePathLoader.add_constructor('tag:yaml.org,2002:python/object/apply:pathlib.WindowsPath', path_constructor)
                
                with open(config_file) as f:
                    config_data = yaml.load(f, Loader=SafePathLoader)
                
                # Clean up config data for display
                if config_data:
                    # Convert any remaining Path objects to strings
                    def clean_paths(obj):
                        if isinstance(obj, dict):
                            return {k: clean_paths(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [clean_paths(item) for item in obj]
                        elif hasattr(obj, '__fspath__'):  # Path-like object
                            return str(obj)
                        else:
                            return obj
                    
                    config_data = clean_paths(config_data)
                    st.json(config_data)
                else:
                    st.info("Configuration file is empty")
                    
            except yaml.constructor.ConstructorError as e:
                st.error(f"YAML parsing error: {e}")
                st.info("The configuration file contains unsupported Python objects. Showing raw content:")
                with open(config_file) as f:
                    st.code(f.read(), language="yaml")
            except ImportError:
                st.info("Install PyYAML to view configuration")
            except Exception as e:
                st.error(f"Error loading configuration: {e}")
                st.info("Showing raw YAML content:")
                try:
                    with open(config_file) as f:
                        st.code(f.read(), language="yaml")
                except Exception:
                    st.error("Could not read configuration file")
        else:
            st.info("No configuration found")
    
    # ── Confusion Matrix ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔲 Confusion Matrix")
    
    with st.expander("Show Confusion Matrix & Per-Class Metrics", expanded=False):
        results_for_cm = st.session_state.get('training_results', {})
        dataset_info_cm = st.session_state.get('dataset_info')
        
        if not results_for_cm or not dataset_info_cm:
            st.info("ℹ️ Train a PyTorch model first, then return here to see the confusion matrix.")
        else:
            model_path_cm = results_for_cm.get('model_path')
            framework_cm = results_for_cm.get('framework', 'PyTorch')
            
            if framework_cm != 'PyTorch':
                st.info("ℹ️ Confusion matrix is currently available for PyTorch models only.")
            elif not model_path_cm or not Path(model_path_cm).exists():
                st.warning("⚠️ Model file not found. Please retrain to generate the confusion matrix.")
            else:
                if st.button("🔲 Compute Confusion Matrix", key="btn_confusion_matrix"):
                    with st.spinner("Running inference on test set…"):
                        try:
                            import torch
                            import torch.nn as nn
                            import torch.nn.functional as F
                            import torchvision.transforms as transforms
                            from torch.utils.data import DataLoader
                            import numpy as np
                            from sklearn.metrics import confusion_matrix, classification_report
                            
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            
                            # Load model
                            checkpoint = torch.load(model_path_cm, map_location=device, weights_only=False)
                            
                            # Rebuild the same AdaptiveCNN architecture used during training
                            num_classes_cm = dataset_info_cm.num_classes
                            
                            # Resolve input shape
                            if hasattr(dataset_info_cm, 'builtin_dataset_name'):
                                _builtin_dims = {'MNIST': (28,28,1), 'Fashion-MNIST': (28,28,1), 'CIFAR-10': (32,32,3), 'CIFAR-100': (32,32,3)}
                                _dims = _builtin_dims.get(dataset_info_cm.builtin_dataset_name, (224,224,3))
                                h_cm, w_cm, ch_cm = _dims
                            else:
                                h_cm, w_cm, ch_cm = 224, 224, 3
                            img_size_cm = (h_cm, w_cm)
                            
                            class AdaptiveCNN_cm(nn.Module):
                                def __init__(self, num_classes, input_channels, input_size):
                                    super().__init__()
                                    h, w = input_size
                                    self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
                                    self.bn1 = nn.BatchNorm2d(32)
                                    self.pool1 = nn.MaxPool2d(2, 2) if min(h, w) >= 32 else nn.Identity()
                                    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                                    self.bn2 = nn.BatchNorm2d(64)
                                    self.pool2 = nn.MaxPool2d(2, 2) if min(h, w) >= 16 else nn.Identity()
                                    if min(h, w) >= 64:
                                        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                                        self.bn3 = nn.BatchNorm2d(128)
                                        self.pool3 = nn.MaxPool2d(2, 2)
                                        final_ch = 128
                                    else:
                                        self.conv3 = None
                                        final_ch = 64
                                    self.global_pool = nn.AdaptiveAvgPool2d(1)
                                    self.dropout1 = nn.Dropout(0.5)
                                    self.fc1 = nn.Linear(final_ch, 256)
                                    self.dropout2 = nn.Dropout(0.3)
                                    self.fc2 = nn.Linear(256, num_classes)
                                
                                def forward(self, x):
                                    x = F.relu(self.bn1(self.conv1(x)))
                                    x = self.pool1(x)
                                    x = F.relu(self.bn2(self.conv2(x)))
                                    x = self.pool2(x)
                                    if self.conv3 is not None:
                                        x = F.relu(self.bn3(self.conv3(x)))
                                        x = self.pool3(x)
                                    x = self.global_pool(x)
                                    x = x.view(x.size(0), -1)
                                    x = self.dropout1(x)
                                    x = F.relu(self.fc1(x))
                                    x = self.dropout2(x)
                                    return self.fc2(x)
                            
                            model_cm = AdaptiveCNN_cm(num_classes_cm, ch_cm, img_size_cm).to(device)
                            
                            # Load weights (handle both full model and state_dict saves)
                            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                                model_cm.load_state_dict(checkpoint['model_state_dict'])
                            elif isinstance(checkpoint, dict):
                                try:
                                    model_cm.load_state_dict(checkpoint)
                                except Exception:
                                    st.error("Could not load model weights. The saved model format is incompatible.")
                                    raise
                            else:
                                model_cm = checkpoint
                            model_cm.eval()
                            
                            # Load test data using same built-in dataset
                            transform_cm = transforms.Compose([
                                transforms.Resize(img_size_cm),
                                transforms.ToTensor(),
                            ])
                            
                            import torchvision.datasets as tv_datasets
                            builtin_loaders = {
                                'MNIST': lambda: tv_datasets.MNIST('./data', train=False, download=True, transform=transform_cm),
                                'Fashion-MNIST': lambda: tv_datasets.FashionMNIST('./data', train=False, download=True, transform=transform_cm),
                                'CIFAR-10': lambda: tv_datasets.CIFAR10('./data', train=False, download=True, transform=transform_cm),
                                'CIFAR-100': lambda: tv_datasets.CIFAR100('./data', train=False, download=True, transform=transform_cm),
                            }
                            
                            ds_name_cm = getattr(dataset_info_cm, 'builtin_dataset_name', None)
                            if ds_name_cm and ds_name_cm in builtin_loaders:
                                test_ds_cm = builtin_loaders[ds_name_cm]()
                                test_loader_cm = DataLoader(test_ds_cm, batch_size=256, shuffle=False, num_workers=0)
                                
                                all_preds, all_labels = [], []
                                with torch.no_grad():
                                    for imgs_b, labels_b in test_loader_cm:
                                        imgs_b = imgs_b.to(device)
                                        if ch_cm == 1 and imgs_b.shape[1] == 3:
                                            imgs_b = imgs_b[:, :1, :, :]
                                        elif ch_cm == 3 and imgs_b.shape[1] == 1:
                                            imgs_b = imgs_b.repeat(1, 3, 1, 1)
                                        out_b = model_cm(imgs_b)
                                        preds_b = out_b.argmax(dim=1).cpu().numpy()
                                        all_preds.extend(preds_b.tolist())
                                        all_labels.extend(labels_b.numpy().tolist())
                                
                                all_preds = np.array(all_preds)
                                all_labels = np.array(all_labels)
                                
                                # Get class names
                                from ui.shared import get_dataset_class_names
                                class_names_cm = get_dataset_class_names(ds_name_cm) or [str(i) for i in range(num_classes_cm)]
                                
                                # Compute confusion matrix
                                cm = confusion_matrix(all_labels, all_preds)
                                overall_acc = (all_preds == all_labels).mean() * 100
                                
                                st.success(f"✅ Test accuracy: **{overall_acc:.2f}%** on {len(all_labels):,} samples")
                                
                                # Plot as heatmap
                                fig_cm = px.imshow(
                                    cm,
                                    x=class_names_cm,
                                    y=class_names_cm,
                                    color_continuous_scale='Blues',
                                    text_auto=True,
                                    labels=dict(x='Predicted', y='Actual', color='Count'),
                                    title=f"Confusion Matrix — {ds_name_cm}",
                                    aspect='auto',
                                )
                                fig_cm.update_layout(height=max(400, num_classes_cm * 30 + 100))
                                st.plotly_chart(fig_cm, use_container_width=True)
                                
                                # Per-class classification report
                                report_txt = classification_report(all_labels, all_preds, target_names=class_names_cm, output_dict=True)
                                report_df = pd.DataFrame(report_txt).transpose()
                                report_df = report_df.drop(['accuracy'], errors='ignore')
                                report_df = report_df[['precision', 'recall', 'f1-score', 'support']].round(3)
                                st.subheader("📋 Per-Class Metrics")
                                st.dataframe(report_df, use_container_width=True)
                            else:
                                st.warning("⚠️ Confusion matrix is currently supported for built-in datasets (MNIST, CIFAR-10, etc.).")
                        
                        except Exception as e_cm:
                            st.error(f"❌ Confusion matrix failed: {e_cm}")
                            logger.exception("Confusion matrix error")
    
    # ── In-UI Inference Widget ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Test Your Model")
    
    with st.expander("Upload an image and get predictions from your trained model", expanded=False):
        infer_results = st.session_state.get('training_results', {})
        infer_dataset_info = st.session_state.get('dataset_info')
        
        if not infer_results:
            st.info("ℹ️ Train a PyTorch model first to use the inference widget.")
        else:
            infer_model_path = infer_results.get('model_path')
            infer_framework = infer_results.get('framework', 'PyTorch')
            
            if infer_framework != 'PyTorch':
                st.info("ℹ️ The inference widget currently supports PyTorch models.")
            elif not infer_model_path or not Path(infer_model_path).exists():
                st.warning(f"⚠️ Model file not found at `{infer_model_path}`. Please retrain.")
            else:
                col_upload, col_result = st.columns([1, 1])
                
                with col_upload:
                    uploaded_img = st.file_uploader(
                        "Upload an image",
                        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                        key="inference_upload",
                        help="Upload any image to classify with your trained model",
                    )
                    top_k = st.slider("Top-K predictions", min_value=1, max_value=10, value=5, key="topk_slider")
                
                if uploaded_img is not None:
                    with col_result:
                        try:
                            import torch
                            import torch.nn as nn
                            import torch.nn.functional as F
                            import torchvision.transforms as transforms
                            import numpy as np
                            from PIL import Image as PILImage
                            
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            
                            # Resolve input shape
                            if infer_dataset_info and hasattr(infer_dataset_info, 'builtin_dataset_name'):
                                _dims = {'MNIST': (28,28,1), 'Fashion-MNIST': (28,28,1), 'CIFAR-10': (32,32,3), 'CIFAR-100': (32,32,3)}
                                h_inf, w_inf, ch_inf = _dims.get(infer_dataset_info.builtin_dataset_name, (224,224,3))
                                num_cls_inf = infer_dataset_info.num_classes
                                ds_name_inf = infer_dataset_info.builtin_dataset_name
                            else:
                                h_inf, w_inf, ch_inf = 224, 224, 3
                                num_cls_inf = infer_results.get('num_classes', 10)
                                ds_name_inf = None
                            
                            img_size_inf = (h_inf, w_inf)
                            
                            # Load and preprocess image
                            pil_img = PILImage.open(uploaded_img).convert('RGB')
                            st.image(pil_img, caption="Uploaded Image", use_container_width=True)
                            
                            transform_inf = transforms.Compose([
                                transforms.Resize(img_size_inf),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,) * ch_inf, (0.5,) * ch_inf),
                            ])
                            
                            # Adapt channels
                            if ch_inf == 1:
                                pil_img_ch = pil_img.convert('L')
                            else:
                                pil_img_ch = pil_img
                            
                            img_tensor = transform_inf(pil_img_ch).unsqueeze(0).to(device)
                            
                            # Rebuild model architecture (same as training)
                            class AdaptiveCNN_inf(nn.Module):
                                def __init__(self, num_classes, input_channels, input_size):
                                    super().__init__()
                                    h, w = input_size
                                    self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
                                    self.bn1 = nn.BatchNorm2d(32)
                                    self.pool1 = nn.MaxPool2d(2, 2) if min(h, w) >= 32 else nn.Identity()
                                    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                                    self.bn2 = nn.BatchNorm2d(64)
                                    self.pool2 = nn.MaxPool2d(2, 2) if min(h, w) >= 16 else nn.Identity()
                                    if min(h, w) >= 64:
                                        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                                        self.bn3 = nn.BatchNorm2d(128)
                                        self.pool3 = nn.MaxPool2d(2, 2)
                                        final_ch = 128
                                    else:
                                        self.conv3 = None
                                        final_ch = 64
                                    self.global_pool = nn.AdaptiveAvgPool2d(1)
                                    self.dropout1 = nn.Dropout(0.5)
                                    self.fc1 = nn.Linear(final_ch, 256)
                                    self.dropout2 = nn.Dropout(0.3)
                                    self.fc2 = nn.Linear(256, num_classes)
                                
                                def forward(self, x):
                                    x = F.relu(self.bn1(self.conv1(x)))
                                    x = self.pool1(x)
                                    x = F.relu(self.bn2(self.conv2(x)))
                                    x = self.pool2(x)
                                    if self.conv3 is not None:
                                        x = F.relu(self.bn3(self.conv3(x)))
                                        x = self.pool3(x)
                                    x = self.global_pool(x)
                                    x = x.view(x.size(0), -1)
                                    x = self.dropout1(x)
                                    x = F.relu(self.fc1(x))
                                    x = self.dropout2(x)
                                    return self.fc2(x)
                            
                            model_inf = AdaptiveCNN_inf(num_cls_inf, ch_inf, img_size_inf).to(device)
                            checkpoint_inf = torch.load(infer_model_path, map_location=device, weights_only=False)
                            if isinstance(checkpoint_inf, dict) and 'model_state_dict' in checkpoint_inf:
                                model_inf.load_state_dict(checkpoint_inf['model_state_dict'])
                            elif isinstance(checkpoint_inf, dict):
                                model_inf.load_state_dict(checkpoint_inf)
                            else:
                                model_inf = checkpoint_inf
                            model_inf.eval()
                            
                            # Run inference
                            with torch.no_grad():
                                logits = model_inf(img_tensor)
                                probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
                            
                            # Get top-K predictions
                            k = min(top_k, num_cls_inf)
                            top_indices = np.argsort(probs)[::-1][:k]
                            
                            from ui.shared import get_dataset_class_names
                            class_names_inf = get_dataset_class_names(ds_name_inf) if ds_name_inf else [str(i) for i in range(num_cls_inf)]
                            
                            # Display predictions
                            st.markdown("**🎯 Top Predictions:**")
                            pred_data = {
                                'Rank': list(range(1, k+1)),
                                'Class': [class_names_inf[i] if i < len(class_names_inf) else str(i) for i in top_indices],
                                'Confidence': [f"{probs[i]*100:.2f}%" for i in top_indices],
                                'Score': [float(f"{probs[i]:.4f}") for i in top_indices],
                            }
                            pred_df = pd.DataFrame(pred_data)
                            st.dataframe(pred_df, use_container_width=True, hide_index=True)
                            
                            # Bar chart of top-K
                            fig_inf = go.Figure(go.Bar(
                                x=[class_names_inf[i] if i < len(class_names_inf) else str(i) for i in top_indices],
                                y=[probs[i]*100 for i in top_indices],
                                marker_color='#3b82f6',
                                text=[f"{probs[i]*100:.1f}%" for i in top_indices],
                                textposition='outside',
                            ))
                            fig_inf.update_layout(
                                title=f"Top-{k} Prediction Confidences",
                                xaxis_title="Class",
                                yaxis_title="Confidence (%)",
                                yaxis=dict(range=[0, 115]),
                                height=350,
                                margin=dict(t=40, b=40),
                            )
                            st.plotly_chart(fig_inf, use_container_width=True)
                        
                        except Exception as e_inf:
                            st.error(f"❌ Inference failed: {e_inf}")
                            logger.exception("Inference widget error")
    
    # ── Experiment Comparison ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Experiment Comparison")
    
    with st.expander("Compare multiple training runs side-by-side", expanded=False):
        # Scan for all *_results.json files
        exp_json_files = {}
        for _p in Path('.').glob('**/*_results.json'):
            if '_backup' not in str(_p):
                exp_json_files[str(_p)] = _p
        
        if not exp_json_files:
            st.info("ℹ️ No experiment result files found. Train models to populate this section.")
        else:
            selected_exps = st.multiselect(
                "Select experiments to compare",
                options=list(exp_json_files.keys()),
                default=list(exp_json_files.keys())[:min(4, len(exp_json_files))],
                help="Choose result files to compare side-by-side",
                key="exp_compare_select",
            )
            
            if selected_exps:
                rows_compare = []
                for exp_path in selected_exps:
                    try:
                        with open(exp_path) as _f:
                            _data = json.load(_f)
                        
                        _run_name = Path(exp_path).stem.replace('_results', '')
                        _acc = _data.get('best_accuracy') or _data.get('val_accuracy') or _data.get('val_acc')
                        _loss = _data.get('best_loss') or _data.get('val_loss')
                        _time = _data.get('training_time')
                        _epochs = _data.get('total_epochs') or (len(_data.get('training_history', [])) or None)
                        _fw = _data.get('framework', 'unknown')
                        
                        if _acc is not None and _acc <= 1.0:
                            _acc *= 100
                        
                        rows_compare.append({
                            'Run': _run_name,
                            'Framework': _fw,
                            'Val Accuracy (%)': round(_acc, 2) if _acc is not None else None,
                            'Val Loss': round(_loss, 4) if _loss is not None else None,
                            'Training Time (s)': round(_time, 1) if _time is not None else None,
                            'Epochs': _epochs,
                        })
                    except Exception as _e:
                        st.warning(f"⚠️ Could not load {exp_path}: {_e}")
                
                if rows_compare:
                    df_compare = pd.DataFrame(rows_compare)
                    
                    # Summary table
                    st.markdown("**📋 Summary Table**")
                    st.dataframe(df_compare, use_container_width=True, hide_index=True)
                    
                    # Accuracy comparison chart
                    df_acc = df_compare.dropna(subset=['Val Accuracy (%)'])
                    if not df_acc.empty:
                        fig_comp_acc = px.bar(
                            df_acc,
                            x='Run', y='Val Accuracy (%)',
                            color='Framework',
                            text='Val Accuracy (%)',
                            title='Validation Accuracy Comparison',
                            labels={'Val Accuracy (%)': 'Accuracy (%)'},
                        )
                        fig_comp_acc.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                        fig_comp_acc.update_layout(height=400, margin=dict(t=40, b=80), xaxis_tickangle=-30)
                        st.plotly_chart(fig_comp_acc, use_container_width=True)
                    
                    # Loss comparison chart
                    df_loss = df_compare.dropna(subset=['Val Loss'])
                    if not df_loss.empty:
                        fig_comp_loss = px.bar(
                            df_loss,
                            x='Run', y='Val Loss',
                            color='Framework',
                            text='Val Loss',
                            title='Validation Loss Comparison',
                        )
                        fig_comp_loss.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                        fig_comp_loss.update_layout(height=400, margin=dict(t=40, b=80), xaxis_tickangle=-30)
                        st.plotly_chart(fig_comp_loss, use_container_width=True)
                    
                    # Training time comparison
                    df_time = df_compare.dropna(subset=['Training Time (s)'])
                    if not df_time.empty:
                        fig_comp_time = px.bar(
                            df_time,
                            x='Run', y='Training Time (s)',
                            color='Framework',
                            title='Training Time Comparison',
                        )
                        fig_comp_time.update_layout(height=350, margin=dict(t=40, b=80), xaxis_tickangle=-30)
                        st.plotly_chart(fig_comp_time, use_container_width=True)
    
    # Start New Project Option
    st.markdown("---")
    st.subheader("🚀 Ready for Another Project?")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.write("Start fresh with a new dataset and model configuration")
        
        # Framework selection for new project
        st.markdown("**Select Framework for New Project:**")
        
        # Initialize the temporary framework selection in session state if not exists
        if 'temp_new_project_framework' not in st.session_state:
            st.session_state.temp_new_project_framework = "PyTorch"
        
        def update_temp_framework():
            """Update temporary framework selection immediately when changed."""
            st.session_state.temp_new_project_framework = st.session_state.new_project_framework_selector
        
        new_project_framework = st.selectbox(
            "Choose ML Framework",
            ["PyTorch", "TensorFlow/Keras", "Scikit-learn"],
            index=["PyTorch", "TensorFlow/Keras", "Scikit-learn"].index(st.session_state.temp_new_project_framework),
            help="Select the machine learning framework for your new training project",
            key="new_project_framework_selector",
            on_change=update_temp_framework
        )
        
        # Update the temporary selection immediately
        st.session_state.temp_new_project_framework = new_project_framework
        
        # Show framework-specific features
        framework_info = {
            "PyTorch": {
                "icon": "🔥",
                "features": ["Deep Learning", "Computer Vision", "NLP", "Custom Architectures", "ONNX Export"],
                "best_for": "Research, Custom Models, Computer Vision"
            },
            "TensorFlow/Keras": {
                "icon": "🧠",
                "features": ["Deep Learning", "Production Deployment", "TensorFlow Lite", "TPU Support"],
                "best_for": "Production, Mobile Deployment, Large Scale"
            },
            "Scikit-learn": {
                "icon": "📊",
                "features": ["Classical ML", "Fast Training", "Tabular Data", "Preprocessing"],
                "best_for": "Traditional ML, Small Datasets, Quick Results"
            }
        }
        
        if new_project_framework in framework_info:
            info = framework_info[new_project_framework]
            
            with st.expander(f"{info['icon']} {new_project_framework} Features", expanded=True):
                st.write(f"**Best for:** {info['best_for']}")
                st.write("**Key Features:**")
                for feature in info['features']:
                    st.write(f"• {feature}")
        
        st.markdown("---")
        
        if st.button("🎯 Start New Project", type="primary", use_container_width=True):
            # Clear all session state variables to reset the project
            keys_to_clear = [
                'dataset_info',
                'selected_builtin_dataset', 
                'model_config',
                'training_completed',
                'training_results',
                'current_step',
                'uploaded_files',
                'dataset_path',
                'selected_framework'  # Also clear previous framework selection
            ]
            
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Set the selected framework for the new project using the current temp selection
            selected_framework = st.session_state.get('temp_new_project_framework', new_project_framework)
            st.session_state.selected_framework = selected_framework
            
            # Also clear the temporary framework selection
            if 'temp_new_project_framework' in st.session_state:
                del st.session_state['temp_new_project_framework']
            
            # Reset to step 0 (Home) but with framework pre-selected
            st.session_state.current_step = 0
            
            st.success(f"✅ New {selected_framework} project started! Redirecting to home page...")
            st.balloons()
            import time
            time.sleep(1)  # Brief pause for user feedback
            st_rerun()


