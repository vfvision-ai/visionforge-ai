"""
Model Selection page – VisionForge (Train Vision Models Effortlessly).
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
    create_model_comparison_data,
)

def show_model_selection():
    """Display model selection interface."""
    apply_theme()

    # Guard: task type must be selected
    if not st.session_state.get('task_type_confirmed', False) or st.session_state.get('selected_task_type') is None:
        hero("🧠 Model Selection",
             "Please select a task type on the Home page first.",
             badges=["Step 3"])
        if st.button("← Go to Home", type="primary"):
            st.session_state.current_step = 0
            st_rerun()
        return

    task = st.session_state.get('selected_task_type', '')
    hero(
        "🧠 Model Selection",
        f"Choose the best model architecture for your {task} task — auto-select or configure manually.",
        badges=[f"🎯 {task}", "Step 3 of 5"],
    )
    step_tracker(["Home","Dataset","Model","Training","Results"],
                 st.session_state.get("current_step", 2))

    # Reset button (top-right)
    _, col_reset = st.columns([5, 1])
    with col_reset:
        if st.button("🔄 Reset", help="Clear current model configuration"):
            if 'model_config' in st.session_state:
                del st.session_state.model_config
                st.success("✅ Configuration reset!")
                st_rerun()

    if not st.session_state.get('dataset_info'):
        st.markdown("""
        <div style="background:rgba(255,107,107,.08);border:1px solid rgba(255,107,107,.3);
             border-radius:12px;padding:1.2rem 1.6rem;margin-bottom:1rem">
          <div style="font-weight:700;font-size:1rem;margin-bottom:.3rem">⚠️ Dataset not analysed yet</div>
          <div style="font-size:.88rem;color:#c0c0d8">Please complete <b>Dataset Analysis</b> (Step 2) first.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("← Go to Dataset Analysis", type="primary"):
            st.session_state.current_step = 1
            st.session_state["current_tool"] = None
            st_rerun()
        return

    dataset_info = st.session_state.dataset_info

    col1, col2 = st.columns([1, 1])

    with col1:
        section("⚙️", "Model Configuration")

        # Display current framework
        framework = st.session_state.get('selected_framework', 'PyTorch')

        # Framework icons
        framework_icons = {
            "PyTorch": "🔥",
            "TensorFlow/Keras": "🧠", 
            "Scikit-learn": "📊"
        }
        
        icon = framework_icons.get(framework, "🛠️")
        st.markdown(f"### {icon} Selected Framework: **{framework}**")
        st.info(f"✅ Framework was selected on the Home page. To change framework, go back to Home.")
        
        # Framework-specific information
        framework_info = {
            "PyTorch": "🔥 Deep learning with dynamic computation graphs. Best for research and custom architectures.",
            "TensorFlow/Keras": "🧠 Production-ready deep learning with static graphs. Best for deployment and scalability.",
            "Scikit-learn": "📊 Traditional ML algorithms. Best for structured data and classical approaches."
        }
        st.info(f"ℹ️ {framework_info[framework]}")
        
        # Model selection method
        model_method = st.radio(
            "Model Selection Method:",
            ["🤖 Auto-Select", "📁 Load Pre-trained Model", "⚙️ Manual Selection"],
            horizontal=True
        )
        
        if model_method == "🤖 Auto-Select":
            # Framework-aware automatic model selection
            st.markdown(f"### 🎯 Auto-Selecting for {framework}")
            
            # Clear manual configuration flag for auto-select
            st.session_state.manual_config_in_progress = False
            
            # Add explicit trigger button for auto-select to prevent immediate execution
            if st.button("🚀 Start Auto-Selection", key="trigger_auto_select", type="primary"):
                
                # Detect channels from dataset if available
                detected_channels = 3  # Default to RGB
                if hasattr(dataset_info, 'image_stats') and 'channels' in dataset_info.image_stats:
                    detected_channels = dataset_info.image_stats['channels']
                
                # Framework-specific model selection
                if framework == "PyTorch":
                    selector = ModelSelector()
                    
                    # DEFENSIVE FIX: Handle legacy "image_classification" task type
                    if hasattr(dataset_info, 'task_type') and dataset_info.task_type == "image_classification":
                        st.warning("⚠️ **Fixing legacy task type:** Converting 'image_classification' to 'classification'")
                        dataset_info.task_type = "classification"
                        st.info("✅ **Task type corrected!** Please continue.")
                    
                    model_config = selector.select_model(dataset_info)
                    
                    # Update config with detected channels
                    model_config.config_params['input_channels'] = detected_channels
                    model_config.config_params['channel_adaptation'] = 'auto' if detected_channels != 3 else 'none'
                    
                    # Update input size to include channels
                    if isinstance(model_config.input_size, tuple) and len(model_config.input_size) == 2:
                        width, height = model_config.input_size
                        model_config.input_size = (detected_channels, width, height)
                        
                    model_config.framework = "PyTorch"
                    
                elif framework == "TensorFlow/Keras":
                    # TensorFlow/Keras model selection
                    model_config = select_tensorflow_model(dataset_info, detected_channels)
                    
                elif framework == "Scikit-learn":
                    # Scikit-learn model selection  
                    model_config = select_sklearn_model(dataset_info)
                
                # Adjust pretrained setting based on channels
                if detected_channels != 3:
                    model_config.pretrained = False
                    st.info(f"🔍 Detected {detected_channels} channel(s). Pretrained weights disabled for non-RGB images.")
                
                st.session_state.model_config = model_config
                
                # CRITICAL FIX: Set manual_config_in_progress to prevent immediate auto-advance to Training
                st.session_state.manual_config_in_progress = True
                
                channel_type = "Grayscale" if detected_channels == 1 else "RGB" if detected_channels == 3 else "RGBA" if detected_channels == 4 else f"{detected_channels}-channel"
                st.success(f"✅ Auto-Selected: **{model_config.architecture}** for **{channel_type}** images")
                
                # Display model details for user review
                st.markdown("---")
                section("🔍", "Selected Model Details")

                d_c1, d_c2, d_c3 = st.columns(3)
                with d_c1:
                    st.markdown(
                        f'<div class="cv-card" style="border-left:3px solid var(--accent)">'
                        f'<div style="font-size:.7rem;text-transform:uppercase;color:var(--text-secondary)">Architecture</div>'
                        f'<div style="font-weight:700;margin-top:.3rem">{model_config.architecture}</div>'
                        f'<div style="font-size:.8rem;color:var(--text-secondary);margin-top:.2rem">{model_config.backbone}</div>'
                        f'</div>', unsafe_allow_html=True)
                with d_c2:
                    st.markdown(
                        f'<div class="cv-card" style="border-left:3px solid var(--accent2)">'
                        f'<div style="font-size:.7rem;text-transform:uppercase;color:var(--text-secondary)">Parameters</div>'
                        f'<div style="font-weight:700;margin-top:.3rem">{model_config.num_parameters:,}</div>'
                        f'<div style="font-size:.8rem;color:var(--text-secondary);margin-top:.2rem">Input: {model_config.input_size}</div>'
                        f'</div>', unsafe_allow_html=True)
                with d_c3:
                    st.markdown(
                        f'<div class="cv-card" style="border-left:3px solid #ffd93d">'
                        f'<div style="font-size:.7rem;text-transform:uppercase;color:var(--text-secondary)">Memory</div>'
                        f'<div style="font-weight:700;margin-top:.3rem">{model_config.memory_requirements:.1f} GB</div>'
                        f'<div style="font-size:.8rem;color:var(--text-secondary);margin-top:.2rem">Pretrained: {"✅" if model_config.pretrained else "❌"}</div>'
                        f'</div>', unsafe_allow_html=True)

                # Manual continue button instead of auto-advance
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("➡️ Continue to Training", key="auto_model_continue", type="primary", use_container_width=True):
                    st.session_state.current_step = 3
                    st.session_state["current_tool"] = None
                    st.session_state.manual_config_in_progress = False
                    st_rerun()
            else:
                # Show instructions when auto-select hasn't been triggered yet
                st.info("🎯 Click 'Start Auto-Selection' to automatically choose the best model for your dataset.")
            
        elif model_method == "📁 Load Pre-trained Model":
            # Pre-trained model file browser
            st.write("📂 **Select Pre-trained Model File:**")
            
            # Clear manual configuration flag for pre-trained model loading
            st.session_state.manual_config_in_progress = False
            
            # Model file selection
            import os
            model_dir = st.selectbox(
                "Model Directory:",
                [os.getcwd(), "./models", "./checkpoints", "./experiments", os.path.expanduser("~/models")],
                help="Choose directory containing model files"
            )
            
            try:
                if os.path.exists(model_dir):
                    model_files = []
                    for item in os.listdir(model_dir):
                        if item.endswith(('.pth', '.pt', '.ckpt', '.h5', '.pkl', '.pb', '.onnx')):
                            model_files.append(item)
                    
                    if model_files:
                        selected_model_file = st.selectbox(
                            "Available Model Files:",
                            ["[Select a model file]"] + sorted(model_files),
                            help="Choose your pre-trained model file"
                        )
                        
                        if selected_model_file != "[Select a model file]":
                            model_path = os.path.join(model_dir, selected_model_file)
                            st.success(f"📁 Selected Model: `{model_path}`")
                            
                            # Show file info
                            file_size = os.path.getsize(model_path) / (1024*1024)  # MB
                            st.info(f"📊 File Size: {file_size:.1f} MB")
                            
                            # Create a basic model config for pre-trained model
                            from core.model_selector import ModelConfig
                            
                            # Use dynamic input size instead of hardcoded (224, 224)
                            dynamic_input_size = get_user_specified_input_size()
                            
                            model_config = ModelConfig(
                                architecture="pretrained_" + selected_model_file.split('.')[0],
                                backbone="custom",
                                num_parameters=1000000,  # Placeholder
                                input_size=dynamic_input_size[:2],  # Use only (height, width) for compatibility
                                pretrained=True,
                                config_params={"model_path": model_path},
                                estimated_flops=1000000000,
                                memory_requirements=2.0
                            )
                            st.session_state.model_config = model_config
                    else:
                        st.warning(f"No model files found in {model_dir}")
                        st.info("Supported formats: .pth, .pt, .ckpt, .h5, .pkl, .pb, .onnx")
                else:
                    st.error(f"Directory does not exist: {model_dir}")
            except Exception as e:
                st.error(f"Error browsing model files: {str(e)}")
                
        else:  # Manual selection
            st.markdown(f"🔧 **Manual {framework} Model Configuration**")
            
            # Set flag to prevent auto-advance to training
            st.session_state.manual_config_in_progress = True
            
            # Framework-specific model options
            available_models = get_framework_models(framework, dataset_info.task_type)
            
            if available_models:
                # Model architecture selection
                manual_model = st.selectbox(
                    "Choose Model Architecture:",
                    available_models,
                    help=f"Available {dataset_info.task_type} models. The backbone architecture will automatically match your selection."
                )
                
                # Custom CNN Builder Configuration (TensorFlow only)
                custom_cnn_config = None
                if framework == "TensorFlow/Keras" and "Custom CNN Builder" in manual_model:
                    st.markdown("---")
                    st.markdown("### 🏗️ Custom CNN Architecture Builder")
                    st.info("💡 **Build your own CNN from scratch!** Compatible with 1-channel (grayscale) and 3-channel (RGB) images.")
                    
                    # Detect channels from dataset
                    detected_channels = 3  # Default
                    if hasattr(dataset_info, 'channels'):
                        detected_channels = dataset_info.channels
                    elif hasattr(dataset_info, 'image_stats') and 'channels' in dataset_info.image_stats:
                        detected_channels = dataset_info.image_stats['channels']
                    
                    st.success(f"📊 **Detected {detected_channels} channel(s)** from your dataset - Architecture will auto-adapt!")
                    
                    cnn_col1, cnn_col2 = st.columns(2)
                    
                    with cnn_col1:
                        st.markdown("**🔲 Convolutional Layers**")
                        
                        num_conv_blocks = st.slider(
                            "Number of Conv Blocks:",
                            min_value=1,
                            max_value=5,
                            value=3,
                            help="Number of convolutional blocks (each block = Conv2D + BatchNorm + Activation + MaxPool)"
                        )
                        
                        conv_layer_configs = []
                        for i in range(num_conv_blocks):
                            st.markdown(f"**Block {i+1}:**")
                            # No nested columns - use simple layout
                            filters = st.selectbox(
                                f"Filters (Block {i+1}):",
                                [16, 32, 64, 128, 256, 512],
                                index=min(i + 1, 5),
                                key=f"conv_filters_{i}"
                            )
                            kernel_size = st.selectbox(
                                f"Kernel Size (Block {i+1}):",
                                [(3, 3), (5, 5), (7, 7)],
                                index=0,
                                format_func=lambda x: f"{x[0]}×{x[1]}",
                                key=f"kernel_size_{i}"
                            )
                            
                            conv_layer_configs.append({
                                'filters': filters,
                                'kernel_size': kernel_size,
                                'activation': 'relu',
                                'use_batch_norm': True,
                                'use_max_pool': True
                            })
                    
                    with cnn_col2:
                        st.markdown("**🔗 Dense Layers**")
                        
                        num_dense_layers = st.slider(
                            "Number of Hidden Dense Layers:",
                            min_value=0,
                            max_value=2,
                            value=1,
                            help="Hidden dense layers (output layer with num_classes units will be added automatically)"
                        )
                        
                        dense_layer_configs = []
                        for i in range(num_dense_layers):
                            units = st.selectbox(
                                f"Units (Hidden Dense {i+1}):",
                                [64, 128, 256, 512, 1024],
                                index=1,
                                key=f"dense_units_{i}"
                            )
                            dense_layer_configs.append({
                                'units': units,
                                'activation': 'relu',
                                'use_dropout': True
                            })
                        
                        # Add output layer info
                        st.info(f"📌 **Output Layer:** {dataset_info.num_classes} units (auto-added)")
                        
                        st.markdown("**⚙️ Regularization**")
                        dropout_rate = st.slider(
                            "Dropout Rate:",
                            min_value=0.0,
                            max_value=0.8,
                            value=0.3,
                            step=0.05,
                            help="Dropout probability for regularization"
                        )
                        
                        use_global_pooling = st.checkbox(
                            "Use Global Average Pooling",
                            value=True,
                            help="Use GAP instead of Flatten for better generalization"
                        )
                    
                    st.markdown("**🎨 Activation & Optimizer**")
                    act_col1, act_col2, act_col3 = st.columns(3)
                    
                    with act_col1:
                        activation_fn = st.selectbox(
                            "Activation Function:",
                            ["relu", "leaky_relu", "elu", "selu", "swish"],
                            help="Activation function for hidden layers"
                        )
                    
                    with act_col2:
                        optimizer_name = st.selectbox(
                            "Optimizer:",
                            ["Adam", "AdamW", "SGD", "RMSprop"],
                            help="Optimization algorithm"
                        )
                    
                    with act_col3:
                        learning_rate_custom = st.number_input(
                            "Learning Rate:",
                            min_value=0.00001,
                            max_value=0.1,
                            value=0.001,
                            step=0.00001,
                            format="%.5f",
                            help="Initial learning rate"
                        )
                    
                    # Store custom CNN configuration
                    custom_cnn_config = {
                        'type': 'custom_cnn',
                        'detected_channels': detected_channels,
                        'conv_blocks': conv_layer_configs,
                        'dense_layers': dense_layer_configs,
                        'dropout_rate': dropout_rate,
                        'use_global_pooling': use_global_pooling,
                        'activation': activation_fn,
                        'optimizer': optimizer_name,
                        'learning_rate': learning_rate_custom,
                        'num_classes': dataset_info.num_classes
                    }
                    
                    # Preview architecture
                    st.markdown("---")
                    st.markdown("### 📐 Architecture Preview")
                    
                    preview_text = f"**Input:** ({detected_channels} channels) → "
                    for i, block in enumerate(conv_layer_configs):
                        preview_text += f"Conv{i+1}({block['filters']}) → BatchNorm → {activation_fn.upper()} → MaxPool → "
                    
                    if use_global_pooling:
                        preview_text += "GlobalAvgPool → "
                    else:
                        preview_text += "Flatten → "
                    
                    for i, dense in enumerate(dense_layer_configs):
                        preview_text += f"Dense({dense['units']}) → {activation_fn.upper()} → Dropout({dropout_rate}) → "
                    
                    preview_text += f"**Output:** Dense({dataset_info.num_classes}) → Softmax/Sigmoid"
                    
                    st.info(preview_text)
                    
                    # Estimate parameters (conv weights + bias + batch norm + dense layers)
                    total_params = 0
                    for i, block in enumerate(conv_layer_configs):
                        k = block['kernel_size']
                        f = block['filters']
                        c_in = detected_channels if i == 0 else conv_layer_configs[i-1]['filters']
                        total_params += (k[0] * k[1] * c_in * f) + f  # Conv weights + bias
                        total_params += 2 * f  # BatchNorm: gamma + beta (trainable)

                    # Dense layer parameters
                    last_conv_filters = conv_layer_configs[-1]['filters'] if conv_layer_configs else detected_channels
                    # With GlobalAvgPool the flattened size = last_conv_filters; without it use rough 4×4 spatial
                    pooled_size = last_conv_filters if use_global_pooling else last_conv_filters * 4 * 4
                    if dense_layer_configs:
                        total_params += pooled_size * dense_layer_configs[0]['units'] + dense_layer_configs[0]['units']
                        for i in range(len(dense_layer_configs) - 1):
                            total_params += (dense_layer_configs[i]['units'] * dense_layer_configs[i+1]['units']
                                             + dense_layer_configs[i+1]['units'])
                        total_params += dense_layer_configs[-1]['units'] * dataset_info.num_classes + dataset_info.num_classes
                    else:
                        total_params += pooled_size * dataset_info.num_classes + dataset_info.num_classes

                    memory_est_mb = total_params * 4 / (1024 ** 2)  # 4 bytes per param → MB
                    # Store back into config so the Apply button uses the correct values
                    custom_cnn_config['estimated_params'] = total_params
                    custom_cnn_config['estimated_memory'] = total_params * 4 / (1024 ** 3)  # GB for ModelConfig

                    param_col1, param_col2 = st.columns(2)
                    with param_col1:
                        st.metric("📊 Estimated Parameters", f"{total_params:,}")
                    with param_col2:
                        st.metric("💾 Estimated Memory", f"{memory_est_mb:.1f} MB")
                    
                    st.markdown("---")
                
                # Immediate customization options
                st.markdown("**Customize Model Settings:**")
                
                col1_manual, col2_manual = st.columns(2)
                
                with col1_manual:
                    if framework == "Scikit-learn":
                        # Scikit-learn specific algorithm parameters
                        st.markdown("**🔧 Algorithm-Specific Configuration:**")
                        
                        # Algorithm-specific parameters based on selected model
                        if "Random Forest" in manual_model or "Extra Trees" in manual_model:
                            n_estimators = st.slider("Number of Trees:", 10, 1000, 100, 10, help="Number of trees in the forest")
                            max_depth = st.selectbox("Max Depth:", [None, 3, 5, 10, 15, 20, 25], index=0, help="Maximum depth of trees")
                            min_samples_split = st.slider("Min Samples Split:", 2, 20, 2, help="Minimum samples required to split a node")
                            min_samples_leaf = st.slider("Min Samples Leaf:", 1, 10, 1, help="Minimum samples required at leaf node")
                            
                        elif "SVM" in manual_model or "Support Vector" in manual_model:
                            C = st.number_input("Regularization (C):", 0.001, 100.0, 1.0, 0.001, format="%.3f", help="Regularization parameter")
                            kernel = st.selectbox("Kernel:", ["rbf", "linear", "poly", "sigmoid"], help="Kernel type for SVM")
                            if kernel == "poly":
                                degree = st.slider("Polynomial Degree:", 2, 8, 3, help="Degree for polynomial kernel")
                            if kernel == "rbf":
                                gamma = st.selectbox("Gamma:", ["scale", "auto", "custom"], help="Kernel coefficient")
                                if gamma == "custom":
                                    gamma_value = st.number_input("Custom Gamma:", 0.0001, 10.0, 0.001, 0.0001, format="%.4f")
                                    
                        elif "Gradient Boosting" in manual_model or "AdaBoost" in manual_model:
                            n_estimators = st.slider("Number of Estimators:", 50, 500, 100, 10, help="Number of boosting stages")
                            learning_rate = st.number_input("Learning Rate:", 0.01, 2.0, 0.1, 0.01, format="%.2f", help="Learning rate shrinks contribution of each tree")
                            max_depth = st.slider("Max Depth:", 1, 10, 3, help="Maximum depth of individual trees")
                            
                        elif "Logistic Regression" in manual_model or "Ridge" in manual_model or "Lasso" in manual_model:
                            C = st.number_input("Regularization (C):", 0.001, 100.0, 1.0, 0.001, format="%.3f", help="Inverse of regularization strength")
                            solver = st.selectbox("Solver:", ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"], help="Algorithm for optimization")
                            max_iter = st.slider("Max Iterations:", 100, 5000, 1000, 100, help="Maximum iterations for solver")
                            
                        elif "Naive Bayes" in manual_model:
                            if "Gaussian" in manual_model:
                                var_smoothing = st.number_input("Smoothing:", 1e-10, 1e-5, 1e-9, 1e-10, format="%.2e", help="Portion of largest variance added to variances")
                            elif "Multinomial" in manual_model:
                                alpha = st.number_input("Alpha (Smoothing):", 0.0, 10.0, 1.0, 0.1, help="Additive smoothing parameter")
                                
                        elif "K-Nearest" in manual_model or "KNN" in manual_model:
                            n_neighbors = st.slider("Number of Neighbors:", 1, 50, 5, help="Number of neighbors to use")
                            weights = st.selectbox("Weights:", ["uniform", "distance"], help="Weight function")
                            algorithm = st.selectbox("Algorithm:", ["auto", "ball_tree", "kd_tree", "brute"], help="Algorithm for nearest neighbor search")
                            metric = st.selectbox("Distance Metric:", ["euclidean", "manhattan", "chebyshev", "minkowski"], help="Distance metric")
                            
                        elif "Neural Network" in manual_model or "MLP" in manual_model:
                            hidden_layer_sizes = st.text_input("Hidden Layers:", "100,", help="Hidden layer sizes (comma-separated, e.g., '100,50,')")
                            activation = st.selectbox("Activation:", ["relu", "tanh", "logistic"], help="Activation function")
                            alpha = st.number_input("L2 Regularization:", 0.0001, 1.0, 0.0001, 0.0001, format="%.4f", help="L2 penalty parameter")
                            learning_rate_init = st.number_input("Initial Learning Rate:", 0.0001, 1.0, 0.001, 0.0001, format="%.4f")
                            
                        elif "Decision Tree" in manual_model:
                            criterion = st.selectbox("Criterion:", ["gini", "entropy", "log_loss"], help="Function to measure quality of split")
                            max_depth = st.selectbox("Max Depth:", [None, 3, 5, 10, 15, 20], index=0, help="Maximum depth of tree")
                            min_samples_split = st.slider("Min Samples Split:", 2, 20, 2, help="Minimum samples required to split")
                            min_samples_leaf = st.slider("Min Samples Leaf:", 1, 10, 1, help="Minimum samples required at leaf")
                            
                        # Feature preprocessing options
                        st.markdown("**🔄 Feature Preprocessing:**")
                        use_scaling = st.checkbox("Feature Scaling", value=True, help="StandardScaler normalization")
                        use_pca = st.checkbox("Principal Component Analysis", value=False, help="Dimensionality reduction")
                        if use_pca:
                            n_components = st.slider("PCA Components:", 2, 100, 10, help="Number of principal components")
                        
                        backbone = "N/A"  # No backbone for traditional ML
                        
                    else:
                        # Deep learning frameworks (PyTorch, TensorFlow/Keras)
                        
                        # Skip backbone selection for Custom CNN Builder
                        if "Custom CNN Builder" in manual_model:
                            backbone = "custom_cnn"
                            st.success(f"🏗️ **Custom CNN Architecture** - Building from your configuration")
                            st.info("💡 No pre-trained backbone - training from scratch with your custom design")
                        # Use the selected model as the backbone to avoid confusion
                        elif "resnet50" in manual_model.lower():
                            backbone = "resnet50"
                            st.info(f"🏗️ **Using {manual_model}** - No additional backbone selection needed")
                        elif "resnet101" in manual_model.lower():
                            backbone = "resnet101"
                            st.info(f"🏗️ **Using {manual_model}** - No additional backbone selection needed")
                        elif "efficientnet" in manual_model.lower():
                            backbone = "efficientnet"
                            st.info(f"🏗️ **Using {manual_model}** - No additional backbone selection needed")
                        elif "mobilenet" in manual_model.lower():
                            backbone = "mobilenet"
                            st.info(f"🏗️ **Using {manual_model}** - No additional backbone selection needed")
                        elif "vgg" in manual_model.lower():
                            backbone = "vgg"
                            st.info(f"🏗️ **Using {manual_model}** - No additional backbone selection needed")
                        elif "densenet" in manual_model.lower():
                            backbone = "densenet"
                            st.info(f"🏗️ **Using {manual_model}** - No additional backbone selection needed")
                        elif "inception" in manual_model.lower():
                            backbone = "inception"
                            st.info(f"🏗️ **Using {manual_model}** - No additional backbone selection needed")
                        elif "simple cnn" in manual_model.lower() or "sequential" in manual_model.lower():
                            st.error("❌ Simple CNN and Sequential CNN are deprecated. Please select EfficientNet, MobileNetV3, or RegNet for better performance and accuracy.")
                        elif "efficientnet-b0" in manual_model.lower():
                            backbone = "efficientnet_b0"
                            st.success(f"🚀 **Using {manual_model}** - State-of-the-art efficiency & accuracy!")
                            st.info("💡 **Benefits**: 40% smaller than ResNet-50 but 8% more accurate")
                        elif "mobilenetv3" in manual_model.lower():
                            backbone = "mobilenet_v3_small"
                            st.success(f"📱 **Using {manual_model}** - Optimized for mobile/edge deployment!")
                            st.info("💡 **Benefits**: 50% faster than MobileNetV2, hardware-optimized")
                        elif "regnet" in manual_model.lower():
                            backbone = "regnet_y_002"
                            st.success(f"⚡ **Using {manual_model}** - Best speed/accuracy balance!")
                            st.info("💡 **Benefits**: Better than EfficientNet at similar computational cost")
                        elif "convnext" in manual_model.lower():
                            backbone = "convnext_tiny"
                            st.success(f"🔬 **Using {manual_model}** - Modern ConvNet architecture!")
                            st.info("💡 **Benefits**: Vision Transformer performance with CNN efficiency")
                        else:
                            # For complex models that might need additional backbone specification
                            st.warning(f"⚠️ Advanced model detected: {manual_model}")
                            backbone_options = ["resnet50", "resnet101", "efficientnet", "mobilenet", "vgg", "densenet"]
                            backbone = st.selectbox("Backbone Architecture:", backbone_options, 
                                                  help="Select backbone architecture for this advanced model")
                    
                    # Input size customization with small image support
                    input_sizes = [
                        (28, 28),    # MNIST
                        (32, 32),    # CIFAR
                        (64, 64),    # Small images
                        (128, 128),  # Medium images
                        (224, 224),  # ImageNet standard
                        (256, 256),  # High resolution
                        (384, 384),  # Very high resolution
                        (512, 512)   # Ultra high resolution
                    ]
                    size_labels = [f"{w}x{h}" + (" (MNIST)" if w==28 else " (CIFAR)" if w==32 else " (ImageNet)" if w==224 else "") for w, h in input_sizes]
                    
                    # Auto-detect from dataset if available
                    default_idx = 4  # Default to 224x224
                    if hasattr(dataset_info, 'image_stats'):
                        detected_size = dataset_info.image_stats.get('avg_size', (224, 224))
                        detected_w, detected_h = detected_size
                        
                        # Find closest standard size
                        closest_idx = 4  # Default to ImageNet
                        min_diff = float('inf')
                        for i, (w, h) in enumerate(input_sizes):
                            diff = abs(w - detected_w) + abs(h - detected_h)
                            if diff < min_diff:
                                min_diff = diff
                                closest_idx = i
                        
                        default_idx = closest_idx
                        
                        # Show dataset type hint if available
                        dataset_hint = dataset_info.image_stats.get('dataset_type_hint', '')
                        if dataset_hint:
                            st.info(f"🔍 Detected: {detected_w}×{detected_h} **{dataset_hint}** → suggesting {input_sizes[closest_idx][0]}×{input_sizes[closest_idx][1]}")
                        else:
                            st.info(f"🔍 Detected image size: {detected_w}×{detected_h}, suggesting {input_sizes[closest_idx][0]}×{input_sizes[closest_idx][1]}")
                    
                    # Input size configuration (only for deep learning frameworks)
                    if framework in ["PyTorch", "TensorFlow/Keras"]:
                        selected_size_idx = st.selectbox("Input Resolution:", range(len(size_labels)), format_func=lambda x: size_labels[x], index=default_idx)
                        input_size = input_sizes[selected_size_idx]
                    else:
                        # For Scikit-learn, input size doesn't apply to traditional ML
                        input_size = (None, None)  # Placeholder
                        st.info("ℹ️ Input resolution not applicable for traditional ML algorithms")
                    
                with col2_manual:
                    if framework == "Scikit-learn":
                        # Scikit-learn specific feature engineering and validation options
                        st.markdown("**📊 Feature Engineering:**")
                        
                        # Feature extraction method for images
                        feature_method = st.selectbox(
                            "Feature Extraction Method:",
                            ["Histogram Features", "HOG (Histogram of Gradients)", "LBP (Local Binary Patterns)", 
                             "Color Moments", "Texture Features", "Statistical Features"],
                            help="Method to extract features from images"
                        )
                        
                        # Image preprocessing
                        image_size = st.selectbox(
                            "Image Resize:",
                            [(32, 32), (64, 64), (128, 128), (224, 224)],
                            index=1,
                            format_func=lambda x: f"{x[0]}×{x[1]}",
                            help="Resize images before feature extraction"
                        )
                        
                        # Feature selection
                        feature_selection = st.selectbox(
                            "Feature Selection:",
                            ["All Features", "Select K Best", "Recursive Feature Elimination", "Variance Threshold"],
                            help="Method to select most relevant features"
                        )
                        
                        if feature_selection == "Select K Best":
                            k_features = st.slider("Number of Features:", 10, 1000, 100, 10, help="Number of top features to select")
                        elif feature_selection == "Variance Threshold":
                            variance_threshold = st.number_input("Variance Threshold:", 0.0, 1.0, 0.01, 0.01, format="%.3f")
                        
                        st.markdown("**🎯 Model Validation:**")
                        
                        # Cross-validation settings
                        cv_folds = st.slider("Cross-Validation Folds:", 3, 10, 5, help="Number of folds for cross-validation")
                        
                        # Validation metrics
                        scoring_metrics = st.multiselect(
                            "Scoring Metrics:",
                            ["accuracy", "precision", "recall", "f1", "roc_auc", "balanced_accuracy"],
                            default=["accuracy", "f1"],
                            help="Metrics to evaluate model performance"
                        )
                        
                        # Class balancing
                        class_weight = st.selectbox(
                            "Class Weight:",
                            ["None", "balanced", "balanced_subsample"],
                            help="Handle class imbalance"
                        )
                        
                        # Random state for reproducibility
                        random_state = st.number_input("Random State:", 0, 1000, 42, help="Seed for reproducible results")
                        
                        input_channels = None
                        use_pretrained = False
                        
                    else:
                        # Deep learning frameworks (PyTorch, TensorFlow/Keras)
                        channel_options = {
                            "Auto-detect": "auto",
                            "Grayscale (1 channel)": 1,
                            "RGB (3 channels)": 3,
                            "RGBA (4 channels)": 4,
                            "Custom": "custom"
                        }
                        
                        channel_selection = st.selectbox(
                            "Input Channels:", 
                            list(channel_options.keys()),
                            index=0,
                            help="Select the number of input channels for your images"
                        )
                        
                        input_channels = channel_options[channel_selection]
                        
                        # Custom channel input
                        if input_channels == "custom":
                            input_channels = st.number_input("Custom Channel Count:", min_value=1, max_value=16, value=3, step=1)
                        
                        # Auto-detect from dataset if available
                        if input_channels == "auto" and hasattr(dataset_info, 'image_stats'):
                            detected_channels = dataset_info.image_stats.get('channels', 3)
                            st.info(f"🔍 Detected: {detected_channels} channels from dataset")
                            input_channels = detected_channels
                        elif input_channels == "auto":
                            input_channels = 3  # Default to RGB
                            st.info("ℹ️ Using default RGB (3 channels)")
                        
                        # Pretrained weights
                        use_pretrained = st.checkbox("Use Pretrained Weights", value=True, help="Start with ImageNet pretrained weights (works best with 3-channel RGB)")
                    
                    # Memory budget (applicable to all frameworks)
                    memory_budget = st.slider("Memory Budget (GB):", 1.0, 16.0, 4.0, 0.5, help="Adjust model complexity based on available memory")
                
                # Advanced manual configuration
                with st.expander("⚙️ Advanced Manual Configuration"):
                    adv_col1, adv_col2 = st.columns(2)
                    
                    with adv_col1:
                        learning_rate = st.number_input("Learning Rate:", 0.0001, 1.0, 0.001, 0.0001, format="%.4f")
                        batch_size = st.selectbox("Batch Size:", [8, 16, 32, 64, 128], index=2)
                        optimizer = st.selectbox("Optimizer:", ["Adam", "SGD", "RMSprop", "AdamW"])
                    
                    with adv_col2:
                        dropout_rate = st.slider("Dropout Rate:", 0.0, 0.8, 0.2, 0.05)
                        weight_decay = st.number_input("Weight Decay:", 0.0, 0.01, 0.0001, 0.00001, format="%.5f")
                        epochs = st.number_input("Training Epochs:", 1, 1000, 50, 1)
                
                # Data Processing Options
                with st.expander("📊 Data Processing Options"):
                    data_col1, data_col2 = st.columns(2)
                    
                    with data_col1:
                        st.subheader("🔧 Normalization Settings")
                        enable_normalization = st.checkbox("Enable Data Normalization", value=True, 
                                                          help="Normalize pixel values to [0,1] range by dividing by 255")
                        
                        if enable_normalization:
                            # Check if ResNet or MobileNet is selected for model-specific preprocessing
                            model_specific_options = []
                            if "ResNet" in manual_model:
                                if framework == "TensorFlow/Keras":
                                    model_specific_options.append("ResNet Preprocessing (TensorFlow)")
                                elif framework == "PyTorch":
                                    model_specific_options.append("ResNet Preprocessing (PyTorch)")
                            
                            if "MobileNet" in manual_model:
                                if framework == "TensorFlow/Keras":
                                    model_specific_options.append("MobileNet Preprocessing (TensorFlow)")
                                elif framework == "PyTorch":
                                    model_specific_options.append("MobileNet Preprocessing (PyTorch)")
                            
                            # Base normalization options
                            base_options = ["Standard (0-1)", "Z-Score (-1 to 1)", "Custom Range"]
                            all_options = base_options + model_specific_options
                            
                            normalization_type = st.selectbox(
                                "Normalization Type:",
                                all_options,
                                help="Choose normalization method. Model-specific preprocessing uses optimized parameters for the selected architecture."
                            )
                            
                            # Show preprocessing details for model-specific options
                            if "ResNet Preprocessing" in normalization_type:
                                st.info("🧠 **ResNet Preprocessing**: Uses ImageNet statistics - Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]")
                            elif "MobileNet Preprocessing" in normalization_type:
                                if framework == "TensorFlow/Keras":
                                    st.info("🧠 **MobileNet Preprocessing**: TensorFlow preprocessing with range [-1, 1] normalization")
                                else:
                                    st.info("🧠 **MobileNet Preprocessing**: PyTorch preprocessing with ImageNet statistics")
                            
                            if normalization_type == "Custom Range":
                                norm_min = st.number_input("Min Value:", -2.0, 2.0, 0.0, 0.1, format="%.1f")
                                norm_max = st.number_input("Max Value:", -2.0, 2.0, 1.0, 0.1, format="%.1f")
                        else:
                            normalization_type = "None"
                            norm_min, norm_max = 0.0, 255.0
                    
                    with data_col2:
                        st.subheader("🎨 Color Space Settings")
                        st.info("💡 These options apply only to folder-based datasets loaded with CV2")
                        
                        use_cv2_loading = st.checkbox("Force CV2 Loading", value=True,
                                                     help="Use OpenCV (CV2) to load images from folders instead of PIL")
                        
                        if use_cv2_loading:
                            color_conversion = st.selectbox(
                                "Color Space Conversion:",
                                ["None", "BGR2RGB", "RGB2BGR", "BGR2GRAY", "RGB2GRAY", "GRAY2RGB", "GRAY2BGR"],
                                index=1,  # Default to BGR2RGB since CV2 loads as BGR
                                help="Convert color space after loading with CV2"
                            )
                            
                            interpolation_method = st.selectbox(
                                "Resize Interpolation:",
                                ["INTER_LINEAR", "INTER_CUBIC", "INTER_AREA", "INTER_NEAREST"],
                                help="CV2 interpolation method for resizing images"
                            )
                        else:
                            color_conversion = "None"
                            interpolation_method = "INTER_LINEAR"
                
                # Create and apply manual configuration
                if st.button("🔧 Apply Manual Configuration", type="primary"):
                    
                    # Check if Custom CNN Builder is selected
                    if custom_cnn_config is not None:
                        # Handle Custom CNN Builder
                        from core.model_selector import ModelConfig
                        
                        st.success("🏗️ **Custom CNN Architecture Created!**")
                        st.info(f"✅ Channel-adaptive architecture for {custom_cnn_config['detected_channels']} channel(s)")
                        
                        # Create model config for custom CNN
                        model_config = ModelConfig(
                            architecture="Custom CNN Builder",
                            backbone="custom_cnn",
                            num_parameters=custom_cnn_config.get('estimated_params', 1000000),
                            memory_requirements=custom_cnn_config.get('estimated_memory', 1.0),
                            estimated_flops=500000000,  # Rough estimate
                            input_size=input_size if isinstance(input_size, tuple) else (224, 224),
                            pretrained=False,  # Custom models start from scratch
                            framework="TensorFlow",
                            config_params={
                                'is_custom_cnn': True,
                                'custom_cnn_config': custom_cnn_config,
                                'num_classes': dataset_info.num_classes,
                                'input_channels': custom_cnn_config['detected_channels'],
                                'learning_rate': custom_cnn_config['learning_rate'],
                                'optimizer': custom_cnn_config['optimizer'],
                                'dropout': custom_cnn_config['dropout_rate'],
                                'activation': custom_cnn_config['activation'],
                                'is_manual_config': True
                            }
                        )
                        
                        st.session_state.model_config = model_config
                        st.session_state.manual_config_in_progress = True
                        
                        st.success("✅ Custom CNN configuration applied! Review and proceed to training.")
                        
                        # Show summary
                        st.markdown("### 📋 Configuration Summary")
                        summary_col1, summary_col2 = st.columns(2)
                        with summary_col1:
                            st.write(f"**Architecture:** Custom CNN")
                            st.write(f"**Conv Blocks:** {len(custom_cnn_config['conv_blocks'])}")
                            st.write(f"**Dense Layers:** {len(custom_cnn_config['dense_layers'])}")
                            st.write(f"**Estimated Parameters:** {custom_cnn_config.get('estimated_params', 0):,}")
                        with summary_col2:
                            st.write(f"**Input Channels:** {custom_cnn_config['detected_channels']}")
                            st.write(f"**Optimizer:** {custom_cnn_config['optimizer']}")
                            st.write(f"**Learning Rate:** {custom_cnn_config['learning_rate']}")
                            _mem_mb = custom_cnn_config.get('estimated_memory', 0) * 1024  # GB → MB
                            st.write(f"**Estimated Memory:** {_mem_mb:.1f} MB")
                        
                        # Continue button
                        if st.button("➡️ Continue to Training", key="custom_cnn_continue", type="primary"):
                            st.session_state.current_step = 3
                            st.session_state["current_tool"] = None
                            st.session_state.manual_config_in_progress = False
                            st_rerun()
                        
                        return  # Exit early to prevent running the rest of the manual config logic
                    
                    # Create selector for manual configuration
                    selector = ModelSelector()
                    
                    # Get base model info or create configurations
                    if manual_model == "Simple CNN":
                        st.error("❌ Simple CNN and Sequential CNN are deprecated. Please select EfficientNet, MobileNetV3, or RegNet for better performance and accuracy.")
                    elif "🚀 EfficientNet-B0" in manual_model:
                        # EfficientNet-B0: Best overall replacement
                        model_info = {
                            'name': 'EfficientNet-B0',
                            'description': 'State-of-the-art compound scaling model - 40% smaller than ResNet-50 but 8% more accurate',
                            'architecture': 'EfficientNet-B0',
                            'backbone': 'efficientnet_b0',
                            'num_parameters': 5300000,  # 5.3M parameters
                            'input_size': get_user_specified_input_size(),
                            'flops': 390000000,  # 390M FLOPs
                            'memory_gb': 1.3,
                            'accuracy_score': 0.95,
                            'speed_score': 0.85,
                            'complexity': 'Medium',
                            'accuracy': 'Excellent',
                            'speed': 'Good',
                            'pretrained': True
                        }
                    elif "📱 MobileNetV3-Small" in manual_model:
                        # MobileNetV3-Small: Best for mobile/edge
                        model_info = {
                            'name': 'MobileNetV3-Small',
                            'description': 'Hardware-optimized neural architecture - 50% faster than MobileNetV2',
                            'architecture': 'MobileNetV3-Small',
                            'backbone': 'mobilenet_v3_small',
                            'num_parameters': 2500000,  # 2.5M parameters
                            'input_size': get_user_specified_input_size(),
                            'flops': 66000000,  # 66M FLOPs
                            'memory_gb': 0.8,
                            'accuracy_score': 0.92,
                            'speed_score': 0.98,
                            'complexity': 'Low',
                            'accuracy': 'Very Good',
                            'speed': 'Excellent',
                            'pretrained': True
                        }
                    elif "⚡ RegNetY-002" in manual_model:
                        # RegNet: Best balance of speed & accuracy
                        model_info = {
                            'name': 'RegNetY-002',
                            'description': 'Design space optimized model - Better than EfficientNet at similar computational cost',
                            'architecture': 'RegNetY-002',
                            'backbone': 'regnet_y_002',
                            'num_parameters': 3200000,  # 3.2M parameters
                            'input_size': get_user_specified_input_size(),
                            'flops': 200000000,  # 200M FLOPs
                            'memory_gb': 1.0,
                            'accuracy_score': 0.94,
                            'speed_score': 0.90,
                            'complexity': 'Medium',
                            'accuracy': 'Excellent',
                            'speed': 'Very Good',
                            'pretrained': True
                        }
                    elif "⚡ ConvNeXt-Tiny" in manual_model:
                        # ConvNeXt: Modern ConvNet architecture
                        model_info = {
                            'name': 'ConvNeXt-Tiny',
                            'description': 'Modern ConvNet with Vision Transformer performance but CNN efficiency',
                            'architecture': 'ConvNeXt-Tiny',
                            'backbone': 'convnext_tiny',
                            'num_parameters': 28600000,  # 28.6M parameters
                            'input_size': get_user_specified_input_size(),
                            'flops': 4460000000,  # 4.46G FLOPs
                            'memory_gb': 2.2,
                            'accuracy_score': 0.96,
                            'speed_score': 0.75,
                            'complexity': 'High',
                            'accuracy': 'Outstanding',
                            'speed': 'Moderate',
                            'pretrained': True
                        }
                    elif dataset_info.task_type == "classification":
                        # Use first available model as fallback if manual_model not found
                        fallback_model = list(selector.classification_models.values())[0]
                        model_info = selector.classification_models.get(manual_model, fallback_model)
                    elif dataset_info.task_type == "detection":
                        model_info = selector.detection_models.get(manual_model, list(selector.detection_models.values())[0])
                    else:
                        model_info = selector.segmentation_models.get(manual_model, list(selector.segmentation_models.values())[0])
                    
                    # Create customized config
                    from core.model_selector import ModelConfig
                    
                    # Framework-specific configuration parameters
                    if framework == "Scikit-learn":
                        # Build comprehensive Scikit-learn configuration
                        custom_config_params = {
                            'learning_rate': learning_rate,
                            'memory_budget': memory_budget,
                            'is_manual_config': True,
                            'framework': 'scikit-learn',
                            'algorithm': manual_model,
                            
                            # Feature engineering parameters
                            'feature_method': locals().get('feature_method', 'Histogram Features'),
                            'image_size': locals().get('image_size', (64, 64)),
                            'feature_selection': locals().get('feature_selection', 'All Features'),
                            'use_scaling': locals().get('use_scaling', True),
                            'use_pca': locals().get('use_pca', False),
                            
                            # Validation parameters
                            'cv_folds': locals().get('cv_folds', 5),
                            'scoring_metrics': locals().get('scoring_metrics', ['accuracy']),
                            'class_weight': locals().get('class_weight', 'None'),
                            'random_state': locals().get('random_state', 42),
                            
                            # Data processing parameters
                            'enable_normalization': enable_normalization,
                            'normalization_type': normalization_type,
                            'norm_min': norm_min if normalization_type == "Custom Range" else None,
                            'norm_max': norm_max if normalization_type == "Custom Range" else None,
                            'use_cv2_loading': use_cv2_loading,
                            'color_conversion': color_conversion,
                            'interpolation_method': interpolation_method,
                        }
                        
                        # Add algorithm-specific parameters
                        if "Random Forest" in manual_model or "Extra Trees" in manual_model:
                            custom_config_params.update({
                                'n_estimators': locals().get('n_estimators', 100),
                                'max_depth': locals().get('max_depth', None),
                                'min_samples_split': locals().get('min_samples_split', 2),
                                'min_samples_leaf': locals().get('min_samples_leaf', 1)
                            })
                        elif "SVM" in manual_model or "Support Vector" in manual_model:
                            custom_config_params.update({
                                'C': locals().get('C', 1.0),
                                'kernel': locals().get('kernel', 'rbf'),
                                'degree': locals().get('degree', 3) if locals().get('kernel') == 'poly' else None,
                                'gamma': locals().get('gamma_value', 'scale') if locals().get('gamma') == 'custom' else locals().get('gamma', 'scale')
                            })
                        elif "Gradient Boosting" in manual_model or "AdaBoost" in manual_model:
                            custom_config_params.update({
                                'n_estimators': locals().get('n_estimators', 100),
                                'learning_rate': locals().get('learning_rate', 0.1),
                                'max_depth': locals().get('max_depth', 3)
                            })
                        elif "Logistic Regression" in manual_model or "Ridge" in manual_model:
                            custom_config_params.update({
                                'C': locals().get('C', 1.0),
                                'solver': locals().get('solver', 'lbfgs'),
                                'max_iter': locals().get('max_iter', 1000)
                            })
                        elif "Naive Bayes" in manual_model:
                            if "Gaussian" in manual_model:
                                custom_config_params['var_smoothing'] = locals().get('var_smoothing', 1e-9)
                            elif "Multinomial" in manual_model:
                                custom_config_params['alpha'] = locals().get('alpha', 1.0)
                        elif "K-Nearest" in manual_model:
                            custom_config_params.update({
                                'n_neighbors': locals().get('n_neighbors', 5),
                                'weights': locals().get('weights', 'uniform'),
                                'algorithm': locals().get('algorithm', 'auto'),
                                'metric': locals().get('metric', 'euclidean')
                            })
                        elif "Neural Network" in manual_model or "MLP" in manual_model:
                            hidden_layers_str = locals().get('hidden_layer_sizes', '100,')
                            try:
                                hidden_layers = tuple(int(x.strip()) for x in hidden_layers_str.split(',') if x.strip())
                            except:
                                hidden_layers = (100,)
                            custom_config_params.update({
                                'hidden_layer_sizes': hidden_layers,
                                'activation': locals().get('activation', 'relu'),
                                'alpha': locals().get('alpha', 0.0001),
                                'learning_rate_init': locals().get('learning_rate_init', 0.001)
                            })
                        elif "Decision Tree" in manual_model:
                            custom_config_params.update({
                                'criterion': locals().get('criterion', 'gini'),
                                'max_depth': locals().get('max_depth', None),
                                'min_samples_split': locals().get('min_samples_split', 2),
                                'min_samples_leaf': locals().get('min_samples_leaf', 1)
                            })
                        
                        # PCA parameters if enabled
                        if locals().get('use_pca', False):
                            custom_config_params['n_components'] = locals().get('n_components', 10)
                        
                        # Feature selection parameters
                        if locals().get('feature_selection') == "Select K Best":
                            custom_config_params['k_features'] = locals().get('k_features', 100)
                        elif locals().get('feature_selection') == "Variance Threshold":
                            custom_config_params['variance_threshold'] = locals().get('variance_threshold', 0.01)
                        
                        # For Scikit-learn, input size is feature-based
                        full_input_size = "Feature Vector"
                    else:
                        # Deep learning frameworks (PyTorch, TensorFlow/Keras)
                        custom_config_params = {
                            'learning_rate': learning_rate,
                            'batch_size': batch_size,
                            'dropout_rate': dropout_rate,
                            'optimizer': optimizer,
                            'weight_decay': weight_decay,
                            'epochs': epochs,
                            'custom_backbone': backbone,
                            'memory_budget': memory_budget,
                            'input_channels': input_channels,
                            'channel_adaptation': 'auto' if input_channels and input_channels != 3 else 'none',
                            # Store custom dimensions for override detection (only for DL frameworks)
                            'custom_height': input_size[1] if input_size[0] is not None else None,
                            'custom_width': input_size[0] if input_size[0] is not None else None,
                            'is_manual_config': True,
                            
                            # Data processing parameters
                            'enable_normalization': enable_normalization,
                            'normalization_type': normalization_type,
                            'norm_min': norm_min if normalization_type == "Custom Range" else None,
                            'norm_max': norm_max if normalization_type == "Custom Range" else None,
                            'use_cv2_loading': use_cv2_loading,
                            'color_conversion': color_conversion,
                            'interpolation_method': interpolation_method,
                        }
                        # Store input size as (channels, width, height) for backward compatibility
                        full_input_size = (input_channels, input_size[0], input_size[1]) if input_channels and input_size[0] is not None else (3, 224, 224)
                    
                    model_config = ModelConfig(
                        architecture=manual_model,
                        backbone=backbone,
                        num_parameters=model_info['num_parameters'],
                        input_size=full_input_size,
                        pretrained=use_pretrained and input_channels == 3 if framework != "Scikit-learn" else False,
                        config_params=custom_config_params,
                        estimated_flops=model_info.get('flops', 0),
                        memory_requirements=memory_budget
                    )
                    
                    st.session_state.model_config = model_config
                    
                    # Framework-specific success message
                    if framework == "Scikit-learn":
                        st.success(f"✅ Manual Configuration Applied: **{manual_model}** algorithm")
                    else:
                        st.success(f"✅ Manual Configuration Applied: **{manual_model}** with **{backbone}** backbone")
                    st_rerun()  # Refresh to show the configuration editor and continue button
            else:
                st.error(f"❌ No models available for task type: {dataset_info.task_type}")
                st.info("💡 Try running dataset analysis first or check if the task type is supported.")
        
        # Display and edit model details if config exists
        if hasattr(st.session_state, 'model_config') and st.session_state.model_config is not None:
            model_config = st.session_state.model_config
            
            st.subheader("📝 Editable Model Details")
            
            # Create editable form
            with st.form("model_config_form"):
                st.markdown("**Edit Model Configuration:**")
                
                # Architecture (read-only for context)
                st.text_input("Architecture", value=model_config.architecture, disabled=True, help="Model architecture (read-only)")
                
                # Framework-aware backbone configuration
                framework = st.session_state.get('selected_framework', 'PyTorch')
                
                if framework == "Scikit-learn":
                    # For Scikit-learn, backbone doesn't apply - show info instead
                    st.info("ℹ️ Backbone architecture not applicable for traditional ML algorithms")
                    new_backbone = "N/A"  # Use N/A for Scikit-learn
                else:
                    # Editable backbone (None option for Simple CNN)
                    backbone_options = ["None", "resnet50", "resnet101", "efficientnet", "mobilenet", "vgg", "custom"]
                    current_backbone_idx = backbone_options.index(model_config.backbone) if model_config.backbone in backbone_options else 0
                    new_backbone = st.selectbox("Backbone", backbone_options, index=current_backbone_idx, 
                                              help="'None' means no pretrained backbone (e.g., Simple CNN)")
                
                # Editable input configuration
                col_ch, col_w, col_h = st.columns(3)
                
                # Framework-aware channel configuration
                with col_ch:
                    if framework == "Scikit-learn":
                        # For Scikit-learn, channels don't apply in the same way
                        st.info("ℹ️ Input channels not applicable for traditional ML (features are extracted)")
                        new_channels = None  # Not applicable for Scikit-learn
                        current_width = None
                        current_height = None
                    else:
                        # Extract current channels (handle both 2D and 3D input_size)
                        if isinstance(model_config.input_size, tuple) and len(model_config.input_size) == 3:
                            # Check if it's (channels, height, width) or (height, width, channels)
                            if model_config.input_size[0] <= 4:  # Likely channels first
                                current_channels = model_config.input_size[0]
                                current_height = model_config.input_size[1]  
                                current_width = model_config.input_size[2]
                            else:  # Likely (height, width, channels)
                                current_height = model_config.input_size[0]
                                current_width = model_config.input_size[1]
                                current_channels = model_config.input_size[2]
                        else:
                            # Legacy format (width, height) - assume RGB
                            current_channels = model_config.config_params.get('input_channels', 3)
                            current_width = model_config.input_size[0] if model_config.input_size[0] is not None else 224
                            current_height = model_config.input_size[1] if model_config.input_size[1] is not None else 224
                        
                        # Ensure valid values for number inputs (avoid API exception)
                        # Convert to int and handle non-numeric values
                        try:
                            current_width = int(current_width) if current_width is not None else 224
                            current_width = max(8, current_width)
                        except (ValueError, TypeError):
                            current_width = 224
                        
                        try:
                            current_height = int(current_height) if current_height is not None else 224
                            current_height = max(8, current_height)
                        except (ValueError, TypeError):
                            current_height = 224
                        
                        try:
                            current_channels = int(current_channels) if current_channels is not None else 3
                            current_channels = max(1, min(16, current_channels))
                        except (ValueError, TypeError):
                            current_channels = 3
                        
                        new_channels = st.selectbox(
                            "Input Channels",
                            [1, 3, 4],
                            index=[1, 3, 4].index(current_channels) if current_channels in [1, 3, 4] else 1,
                            format_func=lambda x: f"{x} ({'Grayscale' if x==1 else 'RGB' if x==3 else 'RGBA'})"
                        )
                
                with col_w:
                    if framework == "Scikit-learn":
                        st.info("ℹ️ Input resolution not applicable for traditional ML")
                        new_width = None
                    else:
                        new_width = st.number_input("Width", min_value=8, max_value=1024, value=current_width, 
                                                  help="Supports small images (MNIST: 28, CIFAR: 32) to large (1024)")
                with col_h:
                    if framework == "Scikit-learn":
                        st.info("ℹ️ Feature extraction handles image dimensions automatically")
                        new_height = None
                    else:
                        new_height = st.number_input("Height", min_value=8, max_value=1024, value=current_height,
                                                   help="Common sizes: 28 (MNIST), 32 (CIFAR), 224 (ImageNet)")
                
                # Framework-aware image size info (only for deep learning)
                if framework != "Scikit-learn" and new_width is not None and new_height is not None:
                    if new_width <= 32 or new_height <= 32:
                        if new_width == 28 and new_height == 28:
                            st.success("📊 MNIST-style configuration (28×28) - Perfect for digit recognition!")
                        elif new_width == 32 and new_height == 32:
                            st.success("📊 CIFAR-style configuration (32×32) - Great for small object classification!")
                        else:
                            st.info("📊 Small image configuration - Higher batch sizes recommended")
                    elif new_width >= 512 or new_height >= 512:
                        st.warning("📊 Large image configuration - Consider reducing batch size")
                
                # Framework-aware channel adaptation info (only for deep learning)
                if framework != "Scikit-learn" and new_channels is not None and new_channels != 3:
                    if new_channels == 1:
                        st.info("🔘 Grayscale mode: Model will adapt for single-channel input")
                    elif new_channels == 4:
                        st.info("🎨 RGBA mode: Model will handle alpha channel")
                    
                    # Pretrained warning
                    st.warning("⚠️ Pretrained weights work best with 3-channel RGB images")
                
                # Framework-aware pretrained configuration
                if framework == "Scikit-learn":
                    st.info("ℹ️ Pretrained weights not applicable for traditional ML algorithms")
                    new_pretrained = False
                else:
                    # Pretrained option with channel-aware logic
                    pretrained_help = "Pretrained weights (works best with RGB images)" if new_channels == 3 else "Pretrained weights (may need adaptation for non-RGB)"
                    new_pretrained = st.checkbox("Use Pretrained Weights", value=model_config.pretrained and new_channels == 3, help=pretrained_help)
                
                # Memory requirements (editable for custom models)
                current_memory = max(0.5, min(32.0, model_config.memory_requirements))  # Ensure valid range
                new_memory = st.number_input("Memory Requirements (GB)", min_value=0.5, max_value=32.0, value=current_memory, step=0.1)
                
                # Advanced configuration parameters
                st.markdown("**Advanced Parameters:**")
                
                # Common editable parameters
                col1_adv, col2_adv = st.columns(2)
                
                with col1_adv:
                    if framework == "Scikit-learn":
                        # Scikit-learn specific advanced parameters
                        current_cv = model_config.config_params.get('cv_folds', 5)
                        new_cv = st.slider("Cross-Validation Folds", 3, 10, current_cv, help="Number of CV folds")
                        
                        current_random = model_config.config_params.get('random_state', 42)
                        new_random = st.number_input("Random State", 0, 1000, current_random, help="Random seed for reproducibility")
                        
                        current_scaling = model_config.config_params.get('use_scaling', True)
                        new_scaling = st.checkbox("Feature Scaling", current_scaling, help="Apply StandardScaler")
                        
                        # Algorithm-specific parameters based on architecture
                        if "Random Forest" in model_config.architecture or "Extra Trees" in model_config.architecture:
                            current_estimators = model_config.config_params.get('n_estimators', 100)
                            new_estimators = st.slider("Number of Trees", 10, 500, current_estimators, 10)
                            
                            current_depth = model_config.config_params.get('max_depth', None)
                            depth_options = [None, 5, 10, 15, 20, 25]
                            depth_index = depth_options.index(current_depth) if current_depth in depth_options else 0
                            new_depth = st.selectbox("Max Depth", depth_options, index=depth_index)
                            
                        elif "SVM" in model_config.architecture:
                            current_C = model_config.config_params.get('C', 1.0)
                            new_C = st.number_input("Regularization (C)", 0.001, 100.0, current_C, 0.001, format="%.3f")
                            
                            current_kernel = model_config.config_params.get('kernel', 'rbf')
                            new_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], 
                                                    index=["rbf", "linear", "poly", "sigmoid"].index(current_kernel) if current_kernel in ["rbf", "linear", "poly", "sigmoid"] else 0)
                        
                        elif "Gradient Boosting" in model_config.architecture:
                            current_estimators = model_config.config_params.get('n_estimators', 100)
                            new_estimators = st.slider("Number of Estimators", 50, 300, current_estimators, 10)
                            
                            current_lr = model_config.config_params.get('learning_rate', 0.1)
                            # Ensure learning rate is within valid range for Gradient Boosting (0.01 to 2.0)
                            current_lr = max(0.01, min(2.0, current_lr))
                            new_lr = st.number_input("Learning Rate", 0.01, 2.0, current_lr, 0.01, format="%.2f")
                            
                        else:
                            # Generic parameters for other algorithms
                            new_lr = 0.001  # Default for compatibility
                            
                    else:
                        # Deep learning frameworks
                        current_lr = model_config.config_params.get('learning_rate', 0.001)
                        current_lr = max(0.0001, min(1.0, current_lr))  # Ensure valid range
                        new_lr = st.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=current_lr, format="%.4f", step=0.0001)
                        
                        # Batch size
                        current_batch = model_config.config_params.get('batch_size', 32)
                        new_batch = st.selectbox("Batch Size", [8, 16, 32, 64, 128], index=[8, 16, 32, 64, 128].index(current_batch) if current_batch in [8, 16, 32, 64, 128] else 2)
                        
                        # Dropout rate
                        current_dropout = model_config.config_params.get('dropout_rate', 0.2)
                        new_dropout = st.slider("Dropout Rate", 0.0, 0.8, current_dropout, 0.05)
                
                with col2_adv:
                    if framework == "Scikit-learn":
                        # Scikit-learn specific validation and feature parameters
                        current_feature_method = model_config.config_params.get('feature_method', 'Histogram Features')
                        feature_methods = ["Histogram Features", "HOG (Histogram of Gradients)", "LBP (Local Binary Patterns)", "Color Moments"]
                        feature_idx = feature_methods.index(current_feature_method) if current_feature_method in feature_methods else 0
                        new_feature_method = st.selectbox("Feature Method", feature_methods, index=feature_idx)
                        
                        current_class_weight = model_config.config_params.get('class_weight', 'None')
                        class_weight_options = ["None", "balanced", "balanced_subsample"]
                        weight_idx = class_weight_options.index(current_class_weight) if current_class_weight in class_weight_options else 0
                        new_class_weight = st.selectbox("Class Weight", class_weight_options, index=weight_idx)
                        
                        current_pca = model_config.config_params.get('use_pca', False)
                        new_pca = st.checkbox("Use PCA", current_pca, help="Principal Component Analysis")
                        
                        if new_pca:
                            current_components = model_config.config_params.get('n_components', 10)
                            new_components = st.slider("PCA Components", 2, 100, current_components)
                        else:
                            new_components = None
                        
                        # Scoring metrics
                        current_metrics = model_config.config_params.get('scoring_metrics', ['accuracy'])
                        new_metrics = st.multiselect(
                            "Scoring Metrics",
                            ["accuracy", "precision", "recall", "f1", "roc_auc"],
                            default=current_metrics if isinstance(current_metrics, list) else ['accuracy']
                        )
                        
                    else:
                        # Deep learning frameworks
                        optimizers = ["Adam", "SGD", "RMSprop", "AdamW"]
                        current_optimizer = model_config.config_params.get('optimizer', 'Adam')
                        optimizer_idx = optimizers.index(current_optimizer) if current_optimizer in optimizers else 0
                        new_optimizer = st.selectbox("Optimizer", optimizers, index=optimizer_idx)
                        
                        # Weight decay
                        current_wd = model_config.config_params.get('weight_decay', 0.0001)
                        current_wd = max(0.0, min(0.01, current_wd))  # Ensure valid range
                        new_wd = st.number_input("Weight Decay", min_value=0.0, max_value=0.01, value=current_wd, format="%.5f", step=0.00001)
                        
                        # Number of epochs
                        current_epochs = model_config.config_params.get('epochs', 50)
                        current_epochs = max(1, min(1000, current_epochs))  # Ensure valid range
                        new_epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=current_epochs, step=1)
                
                # Submit button
                submitted = st.form_submit_button("💾 Update Model Configuration", type="primary")
                
                if submitted:
                    # EXTENSIVE LOGGING TO DEBUG THE BUG
                    print("=" * 80)
                    
                    # Set manual config flag to prevent auto-navigation to Training
                    st.session_state.manual_config_in_progress = True
                    
                    # Update model config with new values
                    from core.model_selector import ModelConfig
                    
                    # Framework-specific parameter updates
                    if framework == "Scikit-learn":
                        # Scikit-learn specific parameters with comprehensive update
                        updated_config_params = {
                            **model_config.config_params,
                            'framework': 'scikit-learn',
                            'model_type': model_config.architecture.lower().replace(' ', '_'),
                            'cv_folds': locals().get('new_cv', model_config.config_params.get('cv_folds', 5)),
                            'random_state': locals().get('new_random', model_config.config_params.get('random_state', 42)),
                            'use_scaling': locals().get('new_scaling', model_config.config_params.get('use_scaling', True)),
                            'feature_method': locals().get('new_feature_method', model_config.config_params.get('feature_method', 'Histogram Features')),
                            'class_weight': locals().get('new_class_weight', model_config.config_params.get('class_weight', 'None')),
                            'use_pca': locals().get('new_pca', model_config.config_params.get('use_pca', False)),
                            'scoring_metrics': locals().get('new_metrics', model_config.config_params.get('scoring_metrics', ['accuracy']))
                        }
                        
                        # Add PCA components if enabled
                        if locals().get('new_pca', False):
                            updated_config_params['n_components'] = locals().get('new_components', 10)
                        
                        # Add algorithm-specific parameters
                        if "Random Forest" in model_config.architecture or "Extra Trees" in model_config.architecture:
                            if 'new_estimators' in locals():
                                updated_config_params['n_estimators'] = locals()['new_estimators']
                            if 'new_depth' in locals():
                                updated_config_params['max_depth'] = locals()['new_depth']
                        elif "SVM" in model_config.architecture:
                            if 'new_C' in locals():
                                updated_config_params['C'] = locals()['new_C']
                            if 'new_kernel' in locals():
                                updated_config_params['kernel'] = locals()['new_kernel']
                        elif "Gradient Boosting" in model_config.architecture:
                            if 'new_estimators' in locals():
                                updated_config_params['n_estimators'] = locals()['new_estimators']
                            if 'new_lr' in locals():
                                updated_config_params['learning_rate'] = locals()['new_lr']
                        
                        # For Scikit-learn, input size is feature-based
                        full_input_size = "Feature Vector"
                    else:
                        # Deep learning frameworks (PyTorch, TensorFlow/Keras)
                        updated_config_params = {
                            **model_config.config_params,
                            'learning_rate': new_lr,
                            'batch_size': new_batch,
                            'dropout_rate': new_dropout,
                            'optimizer': new_optimizer,
                            'weight_decay': new_wd,
                            'epochs': new_epochs,
                            'input_channels': new_channels,
                            'channel_adaptation': 'auto' if new_channels != 3 else 'none'
                        }
                        # Create full input size with channels
                        full_input_size = (new_channels, new_width, new_height)
                    
                    updated_model_config = ModelConfig(
                        architecture=model_config.architecture,
                        backbone=new_backbone,
                        num_parameters=model_config.num_parameters,
                        input_size=full_input_size,
                        pretrained=new_pretrained,
                        config_params=updated_config_params,
                        estimated_flops=model_config.estimated_flops,
                        memory_requirements=new_memory,
                        framework=framework
                    )
                    
                    st.session_state.model_config = updated_model_config
                    
                    # MORE EXTENSIVE LOGGING AFTER UPDATE
                    print("=" * 80)
                    
                    st.success("✅ Model configuration updated successfully!")
                    st.info("💡 Configuration updated! Review the changes below, then use 'Continue to Training' when ready.")
                    # Update the local model_config variable to reflect changes in the display
                    model_config = updated_model_config
            
            # Display current configuration summary
            st.subheader("📊 Current Configuration")
            config_col1, config_col2 = st.columns(2)
            
            with config_col1:
                st.metric("Parameters", f"{model_config.num_parameters:,}")
                
                # Framework-aware input shape display
                if framework == "Scikit-learn":
                    st.metric("Input Type", "Feature Vector")
                    st.metric("Feature Extraction", "Automatic")
                else:
                    # Handle both 2D and 3D input_size formats for deep learning
                    try:
                        if isinstance(model_config.input_size, tuple) and len(model_config.input_size) == 3:
                            channels, width, height = model_config.input_size
                            size_str = f"{channels}×{width}×{height}"
                        elif isinstance(model_config.input_size, tuple) and len(model_config.input_size) == 2:
                            width, height = model_config.input_size
                            channels = model_config.config_params.get('input_channels', 3)
                            size_str = f"{channels}×{width}×{height}"
                        else:
                            # Handle None or other formats
                            channels = model_config.config_params.get('input_channels', 3)
                            size_str = f"{channels}×N/A×N/A"
                    except (ValueError, TypeError):
                        # Fallback for any unpacking errors
                        channels = model_config.config_params.get('input_channels', 3)
                        size_str = f"{channels}×N/A×N/A"
                    
                    st.metric("Input Shape (C×H×W)", size_str)
                
                st.metric("Memory", f"{model_config.memory_requirements:.1f} GB")
            
            with config_col2:
                st.metric("Backbone", model_config.backbone)
                
                # Framework-aware channel type display
                if framework == "Scikit-learn":
                    st.metric("Algorithm Type", model_config.architecture)
                    st.metric("Feature Engineering", "Built-in")
                else:
                    # Channel type display for deep learning
                    try:
                        channel_type = "Grayscale" if channels == 1 else "RGB" if channels == 3 else "RGBA" if channels == 4 else f"{channels}-channel"
                        st.metric("Channel Type", channel_type)
                    except:
                        st.metric("Channel Type", "Auto-detect")
                
                st.metric("Pretrained", "✅ Yes" if model_config.pretrained else "❌ No")
                
                # Framework-specific additional info
                if framework == "Scikit-learn":
                    cv_folds = model_config.config_params.get('cv_folds', 5)
                    st.metric("Cross-Validation", f"{cv_folds}-Fold")
                else:
                    st.metric("Batch Size", model_config.config_params.get('batch_size', 'N/A'))
            
            # Additional Scikit-learn specific configuration display
            if framework == "Scikit-learn":
                st.subheader("🔧 Algorithm Configuration")
                algo_col1, algo_col2 = st.columns(2)
                
                config_params = model_config.config_params
                arch = model_config.architecture
                
                with algo_col1:
                    st.metric("Random State", config_params.get('random_state', 42))
                    st.metric("Feature Scaling", "Enabled" if config_params.get('use_scaling', True) else "Disabled")
                    st.metric("Feature Method", config_params.get('feature_method', 'Histogram Features'))
                    st.metric("Class Weight", config_params.get('class_weight', 'None'))
                    
                with algo_col2:
                    st.metric("PCA", "Enabled" if config_params.get('use_pca', False) else "Disabled")
                    if config_params.get('use_pca', False):
                        st.metric("PCA Components", config_params.get('n_components', 10))
                    
                    # Show algorithm-specific parameters
                    if "Random Forest" in arch or "Extra Trees" in arch:
                        st.metric("Estimators", config_params.get('n_estimators', 100))
                        max_depth = config_params.get('max_depth', None)
                        st.metric("Max Depth", max_depth if max_depth is not None else "Auto")
                    elif "SVM" in arch:
                        st.metric("C Parameter", config_params.get('C', 1.0))
                        st.metric("Kernel", config_params.get('kernel', 'rbf'))
                    elif "Gradient Boosting" in arch or "AdaBoost" in arch:
                        st.metric("Estimators", config_params.get('n_estimators', 100))
                        st.metric("Learning Rate", config_params.get('learning_rate', 0.1))
                    elif "Logistic Regression" in arch or "Ridge" in arch:
                        st.metric("C Parameter", config_params.get('C', 1.0))
                        st.metric("Max Iterations", config_params.get('max_iter', 1000))
                    elif "Neural Network" in arch:
                        hidden_layers = config_params.get('hidden_layer_sizes', (100,))
                        st.metric("Hidden Layers", str(hidden_layers))
                        st.metric("Learning Rate", config_params.get('learning_rate_init', 0.001))
                
                # Scoring metrics display
                metrics_list = config_params.get('scoring_metrics', ['accuracy'])
                if isinstance(metrics_list, list):
                    st.info(f"📊 **Scoring Metrics:** {', '.join(metrics_list)}")
                else:
                    st.info(f"📊 **Scoring Metrics:** {metrics_list}")
        
            with col2:
                section("📊", "Model Comparison")

                comparison_data = create_model_comparison_data(dataset_info.task_type)
                df_cmp = pd.DataFrame(comparison_data)

                if not df_cmp.empty:
                    # Highlight the selected model
                    selected_model = model_config.architecture if model_config else None

                    # ── Radar / spider chart ──────────────────────────────────
                    top3 = df_cmp.nlargest(3, "accuracy_score")
                    if selected_model and selected_model not in top3["model"].values:
                        sel_row = df_cmp[df_cmp["model"] == selected_model]
                        top3 = pd.concat([top3, sel_row]).drop_duplicates()

                    radar_cols = ["accuracy_score", "speed_score"]
                    if "memory_gb" in top3.columns:
                        top3 = top3.copy()
                        top3["efficiency"] = 1 / (top3["memory_gb"].clip(lower=0.1))
                        top3["efficiency"] = top3["efficiency"] / top3["efficiency"].max()
                        radar_cols.append("efficiency")

                    fig_radar = go.Figure()
                    colors = ["#6c63ff", "#00d4aa", "#ffd93d", "#ff6b6b"]
                    for idx, (_, row) in enumerate(top3.iterrows()):
                        vals = [row[c] for c in radar_cols] + [row[radar_cols[0]]]
                        labels = ["Accuracy", "Speed"] + (["Efficiency"] if "efficiency" in radar_cols else []) + ["Accuracy"]
                        is_sel = row["model"] == selected_model
                        fig_radar.add_trace(go.Scatterpolar(
                            r=vals, theta=labels, fill="toself",
                            name=row["model"],
                            line=dict(color=colors[idx % len(colors)], width=3 if is_sel else 1.5),
                            opacity=1.0 if is_sel else 0.6,
                        ))
                    fig_radar.update_layout(
                        polar=dict(
                            bgcolor="rgba(0,0,0,0)",
                            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,.1)"),
                            angularaxis=dict(gridcolor="rgba(255,255,255,.1)"),
                        ),
                        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                        height=280, margin=dict(l=20, r=20, t=20, b=20),
                        legend=dict(orientation="h", y=-0.15),
                        showlegend=True,
                    )
                    st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

                    # ── Comparison table ──────────────────────────────────────
                    df_show = df_cmp[["model", "accuracy_score", "speed_score", "num_parameters", "memory_gb"]].copy()
                    df_show.columns = ["Architecture", "Accuracy↑", "Speed↑", "Params (M)", "VRAM (GB)"]
                    df_show = df_show.sort_values("Accuracy↑", ascending=False).reset_index(drop=True)

                    def _highlight(row):
                        if row["Architecture"] == selected_model:
                            return ["background-color:rgba(108,99,255,.18);font-weight:700"] * len(row)
                        return [""] * len(row)

                    st.dataframe(
                        df_show.style.apply(_highlight, axis=1).format({
                            "Accuracy↑": "{:.2f}",
                            "Speed↑":    "{:.2f}",
                            "Params (M)":"{:.1f}",
                            "VRAM (GB)": "{:.1f}",
                        }),
                        use_container_width=True, hide_index=True,
                    )
                    st.caption("★ Highlighted row = currently selected model")
    
    # Manual progression button (only show if model is configured) - MOVED OUTSIDE column context
    if st.session_state.get('model_config') is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➡️ Continue to Training Configuration", key="continue_to_training",
                     type="primary", use_container_width=True):
            st.session_state.manual_config_in_progress = False
            st.session_state.current_step = 3
            st.session_state["current_tool"] = None
            st_rerun()


