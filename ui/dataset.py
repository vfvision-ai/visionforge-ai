"""
Dataset Analysis page – VisionForge (Train Vision Models Effortlessly).
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
)

def show_dataset_analysis():
    """Display dataset analysis interface."""
    apply_theme()

    # Guard: task type must be selected
    if not st.session_state.get('task_type_confirmed', False) or st.session_state.get('selected_task_type') is None:
        hero("📊 Dataset Analysis",
             "Please select a task type on the Home page before analysing a dataset.",
             badges=["Step 2"])
        if st.button("← Go to Home", type="primary"):
            st.session_state.current_step = 0
            st_rerun()
        return

    task = st.session_state.get('selected_task_type', '')
    hero(
        "📊 Dataset Analysis",
        f"Load and inspect your {task} dataset — the pipeline will auto-detect classes, image stats, and training estimates.",
        badges=[f"🎯 {task}", "Step 2 of 5"],
    )
    step_tracker(["Home","Dataset","Model","Training","Results"],
                 st.session_state.get("current_step", 1))

    col1, col2 = st.columns([1, 1])

    with col1:
        section("📂", "Dataset Configuration")
        
        # Initialize dataset_path to prevent UnboundLocalError
        dataset_path = None
        
        # Get current framework
        framework = st.session_state.get('selected_framework', 'PyTorch')
        
        # EARLY LOGGING: Check framework state
        
        # Framework-aware dataset selection method
        if framework == "TensorFlow/Keras":
            dataset_options = ["🌐 Built-in TensorFlow Datasets", "🤗 Hugging Face Datasets", "📁 Browse Folders", "✏️ Enter Path Manually"]
            builtin_label = "🌐 Built-in TensorFlow Datasets"
        elif framework == "PyTorch":
            dataset_options = ["🌐 Built-in PyTorch Datasets", "🤗 Hugging Face Datasets", "📁 Browse Folders", "✏️ Enter Path Manually"]  
            builtin_label = "🌐 Built-in PyTorch Datasets"
        else:
            dataset_options = ["🤗 Hugging Face Datasets", "📁 Browse Folders", "✏️ Enter Path Manually"]
            builtin_label = None
        
        dataset_source = st.radio(
            "Dataset Source:",
            dataset_options,
            horizontal=False
        )
        
        # Built-in datasets (framework-aware)
        if builtin_label and dataset_source == builtin_label:
            if framework == "TensorFlow/Keras":
                st.subheader(f"🧠 TensorFlow Built-in Datasets - {st.session_state.selected_task_type}")
                
                # All TensorFlow datasets with task type tags
                all_tf_datasets = {
                    "MNIST": {
                        "name": "tf.keras.datasets.mnist",
                        "description": "Handwritten digits (0-9), 28x28 grayscale images",
                        "classes": 10,
                        "task": "classification",
                        "task_type": "Classification",
                        "samples": "60K train + 10K test",
                        "icon": "🔢",
                        "image_size": (28, 28),
                        "channels": 1,
                        "input_shape": (28, 28, 1)
                    },
                    "Fashion-MNIST": {
                        "name": "tf.keras.datasets.fashion_mnist", 
                        "description": "Fashion items, 28x28 grayscale images",
                        "classes": 10,
                        "task": "classification",
                        "task_type": "Classification",
                        "samples": "60K train + 10K test",
                        "icon": "👕",
                        "image_size": (28, 28),
                        "channels": 1,
                        "input_shape": (28, 28, 1)
                    },
                    "CIFAR-10": {
                        "name": "tf.keras.datasets.cifar10",
                        "description": "Objects in 10 classes, 32x32 RGB images",
                        "classes": 10,
                        "task": "classification",
                        "task_type": "Classification",
                        "samples": "50K train + 10K test",
                        "icon": "🖼️",
                        "image_size": (32, 32),
                        "channels": 3,
                        "input_shape": (32, 32, 3),
                    },
                    "CIFAR-100": {
                        "name": "tf.keras.datasets.cifar100",
                        "description": "Objects in 100 classes, 32x32 RGB images",
                        "classes": 100,
                        "task": "classification",
                        "task_type": "Classification",
                        "samples": "50K train + 10K test", 
                        "icon": "🎨",
                        "image_size": (32, 32),
                        "channels": 3,
                        "input_shape": (32, 32, 3)
                    },
                    "Oxford-IIIT Pet (Segmentation)": {
                        "name": "oxford_iiit_pet",
                        "description": "Pet image segmentation with pixel-level masks",
                        "classes": 3,
                        "task": "segmentation",
                        "task_type": "Segmentation",
                        "samples": "3.7K train + 3.7K test",
                        "icon": "🐕",
                        "image_size": (128, 128),
                        "channels": 3,
                        "input_shape": (128, 128, 3),
                        "tf_name": "oxford_iiit_pet",
                        "requires_tfds": True
                    }
                }
                
                # Filter datasets based on selected task type
                datasets_dict = {k: v for k, v in all_tf_datasets.items() 
                               if v.get("task_type") == st.session_state.selected_task_type}
                
                if not datasets_dict:
                    st.warning(f"⚠️ No TensorFlow built-in datasets available for {st.session_state.selected_task_type}. Please use HuggingFace datasets or custom datasets.")
            
            elif framework == "PyTorch":
                st.subheader(f"🔥 PyTorch Built-in Datasets - {st.session_state.selected_task_type}")
                
                # All PyTorch datasets with task type tags
                all_pytorch_datasets = {
                    "MNIST": {
                        "name": "torchvision.datasets.MNIST",
                        "description": "Handwritten digits (0-9), 28x28 grayscale images",
                        "classes": 10,
                        "task": "classification",
                        "task_type": "Classification",
                        "samples": "60K train + 10K test",
                        "icon": "🔢",
                        "image_size": (28, 28),
                        "channels": 1,
                        "input_shape": (1, 28, 28)  # PyTorch format: (C, H, W)
                    },
                    "Fashion-MNIST": {
                        "name": "torchvision.datasets.FashionMNIST",
                        "description": "Fashion items, 28x28 grayscale images",
                        "classes": 10,
                        "task": "classification",
                        "task_type": "Classification",
                        "samples": "60K train + 10K test",
                        "icon": "👕",
                        "image_size": (28, 28),
                        "channels": 1,
                        "input_shape": (1, 28, 28)
                    },
                    "CIFAR-10": {
                        "name": "torchvision.datasets.CIFAR10",
                        "description": "Objects in 10 classes, 32x32 RGB images",
                        "classes": 10,
                        "task": "classification",
                        "task_type": "Classification",
                        "samples": "50K train + 10K test",
                        "icon": "🎯",
                        "image_size": (32, 32),
                        "channels": 3,
                        "input_shape": (3, 32, 32)
                    },
                    "CIFAR-100": {
                        "name": "torchvision.datasets.CIFAR100",
                        "description": "Objects in 100 classes, 32x32 RGB images",
                        "classes": 100,
                        "task": "classification",
                        "task_type": "Classification",
                        "samples": "50K train + 10K test",
                        "icon": "🎨",
                        "image_size": (32, 32),
                        "channels": 3,
                        "input_shape": (3, 32, 32)
                    },
                    "VOC2012 (Segmentation)": {
                        "name": "torchvision.datasets.VOCSegmentation",
                        "description": "Pascal VOC 2012 semantic segmentation dataset",
                        "classes": 21,
                        "task": "segmentation",
                        "task_type": "Segmentation",
                        "samples": "1.5K train + 1.5K val",
                        "icon": "🎨",
                        "image_size": (256, 256),
                        "channels": 3,
                        "input_shape": (3, 256, 256)
                    }
                }
                
                # Filter datasets based on selected task type
                datasets_dict = {k: v for k, v in all_pytorch_datasets.items() 
                               if v.get("task_type") == st.session_state.selected_task_type}
                
                if not datasets_dict:
                    st.warning(f"⚠️ No PyTorch built-in datasets available for {st.session_state.selected_task_type}. Please use HuggingFace datasets or custom datasets.")
            
            # Dataset selection
            dataset_label = f"Choose a {framework} dataset:" if framework != "PyTorch" else "Choose a PyTorch dataset:"
            selected_dataset = st.selectbox(
                dataset_label,
                options=list(datasets_dict.keys()),
                format_func=lambda x: f"{datasets_dict[x]['icon']} {x}",
                help=f"Select from {framework}'s built-in datasets"
            )
            
            if selected_dataset:
                dataset_info = datasets_dict[selected_dataset]
                
                # Display dataset information
                st.info(f"""
                **{dataset_info['icon']} {selected_dataset}**
                
                📋 **Description:** {dataset_info['description']}
                🎯 **Task Type:** {dataset_info['task'].replace('_', ' ').title()}
                📊 **Classes:** {dataset_info['classes']}
                📈 **Samples:** {dataset_info['samples']}
                💾 **Source:** `{dataset_info['name']}`
                """)
                
                # Store the selected built-in dataset info
                dataset_config = {
                    'name': selected_dataset,
                    'tf_name': dataset_info['name'],
                    'task': dataset_info['task'],  # Store lowercase task
                    'task_type': dataset_info['task'],  # Also store as task_type for compatibility
                    'classes': dataset_info['classes'],
                    'description': dataset_info['description'],
                    'input_shape': dataset_info.get('input_shape', (224, 224, 3)),  # Use default if missing
                }
                
                # Add image-specific properties only for image datasets
                if 'image_size' in dataset_info:
                    dataset_config['image_size'] = dataset_info['image_size']
                    dataset_config['channels'] = dataset_info['channels']
                else:
                    # For text datasets like IMDB Reviews, use sensible defaults
                    dataset_config['image_size'] = (224, 224)  # Default fallback
                    dataset_config['channels'] = 3  # Default RGB fallback
                    dataset_config['is_text_dataset'] = True  # Mark as text dataset
                
                st.session_state.selected_builtin_dataset = dataset_config
                
                dataset_path = "builtin_tf_dataset"  # Special marker for built-in datasets
                selection_method = "builtin"  # Mark as built-in selection
        
        elif dataset_source == "🤗 Hugging Face Datasets":
            st.subheader(f"🤗 Hugging Face Dataset Hub - {st.session_state.selected_task_type}")
            
            # Curated dataset selection for seamless experience
            st.markdown(f"### 📋 **Curated {st.session_state.selected_task_type} Datasets** (Recommended)")
            st.info(f"💡 **These {st.session_state.selected_task_type.lower()} datasets are pre-tested and guaranteed to work!**")
            
            # Define curated datasets with metadata - organized by task type
            all_curated_datasets = {
                "Classification": {
                    "🚗 CIFAR-10": {
                        "name": "cifar10", 
                        "description": "Classic image classification - vehicles, animals, objects",
                        "samples": "60K images",
                        "classes": "10 categories",
                        "task": "Multi-class Classification",
                        "size": "32×32 pixels",
                        "channels": "RGB (3)",
                        "verified": True,
                        "task_type": "Classification"
                    },
                    "👕 Fashion Items": {
                        "name": "fashion_mnist",
                        "description": "Fashion and clothing items classification",
                        "samples": "70K images", 
                        "classes": "10 fashion categories",
                        "task": "Fashion Classification",
                        "size": "28×28 pixels",
                        "channels": "Grayscale (1)",
                        "verified": True,
                        "task_type": "Classification"
                    },
                    "🍕 Food Categories": {
                        "name": "food101",
                        "description": "Food and cuisine classification dataset",
                        "samples": "101,000 images",
                        "classes": "101 food types",
                        "task": "Food Classification", 
                        "size": "Variable resolution",
                        "channels": "RGB (3)",
                        "verified": True,
                        "task_type": "Classification"
                    },
                    "🐱 Animals": {
                        "name": "cats_vs_dogs",
                        "description": "Binary classification - cats vs dogs",
                        "samples": "23,410 images",
                        "classes": "2 categories (cat, dog)",
                        "task": "Binary Classification",
                        "size": "Variable resolution", 
                        "channels": "RGB (3)",
                        "verified": True,
                        "task_type": "Classification"
                    },
                    "🏠 Indoor Scenes": {
                        "name": "keremberke/indoor-scene-classification",
                        "description": "Indoor scene recognition dataset",
                        "samples": "15K+ images",
                        "classes": "67 indoor categories",
                        "task": "Indoor Scene Classification",
                        "size": "Variable resolution",
                        "channels": "RGB (3)",
                        "verified": True,
                        "task_type": "Classification"
                    }
                },
                "Segmentation": {
                    "🐾 Oxford Pets": {
                        "name": "oxford_iiit_pet",
                        "description": "Pet breeds with segmentation masks",
                        "samples": "7,390 images",
                        "classes": "37 pet breeds + 3 segmentation classes",
                        "task": "Instance Segmentation",
                        "size": "Variable resolution",
                        "channels": "RGB (3)",
                        "verified": True,
                        "task_type": "Segmentation",
                        "subset": None,
                        "features": ["image", "segmentation_mask", "label"]
                    },
                    "🏙️ Scene Parsing": {
                        "name": "scene_parse_150",
                        "description": "ADE20K scene parsing with semantic categories",
                        "samples": "20,210 train + 2,000 val images",
                        "classes": "150 object categories",
                        "task": "Semantic Segmentation",
                        "size": "Variable resolution",
                        "channels": "RGB (3)",
                        "verified": True,
                        "task_type": "Segmentation",
                        "subset": None,
                        "features": ["image", "annotation"]
                    }
                },
                "Object Detection": {
                    "📦 COCO": {
                        "name": "detection-datasets/coco",
                        "description": "Common Objects in Context - standard detection benchmark",
                        "samples": "118K+ images",
                        "classes": "80 object categories",
                        "task": "Object Detection",
                        "size": "Variable resolution",
                        "channels": "RGB (3)",
                        "verified": True,
                        "task_type": "Object Detection"
                    },
                    "🚗 Vehicle Detection": {
                        "name": "keremberke/vehicle-detection",
                        "description": "Detect vehicles in various environments",
                        "samples": "10K+ images",
                        "classes": "Vehicle classes",
                        "task": "Vehicle Detection",
                        "size": "Variable resolution",
                        "channels": "RGB (3)",
                        "verified": True,
                        "task_type": "Object Detection"
                    },
                    "😷 Face & Mask Detection": {
                        "name": "keremberke/face-mask-detection",
                        "description": "Detect faces and face masks",
                        "samples": "5K+ images",
                        "classes": "3 classes (with mask, without mask, improper)",
                        "task": "Face Mask Detection",
                        "size": "Variable resolution",
                        "channels": "RGB (3)",
                        "verified": True,
                        "task_type": "Object Detection"
                    }
                }
            }
            
            # Filter datasets based on selected task type
            curated_datasets = all_curated_datasets.get(st.session_state.selected_task_type, {})
            
            # Dataset selection method
            selection_mode = st.radio(
                "Choose Dataset Selection Method:",
                ["📋 Curated Datasets (Recommended)", "✏️ Manual Entry (Advanced)"],
                help="Curated datasets are pre-tested and guaranteed to work"
            )
            
            if selection_mode == "📋 Curated Datasets (Recommended)":
                # Curated dataset selection
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    selected_dataset = st.selectbox(
                        "Select Dataset:",
                        list(curated_datasets.keys()),
                        help="Choose from our verified, working datasets"
                    )
                    
                    # DEBUG: Log what user selected
                    logger.info(f"🔍 USER SELECTED DATASET: '{selected_dataset}'")
                
                with col2:
                    if selected_dataset:
                        dataset_info = curated_datasets[selected_dataset]
                        
                        # DEBUG: Log dataset info being retrieved
                        logger.info(f"🔍 RETRIEVED DATASET INFO: {dataset_info}")
                        st.markdown(f"""
                        **📋 {selected_dataset}**
                        
                        🎯 **Task:** {dataset_info['task']}  
                        📊 **Samples:** {dataset_info['samples']}  
                        🏷️ **Classes:** {dataset_info['classes']}  
                        📐 **Size:** {dataset_info['size']}  
                        🎨 **Channels:** {dataset_info['channels']}  
                        ✅ **Status:** Pre-verified
                        """)
                
                # Set the dataset name from curated selection
                if selected_dataset:
                    hf_dataset_name = curated_datasets[selected_dataset]["name"]
                    hf_subset = None  # Curated datasets don't need subsets
                    
                    # DEBUG: Log the final dataset name that will be used
                    logger.info(f"🔍 FINAL DATASET NAME TO BE USED: '{hf_dataset_name}'")
                    logger.info(f"🔍 DATASET SUBSET: {hf_subset}")
                    st.success(f"✅ Selected: `{hf_dataset_name}`")
                else:
                    hf_dataset_name = None
                    hf_subset = None
            
            else:
                # Manual entry mode (original functionality)
                st.markdown("### ✏️ **Manual Dataset Entry**")
                st.warning("⚠️ **Advanced users only.** Manually entered datasets may require authentication or might not exist.")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    hf_dataset_name = st.text_input(
                        "Dataset Name:",
                        placeholder="e.g., username/dataset-name",
                        help="Enter the exact Hugging Face dataset name (format: username/dataset-name)"
                    )
                
                with col2:
                    hf_subset = st.text_input(
                        "Subset (optional):",
                        placeholder="e.g., train, test",
                        help="Dataset subset/configuration if needed"
                    )
            
            # Authentication (only show for manual entry)
            if selection_mode == "✏️ Manual Entry (Advanced)":
                st.markdown("### 🔐 **Authentication (for private/gated datasets):**")
                st.info("💡 **Note:** Curated datasets don't require authentication!")
                
                # Simple vertical layout to avoid nesting columns
                auth_method = st.selectbox(
                    "Authentication Method:",
                    ["None (Public Dataset)", "Hugging Face Token", "Environment Variable"],
                    help="Choose authentication method for private repositories"
                )
                
                hf_token = None
                if auth_method == "Hugging Face Token":
                    hf_token = st.text_input(
                        "Access Token:",
                        type="password",
                        placeholder="hf_xxxxxxxxxxxx",
                        help="Your Hugging Face access token"
                    )
                elif auth_method == "Environment Variable":
                    env_var = st.text_input(
                        "Environment Variable:",
                        value="HF_TOKEN",
                        help="Environment variable containing your HF token"
                    )
                    try:
                        hf_token = os.environ.get(env_var) if env_var else None
                        if hf_token:
                            st.success(f"✅ Token found in {env_var}")
                        else:
                            st.warning(f"⚠️ No token found in {env_var}")
                    except:
                        st.warning("⚠️ Could not access environment variable")
            else:
                # For curated datasets, no authentication needed
                hf_token = None
                st.success("✅ **No authentication required for curated datasets!**")
            
            # Auto-validation for curated datasets or manual validation
            if hf_dataset_name:
                if selection_mode == "📋 Curated Datasets (Recommended)":
                    # Auto-validate curated datasets (they're pre-tested)
                    st.success("✅ **Pre-verified dataset ready to use!**")
                    st.info("💡 **This dataset has been tested and guaranteed to work.**")
                    
                    # DEBUG: Log what's about to be stored
                    logger.info(f"🔍 ========== STORING HF DATASET CONFIG ==========")
                    logger.info(f"🔍 Selected Dataset Display Name: '{selected_dataset}'")
                    logger.info(f"🔍 Dataset Technical Name (hf_dataset_name): '{hf_dataset_name}'")
                    logger.info(f"🔍 Dataset Subset (hf_subset): '{hf_subset}'")
                    
                    # Store HF dataset configuration immediately for curated datasets
                    hf_config = {
                        'dataset_name': hf_dataset_name,  # Technical name for loading
                        'display_name': selected_dataset,  # Display name for UI
                        'subset': hf_subset if hf_subset else None,
                        'token': None,  # Curated datasets are all public
                        'features': ['image', 'label'],  # Standard features
                        'description': curated_datasets[selected_dataset]['description'],
                        'builder_name': hf_dataset_name.split('/')[-1],
                        'is_curated': True
                    }
                    
                    # DEBUG: Log the config being stored
                    logger.info(f"🔍 HF Config to be stored: {hf_config}")
                    st.info(f"🔍 **STORING CONFIG:** Dataset='{hf_config['dataset_name']}', Display='{hf_config['display_name']}'")
                    
                    st.session_state.hf_dataset_config = hf_config
                    
                    # DEBUG: Verify what was stored
                    logger.info(f"🔍 Config stored in session_state: {st.session_state.hf_dataset_config}")
                    logger.info(f"🔍 ========== HF DATASET CONFIG STORED ==========")
                    
                    # Show curated dataset info
                    dataset_info = curated_datasets[selected_dataset]
                    st.markdown(f"""
                    **🤗 {hf_dataset_name}**
                    
                    📋 **Description:** {dataset_info['description']}
                    🎯 **Task Type:** {dataset_info['task']}
                    📊 **Samples:** {dataset_info['samples']}
                    🏷️ **Classes:** {dataset_info['classes']}
                    📐 **Image Size:** {dataset_info['size']}
                    🎨 **Channels:** {dataset_info['channels']}
                    ✅ **Status:** Ready for analysis
                    """)
                    
                else:
                    # Manual validation for advanced users
                    if st.button("🔍 Validate & Preview Dataset", type="primary"):
                        try:
                            with st.spinner("Connecting to Hugging Face Hub..."):
                                # Try to import datasets library
                                try:
                                    from datasets import load_dataset_builder, load_dataset
                                    import datasets
                                except ImportError:
                                    st.error("❌ `datasets` library not installed. Please install with: `pip install datasets`")
                                    st.code("pip install datasets", language="bash")
                                    st.stop()
                                
                                # Build authentication kwargs
                                auth_kwargs = {}
                                if hf_token:
                                    auth_kwargs['token'] = hf_token
                                
                                # Load dataset builder for info
                                if hf_subset:
                                    builder = load_dataset_builder(hf_dataset_name, hf_subset, **auth_kwargs)
                                else:
                                    builder = load_dataset_builder(hf_dataset_name, **auth_kwargs)
                                
                                # Display dataset information
                                st.success("✅ Dataset found and accessible!")
                                
                                st.info(f"""
                                **🤗 {hf_dataset_name}** {f"({hf_subset})" if hf_subset else ""}
                                
                                📋 **Description:** {builder.info.description[:200]}...
                                🎯 **Features:** {list(builder.info.features.keys())}
                                📊 **Dataset Size:** {builder.info.dataset_size or 'Unknown'} bytes
                                💾 **Builder:** {builder.info.builder_name}
                                """)
                                
                                # Load a small sample for preview
                                st.write("📋 **Dataset Preview:**")
                                try:
                                    if hf_subset:
                                        sample_dataset = load_dataset(hf_dataset_name, hf_subset, split='train[:5]', **auth_kwargs)
                                    else:
                                        sample_dataset = load_dataset(hf_dataset_name, split='train[:5]', **auth_kwargs)
                                    
                                    # Display sample
                                    st.write("**First 5 samples:**")
                                    for i, example in enumerate(sample_dataset):
                                        with st.expander(f"Sample {i+1}"):
                                            st.json(example, expanded=False)
                                            
                                except Exception as e:
                                    st.warning(f"Could not load sample data: {str(e)}")
                                
                                # Store HF dataset configuration
                                hf_config = {
                                    'dataset_name': hf_dataset_name,
                                    'subset': hf_subset if hf_subset else None,
                                    'token': hf_token if hf_token else None,
                                    'features': list(builder.info.features.keys()),
                                    'description': builder.info.description,
                                    'builder_name': builder.info.builder_name,
                                    'is_curated': False
                                }
                                
                                st.session_state.hf_dataset_config = hf_config
                                
                        except Exception as e:
                            st.error(f"❌ Failed to access dataset: {str(e)}")
                            
                            # Enhanced error handling with solutions
                            error_msg = str(e).lower()
                            if "doesn't exist" in error_msg:
                                st.markdown("""
                                **🔧 Possible Solutions:**
                                - ✅ **Use curated datasets instead** (recommended)
                                - 🔍 **Check dataset name spelling**
                                - 🌐 **Visit** https://huggingface.co/datasets **to browse available datasets**
                                """)
                            elif "gated" in error_msg or "authenticated" in error_msg:
                                st.markdown("""
                                **🔐 Authentication Required:**
                                - 🎫 **Get a token** from https://huggingface.co/settings/tokens
                                - 🔑 **Enter token** in the authentication section above
                                - ✅ **Or use curated datasets** (no auth needed)
                                """)
                            else:
                                st.markdown("""
                                **🔧 General Solutions:**
                                - ✅ **Try curated datasets** (guaranteed to work)
                                - 🌐 **Check internet connection**
                                - 📝 **Verify dataset name format**: `username/dataset-name`
                                """)
                            
                            # Suggest fallback to curated datasets
                            st.info("💡 **Suggestion:** Switch to 'Curated Datasets' above for guaranteed compatibility!")
            
            # Set dataset path and selection method for HF datasets
            if hf_dataset_name and hasattr(st.session_state, 'hf_dataset_config'):
                dataset_path = f"hf_dataset:{hf_dataset_name}"
                if hf_subset:
                    dataset_path += f":{hf_subset}"
                selection_method = "huggingface"
            else:
                dataset_path = None
                selection_method = None
                
        elif dataset_source == "📁 Browse Folders":
            selection_method = "📁 Browse Folders"
            dataset_path = None
        else:
            selection_method = "✏️ Enter Path Manually"
            dataset_path = None
        
        if builtin_label and dataset_source != builtin_label and selection_method == "📁 Browse Folders":
            # Dynamic folder browser with navigation
            st.write("📂 **Navigate to Dataset Folder:**")
            
            # Initialize current path in session state
            if 'current_browse_path' not in st.session_state:
                st.session_state.current_browse_path = os.getcwd()
            
            current_path = st.session_state.current_browse_path
            
            # Show current location with breadcrumbs
            st.write(f"📍 **Current Location:** `{current_path}`")
            
            # Navigation buttons
            col_nav1, col_nav2, col_nav3, col_nav4 = st.columns([1, 1, 1, 2])
            
            with col_nav1:
                if st.button("🏠 Home"):
                    st.session_state.current_browse_path = os.path.expanduser("~")
                    st.experimental_rerun()
            
            with col_nav2:
                if st.button("💻 Root"):
                    st.session_state.current_browse_path = "/"
                    st.experimental_rerun()
            
            with col_nav3:
                if st.button("⬆️ Up"):
                    parent = os.path.dirname(current_path)
                    if parent != current_path:  # Prevent infinite loop at root
                        st.session_state.current_browse_path = parent
                        st.experimental_rerun()
            
            with col_nav4:
                # Quick path input
                new_path = st.text_input("Quick jump to path:", key="quick_path")
                if new_path and os.path.exists(new_path) and os.path.isdir(new_path):
                    st.session_state.current_browse_path = new_path
                    st.experimental_rerun()
            
            # Get available directories and files
            try:
                items = []
                if os.path.exists(current_path):
                    for item in sorted(os.listdir(current_path)):
                        item_path = os.path.join(current_path, item)
                        if not item.startswith('.'):  # Hide hidden files
                            if os.path.isdir(item_path):
                                items.append(("📁", item, "folder"))
                            elif item.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')):
                                items.append(("🖼️", item, "image"))
                
                # Display folder contents
                if items:
                    st.write("📋 **Contents:**")
                    
                    # Create clickable folder list
                    folders = [item for item in items if item[2] == "folder"]
                    images = [item for item in items if item[2] == "image"]
                    
                    if folders:
                        st.write(f"📁 **Folders ({len(folders)}):**")
                        
                        # Create grid of folder buttons
                        cols_per_row = 3
                        for i in range(0, len(folders), cols_per_row):
                            row_cols = st.columns(cols_per_row)
                            for j, col in enumerate(row_cols):
                                if i + j < len(folders):
                                    folder_name = folders[i + j][1]
                                    with col:
                                        if st.button(f"📁 {folder_name}", key=f"folder_{i+j}"):
                                            st.session_state.current_browse_path = os.path.join(current_path, folder_name)
                                            st.experimental_rerun()
                    
                    if images:
                        st.write(f"🖼️ **Images ({len(images)}):** {', '.join([img[1] for img in images[:5]])}{'...' if len(images) > 5 else ''}")
                    
                    # Select current folder button
                    st.write("---")
                    col_select1, col_select2 = st.columns([1, 1])
                    
                    with col_select1:
                        if st.button("✅ Use This Folder", type="primary", key="select_current"):
                            dataset_path = current_path
                            st.success(f"📁 Selected: `{dataset_path}`")
                    
                    with col_select2:
                        # Show folder stats
                        total_images = len(images)
                        total_folders = len(folders)
                        st.info(f"📊 {total_folders} folders, {total_images} images")
                
                else:
                    st.warning("📭 Folder is empty or inaccessible")
                    
            except PermissionError:
                st.error("🚫 Permission denied to access this folder")
            except Exception as e:
                st.error(f"❌ Error reading folder: {str(e)}")
            
            # Set dataset_path if folder was selected
            if 'dataset_path' not in locals():
                dataset_path = None
            
            # File selection within the current folder
            if st.session_state.current_path and os.path.isdir(st.session_state.current_path):
                st.subheader("📄 Select Files")
                
                # Get all files in current directory with size filtering
                try:
                    all_files = []
                    large_files_count = 0
                    
                    for item in sorted(os.listdir(st.session_state.current_path)):
                        item_path = os.path.join(st.session_state.current_path, item)
                        if os.path.isfile(item_path):
                            file_ext = os.path.splitext(item)[1].lower()
                            file_size = os.path.getsize(item_path)
                            file_size_gb = file_size / (1024 * 1024 * 1024)
                            
                            # Support large archives and images up to 10GB
                            if file_ext in ['.zip', '.tar', '.gz', '.rar', '.7z'] or file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                                if file_size_gb <= 10:  # Max 10GB per file
                                    all_files.append(item)
                                    if file_size_gb > 1:
                                        large_files_count += 1
                                else:
                                    st.warning(f"⚠️ Skipped `{item}` ({file_size_gb:.1f} GB - exceeds 10 GB limit)")
                    
                    # Show large file info
                    if large_files_count > 0:
                        st.info(f"📊 Found {large_files_count} large file(s) (>1 GB) - Processing may take time")
                    
                    if all_files:
                        selected_files = st.multiselect(
                            "Select dataset files (supports ZIP archives and images):",
                            all_files,
                            help="You can select multiple files including ZIP archives"
                        )
                        
                        if selected_files:
                            st.write("Selected files:")
                            total_size = 0
                            
                            for file in selected_files:
                                file_path = os.path.join(st.session_state.current_path, file)
                                file_size = os.path.getsize(file_path)
                                total_size += file_size
                                
                                # Smart size formatting
                                if file_size >= 1024**3:  # GB
                                    size_str = f"{file_size / (1024**3):.2f} GB"
                                elif file_size >= 1024**2:  # MB
                                    size_str = f"{file_size / (1024**2):.1f} MB"
                                else:  # KB
                                    size_str = f"{file_size / 1024:.0f} KB"
                                
                                file_ext = os.path.splitext(file)[1].lower()
                                
                                # Enhanced icons for different file types
                                if file_ext == '.zip':
                                    icon = "📦"
                                elif file_ext in ['.tar', '.gz']:
                                    icon = "🗜️"
                                elif file_ext in ['.rar', '.7z']:
                                    icon = "📁"
                                elif file_ext in ['.jpg', '.jpeg']:
                                    icon = "🖼️"
                                elif file_ext in ['.png']:
                                    icon = "🎨"
                                else:
                                    icon = "📄"
                                
                                # Color-code large files
                                if file_size >= 1024**3:  # >= 1GB
                                    st.write(f"{icon} `{file}` **{size_str}** 🔴")
                                elif file_size >= 100 * 1024**2:  # >= 100MB
                                    st.write(f"{icon} `{file}` **{size_str}** 🟡")
                                else:
                                    st.write(f"{icon} `{file}` {size_str}")
                            
                            # Show total selection size
                            if total_size >= 1024**3:
                                total_str = f"{total_size / (1024**3):.2f} GB"
                            elif total_size >= 1024**2:
                                total_str = f"{total_size / (1024**2):.1f} MB"
                            else:
                                total_str = f"{total_size / 1024:.0f} KB"
                            
                            st.info(f"📊 **Total selected:** {len(selected_files)} files, {total_str}")
                            
                            # Accept selected files button
                            if st.button("✅ Accept Selected Files"):
                                st.session_state.selected_files = [
                                    os.path.join(st.session_state.current_path, f) for f in selected_files
                                ]
                                st.success(f"✅ Accepted {len(selected_files)} file(s) for processing")
                                
                                # Set dataset_path to the directory for processing
                                dataset_path = st.session_state.current_path
                    else:
                        st.info("📭 No supported files found (ZIP archives or images)")
                        
                except Exception as e:
                    st.error(f"❌ Error reading files: {str(e)}")
                
        elif builtin_label and dataset_source != builtin_label:  # Manual path entry
            st.subheader("📝 Manual Path Entry")
            dataset_path = st.text_input(
                "Dataset Path",
                placeholder="/path/to/your/dataset",
                help="Enter the full path to your dataset directory"
            )
            
            # Validate path if entered
            if dataset_path:
                if os.path.exists(dataset_path):
                    if os.path.isdir(dataset_path):
                        st.success("✅ Valid directory path")
                    else:
                        st.error("❌ Path exists but is not a directory")
                else:
                    st.error("❌ Path does not exist")
        
        # Large file upload option (up to 10GB) - only for non-builtin datasets
        uploaded_file = None  # Initialize uploaded_file
        if not builtin_label or dataset_source != builtin_label:
            st.subheader("📦 Upload Large Dataset Archive")
            
            # File size warning
            st.info("📊 **Supported file sizes:** Up to 10 GB • Formats: ZIP, TAR, RAR, 7Z")
            
            uploaded_file = st.file_uploader(
            "Upload your dataset archive (supports large files up to 10GB)",
            type=['zip', 'tar', 'gz', 'rar', '7z'],
            help="Large dataset support: Upload archives up to 10GB containing your training images"
        )
        
        if uploaded_file is not None:
            # CRITICAL FIX: Clear built-in dataset selection immediately when file is uploaded
            if 'selected_builtin_dataset' in st.session_state:
                del st.session_state.selected_builtin_dataset
            # Check file size
            file_size_mb = uploaded_file.size / (1024 * 1024)
            file_size_gb = file_size_mb / 1024
            
            if file_size_gb > 10:
                st.error(f"❌ File too large: {file_size_gb:.1f} GB (max 10 GB)")
                st.stop()
            
            # Display file info
            if file_size_gb >= 1:
                st.success(f"📦 **{uploaded_file.name}** ({file_size_gb:.2f} GB)")
            else:
                st.success(f"📦 **{uploaded_file.name}** ({file_size_mb:.1f} MB)")
            
            # Save uploaded file with progress
            upload_dir = os.path.join(os.getcwd(), "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            
            zip_path = os.path.join(upload_dir, uploaded_file.name)
            
            # Large file upload with progress tracking
            if file_size_mb > 100:  # Show progress for files > 100MB
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with open(zip_path, "wb") as f:
                    buffer = uploaded_file.getbuffer()
                    total_size = len(buffer)
                    chunk_size = 1024 * 1024  # 1MB chunks
                    
                    for i in range(0, total_size, chunk_size):
                        chunk = buffer[i:i + chunk_size]
                        f.write(chunk)
                        progress = min((i + chunk_size) / total_size, 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"Uploading... {progress*100:.1f}% ({(i + chunk_size)/(1024*1024):.1f} MB)")
                
                progress_bar.progress(1.0)
                status_text.text(f"✅ Upload complete: {file_size_gb:.2f} GB" if file_size_gb >= 1 else f"✅ Upload complete: {file_size_mb:.1f} MB")
            else:
                # Direct upload for smaller files
                with open(zip_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"✅ Uploaded: {uploaded_file.name}")
            
            # Extract large archive with progress
            extract_button_text = f"📂 Extract Large Dataset ({file_size_gb:.2f} GB)" if file_size_gb >= 1 else f"📂 Extract Dataset ({file_size_mb:.1f} MB)"
            
            if st.button(extract_button_text, type="primary"):
                try:
                    import zipfile
                    import tarfile
                    extract_dir = os.path.join(upload_dir, "extracted", os.path.splitext(uploaded_file.name)[0])
                    os.makedirs(extract_dir, exist_ok=True)
                    
                    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                    
                    with st.spinner(f"Extracting large dataset... This may take several minutes for {file_size_gb:.1f} GB files"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        if file_ext in ['.zip']:
                            with zipfile.ZipFile(zip_path, 'r') as archive:
                                file_list = archive.namelist()
                                total_files = len(file_list)
                                
                                for i, file_info in enumerate(file_list):
                                    archive.extract(file_info, extract_dir)
                                    if i % 100 == 0 or i == total_files - 1:  # Update every 100 files
                                        progress = (i + 1) / total_files
                                        progress_bar.progress(progress)
                                        status_text.text(f"Extracting... {progress*100:.1f}% ({i+1}/{total_files} files)")
                        
                        elif file_ext in ['.tar', '.gz']:
                            with tarfile.open(zip_path, 'r:*') as archive:
                                members = archive.getmembers()
                                total_files = len(members)
                                
                                for i, member in enumerate(members):
                                    archive.extract(member, extract_dir)
                                    if i % 100 == 0 or i == total_files - 1:
                                        progress = (i + 1) / total_files
                                        progress_bar.progress(progress)
                                        status_text.text(f"Extracting... {progress*100:.1f}% ({i+1}/{total_files} files)")
                        
                        else:
                            # Fallback for other formats
                            st.warning(f"⚠️ Extracting {file_ext} format without progress tracking")
                            with zipfile.ZipFile(zip_path, 'r') as archive:
                                archive.extractall(extract_dir)
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Extraction complete!")
                    st.success(f"✅ Successfully extracted {file_size_gb:.2f} GB dataset to: `{extract_dir}`")
                    
                    # CRITICAL FIX: Clear any previous built-in dataset selection when uploading
                    if 'selected_builtin_dataset' in st.session_state:
                        del st.session_state.selected_builtin_dataset
                    
                    dataset_path = extract_dir
                    st.session_state.uploaded_dataset_path = extract_dir
                    
                    # Show extracted content info
                    try:
                        total_files = sum([len(files) for r, d, files in os.walk(extract_dir)])
                        st.info(f"📊 Extracted content: {total_files:,} files")
                    except:
                        pass
                    
                except Exception as e:
                    st.error(f"❌ Failed to extract large archive: {str(e)}")
                    st.info("💡 Try extracting manually or using a smaller archive")
        
        # Task type - use the pre-selected task type from session state
        # Convert from "Classification" to "classification" for analyzer compatibility
        task_type_map = {
            "Classification": "classification",
            "Segmentation": "segmentation",
            "Object Detection": "detection"
        }
        task_type = task_type_map.get(st.session_state.selected_task_type, "auto")
        
        # Display current task type (read-only)
        st.info(f"📋 **Task Type**: {st.session_state.selected_task_type} (auto-detected as '{task_type}')")
        
        # Show current dataset status
        current_dataset = None
        if dataset_path:
            current_dataset = dataset_path
        elif hasattr(st.session_state, 'uploaded_dataset_path'):
            current_dataset = st.session_state.uploaded_dataset_path
        
        if current_dataset:
            st.info(f"📂 Current dataset: `{os.path.basename(current_dataset)}`")
            
            # Show selected files if any
            if hasattr(st.session_state, 'selected_files') and st.session_state.selected_files:
                st.write("📄 Selected files:")
                for file_path in st.session_state.selected_files:
                    file_name = os.path.basename(file_path)
                    file_ext = os.path.splitext(file_name)[1].lower()
                    
                    if file_ext == '.zip':
                        st.write(f"📦 {file_name}")
                    else:
                        st.write(f"🖼️ {file_name}")
        
        # Analysis button
        # Check if we have either a dataset path or a built-in dataset selected
        has_builtin_dataset = hasattr(st.session_state, 'selected_builtin_dataset') and st.session_state.selected_builtin_dataset
        analyze_disabled = not (current_dataset or has_builtin_dataset)
        
        if has_builtin_dataset:
            framework = st.session_state.get('selected_framework', 'PyTorch')
            button_help = f"Analyze {framework} {st.session_state.selected_builtin_dataset['name']} dataset"
        elif current_dataset:
            button_help = "Analyze the selected dataset"
        else:
            button_help = "Select a dataset first"
        
        if st.button("🔍 Analyze Dataset", disabled=analyze_disabled, help=button_help):
            with st.spinner("Analyzing dataset..."):
                try:
                    # CRITICAL FIX: Priority order - Uploaded > HuggingFace > Built-in datasets
                    # 1. Check if we have an uploaded dataset first
                    if current_dataset and current_dataset != "builtin_tf_dataset":
                        
                        # Handle regular/uploaded datasets
                        analyzer = DatasetAnalyzer()
                        
                        # Handle ZIP files if selected
                        if hasattr(st.session_state, 'selected_files') and st.session_state.selected_files:
                            zip_files = [f for f in st.session_state.selected_files if f.endswith('.zip')]
                            if zip_files:
                                # Extract ZIP files first
                                import zipfile
                                temp_extract_dir = os.path.join(os.getcwd(), "temp_extracted")
                                os.makedirs(temp_extract_dir, exist_ok=True)
                                
                                for zip_file in zip_files:
                                    extract_subdir = os.path.join(temp_extract_dir, os.path.splitext(os.path.basename(zip_file))[0])
                                    os.makedirs(extract_subdir, exist_ok=True)
                                    
                                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                                        zip_ref.extractall(extract_subdir)
                                    
                                    st.info(f"📦 Extracted: {os.path.basename(zip_file)}")
                                
                                # Analyze the extracted content
                                current_dataset = temp_extract_dir
                        
                        dataset_info = analyzer.analyze(Path(current_dataset), task_type)
                        st.session_state.dataset_info = dataset_info
                        st.success("✅ Dataset analysis complete! Use **Continue to Model Selection →** in the panel on the right.")
                    
                    # 2. Check for HuggingFace dataset configuration (curated datasets)
                    elif hasattr(st.session_state, 'hf_dataset_config') and st.session_state.hf_dataset_config:
                        hf_config = st.session_state.hf_dataset_config
                        
                        # DEBUG: Comprehensive logging for HF dataset analysis
                        logger.info(f"🔍 ========== HUGGINGFACE DATASET ANALYSIS START ==========")
                        logger.info(f"🔍 HF Config: {hf_config}")
                        logger.info(f"🔍 Dataset Name: '{hf_config['dataset_name']}'")
                        logger.info(f"🔍 Dataset Subset: '{hf_config.get('subset')}'")
                        logger.info(f"🔍 Is Curated: {hf_config.get('is_curated', False)}")
                        logger.info(f"🔍 Selected Framework: {st.session_state.get('selected_framework', 'PyTorch')}")
                        
                        
                        # Handle HuggingFace dataset analysis
                        from core.dataset_analyzer import DatasetInfo
                        
                        # Map curated dataset names to proper configurations
                        if hf_config['dataset_name'] == 'cifar10':
                            # CIFAR-10 actual class names
                            cifar10_classes = [
                                "airplane", "automobile", "bird", "cat", "deer", 
                                "dog", "frog", "horse", "ship", "truck"
                            ]
                            
                            # FRAMEWORK-AWARE FIX: Create correct image stats for TensorFlow vs PyTorch
                            framework = st.session_state.get('selected_framework', 'PyTorch')
                            if framework == "TensorFlow/Keras":
                                cifar_stats = {
                                    "tensorflow_shape": "(32, 32, 3)",
                                    "display_size": "32×32", 
                                    "channels": 3,
                                    "framework_format": "TensorFlow (H,W,C)",
                                    "mean": 0.5, "std": 0.5
                                }
                            else:
                                cifar_stats = {
                                    "pytorch_shape": "(3, 32, 32)",
                                    "display_size": "32×32",
                                    "channels": 3, 
                                    "framework_format": "PyTorch (C,H,W)",
                                    "mean": 0.5, "std": 0.5
                                }
                            
                            dataset_info = DatasetInfo(
                                task_type=task_type,  # Use user-selected task type
                                num_classes=10,
                                num_samples=60000,
                                class_names=cifar10_classes,
                                class_distribution={class_name: 6000 for class_name in cifar10_classes},
                                image_size=(32, 32),
                                image_stats=cifar_stats,
                                has_annotations=True,
                                annotation_format="classification",
                                recommended_batch_size=32,
                                estimated_training_time=60.0,
                                dataset_path=None,
                                channels=3,
                                is_hf_dataset=True,
                                hf_dataset_name="cifar10",
                                hf_subset=None,
                                hf_features=["image", "label"],
                                hf_description="CIFAR-10 dataset from HuggingFace"
                            )
                            st.info("✅ CIFAR-10 HuggingFace dataset configured: 60K total samples")
                            st.info(f"🛩️ **Classes:** {', '.join(cifar10_classes)}")
                            
                        elif hf_config['dataset_name'] == 'fashion_mnist':
                            # Fashion-MNIST actual class names
                            fashion_classes = [
                                "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
                            ]
                            
                            # FRAMEWORK-AWARE FIX: Create correct image stats for Fashion-MNIST
                            framework = st.session_state.get('selected_framework', 'PyTorch')
                            if framework == "TensorFlow/Keras":
                                fashion_stats = {
                                    "tensorflow_shape": "(28, 28, 1)",
                                    "display_size": "28×28", 
                                    "channels": 1,
                                    "framework_format": "TensorFlow (H,W,C)",
                                    "mean": 0.5, "std": 0.5
                                }
                            else:
                                fashion_stats = {
                                    "pytorch_shape": "(1, 28, 28)",
                                    "display_size": "28×28",
                                    "channels": 1, 
                                    "framework_format": "PyTorch (C,H,W)",
                                    "mean": 0.5, "std": 0.5
                                }
                            
                            dataset_info = DatasetInfo(
                                task_type=task_type,  # Use user-selected task type
                                num_classes=10,
                                num_samples=70000,
                                class_names=fashion_classes,
                                class_distribution={class_name: 7000 for class_name in fashion_classes},
                                image_size=(28, 28),
                                image_stats=fashion_stats,
                                has_annotations=True,
                                annotation_format="classification",
                                recommended_batch_size=32,
                                estimated_training_time=45.0,
                                dataset_path=None,
                                channels=1,
                                is_hf_dataset=True,
                                hf_dataset_name="fashion_mnist",
                                hf_subset=None,
                                hf_features=["image", "label"],
                                hf_description="Fashion-MNIST dataset from HuggingFace"
                            )
                            st.info("✅ Fashion-MNIST HuggingFace dataset configured: 70K total samples")
                            st.info(f"👕 **Classes:** {', '.join(fashion_classes)}")
                            
                        elif hf_config['dataset_name'] == 'cats_vs_dogs':
                            # DEBUG: Log that we're processing cats_vs_dogs
                            logger.info(f"🔍 ========== PROCESSING CATS_VS_DOGS DATASET ==========")
                            logger.info(f"🔍 Creating DatasetInfo for cats_vs_dogs")
                            
                            # Cats vs Dogs actual class names
                            cats_dogs_classes = ["cat", "dog"]
                            
                            # FRAMEWORK-AWARE FIX: Create correct image stats for Cats vs Dogs
                            framework = st.session_state.get('selected_framework', 'PyTorch')
                            if framework == "TensorFlow/Keras":
                                cats_dogs_stats = {
                                    "tensorflow_shape": "(224, 224, 3)",
                                    "display_size": "224×224", 
                                    "channels": 3,
                                    "framework_format": "TensorFlow (H,W,C)",
                                    "mean": 0.485, "std": 0.229
                                }
                            else:
                                cats_dogs_stats = {
                                    "pytorch_shape": "(3, 224, 224)",
                                    "display_size": "224×224",
                                    "channels": 3, 
                                    "framework_format": "PyTorch (C,H,W)",
                                    "mean": 0.485, "std": 0.229
                                }
                            
                            dataset_info = DatasetInfo(
                                task_type=task_type,  # Use user-selected task type
                                num_classes=2,
                                num_samples=23410,
                                class_names=cats_dogs_classes,
                                class_distribution={"cat": 11705, "dog": 11705},
                                image_size=(224, 224),
                                image_stats=cats_dogs_stats,
                                has_annotations=True,
                                annotation_format="classification",
                                recommended_batch_size=32,
                                estimated_training_time=30.0,
                                dataset_path=None,
                                channels=3,
                                is_hf_dataset=True,
                                hf_dataset_name="cats_vs_dogs",
                                hf_subset=None,
                                hf_features=["image", "labels"],  # Note: cats_vs_dogs uses "labels" not "label"
                                hf_description="Cats vs Dogs binary classification dataset"
                            )
                            
                            # DEBUG: Log the created DatasetInfo
                            logger.info(f"🔍 DatasetInfo created:")
                            logger.info(f"🔍   - Task: {dataset_info.task_type}")
                            logger.info(f"🔍   - Classes: {dataset_info.num_classes}")
                            logger.info(f"🔍   - Samples: {dataset_info.num_samples}")
                            logger.info(f"🔍   - Class names: {dataset_info.class_names}")
                            logger.info(f"🔍 ========== CATS_VS_DOGS PROCESSING COMPLETE ==========")
                            
                            st.info("✅ Cats vs Dogs HuggingFace dataset configured: 23,410 total samples")
                            st.info(f"🐱🐶 **Classes:** {', '.join(cats_dogs_classes)}")
                            st.info(f"📊 **Distribution:** ~11,705 images per class (cat/dog)")
                            
                        elif hf_config['dataset_name'] == 'food101':
                            # Food-101 actual class names (101 food categories)
                            food101_classes = [
                                "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
                                "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
                                "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
                                "ceviche", "cheese_plate", "cheesecake", "chicken_curry", "chicken_quesadilla",
                                "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
                                "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
                                "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
                                "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
                                "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
                                "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
                                "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
                                "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
                                "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
                                "mussels", "nachos", "omelette", "onion_rings", "oysters",
                                "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
                                "pho", "pizza", "pork_chop", "poutine", "prime_rib",
                                "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
                                "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
                                "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
                                "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare",
                                "waffles"
                            ]
                            
                            # FRAMEWORK-AWARE FIX: Create correct image stats for Food-101
                            framework = st.session_state.get('selected_framework', 'PyTorch')
                            if framework == "TensorFlow/Keras":
                                food_stats = {
                                    "tensorflow_shape": "(224, 224, 3)",
                                    "display_size": "224×224", 
                                    "channels": 3,
                                    "framework_format": "TensorFlow (H,W,C)",
                                    "mean": 0.485, "std": 0.229
                                }
                            else:
                                food_stats = {
                                    "pytorch_shape": "(3, 224, 224)",
                                    "display_size": "224×224",
                                    "channels": 3, 
                                    "framework_format": "PyTorch (C,H,W)",
                                    "mean": 0.485, "std": 0.229
                                }
                            
                            dataset_info = DatasetInfo(
                                task_type=task_type,  # Use user-selected task type
                                num_classes=101,
                                num_samples=101000,
                                class_names=food101_classes,
                                class_distribution={class_name: 1000 for class_name in food101_classes},
                                image_size=(224, 224),
                                image_stats=food_stats,
                                has_annotations=True,
                                annotation_format="classification",
                                recommended_batch_size=32,
                                estimated_training_time=120.0,
                                dataset_path=None,
                                channels=3,
                                is_hf_dataset=True,
                                hf_dataset_name="food101",
                                hf_subset=None,
                                hf_features=["image", "label"],
                                hf_description="Food-101 dataset with 101 food categories"
                            )
                            st.info("✅ Food-101 HuggingFace dataset configured: 101K total samples")
                            st.info(f"🍕 **Classes:** 101 food categories (apple_pie, baby_back_ribs, baklava, ...)")
                            st.info(f"📊 **Sample distribution:** 1000 images per food category")
                            
                        elif hf_config['dataset_name'] == 'keremberke/indoor-scene-classification':
                            # DEBUG: Log that we're processing indoor scene dataset
                            logger.info(f"🔍 ========== PROCESSING INDOOR SCENE DATASET ==========")
                            logger.info(f"🔍 Creating DatasetInfo for keremberke/indoor-scene-classification")
                            
                            # Indoor scenes classification - 67 indoor categories
                            # Generate generic class names since actual names aren't easily accessible
                            indoor_classes = [f"indoor_scene_{i}" for i in range(67)]
                            
                            # FRAMEWORK-AWARE FIX: Create correct image stats for Indoor Scenes
                            framework = st.session_state.get('selected_framework', 'PyTorch')
                            if framework == "TensorFlow/Keras":
                                indoor_stats = {
                                    "tensorflow_shape": "(224, 224, 3)",
                                    "display_size": "224×224", 
                                    "channels": 3,
                                    "framework_format": "TensorFlow (H,W,C)",
                                    "mean": 0.485, "std": 0.229
                                }
                            else:
                                indoor_stats = {
                                    "pytorch_shape": "(3, 224, 224)",
                                    "display_size": "224×224",
                                    "channels": 3, 
                                    "framework_format": "PyTorch (C,H,W)",
                                    "mean": 0.485, "std": 0.229
                                }
                            
                            dataset_info = DatasetInfo(
                                task_type=task_type,  # Use user-selected task type
                                num_classes=67,
                                num_samples=15620,  # Approximate total samples
                                class_names=indoor_classes,
                                class_distribution={class_name: 233 for class_name in indoor_classes},  # ~15620/67
                                image_size=(224, 224),
                                image_stats=indoor_stats,
                                has_annotations=True,
                                annotation_format="classification",
                                recommended_batch_size=32,
                                estimated_training_time=60.0,
                                dataset_path=None,
                                channels=3,
                                is_hf_dataset=True,
                                hf_dataset_name="keremberke/indoor-scene-classification",
                                hf_subset="full",  # CRITICAL: This dataset requires 'full' subset
                                hf_features=["image", "label"],
                                hf_description="Indoor scene classification dataset with 67 categories"
                            )
                            
                            # DEBUG: Log the created DatasetInfo
                            logger.info(f"🔍 DatasetInfo created:")
                            logger.info(f"🔍   - Task: {dataset_info.task_type}")
                            logger.info(f"🔍   - Classes: {dataset_info.num_classes}")
                            logger.info(f"🔍   - Samples: {dataset_info.num_samples}")
                            logger.info(f"🔍   - Class names: {dataset_info.class_names[:5]}...")
                            logger.info(f"🔍   - Subset: full (REQUIRED)")
                            logger.info(f"🔍 ========== INDOOR SCENE PROCESSING COMPLETE ==========")
                            
                            st.info("✅ Indoor Scene HuggingFace dataset configured: 15,620 total samples")
                            st.info(f"🏠 **Classes:** 67 indoor scene categories")
                            st.info(f"📊 **Distribution:** ~233 images per category")
                            st.info(f"📦 **Subset:** full (required)")
                            
                        # ==================== SEGMENTATION DATASETS ====================
                        elif hf_config['dataset_name'] == 'oxford_iiit_pet':
                            # Oxford-IIIT Pet segmentation - 37 pet breeds + 3 segmentation classes
                            logger.info(f"🔍 ========== PROCESSING OXFORD-IIIT PET SEGMENTATION DATASET ==========")
                            logger.info(f"🔍 Creating DatasetInfo for oxford_iiit_pet")
                            
                            # Segmentation mask classes (not breed classes)
                            pet_seg_classes = ["background", "pet", "border"]
                            
                            # FRAMEWORK-AWARE FIX: Create correct image stats for Oxford Pets
                            framework = st.session_state.get('selected_framework', 'PyTorch')
                            if framework == "TensorFlow/Keras":
                                pet_stats = {
                                    "tensorflow_shape": "(128, 128, 3)",
                                    "display_size": "128×128", 
                                    "channels": 3,
                                    "framework_format": "TensorFlow (H,W,C)",
                                    "mean": 0.485, "std": 0.229
                                }
                            else:
                                pet_stats = {
                                    "pytorch_shape": "(3, 128, 128)",
                                    "display_size": "128×128",
                                    "channels": 3, 
                                    "framework_format": "PyTorch (C,H,W)",
                                    "mean": 0.485, "std": 0.229
                                }
                            
                            dataset_info = DatasetInfo(
                                task_type='segmentation',  # CRITICAL: Must be segmentation
                                num_classes=3,  # Background, Pet, Border for segmentation masks
                                num_samples=7390,  # Total dataset size
                                class_names=pet_seg_classes,  # Segmentation mask classes
                                class_distribution={"background": 7390, "pet": 7390, "border": 7390},
                                image_size=(128, 128),
                                image_stats=pet_stats,
                                has_annotations=True,
                                annotation_format="segmentation_mask",
                                recommended_batch_size=16,
                                estimated_training_time=45.0,
                                dataset_path=None,
                                channels=3,
                                is_hf_dataset=True,
                                hf_dataset_name="oxford_iiit_pet",
                                hf_subset=None,
                                hf_features=["image", "segmentation_mask", "label"],
                                hf_description="Oxford-IIIT Pet segmentation: 37 breeds with 3-class segmentation masks"
                            )
                            
                            logger.info(f"🔍 DatasetInfo created:")
                            logger.info(f"🔍   - Task: {dataset_info.task_type}")
                            logger.info(f"🔍   - Segmentation Classes: {dataset_info.num_classes}")
                            logger.info(f"🔍   - Pet Breeds: 37")
                            logger.info(f"🔍   - Samples: {dataset_info.num_samples}")
                            logger.info(f"🔍   - Class names: {dataset_info.class_names}")
                            logger.info(f"🔍 ========== OXFORD PET SEGMENTATION PROCESSING COMPLETE ==========")
                            
                            st.info("✅ Oxford-IIIT Pet HuggingFace dataset configured: 7,390 total samples")
                            st.info(f"🐾 **Segmentation Classes:** 3 (background, pet, border)")
                            st.info(f"🐕 **Pet Breeds:** 37 different breeds")
                            st.info(f"📊 **Samples:** 7,390 images with segmentation masks")
                            st.info(f"🎯 **Task:** Instance Segmentation")
                            
                        elif hf_config['dataset_name'] == 'scene_parse_150':
                            # ADE20K Scene Parsing - 150 semantic classes
                            logger.info(f"🔍 ========== PROCESSING ADE20K SCENE PARSING DATASET ==========")
                            logger.info(f"🔍 Creating DatasetInfo for scene_parse_150")
                            
                            # Common ADE20K classes (150 total)
                            ade20k_classes = [
                                "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed",
                                "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door",
                                "table", "mountain", "plant", "curtain", "chair", "car", "water",
                                "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field"
                            ] + [f"class_{i}" for i in range(31, 151)]  # Remaining 120 classes
                            
                            framework = st.session_state.get('selected_framework', 'PyTorch')
                            if framework == "TensorFlow/Keras":
                                ade20k_stats = {
                                    "tensorflow_shape": "(256, 256, 3)",
                                    "display_size": "256×256", 
                                    "channels": 3,
                                    "framework_format": "TensorFlow (H,W,C)",
                                    "mean": 0.485, "std": 0.229
                                }
                            else:
                                ade20k_stats = {
                                    "pytorch_shape": "(3, 256, 256)",
                                    "display_size": "256×256",
                                    "channels": 3, 
                                    "framework_format": "PyTorch (C,H,W)",
                                    "mean": 0.485, "std": 0.229
                                }
                            
                            dataset_info = DatasetInfo(
                                task_type='segmentation',
                                num_classes=150,
                                num_samples=22210,  # 20,210 train + 2,000 val (actual ADE20K stats)
                                class_names=ade20k_classes,
                                class_distribution={class_name: 148 for class_name in ade20k_classes},  # ~22210/150
                                image_size=(256, 256),
                                image_stats=ade20k_stats,
                                has_annotations=True,
                                annotation_format="segmentation_mask",
                                recommended_batch_size=8,
                                estimated_training_time=180.0,
                                dataset_path=None,
                                channels=3,
                                is_hf_dataset=True,
                                hf_dataset_name="scene_parse_150",
                                hf_subset=None,
                                hf_features=["image", "annotation"],
                                hf_description="ADE20K Scene Parsing with 150 semantic categories"
                            )
                            
                            logger.info(f"🔍 DatasetInfo created:")
                            logger.info(f"🔍   - Task: {dataset_info.task_type}")
                            logger.info(f"🔍   - Classes: {dataset_info.num_classes}")
                            logger.info(f"🔍   - Samples: {dataset_info.num_samples}")
                            logger.info(f"🔍   - Class names (first 30): {dataset_info.class_names[:30]}")
                            logger.info(f"🔍 ========== ADE20K SCENE PARSING PROCESSING COMPLETE ==========")
                            
                            st.info("✅ ADE20K Scene Parsing HuggingFace dataset configured: 22,210 total samples")
                            st.info(f"🏙️ **Classes:** 150 semantic categories (wall, building, sky, floor, tree, ceiling, road, ...)")
                            st.info(f"📊 **Distribution:** 20,210 train + 2,000 validation images")
                            st.info(f"🎯 **Task:** Semantic Segmentation")
                            
                        else:
                            # Default configuration for other curated datasets
                            # DEBUG: Log when falling back to default configuration
                            logger.warning(f"⚠️ ========== UNKNOWN DATASET - USING DEFAULT CONFIG ==========")
                            logger.warning(f"⚠️ Dataset name '{hf_config['dataset_name']}' not explicitly handled!")
                            logger.warning(f"⚠️ This dataset will use default config (10 classes, 50K samples)")
                            logger.warning(f"⚠️ This is likely a BUG if you expected specific configuration!")
                            logger.warning(f"⚠️ ========== DEFAULT CONFIG WARNING ==========")
                            
                            st.error(f"⚠️ **CRITICAL WARNING:** Dataset '{hf_config['dataset_name']}' not explicitly configured!")
                            st.error(f"⚠️ Using default config (10 classes) - This is likely INCORRECT!")
                            
                            dataset_info = DatasetInfo(
                                task_type=task_type,  # Use user-selected task type
                                num_classes=10,
                                num_samples=50000,
                                class_names=[f"class_{i}" for i in range(10)],
                                class_distribution={f"class_{i}": 5000 for i in range(10)},
                                image_size=(224, 224),
                                image_stats={"mean": 0.5, "std": 0.5},
                                has_annotations=True,
                                annotation_format="classification",
                                recommended_batch_size=32,
                                estimated_training_time=90.0,
                                dataset_path=None,
                                channels=3,
                                is_hf_dataset=True,
                                hf_dataset_name=hf_config['dataset_name'],
                                hf_subset=hf_config.get('subset'),
                                hf_features=hf_config.get('features', ['image', 'label']),
                                hf_description=hf_config.get('description', f"{hf_config['dataset_name']} dataset from HuggingFace")
                            )
                            display_name = hf_config.get('display_name', hf_config['dataset_name'])
                            st.info(f"✅ {display_name} HuggingFace dataset configured")
                        
                        st.session_state.dataset_info = dataset_info
                        st.success("✅ HuggingFace dataset analysis complete! Use **Continue to Model Selection →** in the panel on the right.")
                    
                    # 3. Handle built-in datasets (framework-aware) - only if no other datasets
                    elif has_builtin_dataset:
                        builtin_info = st.session_state.selected_builtin_dataset
                        framework = st.session_state.get('selected_framework', 'PyTorch')
                        
                        # LOGGING: Add comprehensive logging
                        
                        # Create synthetic dataset_info for built-in datasets
                        from core.dataset_analyzer import DatasetInfo
                        import numpy as np
                        
                        # Load dataset info - framework specific
                        try:
                            if framework == "TensorFlow/Keras":
                                import tensorflow as tf
                            
                                # Load the actual TensorFlow dataset first to get accurate dimensions
                                if builtin_info['name'] == 'MNIST':
                                    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
                                    st.info(f"✅ MNIST loaded: {x_train.shape[0]} train, {x_test.shape[0]} test samples")
                                    
                                elif builtin_info['name'] == 'Fashion-MNIST':
                                    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
                                    st.info(f"✅ Fashion-MNIST loaded: {x_train.shape[0]} train, {x_test.shape[0]} test samples")
                                    
                                elif builtin_info['name'] == 'CIFAR-10':
                                    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
                                    st.info(f"✅ CIFAR-10 loaded: {x_train.shape[0]} train, {x_test.shape[0]} test samples")
                                    
                                elif builtin_info['name'] == 'CIFAR-100':
                                    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
                                    st.info(f"✅ CIFAR-100 loaded: {x_train.shape[0]} train, {x_test.shape[0]} test samples")
                                    
                                elif builtin_info['name'] == 'IMDB Reviews':
                                    # Load IMDB dataset
                                    try:
                                        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
                                        st.info(f"✅ IMDB Reviews loaded: {len(x_train)} train, {len(x_test)} test samples")
                                        st.info(f"📝 Text sequences with binary sentiment labels (0=negative, 1=positive)")
                                    except Exception as imdb_error:
                                        st.warning(f"⚠️ IMDB dataset loading failed: {imdb_error}")
                                        # Use fallback data for analysis
                                        x_train, y_train = list(range(25000)), [0] * 12500 + [1] * 12500
                                        x_test, y_test = list(range(25000)), [0] * 12500 + [1] * 12500
                                        st.info(f"📊 Using fallback: 25K train, 25K test samples (synthetic)")
                                        
                                elif builtin_info['name'] == 'Reuters News':
                                    # Load Reuters dataset
                                    try:
                                        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.reuters.load_data(num_words=10000)
                                        st.info(f"✅ Reuters News loaded: {len(x_train)} train, {len(x_test)} test samples")
                                        st.info(f"📰 News articles with 46 topic categories")
                                    except Exception as reuters_error:
                                        st.warning(f"⚠️ Reuters dataset loading failed: {reuters_error}")
                                        # Use fallback data for analysis
                                        x_train, y_train = list(range(8982)), list(range(8982))
                                        x_test, y_test = list(range(2246)), list(range(2246))
                                        st.info(f"📊 Using fallback: 8.9K train, 2.2K test samples (synthetic)")
                                        
                                else:
                                    # Default fallback for unknown datasets
                                    st.warning(f"⚠️ Unknown dataset: {builtin_info['name']}")
                                    x_train, y_train = [], []
                                    
                            elif framework == "PyTorch":
                                import torch
                                import torchvision
                                import torchvision.transforms as transforms
                                from torchvision import datasets
                                
                                
                                # Configure PyTorch datasets (use synthetic data for analysis, no actual downloading)
                                
                                if builtin_info['name'] == 'MNIST':
                                    # Use synthetic data for analysis (no actual dataset loading needed)
                                    x_train = torch.zeros((60000, 28, 28))  # Synthetic for analysis
                                    y_train = torch.zeros(60000)
                                    st.info(f"✅ PyTorch MNIST configured: 60K train, 10K test samples")
                                    
                                elif builtin_info['name'] == 'Fashion-MNIST':
                                    x_train = torch.zeros((60000, 28, 28))  # Synthetic for analysis
                                    y_train = torch.zeros(60000)
                                    st.info(f"✅ PyTorch Fashion-MNIST configured: 60K train, 10K test samples")
                                    
                                elif builtin_info['name'] == 'CIFAR-10':
                                    x_train = torch.zeros((50000, 32, 32, 3))  # Synthetic for analysis
                                    y_train = torch.zeros(50000)
                                    st.info(f"✅ PyTorch CIFAR-10 configured: 50K train, 10K test samples")
                                    
                                elif builtin_info['name'] == 'CIFAR-100':
                                    x_train = torch.zeros((50000, 32, 32, 3))  # Synthetic for analysis
                                    y_train = torch.zeros(50000)
                                    st.info(f"✅ PyTorch CIFAR-100 configured: 50K train, 10K test samples")
                                    
                                elif builtin_info['name'] == 'IMDB Reviews':
                                    # For PyTorch, create synthetic text data for analysis
                                    x_train = list(range(25000))  # Synthetic text sequence IDs
                                    y_train = [0] * 12500 + [1] * 12500  # Binary sentiment labels
                                    st.info(f"✅ PyTorch IMDB Reviews configured: 25K train, 25K test samples")
                                    st.info(f"📝 Text sequences with binary sentiment labels (0=negative, 1=positive)")
                                    
                                elif builtin_info['name'] == 'Reuters News':
                                    # For PyTorch, create synthetic news data for analysis
                                    x_train = list(range(8982))  # Synthetic news article IDs
                                    y_train = list(range(46)) * (8982 // 46 + 1)  # 46 topic categories
                                    y_train = y_train[:8982]  # Trim to exact size
                                    st.info(f"✅ PyTorch Reuters News configured: 8.9K train, 2.2K test samples")
                                    st.info(f"📰 News articles with 46 topic categories")
                                    
                                else:
                                    # Default for unknown datasets
                                    st.warning(f"⚠️ Unknown dataset: {builtin_info['name']}")
                                    x_train, y_train = [], []
                                    
                            else:
                                # Fallback for other frameworks
                                x_train, y_train = [], []
                            
                            # Get original dataset specifications for channels based on builtin_info
                            dataset_spec = builtin_info  # Use builtin_info directly instead of tf_datasets
                            
                            # Smart fallback based on dataset name (in case input_shape is missing)
                            dataset_name = builtin_info.get('name', '')
                            
                            if 'MNIST' in dataset_name or 'Fashion' in dataset_name:
                                default_shape = (28, 28, 1) if framework == "TensorFlow/Keras" else (1, 28, 28)
                            elif 'CIFAR' in dataset_name:
                                default_shape = (32, 32, 3) if framework == "TensorFlow/Keras" else (3, 32, 32)
                            else:
                                default_shape = (224, 224, 3) if framework == "TensorFlow/Keras" else (3, 224, 224)
                            
                            builtin_input_shape = builtin_info.get('input_shape', default_shape)
                            
                            # Handle different format conventions (PyTorch: CHW, TensorFlow: HWC)
                            if framework == "PyTorch" and len(builtin_input_shape) == 3:
                                # PyTorch format: (C, H, W)
                                natural_channels, natural_height, natural_width = builtin_input_shape
                            else:
                                # TensorFlow format: (H, W, C)
                                natural_height, natural_width, natural_channels = builtin_input_shape
                            
                            
                            st.info(f"📏 Original dataset: {x_train.shape[1:] if len(x_train) > 0 else 'N/A'}")
                            st.info(f"🎯 Natural dimensions: {natural_height}×{natural_width}×{natural_channels}")
                            
                            # Create dataset_info object with correct natural dimensions
                            if isinstance(builtin_info['classes'], int):
                                num_classes_val = builtin_info['classes']
                            else:
                                # Extract numeric value from strings like "101 food types" or "10 classes"
                                import re
                                classes_str = builtin_info['classes']
                                match = re.search(r'(\d+)', classes_str)
                                num_classes_val = int(match.group(1)) if match else 2
                            num_samples_val = len(x_train) if hasattr(x_train, '__len__') else 60000
                            
                            # Check if this is a text dataset
                            is_text_dataset = builtin_info.get('task', '').startswith('text') or 'text' in builtin_info.get('task', '').lower()
                            
                            # For text datasets, we don't have natural_height/natural_width, so set defaults
                            if is_text_dataset:
                                natural_height, natural_width = 224, 224  # Default fallback
                                natural_channels = 1  # Default for text

                            if is_text_dataset:
                                # For text datasets, create DatasetInfo with proper class names
                                dataset_name = builtin_info['name']
                                proper_class_names = get_dataset_class_names(dataset_name)
                                
                                if proper_class_names:
                                    class_names = proper_class_names
                                    class_distribution = {class_name: num_samples_val//len(proper_class_names) for class_name in proper_class_names}
                                    st.info(f"📋 **Classes:** {', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''}")
                                else:
                                    # For text datasets, create more meaningful default names
                                    if 'IMDB' in dataset_name:
                                        class_names = ['Negative', 'Positive']
                                    elif 'Reuters' in dataset_name:
                                        class_names = [f"Topic_{i}" for i in range(num_classes_val)]
                                    else:
                                        class_names = [f"Class_{i}" for i in range(num_classes_val)]
                                    class_distribution = {class_name: num_samples_val//len(class_names) for class_name in class_names}
                                
                                dataset_info = DatasetInfo(
                                    task_type=task_type,  # Use user-selected task type (mapped to lowercase)
                                    num_classes=num_classes_val,
                                    num_samples=num_samples_val,
                                    class_names=class_names,
                                    class_distribution=class_distribution,
                                    image_size=(224, 224),  # Default fallback for compatibility
                                    image_stats={'mean': 0.5, 'std': 0.5, 'min': 0.0, 'max': 1.0},
                                    has_annotations=False,
                                    annotation_format=None,
                                    recommended_batch_size=32,
                                    estimated_training_time=5.0,
                                    dataset_path="builtin_text_dataset"
                                )
                                # Mark as text dataset for special handling
                                dataset_info.is_text_dataset = True
                            else:
                                # For image datasets, use natural dimensions and proper class names
                                dataset_name = builtin_info['name']
                                proper_class_names = get_dataset_class_names(dataset_name)
                                
                                if proper_class_names:
                                    class_names = proper_class_names
                                    class_distribution = {class_name: num_samples_val//len(proper_class_names) for class_name in proper_class_names}
                                    st.info(f"📋 **Classes:** {', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''}")
                                else:
                                    # Fallback to generic names
                                    class_names = [f"Class_{i}" for i in range(num_classes_val)]
                                    class_distribution = {f"Class_{i}": num_samples_val//num_classes_val for i in range(num_classes_val)}
                                
                                
                                dataset_info = DatasetInfo(
                                    task_type=builtin_info.get('task', builtin_info.get('task_type', 'classification')),  # Safely get task field
                                    num_classes=num_classes_val,
                                    num_samples=num_samples_val,
                                    class_names=class_names,
                                    class_distribution=class_distribution,
                                    image_size=(natural_height, natural_width),  # Use natural dimensions
                                    image_stats={'mean': 0.5, 'std': 0.5, 'min': 0.0, 'max': 1.0},
                                    has_annotations=False,
                                    annotation_format=None,
                                    recommended_batch_size=32,
                                    estimated_training_time=5.0,
                                    dataset_path="builtin_tf_dataset"
                                )
                            
                            # Add channels information for dynamic input size handling (only for image datasets)
                            if not (hasattr(dataset_info, 'is_text_dataset') and dataset_info.is_text_dataset):
                                dataset_info.channels = natural_channels
                            else:
                                dataset_info.channels = 1  # Default for text datasets
                            
                            
                            # Store additional built-in dataset info
                            dataset_info.builtin_dataset_name = builtin_info['name']
                            # Only set tf_name for TensorFlow datasets
                            if framework == "TensorFlow/Keras" and 'tf_name' in builtin_info:
                                dataset_info.builtin_tf_name = builtin_info['tf_name']
                            dataset_info.is_builtin = True
                            
                            st.session_state.dataset_info = dataset_info
                            st.success(f"✅ {framework} {builtin_info['name']} dataset ready!")
                            
                            
                        except Exception as dataset_error:
                            st.error(f"❌ Failed to load {framework} dataset: {str(dataset_error)}")
                            if framework == "TensorFlow/Keras":
                                st.info("💡 Make sure TensorFlow is properly installed")
                            elif framework == "PyTorch":
                                st.info("💡 Make sure PyTorch and torchvision are properly installed")
                            return
                    
                    # Handle Hugging Face datasets
                    elif hasattr(st.session_state, 'hf_dataset_config') and st.session_state.hf_dataset_config:
                        
                        hf_config = st.session_state.hf_dataset_config
                        framework = st.session_state.get('selected_framework', 'PyTorch')
                        
                        # COMPREHENSIVE LOGGING: Add extensive debugging
                        
                        
                        # Create synthetic dataset_info for HF datasets
                        from core.dataset_analyzer import DatasetInfo
                        import numpy as np
                        
                        # Load Hugging Face dataset for analysis
                        try:
                            # Try to import datasets library
                            try:
                                from datasets import load_dataset
                                import datasets
                            except ImportError:
                                st.error("❌ `datasets` library not installed. Please install with: `pip install datasets`")
                                st.code("pip install datasets", language="bash")
                                return
                            
                            # Build authentication kwargs
                            auth_kwargs = {}
                            if hf_config.get('token'):
                                auth_kwargs['token'] = hf_config['token']
                            
                            # Load dataset sample for analysis
                            
                            with st.spinner(f"Loading Hugging Face dataset: {hf_config['dataset_name']}..."):
                                try:
                                    # Disable datasets caching to prevent resource leaks
                                    from datasets import disable_caching
                                    disable_caching()
                                    
                                    if hf_config.get('subset'):
                                        dataset = load_dataset(
                                            hf_config['dataset_name'], 
                                            hf_config['subset'], 
                                            split='train[:1000]',  # Load sample for analysis
                                            **auth_kwargs
                                        )
                                    else:
                                        dataset = load_dataset(
                                            hf_config['dataset_name'], 
                                            split='train[:1000]',  # Load sample for analysis
                                            **auth_kwargs
                                        )
                                finally:
                                    # Ensure cleanup
                                    import gc
                                    gc.collect()
                            
                            # Analyze dataset structure
                            features = dataset.features
                            num_samples = len(dataset)
                            
                            st.info(f"✅ Hugging Face dataset loaded: {num_samples} samples for analysis")
                            
                            # Auto-detect task type and properties from features
                            task_type = "classification"  # Default
                            num_classes = 2  # Default binary
                            is_text_dataset = False
                            image_size = (224, 224)  # Default
                            channels = 3  # Default RGB
                            
                            # Analyze features to determine task type
                            feature_names = list(features.keys())
                            
                            # Check for image features
                            if any('image' in name.lower() or 'img' in name.lower() or 'pixel' in name.lower() for name in feature_names):
                                # Image dataset
                                is_text_dataset = False
                                
                                # Try to get image properties from first sample
                                try:
                                    first_sample = dataset[0]
                                    for key, value in first_sample.items():
                                        if 'image' in key.lower():
                                            if hasattr(value, 'size'):  # PIL Image
                                                image_size = value.size  # (width, height)
                                                channels = len(value.getbands()) if hasattr(value, 'getbands') else 3
                                            elif isinstance(value, np.ndarray):
                                                if len(value.shape) == 3:
                                                    image_size = (value.shape[1], value.shape[0])  # (width, height)
                                                    channels = value.shape[2]
                                                elif len(value.shape) == 2:
                                                    image_size = (value.shape[1], value.shape[0])
                                                    channels = 1
                                            break
                                except:
                                    pass  # Use defaults
                                    
                            elif any('text' in name.lower() or 'sentence' in name.lower() or 'review' in name.lower() for name in feature_names):
                                # Text dataset
                                is_text_dataset = True
                                image_size = None
                                channels = 1  # Not applicable for text
                            
                            # Check if this is a curated dataset - use metadata instead of sample analysis
                            if hf_config.get('is_curated', False):
                                # Find the curated dataset entry to get correct metadata
                                curated_datasets = {
                                    "🏛️ MIT Places": {
                                        "name": "mit-places",
                                        "description": "Scene recognition dataset",
                                        "samples": "1.8M images",
                                        "classes": "150 scene types",
                                        "task": "Scene Classification", 
                                        "size": "256×256 (min)",
                                        "channels": "RGB (3)",
                                        "verified": True
                                    },
                                    "🎨 CIFAR-10": {
                                        "name": "cifar10",
                                        "description": "Classic object recognition dataset",
                                        "samples": "60K images",
                                        "classes": "10 categories",
                                        "task": "Object Classification", 
                                        "size": "32×32",
                                        "channels": "RGB (3)",
                                        "verified": True
                                    },
                                    "👕 Fashion Items": {
                                        "name": "fashion_mnist",
                                        "description": "Fashion products classification",
                                        "samples": "70K images",
                                        "classes": "10 fashion categories",
                                        "task": "Fashion Classification", 
                                        "size": "28×28",
                                        "channels": "Grayscale (1)",
                                        "verified": True
                                    },
                                    "🍕 Food Categories": {
                                        "name": "food101",
                                        "description": "Food and cuisine classification dataset",
                                        "samples": "101K images",
                                        "classes": "101 food types",
                                        "task": "Food Classification", 
                                        "size": "Variable resolution",
                                        "channels": "RGB (3)",
                                        "verified": True
                                    },
                                    "🐱 Animals": {
                                        "name": "cats_vs_dogs",
                                        "description": "Binary classification of pets",
                                        "samples": "25K images",
                                        "classes": "2 categories",
                                        "task": "Binary Classification", 
                                        "size": "Variable resolution",
                                        "channels": "RGB (3)",
                                        "verified": True
                                    },
                                    "🏠 Indoor Scenes": {
                                        "name": "scene_parse_150",
                                        "description": "Indoor scene understanding",
                                        "samples": "22K images",
                                        "classes": "67 indoor categories",
                                        "task": "Scene Parsing", 
                                        "size": "Variable resolution",
                                        "channels": "RGB (3)",
                                        "verified": True
                                    },
                                }
                                
                                # Find matching curated dataset by name
                                curated_info = None
                                for dataset_display_name, dataset_meta in curated_datasets.items():
                                    if dataset_meta['name'] == hf_config['dataset_name']:
                                        curated_info = dataset_meta
                                        break
                                
                                if curated_info:
                                    # Use curated metadata for class count
                                    if isinstance(curated_info['classes'], int):
                                        num_classes = curated_info['classes']
                                    else:
                                        # Extract numeric value from strings like "101 food types"
                                        import re
                                        classes_str = curated_info['classes']
                                        match = re.search(r'(\d+)', classes_str)
                                        num_classes = int(match.group(1)) if match else 2
                                    
                                else:
                                    # Fallback to sample analysis
                                    if 'label' in feature_names:
                                        try:
                                            labels = [item['label'] for item in dataset]
                                            unique_labels = set(labels)
                                            num_classes = len(unique_labels)
                                        except:
                                            pass
                                    elif 'labels' in feature_names:
                                        try:
                                            labels = [item['labels'] for item in dataset]
                                            unique_labels = set(labels)
                                            num_classes = len(unique_labels)
                                        except:
                                            pass
                            else:
                                # For manual HuggingFace datasets, analyze the sample
                                if 'label' in feature_names:
                                    try:
                                        # Get unique labels from sample
                                        labels = [item['label'] for item in dataset]
                                        unique_labels = set(labels)
                                        num_classes = len(unique_labels)
                                    except:
                                        pass
                                elif 'labels' in feature_names:
                                    try:
                                        labels = [item['labels'] for item in dataset]
                                        unique_labels = set(labels)
                                        num_classes = len(unique_labels)
                                    except:
                                        pass
                            
                            # FRAMEWORK-AWARE FIX: Create correct DatasetInfo for TensorFlow vs PyTorch
                            
                            # FRAMEWORK-SPECIFIC FIXES
                            # 1. Fix image size format for framework compatibility
                            if image_size and isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
                                # Ensure consistent (width, height) format regardless of framework
                                width, height = image_size[0], image_size[1]
                                framework_image_size = (width, height)
                            else:
                                # Default size
                                framework_image_size = (224, 224)
                            
                            # 2. Fix channels detection and validation
                            if channels is None or channels < 1:
                                channels = 3  # Default to RGB
                            
                            # 3. Create framework-aware image stats
                            if framework == "TensorFlow/Keras":
                                # TensorFlow expects data in (height, width, channels) format internally
                                # But image_size should remain (width, height) for consistency
                                tf_shape_info = {
                                    "tensorflow_shape": f"({framework_image_size[1]}, {framework_image_size[0]}, {channels})",  # (H, W, C)
                                    "display_size": f"{framework_image_size[0]}×{framework_image_size[1]}",  # W×H for display
                                    "channels": channels,
                                    "framework_format": "TensorFlow (H,W,C)",
                                    "mean": 0.5, 
                                    "std": 0.5
                                }
                            else:
                                # PyTorch expects (channels, height, width) format
                                tf_shape_info = {
                                    "pytorch_shape": f"({channels}, {framework_image_size[1]}, {framework_image_size[0]})",  # (C, H, W)
                                    "display_size": f"{framework_image_size[0]}×{framework_image_size[1]}",  # W×H for display
                                    "channels": channels,
                                    "framework_format": "PyTorch (C,H,W)",
                                    "mean": 0.5, 
                                    "std": 0.5
                                }
                            
                            
                            dataset_info = DatasetInfo(
                                task_type=task_type,
                                num_classes=num_classes,
                                num_samples=num_samples,
                                class_names=[f"Class_{i}" for i in range(num_classes)],  # Placeholder class names
                                class_distribution={f"Class_{i}": num_samples // num_classes for i in range(num_classes)},  # Estimated
                                image_size=framework_image_size,  # Always (width, height) for consistency
                                image_stats=tf_shape_info,  # Framework-aware stats
                                has_annotations=True,
                                annotation_format="classification",
                                recommended_batch_size=32,
                                estimated_training_time=60.0,
                                dataset_path=None,  # HuggingFace datasets don't have local paths
                                channels=channels,
                                # HuggingFace specific fields (now part of dataclass)
                                is_hf_dataset=True,  # This is the crucial attribute!
                                hf_dataset_name=hf_config['dataset_name'],
                                hf_subset=hf_config.get('subset'),
                                hf_features=list(features.keys()),
                                hf_description=hf_config.get('description', 'Hugging Face Dataset'),
                                # Set text dataset flag if applicable
                                is_text_dataset=is_text_dataset
                            )
                            
                            st.session_state.dataset_info = dataset_info
                            
                            # Verify session state
                            saved_dataset_info = st.session_state.dataset_info
                            if hasattr(saved_dataset_info, 'is_hf_dataset'):
                                pass
                            
                            st.success(f"✅ {framework} Hugging Face dataset '{hf_config['dataset_name']}' analyzed!")
                            
                        except Exception as hf_error:
                            st.error(f"❌ Failed to load Hugging Face dataset: {str(hf_error)}")
                            
                            # Enhanced error handling with intelligent fallbacks
                            error_msg = str(hf_error).lower()
                            if "internet" in error_msg or "network" in error_msg or "connection" in error_msg:
                                st.warning("🌐 **Network Issue Detected**")
                                st.info("💡 **Fallback Option:** Use built-in datasets or upload local files")
                            elif "token" in error_msg or "authenticate" in error_msg:
                                st.warning("🔐 **Authentication Issue**")
                                st.info("💡 **Solution:** Switch to curated datasets (no auth required)")
                            else:
                                st.warning("📊 **Dataset Loading Issue**") 
                                st.info("💡 **Fallback:** Try built-in datasets or local file upload")
                            
                            # Suggest immediate alternatives
                            st.markdown("""
                            **🔄 Quick Alternatives:**
                            - 🌐 **Built-in Datasets:** Switch to TensorFlow/PyTorch built-ins
                            - 📁 **Local Files:** Upload your own images  
                            - 🔄 **Try Again:** Check internet connection and retry
                            """)
                            return
                            
                    else:
                        st.error("❌ No dataset selected for analysis")
                        st.info("💡 Please select a dataset (built-in, upload, or Hugging Face) before analyzing")
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
    
    with col2:
        section("📊", "Analysis Results")

        if hasattr(st.session_state, 'dataset_info') and st.session_state.dataset_info is not None:
            info = st.session_state.dataset_info
            framework = st.session_state.get('selected_framework', 'PyTorch')

            # Display key metrics in a simple vertical layout
            st.metric("Task Type", info.task_type.title())
            st.metric("Number of Classes", info.num_classes)
            st.metric("Number of Samples", f"{info.num_samples:,}")
            
            # FRAMEWORK-AWARE DISPLAY FIX for HuggingFace datasets
            if hasattr(info, 'is_hf_dataset') and info.is_hf_dataset:
                st.markdown("### 🤗 HuggingFace Dataset Information")
                
                # Display HF-specific info with framework awareness
                st.info(f"📊 **Dataset**: {info.hf_dataset_name}")
                if info.hf_subset:
                    st.info(f"📂 **Subset**: {info.hf_subset}")
                
                # Framework-specific image size display
                if hasattr(info, 'image_stats') and isinstance(info.image_stats, dict):
                    display_size = info.image_stats.get('display_size', 'N/A')
                    framework_format = info.image_stats.get('framework_format', 'Unknown')
                    channels = info.image_stats.get('channels', info.channels if hasattr(info, 'channels') else 3)
                    
                    st.metric("Image Size", display_size)
                    st.metric("Channels", f"{channels} ({'Grayscale' if channels == 1 else 'RGB' if channels == 3 else 'RGBA' if channels == 4 else f'{channels}-channel'})")
                    
                    # Show framework-specific tensor shape
                    if framework == "TensorFlow/Keras" and 'tensorflow_shape' in info.image_stats:
                        st.info(f"🧠 **TensorFlow Shape**: {info.image_stats['tensorflow_shape']} (Height, Width, Channels)")
                    elif framework == "PyTorch" and 'pytorch_shape' in info.image_stats:
                        st.info(f"🔥 **PyTorch Shape**: {info.image_stats['pytorch_shape']} (Channels, Height, Width)")
                    
                else:
                    # Fallback display for legacy or incomplete data
                    if hasattr(info, 'image_size') and info.image_size:
                        if isinstance(info.image_size, (list, tuple)) and len(info.image_size) >= 2:
                            st.metric("Image Size", f"{info.image_size[0]}×{info.image_size[1]}")
                        else:
                            st.metric("Image Size", str(info.image_size))
                    
                    if hasattr(info, 'channels') and info.channels:
                        st.metric("Channels", f"{info.channels}")
                
            # Handle different dataset types (non-HuggingFace)
            elif hasattr(info, 'image_size') and info.image_size:
                if isinstance(info.image_size, (list, tuple)) and len(info.image_size) >= 2:
                    st.metric("Image Size", f"{info.image_size[0]}×{info.image_size[1]}")
                    
                    # Show framework-specific tensor format for regular datasets too
                    if hasattr(info, 'channels') and info.channels:
                        channels = info.channels
                        width, height = info.image_size[0], info.image_size[1]
                        
                        if framework == "TensorFlow/Keras":
                            st.info(f"🧠 **TensorFlow Format**: ({height}, {width}, {channels}) - (Height, Width, Channels)")
                        elif framework == "PyTorch":
                            st.info(f"🔥 **PyTorch Format**: ({channels}, {height}, {width}) - (Channels, Height, Width)")
                else:
                    st.metric("Image Size", str(info.image_size))
            elif hasattr(info, 'is_text_dataset') and info.is_text_dataset:
                st.metric("Data Type", "Text Dataset")
            else:
                st.metric("Image Size", "N/A")
            
            # Class distribution
            if info.class_distribution:
                st.markdown("<br>", unsafe_allow_html=True)
                section("📊", "Class Distribution")

                if isinstance(info.class_distribution, dict):
                    df_cls = pd.DataFrame(
                        list(info.class_distribution.items()),
                        columns=['Class', 'Count']
                    ).sort_values('Count', ascending=False)

                    if len(df_cls) > 1:
                        # Class imbalance check
                        max_c, min_c = df_cls['Count'].max(), df_cls['Count'].min()
                        imbalance_ratio = max_c / max(min_c, 1)
                        if imbalance_ratio > 5:
                            st.markdown(
                                f'<div style="background:rgba(255,107,107,.08);border:1px solid rgba(255,107,107,.3);'
                                f'border-radius:8px;padding:.7rem 1rem;font-size:.85rem;margin-bottom:.5rem">'
                                f'⚠️ <b>Class Imbalance Detected</b> — max/min ratio is {imbalance_ratio:.1f}×. '
                                f'Consider oversampling minority classes or using weighted loss.</div>',
                                unsafe_allow_html=True,
                            )
                        elif imbalance_ratio > 2:
                            st.markdown(
                                f'<div style="background:rgba(255,217,61,.08);border:1px solid rgba(255,217,61,.3);'
                                f'border-radius:8px;padding:.7rem 1rem;font-size:.85rem;margin-bottom:.5rem">'
                                f'ℹ️ <b>Mild Imbalance</b> — max/min ratio is {imbalance_ratio:.1f}×. Training should still work well.</div>',
                                unsafe_allow_html=True,
                            )

                        # Styled bar chart
                        n_show = min(len(df_cls), 30)
                        bar_colors = ["#6c63ff" if i < n_show // 2 else "#00d4aa"
                                      for i in range(n_show)]
                        fig_cls = go.Figure(go.Bar(
                            x=df_cls['Class'].iloc[:n_show],
                            y=df_cls['Count'].iloc[:n_show],
                            marker_color=bar_colors,
                            text=df_cls['Count'].iloc[:n_show],
                            textposition="outside",
                        ))
                        fig_cls.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            height=300, margin=dict(l=0, r=0, t=10, b=0),
                            xaxis_tickangle=-30,
                            xaxis=dict(gridcolor="rgba(255,255,255,.04)"),
                            yaxis=dict(title="Samples", gridcolor="rgba(255,255,255,.06)"),
                            showlegend=False,
                        )
                        st.plotly_chart(fig_cls, use_container_width=True, config={"displayModeBar": False})

                        if len(df_cls) > 30:
                            st.caption(f"Showing top 30 of {len(df_cls)} classes.")
            
            # Recommendations
            st.markdown("<br>", unsafe_allow_html=True)
            section("💡", "Recommendations")
            r_c1, r_c2 = st.columns(2)
            with r_c1:
                st.markdown(
                    f'<div class="cv-card" style="border-left:3px solid var(--accent2)">'
                    f'<div style="font-size:.75rem;text-transform:uppercase;color:var(--text-secondary);margin-bottom:.3rem">Batch Size</div>'
                    f'<div style="font-weight:700;font-size:1.1rem">{info.recommended_batch_size}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with r_c2:
                est_h = info.estimated_training_time
                est_label = f"{est_h:.1f}h" if est_h >= 1 else f"{est_h*60:.0f}min"
                st.markdown(
                    f'<div class="cv-card" style="border-left:3px solid var(--accent)">'
                    f'<div style="font-size:.75rem;text-transform:uppercase;color:var(--text-secondary);margin-bottom:.3rem">Est. Training Time</div>'
                    f'<div style="font-weight:700;font-size:1.1rem">{est_label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            
        else:
            st.info("👆 Analyze a dataset to see results here")

    # ── Augmentation Preview ──────────────────────────────────────────────────
    # Lives at the PAGE level (not inside col2) so that:
    #   aug_col1/aug_col2 = level-0 columns (top-level)
    #   grid_cols         = level-1 columns (inside aug_col2) ← Streamlit max
    if st.session_state.get("dataset_info") is not None:
        _aug_info = st.session_state.dataset_info

        st.markdown("---")
        section("🎨", "Augmentation Preview")
        st.markdown(
            "<p style='font-size:.88rem;color:var(--text-secondary);margin-bottom:.8rem'>"
            "See how augmentations will transform your images before training.</p>",
            unsafe_allow_html=True,
        )

        aug_col1, aug_col2 = st.columns([1, 2])
        with aug_col1:
            aug_preset = st.selectbox(
                "Augmentation Preset",
                ["Light", "Medium", "Heavy", "Custom"],
                index=1,
            )
            num_aug_previews = st.slider("Preview count", 4, 16, 8, step=4)

            if aug_preset == "Custom":
                do_hflip  = st.checkbox("Horizontal Flip", value=True)
                do_vflip  = st.checkbox("Vertical Flip",   value=False)
                do_rotate = st.slider("Max Rotation (°)", 0, 90, 15)
                do_bright = st.slider("Brightness Jitter", 0.0, 0.8, 0.2)
                do_crop   = st.slider("Random Crop %",    0, 30, 10)
                do_blur   = st.checkbox("Gaussian Blur",  value=False)
            else:
                _AUG_PRESETS = {
                    "Light":  dict(do_hflip=True,  do_vflip=False, do_rotate=0,
                                   do_bright=0.1, do_crop=5,  do_blur=False),
                    "Medium": dict(do_hflip=True,  do_vflip=False, do_rotate=15,
                                   do_bright=0.3, do_crop=10, do_blur=False),
                    "Heavy":  dict(do_hflip=True,  do_vflip=True,  do_rotate=45,
                                   do_bright=0.5, do_crop=20, do_blur=True),
                }
                _c = _AUG_PRESETS.get(aug_preset, _AUG_PRESETS["Medium"])
                do_hflip  = _c["do_hflip"];  do_vflip  = _c["do_vflip"]
                do_rotate = _c["do_rotate"]; do_bright = _c["do_bright"]
                do_crop   = _c["do_crop"];   do_blur   = _c["do_blur"]

            preview_btn = st.button(
                "🔄 Generate Previews", type="primary", use_container_width=True
            )

        with aug_col2:
            if preview_btn:
                try:
                    from torchvision import transforms
                    from PIL import Image as _PIL
                    import numpy as _np

                    tfs = []
                    if do_crop > 0:
                        tfs.append(
                            transforms.RandomResizedCrop(224, scale=(1 - do_crop / 100, 1.0))
                        )
                    else:
                        tfs.append(transforms.Resize((224, 224)))
                    if do_hflip:  tfs.append(transforms.RandomHorizontalFlip())
                    if do_vflip:  tfs.append(transforms.RandomVerticalFlip())
                    if do_rotate: tfs.append(transforms.RandomRotation(do_rotate))
                    if do_bright:
                        tfs.append(transforms.ColorJitter(
                            brightness=do_bright,
                            contrast=do_bright * 0.5,
                            saturation=do_bright * 0.5,
                        ))
                    if do_blur:   tfs.append(transforms.GaussianBlur(3, sigma=(0.1, 2.0)))
                    transform = transforms.Compose(tfs)

                    src_images = []
                    if hasattr(_aug_info, "dataset_path") and _aug_info.dataset_path:
                        _dp = Path(_aug_info.dataset_path)
                        for _ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                            src_images.extend(list(_dp.rglob(_ext))[:20])

                    # Universal fallback: colourful gradient placeholder
                    if not src_images:
                        _arr = _np.zeros((224, 224, 3), dtype=_np.uint8)
                        for _i in range(224):
                            _arr[_i, :] = [_i, 255 - _i, 128]
                        src_images = [_PIL.fromarray(_arr)]

                    # grid_cols is at nesting level 1 (inside aug_col2 which is level 0) ✓
                    grid_cols = st.columns(4)
                    for k in range(num_aug_previews):
                        _src = src_images[k % len(src_images)]
                        if isinstance(_src, Path):
                            _src = _PIL.open(_src).convert("RGB")
                        grid_cols[k % 4].image(
                            transform(_src),
                            use_column_width=True,
                            caption=f"Aug #{k + 1}",
                        )

                except ImportError:
                    st.warning("⚠️ torchvision is required: pip install torchvision")
                except Exception as _aug_err:
                    st.error(f"Preview failed: {_aug_err}")
            else:
                st.info("👈 Configure augmentations on the left and click **Generate Previews**.")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(
            "➡️ Continue to Model Selection",
            key="continue_to_model",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.current_step = 2
            st.session_state["current_tool"] = None
            st.session_state.project_started = True
            st_rerun()



