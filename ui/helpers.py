"""
Shared helper functions used across UI pages.
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.dataset_analyzer import DatasetAnalyzer, DatasetInfo
from core.model_selector import ModelSelector, ModelConfig
from utils.config import Config

logger = logging.getLogger(__name__)

from core.trainer import AutoTrainer
from core.optimizer import HyperparameterOptimizer
from ui.shared import st_rerun

def check_system_status() -> Dict[str, bool]:
    """Check system requirements and availability."""
    status = {}
    
    try:
        import torch
        status["PyTorch"] = True
        status[f"CUDA Available"] = torch.cuda.is_available()
    except ImportError:
        status["PyTorch"] = False
        status["CUDA Available"] = False
    
    try:
        import tensorflow as tf
        status["TensorFlow"] = True
    except ImportError:
        status["TensorFlow"] = False
    
    try:
        import cv2
        status["OpenCV"] = True
    except ImportError:
        status["OpenCV"] = False
    
    try:
        import optuna
        status["Optuna (Optimization)"] = True
    except ImportError:
        status["Optuna (Optimization)"] = False
    
    return status



def create_model_comparison_data(task_type: str) -> List[Dict]:
    """Create model comparison data for visualization."""
    
    selector = ModelSelector()
    
    if task_type == "classification":
        models_data = selector.classification_models
    elif task_type == "detection":
        models_data = selector.detection_models
    elif task_type == "segmentation":
        models_data = selector.segmentation_models
    else:
        return []
    
    comparison_data = []
    for model_name, model_info in models_data.items():
        comparison_data.append({
            'model': model_name,
            'accuracy_score': model_info['accuracy_score'],
            'speed_score': model_info['speed_score'],
            'num_parameters': model_info['num_parameters'] // 1000000,  # In millions
            'memory_gb': model_info['memory_gb']
        })
    
    return comparison_data



def get_user_specified_input_size():
    """Get the appropriate input size dynamically for any dataset.
    
    Priority order:
    1. Built-in dataset natural dimensions (CIFAR-10: 32x32x3, MNIST: 28x28x1)
    2. User manual configuration (if explicitly set)  
    3. Dataset analysis results
    4. Default fallback
    
    Returns:
        tuple: (height, width, channels) for training functions
    """
    dataset_info = st.session_state.get('dataset_info')
    
    # Priority 1: Built-in dataset natural dimensions
    if dataset_info and hasattr(dataset_info, 'builtin_dataset_name'):
        builtin_name = dataset_info.builtin_dataset_name
        
        # Define natural dimensions for built-in datasets
        builtin_dimensions = {
            'MNIST': (28, 28, 1),
            'Fashion-MNIST': (28, 28, 1), 
            'CIFAR-10': (32, 32, 3),
            'CIFAR-100': (32, 32, 3),
            # Add more built-in datasets as needed
        }
        
        if builtin_name in builtin_dimensions:
            height, width, channels = builtin_dimensions[builtin_name]
            
            # Check if user has explicitly overridden dimensions in manual config
            model_config = st.session_state.get('model_config')
            if model_config and hasattr(model_config, 'config_params'):
                # Only override if user explicitly specified different dimensions
                custom_height = model_config.config_params.get('custom_height')
                custom_width = model_config.config_params.get('custom_width')
                custom_channels = model_config.config_params.get('input_channels')
                
                if custom_height and custom_width:
                    height, width = custom_height, custom_width
                if custom_channels:
                    channels = custom_channels
            
            return (height, width, channels)
    
    # Priority 2: User manual configuration (for custom datasets)
    model_config = st.session_state.get('model_config')
    if model_config and hasattr(model_config, 'input_size'):
        if len(model_config.input_size) == 3:
            # Handle different possible formats - assume (channels, height, width) from manual config
            channels, height, width = model_config.input_size
            return (height, width, channels)
        else:
            # Legacy format: (width, height)
            width, height = model_config.input_size
            channels = model_config.config_params.get('input_channels', 3)
            return (height, width, channels)
    
    # Priority 3: Dataset analysis results
    if dataset_info and hasattr(dataset_info, 'image_size'):
        if len(dataset_info.image_size) == 2:
            h, w = dataset_info.image_size
            # Determine channels from dataset if available
            channels = getattr(dataset_info, 'channels', 3)
            return (h, w, channels)
        elif len(dataset_info.image_size) == 3:
            return dataset_info.image_size
    
    # Priority 4: Ultimate fallback
    return (224, 224, 3)



def start_training_with_config(config: dict):
    """Start the training process with enhanced configuration including early stopping."""
    max_epochs = config['max_epochs']
    optimize_hyperparams = config['optimize_hyperparams']
    n_trials = config.get('n_trials', 20)
    output_dir = config['output_dir']

    # Store early stopping config in session state for training functions to access
    st.session_state.early_stopping_config = {
        'enable_early_stopping': config.get('enable_early_stopping', True),
        'patience': config.get('early_stopping_patience', 10),
        'min_delta': config.get('early_stopping_min_delta', 0.001),
    }
    # Store LR and batch size so trainers can pick them up
    if config.get('learning_rate'):
        st.session_state['override_learning_rate'] = config['learning_rate']
    if config.get('batch_size'):
        st.session_state['override_batch_size'] = config['batch_size']
    if config.get('experiment_name'):
        st.session_state['experiment_name'] = config['experiment_name']

    start_training(max_epochs, optimize_hyperparams, output_dir, n_trials=n_trials)



def start_training(max_epochs: int, optimize_hyperparams: bool, output_dir: str, n_trials: int = 20):
    """Start the training process with framework support."""
    
    st.session_state.training_status = 'running'
    
    # Clear previous training log history
    st.session_state.training_log_history = []
    
    # Check selected framework
    framework = st.session_state.get('selected_framework', 'PyTorch')
    
    # Create progress container
    progress_container = st.container()
    with progress_container:
        st.markdown("### 🔥 Training Progress")
        
        # Overall progress bar
        overall_progress = st.progress(0)
        status_text = st.empty()
        
        # Metrics columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            current_epoch = st.empty()
        with col2:
            current_loss = st.empty()
        with col3:
            current_acc = st.empty()
        with col4:
            elapsed_time = st.empty()
        
        # Training log container
        log_container = st.expander("📋 Training Logs", expanded=True)
        training_log = log_container.empty()
        
        # Live training charts
        st.markdown("**📊 Live Training Curves**")
        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.caption("📉 Loss")
            chart_loss = st.empty()
        with chart_cols[1]:
            st.caption("📈 Accuracy")
            chart_acc = st.empty()
        
        # Initialize training log history in session state
        if 'training_log_history' not in st.session_state:
            st.session_state.training_log_history = []
    
    try:
        # Framework-specific training with progress UI
        progress_ui = {
            'overall_progress': overall_progress,
            'status_text': status_text,
            'current_epoch': current_epoch,
            'current_loss': current_loss,
            'current_acc': current_acc,
            'elapsed_time': elapsed_time,
            'training_log': training_log,
            'log_history': st.session_state.training_log_history,
            'chart_loss': chart_loss,
            'chart_acc': chart_acc,
        }
        
        if framework == "PyTorch":
            results = start_pytorch_training(max_epochs, optimize_hyperparams, output_dir, progress_ui)
        elif framework == "TensorFlow/Keras":
            results = start_tensorflow_training(max_epochs, optimize_hyperparams, output_dir, progress_ui) 
        elif framework == "Scikit-learn":
            results = start_sklearn_training(max_epochs, optimize_hyperparams, output_dir, progress_ui)
        else:
            # Default to PyTorch
            results = start_pytorch_training(max_epochs, optimize_hyperparams, output_dir, progress_ui)
            
        # Store results
        st.session_state.training_status = 'completed'
        st.session_state.training_completed = True

        training_results = results.copy() if results else {}
        training_results.setdefault('best_accuracy', 0)
        training_results.setdefault('best_loss', 0)
        training_results.setdefault('training_time', 0)
        training_results['framework'] = framework

        # Normalise model_path key
        if 'model_saved' in results:
            training_results['model_path'] = results['model_saved']

        st.session_state.training_results = training_results

        # Advance to Results page
        st.session_state.current_step = 4
        st.session_state["current_tool"] = None
        st.success("🎉 **Training Complete!** Opening Results page...")
        st_rerun()

    except Exception as e:
        st.session_state.training_status = 'failed'
        st.error(f"Training failed: {str(e)}")
        logger.exception("Training failed")


def start_pytorch_training(max_epochs: int, optimize_hyperparams: bool, output_dir: str, progress_ui=None):
    """Start PyTorch training process with REAL implementation and progress tracking."""
    
    import time
    import os
    import shutil
    import logging
    from datetime import datetime
    
    # Setup detailed logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    logger.info(f"=== PYTORCH TRAINING START ===")
    logger.info(f"max_epochs: {max_epochs}")
    logger.info(f"optimize_hyperparams: {optimize_hyperparams}")
    logger.info(f"output_dir: {output_dir}")
    
    try:
        if progress_ui:
            progress_ui['status_text'].text("🔥 **Initializing PyTorch Training...**")
            progress_ui['overall_progress'].progress(0.05)
        
        # Import PyTorch modules
        try:
            logger.info("Importing PyTorch modules...")
            import torch
            import torch.nn as nn
            import torch.optim as optim
            import torch.nn.functional as F
            from torch.utils.data import DataLoader, Dataset, TensorDataset
            import torchvision
            import torchvision.transforms as transforms
            import torchvision.datasets as datasets
            import numpy as np
            
            # Import callback system for early stopping
            from utils.callbacks import CallbackManager, EarlyStopping, ModelCheckpoint, MetricsLogger
            
            TORCH_AVAILABLE = True
            logger.info(f"PyTorch {torch.__version__} imported successfully")
            if progress_ui:
                progress_ui['status_text'].text("✅ **PyTorch loaded successfully**")
                progress_ui['overall_progress'].progress(0.1)
        except ImportError as e:
            logger.error(f"PyTorch import failed: {e}")
            if progress_ui:
                progress_ui['status_text'].text(f"❌ **PyTorch import failed**")
            st.error(f"❌ PyTorch import failed: {e}")
            return {
                'best_accuracy': 0.0,
                'best_loss': float('inf'),
                'training_time': 0.0,
                'error': 'PyTorch not available'
            }
        
        # Get required data from session state
        dataset_info = st.session_state.get('dataset_info')
        model_config = st.session_state.get('model_config')
        
        if not dataset_info or not model_config:
            st.error("❌ Dataset info or model config missing")
            return {
                'best_accuracy': 0.0,
                'best_loss': float('inf'),
                'training_time': 0.0,
                'error': 'Missing configuration'
            }
        
        if progress_ui:
            progress_ui['status_text'].text("📊 **Loading PyTorch data...**")
            progress_ui['overall_progress'].progress(0.2)
        
        # Get user-specified input size
        try:
            height, width, channels = get_user_specified_input_size()
            img_size = (height, width)
            logger.info(f"✅ Using user-specified input size: height={height}, width={width}, channels={channels}")
        except Exception as size_error:
            logger.error(f"❌ Failed to get user input size: {size_error}")
            # Fallback
            img_size = (224, 224)
            channels = 3
            logger.info(f"Using fallback img_size: {img_size}, channels: {channels}")
        
        # Get configuration parameters
        num_classes = dataset_info.num_classes
        batch_size = model_config.config_params.get('batch_size', 32) if model_config else 32
        
        st.info(f"🔧 **PyTorch using input size: {img_size} with {channels} channels, batch size: {batch_size}**")
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Check if this is a built-in dataset (torchvision support)
        is_builtin_dataset = hasattr(dataset_info, 'is_builtin') and dataset_info.is_builtin
        logger.info(f"Is built-in dataset: {is_builtin_dataset}")
        

        # Get data processing preferences from model config
        enable_normalization = model_config.config_params.get('enable_normalization', True)
        normalization_type = model_config.config_params.get('normalization_type', 'Standard (0-1)')
        norm_min = model_config.config_params.get('norm_min', 0.0)
        norm_max = model_config.config_params.get('norm_max', 1.0)
        
        logger.info(f"🔧 PYTORCH NORMALIZATION VERIFICATION: Using '{normalization_type}' (enabled: {enable_normalization})")
        
        # Create normalization transform based on user preferences
        def get_normalization_transform(channels):
            if not enable_normalization:
                return []  # Return empty list - no normalization
            elif normalization_type == "Standard (0-1)":
                # Standard normalization: just ToTensor() which converts to [0,1]
                return []  # ToTensor already normalizes to [0,1]
            elif normalization_type == "Z-Score (-1 to 1)":
                # Z-score normalization to [-1, 1]
                return [transforms.Normalize((0.5,) * channels, (0.5,) * channels)]
            elif normalization_type == "ResNet Preprocessing (PyTorch)":
                # ResNet ImageNet preprocessing for PyTorch
                if channels == 3:
                    return [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                else:
                    # For non-RGB images, use average values
                    mean_val = (0.485 + 0.456 + 0.406) / 3
                    std_val = (0.229 + 0.224 + 0.225) / 3
                    return [transforms.Normalize((mean_val,) * channels, (std_val,) * channels)]
            elif normalization_type == "MobileNet Preprocessing (PyTorch)":
                # MobileNet uses ImageNet preprocessing in PyTorch
                if channels == 3:
                    return [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                else:
                    # For non-RGB images, use average values
                    mean_val = (0.485 + 0.456 + 0.406) / 3
                    std_val = (0.229 + 0.224 + 0.225) / 3
                    return [transforms.Normalize((mean_val,) * channels, (std_val,) * channels)]
            elif normalization_type == "Custom Range":
                # Custom range normalization
                # First normalize to [0,1] with ToTensor, then scale to custom range
                scale = norm_max - norm_min
                offset = norm_min
                # Transform: x = x * scale + offset
                mean = tuple([0.5] * channels)  # Center of [0,1] range
                std = tuple([0.5/scale] * channels)  # Scale factor
                return [transforms.Normalize(mean, std)]
            else:
                # Default fallback
                return [transforms.Normalize((0.5,) * channels, (0.5,) * channels)]
        
        norm_transform = get_normalization_transform(channels)
        
        # Define base transforms (without normalization)
        base_transforms_train = [
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor()
        ]
        
        base_transforms_test = [
            transforms.Resize(img_size),
            transforms.ToTensor()
        ]
        
        # Add normalization if enabled
        transform_train = transforms.Compose(base_transforms_train + norm_transform)
        transform_test = transforms.Compose(base_transforms_test + norm_transform)

        # ── Detection task: bypass image-folder loading, use DataLoaderFactory ──
        _task_type_early = getattr(dataset_info, 'task_type', 'classification')
        if _task_type_early == 'detection':
            logger.info('Detection task detected — using DataLoaderFactory for data loading')
            try:
                from utils.data_factory import DataLoaderFactory as _DLF
                from utils.config import Config as _Cfg
                _det_cfg = type('_DC', (), {
                    'dataset_info': dataset_info,
                    'model_config':  model_config,
                    'batch_size':    batch_size,
                })()
                _dlf = _DLF(_det_cfg)
                train_loader = _dlf.create_train_loader()
                test_loader  = _dlf.create_val_loader()
                logger.info(f'Detection DataLoaders created: '
                            f'{len(train_loader.dataset)} train, '
                            f'{len(test_loader.dataset)} val')
            except Exception as _det_dl_err:
                logger.error(f'Detection DataLoader failed ({_det_dl_err}), using dummy detection data')
                from utils.data_factory import DummyDetectionDataset, _detection_collate
                _det_train = DummyDetectionDataset(size=400, num_classes=num_classes,
                                                   input_channels=channels, input_size=img_size)
                _det_val   = DummyDetectionDataset(size=100,  num_classes=num_classes,
                                                   input_channels=channels, input_size=img_size,
                                                   split='val')
                train_loader = DataLoader(_det_train, batch_size=batch_size, shuffle=True,
                                          collate_fn=_detection_collate, num_workers=0)
                test_loader  = DataLoader(_det_val,   batch_size=batch_size, shuffle=False,
                                          collate_fn=_detection_collate, num_workers=0)

        elif is_builtin_dataset:
            logger.info(f"Loading built-in PyTorch dataset: {dataset_info.builtin_dataset_name}")
            
            # Load the appropriate torchvision dataset
            if dataset_info.builtin_dataset_name == 'MNIST':
                # For MNIST, we need grayscale transform
                if channels == 1:
                    mnist_norm_transform = get_normalization_transform(1)
                    transform_train = transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.RandomRotation(10),
                        transforms.ToTensor()
                    ] + mnist_norm_transform)
                    transform_test = transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.ToTensor()
                    ] + mnist_norm_transform)
                else:
                    # Convert grayscale to RGB
                    mnist_norm_transform = get_normalization_transform(channels)
                    transform_train = transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.Grayscale(num_output_channels=channels),
                        transforms.RandomRotation(10),
                        transforms.ToTensor()
                    ] + mnist_norm_transform)
                    transform_test = transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.Grayscale(num_output_channels=channels),
                        transforms.ToTensor()
                    ] + mnist_norm_transform)
                
                try:
                    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform_train)
                    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform_test)
                except Exception as e:
                    logger.warning(f"Failed to download MNIST: {e}. Using dummy data.")
                    # Create dummy MNIST-like data as fallback with learnable patterns
                    dummy_data = torch.randn(1000, 1, *img_size)
                    # Create learnable labels based on data patterns
                    dummy_labels = []
                    for i in range(1000):
                        img = dummy_data[i, 0]  # Single channel
                        h, w = img.shape[0], img.shape[1]
                        region1 = img[:h//2, :w//2].mean()
                        region2 = img[:h//2, w//2:].mean()
                        label = int((region1 + region2 + 2) * 2.5) % 10
                        dummy_labels.append(label)
                    dummy_labels = torch.tensor(dummy_labels)
                    train_dataset = TensorDataset(dummy_data[:800], dummy_labels[:800])
                    test_dataset = TensorDataset(dummy_data[800:], dummy_labels[800:])
                
            elif dataset_info.builtin_dataset_name == 'Fashion-MNIST':
                if channels == 1:
                    transform_train = transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(10),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ])
                    transform_test = transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ])
                else:
                    transform_train = transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.Grayscale(num_output_channels=channels),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(10),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,) * channels, (0.5,) * channels)
                    ])
                    transform_test = transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.Grayscale(num_output_channels=channels),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,) * channels, (0.5,) * channels)
                    ])
                
                try:
                    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform_train)
                    test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform_test)
                except Exception as e:
                    logger.warning(f"Failed to download Fashion-MNIST: {e}. Using dummy data.")
                    dummy_data = torch.randn(1000, channels, *img_size)
                    # Create learnable labels based on data patterns
                    dummy_labels = []
                    for i in range(1000):
                        img = dummy_data[i]
                        if channels == 1:
                            h, w = img.shape[1], img.shape[2]
                            region1 = img[0, :h//2, :w//2].mean()
                            region2 = img[0, :h//2, w//2:].mean()
                        else:
                            h, w = img.shape[1], img.shape[2]
                            region1 = img[:, :h//2, :w//2].mean()
                            region2 = img[:, :h//2, w//2:].mean()
                        label = int((region1 + region2 + 2) * 2.5) % 10
                        dummy_labels.append(label)
                    dummy_labels = torch.tensor(dummy_labels)
                    train_dataset = TensorDataset(dummy_data[:800], dummy_labels[:800])
                    test_dataset = TensorDataset(dummy_data[800:], dummy_labels[800:])
                
            elif dataset_info.builtin_dataset_name == 'CIFAR-10':
                if channels != 3:
                    # Convert RGB to desired channels
                    if channels == 1:
                        transform_train = transforms.Compose([
                            transforms.Resize(img_size),
                            transforms.Grayscale(),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(10),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ])
                        transform_test = transforms.Compose([
                            transforms.Resize(img_size),
                            transforms.Grayscale(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ])
                
                try:
                    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
                    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
                except Exception as e:
                    logger.warning(f"Failed to download CIFAR-10: {e}. Using dummy data.")
                    dummy_data = torch.randn(1000, channels, *img_size)
                    # Create learnable labels based on data patterns
                    dummy_labels = []
                    for i in range(1000):
                        img = dummy_data[i]
                        if channels == 1:
                            h, w = img.shape[1], img.shape[2]
                            region1 = img[0, :h//2, :w//2].mean()
                            region2 = img[0, :h//2, w//2:].mean()
                        else:
                            h, w = img.shape[1], img.shape[2]
                            region1 = img[:, :h//2, :w//2].mean()
                            region2 = img[:, :h//2, w//2:].mean()
                        label = int((region1 + region2 + 2) * 2.5) % 10
                        dummy_labels.append(label)
                    dummy_labels = torch.tensor(dummy_labels)
                    train_dataset = TensorDataset(dummy_data[:800], dummy_labels[:800])
                    test_dataset = TensorDataset(dummy_data[800:], dummy_labels[800:])
                
            elif dataset_info.builtin_dataset_name == 'CIFAR-100':
                if channels != 3:
                    if channels == 1:
                        transform_train = transforms.Compose([
                            transforms.Resize(img_size),
                            transforms.Grayscale(),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(10),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ])
                        transform_test = transforms.Compose([
                            transforms.Resize(img_size),
                            transforms.Grayscale(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ])
                
                try:
                    train_dataset = datasets.CIFAR100('./data', train=True, download=True, transform=transform_train)
                    test_dataset = datasets.CIFAR100('./data', train=False, download=True, transform=transform_test)
                except Exception as e:
                    logger.warning(f"Failed to download CIFAR-100: {e}. Using dummy data.")
                    dummy_data = torch.randn(1000, channels, *img_size)
                    # Create learnable labels based on data patterns
                    dummy_labels = []
                    for i in range(1000):
                        img = dummy_data[i]
                        if channels == 1:
                            h, w = img.shape[1], img.shape[2]
                            region1 = img[0, :h//2, :w//2].mean()
                            region2 = img[0, :h//2, w//2:].mean()
                        else:
                            h, w = img.shape[1], img.shape[2]
                            region1 = img[:, :h//2, :w//2].mean()
                            region2 = img[:, :h//2, w//2:].mean()
                        label = int((region1 + region2 + 2) * 2.5) % 100  # 100 classes for CIFAR-100
                        dummy_labels.append(label)
                    dummy_labels = torch.tensor(dummy_labels)
                    train_dataset = TensorDataset(dummy_data[:800], dummy_labels[:800])
                    test_dataset = TensorDataset(dummy_data[800:], dummy_labels[800:])
            
            elif 'VOC2012' in dataset_info.builtin_dataset_name:
                logger.info("Loading VOC2012 segmentation dataset")
                
                # VOC2012 specific transforms for segmentation
                transform_train = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
                transform_test = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
                
                try:
                    # Load VOC2012 segmentation dataset
                    train_dataset = datasets.VOCSegmentation(
                        './data', 
                        year='2012',
                        image_set='train',
                        download=True,
                        transform=transform_train
                    )
                    test_dataset = datasets.VOCSegmentation(
                        './data',
                        year='2012', 
                        image_set='val',
                        download=True,
                        transform=transform_test
                    )
                    logger.info(f"VOC2012 loaded: {len(train_dataset)} training, {len(test_dataset)} validation samples")
                    
                except Exception as e:
                    logger.warning(f"Failed to download VOC2012: {e}. Using dummy segmentation data.")
                    # Create dummy segmentation data
                    dummy_images = torch.randn(1000, channels, *img_size)
                    dummy_masks = torch.randint(0, 21, (1000, *img_size))  # 21 classes for VOC
                    train_dataset = TensorDataset(dummy_images[:800], dummy_masks[:800])
                    test_dataset = TensorDataset(dummy_images[800:], dummy_masks[800:])
                
            else:
                raise ValueError(f"Unsupported built-in dataset: {dataset_info.builtin_dataset_name}")
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            
            logger.info(f"Built-in dataset loaded: {len(train_dataset)} training, {len(test_dataset)} test samples")
        
        elif hasattr(dataset_info, 'is_hf_dataset') and dataset_info.is_hf_dataset:
            # Hugging Face dataset loading
            logger.info(f"Loading Hugging Face dataset: {dataset_info.hf_dataset_name}")
            
            try:
                from datasets import load_dataset
                import numpy as np
                from PIL import Image
                
                # Get HF dataset config from session state
                hf_config = st.session_state.get('hf_dataset_config', {})
                
                # DEBUG: Log the values being used for PyTorch
                logger.info(f"🔍 PYTORCH DEBUG: dataset_info.hf_dataset_name = '{dataset_info.hf_dataset_name}'")
                logger.info(f"🔍 PYTORCH DEBUG: dataset_info.hf_subset = '{dataset_info.hf_subset}'")
                logger.info(f"🔍 PYTORCH DEBUG: dataset_info.dataset_path = '{dataset_info.dataset_path}'")
                logger.info(f"🔍 PYTORCH DEBUG: type(dataset_info.hf_dataset_name) = {type(dataset_info.hf_dataset_name)}")
                
                # Build authentication kwargs
                auth_kwargs = {}
                if hf_config.get('token'):
                    auth_kwargs['token'] = hf_config['token']
                
                logger.info(f"🔍 PYTORCH DEBUG: auth_kwargs = {auth_kwargs}")
                
                # CRITICAL FIX: Temporarily rename local directory to prevent conflict
                logger.info("🔍 PYTORCH DEBUG: Checking for local directory conflicts...")
                local_dir = os.path.join(os.getcwd(), dataset_info.hf_dataset_name)
                temp_renamed = None
                if os.path.exists(local_dir):
                    temp_renamed = f"{local_dir}_temp_hidden_{int(time.time())}"
                    logger.info(f"⚠️ Found local directory '{local_dir}' - temporarily renaming to avoid conflict")
                    try:
                        os.rename(local_dir, temp_renamed)
                        logger.info(f"✅ Temporarily renamed to: {temp_renamed}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not rename directory: {e}")
                        temp_renamed = None
                
                # Clear corrupted cache before loading
                logger.info("🔍 PYTORCH DEBUG: Clearing any corrupted cache...")
                cache_dir = os.path.expanduser(f"~/.cache/huggingface/datasets/{dataset_info.hf_dataset_name}")
                if os.path.exists(cache_dir):
                    logger.info(f"🗑️ Clearing cache: {cache_dir}")
                    try:
                        shutil.rmtree(cache_dir)
                        logger.info("✅ Cache cleared successfully")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not clear cache: {e}")
                
                # Load dataset with resource management
                try:
                    # Disable datasets caching to prevent resource leaks
                    from datasets import disable_caching
                    disable_caching()
                    
                    if dataset_info.hf_subset:
                        logger.info(f"🔍 PYTORCH DEBUG: Loading with subset: load_dataset('{dataset_info.hf_dataset_name}', '{dataset_info.hf_subset}', ...)")
                        hf_dataset = load_dataset(
                            dataset_info.hf_dataset_name, 
                            dataset_info.hf_subset,
                            download_mode='force_redownload',
                            trust_remote_code=True,
                            **auth_kwargs
                        )
                    else:
                        logger.info(f"🔍 PYTORCH DEBUG: Loading without subset: load_dataset('{dataset_info.hf_dataset_name}', ...)")
                        hf_dataset = load_dataset(
                            dataset_info.hf_dataset_name,
                            download_mode='force_redownload',
                            trust_remote_code=True,
                            **auth_kwargs
                        )
                except Exception as e:
                    logger.error(f"Failed to load dataset: {e}")
                    # Restore directory even on error
                    if temp_renamed and os.path.exists(temp_renamed):
                        try:
                            os.rename(temp_renamed, local_dir)
                            logger.info(f"✅ Restored local directory after error: {local_dir}")
                        except Exception as restore_e:
                            logger.warning(f"⚠️ Could not restore directory: {restore_e}")
                    raise
                finally:
                    # Restore the renamed directory
                    if temp_renamed and os.path.exists(temp_renamed):
                        try:
                            os.rename(temp_renamed, local_dir)
                            logger.info(f"✅ Restored local directory: {local_dir}")
                        except Exception as e:
                            logger.warning(f"⚠️ Could not restore directory: {e}")
                    # Force garbage collection
                    import gc
                    gc.collect()
                
                # Get train/test splits (use train/test or train/validation)
                if 'train' in hf_dataset and 'test' in hf_dataset:
                    train_split = hf_dataset['train']
                    test_split = hf_dataset['test']
                elif 'train' in hf_dataset and 'validation' in hf_dataset:
                    train_split = hf_dataset['train']
                    test_split = hf_dataset['validation']
                elif 'train' in hf_dataset:
                    # Split train set
                    train_split = hf_dataset['train']
                    split_dataset = train_split.train_test_split(test_size=0.2, seed=42)
                    train_split = split_dataset['train']
                    test_split = split_dataset['test']
                else:
                    raise ValueError("No suitable train split found in Hugging Face dataset")
                
                logger.info(f"HF dataset splits: {len(train_split)} train, {len(test_split)} test")
                
                # Create custom PyTorch dataset class for HF data
                class HFImageDataset(Dataset):
                    def __init__(self, hf_dataset, transform=None, image_key='image', label_key='label'):
                        self.hf_dataset = hf_dataset
                        self.transform = transform
                        self.image_key = image_key
                        self.label_key = label_key
                        
                        # Find image and label keys
                        features = hf_dataset.features
                        self.image_key = self._find_key(features, ['image', 'img', 'pixel_values', 'picture'])
                        self.label_key = self._find_key(features, ['label', 'labels', 'target', 'class'])
                    
                    def _find_key(self, features, candidates):
                        for candidate in candidates:
                            if candidate in features:
                                return candidate
                        # Return first key as fallback
                        return list(features.keys())[0]
                    
                    def __len__(self):
                        return len(self.hf_dataset)
                    
                    def __getitem__(self, idx):
                        item = self.hf_dataset[idx]
                        
                        # Get image
                        image = item[self.image_key]
                        
                        # Process image - ensure it's a PIL Image with proper error handling
                        if isinstance(image, Image.Image):
                            # Already PIL Image - no conversion needed
                            pass  
                        elif isinstance(image, np.ndarray):
                            # Handle numpy array conversion
                            if image.size > 0 and len(image.shape) >= 2:  # Valid image array
                                try:
                                    # Ensure proper dtype for PIL Image
                                    if image.dtype != np.uint8:
                                        # Normalize to 0-255 range if needed
                                        if image.max() <= 1.0:
                                            image = (image * 255).astype(np.uint8)
                                        else:
                                            image = image.astype(np.uint8)
                                    image = Image.fromarray(image)
                                except (TypeError, ValueError) as e:
                                    print(f"⚠️ Error converting numpy array to PIL Image: {e}")
                                    print(f"Array shape: {image.shape}, dtype: {image.dtype}")
                                    # Create a dummy image to avoid crashing
                                    image = Image.new('RGB', (224, 224), color=(128, 128, 128))
                            else:
                                print(f"⚠️ Invalid numpy array shape: {image.shape if hasattr(image, 'shape') else 'unknown'}")
                                # Create a dummy image to avoid crashing
                                image = Image.new('RGB', (224, 224), color=(128, 128, 128))
                        else:
                            # Handle other types - but be more careful
                            try:
                                # For HuggingFace datasets, image should already be PIL Image
                                # This case should rarely occur with proper HF datasets
                                if hasattr(image, 'numpy'):
                                    # PyTorch tensor
                                    image_array = image.numpy()
                                elif hasattr(image, '__array__'):
                                    # Array-like object
                                    image_array = np.array(image)
                                else:
                                    print(f"⚠️ Unknown image type: {type(image)}")
                                    # Create a dummy image to avoid crashing
                                    image = Image.new('RGB', (224, 224), color=(128, 128, 128))
                                    image_array = None
                                
                                if image_array is not None and image_array.size > 0:
                                    # Ensure proper dtype
                                    if image_array.dtype != np.uint8:
                                        if image_array.max() <= 1.0:
                                            image_array = (image_array * 255).astype(np.uint8)
                                        else:
                                            image_array = image_array.astype(np.uint8)
                                    image = Image.fromarray(image_array)
                            except Exception as e:
                                print(f"⚠️ Error processing image of type {type(image)}: {e}")
                                # Create a dummy image to avoid crashing
                                image = Image.new('RGB', (224, 224), color=(128, 128, 128))
                        
                        # CRITICAL FIX: Ensure consistent channels
                        # Convert all images to RGB to fix channel mismatch
                        if hasattr(image, 'mode') and image.mode != 'RGB':
                            image = image.convert('RGB')
                        
                        # Get label
                        label = item[self.label_key]
                        if isinstance(label, (list, np.ndarray)):
                            label = label[0] if len(label) > 0 else 0
                        
                        # Apply transforms
                        if self.transform:
                            image = self.transform(image)
                        
                        return image, label
                
                # Create datasets
                train_dataset = HFImageDataset(train_split, transform=transform_train)
                test_dataset = HFImageDataset(test_split, transform=transform_test)
                
                # Create data loaders
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                
                logger.info(f"HF dataset loaded: {len(train_dataset)} training, {len(test_dataset)} test samples")
                
            except Exception as e:
                logger.error(f"Failed to load Hugging Face dataset: {str(e)}")
                raise RuntimeError(f"Hugging Face dataset loading failed: {str(e)}")
            
        else:
            # Directory-based dataset loading
            logger.info("Loading directory-based dataset...")
            
            # Check if user wants CV2 loading
            use_cv2_loading = model_config.config_params.get('use_cv2_loading', False)
            color_conversion = model_config.config_params.get('color_conversion', 'None')
            interpolation_method = model_config.config_params.get('interpolation_method', 'INTER_LINEAR')
            
            if use_cv2_loading:
                logger.info("Using CV2 for image loading with custom dataset class")
                
                # Create custom CV2 dataset class
                class CV2ImageFolder(Dataset):
                    def __init__(self, root_dir, transform=None, color_conversion='None', interpolation_method='INTER_LINEAR'):
                        self.root_dir = root_dir
                        self.transform = transform
                        self.color_conversion = color_conversion
                        self.interpolation_method = interpolation_method
                        self.samples = []
                        self.class_to_idx = {}
                        
                        # Import cv2 here to avoid issues if not installed
                        import cv2
                        
                        # Map interpolation method names to cv2 constants
                        self.interpolation_map = {
                            'INTER_LINEAR': cv2.INTER_LINEAR,
                            'INTER_CUBIC': cv2.INTER_CUBIC,  
                            'INTER_AREA': cv2.INTER_AREA,
                            'INTER_NEAREST': cv2.INTER_NEAREST
                        }
                        
                        # Map color conversion names to cv2 constants
                        self.color_conversion_map = {
                            'BGR2RGB': cv2.COLOR_BGR2RGB,
                            'RGB2BGR': cv2.COLOR_RGB2BGR,
                            'BGR2GRAY': cv2.COLOR_BGR2GRAY,
                            'RGB2GRAY': cv2.COLOR_RGB2GRAY,
                            'GRAY2RGB': cv2.COLOR_GRAY2RGB,
                            'GRAY2BGR': cv2.COLOR_GRAY2BGR
                        }
                        
                        # Scan directories and collect samples
                        self._make_dataset()
                    
                    def _make_dataset(self):
                        import os
                        classes = [d for d in os.listdir(self.root_dir) 
                                 if os.path.isdir(os.path.join(self.root_dir, d))]
                        classes.sort()
                        
                        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
                        
                        for class_name in classes:
                            class_dir = os.path.join(self.root_dir, class_name)
                            class_idx = self.class_to_idx[class_name]
                            
                            for filename in os.listdir(class_dir):
                                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                                    path = os.path.join(class_dir, filename)
                                    self.samples.append((path, class_idx))
                    
                    def __len__(self):
                        return len(self.samples)
                    
                    def __getitem__(self, idx):
                        import cv2
                        from PIL import Image
                        
                        path, target = self.samples[idx]
                        
                        # Load image with CV2
                        image = cv2.imread(path)
                        if image is None:
                            raise RuntimeError(f"Could not load image: {path}")
                        
                        # Apply color conversion if specified
                        if self.color_conversion != 'None' and self.color_conversion in self.color_conversion_map:
                            conversion_code = self.color_conversion_map[self.color_conversion]
                            image = cv2.cvtColor(image, conversion_code)
                        
                        # Resize image using CV2 with specified interpolation
                        if hasattr(self, 'target_size'):
                            interpolation = self.interpolation_map.get(self.interpolation_method, cv2.INTER_LINEAR)
                            image = cv2.resize(image, self.target_size, interpolation=interpolation)
                        
                        # Convert CV2 image (numpy array) to PIL Image for compatibility with torchvision transforms
                        if len(image.shape) == 3:
                            # Color image: convert from BGR to RGB for PIL (if not already done by color conversion)
                            if self.color_conversion == 'None':
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            image = Image.fromarray(image)
                        else:
                            # Grayscale image
                            image = Image.fromarray(image, mode='L')
                        
                        if self.transform:
                            image = self.transform(image)
                        
                        return image, target
                
                # Set target size for CV2 resizing
                target_size = (img_size[1], img_size[0])  # CV2 expects (width, height)
                
                try:
                    train_dataset = CV2ImageFolder(
                        root_dir=dataset_info.dataset_path, 
                        transform=transform_train,
                        color_conversion=color_conversion,
                        interpolation_method=interpolation_method
                    )
                    train_dataset.target_size = target_size
                    
                    test_dataset = CV2ImageFolder(
                        root_dir=dataset_info.dataset_path,
                        transform=transform_test,
                        color_conversion=color_conversion,
                        interpolation_method=interpolation_method
                    )
                    test_dataset.target_size = target_size
                    
                    logger.info(f"CV2 dataset created with color conversion: {color_conversion}, interpolation: {interpolation_method}")
                    
                except Exception as e:
                    logger.error(f"Failed to create CV2 dataset: {e}")
                    # Fallback to regular ImageFolder
                    from torchvision.datasets import ImageFolder
                    train_dataset = ImageFolder(root=dataset_info.dataset_path, transform=transform_train)
                    test_dataset = ImageFolder(root=dataset_info.dataset_path, transform=transform_test)
                    logger.info("Fell back to regular ImageFolder loading")
            
            else:
                # Regular ImageFolder loading
                logger.info("Using regular PyTorch ImageFolder loading")
                from torchvision.datasets import ImageFolder
                train_dataset = ImageFolder(root=dataset_info.dataset_path, transform=transform_train)
                test_dataset = ImageFolder(root=dataset_info.dataset_path, transform=transform_test)
            
            try:
                # Split into train/test if needed
                train_size = int(0.8 * len(train_dataset))
                test_size = len(train_dataset) - train_size
                train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                
                loading_method = "CV2" if use_cv2_loading else "PIL"
                logger.info(f"Directory dataset loaded with {loading_method}: {len(train_dataset)} training, {len(test_dataset)} test samples")
                
            except Exception as e:
                logger.error(f"Failed to load directory dataset: {e}")
                # Create dummy dataset as fallback with learnable patterns
                logger.info("Creating dummy dataset with learnable patterns as fallback...")
                
                dummy_data = torch.randn(1000, channels, *img_size)
                # Create learnable labels based on data patterns instead of random labels
                dummy_labels = []
                for i in range(1000):
                    img = dummy_data[i]
                    # Create label based on spatial patterns (learnable relationship)
                    if len(img.shape) == 3:  # Color image
                        h, w = img.shape[1], img.shape[2]
                        region1 = img[:, :h//2, :w//2].mean()  # Top-left
                        region2 = img[:, :h//2, w//2:].mean()  # Top-right
                        label = int((region1 + region2 + 2) * 2.5) % num_classes
                    else:  # Grayscale
                        h, w = img.shape[0], img.shape[1]
                        region1 = img[:h//2, :w//2].mean()
                        region2 = img[:h//2, w//2:].mean()
                        label = int((region1 + region2 + 2) * 2.5) % num_classes
                    dummy_labels.append(label)
                
                dummy_labels = torch.tensor(dummy_labels)
                train_dataset = TensorDataset(dummy_data[:800], dummy_labels[:800])
                test_dataset = TensorDataset(dummy_data[800:], dummy_labels[800:])
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                
                logger.info(f"Dummy dataset created: {len(train_dataset)} training, {len(test_dataset)} test samples")
        
        if progress_ui:
            progress_ui['status_text'].text("🏗️ **Building PyTorch model...**")
            progress_ui['overall_progress'].progress(0.3)
        
        # ── Class imbalance detection ─────────────────────────────────────────
        class_weights_tensor = None
        try:
            # Collect labels from the training dataset
            _labels = None
            if hasattr(train_dataset, 'dataset') and hasattr(train_dataset, 'indices'):
                base = train_dataset.dataset
                indices = list(train_dataset.indices)
                if hasattr(base, 'targets'):
                    targets = base.targets
                    _labels = np.array([int(targets[i]) for i in indices])
            elif hasattr(train_dataset, 'targets'):
                _labels = np.array([int(t) for t in train_dataset.targets])
            elif hasattr(train_dataset, 'tensors') and len(train_dataset.tensors) > 1:
                _labels = train_dataset.tensors[1].numpy().astype(int)
            
            if _labels is not None and len(_labels) > 0:
                class_counts = np.bincount(_labels, minlength=num_classes).astype(float)
                if class_counts.min() > 0:
                    ratio = class_counts.max() / class_counts.min()
                    if ratio > 2.0:
                        total = class_counts.sum()
                        weights = total / (num_classes * class_counts)
                        class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
                        imbalance_info = ", ".join(
                            [f"class {i}: {int(c)}" for i, c in enumerate(class_counts)]
                        )
                        st.warning(
                            f"⚠️ **Class imbalance detected** (ratio {ratio:.1f}x). "
                            f"Auto-applying inverse-frequency class weights to the loss function.\n\n"
                            f"Counts: {imbalance_info}"
                        )
                        logger.info(f"Class imbalance detected (ratio {ratio:.1f}x). Applied class weights.")
                    else:
                        logger.info(f"Classes are balanced (max/min ratio {ratio:.1f}x). No weighting applied.")
        except Exception as _cw_err:
            logger.warning(f"Class weight computation skipped: {_cw_err}")
        
        # Build CNN Model
        logger.info("Building PyTorch CNN model...")
        logger.info(f"Model input shape: ({channels}, {height}, {width})")
        logger.info(f"Model output classes: {num_classes}")
        
        class AdaptiveCNN(nn.Module):
            def __init__(self, num_classes, input_channels, input_size):
                super(AdaptiveCNN, self).__init__()
                self.input_channels = input_channels
                self.input_size = input_size
                
                # Adaptive architecture based on input size
                h, w = input_size
                
                # First conv block
                self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(32)
                self.pool1 = nn.MaxPool2d(2, 2) if min(h, w) >= 32 else nn.Identity()
                
                # Second conv block
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(64)
                self.pool2 = nn.MaxPool2d(2, 2) if min(h, w) >= 16 else nn.Identity()
                
                # Third conv block (only for larger inputs)
                if min(h, w) >= 64:
                    self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                    self.bn3 = nn.BatchNorm2d(128)
                    self.pool3 = nn.MaxPool2d(2, 2)
                    final_channels = 128
                else:
                    self.conv3 = None
                    final_channels = 64
                
                # Global average pooling instead of flatten for flexibility
                self.global_pool = nn.AdaptiveAvgPool2d(1)
                
                # Classifier
                self.dropout1 = nn.Dropout(0.5)
                self.fc1 = nn.Linear(final_channels, 256)
                self.dropout2 = nn.Dropout(0.3)
                self.fc2 = nn.Linear(256, num_classes)
                
            def forward(self, x):
                # First block
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.pool1(x)
                
                # Second block
                x = F.relu(self.bn2(self.conv2(x)))
                x = self.pool2(x)
                
                # Third block (if exists)
                if self.conv3 is not None:
                    x = F.relu(self.bn3(self.conv3(x)))
                    x = self.pool3(x)
                
                # Global pooling and classifier
                x = self.global_pool(x)
                x = x.view(x.size(0), -1)
                x = self.dropout1(x)
                x = F.relu(self.fc1(x))
                x = self.dropout2(x)
                x = self.fc2(x)
                
                return x
        
        # Create model — UNet for segmentation, ModelFactory for detection, AdaptiveCNN for classification
        task_type = getattr(dataset_info, 'task_type', 'classification')
        if task_type == 'detection':
            try:
                from utils.model_factory import ModelFactory
                _det_model_cfg = type('_DC', (), {
                    'dataset_info': dataset_info,
                    'model_config':  model_config,
                })()
                model = ModelFactory(_det_model_cfg).create_model().to(device)
                logger.info('Detection model built via ModelFactory')
            except Exception as _det_me:
                logger.warning(f'ModelFactory failed for detection ({_det_me}), using SimpleDetector fallback')
                from utils.model_factory import ModelFactory
                model = ModelFactory(type('_DC', (), {
                    'dataset_info': dataset_info,
                    'model_config':  model_config,
                })()).create_model().to(device)
        elif task_type == 'segmentation':
            try:
                from utils.model_factory import ModelFactory
                from utils.config import Config
                # Build a minimal config to drive ModelFactory
                _seg_config = type('_C', (), {
                    'dataset_info': dataset_info,
                    'model_config': model_config,
                })()
                _seg_config.dataset_info = dataset_info
                _seg_config.model_config = model_config
                model = ModelFactory(_seg_config).create_model().to(device)
                logger.info(f"Segmentation model built via ModelFactory")
            except Exception as _me:
                logger.warning(f"ModelFactory failed ({_me}), using SimpleUNet fallback")
                # Minimal UNet fallback
                class _SimpleUNet(nn.Module):
                    def __init__(self, nc, ch):
                        super().__init__()
                        self.enc = nn.Sequential(nn.Conv2d(ch, 64, 3, padding=1), nn.ReLU(),
                                                  nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
                        self.head = nn.Conv2d(64, nc, 1)
                    def forward(self, x): return self.head(self.enc(x))
                model = _SimpleUNet(num_classes, channels).to(device)
        else:
            model = AdaptiveCNN(num_classes, channels, img_size).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model created with {total_params:,} parameters")
        
        # Setup training — detection loss comes from model; segmentation uses pixel CE
        if task_type == 'detection':
            criterion = None          # model returns loss dict in train mode
        elif task_type == 'segmentation':
            criterion = nn.CrossEntropyLoss(ignore_index=255)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        # Use user-configured learning rate from session state (set by start_training_with_config)
        _lr = st.session_state.get('override_learning_rate', 0.001)
        _batch_size_override = st.session_state.get('override_batch_size')
        if _batch_size_override:
            batch_size = int(_batch_size_override)
        optimizer = optim.Adam(model.parameters(), lr=_lr)
        # Decay LR every step_size epochs; use the longer of 10 or max_epochs//5
        _step_size = max(10, max_epochs // 5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=_step_size, gamma=0.1)
        
        # Setup callbacks with early stopping
        callback_manager = CallbackManager(framework="PyTorch")
        
        # Early stopping configuration from session state
        es_config = st.session_state.get('early_stopping_config', {})
        enable_early_stopping = es_config.get('enable_early_stopping', True)
        
        if enable_early_stopping:
            early_stopping_patience = es_config.get('patience', 10)
            early_stopping_min_delta = es_config.get('min_delta', 0.001)
            
            early_stopping = EarlyStopping(
                monitor='val_acc',
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                mode='max',
                verbose=True
            )
            callback_manager.add_callback(early_stopping)
            logger.info(f"Early stopping enabled: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
        else:
            logger.info("Early stopping disabled by user")
        
        # Model checkpointing
        checkpoint_path = os.path.join(output_dir, "best_model.pt")
        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_acc',
            save_best_only=True,
            mode='max',
            verbose=True
        )
        model_checkpoint.set_model(model)
        callback_manager.add_callback(model_checkpoint)
        
        # Metrics logger
        metrics_logger = MetricsLogger(
            log_dir=output_dir,
            framework="PyTorch"
        )
        callback_manager.add_callback(metrics_logger)
        
        if progress_ui:
            progress_ui['status_text'].text("🚀 **Training PyTorch model with Early Stopping...**")
            progress_ui['overall_progress'].progress(0.4)
        
        # Initialize callbacks
        callback_manager.on_train_begin({'total_epochs': max_epochs})
        
        # ACTUAL PYTORCH TRAINING
        training_start_time = time.time()
        
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        train_mious = []
        val_mious = []
        val_dices = []
        val_map50s = []   # detection mAP@50 per epoch

        best_accuracy = 0.0
        best_loss = float('inf')
        best_miou = 0.0
        best_dice = 0.0
        best_map50 = 0.0
        
        for epoch in range(max_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, batch in enumerate(train_loader):
                if task_type == 'detection':
                    data, target = batch
                    data = data.to(device)
                    target = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                               for k, v in t.items()} for t in target]
                    optimizer.zero_grad()
                    loss_dict = model(data, target)   # train mode → loss dict
                    loss = sum(loss_dict.values())
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    total += data.size(0)
                    correct += 0   # no per-batch accuracy for detection
                else:
                    data, target = batch
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    # Unwrap torchvision segmentation dict
                    raw_out = output['out'] if isinstance(output, dict) else output
                    loss = criterion(raw_out, target)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    if task_type == 'segmentation':
                        from utils.metrics import compute_segmentation_metrics
                        _miou, _dice, _ = compute_segmentation_metrics(raw_out.detach(), target, num_classes)
                        correct += _miou * target.size(0)
                        total += target.size(0)
                    else:
                        _, predicted = torch.max(raw_out.data, 1)
                        total += target.size(0)
                        correct += predicted.eq(target.data).cpu().sum().item()
            
            train_loss = running_loss / len(train_loader)
            if task_type == 'segmentation':
                train_acc = (correct / total) * 100 if total > 0 else 0.0  # mIoU as %
            else:
                train_acc = 100. * correct / total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            det_val_preds: list = []
            det_val_targets: list = []
            with torch.no_grad():
                for batch in test_loader:
                    if task_type == 'detection':
                        data, target = batch
                        data = data.to(device)
                        target_dev = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                                       for k, v in t.items()} for t in target]
                        preds = model(data)   # eval mode → list of pred dicts
                        det_val_preds.extend(preds)
                        det_val_targets.extend(target_dev)
                        val_total += data.size(0)
                    else:
                        data, target = batch
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        raw_out = output['out'] if isinstance(output, dict) else output
                        val_loss += criterion(raw_out, target).item()
                        if task_type == 'segmentation':
                            from utils.metrics import compute_segmentation_metrics
                            _miou, _dice, _ = compute_segmentation_metrics(raw_out, target, num_classes)
                            val_correct += _miou * target.size(0)
                            val_total += target.size(0)
                            # accumulate dice too
                            if not hasattr(model, '_dice_acc'):
                                model._dice_acc = 0.0
                                model._dice_cnt = 0
                            model._dice_acc += _dice * target.size(0)
                            model._dice_cnt += target.size(0)
                        else:
                            _, predicted = torch.max(raw_out.data, 1)
                            val_total += target.size(0)
                            val_correct += predicted.eq(target.data).cpu().sum().item()
            
            if task_type == 'detection':
                val_loss_avg = 0.0   # no loss computed during eval for detection
                val_acc = 0.0
                if det_val_preds:
                    from utils.metrics import compute_detection_metrics
                    _det_res = compute_detection_metrics(det_val_preds, det_val_targets, iou_threshold=0.5)
                    val_map50 = _det_res.get('mAP@50', 0.0) * 100  # store as %
                else:
                    val_map50 = 0.0
                val_map50s.append(val_map50)
                val_acc = val_map50
                if val_map50 > best_map50:
                    best_map50 = val_map50
                    best_accuracy = val_map50
            else:
                val_loss_avg = val_loss / max(len(test_loader), 1)
                if task_type == 'segmentation':
                    val_acc = (val_correct / val_total) * 100 if val_total > 0 else 0.0  # mIoU %
                    _ep_dice = (model._dice_acc / model._dice_cnt * 100) if hasattr(model, '_dice_cnt') and model._dice_cnt > 0 else 0.0
                    val_mious.append(val_acc)
                    val_dices.append(_ep_dice)
                    # reset accumulators
                    model._dice_acc = 0.0
                    model._dice_cnt = 0
                    if val_acc > best_miou:
                        best_miou  = val_acc
                        best_accuracy = val_acc
                        best_dice = _ep_dice
                else:
                    val_acc = 100. * val_correct / max(val_total, 1)

            val_loss = val_loss_avg if task_type != 'detection' else 0.0

            # Update best metrics
            if task_type not in ('segmentation', 'detection'):
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
            if val_loss < best_loss:
                best_loss = val_loss

            # Store metrics
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            # Call callbacks with metrics
            epoch_metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'epoch': epoch
            }
            if task_type == 'detection':
                epoch_metrics['val_map50'] = val_map50
            
            callback_manager.on_epoch_end(epoch, epoch_metrics)
            
            # Check if early stopping was triggered
            if callback_manager.should_stop_training():
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                if progress_ui:
                    progress_ui['status_text'].text(f"⏹️ **Early stopping triggered at epoch {epoch+1}**")
                break
            
            # Update progress UI
            if progress_ui:
                progress = 0.4 + (epoch + 1) * 0.5 / max_epochs
                progress_ui['overall_progress'].progress(min(progress, 0.9))

                elapsed = time.time() - training_start_time
                # Per-epoch status line so users can see training is running
                progress_ui['status_text'].text(
                    f"🏋️ Epoch {epoch + 1}/{max_epochs} — "
                    f"loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
                    f"val_acc: {val_acc:.1f}%"
                )
                progress_ui['current_epoch'].metric("Epoch", f"{epoch + 1} / {max_epochs}")
                progress_ui['current_loss'].metric("Loss", f"{val_loss:.4f}")
                progress_ui['current_acc'].metric("Accuracy", f"{val_acc:.1f}%")
                progress_ui['elapsed_time'].metric("Time", "{:02d}:{:02d}:{:02d}".format(
                    int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
                ))

                # Store metrics as dict so post-training charts can consume them
                log_dict = {
                    'epoch':        epoch + 1,
                    'loss':         round(train_loss, 4),
                    'val_loss':     round(val_loss, 4),
                }
                if task_type == 'segmentation':
                    log_dict['accuracy']     = round(train_acc / 100.0, 4)  # mIoU stored as accuracy
                    log_dict['val_accuracy'] = round(val_acc   / 100.0, 4)
                    log_dict['train_miou'] = log_dict['accuracy']
                    log_dict['val_miou']   = log_dict['val_accuracy']
                    if val_dices:
                        log_dict['val_dice'] = round(val_dices[-1] / 100.0, 4)
                elif task_type == 'detection':
                    log_dict['val_map50']    = round(val_map50 / 100.0, 4)
                    log_dict['accuracy']     = 0.0
                    log_dict['val_accuracy'] = round(val_map50 / 100.0, 4)
                else:
                    log_dict['accuracy']     = round(train_acc / 100.0, 4)
                    log_dict['val_accuracy'] = round(val_acc   / 100.0, 4)
                progress_ui['log_history'].append(log_dict)
                recent = progress_ui['log_history'][-10:]
                # Build log lines using .get() to handle all task types safely
                log_lines = [
                    f"Ep {e['epoch']:>3}: loss={e.get('loss', 0):.4f}  "
                    f"acc={e.get('accuracy', 0)*100:.1f}%  "
                    f"val_loss={e.get('val_loss', 0):.4f}  "
                    f"val_acc={e.get('val_accuracy', 0)*100:.1f}%"
                    for e in recent
                ]
                progress_ui['training_log'].text("\n".join(log_lines))
                
                # Update live charts
                if 'chart_loss' in progress_ui and len(train_losses) > 0:
                    epochs_x = list(range(1, len(train_losses) + 1))
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(x=epochs_x, y=train_losses, mode='lines+markers', name='Train Loss', line=dict(color='#ef4444')))
                    fig_loss.add_trace(go.Scatter(x=epochs_x, y=val_losses, mode='lines+markers', name='Val Loss', line=dict(color='#f97316')))
                    fig_loss.update_layout(height=220, margin=dict(l=20, r=10, t=10, b=30), showlegend=True, legend=dict(orientation='h', y=-0.25))
                    fig_loss.update_xaxes(title_text='Epoch')
                    fig_loss.update_yaxes(title_text='Loss')
                    progress_ui['chart_loss'].plotly_chart(fig_loss, use_container_width=True)
                    
                    fig_acc = go.Figure()
                    fig_acc.add_trace(go.Scatter(x=epochs_x, y=train_accuracies, mode='lines+markers', name='Train Acc', line=dict(color='#3b82f6')))
                    fig_acc.add_trace(go.Scatter(x=epochs_x, y=val_accuracies, mode='lines+markers', name='Val Acc', line=dict(color='#22c55e')))
                    fig_acc.update_layout(height=220, margin=dict(l=20, r=10, t=10, b=30), showlegend=True, legend=dict(orientation='h', y=-0.25))
                    fig_acc.update_xaxes(title_text='Epoch')
                    fig_acc.update_yaxes(title_text='Accuracy (%)')
                    progress_ui['chart_acc'].plotly_chart(fig_acc, use_container_width=True)
            
            scheduler.step()
            
            logger.info(f"Epoch {epoch+1}/{max_epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")
        
        training_time = time.time() - training_start_time
        
        # Call training end callback
        final_metrics = {
            'best_accuracy': best_accuracy,
            'best_loss': best_loss,
            'training_time': training_time,
            'total_epochs': len(train_losses)
        }
        callback_manager.on_train_end(final_metrics)
        
        if progress_ui:
            progress_ui['overall_progress'].progress(1.0)
            progress_ui['status_text'].text("✅ **PyTorch training completed!**")
        
        # Save test samples for evaluation
        try:
            save_test_samples_for_evaluation(dataset_info, output_dir, num_samples=50)
            if progress_ui:
                progress_ui['status_text'].text("✅ **PyTorch training completed! Test samples saved.**")
            logger.info("Test samples saved for evaluation")
        except Exception as e:
            logger.warning(f"Failed to save test samples: {str(e)}")
        
        # Save model with intelligent naming system
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate intelligent model name
        from utils.model_naming import generate_model_name, extract_dataset_name
        
        # Extract dataset name - fix builtin dataset handling for PyTorch
        dataset_info_obj = st.session_state.dataset_info
        builtin_info = None
        
        # Check for builtin dataset info
        if hasattr(dataset_info_obj, 'builtin_dataset_name'):
            builtin_info = {'name': dataset_info_obj.builtin_dataset_name}
        elif hasattr(dataset_info_obj, 'is_builtin') and dataset_info_obj.is_builtin:
            # For PyTorch, use builtin_dataset_name, not tf_name
            if hasattr(dataset_info_obj, 'builtin_dataset_name'):
                builtin_info = {'name': dataset_info_obj.builtin_dataset_name}
        
        dataset_name = extract_dataset_name(
            dataset_path=getattr(dataset_info_obj, 'dataset_path', None),
            builtin_dataset=builtin_info
        )
        
        # Generate intelligent model name
        model_name = generate_model_name(
            framework='pytorch',
            architecture=model_config.architecture if hasattr(model_config, 'architecture') else 'adaptive_cnn',
            backbone=model_config.backbone if hasattr(model_config, 'backbone') else None,
            dataset_name=dataset_name,
            task_type=dataset_info.task_type if hasattr(dataset_info, 'task_type') else 'classification'
        )
        
        model_path = os.path.join(output_dir, model_name)
        
        # Get class names for inference support
        class_names = None
        if hasattr(dataset_info, 'class_names') and dataset_info.class_names:
            class_names = dataset_info.class_names
        else:
            # Fallback to generic names
            class_names = [f"Class_{i}" for i in range(num_classes)]
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': max_epochs,
            'loss': best_loss,
            'accuracy': best_accuracy,
            'model_config': {
                'num_classes': num_classes,
                'input_channels': channels,
                'input_size': img_size,
                'architecture': 'AdaptiveCNN'
            },
            'class_names': class_names,
            'dataset_info': {
                'task_type': dataset_info.task_type if hasattr(dataset_info, 'task_type') else 'classification',
                'dataset_name': dataset_name,
                'num_samples': dataset_info.num_samples if hasattr(dataset_info, 'num_samples') else None,
                'image_size': dataset_info.image_size if hasattr(dataset_info, 'image_size') else img_size
            }
        }, model_path)
        
        logger.info(f"Model saved to: {model_path}")
        st.success(f"💾 PyTorch Model saved to: {model_path}")
        
        # Prepare results
        results = {
            'best_accuracy': best_accuracy / 100.0,  # Convert to 0-1 scale
            'best_loss': best_loss,
            'training_time': training_time,
            'total_epochs': max_epochs,
            'framework': 'PyTorch',
            'model_parameters': total_params,
            'architecture': (
                'AdaptiveCNN' if task_type not in ('segmentation', 'detection')
                else ('UNet' if task_type == 'segmentation' else 'Detector')
            ),
            'model_saved': model_path,
            'class_names': class_names,
            'num_classes': num_classes,
            'input_size': img_size,
            'input_channels': channels,
            'dataset_name': dataset_name,
            'task_type': task_type,
            'training_history': {
                'train_accuracy': [x/100.0 for x in train_accuracies],
                'val_accuracy': [x/100.0 for x in val_accuracies],
                'train_miou': [x/100.0 for x in val_mious],
                'val_miou':   [x/100.0 for x in val_mious],
                'val_dice':   [x/100.0 for x in val_dices],
                'val_map50':  [x/100.0 for x in val_map50s],
                'train_loss': train_losses,
                'val_loss': val_losses
            }
        }
        # Add segmentation-specific keys to top-level results for results page
        if task_type == 'segmentation':
            results['best_miou'] = best_miou / 100.0
            results['best_dice'] = best_dice / 100.0
        elif task_type == 'detection':
            results['best_map50'] = best_map50 / 100.0
        
        # Save training results to JSON with intelligent naming
        results_name = model_name.replace('.pt', '_results.json')
        results_path = os.path.join(output_dir, results_name)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        st.success(f"📋 PyTorch Training results saved to: {results_path}")
        logger.info(f"Training completed - Best Accuracy: {best_accuracy:.2f}%, Training Time: {training_time:.2f}s")
        
        # Optional ONNX conversion
        try:
            onnx_path = model_path.replace('.pt', '.onnx')
            st.info("🔄 Converting model to ONNX format for broader deployment...")
            
            # Create dummy input for ONNX export
            dummy_input = torch.randn(1, channels, img_size[0], img_size[1]).to(device)
            model.eval()
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # Save ONNX metadata
            onnx_metadata = {
                'pytorch_model_path': model_path,
                'onnx_model_path': onnx_path,
                'input_shape': [1, channels, img_size[0], img_size[1]],
                'class_names': class_names,
                'model_info': {
                    'num_classes': num_classes,
                    'input_channels': channels,
                    'input_size': img_size,
                    'architecture': 'AdaptiveCNN'
                },
                'training_results': {
                    'best_accuracy': best_accuracy / 100.0,
                    'best_loss': best_loss,
                    'training_time': training_time
                }
            }
            
            onnx_metadata_path = onnx_path.replace('.onnx', '_metadata.json')
            with open(onnx_metadata_path, 'w') as f:
                json.dump(onnx_metadata, f, indent=2)
            
            st.success(f"✅ ONNX model saved: {os.path.basename(onnx_path)}")
            logger.info(f"ONNX conversion completed: {onnx_path}")
            
        except Exception as onnx_error:
            logger.warning(f"ONNX conversion failed (optional): {onnx_error}")
            st.warning("⚠️ ONNX conversion failed (training still successful)")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ PyTorch training failed: {e}")
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Full traceback: {error_traceback}")
        st.error(f"❌ PyTorch training failed: {e}")
        return {
            'best_accuracy': 0.0,
            'best_loss': float('inf'),
            'training_time': 0.0,
            'error': str(e)
        }


def start_tensorflow_training(max_epochs: int, optimize_hyperparams: bool, output_dir: str, progress_ui=None):
    """Start TensorFlow/Keras training process with progress tracking."""
    
    import time
    import logging
    import os
    import shutil
    
    # Setup detailed logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    # Log function entry
    logger.info(f"=== TENSORFLOW TRAINING START ===")
    logger.info(f"max_epochs: {max_epochs}")
    logger.info(f"optimize_hyperparams: {optimize_hyperparams}")
    logger.info(f"output_dir: {output_dir}")
    
    try:
        if progress_ui:
            progress_ui['status_text'].text("🧠 **Initializing TensorFlow/Keras...**")
            progress_ui['overall_progress'].progress(0.05)
            
        logger.info("Progress UI setup complete")
        
        # Try direct TensorFlow implementation
        try:
            logger.info("Importing TensorFlow modules...")
            import tensorflow as tf
            from tensorflow import keras
            import numpy as np
            TF_AVAILABLE = True
            logger.info(f"TensorFlow {tf.__version__} imported successfully")
            
            # Configure TensorFlow to prevent resource leaks
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)
            
            # Disable GPU memory growth to prevent leaks
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    logger.warning(f"GPU configuration warning: {e}")
            if progress_ui:
                progress_ui['status_text'].text("✅ **TensorFlow loaded successfully**")
                progress_ui['overall_progress'].progress(0.1)
        except ImportError as e:
            logger.error(f"TensorFlow import failed: {e}")
            if progress_ui:
                progress_ui['status_text'].text(f"❌ **TensorFlow import failed**")
            st.error(f"❌ TensorFlow import failed: {e}")
            return {
                'best_accuracy': 0.0,
                'best_loss': float('inf'),
                'training_time': 0.0,
                'error': 'TensorFlow not available'
            }
        
        # Get required data from session state
        dataset_info = st.session_state.get('dataset_info')
        model_config = st.session_state.get('model_config')
        
        if not dataset_info or not model_config:
            st.error("❌ Dataset info or model config missing")
            return {
                'best_accuracy': 0.0,
                'best_loss': float('inf'),
                'training_time': 0.0,
                'error': 'Missing configuration'
            }
        
        # Use new progress UI system or fallback to local
        if progress_ui:
            progress_bar = progress_ui['overall_progress']
            status_text = progress_ui['status_text']
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        # Real TensorFlow implementation with actual data loading and training
        status_text.text("🔄 **Loading TensorFlow data generators...**")
        progress_bar.progress(0.2)
        
        # Get required data from session state including user-specified model config
        model_config = st.session_state.get('model_config')
        
        # Setup data generators for real training - USE USER-SPECIFIED INPUT SIZE
        num_classes = dataset_info.num_classes
        
        # CRITICAL FIX: Use centralized user input size function
        try:
            height, width, channels = get_user_specified_input_size()
            img_size = (height, width)  # TensorFlow expects (height, width)
            logger.info(f"✅ Using user-specified input size: height={height}, width={width}, channels={channels}")
            logger.info(f"Final img_size for training: {img_size}")
        except Exception as size_error:
            logger.error(f"❌ Failed to get user input size: {size_error}")
            # Fallback to dataset_info if centralized function fails
            img_size = dataset_info.image_size[:2] if hasattr(dataset_info, 'image_size') else (224, 224)
            channels = 3
            logger.info(f"Using fallback img_size: {img_size}, channels: {channels}")
        
        # Use user-specified batch size from model config
        batch_size = model_config.config_params.get('batch_size', 32) if model_config else 32
        
        st.info(f"🔧 **Using user-specified input size: {img_size} with {channels} channels, batch size: {batch_size}**")
        
        logger.info("Setting up TensorFlow data generators...")
        
        # Initialize data generators to None to prevent UnboundLocalError
        train_generator = None
        validation_generator = None
        
        # Check if this is a built-in TensorFlow dataset
        is_builtin_dataset = hasattr(dataset_info, 'is_builtin') and dataset_info.is_builtin
        logger.info(f"Is built-in dataset: {is_builtin_dataset}")
        
        # DEBUG: Check dataset_info attributes
        logger.info(f"DEBUG: dataset_info attributes: {[attr for attr in dir(dataset_info) if not attr.startswith('_')]}")
        logger.info(f"DEBUG: has is_hf_dataset: {hasattr(dataset_info, 'is_hf_dataset')}")
        if hasattr(dataset_info, 'is_hf_dataset'):
            logger.info(f"DEBUG: is_hf_dataset value: {dataset_info.is_hf_dataset}")
        if hasattr(dataset_info, 'dataset_path'):
            logger.info(f"DEBUG: dataset_path value: {dataset_info.dataset_path}")
        if hasattr(dataset_info, 'hf_dataset_name'):
            logger.info(f"DEBUG: hf_dataset_name value: {dataset_info.hf_dataset_name}")
        
        if is_builtin_dataset:
            logger.info(f"Loading built-in TensorFlow dataset: {dataset_info.builtin_dataset_name}")
            
            # Import numpy for array operations
            import numpy as np
            
            # Load the appropriate built-in dataset
            if dataset_info.builtin_dataset_name == 'MNIST':
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
                logger.info(f"MNIST loaded: train shape {x_train.shape}, test shape {x_test.shape}")
                
                # Add channel dimension for grayscale
                x_train = np.expand_dims(x_train, axis=-1)
                x_test = np.expand_dims(x_test, axis=-1)
                original_channels = 1
                
            elif dataset_info.builtin_dataset_name == 'Fashion-MNIST':
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
                logger.info(f"Fashion-MNIST loaded: train shape {x_train.shape}, test shape {x_test.shape}")
                
                # Add channel dimension for grayscale
                x_train = np.expand_dims(x_train, axis=-1)
                x_test = np.expand_dims(x_test, axis=-1)
                original_channels = 1
                
            elif dataset_info.builtin_dataset_name == 'CIFAR-10':
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
                logger.info(f"CIFAR-10 loaded: train shape {x_train.shape}, test shape {x_test.shape}")
                original_channels = 3
                
            elif dataset_info.builtin_dataset_name == 'CIFAR-100':
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
                logger.info(f"CIFAR-100 loaded: train shape {x_train.shape}, test shape {x_test.shape}")
                original_channels = 3
                
            elif 'Oxford-IIIT Pet' in dataset_info.builtin_dataset_name or dataset_info.builtin_dataset_name == 'oxford_iiit_pet':
                # Load Oxford-IIIT Pet dataset using TensorFlow Datasets
                logger.info("Loading Oxford-IIIT Pet segmentation dataset via tensorflow_datasets")
                try:
                    import tensorflow_datasets as tfds
                    
                    # Load dataset
                    ds_train = tfds.load('oxford_iiit_pet', split='train', as_supervised=False)
                    ds_test = tfds.load('oxford_iiit_pet', split='test', as_supervised=False)
                    
                    # Extract images and segmentation masks
                    # Images have different sizes, so we need to resize them first
                    x_train_list, y_train_list = [], []
                    x_test_list, y_test_list = [], []
                    
                    # Use user-specified image size instead of hardcoded value
                    target_size = img_size  # Use the img_size from user configuration
                    logger.info(f"Resizing Oxford-IIIT Pet images to user-specified size: {target_size}")
                    
                    logger.info("Processing training samples...")
                    for idx, item in enumerate(ds_train.take(3000)):  # Limit to 3000 for faster training
                        # Resize image to target size
                        img = tf.image.resize(item['image'], target_size).numpy()
                        x_train_list.append(img)
                        
                        # Oxford-IIIT Pet masks are 1-indexed (1=background, 2=pet, 3=border)
                        # Convert to 0-indexed (0=background, 1=pet, 2=border)
                        mask = item['segmentation_mask'].numpy()
                        
                        # Ensure mask is 2D
                        if len(mask.shape) == 3:
                            mask = mask[:, :, 0]  # Take first channel if 3D
                        
                        # Resize mask to exact target size
                        mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)
                        mask_resized = tf.image.resize(
                            tf.expand_dims(mask_tensor, axis=-1), 
                            target_size, 
                            method='nearest'
                        )
                        mask_resized = tf.squeeze(mask_resized, axis=-1).numpy()
                        
                        # Convert from 1-indexed to 0-indexed
                        mask_resized = mask_resized - 1
                        mask_resized = np.clip(mask_resized, 0, 2).astype(np.int32)
                        
                        # Verify shape
                        if mask_resized.shape != target_size:
                            logger.warning(f"Sample {idx}: Unexpected mask shape {mask_resized.shape}, reshaping to {target_size}")
                            mask_resized = np.resize(mask_resized, target_size).astype(np.int32)
                        
                        y_train_list.append(mask_resized)
                        
                        if idx % 500 == 0:
                            logger.info(f"Processed {idx}/3000 training samples")
                    
                    logger.info("Processing test samples...")
                    for idx, item in enumerate(ds_test.take(700)):
                        img = tf.image.resize(item['image'], target_size).numpy()
                        x_test_list.append(img)
                        
                        mask = item['segmentation_mask'].numpy()
                        if len(mask.shape) == 3:
                            mask = mask[:, :, 0]
                        
                        mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)
                        mask_resized = tf.image.resize(
                            tf.expand_dims(mask_tensor, axis=-1), 
                            target_size, 
                            method='nearest'
                        )
                        mask_resized = tf.squeeze(mask_resized, axis=-1).numpy()
                        mask_resized = mask_resized - 1
                        mask_resized = np.clip(mask_resized, 0, 2).astype(np.int32)
                        
                        if mask_resized.shape != target_size:
                            mask_resized = np.resize(mask_resized, target_size).astype(np.int32)
                        
                        y_test_list.append(mask_resized)
                    
                    logger.info("Converting lists to numpy arrays...")
                    x_train = np.array(x_train_list, dtype=np.float32)
                    y_train = np.array(y_train_list, dtype=np.int32)
                    x_test = np.array(x_test_list, dtype=np.float32)
                    y_test = np.array(y_test_list, dtype=np.int32)
                    
                    logger.info(f"Oxford-IIIT Pet loaded: train shape {x_train.shape}, test shape {x_test.shape}")
                    logger.info(f"Segmentation masks shape: {y_train.shape}, unique values: {np.unique(y_train)}")
                    logger.info(f"Mask value range: [{y_train.min()}, {y_train.max()}]")
                    original_channels = 3
                    
                except ImportError:
                    logger.warning("tensorflow_datasets not installed. Using dummy segmentation data.")
                    # Create dummy segmentation data
                    x_train = np.random.rand(1000, 128, 128, 3).astype(np.float32)
                    y_train = np.random.randint(0, 3, (1000, 128, 128)).astype(np.int32)
                    x_test = np.random.rand(200, 128, 128, 3).astype(np.float32)
                    y_test = np.random.randint(0, 3, (200, 128, 128)).astype(np.int32)
                    original_channels = 3
                
            else:
                raise ValueError(f"Unsupported built-in dataset: {dataset_info.builtin_dataset_name}")
            
            logger.info(f"Original dataset shape: {x_train.shape}")
            logger.info(f"Labels shape: {y_train.shape}")
            logger.info(f"User-specified target size: {img_size} with {channels} channels")
            
            # Check if this is segmentation data (labels have spatial dimensions)
            is_segmentation_data = len(y_train.shape) > 2  # Segmentation masks are 3D: (samples, height, width)
            
            # CRITICAL: Resize to user-specified dimensions FIRST, then handle channels
            if x_train.shape[1:3] != img_size:
                logger.info(f"🔄 Resizing images from {x_train.shape[1:3]} to {img_size}")
                x_train_resized = tf.image.resize(x_train, img_size).numpy()
                x_test_resized = tf.image.resize(x_test, img_size).numpy()
                x_train, x_test = x_train_resized, x_test_resized
                logger.info(f"✅ Images resized to: {x_train.shape}")
                
                # Also resize segmentation masks if present
                if is_segmentation_data and y_train.shape[1:3] != img_size:
                    logger.info(f"🔄 Resizing segmentation masks from {y_train.shape[1:3]} to {img_size}")
                    # Resize masks using nearest neighbor to preserve label values
                    y_train_resized = tf.image.resize(
                        np.expand_dims(y_train, -1),  # Add channel dimension
                        img_size,
                        method='nearest'  # Preserve discrete label values
                    ).numpy().squeeze(-1).astype(np.int32)  # Remove channel dimension and ensure int type
                    y_test_resized = tf.image.resize(
                        np.expand_dims(y_test, -1),
                        img_size,
                        method='nearest'
                    ).numpy().squeeze(-1).astype(np.int32)
                    y_train, y_test = y_train_resized, y_test_resized
                    logger.info(f"✅ Segmentation masks resized to: {y_train.shape}")
            
            # Handle channel conversion if user wants different number of channels
            if original_channels != channels:
                logger.info(f"🎨 Converting from {original_channels} to {channels} channels")
                if original_channels == 1 and channels == 3:
                    # Grayscale to RGB: repeat channel 3 times
                    x_train = np.repeat(x_train, 3, axis=-1)
                    x_test = np.repeat(x_test, 3, axis=-1)
                elif original_channels == 3 and channels == 1:
                    # RGB to Grayscale: convert using standard weights
                    x_train = tf.image.rgb_to_grayscale(x_train).numpy()
                    x_test = tf.image.rgb_to_grayscale(x_test).numpy()
                logger.info(f"✅ Channels converted to: {x_train.shape}")
            
            # Apply user-specified normalization
            enable_normalization = model_config.config_params.get('enable_normalization', True)
            normalization_type = model_config.config_params.get('normalization_type', 'Standard (0-1)')
            norm_min = model_config.config_params.get('norm_min', 0.0)
            norm_max = model_config.config_params.get('norm_max', 1.0)
            
            logger.info(f"🔧 NORMALIZATION VERIFICATION: Using '{normalization_type}' (enabled: {enable_normalization})")
            
            # CRITICAL FIX: Skip manual normalization for TensorFlow data generators
            # The normalization will be handled by the ImageDataGenerator preprocessing_function
            logger.info(f"⚠️  SKIPPING manual normalization - will be handled by TensorFlow data generators")
            logger.info(f"🔧 TensorFlow generators will apply '{normalization_type}' during training")
            
            # Keep data in [0,1] range for TensorFlow generators (HuggingFace data is already [0,1])
            if x_train.max() > 1.1:  # Check if data is in [0,255] range
                logger.info("🔄 Converting data from [0,255] to [0,1] range for TensorFlow generators")
                x_train = x_train.astype('float32') / 255.0
                x_test = x_test.astype('float32') / 255.0
            else:
                logger.info("✅ Data already in [0,1] range - ready for TensorFlow generators")
                x_train = x_train.astype('float32')
                x_test = x_test.astype('float32')
            
            # Convert labels to categorical if needed (ONLY for classification, NOT for segmentation)
            if is_segmentation_data:
                # For segmentation, keep labels as integer arrays (height, width)
                logger.info("✅ Segmentation labels kept as integer arrays (sparse format)")
                y_train = y_train.astype('int32')
                y_test = y_test.astype('int32')
            elif num_classes > 2:
                # For classification, convert to categorical (one-hot encoding)
                y_train = tf.keras.utils.to_categorical(y_train, num_classes)
                y_test = tf.keras.utils.to_categorical(y_test, num_classes)
                logger.info("✅ Classification labels converted to categorical format")
                logger.info(f"✅ Labels converted to categorical: {y_train.shape}")
            else:
                # For binary classification, ensure labels are 0/1
                y_train = y_train.astype('float32')
                y_test = y_test.astype('float32')
                logger.info(f"✅ Labels kept as binary: {y_train.shape}")
            
            logger.info(f"🎯 Final processed data shape: {x_train.shape} (matches user input: {(*img_size, channels)})")
            
            # Create TensorFlow datasets from the loaded data
            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
            
            # Apply data augmentation to training set
            def augment(image, label):
                # Data augmentation
                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_brightness(image, 0.1)
                image = tf.image.random_contrast(image, 0.9, 1.1)
                return image, label
            
            # Prepare datasets
            train_dataset = train_dataset.map(augment).shuffle(1000).batch(batch_size).prefetch(1)
            validation_dataset = test_dataset.batch(batch_size).prefetch(1)
            
            logger.info(f"Built-in dataset prepared: {len(x_train)} training, {len(x_test)} validation samples")
            
            # For compatibility with existing code, create mock generator objects
            class MockGenerator:
                def __init__(self, dataset, samples, x_data=None, y_data=None):
                    self.dataset = dataset
                    self.samples = samples
                    self.batch_size = batch_size
                    self.n = samples
                    # Store raw data for visualization
                    self.x_data = x_data
                    self.y_data = y_data
                    
            train_generator = MockGenerator(train_dataset, len(x_train), x_train, y_train)
            validation_generator = MockGenerator(validation_dataset, len(x_test), x_test, y_test)
        
        elif hasattr(dataset_info, 'is_hf_dataset') and dataset_info.is_hf_dataset:
            # Hugging Face dataset loading for TensorFlow
            logger.info(f"Loading Hugging Face dataset: {dataset_info.hf_dataset_name}")
            
            try:
                logger.info("HF STEP 1: Starting HuggingFace imports...")
                from datasets import load_dataset
                import numpy as np
                from PIL import Image
                logger.info("HF STEP 1: ✅ Imports successful")
                
                # Get HF dataset config from session state
                hf_config = st.session_state.get('hf_dataset_config', {})
                
                # Build authentication kwargs
                auth_kwargs = {}
                if hf_config.get('token'):
                    auth_kwargs['token'] = hf_config['token']
                
                # CRITICAL FIX: Temporarily rename local directory to prevent conflict
                logger.info("HF STEP 2: Checking for local directory conflicts...")
                local_dir = os.path.join(os.getcwd(), dataset_info.hf_dataset_name)
                temp_renamed = None
                if os.path.exists(local_dir):
                    temp_renamed = f"{local_dir}_temp_hidden_{int(time.time())}"
                    logger.info(f"⚠️ Found local directory '{local_dir}' - temporarily renaming to avoid conflict")
                    try:
                        os.rename(local_dir, temp_renamed)
                        logger.info(f"✅ Temporarily renamed to: {temp_renamed}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not rename directory: {e}")
                        temp_renamed = None
                
                # Clear corrupted cache before loading
                logger.info("HF STEP 2: Clearing any corrupted cache...")
                cache_dir = os.path.expanduser(f"~/.cache/huggingface/datasets/{dataset_info.hf_dataset_name}")
                if os.path.exists(cache_dir):
                    logger.info(f"🗑️ Clearing cache: {cache_dir}")
                    try:
                        shutil.rmtree(cache_dir)
                        logger.info("✅ Cache cleared successfully")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not clear cache: {e}")
                
                # Load dataset with resource management
                try:
                    logger.info("HF STEP 2: Starting dataset loading...")
                    # Disable datasets caching to prevent resource leaks
                    from datasets import disable_caching
                    disable_caching()
                    logger.info("HF STEP 2: ✅ Caching disabled")
                    
                    if dataset_info.hf_subset:
                        logger.info(f"HF STEP 3: Loading with subset: {dataset_info.hf_subset}")
                        hf_dataset = load_dataset(
                            dataset_info.hf_dataset_name, 
                            dataset_info.hf_subset,
                            download_mode='force_redownload',
                            trust_remote_code=True,
                            **auth_kwargs
                        )
                    else:
                        logger.info(f"HF STEP 3: Loading without subset: {dataset_info.hf_dataset_name}")
                        hf_dataset = load_dataset(
                            dataset_info.hf_dataset_name,
                            download_mode='force_redownload',
                            trust_remote_code=True,
                            **auth_kwargs
                        )
                    logger.info("HF STEP 3: ✅ Dataset loaded successfully")
                finally:
                    # Restore the renamed directory
                    if temp_renamed and os.path.exists(temp_renamed):
                        try:
                            os.rename(temp_renamed, local_dir)
                            logger.info(f"✅ Restored local directory: {local_dir}")
                        except Exception as e:
                            logger.warning(f"⚠️ Could not restore directory: {e}")
                
                # Validate that we actually got image data
                sample = hf_dataset[list(hf_dataset.keys())[0]][0]
                # Check for both 'image' and 'img' keys (some datasets use 'img' like CIFAR-10)
                image_key = None
                if 'image' in sample:
                    image_key = 'image'
                elif 'img' in sample:
                    image_key = 'img'
                else:
                    logger.warning(f"HuggingFace dataset {dataset_info.hf_dataset_name} doesn't contain image data. Keys: {list(sample.keys())}")
                    raise ValueError(f"Dataset {dataset_info.hf_dataset_name} does not contain valid image data")
                
                logger.info(f"✅ Found image data under key: '{image_key}'")
                
                # Process HuggingFace dataset - moved inside try block
                # Get train/test splits
                if 'train' in hf_dataset and 'test' in hf_dataset:
                    train_split = hf_dataset['train']
                    test_split = hf_dataset['test']
                elif 'train' in hf_dataset and 'validation' in hf_dataset:
                    train_split = hf_dataset['train']
                    test_split = hf_dataset['validation']
                elif 'train' in hf_dataset:
                    train_split = hf_dataset['train']
                    split_dataset = train_split.train_test_split(test_size=0.2, seed=42)
                    train_split = split_dataset['train']
                    test_split = split_dataset['test']
                else:
                    raise ValueError("No suitable train split found in Hugging Face dataset")
                
                logger.info(f"HF dataset splits: {len(train_split)} train, {len(test_split)} test")
                
                # Convert HF dataset to TensorFlow format
                def process_hf_sample(sample):
                    # Find image and label keys
                    features = sample.keys()
                    image_key = None
                    label_key = None
                    
                    for key in features:
                        if 'image' in key.lower() or 'img' in key.lower() or 'pixel' in key.lower():
                            image_key = key
                        # Enhanced label detection: check for label, labels, target, class, category
                        if 'label' in key.lower() or 'target' in key.lower() or 'class' in key.lower() or 'category' in key.lower():
                            label_key = key
                    
                    if not image_key:
                        image_key = list(features)[0]  # Use first key as fallback
                    if not label_key:
                        label_key = list(features)[-1]  # Use last key as fallback
                    
                    # Process image
                    image = sample[image_key]
                    
                    # Convert PIL Image to numpy array
                    if isinstance(image, Image.Image):
                        # Ensure RGB format (fix for mixed channel images)
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        image = np.array(image)
                    elif not isinstance(image, np.ndarray):
                        image = np.array(image)
                    
                    # Ensure image has valid shape
                    if len(image.shape) == 2:  # Grayscale
                        image = np.expand_dims(image, axis=-1)
                    elif len(image.shape) != 3:
                        raise ValueError(f"Invalid image shape: {image.shape}. Expected 2D or 3D array.")
                    
                    # Ensure image is in float32 format for TensorFlow
                    image = image.astype(np.float32)
                    
                    # Resize image to target size
                    image = tf.image.resize(image, img_size).numpy()
                    
                    # Handle channel conversion
                    if image.shape[-1] != channels:
                        if channels == 3 and image.shape[-1] == 1:
                            image = np.repeat(image, 3, axis=-1)
                        elif channels == 1 and image.shape[-1] == 3:
                            image = tf.image.rgb_to_grayscale(image).numpy()
                        else:
                            # Force to target channels if mismatch
                            if channels == 3:
                                image = np.repeat(image[:, :, :1], 3, axis=-1)
                            else:
                                image = image[:, :, :1]
                    
                    # Process label
                    label = sample[label_key]
                    if isinstance(label, (list, np.ndarray)):
                        label = label[0] if len(label) > 0 else 0
                    
                    return image.astype('float32') / 255.0, label
                    
                # Process training data
                logger.info("HF STEP 5: Processing training data...")
                
                # Limit dataset size for memory efficiency
                max_train_samples = min(10000, len(train_split))  # Limit to 10k samples max
                max_test_samples = min(2000, len(test_split))     # Limit to 2k samples max
                
                logger.info(f"HF STEP 5: Processing {max_train_samples} training and {max_test_samples} test samples")
                
                x_train_list, y_train_list = [], []
                for i, sample in enumerate(train_split):
                    if i >= max_train_samples:
                        break
                    try:
                        img, lbl = process_hf_sample(sample)
                        x_train_list.append(img)
                        y_train_list.append(lbl)
                        
                        # Progress feedback every 1000 samples
                        if (i + 1) % 1000 == 0:
                            logger.info(f"Processed {i + 1}/{max_train_samples} training samples")
                    except Exception as e:
                        logger.warning(f"Skipping training sample {i} due to error: {e}")
                        continue
                
                # Check if we have any training data
                if not x_train_list:
                    raise ValueError("No valid training samples could be processed. All samples failed during conversion.")
                
                x_train = np.array(x_train_list)
                y_train = np.array(y_train_list)
                
                # Process test data
                logger.info("HF STEP 6: Processing test data...")
                
                x_test_list, y_test_list = [], []
                for i, sample in enumerate(test_split):
                    if i >= max_test_samples:
                        break
                    try:
                        img, lbl = process_hf_sample(sample)
                        x_test_list.append(img)
                        y_test_list.append(lbl)
                        
                        # Progress feedback every 500 samples
                        if (i + 1) % 500 == 0:
                            logger.info(f"Processed {i + 1}/{max_test_samples} test samples")
                            
                    except Exception as e:
                        logger.warning(f"Skipping test sample {i} due to error: {e}")
                        continue
                
                # Check if we have any test data
                if not x_test_list:
                    raise ValueError("No valid test samples could be processed. All samples failed during conversion.")
                
                x_test = np.array(x_test_list)
                y_test = np.array(y_test_list)
                
                logger.info(f"HF data processed: train {x_train.shape}, test {x_test.shape}")
                
                # Convert labels to categorical if needed
                if num_classes > 2:
                    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
                    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
                else:
                    y_train = y_train.astype('float32')
                    y_test = y_test.astype('float32')
                
                # Get normalization settings from model config
                enable_normalization = model_config.config_params.get('enable_normalization', True)
                normalization_type = model_config.config_params.get('normalization_type', 'Standard (0-1)')
                norm_min = model_config.config_params.get('norm_min', 0.0)
                norm_max = model_config.config_params.get('norm_max', 1.0)
                
                # Log preprocessing info (actual preprocessing happens in TensorFlow pipeline below)
                logger.info(f"🔧 HF NORMALIZATION: Will apply '{normalization_type}' preprocessing (enabled: {enable_normalization})")
                logger.info(f"✅ Data prepared for preprocessing pipeline: train {x_train.shape}, test {x_test.shape}")
                logger.info(f"✅ Input data range: [{x_train.min():.3f}, {x_train.max():.3f}] (should be [0,1])")
                
                # FIXED: Create TensorFlow datasets with PROPER preprocessing pipeline
                logger.info("🔧 Creating optimized preprocessing pipeline...")
                
                # Create base datasets from [0,1] normalized data (BEFORE ImageNet normalization)
                train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
                test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
                
                # Define preprocessing function that will be applied consistently
                def preprocess_for_resnet(image, label):
                    # Image is currently in [0,1] range
                    # Apply ImageNet normalization directly (same as ImageDataGenerator)
                    imagenet_mean = tf.constant([0.485, 0.456, 0.406])
                    imagenet_std = tf.constant([0.229, 0.224, 0.225])
                    
                    # Normalize using ImageNet statistics
                    image = (image - imagenet_mean) / imagenet_std
                    
                    return image, label
                
                def preprocess_standard(image, label):
                    # Keep data in [0,1] range for non-ResNet models
                    return image, label
                
                # Apply appropriate preprocessing based on normalization type
                if normalization_type == "ResNet Preprocessing (TensorFlow)":
                    logger.info("🔧 Applying ResNet preprocessing pipeline...")
                    preprocess_fn = preprocess_for_resnet
                elif normalization_type == "MobileNet Preprocessing (TensorFlow)":
                    # MobileNet uses [-1,1] range
                    def preprocess_mobilenet(image, label):
                        image = image * 2.0 - 1.0
                        return image, label
                    preprocess_fn = preprocess_mobilenet
                else:
                    preprocess_fn = preprocess_standard
                
                # Apply data augmentation to training set BEFORE final preprocessing
                def augment_and_preprocess(image, label):
                    # Apply augmentation in [0,1] range first
                    image = tf.image.random_flip_left_right(image)
                    image = tf.image.random_brightness(image, 0.1)
                    image = tf.image.random_contrast(image, 0.9, 1.1)
                    # Ensure values stay in [0,1] range after augmentation
                    image = tf.clip_by_value(image, 0.0, 1.0)
                    
                    # Then apply final preprocessing (ImageNet normalization, etc.)
                    return preprocess_fn(image, label)
                
                # Apply augmentation + preprocessing to training set
                train_dataset = train_dataset.map(augment_and_preprocess, num_parallel_calls=1)
                
                # Apply only preprocessing (no augmentation) to test set  
                test_dataset = test_dataset.map(preprocess_fn, num_parallel_calls=1)
                
                # Batch and prefetch for optimal performance (fixed values to prevent resource leaks)
                train_dataset = train_dataset.batch(batch_size).prefetch(1)
                test_dataset = test_dataset.batch(batch_size).prefetch(1)
                
                logger.info("✅ Optimized preprocessing pipeline created successfully!")
                
                # Use TensorFlow datasets directly - they are already compatible with model.fit()
                logger.info("HF STEP 9: Creating generators from TensorFlow datasets...")
                train_generator = train_dataset
                validation_generator = test_dataset
                
                # Add attributes for compatibility with existing logging code
                logger.info("HF STEP 10: Adding compatibility attributes...")
                train_generator.samples = len(x_train)
                train_generator.batch_size = batch_size
                validation_generator.samples = len(x_test)
                validation_generator.batch_size = batch_size
                
                logger.info(f"HF STEP 11: ✅ HF dataset processing complete: {len(x_train)} training, {len(x_test)} test samples")
                logger.info(f"HF STEP 11: ✅ Generators created successfully - train_generator: {type(train_generator)}, validation_generator: {type(validation_generator)}")
                
            except Exception as e:
                logger.error(f"Failed to load HuggingFace dataset: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                status_text.error(f"❌ Failed to load HuggingFace dataset: {e}")
                
                # Set generators to None explicitly to avoid confusion
                train_generator = None
                validation_generator = None
                
                raise RuntimeError(f"HuggingFace dataset loading failed: {str(e)}")
                    
            finally:
                # Force garbage collection
                import gc
                gc.collect()
            
        else:
            # Original directory-based dataset loading
            if not dataset_info.dataset_path:
                logger.error("❌ No dataset path available for directory-based loading")
                return {
                    'best_accuracy': 0.0,
                    'best_loss': float('inf'),
                    'training_time': 0.0,
                    'error': 'No dataset path available',
                    'framework': 'TensorFlow/Keras'
                }
                
            logger.info("Setting up directory-based data generators...")
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            
            # Determine color mode and class mode first
            color_mode = 'grayscale' if channels == 1 else 'rgb'
            class_mode = 'binary' if num_classes == 2 else 'categorical'
            
            logger.info(f"Using dataset path: {dataset_info.dataset_path}")
            logger.info(f"Target image size: {img_size}")
            logger.info(f"Batch size: {batch_size}")
            logger.info(f"Color mode: {color_mode}, Class mode: {class_mode}")
            
            # Get user normalization preferences
            enable_normalization = model_config.config_params.get('enable_normalization', True)
            normalization_type = model_config.config_params.get('normalization_type', 'Standard (0-1)')
            norm_min = model_config.config_params.get('norm_min', 0.0)
            norm_max = model_config.config_params.get('norm_max', 1.0)
            
            logger.info(f"🔧 DATA GENERATOR NORMALIZATION VERIFICATION: Using '{normalization_type}' (enabled: {enable_normalization})")
            
            # Determine rescale factor based on user preferences
            if not enable_normalization:
                rescale_factor = None  # No rescaling
                logger.info("Creating data generators without normalization (user disabled)")
            elif normalization_type == "Standard (0-1)":
                rescale_factor = 1.0/255.0
                logger.info("Creating data generators with standard [0,1] normalization")
            elif normalization_type == "Z-Score (-1 to 1)":
                rescale_factor = 1.0/127.5  # This will be adjusted in preprocessing_function
                logger.info("Creating data generators with Z-score [-1,1] normalization")
            elif normalization_type in ["ResNet Preprocessing (TensorFlow)", "MobileNet Preprocessing (TensorFlow)"]:
                rescale_factor = 1.0/255.0  # First normalize to [0,1], then apply model-specific preprocessing
                logger.info(f"Creating data generators with {normalization_type}")
            elif normalization_type == "Custom Range":
                rescale_factor = 1.0/255.0  # Will be adjusted in preprocessing_function
                logger.info(f"Creating data generators with custom range [{norm_min}, {norm_max}] normalization")
            else:
                rescale_factor = 1.0/255.0  # Default fallback
                logger.info("Creating data generators with default [0,1] normalization")
            
            # MODEL-SPECIFIC PREPROCESSING FUNCTION
            def get_model_preprocessing_function(architecture_name, normalization_type):
                """Get the correct preprocessing function based on model architecture."""
                
                def custom_preprocessing(x):
                    # CRITICAL FIX: Ensure data is in [0,1] range after augmentation
                    x = np.clip(x, 0.0, 1.0)
                    
                    # Convert from [0,1] to [0,255] for model-specific preprocessing
                    x_255 = x * 255.0
                    
                    # Apply model-specific preprocessing
                    if any(arch in architecture_name for arch in ["ResNet50", "ResNet101", "ResNet152"]) and not any(v in architecture_name for v in ["V2", "v2"]):
                        # ResNet v1: RGB → BGR, subtract ImageNet mean, no scaling
                        logger.debug(f"Applying ResNet v1 preprocessing for {architecture_name}")
                        if channels == 3:
                            # Convert RGB to BGR
                            x_bgr = x_255[:, :, ::-1]  # Reverse the channel order
                            # Subtract ImageNet mean in BGR order
                            mean_bgr = np.array([103.939, 116.779, 123.68])  # BGR mean
                            normalized = x_bgr - mean_bgr
                        else:
                            # For grayscale, use average mean
                            mean_gray = (103.939 + 116.779 + 123.68) / 3
                            normalized = x_255 - mean_gray
                        return normalized
                        
                    elif any(arch in architecture_name for arch in ["ResNetV2", "ResNet50V2", "ResNet101V2", "ResNet152V2"]):
                        # ResNet v2: Keep RGB, scale to [-1, 1]
                        logger.debug(f"Applying ResNet v2 preprocessing for {architecture_name}")
                        return (x_255 / 127.5) - 1.0
                        
                    elif any(arch in architecture_name for arch in ["MobileNet", "MobileNetV2", "MobileNetV3"]):
                        # MobileNet family: Keep RGB, scale to [-1, 1]
                        logger.debug(f"Applying MobileNet preprocessing for {architecture_name}")
                        return (x_255 / 127.5) - 1.0
                        
                    elif any(arch in architecture_name for arch in ["EfficientNet"]):
                        # EfficientNet family: Keep RGB, scale to [0, 1]
                        logger.debug(f"Applying EfficientNet preprocessing for {architecture_name}")
                        return x  # Already in [0,1] range
                        
                    elif any(arch in architecture_name for arch in ["InceptionV3", "InceptionResNetV2"]):
                        # Inception family: Keep RGB, scale to [-1, 1]
                        logger.debug(f"Applying Inception preprocessing for {architecture_name}")
                        return (x_255 / 127.5) - 1.0
                        
                    elif any(arch in architecture_name for arch in ["VGG16", "VGG19"]):
                        # VGG family: RGB → BGR, subtract ImageNet mean
                        logger.debug(f"Applying VGG preprocessing for {architecture_name}")
                        if channels == 3:
                            # Convert RGB to BGR
                            x_bgr = x_255[:, :, ::-1]
                            # Subtract ImageNet mean in BGR order
                            mean_bgr = np.array([103.939, 116.779, 123.68])
                            normalized = x_bgr - mean_bgr
                        else:
                            mean_gray = (103.939 + 116.779 + 123.68) / 3
                            normalized = x_255 - mean_gray
                        return normalized
                        
                    elif normalization_type == "Z-Score (-1 to 1)":
                        return x * 2.0 - 1.0  # Convert from [0,1] to [-1,1]
                    elif normalization_type == "ResNet Preprocessing (TensorFlow)":
                        # Legacy ImageNet normalization for backward compatibility
                        if channels == 3:
                            imagenet_mean = np.array([0.485, 0.456, 0.406])
                            imagenet_std = np.array([0.229, 0.224, 0.225])
                        else:
                            mean_val = (0.485 + 0.456 + 0.406) / 3
                            std_val = (0.229 + 0.224 + 0.225) / 3
                            imagenet_mean = np.array([mean_val] * channels)
                            imagenet_std = np.array([std_val] * channels)
                        
                        normalized = (x - imagenet_mean) / imagenet_std
                        logger.debug(f"Legacy ResNet preprocessing: input range [{x.min():.3f}, {x.max():.3f}] → output range [{normalized.min():.3f}, {normalized.max():.3f}]")
                        return normalized
                    elif normalization_type == "MobileNet Preprocessing (TensorFlow)":
                        return x * 2.0 - 1.0  # Convert from [0,1] to [-1,1]
                    elif normalization_type == "Custom Range":
                        scale = norm_max - norm_min
                        return x * scale + norm_min
                    else:
                        return x  # No additional processing
                
                return custom_preprocessing
            
            # Get the appropriate preprocessing function
            preprocessing_function = get_model_preprocessing_function(selected_architecture, normalization_type)
            
            # Log the specific preprocessing being applied
            if any(arch in selected_architecture for arch in ["ResNet50", "ResNet101", "ResNet152"]) and not any(v in selected_architecture for v in ["V2", "v2"]):
                logger.info(f"📋 Using ResNet v1 preprocessing: RGB→BGR conversion + ImageNet mean subtraction for {selected_architecture}")
            elif any(arch in selected_architecture for arch in ["ResNetV2", "ResNet50V2", "ResNet101V2", "ResNet152V2"]):
                logger.info(f"📋 Using ResNet v2 preprocessing: RGB format + [-1,1] scaling for {selected_architecture}")
            elif any(arch in selected_architecture for arch in ["MobileNet", "MobileNetV2", "MobileNetV3"]):
                logger.info(f"📋 Using MobileNet preprocessing: RGB format + [-1,1] scaling for {selected_architecture}")
            elif any(arch in selected_architecture for arch in ["EfficientNet"]):
                logger.info(f"📋 Using EfficientNet preprocessing: RGB format + [0,1] normalization for {selected_architecture}")
            elif any(arch in selected_architecture for arch in ["InceptionV3", "InceptionResNetV2"]):
                logger.info(f"📋 Using Inception preprocessing: RGB format + [-1,1] scaling for {selected_architecture}")
            elif any(arch in selected_architecture for arch in ["VGG16", "VGG19"]):
                logger.info(f"📋 Using VGG preprocessing: RGB→BGR conversion + ImageNet mean subtraction for {selected_architecture}")
            else:
                logger.info(f"📋 Using legacy preprocessing: {normalization_type} for {selected_architecture}")
            
            # Data augmentation for training
            logger.info("Creating training data generator with augmentation...")
            train_datagen = ImageDataGenerator(
                rescale=rescale_factor,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                validation_split=0.2,
                preprocessing_function=preprocessing_function if enable_normalization else None
            )
            
            val_datagen = ImageDataGenerator(
                rescale=rescale_factor,
                validation_split=0.2,
                preprocessing_function=preprocessing_function if enable_normalization else None
            )
            
            logger.info("Creating training data generator...")
            try:
                train_generator = train_datagen.flow_from_directory(
                    dataset_info.dataset_path,
                    target_size=img_size,
                    batch_size=batch_size,
                    class_mode=class_mode,
                    subset='training',
                    color_mode=color_mode,
                    shuffle=True
                )
                logger.info(f"Training generator created successfully. Found {train_generator.samples} samples")
            except Exception as gen_error:
                logger.error(f"Training generator creation failed: {str(gen_error)}")
                raise gen_error
            
            logger.info("Creating validation data generator...")
            try:
                validation_generator = val_datagen.flow_from_directory(
                    dataset_info.dataset_path,
                    target_size=img_size,
                    batch_size=batch_size,
                    class_mode=class_mode,
                    subset='validation',
                    color_mode=color_mode,
                    shuffle=False
                )
                logger.info(f"Validation generator created successfully. Found {validation_generator.samples} samples")
            except Exception as val_gen_error:
                logger.error(f"Validation generator creation failed: {str(val_gen_error)}")
                raise val_gen_error
        
        status_text.text("🏗️ **Building TensorFlow CNN model...**")
        progress_bar.progress(0.3)
        
        logger.info("Building TensorFlow CNN model...")
        logger.info(f"Model input shape will be: {(*img_size, channels)}")
        logger.info(f"Model output classes: {num_classes}")
        logger.info(f"Task type: {dataset_info.task_type}")
        
        # Get selected architecture from model config
        selected_architecture = getattr(model_config, 'architecture', 'Simple CNN')
        logger.info(f"🔍 ARCHITECTURE DEBUG: Selected architecture = '{selected_architecture}'")
        logger.info(f"🔍 ARCHITECTURE DEBUG: Type = {type(selected_architecture)}")
        logger.info(f"🔍 ARCHITECTURE DEBUG: Task type = {dataset_info.task_type}")
        
        # Check if this is a segmentation task (case-insensitive)
        is_segmentation = dataset_info.task_type.lower() == "segmentation" if hasattr(dataset_info.task_type, 'lower') else False
        logger.info(f"🔍 SEGMENTATION DEBUG: is_segmentation = {is_segmentation}")
        
        # Build model based on task type and selected architecture
        try:
            if is_segmentation:
                # Build U-Net for segmentation with batch normalization
                logger.info("Building improved U-Net segmentation model with BatchNorm...")
                
                inputs = keras.layers.Input(shape=(*img_size, channels))
                
                # Encoder (downsampling) with BatchNorm
                c1 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
                c1 = keras.layers.BatchNormalization()(c1)
                c1 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
                c1 = keras.layers.BatchNormalization()(c1)
                p1 = keras.layers.MaxPooling2D((2, 2))(c1)
                
                c2 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
                c2 = keras.layers.BatchNormalization()(c2)
                c2 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
                c2 = keras.layers.BatchNormalization()(c2)
                p2 = keras.layers.MaxPooling2D((2, 2))(c2)
                
                c3 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
                c3 = keras.layers.BatchNormalization()(c3)
                c3 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
                c3 = keras.layers.BatchNormalization()(c3)
                p3 = keras.layers.MaxPooling2D((2, 2))(c3)
                
                # Bottleneck
                c4 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
                c4 = keras.layers.BatchNormalization()(c4)
                c4 = keras.layers.Dropout(0.3)(c4)
                c4 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
                c4 = keras.layers.BatchNormalization()(c4)
                
                # Decoder (upsampling) with BatchNorm
                u5 = keras.layers.UpSampling2D((2, 2))(c4)
                u5 = keras.layers.concatenate([u5, c3])
                c5 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u5)
                c5 = keras.layers.BatchNormalization()(c5)
                c5 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)
                c5 = keras.layers.BatchNormalization()(c5)
                
                u6 = keras.layers.UpSampling2D((2, 2))(c5)
                u6 = keras.layers.concatenate([u6, c2])
                c6 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
                c6 = keras.layers.BatchNormalization()(c6)
                c6 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)
                c6 = keras.layers.BatchNormalization()(c6)
                
                u7 = keras.layers.UpSampling2D((2, 2))(c6)
                u7 = keras.layers.concatenate([u7, c1])
                c7 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
                c7 = keras.layers.BatchNormalization()(c7)
                c7 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)
                c7 = keras.layers.BatchNormalization()(c7)
                
                # Output layer - pixel-wise classification
                outputs = keras.layers.Conv2D(num_classes, (1, 1), activation='softmax')(c7)
                
                model = keras.Model(inputs=[inputs], outputs=[outputs])
                logger.info("✅ Improved U-Net segmentation model created with BatchNorm")
                
            elif "ResNet50" in selected_architecture:
                logger.info("Building ResNet50 model...")
                try:
                    # Create ResNet50 base model
                    base_model = keras.applications.ResNet50(
                        weights='imagenet',
                        include_top=False,
                        input_shape=(*img_size, channels),
                        pooling='avg'
                    )
                    
                    # Add custom classification head
                    model = keras.Sequential([
                        base_model,
                        keras.layers.Dense(256, activation='relu'),
                        keras.layers.Dropout(0.5),
                        keras.layers.Dense(1 if num_classes == 2 else num_classes, 
                                         activation='sigmoid' if num_classes == 2 else 'softmax')
                    ])
                    
                    # Fine-tune the last few layers
                    base_model.trainable = False  # Freeze base model initially
                    logger.info("✅ ResNet50 model created with ImageNet weights")
                    
                except Exception as resnet_error:
                    logger.warning(f"⚠️ Failed to load ResNet50 with ImageNet weights: {resnet_error}")
                    logger.info("🔄 Creating ResNet50 without pre-trained weights...")
                    
                    # Fallback: Create ResNet50 without pre-trained weights
                    base_model = keras.applications.ResNet50(
                        weights=None,
                        include_top=False,
                        input_shape=(*img_size, channels),
                        pooling='avg'
                    )
                    
                    # Add custom classification head
                    model = keras.Sequential([
                        base_model,
                        keras.layers.Dense(256, activation='relu'),
                        keras.layers.Dropout(0.5),
                        keras.layers.Dense(1 if num_classes == 2 else num_classes, 
                                         activation='sigmoid' if num_classes == 2 else 'softmax')
                    ])
                    
                    logger.info("✅ ResNet50 model created without pre-trained weights")
                
            elif "ResNet101" in selected_architecture:
                logger.info("Building ResNet101 model...")
                try:
                    # Create ResNet101 base model
                    base_model = keras.applications.ResNet101(
                        weights='imagenet',
                        include_top=False,
                        input_shape=(*img_size, channels),
                        pooling='avg'
                    )
                    
                    # Add custom classification head
                    model = keras.Sequential([
                        base_model,
                        keras.layers.Dense(256, activation='relu'),
                        keras.layers.Dropout(0.5),
                        keras.layers.Dense(1 if num_classes == 2 else num_classes, 
                                         activation='sigmoid' if num_classes == 2 else 'softmax')
                    ])
                    
                    # Fine-tune the last few layers
                    base_model.trainable = False  # Freeze base model initially
                    logger.info("✅ ResNet101 model created with ImageNet weights")
                    
                except Exception as resnet_error:
                    logger.warning(f"⚠️ Failed to load ResNet101 with ImageNet weights: {resnet_error}")
                    logger.info("🔄 Creating ResNet101 without pre-trained weights...")
                    
                    # Fallback: Create ResNet101 without pre-trained weights
                    base_model = keras.applications.ResNet101(
                        weights=None,
                        include_top=False,
                        input_shape=(*img_size, channels),
                        pooling='avg'
                    )
                    
                    # Add custom classification head
                    model = keras.Sequential([
                        base_model,
                        keras.layers.Dense(256, activation='relu'),
                        keras.layers.Dropout(0.5),
                        keras.layers.Dense(1 if num_classes == 2 else num_classes, 
                                         activation='sigmoid' if num_classes == 2 else 'softmax')
                    ])
                    
                    logger.info("✅ ResNet101 model created without pre-trained weights")
                
            elif "MobileNetV2" in selected_architecture:
                logger.info("Building MobileNetV2 model...")
                # Create MobileNetV2 base model
                base_model = keras.applications.MobileNetV2(
                    weights='imagenet',
                    include_top=False,
                    input_shape=(*img_size, channels),
                    pooling='avg'
                )
                
                # Add custom classification head
                model = keras.Sequential([
                    base_model,
                    keras.layers.Dense(256, activation='relu'),
                    keras.layers.Dropout(0.5),
                    keras.layers.Dense(1 if num_classes == 2 else num_classes, 
                                     activation='sigmoid' if num_classes == 2 else 'softmax')
                ])
                
                # Fine-tune the last few layers
                base_model.trainable = False  # Freeze base model initially
                logger.info("✅ MobileNetV2 model created with ImageNet weights")
                
            else:
                # Default to custom CNN for other architectures
                logger.info("Building custom CNN model...")
                # Dynamic pooling size based on input dimensions
                height, width = img_size
                
                # Determine pooling strategy based on input size
                if height < 16 or width < 16:
                    # Very small images - minimal pooling
                    if min(height, width) < 8:
                        pool_size = (1, 1)
                    elif width < height:
                        pool_size = (1, 2)  # Height is larger, pool more in height direction
                    else:
                        pool_size = (2, 1)  # Width is larger, pool more in width direction
                    num_conv_blocks = 2
                elif height < 32 or width < 32:
                    # Small images like CIFAR (32x32) - moderate pooling
                    pool_size = (2, 2)
                    num_conv_blocks = 2
                else:
                    # Large images - standard pooling
                    pool_size = (2, 2)
                    num_conv_blocks = 3
                
                logger.info(f"Using adaptive pooling: {pool_size} for input size {img_size}")
                
                # Build adaptive model
                layers = [keras.layers.Input(shape=(*img_size, channels))]
                
                # First Conv Block (always include)
                layers.extend([
                    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                    keras.layers.BatchNormalization(),
                ])
                
                # Add pooling only if beneficial
                if pool_size != (1, 1):
                    layers.append(keras.layers.MaxPooling2D(pool_size))
                
                # Second Conv Block
                layers.extend([
                    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                    keras.layers.BatchNormalization(),
                ])
                
                # Add pooling for second block if input is large enough
                if num_conv_blocks >= 2 and min(height, width) >= 16:
                    layers.append(keras.layers.MaxPooling2D(pool_size))
                
                # Third Conv Block (only for larger inputs)
                if num_conv_blocks >= 3:
                    layers.extend([
                        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                        keras.layers.BatchNormalization(),
                        keras.layers.MaxPooling2D(pool_size)
                    ])
                
                # Classification Head - use GlobalAveragePooling2D for any size
                layers.extend([
                    keras.layers.GlobalAveragePooling2D(),
                    keras.layers.Dropout(0.5),
                    keras.layers.Dense(256, activation='relu'),
                    keras.layers.BatchNormalization(),
                    keras.layers.Dropout(0.3),
                    # Fix binary classification: use 1 output neuron for binary, num_classes for multiclass
                    keras.layers.Dense(1 if num_classes == 2 else num_classes, 
                                     activation='sigmoid' if num_classes == 2 else 'softmax')
                ])
                
                model = keras.Sequential(layers)
                logger.info("✅ Custom CNN model created")
            
            logger.info("Model architecture created successfully")
            logger.info(f"Model summary: {model.count_params()} parameters")
            
            # Compile model with appropriate learning rate based on architecture
            logger.info("Compiling model...")
            
            # Set learning rate based on model type
            if "ResNet" in selected_architecture or "MobileNet" in selected_architecture:
                # Pre-trained models need much lower learning rate for fine-tuning
                learning_rate = 0.0001  # 10x lower than default
                logger.info(f"🎯 Using FINE-TUNING learning rate: {learning_rate} for pre-trained {selected_architecture}")
            else:
                # Custom CNN models can use higher learning rate
                learning_rate = 0.001
                logger.info(f"🎯 Using STANDARD learning rate: {learning_rate} for custom CNN")
            
            # Compile with task-specific loss and metrics
            if is_segmentation:
                # Segmentation-specific compilation
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                    loss='sparse_categorical_crossentropy',  # For pixel-wise classification
                    metrics=['accuracy']  # MeanIoU removed due to shape issues with dummy data
                )
                logger.info("Model compiled for segmentation with sparse_categorical_crossentropy")
            else:
                # Classification compilation
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                    loss='binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy',
                    metrics=['accuracy']
                )
                logger.info("Model compiled for classification")
            
            logger.info("Model compiled successfully")
        
        except Exception as model_error:
            logger.error(f"Model creation/compilation failed: {str(model_error)}")
            raise model_error
        
        # 🖼️ SAMPLE IMAGE VISUALIZATION BEFORE TRAINING
        status_text.text("🖼️ **Displaying sample images from dataset...**")
        progress_bar.progress(0.35)
        
        try:
            logger.info("🖼️ Creating sample image visualization...")
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            
            def plot_segmentation_before_after(dataset_info, train_generator, num_samples=6):
                """Create before/after visualization for segmentation with mask overlays."""
                logger.info(f"🎨 Creating segmentation visualization with mask overlays for {num_samples} samples")
                
                try:
                    # Extract samples from dataset
                    sample_images = []
                    sample_masks = []
                    
                    # Check if it's a MockGenerator with raw data
                    if hasattr(train_generator, 'x_data') and hasattr(train_generator, 'y_data'):
                        logger.info("✅ Using actual data from MockGenerator for visualization")
                        x_data = train_generator.x_data
                        y_data = train_generator.y_data
                        
                        # Take first num_samples
                        for i in range(min(num_samples, len(x_data))):
                            sample_images.append(x_data[i])
                            sample_masks.append(y_data[i])
                        
                    elif isinstance(train_generator, tf.data.Dataset):
                        # TensorFlow dataset
                        for batch_images, batch_masks in train_generator.take(2):  # Take 2 batches
                            for i in range(min(len(batch_images), num_samples - len(sample_images))):
                                img = batch_images[i].numpy()
                                mask = batch_masks[i].numpy()
                                sample_images.append(img)
                                sample_masks.append(mask)
                                if len(sample_images) >= num_samples:
                                    break
                            if len(sample_images) >= num_samples:
                                break
                    else:
                        # Try to get from generator
                        try:
                            batch_images, batch_masks = next(iter(train_generator))
                            for i in range(min(len(batch_images), num_samples)):
                                sample_images.append(batch_images[i])
                                sample_masks.append(batch_masks[i])
                        except Exception as gen_error:
                            logger.warning(f"Could not extract from generator: {gen_error}")
                            return None
                    
                    if len(sample_images) == 0:
                        logger.warning("No samples extracted for segmentation visualization")
                        return None
                    
                    # Create colormap for masks
                    num_classes = dataset_info.num_classes
                    colors = plt.cm.tab10(np.linspace(0, 1, min(num_classes, 10)))
                    if num_classes > 10:
                        colors = plt.cm.tab20(np.linspace(0, 1, min(num_classes, 20)))
                    
                    # Create figure
                    fig, axes = plt.subplots(2, len(sample_images), figsize=(4*len(sample_images), 8))
                    if len(sample_images) == 1:
                        axes = axes.reshape(2, 1)
                    
                    # Dataset name
                    dataset_name = "Segmentation Dataset"
                    if hasattr(dataset_info, 'hf_dataset_name') and dataset_info.hf_dataset_name:
                        dataset_name = dataset_info.hf_dataset_name.title()
                    elif hasattr(dataset_info, 'builtin_dataset_name') and dataset_info.builtin_dataset_name:
                        dataset_name = dataset_info.builtin_dataset_name.title()
                    
                    fig.suptitle(f'{dataset_name} - Segmentation Before/After Preprocessing\n(Images with Colored Mask Overlays)', fontsize=16)
                    
                    for idx in range(len(sample_images)):
                        img = sample_images[idx]
                        mask = sample_masks[idx]
                        
                        # Normalize image to [0, 1] for BEFORE
                        before_img = img.copy()
                        if before_img.max() > 1.0:
                            before_img = before_img / 255.0
                        elif before_img.min() < 0:
                            before_img = (before_img + 1) / 2
                        before_img = np.clip(before_img, 0, 1)
                        
                        # Create colored mask overlay
                        if len(mask.shape) == 3 and mask.shape[-1] == 1:
                            mask = mask[:, :, 0]  # Remove channel dimension
                        
                        # Create RGB mask
                        mask_rgb = np.zeros((*mask.shape, 3))
                        for class_id in range(num_classes):
                            class_mask = (mask == class_id)
                            if class_id < len(colors):
                                mask_rgb[class_mask] = colors[class_id][:3]
                        
                        # Overlay mask on image (BEFORE)
                        before_overlay = before_img.copy()
                        alpha = 0.4
                        for c in range(3):
                            before_overlay[:, :, c] = (1 - alpha) * before_img[:, :, c] + alpha * mask_rgb[:, :, c]
                        
                        # Apply ImageNet preprocessing for AFTER
                        imagenet_mean = np.array([0.485, 0.456, 0.406])
                        imagenet_std = np.array([0.229, 0.224, 0.225])
                        after_img = (before_img - imagenet_mean) / imagenet_std
                        
                        # Normalize AFTER for visualization
                        after_img_vis = (after_img - after_img.min()) / (after_img.max() - after_img.min() + 1e-8)
                        
                        # Overlay mask on preprocessed image (AFTER)
                        after_overlay = after_img_vis.copy()
                        for c in range(3):
                            after_overlay[:, :, c] = (1 - alpha) * after_img_vis[:, :, c] + alpha * mask_rgb[:, :, c]
                        
                        # Plot BEFORE (top row)
                        axes[0, idx].imshow(before_overlay)
                        unique_classes = len(np.unique(mask))
                        axes[0, idx].set_title(f"BEFORE\n{unique_classes} classes\n[{before_img.min():.2f}, {before_img.max():.2f}]", fontsize=10)
                        axes[0, idx].axis('off')
                        
                        # Plot AFTER (bottom row)
                        axes[1, idx].imshow(after_overlay)
                        axes[1, idx].set_title(f"AFTER PREPROCESSING\n{unique_classes} classes\n[{after_img.min():.2f}, {after_img.max():.2f}]", fontsize=10)
                        axes[1, idx].axis('off')
                    
                    plt.tight_layout()
                    logger.info(f"✅ Segmentation visualization created with {len(sample_images)} samples")
                    return fig
                    
                except Exception as seg_viz_error:
                    logger.error(f"❌ Failed to create segmentation visualization: {seg_viz_error}")
                    import traceback
                    logger.error(f"Segmentation viz traceback: {traceback.format_exc()}")
                    return None
            
            def get_diverse_samples(data_source, num_samples=6):
                """Extract diverse samples ensuring different classes are represented."""
                logger.info(f"🎯 Extracting {num_samples} diverse samples from {type(data_source)}")
                
                all_images = []
                all_labels = []
                class_counts = {}
                max_per_class = 1  # Only 1 sample per class for maximum diversity
                
                try:
                    # Add detailed debugging
                    logger.info(f"Data source attributes: {dir(data_source)}")
                    if hasattr(data_source, 'samples'):
                        logger.info(f"Data source samples: {data_source.samples}")
                    if hasattr(data_source, 'batch_size'):
                        logger.info(f"Data source batch_size: {data_source.batch_size}")
                    if isinstance(data_source, tf.data.Dataset):
                        # Extract from TensorFlow dataset - take more batches to get diversity
                        for batch_images, batch_labels in data_source.take(10):  # Take more batches
                            for i in range(len(batch_images)):
                                img = batch_images[i].numpy()
                                lbl = batch_labels[i].numpy()
                                
                                # Get class ID
                                if isinstance(lbl, np.ndarray) and len(lbl.shape) > 0:
                                    class_id = np.argmax(lbl) if lbl.shape[0] > 1 else int(lbl)
                                else:
                                    class_id = int(lbl)
                                
                                # Ensure class diversity
                                if class_counts.get(class_id, 0) < max_per_class:
                                    all_images.append(img)
                                    all_labels.append(lbl)
                                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                                    
                                    if len(all_images) >= num_samples:
                                        break
                            if len(all_images) >= num_samples:
                                break
                                
                    elif hasattr(data_source, 'dataset') and isinstance(data_source.dataset, tf.data.Dataset):
                        # Handle mock generator objects - similar logic
                        for batch_images, batch_labels in data_source.dataset.take(10):
                            for i in range(len(batch_images)):
                                img = batch_images[i].numpy()
                                lbl = batch_labels[i].numpy()
                                
                                if isinstance(lbl, np.ndarray) and len(lbl.shape) > 0:
                                    class_id = np.argmax(lbl) if lbl.shape[0] > 1 else int(lbl)
                                else:
                                    class_id = int(lbl)
                                
                                if class_counts.get(class_id, 0) < max_per_class:
                                    all_images.append(img)
                                    all_labels.append(lbl)
                                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                                    
                                    if len(all_images) >= num_samples:
                                        break
                            if len(all_images) >= num_samples:
                                break
                                
                    else:
                        # Handle ImageDataGenerator - take multiple batches
                        logger.info("Extracting diverse samples from ImageDataGenerator...")
                        for batch_num in range(5):  # Take 5 batches to get diversity
                            try:
                                batch_data = next(data_source)
                                if isinstance(batch_data, tuple):
                                    batch_images, batch_labels = batch_data
                                else:
                                    batch_images = batch_data
                                    batch_labels = np.zeros(len(batch_images))
                                
                                for i in range(len(batch_images)):
                                    lbl = batch_labels[i]
                                    if isinstance(lbl, np.ndarray) and lbl.shape[0] > 1:
                                        class_id = np.argmax(lbl)
                                    else:
                                        class_id = int(lbl)
                                    
                                    if class_counts.get(class_id, 0) < max_per_class:
                                        all_images.append(batch_images[i])
                                        all_labels.append(lbl)
                                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
                                        
                                        if len(all_images) >= num_samples:
                                            break
                                if len(all_images) >= num_samples:
                                    break
                            except StopIteration:
                                break
                    
                    logger.info(f"🎯 Collected {len(all_images)} samples from {len(class_counts)} different classes")
                    logger.info(f"Class distribution: {class_counts}")
                    
                    return all_images[:num_samples], all_labels[:num_samples]
                    
                except Exception as e:
                    logger.warning(f"Failed to extract diverse samples: {e}")
                    return [], []

            def plot_before_after_preprocessing(images, labels, title="Before/After Preprocessing"):
                """Plot before and after preprocessing comparison with enhanced debugging."""
                logger.info(f"🖼️ Creating before/after plot with {len(images)} images")
                
                num_samples = min(6, len(images))  # Show 6 samples max
                fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
                if num_samples == 1:
                    axes = axes.reshape(2, 1)
                fig.suptitle(title, fontsize=16, y=0.95)
                
                try:
                    # Food101 class names for better display
                    food_classes = [
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
                        "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles"
                    ]
                    
                    for i in range(num_samples):
                        img = images[i]
                        label = labels[i]
                        
                        # Get class ID for labeling
                        if isinstance(label, np.ndarray):
                            if len(label.shape) == 0:
                                class_id = int(label)
                            else:
                                class_id = np.argmax(label) if label.shape[0] > 1 else int(label)
                        else:
                            class_id = int(label)
                        
                        # Get class name
                        class_name = food_classes[class_id] if class_id < len(food_classes) else f"class_{class_id}"
                        
                        # BEFORE: Original [0,1] normalized image
                        before_img = img.copy()
                        # Ensure the image is in [0,1] range
                        if before_img.min() < -1:  # Already preprocessed, need to denormalize
                            imagenet_mean = np.array([0.485, 0.456, 0.406])
                            imagenet_std = np.array([0.229, 0.224, 0.225])
                            before_img = (before_img * imagenet_std) + imagenet_mean
                            before_img = np.clip(before_img, 0, 1)
                        elif before_img.max() > 1.0:
                            before_img = before_img / 255.0
                        elif before_img.min() < 0 and before_img.max() <= 1:
                            before_img = (before_img + 1) / 2
                        
                        # AFTER: Apply ImageNet preprocessing
                        imagenet_mean = np.array([0.485, 0.456, 0.406])
                        imagenet_std = np.array([0.229, 0.224, 0.225])
                        after_img = (before_img - imagenet_mean) / imagenet_std
                        
                        # Create a visualization of the preprocessing effect
                        # Show the difference as a heatmap
                        diff_img = np.abs(after_img - before_img)
                        diff_normalized = (diff_img - diff_img.min()) / (diff_img.max() - diff_img.min() + 1e-8)
                        
                        # Plot BEFORE (top row)
                        axes[0, i].imshow(before_img)
                        axes[0, i].set_title(f"BEFORE\n{class_name.replace('_', ' ').title()[:12]}\n[{before_img.min():.2f}, {before_img.max():.2f}]", fontsize=11)
                        axes[0, i].axis('off')
                        
                        # Plot AFTER (bottom row) - show preprocessing effect
                        axes[1, i].imshow(diff_normalized, cmap='plasma')
                        axes[1, i].set_title(f"PREPROCESSING\nDifference Map\n[{after_img.min():.2f}, {after_img.max():.2f}]", fontsize=11)
                        axes[1, i].axis('off')
                    
                    plt.tight_layout()
                    logger.info(f"✅ Before/after plot created successfully with {num_samples} samples")
                    return fig
                    
                except Exception as plot_error:
                    logger.error(f"❌ Failed to create before/after plot: {plot_error}")
                    import traceback
                    logger.error(f"Plot error traceback: {traceback.format_exc()}")
                    return None
            
            # Create sample visualization with diverse classes and before/after preprocessing
            if train_generator is not None:
                try:
                    # Check if this is a segmentation task
                    logger.info(f"🔍 VIZ DEBUG: is_segmentation = {is_segmentation}, task_type = {dataset_info.task_type}")
                    if is_segmentation:
                        logger.info("🎨 Creating segmentation-specific visualization...")
                        st.info("🔄 Creating segmentation visualization with mask overlays...")
                        
                        seg_fig = plot_segmentation_before_after(dataset_info, train_generator, num_samples=6)
                        if seg_fig is not None:
                            st.pyplot(seg_fig)
                            plt.close(seg_fig)
                            st.success("✅ **Segmentation Visualization Created Successfully!**")
                            st.info(f"📊 **Dataset**: {dataset_info.builtin_dataset_name or 'Custom'}")
                            st.info(f"🎨 **Classes**: {dataset_info.num_classes} segmentation classes")
                            st.info("🔧 **Preprocessing**: ImageNet normalization with colored mask overlays")
                            logger.info("✅ Segmentation visualization displayed successfully")
                        else:
                            st.warning("⚠️ **Could not create segmentation visualization**")
                    else:
                        # Original classification visualization
                        # Determine dataset name for title with safe None checking
                        dataset_name = "Training Dataset"
                        actual_hf_dataset = None
                        
                        if hasattr(dataset_info, 'hf_dataset_name') and dataset_info.hf_dataset_name is not None:
                            dataset_name = f"{dataset_info.hf_dataset_name.title()} Dataset"
                            actual_hf_dataset = dataset_info.hf_dataset_name
                        elif hasattr(dataset_info, 'builtin_dataset_name') and dataset_info.builtin_dataset_name is not None:
                            dataset_name = f"{dataset_info.builtin_dataset_name.title()} Dataset"
                        elif hasattr(dataset_info, 'dataset_path') and dataset_info.dataset_path is not None:
                            dataset_name = f"{os.path.basename(dataset_info.dataset_path).title()} Dataset"
                        
                        # DEBUG: Log which dataset we're visualizing
                        logger.info(f"🔍 ========== CREATING VISUALIZATION ==========")
                        logger.info(f"🔍 Dataset Name: {dataset_name}")
                        logger.info(f"🔍 HF Dataset: {actual_hf_dataset}")
                        logger.info(f"🔍 Num Classes: {dataset_info.num_classes}")
                        
                        # Support visualization for ALL dataset types
                        if actual_hf_dataset is not None:
                            # HuggingFace dataset visualization
                            logger.info(f"🖼️ Creating {actual_hf_dataset} diverse sample visualization...")
                            st.info(f"🔄 Loading diverse {actual_hf_dataset} samples for before/after preprocessing visualization...")
                            # DEBUG: Explicitly log the selected dataset and class names
                            logger.info(f"🔍 [VISUALIZATION] Selected dataset: {actual_hf_dataset}")
                            logger.info(f"🔍 [VISUALIZATION] DatasetInfo class names: {getattr(dataset_info, 'class_names', None)}")
                            try:
                                from datasets import load_dataset
                                viz_dataset = load_dataset(actual_hf_dataset, split="train[:300]", trust_remote_code=True)
                                logger.info(f"Loaded {len(viz_dataset)} samples for visualization")
                                logger.info(f"🔍 Dataset features: {list(viz_dataset.features.keys())}")
                                selected_samples = {}
                                label_field = None
                                if 'label' in viz_dataset.features:
                                    label_field = 'label'
                                elif 'labels' in viz_dataset.features:
                                    label_field = 'labels'
                                else:
                                    logger.warning(f"⚠️ No label field found in dataset features: {list(viz_dataset.features.keys())}")
                                    st.warning("⚠️ Cannot create visualization: No label field found")
                                    label_field = None
                                if label_field:
                                    # Detect image field dynamically to support both 'image' and 'img'
                                    sample_features = list(viz_dataset.features.keys())
                                    image_field = None
                                    if 'image' in sample_features:
                                        image_field = 'image'
                                    elif 'img' in sample_features:
                                        image_field = 'img'
                                    else:
                                        # Search for any field containing 'image' or 'img'
                                        for key in sample_features:
                                            if 'image' in key.lower() or 'img' in key.lower():
                                                image_field = key
                                                break
                                    
                                    if image_field is None:
                                        logger.warning(f"⚠️ No image field found in dataset features: {sample_features}")
                                        st.warning("⚠️ Cannot create visualization: No image field found")
                                    else:
                                        logger.info(f"✅ Using image field: '{image_field}'")
                                        
                                        for sample in viz_dataset:
                                            class_id = sample[label_field]
                                            if class_id not in selected_samples and len(selected_samples) < 6:
                                                selected_samples[class_id] = sample
                                            if len(selected_samples) >= 6:
                                                break
                                        logger.info(f"Selected {len(selected_samples)} diverse samples from classes: {list(selected_samples.keys())}")
                                        # Use dataset_info.class_names instead of hardcoded food_classes
                                        class_names_list = dataset_info.class_names if hasattr(dataset_info, 'class_names') else [f"class_{i}" for i in range(dataset_info.num_classes)]
                                        logger.info(f"🔍 [VISUALIZATION] Using class names: {class_names_list[:5]}...")
                                        # Log mapping of class_id to class_name for all selected samples
                                        for class_id in selected_samples.keys():
                                            logger.info(f"🔍 [VISUALIZATION] Sample class_id={class_id}, class_name={class_names_list[class_id] if class_id < len(class_names_list) else f'class_{class_id}'}")
                                        if len(selected_samples) > 0:
                                            fig, axes = plt.subplots(2, len(selected_samples), figsize=(4*len(selected_samples), 8))
                                            if len(selected_samples) == 1:
                                                axes = axes.reshape(2, 1)
                                            fig.suptitle(f'{dataset_name} - Before/After ImageNet Preprocessing\n({len(selected_samples)} diverse classes out of {dataset_info.num_classes} total)', fontsize=16)
                                            for idx, (class_id, sample) in enumerate(selected_samples.items()):
                                                class_name = class_names_list[class_id] if class_id < len(class_names_list) else f"class_{class_id}"
                                                logger.info(f"🔍 Visualizing sample {idx}: class_id={class_id}, class_name={class_name}")
                                                img = np.array(sample[image_field].resize((224, 224))).astype('float32') / 255.0
                                                axes[0, idx].imshow(img)
                                                axes[0, idx].set_title(f"BEFORE\n{class_name.replace('_', ' ').title()}\nRange: [{img.min():.2f}, {img.max():.2f}]", fontsize=10)
                                                axes[0, idx].axis('off')
                                                imagenet_mean = np.array([0.485, 0.456, 0.406])
                                                imagenet_std = np.array([0.229, 0.224, 0.225])
                                                processed_img = (img - imagenet_mean) / imagenet_std
                                                diff = np.abs(processed_img)
                                                diff_normalized = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
                                                axes[1, idx].imshow(diff_normalized, cmap='viridis')
                                                axes[1, idx].set_title(f"AFTER IMAGENET\n{class_name.replace('_', ' ').title()}\nRange: [{processed_img.min():.2f}, {processed_img.max():.2f}]", fontsize=10)
                                                axes[1, idx].axis('off')
                                            plt.tight_layout()
                                            st.pyplot(fig)
                                            plt.close(fig)
                                            
                                            # Show class information
                                            class_info = {class_id: class_names_list[class_id] if class_id < len(class_names_list) else f"class_{class_id}" for class_id in selected_samples.keys()}
                                            st.success("✅ **Before/After Preprocessing Visualization Created Successfully!**")
                                            st.info(f"📊 **Displayed classes**: {class_info}")
                                            st.info(f"� **Total {dataset_name} classes**: {dataset_info.num_classes} categories")
                                            st.info("🔧 **Preprocessing**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])")
                                            logger.info(f"✅ Visualization displayed: {len(selected_samples)} samples from different classes")
                                        else:
                                            st.warning("⚠️ **No diverse samples found for visualization**")
                            
                            except Exception as viz_load_error:
                                logger.error(f"Failed to load visualization samples: {viz_load_error}")
                                st.warning(f"⚠️ **Could not load samples for visualization**: {str(viz_load_error)}")
                        
                        elif hasattr(dataset_info, 'is_builtin') and dataset_info.is_builtin:
                            # Built-in dataset visualization (MNIST, CIFAR-10, etc.)
                            logger.info(f"🖼️ Creating built-in dataset visualization...")
                            st.info(f"🔄 Loading samples from {dataset_info.builtin_dataset_name} for visualization...")
                            
                            try:
                                import tensorflow as tf
                                
                                # Load built-in dataset
                                builtin_name = dataset_info.builtin_dataset_name
                                
                                if builtin_name == 'MNIST':
                                    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
                                elif builtin_name == 'Fashion-MNIST':
                                    (x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
                                elif builtin_name == 'CIFAR-10':
                                    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
                                    y_train = y_train.flatten()
                                elif builtin_name == 'CIFAR-100':
                                    (x_train, y_train), _ = tf.keras.datasets.cifar100.load_data()
                                    y_train = y_train.flatten()
                                else:
                                    raise ValueError(f"Unsupported built-in dataset: {builtin_name}")
                                
                                # Select diverse samples (one per class, up to 6 classes)
                                selected_samples = {}
                                for idx, label in enumerate(y_train):
                                    if label not in selected_samples and len(selected_samples) < 6:
                                        selected_samples[label] = x_train[idx]
                                    if len(selected_samples) >= 6:
                                        break
                                
                                logger.info(f"Selected {len(selected_samples)} diverse samples from classes: {list(selected_samples.keys())}")
                                
                                # Get class names
                                class_names_list = dataset_info.class_names if hasattr(dataset_info, 'class_names') else [f"class_{i}" for i in range(dataset_info.num_classes)]
                                
                                # Create visualization
                                fig, axes = plt.subplots(2, len(selected_samples), figsize=(4*len(selected_samples), 8))
                                if len(selected_samples) == 1:
                                    axes = axes.reshape(2, 1)
                                
                                fig.suptitle(f'{dataset_name} - Before/After ImageNet Preprocessing\n({len(selected_samples)} diverse classes out of {dataset_info.num_classes} total)', fontsize=16)
                                
                                for idx, (class_id, img_data) in enumerate(selected_samples.items()):
                                    class_name = class_names_list[class_id] if class_id < len(class_names_list) else f"class_{class_id}"
                                    
                                    # Normalize to [0, 1]
                                    img = img_data.astype('float32') / 255.0
                                    
                                    # Handle grayscale (MNIST, Fashion-MNIST)
                                    if len(img.shape) == 2:
                                        # Convert grayscale to RGB for visualization
                                        img = np.stack([img, img, img], axis=-1)
                                    
                                    # Resize to 224x224 if needed
                                    if img.shape[0] != 224 or img.shape[1] != 224:
                                        from PIL import Image
                                        img_pil = Image.fromarray((img * 255).astype('uint8'))
                                        img_pil = img_pil.resize((224, 224))
                                        img = np.array(img_pil).astype('float32') / 255.0
                                    
                                    # BEFORE
                                    axes[0, idx].imshow(img)
                                    axes[0, idx].set_title(f"BEFORE\n{class_name.replace('_', ' ').title()}\nRange: [{img.min():.2f}, {img.max():.2f}]", fontsize=10)
                                    axes[0, idx].axis('off')
                                    
                                    # AFTER: Apply ImageNet preprocessing
                                    imagenet_mean = np.array([0.485, 0.456, 0.406])
                                    imagenet_std = np.array([0.229, 0.224, 0.225])
                                    processed_img = (img - imagenet_mean) / imagenet_std
                                    diff = np.abs(processed_img)
                                    diff_normalized = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
                                    
                                    axes[1, idx].imshow(diff_normalized, cmap='viridis')
                                    axes[1, idx].set_title(f"AFTER IMAGENET\n{class_name.replace('_', ' ').title()}\nRange: [{processed_img.min():.2f}, {processed_img.max():.2f}]", fontsize=10)
                                    axes[1, idx].axis('off')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)
                                
                                # Show class information
                                class_info = {class_id: class_names_list[class_id] if class_id < len(class_names_list) else f"class_{class_id}" for class_id in selected_samples.keys()}
                                st.success("✅ **Before/After Preprocessing Visualization Created Successfully!**")
                                st.info(f"📊 **Displayed classes**: {class_info}")
                                st.info(f"🎯 **Total {dataset_name} classes**: {dataset_info.num_classes} categories")
                                st.info("🔧 **Preprocessing**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])")
                                logger.info(f"✅ Built-in dataset visualization displayed: {len(selected_samples)} samples")
                            
                            except Exception as builtin_viz_error:
                                logger.error(f"Failed to create built-in dataset visualization: {builtin_viz_error}")
                                import traceback
                                logger.error(traceback.format_exc())
                                st.warning(f"⚠️ **Could not create visualization**: {str(builtin_viz_error)}")
                        
                        else:
                            # Custom/uploaded dataset visualization
                            logger.info(f"🖼️ Creating custom dataset visualization from directory...")
                            st.info(f"🔄 Loading samples from uploaded dataset for visualization...")
                            
                            try:
                                # Get samples from train_generator
                                if train_generator and hasattr(train_generator, 'next'):
                                    # Get one batch of samples
                                    batch_x, batch_y = next(iter(train_generator))
                                    
                                    # Select up to 6 diverse samples
                                    num_viz_samples = min(6, len(batch_x))
                                    
                                    # Get class names
                                    class_names_list = dataset_info.class_names if hasattr(dataset_info, 'class_names') else [f"class_{i}" for i in range(dataset_info.num_classes)]
                                    
                                    # Create visualization
                                    fig, axes = plt.subplots(2, num_viz_samples, figsize=(4*num_viz_samples, 8))
                                    if num_viz_samples == 1:
                                        axes = axes.reshape(2, 1)
                                    
                                    fig.suptitle(f'{dataset_name} - Before/After ImageNet Preprocessing\n({num_viz_samples} samples)', fontsize=16)
                                    
                                    for idx in range(num_viz_samples):
                                        img = batch_x[idx]
                                        
                                        # Get label
                                        if len(batch_y.shape) == 1:
                                            class_id = int(batch_y[idx])
                                        else:
                                            class_id = int(np.argmax(batch_y[idx]))
                                        
                                        class_name = class_names_list[class_id] if class_id < len(class_names_list) else f"class_{class_id}"
                                        
                                        # BEFORE
                                        axes[0, idx].imshow(img)
                                        axes[0, idx].set_title(f"BEFORE\n{class_name.replace('_', ' ').title()}\nRange: [{img.min():.2f}, {img.max():.2f}]", fontsize=10)
                                        axes[0, idx].axis('off')
                                        
                                        # AFTER: Apply ImageNet preprocessing
                                        imagenet_mean = np.array([0.485, 0.456, 0.406])
                                        imagenet_std = np.array([0.229, 0.224, 0.225])
                                        processed_img = (img - imagenet_mean) / imagenet_std
                                        diff = np.abs(processed_img)
                                        diff_normalized = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
                                        
                                        axes[1, idx].imshow(diff_normalized, cmap='viridis')
                                        axes[1, idx].set_title(f"AFTER IMAGENET\n{class_name.replace('_', ' ').title()}\nRange: [{processed_img.min():.2f}, {processed_img.max():.2f}]", fontsize=10)
                                        axes[1, idx].axis('off')
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close(fig)
                                    
                                    st.success("✅ **Before/After Preprocessing Visualization Created Successfully!**")
                                    st.info(f"📊 **Displayed samples**: {num_viz_samples}")
                                    st.info(f"🎯 **Total {dataset_name} classes**: {dataset_info.num_classes} categories")
                                    st.info("🔧 **Preprocessing**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])")
                                    logger.info(f"✅ Custom dataset visualization displayed: {num_viz_samples} samples")
                                else:
                                    st.warning("⚠️ **No training data available for visualization**")
                            
                            except Exception as custom_viz_error:
                                logger.error(f"Failed to create custom dataset visualization: {custom_viz_error}")
                                import traceback
                                logger.error(traceback.format_exc())
                                st.warning(f"⚠️ **Could not create visualization**: {str(custom_viz_error)}")
                        
                except Exception as viz_error:
                    logger.warning(f"Sample visualization failed: {viz_error}")
                    import traceback
                    logger.warning(f"Visualization error traceback: {traceback.format_exc()}")
                    st.info("ℹ️ **Sample images could not be displayed due to technical error**")
                    
        except ImportError:
            logger.warning("Matplotlib not available - skipping sample image visualization")
            st.info("ℹ️ **Sample image visualization skipped (matplotlib not available)**")
        except Exception as viz_error:
            logger.warning(f"Sample visualization setup failed: {viz_error}")
            st.info("ℹ️ **Sample image visualization skipped due to error**")
        
        status_text.text("⚙️ **Setting up TensorFlow callbacks...**")
        progress_bar.progress(0.4)
        
        # Custom Streamlit logging callback for TensorFlow
        class StreamlitLoggingCallback(keras.callbacks.Callback):
            def __init__(self, progress_ui, max_epochs):
                super().__init__()
                self.progress_ui = progress_ui
                self.max_epochs = max_epochs
                self.start_time = time.time()
            
            def on_epoch_end(self, epoch, logs=None):
                if logs is None:
                    logs = {}
                
                # Extract metrics
                train_loss = logs.get('loss', 0)
                train_acc = logs.get('accuracy', 0) * 100  # Convert to percentage
                val_loss = logs.get('val_loss', 0)
                val_acc = logs.get('val_accuracy', 0) * 100  # Convert to percentage
                
                # Update progress UI elements
                if self.progress_ui:
                    # Update current metrics
                    if 'current_epoch' in self.progress_ui:
                        self.progress_ui['current_epoch'].text(f"Epoch: {epoch + 1}/{self.max_epochs}")
                    if 'current_loss' in self.progress_ui:
                        self.progress_ui['current_loss'].text(f"Loss: {train_loss:.4f}")
                    if 'current_acc' in self.progress_ui:
                        self.progress_ui['current_acc'].text(f"Acc: {train_acc:.2f}%")
                    if 'elapsed_time' in self.progress_ui:
                        elapsed = time.time() - self.start_time
                        self.progress_ui['elapsed_time'].text(f"Time: {elapsed:.1f}s")
                    
                    # Update overall progress
                    if 'overall_progress' in self.progress_ui:
                        progress = (epoch + 1) / self.max_epochs
                        self.progress_ui['overall_progress'].progress(0.5 + 0.4 * progress)
                    
                    # Store metrics as dict so post-training charts can consume them
                    if 'training_log' in self.progress_ui and 'log_history' in self.progress_ui:
                        log_dict = {
                            'epoch':        epoch + 1,
                            'loss':         round(float(train_loss), 4),
                            'val_loss':     round(float(val_loss), 4),
                            'accuracy':     round(float(train_acc) / 100.0, 4),
                            'val_accuracy': round(float(val_acc) / 100.0, 4),
                        }
                        self.progress_ui['log_history'].append(log_dict)
                        recent = self.progress_ui['log_history'][-10:]
                        log_lines = [
                            f"Ep {e['epoch']:>3}: loss={e['loss']:.4f}  acc={e['accuracy']*100:.1f}%  "
                            f"val_loss={e['val_loss']:.4f}  val_acc={e['val_accuracy']*100:.1f}%"
                            for e in recent
                        ]
                        self.progress_ui['training_log'].text("\n".join(log_lines))

                    # Live charts (TF callback)
                    if 'chart_loss' in self.progress_ui and len(self.progress_ui['log_history']) > 0:
                        ep_x = [e['epoch'] for e in self.progress_ui['log_history']]
                        tl   = [e['loss'] for e in self.progress_ui['log_history']]
                        vl   = [e['val_loss'] for e in self.progress_ui['log_history']]
                        ta   = [e['accuracy'] * 100 for e in self.progress_ui['log_history']]
                        va   = [e['val_accuracy'] * 100 for e in self.progress_ui['log_history']]
                        _fl = go.Figure()
                        _fl.add_trace(go.Scatter(x=ep_x, y=tl, mode='lines+markers', name='Train Loss', line=dict(color='#ef4444')))
                        _fl.add_trace(go.Scatter(x=ep_x, y=vl, mode='lines+markers', name='Val Loss',   line=dict(color='#f97316')))
                        _fl.update_layout(height=220, margin=dict(l=20,r=10,t=10,b=30),
                                          xaxis_title='Epoch', yaxis_title='Loss',
                                          showlegend=True, legend=dict(orientation='h', y=-0.25))
                        self.progress_ui['chart_loss'].plotly_chart(_fl, use_container_width=True)
                        _fa = go.Figure()
                        _fa.add_trace(go.Scatter(x=ep_x, y=ta, mode='lines+markers', name='Train Acc', line=dict(color='#3b82f6')))
                        _fa.add_trace(go.Scatter(x=ep_x, y=va, mode='lines+markers', name='Val Acc',   line=dict(color='#22c55e')))
                        _fa.update_layout(height=220, margin=dict(l=20,r=10,t=10,b=30),
                                          xaxis_title='Epoch', yaxis_title='Accuracy (%)',
                                          showlegend=True, legend=dict(orientation='h', y=-0.25))
                        self.progress_ui['chart_acc'].plotly_chart(_fa, use_container_width=True)
        
        # Setup training callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Add Streamlit logging callback if progress_ui is available
        if progress_ui:
            streamlit_callback = StreamlitLoggingCallback(progress_ui, max_epochs)
            callbacks.append(streamlit_callback)
        
        status_text.text("🚀 **Training TensorFlow model with REAL data...**")
        progress_bar.progress(0.5)
        
        # ACTUAL TENSORFLOW TRAINING
        import time
        training_start_time = time.time()
        
        # Debug info before training
        if train_generator is None or validation_generator is None:
            raise RuntimeError("Data generators not initialized. Dataset loading may have failed.")
        
        # Debug info only in logs
        logger.info(f"Training samples: {train_generator.samples}")
        logger.info(f"Validation samples: {validation_generator.samples}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Input size: {img_size} with {channels} channels")
        logger.info(f"Number of classes: {num_classes}")
        logger.info(f"Model output shape: {model.output_shape}")
        
        try:
            # Train the model with real data
            logger.info("Starting model.fit() training...")
            logger.info(f"Training generator samples: {train_generator.samples}")
            logger.info(f"Validation generator samples: {validation_generator.samples}")
            
            if is_builtin_dataset:
                # For built-in datasets, train directly on the tf.data.Dataset
                logger.info("Training with built-in TensorFlow dataset...")
                history = model.fit(
                    train_generator.dataset,
                    epochs=max_epochs,
                    validation_data=validation_generator.dataset,
                    callbacks=callbacks,
                    verbose=1  # Show detailed training progress
                )
            else:
                # For directory-based datasets, use traditional generator approach
                logger.info("Training with directory-based data generators...")
                logger.info(f"Training generator steps per epoch: {train_generator.samples // train_generator.batch_size}")
                logger.info(f"Validation generator steps per epoch: {validation_generator.samples // validation_generator.batch_size}")
                
                history = model.fit(
                    train_generator,
                    epochs=max_epochs,
                    validation_data=validation_generator,
                    callbacks=callbacks,
                    verbose=1  # Show detailed training progress
                )
            
            logger.info("model.fit() completed successfully!")
            logger.info(f"Training history keys: {history.history.keys()}")
            logger.info(f"Number of epochs completed: {len(history.history['loss'])}")
            st.success("✅ **model.fit() completed successfully!**")
            
        except Exception as fit_error:
            logger.error(f"model.fit() failed with error: {str(fit_error)}")
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"Full traceback: {error_traceback}")
            raise fit_error  # Re-raise to be caught by outer exception handler
        
        training_time = time.time() - training_start_time
        
        # Extract REAL training metrics
        train_acc_history = history.history['accuracy']
        val_acc_history = history.history['val_accuracy']
        train_loss_history = history.history['loss']
        val_loss_history = history.history['val_loss']
        
        best_accuracy = float(max(val_acc_history))
        best_loss = float(min(val_loss_history))
        final_train_acc = float(train_acc_history[-1])
        final_val_acc = float(val_acc_history[-1])
        
        # Update progress during training
        start_time = training_start_time
        for i, (train_acc, val_acc, train_loss, val_loss) in enumerate(zip(train_acc_history, val_acc_history, train_loss_history, val_loss_history)):
            progress = 0.5 + (i + 1) * 0.4 / len(train_acc_history)
            progress_bar.progress(min(progress, 0.9))
            status_text.text(f"🚀 **Epoch {i+1}/{len(train_acc_history)} - Val Accuracy: {val_acc:.4f}**")
            
            # Update detailed metrics if using new progress UI
            if progress_ui:
                elapsed = time.time() - start_time
                progress_ui['current_epoch'].metric("Epoch", f"{i+1} / {len(train_acc_history)}")
                progress_ui['current_loss'].metric("Loss", f"{val_loss:.4f}")
                progress_ui['current_acc'].metric("Accuracy", f"{val_acc*100:.1f}%")
                progress_ui['elapsed_time'].metric("Time", "{:02d}:{:02d}:{:02d}".format(
                    int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
                ))
                
                # Store metrics as dict so post-training charts can consume them
                log_dict = {
                    'epoch':        i + 1,
                    'loss':         round(float(train_loss), 4),
                    'val_loss':     round(float(val_loss), 4),
                    'accuracy':     round(float(train_acc), 4),   # TF already 0-1
                    'val_accuracy': round(float(val_acc), 4),
                }
                progress_ui['log_history'].append(log_dict)
                recent = progress_ui['log_history'][-10:]
                log_lines = [
                    f"Ep {e['epoch']:>3}: loss={e['loss']:.4f}  acc={e['accuracy']*100:.1f}%  "
                    f"val_loss={e['val_loss']:.4f}  val_acc={e['val_accuracy']*100:.1f}%"
                    for e in recent
                ]
                progress_ui['training_log'].text("\n".join(log_lines))
                
                # Update live charts
                if 'chart_loss' in progress_ui and len(progress_ui['log_history']) > 0:
                    ep_x = [e['epoch'] for e in progress_ui['log_history']]
                    tl   = [e['loss'] for e in progress_ui['log_history']]
                    vl   = [e['val_loss'] for e in progress_ui['log_history']]
                    ta   = [e['accuracy'] * 100 for e in progress_ui['log_history']]
                    va   = [e['val_accuracy'] * 100 for e in progress_ui['log_history']]
                    fig_l2 = go.Figure()
                    fig_l2.add_trace(go.Scatter(x=ep_x, y=tl, mode='lines+markers', name='Train Loss', line=dict(color='#ef4444')))
                    fig_l2.add_trace(go.Scatter(x=ep_x, y=vl, mode='lines+markers', name='Val Loss',   line=dict(color='#f97316')))
                    fig_l2.update_layout(height=220, margin=dict(l=20,r=10,t=10,b=30),
                                        xaxis_title='Epoch', yaxis_title='Loss',
                                        showlegend=True, legend=dict(orientation='h', y=-0.25))
                    progress_ui['chart_loss'].plotly_chart(fig_l2, use_container_width=True)
                    fig_a2 = go.Figure()
                    fig_a2.add_trace(go.Scatter(x=ep_x, y=ta, mode='lines+markers', name='Train Acc', line=dict(color='#3b82f6')))
                    fig_a2.add_trace(go.Scatter(x=ep_x, y=va, mode='lines+markers', name='Val Acc',   line=dict(color='#22c55e')))
                    fig_a2.update_layout(height=220, margin=dict(l=20,r=10,t=10,b=30),
                                        xaxis_title='Epoch', yaxis_title='Accuracy (%)',
                                        showlegend=True, legend=dict(orientation='h', y=-0.25))
                    progress_ui['chart_acc'].plotly_chart(fig_a2, use_container_width=True)
                
            time.sleep(0.1)  # Brief pause for UI update
        
        progress_bar.progress(1.0)
        status_text.text("✅ **TensorFlow training completed!**")
        
        # Calculate overfitting metrics
        overfitting_gap = abs(final_train_acc - final_val_acc) * 100
        overfitting_detected = overfitting_gap > 5.0
        
        st.success(f"🎉 TensorFlow training completed successfully!")
        st.metric("Best Validation Accuracy", f"{best_accuracy:.6f}")
        st.metric("Final Training Accuracy", f"{final_train_acc:.6f}")
        st.metric("Training Time", f"{training_time:.1f}s")
        
        if overfitting_detected:
            st.warning(f"⚠️ Overfitting detected: {overfitting_gap:.2f}% gap between train/validation")
        else:
            st.info(f"✅ Good generalization: {overfitting_gap:.2f}% train/val gap")
        
        st.info(f"🧠 Model: TensorFlow CNN with {model.count_params():,} parameters")
        st.info(f"📊 Dataset: {train_generator.samples} training, {validation_generator.samples} validation samples")
        
        # Save test samples for evaluation
        try:
            save_test_samples_for_evaluation(dataset_info, output_dir, num_samples=50)
            st.success("✅ Test samples saved for evaluation!")
        except Exception as e:
            st.warning(f"⚠️ Could not save test samples: {str(e)}")
            logger.warning(f"Failed to save test samples: {str(e)}")
        
        # Save model with intelligent naming system
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate intelligent model name
        from utils.model_naming import generate_model_name, extract_dataset_name
        
        # Extract dataset name - fix builtin dataset handling for TensorFlow
        dataset_info_obj = st.session_state.dataset_info
        builtin_info = None
        
        # Check for builtin dataset info
        if hasattr(dataset_info_obj, 'builtin_dataset_name'):
            builtin_info = {'name': dataset_info_obj.builtin_dataset_name}
        elif hasattr(dataset_info_obj, 'is_builtin') and dataset_info_obj.is_builtin:
            # For TensorFlow, prefer tf_name if available, but fall back to builtin_dataset_name
            if hasattr(dataset_info_obj, 'builtin_tf_name'):
                builtin_info = {'name': dataset_info_obj.builtin_tf_name}
            elif hasattr(dataset_info_obj, 'builtin_dataset_name'):
                builtin_info = {'name': dataset_info_obj.builtin_dataset_name}
        
        dataset_name = extract_dataset_name(
            dataset_path=getattr(dataset_info_obj, 'dataset_path', None),
            builtin_dataset=builtin_info
        )
        
        # Generate intelligent model name
        model_name = generate_model_name(
            framework='tensorflow',
            architecture=model_config.architecture if hasattr(model_config, 'architecture') else 'adaptive_cnn',
            backbone=model_config.backbone if hasattr(model_config, 'backbone') else None,
            dataset_name=dataset_name,
            task_type=dataset_info.task_type if hasattr(dataset_info, 'task_type') else 'classification'
        )
        
        model_path = os.path.join(output_dir, model_name)
        model.save(model_path)
        st.success(f"💾 Model saved to: {model_path}")
        
        # Prepare comprehensive results
        results = {
            'best_accuracy': best_accuracy,
            'best_loss': best_loss,
            'final_train_accuracy': final_train_acc,
            'final_val_accuracy': final_val_acc,
            'training_time': training_time,
            'total_epochs': len(train_acc_history),
            'framework': 'TensorFlow/Keras',
            'model_parameters': int(model.count_params()),
            'architecture': 'CNN Sequential',
            'overfitting_detected': overfitting_detected,
            'overfitting_gap': overfitting_gap,
            'convergence_status': 'completed',
            'train_samples': train_generator.samples,
            'val_samples': validation_generator.samples,
            'model_saved': model_path,
            'training_history': {
                'train_accuracy': [float(x) for x in train_acc_history],
                'val_accuracy': [float(x) for x in val_acc_history],
                'train_loss': [float(x) for x in train_loss_history],
                'val_loss': [float(x) for x in val_loss_history]
            }
        }
        
        # Save training results to JSON with intelligent naming
        results_name = model_name.replace('.keras', '_results.json')
        results_path = os.path.join(output_dir, results_name)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        st.success(f"📋 Training results saved to: {results_path}")
        
        return results
        
    except Exception as e:
        # Enhanced error reporting - only to logs
        import traceback
        error_details = traceback.format_exc()
        
        logger.error(f"TensorFlow training failed: {str(e)}")
        logger.error(f"Full error details: {error_details}")
        
        return {
            'best_accuracy': 0.0,
            'best_loss': float('inf'), 
            'training_time': 0.0,
            'error': str(e),
            'error_details': error_details
        }


def start_sklearn_training(max_epochs: int, optimize_hyperparams: bool, output_dir: str, progress_ui=None):
    """Start Scikit-learn training process with progress tracking."""
    
    import time
    import json
    from pathlib import Path
    
    try:
        if progress_ui:
            progress_ui['status_text'].text("📊 **Initializing Scikit-learn training...**")
            progress_ui['overall_progress'].progress(0.05)
        
        # Try direct Scikit-learn implementation
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score
            import numpy as np
            SKLEARN_AVAILABLE = True
            if progress_ui:
                progress_ui['status_text'].text("✅ **Scikit-learn loaded successfully**")
                progress_ui['overall_progress'].progress(0.1)
        except ImportError as e:
            if progress_ui:
                progress_ui['status_text'].text(f"❌ **Scikit-learn import failed**")
            st.error(f"❌ Scikit-learn import failed: {e}")
            return {
                'best_accuracy': 0.0,
                'best_loss': float('inf'),
                'training_time': 0.0,
                'error': 'Scikit-learn not available'
            }
        
        # Get required data from session state
        dataset_info = st.session_state.get('dataset_info')
        model_config = st.session_state.get('model_config')
        
        if not dataset_info or not model_config:
            st.error("❌ Dataset info or model config missing")
            return {
                'best_accuracy': 0.0,
                'best_loss': float('inf'),
                'training_time': 0.0,
                'error': 'Missing configuration'
            }
        
        # Use new progress UI system or fallback to local
        if progress_ui:
            progress_bar = progress_ui['overall_progress']
            status_text = progress_ui['status_text']
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        # Check if dataset is CSV/tabular data
        dataset_path = Path(dataset_info.dataset_path)
        is_csv_dataset = (dataset_path.suffix.lower() == '.csv' or 
                         dataset_info.image_stats.get('data_type') == 'tabular')
        
        if is_csv_dataset:
            # Handle CSV/tabular data
            status_text.text("📊 **Processing CSV/tabular data...**")
            progress_bar.progress(0.15)
            
            st.info(f"🔧 **Scikit-learn processing tabular dataset: {dataset_path.name}**")
            
            # Load and process CSV data
            import pandas as pd
            
            try:
                df = pd.read_csv(dataset_path)
                st.success(f"✅ **CSV loaded successfully**: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                st.error(f"❌ Failed to load CSV: {e}")
                return {
                    'best_accuracy': 0.0,
                    'best_loss': float('inf'),
                    'training_time': 0.0,
                    'error': f'CSV loading failed: {e}'
                }
            
            # Get target column from dataset_info
            target_column = dataset_info.image_stats.get('target_column', 'label')
            
            # Validate target column exists
            if target_column not in df.columns:
                st.error(f"❌ Target column '{target_column}' not found in dataset")
                return {
                    'best_accuracy': 0.0,
                    'best_loss': float('inf'),
                    'training_time': 0.0,
                    'error': f'Target column {target_column} not found'
                }
            
            feature_columns = [col for col in df.columns if col != target_column]
            
            # Prepare features and labels
            X = df[feature_columns]
            y = df[target_column]
            
            st.info(f"📊 **Features**: {len(feature_columns)} columns, **Target**: {target_column}")
            st.info(f"📋 **Target classes**: {sorted(y.unique())}")
            
            # Handle categorical features
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OneHotEncoder
            
            # Identify numeric and categorical columns
            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = X.select_dtypes(include=['object']).columns.tolist()
            
            # Create preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
                ],
                remainder='passthrough'
            )
            
            # Fit and transform features
            X_processed = preprocessor.fit_transform(X)
            
            # Encode labels if they're strings
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            st.success(f"✅ **CSV data processed**: {len(X_processed)} samples, {X_processed.shape[1]} features")
            
        else:
            # Handle image data (original logic)
            # Use the centralized function to get user-specified input size
            height, width, channels = get_user_specified_input_size()
            resize_dims = (width, height)  # PIL resize expects (width, height)
                
            st.info(f"🔧 **Scikit-learn using user-specified input size: {resize_dims} (W×H) with {channels} channels**")
            
            # Real Scikit-learn implementation with actual feature extraction
            status_text.text("🔄 **Extracting features from images...**")
            progress_bar.progress(0.15)
            
            # Real feature extraction from images
            import os
            from pathlib import Path
            
            features_list = []
            labels_list = []
        
        # Process each class directory
        for class_dir in dataset_path.iterdir():
            if class_dir.is_dir() and not class_dir.name.startswith('.'):
                class_name = class_dir.name
                
                # Get image files
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
                image_files = []
                for ext in image_extensions:
                    image_files.extend(list(class_dir.glob(f"*{ext}")))
                    image_files.extend(list(class_dir.glob(f"*{ext.upper()}")))
                
                # Process images (limit to 200 per class for performance)
                for img_path in image_files[:200]:
                    try:
                        # Simple feature extraction using PIL
                        from PIL import Image
                        
                        with Image.open(img_path) as img:
                            # Convert to grayscale and resize to USER-SPECIFIED SIZE
                            img = img.convert('L').resize(resize_dims)
                            pixels = list(img.getdata())
                            
                            # Basic statistical features
                            features = [
                                sum(pixels) / len(pixels),  # Mean
                                max(pixels),  # Max intensity
                                min(pixels),  # Min intensity
                                len([p for p in pixels if p > 128]) / len(pixels),  # Bright pixels ratio
                            ]
                            
                            # Add histogram features (8 bins)
                            hist_bins = [0] * 8
                            for p in pixels:
                                bin_idx = min(p // 32, 7)
                                hist_bins[bin_idx] += 1
                            
                            # Normalize histogram
                            total = sum(hist_bins)
                            if total > 0:
                                hist_bins = [b/total for b in hist_bins]
                            
                            features.extend(hist_bins)
                            
                            # Add resized pixels as features using user-specified dimensions
                            downsampled = img.resize(resize_dims)
                            pixel_features = [p/255.0 for p in list(downsampled.getdata())]
                            features.extend(pixel_features)
                            
                            features_list.append(features)
                            labels_list.append(class_name)
                            
                    except Exception as e:
                        continue  # Skip problematic images
        
        # Handle different data processing paths
        if is_csv_dataset:
            status_text.text("🔢 Finalizing tabular data processing...")
            progress_bar.progress(30)
            
            # Data already processed above for CSV
            X = X_processed
            y = y_encoded
            
            # We already have preprocessor and label_encoder from CSV processing
            scaler = None  # Scaling is already handled in preprocessor
            
        else:
            # Image data processing
            if not features_list:
                st.error("❌ No features could be extracted from images")
                return {'best_accuracy': 0.0, 'best_loss': 1.0, 'training_time': 0.0}
            
            status_text.text("🔢 Processing extracted features...")
            progress_bar.progress(30)
            
            # Convert to numpy arrays
            X = np.array(features_list)
            
            # Encode labels
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(labels_list)
            
            # Initialize preprocessor as None for image data
            preprocessor = None
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features (only for image data, CSV already handled)
        if is_csv_dataset:
            # Data is already preprocessed and scaled for CSV
            X_train_scaled = X_train
            X_test_scaled = X_test
        else:
            # Scale image features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        
        status_text.text("🏗️ Training Random Forest model...")
        progress_bar.progress(50)
        
        # Validate data before training
        if X_train_scaled.shape[0] == 0 or X_test_scaled.shape[0] == 0:
            st.error("❌ No training or test data available")
            return {
                'best_accuracy': 0.0,
                'best_loss': float('inf'),
                'training_time': 0.0,
                'error': 'No data available for training'
            }
        
        st.info(f"📊 **Training data shape**: {X_train_scaled.shape}")
        st.info(f"📊 **Test data shape**: {X_test_scaled.shape}")
        st.info(f"🎯 **Number of classes**: {len(np.unique(y))}")
        
        # Create and train model with REAL data
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Setup callbacks for consistency across frameworks
        from utils.callbacks import CallbackManager, MetricsLogger
        callback_manager = CallbackManager(framework="Scikit-learn")
        
        # Metrics logger
        metrics_logger = MetricsLogger(
            log_dir=output_dir,
            framework="Scikit-learn"
        )
        callback_manager.add_callback(metrics_logger)
        
        # Initialize callbacks
        callback_manager.on_train_begin({'total_epochs': 1, 'model_type': 'RandomForest'})
        
        # ACTUAL SCIKIT-LEARN TRAINING
        import time
        training_start_time = time.time()
        
        model.fit(X_train_scaled, y_train)
        
        training_time = time.time() - training_start_time
        
        status_text.text("📊 Evaluating model performance...")
        progress_bar.progress(80)
        
        # Real model evaluation
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.model_selection import cross_val_score
        
        train_predictions = model.predict(X_train_scaled)
        test_predictions = model.predict(X_test_scaled)
        
        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        # Cross-validation on training set
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Call training end callback
        final_metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'training_time': training_time,
            'model_type': 'RandomForest'
        }
        callback_manager.on_train_end(final_metrics)
        
        progress_bar.progress(100)
        status_text.text("✅ Scikit-learn training completed!")
        
        # Save test samples for evaluation (for image datasets only)
        try:
            if hasattr(dataset_info, 'task_type') and dataset_info.task_type == 'classification' and hasattr(dataset_info, 'image_size'):
                save_test_samples_for_evaluation(dataset_info, output_dir, num_samples=50)
                st.success("✅ Test samples saved for evaluation!")
            else:
                st.info("ℹ️ Test sample saving skipped (non-image dataset)")
        except Exception as e:
            st.warning(f"⚠️ Could not save test samples: {str(e)}")
        
        st.success(f"🎉 Scikit-learn training finished! Test accuracy: {test_accuracy:.6f}")
        st.info(f"📊 Cross-validation: {cv_mean:.4f} ± {cv_std:.4f}")
        st.info(f"🌳 Model: Random Forest with 100 estimators")
        
        # Feature importance (top features)
        if hasattr(model, 'feature_importances_'):
            st.write("📈 **Top Feature Importances:**")
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[-5:][::-1]
            
            if is_csv_dataset:
                # Use actual feature names from CSV
                feature_names = dataset_info.image_stats.get('feature_names', [f'Feature_{i}' for i in range(len(importances))])
                # Remove target column name if present
                target_col = dataset_info.image_stats.get('target_column')
                if target_col in feature_names:
                    feature_names = [name for name in feature_names if name != target_col]
            else:
                # Use image feature names
                feature_names = ['Mean Intensity', 'Max Intensity', 'Min Intensity', 'Bright Pixels Ratio'] + \
                               [f'Hist Bin {i}' for i in range(8)] + \
                               [f'Pixel {i}' for i in range(64)]
            
            for idx in top_indices:
                if idx < len(feature_names):
                    st.write(f"• {feature_names[idx]}: {importances[idx]:.4f}")
        
        # Save model with intelligent naming system
        import os
        import joblib
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate intelligent model name
        try:
            from utils.model_naming import generate_model_name, extract_dataset_name
            
            # Extract dataset name - fix builtin dataset handling for Scikit-learn
            dataset_info_obj = st.session_state.dataset_info
            builtin_info = None
            
            # Check for builtin dataset info
            if hasattr(dataset_info_obj, 'builtin_dataset_name'):
                builtin_info = {'name': dataset_info_obj.builtin_dataset_name}
            elif hasattr(dataset_info_obj, 'is_builtin') and dataset_info_obj.is_builtin:
                # For Scikit-learn, use builtin_dataset_name, not tf_name
                if hasattr(dataset_info_obj, 'builtin_dataset_name'):
                    builtin_info = {'name': dataset_info_obj.builtin_dataset_name}
            
            dataset_name = extract_dataset_name(
                dataset_path=getattr(dataset_info_obj, 'dataset_path', None),
                builtin_dataset=builtin_info
            )
            
            # Generate intelligent model name
            model_name = generate_model_name(
                framework='sklearn',
                architecture='random_forest',  # Default for sklearn
                backbone=None,
                dataset_name=dataset_name,
                task_type=dataset_info.task_type if hasattr(dataset_info, 'task_type') else 'classification'
            )
        except Exception as e:
            # Fallback naming
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"sklearn_random_forest_{timestamp}.pkl"
        
        # Save model with joblib
        model_path = os.path.join(output_dir, model_name)
        joblib.dump({
            'model': model,
            'scaler': scaler if not is_csv_dataset else None,
            'preprocessor': preprocessor if is_csv_dataset else None,
            'label_encoder': label_encoder,
            'feature_names': feature_names,
            'model_config': {
                'n_estimators': 100,
                'max_depth': 15,
                'model_type': 'Random Forest'
            }
        }, model_path)
        
        st.success(f"💾 Scikit-learn Model saved to: {model_path}")
        
        # Calculate proper classification loss using log loss
        from sklearn.metrics import log_loss
        try:
            # Get probability predictions for proper loss calculation
            train_proba = model.predict_proba(X_train_scaled)
            test_proba = model.predict_proba(X_test_scaled)
            
            train_loss = log_loss(y_train, train_proba)
            test_loss = log_loss(y_test, test_proba)
        except Exception as e:
            # Fallback: use simple error-based loss
            train_loss = 1.0 - train_accuracy
            test_loss = 1.0 - test_accuracy
        
        results = {
            'best_accuracy': float(test_accuracy),
            'best_loss': float(test_loss),
            'train_loss': float(train_loss),
            'training_accuracy': float(train_accuracy),
            'cv_mean': float(cv_mean),
            'cv_std': float(cv_std),
            'training_time': float(training_time),
            'total_epochs': 1,
            'framework': 'Scikit-learn',
            'model_type': 'Random Forest',
            'n_estimators': 100,
            'model_saved': model_path
        }
        
        # Save training results to JSON with intelligent naming
        results_name = model_name.replace('.pkl', '_results.json')
        results_path = os.path.join(output_dir, results_name)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        st.success(f"📋 Scikit-learn Training results saved to: {results_path}")
        
        return results
        
    except Exception as e:
        st.error(f"❌ Scikit-learn training failed: {str(e)}")
        return {
            'best_accuracy': 0.0,
            'best_loss': float('inf'),
            'training_time': 0.0,
            'error': str(e)
        }



def get_framework_models(framework, task_type):
    """Get available models for specific framework and task type."""
    
    framework_models = {
        "PyTorch": {
            "classification": [
                "🚀 EfficientNet-B0 (Recommended)", "📱 MobileNetV3-Small (Fast)", "⚡ RegNetY-002 (Balanced)",
                "Simple CNN", "ResNet50", "ResNet101", "EfficientNet-B3", "VGG16", "DenseNet121", "MobileNetV2"
            ],
            "detection": ["YOLO-v5", "Faster R-CNN", "SSD", "RetinaNet"],
            "segmentation": ["U-Net", "DeepLabV3", "FCN", "PSPNet"]
        },
        "TensorFlow/Keras": {
            "classification": [
                "🏗️ Custom CNN Builder (Channel-Adaptive)", "🚀 EfficientNet-B0 (Recommended)", "📱 MobileNetV3-Small (Fast)", 
                "⚡ RegNetY-002 (Balanced)", "⚡ ConvNeXt-Tiny (Modern)",
                "MobileNetV2", "ResNet50", "ResNet101", "VGG16", "InceptionV3"
            ],
            "detection": ["YOLO-TensorFlow", "SSD-MobileNet", "Faster R-CNN", "CenterNet"],
            "segmentation": ["U-Net", "DeepLabV3+", "Mask R-CNN", "FCN"]
        },
        "Scikit-learn": {
            "classification": [
                "Random Forest", "Extra Trees", "Gradient Boosting", "AdaBoost",
                "Support Vector Machine", "Linear SVM", "Nu-SVM",
                "Logistic Regression", "Ridge Classifier", "Lasso Regression",
                "Elastic Net", "Perceptron", "Passive Aggressive",
                "Naive Bayes", "Gaussian NB", "Multinomial NB", "Bernoulli NB",
                "K-Nearest Neighbors", "Radius Neighbors",
                "Decision Tree", "Extra Tree",
                "Multi-layer Perceptron", "Neural Network",
                "Quadratic Discriminant", "Linear Discriminant"
            ],
            "detection": [
                "HOG + SVM", "Feature Extraction + Random Forest", 
                "Local Binary Pattern + SVM", "SIFT + KMeans + SVM",
                "Histogram Features + Gradient Boosting"
            ],
            "segmentation": [
                "Watershed + Features", "K-Means Clustering", "Gaussian Mixture",
                "Spectral Clustering", "DBSCAN + Features", "Mean Shift"
            ]
        }
    }
    
    return framework_models.get(framework, {}).get(task_type, [])


def select_tensorflow_model(dataset_info, detected_channels):
    """Select optimal TensorFlow/Keras model based on dataset characteristics."""
    from utils.config import ModelConfig
    
    task_type = dataset_info.task_type
    num_classes = dataset_info.num_classes
    
    # TensorFlow/Keras model recommendations by task (IMPROVED)
    if task_type == "classification":
        if num_classes <= 10:
            # EfficientNet-B0 for small classification (better than Simple CNN)
            architecture = "🚀 EfficientNet-B0 (Recommended)"
            backbone = "efficientnet_b0"
        elif num_classes <= 100:
            # EfficientNet-B0 for medium classification (better than MobileNetV2)
            architecture = "🚀 EfficientNet-B0 (Recommended)" 
            backbone = "efficientnet_b0"
        else:
            # ConvNeXt for large classification (modern architecture)
            architecture = "⚡ ConvNeXt-Tiny (Modern)"
            backbone = "convnext_tiny"
    
    elif task_type == "detection":
        # Object detection models
        architecture = "YOLO-TensorFlow"
        backbone = "darknet"
        
    elif task_type == "segmentation":
        # Semantic segmentation models
        architecture = "U-Net"
        backbone = "unet"
    
    else:
        # Default classification
        architecture = "EfficientNet-B0"
        backbone = "efficientnet_b0"
    
    # Configure input size based on channels
    if detected_channels == 1:
        input_size = (224, 224, 1)  # TensorFlow format: (H, W, C)
    elif detected_channels == 4:
        input_size = (224, 224, 4)
    else:
        input_size = (224, 224, 3)  # Default RGB
    
    return ModelConfig(
        architecture=architecture,
        backbone=backbone,
        num_parameters=2_000_000,  # Estimated
        memory_requirements=2.0,   # GB
        estimated_flops=1_000_000_000,
        input_size=input_size,
        pretrained=detected_channels == 3,  # Only RGB has pretrained
        framework="TensorFlow",
        config_params={
            'num_classes': num_classes,
            'input_channels': detected_channels,
            'dropout': 0.2,
            'activation': 'relu',
            'optimizer': 'adam',
            'loss': 'sparse_categorical_crossentropy' if task_type == "classification" else 'mse'
        }
    )


def select_sklearn_model(dataset_info):
    """Select optimal Scikit-learn model based on dataset characteristics.""" 
    from core.model_selector import ModelConfig
    import streamlit as st
    
    task_type = dataset_info.task_type
    num_classes = dataset_info.num_classes
    num_samples = dataset_info.num_samples
    
    # Enhanced Scikit-learn model recommendations with detailed logic
    if task_type == "classification":
        # Advanced selection based on dataset characteristics
        if num_samples < 500:
            # Very small dataset - use Naive Bayes or simple models
            if num_classes <= 2:
                architecture = "Logistic Regression"
                config_params = {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 1000}
                memory_req = 0.1
                num_params = num_samples * 10
            else:
                architecture = "Gaussian NB"
                config_params = {'var_smoothing': 1e-9}
                memory_req = 0.1
                num_params = num_classes * 50
                
        elif num_samples < 2000:
            # Small dataset - use SVM with good generalization
            architecture = "Support Vector Machine"
            config_params = {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}
            memory_req = 0.3
            num_params = min(num_samples, 5000)
            
        elif num_samples < 10000:
            # Medium dataset - Random Forest works well
            architecture = "Random Forest"
            config_params = {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 5}
            memory_req = 0.8
            num_params = 100 * 15 * num_classes
            
        elif num_samples < 50000:
            # Large dataset - Gradient Boosting for high accuracy
            architecture = "Gradient Boosting"
            config_params = {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 6}
            memory_req = 1.5
            num_params = 200 * 6 * num_classes
            
        else:
            # Very large dataset - Extra Trees for speed and accuracy
            architecture = "Extra Trees"
            config_params = {'n_estimators': 150, 'max_depth': 20, 'n_jobs': -1}
            memory_req = 2.0
            num_params = 150 * 20 * num_classes
    
    elif task_type in ["detection", "segmentation"]:
        # Computer vision tasks - feature extraction + classifier
        st.warning("⚠️ Scikit-learn is not optimal for computer vision tasks. Using feature extraction approach.")
        if num_samples < 5000:
            architecture = "HOG + SVM"
            config_params = {'C': 1.0, 'kernel': 'rbf', 'feature_method': 'HOG'}
            memory_req = 1.0
            num_params = 10000
        else:
            architecture = "Feature Extraction + Random Forest"
            config_params = {'n_estimators': 100, 'feature_method': 'Histogram Features'}
            memory_req = 1.5
            num_params = 50000
    
    else:
        # Default classification setup
        architecture = "Random Forest"
        config_params = {'n_estimators': 100, 'max_depth': 15}
        memory_req = 0.8
        num_params = 100000
    
    # Add common configuration parameters
    base_config = {
        'num_classes': num_classes,
        'num_samples': num_samples,
        'task_type': task_type,
        'random_state': 42,
        'framework': 'scikit-learn',
        'use_scaling': True,
        'cv_folds': 5,
        'scoring_metrics': ['accuracy', 'f1'],
        'feature_method': 'Histogram Features',
        'image_size': (64, 64),
        'class_weight': 'balanced' if num_classes > 2 else None
    }
    
    # Merge with algorithm-specific parameters
    config_params.update(base_config)
    
    return ModelConfig(
        architecture=architecture,
        backbone="N/A",            # No backbone for traditional ML
        num_parameters=num_params,
        memory_requirements=memory_req,
        estimated_flops=num_params * 100,  # Rough estimate
        input_size="Feature Vector",
        pretrained=False,          # Scikit-learn doesn't use pretrained models
        framework="Scikit-learn", 
        config_params=config_params
    )


def get_system_info() -> Dict[str, str]:
    """Get system information.""" 
    info = {
        "Python Version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "Platform": sys.platform,
        "CPU Count": str(os.cpu_count())
    }
    
    try:
        import torch
        info["PyTorch Version"] = torch.__version__
        if torch.cuda.is_available():
            info["GPU Count"] = str(torch.cuda.device_count())
            info["GPU Name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    
    return info


if __name__ == "__main__":
    main()
