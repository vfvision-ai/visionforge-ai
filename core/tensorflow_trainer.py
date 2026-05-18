#!/usr/bin/env python3
"""
TensorFlow/Keras Training Pipeline
Complete implementation for TensorFlow model training with enhanced metrics.
"""
from __future__ import annotations

import logging
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import (
        ResNet50, EfficientNetB0, MobileNetV2, VGG16, 
        InceptionV3, Xception, DenseNet121
    )
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None
    keras = None

from utils.config import Config
from utils.logger import setup_logger
from utils.metrics import MetricsTracker
from core.dataset_analyzer import DatasetInfo


class TensorFlowTrainer:
    """Complete TensorFlow/Keras training pipeline with enhanced accuracy tracking."""
    
    def __init__(self, config: Config):
        """Initialize TensorFlow trainer."""
        self.config = config
        self.logger = setup_logger(__name__)
        self.metrics_tracker = MetricsTracker()
        
        # Check TensorFlow availability
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is not installed. Please install with: pip install tensorflow"
            )
        
        # Configure TensorFlow
        self._configure_tensorflow()
        
        # Initialize training state
        self.model = None
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        self.history = None
        
    def _configure_tensorflow(self):
        """Configure TensorFlow settings for optimal performance."""
        try:
            # Set memory growth for GPU if available
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info(f"✅ Configured {len(gpus)} GPU(s) for TensorFlow")
            else:
                self.logger.info("🔧 Using CPU for TensorFlow training")
                
            # Set mixed precision for better performance
            if gpus:
                tf.config.optimizer.set_jit(True)
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                self.logger.info("✅ Enabled mixed precision training")
                
        except Exception as e:
            self.logger.warning(f"⚠️ TensorFlow configuration warning: {e}")
    
    # Canonical map: various user-facing names -> tf.keras.datasets module name
    _KERAS_DATASET_MAP = {
        "mnist":         "mnist",
        "MNIST":         "mnist",
        "fashion_mnist": "fashion_mnist",
        "fashion-mnist": "fashion_mnist",
        "Fashion MNIST": "fashion_mnist",
        "cifar10":       "cifar10",
        "cifar-10":      "cifar10",
        "CIFAR-10":      "cifar10",
        "cifar100":      "cifar100",
        "cifar-100":     "cifar100",
        "CIFAR-100":     "cifar100",
    }

    def prepare_data(self, dataset_info: DatasetInfo, batch_size: int = 32,
                    validation_split: float = 0.2) -> Dict[str, Any]:
        """Prepare TensorFlow data generators for training."""
        try:
            self.logger.info("🔄 Preparing TensorFlow data generators...")

            # ── Builtin dataset (MNIST, CIFAR-10, …) ──────────────────────────
            dataset_path = str(getattr(dataset_info, 'dataset_path', '') or '')
            is_builtin = getattr(dataset_info, 'is_builtin', False)
            keras_name = self._KERAS_DATASET_MAP.get(dataset_path) or \
                         self._KERAS_DATASET_MAP.get(
                             getattr(dataset_info, 'builtin_dataset_name', '') or '')
            if is_builtin or keras_name or not Path(dataset_path).is_dir():
                return self._prepare_builtin_data(dataset_info, batch_size, keras_name)
            # ──────────────────────────────────────────────────────────────────
            
            # Determine input shape
            if len(dataset_info.image_size) == 2:
                input_shape = (*dataset_info.image_size, 1)  # Grayscale
            else:
                input_shape = dataset_info.image_size
                
            # Data augmentation for training
            train_datagen = ImageDataGenerator(
                rescale=1.0/255.0,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                fill_mode='nearest',
                validation_split=validation_split
            )
            
            # No augmentation for validation/test
            val_datagen = ImageDataGenerator(
                rescale=1.0/255.0,
                validation_split=validation_split
            )
            
            # Create generators
            self.train_generator = train_datagen.flow_from_directory(
                dataset_info.dataset_path,
                target_size=dataset_info.image_size[:2],
                batch_size=batch_size,
                class_mode='categorical' if dataset_info.num_classes > 2 else 'binary',
                subset='training',
                color_mode='grayscale' if len(dataset_info.image_size) == 2 else 'rgb'
            )
            
            self.val_generator = val_datagen.flow_from_directory(
                dataset_info.dataset_path,
                target_size=dataset_info.image_size[:2],
                batch_size=batch_size,
                class_mode='categorical' if dataset_info.num_classes > 2 else 'binary',
                subset='validation',
                color_mode='grayscale' if len(dataset_info.image_size) == 2 else 'rgb'
            )
            
            # Store data info
            data_info = {
                'input_shape': input_shape,
                'num_classes': dataset_info.num_classes,
                'train_samples': self.train_generator.samples,
                'val_samples': self.val_generator.samples,
                'class_indices': self.train_generator.class_indices
            }
            
            self.logger.info(f"✅ Data prepared - Train: {data_info['train_samples']}, Val: {data_info['val_samples']}")
            return data_info

        except Exception as e:
            self.logger.error(f"❌ Data preparation failed: {e}")
            raise

    def _prepare_builtin_data(self, dataset_info: DatasetInfo,
                              batch_size: int, keras_name: str) -> Dict[str, Any]:
        """Load a built-in Keras dataset (MNIST, CIFAR-10, …) instead of a directory."""
        if not keras_name:
            keras_name = self._KERAS_DATASET_MAP.get(
                str(getattr(dataset_info, 'dataset_path', '') or '').lower(), 'mnist')

        self.logger.info(f"🔄 Loading builtin Keras dataset: {keras_name}")
        loader = getattr(tf.keras.datasets, keras_name)
        (x_train, y_train), (x_test, y_test) = loader.load_data()

        # Normalize to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test  = x_test.astype('float32')  / 255.0

        # Add channel dim for grayscale: (N, H, W) -> (N, H, W, 1)
        if x_train.ndim == 3:
            x_train = x_train[..., np.newaxis]
            x_test  = x_test[..., np.newaxis]

        num_classes = int(y_train.max()) + 1
        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test_cat  = tf.keras.utils.to_categorical(y_test,  num_classes)

        # Store as tf.data.Dataset so model.fit() receives batches correctly
        self.train_generator = (
            tf.data.Dataset
            .from_tensor_slices((x_train, y_train_cat))
            .shuffle(10000)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        self.val_generator = (
            tf.data.Dataset
            .from_tensor_slices((x_test, y_test_cat))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        input_shape = x_train.shape[1:]  # e.g. (28, 28, 1)
        data_info = {
            'input_shape':   input_shape,
            'num_classes':   num_classes,
            'train_samples': len(x_train),
            'val_samples':   len(x_test),
            'class_indices': {str(i): i for i in range(num_classes)},
        }
        self.logger.info(
            f"✅ Builtin {keras_name}: train={len(x_train)}, val={len(x_test)}, "
            f"classes={num_classes}, shape={input_shape}"
        )
        return data_info

    def build_model(self, model_config: Dict[str, Any], data_info: Dict[str, Any]) -> keras.Model:
        """Build TensorFlow/Keras model architecture."""
        try:
            self.logger.info(f"🏗️ Building {model_config['architecture']} model...")
            
            input_shape = data_info['input_shape']
            num_classes = data_info['num_classes']
            
            # Check for custom CNN builder
            config_params = model_config.get('config_params', {})
            if config_params.get('is_custom_cnn', False):
                model = self._build_custom_cnn(input_shape, num_classes, config_params)
            elif model_config['architecture'] == 'Sequential CNN':
                model = self._build_sequential_cnn(input_shape, num_classes, model_config)
            elif model_config['architecture'] == 'ResNet50':
                model = self._build_resnet50(input_shape, num_classes, model_config)
            elif model_config['architecture'] == 'EfficientNetB0':
                model = self._build_efficientnet(input_shape, num_classes, model_config)
            elif model_config['architecture'] == 'MobileNetV2':
                model = self._build_mobilenet(input_shape, num_classes, model_config)
            elif model_config['architecture'] == 'VGG16':
                model = self._build_vgg16(input_shape, num_classes, model_config)
            elif model_config['architecture'] == 'InceptionV3':
                model = self._build_inception(input_shape, num_classes, model_config)
            else:
                # Fallback to sequential CNN
                self.logger.warning(f"⚠️ Unknown architecture {model_config['architecture']}, using Sequential CNN")
                model = self._build_sequential_cnn(input_shape, num_classes, model_config)
            
            self.model = model

            # Compile immediately so model.fit() can be called without extra steps
            lr = getattr(self.config, 'learning_rate', 0.001)
            num_classes = data_info['num_classes']
            loss = 'categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss=loss,
                metrics=['accuracy'],
            )
            self.logger.info(f"✅ Model built & compiled — params={model.count_params():,}, loss={loss}, lr={lr}")
            return model

        except Exception as e:
            self.logger.error(f"❌ Model building failed: {e}")
            raise
    
    def _build_custom_cnn(self, input_shape: Tuple[int, ...], num_classes: int, config: Dict[str, Any]) -> keras.Model:
        """Build custom CNN architecture from user configuration."""
        self.logger.info("🏗️ Building custom CNN from user configuration...")
        
        custom_config = config.get('custom_cnn_config', {})
        
        # Get activation function
        activation = custom_config.get('activation', 'relu')
        dropout_rate = custom_config.get('dropout_rate', 0.3)
        use_global_pooling = custom_config.get('use_global_pooling', True)
        
        # Build model layer by layer
        model_layers = []
        model_layers.append(layers.Input(shape=input_shape))
        
        # Add convolutional blocks
        conv_blocks = custom_config.get('conv_blocks', [])
        for i, block in enumerate(conv_blocks):
            filters = block['filters']
            kernel_size = block['kernel_size']
            
            # Conv2D layer
            model_layers.append(layers.Conv2D(
                filters, 
                kernel_size, 
                padding='same',
                activation=None,  # Activation applied after batch norm
                name=f'conv_block_{i+1}_conv'
            ))
            
            # Batch normalization
            if block.get('use_batch_norm', True):
                model_layers.append(layers.BatchNormalization(name=f'conv_block_{i+1}_bn'))
            
            # Activation
            if activation == 'leaky_relu':
                model_layers.append(layers.LeakyReLU(alpha=0.1, name=f'conv_block_{i+1}_activation'))
            elif activation == 'elu':
                model_layers.append(layers.ELU(name=f'conv_block_{i+1}_activation'))
            elif activation == 'selu':
                model_layers.append(layers.Activation('selu', name=f'conv_block_{i+1}_activation'))
            elif activation == 'swish':
                model_layers.append(layers.Activation('swish', name=f'conv_block_{i+1}_activation'))
            else:
                model_layers.append(layers.Activation('relu', name=f'conv_block_{i+1}_activation'))
            
            # MaxPooling
            if block.get('use_max_pool', True):
                model_layers.append(layers.MaxPooling2D((2, 2), name=f'conv_block_{i+1}_pool'))
            
            # Spatial dropout between blocks
            if dropout_rate > 0:
                model_layers.append(layers.SpatialDropout2D(dropout_rate * 0.5, name=f'conv_block_{i+1}_dropout'))
        
        # Pooling layer (transition from conv to dense)
        if use_global_pooling:
            model_layers.append(layers.GlobalAveragePooling2D(name='global_avg_pool'))
        else:
            model_layers.append(layers.Flatten(name='flatten'))
        
        # Add hidden dense layers (not including output)
        dense_layers = custom_config.get('dense_layers', [])
        for i, dense in enumerate(dense_layers):
            units = dense['units']
            
            # Dense layer
            model_layers.append(layers.Dense(
                units,
                activation=None,  # Apply activation separately
                name=f'hidden_dense_{i+1}'
            ))
            
            # Batch normalization for dense layers
            model_layers.append(layers.BatchNormalization(name=f'hidden_dense_{i+1}_bn'))
            
            # Activation
            if activation == 'leaky_relu':
                model_layers.append(layers.LeakyReLU(alpha=0.1, name=f'hidden_dense_{i+1}_activation'))
            elif activation == 'elu':
                model_layers.append(layers.ELU(name=f'hidden_dense_{i+1}_activation'))
            elif activation == 'selu':
                model_layers.append(layers.Activation('selu', name=f'hidden_dense_{i+1}_activation'))
            elif activation == 'swish':
                model_layers.append(layers.Activation('swish', name=f'hidden_dense_{i+1}_activation'))
            else:
                model_layers.append(layers.Activation('relu', name=f'hidden_dense_{i+1}_activation'))
            
            # Dropout
            if dense.get('use_dropout', True) and dropout_rate > 0:
                model_layers.append(layers.Dropout(dropout_rate, name=f'hidden_dense_{i+1}_dropout'))
        
        # Output layer (final dense layer with num_classes units)
        output_activation = 'softmax' if num_classes > 2 else 'sigmoid'
        model_layers.append(layers.Dense(
            num_classes, 
            activation=output_activation,
            name='output_dense'
        ))
        
        # Create sequential model
        model = keras.Sequential(model_layers, name='custom_cnn')
        
        self.logger.info(f"✅ Custom CNN built with {len(conv_blocks)} conv blocks and {len(dense_layers)} dense layers")
        return model
    
    def _build_sequential_cnn(self, input_shape: Tuple[int, ...], num_classes: int, config: Dict[str, Any]) -> keras.Model:
        """Build a sequential CNN architecture."""
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation=config.get('activation', 'relu')),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation=config.get('activation', 'relu')),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation=config.get('activation', 'relu')),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Classification head
            layers.GlobalAveragePooling2D(),
            layers.Dropout(config.get('dropout', 0.2)),
            layers.Dense(128, activation=config.get('activation', 'relu')),
            layers.Dropout(config.get('dropout', 0.2)),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        return model
    
    def _build_resnet50(self, input_shape: Tuple[int, ...], num_classes: int, config: Dict[str, Any]) -> keras.Model:
        """Build ResNet50 architecture."""
        base_model = ResNet50(
            weights='imagenet' if input_shape[-1] == 3 else None,
            include_top=False,
            input_shape=input_shape
        )
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(config.get('dropout', 0.2)),
            layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        return model
    
    def _build_efficientnet(self, input_shape: Tuple[int, ...], num_classes: int, config: Dict[str, Any]) -> keras.Model:
        """Build EfficientNet architecture."""
        base_model = EfficientNetB0(
            weights='imagenet' if input_shape[-1] == 3 else None,
            include_top=False,
            input_shape=input_shape
        )
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(config.get('dropout', 0.2)),
            layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        return model
    
    def _build_mobilenet(self, input_shape: Tuple[int, ...], num_classes: int, config: Dict[str, Any]) -> keras.Model:
        """Build MobileNetV2 architecture."""
        base_model = MobileNetV2(
            weights='imagenet' if input_shape[-1] == 3 else None,
            include_top=False,
            input_shape=input_shape
        )
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(config.get('dropout', 0.2)),
            layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        return model
    
    def _build_vgg16(self, input_shape: Tuple[int, ...], num_classes: int, config: Dict[str, Any]) -> keras.Model:
        """Build VGG16 architecture."""
        base_model = VGG16(
            weights='imagenet' if input_shape[-1] == 3 else None,
            include_top=False,
            input_shape=input_shape
        )
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(config.get('dropout', 0.2)),
            layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        return model
    
    def _build_inception(self, input_shape: Tuple[int, ...], num_classes: int, config: Dict[str, Any]) -> keras.Model:
        """Build InceptionV3 architecture."""
        base_model = InceptionV3(
            weights='imagenet' if input_shape[-1] == 3 else None,
            include_top=False,
            input_shape=input_shape
        )
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(config.get('dropout', 0.2)),
            layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        return model
    
    def compile_model(self, model_config: Dict[str, Any], data_info: Dict[str, Any]):
        """Compile the TensorFlow model with optimizer and loss."""
        try:
            # Determine optimizer
            optimizer_name = model_config.get('optimizer', 'adam').lower()
            learning_rate = model_config.get('learning_rate', 0.001)
            
            if optimizer_name == 'adam':
                optimizer = optimizers.Adam(learning_rate=learning_rate)
            elif optimizer_name == 'sgd':
                optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
            elif optimizer_name == 'rmsprop':
                optimizer = optimizers.RMSprop(learning_rate=learning_rate)
            else:
                optimizer = optimizers.Adam(learning_rate=learning_rate)
            
            # Determine loss function
            num_classes = data_info['num_classes']
            if num_classes > 2:
                loss = 'categorical_crossentropy'
                metrics = ['accuracy', 'top_k_categorical_accuracy']
            else:
                loss = 'binary_crossentropy'
                metrics = ['accuracy', 'binary_accuracy']
            
            # Compile model
            self.model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics
            )
            
            self.logger.info(f"✅ Model compiled with {optimizer_name} optimizer and {loss} loss")
            
        except Exception as e:
            self.logger.error(f"❌ Model compilation failed: {e}")
            raise
    
    def train(self, epochs: int = 10, save_model: bool = True, 
              model_save_dir: str = "./experiments") -> Dict[str, Any]:
        """Train the TensorFlow model with enhanced metrics tracking."""
        try:
            self.logger.info(f"🚀 Starting TensorFlow training for {epochs} epochs...")
            
            # Create save directory
            save_dir = Path(model_save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup callbacks
            callback_list = self._setup_callbacks(save_dir, save_model)
            
            # Start training timer
            training_start_time = time.time()
            
            # Initialize metrics tracker
            self.metrics_tracker.reset()
            
            # Custom training loop with enhanced metrics
            best_accuracy = 0.0
            best_loss = float('inf')
            
            # Train the model
            self.history = self.model.fit(
                self.train_generator,
                epochs=epochs,
                validation_data=self.val_generator,
                callbacks=callback_list,
                verbose=1
            )
            
            # Calculate training time
            training_time = time.time() - training_start_time
            
            # Extract metrics from history
            train_acc = self.history.history['accuracy']
            val_acc = self.history.history['val_accuracy']
            train_loss = self.history.history['loss']
            val_loss = self.history.history['val_loss']
            
            # Calculate best metrics
            best_accuracy = float(max(val_acc))
            best_loss = float(min(val_loss))
            final_train_acc = float(train_acc[-1])
            final_val_acc = float(val_acc[-1])
            
            # Enhanced metrics tracking
            for epoch in range(len(train_acc)):
                epoch_metrics = {
                    'train_loss': float(train_loss[epoch]),
                    'train_acc': float(train_acc[epoch]),
                    'train_accuracy': float(train_acc[epoch]),
                    'val_loss': float(val_loss[epoch]),
                    'val_acc': float(val_acc[epoch]),
                    'val_accuracy': float(val_acc[epoch])
                }

                self.metrics_tracker.update(epoch_metrics)

                # Log detailed metrics
                overfitting_gap = abs(epoch_metrics['train_accuracy'] - epoch_metrics['val_accuracy']) * 100
                if overfitting_gap > 5.0:
                    self.logger.warning(f"   ⚠️ Overfitting Gap: {overfitting_gap:.4f}% (Train-Val)")

                self.logger.info(f"Epoch {epoch + 1} - {epoch_metrics}")
            
            # Prepare comprehensive results
            results = {
                'best_accuracy': best_accuracy,
                'best_loss': best_loss,
                'final_train_accuracy': final_train_acc,
                'final_val_accuracy': final_val_acc,
                'training_time': training_time,
                'total_epochs': epochs,
                'framework': 'TensorFlow/Keras',
                'model_architecture': self.model.__class__.__name__,
                'total_parameters': int(self.model.count_params()),
                'overfitting_detected': abs(final_train_acc - final_val_acc) > 0.1,
                'convergence_status': 'converged' if len(train_acc) < epochs else 'completed',
                'training_history': {
                    'train_accuracy': [float(x) for x in train_acc],
                    'val_accuracy': [float(x) for x in val_acc],
                    'train_loss': [float(x) for x in train_loss],
                    'val_loss': [float(x) for x in val_loss]
                }
            }
            
            # Save training results
            results_path = save_dir / "tensorflow_training_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"🎉 TensorFlow training completed!")
            self.logger.info(f"   📊 Best Accuracy: {best_accuracy:.6f}")
            self.logger.info(f"   📉 Best Loss: {best_loss:.6f}")
            self.logger.info(f"   ⏱️ Training Time: {training_time:.2f}s")
            self.logger.info(f"   💾 Results saved to: {results_path}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ TensorFlow training failed: {e}")
            raise
    
    def _setup_callbacks(self, save_dir: Path, save_model: bool) -> List[callbacks.Callback]:
        """Setup TensorFlow training callbacks."""
        callback_list = []
        
        try:
            # Early stopping
            early_stopping = callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
            callback_list.append(early_stopping)
            
            # Reduce learning rate on plateau
            lr_reducer = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
            callback_list.append(lr_reducer)
            
            # Model checkpointing with intelligent naming
            if save_model:
                from utils.model_naming import generate_comprehensive_model_paths, extract_dataset_name
                
                # Extract dataset name and architecture info
                dataset_name = "unknown"
                if hasattr(self, 'dataset_path') and self.dataset_path:
                    dataset_name = extract_dataset_name(dataset_path=self.dataset_path)
                elif hasattr(self, 'config') and hasattr(self.config, 'dataset_path'):
                    dataset_name = extract_dataset_name(dataset_path=self.config.dataset_path)
                
                # Get architecture name from model
                architecture = "unknown"
                if hasattr(self, 'model_config') and hasattr(self.model_config, 'architecture'):
                    architecture = self.model_config.architecture
                elif self.model:
                    architecture = self.model.__class__.__name__
                
                # Get backbone if available
                backbone = None
                if hasattr(self, 'model_config') and hasattr(self.model_config, 'backbone'):
                    backbone = self.model_config.backbone
                
                # Generate comprehensive paths
                model_paths = generate_comprehensive_model_paths(
                    base_dir=save_dir,
                    framework="tensorflow",
                    architecture=architecture,
                    backbone=backbone,
                    dataset_name=dataset_name,
                    task_type="classification"  # Default for TensorFlow trainer
                )
                
                checkpoint_path = model_paths['model']
                model_checkpoint = callbacks.ModelCheckpoint(
                    str(checkpoint_path),
                    monitor='val_accuracy',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1
                )
                callback_list.append(model_checkpoint)
                
                # Store model path for later use
                self.latest_model_path = str(checkpoint_path)
                self.model_base_name = model_paths['base_name']
            
            # CSV logger
            csv_logger = callbacks.CSVLogger(str(save_dir / "tensorflow_training_log.csv"))
            callback_list.append(csv_logger)
            
            self.logger.info(f"✅ Setup {len(callback_list)} TensorFlow callbacks")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Callback setup warning: {e}")
        
        return callback_list
    
    def evaluate(self, test_data_path: Optional[str] = None) -> Dict[str, float]:
        """Evaluate the trained model on test data."""
        try:
            if self.model is None:
                raise ValueError("Model not trained yet. Call train() first.")
            
            # Use validation data if no test data provided
            eval_generator = self.val_generator
            if test_data_path and Path(test_data_path).exists():
                # Create test generator
                test_datagen = ImageDataGenerator(rescale=1.0/255.0)
                eval_generator = test_datagen.flow_from_directory(
                    test_data_path,
                    target_size=self.train_generator.target_size,
                    batch_size=self.train_generator.batch_size,
                    class_mode=self.train_generator.class_mode,
                    shuffle=False
                )
            
            # Evaluate model
            self.logger.info("🔍 Evaluating TensorFlow model...")
            evaluation = self.model.evaluate(eval_generator, verbose=1)
            
            # Prepare results
            metric_names = self.model.metrics_names
            eval_results = dict(zip(metric_names, evaluation))
            
            self.logger.info(f"✅ Evaluation completed: {eval_results}")
            return eval_results
            
        except Exception as e:
            self.logger.error(f"❌ Model evaluation failed: {e}")
            return {}
    
    def save_model(self, model_path: str):
        """Save the trained TensorFlow model."""
        try:
            if self.model is None:
                raise ValueError("No model to save. Train a model first.")
            
            # Ensure .keras extension
            if not model_path.endswith('.keras'):
                model_path += '.keras'
            
            self.model.save(model_path)
            self.logger.info(f"💾 Model saved to: {model_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Model saving failed: {e}")
            raise
    
    def load_model(self, model_path: str):
        """Load a saved TensorFlow model."""
        try:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.model = keras.models.load_model(model_path)
            self.logger.info(f"📂 Model loaded from: {model_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Model loading failed: {e}")
            raise


def create_tensorflow_trainer(config: Config) -> TensorFlowTrainer:
    """Factory function to create TensorFlow trainer."""
    return TensorFlowTrainer(config)