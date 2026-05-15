"""
Auto Trainer - Automated training pipeline with intelligent monitoring.
"""

import logging
import time
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

# Setup platform-specific library paths (Windows DLL, Linux/macOS lib paths)
# Import platform utilities for cross-platform compatibility
try:
    from utils.platform_utils import setup_dll_directories
    setup_dll_directories()
except Exception:
    # Silently continue if platform utils unavailable
    pass

# Import PyTorch with error handling
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.amp import autocast, GradScaler
    TORCH_AVAILABLE = True
except Exception as e:
    # Create dummy classes for type hints when PyTorch unavailable
    class torch:
        class device: pass
        class Tensor: pass
        @staticmethod
        def cuda(): return None
    class nn:
        class CrossEntropyLoss: pass
    class optim:
        class AdamW: pass
        class SGD: pass
        class lr_scheduler:
            class CosineAnnealingLR: pass
            class ReduceLROnPlateau: pass
    class DataLoader: pass
    class autocast: pass
    class GradScaler: pass
    TORCH_AVAILABLE = False
    print(f"⚠️ PyTorch not available in trainer: {e}")

from utils.config import Config
from utils.metrics import MetricsTracker, compute_segmentation_metrics, compute_detection_metrics
from utils.callbacks import CallbackManager
from utils.model_factory import ModelFactory
from utils.data_factory import DataLoaderFactory


@dataclass
class TrainingResults:
    """Training results and artifacts."""
    best_accuracy: float
    best_loss: float
    training_time: float
    model_path: Path
    log_path: Path
    metrics_history: Dict[str, List[float]]
    final_model_state: Dict[str, Any]


class AutoTrainer:
    """Automated training pipeline with intelligent monitoring and optimization."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup device
        self.device = self._setup_device()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = GradScaler() if self.config.use_mixed_precision else None
        
        # Tracking
        self.metrics_tracker = MetricsTracker()
        self.callback_manager = CallbackManager()
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.training_start_time = None
        
    def train(self) -> TrainingResults:
        """
        Execute automated training pipeline.
        
        Returns:
            TrainingResults with training artifacts and metrics
        """
        self.logger.info("🔥 Starting automated training pipeline")
        self.training_start_time = time.time()
        
        try:
            # Setup training components
            self._setup_training()
            
            # Execute training loop
            self._training_loop()
            
            # Generate results
            results = self._generate_results()
            
            self.logger.info("✅ Training completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {str(e)}")
            raise
    
    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self.config.gpu_id}")
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
        else:
            device = torch.device("cpu")
            self.logger.info("Using CPU for training")
            
        return device
    
    def _setup_training(self):
        """Setup all training components."""
        self.logger.info("🔧 Setting up training components...")
        
        # Create data loaders
        self._setup_data_loaders()
        
        # Create model
        self._setup_model()
        
        # Setup optimization
        self._setup_optimization()
        
        # Setup callbacks
        self._setup_callbacks()
        
        self.logger.info("✅ Training setup complete")
    
    def _setup_data_loaders(self):
        """Create training and validation data loaders."""
        try:
            factory = DataLoaderFactory(self.config)
            
            self.train_loader = factory.create_train_loader()
            self.val_loader = factory.create_val_loader()
            
            self.logger.info(f"✅ Data loaders created successfully")
            self.logger.info(f"📊 Training samples: {len(self.train_loader.dataset)}")
            self.logger.info(f"📊 Validation samples: {len(self.val_loader.dataset)}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create data loaders: {e}")
            # Fallback: create minimal data loaders without multiprocessing
            try:
                self.logger.info("🔄 Trying fallback data loader configuration...")
                factory = DataLoaderFactory(self.config)
                
                # Force single-threaded operation
                self.train_loader = factory.create_train_loader()
                self.val_loader = factory.create_val_loader()
                
                self.logger.info(f"✅ Fallback data loaders created")
                self.logger.info(f"📊 Training samples: {len(self.train_loader.dataset)}")
                self.logger.info(f"📊 Validation samples: {len(self.val_loader.dataset)}")
                
            except Exception as fallback_e:
                self.logger.error(f"❌ Fallback data loader creation failed: {fallback_e}")
                raise RuntimeError(f"Data loader creation failed: {e}. Fallback also failed: {fallback_e}") from e
    
    def _setup_model(self):
        """Create and initialize model."""
        try:
            factory = ModelFactory(self.config)
            self.model = factory.create_model()
            self.model = self.model.to(self.device)
            
            # Print model info
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            self.logger.info(f"✅ Model created: {self.config.model_config.architecture}")
            self.logger.info(f"📊 Total parameters: {total_params:,}")
            self.logger.info(f"🎯 Trainable parameters: {trainable_params:,}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create model: {e}")
            raise RuntimeError(f"Model creation failed: {e}") from e
    
    def _setup_optimization(self):
        """Setup optimizer, scheduler, and loss function."""
        
        # Create optimizer
        if self.config.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999)
            )
        elif self.config.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
                nesterov=True
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
        
        # Create scheduler with error handling
        try:
            if self.config.scheduler == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config.max_epochs,
                    eta_min=self.config.learning_rate * 0.01
                )
            elif self.config.scheduler == 'plateau':
                # Try with minimal parameters to avoid deprecated arguments
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='max',
                    factor=0.5,
                    patience=5
                )
            self.logger.info(f"✅ Created {self.config.scheduler} scheduler successfully")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create scheduler: {e}. Using no scheduler.")
            self.scheduler = None
        
        # Create loss function
        self._setup_loss_function()
    
    def _setup_loss_function(self):
        """Setup task-specific loss function."""
        task_type = self.config.dataset_info.task_type
        
        if task_type == "classification":
            if self.config.model_config.config_params.get('label_smoothing', 0) > 0:
                self.criterion = nn.CrossEntropyLoss(
                    label_smoothing=self.config.model_config.config_params['label_smoothing']
                )
            else:
                self.criterion = nn.CrossEntropyLoss()
                
        elif task_type == "detection":
            # Detection loss is typically built into the model
            self.criterion = None
            
        elif task_type == "segmentation":
            # Use weighted cross-entropy for segmentation
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=self.config.model_config.config_params.get('ignore_index', 255)
            )
        
        if self.criterion:
            self.criterion = self.criterion.to(self.device)
    
    def _setup_callbacks(self):
        """Setup training callbacks."""
        from utils.callbacks import (
            EarlyStopping, ModelCheckpoint, LearningRateMonitor,
            ProgressBar, MetricsLogger
        )
        
        # Early stopping
        self.callback_manager.add_callback(
            EarlyStopping(
                monitor='val_acc',
                patience=self.config.early_stopping_patience,
                min_delta=0.001,
                mode='max'
            )
        )
        
        # Model checkpointing with intelligent naming
        from utils.model_naming import generate_comprehensive_model_paths, extract_dataset_name
        
        # Extract dataset name and architecture info
        dataset_name = "unknown"
        if hasattr(self.config, 'dataset_path') and self.config.dataset_path:
            dataset_name = extract_dataset_name(dataset_path=self.config.dataset_path)
        
        # Get architecture name from model config
        architecture = "unknown"
        if hasattr(self.config, 'model_name') and self.config.model_name:
            architecture = self.config.model_name
        elif hasattr(self, 'model') and self.model:
            architecture = self.model.__class__.__name__
        
        # Get backbone if available
        backbone = None
        if hasattr(self.config, 'backbone') and self.config.backbone:
            backbone = self.config.backbone
        
        # Generate comprehensive paths
        model_paths = generate_comprehensive_model_paths(
            base_dir=self.config.output_dir,
            framework="pytorch",
            architecture=architecture,
            backbone=backbone,
            dataset_name=dataset_name,
            task_type=getattr(self.config, 'task_type', 'classification')
        )
        
        # Store paths for later use
        self.intelligent_model_path = model_paths['model']
        self.model_base_name = model_paths['base_name']
        
        self.callback_manager.add_callback(
            ModelCheckpoint(
                filepath=self.intelligent_model_path,
                monitor='val_acc',
                save_best_only=True,
                mode='max'
            )
        )
        
        # Learning rate monitoring
        self.callback_manager.add_callback(LearningRateMonitor())
        
        # Progress bar
        self.callback_manager.add_callback(ProgressBar())
        
        # Metrics logging
        self.callback_manager.add_callback(
            MetricsLogger(
                log_dir=self.config.output_dir / 'logs',
                log_interval=10
            )
        )
    
    def _training_loop(self):
        """Main training loop with automatic monitoring."""
        
        self.callback_manager.on_train_begin()
        
        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            
            self.callback_manager.on_epoch_begin(epoch)
            
            # Training phase
            train_metrics = self._train_epoch()
            
            # Validation phase
            val_metrics = self._validate_epoch()
            
            # Update metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            self.metrics_tracker.update(epoch_metrics)
            
            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    monitor_val = val_metrics.get('val_map50',
                                  val_metrics.get('val_miou',
                                  val_metrics.get('val_acc', 0)))
                    self.scheduler.step(monitor_val)
                else:
                    self.scheduler.step()
            
            # Callbacks
            self.callback_manager.on_epoch_end(epoch, epoch_metrics)
            
            # Check if training should stop
            if self.callback_manager.should_stop_training():
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        self.callback_manager.on_train_end()
    
    def _train_epoch(self) -> Dict[str, float]:
        """Execute one training epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        seg_iou_sum = 0.0
        seg_dice_sum = 0.0
        seg_batches = 0
        
        self.callback_manager.on_train_epoch_begin()
        
        for batch_idx, batch in enumerate(self.train_loader):
            self.callback_manager.on_train_batch_begin(batch_idx)
            
            # Move data to device
            inputs, targets = self._prepare_batch(batch)

            # Forward pass
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu',
                         enabled=self.config.use_mixed_precision):
                if self.config.dataset_info.task_type == "detection" and targets is not None:
                    # Torchvision / SimpleDetector train mode: pass targets to model
                    outputs = self.model(inputs, targets)
                else:
                    outputs = self.model(inputs)
                loss = self._compute_loss(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            # Calculate accuracy for classification
            if self.config.dataset_info.task_type == "classification":
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == targets).sum().item()
            elif self.config.dataset_info.task_type == "segmentation":
                seg_out = outputs['out'] if isinstance(outputs, dict) else outputs
                miou, dice, _ = compute_segmentation_metrics(
                    seg_out.detach(), targets,
                    num_classes=self.config.dataset_info.num_classes
                )
                seg_iou_sum  += miou
                seg_dice_sum += dice
                seg_batches  += 1
            
            self.callback_manager.on_train_batch_end(batch_idx, {'loss': loss.item()})
        
        metrics = {
            'train_loss': total_loss / total_samples,
        }
        
        if self.config.dataset_info.task_type == "classification":
            accuracy = correct_predictions / total_samples
            metrics['train_acc'] = accuracy
            metrics['train_accuracy'] = accuracy  # Alternative key for compatibility
            
            # Log detailed training accuracy
            self.logger.info(f"🚀 Training - Loss: {metrics['train_loss']:.6f}, Accuracy: {accuracy:.6f} ({accuracy*100:.4f}%)")
        elif self.config.dataset_info.task_type == "segmentation" and seg_batches > 0:
            miou = seg_iou_sum / seg_batches
            dice = seg_dice_sum / seg_batches
            metrics['train_miou']     = miou
            metrics['train_dice']     = dice
            metrics['train_accuracy'] = miou  # compatibility alias
            self.logger.info(f"🚀 Training - Loss: {metrics['train_loss']:.6f}, mIoU: {miou:.4f} ({miou*100:.2f}%), Dice: {dice*100:.2f}%")
        elif self.config.dataset_info.task_type == "detection":
            self.logger.info(f"🚀 Training - Loss: {metrics['train_loss']:.6f} (detection)")
        
        self.callback_manager.on_train_epoch_end(metrics)
        return metrics
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Execute one validation epoch."""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        seg_iou_sum = 0.0
        seg_dice_sum = 0.0
        seg_batches = 0
        det_preds_all: list = []
        det_targets_all: list = []
        
        self.callback_manager.on_val_epoch_begin()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Move data to device
                inputs, targets = self._prepare_batch(batch)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self._compute_loss(outputs, targets)
                
                # Update statistics
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
                
                # Calculate accuracy for classification / collect detection preds
                if self.config.dataset_info.task_type == "classification":
                    _, predicted = torch.max(outputs.data, 1)
                    correct_predictions += (predicted == targets).sum().item()
                elif self.config.dataset_info.task_type == "segmentation":
                    seg_out = outputs['out'] if isinstance(outputs, dict) else outputs
                    miou, dice, _ = compute_segmentation_metrics(
                        seg_out, targets,
                        num_classes=self.config.dataset_info.num_classes
                    )
                    seg_iou_sum  += miou
                    seg_dice_sum += dice
                    seg_batches  += 1
                elif self.config.dataset_info.task_type == "detection" and isinstance(outputs, list):
                    det_preds_all.extend(outputs)
                    det_targets_all.extend(targets)

        metrics = {
            'val_loss': total_loss / max(total_samples, 1),
        }

        if self.config.dataset_info.task_type == "classification":
            accuracy = correct_predictions / total_samples
            metrics['val_acc'] = accuracy
            metrics['val_accuracy'] = accuracy  # Alternative key for compatibility
            
            # Log detailed validation accuracy
            self.logger.info(f"✅ Validation - Loss: {metrics['val_loss']:.6f}, Accuracy: {accuracy:.6f} ({accuracy*100:.4f}%)")
            
            # Check if this is a new best accuracy
            if accuracy > self.best_metric:
                improvement = accuracy - self.best_metric
                self.logger.info(f"🎉 New Best Validation Accuracy! Improved by {improvement*100:.4f}%")
                self.best_metric = accuracy

        elif self.config.dataset_info.task_type == "segmentation" and seg_batches > 0:
            miou = seg_iou_sum / seg_batches
            dice = seg_dice_sum / seg_batches
            metrics['val_miou']     = miou
            metrics['val_dice']     = dice
            metrics['val_accuracy'] = miou  # compatibility alias
            self.logger.info(f"✅ Validation - Loss: {metrics['val_loss']:.6f}, mIoU: {miou:.4f} ({miou*100:.2f}%), Dice: {dice*100:.2f}%")
            if miou > self.best_metric:
                self.logger.info(f"🎉 New Best mIoU: {miou*100:.4f}%!")
                self.best_metric = miou
        elif self.config.dataset_info.task_type == "detection" and det_preds_all:
            det_result = compute_detection_metrics(det_preds_all, det_targets_all, iou_threshold=0.5)
            map50 = det_result.get('mAP@50', 0.0)
            metrics['val_map50']   = map50
            metrics['val_accuracy'] = map50  # compatibility alias
            self.logger.info(f"✅ Validation - Loss: {metrics['val_loss']:.6f}, mAP@50: {map50:.4f} ({map50*100:.2f}%)")
            if map50 > self.best_metric:
                self.logger.info(f"🎉 New Best mAP@50: {map50*100:.4f}%!")
                self.best_metric = map50
        else:
            # Update best metric for classification / unknown tasks
            current_metric = metrics.get('val_acc', -metrics.get('val_loss', float('inf')))
            if current_metric > self.best_metric:
                self.best_metric = current_metric
        
        self.callback_manager.on_val_epoch_end(metrics)
        return metrics
    
    def _prepare_batch(self, batch):
        """Prepare batch data for training."""
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
        else:
            inputs, targets = batch, None

        # Stack inputs if they arrived as a tuple/list of tensors (detection collate)
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(list(inputs))
        inputs = inputs.to(self.device, non_blocking=True)

        if targets is not None:
            if isinstance(targets, torch.Tensor):
                targets = targets.to(self.device, non_blocking=True)
            elif isinstance(targets, (list, tuple)):
                # Detection targets: list of dicts with 'boxes', 'labels', etc.
                moved = []
                for t in targets:
                    if isinstance(t, dict):
                        moved.append({k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                       for k, v in t.items()})
                    elif isinstance(t, torch.Tensor):
                        moved.append(t.to(self.device))
                    else:
                        moved.append(t)
                targets = moved

        return inputs, targets
    
    def _compute_loss(self, outputs, targets) -> torch.Tensor:
        """Compute loss based on task type."""
        # Torchvision / SimpleDetector TRAIN mode: returns a dict of scalar loss tensors.
        if isinstance(outputs, dict) and any(k.startswith('loss_') for k in outputs):
            return sum(outputs.values())

        # Torchvision SEGMENTATION models (DeepLabV3, FCN): dict with 'out'.
        if isinstance(outputs, dict) and 'out' in outputs:
            main_out = outputs['out']
            loss = self.criterion(main_out, targets) if self.criterion else main_out.sum()
            if 'aux' in outputs and self.criterion:
                loss = loss + 0.4 * self.criterion(outputs['aux'], targets)
            return loss

        # Detection EVAL mode: model returns a list of prediction dicts — no loss.
        if isinstance(outputs, list):
            return torch.tensor(0.0, device=self.device)

        if self.criterion:
            return self.criterion(outputs, targets)
        return torch.tensor(0.0, device=self.device)
    
    def _generate_results(self) -> TrainingResults:
        """Generate final training results with detailed accuracy analysis."""
        training_time = time.time() - self.training_start_time
        
        # Get comprehensive accuracy summary
        accuracy_summary = self.metrics_tracker.get_accuracy_summary()
        
        # Get best metrics — use mIoU for segmentation, accuracy for classification
        best_val_loss, best_loss_epoch = self.metrics_tracker.get_best('val_loss', 'min')
        is_segmentation = self.config.dataset_info.task_type == 'segmentation'
        is_detection    = self.config.dataset_info.task_type == 'detection'
        if is_detection and 'val_map50' in self.metrics_tracker.history:
            best_val_acc, best_acc_epoch = self.metrics_tracker.get_best('val_map50', 'max')
        elif is_segmentation and 'val_miou' in self.metrics_tracker.history:
            best_val_acc, best_acc_epoch = self.metrics_tracker.get_best('val_miou', 'max')
        elif 'val_acc' in self.metrics_tracker.history:
            best_val_acc, best_acc_epoch = self.metrics_tracker.get_best('val_acc', 'max')
        else:
            best_val_acc, best_acc_epoch = 0, 0
        
        # Log final training summary
        self.logger.info("=" * 60)
        self.logger.info("🎉 TRAINING COMPLETED - Final Results Summary")
        self.logger.info("=" * 60)
        self.logger.info(f"⏱️  Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        self.logger.info(f"📊 Total Epochs: {self.metrics_tracker.epoch_count}")
        
        if accuracy_summary:
            if 'best_validation_accuracy' in accuracy_summary:
                best_acc_info = accuracy_summary['best_validation_accuracy']
                self.logger.info(f"🏆 Best Validation Accuracy: {best_acc_info['percentage']:.4f}% (epoch {best_acc_info['epoch']})")
            
            if 'current_validation_accuracy' in accuracy_summary:
                current_acc_info = accuracy_summary['current_validation_accuracy']
                self.logger.info(f"📈 Final Validation Accuracy: {current_acc_info['percentage']:.4f}%")
            
            if 'total_improvement' in accuracy_summary:
                improvement_info = accuracy_summary['total_improvement']
                self.logger.info(f"📊 Total Improvement: {improvement_info['percentage']:+.4f}%")
            
            if 'overfitting_analysis' in accuracy_summary:
                overfitting_info = accuracy_summary['overfitting_analysis']
                status_emoji = "✅" if overfitting_info['status'] == 'healthy' else "⚠️" if overfitting_info['status'] == 'monitoring' else "❌"
                self.logger.info(f"{status_emoji} Overfitting Status: {overfitting_info['status']} (gap: {overfitting_info['gap']:.4f}%)")
        
        # Use intelligent model path if available
        model_save_path = getattr(self, 'intelligent_model_path', self.config.output_dir / 'best_model.pth')
        
        self.logger.info(f"💾 Model saved to: {model_save_path}")
        if hasattr(self, 'model_base_name'):
            self.logger.info(f"📝 Model base name: {self.model_base_name}")
        self.logger.info("=" * 60)
        
        return TrainingResults(
            best_accuracy=self.best_metric,
            best_loss=best_val_loss,
            training_time=training_time,
            model_path=model_save_path,
            log_path=self.config.output_dir / 'logs',
            metrics_history=self.metrics_tracker.get_history(),
            final_model_state=self.model.state_dict()
        )