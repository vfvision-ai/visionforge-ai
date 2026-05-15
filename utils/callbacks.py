"""
Training callbacks for monitoring and control across all frameworks.
Universal callback system for PyTorch, TensorFlow, and Scikit-learn.
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import time
import numpy as np


class Callback:
    """Base callback class with enhanced framework support."""
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each epoch."""
        pass
    
    def on_train_epoch_begin(self):
        pass
    
    def on_train_epoch_end(self, metrics: Dict[str, float]):
        pass
    
    def on_val_epoch_begin(self):
        pass
    
    def on_val_epoch_end(self, metrics: Dict[str, float]):
        pass
    
    def on_train_batch_begin(self, batch_idx: int):
        pass
    
    def on_train_batch_end(self, batch_idx: int, metrics: Dict[str, float]):
        pass


class CallbackManager:
    """Universal callback manager for all frameworks."""
    
    def __init__(self, framework: str = "PyTorch"):
        self.callbacks: List[Callback] = []
        self._stop_training = False
        self.framework = framework
        self.training_logs = []
        
    def add_callback(self, callback: Callback):
        """Add a callback."""
        self.callbacks.append(callback)
        
    def add_callbacks(self, callbacks: List[Callback]):
        """Add multiple callbacks."""
        for callback in callbacks:
            self.add_callback(callback)
    
    def should_stop_training(self) -> bool:
        """Check if training should be stopped."""
        return self._stop_training
    
    def stop_training(self):
        """Signal to stop training."""
        self._stop_training = True
        
    def reset(self):
        """Reset callback manager state."""
        self._stop_training = False
        self.training_logs.clear()
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Call on_train_begin for all callbacks."""
        for callback in self.callbacks:
            try:
                callback.on_train_begin(logs)
            except Exception as e:
                print(f"Warning: Callback {callback.__class__.__name__} failed in on_train_begin: {e}")
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            try:
                callback.on_train_end(logs)
            except Exception as e:
                print(f"Warning: Callback {callback.__class__.__name__} failed in on_train_end: {e}")
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Call on_epoch_begin for all callbacks."""
        for callback in self.callbacks:
            try:
                callback.on_epoch_begin(epoch, logs)
            except Exception as e:
                print(f"Warning: Callback {callback.__class__.__name__} failed in on_epoch_begin: {e}")
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], logs: Optional[Dict[str, Any]] = None):
        """Call on_epoch_end for all callbacks."""
        # Store training logs
        epoch_log = {'epoch': epoch, 'metrics': metrics, 'timestamp': time.time()}
        if logs:
            epoch_log.update(logs)
        self.training_logs.append(epoch_log)
        
        for callback in self.callbacks:
            try:
                result = callback.on_epoch_end(epoch, metrics, logs)
                # Handle early stopping signal
                if result is True:  # Callback requests training stop
                    self._stop_training = True
            except Exception as e:
                print(f"Warning: Callback {callback.__class__.__name__} failed in on_epoch_end: {e}")
    
    def on_train_epoch_begin(self):
        for callback in self.callbacks:
            callback.on_train_epoch_begin()
    
    def on_train_epoch_end(self, metrics: Dict[str, float]):
        for callback in self.callbacks:
            callback.on_train_epoch_end(metrics)
    
    def on_val_epoch_begin(self):
        for callback in self.callbacks:
            callback.on_val_epoch_begin()
    
    def on_val_epoch_end(self, metrics: Dict[str, float]):
        for callback in self.callbacks:
            callback.on_val_epoch_end(metrics)
    
    def on_train_batch_begin(self, batch_idx: int):
        for callback in self.callbacks:
            callback.on_train_batch_begin(batch_idx)
    
    def on_train_batch_end(self, batch_idx: int, metrics: Dict[str, float]):
        for callback in self.callbacks:
            callback.on_train_batch_end(batch_idx, metrics)


class EarlyStopping(Callback):
    """Universal early stopping callback for all frameworks."""
    
    def __init__(self, monitor: str = 'val_acc', patience: int = 10, 
                 min_delta: float = 0.001, mode: str = 'max', verbose: bool = True):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.best_score = None
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = False
        
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Reset early stopping state."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = None
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], logs: Optional[Dict[str, Any]] = None):
        """Check if training should stop based on monitored metric."""
        current = metrics.get(self.monitor)
        if current is None:
            if self.verbose:
                print(f"Warning: Early stopping metric '{self.monitor}' not found in metrics: {list(metrics.keys())}")
            return False
        
        if self.best_score is None:
            self.best_score = current
            if self.verbose:
                print(f"EarlyStopping: Initial {self.monitor}: {current:.6f}")
        elif self._is_improvement(current, self.best_score):
            self.best_score = current
            self.wait = 0
            if self.verbose:
                print(f"EarlyStopping: {self.monitor} improved to {current:.6f}")
        else:
            self.wait += 1
            if self.verbose:
                print(f"EarlyStopping: {self.monitor} did not improve from {self.best_score:.6f}, waiting {self.wait}/{self.patience}")
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.verbose:
                    print(f"EarlyStopping: Stopping training at epoch {epoch+1}. Best {self.monitor}: {self.best_score:.6f}")
                return True  # Signal to stop
        
        return False
    
    def _is_improvement(self, current: float, best: float) -> bool:
        """Check if current value is an improvement over best."""
        if self.mode == 'max':
            return current > best + self.min_delta
        else:
            return current < best - self.min_delta


class ModelCheckpoint(Callback):
    """Universal model checkpointing callback for all frameworks."""
    
    def __init__(self, filepath: Union[str, Path], monitor: str = 'val_acc', 
                 save_best_only: bool = True, mode: str = 'max', verbose: bool = True):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.verbose = verbose
        
        self.best_score = None
        self.model_to_save = None  # Will be set by trainer
        
    def set_model(self, model):
        """Set the model to save (called by trainer)."""
        self.model_to_save = model
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], logs: Optional[Dict[str, Any]] = None):
        """Save model if criteria are met."""
        current = metrics.get(self.monitor)
        if current is None:
            if self.verbose:
                print(f"Warning: ModelCheckpoint metric '{self.monitor}' not found")
            return False
        
        should_save = not self.save_best_only or self._is_improvement(current)
        
        if should_save and self.model_to_save is not None:
            # Create directory if it doesn't exist
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Save based on framework type
            try:
                if hasattr(self.model_to_save, 'save'):  # TensorFlow/Keras
                    self.model_to_save.save(str(self.filepath))
                elif hasattr(self.model_to_save, 'state_dict'):  # PyTorch
                    import torch
                    torch.save(self.model_to_save.state_dict(), str(self.filepath))
                else:  # Scikit-learn
                    import joblib
                    joblib.dump(self.model_to_save, str(self.filepath))
                
                if self.verbose:
                    print(f"ModelCheckpoint: Saved model to {self.filepath} (epoch {epoch+1}, {self.monitor}: {current:.6f})")
                    
            except Exception as e:
                print(f"Warning: ModelCheckpoint failed to save model: {e}")
        
        return False
    
    def _is_improvement(self, current: float) -> bool:
        """Check if current value is an improvement over best."""
        if self.best_score is None:
            self.best_score = current
            return True
        
        if self.mode == 'max':
            improved = current > self.best_score
        else:
            improved = current < self.best_score
        
        if improved:
            self.best_score = current
        
        return improved


class LearningRateMonitor(Callback):
    """Monitor learning rate changes."""
    
    def __init__(self):
        self.lr_history = []
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        # Get current learning rate (implementation depends on optimizer)
        pass


class ProgressBar(Callback):
    """Enhanced progress bar callback for all frameworks."""
    
    def __init__(self, verbose: bool = True):
        self.current_epoch = 0
        self.total_epochs = 0
        self.verbose = verbose
        self.start_time = None
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Initialize training progress."""
        if self.verbose:
            print("🚀 Starting training...")
            self.start_time = time.time()
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Display epoch start."""
        self.current_epoch = epoch
        if self.verbose:
            print(f"\n📊 Epoch {epoch + 1}/{logs.get('total_epochs', '?') if logs else '?'}")
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], logs: Optional[Dict[str, Any]] = None):
        """Display epoch results."""
        if self.verbose:
            # Format metrics for display
            metric_str = " - ".join([f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}" 
                                   for k, v in metrics.items()])
            print(f"Epoch {epoch + 1} - {metric_str}")
            
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Display training completion."""
        if self.verbose and self.start_time:
            elapsed = time.time() - self.start_time
            print(f"✅ Training completed in {elapsed:.2f}s")


class MetricsLogger(Callback):
    """Enhanced metrics logger for all frameworks."""
    
    def __init__(self, log_dir: Union[str, Path], log_interval: int = 1, 
                 framework: str = "PyTorch"):
        self.log_dir = Path(log_dir)
        self.log_interval = log_interval
        self.framework = framework
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_history = []
        self.best_metrics = {}
        
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Initialize logging."""
        self.metrics_history.clear()
        self.best_metrics.clear()
        
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], logs: Optional[Dict[str, Any]] = None):
        """Log epoch metrics."""
        # Add epoch number and timestamp to metrics
        epoch_metrics = {
            'epoch': epoch,
            'timestamp': time.time(),
            'framework': self.framework,
            **metrics
        }
        
        if logs:
            epoch_metrics.update(logs)
            
        self.metrics_history.append(epoch_metrics)
        
        # Track best metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if key not in self.best_metrics:
                    self.best_metrics[key] = {'value': value, 'epoch': epoch}
                else:
                    # Assume higher is better for acc/accuracy, lower is better for loss
                    if 'acc' in key.lower():
                        if value > self.best_metrics[key]['value']:
                            self.best_metrics[key] = {'value': value, 'epoch': epoch}
                    elif 'loss' in key.lower():
                        if value < self.best_metrics[key]['value']:
                            self.best_metrics[key] = {'value': value, 'epoch': epoch}
        
        # Save to file every log_interval epochs
        if (epoch + 1) % self.log_interval == 0:
            self._save_metrics()
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Save final metrics."""
        self._save_metrics()
        self._save_best_metrics()
    
    def _save_metrics(self):
        """Save metrics history to JSON file."""
        try:
            metrics_file = self.log_dir / f'{self.framework.lower()}_training_metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Failed to save metrics: {e}")
    
    def _save_best_metrics(self):
        """Save best metrics summary."""
        try:
            best_file = self.log_dir / f'{self.framework.lower()}_best_metrics.json'
            with open(best_file, 'w') as f:
                json.dump(self.best_metrics, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Failed to save best metrics: {e}")


class LearningRateScheduler(Callback):
    """Learning rate scheduling callback."""
    
    def __init__(self, schedule_func, verbose: bool = True):
        self.schedule_func = schedule_func
        self.verbose = verbose
        self.lr_history = []
        
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], logs: Optional[Dict[str, Any]] = None):
        """Update learning rate based on schedule."""
        try:
            new_lr = self.schedule_func(epoch, metrics)
            self.lr_history.append({'epoch': epoch, 'lr': new_lr})
            
            if self.verbose:
                print(f"LearningRateScheduler: Updated learning rate to {new_lr:.6f}")
                
            # Store in logs for trainer to use
            if logs is not None:
                logs['new_lr'] = new_lr
                
        except Exception as e:
            print(f"Warning: LearningRateScheduler failed: {e}")


class PerformanceMonitor(Callback):
    """Monitor training performance and detect issues."""
    
    def __init__(self, patience: int = 5, verbose: bool = True):
        self.patience = patience
        self.verbose = verbose
        self.loss_history = []
        self.accuracy_history = []
        self.stagnation_count = 0
        
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], logs: Optional[Dict[str, Any]] = None):
        """Monitor performance metrics."""
        # Track loss
        train_loss = metrics.get('train_loss', metrics.get('loss'))
        if train_loss is not None:
            self.loss_history.append(train_loss)
            
        # Track accuracy  
        train_acc = metrics.get('train_acc', metrics.get('train_accuracy', metrics.get('accuracy')))
        if train_acc is not None:
            self.accuracy_history.append(train_acc)
            
        # Detect issues
        if len(self.loss_history) >= 3:
            self._detect_issues(epoch, metrics)
    
    def _detect_issues(self, epoch: int, metrics: Dict[str, float]):
        """Detect common training issues."""
        if len(self.loss_history) < 3:
            return
            
        recent_losses = self.loss_history[-3:]
        
        # Check for exploding gradients
        if recent_losses[-1] > recent_losses[-2] * 2:
            if self.verbose:
                print(f"⚠️ Warning: Possible exploding gradients detected at epoch {epoch+1}")
                
        # Check for stagnation
        if len(self.loss_history) >= 5:
            recent_5 = self.loss_history[-5:]
            if max(recent_5) - min(recent_5) < 0.001:
                self.stagnation_count += 1
                if self.stagnation_count >= self.patience and self.verbose:
                    print(f"⚠️ Warning: Training appears to have stagnated (epoch {epoch+1})")
            else:
                self.stagnation_count = 0