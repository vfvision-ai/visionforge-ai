"""
Metrics tracking utilities with detailed accuracy analysis.
"""

from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import logging


class MetricsTracker:
    """Track training metrics over time with detailed accuracy analysis."""
    
    def __init__(self):
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.current_metrics: Dict[str, float] = {}
        self.epoch_count = 0
        self.logger = logging.getLogger(__name__)
    
    def update(self, metrics: Dict[str, float]):
        """Update metrics for current step with enhanced accuracy tracking."""
        self.epoch_count += 1
        self.current_metrics = metrics.copy()
        
        # Add epoch number to metrics
        metrics_with_epoch = metrics.copy()
        metrics_with_epoch['epoch'] = self.epoch_count
        
        # Calculate additional derived metrics
        enhanced_metrics = self._calculate_enhanced_metrics(metrics_with_epoch)
        
        # Store all metrics
        for key, value in enhanced_metrics.items():
            self.history[key].append(value)
        
        # Log detailed accuracy information
        self._log_accuracy_details(enhanced_metrics)
    
    def _calculate_enhanced_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate additional accuracy metrics and analysis."""
        enhanced = metrics.copy()
        
        # Calculate accuracy improvements
        if 'val_acc' in metrics and len(self.history['val_acc']) > 0:
            prev_val_acc = self.history['val_acc'][-1]
            enhanced['val_acc_improvement'] = metrics['val_acc'] - prev_val_acc
        
        if 'train_acc' in metrics and len(self.history['train_acc']) > 0:
            prev_train_acc = self.history['train_acc'][-1]
            enhanced['train_acc_improvement'] = metrics['train_acc'] - prev_train_acc
        
        # Calculate overfitting indicator (train vs val accuracy gap)
        if 'train_acc' in metrics and 'val_acc' in metrics:
            enhanced['overfitting_gap'] = metrics['train_acc'] - metrics['val_acc']
        
        # Calculate loss improvement
        if 'val_loss' in metrics and len(self.history['val_loss']) > 0:
            prev_val_loss = self.history['val_loss'][-1]
            enhanced['val_loss_improvement'] = prev_val_loss - metrics['val_loss']  # Positive is better
        
        # Calculate training efficiency (accuracy/loss ratio)
        if 'val_acc' in metrics and 'val_loss' in metrics and metrics['val_loss'] > 0:
            enhanced['efficiency_ratio'] = metrics['val_acc'] / metrics['val_loss']
        
        # Calculate learning trend (moving average of last 3 epochs)
        if len(self.history['val_acc']) >= 2:
            recent_epochs = min(3, len(self.history['val_acc']) + 1)
            recent_acc = self.history['val_acc'][-(recent_epochs-1):] + [metrics.get('val_acc', 0)]
            enhanced['val_acc_trend'] = sum(recent_acc) / len(recent_acc)
        
        return enhanced
    
    def _log_accuracy_details(self, metrics: Dict[str, float]):
        """Log detailed accuracy information."""
        epoch = metrics.get('epoch', self.epoch_count)
        
        # Basic accuracy info
        train_acc = metrics.get('train_acc', 0) * 100
        val_acc = metrics.get('val_acc', 0) * 100
        
        self.logger.info(f"📊 Epoch {epoch} Accuracy Details:")
        self.logger.info(f"   🎯 Training Accuracy: {train_acc:.4f}%")
        self.logger.info(f"   ✅ Validation Accuracy: {val_acc:.4f}%")
        
        # Improvement tracking
        if 'val_acc_improvement' in metrics:
            improvement = metrics['val_acc_improvement'] * 100
            trend = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
            self.logger.info(f"   {trend} Val Accuracy Change: {improvement:+.4f}%")
        
        # Overfitting warning
        if 'overfitting_gap' in metrics:
            gap = metrics['overfitting_gap'] * 100
            if gap > 0.1:  # 10% gap indicates potential overfitting
                self.logger.warning(f"   ⚠️ Overfitting Gap: {gap:.4f}% (Train-Val)")
            else:
                self.logger.info(f"   ✅ Healthy Gap: {gap:.4f}% (Train-Val)")
    
    def get_current(self, key: str) -> float:
        """Get current value for a metric."""
        return self.current_metrics.get(key, 0.0)
    
    def get_history(self, key: str = None) -> Dict[str, List[float]]:
        """Get history for a specific metric or all metrics."""
        if key:
            return {key: self.history.get(key, [])}
        return dict(self.history)
    
    def get_best(self, key: str, mode: str = 'max') -> Tuple[float, int]:
        """Get best value for a metric and the epoch it occurred."""
        if key not in self.history or not self.history[key]:
            return 0.0, 0
        
        values = self.history[key]
        if mode == 'max':
            best_value = max(values)
            best_epoch = values.index(best_value) + 1
        else:
            best_value = min(values)
            best_epoch = values.index(best_value) + 1
        
        return best_value, best_epoch
    
    def get_latest(self, key: str) -> float:
        """Get latest value for a metric."""
        if key not in self.history or not self.history[key]:
            return 0.0
        return self.history[key][-1]
    
    def get_accuracy_summary(self) -> Dict[str, Any]:
        """Get comprehensive accuracy summary."""
        summary = {}
        
        # Best accuracies
        if 'val_acc' in self.history:
            best_val_acc, best_val_epoch = self.get_best('val_acc', 'max')
            summary['best_validation_accuracy'] = {
                'value': best_val_acc,
                'percentage': best_val_acc * 100,
                'epoch': best_val_epoch
            }
        
        if 'train_acc' in self.history:
            best_train_acc, best_train_epoch = self.get_best('train_acc', 'max')
            summary['best_training_accuracy'] = {
                'value': best_train_acc,
                'percentage': best_train_acc * 100,
                'epoch': best_train_epoch
            }
        
        # Current accuracies
        if 'val_acc' in self.history:
            current_val_acc = self.get_latest('val_acc')
            summary['current_validation_accuracy'] = {
                'value': current_val_acc,
                'percentage': current_val_acc * 100
            }
        
        if 'train_acc' in self.history:
            current_train_acc = self.get_latest('train_acc')
            summary['current_training_accuracy'] = {
                'value': current_train_acc,
                'percentage': current_train_acc * 100
            }
        
        # Learning progress
        if 'val_acc' in self.history and len(self.history['val_acc']) >= 2:
            first_acc = self.history['val_acc'][0]
            latest_acc = self.history['val_acc'][-1]
            improvement = latest_acc - first_acc
            summary['total_improvement'] = {
                'value': improvement,
                'percentage': improvement * 100
            }
        
        # Overfitting analysis
        if 'overfitting_gap' in self.history:
            latest_gap = self.get_latest('overfitting_gap')
            summary['overfitting_analysis'] = {
                'gap': latest_gap * 100,
                'status': 'healthy' if latest_gap < 0.1 else 'overfitting' if latest_gap > 0.2 else 'monitoring'
            }
        
        return summary
    
    def reset(self):
        """Reset all metrics."""
        self.history.clear()
        self.current_metrics.clear()
        self.epoch_count = 0