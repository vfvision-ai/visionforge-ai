"""
Metrics tracking utilities with detailed accuracy analysis.
"""

from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import logging


def compute_detection_metrics(preds, targets, iou_threshold: float = 0.5):
    """Compute mAP@50 for a batch of detections.

    Args:
        preds:   list of dicts with keys 'boxes' (N,4), 'labels' (N,), 'scores' (N,)
        targets: list of dicts with keys 'boxes' (M,4), 'labels' (M,)
        iou_threshold: IoU threshold for a true positive (default 0.5)

    Returns:
        map50 (float): mean Average Precision @ IoU 0.50
        per_class_ap (dict): AP per class index
    """
    try:
        import torch

        # Collect per-class TP/FP/scores over the whole batch
        class_data: dict = {}  # class_id → {'scores': [], 'tp': [], 'n_gt': 0}

        for pred, tgt in zip(preds, targets):
            gt_boxes  = tgt['boxes']   # (M, 4)
            gt_labels = tgt['labels']  # (M,)

            pred_boxes  = pred.get('boxes',  torch.zeros(0, 4))
            pred_labels = pred.get('labels', torch.zeros(0, dtype=torch.long))
            pred_scores = pred.get('scores', torch.zeros(0))

            # Track which gt boxes have been matched
            matched_gt = set()

            for cls in gt_labels.unique().tolist():
                cls = int(cls)
                if cls not in class_data:
                    class_data[cls] = {'scores': [], 'tp': [], 'n_gt': 0}
                class_data[cls]['n_gt'] += int((gt_labels == cls).sum().item())

            # Sort predictions by score descending
            if len(pred_scores) > 0:
                order = pred_scores.argsort(descending=True)
                pred_boxes  = pred_boxes[order]
                pred_labels = pred_labels[order]
                pred_scores = pred_scores[order]

            for pb, pl, ps in zip(pred_boxes, pred_labels, pred_scores):
                cls = int(pl.item())
                if cls not in class_data:
                    class_data[cls] = {'scores': [], 'tp': [], 'n_gt': 0}
                class_data[cls]['scores'].append(ps.item())

                # Find matching gt box of same class
                gt_cls_mask = gt_labels == cls
                gt_cls_boxes = gt_boxes[gt_cls_mask]
                gt_cls_idxs  = gt_cls_mask.nonzero(as_tuple=False).squeeze(1).tolist()

                matched = False
                best_iou = 0.0
                best_gt_idx = -1
                for i, (gbi, gb) in enumerate(zip(gt_cls_idxs, gt_cls_boxes)):
                    if gbi in matched_gt:
                        continue
                    iou = _box_iou(pb.unsqueeze(0), gb.unsqueeze(0)).item()
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gbi

                if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
                    class_data[cls]['tp'].append(1)
                    matched_gt.add(best_gt_idx)
                    matched = True
                if not matched:
                    class_data[cls]['tp'].append(0)

        # Compute AP per class
        per_class_ap = {}
        for cls, data in class_data.items():
            if data['n_gt'] == 0 or not data['scores']:
                per_class_ap[cls] = 0.0
                continue
            import numpy as np
            scores = np.array(data['scores'])
            tp     = np.array(data['tp'])
            order  = np.argsort(-scores)
            tp     = tp[order]
            fp     = 1 - tp
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            precision = cum_tp / (cum_tp + cum_fp + 1e-9)
            recall    = cum_tp / (data['n_gt'] + 1e-9)
            # 11-point interpolation
            ap = 0.0
            for thr in np.linspace(0, 1, 11):
                p = precision[recall >= thr].max() if (recall >= thr).any() else 0.0
                ap += p / 11.0
            per_class_ap[cls] = float(ap)

        map50 = float(sum(per_class_ap.values()) / len(per_class_ap)) if per_class_ap else 0.0
        return map50, per_class_ap

    except Exception:
        return 0.0, {}


def _box_iou(box1, box2):
    """Compute IoU between two boxes in (x1,y1,x2,y2) format."""
    try:
        import torch
        inter_x1 = torch.max(box1[:, 0], box2[:, 0])
        inter_y1 = torch.max(box1[:, 1], box2[:, 1])
        inter_x2 = torch.min(box1[:, 2], box2[:, 2])
        inter_y2 = torch.min(box1[:, 3], box2[:, 3])
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        return inter_area / (area1 + area2 - inter_area + 1e-9)
    except Exception:
        import torch
        return torch.tensor(0.0)


def compute_segmentation_metrics(preds, targets, num_classes: int, ignore_index: int = 255):
    """Compute per-class IoU, mean IoU, and mean Dice for a batch.

    Args:
        preds:  (B, C, H, W) logits  **or**  (B, H, W) integer class map
        targets: (B, H, W) integer class map
        num_classes: number of foreground + background classes
        ignore_index: pixel value to exclude (e.g. 255 for Pascal VOC border)

    Returns:
        miou (float), mean_dice (float), iou_per_class (List[float])
    """
    try:
        import torch
        if preds.dim() == 4:
            preds = preds.argmax(dim=1)          # (B, H, W)

        valid_mask = targets != ignore_index     # exclude void/border pixels

        iou_per_class: List[float] = []
        dice_per_class: List[float] = []

        for cls in range(num_classes):
            pred_c   = (preds   == cls) & valid_mask
            target_c = (targets == cls) & valid_mask

            intersection = (pred_c & target_c).sum().float()
            union        = (pred_c | target_c).sum().float()

            if union > 0:
                iou_per_class.append((intersection / union).item())

            denom = pred_c.sum().float() + target_c.sum().float()
            if denom > 0:
                dice_per_class.append((2.0 * intersection / denom).item())

        miou      = sum(iou_per_class)  / len(iou_per_class)  if iou_per_class  else 0.0
        mean_dice = sum(dice_per_class) / len(dice_per_class) if dice_per_class else 0.0
        return miou, mean_dice, iou_per_class
    except Exception:
        return 0.0, 0.0, []


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
        """Calculate additional accuracy / IoU metrics and analysis."""
        enhanced = metrics.copy()

        # ── Detection path ──────────────────────────────────────────────────
        if 'val_map50' in metrics:
            if len(self.history['val_map50']) > 0:
                enhanced['val_map50_improvement'] = metrics['val_map50'] - self.history['val_map50'][-1]
            return enhanced

        # ── Segmentation path ──────────────────────────────────────────────
        if 'val_miou' in metrics:
            if len(self.history['val_miou']) > 0:
                enhanced['val_miou_improvement'] = metrics['val_miou'] - self.history['val_miou'][-1]
            if len(self.history['val_miou']) >= 2:
                recent = self.history['val_miou'][-(min(3, len(self.history['val_miou']))):] + [metrics['val_miou']]
                enhanced['val_miou_trend'] = sum(recent) / len(recent)
            return enhanced

        # ── Classification path (original) ────────────────────────────────
        if 'val_acc' in metrics and len(self.history['val_acc']) > 0:
            enhanced['val_acc_improvement'] = metrics['val_acc'] - self.history['val_acc'][-1]

        if 'train_acc' in metrics and len(self.history['train_acc']) > 0:
            enhanced['train_acc_improvement'] = metrics['train_acc'] - self.history['train_acc'][-1]

        if 'train_acc' in metrics and 'val_acc' in metrics:
            enhanced['overfitting_gap'] = metrics['train_acc'] - metrics['val_acc']

        if 'val_loss' in metrics and len(self.history['val_loss']) > 0:
            enhanced['val_loss_improvement'] = self.history['val_loss'][-1] - metrics['val_loss']

        if 'val_acc' in metrics and 'val_loss' in metrics and metrics['val_loss'] > 0:
            enhanced['efficiency_ratio'] = metrics['val_acc'] / metrics['val_loss']

        if len(self.history['val_acc']) >= 2:
            recent_epochs = min(3, len(self.history['val_acc']) + 1)
            recent_acc = self.history['val_acc'][-(recent_epochs - 1):] + [metrics.get('val_acc', 0)]
            enhanced['val_acc_trend'] = sum(recent_acc) / len(recent_acc)

        return enhanced
    
    def _log_accuracy_details(self, metrics: Dict[str, float]):
        """Log detailed accuracy / IoU information."""
        epoch = metrics.get('epoch', self.epoch_count)

        # Segmentation metrics
        if 'val_miou' in metrics or 'train_miou' in metrics:
            train_miou = metrics.get('train_miou', 0) * 100
            val_miou   = metrics.get('val_miou',   0) * 100
            train_dice = metrics.get('train_dice', 0) * 100
            val_dice   = metrics.get('val_dice',   0) * 100
            self.logger.info(f"📊 Epoch {epoch} Segmentation Details:")
            self.logger.info(f"   🎯 Train mIoU: {train_miou:.4f}%  |  Dice: {train_dice:.4f}%")
            self.logger.info(f"   ✅ Val   mIoU: {val_miou:.4f}%   |  Dice: {val_dice:.4f}%")
            if 'val_miou_improvement' in metrics:
                imp = metrics['val_miou_improvement'] * 100
                trend = "📈" if imp > 0 else "📉" if imp < 0 else "➡️"
                self.logger.info(f"   {trend} Val mIoU Change: {imp:+.4f}%")
            return

        # Detection metrics
        if 'val_map50' in metrics or 'train_map50' in metrics:
            t_map = metrics.get('train_map50', 0) * 100
            v_map = metrics.get('val_map50',   0) * 100
            self.logger.info(f"📊 Epoch {epoch} Detection Details:")
            self.logger.info(f"   🎯 Train mAP@50: {t_map:.4f}%")
            self.logger.info(f"   ✅ Val   mAP@50: {v_map:.4f}%")
            if 'val_map50_improvement' in metrics:
                imp = metrics['val_map50_improvement'] * 100
                trend = "📈" if imp > 0 else "📉" if imp < 0 else "➡️"
                self.logger.info(f"   {trend} Val mAP@50 Change: {imp:+.4f}%")
            return

        # Classification metrics (original)
        train_acc = metrics.get('train_acc', 0) * 100
        val_acc   = metrics.get('val_acc',   0) * 100

        self.logger.info(f"📊 Epoch {epoch} Accuracy Details:")
        self.logger.info(f"   🎯 Training Accuracy: {train_acc:.4f}%")
        self.logger.info(f"   ✅ Validation Accuracy: {val_acc:.4f}%")

        if 'val_acc_improvement' in metrics:
            improvement = metrics['val_acc_improvement'] * 100
            trend = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
            self.logger.info(f"   {trend} Val Accuracy Change: {improvement:+.4f}%")

        if 'overfitting_gap' in metrics:
            gap = metrics['overfitting_gap'] * 100
            if gap > 0.1:
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
        """Get comprehensive accuracy / IoU summary."""
        summary = {}

        # ── Detection summary ─────────────────────────────────────────────
        if 'val_map50' in self.history and self.history['val_map50']:
            best_map, best_epoch = self.get_best('val_map50', 'max')
            summary['best_val_map50'] = {'value': best_map, 'percentage': best_map * 100, 'epoch': best_epoch}
            summary['current_val_map50'] = {'value': self.get_latest('val_map50'), 'percentage': self.get_latest('val_map50') * 100}
            if len(self.history['val_map50']) >= 2:
                improvement = self.history['val_map50'][-1] - self.history['val_map50'][0]
                summary['total_improvement'] = {'value': improvement, 'percentage': improvement * 100}
            return summary

        # ── Segmentation summary ──────────────────────────────────────────
        if 'val_miou' in self.history and self.history['val_miou']:
            best_miou, best_epoch = self.get_best('val_miou', 'max')
            summary['best_val_miou'] = {'value': best_miou, 'percentage': best_miou * 100, 'epoch': best_epoch}
            summary['current_val_miou'] = {'value': self.get_latest('val_miou'), 'percentage': self.get_latest('val_miou') * 100}

            if 'val_dice' in self.history and self.history['val_dice']:
                best_dice, _ = self.get_best('val_dice', 'max')
                summary['best_val_dice'] = {'value': best_dice, 'percentage': best_dice * 100}

            if len(self.history['val_miou']) >= 2:
                improvement = self.history['val_miou'][-1] - self.history['val_miou'][0]
                summary['total_improvement'] = {'value': improvement, 'percentage': improvement * 100}
            return summary

        # ── Classification summary (original) ─────────────────────────────
        if 'val_acc' in self.history:
            best_val_acc, best_val_epoch = self.get_best('val_acc', 'max')
            summary['best_validation_accuracy'] = {'value': best_val_acc, 'percentage': best_val_acc * 100, 'epoch': best_val_epoch}

        if 'train_acc' in self.history:
            best_train_acc, best_train_epoch = self.get_best('train_acc', 'max')
            summary['best_training_accuracy'] = {'value': best_train_acc, 'percentage': best_train_acc * 100, 'epoch': best_train_epoch}

        if 'val_acc' in self.history:
            current_val_acc = self.get_latest('val_acc')
            summary['current_validation_accuracy'] = {'value': current_val_acc, 'percentage': current_val_acc * 100}

        if 'train_acc' in self.history:
            current_train_acc = self.get_latest('train_acc')
            summary['current_training_accuracy'] = {'value': current_train_acc, 'percentage': current_train_acc * 100}

        if 'val_acc' in self.history and len(self.history['val_acc']) >= 2:
            improvement = self.history['val_acc'][-1] - self.history['val_acc'][0]
            summary['total_improvement'] = {'value': improvement, 'percentage': improvement * 100}

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