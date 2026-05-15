"""
Model factory for creating different architectures.
Supports torchvision pretrained models for classification,
and simple baselines for detection/segmentation.
"""

import logging
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torchvision.models as tv_models
    TORCH_AVAILABLE = True
except Exception:
    class nn:
        class Module: pass
    TORCH_AVAILABLE = False

from utils.config import Config

logger = logging.getLogger(__name__)

# Map architecture names (from ModelSelector) → torchvision builder
_CLASSIFICATION_ARCH_MAP = {
    # EfficientNet
    "efficientnet_b0": lambda w: tv_models.efficientnet_b0(weights=w),
    "efficientnet_b1": lambda w: tv_models.efficientnet_b1(weights=w),
    "efficientnet_b2": lambda w: tv_models.efficientnet_b2(weights=w),
    "efficientnet_b3": lambda w: tv_models.efficientnet_b3(weights=w),
    "efficientnet_b4": lambda w: tv_models.efficientnet_b4(weights=w),
    # MobileNet
    "mobilenet_v3_small": lambda w: tv_models.mobilenet_v3_small(weights=w),
    "mobilenet_v3_large": lambda w: tv_models.mobilenet_v3_large(weights=w),
    # ResNet
    "resnet18": lambda w: tv_models.resnet18(weights=w),
    "resnet34": lambda w: tv_models.resnet34(weights=w),
    "resnet50": lambda w: tv_models.resnet50(weights=w),
    "resnet101": lambda w: tv_models.resnet101(weights=w),
    # RegNet
    "regnet_y_400mf": lambda w: tv_models.regnet_y_400mf(weights=w),
    "regnet_y_800mf": lambda w: tv_models.regnet_y_800mf(weights=w),
    # ConvNeXt
    "convnext_tiny": lambda w: tv_models.convnext_tiny(weights=w),
    "convnext_small": lambda w: tv_models.convnext_small(weights=w),
    # ViT
    "vit_b_16": lambda w: tv_models.vit_b_16(weights=w),
}


class ModelFactory:
    """Factory for creating model architectures with optional pretrained weights."""

    def __init__(self, config: Config):
        self.config = config

    def create_model(self) -> nn.Module:
        arch = self.config.model_config.architecture
        task = self.config.dataset_info.task_type
        if task == "classification":
            return self._create_classification_model(arch)
        elif task == "detection":
            return self._create_detection_model(arch)
        elif task == "segmentation":
            return self._create_segmentation_model(arch)
        else:
            raise ValueError(f"Unsupported task type: {task}")

    # ──────────────────────────────────────────────────────────────────────────
    def _get_channels(self) -> int:
        cfg = self.config
        if hasattr(cfg, 'model_config') and hasattr(cfg.model_config, 'config_params'):
            ch = cfg.model_config.config_params.get('input_channels', None)
            if ch:
                return int(ch)
        if hasattr(cfg, 'dataset_info'):
            ch = getattr(cfg.dataset_info, 'channels', None)
            if ch:
                return int(ch)
            if hasattr(cfg.dataset_info, 'image_stats'):
                return int(cfg.dataset_info.image_stats.get('channels', 3))
        return 3

    # ──────────────────────────────────────────────────────────────────────────
    def _create_classification_model(self, architecture: str) -> nn.Module:
        num_classes = self.config.dataset_info.num_classes
        in_channels = self._get_channels()
        use_pretrained = getattr(self.config.model_config, 'pretrained', True)

        arch_key = architecture.lower().replace("-", "_").replace(" ", "_")
        builder = _CLASSIFICATION_ARCH_MAP.get(arch_key)

        if builder and TORCH_AVAILABLE:
            try:
                weights = "DEFAULT" if use_pretrained else None
                model = builder(weights)
                model = self._adapt_classifier_head(model, num_classes, in_channels, arch_key)
                logger.info(f"Created {architecture} ({'pretrained' if use_pretrained else 'random'}) "
                            f"with {num_classes} classes, {in_channels} input channels")
                return model
            except Exception as e:
                logger.warning(f"Could not build {architecture} via torchvision ({e}), falling back to AdaptiveCNN")

        # Fallback: custom adaptive CNN
        return self._build_adaptive_cnn(num_classes, in_channels)

    def _adapt_classifier_head(self, model, num_classes: int, in_channels: int, arch_key: str):
        """Replace the final classification head to match num_classes, and first conv if in_channels != 3."""
        # --- fix first conv for non-RGB inputs ---
        if in_channels != 3:
            if hasattr(model, 'features') and hasattr(model.features, '0'):
                first = model.features[0]
                if hasattr(first, '0') and isinstance(first[0], nn.Conv2d):
                    old = first[0]
                    first[0] = nn.Conv2d(in_channels, old.out_channels, old.kernel_size,
                                         old.stride, old.padding, bias=old.bias is not None)
            elif hasattr(model, 'conv1') and isinstance(model.conv1, nn.Conv2d):
                old = model.conv1
                model.conv1 = nn.Conv2d(in_channels, old.out_channels, old.kernel_size,
                                        old.stride, old.padding, bias=old.bias is not None)

        # --- replace head ---
        if hasattr(model, 'classifier'):
            head = model.classifier
            if isinstance(head, nn.Sequential):
                last_linear_idx = None
                for i, layer in enumerate(head):
                    if isinstance(layer, nn.Linear):
                        last_linear_idx = i
                if last_linear_idx is not None:
                    in_feat = head[last_linear_idx].in_features
                    head[last_linear_idx] = nn.Linear(in_feat, num_classes)
            elif isinstance(head, nn.Linear):
                model.classifier = nn.Linear(head.in_features, num_classes)
        elif hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif hasattr(model, 'heads') and hasattr(model.heads, 'head'):
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

        return model

    def _build_adaptive_cnn(self, num_classes: int, in_channels: int) -> nn.Module:
        """Simple adaptive CNN baseline (used when torchvision arch is unavailable)."""
        class AdaptiveCNN(nn.Module):
            def __init__(self, num_classes, in_ch):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(0.5),
                    nn.Linear(128, 256), nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes),
                )

            def forward(self, x):
                return self.classifier(self.features(x))

        logger.info(f"Using AdaptiveCNN fallback ({in_channels} ch → {num_classes} classes)")
        return AdaptiveCNN(num_classes, in_channels)

    # ──────────────────────────────────────────────────────────────────────────
    def _create_detection_model(self, architecture: str) -> nn.Module:
        num_classes = self.config.dataset_info.num_classes
        in_channels = self._get_channels()

        if TORCH_AVAILABLE and in_channels == 3:
            try:
                arch_key = architecture.lower()
                if "fcos" in arch_key:
                    from torchvision.models.detection import fcos_resnet50_fpn
                    model = fcos_resnet50_fpn(weights=None, num_classes=num_classes)
                else:
                    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
                    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, num_classes=num_classes)
                logger.info(f"Created detection model: {architecture}")
                return model
            except Exception as e:
                logger.warning(f"Detection model build failed ({e}), using simple baseline")

        class SimpleDetector(nn.Module):
            def __init__(self, num_classes, in_ch):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(in_ch, 64, 3, padding=1), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                )
                self.head = nn.Linear(64, num_classes * 5)  # (cls + 4 bbox coords)

            def forward(self, x):
                return self.head(self.backbone(x))

        return SimpleDetector(num_classes, in_channels)

    # ──────────────────────────────────────────────────────────────────────────
    def _create_segmentation_model(self, architecture: str) -> nn.Module:
        num_classes = self.config.dataset_info.num_classes
        in_channels = self._get_channels()

        if TORCH_AVAILABLE and in_channels == 3:
            try:
                arch_key = architecture.lower()
                if "deeplabv3" in arch_key:
                    from torchvision.models.segmentation import deeplabv3_resnet50
                    model = deeplabv3_resnet50(weights=None, num_classes=num_classes)
                else:
                    from torchvision.models.segmentation import fcn_resnet50
                    model = fcn_resnet50(weights=None, num_classes=num_classes)
                logger.info(f"Created segmentation model: {architecture}")
                return model
            except Exception as e:
                logger.warning(f"Segmentation model build failed ({e}), using simple baseline")

        class SimpleSegmenter(nn.Module):
            def __init__(self, num_classes, in_ch):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(in_ch, 64, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
                )
                self.head = nn.Conv2d(32, num_classes, 1)

            def forward(self, x):
                return self.head(self.encoder(x))

        return SimpleSegmenter(num_classes, in_channels)