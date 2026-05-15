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
            """Minimal detection model whose API mirrors torchvision detection models.

            Train mode  (with targets): returns a dict of loss tensors.
            Eval  mode  (no targets)  : returns a list of prediction dicts
                                        [{'boxes': (N,4), 'labels': (N,), 'scores': (N,)}].
            """
            def __init__(self, n_cls, in_ch):
                super().__init__()
                self.n_cls = n_cls
                self.backbone = nn.Sequential(
                    nn.Conv2d(in_ch, 64, 3, padding=1), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(4),
                )
                self.cls_head = nn.Linear(64 * 4 * 4, n_cls)
                self.box_head = nn.Linear(64 * 4 * 4, 4)

            def forward(self, images, targets=None):
                import torch.nn.functional as F
                feats = self.backbone(images).flatten(1)
                cls_logits = self.cls_head(feats)           # (B, n_cls)
                box_pred   = torch.sigmoid(self.box_head(feats))  # (B, 4) in [0, 1]

                if self.training and targets is not None:
                    # --- build simple surrogate losses ---
                    device = images.device
                    cls_tgts = torch.tensor(
                        [int(t['labels'][0]) if len(t['labels']) > 0 else 0 for t in targets],
                        dtype=torch.long, device=device,
                    )
                    loss_cls = F.cross_entropy(cls_logits, cls_tgts)
                    # Regression: pull predicted box toward GT box of first object
                    # (normalised to [0,1] by dividing by image W/H)
                    h, w = images.shape[2], images.shape[3]
                    gt_boxes = []
                    for t in targets:
                        if len(t['boxes']) > 0:
                            b = t['boxes'][0].float()  # xyxy
                            gt_boxes.append(torch.stack([b[0]/w, b[1]/h, b[2]/w, b[3]/h]))
                        else:
                            gt_boxes.append(torch.zeros(4, device=device))
                    gt_boxes = torch.stack(gt_boxes)
                    loss_box = F.l1_loss(box_pred, gt_boxes)
                    return {'loss_classifier': loss_cls, 'loss_box_reg': loss_box}

                # --- eval: return list of prediction dicts ---
                h, w = images.shape[2], images.shape[3]
                scores = torch.softmax(cls_logits, dim=-1)  # (B, n_cls)
                results = []
                for i in range(images.shape[0]):
                    # Denormalise box
                    b = box_pred[i]
                    box_xyxy = torch.stack([b[0]*w, b[1]*h, b[2]*w, b[3]*h]).unsqueeze(0)
                    top_score, top_label = scores[i].max(dim=0)
                    results.append({
                        'boxes':  box_xyxy,
                        'labels': top_label.unsqueeze(0),
                        'scores': top_score.unsqueeze(0),
                    })
                return results

        return SimpleDetector(num_classes, in_channels)

    # ──────────────────────────────────────────────────────────────────────────
    def _create_segmentation_model(self, architecture: str) -> nn.Module:
        num_classes = self.config.dataset_info.num_classes
        in_channels = self._get_channels()

        arch_key = architecture.lower()

        if TORCH_AVAILABLE:
            # ── UNet (custom, works for any channel count) ─────────────────
            if "unet" in arch_key:
                logger.info(f"Building UNet segmentation model ({in_channels} ch → {num_classes} classes)")
                return self._build_unet(num_classes, in_channels)

            # ── torchvision DeepLabV3 / FCN (3-channel only natively) ──────
            if in_channels == 3:
                try:
                    from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50

                    if "deeplabv3" in arch_key or "segformer" in arch_key:
                        base = deeplabv3_resnet50(weights=None, num_classes=num_classes)
                    else:
                        base = fcn_resnet50(weights=None, num_classes=num_classes)

                    # Wrap to unwrap the output dict so the trainer receives a plain tensor
                    class _DictWrapper(nn.Module):
                        def __init__(self, model):
                            super().__init__()
                            self.model = model
                        def forward(self, x):
                            out = self.model(x)
                            return out  # keep dict; trainer._compute_loss handles it

                    logger.info(f"Created torchvision segmentation model: {architecture}")
                    return _DictWrapper(base)
                except Exception as e:
                    logger.warning(f"Torchvision segmentation build failed ({e}), falling back to UNet")

            # Fallback: UNet works for any channel count
            logger.info(f"Building UNet fallback ({in_channels} ch → {num_classes} classes)")
            return self._build_unet(num_classes, in_channels)

        # Minimal baseline when torch is unavailable
        class SimpleSegmenter(nn.Module):
            def __init__(self, num_classes, in_ch):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(in_ch, 64, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                )
                self.head = nn.Conv2d(64, num_classes, 1)
            def forward(self, x):
                return self.head(self.encoder(x))

        return SimpleSegmenter(num_classes, in_channels)

    def _build_unet(self, num_classes: int, in_channels: int) -> nn.Module:
        """Lightweight UNet-style encoder-decoder for any channel count."""

        class DoubleConv(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                )
            def forward(self, x):
                return self.block(x)

        class UNet(nn.Module):
            def __init__(self, in_ch, num_classes, base=32):
                super().__init__()
                self.enc1 = DoubleConv(in_ch,     base)
                self.enc2 = DoubleConv(base,      base * 2)
                self.enc3 = DoubleConv(base * 2,  base * 4)
                self.enc4 = DoubleConv(base * 4,  base * 8)
                self.pool = nn.MaxPool2d(2)

                self.bottleneck = DoubleConv(base * 8, base * 16)

                self.up4   = nn.ConvTranspose2d(base * 16, base * 8,  2, stride=2)
                self.dec4  = DoubleConv(base * 16, base * 8)
                self.up3   = nn.ConvTranspose2d(base * 8,  base * 4,  2, stride=2)
                self.dec3  = DoubleConv(base * 8,  base * 4)
                self.up2   = nn.ConvTranspose2d(base * 4,  base * 2,  2, stride=2)
                self.dec2  = DoubleConv(base * 4,  base * 2)
                self.up1   = nn.ConvTranspose2d(base * 2,  base,      2, stride=2)
                self.dec1  = DoubleConv(base * 2,  base)

                self.head  = nn.Conv2d(base, num_classes, 1)

            def forward(self, x):
                import torch.nn.functional as F
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                e3 = self.enc3(self.pool(e2))
                e4 = self.enc4(self.pool(e3))
                b  = self.bottleneck(self.pool(e4))

                d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
                d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
                d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
                d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
                return self.head(d1)

        return UNet(in_channels, num_classes)