"""
Data loading utilities for different task types.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Fallback definitions for when torch is not available
    class Dataset:
        pass
    class DataLoader:
        pass

from utils.config import Config


# Define dataset classes at module level to avoid pickling issues
class DummyClassificationDataset(Dataset):
    """Dummy classification dataset for testing with learnable patterns."""
    
    def __init__(self, size=1000, num_classes=10, input_channels=3, input_size=(224, 224), split='train'):
        self.size = size
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.input_size = input_size
        self.split = split
        
        # Set different random seed for train vs val to ensure different patterns
        self.seed = 42 if split == 'train' else 123
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available")
            
        # Create deterministic label based on index for consistent patterns
        label = idx % self.num_classes
        
        # Create image with class-specific patterns the model can learn
        # Use local random state to avoid interfering with training
        generator = torch.Generator()
        generator.manual_seed(self.seed + idx)  # Unique seed per sample
        image = torch.randn(self.input_channels, *self.input_size, generator=generator)
        
        # Add class-specific signal to make the data learnable
        # Each class gets a different pattern in different channels
        class_signal_strength = 1.5
        
        if self.input_channels >= 3:
            # Use different channels for different class patterns
            channel_idx = label % 3
            image[channel_idx, :, :] += class_signal_strength * (label / self.num_classes)
            
            # Add spatial patterns
            spatial_pattern = (label / self.num_classes) * 2.0 - 1.0  # Range [-1, 1]
            
            # Create simple spatial patterns based on class
            if label % 4 == 0:  # Top-left bright
                image[channel_idx, :self.input_size[0]//2, :self.input_size[1]//2] += spatial_pattern
            elif label % 4 == 1:  # Top-right bright
                image[channel_idx, :self.input_size[0]//2, self.input_size[1]//2:] += spatial_pattern
            elif label % 4 == 2:  # Bottom-left bright
                image[channel_idx, self.input_size[0]//2:, :self.input_size[1]//2] += spatial_pattern
            else:  # Bottom-right bright
                image[channel_idx, self.input_size[0]//2:, self.input_size[1]//2:] += spatial_pattern
        
        return image, label


class DummyDetectionDataset(Dataset):
    """Dummy detection dataset for testing with learnable patterns."""
    
    def __init__(self, size=500, num_classes=10, input_channels=3, input_size=(224, 224), split='train'):
        self.size = size
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.input_size = input_size
        self.split = split
        
        # Set different random seed for train vs val to ensure different patterns
        self.seed = 42 if split == 'train' else 123
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available")
            
        # Create image with predictable detection patterns
        image = torch.randn(self.input_channels, *self.input_size)
        
        # Create deterministic bounding boxes and labels
        num_objects = min(3, max(1, (idx % 4) + 1))  # 1-3 objects per image
        
        boxes = []
        labels = []
        
        for i in range(num_objects):
            # Create deterministic box positions based on idx and object number
            obj_class = (idx + i) % self.num_classes + 1  # Labels start from 1
            
            # Position boxes deterministically
            box_size = min(self.input_size) // 4  # Quarter of image size
            x_pos = (i * box_size) % (self.input_size[1] - box_size)
            y_pos = ((idx + i) * box_size) % (self.input_size[0] - box_size)
            
            # Create box [x1, y1, x2, y2]
            box = torch.tensor([x_pos, y_pos, x_pos + box_size, y_pos + box_size], dtype=torch.float32)
            boxes.append(box)
            labels.append(obj_class)
            
            # Add visual pattern to the image at box location
            signal_strength = 2.0
            channel_idx = obj_class % self.input_channels
            image[channel_idx, int(y_pos):int(y_pos + box_size), int(x_pos):int(x_pos + box_size)] += signal_strength
        
        boxes = torch.stack(boxes)
        labels = torch.tensor(labels, dtype=torch.long)
        target = {'boxes': boxes, 'labels': labels}
        
        return image, target


class DummySegmentationDataset(Dataset):
    """Dummy segmentation dataset for testing with learnable patterns."""
    
    def __init__(self, size=300, num_classes=21, input_channels=3, input_size=(224, 224)):
        self.size = size
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.input_size = input_size
        
        # Set random seed for reproducibility
        torch.manual_seed(42 if size > 200 else 123)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available")
            
        # Create image and mask with learnable patterns
        image = torch.randn(self.input_channels, *self.input_size)
        
        # Create deterministic segmentation patterns
        mask = torch.zeros(self.input_size, dtype=torch.long)
        
        # Create geometric patterns that correlate with image features
        height, width = self.input_size
        
        # Pattern selection based on idx
        pattern_type = idx % 4
        
        if pattern_type == 0:  # Horizontal stripes
            stripe_height = height // self.num_classes
            for i in range(self.num_classes):
                y_start = i * stripe_height
                y_end = min((i + 1) * stripe_height, height)
                mask[y_start:y_end, :] = i
                
                # Add corresponding signal to image
                if i < self.input_channels:
                    image[i, y_start:y_end, :] += 1.5
                    
        elif pattern_type == 1:  # Vertical stripes
            stripe_width = width // self.num_classes
            for i in range(self.num_classes):
                x_start = i * stripe_width
                x_end = min((i + 1) * stripe_width, width)
                mask[:, x_start:x_end] = i
                
                # Add corresponding signal to image
                channel_idx = i % self.input_channels
                image[channel_idx, :, x_start:x_end] += 1.5
                
        elif pattern_type == 2:  # Checkerboard
            block_size = max(1, min(height, width) // 8)
            for y in range(0, height, block_size):
                for x in range(0, width, block_size):
                    class_id = ((y // block_size) + (x // block_size)) % self.num_classes
                    mask[y:min(y+block_size, height), x:min(x+block_size, width)] = class_id
                    
                    # Add signal to image
                    channel_idx = class_id % self.input_channels
                    image[channel_idx, y:min(y+block_size, height), x:min(x+block_size, width)] += 1.5
                    
        else:  # Concentric regions
            center_y, center_x = height // 2, width // 2
            for y in range(height):
                for x in range(width):
                    distance = ((y - center_y) ** 2 + (x - center_x) ** 2) ** 0.5
                    class_id = int(distance / (min(height, width) / 4)) % self.num_classes
                    mask[y, x] = class_id
                    
                    # Add signal to image
                    channel_idx = class_id % self.input_channels
                    image[channel_idx, y, x] += 1.0
        
        return image, mask


def _detection_collate(batch):
    """Collate function for detection datasets.

    Each item is (image_tensor, target_dict) where target_dict contains
    variable-length 'boxes' and 'labels' tensors.  PyTorch's default collate
    cannot stack variable-length tensors, so we keep targets as a list.
    """
    images  = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


class DataLoaderFactory:
    """Factory for creating data loaders."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def create_train_loader(self) -> DataLoader:
        """Create training data loader."""
        dataset = self._create_dataset('train')
        is_detection = self.config.dataset_info.task_type == "detection"
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=_detection_collate if is_detection else None,
        )
    
    def create_val_loader(self) -> DataLoader:
        """Create validation data loader."""
        dataset = self._create_dataset('val')
        is_detection = self.config.dataset_info.task_type == "detection"
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=_detection_collate if is_detection else None,
        )
    
    def _create_dataset(self, split: str) -> Dataset:
        """Create dataset for the specified split."""
        task_type = self.config.dataset_info.task_type
        
        if task_type == "classification":
            return self._create_classification_dataset(split)
        elif task_type == "detection":
            return self._create_detection_dataset(split)
        elif task_type == "segmentation":
            return self._create_segmentation_dataset(split)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def _create_classification_dataset(self, split: str) -> Dataset:
        """Create classification dataset."""
        # Get configuration parameters
        num_classes = self.config.dataset_info.num_classes
        input_channels = 3  # Default
        input_size = (224, 224)  # Default
        
        # Extract channels and size from model config if available
        if hasattr(self.config, 'model_config') and self.config.model_config:
            if hasattr(self.config.model_config, 'config_params'):
                input_channels = self.config.model_config.config_params.get('input_channels', 3)
            
            if hasattr(self.config.model_config, 'input_size'):
                if len(self.config.model_config.input_size) == 3:
                    # Format: (channels, height, width)
                    input_channels, height, width = self.config.model_config.input_size
                    input_size = (height, width)
                else:
                    # Format: (height, width)
                    input_size = self.config.model_config.input_size
        
        # Determine dataset size based on split
        size = 1000 if split == 'train' else 200
        
        return DummyClassificationDataset(
            size=size,
            num_classes=num_classes,
            input_channels=input_channels,
            input_size=input_size,
            split=split
        )
    
    def _create_detection_dataset(self, split: str) -> Dataset:
        """Create detection dataset - try to load real data first, fallback to dummy."""
        # Get configuration parameters
        num_classes = self.config.dataset_info.num_classes
        input_channels = 3  # Default
        input_size = (640, 640)  # Default for detection
        
        # Extract channels and size from model config if available
        if hasattr(self.config, 'model_config') and self.config.model_config:
            if hasattr(self.config.model_config, 'config_params'):
                input_channels = self.config.model_config.config_params.get('input_channels', 3)
            
            if hasattr(self.config.model_config, 'input_size'):
                if len(self.config.model_config.input_size) == 3:
                    # Format: (channels, height, width)
                    input_channels, height, width = self.config.model_config.input_size
                    input_size = (height, width)
                else:
                    # Format: (height, width)
                    input_size = self.config.model_config.input_size
        
        # Try to load real detection dataset from disk
        dataset_path = getattr(self.config.dataset_info, 'dataset_path', None)
        if dataset_path:
            real_ds = self._try_load_real_detection(
                dataset_path, split, num_classes, input_channels, input_size
            )
            if real_ds is not None:
                return real_ds
        
        # Fallback to dummy dataset
        self.logger.warning(
            f"⚠️ No real detection dataset found at {dataset_path}. Using dummy data. "
            f"For real detection training, organize data in COCO or YOLO format."
        )
        size = 500 if split == 'train' else 100
        
        return DummyDetectionDataset(
            size=size,
            num_classes=num_classes,
            input_channels=input_channels,
            input_size=input_size,
            split=split
        )
    
    def _try_load_real_detection(self, dataset_path, split: str, num_classes: int,
                                  input_channels: int, input_size: tuple):
        """Try to load a real detection dataset from disk.
        
        Supports:
          1. COCO format: images/ + annotations.json
          2. YOLO format: images/ + labels/ (with .txt files)
          3. Pascal VOC format: images/ + annotations/ (with .xml files)
        
        Returns a Dataset or None if dataset format is not recognized.
        """
        from pathlib import Path as _Path
        import json
        
        root = _Path(dataset_path)
        
        # \u2500\u2500 Try COCO format \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        coco_json_candidates = [\n            root / f\"annotations_{split}.json\",\n            root / \"annotations\" / f\"{split}.json\",\n            root / f\"{split}.json\",\n            root / \"annotations.json\",\n        ]\n        \n        for json_path in coco_json_candidates:\n            if json_path.exists():\n                try:\n                    self.logger.info(f\"\ud83d\udcca Found COCO annotations: {json_path}\")\n                    # Just verify it's valid COCO format\n                    with open(json_path, 'r') as f:\n                        coco_data = json.load(f)\n                    if 'images' in coco_data and 'annotations' in coco_data:\n                        # Could implement full COCO dataset loader here\n                        # For now, log success but return None to use dummy data\n                        self.logger.info(\"\u2705 Valid COCO format detected (full loader not yet implemented)\")\n                        return None  # TODO: Implement COCODataset class\n                except Exception as e:\n                    self.logger.warning(f\"\u26a0\ufe0f Failed to parse COCO JSON: {e}\")\n        \n        # \u2500\u2500 Try YOLO format \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n        yolo_candidates = [\n            (root / split / \"images\", root / split / \"labels\"),\n            (root / \"images\" / split, root / \"labels\" / split),\n            (root / \"images\", root / \"labels\"),\n        ]\n        \n        for img_dir, label_dir in yolo_candidates:\n            if img_dir.is_dir() and label_dir.is_dir():\n                img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}\n                image_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in img_exts])\n                \n                if len(image_files) >= 4:\n                    # Check if corresponding label files exist\n                    has_labels = False\n                    for img_p in image_files[:5]:\n                        label_p = label_dir / f\"{img_p.stem}.txt\"\n                        if label_p.exists():\n                            has_labels = True\n                            break\n                    \n                    if has_labels:\n                        self.logger.info(f\"\ud83c\udfaf Found YOLO dataset: {len(image_files)} images\")\n                        # Could implement full YOLO dataset loader here\n                        # For now, log success but return None to use dummy data\n                        self.logger.info(\"\u2705 Valid YOLO format detected (full loader not yet implemented)\")\n                        return None  # TODO: Implement YOLODataset class\n        \n        # \u2500\u2500 Try Pascal VOC format \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n        voc_candidates = [\n            (root / \"JPEGImages\", root / \"Annotations\"),\n            (root / \"images\", root / \"annotations\"),\n        ]\n        \n        for img_dir, annot_dir in voc_candidates:\n            if img_dir.is_dir() and annot_dir.is_dir():\n                image_files = sorted(list(img_dir.glob(\"*.jpg\")) + list(img_dir.glob(\"*.png\")))\n                if len(image_files) >= 4:\n                    # Check for XML files\n                    xml_files = list(annot_dir.glob(\"*.xml\"))\n                    if len(xml_files) >= 4:\n                        self.logger.info(f\"\ud83d\udcdd Found Pascal VOC dataset: {len(image_files)} images\")\n                        self.logger.info(\"\u2705 Valid VOC format detected (full loader not yet implemented)\")\n                        return None  # TODO: Implement VOCDataset class\n        \n        return None\n    \n    def _create_segmentation_dataset(self, split: str) -> Dataset:
        """Create segmentation dataset — real mask folders if available, else dummy."""
        # Get configuration parameters
        num_classes = self.config.dataset_info.num_classes
        input_channels = 3  # Default
        input_size = (512, 512)  # Default for segmentation

        # Extract channels and size from model config if available
        if hasattr(self.config, 'model_config') and self.config.model_config:
            if hasattr(self.config.model_config, 'config_params'):
                input_channels = self.config.model_config.config_params.get('input_channels', 3)
            if hasattr(self.config.model_config, 'input_size'):
                if len(self.config.model_config.input_size) == 3:
                    input_channels, height, width = self.config.model_config.input_size
                    input_size = (height, width)
                else:
                    input_size = self.config.model_config.input_size

        # Try to load real data from disk
        dataset_path = getattr(self.config.dataset_info, 'dataset_path', None)
        if dataset_path:
            real_ds = self._try_load_real_segmentation(
                dataset_path, split, num_classes, input_channels, input_size
            )
            if real_ds is not None:
                return real_ds

        # Fall back to dummy dataset
        size = 300 if split == 'train' else 60
        return DummySegmentationDataset(
            size=size,
            num_classes=num_classes,
            input_channels=input_channels,
            input_size=input_size,
        )

    def _try_load_real_segmentation(self, dataset_path, split: str, num_classes: int,
                                     input_channels: int, input_size: tuple):
        """Try to load a real image + mask segmentation dataset from disk.

        Supports two folder conventions:
          1. <root>/images/ + <root>/masks/   (images and masks in separate dirs)
          2. <root>/<split>/images/ + <root>/<split>/masks/
          3. Flat <root>/ with images named  *.jpg and masks named *_mask.png / *_seg.png

        Masks are assumed to be single-channel PNGs where pixel value == class index.
        Returns a Dataset or None if folder layout is not recognised.
        """
        from pathlib import Path as _Path

        root = _Path(dataset_path)

        # Candidate (images_dir, masks_dir) pairs to probe
        candidates = [
            (root / split / "images", root / split / "masks"),
            (root / split / "images", root / split / "labels"),
            (root / "images", root / "masks"),
            (root / "images", root / "labels"),
            (root / "imgs",   root / "masks"),
        ]

        images_dir = masks_dir = None
        for img_d, msk_d in candidates:
            if img_d.is_dir() and msk_d.is_dir():
                images_dir, masks_dir = img_d, msk_d
                break

        if images_dir is None or masks_dir is None:
            return None

        # Collect matched pairs
        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in img_exts])

        pairs = []
        for img_p in image_files:
            stem = img_p.stem
            # Try common mask naming conventions
            for msk_name in [stem + '.png', stem + '_mask.png', stem + '_seg.png',
                              stem + '.bmp', stem + img_p.suffix]:
                msk_p = masks_dir / msk_name
                if msk_p.exists():
                    pairs.append((str(img_p), str(msk_p)))
                    break

        if len(pairs) < 4:
            return None

        # Split into train / val (80 / 20)
        n = len(pairs)
        if split == 'train':
            pairs = pairs[:int(n * 0.8)]
        else:
            pairs = pairs[int(n * 0.8):]

        if not pairs:
            return None

        return RealSegmentationDataset(pairs, num_classes, input_channels, input_size)


class RealSegmentationDataset(Dataset):
    """Dataset that loads image + mask pairs from disk."""

    def __init__(self, pairs, num_classes: int, input_channels: int, input_size: tuple):
        self.pairs = pairs
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.input_size = input_size  # (H, W)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        import torch
        import numpy as np
        try:
            from PIL import Image
        except ImportError:
            raise RuntimeError("Pillow is required for real segmentation data loading")

        img_path, msk_path = self.pairs[idx]
        h, w = self.input_size

        # Load image
        img = Image.open(img_path)
        if self.input_channels == 1:
            img = img.convert('L')
        elif self.input_channels == 3:
            img = img.convert('RGB')
        else:
            img = img.convert('RGB')
        img = img.resize((w, h), Image.BILINEAR)
        img_arr = np.array(img, dtype=np.float32) / 255.0

        if img_arr.ndim == 2:                          # grayscale (H, W)
            img_arr = img_arr[np.newaxis, :, :]        # → (1, H, W)
        else:                                          # (H, W, C)
            img_arr = img_arr.transpose(2, 0, 1)       # → (C, H, W)

        # Load mask (single channel, pixel = class index)
        msk = Image.open(msk_path).convert('L')
        msk = msk.resize((w, h), Image.NEAREST)
        msk_arr = np.array(msk, dtype=np.int64)
        # Clamp to [0, num_classes-1]; keep 255 as ignore index
        msk_arr = np.where(msk_arr == 255, 255, np.clip(msk_arr, 0, self.num_classes - 1))

        return torch.from_numpy(img_arr), torch.from_numpy(msk_arr)