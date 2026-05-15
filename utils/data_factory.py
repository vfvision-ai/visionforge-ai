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


class DataLoaderFactory:
    """Factory for creating data loaders."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def create_train_loader(self) -> DataLoader:
        """Create training data loader."""
        dataset = self._create_dataset('train')
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing/pickling issues
            pin_memory=False  # Disable for compatibility
        )
    
    def create_val_loader(self) -> DataLoader:
        """Create validation data loader."""
        dataset = self._create_dataset('val')
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid multiprocessing/pickling issues
            pin_memory=False  # Disable for compatibility
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
        """Create detection dataset."""
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
        
        # Determine dataset size based on split
        size = 500 if split == 'train' else 100
        
        return DummyDetectionDataset(
            size=size,
            num_classes=num_classes,
            input_channels=input_channels,
            input_size=input_size
        )
    
    def _create_segmentation_dataset(self, split: str) -> Dataset:
        """Create segmentation dataset."""
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
                    # Format: (channels, height, width)
                    input_channels, height, width = self.config.model_config.input_size
                    input_size = (height, width)
                else:
                    # Format: (height, width)
                    input_size = self.config.model_config.input_size
        
        # Determine dataset size based on split
        size = 300 if split == 'train' else 60
        
        return DummySegmentationDataset(
            size=size,
            num_classes=num_classes,
            input_channels=input_channels,
            input_size=input_size
        )