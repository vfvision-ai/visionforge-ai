"""
Model Selector - Intelligently selects optimal model architectures.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core.dataset_analyzer import DatasetInfo


@dataclass
class ModelConfig:
    """Model configuration with architecture details."""
    architecture: str
    backbone: str
    num_parameters: int
    input_size: Tuple[int, int]
    pretrained: bool
    config_params: Dict
    estimated_flops: int
    memory_requirements: float  # GB
    framework: str = "PyTorch"  # Default framework


class ModelSelector:
    """Intelligent model architecture selector based on dataset characteristics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._init_model_database()
    
    def _init_model_database(self):
        """Initialize database of available models and their characteristics."""
        
        # Classification models
        self.classification_models = {
            'Simple CNN': {
                'backbone': 'None',
                'num_parameters': 50000,
                'input_size': (224, 224),  # Default, will be overridden by user input
                'flops': 10000000,
                'memory_gb': 0.1,
                'accuracy_score': 0.7,
                'speed_score': 0.95
            },
            'efficientnet_b0': {
                'backbone': 'efficientnet_b0',
                'num_parameters': 5300000,
                'input_size': (224, 224),
                'flops': 390000000,
                'memory_gb': 0.5,
                'accuracy_score': 0.85,
                'speed_score': 0.9
            },
            'efficientnet_b3': {
                'backbone': 'efficientnet_b3',
                'num_parameters': 12000000,
                'input_size': (300, 300),
                'flops': 1800000000,
                'memory_gb': 1.2,
                'accuracy_score': 0.9,
                'speed_score': 0.7
            },
            'resnet50': {
                'backbone': 'resnet50',
                'num_parameters': 25600000,
                'input_size': (224, 224),
                'flops': 4100000000,
                'memory_gb': 2.0,
                'accuracy_score': 0.88,
                'speed_score': 0.8
            },
            'vit_base': {
                'backbone': 'vit_base_patch16_224',
                'num_parameters': 86400000,
                'input_size': (224, 224),
                'flops': 17600000000,
                'memory_gb': 4.5,
                'accuracy_score': 0.92,
                'speed_score': 0.5
            },
            'convnext_tiny': {
                'backbone': 'convnext_tiny',
                'num_parameters': 28600000,
                'input_size': (224, 224),
                'flops': 4500000000,
                'memory_gb': 2.2,
                'accuracy_score': 0.89,
                'speed_score': 0.75
            }
        }
        
        # Detection models
        self.detection_models = {
            'yolov8n': {
                'backbone': 'yolov8n',
                'num_parameters': 3200000,
                'input_size': (640, 640),
                'flops': 8700000000,
                'memory_gb': 1.5,
                'accuracy_score': 0.75,
                'speed_score': 0.95
            },
            'yolov8s': {
                'backbone': 'yolov8s',
                'num_parameters': 11200000,
                'input_size': (640, 640),
                'flops': 28600000000,
                'memory_gb': 3.0,
                'accuracy_score': 0.8,
                'speed_score': 0.85
            },
            'yolov8m': {
                'backbone': 'yolov8m',
                'num_parameters': 25900000,
                'input_size': (640, 640),
                'flops': 78900000000,
                'memory_gb': 5.5,
                'accuracy_score': 0.83,
                'speed_score': 0.7
            },
            'faster_rcnn_r50': {
                'backbone': 'resnet50',
                'num_parameters': 41800000,
                'input_size': (800, 600),
                'flops': 134000000000,
                'memory_gb': 8.0,
                'accuracy_score': 0.85,
                'speed_score': 0.4
            },
            'detr_r50': {
                'backbone': 'resnet50',
                'num_parameters': 41300000,
                'input_size': (800, 600),
                'flops': 86000000000,
                'memory_gb': 6.5,
                'accuracy_score': 0.82,
                'speed_score': 0.5
            }
        }
        
        # Segmentation models
        self.segmentation_models = {
            'unet_resnet34': {
                'backbone': 'resnet34',
                'num_parameters': 24400000,
                'input_size': (512, 512),
                'flops': 120000000000,
                'memory_gb': 4.0,
                'accuracy_score': 0.85,
                'speed_score': 0.7
            },
            'deeplabv3_resnet50': {
                'backbone': 'resnet50',
                'num_parameters': 39600000,
                'input_size': (512, 512),
                'flops': 180000000000,
                'memory_gb': 6.0,
                'accuracy_score': 0.87,
                'speed_score': 0.6
            },
            'segformer_b0': {
                'backbone': 'mit_b0',
                'num_parameters': 3700000,
                'input_size': (512, 512),
                'flops': 15800000000,
                'memory_gb': 2.5,
                'accuracy_score': 0.83,
                'speed_score': 0.8
            },
            'segformer_b2': {
                'backbone': 'mit_b2',
                'num_parameters': 27300000,
                'input_size': (512, 512),
                'flops': 62400000000,
                'memory_gb': 5.5,
                'accuracy_score': 0.88,
                'speed_score': 0.65
            }
        }
    
    def select_model(self, dataset_info: DatasetInfo, framework: str = "pytorch") -> ModelConfig:
        """
        Select optimal model architecture based on dataset characteristics.
        
        Args:
            dataset_info: Analyzed dataset information
            framework: Target framework ("pytorch" or "tensorflow")
            
        Returns:
            ModelConfig with selected architecture
        """
        self.logger.info(f"Selecting model for {dataset_info.task_type} task with {framework} framework")
        
        if dataset_info.task_type == "classification":
            return self._select_classification_model(dataset_info, framework)
        elif dataset_info.task_type == "detection":
            return self._select_detection_model(dataset_info, framework)
        elif dataset_info.task_type == "segmentation":
            return self._select_segmentation_model(dataset_info, framework)
        else:
            raise ValueError(f"Unsupported task type: {dataset_info.task_type}")
    
    def _select_classification_model(self, dataset_info: DatasetInfo, framework: str = "pytorch") -> ModelConfig:
        """Select classification model based on dataset characteristics."""
        
        # Scoring factors
        dataset_size_factor = self._get_dataset_size_factor(dataset_info.num_samples)
        complexity_factor = self._get_complexity_factor(dataset_info.num_classes)
        
        best_model = None
        best_score = -1
        
        for model_name, model_info in self.classification_models.items():
            # Calculate composite score
            accuracy_weight = 0.4
            speed_weight = 0.3
            efficiency_weight = 0.3
            
            # Adjust weights based on dataset size
            if dataset_info.num_samples < 1000:
                speed_weight = 0.5  # Prioritize speed for small datasets
                accuracy_weight = 0.3
            elif dataset_info.num_samples > 50000:
                accuracy_weight = 0.6  # Prioritize accuracy for large datasets
                speed_weight = 0.2
            
            score = (
                model_info['accuracy_score'] * accuracy_weight +
                model_info['speed_score'] * speed_weight +
                self._calculate_efficiency_score(model_info) * efficiency_weight
            )
            
            # Adjust for dataset characteristics
            score *= dataset_size_factor * complexity_factor
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        model_info = self.classification_models[best_model]
        
        # Adapt input size to dataset
        input_size = self._adapt_input_size(
            model_info['input_size'], 
            dataset_info.image_size
        )
        
        self.logger.info(f"Selected classification model: {best_model} (score: {best_score:.3f})")
        
        return ModelConfig(
            architecture=best_model,
            backbone=model_info['backbone'],
            num_parameters=model_info['num_parameters'],
            input_size=input_size,
            pretrained=True,
            config_params=self._get_classification_config(dataset_info, model_info),
            estimated_flops=model_info['flops'],
            memory_requirements=model_info['memory_gb']
        )
    
    def _select_detection_model(self, dataset_info: DatasetInfo, framework: str = "pytorch") -> ModelConfig:
        """Select detection model based on dataset characteristics."""
        
        best_model = None
        best_score = -1
        
        # Adjust selection criteria for detection
        for model_name, model_info in self.detection_models.items():
            # For detection, balance accuracy and speed
            accuracy_weight = 0.5
            speed_weight = 0.5
            
            # Adjust based on dataset size and complexity
            if dataset_info.num_samples < 500:
                speed_weight = 0.7  # Fast training for small datasets
                accuracy_weight = 0.3
            elif dataset_info.num_classes > 50:
                accuracy_weight = 0.7  # Better accuracy for complex datasets
                speed_weight = 0.3
            
            score = (
                model_info['accuracy_score'] * accuracy_weight +
                model_info['speed_score'] * speed_weight
            )
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        model_info = self.detection_models[best_model]
        
        self.logger.info(f"Selected detection model: {best_model} (score: {best_score:.3f})")
        
        return ModelConfig(
            architecture=best_model,
            backbone=model_info['backbone'],
            num_parameters=model_info['num_parameters'],
            input_size=model_info['input_size'],
            pretrained=True,
            config_params=self._get_detection_config(dataset_info, model_info),
            estimated_flops=model_info['flops'],
            memory_requirements=model_info['memory_gb']
        )
    
    def _select_segmentation_model(self, dataset_info: DatasetInfo, framework: str = "pytorch") -> ModelConfig:
        """Select segmentation model based on dataset characteristics."""
        
        best_model = None
        best_score = -1
        
        for model_name, model_info in self.segmentation_models.items():
            # Balance accuracy and efficiency for segmentation
            accuracy_weight = 0.6
            speed_weight = 0.4
            
            # Adjust for image resolution
            if max(dataset_info.image_size) > 1024:
                speed_weight = 0.6  # Prioritize speed for high-res images
                accuracy_weight = 0.4
            
            score = (
                model_info['accuracy_score'] * accuracy_weight +
                model_info['speed_score'] * speed_weight
            )
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        model_info = self.segmentation_models[best_model]
        
        self.logger.info(f"Selected segmentation model: {best_model} (score: {best_score:.3f})")
        
        return ModelConfig(
            architecture=best_model,
            backbone=model_info['backbone'],
            num_parameters=model_info['num_parameters'],
            input_size=model_info['input_size'],
            pretrained=True,
            config_params=self._get_segmentation_config(dataset_info, model_info),
            estimated_flops=model_info['flops'],
            memory_requirements=model_info['memory_gb']
        )
    
    def _get_dataset_size_factor(self, num_samples: int) -> float:
        """Get scaling factor based on dataset size."""
        if num_samples < 100:
            return 0.7  # Very small dataset
        elif num_samples < 1000:
            return 0.85  # Small dataset
        elif num_samples < 10000:
            return 1.0  # Medium dataset
        elif num_samples < 100000:
            return 1.1  # Large dataset
        else:
            return 1.2  # Very large dataset
    
    def _get_complexity_factor(self, num_classes: int) -> float:
        """Get scaling factor based on problem complexity."""
        if num_classes < 5:
            return 0.9  # Simple problem
        elif num_classes < 20:
            return 1.0  # Medium complexity
        elif num_classes < 100:
            return 1.1  # High complexity
        else:
            return 1.15  # Very high complexity
    
    def _calculate_efficiency_score(self, model_info: Dict) -> float:
        """Calculate efficiency score (accuracy per parameter)."""
        return model_info['accuracy_score'] / (model_info['num_parameters'] / 1000000)
    
    def _adapt_input_size(self, model_default: Tuple[int, int], 
                         dataset_avg: Tuple[int, int]) -> Tuple[int, int]:
        """Adapt input size based on dataset characteristics."""
        
        # If dataset images are much larger, increase input size
        if min(dataset_avg) > model_default[0] * 1.5:
            # Scale up input size
            scale_factor = min(2.0, min(dataset_avg) / model_default[0])
            new_size = int(model_default[0] * scale_factor)
            # Round to nearest 32 for efficiency
            new_size = ((new_size + 31) // 32) * 32
            return (new_size, new_size)
        
        return model_default
    
    def _get_classification_config(self, dataset_info: DatasetInfo, 
                                 model_info: Dict) -> Dict:
        """Get configuration parameters for classification."""
        return {
            'num_classes': dataset_info.num_classes,
            'dropout': 0.2 if dataset_info.num_samples < 1000 else 0.1,
            'label_smoothing': 0.1 if dataset_info.num_classes > 10 else 0.0,
            'mixup_alpha': 0.2 if dataset_info.num_samples > 1000 else 0.0,
            'cutmix_alpha': 1.0 if dataset_info.num_samples > 5000 else 0.0
        }
    
    def _get_detection_config(self, dataset_info: DatasetInfo, 
                            model_info: Dict) -> Dict:
        """Get configuration parameters for detection."""
        return {
            'num_classes': dataset_info.num_classes,
            'anchor_sizes': 'auto',  # Auto-calculate based on dataset
            'nms_threshold': 0.5,
            'score_threshold': 0.05,
            'max_detections': 100,
            'mosaic_prob': 0.5 if dataset_info.num_samples > 1000 else 0.0
        }
    
    def _get_segmentation_config(self, dataset_info: DatasetInfo,
                               model_info: Dict) -> Dict:
        """Get configuration parameters for segmentation."""
        return {
            'num_classes': dataset_info.num_classes,
            'ignore_index': 255,
            'use_auxiliary_loss': True if dataset_info.num_samples > 1000 else False,
            'deep_supervision': True if 'unet' in model_info['backbone'] else False,
            'class_weights': 'auto'  # Auto-calculate from dataset
        }