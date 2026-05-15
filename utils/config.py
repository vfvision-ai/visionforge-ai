"""
Configuration management for the intelligent training system.
"""

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional
import json
import yaml

from core.dataset_analyzer import DatasetInfo
from core.model_selector import ModelConfig


@dataclass
class Config:
    """Central configuration class for training pipeline."""
    
    # Dataset and model info
    dataset_info: DatasetInfo
    model_config: ModelConfig
    
    # Paths
    dataset_path: Path
    output_dir: Path
    
    # Training parameters
    max_epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 32
    weight_decay: float = 1e-4
    
    # Optimization
    optimizer: str = 'adamw'  # adamw, sgd
    scheduler: str = 'cosine'  # cosine, plateau, step
    
    # Training options
    use_mixed_precision: bool = True
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 10
    
    # Hardware
    gpu_id: int = 0
    num_workers: int = 4
    
    # Logging and monitoring
    log_interval: int = 10
    save_interval: int = 5
    
    @classmethod
    def create_auto_config(cls, dataset_info: DatasetInfo, model_config: ModelConfig, 
                          args) -> 'Config':
        """
        Create automatic configuration based on dataset analysis and model selection.
        
        Args:
            dataset_info: Analyzed dataset information
            model_config: Selected model configuration
            args: Command line arguments
            
        Returns:
            Complete configuration object
        """
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use recommended batch size from dataset analysis
        batch_size = min(dataset_info.recommended_batch_size, 64)  # Cap at 64
        
        # Adjust epochs based on dataset size
        max_epochs = args.max_epochs
        if dataset_info.num_samples < 500:
            max_epochs = min(max_epochs, 50)  # Fewer epochs for small datasets
        elif dataset_info.num_samples > 50000:
            max_epochs = min(max_epochs, 200)  # More epochs for large datasets
        
        # Adjust learning rate based on batch size
        base_lr = 0.001
        lr_scale = batch_size / 32.0  # Scale relative to batch size 32
        learning_rate = base_lr * lr_scale
        
        # Task-specific adjustments
        if dataset_info.task_type == "detection":
            learning_rate *= 0.5  # Lower LR for detection
            batch_size = min(batch_size, 16)  # Smaller batches for detection
        elif dataset_info.task_type == "segmentation":
            learning_rate *= 0.7  # Lower LR for segmentation
            batch_size = min(batch_size, 8)  # Smaller batches for segmentation
        
        config = cls(
            dataset_info=dataset_info,
            model_config=model_config,
            dataset_path=Path(args.dataset_path),
            output_dir=output_dir,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gpu_id=args.gpu_id,
            use_mixed_precision=True,  # Enable by default if supported
        )
        
        # Save configuration
        config.save(output_dir / 'config.yaml')
        
        return config
    
    def save(self, filepath: Path):
        """Save configuration to file."""
        config_dict = self.to_dict()
        
        # Clean up config dict for safe YAML/JSON serialization
        def make_serializable(obj):
            """Convert non-serializable objects to serializable forms."""
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return list(obj)  # Convert tuples to lists
            elif hasattr(obj, '__fspath__'):  # Path-like objects
                return str(obj)
            elif hasattr(obj, 'tolist'):  # NumPy arrays
                return obj.tolist()
            else:
                return obj
        
        config_dict = make_serializable(config_dict)
        
        if filepath.suffix in ['.yaml', '.yml']:
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2, allow_unicode=True)
        else:
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: Path) -> 'Config':
        """Load configuration from file."""
        if filepath.suffix in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary."""
        # Convert nested objects
        if 'dataset_info' in config_dict:
            config_dict['dataset_info'] = DatasetInfo(**config_dict['dataset_info'])
        if 'model_config' in config_dict:
            config_dict['model_config'] = ModelConfig(**config_dict['model_config'])
        
        # Convert paths
        if 'dataset_path' in config_dict:
            config_dict['dataset_path'] = Path(config_dict['dataset_path'])
        if 'output_dir' in config_dict:
            config_dict['output_dir'] = Path(config_dict['output_dir'])
        
        return cls(**config_dict)
    
    def update_params(self, params: Dict[str, Any]):
        """Update configuration with new parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Handle nested model config parameters
                if key in ['dropout', 'label_smoothing', 'mixup_alpha', 'nms_threshold',
                          'score_threshold', 'mosaic_prob', 'deep_supervision']:
                    self.model_config.config_params[key] = value
    
    def get_effective_batch_size(self) -> int:
        """Get effective batch size considering GPU memory."""
        # Estimate memory usage and adjust batch size if needed
        estimated_memory = (
            self.model_config.memory_requirements * 
            (self.batch_size / 16)  # Scale from base estimate
        )
        
        # If memory usage is too high, reduce batch size
        if estimated_memory > 8.0:  # 8GB threshold
            reduction_factor = estimated_memory / 8.0
            return max(1, int(self.batch_size / reduction_factor))
        
        return self.batch_size
    
    def validate(self):
        """Validate configuration parameters."""
        errors = []
        
        # Check paths
        if not self.dataset_path.exists():
            errors.append(f"Dataset path does not exist: {self.dataset_path}")
        
        # Check parameters
        if self.learning_rate <= 0:
            errors.append("Learning rate must be positive")
        
        if self.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        if self.max_epochs <= 0:
            errors.append("Max epochs must be positive")
        
        # Check hardware
        try:
            import torch
            if self.gpu_id >= torch.cuda.device_count():
                errors.append(f"GPU {self.gpu_id} not available")
        except ImportError:
            logging.warning("PyTorch not available for GPU validation")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Ensure paths are Path objects
        if isinstance(self.dataset_path, str):
            self.dataset_path = Path(self.dataset_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)