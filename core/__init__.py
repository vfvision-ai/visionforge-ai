"""
Core modules for ML training pipeline.

This package contains the core functionality for:
- Dataset analysis and preprocessing
- Model selection and architecture search
- Training orchestration
- Hyperparameter optimization

Modules:
    dataset_analyzer: Automatic dataset analysis and task detection
    model_selector: Intelligent model architecture selection
    trainer: PyTorch training pipeline
    tensorflow_trainer: TensorFlow/Keras training pipeline  
    sklearn_trainer: Scikit-learn training pipeline
    optimizer: Hyperparameter optimization with Bayesian methods
"""

from pathlib import Path

__all__ = [
    "dataset_analyzer",
    "model_selector",
    "trainer",
    "tensorflow_trainer",
    "sklearn_trainer",
    "optimizer",
]

# Package version
__version__ = "1.0.0"
