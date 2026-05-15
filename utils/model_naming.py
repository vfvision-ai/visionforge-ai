"""
Model naming utilities for consistent and informative model file names.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional


def generate_model_name(
    framework: str,
    architecture: Optional[str] = None,
    backbone: Optional[str] = None,
    dataset_name: Optional[str] = None,
    task_type: Optional[str] = None,
    timestamp: Optional[str] = None,
    extension: Optional[str] = None
) -> str:
    """
    Generate a comprehensive model name with framework, architecture, dataset, and timestamp.
    
    Args:
        framework: ML framework (pytorch, tensorflow, scikit-learn)
        architecture: Model architecture (resnet50, efficientnet, etc.)
        backbone: Backbone name (if different from architecture)
        dataset_name: Name of the dataset used for training
        task_type: Type of task (classification, detection, segmentation)
        timestamp: Custom timestamp (if None, uses current time)
        extension: File extension (if None, inferred from framework)
    
    Returns:
        str: Formatted model filename
        
    Example:
        >>> generate_model_name("pytorch", "resnet50", "resnet", "cifar10", "classification")
        'pytorch_resnet50_resnet_cifar10_classification_20251011_142530'
    """
    
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Clean and format components
    components = []
    
    # Framework (always first)
    if framework:
        components.append(clean_name_component(framework.lower()))
    
    # Architecture
    if architecture:
        components.append(clean_name_component(architecture))
    
    # Backbone (only if different from architecture)
    if backbone and backbone != architecture:
        components.append(clean_name_component(backbone))
    
    # Dataset name
    if dataset_name:
        components.append(clean_name_component(dataset_name))
    
    # Task type
    if task_type:
        components.append(clean_name_component(task_type))
    
    # Timestamp (always last before extension)
    components.append(timestamp)
    
    # Join components with underscores
    model_name = "_".join(components)
    
    # Add extension if provided
    if extension and extension is not False:
        if not extension.startswith('.'):
            extension = '.' + extension
        model_name += extension
    elif framework and extension is not False:
        # Infer extension from framework
        framework_extensions = {
            'pytorch': '.pt',
            'tensorflow': '.keras',
            'keras': '.keras',
            'scikit-learn': '.joblib',
            'sklearn': '.joblib',
            'onnx': '.onnx'
        }
        ext = framework_extensions.get(framework.lower())
        if ext:
            model_name += ext
    
    return model_name


def clean_name_component(name: str) -> str:
    """
    Clean a name component for use in filenames.
    
    Args:
        name: Raw name component
        
    Returns:
        str: Cleaned name safe for filenames
    """
    if not name:
        return ""
    
    # Convert to lowercase and replace problematic characters
    cleaned = re.sub(r'[^\w\-_]', '_', str(name).lower())
    
    # Remove multiple consecutive underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    
    # Limit length to prevent overly long filenames
    if len(cleaned) > 20:
        cleaned = cleaned[:20].rstrip('_')
    
    return cleaned


def extract_dataset_name(dataset_path: Optional[str] = None, 
                        builtin_dataset: Optional[dict] = None) -> str:
    """
    Extract a clean dataset name from path or builtin dataset info.
    
    Args:
        dataset_path: Path to dataset directory
        builtin_dataset: Builtin dataset info dict
        
    Returns:
        str: Clean dataset name
    """
    if builtin_dataset and 'name' in builtin_dataset:
        return clean_name_component(builtin_dataset['name'])
    
    if dataset_path:
        # Extract folder name from path
        path_obj = Path(dataset_path)
        dataset_name = path_obj.name
        
        # Handle common dataset patterns
        if dataset_name.lower() in ['train', 'training', 'data', 'images', 'raw']:
            # Use parent directory name
            dataset_name = path_obj.parent.name
            
            # If parent is also generic, go up one more level
            if dataset_name.lower() in ['data', 'datasets', 'extracted']:
                dataset_name = path_obj.parent.parent.name
        
        # Clean up common dataset suffixes/prefixes  
        dataset_name = re.sub(r'[-_](train|training|test|val|validation|data|dataset|images|batches|py)$', '', dataset_name, flags=re.IGNORECASE)
        
        # Handle specific known datasets
        if 'cifar' in dataset_name.lower():
            if '10' in dataset_name:
                dataset_name = 'cifar10'
            elif '100' in dataset_name:
                dataset_name = 'cifar100'
            else:
                dataset_name = 'cifar'
        elif 'mnist' in dataset_name.lower():
            dataset_name = 'mnist'
        elif 'imagenet' in dataset_name.lower():
            dataset_name = 'imagenet'
        
        return clean_name_component(dataset_name)
    
    return "unknown"


def generate_comprehensive_model_paths(
    base_dir: Path,
    framework: str,
    architecture: Optional[str] = None,
    backbone: Optional[str] = None,
    dataset_name: Optional[str] = None,
    task_type: Optional[str] = None,
    timestamp: Optional[str] = None
) -> dict:
    """
    Generate comprehensive model file paths for a training session.
    
    Args:
        base_dir: Base directory for saving models
        framework: ML framework name
        architecture: Model architecture
        backbone: Backbone name
        dataset_name: Dataset name
        task_type: Task type
        timestamp: Custom timestamp
        
    Returns:
        dict: Dictionary with model, config, and results paths
    """
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate base name without extension
    base_name = generate_model_name(
        framework=framework,
        architecture=architecture,
        backbone=backbone,
        dataset_name=dataset_name,
        task_type=task_type,
        timestamp=timestamp,
        extension=False  # No extension for base name
    )
    
    # Framework-specific extensions
    if framework.lower() in ['pytorch']:
        model_ext = '.pt'
    elif framework.lower() in ['tensorflow', 'keras']:
        model_ext = '.keras'
    elif framework.lower() in ['scikit-learn', 'sklearn']:
        model_ext = '.joblib'
    else:
        model_ext = '.model'
    
    paths = {
        'model': base_dir / f"{base_name}{model_ext}",
        'config': base_dir / f"{base_name}_config.yaml",
        'results': base_dir / f"{base_name}_results.json",
        'logs': base_dir / f"{base_name}_logs.txt",
        'base_name': base_name,  # Base name without extension
        'timestamp': timestamp
    }
    
    # Add framework-specific additional files
    if framework.lower() in ['scikit-learn', 'sklearn']:
        paths['scaler'] = base_dir / f"{base_name}_scaler.joblib"
        paths['encoder'] = base_dir / f"{base_name}_encoder.joblib"
    
    return paths


def suggest_model_naming_improvements():
    """
    Suggest additional improvements for model naming and organization.
    
    Returns:
        dict: Dictionary with improvement suggestions
    """
    suggestions = {
        'hierarchical_organization': {
            'description': 'Organize models in hierarchical directories',
            'structure': 'experiments/{framework}/{dataset}/{architecture}/{timestamp}/',
            'benefits': ['Better organization', 'Easy filtering', 'Reduced clutter']
        },
        
        'version_control': {
            'description': 'Add version control integration',
            'features': ['Git commit hash in filename', 'Branch name tracking', 'Experiment lineage'],
            'example': 'pytorch_resnet50_cifar10_classification_20251011_142530_abc123.pt'
        },
        
        'metadata_integration': {
            'description': 'Embed metadata in model files',
            'features': ['Training config', 'Dataset stats', 'Performance metrics', 'Environment info'],
            'benefits': ['Self-contained models', 'Easy comparison', 'Reproducibility']
        },
        
        'performance_tagging': {
            'description': 'Include performance metrics in filename',
            'example': 'pytorch_resnet50_cifar10_acc9534_loss0241_20251011_142530.pt',
            'benefits': ['Quick performance identification', 'Easy model comparison']
        },
        
        'automatic_cleanup': {
            'description': 'Implement automatic model cleanup policies',
            'features': ['Keep top N models', 'Delete models older than X days', 'Size-based cleanup'],
            'benefits': ['Prevent disk overflow', 'Maintain only relevant models']
        },
        
        'cloud_integration': {
            'description': 'Cloud storage and model registry integration',
            'features': ['Auto-upload to cloud', 'Model versioning', 'Shared model registry'],
            'platforms': ['MLflow', 'Weights & Biases', 'Hugging Face Hub', 'AWS SageMaker']
        }
    }
    
    return suggestions


# Example usage and testing
if __name__ == "__main__":
    # Test model name generation
    test_cases = [
        {
            'framework': 'pytorch',
            'architecture': 'resnet50',
            'backbone': 'resnet',
            'dataset_name': 'cifar-10',
            'task_type': 'classification'
        },
        {
            'framework': 'tensorflow',
            'architecture': 'efficientnet_b0',
            'dataset_name': 'ImageNet',
            'task_type': 'classification'
        },
        {
            'framework': 'scikit-learn',
            'architecture': 'random_forest',
            'dataset_name': 'custom_dataset',
            'task_type': 'classification'
        }
    ]
    
    print("🔍 Testing Model Name Generation:")
    print("=" * 60)
    
    for i, case in enumerate(test_cases, 1):
        model_name = generate_model_name(**case)
        print(f"{i}. {model_name}")
    
    print("\n📊 Model Naming Improvement Suggestions:")
    print("=" * 60)
    
    suggestions = suggest_model_naming_improvements()
    for key, suggestion in suggestions.items():
        print(f"\n🚀 {suggestion['description']}")
        if 'example' in suggestion:
            print(f"   Example: {suggestion['example']}")
        if 'benefits' in suggestion:
            print(f"   Benefits: {', '.join(suggestion['benefits'])}")