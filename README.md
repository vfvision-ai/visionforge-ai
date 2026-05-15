# 🔭 VisionForge — Train Vision Models Effortlessly

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-red.svg)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange.svg)](https://pytorch.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Latest-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Production Ready](https://img.shields.io/badge/Production-Ready-brightgreen.svg)](PRODUCTION_CHECKLIST.md)

**Train Vision Models Effortlessly** — VisionForge is a comprehensive, zero-configuration AutoML platform that automates the entire machine learning pipeline for computer vision tasks.

> **🚀 Production Ready!** See [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment guide and [PRODUCTION_CHECKLIST.md](PRODUCTION_CHECKLIST.md) for the complete readiness checklist.

## 🚀 Quick Start with Docker

```bash
# Clone the repository
git clone https://github.com/vfvision-ai/visionforge-ai.git
cd visionforge

# Deploy with one command
./deploy.sh dev

# Access at http://localhost:8501
```

For detailed Docker deployment, see the included `docker-compose.yml` and `Dockerfile`.

## 🔮 Quick Start with Inference

After training a model, use it for predictions:

```bash
# Single image inference
python inference.py --model experiments/your_model.pt --image test.jpg --visualize

# Batch inference
python inference.py --model experiments/your_model.pt --batch test_images/ --output results/

# Convert to ONNX for faster deployment
python inference.py --model experiments/your_model.pt --convert-onnx experiments/model.onnx

# Use ONNX model for inference
python inference.py --use-onnx experiments/model.onnx --image test.jpg

# Python API
from inference import PyTorchInference
inference = PyTorchInference('experiments/your_model.pt')
result = inference.predict_image('test.jpg', top_k=5)

# ONNX conversion
onnx_path = inference.convert_to_onnx('model.onnx')
```

The comprehensive inference functionality is built into the web interface and `inference.py` script.

## 🌟 Key Features

### 🔍 **Automatic Dataset Analysis**
- **Smart Task Detection**: Automatically identifies classification, detection, or segmentation tasks
- **Dataset Statistics**: Analyzes image dimensions, channels, class distribution, and quality
- **Performance Predictions**: Estimates training time and resource requirements
- **Multi-format Support**: Handles various image formats (JPEG, PNG, BMP, TIFF, etc.)
- **Meaningful Class Names**: Displays actual class names (e.g., "airplane", "cat", "dog") instead of generic labels ("class_0", "class_1")

### 🤗 **Seamless Hugging Face Integration**
- **Curated Datasets**: Pre-tested, verified datasets that work out-of-the-box
- **Public Datasets Only**: Reliable, accessible datasets with no authentication required
- **One-Click Selection**: 10+ optimized datasets across all CV tasks
- **Classification**: CIFAR-10, Fashion-MNIST, Cats vs Dogs, Food-101, Indoor Scene
- **Segmentation**: Oxford-IIIT Pet (37 breeds, 3-class masks), ADE20K Scene Parsing (150 categories)
- **Smart Error Handling**: Intelligent fallbacks and helpful solutions
- **Manual Override**: Advanced users can still access any HF dataset

### 🧠 **Intelligent Model Selection**
- **🚀 Modern Architectures**: State-of-the-art EfficientNet, MobileNetV3, RegNet, and ConvNeXt models
- **⚡ Optimized Performance**: Better accuracy and efficiency than legacy Simple CNN models
- **🎯 Smart Recommendations**: Automatically selects the best model architecture for your data characteristics
- **Multi-Framework Support**: PyTorch, TensorFlow/Keras, and Scikit-learn with unified interface
- **🏗️ Transfer Learning**: Leverages pretrained models for faster training and better results
- **⚙️ Custom Configuration**: Manual model selection with advanced parameter tuning and real-time guidance

### 🚀 **Zero-Configuration Training**
- **Hyperparameter Optimization**: Bayesian optimization for best results
- **Real-time Monitoring**: Live training progress with rich visualizations
- **Automatic Callbacks**: Early stopping, learning rate scheduling, model checkpointing
- **GPU/CPU Support**: Automatic hardware detection and optimization

### 🔮 **Production-Ready Inference**
- **Easy Model Loading**: Load trained models with complete metadata
- **Intelligent Class Names**: Automatic detection of proper class names (e.g., "airplane", "cat", "dog")
- **ONNX Export**: Convert PyTorch models to ONNX for cross-platform deployment
- **Flexible API**: Command-line interface and Python API for integration
- **Batch Processing**: Efficient inference on multiple images
- **Performance Optimization**: ONNX models often provide faster inference than PyTorch
- **Visualization**: Generate prediction charts and confidence visualizations

### 📊 **Comprehensive Results Analysis**
- **Performance Metrics**: Accuracy, loss, precision, recall, F1-score
- **Visual Analytics**: Training curves, confusion matrices, class predictions
- **Model Comparison**: Side-by-side comparison with performance benchmarks
- **Export Options**: Download models, results, and configuration files

### 🔄 **Advanced Model Management**
- **Intelligent Naming**: `framework_architecture_backbone_dataset_task_timestamp.ext`
- **Bulk Operations**: Download multiple models as organized ZIP archives
- **Version Control**: Automatic model versioning with metadata
- **Cloud Integration**: Ready for cloud storage and model registry

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [User Interface Guide](#-user-interface-guide)
- [Command Line Interface](#-command-line-interface)
- [Supported Tasks & Models](#-supported-tasks--models)
- [Dataset Formats](#-dataset-formats)
- [Configuration](#-configuration)
- [Examples](#-examples)
- [API Reference](#-api-reference)
- [Troubleshooting](#-troubleshooting)
- [Recent Updates](#-recent-updates)
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Installation

### Prerequisites

- **Python 3.9+** (Python 3.10+ recommended for best performance)
- **CUDA 11.0+** (optional, for GPU acceleration)
- **8GB+ RAM** (16GB+ recommended for large datasets)
- **10GB+ Disk Space** (for models and datasets)

### Method 1: Conda Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/vfvision-ai/visionforge-ai.git
cd visionforge

# Create conda environment
conda create -n mlplatform python=3.9
conda activate mlplatform

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### Method 2: Virtual Environment

```bash
# Clone the repository
git clone https://github.com/vfvision-ai/visionforge-ai.git
cd visionforge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## ⚡ Quick Start

### 1. Web Interface (Recommended)

```bash
# Activate your environment
conda activate mlplatform  # or source venv/bin/activate

# Launch the web interface
streamlit run app.py

# Open your browser to http://localhost:8501
```

### 2. Features at a Glance

- **🚀 Modern Models**: EfficientNet-B0, MobileNetV3-Small, RegNetY-002, ConvNeXt-Tiny
- **⚡ Smart Training**: Automatic dataset analysis and model selection
- **🎯 Multi-Framework**: PyTorch, TensorFlow/Keras, Scikit-learn support
- **📊 Real-Time Monitoring**: Live training progress and interactive visualizations
- **🔧 Zero Configuration**: Works out-of-the-box with intelligent defaults

### 3. Python API

```python
from core.dataset_analyzer import DatasetAnalyzer
from core.model_selector import ModelSelector
from core.trainer import AutoTrainer

# Analyze your dataset
analyzer = DatasetAnalyzer()
dataset_info = analyzer.analyze_dataset("path/to/dataset")

# Select optimal model
selector = ModelSelector()
model_config = selector.select_model(dataset_info)

# Train the model
trainer = AutoTrainer(dataset_info, model_config, "path/to/dataset", "./experiments")
results = trainer.train()

print(f"Training completed! Best accuracy: {results.best_accuracy:.3f}")
```

## 🖥️ User Interface Guide

### Home Page 🏠
- **Framework Selection**: Choose between PyTorch, TensorFlow/Keras, or Scikit-learn
- **System Status**: Check GPU availability and dependencies
- **Quick Actions**: Start new project or load existing results

### Dataset Analysis 📊
- **Upload Options**: 
  - Browse local folders
  - Upload ZIP archives (up to 10GB)
  - Use built-in datasets (CIFAR-10, CIFAR-100, MNIST)
  - Manual path entry
- **Analysis Results**: Task type, class distribution, image statistics
- **Recommendations**: Optimal batch size and training time estimates

### Model Selection 🧠
- **Auto-Select**: Intelligent model recommendation based on dataset
- **Manual Selection**: Choose from 20+ architectures with customization
- **Pre-trained Models**: Load existing model files
- **Configuration Editor**: Fine-tune hyperparameters and architecture

### Training 🔥
- **Progress Monitoring**: Real-time training metrics and visualizations
- **Resource Usage**: GPU/CPU utilization and memory consumption
- **Early Stopping**: Automatic training termination for optimal results
- **Output Management**: Organized experiment directories

### Results 📈
- **Performance Metrics**: Comprehensive evaluation with charts
- **Model Comparison**: Benchmark against other architectures
- **Download Options**: Individual models, bulk ZIP, JSON metadata
- **Historical Analysis**: Browse previous training sessions

### Settings ⚙️
- **Global Configuration**: System-wide preferences
- **Model Management**: Browse and manage all trained models
- **Export Tools**: Bulk operations and cleanup utilities

## 💻 Web Interface (Primary Method)

### Launch the Application

```bash
# Start the web interface
streamlit run app.py

# Access at http://localhost:8501
# Use the intuitive web interface for all training tasks
```

### Streamlit Interface Benefits
- **🖱️ Point-and-Click**: No command line required
- **📊 Real-Time Monitoring**: Live training progress and visualizations  
- **🎯 Smart Guidance**: Automated model selection and configuration
- **📁 Easy Dataset Upload**: Drag-and-drop file upload support
- **🔄 Interactive Configuration**: Dynamic parameter adjustment

### Alternative: Inference CLI

```bash
# For trained model inference only
python inference.py --model experiments/your_model.pt --image test.jpg --visualize

# Batch inference
python inference.py --model experiments/your_model.pt --batch test_images/ --output results/

# Convert to ONNX for deployment
python inference.py --model experiments/your_model.pt --convert-onnx model.onnx
```

### Python API Integration

```python
# Import and use in your own projects
from core.dataset_analyzer import DatasetAnalyzer
from core.model_selector import ModelSelector
from core.trainer import AutoTrainer

# Programmatic training
analyzer = DatasetAnalyzer()
dataset_info = analyzer.analyze('./data/my_dataset')
selector = ModelSelector()
model_config = selector.select_optimal_model(dataset_info)
```

## 🎯 Supported Tasks & Models

### Image Classification 📸

| Framework | Architecture | Backbone | Parameters | Speed | Accuracy | Status |
|-----------|-------------|----------|------------|-------|----------|---------|
| PyTorch | **🚀 EfficientNet-B0** | EfficientNet | 5.3M | Good | **Excellent** | ✅ **Recommended** |
| PyTorch | **📱 MobileNetV3-Small** | MobileNet | 2.5M | **Excellent** | Very Good | ✅ **Fast** |
| PyTorch | **⚡ RegNetY-002** | RegNet | 3.2M | Very Good | **Excellent** | ✅ **Balanced** |
| PyTorch | ResNet50 | ResNet | 25.6M | Fast | High | ✅ Available |
| TensorFlow | **🚀 EfficientNet-B0** | EfficientNet | 5.3M | Good | **Excellent** | ✅ **Recommended** |
| TensorFlow | **📱 MobileNetV3-Small** | MobileNet | 2.5M | **Excellent** | Very Good | ✅ **Fast** |
| TensorFlow | **⚡ RegNetY-002** | RegNet | 3.2M | Very Good | **Excellent** | ✅ **Balanced** |
| TensorFlow | **⚡ ConvNeXt-Tiny** | ConvNeXt | 28.6M | Good | **Outstanding** | ✅ **Modern** |
| TensorFlow | ResNet50 | ResNet | 25.6M | Fast | High | ✅ Available |
| Scikit-learn | Random Forest | N/A | Variable | Fast | Medium | ✅ Available |
| Scikit-learn | SVM | N/A | Variable | Medium | Medium | ✅ Available |

### Object Detection 🎯

| Framework | Architecture | Backbone | mAP | Speed |
|-----------|-------------|----------|-----|-------|
| PyTorch | YOLOv5 | CSPDarknet | 0.65 | Fast |
| PyTorch | Faster R-CNN | ResNet50 | 0.70 | Medium |
| PyTorch | RetinaNet | ResNet50 | 0.68 | Medium |
| TensorFlow | SSD MobileNet | MobileNet | 0.55 | Very Fast |

### Semantic Segmentation 🖼️

| Framework | Architecture | Backbone | mIoU | Speed |
|-----------|-------------|----------|------|-------|
| PyTorch | U-Net | ResNet34 | 0.75 | Fast |
| PyTorch | DeepLabV3+ | ResNet50 | 0.82 | Medium |
| PyTorch | PSPNet | ResNet50 | 0.80 | Medium |

## 📁 Dataset Formats

### Directory Structure

The system supports multiple dataset organization formats:

#### Classification Format
```
dataset/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── image1.jpg
│   │   └── ...
│   └── ...
├── val/
│   ├── class1/
│   └── class2/
└── test/ (optional)
    ├── class1/
    └── class2/
```

#### Detection Format (COCO/YOLO)
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── annotations/
│   ├── instances_train.json
│   ├── instances_val.json
│   └── instances_test.json
└── labels/ (for YOLO format)
    ├── train/
    └── val/
```

#### Segmentation Format
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── masks/
│   ├── train/
│   ├── val/
│   └── test/
└── annotations/ (optional)
```

### Built-in Datasets

The system includes several popular datasets:

- **CIFAR-10**: 60,000 32×32 color images in 10 classes
- **CIFAR-100**: 60,000 32×32 color images in 100 classes  
- **MNIST**: 70,000 28×28 grayscale digit images
- **Fashion-MNIST**: 70,000 28×28 fashion item images
- **ImageNet** (subset): High-resolution natural images

### Supported File Formats

- **Images**: JPEG, PNG, BMP, TIFF, WebP, GIF
- **Archives**: ZIP, TAR, RAR, 7Z (up to 10GB)
- **Annotations**: JSON (COCO), XML (PASCAL VOC), TXT (YOLO)

## ⚙️ Configuration

### Configuration Files

The system uses several configuration files:

#### `pyproject.toml` - Project Configuration
```toml
[tool.mlplatform]
default_framework = "pytorch"
default_batch_size = 32
default_epochs = 100
auto_gpu = true
experiment_dir = "./experiments"
```

#### `experiments/config.yaml` - Training Configuration
```yaml
dataset:
  path: "./data/my_dataset"
  task_type: "classification"
  num_classes: 10

model:
  framework: "pytorch"
  architecture: "resnet50"
  backbone: "resnet"
  pretrained: true
  input_size: [3, 224, 224]

training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  optimizer: "adam"
  scheduler: "cosine"

system:
  device: "cuda"
  num_workers: 4
  mixed_precision: true
```

### Environment Variables

```bash
# Optional environment variables
export ML_DATA_DIR="/path/to/datasets"
export ML_MODEL_DIR="/path/to/models"
export ML_EXPERIMENT_DIR="/path/to/experiments"
export CUDA_VISIBLE_DEVICES="0,1"  # GPU selection
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
```

## 📚 Examples

### Example 1: Image Classification

```python
from core.dataset_analyzer import DatasetAnalyzer
from core.model_selector import ModelSelector
from core.trainer import AutoTrainer

# Analyze CIFAR-10 dataset
analyzer = DatasetAnalyzer()
dataset_info = analyzer.analyze_dataset(
    dataset_path="./data/cifar-10-batches-py",
    dataset_type="builtin",
    builtin_name="CIFAR-10"
)

print(f"Task: {dataset_info.task_type}")
print(f"Classes: {dataset_info.num_classes}")
print(f"Samples: {dataset_info.num_samples}")

# Auto-select best model
selector = ModelSelector()
model_config = selector.select_model(dataset_info)

print(f"Selected: {model_config.architecture}")
print(f"Parameters: {model_config.num_parameters:,}")

# Train the model
trainer = AutoTrainer(
    dataset_info=dataset_info,
    model_config=model_config,
    dataset_path="./data/cifar-10-batches-py",
    output_dir="./experiments"
)

results = trainer.train()
print(f"Best accuracy: {results.best_accuracy:.3f}")
```

### Example 2: Custom Dataset Classification

```python
import os
from pathlib import Path

# Prepare your dataset in the correct format
dataset_path = "./data/my_custom_dataset"

# Directory structure:
# my_custom_dataset/
# ├── train/
# │   ├── cats/
# │   └── dogs/
# └── val/
#     ├── cats/
#     └── dogs/

# Analyze custom dataset
analyzer = DatasetAnalyzer()
dataset_info = analyzer.analyze_dataset(dataset_path)

# Manual model selection
selector = ModelSelector()
model_config = selector.select_model(
    dataset_info,
    architecture_preference="efficientnet_b0",
    performance_priority="accuracy"  # or "speed" or "balanced"
)

# Custom training configuration
model_config.config_params.update({
    'learning_rate': 0.0001,
    'batch_size': 16,
    'epochs': 50,
    'optimizer': 'adamw',
    'scheduler': 'cosine',
    'augmentation': True
})

# Train with custom settings
trainer = AutoTrainer(dataset_info, model_config, dataset_path, "./experiments")
results = trainer.train()
```

### Example 3: Batch Processing Multiple Datasets

```python
import glob
from pathlib import Path

datasets = glob.glob("./data/*/")  # Find all dataset directories

results_summary = []

for dataset_path in datasets:
    dataset_name = Path(dataset_path).name
    print(f"Processing dataset: {dataset_name}")
    
    try:
        # Analyze dataset
        analyzer = DatasetAnalyzer()
        dataset_info = analyzer.analyze_dataset(dataset_path)
        
        # Auto-select model
        selector = ModelSelector()
        model_config = selector.select_model(dataset_info)
        
        # Train model
        trainer = AutoTrainer(
            dataset_info, 
            model_config, 
            dataset_path, 
            f"./experiments/{dataset_name}"
        )
        results = trainer.train()
        
        results_summary.append({
            'dataset': dataset_name,
            'architecture': model_config.architecture,
            'accuracy': results.best_accuracy,
            'training_time': results.training_time
        })
        
    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")

# Print summary
for result in results_summary:
    print(f"{result['dataset']}: {result['accuracy']:.3f} accuracy with {result['architecture']}")
```

## 📖 API Reference

### Core Classes

#### `DatasetAnalyzer`
```python
class DatasetAnalyzer:
    def analyze_dataset(self, dataset_path: str, dataset_type: str = "auto", builtin_name: str = None) -> DatasetInfo:
        """Analyze dataset and extract metadata."""
        
    def get_class_distribution(self, dataset_path: str) -> Dict[str, int]:
        """Get class distribution statistics."""
        
    def estimate_training_time(self, dataset_info: DatasetInfo, model_config: ModelConfig) -> float:
        """Estimate training time in hours."""
```

#### `ModelSelector`
```python
class ModelSelector:
    def select_model(self, dataset_info: DatasetInfo, architecture_preference: str = None) -> ModelConfig:
        """Select optimal model for dataset."""
        
    def get_available_models(self, task_type: str, framework: str = "pytorch") -> List[str]:
        """Get list of available models."""
        
    def compare_models(self, dataset_info: DatasetInfo) -> pd.DataFrame:
        """Compare all available models."""
```

#### `AutoTrainer`
```python
class AutoTrainer:
    def __init__(self, dataset_info: DatasetInfo, model_config: ModelConfig, dataset_path: str, output_dir: str):
        """Initialize trainer with configuration."""
        
    def train(self) -> TrainingResults:
        """Start training process."""
        
    def evaluate(self, model_path: str) -> Dict[str, float]:
        """Evaluate trained model."""
        
    def predict(self, image_path: str, model_path: str) -> Dict[str, Any]:
        """Make prediction on single image."""
```

### Utility Functions

#### Model Naming
```python
from utils.model_naming import generate_model_name, extract_dataset_name

# Generate intelligent model names
model_name = generate_model_name(
    framework="pytorch",
    architecture="resnet50",
    backbone="resnet",
    dataset_name="cifar10",
    task_type="classification"
)
# Output: pytorch_resnet50_resnet_cifar10_classification_20251011_143022.pt

# Extract dataset names from paths
dataset_name = extract_dataset_name(dataset_path="./data/cifar-10-batches-py")
# Output: "cifar10"
```

#### Configuration Management
```python
from utils.config import Config

config = Config()
config.update_model_config("resnet50", {"learning_rate": 0.001})
config.save_experiment_config("./experiments/exp_001")
```

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```
Error: CUDA out of memory. Tried to allocate X GB
```
**Solutions:**
- Reduce batch size: `--batch_size 16` or `--batch_size 8`
- Use mixed precision: Enable in settings or add `--mixed_precision`
- Use CPU training: `--device cpu`
- Reduce image resolution in model configuration

#### 2. Dataset Not Found
```
Error: Dataset directory does not exist or is empty
```
**Solutions:**
- Check dataset path: Ensure correct absolute or relative path
- Verify directory structure: Follow supported formats
- Check permissions: Ensure read access to dataset directory
- Use built-in datasets for testing: Select CIFAR-10 or MNIST

#### 3. Model Loading Failed
```
Error: Failed to load pretrained model
```
**Solutions:**
- Check internet connection for downloading pretrained weights
- Disable pretrained weights: `use_pretrained=False`
- Update PyTorch/TensorFlow to latest version
- Clear model cache: Delete `~/.cache/torch/hub/` or `~/.keras/`

#### 4. Streamlit Port Issues
```
Error: Port 8501 is already in use
```
**Solutions:**
```bash
# Use different port
streamlit run app.py --server.port 8502

# Kill existing Streamlit processes
pkill -f streamlit

# Find and kill process using port
lsof -ti:8501 | xargs kill -9
```

#### 5. Import Errors
```
ImportError: No module named 'torch'
```
**Solutions:**
```bash
# Ensure environment is activated
conda activate mlplatform

# Reinstall dependencies
pip install -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

### Performance Optimization

#### GPU Optimization
```python
# Enable mixed precision
model_config.config_params['mixed_precision'] = True

# Optimize CUDA settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Use optimal number of workers
model_config.config_params['num_workers'] = min(8, os.cpu_count())
```

#### Memory Management
```python
# Gradient accumulation for large batch sizes
model_config.config_params['gradient_accumulation_steps'] = 4

# Enable gradient checkpointing
model_config.config_params['gradient_checkpointing'] = True

# Use efficient data loading
model_config.config_params['pin_memory'] = True
model_config.config_params['persistent_workers'] = True
```

### Debug Mode

Enable verbose logging for debugging:

```bash
# Enable debug mode via environment variable
export ML_DEBUG=1
streamlit run app.py

# Or use the debug options in the web interface settings
```

```python
# Python API
import logging
logging.basicConfig(level=logging.DEBUG)
```



### Enhanced Accuracy Tracking
- **📊 Detailed Metrics**: 6-decimal precision accuracy tracking with percentage display
- **⚡ Real-Time Analysis**: Live overfitting detection and health status monitoring
- **📈 Performance Dashboard**: Comprehensive accuracy visualization with interactive charts
- **🏆 Best Performance Tracking**: Automatic identification of peak accuracy and corresponding epochs
- **📊 Trend Analysis**: Moving averages, improvement tracking, and learning efficiency metrics
- **💾 Export Capabilities**: JSON-serializable metrics for external analysis and research

### Large File & Multi-Channel Support
- **📦 Large File Support**: Files up to **10GB** with progress tracking
- **🔍 Multi-Channel Images**: Support for grayscale (1), RGB (3), RGBA (4), and custom channels
- **🎯 Small Image Optimization**: MNIST (28×28) and CIFAR (32×32) optimizations
- **⚡ Memory Efficiency**: Chunked processing and smart resource management
- **📁 Multiple Formats**: ZIP, TAR, RAR, 7Z archive support

### System Architecture

```
visionforge/
├── core/                 # Core training modules
│   ├── dataset_analyzer.py
│   ├── model_selector.py
│   ├── trainer.py
│   ├── tensorflow_trainer.py
│   ├── sklearn_trainer.py
│   └── optimizer.py
├── utils/                # Utility functions
│   ├── config.py
│   ├── model_naming.py   # Intelligent naming system
│   ├── callbacks.py
│   ├── metrics.py
│   ├── model_factory.py
│   ├── data_factory.py
│   └── logger.py
├── data/                 # Built-in datasets storage
│   ├── MNIST/           # MNIST dataset cache
│   ├── cifar-10-batches-py/   # CIFAR-10 dataset
│   └── cifar-100-python/      # CIFAR-100 dataset
├── .github/              # GitHub configuration
│   └── copilot-instructions.md
├── app.py               # Streamlit web interface (PRIMARY)
├── inference.py         # Model inference and ONNX conversion
├── requirements.txt     # Production dependencies
├── requirements-dev.txt # Development dependencies
├── pyproject.toml       # Project configuration
├── Dockerfile           # Docker container definition
├── docker-compose.yml   # Docker orchestration
├── deploy.sh            # Deployment script
├── nginx.conf           # Nginx configuration
└── README.md            # This file
```

**Note**: The following directories are auto-created during usage:
- `uploads/` - Temporary storage for uploaded datasets (created on first upload)
- `experiments/` - Training outputs with intelligent naming (created on first training)
- `.venv/` - Python virtual environment (if using venv)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/vfvision-ai/visionforge-ai.git
cd visionforge

# Create development environment
conda env create -f environment-dev.yml
conda activate mlplatform-dev

# Install in development mode
pip install -e .

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Run linting
flake8 src/
black src/
mypy src/
```

### Testing

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/test_dataset_analyzer.py
pytest tests/test_model_selector.py
pytest tests/test_trainer.py

# Run with coverage
pytest --cov=src --cov-report=html

# Performance tests
pytest tests/test_performance.py --benchmark-only
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Team**: For the excellent deep learning framework
- **Streamlit Team**: For the amazing web app framework
- **Hugging Face**: For pretrained models and datasets
- **OpenCV**: For computer vision utilities
- **scikit-learn**: For traditional ML algorithms
- **Plotly**: For interactive visualizations

## 📞 Support

- **Documentation**: [Full Documentation](https://docs.example.com)
- **Issues**: [GitHub Issues](https://github.com/vfvision-ai/visionforge-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/vfvision-ai/visionforge-ai/discussions)
- **Email**: support@example.com

## 🗺️ Roadmap

### Version 2.0 (Coming Soon)
- [ ] Docker containerization
- [ ] Kubernetes deployment support
- [ ] Advanced model architectures (Transformers, CLIP)
- [ ] Multi-GPU training support
- [ ] Cloud storage integration (AWS S3, Google Cloud, Azure)

### Version 2.1
- [ ] Model serving and deployment
- [ ] REST API for model inference
- [ ] Mobile app support
- [ ] Advanced augmentation techniques
- [ ] Federated learning support

### Version 3.0
- [ ] AutoML capabilities
- [ ] Neural architecture search
- [ ] Model compression and quantization
- [ ] Edge deployment optimization
- [ ] Real-time inference pipeline

---

**Happy Training! 🚀**

*Made with ❤️*
