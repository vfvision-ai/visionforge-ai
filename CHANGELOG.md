# Changelog

All notable changes to VisionForge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2026-06-08

### 🎉 Major New Features

#### Model Explainability & Interpretability
- **Added** Grad-CAM (Gradient-weighted Class Activation Mapping) implementation
- **Added** Saliency map generation for pixel-level importance
- **Added** Integrated Gradients for stable attribution
- **Added** Framework-agnostic interface supporting both PyTorch and TensorFlow
- **Added** Automatic visualization overlays on original images
- **Added** `core/explainability.py` module with `ExplainabilityManager` class

#### Model Versioning & Experiment Tracking
- **Added** Complete model registry with semantic versioning
- **Added** Experiment tracking system with full metadata
- **Added** Model lineage tracking (parent-child relationships)
- **Added** Model stage promotion (development → staging → production)
- **Added** Version comparison and diff capabilities
- **Added** Best experiment selection based on custom metrics
- **Added** `core/versioning.py` module with `ModelRegistry` and `ExperimentTracker`

#### Enterprise Security Features
- **Added** Sliding window rate limiter with burst support
- **Added** API key management with rotation and expiration
- **Added** Request signature validation using HMAC-SHA256
- **Added** IP-based access control (whitelist/blacklist)
- **Added** Comprehensive security audit logging
- **Added** Protection against replay attacks
- **Added** `utils/security.py` module with all security components

#### Model Compression & Optimization
- **Added** Post-training quantization (INT8, FP16)
- **Added** Quantization-aware training (QAT) support
- **Added** Model pruning (structured and unstructured)
- **Added** Knowledge distillation for model compression
- **Added** ONNX export with optimization
- **Added** Inference speed benchmarking utilities
- **Added** `core/compression.py` module with `ModelCompressor` class

#### Production Monitoring & Alerting
- **Added** Real-time metrics collection (Prometheus compatible)
- **Added** System resource monitoring (CPU, GPU, memory, disk)
- **Added** Custom alert rules with configurable thresholds
- **Added** Health check system with dependency verification
- **Added** Performance profiling and tracking
- **Added** Automatic metric aggregation
- **Added** `utils/monitoring.py` module with monitoring components

#### Data Versioning & Lineage
- **Added** Dataset version control system
- **Added** Data lineage tracking and provenance
- **Added** Automatic change detection with hash verification
- **Added** Dataset comparison and diff capabilities
- **Added** Data quality validation with corrupt image detection
- **Added** Stage promotion for datasets
- **Added** `core/data_versioning.py` module

### 🔧 Enhancements

- **Updated** Project version to 2.1.0
- **Enhanced** Documentation with v2.1 features
- **Added** New dependencies for compression and monitoring
- **Improved** Overall system robustness and production-readiness

### 📚 Documentation

- **Updated** README.md with comprehensive v2.1 feature list
- **Added** CHANGELOG.md for version tracking
- **Added** Inline documentation for all new modules

### 🔒 Security

- **Added** Rate limiting to prevent API abuse
- **Added** API key authentication system
- **Added** Security audit logging
- **Added** Request signature validation
- **Added** IP-based access control

### 🐛 Bug Fixes

- None (this is a feature release)

### 🔄 Migration Guide

#### From 2.0.0 to 2.1.0

**Backwards Compatibility**: Version 2.1.0 is fully backwards compatible with 2.0.0. All existing code will continue to work.

**Optional Upgrades**:

1. **Enable Model Versioning**:
   ```python
   from core.versioning import ModelRegistry
   
   registry = ModelRegistry("./model_registry")
   model_version = registry.register_model(model_version_metadata, model_path)
   ```

2. **Add Explainability**:
   ```python
   from core.explainability import explain_prediction
   
   results = explain_prediction(model, image, method='gradcam')
   ```

3. **Enable Monitoring**:
   ```python
   from utils.monitoring import get_system_monitor
   
   monitor = get_system_monitor()
   monitor.start_monitoring(interval_seconds=10)
   ```

4. **Add Security Features**:
   ```python
   from utils.security import get_rate_limiter, RateLimitConfig
   
   config = RateLimitConfig(max_requests=100, window_seconds=60)
   limiter = get_rate_limiter("api", config)
   ```

5. **Install New Dependencies**:
   ```bash
   pip install -r requirements.txt
   
   # Optional for enhanced features:
   pip install GPUtil  # GPU monitoring
   pip install tf2onnx tensorflow-model-optimization  # TF compression
   ```

**No Breaking Changes**: All existing APIs remain unchanged.

---

## [2.0.0] - Previous Release

### Initial Production Release
- Full-stack AutoML platform
- Streamlit web UI
- FastAPI REST API
- Next.js dashboard
- JWT authentication
- Celery async workers
- Multi-framework support (PyTorch, TensorFlow, Scikit-learn)
- Automated dataset analysis
- Hyperparameter optimization
- Production deployment with Docker

---

## Version Numbering

VisionForge follows [Semantic Versioning](https://semver.org/):

- **MAJOR** version (X.0.0): Incompatible API changes
- **MINOR** version (0.X.0): New functionality, backwards compatible
- **PATCH** version (0.0.X): Backwards compatible bug fixes
