# VisionForge — Train Vision Models Effortlessly

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![Version](https://img.shields.io/badge/Version-2.1.0-blue.svg)](pyproject.toml)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-red.svg)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.124-009688.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange.svg)](https://pytorch.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Latest-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

VisionForge is a comprehensive, production-ready AutoML platform that automates the entire machine learning pipeline for computer vision tasks. It ships with a Streamlit web UI, a FastAPI REST API, a Next.js dashboard, async training workers via Celery, and JWT-based authentication — all orchestrated with Docker Compose.

## Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [REST API](#rest-api)
- [Authentication](#authentication)
- [Database Migrations](#database-migrations)
- [Supported Models](#supported-models)
- [Dataset Formats](#dataset-formats)
- [Configuration](#configuration)
- [Docker Deployment](#docker-deployment)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Architecture

VisionForge is composed of several cooperating services:

| Service | Technology | Default Port |
|---------|-----------|-------------|
| Web UI | Streamlit | 8501 |
| REST API | FastAPI | 8000 |
| Frontend Dashboard | Next.js 14 | 3000 |
| Async Workers | Celery | — |
| Task Queue / Cache | Redis 7 | 6379 |
| Database (dev) | SQLite | — |
| Database (production) | PostgreSQL 16 | 5432 |
| Worker Monitoring | Celery Flower | 5555 |

## Features

### 🆕 New in v2.1.0

#### Model Explainability & Interpretability
- **Grad-CAM** (Gradient-weighted Class Activation Mapping) for visual explanations
- **Saliency Maps** showing pixel-level importance
- **Integrated Gradients** for stable attribution
- Framework-agnostic interface for PyTorch and TensorFlow
- Automatic visualization overlays on original images

#### Advanced Model Versioning & Experiment Tracking
- Semantic versioning for models (major.minor.patch)
- Complete experiment tracking with metadata and lineage
- Model registry with stage promotion (development → staging → production)
- Experiment comparison and version diffing
- Parent-child relationships for model genealogy
- Best model selection based on custom metrics

#### Enterprise-Grade Security
- **Rate Limiting** with sliding window algorithm and burst support
- **API Key Management** with rotation, expiration, and permissions
- **Request Signature Validation** using HMAC-SHA256
- **IP Whitelisting/Blacklisting** for access control
- **Security Audit Logging** with comprehensive event tracking
- Protection against replay attacks and request tampering

#### Model Compression & Optimization
- **Quantization** (INT8, FP16) for PyTorch and TensorFlow
- **Model Pruning** (structured and unstructured)
- **Knowledge Distillation** for teacher-student training
- **ONNX Export** with optimization
- Model size reduction up to 4x with minimal accuracy loss
- Inference speed benchmarking

#### Production Monitoring & Alerting
- **Real-time Metrics Collection** (Prometheus compatible)
- **System Resource Monitoring** (CPU, GPU, memory, disk)
- **Custom Alert Rules** with configurable thresholds
- **Health Check Endpoints** with dependency verification
- **Performance Profiling** and resource tracking
- Automatic metric aggregation and visualization

#### Data Versioning & Lineage Tracking
- **Dataset Version Control** with semantic versioning
- **Data Lineage Tracking** for full provenance
- **Automatic Change Detection** with hash-based verification
- **Dataset Comparison** and diff capabilities
- **Quality Validation** with corrupt image detection
- Stage promotion (raw → processed → validated → production)

---

### Core Features

#### Automatic Dataset Analysis
- Identifies classification, detection, or segmentation tasks automatically
- Analyzes image dimensions, channels, class distribution, and data quality
- Estimates training time and hardware requirements
- Supports JPEG, PNG, BMP, TIFF, WebP, and ZIP/TAR/RAR/7Z archives up to 10 GB
- Displays meaningful class names (e.g. "airplane", "cat") rather than generic labels

#### Intelligent Model Selection
- Recommends the best architecture based on dataset characteristics
- Supports PyTorch, TensorFlow/Keras, and Scikit-learn with a unified interface
- Leverages pretrained models (ImageNet weights) for transfer learning
- Manual model selection with real-time hyperparameter guidance

#### Zero-Configuration Training
- Bayesian hyperparameter optimization via Optuna
- Real-time training progress with interactive Plotly visualizations
- Automatic callbacks: early stopping, LR scheduling, model checkpointing
- Async job execution through Celery — training runs in the background

#### Authentication & Multi-User Support
- JWT access + refresh token authentication
- Role-based access control (user / admin)
- User registration, login, token refresh, and logout endpoints
- One-time bootstrap endpoint to promote the first admin

#### Production-Ready Inference
- Load trained PyTorch or TensorFlow models with full metadata
- Export PyTorch models to ONNX for cross-platform deployment
- Batch inference on directories of images
- Confidence visualization and top-k predictions

#### REST API
- Full OpenAPI 3.0 documentation at `/docs` and `/redoc`
- Endpoints for experiments, training jobs, models, inference, and auth
- Async job submission — poll job status without blocking

#### Hugging Face Dataset Integration
- One-click access to CIFAR-10, Fashion-MNIST, Cats vs Dogs, Food-101, Indoor Scene, Oxford-IIIT Pet, ADE20K, and more
- No authentication required; all datasets are publicly accessible

## Installation

### Prerequisites

- Python 3.9+ (3.10+ recommended)
- CUDA 11.0+ (optional, for GPU acceleration)
- 8 GB RAM minimum (16 GB+ recommended for large datasets)
- Docker 20.10+ and Docker Compose 2.0+ (for containerized deployment)

### Option A — Virtual Environment

```bash
git clone https://github.com/vfvision-ai/visionforge-ai.git
cd visionforge-ai

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# CPU-only
pip install -r requirements.txt -r requirements-cpu.txt

# GPU (CUDA 12.1)
pip install -r requirements.txt -r requirements-gpu.txt
```

### Option B — Conda

```bash
git clone https://github.com/vfvision-ai/visionforge-ai.git
cd visionforge-ai

conda create -n visionforge python=3.10
conda activate visionforge

pip install -r requirements.txt -r requirements-cpu.txt  # or requirements-gpu.txt
```

### Development dependencies

```bash
pip install -r requirements-dev.txt
```

## Quick Start

### 1. Streamlit Web UI

```bash
streamlit run app.py
# Open http://localhost:8501
```

### 2. FastAPI REST API

```bash
uvicorn api.main:app --reload --port 8000
# Docs at http://localhost:8000/docs
```

### 3. Next.js Frontend

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:3000
```

### 4. Celery Worker (for async training jobs)

```bash
# Start Redis first (or use Docker)
redis-server

# Start worker
celery -A workers.celery_app worker --loglevel=info -Q training
```

### 5. Python API

```python
from core.dataset_analyzer import DatasetAnalyzer
from core.model_selector import ModelSelector
from core.trainer import AutoTrainer

analyzer = DatasetAnalyzer()
dataset_info = analyzer.analyze_dataset("./data/cifar-10-batches-py")

selector = ModelSelector()
model_config = selector.select_model(dataset_info)

trainer = AutoTrainer(dataset_info, model_config, "./data/cifar-10-batches-py", "./experiments")
results = trainer.train()

print(f"Best accuracy: {results.best_accuracy:.4f}")
```

## REST API

The FastAPI service exposes a versioned REST API documented with OpenAPI 3.0.

| URL | Description |
|-----|-------------|
| `http://localhost:8000/docs` | Swagger UI |
| `http://localhost:8000/redoc` | ReDoc |
| `http://localhost:8000/openapi.json` | OpenAPI schema |

### Key endpoint groups

| Prefix | Description |
|--------|-------------|
| `POST /auth/register` | Create a new user account |
| `POST /auth/login` | Obtain access + refresh tokens |
| `POST /auth/refresh` | Refresh an access token |
| `GET /auth/me` | Current authenticated user |
| `GET /health` | Service health check |
| `POST /training/submit` | Submit an async training job |
| `GET /training/{job_id}` | Poll training job status |
| `GET /models` | List trained models |
| `GET /experiments` | List experiments |
| `POST /inference` | Run inference on an uploaded image |

All endpoints except `/health`, `/auth/register`, and `/auth/login` require a Bearer token.

## Authentication

VisionForge uses JWT (RS256-compatible) access and refresh tokens.

```bash
# Register
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "StrongPass1!"}'

# Login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "StrongPass1!"}'

# Use token
curl http://localhost:8000/auth/me \
  -H "Authorization: Bearer <access_token>"

# Bootstrap first admin (one-time, when no admin exists)
curl -X POST http://localhost:8000/auth/bootstrap-admin \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@example.com", "secret": "<SECRET_KEY>"}'
```

Required environment variables:

```env
SECRET_KEY=<long-random-string>
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
```

## Database Migrations

VisionForge uses Alembic for schema migrations backed by SQLAlchemy.

```bash
# Apply all pending migrations
alembic upgrade head

# Create a new migration after changing db/models.py
alembic revision --autogenerate -m "describe your change"

# Roll back one step
alembic downgrade -1
```

The default database is SQLite (`visionforge.db`). Set `DATABASE_URL` in `.env` for PostgreSQL:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/visionforge
```

## Supported Models

### Image Classification

| Framework | Architecture | Parameters | Notes |
|-----------|-------------|------------|-------|
| PyTorch | EfficientNet-B0 | 5.3 M | Recommended |
| PyTorch | MobileNetV3-Small | 2.5 M | Best speed |
| PyTorch | RegNetY-002 | 3.2 M | Balanced |
| PyTorch | ResNet-50 | 25.6 M | Widely tested |
| TensorFlow | EfficientNet-B0 | 5.3 M | Recommended |
| TensorFlow | MobileNetV3-Small | 2.5 M | Best speed |
| TensorFlow | ConvNeXt-Tiny | 28.6 M | Highest accuracy |
| TensorFlow | ResNet-50 | 25.6 M | — |
| Scikit-learn | Random Forest | — | CPU only |
| Scikit-learn | SVM | — | CPU only |

### Object Detection

| Framework | Architecture | Backbone |
|-----------|-------------|----------|
| PyTorch | YOLOv5 | CSPDarknet |
| PyTorch | Faster R-CNN | ResNet-50 |
| PyTorch | RetinaNet | ResNet-50 |
| TensorFlow | SSD MobileNet | MobileNet |

### Semantic Segmentation

| Framework | Architecture | Backbone |
|-----------|-------------|----------|
| PyTorch | U-Net | ResNet-34 |
| PyTorch | DeepLabV3+ | ResNet-50 |
| PyTorch | PSPNet | ResNet-50 |

## Dataset Formats

### Classification

```
dataset/
├── train/
│   ├── cat/
│   └── dog/
├── val/
│   ├── cat/
│   └── dog/
└── test/          # optional
```

### Detection (COCO / YOLO)

```
dataset/
├── images/train/  val/  test/
├── annotations/   # COCO JSON
└── labels/        # YOLO TXT
```

### Segmentation

```
dataset/
├── images/train/  val/  test/
└── masks/train/   val/  test/
```

### Built-in datasets

| Dataset | Images | Classes | Size |
|---------|--------|---------|------|
| CIFAR-10 | 60 000 | 10 | 32×32 |
| CIFAR-100 | 60 000 | 100 | 32×32 |
| MNIST | 70 000 | 10 | 28×28 |
| Fashion-MNIST | 70 000 | 10 | 28×28 |

Hugging Face datasets (CIFAR-10, Food-101, Oxford-IIIT Pet, ADE20K, etc.) are available from the web UI with one click.

## Configuration

### Environment variables

Create a `.env` file in the project root:

```env
# Application
ENVIRONMENT=development    # development | production
DEBUG=true
SECRET_KEY=change-me-in-production

# Database
DATABASE_URL=sqlite:///./visionforge.db

# Redis / Celery
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# Auth tokens
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# CORS (comma-separated)
ALLOWED_ORIGINS=http://localhost:8501,http://localhost:3000

# GPU selection (optional)
CUDA_VISIBLE_DEVICES=0
```

### Training configuration (`experiments/config.yaml`)

```yaml
dataset:
  path: "./data/my_dataset"
  task_type: "classification"
  num_classes: 10

model:
  framework: "pytorch"
  architecture: "efficientnet_b0"
  pretrained: true

training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  optimizer: "adamw"
  scheduler: "cosine"
  mixed_precision: true

system:
  device: "cuda"
  num_workers: 4
```

## Docker Deployment

### Local (default profile — SQLite + Redis)

```bash
docker compose up
```

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| Flower (worker monitor) | http://localhost:5555 |

### Production (Nginx + PostgreSQL)

```bash
docker compose --profile production up -d
```

### GPU workers

```bash
docker compose --profile gpu up -d
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for full production configuration including SSL/TLS, secrets management, and scaling.

## Project Structure

```
visionforge-ai/
├── api/                    # FastAPI application
│   ├── main.py             # App factory, CORS, lifespan
│   ├── auth.py             # JWT helpers
│   ├── dependencies.py     # Dependency injection
│   ├── schemas.py          # Pydantic request/response models
│   └── routes/             # Routers: auth, training, models, experiments, inference, health
├── core/                   # ML pipeline
│   ├── dataset_analyzer.py # Dataset introspection
│   ├── model_selector.py   # Architecture recommendation
│   ├── trainer.py          # PyTorch training loop
│   ├── tensorflow_trainer.py
│   ├── sklearn_trainer.py
│   └── optimizer.py        # Optuna hyperparameter search
├── db/                     # Database layer
│   ├── database.py         # SQLAlchemy engine + session
│   ├── models.py           # ORM models (User, Job, Experiment)
│   ├── crud.py             # Training job / experiment CRUD
│   └── auth_crud.py        # User CRUD
├── workers/                # Celery async workers
│   ├── celery_app.py       # Celery factory
│   └── training_tasks.py   # Training job tasks
├── ui/                     # Streamlit page modules
├── frontend/               # Next.js 14 dashboard (TypeScript + Tailwind)
├── utils/                  # Shared utilities
│   ├── config.py
│   ├── model_naming.py     # Intelligent model file naming
│   ├── callbacks.py
│   ├── metrics.py
│   ├── model_factory.py
│   ├── data_factory.py
│   └── logger.py
├── alembic/                # Database migrations
│   └── versions/
├── tests/                  # Pytest test suite
├── data/                   # Built-in dataset cache
│   ├── MNIST/
│   ├── cifar-10-batches-py/
│   └── cifar-100-python/
├── deploy/                 # Cloud deployment scripts (AWS)
├── app.py                  # Streamlit entry point
├── docker-compose.yml      # Multi-service orchestration
├── Dockerfile
├── requirements.txt        # Core dependencies
├── requirements-gpu.txt    # CUDA extras
├── requirements-cpu.txt    # CPU-only extras
├── requirements-dev.txt    # Development tools
├── alembic.ini
└── pyproject.toml
```

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.

```bash
# Set up development environment
git clone https://github.com/vfvision-ai/visionforge-ai.git
cd visionforge-ai
python -m venv venv && venv\Scripts\activate   # or source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# Run tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [PyTorch](https://pytorch.org) — deep learning framework
- [TensorFlow](https://tensorflow.org) — deep learning framework
- [Streamlit](https://streamlit.io) — web app framework
- [FastAPI](https://fastapi.tiangolo.com) — REST API framework
- [Hugging Face](https://huggingface.co) — pretrained models and datasets
- [Optuna](https://optuna.org) — hyperparameter optimization
- [Celery](https://docs.celeryq.dev) — distributed task queue

---

*For deployment instructions see [DEPLOYMENT.md](DEPLOYMENT.md). For security disclosures see [SECURITY.md](SECURITY.md).*

