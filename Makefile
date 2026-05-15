# Makefile for ML Platform
# Provides convenient commands for development and deployment

.PHONY: help install test lint format clean docker-build docker-run docker-stop deploy-dev deploy-prod backup

# Default target
help:
	@echo ""
	@echo "CV Training Pipeline — Make Targets"
	@echo "=================================================="
	@echo ""
	@echo "Development:"
	@echo "  make install        Install all dependencies (CPU)"
	@echo "  make install-gpu    Install GPU (CUDA 12.1) dependencies"
	@echo "  make run            Start Streamlit UI (dev)"
	@echo "  make run-api        Start FastAPI server (dev)"
	@echo "  make run-worker     Start Celery worker (dev)"
	@echo "  make run-all        Start all services locally"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all tests"
	@echo "  make test-cov       Run tests with HTML coverage report"
	@echo "  make test-api       Run API-specific tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           Check code style (black + flake8)"
	@echo "  make format         Auto-format code with black"
	@echo "  make typecheck      Run mypy type checking"
	@echo ""
	@echo "Database:"
	@echo "  make db-init        Create/update database tables"
	@echo "  make db-migrate     Run Alembic migrations"
	@echo "  make db-reset       Drop and recreate all tables (dev only!)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   Build Docker image"
	@echo "  make up             Start all services (docker compose)"
	@echo "  make up-prod        Start full production stack (+ nginx + postgres)"
	@echo "  make up-gpu         Start stack with GPU worker"
	@echo "  make down           Stop and remove containers"
	@echo "  make logs           Tail all container logs"
	@echo "  make ps             Show container status"
	@echo ""
	@echo "Deployment:"
	@echo "  make deploy-ec2     Deploy to EC2 via SSH"
	@echo "  make backup         Backup experiments and uploads"
	@echo ""
	@echo "Utilities:"
	@echo "  make health         Check health of all services"
	@echo "  make clean          Remove cache files"
	@echo ""

# Development commands
install:
	@echo "Installing CPU dependencies..."
	python -m pip install --upgrade pip
	pip install torch==2.5.1+cpu torchvision==0.20.1+cpu \
		--extra-index-url https://download.pytorch.org/whl/cpu
	pip install tensorflow-cpu==2.15.0
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	@echo "✅ CPU install complete"

install-gpu:
	@echo "Installing GPU (CUDA 12.1) dependencies..."
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-gpu.txt
	pip install -r requirements-dev.txt
	@echo "✅ GPU install complete"

run:
	@echo "Starting Streamlit UI..."
	streamlit run app.py --server.port=8501

run-api:
	@echo "Starting FastAPI server..."
	uvicorn api.main:app --reload --port 8000

run-worker:
	@echo "Starting Celery worker..."
	celery -A workers.celery_app worker --loglevel=info --concurrency=2

run-all:
	@echo "Starting all services locally (requires Redis running)..."
	@make -j3 run run-api run-worker

test:
	@echo "Running tests..."
	pytest tests/ -v

test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term

test-api:
	@echo "Running API tests..."
	pytest tests/ -v -k "api or test_api"

lint:
	@echo "Checking code style..."
	black --check .
	flake8 . --max-line-length=120 --exclude=app_legacy.py
	@echo "✅ Lint passed"

format:
	@echo "Formatting code..."
	black .
	@echo "✅ Code formatted"

typecheck:
	@echo "Running type checks..."
	mypy api/ db/ workers/ --ignore-missing-imports || true

# Database commands
db-init:
	@echo "Initialising database..."
	python -c "from db.database import init_db; init_db(); print('✅ Database initialised')"

db-migrate:
	@echo "Running Alembic migrations..."
	alembic upgrade head

db-reset:
	@echo "⚠️  Resetting database (dev only!)..."
	@read -p "Are you sure? [y/N] " ans && [ "$$ans" = "y" ]
	python -c "from db.database import Base, engine; Base.metadata.drop_all(bind=engine); print('Tables dropped')"
	make db-init

clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache htmlcov .coverage ml_platform.db
	@echo "✅ Cleanup complete"

# Docker commands
docker-build:
	@echo "Building Docker image..."
	docker build -t cv-training-pipeline:latest .
	@echo "✅ Image built"

up:
	@echo "Starting services..."
	docker compose up -d
	@echo "✅ Services started — UI: http://localhost:8501  API: http://localhost:8000/docs"

up-prod:
	@echo "Starting production stack (nginx + postgres)..."
	docker compose --profile production up -d
	@echo "✅ Production stack started — http://localhost:80"

up-gpu:
	@echo "Starting GPU stack..."
	docker compose --profile production,gpu up -d
	@echo "✅ GPU stack started"

down:
	@echo "Stopping containers..."
	docker compose down
	@echo "✅ Containers stopped"

docker-stop: down

logs:
	docker compose logs -f

docker-logs: logs

ps:
	docker compose ps

docker-shell:
	docker exec -it cv-streamlit /bin/bash

# Deployment commands
deploy-ec2:
	@echo "Deploying to EC2..."
	@[ -n "$(EC2_HOST)" ] || (echo "❌ Set EC2_HOST=user@host.example.com" && exit 1)
	ssh $(EC2_HOST) "cd /opt/ml-platform && ./deploy/aws/deploy.sh --tag $(TAG)"

deploy-dev:
	make up

deploy-prod:
	make up-prod
	@echo "Stopping Docker containers..."
	docker-compose down
	@echo "✓ Containers stopped"

docker-logs:
	@echo "Viewing logs (Ctrl+C to exit)..."
	docker-compose logs -f

docker-shell:
	@echo "Opening shell in container..."
	docker exec -it ml-platform /bin/bash

# Deployment commands
deploy-dev:
	@echo "Deploying in development mode..."
	docker-compose up -d
	@echo "✓ Development deployment complete"

deploy-prod:
	@echo "Deploying in production mode..."
	docker-compose --profile production up -d
	@echo "✓ Production deployment complete"

# Backup and restore
backup:
	@echo "Creating backup..."
	@mkdir -p backups
	@tar -czf backups/experiments-$$(date +%Y%m%d-%H%M%S).tar.gz experiments/ 2>/dev/null || true
	@tar -czf backups/uploads-$$(date +%Y%m%d-%H%M%S).tar.gz uploads/ 2>/dev/null || true
	@[ -f .env ] && cp .env backups/.env-$$(date +%Y%m%d-%H%M%S) || true
	@echo "✅ Backup created in backups/"

restore:
	@echo "Restore functionality - please specify backup file manually"
	@ls -lt backups/

# Utilities
health:
	@echo "Checking service health..."
	@curl -sf http://localhost:8000/health   && echo "✅ API healthy"          || echo "❌ API unreachable"
	@curl -sf http://localhost:8501/_stcore/health && echo "✅ Streamlit healthy" || echo "❌ Streamlit unreachable"
	@docker exec cv-redis redis-cli ping    2>/dev/null && echo "✅ Redis healthy"    || echo "⚠️  Redis container not running"

metrics:
	@curl -s http://localhost:9090/metrics 2>/dev/null || echo "Prometheus not running"

metrics:
	@echo "Fetching metrics..."
	@curl -s http://localhost:9090/metrics 2>/dev/null || echo "Metrics endpoint not available"

# Database/Data management (if needed)
create-dirs:
	@echo "Creating required directories..."
	@mkdir -p data uploads experiments logs models
	@echo "✓ Directories created"

# Security
security-scan:
	@echo "Running security scan..."
	docker run --rm -v $$(pwd):/app aquasec/trivy fs --severity HIGH,CRITICAL /app
	@echo "✓ Security scan complete"

# Generate secret key
generate-secret:
	@echo "Generating secret key..."
	@python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))"

# Local development server
run-local:
	@echo "Starting local development server..."
	streamlit run app.py

# Install pre-commit hooks
setup-hooks:
	@echo "Setting up git hooks..."
	@echo "#!/bin/sh\nmake lint" > .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "✓ Git hooks installed"
