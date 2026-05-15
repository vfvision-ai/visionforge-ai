#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# deploy.sh — Deploy / update the ML Platform on an EC2 instance
#
# Usage:
#   ./deploy.sh [--tag v1.2.3] [--gpu]
#
# Environment vars (from .env or CI/CD secrets):
#   ECR_REGISTRY   — e.g. 123456789012.dkr.ecr.us-east-1.amazonaws.com
#   ECR_REPO       — e.g. ml-platform
#   AWS_REGION     — e.g. us-east-1
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
DEPLOY_TAG="${DEPLOY_TAG:-latest}"
USE_GPU=false
COMPOSE_PROFILES="production"
APP_DIR="${APP_DIR:-/opt/ml-platform}"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --tag)   DEPLOY_TAG="$2"; shift 2 ;;
        --gpu)   USE_GPU=true;    shift   ;;
        *)       echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if $USE_GPU; then
    COMPOSE_PROFILES="production,gpu"
fi

echo "🚀 Deploying CV Training Pipeline"
echo "   Tag     : $DEPLOY_TAG"
echo "   GPU     : $USE_GPU"
echo "   Profiles: $COMPOSE_PROFILES"
echo ""

# ── Pull latest code ──────────────────────────────────────────────────────────
cd "$APP_DIR"
git fetch --all
git reset --hard "origin/main"
echo "✅ Code updated ($(git rev-parse --short HEAD))"

# ── Login to ECR (if ECR_REGISTRY is set) ─────────────────────────────────────
if [[ -n "${ECR_REGISTRY:-}" ]]; then
    aws ecr get-login-password --region "${AWS_REGION:-us-east-1}" \
        | docker login --username AWS --password-stdin "$ECR_REGISTRY"
    echo "✅ ECR login successful"

    # Pull pre-built images instead of local build
    export IMAGE_TAG="$ECR_REGISTRY/${ECR_REPO:-ml-platform}:$DEPLOY_TAG"
    docker pull "$IMAGE_TAG"
    echo "✅ Pulled image: $IMAGE_TAG"
fi

# ── Backup .env if it exists ──────────────────────────────────────────────────
if [ -f .env ]; then
    cp .env ".env.backup.$(date +%Y%m%d_%H%M%S)"
fi

# ── Run DB migrations (Alembic) ───────────────────────────────────────────────
echo "🔄 Running database migrations..."
docker compose --profile "$COMPOSE_PROFILES" run --rm api alembic upgrade head || \
    echo "⚠️  Migration failed or Alembic not configured — skipping"

# ── Rolling restart ───────────────────────────────────────────────────────────
echo "🔄 Restarting services..."
docker compose --profile "$COMPOSE_PROFILES" up -d --build --remove-orphans

# Remove dangling images to free disk space
docker image prune -f

# ── Health check ─────────────────────────────────────────────────────────────
echo "⏳ Waiting for services to be healthy..."
sleep 15

API_URL="http://localhost:8000/health"
for i in {1..10}; do
    if curl -sf "$API_URL" > /dev/null; then
        echo "✅ API health check passed (attempt $i)"
        break
    fi
    echo "   Attempt $i/10 — retrying in 10s..."
    sleep 10
done

UI_URL="http://localhost:8501/_stcore/health"
if curl -sf "$UI_URL" > /dev/null; then
    echo "✅ Streamlit UI health check passed"
else
    echo "⚠️  Streamlit UI not yet healthy — check logs: docker compose logs streamlit"
fi

echo ""
echo "✅ Deployment complete!"
echo "   Streamlit : http://$(curl -s ifconfig.me):80"
echo "   API docs  : http://$(curl -s ifconfig.me):80/docs"
echo "   Flower    : http://$(curl -s ifconfig.me):80/flower"
