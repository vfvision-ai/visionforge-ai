# Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying VisionForge to production environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Security Hardening](#security-hardening)
- [Monitoring & Logging](#monitoring--logging)
- [Backup & Recovery](#backup--recovery)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended) or Windows Server
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 50GB+ available space
- **Network**: Public IP or load balancer for external access

### Software Requirements

- Docker 20.10+ and Docker Compose 2.0+
- Python 3.9+ (if running without Docker)
- NVIDIA GPU with CUDA support (optional, for GPU acceleration)
- Nginx or similar reverse proxy (for SSL/TLS termination)

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/vfvision-ai/visionforge-ai.git
cd visionforge
```

### 2. Configure Environment Variables

Copy the example environment file and customize it:

```bash
cp .env.example .env
```

**Critical Production Settings:**

```env
# Application
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=<generate-a-secure-random-key>

# Security
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
ALLOWED_ORIGINS=https://your-domain.com
MAX_FILE_SIZE_MB=500

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/app/logs/app.log

# Monitoring
METRICS_ENABLED=true
PROMETHEUS_PORT=9090
```

**Generate SECRET_KEY:**

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 3. Create Required Directories

```bash
mkdir -p data uploads experiments logs ssl
chmod 755 data uploads experiments logs
```

## Docker Deployment

### Quick Start (Development)

```bash
./deploy.sh dev
```

### Production Deployment

#### 1. Build the Docker Image

```bash
docker build -t mlplatform:latest .
```

#### 2. Run with Docker Compose

```bash
# Start the application
docker-compose up -d

# With Nginx reverse proxy
docker-compose --profile production up -d
```

#### 3. Verify Deployment

```bash
# Check container status
docker ps

# View logs
docker logs visionforge

# Check health
curl http://localhost:8501/_stcore/health
```

### Production Docker Compose Configuration

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  mlplatform:
    image: mlplatform:latest
    container_name: ml-prod
    restart: always
    env_file:
      - .env
    volumes:
      - ./data:/app/data:ro
      - ./uploads:/app/uploads
      - ./experiments:/app/experiments
      - ./logs:/app/logs
    networks:
      - ml-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G

  nginx:
    image: nginx:alpine
    container_name: ml-nginx
    restart: always
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - mlplatform
    networks:
      - ml-network

  prometheus:
    image: prom/prometheus:latest
    container_name: ml-prometheus
    restart: always
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    networks:
      - ml-network

networks:
  ml-network:
    driver: bridge

volumes:
  prometheus-data:
```

## Kubernetes Deployment

### 1. Create Namespace

```bash
kubectl create namespace mlplatform
```

### 2. Create ConfigMap

```bash
kubectl create configmap ml-config \
  --from-env-file=.env \
  -n mlplatform
```

### 3. Create Deployment

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlplatform
  namespace: mlplatform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mlplatform
  template:
    metadata:
      labels:
        app: mlplatform
    spec:
      containers:
      - name: mlplatform
        image: mlplatform:latest
        ports:
        - containerPort: 8501
        envFrom:
        - configMapRef:
            name: ml-config
        volumeMounts:
        - name: data
          mountPath: /app/data
        - name: uploads
          mountPath: /app/uploads
        - name: experiments
          mountPath: /app/experiments
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: ml-data-pvc
      - name: uploads
        persistentVolumeClaim:
          claimName: ml-uploads-pvc
      - name: experiments
        persistentVolumeClaim:
          claimName: ml-experiments-pvc
```

### 4. Create Service

Create `k8s/service.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-service
  namespace: mlplatform
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8501
    protocol: TCP
  selector:
    app: mlplatform
```

### 5. Deploy

```bash
kubectl apply -f k8s/
```

## Security Hardening

### 1. SSL/TLS Configuration

Generate SSL certificates:

```bash
# Using Let's Encrypt (recommended)
certbot certonly --standalone -d your-domain.com

# Or use self-signed for testing
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/key.pem \
  -out ssl/cert.pem
```

Update Nginx configuration with SSL settings.

### 2. Firewall Configuration

```bash
# Allow only necessary ports
ufw allow 80/tcp
ufw allow 443/tcp
ufw enable
```

### 3. Security Best Practices

- ✅ Use strong SECRET_KEY
- ✅ Enable XSRF protection
- ✅ Set appropriate ALLOWED_ORIGINS
- ✅ Run containers as non-root user
- ✅ Regularly update dependencies
- ✅ Scan images for vulnerabilities
- ✅ Implement rate limiting
- ✅ Use secure file upload validation

## Monitoring & Logging

### Prometheus Metrics

Access metrics at: `http://your-domain:9090/metrics`

Key metrics:
- `ml_requests_total` - Total requests
- `ml_training_jobs_total` - Training jobs
- `ml_errors_total` - Errors
- `ml_system_cpu_usage_percent` - CPU usage
- `ml_system_memory_usage_percent` - Memory usage

### Logging

Logs are stored in `/app/logs/app.log` in JSON format:

```bash
# View logs
docker logs visionforge

# Follow logs
docker logs -f visionforge

# Filter errors
docker logs visionforge 2>&1 | grep ERROR
```

### Health Checks

```bash
# Application health
curl http://localhost:8501/_stcore/health

# Detailed health status (if exposed)
curl http://localhost:8501/health
```

## Backup & Recovery

### Backup Strategy

**Daily backups** of:
- Trained models (`/app/experiments`)
- Uploaded data (`/app/uploads`)
- Configuration files

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup experiments
tar -czf $BACKUP_DIR/experiments.tar.gz ./experiments

# Backup uploads
tar -czf $BACKUP_DIR/uploads.tar.gz ./uploads

# Backup configuration
cp .env $BACKUP_DIR/.env
```

### Restore Procedure

```bash
# Stop services
docker-compose down

# Restore from backup
tar -xzf backup/experiments.tar.gz -C ./
tar -xzf backup/uploads.tar.gz -C ./

# Restart services
docker-compose up -d
```

## Troubleshooting

### Common Issues

#### 1. Container Won't Start

```bash
# Check logs
docker logs visionforge

# Check disk space
df -h

# Check memory
free -m
```

#### 2. Out of Memory

Increase Docker memory limits in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 16G
```

#### 3. Slow Performance

- Check CPU/memory usage
- Enable GPU if available
- Reduce batch size
- Increase worker processes

#### 4. SSL Certificate Issues

```bash
# Verify certificate
openssl x509 -in ssl/cert.pem -text -noout

# Renew Let's Encrypt certificate
certbot renew
```

### Getting Help

- Check logs: `docker logs visionforge`
- Check GitHub Issues
- Review health status
- Enable DEBUG mode temporarily for detailed logs

## Performance Optimization

### 1. Enable GPU Support

Update `docker-compose.yml`:

```yaml
services:
  mlplatform:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 2. Caching

Enable Redis for caching (optional):

```yaml
services:
  redis:
    image: redis:alpine
    restart: always
```

### 3. Load Balancing

For high availability, deploy multiple replicas behind a load balancer.

## Production Checklist

Before going live:

- [ ] Environment variables configured
- [ ] SECRET_KEY set to secure random value
- [ ] SSL/TLS certificates installed
- [ ] Firewall rules configured
- [ ] Backup strategy implemented
- [ ] Monitoring enabled
- [ ] Log rotation configured
- [ ] Health checks working
- [ ] Rate limiting enabled
- [ ] Security scan passed
- [ ] Load testing completed
- [ ] Documentation updated
- [ ] Team trained on operations

## Support & Maintenance

### Regular Maintenance Tasks

- **Daily**: Check logs for errors
- **Weekly**: Review metrics and  performance
- **Monthly**: Update dependencies, renew certificates
- **Quarterly**: Security audit, backup testing

### Version Updates

```bash
# Pull latest changes
git pull origin main

# Rebuild image
docker-compose build

# Restart with zero downtime
docker-compose up -d --no-deps --build mlplatform
```

---

For additional help, consult the [README.md](README.md) and project documentation.
