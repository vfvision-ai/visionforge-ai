# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AWS EC2 Deployment Guide — VisionForge
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Quick Start

### 1. Launch an EC2 Instance

Recommended AMIs:
- **Amazon Linux 2023** (easiest, native AWS)
- **Ubuntu 22.04 LTS** (most compatible)

Recommended instance types:

| Use case          | Instance       | vCPU | RAM  | Cost (us-east-1) |
|-------------------|----------------|------|------|-----------------|
| Development/Demo  | t3.xlarge      | 4    | 16GB | ~$0.17/hr        |
| CPU Training      | c6i.4xlarge    | 16   | 32GB | ~$0.68/hr        |
| GPU Training      | g4dn.xlarge    | 4    | 16GB | ~$0.53/hr (T4)   |
| Heavy GPU         | g5.2xlarge     | 8    | 32GB | ~$1.21/hr (A10G) |

**Security Group settings:**

| Port | Protocol | Source      | Purpose              |
|------|----------|-------------|----------------------|
| 22   | TCP      | Your IP     | SSH admin access     |
| 80   | TCP      | 0.0.0.0/0   | HTTP (Nginx)         |
| 443  | TCP      | 0.0.0.0/0   | HTTPS (Nginx)        |

> **Do not** expose ports 8000, 8501, 5555 directly. Let Nginx proxy them.

---

### 2. Bootstrap the Instance

```bash
# SSH into your instance
ssh -i your-key.pem ec2-user@<EC2_PUBLIC_IP>

# Download and run the bootstrap script
curl -fsSL https://raw.githubusercontent.com/vfvision-ai/visionforge-ai/main/deploy/aws/setup-ec2.sh | sudo bash
```

---

### 3. Clone the Repo and Configure

```bash
cd /opt/ml-platform
sudo git clone https://github.com/vfvision-ai/visionforge-ai.git .

# Configure environment
sudo cp .env.example .env
sudo nano .env   # Fill in DATABASE_URL, API_KEY, SECRET_KEY, etc.
```

---

### 4. Start the Stack

```bash
# CPU stack (default)
sudo docker compose --profile production up -d

# GPU stack
sudo docker compose --profile production,gpu up -d
```

Verify all services are healthy:

```bash
docker compose ps
curl http://localhost:8000/health
curl http://localhost:8501/_stcore/health
```

---

### 5. Enable HTTPS (Recommended for Production)

**Option A — AWS Certificate Manager + ALB (recommended)**

1. Create an Application Load Balancer (ALB) in front of the instance
2. Request a free ACM certificate for your domain
3. Add HTTPS listener (port 443) on the ALB pointing to the instance on port 80
4. Update your DNS A record to the ALB DNS name

**Option B — Let's Encrypt (Certbot)**

```bash
# On the instance
sudo dnf install -y certbot  # AL2023
sudo certbot certonly --standalone -d your-domain.com

# Copy certs to the Nginx SSL dir
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem deploy/nginx/ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem   deploy/nginx/ssl/key.pem

# Uncomment the HTTPS server block in deploy/nginx/nginx.conf
# Then restart Nginx:
docker compose --profile production restart nginx
```

---

### 6. Updating the Deployment

```bash
cd /opt/ml-platform
sudo ./deploy/aws/deploy.sh
```

Or with a specific version tag:

```bash
sudo ./deploy/aws/deploy.sh --tag v1.2.0
```

---

### 7. ECR-Based Deployment (CI/CD)

Push your Docker image to ECR from GitHub Actions (see `.github/workflows/ci-cd.yml`),
then pull and deploy on the instance:

```bash
export ECR_REGISTRY=123456789012.dkr.ecr.us-east-1.amazonaws.com
export ECR_REPO=visionforge
export DEPLOY_TAG=v1.2.0
sudo ./deploy/aws/deploy.sh --tag $DEPLOY_TAG
```

---

### Useful Commands

```bash
# View all service logs
docker compose logs -f

# View specific service
docker compose logs -f worker

# Scale workers (CPU-only)
docker compose up -d --scale worker=3

# Stop everything
docker compose --profile production down

# Upgrade DB schema after code update
docker compose run --rm api alembic upgrade head
```

---

### Monitoring

| URL                          | Service          |
|------------------------------|------------------|
| `http://<IP>/`               | Streamlit UI     |
| `http://<IP>/docs`           | API (Swagger)    |
| `http://<IP>/flower/`        | Celery monitoring|
| `http://<IP>:9090`           | Prometheus       |

---

### Cost Optimisation Tips

- Use **Spot Instances** for non-critical training workers (up to 90% cheaper)
- Stop/start the instance when not in use — data persists on the EBS volume
- Use **EFS** or **S3** for shared dataset storage across instances
- Enable **CloudWatch billing alerts** to avoid surprises
