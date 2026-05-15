#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# setup-ec2.sh — Bootstrap a new AWS EC2 instance (Amazon Linux 2023 / Ubuntu)
#
# Run once after launching the EC2 instance:
#   chmod +x setup-ec2.sh
#   sudo ./setup-ec2.sh
#
# Recommended instance types:
#   CPU only : t3.xlarge (4 vCPU, 16 GB) or c6i.2xlarge
#   GPU      : g4dn.xlarge  (T4  GPU, 16 GB)   cheapest
#              g5.2xlarge   (A10G GPU, 24 GB)   faster
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
set -euo pipefail

# ── Detect OS ─────────────────────────────────────────────────────────────────
if   [ -f /etc/amazon-linux-release ]; then os=amzn
elif [ -f /etc/lsb-release ];          then os=ubuntu
else echo "Unsupported OS" && exit 1
fi
echo "🖥️  Detected OS: $os"

# ── System packages ───────────────────────────────────────────────────────────
if [ "$os" = "amzn" ]; then
    dnf update -y
    dnf install -y \
        git curl wget htop unzip tar make \
        docker amazon-cloudwatch-agent
    systemctl enable --now docker
    usermod -aG docker ec2-user
else
    apt-get update -y
    apt-get install -y \
        git curl wget htop unzip tar make \
        ca-certificates gnupg lsb-release
    # Docker via official repo
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) \
        signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list
    apt-get update -y
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    usermod -aG docker ubuntu
fi

# ── Docker Compose v2 plugin ──────────────────────────────────────────────────
# Already installed via docker-compose-plugin on Ubuntu above.
# For AL2023:
if [ "$os" = "amzn" ]; then
    COMPOSE_VERSION=2.27.0
    mkdir -p /usr/local/lib/docker/cli-plugins
    curl -SL "https://github.com/docker/compose/releases/download/v${COMPOSE_VERSION}/docker-compose-linux-$(uname -m)" \
        -o /usr/local/lib/docker/cli-plugins/docker-compose
    chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
fi

docker compose version

# ── NVIDIA driver + container toolkit (GPU instances only) ───────────────────
if lspci | grep -q NVIDIA 2>/dev/null; then
    echo "🎮 GPU detected — installing NVIDIA drivers and container toolkit..."
    if [ "$os" = "amzn" ]; then
        dnf install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r)
        dnf config-manager --add-repo \
            https://developer.download.nvidia.com/compute/cuda/repos/amzn2023/x86_64/cuda-amzn2023.repo
        dnf clean expire-cache
        dnf install -y cuda-toolkit-12-1 nvidia-container-toolkit
    else
        distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
            | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L "https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list" \
            | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
            | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        apt-get update -y
        apt-get install -y nvidia-container-toolkit
    fi
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
fi

# ── Project directory ─────────────────────────────────────────────────────────
APP_DIR=/opt/ml-platform
mkdir -p "$APP_DIR"
chown -R "${SUDO_USER:-ubuntu}:${SUDO_USER:-ubuntu}" "$APP_DIR"

# ── systemd unit for the stack ────────────────────────────────────────────────
cat > /etc/systemd/system/ml-platform.service << 'UNIT'
[Unit]
Description=CV Training Pipeline Docker Stack
Requires=docker.service
After=docker.service network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/ml-platform
ExecStart=/usr/local/lib/docker/cli-plugins/docker-compose --profile production up -d
ExecStop=/usr/local/lib/docker/cli-plugins/docker-compose down
TimeoutStartSec=300
Restart=on-failure

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable ml-platform.service

echo ""
echo "✅ EC2 bootstrap complete!"
echo ""
echo "Next steps:"
echo "  1. Clone the repo into /opt/ml-platform/"
echo "  2. Copy .env.example → .env and fill in your secrets"
echo "  3. sudo systemctl start ml-platform"
echo ""
echo "Open ports (Security Group):"
echo "  80   — HTTP (Nginx)"
echo "  443  — HTTPS (Nginx)"
echo "  8501 — Streamlit (optional, if not behind Nginx)"
echo "  8000 — FastAPI  (optional, if not behind Nginx)"
