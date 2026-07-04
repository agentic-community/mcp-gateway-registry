# Installation Guide

Complete installation instructions for the MCP Gateway & Registry on various platforms.

## Prerequisites

- **Node.js 16+**: Required for building the React frontend (not needed with `--prebuilt` flag)
- **Container Runtime**: Choose one:
  - **Docker & Docker Compose**: Standard container runtime
  - **Podman & Podman Compose**: Rootless alternative (recommended for macOS)
- **Amazon Cognito or Keycloak**: Identity provider for authentication (see [Cognito Setup Guide](cognito.md), [Keycloak: MCP Client Guide](keycloak-mcp-clients.md), or [Keycloak: Agent M2M & Operations Guide](keycloak-agent-m2m.md))
- **SSL Certificate**: Optional for HTTPS deployment in production

## Quick Start

### Docker Installation (Default)

```bash
# 1. Clone and setup
git clone https://github.com/agentic-community/mcp-gateway-registry.git
cd mcp-gateway-registry
cp .env.example .env

# 2. Setup Python virtual environment
uv sync
source .venv/bin/activate

# 3. Download embeddings model
uv pip install -U huggingface_hub
hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir ${HOME}/mcp-gateway/models/all-MiniLM-L6-v2

# 4. Configure environment - edit .env with your passwords
nano .env
# Set: KEYCLOAK_ADMIN_PASSWORD, INITIAL_ADMIN_PASSWORD (must match), KEYCLOAK_DB_PASSWORD
# Set: SESSION_COOKIE_SECURE=false (for HTTP localhost)

# Generate SECRET_KEY
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(64))")
sed -i "s/^#*\s*SECRET_KEY=.*/SECRET_KEY=$SECRET_KEY/" .env

# 5. Deploy with pre-built images
export DOCKERHUB_ORG=mcpgateway
source .env
export KEYCLOAK_ADMIN="${KEYCLOAK_ADMIN:-admin}"
./build_and_run.sh --prebuilt
# Press Ctrl+C when logs are streaming - containers continue running

# 6. Initialize MongoDB
docker compose up mongodb-init
docker compose restart auth-server

# 7. Initialize Keycloak (wait for Keycloak to start first)
# Note: both realms ship with sslRequired=external, which requires TLS for
# external requests but allows plaintext HTTP from loopback. These commands talk
# to Keycloak over http://localhost, so no SSL change is needed. Do NOT set
# sslRequired=none (that would disable TLS enforcement for external requests too).

# Initialize realm and clients
chmod +x keycloak/setup/init-keycloak.sh
./keycloak/setup/init-keycloak.sh

# Get client credentials
chmod +x keycloak/setup/get-all-client-credentials.sh
./keycloak/setup/get-all-client-credentials.sh

# Update .env with client secrets from .oauth-tokens/keycloak-client-secrets.txt
cat .oauth-tokens/keycloak-client-secrets.txt
nano .env  # Update KEYCLOAK_CLIENT_SECRET and KEYCLOAK_M2M_CLIENT_SECRET

# Recreate containers with new credentials
./build_and_run.sh --prebuilt

# 8. Setup users and service accounts
chmod +x ./cli/bootstrap_user_and_m2m_setup.sh
./cli/bootstrap_user_and_m2m_setup.sh

# 9. Access registry
open http://localhost  # macOS
# xdg-open http://localhost  # Linux
# Login: admin / <KEYCLOAK_ADMIN_PASSWORD>
```

For the complete step-by-step guide with detailed explanations, see the [Quick Start Guide](quickstart.md).

### Podman Installation (Rootless Alternative)

**Recommended for macOS and rootless Linux environments**

```bash
# 1. Clone and setup
git clone https://github.com/agentic-community/mcp-gateway-registry.git
cd mcp-gateway-registry
cp .env.example .env

# 2. Install Podman (macOS)
brew install podman-desktop
# OR download from: https://podman-desktop.io/

# 3. Initialize Podman machine (macOS)
podman machine init --cpus 4 --memory 8192 --disk-size 50
podman machine start

# 4. Setup Python virtual environment
uv sync
source .venv/bin/activate

# 5. Download embeddings model
uv pip install -U huggingface_hub
hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir ${HOME}/mcp-gateway/models/all-MiniLM-L6-v2

# 6. Configure environment - edit .env with your passwords
nano .env
# Set: KEYCLOAK_ADMIN_PASSWORD, INITIAL_ADMIN_PASSWORD (must match), KEYCLOAK_DB_PASSWORD
# Set: SESSION_COOKIE_SECURE=false (for HTTP localhost)
# For Podman: Set KEYCLOAK_URL=http://localhost:18080

# Generate SECRET_KEY
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(64))")
sed -i "s/^#*\s*SECRET_KEY=.*/SECRET_KEY=$SECRET_KEY/" .env

# 7. Deploy with Podman
export DOCKERHUB_ORG=mcpgateway
source .env
export KEYCLOAK_ADMIN="${KEYCLOAK_ADMIN:-admin}"
./build_and_run.sh --prebuilt --podman
# Apple Silicon: Use ./build_and_run.sh --podman (without --prebuilt)
# Press Ctrl+C when logs are streaming - containers continue running

# 8. Initialize MongoDB
podman compose up mongodb-init
podman compose restart auth-server

# 9. Initialize Keycloak (wait for Keycloak to start first)
# Note: Podman uses port 18080 for Keycloak.
# Both realms ship with sslRequired=external (TLS required for external requests,
# plaintext allowed from loopback). These commands talk to Keycloak over
# http://localhost, so no SSL change is needed. Do NOT set sslRequired=none.

# Initialize realm and clients
chmod +x keycloak/setup/init-keycloak.sh
KEYCLOAK_URL=http://localhost:18080 ./keycloak/setup/init-keycloak.sh

# Get client credentials
chmod +x keycloak/setup/get-all-client-credentials.sh
KEYCLOAK_URL=http://localhost:18080 ./keycloak/setup/get-all-client-credentials.sh

# Update .env with client secrets
cat .oauth-tokens/keycloak-client-secrets.txt
nano .env  # Update KEYCLOAK_CLIENT_SECRET and KEYCLOAK_M2M_CLIENT_SECRET

# Recreate containers with new credentials
./build_and_run.sh --prebuilt --podman

# 10. Setup users and service accounts
chmod +x ./cli/bootstrap_user_and_m2m_setup.sh
KEYCLOAK_URL=http://localhost:18080 ./cli/bootstrap_user_and_m2m_setup.sh

# 11. Access registry (note the different port for Podman)
open http://localhost:8080  # macOS
# xdg-open http://localhost:8080  # Linux
# Login: admin / <KEYCLOAK_ADMIN_PASSWORD>
```

> **Note for Apple Silicon:** Don't use `--prebuilt` with Podman on ARM64. Use `./build_and_run.sh --podman` instead. See [Podman on Apple Silicon Guide](podman-apple-silicon.md).

**Podman Port Mapping:**
- Main interface: `http://localhost:8080` (HTTP) or `https://localhost:8443` (HTTPS)
- Registry API: `http://localhost:8080` (served through nginx on the main interface port)
- Keycloak: `http://localhost:18080` (instead of 8080)
- All other internal services: unchanged ports

## Installation on Amazon EC2

### System Requirements

**Minimum (Development)**:
- EC2 Instance: `t3.large` (2 vCPU, 8GB RAM)
- Storage: 20GB SSD
- Network: Ports 80, 443, 7860, 8080 accessible

**Recommended (Production)**:
- EC2 Instance: `t3.2xlarge` (8 vCPU, 32GB RAM)  
- Storage: 50GB+ SSD
- Network: Multi-AZ with load balancer

### Detailed Setup Steps

1. **Create Local Directories**
   ```bash
   mkdir -p ${HOME}/mcp-gateway/{servers,auth_server,secrets,logs}
   cp -r registry/servers ${HOME}/mcp-gateway/
   ```

   Note: authorization scopes are no longer copied as a file. They live in the `mcp_scopes` collection in DocumentDB / MongoDB and are seeded at init time from the JSON scope seed files in `scripts/` (or managed via the scope management API).

2. **Configure Environment Variables**
   ```bash
   cp .env.example .env
   nano .env  # Configure required values
   ```

   **Required Configuration:**
   - `KEYCLOAK_ADMIN_PASSWORD`: Keycloak admin password (if using Keycloak)
   - `COGNITO_USER_POOL_ID`: Amazon Cognito User Pool ID
   - `COGNITO_CLIENT_ID`: Cognito App Client ID
   - `COGNITO_CLIENT_SECRET`: Cognito App Client Secret
   - `AWS_REGION`: AWS region for Cognito

3. **Generate Authentication Credentials**
   ```bash
   # Configure OAuth credentials
   cp credentials-provider/oauth/.env.example credentials-provider/oauth/.env
   nano credentials-provider/oauth/.env
   
   # Generate tokens and client configurations
   ./credentials-provider/generate_creds.sh
   ```

4. **Install Dependencies**
   ```bash
   # Install uv (Python package manager)
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source $HOME/.local/bin/env
   uv venv --python 3.14 && source .venv/bin/activate
   
   # Install Docker
   sudo apt-get update
   sudo apt-get install --reinstall docker.io -y
   sudo apt-get install -y docker-compose
   sudo usermod -a -G docker $USER
   newgrp docker
   ```

5. **Deploy Services**
   ```bash
   ./build_and_run.sh
   ```

## Podman Installation (Rootless Containers)

Podman is a daemonless container engine that provides rootless container execution, making it ideal for macOS and environments where Docker requires privileged access.

### Benefits of Podman

- ✅ **Rootless Execution**: No sudo or privileged ports required
- ✅ **macOS Native**: Works seamlessly with Podman Desktop on macOS
- ✅ **Security**: Enhanced container isolation without root privileges
- ✅ **Compatibility**: Drop-in replacement for Docker with similar CLI commands

### Installation on macOS

**Option 1: Podman Desktop (Recommended)**

```bash
# Install via Homebrew
brew install podman-desktop

# Or download directly from:
# https://podman-desktop.io/
```

**Option 2: Podman CLI Only**

```bash
# Install Podman
brew install podman

# Initialize Podman machine
podman machine init --cpus 4 --memory 8192 --disk-size 50
podman machine start

# Verify installation
podman --version
podman compose version
```

### Installation on Linux

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y podman podman-compose

# Fedora/RHEL
sudo dnf install -y podman podman-compose

# Arch Linux
sudo pacman -S podman podman-compose

# Verify installation
podman --version
podman compose version
```

### Deploying with Podman

```bash
# Navigate to repository
cd mcp-gateway-registry

# Configure environment
cp .env.example .env
nano .env  # Configure required values
# Important: Set KEYCLOAK_URL=http://localhost:18080 for Podman

# Deploy with Podman (explicit)
./build_and_run.sh --prebuilt --podman

# Apple Silicon: Use without --prebuilt
# ./build_and_run.sh --podman

# Or let the script auto-detect (will use Podman if Docker not available)
./build_and_run.sh --prebuilt
```

After initial deployment, you must complete the MongoDB and Keycloak initialization steps. See the [Podman Installation Quick Start](#podman-installation-rootless-alternative) above for the complete sequence including:
- MongoDB initialization (`podman compose up mongodb-init`)
- Keycloak realm setup (using port 18080)
- Client credential retrieval and .env update
- Container recreation to apply credentials
- User and service account setup

> **Apple Silicon Warning:** Don't use `--prebuilt` with Podman on Apple Silicon Macs. Use `./build_and_run.sh --podman` instead. See [Podman on Apple Silicon Guide](podman-apple-silicon.md).

### Accessing Services with Podman

**Important Port Differences:**

Podman uses non-privileged host ports to avoid requiring root access:

| Service | Docker Port | Podman Port | Description |
|---------|-------------|-------------|-------------|
| Main UI (HTTP) | `http://localhost` | `http://localhost:8080` | Web interface |
| Main UI (HTTPS) | `https://localhost` | `https://localhost:8443` | Secure web interface |
| Registry API | `http://localhost` | `http://localhost:8080` | API endpoint (served through nginx on the Main UI port) |
| Auth Server | `http://localhost:8888` | `http://localhost:8888` | Auth service (unchanged) |
| Keycloak | `http://localhost:8080` | `http://localhost:18080` | IdP (Podman uses 18080 because 8080 is used by the Registry UI) |
| Prometheus | `http://localhost:9090` | `http://localhost:9090` | Metrics (unchanged) |
| Grafana | `http://localhost:3000` | `http://localhost:3000` | Dashboards (unchanged) |

**Access the registry:**
```bash
# With Podman
open http://localhost:8080

# With Docker
open http://localhost
```

### Podman-Specific Configuration

The deployment uses `docker-compose.podman.yml` when using Podman, which:

1. **Remaps privileged ports**: Maps container ports 80→8080 and 443→8443 on the host
2. **Adds SELinux labels**: Adds `:z` mount options for SELinux compatibility (Linux)
3. **Maintains compatibility**: All internal service-to-service communication unchanged

### Troubleshooting Podman

**Issue: Permission denied on volume mounts**

```bash
# Ensure directories exist with proper permissions
mkdir -p ${HOME}/mcp-gateway/{servers,agents,models,logs,security_scans,auth_server,ssl}
chmod -R 755 ${HOME}/mcp-gateway
```

**Issue: Podman machine not starting (macOS)**

```bash
# Reset Podman machine
podman machine stop
podman machine rm
podman machine init --cpus 4 --memory 8192 --disk-size 50
podman machine start
```

**Issue: Port conflicts**

```bash
# Check what's using ports 8080 or 8443
lsof -i :8080
lsof -i :8443

# Stop conflicting services or use different ports by editing docker-compose.podman.yml
```

**Issue: Podman compose command not found**

```bash
# Install podman-compose separately
pip install podman-compose

# Or use podman-compose wrapper
brew install podman-compose
```

### HTTPS Configuration

By default, MCP Gateway runs on HTTP (port 80). To enable HTTPS for production deployments:

#### 1. Obtain SSL Certificates

**Option A: Let's Encrypt (Recommended)**
```bash
# Install certbot
sudo apt-get update
sudo apt-get install -y certbot

# Get certificate (requires domain and port 80 accessible)
sudo certbot certonly --standalone -d your-domain.com
```

**Option B: Commercial CA**
Purchase SSL certificate from a trusted Certificate Authority.

#### 2. Copy Certificates to Expected Location

MCP Gateway expects SSL certificates at `${HOME}/mcp-gateway/ssl/`. The `build_and_run.sh` script will automatically set up the proper directory structure.

```bash
# Create the ssl directory structure
mkdir -p ${HOME}/mcp-gateway/ssl/certs
mkdir -p ${HOME}/mcp-gateway/ssl/private

# Copy your certificates to the expected location
# Replace paths below with your actual certificate locations
cp /etc/letsencrypt/live/your-domain/fullchain.pem ${HOME}/mcp-gateway/ssl/certs/fullchain.pem
cp /etc/letsencrypt/live/your-domain/privkey.pem ${HOME}/mcp-gateway/ssl/private/privkey.pem

# Set proper permissions
chmod 644 ${HOME}/mcp-gateway/ssl/certs/fullchain.pem
chmod 600 ${HOME}/mcp-gateway/ssl/private/privkey.pem
```

**Note**: If SSL certificates are not present at `${HOME}/mcp-gateway/ssl/certs/fullchain.pem` and `${HOME}/mcp-gateway/ssl/private/privkey.pem`, the MCP Gateway will automatically run in HTTP-only mode.

#### 3. Configure Security Group

- Enable TCP port 443 for HTTPS access
- Restrict access to authorized IP ranges
- Keep port 80 open for HTTP and Let's Encrypt renewals

#### 4. Deploy and Verify

```bash
# Start/restart the services
./build_and_run.sh

# Check logs for SSL certificate detection
docker compose logs registry | grep -i ssl

# Expected output:
# "SSL certificates found - HTTPS enabled"
# "HTTPS server will be available on port 443"

# Test HTTPS access
curl https://your-domain.com
```

#### Certificate Renewal (Let's Encrypt)

Let's Encrypt certificates expire after 90 days. Set up automatic renewal:

```bash
# Add to crontab
sudo crontab -e

# Add this line (checks twice daily, renews if needed)
0 0,12 * * * certbot renew --quiet && cp /etc/letsencrypt/live/your-domain/fullchain.pem ${HOME}/mcp-gateway/ssl/certs/fullchain.pem && cp /etc/letsencrypt/live/your-domain/privkey.pem ${HOME}/mcp-gateway/ssl/private/privkey.pem && docker compose restart registry
```

#### Troubleshooting

**HTTPS not working?**
- Check certificate files exist: `ls -la ${HOME}/mcp-gateway/ssl/certs/ ${HOME}/mcp-gateway/ssl/private/`
- Verify certificates are present: `${HOME}/mcp-gateway/ssl/certs/fullchain.pem` and `${HOME}/mcp-gateway/ssl/private/privkey.pem`
- Check container logs: `docker compose logs registry | grep -i ssl`
- Verify port 443 is accessible: `sudo netstat -tlnp | grep 443`
- Ensure certificates are from a trusted CA

## Installation on Amazon EKS

For production Kubernetes deployments, see the [EKS deployment guide](https://github.com/aws-samples/amazon-eks-machine-learning-with-terraform-and-kubeflow/tree/master/examples/agentic/mcp-gateway-microservices).

### Architecture Overview

```mermaid
graph TB
    subgraph "EKS Cluster"
        subgraph "Ingress"
            ALB[Application Load Balancer]
            IC[Ingress Controller]
        end
        
        subgraph "Application Pods"
            RP[Registry Pod]
            AS[Auth Server Pod]
            NG[Nginx Pod]
        end
        
        subgraph "MCP Servers"
            MS1[MCP Server 1]
            MS2[MCP Server 2]
            MSN[MCP Server N]
        end
    end
    
    subgraph "AWS Services"
        COG[Amazon Cognito]
        CW[CloudWatch]
        ECR[Amazon ECR]
    end
    
    ALB --> IC
    IC --> RP
    IC --> AS
    IC --> NG
    NG --> MS1
    NG --> MS2
    NG --> MSN
    AS --> COG
    RP --> CW
```

### Key Benefits of EKS Deployment

- **Multi-AZ Support**: Pod distribution across availability zones
- **Auto Scaling**: Horizontal pod autoscaling based on metrics
- **Service Mesh**: Istio integration for advanced traffic management
- **Observability**: Native integration with CloudWatch and Prometheus
- **Security**: Pod security policies and network policies

## Post-Installation

### Verify Installation

1. **Check Service Status**
   ```bash
   docker-compose ps
   docker-compose logs -f
   ```

2. **Test Web Interface**
   - Navigate to `http://localhost`
   - Login with admin credentials
   - Verify MCP server health status

3. **Test Authentication**
   ```bash
   cd tests
   ./mcp_cmds.sh ping
   ```

### Configure AI Coding Assistants

1. **Generate Client Configurations**
   ```bash
   ./credentials-provider/generate_creds.sh
   ls .oauth-tokens/  # View generated configurations
   ```

2. **Setup VS Code**
   ```bash
   cp .oauth-tokens/vscode-mcp.json ~/.vscode/settings.json
   ```

3. **Setup Roo Code**
   ```bash
   cp .oauth-tokens/mcp.json ~/.vscode/mcp-settings.json
   ```

For detailed AI assistant setup, see [AI Coding Assistants Setup Guide](ai-coding-assistants-setup.md).

## Troubleshooting

### Common Issues

**Services won't start:**
```bash
# Check Docker daemon
sudo systemctl status docker

# Check environment variables
cat .env | grep -v SECRET

# View detailed logs
docker-compose logs --tail=50
```

**Authentication failures:**
```bash
# Verify Cognito configuration
aws cognito-idp describe-user-pool --user-pool-id YOUR_POOL_ID

# Test credential generation
cd credentials-provider && ./generate_creds.sh --verbose
```

**Network connectivity issues:**
```bash
# Check port availability
sudo netstat -tlnp | grep -E ':(80|443|7860|8080)'

# Test the registry through nginx
curl -v http://localhost/health
```

For more troubleshooting help, see [Troubleshooting Guide](troubleshooting.md).

## Next Steps

- [Authentication Setup](auth.md) - Configure identity providers
- [AI Assistant Integration](ai-coding-assistants-setup.md) - Setup development tools
- [AWS ECS Deployment](../terraform/aws-ecs/README.md) - Multi-instance configuration
- [API Reference](registry_api.md) - Programmatic management