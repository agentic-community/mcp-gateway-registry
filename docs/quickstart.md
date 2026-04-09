# Quick Start Guide

This guide walks you through setting up the MCP Gateway & Registry using pre-built Docker images. For other deployment options, see the [Installation Guide](installation.md).

## Prerequisites

<details>
<summary><strong>Click to expand: Install Docker, Node.js, Python, and UV</strong></summary>

**Install Docker and Docker Compose:**
```bash
# Install Docker
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add Docker repository
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add user to docker group (logout/login required, or use newgrp)
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker compose version
```

**Install Node.js 20.x:**
```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
node --version  # Should show v20.x.x
```

**Install Python and UV:**
```bash
sudo apt-get install -y python3.12 python3.12-venv python3-pip

# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
uv --version
```

**Install additional tools:**
```bash
sudo apt-get install -y git jq curl wget
```

</details>

---

## Step 1: Clone and Setup

```bash
git clone https://github.com/agentic-community/mcp-gateway-registry.git
cd mcp-gateway-registry
cp .env.example .env

# Setup Python virtual environment
# (Enterprise Macs: export UV_NATIVE_TLS=true if you hit TLS certificate errors)
uv sync
source .venv/bin/activate
```

---

## Step 2: Download Embeddings Model

Download the required sentence-transformers model using the [HuggingFace CLI](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli):
```bash
# Install huggingface_hub if not already installed
uv pip install -U huggingface_hub

# Download the model
hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir ${HOME}/mcp-gateway/models/all-MiniLM-L6-v2
```

---

## Step 3: Configure Environment

<details>
<summary><strong>Click to expand: Edit .env file with your settings</strong></summary>

Edit the `.env` file with your preferred editor:
```bash
nano .env
```

**Required changes:**
```bash
# Authentication provider (do not change)
AUTH_PROVIDER=keycloak

# Set secure passwords (CHANGE THESE!)
KEYCLOAK_ADMIN_PASSWORD=YourSecureAdminPassword123!
INITIAL_ADMIN_PASSWORD=YourSecureAdminPassword123!  # MUST match KEYCLOAK_ADMIN_PASSWORD
KEYCLOAK_DB_PASSWORD=SecureKeycloakDB123!

# Session cookie security (CRITICAL for local development)
# For HTTP access (localhost): MUST be false
SESSION_COOKIE_SECURE=false

# For HTTPS access (production): set to true
# SESSION_COOKIE_SECURE=true

# Leave these as defaults
KEYCLOAK_URL=http://localhost:8080
KEYCLOAK_REALM=mcp-gateway
KEYCLOAK_CLIENT_ID=mcp-gateway-client
```

**Generate and set SECRET_KEY:**
```bash
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(64))")
sed -i "s/^#*\s*SECRET_KEY=.*/SECRET_KEY=$SECRET_KEY/" .env
echo "Generated SECRET_KEY: $SECRET_KEY"
```

Save and exit (Ctrl+X, then Y, then Enter if using nano).

</details>

**Set environment variables for deployment:**
```bash
export DOCKERHUB_ORG=mcpgateway
source .env
export KEYCLOAK_ADMIN="${KEYCLOAK_ADMIN:-admin}"
```

---

## Step 4: Deploy with Pre-built Images

```bash
./build_and_run.sh --prebuilt
```

> **Port Differences:**
> - **Docker**: Services run on privileged ports (`http://localhost`, `https://localhost`)
> - **Podman**: Services run on non-privileged ports (`http://localhost:8080`, `https://localhost:8443`)

Once the build completes and you see the container logs streaming, you can press **Ctrl+C** to exit the log view and continue with the next steps. The containers will continue running in the background.

Wait for all services to start (2-3 minutes), then verify:
```bash
docker compose ps
# All services should show as "Up"
```

---

## Step 5: Initialize MongoDB

Initialize the MongoDB database with required collections, indexes, and default scopes:

```bash
# Run the MongoDB initialization container
docker compose up mongodb-init

# Verify collections were created
docker exec mcp-mongodb mongosh --eval "use mcp_registry; show collections"
# Should show: mcp_servers_default, mcp_agents_default, mcp_scopes_default, etc.

# Restart auth-server to load the new scopes
docker compose restart auth-server
```

---

## Step 6: Initialize Keycloak

<details>
<summary><strong>Click to expand: Complete Keycloak setup instructions</strong></summary>

**6a. Wait for Keycloak to be ready:**
```bash
# Monitor logs until you see "Keycloak started"
docker compose logs -f keycloak
# Press Ctrl+C when you see "Keycloak 25.x.x started"

# Or check health endpoint
curl http://localhost:8080/realms/master
# Should return JSON with realm information
```

**6b. Disable SSL for master realm (required for HTTP access):**
```bash
ADMIN_TOKEN=$(curl -s -X POST "http://localhost:8080/realms/master/protocol/openid-connect/token" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=${KEYCLOAK_ADMIN}" \
    -d "password=${KEYCLOAK_ADMIN_PASSWORD}" \
    -d "grant_type=password" \
    -d "client_id=admin-cli" | \
    jq -r '.access_token') && \
curl -X PUT "http://localhost:8080/admin/realms/master" \
    -H "Authorization: Bearer $ADMIN_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"sslRequired": "none"}'
```

**6c. Initialize Keycloak realm and clients:**
```bash
chmod +x keycloak/setup/init-keycloak.sh
./keycloak/setup/init-keycloak.sh
```

**6d. Disable SSL for application realm:**
```bash
ADMIN_TOKEN=$(curl -s -X POST "http://localhost:8080/realms/master/protocol/openid-connect/token" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=${KEYCLOAK_ADMIN}" \
    -d "password=${KEYCLOAK_ADMIN_PASSWORD}" \
    -d "grant_type=password" \
    -d "client_id=admin-cli" | \
    jq -r '.access_token') && \
curl -X PUT "http://localhost:8080/admin/realms/mcp-gateway" \
    -H "Authorization: Bearer $ADMIN_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"sslRequired": "none"}'
```

**6e. Retrieve and save client credentials:**
```bash
chmod +x keycloak/setup/get-all-client-credentials.sh
./keycloak/setup/get-all-client-credentials.sh
```

**6f. Update .env with client secrets:**
```bash
# View the retrieved secrets
cat .oauth-tokens/keycloak-client-secrets.txt

# Update .env with the actual secret values shown above
nano .env
# Find and update: KEYCLOAK_CLIENT_SECRET and KEYCLOAK_M2M_CLIENT_SECRET
```

**6g. Recreate containers to apply new credentials:**
```bash
# Recreate containers to pick up the updated .env values
./build_and_run.sh --prebuilt
```
Once logs are streaming, press **Ctrl+C** to exit - containers will continue running.

</details>

---

## Step 7: Set Up Users and Service Accounts

```bash
chmod +x ./cli/bootstrap_user_and_m2m_setup.sh
./cli/bootstrap_user_and_m2m_setup.sh
```

This creates:
- **3 groups**: `registry-users-lob1`, `registry-users-lob2`, `registry-admins`
- **6 users**:
  - **LOB1**: `lob1-bot` (M2M) and `lob1-user` (human)
  - **LOB2**: `lob2-bot` (M2M) and `lob2-user` (human)
  - **Admin**: `admin-bot` (M2M) and `admin-user` (human)

All credentials are saved to `.oauth-tokens/` directory.

---

## Step 8: Create AI Agent Account (Optional)

<details>
<summary><strong>Click to expand: Create additional agent accounts</strong></summary>

```bash
chmod +x keycloak/setup/setup-agent-service-account.sh

# Create a test agent with full access
./keycloak/setup/setup-agent-service-account.sh \
  --agent-id test-agent \
  --group mcp-servers-unrestricted

# Create an agent for AI coding assistants
./keycloak/setup/setup-agent-service-account.sh \
  --agent-id ai-coding-assistant \
  --group mcp-servers-unrestricted

# Retrieve credentials for the new agents
./keycloak/setup/get-all-client-credentials.sh
```

</details>

---

## Step 9: Access the Registry

<details>
<summary><strong>Click to expand: Remote Access Options (EC2, Port Forwarding, etc.)</strong></summary>

The method to access the web UI depends on where you're running the MCP Gateway:

**Option A: Local Machine (Linux/macOS)**

If you're running on your local machine, simply open a browser - you're already on localhost.

**Option B: AWS EC2 with Port Forwarding**

If you're running on EC2 and want to access from your local machine via SSH tunnels:

```bash
# From your local machine, create SSH tunnels
ssh -i your-key.pem -L 7860:localhost:7860 -L 8080:localhost:8080 -L 8888:localhost:8888 -L 80:localhost:80 ubuntu@your-ec2-ip

# Then access in your local browser: http://localhost:7860
```

**Option C: AWS EC2 with Remote Desktop (GUI Access)**

If you prefer a full desktop environment on your EC2 instance:

```bash
# Install XFCE desktop and XRDP
sudo apt update && sudo apt install -y xfce4 xfce4-goodies xrdp firefox
echo "xfce4-session" > ~/.xsession
sudo systemctl enable xrdp && sudo systemctl start xrdp
sudo passwd ubuntu  # Set password for RDP login
```

**AWS Security Group**: Add inbound rule for port 3389 (RDP) from your IP.

**Connect**: Use Remote Desktop Connection (Windows) or Microsoft Remote Desktop (macOS) with `your-ec2-ip:3389`, username `ubuntu`.

See [Remote Desktop Setup Guide](remote-desktop-setup.md) for detailed instructions.

</details>

```bash
# On macOS:
open http://localhost:7860

# On Linux (install xdg-utils if xdg-open is not available):
# sudo apt install xdg-utils
xdg-open http://localhost:7860

# Or open http://localhost:7860 in your browser
```

Login with:
- **Username**: `admin` (or any user created in Step 6)
- **Password**: The `KEYCLOAK_ADMIN_PASSWORD` you set in Step 3

---

## Step 10: Register Example Servers and Agents (Optional)

To register example MCP servers and A2A agents, first get a JWT token from the Registry UI:

1. In the Registry UI, click the **"Get JWT Token"** button (top-left corner)
2. In the popup, click **"Copy JSON"** to copy the full token JSON
3. Save it to a `.token` file:

```bash
# Create .token file with the copied JSON
# Note: .token is already in .gitignore so it won't be committed to the repo
cat > .token << 'EOF'
<paste the copied JSON here>
EOF
```

Then register servers and agents using the Registry Management CLI:

> **Note:** Registration includes automatic security scanning using [Cisco AI Defense MCP Scanner](https://github.com/cisco-ai-defense/mcp-scanner) for servers and [Cisco AI Defense A2A Scanner](https://github.com/cisco-ai-defense/a2a-scanner) for agents. Each registration may take a few seconds while the security scan completes.

```bash
# Register MCP servers
uv run python api/registry_management.py --registry-url http://localhost --token-file .token \
    register --config cli/examples/mcpgw.json

uv run python api/registry_management.py --registry-url http://localhost --token-file .token \
    register --config cli/examples/cloudflare-docs-server-config.json

uv run python api/registry_management.py --registry-url http://localhost --token-file .token \
    register --config cli/examples/context7-server-config.json

uv run python api/registry_management.py --registry-url http://localhost --token-file .token \
    register --config cli/examples/currenttime.json

# Register A2A agents
uv run python api/registry_management.py --registry-url http://localhost --token-file .token \
    agent-register --config cli/examples/travel_assistant_agent_card.json

uv run python api/registry_management.py --registry-url http://localhost --token-file .token \
    agent-register --config cli/examples/flight_booking_agent_card.json

# Verify registrations
uv run python api/registry_management.py --registry-url http://localhost --token-file .token list
uv run python api/registry_management.py --registry-url http://localhost --token-file .token agent-list
```

Servers and agents are registered as **disabled** by default. Refresh the Registry UI to see them, then enable them using the toggle controls on each server/agent card.

---

## Step 11: Test the Setup

Test the registry using the Registry Management CLI:

```bash
# List registered servers
uv run python api/registry_management.py --registry-url http://localhost --token-file .token list

# List registered agents
uv run python api/registry_management.py --registry-url http://localhost --token-file .token agent-list

# Search for servers by natural language
uv run python api/registry_management.py --registry-url http://localhost --token-file .token \
    server-search --query "documentation tools"

# Search for agents by natural language
uv run python api/registry_management.py --registry-url http://localhost --token-file .token \
    agent-search --query "travel booking"

# Invoke a tool on an MCP server (e.g., get current time)
# This exercises the "Gateway" functionality - the request is routed through the
# MCP Gateway to the backend currenttime server, demonstrating centralized access
uv run python cli/mcp_client.py --url http://localhost/currenttime/mcp --token-file .token \
    call --tool current_time_by_timezone --args '{"tz_name": "America/New_York"}'
```

---

## Next Steps

- [Authentication Setup](auth.md) - Configure OAuth and identity providers
- [AI Coding Assistants Setup](ai-coding-assistants-setup.md) - Integrate with VS Code, Cursor, Claude Code
- [Complete Installation Guide](installation.md) - Additional deployment options
- [Configuration Reference](configuration.md) - Environment variables and settings

## Alternative Deployment Options

### Podman (Rootless)

For macOS and rootless Linux environments, see the [Installation Guide](installation.md#podman-installation) and [macOS Setup Guide](macos-setup-guide.md#podman-deployment).

### Build from Source

For customization or development, see the [Complete Setup Guide](complete-setup-guide.md).
