---
name: macos-setup
description: "Complete macOS setup and teardown for MCP Gateway & Registry. Guides users through the full installation (Docker containers, Keycloak auth, Python environment, embeddings model, all services) or complete removal of everything. Uses progressive disclosure - presents each phase, asks for confirmation, and executes only after approval. Handles both /macos-setup setup and /macos-setup teardown modes."
license: Apache-2.0
metadata:
  author: mcp-gateway-registry
  version: "1.0"
---

# MCP Gateway & Registry - macOS Setup & Teardown Skill

Use this skill when the user wants to:
- **Set up** the MCP Gateway & Registry from scratch on macOS so they can access `http://localhost` and see the registry fully running
- **Tear down** and completely remove all MCP Gateway components from their macOS system

## Overview

This skill follows `docs/macos-setup-guide.md` with **progressive disclosure**: each phase is announced to the user with a description of what will happen, confirmation is requested using `AskUserQuestion`, and only then are commands executed. No system changes happen without explicit user approval.

## Step 0: Determine Mode

**Before doing anything else**, use `AskUserQuestion` to ask the user which operation they want:

```
Which operation would you like to perform?

  Setup   - Install and configure MCP Gateway & Registry from scratch.
            Requires: Docker Desktop running, ~10GB free disk, ~15-20 min.

  Teardown - Stop all services, remove all containers, volumes, credentials,
             and generated files. This is irreversible.
```

If the user invoked the skill with an argument (e.g., `/macos-setup setup` or `/macos-setup teardown`), skip this question and go directly to the appropriate workflow.

---

## SETUP WORKFLOW

Work through each phase in order. **Never skip ahead.** After each phase succeeds, announce the next phase and ask for confirmation before executing.

---

### Phase 1: Prerequisites Check

**Announce to the user:**
> "Phase 1 of 13: Prerequisites Check. I will verify that all required software is installed and Docker Desktop is running. No files will be modified."

Use `AskUserQuestion` with options: "Check prerequisites" / "Skip (I know everything is installed)".

If user confirms, run the following checks:

```bash
echo "=== Checking Docker ==="
docker --version 2>/dev/null && docker ps >/dev/null 2>&1 && echo "DOCKER_OK" || echo "DOCKER_FAIL"

echo "=== Checking Python 3.12+ ==="
python3 --version 2>/dev/null && echo "PYTHON_OK" || echo "PYTHON_FAIL"

echo "=== Checking uv ==="
uv --version 2>/dev/null && echo "UV_OK" || echo "UV_FAIL"

echo "=== Checking jq ==="
jq --version 2>/dev/null && echo "JQ_OK" || echo "JQ_FAIL"

echo "=== Checking git ==="
git --version 2>/dev/null && echo "GIT_OK" || echo "GIT_FAIL"

echo "=== Checking we are in the repo root ==="
ls docker-compose.yml pyproject.toml build_and_run.sh 2>/dev/null && echo "REPO_OK" || echo "REPO_FAIL"
```

**Evaluate results and report clearly:**

| Check | Fail Action |
|-------|-------------|
| DOCKER_FAIL | "Docker Desktop is not running. Open Docker Desktop from Applications and wait for the whale icon in the menu bar, then retry." |
| PYTHON_FAIL | "Install Python 3.12+: `brew install python@3.12`" |
| UV_FAIL | "Install uv: `curl -LsSf https://astral.sh/uv/install.sh \| sh` then restart your terminal" |
| JQ_FAIL | "Install jq: `brew install jq`" |
| GIT_FAIL | "Install Git: `xcode-select --install`" |
| REPO_FAIL | "You are not in the repository root. Navigate to the mcp-gateway-registry directory with `cd ~/workspace/mcp-gateway-registry` (or wherever you cloned it) and run this skill again." |

**Do not proceed to Phase 2 if any check fails.** Ask the user to fix the issues and confirm before retrying.

---

### Phase 2: Collect Required Configuration

**Announce to the user:**
> "Phase 2 of 13: Configuration Collection. I need two passwords before we can proceed. These will be set for Keycloak (the authentication server). No files are modified in this phase."

Use `AskUserQuestion` to collect:

**Question 1 - Keycloak Admin Password:**
```
Enter a password for the Keycloak admin account (minimum 8 characters).

This is REQUIRED. There is no default - you must set a strong, memorable password.
You will use this to log in to the Keycloak admin console at http://localhost:8080/admin.
```

**Question 2 - Keycloak Database Password:**
```
Enter a password for the internal Keycloak PostgreSQL database (minimum 8 characters).

This is REQUIRED. There is no default. This password is used internally by Keycloak to connect to its database. You won't need to type this again after setup.
```

Store both responses as variables named `KEYCLOAK_ADMIN_PASSWORD` and `KEYCLOAK_DB_PASSWORD`.

**Validate passwords**: If either password is fewer than 8 characters or empty, inform the user and ask again. Do NOT proceed with empty or short passwords.

**Confirm with the user** by presenting a summary (never show actual password values):
```
Configuration Summary:
  - Keycloak Admin Password: [set - N characters]
  - Keycloak Database Password: [set - N characters]
  - Auth Provider: keycloak (default)
  - Auth Server URL: http://localhost (default)
  - Secret Key: will be auto-generated

Shall I proceed to create the environment file?
```

---

### Phase 3: Create Environment Configuration

**Announce to the user:**
> "Phase 3 of 13: Environment Configuration. I will copy `.env.example` to `.env` and configure it with your settings. If `.env` already exists, I will ask before overwriting."

Use `AskUserQuestion` to confirm: "Create/update .env file" / "Cancel".

Check for existing `.env`:
```bash
ls -la .env 2>/dev/null && echo "ENV_EXISTS" || echo "ENV_MISSING"
```

If `.env` exists, warn the user:
> "A `.env` file already exists. This may be from a previous setup. Overwriting it will reset all configuration. Do you want to overwrite it?"

Use `AskUserQuestion`: "Overwrite existing .env" / "Keep existing .env and skip this phase".

If proceeding:

```bash
# Copy template
cp .env.example .env

# Generate a secure SECRET_KEY
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(64))")
echo "SECRET_KEY generated (${#SECRET_KEY} characters)"
```

Now update the `.env` file. Use `sed -i ''` (macOS syntax). Run each line separately:

```bash
# Set auth provider
sed -i '' "s|^AUTH_PROVIDER=.*|AUTH_PROVIDER=keycloak|" .env
grep -q "^AUTH_PROVIDER=" .env || echo "AUTH_PROVIDER=keycloak" >> .env

# Set auth server URL
sed -i '' "s|^AUTH_SERVER_EXTERNAL_URL=.*|AUTH_SERVER_EXTERNAL_URL=http://localhost|" .env
grep -q "^AUTH_SERVER_EXTERNAL_URL=" .env || echo "AUTH_SERVER_EXTERNAL_URL=http://localhost" >> .env

# Set admin password (use printf to handle special characters safely)
python3 -c "
import re, sys
content = open('.env').read()
pwd = '''${KEYCLOAK_ADMIN_PASSWORD}'''
content = re.sub(r'^KEYCLOAK_ADMIN_PASSWORD=.*', f'KEYCLOAK_ADMIN_PASSWORD={pwd}', content, flags=re.MULTILINE)
if 'KEYCLOAK_ADMIN_PASSWORD=' not in content:
    content += f'\nKEYCLOAK_ADMIN_PASSWORD={pwd}'
open('.env', 'w').write(content)
print('Admin password set in .env')
"

# Set DB password
python3 -c "
import re
content = open('.env').read()
pwd = '''${KEYCLOAK_DB_PASSWORD}'''
content = re.sub(r'^KEYCLOAK_DB_PASSWORD=.*', f'KEYCLOAK_DB_PASSWORD={pwd}', content, flags=re.MULTILINE)
if 'KEYCLOAK_DB_PASSWORD=' not in content:
    content += f'\nKEYCLOAK_DB_PASSWORD={pwd}'
open('.env', 'w').write(content)
print('DB password set in .env')
"

# Set SECRET_KEY
python3 -c "
import re
content = open('.env').read()
key = '${SECRET_KEY}'
content = re.sub(r'^SECRET_KEY=.*', f'SECRET_KEY={key}', content, flags=re.MULTILINE)
if 'SECRET_KEY=' not in content:
    content += f'\nSECRET_KEY={key}'
open('.env', 'w').write(content)
print('SECRET_KEY set in .env')
"
```

Verify the key settings (without showing password values):
```bash
echo "=== Verifying .env settings ==="
grep "^AUTH_PROVIDER=" .env
grep "^AUTH_SERVER_EXTERNAL_URL=" .env
grep "^KEYCLOAK_ADMIN_PASSWORD=" .env | sed 's/=.*/=[set]/'
grep "^KEYCLOAK_DB_PASSWORD=" .env | sed 's/=.*/=[set]/'
grep "^SECRET_KEY=" .env | sed 's/=.*/=[set - 88 chars]/'
```

Report: "Environment file configured at `.env`."

---

### Phase 4: Python Virtual Environment

**Announce to the user:**
> "Phase 4 of 13: Python Virtual Environment. I will run `uv sync` to install all Python dependencies into a `.venv` directory. This may take 1-2 minutes on first run."

Use `AskUserQuestion` to confirm: "Install Python dependencies" / "Skip (already set up)".

```bash
uv sync
echo "Python venv exit code: $?"
```

Verify:
```bash
ls -la .venv/bin/python 2>/dev/null && echo "VENV_OK" || echo "VENV_FAIL"
```

Report: "Python virtual environment ready at `.venv/`."

---

### Phase 5: Download Embeddings Model

**Announce to the user:**
> "Phase 5 of 13: Embeddings Model Download. The MCP Gateway requires a sentence-transformers model (~90MB) for intelligent tool discovery. It will be saved to `~/mcp-gateway/models/all-MiniLM-L6-v2/`."

Use `AskUserQuestion` to confirm: "Download embeddings model (~90MB)" / "Skip (already downloaded)".

If skipping, verify the model directory exists:
```bash
ls ${HOME}/mcp-gateway/models/all-MiniLM-L6-v2/ 2>/dev/null && echo "MODEL_EXISTS" || echo "MODEL_MISSING"
```

If MODEL_MISSING and user skipped, warn: "The embeddings model is required. Services may fail to start without it. Consider running this phase."

If downloading:
```bash
# Create directory
mkdir -p ${HOME}/mcp-gateway/models/all-MiniLM-L6-v2

# Try huggingface-cli first
if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
        --local-dir ${HOME}/mcp-gateway/models/all-MiniLM-L6-v2
else
    # Fall back to Python API
    uv run python -c "
from huggingface_hub import snapshot_download
import os
path = snapshot_download(
    'sentence-transformers/all-MiniLM-L6-v2',
    local_dir=os.path.expanduser('~/mcp-gateway/models/all-MiniLM-L6-v2')
)
print(f'Downloaded to: {path}')
"
fi
```

Verify:
```bash
ls -la ${HOME}/mcp-gateway/models/all-MiniLM-L6-v2/ | head -10
echo "Total files: $(ls ${HOME}/mcp-gateway/models/all-MiniLM-L6-v2/ | wc -l)"
```

Report: "Embeddings model downloaded to `~/mcp-gateway/models/all-MiniLM-L6-v2/`."

---

### Phase 6: Create Required Directories

**Announce to the user:**
> "Phase 6 of 13: Directory Setup. I will create the required Docker volume mount directories under `~/mcp-gateway/`."

Use `AskUserQuestion` to confirm: "Create directories" / "Skip".

```bash
mkdir -p ${HOME}/mcp-gateway/{servers,models,auth_server,secrets/fininfo,logs,ssl}
ls -la ${HOME}/mcp-gateway/
```

Report: "Required directories created."

---

### Phase 7: Start Keycloak Services

**Announce to the user:**
> "Phase 7 of 13: Starting Keycloak. I will start the Keycloak database and Keycloak server containers. Allow 2-3 minutes for full initialization. The terminal will show polling output while waiting."

Use `AskUserQuestion` to confirm: "Start Keycloak services" / "Cancel".

```bash
# Export passwords - these must be set before docker compose reads them
export KEYCLOAK_ADMIN_PASSWORD="${KEYCLOAK_ADMIN_PASSWORD}"
export KEYCLOAK_DB_PASSWORD="${KEYCLOAK_DB_PASSWORD}"

# Start only Keycloak and its database
docker compose up -d keycloak-db keycloak
echo "Docker compose exit code: $?"
```

Wait for Keycloak to be ready by polling (max 180 seconds):

```bash
echo "Waiting for Keycloak to become ready (this takes 1-3 minutes)..."
TIMEOUT=180
ELAPSED=0
READY=false

while [ $ELAPSED -lt $TIMEOUT ]; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/realms/master 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        echo "Keycloak is ready! (HTTP 200 from /realms/master)"
        READY=true
        break
    fi
    echo "  Waiting... ($ELAPSED/${TIMEOUT}s) - HTTP status: $HTTP_CODE"
    sleep 10
    ELAPSED=$((ELAPSED + 10))
done

if [ "$READY" = "false" ]; then
    echo "ERROR: Keycloak did not respond within ${TIMEOUT} seconds"
    docker compose logs keycloak --tail 20
fi
```

Verify the master realm is accessible:
```bash
curl -s http://localhost:8080/realms/master | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print('Keycloak master realm:', d.get('realm', 'unknown'))
except:
    print('Could not parse Keycloak response')
"
```

**If Keycloak fails to start**, show the logs and stop:
```bash
docker compose logs keycloak --tail 30
docker compose ps
```
Inform the user of the error and ask them to investigate before retrying.

---

### Phase 8: Fix macOS SSL Requirement (Master Realm)

**Announce to the user:**
> "Phase 8 of 13: Keycloak SSL Configuration. On macOS, Docker runs in a VM which causes Keycloak to require HTTPS even for localhost. I will disable this requirement for local development. This is a standard macOS-specific configuration step."

Use `AskUserQuestion` to confirm: "Disable SSL requirement for Keycloak" / "Cancel".

First, detect the Keycloak container name dynamically:
```bash
KEYCLOAK_CONTAINER=$(docker ps --format "{{.Names}}" | grep keycloak | grep -v db | head -1)
echo "Detected Keycloak container: ${KEYCLOAK_CONTAINER}"

if [ -z "$KEYCLOAK_CONTAINER" ]; then
    echo "ERROR: No running Keycloak container found"
    docker ps
    exit 1
fi
```

Configure Keycloak admin CLI:
```bash
docker exec ${KEYCLOAK_CONTAINER} /opt/keycloak/bin/kcadm.sh config credentials \
    --server http://localhost:8080 \
    --realm master \
    --user admin \
    --password "${KEYCLOAK_ADMIN_PASSWORD}"
echo "Admin CLI config exit code: $?"
```

Disable SSL for master realm:
```bash
docker exec ${KEYCLOAK_CONTAINER} /opt/keycloak/bin/kcadm.sh update realms/master -s sslRequired=NONE
echo "SSL disable for master realm exit code: $?"
```

Verify:
```bash
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:8080/admin/")
echo "Admin endpoint HTTP status: ${HTTP_CODE} (302 = success/redirect to login)"
```

A `302` response means the admin endpoint is accessible (redirects to login page). Any `403` or `400` indicates SSL is still required.

Report: "SSL requirement disabled for Keycloak master realm."

---

### Phase 9: Initialize Keycloak Realm and Clients

**Announce to the user:**
> "Phase 9 of 13: Keycloak Initialization. I will run the initialization script to create the 'mcp-gateway' realm, the web OAuth client (mcp-gateway-web), and the machine-to-machine client (mcp-gateway-m2m). This takes about 30-60 seconds."

Use `AskUserQuestion` to confirm: "Initialize Keycloak realm and clients" / "Cancel".

```bash
chmod +x keycloak/setup/init-keycloak.sh

# The init script needs the admin password available
export KEYCLOAK_ADMIN_PASSWORD="${KEYCLOAK_ADMIN_PASSWORD}"

./keycloak/setup/init-keycloak.sh
echo "Init script exit code: $?"
```

**If the init script fails**, common causes are:
- SSL still required (re-run Phase 8)
- Keycloak not fully ready (wait 30 more seconds and retry)
- Admin password incorrect (check the `.env` file)

After initialization, disable SSL for the newly created `mcp-gateway` realm:

```bash
KEYCLOAK_CONTAINER=$(docker ps --format "{{.Names}}" | grep keycloak | grep -v db | head -1)

# Re-authenticate (session may have expired)
docker exec ${KEYCLOAK_CONTAINER} /opt/keycloak/bin/kcadm.sh config credentials \
    --server http://localhost:8080 \
    --realm master \
    --user admin \
    --password "${KEYCLOAK_ADMIN_PASSWORD}"

# Disable SSL for the mcp-gateway realm
docker exec ${KEYCLOAK_CONTAINER} /opt/keycloak/bin/kcadm.sh update realms/mcp-gateway -s sslRequired=NONE
echo "SSL disable for mcp-gateway realm exit code: $?"
```

Verify both realms are accessible:
```bash
echo "=== Verifying realms ==="
curl -s http://localhost:8080/realms/master | python3 -c "import sys,json; d=json.load(sys.stdin); print('master realm:', d.get('realm'))"
curl -s http://localhost:8080/realms/mcp-gateway | python3 -c "import sys,json; d=json.load(sys.stdin); print('mcp-gateway realm:', d.get('realm'))"
```

Report: "Keycloak realm 'mcp-gateway' created with web and M2M clients."

---

### Phase 10: Retrieve and Save Client Credentials

**Announce to the user:**
> "Phase 10 of 13: Client Credentials. I will retrieve the auto-generated OAuth client secrets from Keycloak and save them to `.oauth-tokens/`. Then I will update `.env` with these secrets so the services can authenticate."

Use `AskUserQuestion` to confirm: "Retrieve and save client credentials" / "Cancel".

```bash
chmod +x keycloak/setup/get-all-client-credentials.sh
./keycloak/setup/get-all-client-credentials.sh
echo "Credentials retrieval exit code: $?"
```

Display the saved credential summary (these are secrets - inform the user to keep them safe):
```bash
echo "=== Saved credentials ==="
cat .oauth-tokens/keycloak-client-secrets.txt 2>/dev/null || echo "Credentials file not found"
```

Parse and update `.env` with the client secrets:
```bash
# Extract the web client secret
WEB_SECRET=$(grep "^KEYCLOAK_CLIENT_SECRET=" .oauth-tokens/keycloak-client-secrets.txt 2>/dev/null | head -1 | cut -d'=' -f2)
M2M_SECRET=$(grep "^KEYCLOAK_M2M_CLIENT_SECRET=" .oauth-tokens/keycloak-client-secrets.txt 2>/dev/null | head -1 | cut -d'=' -f2)

echo "Web secret retrieved: ${#WEB_SECRET} characters"
echo "M2M secret retrieved: ${#M2M_SECRET} characters"

# Update .env using Python to handle special characters safely
python3 -c "
import re
content = open('.env').read()
web_secret = '${WEB_SECRET}'
m2m_secret = '${M2M_SECRET}'

content = re.sub(r'^KEYCLOAK_CLIENT_SECRET=.*', f'KEYCLOAK_CLIENT_SECRET={web_secret}', content, flags=re.MULTILINE)
if 'KEYCLOAK_CLIENT_SECRET=' not in content:
    content += f'\nKEYCLOAK_CLIENT_SECRET={web_secret}'

content = re.sub(r'^KEYCLOAK_M2M_CLIENT_SECRET=.*', f'KEYCLOAK_M2M_CLIENT_SECRET={m2m_secret}', content, flags=re.MULTILINE)
if 'KEYCLOAK_M2M_CLIENT_SECRET=' not in content:
    content += f'\nKEYCLOAK_M2M_CLIENT_SECRET={m2m_secret}'

open('.env', 'w').write(content)
print('Secrets updated in .env')
"
```

Verify:
```bash
grep "^KEYCLOAK_CLIENT_SECRET=" .env | sed 's/=.*/=[set]/'
grep "^KEYCLOAK_M2M_CLIENT_SECRET=" .env | sed 's/=.*/=[set]/'
```

Report: "Client credentials retrieved and saved to `.oauth-tokens/`. `.env` updated with secrets."

---

### Phase 11: Create Test Agents

**Announce to the user:**
> "Phase 11 of 13: Test Agent Setup. I will create service account agents that can authenticate with the MCP Gateway. These agents are used by AI coding assistants and for testing."

Use `AskUserQuestion` to confirm: "Create test agents" / "Skip".

```bash
chmod +x keycloak/setup/setup-agent-service-account.sh

export KEYCLOAK_ADMIN_PASSWORD="${KEYCLOAK_ADMIN_PASSWORD}"

# Create a test agent with unrestricted access
echo "Creating test-agent..."
./keycloak/setup/setup-agent-service-account.sh \
    --agent-id test-agent \
    --group mcp-servers-unrestricted
echo "test-agent creation exit code: $?"

# Create an AI coding assistant agent
echo "Creating ai-coding-assistant..."
./keycloak/setup/setup-agent-service-account.sh \
    --agent-id ai-coding-assistant \
    --group mcp-servers-unrestricted
echo "ai-coding-assistant creation exit code: $?"
```

Retrieve updated credentials including new agents:
```bash
./keycloak/setup/get-all-client-credentials.sh
echo "Updated credentials in .oauth-tokens/"
ls .oauth-tokens/
```

Report: "Test agents created. Credentials saved to `.oauth-tokens/`."

---

### Phase 12: Start All Services

**Announce to the user:**
> "Phase 12 of 13: Starting All Services. I will start the complete MCP Gateway stack using pre-built Docker images. This includes: registry, auth-server, nginx proxy, currenttime-server, fininfo-server, mcpgw-server, and realserverfaketools-server. First-run image pulls may take 3-5 minutes."

Use `AskUserQuestion` to confirm: "Start all MCP Gateway services" / "Cancel".

```bash
chmod +x build_and_run.sh

# Start all services with pre-built images (fastest option, no local build required)
./build_and_run.sh --prebuilt
echo "build_and_run.sh exit code: $?"
```

Wait for services to start:
```bash
echo "Waiting 20 seconds for services to initialize..."
sleep 20

echo "=== Service Status ==="
docker compose ps
```

Check core endpoints:
```bash
echo "=== Health Checks ==="

# Registry health
REGISTRY_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost/health 2>/dev/null || echo "000")
echo "Registry health (http://localhost/health): HTTP ${REGISTRY_STATUS}"

# Keycloak realm
KEYCLOAK_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/realms/mcp-gateway 2>/dev/null || echo "000")
echo "Keycloak mcp-gateway realm: HTTP ${KEYCLOAK_STATUS}"

# Main UI
UI_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost/ 2>/dev/null || echo "000")
echo "Main UI (http://localhost/): HTTP ${UI_STATUS}"
```

If `REGISTRY_STATUS` is not `200`, show logs:
```bash
docker compose logs registry --tail 20
docker compose logs auth-server --tail 20
```

Report services that are up and any that may need attention.

---

### Phase 13: Verification and Summary

**Announce to the user:**
> "Phase 13 of 13: Final Verification. I will run a complete health check across all components and provide a summary of your MCP Gateway installation."

```bash
echo "=== Complete Service Status ==="
docker compose ps

echo ""
echo "=== Endpoint Verification ==="
for URL in "http://localhost/health" "http://localhost:8080/realms/mcp-gateway" "http://localhost/"; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$URL" 2>/dev/null || echo "000")
    echo "  ${URL}: HTTP ${STATUS}"
done

echo ""
echo "=== OAuth Token Files ==="
ls .oauth-tokens/ 2>/dev/null | head -20

echo ""
echo "=== Key .env Settings ==="
grep -E "^(AUTH_PROVIDER|AUTH_SERVER_EXTERNAL_URL|KEYCLOAK_REALM|KEYCLOAK_CLIENT_ID)=" .env
grep "^KEYCLOAK_ADMIN_PASSWORD=" .env | sed 's/=.*/=[set]/'
grep "^KEYCLOAK_CLIENT_SECRET=" .env | sed 's/=.*/=[set]/'
```

Present the final summary to the user:

```
Setup Complete! MCP Gateway & Registry is running on your Mac.

Access Points:
  Main UI:           http://localhost
  Keycloak Admin:    http://localhost:8080/admin
  Registry API:      http://localhost/health
  API Gateway:       http://localhost/mcpgw/mcp

Login Credentials:
  Username: admin
  Password: [the KEYCLOAK_ADMIN_PASSWORD you set in Phase 2]

Agent Credentials (for AI tools like Claude, Cursor, VS Code):
  Test Agent:          .oauth-tokens/agent-test-agent-m2m.env
  AI Coding Assistant: .oauth-tokens/agent-ai-coding-assistant-m2m.env

Quick Test (run in a new terminal):
  source .venv/bin/activate
  source .oauth-tokens/agent-test-agent-m2m.env
  uv run cli/mcp_client.py ping

Next Steps:
  1. Open http://localhost in your browser
  2. Click "Login with Keycloak"
  3. Use admin / [your password] to access the registry
  4. Explore the web interface to register and manage MCP servers
  5. Configure your AI coding assistant using the agent credentials above

To completely remove the installation later, run: /macos-setup teardown
```

---

## TEARDOWN WORKFLOW

The teardown workflow removes all MCP Gateway components. It asks for confirmation at each step and presents options for how thorough the cleanup should be.

---

### Phase T1: Confirm Scope of Removal

**Announce to the user:**
> "Teardown Mode: I will help you remove the MCP Gateway & Registry components from your system."

Use `AskUserQuestion` to ask the user about the scope of removal. Present each as a separate yes/no question:

**Question 1 - Core teardown (always required):**
```
This will:
  - Stop and remove ALL running MCP Gateway Docker containers
  - Remove ALL Docker volumes (Keycloak config, database data - IRREVERSIBLE)
  - Remove .oauth-tokens/ directory (all credentials)
  - Remove .env configuration file

This is required for a proper teardown. Do you want to proceed?
```

**If user says No, stop immediately. Do not proceed.**

**Question 2 - Model files:**
```
Remove the embeddings model files (~90MB) at ~/mcp-gateway/?

  Yes - Remove ~/mcp-gateway/ directory (frees ~90MB, saves time to re-download if you set up again)
  No  - Keep ~/mcp-gateway/ (faster re-setup if you plan to reinstall)
```

Store answer as `REMOVE_MODELS`.

**Question 3 - Docker images:**
```
Remove MCP Gateway Docker images from your local Docker cache?

  Yes - Remove downloaded images (frees several GB of disk space; images will re-download on next setup)
  No  - Keep images cached (much faster if you re-install later)
```

Store answer as `REMOVE_IMAGES`.

**Question 4 - Docker system prune (only ask if user said Yes to images):**
```
Run 'docker system prune' to also remove ALL unused Docker resources
(not just MCP Gateway - any unused container, image, volume, or network)?

  Yes - Full Docker cleanup (maximum disk recovery)
  No  - Only remove MCP Gateway images
```

Store answer as `DOCKER_PRUNE`.

Present a final confirmation:
```
About to perform the following:
  - Stop and remove all MCP Gateway containers and volumes [ALWAYS]
  - Remove .env and .oauth-tokens/ [ALWAYS]
  - Remove ~/mcp-gateway/ model files: [YES/NO]
  - Remove Docker images: [YES/NO]
  - Docker system prune: [YES/NO]

THIS CANNOT BE UNDONE. Confirm to proceed.
```

Use `AskUserQuestion`: "Yes, remove everything I selected" / "Cancel - do not change anything".

---

### Phase T2: Stop All Services and Remove Volumes

**Announce to the user:**
> "Stopping all containers and removing Docker volumes..."

```bash
# Stop all services and remove volumes (-v removes named volumes)
docker compose down -v 2>/dev/null || docker-compose down -v 2>/dev/null || echo "Note: docker compose returned non-zero (may be no services were running)"

echo "=== Verifying containers are stopped ==="
docker ps | grep -E "keycloak|registry|auth-server|nginx|mcpgw|fininfo|currenttime|realserver" && echo "WARNING: Some containers still running" || echo "All MCP Gateway containers stopped"

echo "=== Verifying volumes removed ==="
docker volume ls | grep -E "mcp.gateway|keycloak" && echo "WARNING: Some volumes still exist" || echo "All MCP Gateway volumes removed"
```

---

### Phase T3: Remove Generated Files

**Announce to the user:**
> "Removing generated credentials and configuration files..."

```bash
# Remove OAuth token files
if [ -d ".oauth-tokens" ]; then
    rm -rf .oauth-tokens/
    echo "Removed .oauth-tokens/ directory"
else
    echo ".oauth-tokens/ directory not found (already removed or never created)"
fi

# Remove .env file
if [ -f ".env" ]; then
    rm .env
    echo "Removed .env file"
else
    echo ".env file not found (already removed or never created)"
fi
```

---

### Phase T4: Remove Model Files (if selected)

**Only execute if user chose to remove models in Phase T1.**

```bash
if [ -d "${HOME}/mcp-gateway" ]; then
    rm -rf ${HOME}/mcp-gateway/
    echo "Removed ${HOME}/mcp-gateway/ directory"
else
    echo "${HOME}/mcp-gateway/ directory not found (already removed or never created)"
fi
```

---

### Phase T5: Remove Docker Images (if selected)

**Only execute if user chose to remove images in Phase T1.**

```bash
echo "=== Finding MCP Gateway Docker images ==="
docker images | grep -E "mcpgateway|mcp-gateway-registry" | awk '{print $1":"$2}' | head -20

# Remove MCP Gateway images
docker images | grep -E "mcpgateway|mcp-gateway-registry" | awk '{print $3}' | sort -u | xargs -r docker rmi -f 2>/dev/null
echo "MCP Gateway image removal exit code: $?"

echo "=== Remaining images ==="
docker images | grep -E "mcpgateway|mcp-gateway-registry" && echo "Some images remain" || echo "All MCP Gateway images removed"
```

If user also selected Docker system prune:

```bash
echo "Running Docker system prune (removes ALL unused Docker resources)..."
docker system prune -a --volumes --force
echo "Docker system prune exit code: $?"
```

---

### Phase T6: Verify Cleanup

```bash
echo "=== Teardown Verification ==="

echo ""
echo "Running containers:"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "keycloak|registry|auth-server" && echo "WARNING: MCP Gateway containers still running" || echo "No MCP Gateway containers running"

echo ""
echo "Docker volumes:"
docker volume ls | grep -E "mcp.gateway|keycloak" && echo "WARNING: MCP Gateway volumes still exist" || echo "No MCP Gateway volumes"

echo ""
echo "Configuration files:"
ls .env 2>/dev/null && echo "WARNING: .env still exists" || echo ".env removed"
ls -d .oauth-tokens/ 2>/dev/null && echo "WARNING: .oauth-tokens/ still exists" || echo ".oauth-tokens/ removed"

echo ""
echo "Model files:"
ls -d ${HOME}/mcp-gateway/ 2>/dev/null && echo "~/mcp-gateway/ still exists (was not selected for removal or removal failed)" || echo "~/mcp-gateway/ removed"
```

Present the teardown summary:

```
Teardown Complete.

Removed:
  - All Docker containers (keycloak, registry, auth-server, and others)
  - All Docker volumes (all Keycloak and database data)
  - .env configuration file
  - .oauth-tokens/ credential directory
  [- ~/mcp-gateway/ model files (if selected)]
  [- Docker images (if selected)]

The repository source code remains at its current location.

To reinstall, run: /macos-setup setup
```

---

## Error Handling Reference

### Docker Not Running
If any docker command fails with "Cannot connect to the Docker daemon":
> "Docker Desktop is not running. Please open Docker Desktop from your Applications folder and wait for the whale icon to appear in the Mac menu bar (top right). Then confirm to retry."

### Port Conflicts
If services fail to start due to port conflicts:
```bash
lsof -i :80 && echo "Port 80 in use"
lsof -i :8080 && echo "Port 8080 in use"
lsof -i :7860 && echo "Port 7860 in use"
```
Inform the user which process holds the port and suggest stopping it.

### Keycloak Container Name Varies
The Keycloak container may be named differently. Always detect dynamically:
```bash
KEYCLOAK_CONTAINER=$(docker ps --format "{{.Names}}" | grep keycloak | grep -v db | head -1)
```
Use `${KEYCLOAK_CONTAINER}` in all `docker exec` commands.

### init-keycloak.sh Fails
1. Check Keycloak logs: `docker compose logs keycloak --tail 30`
2. Verify SSL was disabled: re-run Phase 8 commands
3. Verify Keycloak is fully ready: `curl -s http://localhost:8080/realms/master | python3 -m json.tool`
4. Retry the init script after confirming Keycloak is healthy

### Partial/Interrupted Setup
If the user runs setup again after a failure, check what already exists:
```bash
ls .env 2>/dev/null && echo ".env exists"
ls .oauth-tokens/ 2>/dev/null && echo ".oauth-tokens exists"
docker ps | grep keycloak && echo "Keycloak running"
```
Offer to skip phases that are already complete, or start fresh with a teardown.

### sed Command Failures on macOS
Always use `sed -i ''` (with empty string argument) for macOS. Linux uses `sed -i`. When in doubt, use the Python-based `.env` update approach shown in Phase 3.

---

## Important Rules

- **Never execute a phase without user confirmation** - always use `AskUserQuestion` before any system-modifying action
- **Never show passwords** in terminal output or summaries - always mask with `[set]`
- **Both Keycloak passwords are mandatory** - there are no acceptable defaults for security credentials
- **Export passwords before docker compose** - they must be in the environment when Docker Compose reads them
- **macOS sed syntax** - always `sed -i ''` not `sed -i`
- **Detect container names dynamically** - never hardcode `mcp-gateway-registry-keycloak-1`, always query with `docker ps`
- **One phase at a time** - complete and verify each phase before announcing the next
- **Teardown is irreversible** - always present a final confirmation with a full list of what will be removed before executing any teardown action
- **Carry variables across phases** - `KEYCLOAK_ADMIN_PASSWORD`, `KEYCLOAK_DB_PASSWORD`, and `KEYCLOAK_CONTAINER` must persist across all phases. Re-export them at the start of any phase that uses them in shell commands.
