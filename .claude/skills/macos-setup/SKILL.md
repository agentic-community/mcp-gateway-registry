---
name: macos-setup
description: "Complete macOS setup and teardown for MCP Gateway & Registry (AI-registry). Clones the repository, installs all services, configures Keycloak auth, registers the Cloudflare docs server, and verifies the full stack. Also supports complete teardown. Can be run directly from its GitHub URL without the repository already cloned. Uses an interactive or default-values mode chosen at startup."
license: Apache-2.0
metadata:
  author: mcp-gateway-registry
  version: "2.0"
---

# MCP Gateway & Registry — macOS Setup Skill

**Repository:** https://github.com/agentic-community/mcp-gateway-registry
**This skill:** https://github.com/agentic-community/mcp-gateway-registry/blob/main/.claude/skills/macos-setup/SKILL.md
**Full macOS guide:** https://github.com/agentic-community/mcp-gateway-registry/blob/main/docs/macos-setup-guide.md

## How to run this skill without cloning the repository

This skill is self-contained. You can invoke it from any directory in Claude Code using the GitHub URL. It will clone the repository for you.

```
/macos-setup
```

Or reference it remotely if you have not installed this repo:

```
@https://raw.githubusercontent.com/agentic-community/mcp-gateway-registry/main/.claude/skills/macos-setup/SKILL.md
```

---

## What this skill does

**`/macos-setup setup`** — Full guided installation on a fresh macOS machine:
- Clones the MCP Gateway & Registry repository
- Installs and configures all services (Keycloak, registry, auth-server, MCP servers)
- Builds all Docker images from source
- Registers the Cloudflare Documentation MCP server so it appears immediately on login
- Ends with a complete summary of every step taken

**`/macos-setup teardown`** — Removes all MCP Gateway components from your system

---

## CRITICAL: First action is ALWAYS Step 0

**DO NOT run any Bash commands. DO NOT check prerequisites. DO NOT read any files.**
**The very first action when this skill is invoked MUST be using `AskUserQuestion` to complete Step 0.**
**Nothing else happens until the user has answered all three Step 0 questions.**

---

## Step tracking

Throughout the entire execution, Claude must maintain an internal step log. After every phase completes (success, skip, or failure), append an entry to this log. Display the full log as a formatted table in the Final Summary phase.

Step log format: `{ phase, name, status (DONE / SKIPPED / FAILED), notes }`

---

## Step 0: Determine Mode — MUST BE FIRST, NO EXCEPTIONS

**STOP. Do not run any commands. Use `AskUserQuestion` right now to ask all three questions below before taking any other action.**

**Question 1 — Operation:**
```
Which operation would you like to perform?

  Setup   - Install the MCP Gateway & Registry (AI-registry) from scratch.
            Builds all services from source. Estimated time: 20-40 minutes.

  Teardown - Stop all services and remove all components. Irreversible.
```

If the user invoked the skill with an argument (`/macos-setup setup` or `/macos-setup teardown`), skip this question.

**Question 2 — Execution mode (Setup only):**
```
How should I run the setup?

  Default (recommended) - Use sensible defaults for all prompts. I will only
    pause for decisions that truly require your input. Passwords will be
    auto-generated and shown in the final summary.

  Interactive - Ask for your confirmation and input before every phase.
    You control each step individually.
```

Store the answer as `EXECUTION_MODE` = `default` or `interactive`.

**Question 3 — Installation directory (Setup only):**
```
Where should the AI-registry project be installed?

Default: ~/AI-registry

Enter a path, or press Enter / select the default option to use ~/AI-registry.
```

Store the answer as `INSTALL_DIR`. If the user provides no input or selects the default, set `INSTALL_DIR=~/AI-registry` and inform the user:
> "Using default installation directory: ~/AI-registry"

**Expand the path immediately:**
```bash
INSTALL_DIR=$(eval echo "${INSTALL_DIR}")
echo "Installation directory: ${INSTALL_DIR}"
```

Log: `{ 0, "Mode & Directory Selection", DONE, "Mode: ${EXECUTION_MODE}, Dir: ${INSTALL_DIR}" }`

---

## SETUP WORKFLOW

For every phase below, apply this rule:
- **Interactive mode**: announce the phase, use `AskUserQuestion` to confirm before executing
- **Default mode**: announce what you are doing with a one-line message, then execute immediately without asking confirmation

---

### Phase 1: Prerequisites Check

**Announce:** "Checking prerequisites..."

```bash
echo "=== Docker ==="
docker --version 2>/dev/null && docker ps >/dev/null 2>&1 && echo "DOCKER_OK" || echo "DOCKER_FAIL"

echo "=== Python ==="
python3 --version 2>/dev/null && echo "PYTHON_OK" || echo "PYTHON_FAIL"

echo "=== uv ==="
uv --version 2>/dev/null && echo "UV_OK" || echo "UV_FAIL"

echo "=== Node.js (required for building from source) ==="
node --version 2>/dev/null && echo "NODE_OK" || echo "NODE_FAIL"

echo "=== git ==="
git --version 2>/dev/null && echo "GIT_OK" || echo "GIT_FAIL"

echo "=== jq ==="
jq --version 2>/dev/null && echo "JQ_OK" || echo "JQ_FAIL"
```

For any failed check, display the install instructions:

| Check | Install command |
|-------|----------------|
| DOCKER_FAIL | "Install Docker Desktop from https://www.docker.com/products/docker-desktop/ then start it and wait for the whale icon in the menu bar" |
| PYTHON_FAIL | `brew install python@3.12` |
| UV_FAIL | `curl -LsSf https://astral.sh/uv/install.sh \| sh` — then restart your terminal |
| NODE_FAIL | `brew install node@20` or download from https://nodejs.org/ |
| GIT_FAIL | `xcode-select --install` |
| JQ_FAIL | `brew install jq` |

**Do not proceed if Docker or git fail.** Python, uv, Node.js, and jq must also be present before continuing. Ask the user to install missing tools and retry.

Log: `{ 1, "Prerequisites Check", DONE/FAILED, list of what passed/failed }`

---

### Phase 2: Clone Repository

**Announce:** "Cloning the MCP Gateway & Registry repository to `${INSTALL_DIR}`..."

First check if the directory already exists:

```bash
if [ -d "${INSTALL_DIR}" ]; then
    echo "ALREADY_EXISTS"
    ls "${INSTALL_DIR}/docker-compose.yml" 2>/dev/null && echo "REPO_OK" || echo "NOT_A_REPO"
else
    echo "WILL_CLONE"
fi
```

**If `ALREADY_EXISTS` and `REPO_OK`:** Inform the user and ask (both modes):
```
The directory ${INSTALL_DIR} already contains the repository.

  Use existing - Continue setup with the existing copy
  Re-clone     - Remove it and clone fresh (WARNING: deletes existing data)
```

**If `WILL_CLONE`:** Clone the repository:
```bash
# Create parent directory if needed
mkdir -p "$(dirname "${INSTALL_DIR}")"

# Clone
git clone https://github.com/agentic-community/mcp-gateway-registry.git "${INSTALL_DIR}"
echo "Clone exit code: $?"
```

After cloning or confirming existing, change into the directory:
```bash
cd "${INSTALL_DIR}"
echo "Working directory: $(pwd)"
ls docker-compose.yml build_and_run.sh 2>/dev/null && echo "REPO_VERIFIED" || echo "REPO_INVALID"
```

All subsequent phases run commands from within `${INSTALL_DIR}`.

Key files now available locally (GitHub references for documentation):
- [`build_and_run.sh`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/build_and_run.sh)
- [`docker-compose.yml`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/docker-compose.yml)
- [`.env.example`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/.env.example)

Log: `{ 2, "Repository Clone", DONE/SKIPPED, "Cloned to ${INSTALL_DIR} / Used existing" }`

---

### Phase 3: Credentials Configuration

**Announce:** "Configuring credentials..."

**In default mode:** Auto-generate both passwords using Python. Store them for the final summary.

```bash
KEYCLOAK_ADMIN_PASSWORD=$(python3 -c "import secrets, string; print(''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(20)))")
KEYCLOAK_DB_PASSWORD=$(python3 -c "import secrets, string; print(''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(20)))")
echo "Passwords auto-generated (will be shown in final summary)"
echo "Admin password length: ${#KEYCLOAK_ADMIN_PASSWORD}"
echo "DB password length: ${#KEYCLOAK_DB_PASSWORD}"
```

**In interactive mode:** Use `AskUserQuestion` to collect:

- **Keycloak Admin Password** — minimum 8 characters, REQUIRED, no default. Used to log in at `http://localhost:8080/admin`.
- **Keycloak Database Password** — minimum 8 characters, REQUIRED, no default. Internal Keycloak database credential.

Validate: if either password is fewer than 8 characters or empty, re-prompt. Do not proceed with weak passwords.

Log: `{ 3, "Credentials Configuration", DONE, "default-generated / user-provided" }`

---

### Phase 4: Environment File Setup

**Announce:** "Creating `.env` configuration file..."

Check for existing `.env`:
```bash
ls -la "${INSTALL_DIR}/.env" 2>/dev/null && echo "ENV_EXISTS" || echo "ENV_MISSING"
```

In **interactive mode** with existing `.env`, ask to overwrite. In **default mode**, overwrite automatically and note it in the log.

```bash
cd "${INSTALL_DIR}"

# Copy template
cp .env.example .env

# Generate SECRET_KEY
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(64))")
echo "SECRET_KEY generated: ${#SECRET_KEY} characters"
```

Update `.env` using Python to handle special characters safely:

```bash
cd "${INSTALL_DIR}"

python3 << 'PYEOF'
import re, os

env_path = '.env'
content = open(env_path).read()

updates = {
    'AUTH_PROVIDER': 'keycloak',
    'AUTH_SERVER_EXTERNAL_URL': 'http://localhost',
    'KEYCLOAK_ADMIN_PASSWORD': os.environ.get('KEYCLOAK_ADMIN_PASSWORD', ''),
    'KEYCLOAK_DB_PASSWORD': os.environ.get('KEYCLOAK_DB_PASSWORD', ''),
    'SECRET_KEY': os.environ.get('SECRET_KEY', ''),
}

for key, value in updates.items():
    pattern = rf'^{key}=.*'
    replacement = f'{key}={value}'
    if re.search(pattern, content, flags=re.MULTILINE):
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    else:
        content += f'\n{key}={value}'

open(env_path, 'w').write(content)
print('Environment file updated successfully')
PYEOF
```

Verify (without exposing values):
```bash
cd "${INSTALL_DIR}"
for KEY in AUTH_PROVIDER AUTH_SERVER_EXTERNAL_URL KEYCLOAK_ADMIN_PASSWORD KEYCLOAK_DB_PASSWORD SECRET_KEY; do
    VALUE=$(grep "^${KEY}=" .env | cut -d'=' -f2)
    if [ -n "$VALUE" ]; then
        echo "${KEY}=[set]"
    else
        echo "${KEY}=[MISSING - ERROR]"
    fi
done
```

The template is at: [`.env.example`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/.env.example)

Log: `{ 4, "Environment File Setup", DONE, ".env created and configured" }`

---

### Phase 5: Python Virtual Environment

**Announce:** "Installing Python dependencies via `uv sync`..."

```bash
cd "${INSTALL_DIR}"
uv sync
echo "uv sync exit code: $?"
ls -la .venv/bin/python 2>/dev/null && echo "VENV_OK" || echo "VENV_FAIL"
```

Log: `{ 5, "Python Virtual Environment", DONE/FAILED, "" }`

---

### Phase 6: Download Embeddings Model

**Announce:** "Downloading sentence-transformers embeddings model (~90MB) to `~/mcp-gateway/models/`..."

This model powers intelligent tool discovery. It is downloaded from HuggingFace.

```bash
mkdir -p "${HOME}/mcp-gateway/models/all-MiniLM-L6-v2"

cd "${INSTALL_DIR}"

# Try huggingface-cli first, fall back to Python API
if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
        --local-dir "${HOME}/mcp-gateway/models/all-MiniLM-L6-v2"
else
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

echo "Model files: $(ls ${HOME}/mcp-gateway/models/all-MiniLM-L6-v2/ | wc -l | tr -d ' ') files"
```

Log: `{ 6, "Embeddings Model Download", DONE/FAILED, "~/mcp-gateway/models/all-MiniLM-L6-v2" }`

---

### Phase 7: Create Required Directories

**Announce:** "Creating Docker volume mount directories..."

```bash
mkdir -p "${HOME}/mcp-gateway/{servers,models,auth_server,secrets/fininfo,logs,ssl}"
ls -la "${HOME}/mcp-gateway/"
```

Log: `{ 7, "Directory Creation", DONE, "~/mcp-gateway/{servers,models,auth_server,secrets/fininfo,logs,ssl}" }`

---

### Phase 8: Start Keycloak Services

**Announce:** "Starting Keycloak authentication services (1-3 minute wait)..."

```bash
cd "${INSTALL_DIR}"

export KEYCLOAK_ADMIN_PASSWORD="${KEYCLOAK_ADMIN_PASSWORD}"
export KEYCLOAK_DB_PASSWORD="${KEYCLOAK_DB_PASSWORD}"

docker compose up -d keycloak-db keycloak
echo "Docker compose exit code: $?"
```

Poll until Keycloak responds (max 180 seconds):

```bash
echo "Waiting for Keycloak to be ready..."
TIMEOUT=180
ELAPSED=0
READY=false

while [ $ELAPSED -lt $TIMEOUT ]; do
    HTTP=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/realms/master 2>/dev/null || echo "000")
    if [ "$HTTP" = "200" ]; then
        echo "Keycloak ready after ${ELAPSED}s"
        READY=true
        break
    fi
    echo "  ${ELAPSED}s — HTTP ${HTTP}, still waiting..."
    sleep 10
    ELAPSED=$((ELAPSED + 10))
done

[ "$READY" = "false" ] && echo "ERROR: Keycloak did not start within ${TIMEOUT}s" && docker compose logs keycloak --tail 20
```

Verify:
```bash
curl -s http://localhost:8080/realms/master | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print('Keycloak master realm:', d.get('realm'))
except Exception as e:
    print('Parse error:', e)
"
```

Log: `{ 8, "Keycloak Startup", DONE/FAILED, "Ready in Xs / timed out" }`

---

### Phase 9: Fix macOS SSL Requirement

**Announce:** "Disabling Keycloak HTTPS requirement for local macOS development..."

On macOS, Docker's VM causes Keycloak to enforce HTTPS on all connections. This must be disabled for local development.

Detect the Keycloak container name:
```bash
KEYCLOAK_CONTAINER=$(docker ps --format "{{.Names}}" | grep keycloak | grep -v db | head -1)
echo "Keycloak container: ${KEYCLOAK_CONTAINER}"
[ -z "$KEYCLOAK_CONTAINER" ] && echo "ERROR: No Keycloak container running" && docker ps && exit 1
```

Disable SSL on master realm:
```bash
docker exec ${KEYCLOAK_CONTAINER} /opt/keycloak/bin/kcadm.sh config credentials \
    --server http://localhost:8080 --realm master \
    --user admin --password "${KEYCLOAK_ADMIN_PASSWORD}"

docker exec ${KEYCLOAK_CONTAINER} /opt/keycloak/bin/kcadm.sh update realms/master -s sslRequired=NONE
echo "SSL disabled for master realm, exit code: $?"
```

Verify:
```bash
HTTP=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:8080/admin/")
echo "Admin endpoint: HTTP ${HTTP} (302 = success)"
```

Log: `{ 9, "Keycloak SSL Fix (master realm)", DONE/FAILED, "HTTP ${HTTP}" }`

---

### Phase 10: Initialize Keycloak Realm and Clients

**Announce:** "Initializing Keycloak — creating mcp-gateway realm and OAuth clients..."

Script: [`keycloak/setup/init-keycloak.sh`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/keycloak/setup/init-keycloak.sh)

```bash
cd "${INSTALL_DIR}"
chmod +x keycloak/setup/init-keycloak.sh

export KEYCLOAK_ADMIN_PASSWORD="${KEYCLOAK_ADMIN_PASSWORD}"
./keycloak/setup/init-keycloak.sh
echo "Init exit code: $?"
```

After initialization, disable SSL on the newly created `mcp-gateway` realm:

```bash
KEYCLOAK_CONTAINER=$(docker ps --format "{{.Names}}" | grep keycloak | grep -v db | head -1)

docker exec ${KEYCLOAK_CONTAINER} /opt/keycloak/bin/kcadm.sh config credentials \
    --server http://localhost:8080 --realm master \
    --user admin --password "${KEYCLOAK_ADMIN_PASSWORD}"

docker exec ${KEYCLOAK_CONTAINER} /opt/keycloak/bin/kcadm.sh update realms/mcp-gateway -s sslRequired=NONE
echo "SSL disabled for mcp-gateway realm, exit code: $?"
```

Verify both realms:
```bash
curl -s http://localhost:8080/realms/master | python3 -c "import sys,json; print('master:', json.load(sys.stdin).get('realm'))"
curl -s http://localhost:8080/realms/mcp-gateway | python3 -c "import sys,json; print('mcp-gateway:', json.load(sys.stdin).get('realm'))"
```

**Common failures and fixes:**
- If the script fails with an SSL error: re-run Phase 9 before retrying
- If Keycloak is not responding: wait 30 seconds and retry — it may still be initializing
- If admin credentials are rejected: verify `KEYCLOAK_ADMIN_PASSWORD` matches what was set in Phase 3

Log: `{ 10, "Keycloak Init (realm + clients)", DONE/FAILED, "mcp-gateway realm created" }`

---

### Phase 11: Retrieve Client Credentials

**Announce:** "Retrieving OAuth client secrets from Keycloak and updating `.env`..."

Script: [`keycloak/setup/get-all-client-credentials.sh`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/keycloak/setup/get-all-client-credentials.sh)

```bash
cd "${INSTALL_DIR}"
chmod +x keycloak/setup/get-all-client-credentials.sh
./keycloak/setup/get-all-client-credentials.sh
echo "Credentials retrieval exit code: $?"
```

Parse the retrieved secrets and update `.env`:

```bash
cd "${INSTALL_DIR}"

WEB_SECRET=$(grep "^KEYCLOAK_CLIENT_SECRET=" .oauth-tokens/keycloak-client-secrets.txt 2>/dev/null | head -1 | cut -d'=' -f2)
M2M_SECRET=$(grep "^KEYCLOAK_M2M_CLIENT_SECRET=" .oauth-tokens/keycloak-client-secrets.txt 2>/dev/null | head -1 | cut -d'=' -f2)

echo "Web client secret: ${#WEB_SECRET} characters"
echo "M2M client secret: ${#M2M_SECRET} characters"

python3 << PYEOF
import re

content = open('.env').read()
web = '${WEB_SECRET}'
m2m = '${M2M_SECRET}'

for key, val in [('KEYCLOAK_CLIENT_SECRET', web), ('KEYCLOAK_M2M_CLIENT_SECRET', m2m)]:
    if re.search(rf'^{key}=', content, flags=re.MULTILINE):
        content = re.sub(rf'^{key}=.*', f'{key}={val}', content, flags=re.MULTILINE)
    else:
        content += f'\n{key}={val}'

open('.env', 'w').write(content)
print('Secrets written to .env')
PYEOF
```

Log: `{ 11, "Client Credentials Retrieved", DONE/FAILED, ".oauth-tokens/ populated, .env updated" }`

---

### Phase 12: Create Test Agents

**Announce:** "Creating service account agents for MCP Gateway access..."

Script: [`keycloak/setup/setup-agent-service-account.sh`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/keycloak/setup/setup-agent-service-account.sh)

```bash
cd "${INSTALL_DIR}"
chmod +x keycloak/setup/setup-agent-service-account.sh

export KEYCLOAK_ADMIN_PASSWORD="${KEYCLOAK_ADMIN_PASSWORD}"

echo "Creating test-agent..."
./keycloak/setup/setup-agent-service-account.sh \
    --agent-id test-agent \
    --group mcp-servers-unrestricted
echo "test-agent exit code: $?"

echo "Creating ai-coding-assistant..."
./keycloak/setup/setup-agent-service-account.sh \
    --agent-id ai-coding-assistant \
    --group mcp-servers-unrestricted
echo "ai-coding-assistant exit code: $?"

# Refresh credentials to include new agents
./keycloak/setup/get-all-client-credentials.sh
echo "Credentials refreshed"
ls .oauth-tokens/
```

Log: `{ 12, "Test Agents Created", DONE/FAILED, "test-agent, ai-coding-assistant" }`

---

### Phase 13: Build and Start All Services

**Announce:** "Building all Docker images from source and starting all services. This builds the React frontend and all containers locally — this will take 20-40 minutes on first run."

Script: [`build_and_run.sh`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/build_and_run.sh)

```bash
cd "${INSTALL_DIR}"
chmod +x build_and_run.sh

# Build from source (no --prebuilt flag)
./build_and_run.sh
echo "build_and_run.sh exit code: $?"
```

After the build completes, wait for services to initialize:

```bash
echo "Waiting 30 seconds for all services to start..."
sleep 30

echo "=== Service Status ==="
docker compose ps

echo "=== Health Checks ==="
for URL in "http://localhost/health" "http://localhost/" "http://localhost:8080/realms/mcp-gateway"; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${URL}" 2>/dev/null || echo "000")
    echo "  ${URL}: HTTP ${STATUS}"
done
```

If services are not healthy, show logs:
```bash
docker compose logs registry --tail 20
docker compose logs auth-server --tail 20
```

Log: `{ 13, "Build and Start All Services", DONE/FAILED, "Build time: ~Xmin, services: up/partial" }`

---

### Phase 14: Generate Access Tokens

**Announce:** "Generating access tokens for all agents..."

Script: [`credentials-provider/keycloak/generate_tokens.py`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/credentials-provider/keycloak/generate_tokens.py)

```bash
cd "${INSTALL_DIR}"

uv run credentials-provider/keycloak/generate_tokens.py --all-agents 2>/dev/null
echo "Token generation exit code: $?"

echo "=== Available token files ==="
ls .oauth-tokens/*.env 2>/dev/null | head -10
```

Log: `{ 14, "Access Token Generation", DONE/FAILED, "Tokens in .oauth-tokens/" }`

---

### Phase 15: Register Cloudflare Documentation Server

**Announce:** "Registering Cloudflare Documentation MCP Server so it appears immediately on login..."

Config file: [`cli/examples/cloudflare-docs-server-config.json`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/cli/examples/cloudflare-docs-server-config.json)

Registration CLI: [`api/registry_management.py`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/api/registry_management.py)

```bash
cd "${INSTALL_DIR}"

# Verify token file exists for test-agent
TOKEN_FILE=".oauth-tokens/agent-test-agent-m2m.env"
if [ ! -f "$TOKEN_FILE" ]; then
    echo "ERROR: Token file not found: $TOKEN_FILE"
    ls .oauth-tokens/
    exit 1
fi

echo "Token file found: ${TOKEN_FILE}"

# Register the Cloudflare Documentation server
uv run python api/registry_management.py \
    --token-file "${TOKEN_FILE}" \
    --registry-url http://localhost \
    register \
    --config cli/examples/cloudflare-docs-server-config.json \
    --overwrite

echo "Cloudflare registration exit code: $?"
```

Verify the server was registered:

```bash
cd "${INSTALL_DIR}"

uv run python api/registry_management.py \
    --token-file ".oauth-tokens/agent-test-agent-m2m.env" \
    --registry-url http://localhost \
    list 2>/dev/null | grep -i cloudflare && echo "Cloudflare server confirmed in registry" || echo "WARNING: Cloudflare server not found in list"
```

The server configuration that was registered:
```json
{
  "server_name": "Cloudflare Documentation MCP Server",
  "description": "Search Cloudflare documentation and get migration guides",
  "path": "/cloudflare-docs",
  "proxy_pass_url": "https://docs.mcp.cloudflare.com/mcp",
  "supported_transports": ["streamable-http"],
  "tags": ["documentation", "cloudflare", "cdn", "workers", "pages", "migration-guide"]
}
```

Log: `{ 15, "Cloudflare Server Registration", DONE/FAILED, "Registered at /cloudflare-docs" }`

---

### Phase 16: Final Verification and Summary

**Announce:** "Running final verification and preparing your summary..."

```bash
cd "${INSTALL_DIR}"

echo "=== All Services ==="
docker compose ps

echo ""
echo "=== Endpoint Health ==="
declare -A ENDPOINTS=(
    ["Main UI"]="http://localhost/"
    ["Registry Health"]="http://localhost/health"
    ["Keycloak mcp-gateway realm"]="http://localhost:8080/realms/mcp-gateway"
    ["Cloudflare MCP endpoint"]="http://localhost/cloudflare-docs/mcp"
)

for NAME in "${!ENDPOINTS[@]}"; do
    URL="${ENDPOINTS[$NAME]}"
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${URL}" 2>/dev/null || echo "000")
    echo "  ${NAME}: ${URL} — HTTP ${STATUS}"
done

echo ""
echo "=== Registered Servers ==="
uv run python api/registry_management.py \
    --token-file ".oauth-tokens/agent-test-agent-m2m.env" \
    --registry-url http://localhost \
    list 2>/dev/null || echo "Could not retrieve server list"
```

**Display the complete step summary table:**

Present a formatted summary of every phase from the internal step log:

```
========================================
   AI-REGISTRY SETUP COMPLETE
========================================

Installation Directory: ${INSTALL_DIR}

Step Summary:
+-------+--------------------------------------------+----------+-----------------------------+
| Phase | Name                                       | Status   | Notes                       |
+-------+--------------------------------------------+----------+-----------------------------+
|  0    | Mode & Directory Selection                 | DONE     | mode, dir                   |
|  1    | Prerequisites Check                        | DONE     | all passed                  |
|  2    | Repository Clone                           | DONE     | ${INSTALL_DIR}              |
|  3    | Credentials Configuration                  | DONE     | default-generated/provided  |
|  4    | Environment File Setup                     | DONE     | .env configured             |
|  5    | Python Virtual Environment                 | DONE     | .venv created               |
|  6    | Embeddings Model Download                  | DONE     | ~90MB downloaded            |
|  7    | Directory Creation                         | DONE     | ~/mcp-gateway/...           |
|  8    | Keycloak Startup                           | DONE     | ready in Xs                 |
|  9    | Keycloak SSL Fix (master)                  | DONE     | sslRequired=NONE            |
| 10    | Keycloak Init (realm + clients)            | DONE     | mcp-gateway realm           |
| 11    | Client Credentials Retrieved               | DONE     | .oauth-tokens/ updated      |
| 12    | Test Agents Created                        | DONE     | test-agent, ai-assistant    |
| 13    | Build and Start All Services               | DONE     | all containers up           |
| 14    | Access Token Generation                    | DONE     | .oauth-tokens/*.env         |
| 15    | Cloudflare Server Registration             | DONE     | /cloudflare-docs            |
| 16    | Final Verification                         | DONE     | all checks passed           |
+-------+--------------------------------------------+----------+-----------------------------+

Access Points:
  Main UI (login here):   http://localhost
  Keycloak Admin:         http://localhost:8080/admin
  Registry API:           http://localhost/health
  MCP Gateway:            http://localhost/mcpgw/mcp
  Cloudflare MCP server:  http://localhost/cloudflare-docs/mcp

Login Credentials:
  URL:      http://localhost
  Username: admin
  Password: [KEYCLOAK_ADMIN_PASSWORD shown only in default mode — see below]

Agent Credentials:
  Test agent:    ${INSTALL_DIR}/.oauth-tokens/agent-test-agent-m2m.env
  AI assistant:  ${INSTALL_DIR}/.oauth-tokens/agent-ai-coding-assistant-m2m.env

Registered Servers:
  - Cloudflare Documentation MCP Server (/cloudflare-docs) — visible immediately on login

Quick Test:
  cd ${INSTALL_DIR}
  source .venv/bin/activate
  source .oauth-tokens/agent-test-agent-m2m.env
  uv run cli/mcp_client.py ping
```

**In default mode only**, display the auto-generated passwords clearly since the user never set them:

```
Generated Credentials (SAVE THESE):
  Keycloak Admin Password: ${KEYCLOAK_ADMIN_PASSWORD}
  Keycloak DB Password:    ${KEYCLOAK_DB_PASSWORD}

These passwords are also stored in: ${INSTALL_DIR}/.env
```

---

## TEARDOWN WORKFLOW

### Phase T1: Confirm Scope

Use `AskUserQuestion` (always, regardless of mode) to ask:

**Required confirmation:**
```
This will permanently remove:
  - All running MCP Gateway Docker containers
  - All Docker volumes (Keycloak config, database — IRREVERSIBLE)
  - .env configuration file
  - .oauth-tokens/ directory

This cannot be undone. Proceed?
```

Also ask:
- Remove model files at `~/mcp-gateway/`? (Yes / No)
- Remove cached Docker images? (Yes / No)

**Only proceed if the user explicitly confirms.**

---

### Phase T2: Stop All Services and Remove Volumes

```bash
# Detect install dir if not set
INSTALL_DIR="${INSTALL_DIR:-~/AI-registry}"
INSTALL_DIR=$(eval echo "${INSTALL_DIR}")

cd "${INSTALL_DIR}" 2>/dev/null || echo "Directory not found, skipping cd"

docker compose down -v 2>/dev/null || docker-compose down -v 2>/dev/null || echo "No services were running"

docker ps | grep -E "keycloak|registry|auth-server|nginx|mcpgw|fininfo|currenttime" \
    && echo "WARNING: some containers still running" \
    || echo "All MCP Gateway containers stopped"
```

---

### Phase T3: Remove Generated Files

```bash
cd "${INSTALL_DIR}" 2>/dev/null || true

rm -rf .oauth-tokens/ && echo "Removed .oauth-tokens/"
rm -f .env && echo "Removed .env"
```

---

### Phase T4: Remove Model Files (if selected)

```bash
rm -rf "${HOME}/mcp-gateway/" && echo "Removed ~/mcp-gateway/"
```

---

### Phase T5: Remove Docker Images (if selected)

```bash
docker images | grep -E "mcpgateway|mcp-gateway-registry" | awk '{print $3}' | sort -u | xargs -r docker rmi -f
echo "Docker image removal complete"
```

---

### Phase T6: Teardown Summary

```bash
echo "=== Remaining containers ==="
docker ps -a | grep -E "keycloak|registry|auth-server" || echo "None"

echo "=== Remaining volumes ==="
docker volume ls | grep -E "mcp.gateway|keycloak" || echo "None"

echo "=== Files ==="
ls "${INSTALL_DIR}/.env" 2>/dev/null && echo "WARNING: .env still exists" || echo ".env removed"
ls -d "${INSTALL_DIR}/.oauth-tokens/" 2>/dev/null && echo "WARNING: .oauth-tokens/ still exists" || echo ".oauth-tokens/ removed"
```

Present final teardown summary to the user listing everything that was removed.

---

## Error Handling Reference

### Docker not running
> "Docker Desktop is not running. Open it from Applications and wait for the whale icon in the menu bar."

### Port conflict
```bash
lsof -i :80 && echo "Port 80 in use"
lsof -i :8080 && echo "Port 8080 in use"
```

### Keycloak container name varies
Always detect dynamically:
```bash
KEYCLOAK_CONTAINER=$(docker ps --format "{{.Names}}" | grep keycloak | grep -v db | head -1)
```

### init-keycloak.sh fails
1. Check logs: `docker compose logs keycloak --tail 30`
2. Re-run Phase 9 SSL fix
3. Verify Keycloak is responding: `curl -s http://localhost:8080/realms/master`
4. Retry Phase 10

### Cloudflare registration fails
1. Verify services are healthy: `docker compose ps`
2. Verify token file exists and is valid
3. Check registry logs: `docker compose logs registry --tail 20`
4. Retry: `uv run python api/registry_management.py --token-file .oauth-tokens/agent-test-agent-m2m.env --registry-url http://localhost register --config cli/examples/cloudflare-docs-server-config.json --overwrite`

---

## Important Rules

- **EXECUTION_MODE and INSTALL_DIR must be established in Step 0** before any other phase runs
- **In default mode**: execute phases immediately after a brief announcement, no confirmation prompts — the only questions asked are in Step 0
- **In interactive mode**: use `AskUserQuestion` before each phase
- **Never display passwords** in output except in the Final Summary for default-mode auto-generated passwords
- **All commands run from within `${INSTALL_DIR}`** — `cd "${INSTALL_DIR}"` at the start of every phase that runs commands
- **NO `--prebuilt` flag** on `build_and_run.sh` — always build from source
- **Carry `KEYCLOAK_ADMIN_PASSWORD`, `KEYCLOAK_DB_PASSWORD`, `KEYCLOAK_CONTAINER`, `INSTALL_DIR`, `EXECUTION_MODE`** across all phases — re-export shell variables at the start of any phase that uses them
- **macOS `sed` syntax**: always `sed -i ''`, never `sed -i` — or use the Python `.env` update approach
- **Log every phase** to the internal step log and display the full table in Phase 16
