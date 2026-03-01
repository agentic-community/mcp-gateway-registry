#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "Starting Registry Service Setup..."

# --- DocumentDB CA Bundle Download (needed for both init mode and normal mode) ---
if [[ "${DOCUMENTDB_HOST}" == *"docdb-elastic.amazonaws.com"* ]]; then
    echo "Detected DocumentDB Elastic cluster"
    echo "Downloading DocumentDB Elastic CA bundle..."
    CA_BUNDLE_URL="https://www.amazontrust.com/repository/SFSRootCAG2.pem"
    CA_BUNDLE_PATH="/app/global-bundle.pem"
    if [ ! -f "$CA_BUNDLE_PATH" ]; then
        curl -fsSL "$CA_BUNDLE_URL" -o "$CA_BUNDLE_PATH"
        echo "DocumentDB Elastic CA bundle (SFSRootCAG2.pem) downloaded successfully to $CA_BUNDLE_PATH"
    fi
elif [[ "${DOCUMENTDB_HOST}" == *"docdb.amazonaws.com"* ]]; then
    echo "Detected regular DocumentDB cluster"
    echo "Downloading regular DocumentDB CA bundle..."
    CA_BUNDLE_URL="https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem"
    CA_BUNDLE_PATH="/app/global-bundle.pem"
    if [ ! -f "$CA_BUNDLE_PATH" ]; then
        curl -fsSL "$CA_BUNDLE_URL" -o "$CA_BUNDLE_PATH"
        echo "DocumentDB CA bundle (global-bundle.pem) downloaded successfully to $CA_BUNDLE_PATH"
    fi
fi

# Check if we're in init mode (for running DocumentDB initialization scripts)
if [ "$RUN_INIT_SCRIPTS" = "true" ]; then
    echo "Running in init mode - executing initialization scripts..."
    exec "$@"
fi

# --- Environment Variable Setup ---
echo "Setting up environment variables..."

# Get deployment mode (default: with-gateway)
DEPLOYMENT_MODE="${DEPLOYMENT_MODE:-with-gateway}"
REGISTRY_MODE="${REGISTRY_MODE:-full}"

echo "============================================================"
echo "Starting MCP Gateway Registry"
echo "  DEPLOYMENT_MODE: ${DEPLOYMENT_MODE}"
echo "  REGISTRY_MODE: ${REGISTRY_MODE}"
if [ "$DEPLOYMENT_MODE" = "registry-only" ]; then
    echo "  Note: Dynamic MCP server location blocks will NOT be generated"
fi
echo "============================================================"

# Generate secret key if not provided
if [ -z "$SECRET_KEY" ]; then
    SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
fi

ADMIN_USER_VALUE=${ADMIN_USER:-admin}

# Check if ADMIN_PASSWORD is set
if [ -z "$ADMIN_PASSWORD" ]; then
    echo "ERROR: ADMIN_PASSWORD environment variable is not set."
    echo "Please set ADMIN_PASSWORD to a secure value before running the container."
    exit 1
fi

# Create .env file for registry
REGISTRY_ENV_FILE="/app/registry/.env"
echo "Creating Registry .env file..."
echo "SECRET_KEY=${SECRET_KEY}" > "$REGISTRY_ENV_FILE"
echo "ADMIN_USER=${ADMIN_USER_VALUE}" >> "$REGISTRY_ENV_FILE"
echo "ADMIN_PASSWORD=${ADMIN_PASSWORD}" >> "$REGISTRY_ENV_FILE"
echo "Registry .env created."

# DocumentDB CA Bundle already downloaded at the beginning of this script

# --- SSL Certificate Check ---
# These paths match REGISTRY_CONSTANTS.SSL_CERT_PATH and SSL_KEY_PATH in registry/constants.py
SSL_CERT_PATH="/etc/ssl/certs/fullchain.pem"
SSL_KEY_PATH="/etc/ssl/private/privkey.pem"

echo "Checking for SSL certificates..."
if [ ! -f "$SSL_CERT_PATH" ] || [ ! -f "$SSL_KEY_PATH" ]; then
    echo "=========================================="
    echo "SSL certificates not found - HTTPS will not be available"
    echo "=========================================="
    echo ""
    echo "To enable HTTPS, mount your certificates to:"
    echo "  - $SSL_CERT_PATH"
    echo "  - $SSL_KEY_PATH"
    echo ""
    echo "Example for docker-compose.yml:"
    echo "  volumes:"
    echo "    - /path/to/fullchain.pem:/etc/ssl/certs/fullchain.pem:ro"
    echo "    - /path/to/privkey.pem:/etc/ssl/private/privkey.pem:ro"
    echo ""
    echo "HTTP server will be available on port 80"
    echo "=========================================="
else
    echo "=========================================="
    echo "SSL certificates found - HTTPS enabled"
    echo "=========================================="
    echo "Certificate: $SSL_CERT_PATH"
    echo "Private key: $SSL_KEY_PATH"
    echo "HTTPS server will be available on port 443"
    echo "=========================================="
fi

# --- Lua Module Setup ---
echo "Setting up Lua support for nginx..."
LUA_SCRIPTS_DIR="/etc/nginx/lua"
mkdir -p "$LUA_SCRIPTS_DIR"
mkdir -p "$LUA_SCRIPTS_DIR/virtual_mappings"

# Copy Lua scripts from the docker/lua directory (standalone files, not heredocs)
LUA_SOURCE_DIR="/app/docker/lua"
cp "$LUA_SOURCE_DIR/capture_body.lua" "$LUA_SCRIPTS_DIR/capture_body.lua"
cp "$LUA_SOURCE_DIR/virtual_router.lua" "$LUA_SCRIPTS_DIR/virtual_router.lua"

cp "$LUA_SOURCE_DIR/emit_metrics.lua" "$LUA_SCRIPTS_DIR/emit_metrics.lua"
cp "$LUA_SOURCE_DIR/flush_metrics.lua" "$LUA_SCRIPTS_DIR/flush_metrics.lua"

echo "Lua scripts copied from $LUA_SOURCE_DIR to $LUA_SCRIPTS_DIR."

# --- Nginx Configuration ---
echo "Preparing Nginx configuration..."

# Pass environment variables through to Lua workers (nginx strips them by default)
for envvar in METRICS_API_KEY METRICS_SERVICE_URL; do
    grep -q "^env ${envvar};" /etc/nginx/nginx.conf 2>/dev/null || \
        sed -i "1i env ${envvar};" /etc/nginx/nginx.conf
done

# Raise main-context error_log to 'warn' so Lua init_worker/timer messages
# (e.g. flush_metrics.lua startup confirmation and connection errors) are visible.
# The default nginx.conf ships with 'error' level which suppresses WARN/INFO.
sed -i 's|error_log /var/log/nginx/error.log;|error_log /var/log/nginx/error.log warn;|' /etc/nginx/nginx.conf

# Remove default nginx site to prevent conflicts with our config
echo "Removing default nginx site configuration..."
rm -f /etc/nginx/sites-enabled/default
rm -f /etc/nginx/sites-available/default

# Template paths matching REGISTRY_CONSTANTS in registry/constants.py
NGINX_TEMPLATE_HTTP_ONLY="/app/docker/nginx_rev_proxy_http_only.conf"
NGINX_TEMPLATE_HTTP_AND_HTTPS="/app/docker/nginx_rev_proxy_http_and_https.conf"
NGINX_CONFIG_PATH="/etc/nginx/conf.d/nginx_rev_proxy.conf"

# Check if SSL certificates exist and use appropriate config
if [ ! -f "$SSL_CERT_PATH" ] || [ ! -f "$SSL_KEY_PATH" ]; then
    echo "Using HTTP-only Nginx configuration (no SSL certificates)..."
    cp "$NGINX_TEMPLATE_HTTP_ONLY" "$NGINX_CONFIG_PATH"
    echo "HTTP-only Nginx configuration installed."
else
    echo "Using HTTP + HTTPS Nginx configuration (SSL certificates found)..."
    cp "$NGINX_TEMPLATE_HTTP_AND_HTTPS" "$NGINX_CONFIG_PATH"
    echo "HTTP + HTTPS Nginx configuration installed."
fi

# --- Embeddings Configuration ---
# Get embeddings configuration from environment or use defaults
EMBEDDINGS_PROVIDER="${EMBEDDINGS_PROVIDER:-sentence-transformers}"
EMBEDDINGS_MODEL_NAME="${EMBEDDINGS_MODEL_NAME:-all-MiniLM-L6-v2}"
EMBEDDINGS_MODEL_DIMENSIONS="${EMBEDDINGS_MODEL_DIMENSIONS:-384}"

echo "Embeddings Configuration:"
echo "  Provider: $EMBEDDINGS_PROVIDER"
echo "  Model: $EMBEDDINGS_MODEL_NAME"
echo "  Dimensions: $EMBEDDINGS_MODEL_DIMENSIONS"

# Only check for local model if using sentence-transformers
if [ "$EMBEDDINGS_PROVIDER" = "sentence-transformers" ]; then
    EMBEDDINGS_MODEL_DIR="/app/registry/models/$EMBEDDINGS_MODEL_NAME"

    echo "Checking for sentence-transformers model..."
    if [ ! -d "$EMBEDDINGS_MODEL_DIR" ] || [ -z "$(ls -A "$EMBEDDINGS_MODEL_DIR")" ]; then
        echo "=========================================="
        echo "WARNING: Embeddings model not found!"
        echo "=========================================="
        echo ""
        echo "The registry requires the sentence-transformers model to function properly."
        echo "Please download the model to: $EMBEDDINGS_MODEL_DIR"
        echo ""
        echo "Run this command to download the model:"
        echo "  docker run --rm -v \$(pwd)/models:/models huggingface/transformers-pytorch-cpu python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/$EMBEDDINGS_MODEL_NAME').save('/models/$EMBEDDINGS_MODEL_NAME')\""
        echo ""
        echo "Or see the README for alternative download methods."
        echo "=========================================="
    else
        echo "Embeddings model found at $EMBEDDINGS_MODEL_DIR"
    fi
elif [ "$EMBEDDINGS_PROVIDER" = "litellm" ]; then
    echo "Using LiteLLM provider - no local model download required"
    echo "Model: $EMBEDDINGS_MODEL_NAME"
    if [[ "$EMBEDDINGS_MODEL_NAME" == bedrock/* ]]; then
        echo "Bedrock model will use AWS credential chain for authentication"
    elif [ ! -z "$EMBEDDINGS_API_KEY" ]; then
        echo "API key configured for cloud embeddings"
    else
        echo "WARNING: No EMBEDDINGS_API_KEY set for cloud provider"
    fi
fi

# --- Environment Variable Substitution for MCP Server Auth Tokens ---
echo "Processing MCP Server configuration files..."
for i in $(seq 1 99); do
    env_var_name="MCP_SERVER${i}_AUTH_TOKEN"
    env_var_value=$(eval echo \$$env_var_name)
    
    if [ ! -z "$env_var_value" ]; then
        echo "Found $env_var_name, substituting in server JSON files..."
        # Replace the literal environment variable name with its value in all JSON files
        find /app/registry/servers -name "*.json" -type f -exec sed -i "s|$env_var_name|$env_var_value|g" {} \;
    fi
done
echo "MCP Server configuration processing completed."

# --- Start Background Services ---
# Export embeddings configuration for the registry service
export EMBEDDINGS_PROVIDER=$EMBEDDINGS_PROVIDER
export EMBEDDINGS_MODEL_NAME=$EMBEDDINGS_MODEL_NAME
export EMBEDDINGS_MODEL_DIMENSIONS=$EMBEDDINGS_MODEL_DIMENSIONS

echo "Starting MCP Registry in the background..."
cd /app
source /app/.venv/bin/activate
uvicorn registry.main:app --host 0.0.0.0 --port 7860 --proxy-headers --forwarded-allow-ips='*' &
echo "MCP Registry started."

# Wait for nginx config to be generated (check that placeholders are replaced)
echo "Waiting for nginx configuration to be generated..."
WAIT_TIME=0
MAX_WAIT=120
while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if [ -f "/etc/nginx/conf.d/nginx_rev_proxy.conf" ]; then
        # Check if placeholders have been replaced
        if ! grep -q "{{ADDITIONAL_SERVER_NAMES}}" "/etc/nginx/conf.d/nginx_rev_proxy.conf" && \
           ! grep -q "{{ANTHROPIC_API_VERSION}}" "/etc/nginx/conf.d/nginx_rev_proxy.conf" && \
           ! grep -q "{{LOCATION_BLOCKS}}" "/etc/nginx/conf.d/nginx_rev_proxy.conf" && \
           ! grep -q "{{VIRTUAL_SERVER_BLOCKS}}" "/etc/nginx/conf.d/nginx_rev_proxy.conf"; then
            echo "Nginx configuration generated successfully"
            break
        fi
    fi
    sleep 2
    WAIT_TIME=$((WAIT_TIME + 2))
done

if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    echo "WARNING: Timeout waiting for nginx configuration. Starting nginx anyway..."
fi

# Resolve METRICS_SERVICE_URL hostname to IPv4 before nginx starts.
# Lua cosockets use the nginx resolver (VPC DNS 169.254.169.253), which cannot
# resolve Service Connect names (only the Envoy sidecar can).  By substituting
# the hostname with its IPv4 Service Connect VIP (127.255.0.x) in the env var,
# flush_metrics.lua connects directly to the IP, bypassing DNS entirely.
if [ -n "$METRICS_SERVICE_URL" ]; then
    metrics_host=$(echo "$METRICS_SERVICE_URL" | sed 's|http://||;s|:.*||')
    if ! echo "$metrics_host" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$'; then
        resolved=$(getent ahostsv4 "$metrics_host" 2>/dev/null | head -1 | awk '{print $1}')
        if [ -n "$resolved" ]; then
            export METRICS_SERVICE_URL=$(echo "$METRICS_SERVICE_URL" | sed "s|$metrics_host|$resolved|")
            echo "Resolved METRICS_SERVICE_URL: $metrics_host -> $resolved ($METRICS_SERVICE_URL)"
        else
            echo "WARNING: Could not resolve $metrics_host to IPv4 -- metrics flush may fail"
        fi
    fi
fi

# Add FQDN aliases for Service Connect entries in /etc/hosts.
# Service Connect only registers short names (e.g., "auth-server"), but servers
# may be registered with Cloud Map FQDNs (e.g., "auth-server.mcp-gateway.local").
# The Python health checker resolves proxy_pass_url hostnames via system DNS,
# which only finds /etc/hosts entries.  Adding FQDN aliases ensures both short
# names and FQDNs resolve to the IPv4 Service Connect VIP.
# Gated on SERVICE_CONNECT_NAMESPACE -- only set in ECS Terraform deployments.
if [ -n "${SERVICE_CONNECT_NAMESPACE:-}" ]; then
    fqdn_count=0
    grep '^127\.255\.0\.' /etc/hosts | while read -r ip name _rest; do
        echo "$ip ${name}.${SERVICE_CONNECT_NAMESPACE}" >> /etc/hosts
        fqdn_count=$((fqdn_count + 1))
    done
    echo "Added FQDN aliases for Service Connect entries (namespace: ${SERVICE_CONNECT_NAMESPACE})"
fi

echo "Starting Nginx..."
nginx

echo "Registry service fully started. Keeping container alive..."
# Keep the container running indefinitely
tail -f /dev/null 
