#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "Starting Auth Server Setup..."

# SECRET_KEY is required. The application also enforces this at startup, but
# checking here surfaces the misconfiguration as a shell error instead of a
# Python traceback. Must match across all auth_server and registry replicas
# or session cookies will fail with BadSignature.
if [ -z "$SECRET_KEY" ]; then
    echo "ERROR: SECRET_KEY environment variable is required but not set." >&2
    echo "  Set it to a value at least 32 bytes long, identical across all" >&2
    echo "  auth_server and registry replicas. Generate one with:" >&2
    echo "    python3 -c 'import secrets; print(secrets.token_urlsafe(32))'" >&2
    exit 1
fi

# --- DocumentDB CA Bundle Download ---
if [[ "${DOCUMENTDB_HOST}" == *"docdb-elastic.amazonaws.com"* ]]; then
    echo "Detected DocumentDB Elastic cluster"
    echo "Downloading DocumentDB Elastic CA bundle..."
    CA_BUNDLE_URL="https://www.amazontrust.com/repository/SFSRootCAG2.pem"
    CA_BUNDLE_PATH="/app/certs/global-bundle.pem"
    if [ ! -f "$CA_BUNDLE_PATH" ]; then
        curl -fsSL "$CA_BUNDLE_URL" -o "$CA_BUNDLE_PATH"
        echo "DocumentDB Elastic CA bundle (SFSRootCAG2.pem) downloaded successfully to $CA_BUNDLE_PATH"
    fi
elif [[ "${DOCUMENTDB_HOST}" == *"docdb.amazonaws.com"* ]]; then
    echo "Detected regular DocumentDB cluster"
    echo "Downloading regular DocumentDB CA bundle..."
    CA_BUNDLE_URL="https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem"
    CA_BUNDLE_PATH="/app/certs/global-bundle.pem"
    if [ ! -f "$CA_BUNDLE_PATH" ]; then
        curl -fsSL "$CA_BUNDLE_URL" -o "$CA_BUNDLE_PATH"
        echo "DocumentDB CA bundle (global-bundle.pem) downloaded successfully to $CA_BUNDLE_PATH"
    fi
else
    echo "No DocumentDB host detected or DOCUMENTDB_HOST is empty - skipping CA bundle download"
fi

# --- Wait for MongoDB ---
if [ -n "$MONGODB_CONNECTION_STRING" ] || [ -n "$DOCUMENTDB_HOST" ]; then
    if [ -n "$MONGODB_CONNECTION_STRING" ]; then
        echo "Waiting for MongoDB via connection string override..."
    else
        echo "Waiting for MongoDB replica set at ${DOCUMENTDB_HOST}:${DOCUMENTDB_PORT:-27017}..."
    fi
    source /app/.venv/bin/activate
    python3 -c "
import pymongo, os, re, time
from urllib.parse import urlsplit

override = os.getenv('MONGODB_CONNECTION_STRING', '')
if override:
    uri = override
    tls_options = {}
    skip_replset_check = True
    display_host = urlsplit(uri).hostname or '(override)'
else:
    host = os.getenv('DOCUMENTDB_HOST', 'mongodb')
    port = int(os.getenv('DOCUMENTDB_PORT', '27017'))
    user = os.getenv('DOCUMENTDB_USERNAME', '')
    pwd = os.getenv('DOCUMENTDB_PASSWORD', '')
    backend = os.getenv('STORAGE_BACKEND', 'mongodb-ce')
    use_tls = os.getenv('DOCUMENTDB_USE_TLS', 'true').lower() == 'true'
    ca_file = os.getenv('DOCUMENTDB_TLS_CA_FILE', '/app/certs/global-bundle.pem')
    auth = 'SCRAM-SHA-256' if backend == 'mongodb-ce' else 'SCRAM-SHA-1'
    if user and pwd:
        uri = f'mongodb://{user}:{pwd}@{host}:{port}/?authMechanism={auth}&authSource=admin'
    else:
        uri = f'mongodb://{host}:{port}/'
    tls_options = {}
    if use_tls:
        tls_options['tls'] = True
        tls_options['tlsCAFile'] = ca_file
    skip_replset_check = False
    display_host = f'{host}:{port}'

def _redact(msg):
    return re.sub(r'mongodb(?:\+srv)?://[^\s]*', '<redacted-uri>', str(msg))

while True:
    try:
        c = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000, **tls_options)
        c.admin.command('ping')
        if skip_replset_check:
            print(f'MongoDB is ready ({display_host})')
            c.close()
            break
        try:
            st = c.admin.command('replSetGetStatus')
            ready = [m for m in st['members'] if m['state'] in [1, 2]]
            total = len(st['members'])
            if st['ok'] == 1 and len(ready) == total:
                print(f'MongoDB replica set ready ({len(ready)}/{total} members)')
                c.close()
                break
            print(f'Waiting for replica set: {len(ready)}/{total} ready')
        except pymongo.errors.OperationFailure:
            # Standalone mode (no replica set) - ping succeeded so we're good
            print('MongoDB is ready (standalone mode)')
            c.close()
            break
    except Exception as e:
        print(f'MongoDB not ready yet: {_redact(e)}')
    time.sleep(5)
"
    deactivate
    echo "MongoDB is ready."
fi

# Default binds to all IPv4 interfaces inside the container. Operators who
# need IPv6 dual-stack can set BIND_HOST=:: — but see docs/TELEMETRY.md for
# the net.ipv6.bindv6only=0 host-side requirement.
BIND_HOST="${BIND_HOST:-0.0.0.0}"

# Internal listen port. Defaults to 8888 (Docker Compose, Helm). On ECS with
# Service Connect, set AUTH_LISTEN_PORT=18888 to avoid conflict with Envoy's
# outbound interceptor which binds 127.0.0.1:8888.
AUTH_LISTEN_PORT="${AUTH_LISTEN_PORT:-8888}"

echo "Starting Auth Server (host=$BIND_HOST, port=$AUTH_LISTEN_PORT)..."
cd /app
source .venv/bin/activate

# Free-threaded Python 3.14t trial (issue #1316). Report whether this is a
# free-threaded build and whether the GIL is actually disabled at runtime. On a
# free-threaded build, importing a C extension that did not declare support via
# Py_mod_gil silently re-enables the GIL for the whole process, which would
# invalidate the trial. Set AUTH_REQUIRE_FREE_THREADING=true to fail startup if
# the GIL is unexpectedly re-enabled on a free-threaded build.
python3 - <<'PYEOF'
import sys
import sysconfig

gil_disabled_build = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
# sys._is_gil_enabled exists only on free-threaded builds.
is_gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()

print(
    f"Python {sys.version.split()[0]} | free-threaded build={gil_disabled_build} | "
    f"GIL enabled at runtime={is_gil_enabled}"
)

import os

strict = os.getenv("AUTH_REQUIRE_FREE_THREADING", "false").lower() == "true"
if gil_disabled_build and is_gil_enabled:
    msg = (
        "WARNING: running a free-threaded build but the GIL was re-enabled "
        "(likely a C extension without Py_mod_gil support was imported). "
        "The free-threading trial is not active. See issue #1316."
    )
    print(msg, file=sys.stderr)
    if strict:
        sys.exit(1)
PYEOF

if [ -n "${OTEL_EXPORTER_OTLP_ENDPOINT}" ] && command -v opentelemetry-instrument >/dev/null 2>&1; then
    echo "Using OTEL_EXPORTER_OTLP_ENDPOINT at ${OTEL_EXPORTER_OTLP_ENDPOINT}"
    UVICORN_CMD="opentelemetry-instrument uvicorn server:app --host $BIND_HOST --port $AUTH_LISTEN_PORT --proxy-headers --forwarded-allow-ips=*"
else
    echo "OTEL_EXPORTER_OTLP_ENDPOINT not found, not using OTEL"
    UVICORN_CMD="uvicorn server:app --host $BIND_HOST --port $AUTH_LISTEN_PORT --proxy-headers --forwarded-allow-ips=*"
fi
exec $UVICORN_CMD
