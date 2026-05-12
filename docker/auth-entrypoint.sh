#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "Starting Auth Server Setup..."

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

echo "Starting Auth Server (host=$BIND_HOST)..."
cd /app
source .venv/bin/activate
exec uvicorn server:app --host "$BIND_HOST" --port 8888 --proxy-headers --forwarded-allow-ips='*'
