#!/bin/bash

# Initialize DocumentDB collections and indexes for MCP Gateway Registry
# This script downloads the CA bundle (if needed) and runs the Python initialization script

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
CA_BUNDLE_URL="https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem"
CA_BUNDLE_FILE="${DOCUMENTDB_TLS_CA_FILE:-global-bundle.pem}"
CA_BUNDLE_PATH="${PARENT_DIR}/${CA_BUNDLE_FILE}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "DocumentDB Initialization Script"
echo "================================="
echo ""

# Check if DocumentDB host is set (not required when MONGODB_CONNECTION_STRING is provided)
if [ -z "$DOCUMENTDB_HOST" ] && [ -z "${MONGODB_CONNECTION_STRING:-}" ]; then
    echo "${RED}Error: DOCUMENTDB_HOST environment variable is not set${NC}"
    echo ""
    echo "Please set the required environment variables:"
    echo "  export DOCUMENTDB_HOST=your-cluster.docdb.amazonaws.com"
    echo "  export DOCUMENTDB_USERNAME=admin"
    echo "  export DOCUMENTDB_PASSWORD=yourpassword"
    echo ""
    echo "Or provide a full connection URI:"
    echo "  export MONGODB_CONNECTION_STRING='mongodb+srv://user:pass@cluster.mongodb.net/db'"
    echo ""
    echo "Or use command-line arguments:"
    echo "  $0 --host your-cluster.docdb.amazonaws.com --username admin --password yourpassword"
    exit 1
fi

# Download CA bundle if it doesn't exist and TLS is enabled
USE_TLS="${DOCUMENTDB_USE_TLS:-true}"
if [ "$USE_TLS" = "true" ] && [ ! -f "$CA_BUNDLE_PATH" ]; then
    echo "${YELLOW}TLS is enabled but CA bundle not found${NC}"
    echo "Downloading AWS DocumentDB CA bundle..."
    echo "Source: ${CA_BUNDLE_URL}"
    echo "Destination: ${CA_BUNDLE_PATH}"
    echo ""

    if command -v wget &> /dev/null; then
        wget -O "$CA_BUNDLE_PATH" "$CA_BUNDLE_URL"
    elif command -v curl &> /dev/null; then
        curl -o "$CA_BUNDLE_PATH" "$CA_BUNDLE_URL"
    else
        echo "${RED}Error: Neither wget nor curl is available. Please install one of them.${NC}"
        exit 1
    fi

    if [ -f "$CA_BUNDLE_PATH" ]; then
        FILE_SIZE=$(stat -f%z "$CA_BUNDLE_PATH" 2>/dev/null || stat -c%s "$CA_BUNDLE_PATH" 2>/dev/null)
        if [ "$FILE_SIZE" -gt 0 ]; then
            echo "${GREEN}Successfully downloaded CA bundle (${FILE_SIZE} bytes)${NC}"
            echo ""
        else
            echo "${RED}Error: Downloaded file is empty${NC}"
            rm -f "$CA_BUNDLE_PATH"
            exit 1
        fi
    else
        echo "${RED}Error: Failed to download CA bundle${NC}"
        exit 1
    fi
elif [ "$USE_TLS" = "true" ]; then
    echo "${GREEN}CA bundle found at: ${CA_BUNDLE_PATH}${NC}"
    echo ""
fi

# Set up environment variables for the Python script
export DOCUMENTDB_TLS_CA_FILE="$CA_BUNDLE_PATH"

echo "Environment Configuration:"
if [ -n "${MONGODB_CONNECTION_STRING:-}" ]; then
    echo "  MONGODB_CONNECTION_STRING: (override set, host/port/auth derived from URI)"
else
    echo "  DOCUMENTDB_HOST: ${DOCUMENTDB_HOST}"
    echo "  DOCUMENTDB_PORT: ${DOCUMENTDB_PORT:-27017}"
    echo "  DOCUMENTDB_USE_IAM: ${DOCUMENTDB_USE_IAM:-false}"
    if [ -n "$DOCUMENTDB_USERNAME" ]; then
        echo "  DOCUMENTDB_USERNAME: ${DOCUMENTDB_USERNAME}"
    fi
fi
echo "  DOCUMENTDB_DATABASE: ${DOCUMENTDB_DATABASE:-mcp_registry}"
echo "  DOCUMENTDB_NAMESPACE: ${DOCUMENTDB_NAMESPACE:-default}"
echo "  DOCUMENTDB_USE_TLS: ${USE_TLS}"

echo ""
echo "Step 1: Creating collections and indexes..."
echo ""

# Run the Python initialization script
cd "$PARENT_DIR"

if command -v uv &> /dev/null; then
    PYTHON_CMD="uv run python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "${RED}Error: Neither uv nor python3 is available${NC}"
    exit 1
fi

# Create collections and indexes
$PYTHON_CMD scripts/init-documentdb-indexes.py "$@"

echo ""
echo "${GREEN}Collections and indexes created successfully!${NC}"
echo ""

echo ""
echo "${GREEN}DocumentDB initialization complete!${NC}"
