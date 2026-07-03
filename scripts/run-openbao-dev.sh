#!/bin/bash
#
# Start (or stop) a local dev OpenBao for the egress-credential-vault tests.
#
#   scripts/run-openbao-dev.sh start   # launch on :8200, KV v2 at secret/
#   scripts/run-openbao-dev.sh stop    # remove the container
#
# Then run the integration suite:
#   export OPENBAO_TEST_ADDR=http://127.0.0.1:8200
#   export OPENBAO_TEST_TOKEN=dev-root-token
#   uv run pytest tests/integration/test_openbao_secret_store.py -v
#
# Dev mode is in-memory and unsealed with a known root token. DEV ONLY.

set -e

CONTAINER_NAME="mcp-openbao-dev"
IMAGE="openbao/openbao:latest"
PORT="8200"
ROOT_TOKEN="dev-root-token"

action="${1:-start}"

case "$action" in
  start)
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
      echo "OpenBao dev container already running on :${PORT}"
      exit 0
    fi
    echo "Starting OpenBao dev container on :${PORT}..."
    docker run -d --rm --name "$CONTAINER_NAME" \
      -p "${PORT}:8200" \
      -e "BAO_DEV_ROOT_TOKEN_ID=${ROOT_TOKEN}" \
      -e "BAO_DEV_LISTEN_ADDRESS=0.0.0.0:8200" \
      "$IMAGE" server -dev >/dev/null

    echo "Waiting for OpenBao to become healthy..."
    for _ in $(seq 1 30); do
      code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 \
        "http://127.0.0.1:${PORT}/v1/sys/health" 2>/dev/null || echo "000")
      if [ "$code" != "000" ]; then
        echo "OpenBao is up (health ${code})."
        echo
        echo "  export OPENBAO_TEST_ADDR=http://127.0.0.1:${PORT}"
        echo "  export OPENBAO_TEST_TOKEN=${ROOT_TOKEN}"
        exit 0
      fi
      sleep 1
    done
    echo "OpenBao did not become healthy in time." >&2
    exit 1
    ;;
  stop)
    echo "Stopping OpenBao dev container..."
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || echo "(not running)"
    ;;
  *)
    echo "Usage: $0 {start|stop}" >&2
    exit 1
    ;;
esac
