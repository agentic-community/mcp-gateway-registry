#!/usr/bin/env bash
#
# prepare-log-dirs.sh (Issue #987)
#
# Creates and chowns the host log directory used by AI Registry containers.
# The registry, auth-server, and mcpgw containers bind-mount
# /var/log/containers/ai-registry/ from the host so customer Splunk
# forwarders can ingest the .log files directly.
#
# Containers run as uid 1000 (appuser), so the host directory must be owned
# by 1000:1000 for the non-root container user to write into it.
#
# Safe to run multiple times. Idempotent.

set -euo pipefail

LOG_BASE="${APP_LOG_DIR:-/var/log/containers/ai-registry}"
OWNER_UID="${LOG_DIR_OWNER_UID:-1000}"
OWNER_GID="${LOG_DIR_OWNER_GID:-1000}"
DIR_MODE="${LOG_DIR_MODE:-0750}"

if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

echo "Preparing host log directory for AI Registry..."
echo "  Path:  ${LOG_BASE}"
echo "  Owner: ${OWNER_UID}:${OWNER_GID}"
echo "  Mode:  ${DIR_MODE}"

if [ ! -d "${LOG_BASE}" ]; then
    echo "Creating ${LOG_BASE}"
    ${SUDO} mkdir -p "${LOG_BASE}"
fi

${SUDO} chown -R "${OWNER_UID}:${OWNER_GID}" "${LOG_BASE}"
${SUDO} chmod "${DIR_MODE}" "${LOG_BASE}"

echo "OK: ${LOG_BASE} prepared"
ls -ld "${LOG_BASE}"
