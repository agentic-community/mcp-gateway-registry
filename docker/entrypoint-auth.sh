#!/usr/bin/env bash

set -euo pipefail

if [[ -f "/app/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "/app/.venv/bin/activate"
fi

if [[ "${ENFORCEAI_AUTO_BOOTSTRAP:-false}" == "true" ]]; then
  state_dir="${ENFORCEAI_STATE_DIR:-}"
  if [[ -z "$state_dir" ]]; then
    echo "ENFORCEAI_AUTO_BOOTSTRAP=true requires ENFORCEAI_STATE_DIR to be set" >&2
    exit 2
  fi

  args=(
    "--state-dir" "$state_dir"
    "--active-kid" "${ENFORCEAI_GATEWAY_ACTIVE_KID:-kid-prod-1}"
    "--gateway-issuer" "${ENFORCEAI_GATEWAY_ISSUER:-enforceai-gateway}"
  )

  if [[ -n "${ENFORCEAI_BOOTSTRAP_USER_ID:-}" ]]; then
    args+=("--bootstrap-user-id" "$ENFORCEAI_BOOTSTRAP_USER_ID")
  fi

  if [[ -n "${ENFORCEAI_BOOTSTRAP_AGENT_ID:-}" ]]; then
    args+=("--bootstrap-agent-id" "$ENFORCEAI_BOOTSTRAP_AGENT_ID")
  fi

  python -m auth_server.enforceai.bootstrap "${args[@]}"
fi

exec uvicorn auth_server.server:app --host 0.0.0.0 --port 8888

