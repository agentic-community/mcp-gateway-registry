#!/bin/bash
# Shared helpers for deploy.sh / post-deploy.sh.
# Source with: source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

# Disable AWS CLI pager (table output otherwise pipes into `less`).
export AWS_PAGER=""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

_log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
_log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
_log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
_log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# _timed "<success-label>" <command...> — run <command...>, log "<label> in Xm Ys" on success.
_timed() {
  local label="$1"; shift
  local start; start=$(date +%s)
  "$@"
  local elapsed=$(( $(date +%s) - start ))
  _log_success "$label in $((elapsed / 60))m $((elapsed % 60))s"
}

# _retry "<label>" <max_attempts> <sleep_seconds> <function_name>
_retry() {
  local label="$1" max="$2" delay="$3" fn="$4"
  local attempt=0
  while [ $attempt -lt "$max" ]; do
    if "$fn"; then return 0; fi
    attempt=$((attempt + 1))
    if [ $attempt -lt "$max" ]; then
      _log_warn "$label failed (attempt $attempt/$max). Retrying in ${delay}s..."
      sleep "$delay"
    fi
  done
  return 1
}
