#!/usr/bin/env bash
# Preflight gate for the gateway generic-proxy regression suite.
#
# Run this BEFORE any test in tests/scripts/gateway-proxy-regression.md. It fails
# loudly and exits non-zero when a prerequisite is missing, so a run cannot start
# half-configured and produce a result nobody can trust.
#
# Usage (from the repository root):
#   ./tests/scripts/gateway-proxy-preflight.sh              # gate every section
#   ./tests/scripts/gateway-proxy-preflight.sh 3 4 5 6      # gate only these sections
#
# Exit codes: 0 all required prerequisites present. 1 at least one is missing.

set -uo pipefail

RED=$'\033[0;31m'; YEL=$'\033[0;33m'; GRN=$'\033[0;32m'; BLD=$'\033[1m'; NC=$'\033[0m'
FAILED=0
SKIPPED=()

die()  { printf '%s\n' "${RED}${BLD}FATAL${NC} ${RED}$1${NC}" >&2; FAILED=1; }
warn() { printf '%s\n' "${YEL}WARN ${NC} $1" >&2; }
ok()   { printf '%s\n' "${GRN}ok   ${NC} $1"; }
head2() { printf '\n%s\n' "${BLD}$1${NC}"; }

# --- where am I -------------------------------------------------------------
if [ ! -f pyproject.toml ] || [ ! -d registry ]; then
  die "run this from the repository root (pyproject.toml and registry/ must be here); every path in the suite is root-relative"
  exit 1
fi

SECTIONS=("$@")
want() {  # want <section-number> -> 0 if that section is in scope
  [ ${#SECTIONS[@]} -eq 0 ] && return 0
  local s; for s in "${SECTIONS[@]}"; do [ "$s" = "$1" ] && return 0; done
  return 1
}

# --- credential and key files ----------------------------------------------
# Every file the suite reads a secret from, with the sections that need it.
# Format: path|sections|what it is|how to get it
CRED_FILES=(
".token|all|gateway JWT (admin, group authorizing HTTP verbs)|generate from the registry UI sidebar, or ./api/get-m2m-token.sh"
".scratchpad/pr-1714/api-ninja|5 6|api-ninjas.com API key, one line|sign up at api-ninjas.com and write the key to this file"
".scratchpad/.oai|7|OpenAI API key, one line|platform.openai.com API keys"
".scratchpad/.bedrock|7|Amazon Bedrock long-term API key for us-east-2|Bedrock console, long-term API key"
)

head2 "CREDENTIAL FILES (secrets — never commit, never echo)"
for row in "${CRED_FILES[@]}"; do
  IFS='|' read -r path secs what how <<< "$row"
  in_scope=0
  if [ "$secs" = "all" ]; then in_scope=1; else
    for s in $secs; do want "$s" && in_scope=1; done
  fi
  if [ "$in_scope" -eq 0 ]; then
    printf '%s\n' "     ${path} — not needed for the requested sections"
    continue
  fi
  if [ ! -e "$path" ]; then
    die "MISSING CREDENTIAL FILE  ${path}
          needed by section(s): ${secs}
          contents: ${what}
          obtain:   ${how}"
    SKIPPED+=("$secs")
  elif [ ! -s "$path" ]; then
    die "EMPTY CREDENTIAL FILE    ${path} (needed by section(s) ${secs}: ${what})"
    SKIPPED+=("$secs")
  else
    ok "${path} present, $(wc -c < "$path" | tr -d ' ') bytes — ${what}"
  fi
done

# --- gateway token validity -------------------------------------------------
head2 "GATEWAY TOKEN"
if [ -s .token ]; then
  TOKEN_LEFT=$(python3 - <<'PY' 2>/dev/null
import base64, json, time
try:
    t = json.load(open('.token'))['tokens']['access_token']
    p = t.split('.')[1]; p += '=' * (-len(p) % 4)
    print(json.loads(base64.urlsafe_b64decode(p))['exp'] - int(time.time()))
except Exception:
    print('unreadable')
PY
)
  case "$TOKEN_LEFT" in
    unreadable) die ".token does not contain tokens.access_token as a decodable JWT" ;;
    -*)         die "GATEWAY TOKEN EXPIRED $(( ${TOKEN_LEFT#-} / 60 )) minutes ago — regenerate before running anything, or every test returns 401" ;;
    *)          if [ "$TOKEN_LEFT" -lt 900 ]; then
                  warn "gateway token expires in $(( TOKEN_LEFT / 60 )) minutes; a full run may outlive it"
                else
                  ok "gateway token valid for $(( TOKEN_LEFT / 60 )) more minutes"
                fi ;;
  esac
fi

# --- configuration ----------------------------------------------------------
head2 "CONFIGURATION (.env)"
if [ ! -f .env ]; then
  die "no .env at the repository root"
else
  getenv() { grep -m1 "^$1=" .env 2>/dev/null | cut -d= -f2- ; }
  [ "$(getenv GATEWAY_GENERIC_PROXY_ENABLED)" = "true" ] \
    && ok "GATEWAY_GENERIC_PROXY_ENABLED=true" \
    || die "GATEWAY_GENERIC_PROXY_ENABLED is not true — the feature ships off and every route 404s"
  [ "$(getenv DEPLOYMENT_MODE)" = "with-gateway" ] \
    && ok "DEPLOYMENT_MODE=with-gateway" \
    || die "DEPLOYMENT_MODE is not with-gateway — the generic proxy only renders in gateway mode"
  [ -n "$(getenv SECRET_KEY)" ] \
    && ok "SECRET_KEY set (derives the upstream-credential encryption key)" \
    || die "SECRET_KEY is empty — sections 5 and 6 cannot encrypt or decrypt a stored credential"
  if want 5 || want 6; then
    [ -n "$(getenv DOCUMENTDB_PASSWORD)" ] \
      && ok "DOCUMENTDB_PASSWORD set (sections 5 and 6 read Mongo directly)" \
      || die "DOCUMENTDB_PASSWORD is empty — sections 5 and 6 inspect stored ciphertext"
  fi
fi

# --- stack ------------------------------------------------------------------
head2 "STACK"
ORIGIN="${ORIGIN:-http://localhost}"
HEALTH=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$ORIGIN/health" 2>/dev/null)
[ "$HEALTH" = "200" ] \
  && ok "$ORIGIN/health -> 200" \
  || die "$ORIGIN/health -> ${HEALTH:-no response} — start the stack (docker compose up -d) before testing"

if command -v docker >/dev/null 2>&1; then
  # Match the literal text auth_server/server.py emits. An earlier invented
  # string ("generic-proxy feature DISABLED") matched nothing, so this gate
  # always reported ok and the latched-off case reached the suite as a 404.
  if docker compose logs auth-server 2>/dev/null | grep -qF "Generic proxy egress self-check FAILED"; then
    die "the auth-server logged 'Generic proxy egress self-check FAILED' — a cloud metadata IP is reachable from the container, so the feature latched off for the process and every route 404s. Set GATEWAY_EGRESS_SELFCHECK_ENABLED=false for local runs and restart."
  else
    ok "no 'Generic proxy egress self-check FAILED' line in the auth-server log"
  fi
fi

# --- tooling ----------------------------------------------------------------
head2 "TOOLING"
for bin in curl jq python3 docker; do
  command -v "$bin" >/dev/null 2>&1 && ok "$bin present" || die "$bin not on PATH — the suite needs it"
done

# --- verdict ----------------------------------------------------------------
printf '\n'
if [ "$FAILED" -ne 0 ]; then
  printf '%s\n' "${RED}${BLD}=============================================================${NC}"
  printf '%s\n' "${RED}${BLD} PREFLIGHT FAILED — DO NOT RUN THE SUITE${NC}"
  printf '%s\n' "${RED}${BLD} Fix every FATAL above. A partial run produces results${NC}"
  printf '%s\n' "${RED}${BLD} nobody can trust: a missing key looks like a proxy bug.${NC}"
  printf '%s\n' "${RED}${BLD}=============================================================${NC}"
  exit 1
fi
printf '%s\n' "${GRN}${BLD}PREFLIGHT PASSED — the suite may run.${NC}"
exit 0
