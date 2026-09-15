#!/bin/bash
# init-pingfederate.sh
#
# Configures a PingFederate instance (baseline profile) for MCP Gateway Registry.
# Idempotent: safe to run multiple times.
#
# Prerequisites:
#   - PingFederate container running and healthy (baseline server profile)
#   - .env file with PINGFEDERATE_* variables set
#
# Required environment variables (no defaults; the script fails closed if any
# is unset, empty, or set to a known-weak value):
#   - PF_ADMIN_PASS                 admin-API password of the PingFederate console
#   - PINGFEDERATE_CLIENT_SECRET    OAuth client secret for the mcp-gateway client
#   - PF_REGISTRY_ADMIN_PASSWORD    password for the seeded 'admin' browser login
#
# Usage:
#   bash pingfederate/setup/init-pingfederate.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    source "$REPO_ROOT/.env"
    set +a
fi

PF_ADMIN_URL="${PF_ADMIN_URL:-https://localhost:9999}"
PF_ADMIN_USER="${PF_ADMIN_USER:-administrator}"
PF_ADMIN_PASS="${PF_ADMIN_PASS:-}"
PF_EXTERNAL_URL="${PINGFEDERATE_EXTERNAL_URL:-https://localhost:9031}"
PF_CLIENT_ID="${PINGFEDERATE_CLIENT_ID:-mcp-gateway}"
PF_CLIENT_SECRET="${PINGFEDERATE_CLIENT_SECRET:-}"
PF_REGISTRY_ADMIN_PASSWORD="${PF_REGISTRY_ADMIN_PASSWORD:-}"
PF_CALLBACK_URL="${PF_EXTERNAL_URL}/oauth2/callback/pingfederate"

# ---------------------------------------------------------------------------
# Fail-closed credential validation.
#
# Every credential this bootstrap consumes MUST be supplied by the operator via
# the environment with no shipped default. A value that is unset, empty, or a
# known-weak literal aborts the run before any PingFederate object is created or
# any group mapping is seeded, so the deployment can never come up with default
# credentials.
# ---------------------------------------------------------------------------

# Known-weak values live in one shared file so the pre-deploy preflight in
# build_and_run.sh and this post-deploy gate cannot drift. Missing file is fatal:
# silently validating against an empty denylist would be a fail-open bootstrap.
WEAK_CREDENTIALS_LIB="$REPO_ROOT/scripts/weak-credentials.sh"
if [ ! -f "$WEAK_CREDENTIALS_LIB" ]; then
    echo "ERROR: missing ${WEAK_CREDENTIALS_LIB}." >&2
    echo "       Run this script from a full checkout; it must not be copied alone." >&2
    exit 1
fi
# shellcheck source=scripts/weak-credentials.sh
. "$WEAK_CREDENTIALS_LIB"

require_strong_secret() {
    # Usage: require_strong_secret <ENV_VAR_NAME>
    # Reads the named variable, rejects unset/empty/whitespace-only, a known-weak
    # literal, or a value shorter than the minimum length, then writes the
    # whitespace-trimmed value back to the variable so what was validated is
    # exactly what gets transmitted. Exits non-zero (fail closed) on any failure
    # so nothing is seeded.
    local name="$1"
    local value="${!name-}"
    local min_len="$WEAK_CREDENTIAL_MIN_LEN"

    # Trim leading/trailing whitespace so a padded weak value (e.g. "  changeme  ")
    # cannot slip past the denylist and so a whitespace-only value is treated as
    # empty rather than as a long "strong" secret.
    local stripped="$value"
    stripped="${stripped#"${stripped%%[![:space:]]*}"}"
    stripped="${stripped%"${stripped##*[![:space:]]}"}"

    if [ -z "$stripped" ]; then
        echo "ERROR: ${name} is required and must be set to a non-empty value." >&2
        echo "       Refusing to bootstrap PingFederate with a missing credential." >&2
        exit 1
    fi

    # The weak-value check runs BEFORE the length check so a known placeholder
    # produces the precise "known-weak" error even when it is also short.
    if is_weak_credential "$stripped"; then
        echo "ERROR: ${name} is set to a known-weak value; choose a strong secret." >&2
        echo "       Refusing to bootstrap PingFederate with a default credential." >&2
        exit 1
    fi

    if [ "${#stripped}" -lt "$min_len" ]; then
        echo "ERROR: ${name} must be at least ${min_len} characters." >&2
        exit 1
    fi

    printf -v "$name" '%s' "$stripped"
}

require_strong_secret "PF_ADMIN_PASS"
require_strong_secret "PINGFEDERATE_CLIENT_SECRET"
require_strong_secret "PF_REGISTRY_ADMIN_PASSWORD"
# PF_CLIENT_SECRET is the local alias used by the payload builder below; re-read
# it after validation so it carries the trimmed value.
PF_CLIENT_SECRET="$PINGFEDERATE_CLIENT_SECRET"

pf_api() {
    # Perform a PingFederate admin-API call. The response body is written to
    # stdout; a non-2xx HTTP status (or a transport failure) makes the function
    # return non-zero WITHOUT emitting a body. Because the script runs under
    # `set -e`, a failed write (e.g. a rejected password rotation) aborts the run
    # instead of being silently swallowed and reported as success -- so a legacy
    # weak credential can never survive behind a "done" message.
    local method="$1"
    local path="$2"
    local data="$3"
    # -S so curl still reports a transport failure while -s suppresses its progress
    # meter; its stderr is deliberately NOT discarded, otherwise a refused admin
    # port renders as a bare "HTTP 000" with no cause. Step 1 waits on the runtime
    # port (9031), not the admin port (9999), so a connection refusal here is a
    # realistic first failure.
    local args=(-ksS -w '\n%{http_code}' -u "${PF_ADMIN_USER}:${PF_ADMIN_PASS}"
        -H "X-XSRF-Header: PingFederate"
        -H "Content-Type: application/json"
        -X "$method"
        "${PF_ADMIN_URL}/pf-admin-api/v1${path}")
    if [ -n "$data" ]; then
        args+=(-d "$data")
    fi
    local response http_code body
    response="$(curl "${args[@]}")"
    http_code="${response##*$'\n'}"
    body="${response%$'\n'*}"
    if ! [[ "$http_code" =~ ^[0-9]{3}$ ]] || [ "$http_code" -lt 200 ] || [ "$http_code" -ge 300 ]; then
        echo "ERROR: PingFederate API ${method} ${path} failed (HTTP ${http_code:-000})." >&2
        # Echo the response body: PingFederate returns {resultId, message} on 4xx,
        # and the most likely failure here is the seeded password being rejected by
        # the server's own password policy, which is otherwise indistinguishable
        # from a malformed payload. Error responses echo no resource, so no
        # credential material is disclosed.
        if [ -n "$body" ]; then
            echo "       response: ${body}" >&2
        fi
        return 1
    fi
    printf '%s' "$body"
}

pf_exists() {
    local path="$1"
    local status
    status=$(curl -ks -u "${PF_ADMIN_USER}:${PF_ADMIN_PASS}" \
        -H "X-XSRF-Header: PingFederate" \
        -o /dev/null -w "%{http_code}" \
        "${PF_ADMIN_URL}/pf-admin-api/v1${path}" 2>/dev/null)
    [ "$status" = "200" ]
}

echo "=== MCP Gateway Registry: PingFederate Initialization (baseline profile) ==="
echo ""
echo "Admin URL:    $PF_ADMIN_URL"
echo "External URL: $PF_EXTERNAL_URL"
echo "Client ID:    $PF_CLIENT_ID"
echo "Callback:     $PF_CALLBACK_URL"
echo ""

# Step 1: Wait for PingFederate
echo "[1/8] Waiting for PingFederate to be healthy..."
MAX_WAIT=300
ELAPSED=0
while ! curl -ksf "https://localhost:9031/pf/heartbeat.ping" > /dev/null 2>&1; do
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "ERROR: PingFederate did not become healthy within ${MAX_WAIT}s"
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "  Waiting... (${ELAPSED}s)"
done
echo "  PingFederate is healthy."

# Step 2: Extract TLS cert and create CA bundle
echo "[2/8] Extracting PingFederate TLS certificate..."
echo | openssl s_client -connect localhost:9031 -servername localhost 2>/dev/null \
    | openssl x509 > /tmp/pf-cert.pem 2>/dev/null
cat /etc/ssl/certs/ca-certificates.crt /tmp/pf-cert.pem \
    > "$SCRIPT_DIR/pingfederate-ca-bundle.pem"
echo "  CA bundle written to pingfederate/setup/pingfederate-ca-bundle.pem"

# Step 3: Set base URL
echo "[3/8] Setting federation base URL to: $PF_EXTERNAL_URL"
SETTINGS=$(pf_api GET "/serverSettings")
UPDATED=$(echo "$SETTINGS" | python3 -c "
import json, sys
d = json.load(sys.stdin)
d['federationInfo']['baseUrl'] = '${PF_EXTERNAL_URL}'
print(json.dumps(d))
")
pf_api PUT "/serverSettings" "$UPDATED" > /dev/null
echo "  Done."

# Step 4: Add groups scope
echo "[4/8] Configuring OAuth scopes..."
AUTH_SETTINGS=$(pf_api GET "/oauth/authServerSettings")
UPDATED=$(echo "$AUTH_SETTINGS" | python3 -c "
import json, sys
d = json.load(sys.stdin)
names = {s['name'] for s in d.get('scopes', [])}
if 'groups' not in names:
    d['scopes'].append({'name': 'groups', 'description': 'Groups', 'dynamic': False})
print(json.dumps(d))
")
pf_api PUT "/oauth/authServerSettings" "$UPDATED" > /dev/null
echo "  Done (groups scope added)."

# Step 5: Seed the registry admin login into the simple PCV and switch HTMLFormPD
# to use it. Only a single 'admin' login is seeded, and its password comes from
# the required PF_REGISTRY_ADMIN_PASSWORD env var (validated above). The PCV row
# does NOT relax password requirements: PingFederate enforces its own password
# policy on the seeded credential.
echo "[5/8] Seeding admin login and configuring adapter..."
PCV=$(pf_api GET "/passwordCredentialValidators/simple")
UPDATED=$(echo "$PCV" | PF_SEED_ADMIN_PW="$PF_REGISTRY_ADMIN_PASSWORD" python3 -c "
import json, os, sys
d = json.load(sys.stdin)
users_table = next(t for t in d['configuration']['tables'] if t['name'] == 'Users')
admin_pw = os.environ['PF_SEED_ADMIN_PW']

def _username(row):
    return next((f['value'] for f in row['fields'] if f['name'] == 'Username'), None)

# Drop any legacy demo rows (e.g. a previously-seeded weak 'testuser').
users_table['rows'] = [r for r in users_table['rows'] if _username(r) != 'testuser']

# Upsert the 'admin' row with the operator-supplied password and without the
# policy-relaxing flag. Overwriting rather than skipping rotates a
# pre-existing weak credential (e.g. a legacy admin/admin123) on every run, so a
# previously-bootstrapped instance is remediated the next time this script runs.
admin_row = {'fields': [
    {'name': 'Username', 'value': 'admin'},
    {'name': 'Password', 'value': admin_pw},
    {'name': 'Confirm Password', 'value': admin_pw},
]}
for i, row in enumerate(users_table['rows']):
    if _username(row) == 'admin':
        users_table['rows'][i] = admin_row
        break
else:
    users_table['rows'].append(admin_row)
print(json.dumps(d))
")
pf_api PUT "/passwordCredentialValidators/simple" "$UPDATED" > /dev/null
echo "  Seeded browser login: admin (password from PF_REGISTRY_ADMIN_PASSWORD)"

# Switch HTMLFormPD adapter to use simple PCV (baseline uses pingdirectory)
ADAPTER=$(pf_api GET "/idp/adapters/HTMLFormPD")
UPDATED=$(echo "$ADAPTER" | python3 -c "
import json, sys
d = json.load(sys.stdin)
for table in d['configuration']['tables']:
    if table['name'] == 'Credential Validators':
        for row in table['rows']:
            for f in row['fields']:
                if f['name'] == 'Password Credential Validator Instance':
                    f['value'] = 'simple'
print(json.dumps(d))
")
pf_api PUT "/idp/adapters/HTMLFormPD" "$UPDATED" > /dev/null
echo "  HTMLFormPD adapter switched to simple PCV."

# Step 6: Create OAuth client
echo "[6/8] Creating OAuth client: $PF_CLIENT_ID"
# Build the client payload with a real JSON serializer, taking the secret from
# the environment. This keeps the stored secret byte-identical to the value that
# require_strong_secret validated (a raw JSON-source interpolation would let a
# JSON-escaped value decode to a weak secret PingFederate-side, and would break
# on any secret containing a quote or backslash).
CLIENT_PAYLOAD=$(
    PF_CLIENT_ID="$PF_CLIENT_ID" \
    PF_CLIENT_SECRET="$PF_CLIENT_SECRET" \
    PF_CALLBACK_URL="$PF_CALLBACK_URL" \
    python3 -c "
import json, os
print(json.dumps({
    'clientId': os.environ['PF_CLIENT_ID'],
    'name': 'MCP Gateway Registry',
    'clientAuth': {'type': 'SECRET', 'secret': os.environ['PF_CLIENT_SECRET']},
    'grantTypes': ['AUTHORIZATION_CODE', 'CLIENT_CREDENTIALS', 'REFRESH_TOKEN'],
    'redirectUris': [os.environ['PF_CALLBACK_URL']],
    'enabled': True,
    'defaultAccessTokenManagerRef': {'id': 'jwt'},
}))
"
)
if pf_exists "/oauth/clients/${PF_CLIENT_ID}"; then
    echo "  Already exists, updating..."
    pf_api PUT "/oauth/clients/${PF_CLIENT_ID}" "$CLIENT_PAYLOAD" > /dev/null
else
    pf_api POST "/oauth/clients" "$CLIENT_PAYLOAD" > /dev/null
fi
echo "  Done."

# Step 7: Wire auth policy + adapter mapping + access token mapping
echo "[7/8] Wiring authentication policy and token mappings..."

# Set HTMLFormPD as default auth source
pf_api PUT "/authenticationPolicies/default" '{
    "failIfNoSelection": false,
    "authnSelectionTrees": [],
    "defaultAuthenticationSources": [{"type": "IDP_ADAPTER", "sourceRef": {"id": "HTMLFormPD"}}],
    "trackedHttpParameters": []
}' > /dev/null
echo "  Default auth source: HTMLFormPD"

# Create IdP adapter grant mapping (if not exists)
EXISTING=$(pf_api GET "/oauth/idpAdapterMappings" | python3 -c "
import json, sys
d = json.load(sys.stdin)
ids = [m['id'] for m in d.get('items', [])]
print('yes' if 'HTMLFormPD' in ids else 'no')
" 2>/dev/null)
if [ "$EXISTING" != "yes" ]; then
    pf_api POST "/oauth/idpAdapterMappings" '{
        "id": "HTMLFormPD",
        "idpAdapterRef": {"id": "HTMLFormPD"},
        "attributeSources": [],
        "attributeContractFulfillment": {
            "USER_KEY": {"source": {"type": "ADAPTER"}, "value": "username"},
            "USER_NAME": {"source": {"type": "ADAPTER"}, "value": "username"}
        },
        "issuanceCriteria": {"conditionalCriteria": []}
    }' > /dev/null
    echo "  IdP adapter grant mapping created."
else
    echo "  IdP adapter grant mapping already exists."
fi

# Create access token mapping (HTMLFormPD -> jwt ATM)
EXISTING_ATM=$(pf_api GET "/oauth/accessTokenMappings" | python3 -c "
import json, sys
d = json.load(sys.stdin)
items = d if isinstance(d, list) else d.get('items', [])
found = any('HTMLFormPD' in m.get('id','') and 'jwt' in m.get('id','') for m in items)
print('yes' if found else 'no')
" 2>/dev/null)
if [ "$EXISTING_ATM" != "yes" ]; then
    pf_api POST "/oauth/accessTokenMappings" '{
        "context": {"type": "IDP_ADAPTER", "contextRef": {"id": "HTMLFormPD"}},
        "accessTokenManagerRef": {"id": "jwt"},
        "attributeSources": [],
        "attributeContractFulfillment": {
            "Username": {"source": {"type": "ADAPTER"}, "value": "username"},
            "OrgName": {"source": {"type": "TEXT"}, "value": "MCP-Gateway"}
        },
        "issuanceCriteria": {"conditionalCriteria": []}
    }' > /dev/null
    echo "  Access token mapping created."
else
    echo "  Access token mapping already exists."
fi

# Step 8: Seed registry's idp_user_groups collection so the seeded PingFederate
# admin login (whose JWT comes back with an empty `groups` claim) gets mapped to
# the registry-admins group. The auth-server enrichment looks up by username at
# JWT-validation time and uses these `groups` when the token's groups claim is
# empty (issue #1127).
#
# - admin -> registry-admins (full registry admin, matches scripts/registry-admins.json)
echo "[8/8] Seeding registry idp_user_groups for the admin login..."
MONGO_DB="${DOCUMENTDB_DATABASE:-mcp_registry}"
MONGO_CONTAINER="mcp-mongodb"
if docker ps --format "{{.Names}}" | grep -q "^${MONGO_CONTAINER}$"; then
    docker exec "${MONGO_CONTAINER}" mongosh --quiet --eval "
db = db.getSiblingDB('${MONGO_DB}');
const now = new Date();
db.idp_user_groups.deleteOne({username: 'testuser', created_by: 'init-pingfederate.sh'});
[
  {username: 'admin', groups: ['registry-admins'], email: null}
].forEach(function(u) {
    db.idp_user_groups.updateOne(
        {username: u.username},
        {\$set: {
            username: u.username,
            groups: u.groups,
            email: u.email,
            enabled: true,
            provider: 'pingfederate',
            created_by: 'init-pingfederate.sh',
            updated_at: now
        }, \$setOnInsert: {created_at: now}},
        {upsert: true}
    );
});
print('  idp_user_groups seeded for admin.');
" || {
    SEED_FAILED=1
    echo "  Warning: failed to seed idp_user_groups." >&2
}
else
    SEED_FAILED=1
    echo "  Warning: ${MONGO_CONTAINER} container not running; skipping idp_user_groups seed."
    echo "           Start the stack and re-run this script to seed."
fi

echo ""
if [ "${SEED_FAILED:-0}" = "1" ]; then
    # The IdP objects committed but the group mapping did not, so the seeded admin
    # would authenticate with an empty groups claim and then be denied everything.
    # Do not claim success: that "login works but nothing works" state is expensive
    # to diagnose precisely because the bootstrap reported completion.
    echo "=== PingFederate initialization INCOMPLETE ===" >&2
    echo "" >&2
    echo "PingFederate is configured and the 'admin' browser login is seeded, but" >&2
    echo "the registry idp_user_groups mapping was NOT written. That login will" >&2
    echo "authenticate with an empty groups claim and be denied every action." >&2
    echo "" >&2
    echo "Start the ${MONGO_CONTAINER} container and re-run this script." >&2
    exit 1
fi

echo "=== PingFederate initialization complete ==="
echo ""
echo "Browser login seeded: admin (password taken from PF_REGISTRY_ADMIN_PASSWORD)"
echo ""
echo "Registry group mappings:"
echo "  admin -> registry-admins"
echo ""
echo "IMPORTANT: Restart auth-server to pick up the CA bundle:"
echo "  docker compose up -d --build auth-server"
echo ""
echo "Then visit: ${PF_EXTERNAL_URL}"
