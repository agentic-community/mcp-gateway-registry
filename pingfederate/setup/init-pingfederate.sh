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
PF_ADMIN_PASS="${PF_ADMIN_PASS:-2FederateM0re}"
PF_EXTERNAL_URL="${PINGFEDERATE_EXTERNAL_URL:-https://localhost:9031}"
PF_CLIENT_ID="${PINGFEDERATE_CLIENT_ID:-mcp-gateway}"
PF_CLIENT_SECRET="${PINGFEDERATE_CLIENT_SECRET:-changeme}"
PF_CALLBACK_URL="${PF_EXTERNAL_URL}/oauth2/callback/pingfederate"

pf_api() {
    local method="$1"
    local path="$2"
    local data="$3"
    local args=(-ks -u "${PF_ADMIN_USER}:${PF_ADMIN_PASS}"
        -H "X-XSRF-Header: PingFederate"
        -H "Content-Type: application/json"
        -X "$method"
        "${PF_ADMIN_URL}/pf-admin-api/v1${path}")
    if [ -n "$data" ]; then
        args+=(-d "$data")
    fi
    curl "${args[@]}" 2>/dev/null
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

# Step 5: Add test users to simple PCV and switch HTMLFormPD to use it
echo "[5/8] Adding test users and configuring adapter..."
PCV=$(pf_api GET "/passwordCredentialValidators/simple")
UPDATED=$(echo "$PCV" | python3 -c "
import json, sys
d = json.load(sys.stdin)
users_table = next(t for t in d['configuration']['tables'] if t['name'] == 'Users')
existing = {f['value'] for row in users_table['rows'] for f in row['fields'] if f['name'] == 'Username'}
for user, pw in [('admin', 'admin123'), ('testuser', 'changeme')]:
    if user not in existing:
        users_table['rows'].append({'fields': [
            {'name': 'Username', 'value': user},
            {'name': 'Password', 'value': pw},
            {'name': 'Confirm Password', 'value': pw},
            {'name': 'Relax Password Requirements', 'value': 'true'}
        ]})
print(json.dumps(d))
")
pf_api PUT "/passwordCredentialValidators/simple" "$UPDATED" > /dev/null
echo "  Test users: admin/admin123, testuser/changeme"

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
if pf_exists "/oauth/clients/${PF_CLIENT_ID}"; then
    echo "  Already exists, updating..."
    pf_api PUT "/oauth/clients/${PF_CLIENT_ID}" "{
        \"clientId\": \"${PF_CLIENT_ID}\",
        \"name\": \"MCP Gateway Registry\",
        \"clientAuth\": {\"type\": \"SECRET\", \"secret\": \"${PF_CLIENT_SECRET}\"},
        \"grantTypes\": [\"AUTHORIZATION_CODE\", \"CLIENT_CREDENTIALS\", \"REFRESH_TOKEN\"],
        \"redirectUris\": [\"${PF_CALLBACK_URL}\"],
        \"enabled\": true,
        \"defaultAccessTokenManagerRef\": {\"id\": \"jwt\"}
    }" > /dev/null
else
    pf_api POST "/oauth/clients" "{
        \"clientId\": \"${PF_CLIENT_ID}\",
        \"name\": \"MCP Gateway Registry\",
        \"clientAuth\": {\"type\": \"SECRET\", \"secret\": \"${PF_CLIENT_SECRET}\"},
        \"grantTypes\": [\"AUTHORIZATION_CODE\", \"CLIENT_CREDENTIALS\", \"REFRESH_TOKEN\"],
        \"redirectUris\": [\"${PF_CALLBACK_URL}\"],
        \"enabled\": true,
        \"defaultAccessTokenManagerRef\": {\"id\": \"jwt\"}
    }" > /dev/null
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

# Step 8: Seed registry's idp_user_groups collection so PingFederate's
# test users (whose JWTs come back with empty `groups`) get mapped to
# real registry groups. The auth-server enrichment looks up by username
# at JWT-validation time and uses these `groups` when the token's
# groups claim is empty (issue #1127).
#
# - admin    -> registry-admins  (full registry admin, matches scripts/registry-admins.json)
# - testuser -> public-mcp-users (read-only MCP server access)
echo "[8/8] Seeding registry idp_user_groups for test users..."
MONGO_DB="${DOCUMENTDB_DATABASE:-mcp_registry}"
MONGO_CONTAINER="mcp-mongodb"
if docker ps --format "{{.Names}}" | grep -q "^${MONGO_CONTAINER}$"; then
    docker exec "${MONGO_CONTAINER}" mongosh --quiet --eval "
db = db.getSiblingDB('${MONGO_DB}');
const now = new Date();
[
  {username: 'admin',    groups: ['registry-admins'],   email: null},
  {username: 'testuser', groups: ['public-mcp-users'],  email: null}
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
print('  idp_user_groups seeded for admin and testuser.');
" || echo "  Warning: failed to seed idp_user_groups (auth-server enrichment will return empty groups)."
else
    echo "  Warning: ${MONGO_CONTAINER} container not running; skipping idp_user_groups seed."
    echo "           Start the stack and re-run this script to seed."
fi

echo ""
echo "=== PingFederate initialization complete ==="
echo ""
echo "Test users: admin/admin123, testuser/changeme, joe/2FederateM0re"
echo ""
echo "Registry group mappings:"
echo "  admin    -> registry-admins"
echo "  testuser -> public-mcp-users"
echo ""
echo "IMPORTANT: Restart auth-server to pick up the CA bundle:"
echo "  docker compose up -d --build auth-server"
echo ""
echo "Then visit: ${PF_EXTERNAL_URL}"
