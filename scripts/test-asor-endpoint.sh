#!/bin/bash
#
# Test ASOR endpoint connectivity
#
# Usage: ./scripts/test-asor-endpoint.sh CLIENT_ID CLIENT_SECRET
#

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          ASOR Endpoint Test                                   ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get credentials
if [ -z "$1" ] || [ -z "$2" ]; then
    echo -e "${YELLOW}Usage: $0 CLIENT_ID CLIENT_SECRET${NC}"
    echo ""
    echo "Or set environment variables:"
    echo "  ASOR_CLIENT_ID=your-client-id"
    echo "  ASOR_CLIENT_SECRET=your-client-secret"
    echo ""

    # Try to get from .env
    if [ -f ".env" ]; then
        echo -e "${BLUE}Checking .env file...${NC}"
        if grep -q "^ASOR_CLIENT_CREDENTIALS=" .env; then
            CREDENTIALS=$(grep "^ASOR_CLIENT_CREDENTIALS=" .env | cut -d'=' -f2)
            CLIENT_ID=$(echo "$CREDENTIALS" | cut -d':' -f1)
            CLIENT_SECRET=$(echo "$CREDENTIALS" | cut -d':' -f2-)
            echo -e "${GREEN}✅ Found credentials in .env${NC}"
        else
            echo -e "${RED}❌ ASOR_CLIENT_CREDENTIALS not found in .env${NC}"
            exit 1
        fi
    else
        exit 1
    fi
else
    CLIENT_ID="$1"
    CLIENT_SECRET="$2"
fi

# Workday tenant details (per Madhur's documentation)
TENANT_URL="https://wcpdev.wd103.myworkday.com"
TENANT_NAME="awsasor_wcpdev1"
TOKEN_URL="${TENANT_URL}/ccx/oauth2/token"
ASOR_ENDPOINT="${TENANT_URL}/${TENANT_NAME}/api/asor/v1/agentDefinition"

echo -e "${BLUE}🔐 Step 1: Getting OAuth2 Token${NC}"
echo "  Token URL: ${TOKEN_URL}"
echo ""

# Get token
TOKEN_RESPONSE=$(curl -s -w "\nHTTP_STATUS:%{http_code}" -X POST "${TOKEN_URL}" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "grant_type=client_credentials" \
    -d "client_id=${CLIENT_ID}" \
    -d "client_secret=${CLIENT_SECRET}")

HTTP_STATUS=$(echo "$TOKEN_RESPONSE" | grep "HTTP_STATUS:" | cut -d':' -f2)
RESPONSE_BODY=$(echo "$TOKEN_RESPONSE" | sed '/HTTP_STATUS:/d')

if [ "$HTTP_STATUS" != "200" ]; then
    echo -e "${RED}❌ Token request failed with status ${HTTP_STATUS}${NC}"
    echo "Response:"
    echo "$RESPONSE_BODY" | jq '.' 2>/dev/null || echo "$RESPONSE_BODY"
    exit 1
fi

ACCESS_TOKEN=$(echo "$RESPONSE_BODY" | jq -r '.access_token' 2>/dev/null)

if [ -z "$ACCESS_TOKEN" ] || [ "$ACCESS_TOKEN" = "null" ]; then
    echo -e "${RED}❌ Failed to extract access token${NC}"
    echo "Response:"
    echo "$RESPONSE_BODY" | jq '.' 2>/dev/null || echo "$RESPONSE_BODY"
    exit 1
fi

echo -e "${GREEN}✅ Successfully obtained access token${NC}"
echo "  Token length: ${#ACCESS_TOKEN} characters"
echo "  Token preview: ${ACCESS_TOKEN:0:50}..."
echo ""

echo -e "${BLUE}🔍 Step 2: Testing ASOR Endpoint${NC}"
echo "  ASOR URL: ${ASOR_ENDPOINT}"
echo ""

# Test ASOR endpoint
ASOR_RESPONSE=$(curl -s -w "\nHTTP_STATUS:%{http_code}" -X GET "${ASOR_ENDPOINT}" \
    -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    -H "Content-Type: application/json")

HTTP_STATUS=$(echo "$ASOR_RESPONSE" | grep "HTTP_STATUS:" | cut -d':' -f2)
RESPONSE_BODY=$(echo "$ASOR_RESPONSE" | sed '/HTTP_STATUS:/d')

echo "Status: ${HTTP_STATUS}"
echo ""

if [ "$HTTP_STATUS" = "200" ]; then
    echo -e "${GREEN}✅ SUCCESS! ASOR endpoint is working${NC}"
    echo ""
    echo "Response:"
    echo "$RESPONSE_BODY" | jq '.' 2>/dev/null || echo "$RESPONSE_BODY"
    echo ""

    # Count agents
    AGENT_COUNT=$(echo "$RESPONSE_BODY" | jq '.data | length' 2>/dev/null || echo "0")
    TOTAL=$(echo "$RESPONSE_BODY" | jq '.total' 2>/dev/null || echo "unknown")

    echo -e "${GREEN}📊 Found ${AGENT_COUNT} agents (total: ${TOTAL})${NC}"

    # List agent names
    if [ "$AGENT_COUNT" != "0" ]; then
        echo ""
        echo "Agent Names:"
        echo "$RESPONSE_BODY" | jq -r '.data[] | "  - \(.name) (v\(.version))"' 2>/dev/null || echo "  (unable to parse)"
    fi

elif [ "$HTTP_STATUS" = "401" ]; then
    echo -e "${RED}❌ UNAUTHORIZED (401)${NC}"
    echo "  Problem: Invalid credentials or token"
    echo "  Check: Client ID and Secret are correct"
    echo ""
    echo "Response:"
    echo "$RESPONSE_BODY" | jq '.' 2>/dev/null || echo "$RESPONSE_BODY"

elif [ "$HTTP_STATUS" = "403" ]; then
    echo -e "${RED}❌ FORBIDDEN (403)${NC}"
    echo "  Problem: Insufficient permissions"
    echo "  Check: API client has 'Agent System of Record' scope"
    echo "  Action: Update API client in Workday and add required scope"
    echo ""
    echo "Response:"
    echo "$RESPONSE_BODY" | jq '.' 2>/dev/null || echo "$RESPONSE_BODY"

elif [ "$HTTP_STATUS" = "404" ]; then
    echo -e "${RED}❌ NOT FOUND (404)${NC}"
    echo "  Problem: ASOR API not available on this tenant"
    echo "  Check:"
    echo "    1. Is ASOR enabled on tenant awsasor_wcpdev1?"
    echo "    2. Login to Workday and search for 'Agent System of Record'"
    echo "    3. Contact Madhur or Workday team to enable ASOR"
    echo ""
    echo "Response:"
    echo "$RESPONSE_BODY" | jq '.' 2>/dev/null || echo "$RESPONSE_BODY"

else
    echo -e "${RED}❌ UNEXPECTED STATUS: ${HTTP_STATUS}${NC}"
    echo ""
    echo "Response:"
    echo "$RESPONSE_BODY" | jq '.' 2>/dev/null || echo "$RESPONSE_BODY"
fi

echo ""
echo -e "${BLUE}📝 Summary${NC}"
echo "  Tenant: wcpdev.wd103.myworkday.com"
echo "  Endpoint: /asor/v1/agentDefinition"
echo "  Status: ${HTTP_STATUS}"
echo ""

if [ "$HTTP_STATUS" = "200" ]; then
    echo -e "${GREEN}✨ ASOR is ready to use!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Set ASOR_CLIENT_CREDENTIALS in .env"
    echo "  2. Ensure 'enabled: true' in config/federation.json"
    echo "  3. Run: ./build_and_run.sh"
    echo "  4. Check logs: docker compose logs registry | grep -i asor"
    exit 0
else
    echo -e "${YELLOW}⚠️  ASOR needs configuration${NC}"
    echo ""
    echo "See docs/asor-setup-guide.md for detailed instructions"
    exit 1
fi
