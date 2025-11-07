#!/bin/bash
#
# Setup script for Workday ASOR integration
#
# This script helps you configure API client credentials for Workday ASOR
#

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          Workday ASOR Federation Setup                       ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}📋 Workday Tenant Information:${NC}"
echo "  Tenant URL: https://wcpdev.wd103.myworkday.com"
echo "  Tenant Name: awsasor_wcpdev1"
echo "  Test User: lmcneil / AWS_asor!123"
echo ""

echo -e "${BLUE}🔑 API Client Setup Required:${NC}"
echo "  1. Log into Workday tenant as an admin"
echo "  2. Navigate to: Create API Client for Integrations"
echo "  3. Grant required scopes (see REST API documentation)"
echo "  4. Copy the Client ID and Client Secret"
echo ""

# Prompt for API client credentials
echo -e "${YELLOW}Enter your Workday API Client credentials:${NC}"
read -p "Client ID: " CLIENT_ID
read -sp "Client Secret: " CLIENT_SECRET
echo ""

if [ -z "$CLIENT_ID" ] || [ -z "$CLIENT_SECRET" ]; then
    echo -e "${YELLOW}⚠️  Credentials not provided. Skipping...${NC}"
    exit 0
fi

# Update .env file
ENV_FILE="$PROJECT_ROOT/.env"

if grep -q "^ASOR_CLIENT_CREDENTIALS=" "$ENV_FILE" 2>/dev/null; then
    # Update existing
    sed -i.backup "s|^ASOR_CLIENT_CREDENTIALS=.*|ASOR_CLIENT_CREDENTIALS=${CLIENT_ID}:${CLIENT_SECRET}|" "$ENV_FILE"
    echo -e "${GREEN}✅ Updated ASOR credentials in .env${NC}"
else
    # Add new
    echo "" >> "$ENV_FILE"
    echo "# Workday ASOR Configuration" >> "$ENV_FILE"
    echo "ASOR_CLIENT_CREDENTIALS=${CLIENT_ID}:${CLIENT_SECRET}" >> "$ENV_FILE"
    echo -e "${GREEN}✅ Added ASOR credentials to .env${NC}"
fi

echo ""
echo -e "${BLUE}🔄 Testing authentication...${NC}"

# Test token endpoint
TENANT_URL="https://wcpdev.wd103.myworkday.com"
TOKEN_URL="${TENANT_URL}/ccx/oauth2/token"

echo "  Token URL: $TOKEN_URL"

# Try to get token
RESPONSE=$(curl -s -X POST "$TOKEN_URL" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "grant_type=client_credentials&client_id=${CLIENT_ID}&client_secret=${CLIENT_SECRET}" \
    2>&1) || true

if echo "$RESPONSE" | grep -q "access_token"; then
    echo -e "${GREEN}✅ Authentication successful!${NC}"
    ACCESS_TOKEN=$(echo "$RESPONSE" | jq -r '.access_token' 2>/dev/null || echo "")
    if [ -n "$ACCESS_TOKEN" ]; then
        echo "  Token obtained (${#ACCESS_TOKEN} characters)"
    fi
else
    echo -e "${YELLOW}⚠️  Authentication may have failed. Response:${NC}"
    echo "$RESPONSE" | head -5
fi

echo ""
echo -e "${BLUE}📝 Next Steps:${NC}"
echo "  1. Verify agent IDs in Workday ASOR"
echo "  2. Update config/federation.json with actual agent IDs"
echo "  3. Restart services: ./build_and_run.sh"
echo "  4. Check logs: docker compose logs -f registry | grep -i asor"
echo ""
echo -e "${GREEN}✨ Setup complete!${NC}"
