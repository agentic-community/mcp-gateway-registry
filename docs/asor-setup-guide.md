# Workday ASOR Setup Guide

Complete guide to set up and configure Workday Agent System of Record (ASOR) integration.

## Overview

This guide helps you:
1. Enable ASOR on your Workday tenant
2. Configure security and permissions
3. Register API client credentials
4. Test ASOR endpoints
5. Integrate with MCP Gateway Registry

## Prerequisites

- Workday tenant with ASOR enabled
- Admin access to Workday
- Access to create API clients
- MCP Gateway Registry deployed

## Step 1: Verify ASOR is Enabled

### Check Tenant Configuration

1. **Login to Workday**
   ```
   URL: https://wcpdev.wd103.myworkday.com/awsasor_wcpdev1
   User: lmcneil
   Password: AWS_asor!123
   ```

2. **Search for ASOR Tasks**
   - In search box, type: "Agent System of Record"
   - Look for tasks like:
     - View Agent Definitions
     - Register Agent
     - Manage Agent Registry

3. **If ASOR tasks are NOT visible:**
   - Contact your Workday implementation team
   - Request ASOR feature to be enabled
   - May require tenant upgrade or feature flag

## Step 2: Configure Security (Functional Areas)

### Grant ASOR Permissions

1. **Navigate to Security Setup**
   - Search: "Domain Security Policies"
   - Find: "Agent System of Record"

2. **Create Security Group** (if needed)
   - Task: "Create Security Group"
   - Name: "ASOR_API_Access"
   - Type: "Integration System Security Group"

3. **Assign Permissions**
   - Domain: "Agent System of Record"
   - Permissions needed:
     - `Get` - View agent definitions
     - `Put` - Register/update agents
     - `Delete` - Remove agents (optional)

4. **Add Security Group to Domain**
   - Search: "Activate Pending Security Policy Changes"
   - Review and activate changes

## Step 3: Register API Client

### Option A: Use Existing API Client (if available)

If Madhur or your Workday team has already created an API client for ASOR:

1. **Request Client Credentials**
   - Contact Madhur Prashant or your Workday administrator
   - Ask for the existing API Client ID and Client Secret
   - Verify the client has "Agent System of Record" scope enabled
   - Skip to Step 4 to test the credentials

### Option B: Create New API Client

1. **Navigate to API Client Registration**
   - Search: "Register API Client for Integrations"
   - Click on the task

2. **Fill Out Registration Form**

   **Basic Settings:**
   - **Client Name**: `MCP-Gateway-ASOR-Client`
   - **Client Grant Type**: Select `Authorization Code Grant`
     - ✅ Check "Support Proof Key for Code Exchange (PKCE)"
   - **Access Token Type**: `Bearer`
   - **Enforce 60 Minute Access Token Expiry**: Your choice
     - Uncheck for longer sessions during development

   **Authentication:**
   - **Integration System User**: Leave blank (for client credentials)
   - **Assertion Verification**: `None of the above`
   - **Allow Access to All System Users**: ✅ Checked (or configure specific users)

   **OAuth Settings:**
   - **Redirection URI**: Leave blank (not needed for client credentials flow)
   - **Refresh Token Timeout**: `30` days
   - **Non-Expiring Refresh Tokens**: ⬜ Unchecked

   **Scopes (CRITICAL):**
   - **Scope (Functional Areas)**: Click to add
     - Search and select: `Agent System of Record`
     - **Note**: Check the Workday REST API documentation for exact scope names
     - REST API Directory: https://community.workday.com/sites/default/files/file-hosting/restapi/
     - Look for ASOR or Agent System of Record specific scopes
     - Add any other required scopes for your agents
   - **Include Workday Owned Scope**: ✅ Checked

   **Security:**
   - **Restricted to IP Ranges**: Optional
   - **Grant Administrative Consent**: Based on requirements

3. **Submit and Save Credentials**
   - Click **Done**
   - **IMMEDIATELY COPY**:
     - Client ID
     - Client Secret (you cannot view this again!)
   - Store securely

## Step 4: Test ASOR Endpoints

### Get OAuth Token

```bash
# Set your credentials
CLIENT_ID="your-client-id-here"
CLIENT_SECRET="your-client-secret-here"
TENANT_URL="https://wcpdev.wd103.myworkday.com"
TENANT_NAME="awsasor_wcpdev1"

# Get access token
TOKEN_RESPONSE=$(curl -s -X POST "${TENANT_URL}/ccx/oauth2/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials" \
  -d "client_id=${CLIENT_ID}" \
  -d "client_secret=${CLIENT_SECRET}")

# Extract token
ACCESS_TOKEN=$(echo $TOKEN_RESPONSE | jq -r '.access_token')

echo "Access Token: ${ACCESS_TOKEN:0:50}..."
```

### Test ASOR API Endpoints

Based on standard Workday REST API patterns (like absenceManagement), try these formats:

```bash
# Variation 1: Standard Workday REST format (most likely based on absenceManagement API)
# Pattern: https://<tenantHost>/{serviceName}/{version}/{resource}
curl -v "${TENANT_URL}/asor/v1/agentDefinitions" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json"

# Variation 2: Singular resource name
curl -v "${TENANT_URL}/asor/v1/agentDefinition" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json"

# Variation 3: Different service name
curl -v "${TENANT_URL}/agentSystemOfRecord/v1/agentDefinitions" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json"

# Variation 4: Check if using older ccx path style (less likely)
curl -v "${TENANT_URL}/${TENANT_NAME}/ccx/api/asor/v1/agentDefinition" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json"
```

**Key Insight from absenceManagement API**:
- Standard Workday REST APIs do NOT include tenant name in path
- Format: `https://wcpdev.wd103.myworkday.com/{serviceName}/{version}/...`
- Example: `https://wcpdev.wd103.myworkday.com/absenceManagement/v3/workers`

**Note**: If all variations return 404, ASOR may not be enabled yet. Ask Madhur:
1. What is the exact service name? (asor? agentSystemOfRecord? something else?)
2. Is there an OpenAPI spec like absenceManagement has?
3. What are the available resource endpoints?

**Expected Responses:**
- ✅ `200 OK` - Endpoint works! (may return empty list `[]`)
- ❌ `401 Unauthorized` - Auth problem (check token/scopes)
- ❌ `403 Forbidden` - Permission problem (check security groups)
- ❌ `404 Not Found` - Wrong endpoint or ASOR not enabled

### Verify Successful Response

```bash
# If you get a 200, test creating an agent
curl -X POST "${WORKING_ENDPOINT_URL}/agentDefinition" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Agent",
    "description": "Test agent for validation",
    "url": "https://example.com/agent",
    "provider": {
      "organization": "Test Org",
      "url": "https://example.com"
    },
    "version": "1.0.0",
    "capabilities": {
      "pushNotifications": false,
      "streaming": false,
      "stateTransitionHistory": false
    },
    "defaultInputModes": [{"type": "text/plain"}],
    "defaultOutputModes": [{"type": "text/plain"}],
    "skills": [],
    "supportsAuthenticatedExtendedCard": false
  }'
```

## Step 5: Configure MCP Gateway Registry

### Update Environment Variables

1. **Edit `.env` file:**
   ```bash
   cd /Users/nishdeb/workspace/mcp-gateway-registry
   nano .env
   ```

2. **Add ASOR credentials:**
   ```bash
   # Workday ASOR Configuration
   ASOR_CLIENT_CREDENTIALS=your-client-id:your-client-secret
   WORKDAY_TENANT_URL=https://wcpdev.wd103.myworkday.com
   WORKDAY_TENANT_NAME=awsasor_wcpdev1
   ```

### Update Federation Configuration

1. **Edit `config/federation.json`:**
   ```bash
   nano config/federation.json
   ```

2. **Update ASOR section with working endpoint:**
   ```json
   {
     "asor": {
       "enabled": true,
       "endpoint": "https://wcpdev.wd103.myworkday.com/awsasor_wcpdev1/api/asor/v1",
       "agents": [],
       "cache_ttl_seconds": 3600,
       "sync_interval_seconds": 300,
       "sync_on_startup": true,
       "display_options": {
         "mark_as_federated": true,
         "attribution_label": "Workday ASOR",
         "separate_section": true,
         "read_only": true
       },
       "auth_type": "oauth2",
       "auth_env_var": "ASOR_CLIENT_CREDENTIALS",
       "timeout_seconds": 30,
       "retry_attempts": 3
     }
   }
   ```

3. **If you want to sync specific agents, add them:**
   ```json
   "agents": [
     {
       "id": "agent-123",
       "endpoint": "https://wcpdev.wd103.myworkday.com/awsasor_wcpdev1/api/asor/v1/agentDefinition/agent-123",
       "enabled": true,
       "metadata": {
         "description": "HR Assistant Agent",
         "category": "hr"
       }
     }
   ]
   ```

4. **Leave agents array empty to sync ALL agents:**
   ```json
   "agents": []
   ```

## Step 6: Deploy and Test

### Rebuild and Start Services

```bash
# Rebuild with new configuration
./build_and_run.sh

# Wait for startup (30-60 seconds)
```

### Verify Federation

```bash
# Check startup logs for ASOR
docker compose logs registry | grep -i asor

# Expected logs:
# "Federation enabled for: anthropic, asor"
# "Syncing agents from ASOR..."
# "Synced X agents from asor"
```

### Check Cache

```bash
# View cached ASOR data
docker compose exec registry cat /app/.cache/federation/asor_cache.json | jq '.'

# Should show:
# - cached_at timestamp
# - source: "asor"
# - servers: [array of agents]
```

### View in UI

1. **Open MCP Gateway:**
   ```
   http://localhost:7860
   ```

2. **Look for ASOR agents:**
   - Should appear in server list
   - Tagged with "Workday ASOR"
   - Marked as "federated" and "read-only"

## Troubleshooting

### Issue: 404 Not Found on ASOR Endpoint

**Possible Causes:**
1. ASOR not enabled on tenant
2. Wrong API endpoint path
3. Missing feature flags

**Solutions:**
- Contact Workday implementation team
- Verify ASOR is part of your tenant subscription
- Ask for correct API endpoint documentation
- Check if tenant needs to be on specific Workday version

### Issue: 401 Unauthorized

**Possible Causes:**
1. Invalid credentials
2. Token expired
3. Wrong token endpoint

**Solutions:**
```bash
# Test token generation
curl -v -X POST "https://wcpdev.wd103.myworkday.com/ccx/oauth2/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials" \
  -d "client_id=YOUR_CLIENT_ID" \
  -d "client_secret=YOUR_CLIENT_SECRET"

# Check response for errors
```

### Issue: 403 Forbidden

**Possible Causes:**
1. Missing "Agent System of Record" scope
2. Security group not configured
3. User doesn't have permissions

**Solutions:**
1. Go back to API Client registration
2. Edit client
3. Verify "Agent System of Record" is in Scopes
4. Check security group assignments
5. Activate pending security policy changes

### Issue: Empty Agent List

**Expected Behavior:**
- If no agents are registered in ASOR yet, you'll get an empty array `[]`
- This is normal for a new tenant

**To Register Test Agent:**
- Use Madhur's tool: https://github.com/madhurprash/agentcore_asor_integration
- Or manually POST an agent definition via API

### Issue: No ASOR Logs in Registry

**Check:**
```bash
# Is ASOR enabled in config?
docker compose exec registry cat /app/config/federation.json | jq '.asor.enabled'

# Should return: true

# Check for errors
docker compose logs registry | grep -i "error\|fail" | grep -i asor
```

## Reference Information

### ASOR REST API Specification

**Base Endpoint Format:**
```
https://{tenant-host}/{tenant-name}/api/asor/v1
```

**Operations:**
- `GET /agentDefinition` - List all agents
- `GET /agentDefinition/{id}` - Get specific agent
- `POST /agentDefinition` - Register new agent
- `PUT /agentDefinition/{id}` - Update agent
- `DELETE /agentDefinition/{id}` - Delete agent

### Agent Card Schema

```json
{
  "name": "Agent Name",
  "description": "Agent description",
  "url": "https://agent-endpoint.com",
  "iconUrl": "https://icon-url.com/icon.png",
  "provider": {
    "organization": "Organization Name",
    "url": "https://org.com"
  },
  "version": "1.0.0",
  "capabilities": {
    "pushNotifications": false,
    "streaming": true,
    "stateTransitionHistory": false
  },
  "defaultInputModes": [
    { "type": "text/plain" }
  ],
  "defaultOutputModes": [
    { "type": "text/markdown" }
  ],
  "skills": [
    {
      "id": "skill-1",
      "name": "Skill Name",
      "description": "Skill description",
      "tags": [
        { "tag": "workday" },
        { "tag": "hr" }
      ],
      "inputModes": [
        { "type": "application/json" }
      ],
      "outputModes": [
        { "type": "application/json" }
      ]
    }
  ],
  "supportsAuthenticatedExtendedCard": false,
  "workdayConfig": []
}
```

### Useful Workday Contacts

- **Madhur Prashant** - ASOR/AgentCore integration
- **Workday Support** - For tenant configuration
- **Implementation Team** - For feature enablement

### External Resources

- **ASOR Documentation**: (Internal Workday docs)
- **Madhur's GitHub**: https://github.com/madhurprash/agentcore_asor_integration
- **Workday REST API**: https://community.workday.com/sites/default/files/file-hosting/restapi/
- **Issue #204**: MCP Gateway Federation spec

## Next Steps

Once ASOR is working:

1. **Register AgentCore Agents**
   - Use Madhur's tool to register Bedrock AgentCore agents
   - Agents will appear in both ASOR and MCP Gateway

2. **Configure Agent Gateway**
   - Set up secure tool access
   - Configure authentication

3. **Test Agent-to-Agent (A2A)**
   - Enable A2A protocol
   - Test cross-agent communication

4. **Production Deployment**
   - Move from dev tenant to production
   - Update credentials
   - Configure production endpoints

## Support

For issues:
- **MCP Gateway**: GitHub Issues
- **ASOR Setup**: Contact Madhur or Workday team
- **Tenant Access**: Workday Support
