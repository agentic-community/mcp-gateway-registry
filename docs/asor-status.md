# ASOR Integration - Current Status

**Last Updated**: 2025-11-06

## ✅ Completed

1. **Federation Framework** - Fully implemented and working
   - Configuration-driven federation system (follows issue #204 pattern)
   - Base federation client with retry logic
   - Anthropic federation client (6 servers synced successfully)
   - ASOR federation client (code complete, ready to use)
   - Caching with TTL and disk persistence
   - Federation service orchestration

2. **Anthropic Federation** - Live and Working
   - Syncing 6 servers from Anthropic MCP Registry
   - Cache location: `/app/.cache/federation/anthropic_cache.json`
   - Servers visible in UI with "Anthropic MCP Registry" attribution
   - Read-only display mode working correctly

3. **ASOR Client Implementation** - Code Complete
   - OAuth2 client credentials flow implemented
   - Token caching with expiry handling
   - Agent fetching and listing capabilities
   - Response transformation to internal format
   - Configuration: `config/federation.json` (currently disabled)

4. **Documentation** - Comprehensive Guides Created
   - `docs/asor-setup-guide.md` - Complete setup instructions
   - `docs/federation.md` - Federation system overview
   - `scripts/setup-workday-asor.sh` - Automated credential setup

5. **Environment Configuration** - Ready
   - `.env` configured with Workday tenant info:
     - `WORKDAY_TENANT_URL=https://wcpdev.wd103.myworkday.com`
     - `WORKDAY_TENANT_NAME=awsasor_wcpdev1`
     - `ASOR_CLIENT_CREDENTIALS=your-client-id:your-client-secret` (placeholder)

## 🚧 Blocked - Waiting on Workday

### Primary Blocker: ASOR API Endpoint Not Available

**Problem**: All tested ASOR endpoint variations return `404 Not Found`:
- ❌ `https://wcpdev.wd103.myworkday.com/awsasor_wcpdev1/api/asor/v1/agentDefinition`
- ❌ `https://wcpdev.wd103.myworkday.com/api/asor/v1/agentDefinition`
- ❌ Multiple other path variations (see setup guide)

**New Discovery**: After analyzing Workday's absenceManagement API structure, standard Workday REST APIs follow this pattern:
- ✅ Format: `https://<tenantHost>/{serviceName}/{version}/{resource}`
- ✅ Example: `https://wcpdev.wd103.myworkday.com/absenceManagement/v3/workers`
- ❌ **No tenant name in path** (unlike SOAP/WSDL services)

**Root Cause**: ASOR API feature likely not enabled on tenant `awsasor_wcpdev1` yet.

### Required Actions (Contact Madhur/Workday Team)

1. **Verify ASOR is enabled**
   - Login: `https://wcpdev.wd103.myworkday.com/awsasor_wcpdev1`
   - User: `lmcneil` / Password: `AWS_asor!123`
   - Search for: "Agent System of Record" tasks
   - Should see: "View Agent Definitions", "Register Agent", "Manage Agent Registry"

2. **Get correct API endpoint and service name**
   - Ask Madhur: What is the ASOR service name? (asor? agentSystemOfRecord?)
   - Expected format: `https://wcpdev.wd103.myworkday.com/{serviceName}/v1/...`
   - Check if ASOR has OpenAPI spec: `https://community.workday.com/sites/default/files/file-hosting/restapi/asor_v1_YYYYMMDD_oas2.json`
   - Confirm resource names (agentDefinitions? agentDefinition? agents?)

3. **Verify API client credentials**
   - Does Madhur have existing API client credentials for ASOR?
   - If yes: Get Client ID and Client Secret
   - If no: Follow `docs/asor-setup-guide.md` to create new client
   - Ensure "Agent System of Record" scope is granted

4. **Get list of registered agents**
   - How many agents are currently in ASOR?
   - What are their agent IDs?
   - This will help test the federation sync

## 📋 Next Steps (Once Endpoint Available)

### Step 1: Update Configuration
```bash
# Update config/federation.json
"asor": {
  "enabled": true,  # Change from false
  "endpoint": "https://CORRECT_ENDPOINT_HERE",  # Update with working endpoint
  ...
}
```

### Step 2: Add API Credentials
```bash
# Update .env with actual credentials
ASOR_CLIENT_CREDENTIALS=actual-client-id:actual-client-secret
```

### Step 3: Test Authentication
```bash
# Test token endpoint
curl -X POST "https://wcpdev.wd103.myworkday.com/ccx/oauth2/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials" \
  -d "client_id=YOUR_CLIENT_ID" \
  -d "client_secret=YOUR_CLIENT_SECRET"

# Should return access token
```

### Step 4: Test ASOR Endpoint
```bash
# Use the working endpoint from Madhur
curl -v "WORKING_ASOR_ENDPOINT/agentDefinition" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json"

# Should return 200 OK with list of agents (or empty array [])
```

### Step 5: Deploy and Verify
```bash
# Rebuild with new config
./build_and_run.sh

# Wait 30-60 seconds for startup

# Check logs for ASOR sync
docker compose logs registry | grep -i asor

# Expected logs:
# "Federation enabled for: anthropic, asor"
# "Syncing agents from ASOR..."
# "Synced X agents from asor"

# Verify cache created
docker compose exec registry cat /app/.cache/federation/asor_cache.json | jq '.'

# Should show agents synced from ASOR
```

### Step 6: View in UI
```bash
# Open browser
http://localhost:7860

# Look for ASOR agents:
# - Tagged with "Workday ASOR"
# - Marked as "federated" and "read-only"
# - Separate section if configured
```

## 🔍 Troubleshooting

### If still getting 404 after endpoint update:
1. Verify ASOR feature is enabled on tenant (contact Workday support)
2. Check tenant version supports ASOR
3. Confirm API endpoint format with Workday documentation
4. Try logging into Workday UI and search for "Agent System of Record" tasks

### If getting 401 Unauthorized:
1. Verify credentials are correct
2. Check token generation works
3. Ensure client has "Agent System of Record" scope

### If getting 403 Forbidden:
1. Check API client has proper scopes granted
2. Verify security group permissions in Workday
3. Activate pending security policy changes
4. Ensure functional area "Agent System of Record" is accessible

### If no logs appear:
1. Check `config/federation.json` has `"enabled": true` for asor
2. Verify `sync_on_startup: true` is set
3. Check for errors: `docker compose logs registry | grep -i error | grep -i asor`

## 📚 Reference Documents

- **Setup Guide**: `docs/asor-setup-guide.md` - Complete step-by-step instructions
- **Federation Overview**: `docs/federation.md` - How federation system works
- **Setup Script**: `scripts/setup-workday-asor.sh` - Automated credential setup
- **Configuration**: `config/federation.json` - Federation settings
- **ASOR Client Code**: `registry/services/federation/asor_client.py` - Implementation

## 👥 Contacts

- **Madhur Prashant** - ASOR/AgentCore integration expert
  - GitHub: https://github.com/madhurprash/agentcore_asor_integration
- **Workday Support** - For tenant configuration and feature enablement
- **Workday REST API Docs** - https://community.workday.com/sites/default/files/file-hosting/restapi/

## 📝 Questions for Madhur/Workday Team

When contacting Madhur or the Workday team, ask:

1. **Is ASOR enabled on tenant `awsasor_wcpdev1`?**
   - If no, how do we enable it?
   - What tenant version is required?

2. **What is the correct ASOR REST API endpoint?**
   - Base URL format?
   - API version (v1, v2)?
   - Path structure for agentDefinition operations?

3. **Do you have existing API client credentials for ASOR?**
   - If yes, can you share Client ID and Secret?
   - What scopes are granted?

4. **Are there any registered agents in ASOR currently?**
   - How many agents?
   - What are their IDs?
   - Can I get a sample agent response?

5. **Any special authentication requirements?**
   - Beyond standard OAuth2 client credentials?
   - IP restrictions?
   - Additional headers needed?

---

**Current Status**: ASOR integration is **code-complete and ready**, just waiting for working API endpoint from Workday team. Anthropic federation is **live and working** with 6 servers synced.
