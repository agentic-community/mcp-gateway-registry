# Workday REST API Pattern Discovery

**Date**: 2025-11-06

## Key Discovery: Standard Workday REST API Format

After analyzing the Workday absenceManagement API OpenAPI specification, we discovered the standard Workday REST API URL pattern.

## Standard Workday REST API Pattern

### URL Format
```
https://<tenantHost>/{serviceName}/{version}/{resource}
```

### Example from absenceManagement API
```json
{
  "swagger": "2.0",
  "host": "<tenantHostname>",
  "basePath": "/absenceManagement/v3",
  "schemes": ["https"]
}
```

**Full URL**: `https://wcpdev.wd103.myworkday.com/absenceManagement/v3/workers`

### Key Observations

1. **No Tenant Name in Path**
   - ❌ Wrong: `https://wcpdev.wd103.myworkday.com/awsasor_wcpdev1/api/asor/v1/...`
   - ✅ Correct: `https://wcpdev.wd103.myworkday.com/asor/v1/...`

2. **Service Name + Version Pattern**
   - Format: `/{serviceName}/{version}/...`
   - Examples:
     - `/absenceManagement/v3/...`
     - `/asor/v1/...` (expected for ASOR)
     - `/staffing/v2/...`

3. **Resource Names**
   - Typically plural: `/workers`, `/balances`, `/agentDefinitions`
   - Some singular: `/agentDefinition` (need to verify with Madhur)

4. **OpenAPI Specs Available**
   - Located at: `https://community.workday.com/sites/default/files/file-hosting/restapi/`
   - Format: `{serviceName}_{version}_{date}_oas2.json`
   - Example: `absenceManagement_v3_20251101_oas2.json`

## Applied to ASOR Integration

### Updated ASOR Endpoint (Best Guess)

Based on standard Workday pattern:
```
https://wcpdev.wd103.myworkday.com/asor/v1
```

### Possible Resource Endpoints

```bash
# List all agents
GET https://wcpdev.wd103.myworkday.com/asor/v1/agentDefinitions

# Get specific agent
GET https://wcpdev.wd103.myworkday.com/asor/v1/agentDefinitions/{id}

# Create agent
POST https://wcpdev.wd103.myworkday.com/asor/v1/agentDefinitions

# Update agent
PUT https://wcpdev.wd103.myworkday.com/asor/v1/agentDefinitions/{id}

# Delete agent
DELETE https://wcpdev.wd103.myworkday.com/asor/v1/agentDefinitions/{id}
```

### Alternative Service Names to Try

If `asor` doesn't work:
- `agentSystemOfRecord`
- `agents`
- `agentRegistry`

## Comparison: Old vs New Understanding

### What We Tried Before
```bash
# Including tenant name (WRONG)
https://wcpdev.wd103.myworkday.com/awsasor_wcpdev1/api/asor/v1/agentDefinition

# Various incorrect formats
https://wcpdev.wd103.myworkday.com/awsasor_wcpdev1/ccx/api/asor/v1/...
https://wcpdev.wd103.myworkday.com/api/asor/v1/...
```

### What to Try Now
```bash
# Standard Workday REST format (CORRECT)
https://wcpdev.wd103.myworkday.com/asor/v1/agentDefinitions
https://wcpdev.wd103.myworkday.com/asor/v1/agentDefinition
https://wcpdev.wd103.myworkday.com/agentSystemOfRecord/v1/agentDefinitions
```

## Questions for Madhur

Now that we understand the pattern, we need specific ASOR details:

1. **Service Name**: What is the exact service name?
   - Is it `asor`?
   - Is it `agentSystemOfRecord`?
   - Something else?

2. **OpenAPI Spec**: Does ASOR have a published OpenAPI spec?
   - Looking for: `https://community.workday.com/sites/default/files/file-hosting/restapi/asor_v1_YYYYMMDD_oas2.json`
   - Or equivalent documentation

3. **Resource Names**: What are the resource endpoints?
   - Plural `agentDefinitions` or singular `agentDefinition`?
   - What operations are supported (GET, POST, PUT, DELETE)?

4. **Scope Name**: What functional area scope is needed?
   - absenceManagement uses "Time Off and Leave"
   - What does ASOR use?

5. **API Version**: Confirm it's v1 or if there's a different version

## Authentication Pattern (Confirmed Working)

The OAuth2 token endpoint is standard across all Workday APIs:

```bash
POST https://wcpdev.wd103.myworkday.com/ccx/oauth2/token
Content-Type: application/x-www-form-urlencoded

grant_type=client_credentials
&client_id=YOUR_CLIENT_ID
&client_secret=YOUR_CLIENT_SECRET
```

This part we have working - just need the correct ASOR service endpoint!

## Code Updates Made

1. **config/federation.json**
   - Updated endpoint: `https://wcpdev.wd103.myworkday.com/asor/v1`
   - Removed tenant name from path

2. **asor_client.py**
   - Updated resource paths to `/agentDefinitions` (plural)
   - Added comments explaining Workday REST pattern

3. **docs/asor-setup-guide.md**
   - Updated endpoint testing section with correct patterns
   - Added reference to absenceManagement API as example

4. **docs/asor-status.md**
   - Added API discovery findings
   - Updated questions for Madhur

## Next Steps

1. **Ask Madhur** the 5 questions above
2. **Test with correct service name** once confirmed
3. **Update configuration** with working endpoint
4. **Deploy and verify** ASOR integration

## References

- **absenceManagement API Spec**: https://community.workday.com/sites/default/files/file-hosting/restapi/absenceManagement_v3_20251101_oas2.json
- **Workday REST API Directory**: https://community.workday.com/sites/default/files/file-hosting/restapi/
- **Issue #204**: ASOR Federation for Enterprise Agent Discovery

---

**Status**: Waiting for ASOR service name and endpoint confirmation from Madhur
