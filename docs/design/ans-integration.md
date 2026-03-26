# ANS (Agent Name Service) Integration

This document describes the ANS integration architecture, configuration, API usage, and operational procedures for the MCP Gateway Registry.

## Overview

ANS (Agent Name Service) is a PKI-based trust verification service operated by GoDaddy that provides cryptographic identity verification for AI agents. The MCP Gateway Registry integrates with ANS using a **read-only "Bring Your Own ANS ID"** approach -- the registry never manages PKI certificates or identities directly. Instead, agent owners register with ANS independently and then link their ANS Agent ID to their registry entry for trust verification.

### What ANS Provides

- Cryptographic identity verification for AI agents
- Domain ownership proof via PKI certificates
- Agent identity metadata (name, description, version, organization)
- Endpoint and protocol registration (A2A, MCP, HTTP-API)
- Certificate lifecycle management (issuance, expiration, revocation)

### What the Registry Does

- Stores ANS verification metadata on agent and server records
- Displays trust badges on agent/server cards in the UI
- Periodically re-verifies ANS status via background sync
- Provides admin visibility into ANS integration health

```
Integration Architecture:

  Agent Owner                 AI Registry                 GoDaddy ANS API
  -----------                 -----------                 ---------------
       |                           |                            |
       |  1. Register with ANS     |                            |
       |  (out-of-band)           ========================>     |
       |                           |                            |
       |  2. Link ANS ID          |                            |
       |  POST /agents/{p}/ans/link                            |
       | ======================>   |                            |
       |                           |  3. Verify with ANS API   |
       |                           | ========================> |
       |                           |  <== ANS Metadata ======  |
       |                           |                            |
       |  4. Trust badge shown     |                            |
       |  <=====================   |                            |
       |                           |                            |
       |                           |  5. Background re-verify  |
       |                           |  (every 6 hours)          |
       |                           | ========================> |
```

## Configuration

All ANS configuration is managed via environment variables that map to Pydantic Settings fields in `registry/core/config.py`.

### Required Configuration

| Parameter | Environment Variable | Description | Default |
|-----------|---------------------|-------------|---------|
| `ans_integration_enabled` | `ANS_INTEGRATION_ENABLED` | Master switch for ANS integration | `false` |
| `ans_api_key` | `ANS_API_KEY` | GoDaddy API key for authentication | `""` |
| `ans_api_secret` | `ANS_API_SECRET` | GoDaddy API secret for authentication | `""` |

### Optional Configuration

| Parameter | Environment Variable | Description | Default |
|-----------|---------------------|-------------|---------|
| `ans_api_endpoint` | `ANS_API_ENDPOINT` | ANS API base URL | `https://api.godaddy.com` |
| `ans_api_timeout_seconds` | `ANS_API_TIMEOUT_SECONDS` | HTTP request timeout for ANS calls | `30` |
| `ans_sync_interval_hours` | `ANS_SYNC_INTERVAL_HOURS` | Background verification sync interval | `6` |
| `ans_verification_cache_ttl_seconds` | `ANS_VERIFICATION_CACHE_TTL_SECONDS` | Cache TTL for verification results | `3600` |

### Environment File Example

```bash
# ANS Integration
ANS_INTEGRATION_ENABLED=true
ANS_API_ENDPOINT=https://api.godaddy.com
ANS_API_KEY=your-godaddy-api-key
ANS_API_SECRET=your-godaddy-api-secret
ANS_API_TIMEOUT_SECONDS=30
ANS_SYNC_INTERVAL_HOURS=6
```

### Terraform Configuration

For ECS deployments, set these in `terraform/aws-ecs/terraform.tfvars`:

```hcl
ans_integration_enabled = true
ans_api_endpoint        = "https://api.godaddy.com"
ans_api_key             = "your-api-key"
ans_api_secret          = "your-api-secret"
```

### System Configuration Page

ANS configuration is visible in the admin System Configuration page under the "ANS Integration" group. Navigate to the registry UI and open the system configuration panel to view and export current ANS settings.

## API Endpoints

### Agent ANS Endpoints

#### Link ANS ID to Agent

Links an ANS Agent ID to a registered agent. The registry calls the ANS API to verify the identity and stores the metadata.

```bash
POST /api/agents/{agent_path}/ans/link
Content-Type: application/json
Authorization: Bearer <token>
X-CSRF-Token: <csrf_token>

{
  "ans_agent_id": "ans://v1.0.0.myagent.example.com"
}
```

**Response (200):**
```json
{
  "success": true,
  "message": "ANS identity linked and verified",
  "ans_metadata": {
    "ans_agent_id": "89a5061b-4f89-452b-9b66-dd9ca8baad7f",
    "status": "verified",
    "domain": "example.com",
    "organization": "Example Corp",
    "ans_name": "myagent.example.com",
    "ans_display_name": "My Agent",
    "certificate": {
      "not_before": "2025-01-01T00:00:00Z",
      "not_after": "2026-01-01T00:00:00Z",
      "subject_dn": "CN=myagent.example.com",
      "issuer_dn": "CN=ANS CA"
    },
    "endpoints": [],
    "linked_at": "2026-03-26T12:00:00Z",
    "last_verified": "2026-03-26T12:00:00Z"
  }
}
```

**Requirements:**
- User must be authenticated
- User must own the agent (`registered_by` field matches username)
- Rate limited: 10 link operations per user per hour
- CSRF token required

#### Get ANS Status

```bash
GET /api/agents/{agent_path}/ans/status
Authorization: Bearer <token>
```

**Response (200):** Returns full ANS metadata for the agent.
**Response (404):** Agent has no ANS link.

#### Unlink ANS from Agent

```bash
DELETE /api/agents/{agent_path}/ans/link
Authorization: Bearer <token>
X-CSRF-Token: <csrf_token>
```

**Response (200):**
```json
{
  "success": true,
  "message": "ANS identity unlinked"
}
```

### Server ANS Endpoints

Servers follow the same pattern as agents:

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/servers/{path}/ans/link` | Link ANS ID to server |
| GET | `/api/servers/{path}/ans/status` | Get server ANS status |
| DELETE | `/api/servers/{path}/ans/link` | Unlink ANS from server |

### Admin Endpoints

#### Trigger Manual Sync

Forces an immediate re-verification of all linked ANS identities.

```bash
POST /api/admin/ans/sync
Authorization: Bearer <admin_token>
X-CSRF-Token: <csrf_token>
```

**Response (200):**
```json
{
  "total": 15,
  "updated": 12,
  "errors": 1,
  "duration_seconds": 4.2
}
```

**Requires:** Admin group membership or `ans-admin/manage` scope.

#### Get ANS Metrics

```bash
GET /api/admin/ans/metrics
Authorization: Bearer <admin_token>
```

**Response (200):**
```json
{
  "total_linked": 15,
  "by_status": {
    "verified": 12,
    "expired": 2,
    "not_found": 1
  },
  "by_asset_type": {
    "agent": 10,
    "server": 5
  },
  "sync_history": [
    {
      "timestamp": "2026-03-26T06:00:00Z",
      "total": 15,
      "updated": 2,
      "errors": 0,
      "duration_seconds": 3.8
    }
  ]
}
```

#### Check ANS API Health

```bash
GET /api/admin/ans/health
Authorization: Bearer <admin_token>
```

**Response (200):**
```json
{
  "status": "healthy",
  "api_reachable": true,
  "api_status_code": 200
}
```

Possible status values: `healthy`, `degraded`, `unhealthy`.

## CLI Usage

The registry management CLI can be used to interact with ANS endpoints.

### Link an Agent to ANS

```bash
# Using curl with token file
TOKEN=$(cat .token)
REGISTRY_URL="https://your-registry.example.com"

# Link ANS identity
curl -X POST "${REGISTRY_URL}/api/agents/my-agent/ans/link" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"ans_agent_id": "ans://v1.0.0.myagent.example.com"}'
```

### Check ANS Status

```bash
curl -s "${REGISTRY_URL}/api/agents/my-agent/ans/status" \
  -H "Authorization: Bearer ${TOKEN}" | python -m json.tool
```

### Admin: Trigger Manual Sync

```bash
curl -X POST "${REGISTRY_URL}/api/admin/ans/sync" \
  -H "Authorization: Bearer ${TOKEN}"
```

### Admin: View Metrics

```bash
curl -s "${REGISTRY_URL}/api/admin/ans/metrics" \
  -H "Authorization: Bearer ${TOKEN}" | python -m json.tool
```

### Admin: Check API Health

```bash
curl -s "${REGISTRY_URL}/api/admin/ans/health" \
  -H "Authorization: Bearer ${TOKEN}" | python -m json.tool
```

### Verify Agent Has ANS Data

List an agent and check the `ans_metadata` field:

```bash
curl -s "${REGISTRY_URL}/api/agents/my-agent" \
  -H "Authorization: Bearer ${TOKEN}" | python -m json.tool | grep -A 20 ans_metadata
```

## ANS Agent ID Format

ANS Agent IDs follow the URI format:

```
ans://v1.0.0.agentname.domain.com
```

Components:
- `ans://` - Protocol scheme
- `v1.0.0` - Version identifier
- `agentname` - Agent name within the domain
- `domain.com` - Verified domain

The registry also accepts raw UUIDs (e.g., `89a5061b-4f89-452b-9b66-dd9ca8baad7f`). When an `ans://` URI is provided, the client resolves it to a UUID by searching the ANS API.

## Verification Status Values

| Status | Meaning | Badge Color |
|--------|---------|-------------|
| `verified` | Agent identity is valid and certificate is current | Green |
| `expired` | Certificate has passed its `notAfter` date | Yellow |
| `revoked` | Agent or certificate has been explicitly revoked | Red |
| `not_found` | ANS Agent ID no longer exists in ANS | Gray |
| `pending` | Verification is in progress | Blue |

## Architecture Components

### Service Layer

| File | Purpose |
|------|---------|
| `registry/services/ans_client.py` | Low-level HTTP client for GoDaddy ANS API |
| `registry/services/ans_service.py` | Business logic for link/unlink/sync operations |
| `registry/services/ans_sync_scheduler.py` | Background task that re-verifies all linked identities |

### API Layer

| File | Purpose |
|------|---------|
| `registry/api/ans_routes.py` | FastAPI router with all ANS endpoints |

### Data Models

| File | Purpose |
|------|---------|
| `registry/schemas/ans_models.py` | Pydantic models for ANS metadata, certificates, endpoints |
| `registry/schemas/agent_models.py` | Agent model with `ans_metadata` field |
| `registry/core/schemas.py` | Server model with `ans_metadata` field |

### Frontend

| File | Purpose |
|------|---------|
| `frontend/src/components/ANSBadge.tsx` | Badge component and certificate detail modal |
| `frontend/src/components/AgentCard.tsx` | Displays ANS badge on agent cards |
| `frontend/src/components/ServerCard.tsx` | Displays ANS badge on server cards |

## Background Sync

The ANS sync scheduler runs as an async background task within the FastAPI application lifecycle.

### How It Works

1. On application startup, if `ans_integration_enabled=True`, the scheduler starts
2. Every `ans_sync_interval_hours` (default 6), it runs `sync_all_ans_status()`
3. The sync queries all agents and servers with non-null `ans_metadata`
4. For each linked asset, it calls the ANS API to re-verify the identity
5. Updates `ans_metadata.status` and `ans_metadata.last_verified` timestamp
6. Stores sync results in memory (last 20 runs), viewable via admin metrics endpoint

### Sync Lifecycle

```
Application Start
      |
      v
  [ANS Enabled?] --No--> Skip
      |
     Yes
      v
  Start Scheduler Loop
      |
      v
  Sleep(sync_interval_hours)
      |
      v
  sync_all_ans_status()
      |
      +---> For each agent with ans_metadata:
      |         verify_ans_agent(ans_agent_id)
      |         Update status + last_verified
      |
      v
  Store sync stats
      |
      v
  Loop back to Sleep
```

## Resilience Features

### Circuit Breaker

The ANS client implements a circuit breaker to prevent cascading failures when the ANS API is unavailable.

| Parameter | Value |
|-----------|-------|
| Failure threshold | 5 consecutive failures |
| Reset timeout | 3600 seconds (1 hour) |
| Behavior when open | Returns `None` immediately without calling API |

### Retry Logic

Each ANS API call includes automatic retries:

| Parameter | Value |
|-----------|-------|
| Max retries | 3 |
| Backoff strategy | Exponential (1s, 2s, 4s) |
| Timeout per request | `ans_api_timeout_seconds` (default 30s) |

### Rate Limiting

Per-user rate limiting on link operations prevents abuse:

| Parameter | Value |
|-----------|-------|
| Max requests | 10 per user |
| Window | 3600 seconds (1 hour) |

## Authentication with ANS API

The registry authenticates with GoDaddy's ANS API using SSO-key authentication:

```
Authorization: sso-key {ans_api_key}:{ans_api_secret}
```

This is a GoDaddy-specific authentication scheme. API keys are obtained from the GoDaddy developer portal.

### API Endpoints Used

| Method | ANS API Path | Purpose |
|--------|-------------|---------|
| GET | `/v1/agents/{uuid}` | Fetch agent details and certificate info |
| GET | `/v1/agents?name={name}` | Resolve `ans://` URI to UUID |

## Data Storage

ANS metadata is stored as a `dict[str, Any]` field on both agent and server MongoDB documents. This allows schema evolution without database migrations.

### MongoDB Field

```json
{
  "path": "/my-agent",
  "name": "My Agent",
  "ans_metadata": {
    "ans_agent_id": "89a5061b-4f89-452b-9b66-dd9ca8baad7f",
    "status": "verified",
    "domain": "example.com",
    "organization": "Example Corp",
    "ans_name": "myagent.example.com",
    "certificate": { ... },
    "endpoints": [ ... ],
    "linked_at": "2026-03-26T12:00:00Z",
    "last_verified": "2026-03-26T12:00:00Z"
  }
}
```

### ANS Metadata in Agent and Server List APIs

The agent list (`GET /api/agents`) and server list (`GET /api/servers`) API responses now include `ans_metadata` for each entry. This is reflected in the OpenAPI spec (`openapi.json`). When an agent or server has a linked ANS identity, the full metadata object is returned inline, allowing API consumers to display trust information without making additional calls.

```bash
# List agents - each entry includes ans_metadata when linked
curl -s "${REGISTRY_URL}/api/agents" \
  -H "Authorization: Bearer ${TOKEN}" | python -m json.tool
```

Example agent entry in the list response:

```json
{
  "path": "/jewel-homes-support-agent",
  "name": "Jewel Homes Support Agent",
  "description": "Real estate support agent",
  "ans_metadata": {
    "ans_agent_id": "89a5061b-4f89-452b-9b66-dd9ca8baad7f",
    "status": "verified",
    "domain": "helpagent.club",
    "organization": "Jewel Homes",
    "last_verified": "2026-03-26T12:00:00Z"
  }
}
```

When no ANS identity is linked, `ans_metadata` is `null`.

### ANS Trust Verified in Semantic Search Results

The semantic search API (`GET /api/search`) now returns a `trust_verified` boolean field in each result. This field is derived from `ans_metadata.status == "verified"` and provides a simple flag for consumers to identify agents with valid ANS verification without needing to parse the full metadata.

```bash
# Semantic search - results include trust_verified field
curl -s "${REGISTRY_URL}/api/search?q=real+estate+support" \
  -H "Authorization: Bearer ${TOKEN}" | python -m json.tool
```

Example search result:

```json
{
  "results": [
    {
      "path": "/jewel-homes-support-agent",
      "name": "Jewel Homes Support Agent",
      "description": "Real estate support agent",
      "score": 0.92,
      "trust_verified": true
    },
    {
      "path": "/generic-helper",
      "name": "Generic Helper",
      "description": "General purpose helper",
      "score": 0.78,
      "trust_verified": false
    }
  ]
}
```

This allows search consumers to prioritize or filter results by trust status. Agents with `trust_verified: true` have a valid, non-expired, non-revoked ANS identity.

## Linking During Agent Registration

When registering a new agent, an optional `ans_agent_id` field can be included in the request body. If provided and ANS integration is enabled, the registry will attempt to link and verify the ANS identity as part of registration. This is a best-effort operation -- if ANS verification fails, the agent is still registered and can be linked later via the dedicated endpoint.

```bash
curl -X POST "${REGISTRY_URL}/api/agents" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Agent",
    "path": "/my-agent",
    "description": "An example agent",
    "ans_agent_id": "ans://v1.0.0.myagent.example.com"
  }'
```

## Route Ordering

ANS routes must be registered in `registry/main.py` **before** the agent router because the agent router contains a catch-all `{path:path}` route that would otherwise consume ANS-specific paths like `/agents/{path}/ans/status`.

```python
# In registry/main.py - order matters
app.include_router(ans_router, prefix="/api", tags=["ANS Integration"])    # BEFORE agent_router
app.include_router(agent_router, prefix="/api", tags=["Agent Management"])  # catch-all {path:path} here
```

## Troubleshooting

### ANS Badge Not Showing

1. **Check ANS is enabled:** Verify `ANS_INTEGRATION_ENABLED=true` in environment
2. **Check API credentials:** Verify `ANS_API_KEY` and `ANS_API_SECRET` are set
3. **Check route ordering:** ANS router must be registered before agent router in `main.py`
4. **Check agent has metadata:** `GET /api/agents/{path}` should show `ans_metadata` field
5. **Check API health:** `GET /api/admin/ans/health` should return `healthy`

### ANS API Returning Errors

1. **Check circuit breaker:** If 5+ consecutive failures, circuit opens for 1 hour
2. **Check API endpoint:** Verify `ANS_API_ENDPOINT` points to correct URL
3. **Check credentials:** Test with `GET /api/admin/ans/health`
4. **Check timeout:** Increase `ANS_API_TIMEOUT_SECONDS` if requests are timing out

### Verification Status Stuck on "pending"

This typically means the initial verification call failed or timed out. Try:

1. Unlink: `DELETE /api/agents/{path}/ans/link`
2. Re-link: `POST /api/agents/{path}/ans/link` with the ANS Agent ID
3. Or trigger manual sync: `POST /api/admin/ans/sync`

### Security Scan Routes Returning 404

Similar to ANS routes, security scan routes (`/agents/{path}/security-scan`) must be defined before the catch-all `{path:path}` route in `agent_routes.py`. If these return 404, check route ordering.
