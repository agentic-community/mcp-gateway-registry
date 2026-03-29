# Auth0 M2M Client Management

This guide explains how to manage Auth0 Machine-to-Machine (M2M) client applications and their group mappings in the MCP Gateway Registry.

## Overview

Auth0 M2M tokens do not include groups in the JWT payload (similar to Okta). The MCP Gateway Registry solves this by:

1. **Syncing M2M clients** from Auth0 Management API to MongoDB
2. **Storing group mappings** in the `idp_m2m_clients` collection
3. **Enriching tokens** with groups during authentication

This enables group-based authorization for Auth0 M2M clients without modifying Auth0 configuration.

## Architecture

### Collections

**`auth0_m2m_clients`** (Auth0-specific):
- Stores Auth0 M2M application metadata
- Synced via Auth0 Management API
- Used for listing and managing Auth0 clients

**`idp_m2m_clients`** (Provider-agnostic):
- Generic collection for all IdP providers (Keycloak, Okta, Entra, Auth0)
- Used by auth-server for groups enrichment
- Schema: `{client_id, name, groups, provider, enabled, ...}`

### Flow

```
┌─────────────────┐
│  Auth0 M2M App  │
│  (no groups)    │
└────────┬────────┘
         │ 1. M2M Token (JWT)
         │    - iss: https://domain.auth0.com/
         │    - sub: client_id@clients
         │    - aud: https://domain.auth0.com/api/v2/
         │    - groups: [] (empty)
         │
         v
┌─────────────────────────┐
│   Auth Server           │
│  (validate_token)       │
└────────┬────────────────┘
         │ 2. Groups enrichment
         │    - Query: db.idp_m2m_clients.find_one({client_id})
         │    - Return: ["registry-admins"]
         │
         v
┌─────────────────────────┐
│   Authorization         │
│   (with groups)         │
└─────────────────────────┘
```

## Prerequisites

### 1. Auth0 M2M Application

Create a Machine-to-Machine application in Auth0:

1. Navigate to **Applications** > **Applications** > **Create Application**
2. Select **Machine to Machine Applications**
3. Name it (e.g., "MCP Gateway M2M Sync")
4. Authorize for **Auth0 Management API**
5. Grant required scopes:
   - `read:clients` - Read client applications
   - `read:client_grants` - Read client grants (optional)

### 2. Environment Variables

Configure the following in `.env`:

```bash
# Auth0 M2M credentials for Management API access
AUTH0_DOMAIN=dev-abc123.us.auth0.com
AUTH0_M2M_CLIENT_ID=your_m2m_client_id
AUTH0_M2M_CLIENT_SECRET=your_m2m_client_secret
```

These credentials are used by the sync service to query the Auth0 Management API.

## API Endpoints

### Sync M2M Clients

Fetch all M2M applications from Auth0 and store in MongoDB.

**Request:**
```http
POST /api/iam/auth0/m2m/sync
Authorization: Bearer <admin_token>
Content-Type: application/json

{
  "force_full_sync": false
}
```

**Response:**
```json
{
  "synced_count": 3,
  "added_count": 2,
  "updated_count": 1,
  "removed_count": 0,
  "errors": []
}
```

### List M2M Clients

Get all synced Auth0 M2M clients.

**Request:**
```http
GET /api/iam/auth0/m2m/clients
Authorization: Bearer <token>
```

**Response:**
```json
[
  {
    "client_id": "KhZMijfKUcl2TEJqZzrzVJb8rmwk6Qcd",
    "name": "MCP Gateway M2M",
    "description": "M2M client for registry access",
    "groups": ["registry-admins"],
    "enabled": true,
    "provider": "auth0",
    "created_at": "2026-03-29T00:00:00Z",
    "updated_at": "2026-03-29T00:00:00Z"
  }
]
```

### Get Client Groups

Get groups for a specific M2M client.

**Request:**
```http
GET /api/iam/auth0/m2m/clients/{client_id}/groups
Authorization: Bearer <token>
```

**Response:**
```json
["registry-admins", "public-mcp-users"]
```

### Update Client Groups

Update groups for an M2M client (admin only).

**Request:**
```http
PATCH /api/iam/auth0/m2m/clients/{client_id}/groups
Authorization: Bearer <admin_token>
Content-Type: application/json

{
  "groups": ["registry-admins", "developers"]
}
```

**Response:**
```json
{
  "client_id": "KhZMijfKUcl2TEJqZzrzVJb8rmwk6Qcd",
  "groups": ["registry-admins", "developers"],
  "message": "Groups updated successfully"
}
```

## Usage

### 1. Initial Sync

After configuring Auth0 credentials, perform an initial sync:

```bash
curl -X POST https://registry.example.com/api/iam/auth0/m2m/sync \
  -H "Authorization: Bearer <admin_token>" \
  -H "Content-Type: application/json" \
  -d '{"force_full_sync": true}'
```

This will:
1. Fetch all M2M applications from Auth0
2. Store them in `auth0_m2m_clients` collection
3. Write to `idp_m2m_clients` collection for groups enrichment

### 2. Assign Groups

Update groups for M2M clients:

```bash
curl -X PATCH https://registry.example.com/api/iam/auth0/m2m/clients/{client_id}/groups \
  -H "Authorization: Bearer <admin_token>" \
  -H "Content-Type: application/json" \
  -d '{"groups": ["registry-admins"]}'
```

### 3. Verify Groups Enrichment

Test an M2M token:

```bash
# Get M2M token from Auth0
TOKEN=$(curl -X POST https://dev-abc123.us.auth0.com/oauth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials" \
  -d "client_id=${CLIENT_ID}" \
  -d "client_secret=${CLIENT_SECRET}" \
  -d "audience=https://dev-abc123.us.auth0.com/api/v2/" \
  | jq -r '.access_token')

# Use token to access registry
curl https://registry.example.com/api/servers \
  -H "Authorization: Bearer $TOKEN"
```

The auth-server will:
1. Validate the JWT signature
2. Detect empty groups claim
3. Query `idp_m2m_clients` for the client ID
4. Enrich with groups from database
5. Apply group-based authorization

## Default Groups

You can configure default groups for specific client IDs in `registry/services/auth0_m2m_sync.py`:

```python
DEFAULT_CLIENT_GROUPS = {
    "KhZMijfKUcl2TEJqZzrzVJb8rmwk6Qcd": ["registry-admins"],
    "another_client_id": ["public-mcp-users"],
}
```

These groups are assigned during sync and can be overridden via the API.

## Troubleshooting

### Sync Returns Empty List

**Problem:** No M2M clients found

**Solutions:**
1. Verify Auth0 has M2M applications (app_type: "non_interactive")
2. Check Management API credentials have `read:clients` scope
3. Review logs: `docker logs mcp-gateway-registry 2>&1 | grep "Auth0 M2M"`

### Token Has No Groups

**Problem:** M2M token works but has no authorization

**Solutions:**
1. Verify client is synced: `GET /api/iam/auth0/m2m/clients`
2. Check `idp_m2m_clients` collection in MongoDB:
   ```javascript
   db.idp_m2m_clients.find({ client_id: "your_client_id" })
   ```
3. Assign groups via API: `PATCH /api/iam/auth0/m2m/clients/{id}/groups`
4. Check auth-server logs for groups enrichment messages

### Permission Denied

**Problem:** 403 Forbidden despite having correct groups

**Solutions:**
1. Verify groups are mapped to scopes in `group_to_scope_mappings` collection
2. Check auth-server includes "auth0" in provider list (line 1557 in server.py)
3. Ensure `AUTH_PROVIDER=auth0` in environment variables

## MongoDB Queries

### Check M2M Client

```javascript
db.idp_m2m_clients.find({
  provider: "auth0",
  client_id: "your_client_id"
}).pretty()
```

### Update Groups Manually

```javascript
db.idp_m2m_clients.updateOne(
  { client_id: "your_client_id" },
  {
    $set: {
      groups: ["registry-admins"],
      updated_at: new Date()
    }
  }
)
```

### List All Auth0 M2M Clients

```javascript
db.idp_m2m_clients.find({ provider: "auth0" }).pretty()
```

## Related Documentation

- [Auth0 Management API](https://auth0.com/docs/api/management/v2)
- [Auth0 M2M Applications](https://auth0.com/docs/get-started/applications/application-types#machine-to-machine-applications)
- [Groups Enrichment](../auth_server/mongodb_groups_enrichment.py)
- [Okta M2M Setup](okta-setup.md) - Similar pattern for Okta
