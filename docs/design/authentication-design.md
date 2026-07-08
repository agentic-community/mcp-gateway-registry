# Authentication and Authorization Design

**Version:** 1.0
**Last Updated:** 2026-01-18

## Related Documentation

- [Browser Login: Cookie + Session Flow](./session-flow-cookie-based.md) - End-to-end cookie/session design (post-#1042 server-side store)
- [JWT / Bearer Token Login Flow](./session-flow-jwt-bearer.md) - End-to-end JWT/Bearer design (programmatic, M2M, federation)
- [Multi-Provider IdP Support](./idp-provider-support.md) - Architecture for supporting multiple identity providers
- [Internal Hop Authentication](./internal-hop-authentication.md) - How components authenticate to each other via signed inter-component tokens (#1260 / #1262)
- [Authentication & Authorization Guide](../auth.md) - Operational guide with setup instructions
- [Microsoft Entra ID Integration](../entra.md) - Entra ID-specific setup and configuration

## Overview

The MCP Gateway Registry implements a comprehensive authentication and authorization system supporting three distinct identity scenarios:

1. **Human Users** - Interactive users accessing the Registry UI and generating API tokens
2. **Programmatic Access (API Tokens)** - Self-signed JWT tokens for CLI tools and AI coding assistants
3. **Workload Identity (M2M)** - Service accounts for AI agents and automated systems

## Identity Types

```
+------------------+------------------+------------------+
|   Human Users    | Programmatic     | Workload         |
|                  | Access           | Identity (M2M)   |
+------------------+------------------+------------------+
|                  |                  |                  |
| - Interactive    | - CLI tools      | - AI Agents      |
|   browser login  | - AI coding      | - Automated      |
| - OAuth2 flow    |   assistants     |   pipelines      |
| - Session-based  | - Scripts        | - Service-to-    |
|                  |                  |   service        |
+------------------+------------------+------------------+
|                  |                  |                  |
| Auth Method:     | Auth Method:     | Auth Method:     |
| Authorization    | Self-signed      | OAuth2 Client    |
| Code Flow        | JWT (HS256)      | Credentials Flow |
|                  |                  | (RS256)          |
+------------------+------------------+------------------+
```

---

## Part 1: Human User Authentication

### 1.1 OAuth2 Authorization Code Flow

Human users authenticate via the configured identity provider (Keycloak or Entra ID) using the standard OAuth2 Authorization Code flow.

```
+-------------+     +--------------+     +--------------+     +-------------+
|   Browser   |     |  Registry    |     | Auth Server  |     | Identity    |
|   (User)    |     |  Frontend    |     |              |     | Provider    |
+------+------+     +------+-------+     +------+-------+     +------+------+
       |                   |                    |                    |
       | 1. Click "Login"  |                    |                    |
       +------------------>|                    |                    |
       |                   |                    |                    |
       | 2. Redirect to    |                    |                    |
       |    Auth Server    |                    |                    |
       |<------------------+                    |                    |
       |                   |                    |                    |
       | 3. GET /oauth2/login/entra             |                    |
       +--------------------------------------->|                    |
       |                   |                    |                    |
       |         4. Redirect to IdP authorize endpoint               |
       |<-----------------------------------------------------------+
       |                   |                    |                    |
       |                   5. User authenticates with IdP            |
       +------------------------------------------------------------>|
       |<------------------------------------------------------------+
       |                   |                    |                    |
       | 6. Redirect with authorization code    |                    |
       +--------------------------------------->|                    |
       |                   |                    |                    |
       |                   |  7. Exchange code  |                    |
       |                   |     for tokens     |                    |
       |                   |                    +------------------->|
       |                   |                    |<-------------------+
       |                   |                    |  (ID token +       |
       |                   |                    |   access token)    |
       |                   |                    |                    |
       |    8. Set session cookie + redirect    |                    |
       |<---------------------------------------+                    |
       |                   |                    |                    |
       | 9. Access Registry with session cookie |                    |
       +------------------>|                    |                    |
       |                   |                    |                    |
```

### 1.2 Session Data

After successful authentication, the auth server creates a session containing:

```json
{
  "user_id": "user@example.onmicrosoft.com",
  "email": "user@example.com",
  "groups": ["5f605d68-06bc-4208-b992-bb378eee12c5"],
  "provider": "entra",
  "scopes": ["public-mcp-users"],
  "is_admin": false,
  "ui_permissions": {
    "list_service": ["all"],
    "list_agents": ["/flight-booking"],
    "get_agent": ["/flight-booking"]
  }
}
```

### 1.3 UI Permission Enforcement

The Registry UI enforces feature access based on `ui_permissions` from the user's mapped scopes.

#### Example: `public-mcp-users` Scope

From `cli/examples/public-mcp-users.json`:

```json
{
  "scope_name": "public-mcp-users",
  "ui_permissions": {
    "list_service": ["all"],
    "list_agents": ["/flight-booking"],
    "get_agent": ["/flight-booking"]
  }
}
```

**What this user CAN do:**
- View all MCP servers in the dashboard (`list_service: ["all"]`)
- View the flight-booking agent details (`list_agents`, `get_agent` for `/flight-booking`)
- Access public MCP servers: context7, cloudflare-docs (via `server_access` rules)

**What this user CANNOT do:**
- Publish, modify, or delete agents
- Register or modify MCP servers
- Toggle services on/off
- Access health check for all servers (only context7, cloudflare-docs)
- Access IAM management features

#### Example: `registry-admins` Scope

Admins have unrestricted access to all UI features:

```yaml
# From the mcp_scopes collection in DocumentDB (seeded from JSON scope files in scripts/)
UI-Scopes:
  registry-admins:
    list_agents: [all]
    get_agent: [all]
    publish_agent: [all]
    modify_agent: [all]
    delete_agent: [all]
    list_service: [all]
    register_service: [all]
    health_check_service: [all]
    toggle_service: [all]
    modify_service: [all]
```

### 1.4 Frontend Permission Checks

The frontend checks `ui_permissions` before rendering features:

```typescript
// From Dashboard.tsx
const hasUiPermission = useCallback((permission: string, servicePath: string): boolean => {
  const permissions = user?.ui_permissions?.[permission];
  if (!permissions) return false;

  const serviceName = servicePath.replace(/^\//, '');
  return permissions.includes('all') || permissions.includes(serviceName);
}, [user?.ui_permissions]);

// Usage in JSX
<ServerCard
  canHealthCheck={hasUiPermission('health_check_service', server.path)}
  canToggle={hasUiPermission('toggle_service', server.path)}
  // ...
/>
```

---

## Part 2: Programmatic Access (Self-Signed JWT Tokens)

Human users can generate API tokens for programmatic access (CLI tools, AI coding assistants) via the "Get JWT Token" button in the UI.

### 2.1 Token Generation Flow

```
+-------------+     +--------------+     +--------------+     +-------------+
|   Browser   |     |  Registry    |     | Auth Server  |     | MongoDB     |
|   (User)    |     |  Backend     |     |              |     | (Scopes)    |
+------+------+     +------+-------+     +------+-------+     +------+------+
       |                   |                    |                    |
       | 1. Click "Get JWT Token"               |                    |
       +------------------>|                    |                    |
       |                   |                    |                    |
       |                   | 2. POST /api/tokens/generate            |
       |                   |    (with session cookie)                |
       |                   +------------------->|                    |
       |                   |                    |                    |
       |                   |   3. Validate session                   |
       |                   |   Extract: username, groups, provider   |
       |                   |                    |                    |
       |                   |                    | 4. Query group     |
       |                   |                    |    mappings        |
       |                   |                    +------------------->|
       |                   |                    |<-------------------+
       |                   |                    |  (scopes for       |
       |                   |                    |   user's groups)   |
       |                   |                    |                    |
       |                   |   5. Build JWT claims:                  |
       |                   |   - iss: mcp-auth-server                |
       |                   |   - aud: mcp-registry                   |
       |                   |   - sub: username                       |
       |                   |   - groups: [group IDs]                 |
       |                   |   - scope: mapped scopes                |
       |                   |   - exp: 8 hours                        |
       |                   |                    |                    |
       |                   |   6. Sign JWT with SECRET_KEY (HS256)   |
       |                   |                    |                    |
       |                   | 7. Return JWT      |                    |
       |                   |<-------------------+                    |
       |                   |                    |                    |
       | 8. Display token  |                    |                    |
       |<------------------+                    |                    |
```

### 2.2 Self-Signed JWT Structure

```json
{
  "iss": "mcp-auth-server",
  "aud": "mcp-registry",
  "sub": "user@example.onmicrosoft.com",
  "preferred_username": "user@example.onmicrosoft.com",
  "email": "user@example.com",
  "groups": ["5f605d68-06bc-4208-b992-bb378eee12c5"],
  "scope": "public-mcp-users",
  "token_use": "access",
  "auth_method": "oauth2",
  "provider": "entra",
  "iat": 1768685565,
  "exp": 1768714365,
  "description": "Generated via sidebar"
}
```

### 2.3 Using the Token with CLI Tools

```bash
# Save token to file
echo "eyJhbGciOiJIUzI1NiIs..." > .token

# Use with registry_management.py
uv run python api/registry_management.py \
  --token-file .token \
  --registry-url http://localhost \
  server-search --query "documentation"

# Use with curl
curl -H "Authorization: Bearer $(cat .token)" \
  http://localhost/api/servers
```

### 2.4 Token Validation Flow (API Usage)

```
+-------------+     +--------------+     +--------------+     +-------------+
|   CLI /     |     |    NGINX     |     | Auth Server  |     | MCP Server  |
|   Client    |     |   Gateway    |     |              |     |             |
+------+------+     +------+-------+     +------+-------+     +------+------+
       |                   |                    |                    |
       | 1. API Request    |                    |                    |
       |    Authorization: Bearer <JWT>         |                    |
       +------------------>|                    |                    |
       |                   |                    |                    |
       |                   | 2. auth_request /validate               |
       |                   +------------------->|                    |
       |                   |                    |                    |
       |                   |   3. Check token issuer                 |
       |                   |   iss == "mcp-auth-server"?             |
       |                   |                    |                    |
       |                   |   4. If yes: validate with              |
       |                   |      SECRET_KEY (HS256)                 |
       |                   |                    |                    |
       |                   |   5. If no: try IdP JWKS                |
       |                   |      validation (RS256)                 |
       |                   |                    |                    |
       |                   |   6. Extract scopes, validate           |
       |                   |      server/tool access                 |
       |                   |                    |                    |
       |                   | 7. 200 OK + X-User headers              |
       |                   |<-------------------+                    |
       |                   |                    |                    |
       |                   | 8. Proxy request   |                    |
       |                   +--------------------------------------->|
       |                   |                    |                    |
       | 9. Response       |                    |                    |
       |<------------------+                    |                    |
```

---

## Part 3: Workload Identity (M2M / Service Accounts)

AI agents and automated systems use service accounts with client credentials for authentication.

### 3.1 M2M Identity in Identity Providers

```
+---------------------------+---------------------------+
|        Keycloak           |      Microsoft Entra ID   |
+---------------------------+---------------------------+
|                           |                           |
| Service Account Client:   | App Registration:         |
| - Client ID               | - Application (client) ID |
| - Client Secret           | - Client Secret           |
| - Service Account User    | - Service Principal       |
| - Group Memberships       | - Group Memberships       |
|                           |                           |
+---------------------------+---------------------------+
```

### 3.2 M2M Account Creation Flow (Entra ID)

```
+-------------+     +--------------+     +--------------+     +-------------+
|   Admin     |     |  Registry    |     | Entra ID     |     | MongoDB     |
|   CLI       |     |  Backend     |     | Graph API    |     |             |
+------+------+     +------+-------+     +------+-------+     +------+------+
       |                   |                    |                    |
       | 1. user-create-m2m --name pub-m2m-bot --groups public-mcp-users
       +------------------>|                    |                    |
       |                   |                    |                    |
       |                   | 2. Create App Registration              |
       |                   +------------------->|                    |
       |                   |<-------------------+                    |
       |                   |   (app_id, object_id)                   |
       |                   |                    |                    |
       |                   | 3. Create Service Principal             |
       |                   +------------------->|                    |
       |                   |<-------------------+                    |
       |                   |   (service_principal_id)                |
       |                   |                    |                    |
       |                   | 4. Create Client Secret                 |
       |                   +------------------->|                    |
       |                   |<-------------------+                    |
       |                   |   (client_secret)                       |
       |                   |                    |                    |
       |                   | 5. Add SP to group (with retry)         |
       |                   +------------------->|                    |
       |                   |<-------------------+                    |
       |                   |                    |                    |
       | 6. Return credentials                  |                    |
       |   client_id, client_secret             |                    |
       |<------------------+                    |                    |
```

### 3.3 M2M Token Request Flow

AI agents use OAuth2 Client Credentials flow to obtain access tokens:

```
+-------------+     +--------------+     +--------------+
| AI Agent    |     | Identity     |     | Registry     |
| (M2M)       |     | Provider     |     | API          |
+------+------+     +------+-------+     +------+-------+
       |                   |                    |
       | 1. POST /oauth2/v2.0/token             |
       |    grant_type=client_credentials       |
       |    client_id=...                       |
       |    client_secret=...                   |
       |    scope=api://.../.default            |
       +------------------>|                    |
       |                   |                    |
       | 2. Access Token (RS256, 1 hour)        |
       |<------------------+                    |
       |                   |                    |
       | 3. API Request with token              |
       +--------------------------------------->|
       |                   |                    |
       |                   |   4. Validate via  |
       |                   |      IdP JWKS      |
       |                   |                    |
       | 5. Response       |                    |
       |<--------------------------------------+
```

### 3.4 Generating M2M Tokens

Use the credentials provider script to generate tokens:

**Identities File** (`.oauth-tokens/entra-identities.json`):

```json
[
  {
    "identity_name": "pub-m2m-bot",
    "tenant_id": "6e6ee81b-6bf3-495d-a7fc-d363a551f765",
    "client_id": "c50b03cf-6f7b-4fae-846e-7910a4100020",
    "client_secret": "your-client-secret",
    "scope": "api://1bd17ba1-aad3-447f-be0b-26f8f9ee859f/.default"
  }
]
```

**Generate Token:**

```bash
cd credentials-provider
uv run python entra/generate_tokens.py \
  --identities-file ../.oauth-tokens/entra-identities.json \
  --output-dir ../.oauth-tokens
```

**Output** (`.oauth-tokens/pub-m2m-bot.json`):

```json
{
  "identity_name": "pub-m2m-bot",
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIs...",
  "token_type": "Bearer",
  "expires_in": 3599,
  "generated_at": "2026-01-18T02:30:56.123456+00:00",
  "expires_at": "2026-01-18T03:30:55.123456+00:00",
  "provider": "entra",
  "tenant_id": "6e6ee81b-6bf3-495d-a7fc-d363a551f765",
  "client_id": "c50b03cf-6f7b-4fae-846e-7910a4100020",
  "scope": "api://1bd17ba1-aad3-447f-be0b-26f8f9ee859f/.default"
}
```

### 3.5 M2M Token Structure (from Entra ID)

```json
{
  "aud": "api://1bd17ba1-aad3-447f-be0b-26f8f9ee859f",
  "iss": "https://sts.windows.net/6e6ee81b-6bf3-495d-a7fc-d363a551f765/",
  "iat": 1768703056,
  "nbf": 1768703056,
  "exp": 1768706956,
  "appid": "c50b03cf-6f7b-4fae-846e-7910a4100020",
  "appidacr": "1",
  "groups": ["5f605d68-06bc-4208-b992-bb378eee12c5"],
  "idp": "https://sts.windows.net/6e6ee81b-6bf3-495d-a7fc-d363a551f765/",
  "oid": "5d3d562c-4449-413a-9791-86920d4bf75f",
  "sub": "5d3d562c-4449-413a-9791-86920d4bf75f",
  "tid": "6e6ee81b-6bf3-495d-a7fc-d363a551f765",
  "ver": "1.0"
}
```

**Key Differences from Self-Signed Tokens:**

| Aspect | Self-Signed (Human) | IdP Token (M2M) |
|--------|---------------------|-----------------|
| Issuer | `mcp-auth-server` | `https://sts.windows.net/{tenant}/` |
| Algorithm | HS256 (symmetric) | RS256 (asymmetric) |
| Validation | SECRET_KEY | IdP JWKS endpoint |
| Expiry | 8 hours | 1 hour |
| Subject | username/email | Service principal object ID |

---

## Part 4: Authorization - Scope-Based Access Control

### 4.1 Scope Storage in MongoDB-CE/Amazon DocumentDB

Scopes are stored in the `mcp_scopes_default` collection:

```
+-----------------------------------------------------------------------+
|                    mcp_scopes_default collection                       |
+-----------------------------------------------------------------------+
| {                                                                     |
|   "_id": "public-mcp-users",                                          |
|   "group_mappings": [                                                 |
|     "public-mcp-users",                        <-- Keycloak group     |
|     "5f605d68-06bc-4208-b992-bb378eee12c5"     <-- Entra ID Object ID |
|   ],                                                                  |
|   "server_access": [                                                  |
|     {                                                                 |
|       "server": "context7",                                           |
|       "methods": ["initialize", "tools/list", "tools/call"],          |
|       "tools": ["*"]                                                  |
|     },                                                                |
|     {                                                                 |
|       "server": "api",                                                |
|       "methods": ["initialize", "GET", "POST", "servers", ...],       |
|       "tools": []                                                     |
|     }                                                                 |
|   ],                                                                  |
|   "ui_permissions": {                                                 |
|     "list_service": ["all"],                                          |
|     "list_agents": ["/flight-booking"],                               |
|     "get_agent": ["/flight-booking"]                                  |
|   }                                                                   |
| }                                                                     |
+-----------------------------------------------------------------------+
```

### 4.2 Group-to-Scope Mapping Query

The auth server queries MongoDB-CE/Amazon DocumentDB to find scopes for a given group:

```python
# From scope_repository.py
async def get_group_mappings(self, keycloak_group: str) -> List[str]:
    """Find all scopes where group_mappings array contains this group."""
    collection = await self._get_collection()
    cursor = collection.find({"group_mappings": keycloak_group})
    scope_names = [doc["_id"] async for doc in cursor]
    return scope_names
```

### 4.3 Server/Tool Access Validation

```
+------------------+     +------------------+     +------------------+
|                  |     |                  |     |                  |
|  Request:        |     |  User Scopes:    |     |  Access          |
|  POST /context7  |     |  [public-mcp-    |     |  Decision:       |
|  tools/call      |     |   users]         |     |  GRANTED         |
|                  |     |                  |     |                  |
+--------+---------+     +--------+---------+     +--------+---------+
         |                        |                        |
         |   1. Extract server    |                        |
         |      and method        |                        |
         +----------------------->|                        |
         |                        |                        |
         |   2. For each scope,   |                        |
         |      check server_     |                        |
         |      access rules      |                        |
         |                        +----------------------->|
         |                        |                        |
         |   3. public-mcp-users  |                        |
         |      allows context7   |                        |
         |      with tools/call   |                        |
         |                        |                        |
```

**Validation Logic** (from `server.py`):

```python
def validate_server_tool_access(
    server_name: str,
    method: str,
    tool_name: Optional[str],
    user_scopes: List[str]
) -> bool:
    """Check if user has access to server/method/tool."""
    for scope_name in user_scopes:
        scope_config = get_scope_config(scope_name)
        for server_rule in scope_config.get("server_access", []):
            # Check server name match (exact or wildcard)
            if server_rule["server"] in (server_name, "*"):
                # Check method is allowed
                if method in server_rule["methods"] or "all" in server_rule["methods"]:
                    # Check tool access if specified
                    if tool_name is None or server_rule["tools"] == "*":
                        return True
                    if tool_name in server_rule["tools"]:
                        return True
    return False
```

---

## Summary Comparison

| Aspect | Human User | Programmatic (API Token) | M2M (Workload) |
|--------|------------|--------------------------|----------------|
| **Use Case** | Browser UI | CLI, AI assistants | AI agents, automation |
| **Auth Flow** | OAuth2 Authorization Code | N/A (derived from session) | OAuth2 Client Credentials |
| **Token Issuer** | IdP (Keycloak/Entra) | `mcp-auth-server` | IdP (Keycloak/Entra) |
| **Token Signing** | RS256 (IdP) | HS256 (SECRET_KEY) | RS256 (IdP) |
| **Token Lifetime** | Session-based | 8 hours | 1 hour |
| **Validation** | Session cookie | SECRET_KEY | IdP JWKS |
| **Groups Source** | IdP groups claim | Copied from session | IdP groups claim |
| **Scope Mapping** | Groups -> Datastore scopes | Embedded in token | Groups -> Datastore scopes |
| **Credential Storage** | Browser session | Token file | Client ID/Secret |

---

## Related Documentation

- [Multi-Provider IdP Support](./idp-provider-support.md)
- [Microsoft Entra ID Integration](../entra.md)
- [Management API Testing Guide](https://github.com/agentic-community/mcp-gateway-registry/blob/main/api/test-management-api-e2e.md)
