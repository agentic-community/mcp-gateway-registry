# Multi-Provider Identity Provider (IdP) Support

**Version:** 1.0
**Last Updated:** 2026-01-18

## Related Documentation

- [Authentication Design](./authentication-design.md) - Auth flows for human users, programmatic access, and M2M workloads
- [Authentication & Authorization Guide](../auth.md) - Operational guide with setup instructions
- [Microsoft Entra ID Integration](../entra.md) - Entra ID-specific setup and configuration
- [Okta Integration](../okta-setup.md) - Okta-specific setup and configuration

## Overview

The MCP Gateway Registry supports multiple identity providers (IdPs) through a pluggable architecture. This design enables organizations to use their existing enterprise identity infrastructure (Keycloak, Microsoft Entra ID, Okta) for authentication and authorization.

## Architecture

### High-Level Component Diagram

```
+------------------+     +------------------+     +------------------+
|                  |     |                  |     |                  |
|   Registry UI    |     |   CLI Tools      |     |   AI Agents      |
|   (Frontend)     |     | (registry_mgmt)  |     |  (M2M Clients)   |
|                  |     |                  |     |                  |
+--------+---------+     +--------+---------+     +--------+---------+
         |                        |                        |
         |   HTTP + JWT Token     |   HTTP + JWT Token     |
         |                        |                        |
         v                        v                        v
+--------+------------------------+------------------------+---------+
|                                                                    |
|                         NGINX Gateway                              |
|                     (auth_request /validate)                       |
|                                                                    |
+--------+-----------------------------------------------------------+
         |
         | /validate
         v
+--------+-----------------------------------------------------------+
|                                                                    |
|                         Auth Server                                |
|                                                                    |
|  +----------------+    +------------------+    +----------------+  |
|  |                |    |                  |    |                |  |
|  | AuthProvider   |    | AuthProvider     |    | AuthProvider   |  |
|  | Factory        +--->+ Protocol         +--->+ Implementations|  |
|  |                |    | (Base Class)     |    |                |  |
|  +----------------+    +------------------+    +----------------+  |
|                                                      |             |
|                           +----------------+---------+--------+    |
|                           |                |                  |    |
|                           v                v                  v    |
|                    +------+------+  +------+------+  +--------+-+  |
|                    |             |  |             |  |           |  |
|                    |  Keycloak   |  |  Entra ID   |  |   Okta    |  |
|                    |  Provider   |  |  Provider   |  |  Provider |  |
|                    |             |  |             |  |           |  |
|                    +-------------+  +-------------+  +-----------+  |
|                                                                    |
+--------------------------------------------------------------------+
         |
         | Group-to-Scope Mapping
         v
+--------+-----------------------------------------------------------+
|                                                                    |
|               MongoDB-CE / Amazon DocumentDB                       |
|                                                                    |
|  +---------------------------+  +---------------------------+      |
|  | mcp_scopes_default        |  | mcp_servers_default       |      |
|  |                           |  |                           |      |
|  | - scope definitions       |  | - server configurations   |      |
|  | - group_mappings          |  | - tool definitions        |      |
|  | - server_access rules     |  |                           |      |
|  | - ui_permissions          |  |                           |      |
|  +---------------------------+  +---------------------------+      |
|                                                                    |
+--------------------------------------------------------------------+
```

## Provider Selection

The active identity provider is determined by the `AUTH_PROVIDER` environment variable:

```
AUTH_PROVIDER=keycloak   # Use Keycloak
AUTH_PROVIDER=entra      # Use Microsoft Entra ID
AUTH_PROVIDER=okta       # Use Okta
```

### Provider Factory Pattern

```
+-------------------+     +--------------------------+
|                   |     |                          |
| AUTH_PROVIDER env +---->+  AuthProviderFactory     |
|                   |     |                          |
+-------------------+     +------------+-------------+
                                       |
                +----------------------+----------------------+
                |                      |                      |
                v                      v                      v
    +-------------------+  +-----------------------+  +-----------------------+
    |                   |  |                       |  |                       |
    | KeycloakProvider  |  | EntraIdProvider       |  | OktaProvider          |
    |                   |  |                       |  |                       |
    | - OIDC endpoints  |  | - Microsoft Graph API |  | - Okta OAuth2/OIDC    |
    | - JWKS validation |  | - JWKS validation     |  | - JWKS validation     |
    | - Realm-based     |  | - Tenant-based        |  | - Domain-based        |
    |                   |  |                       |  |                       |
    +-------------------+  +-----------------------+  +-----------------------+
```

## IAM Manager Interface

For administrative operations (user/group CRUD), the system uses the IAM Manager abstraction:

```python
@runtime_checkable
class IAMManager(Protocol):
    """Protocol defining the IAM manager interface."""

    async def list_users(
        self,
        search: str | None = None,
        max_results: int = 500,
        include_groups: bool = True
    ) -> list[dict[str, Any]]: ...

    async def create_human_user(
        self,
        username: str,
        email: str,
        first_name: str,
        last_name: str,
        groups: list[str],
        password: str | None = None,
    ) -> dict[str, Any]: ...

    async def delete_user(self, username: str) -> bool: ...

    async def list_groups(self) -> list[dict[str, Any]]: ...

    async def create_group(
        self,
        group_name: str,
        description: str = ""
    ) -> dict[str, Any]: ...

    async def delete_group(self, group_name: str) -> bool: ...

    async def create_service_account(
        self,
        client_id: str,
        groups: list[str],
        description: str | None = None
    ) -> dict[str, Any]: ...
```

### Implementation Classes

```
+------------------+          +------------------+          +------------------+
|                  |          |                  |          |                  |
| KeycloakIAM      |          | EntraIAM         |          | OktaIAM          |
| Manager          |          | Manager          |          | Manager          |
|                  |          |                  |          |                  |
+--------+---------+          +--------+---------+          +--------+---------+
         |                             |                             |
         | Delegates to                | Delegates to                | Delegates to
         v                             v                             v
+--------+---------+          +--------+---------+          +--------+---------+
|                  |          |                  |          |                  |
| keycloak_manager |          | entra_manager    |          | okta_manager     |
| .py              |          | .py              |          | .py              |
|                  |          |                  |          |                  |
| - Keycloak Admin |          | - Microsoft      |          | - Okta Admin     |
|   REST API       |          |   Graph API      |          |   REST API       |
| - Realm mgmt     |          | - App registr.   |          | - SSWS auth      |
| - Client mgmt    |          | - Service        |          | - OIDC service   |
|                  |          |   principals     |          |   apps           |
+------------------+          +------------------+          +------------------+
```

## Provider-Specific Details

### Keycloak Provider

**Authentication Flow:**
- Uses OIDC Authorization Code flow
- Tokens issued by Keycloak realm
- JWKS endpoint: `{keycloak_url}/realms/{realm}/protocol/openid-connect/certs`

**Group Identifier in Tokens:**
- Group names (e.g., `registry-admins`, `public-mcp-users`)
- Stored in `groups` claim of JWT

**IAM Operations:**
- Uses Keycloak Admin REST API
- Requires admin credentials or service account with realm-admin role

### Microsoft Entra ID Provider

**Authentication Flow:**
- Uses OAuth2 Authorization Code flow (users)
- Uses OAuth2 Client Credentials flow (M2M)
- Tokens issued by Microsoft STS
- JWKS endpoint: `https://login.microsoftonline.com/{tenant_id}/discovery/v2.0/keys`

**Group Identifier in Tokens:**
- Group Object IDs (GUIDs) like `5f605d68-06bc-4208-b992-bb378eee12c5`
- Stored in `groups` claim of JWT
- Object IDs must be mapped to scope names in MongoDB

**IAM Operations:**
- Uses Microsoft Graph API
- Requires App Registration with appropriate permissions:
  - `Application.ReadWrite.All`
  - `Directory.ReadWrite.All`
  - `Group.ReadWrite.All`
  - `User.ReadWrite.All`

### Okta Provider

**Authentication Flow:**
- Uses OAuth2 Authorization Code flow (users)
- Uses OAuth2 Client Credentials flow (M2M)
- Tokens issued by Okta org authorization server
- JWKS endpoint: `https://{okta_domain}/oauth2/v1/keys`

**Group Identifier in Tokens:**
- Group names (e.g., `mcp-admin`, `mcp-user`) — similar to Keycloak
- Stored in `groups` claim of JWT
- Requires groups claim to be configured in the Okta Authorization Server

**Key Differences from Other Providers:**
- Single issuer format: `https://{okta_domain}` (unlike Entra ID's dual v1.0/v2.0)
- Uses `scp` claim for scopes in access tokens (fallback to `scope`)
- Uses `cid` claim for client ID
- Admin API uses a separate API token (`SSWS` scheme), not OAuth2 credentials

**IAM Operations:**
- Uses Okta Admin REST API (`/api/v1/*`)
- Requires dedicated API token (`OKTA_API_TOKEN`) with `SSWS` authorization
- User deletion requires deactivate-then-delete two-step flow
- See [Okta Setup Guide](../okta-setup.md) for configuration details

## Group-to-Scope Mapping

The mapping between IdP groups and registry scopes is stored in MongoDB-CE/Amazon DocumentDB (`mcp_scopes_default` collection):

```
+---------------------------------------------------+
| MongoDB-CE/Amazon DocumentDB: mcp_scopes_default  |
+---------------------------------------------------+
| Document Structure:                               |
|                                                   |
| {                                         |
|   "_id": "registry-admins",               |  <-- Scope name
|   "group_mappings": [                     |
|     "registry-admins",                    |  <-- Keycloak group name
|     "4c46ec66-a4f7-4b62-9095-..."         |  <-- Entra ID group Object ID
|   ],                                      |
|   "server_access": [ ... ],               |  <-- MCP server permissions
|   "ui_permissions": { ... }               |  <-- UI feature access
| }                                         |
+-------------------------------------------+
```

### Mapping Flow

```
+------------------+     +------------------+     +------------------+
|                  |     |                  |     |                  |
|  JWT Token       |     |  Scope           |     |  Access          |
|  from IdP        +---->+  Repository      +---->+  Decision        |
|                  |     |                  |     |                  |
+------------------+     +------------------+     +------------------+
        |                        |                        |
        | groups claim:          | Query:                 | Result:
        | ["5f605d68-..."]       | Find scopes where      | scopes=["public-
        |                        | group_mappings         |  mcp-users"]
        |                        | contains "5f605d68-"   |
        v                        v                        v

Keycloak example:              Entra ID example:
groups: ["public-mcp-users"]   groups: ["5f605d68-06bc-4208-b992-bb378eee12c5"]
        |                              |
        +------------------------------+
                      |
                      v
              +-------+-------+
              |               |
              | Mapped to:    |
              | public-mcp-   |
              | users scope   |
              |               |
              +---------------+
```

## Configuration

### Environment Variables

```bash
# Provider Selection
AUTH_PROVIDER=entra              # or "keycloak" or "okta"

# Keycloak Configuration
KEYCLOAK_URL=https://keycloak.example.com
KEYCLOAK_REALM=mcp-gateway
KEYCLOAK_CLIENT_ID=mcp-registry
KEYCLOAK_CLIENT_SECRET=...

# Entra ID Configuration
ENTRA_TENANT_ID=6e6ee81b-6bf3-495d-a7fc-d363a551f765
ENTRA_CLIENT_ID=1bd17ba1-aad3-447f-be0b-26f8f9ee859f
ENTRA_CLIENT_SECRET=...

# Okta Configuration
OKTA_DOMAIN=dev-123456.okta.com
OKTA_CLIENT_ID=0oa1234567890abcdef
OKTA_CLIENT_SECRET=...
# OKTA_M2M_CLIENT_ID=...        # Optional separate M2M credentials
# OKTA_M2M_CLIENT_SECRET=...
# OKTA_API_TOKEN=...             # Optional, for IAM operations

# Token Validation
SECRET_KEY=...                   # For self-signed tokens
JWT_ISSUER=mcp-auth-server
JWT_AUDIENCE=mcp-registry
```

### Scopes Configuration (the `mcp_scopes` collection in DocumentDB / MongoDB)

```yaml
# Group mappings - maps IdP group identifiers to scope names
group_mappings:
  # Entra ID uses Object IDs (GUIDs)
  "4c46ec66-a4f7-4b62-9095-b7958662f4b6":
    - registry-admins
    - mcp-servers-unrestricted/read
    - mcp-servers-unrestricted/execute

  "5f605d68-06bc-4208-b992-bb378eee12c5":
    - public-mcp-users

  # Keycloak uses group names
  "registry-admins":
    - registry-admins

  "public-mcp-users":
    - public-mcp-users

  # Okta also uses group names (same format as Keycloak)
  "mcp-admin":
    - registry-admins

  "mcp-user":
    - public-mcp-users
```

## Adding a New Provider

To add support for a new identity provider:

1. **Create Provider Class** (`auth_server/providers/new_provider.py`):
   - Implement `AuthProvider` base class
   - Handle OIDC/OAuth2 flows
   - Implement token validation via JWKS

2. **Create IAM Manager** (`registry/utils/new_provider_manager.py`):
   - Implement user/group CRUD operations
   - Handle provider-specific API calls

3. **Update Factory** (`registry/utils/iam_manager.py`):
   - Add new provider case to `get_iam_manager()`

4. **Update Auth Factory** (`auth_server/providers/factory.py`):
   - Add new provider case to factory function

5. **Configure Group Mappings**:
   - Add group identifiers to the `mcp_scopes` collection in MongoDB-CE/Amazon DocumentDB (seeded from JSON scope files in `scripts/` or managed via the scope management API)
   - Document group identifier format (names vs IDs)

## Security Considerations

1. **Token Validation**: Always validate JWT signatures against provider JWKS
2. **Admin Credentials**: Store IdP admin credentials securely (environment variables, secrets manager)
3. **Principle of Least Privilege**: Request minimal permissions for IAM operations
4. **Eventual Consistency**: Handle Entra ID's eventual consistency with retry logic
5. **Token Expiry**: Respect token expiration times; implement refresh where needed
