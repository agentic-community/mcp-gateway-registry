# Microsoft Entra ID Integration for MCP Gateway Registry

This document describes the integration between Microsoft Entra ID and the MCP Gateway Registry, including the JWT token generation flow for programmatic API access.

## Overview

The MCP Gateway Registry supports Microsoft Entra ID as an OAuth2 identity provider. Users can authenticate via Entra ID and obtain JWT tokens for programmatic access to the gateway APIs (CLI tools, coding assistants, etc.).

## Architecture

### Authentication Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Browser   │     │  Registry   │     │ Auth Server │     │  Entra ID   │
│   (User)    │     │  Frontend   │     │             │     │  (Microsoft)│
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │                   │
       │  1. Click Login   │                   │                   │
       │──────────────────>│                   │                   │
       │                   │                   │                   │
       │  2. Redirect to Auth Server          │                   │
       │<──────────────────│                   │                   │
       │                   │                   │                   │
       │  3. /oauth2/login/entra              │                   │
       │──────────────────────────────────────>│                   │
       │                   │                   │                   │
       │  4. Redirect to Entra ID authorize endpoint              │
       │<─────────────────────────────────────────────────────────>│
       │                   │                   │                   │
       │  5. User authenticates with Microsoft │                   │
       │<─────────────────────────────────────────────────────────>│
       │                   │                   │                   │
       │  6. Redirect with auth code           │                   │
       │──────────────────────────────────────>│                   │
       │                   │                   │                   │
       │                   │  7. Exchange code │                   │
       │                   │  for tokens       │                   │
       │                   │                   │──────────────────>│
       │                   │                   │<──────────────────│
       │                   │                   │  (ID token +      │
       │                   │                   │   access token)   │
       │                   │                   │                   │
       │  8. Set session cookie + redirect     │                   │
       │<──────────────────────────────────────│                   │
       │                   │                   │                   │
       │  9. Access Registry with session      │                   │
       │──────────────────>│                   │                   │
       │                   │                   │                   │
```

### JWT Token Generation Flow (Get JWT Token Button)

When an OAuth-authenticated user clicks "Get JWT Token" in the UI:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Browser   │     │  Registry   │     │ Auth Server │     │  DocumentDB │
│   (User)    │     │  Backend    │     │             │     │  (Scopes)   │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │                   │
       │  1. Click "Get JWT Token"            │                   │
       │──────────────────>│                   │                   │
       │                   │                   │                   │
       │                   │  2. POST /api/tokens/generate        │
       │                   │  (with session cookie)               │
       │                   │──────────────────>│                   │
       │                   │                   │                   │
       │                   │                   │  3. Validate session
       │                   │                   │  Extract: username,
       │                   │                   │  groups, provider  │
       │                   │                   │                   │
       │                   │                   │  4. Query group    │
       │                   │                   │  mappings          │
       │                   │                   │──────────────────>│
       │                   │                   │<──────────────────│
       │                   │                   │  (scopes for      │
       │                   │                   │   user's groups)  │
       │                   │                   │                   │
       │                   │                   │  5. Build JWT claims:
       │                   │                   │  - iss: mcp-auth-server
       │                   │                   │  - aud: mcp-registry
       │                   │                   │  - sub: username
       │                   │                   │  - groups: [group IDs]
       │                   │                   │  - scope: mapped scopes
       │                   │                   │  - exp: 8 hours
       │                   │                   │                   │
       │                   │                   │  6. Sign JWT with
       │                   │                   │  SECRET_KEY (HS256)
       │                   │                   │                   │
       │                   │  7. Return JWT    │                   │
       │                   │<──────────────────│                   │
       │                   │                   │                   │
       │  8. Display token │                   │                   │
       │<──────────────────│                   │                   │
       │                   │                   │                   │
```

### Token Validation Flow (CLI/API Usage)

When a user uses the self-signed JWT token with the CLI or API:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    CLI /    │     │   NGINX     │     │ Auth Server │     │  MCP Server │
│   Client    │     │  Gateway    │     │             │     │             │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │                   │
       │  1. API Request   │                   │                   │
       │  Authorization:   │                   │                   │
       │  Bearer <JWT>     │                   │                   │
       │──────────────────>│                   │                   │
       │                   │                   │                   │
       │                   │  2. auth_request  │                   │
       │                   │  /validate        │                   │
       │                   │──────────────────>│                   │
       │                   │                   │                   │
       │                   │                   │  3. Check token issuer
       │                   │                   │  iss == "mcp-auth-server"?
       │                   │                   │                   │
       │                   │                   │  4. If yes: validate
       │                   │                   │  with SECRET_KEY (HS256)
       │                   │                   │                   │
       │                   │                   │  5. If no: try Entra
       │                   │                   │  JWKS validation (RSA)
       │                   │                   │                   │
       │                   │                   │  6. Extract scopes,
       │                   │                   │  validate server/tool
       │                   │                   │  access permissions
       │                   │                   │                   │
       │                   │  7. 200 OK +      │                   │
       │                   │  X-User headers   │                   │
       │                   │<──────────────────│                   │
       │                   │                   │                   │
       │                   │  8. Proxy request │                   │
       │                   │──────────────────────────────────────>│
       │                   │                   │                   │
       │  9. Response      │                   │                   │
       │<──────────────────────────────────────────────────────────│
       │                   │                   │                   │
```

## Token Types

### 1. Entra ID Tokens (from Microsoft)

When users authenticate via Entra ID, Microsoft issues:

- **ID Token**: Contains user identity claims (username, email, groups)
- **Access Token**: Scoped for Microsoft Graph API (not usable for our gateway)

These tokens are:
- RSA-signed (RS256) with Microsoft's keys
- Validated against Microsoft's JWKS endpoint
- Contain group Object IDs (not group names)

### 2. Self-Signed JWT Tokens (from Auth Server)

When users click "Get JWT Token", the auth server generates:

- **Self-Signed JWT**: Contains user identity + gateway-specific scopes
- Signed with HS256 using `SECRET_KEY`
- Contains: username, groups, mapped scopes, provider info

## Security Analysis

### Why We Use Self-Signed Tokens (Not IdP Tokens Directly)

**The IdP tokens don't work for our use case:**

1. **Entra Access Token is for Microsoft Graph API** - When you authenticate with Entra ID, the access token you receive is scoped for Microsoft's APIs (like Graph API for reading user profiles). It's not meant for your custom gateway.

2. **IdP tokens don't contain your scopes** - Entra doesn't know about your `public-mcp-users` scope or your MCP server permissions. Those mappings exist only in your system (the `mcp_scopes` collection in DocumentDB).

3. **Group-to-scope mapping is custom** - The translation from Entra Group Object ID (`5f605d68-06bc-4208-b992-bb378eee12c5`) to gateway scopes (`public-mcp-users`) happens in your auth server, not in Entra.

### Is the Self-Signed Approach Secure?

**Yes, with proper implementation.** Here's why:

| Security Aspect | Implementation |
|-----------------|----------------|
| **Secret Management** | SECRET_KEY from environment variable, not hardcoded |
| **Token Validation** | Every request validates signature, expiry, issuer, audience |
| **Short Expiry** | 8-hour token lifetime limits exposure window |
| **No Credential Storage** | Users don't store passwords; token is derived from OAuth session |
| **Auditable Claims** | Token contains username, groups, provider - traceable |
| **Rate Limiting** | Token generation rate-limited per user (100/hour default) |

### Comparison: Self-Signed vs Direct IdP Tokens

| Aspect | Self-Signed JWT | Direct IdP Token |
|--------|-----------------|------------------|
| **Signing** | HS256 (symmetric) | RS256 (asymmetric) |
| **Key Management** | Single SECRET_KEY | IdP manages key rotation |
| **Scope Mapping** | Done at generation time | Would need separate mapping layer |
| **Token Revocation** | Expiry-based only | Could use IdP revocation |
| **Complexity** | Simple | Requires IdP API registration |
| **Audit Trail** | In auth server logs | In IdP audit logs |

### What Would Be "More Secure"?

Getting tokens directly from IdP would require:

1. **Registering your gateway as an API in Entra** - Defining your own scopes in Azure AD
2. **Users requesting your API scopes** - During OAuth login
3. **Entra issuing tokens for your API** - Instead of for Graph API

This provides:
- Tokens signed by Microsoft's keys (asymmetric RSA)
- Centralized token revocation through Entra
- Entra's audit logs for token issuance

**However**, you'd still need to map Entra groups to your MCP permissions somewhere, so the complexity often isn't worth it for internal/enterprise use cases.

### Security Best Practices

1. **Rotate SECRET_KEY periodically** - Update via environment variable
2. **Use HTTPS everywhere** - Tokens in transit must be encrypted
3. **Monitor token usage** - Log and alert on unusual patterns
4. **Short token lifetime** - 8 hours default, configurable
5. **Scope minimization** - Tokens only get scopes user already has

## Configuration

### Environment Variables

```bash
# Auth Server
SECRET_KEY=your-secure-random-key-here
JWT_ISSUER=mcp-auth-server
JWT_AUDIENCE=mcp-registry
MAX_TOKENS_PER_USER_PER_HOUR=100

# Entra ID
ENTRA_ENABLED=true
ENTRA_CLIENT_ID=your-client-id
ENTRA_CLIENT_SECRET=your-client-secret
ENTRA_TENANT_ID=your-tenant-id
```

### Group Mappings (the `mcp_scopes` collection in DocumentDB)

```yaml
group_mappings:
  # Entra ID Group Object ID -> Gateway Scopes
  "5f605d68-06bc-4208-b992-bb378eee12c5":
    - public-mcp-users

  "4c46ec66-a4f7-4b62-9095-b7958662f4b6":
    - registry-admins
    - mcp-servers-unrestricted/read
    - mcp-servers-unrestricted/execute
```

### Entra ID App Registration Requirements

#### User Authentication App (OAuth Login)

1. **Redirect URIs**: Add your auth server callback URLs
   - `https://your-domain.com/oauth2/callback/entra`

2. **Token Configuration**:
   - Enable ID tokens
   - Add `groups` claim to ID token

3. **API Permissions (Delegated)**:
   - `openid` (delegated)
   - `email` (delegated)
   - `profile` (delegated)

4. **Group Claims**:
   - Configure "Groups assigned to the application" or "All groups"
   - Emit groups as Object IDs (not names)

#### Admin App (IAM Management - M2M Account Creation)

To create M2M service accounts via the Management API, the admin app registration needs additional **Application permissions** (not delegated):

1. **API Permissions (Application - requires admin consent)**:
   - `Application.ReadWrite.All` - Create/manage app registrations
   - `Directory.ReadWrite.All` - Create service principals and manage group memberships
   - `Group.ReadWrite.All` - Create and manage groups
   - `User.ReadWrite.All` - Create and manage users

2. **Grant Admin Consent**:
   - After adding permissions, click "Grant admin consent for [Tenant]"
   - Requires Global Administrator or Privileged Role Administrator

3. **Client Secret**:
   - Create a client secret under "Certificates & secrets"
   - Set as `ENTRA_CLIENT_SECRET` environment variable

**Note**: The admin app is used by the registry backend for IAM operations. It's separate from the user-facing OAuth app (though they can be the same app registration with both delegated and application permissions).

## JWT Token Structure

### Claims in Self-Signed JWT

```json
{
  "iss": "mcp-auth-server",
  "aud": "mcp-registry",
  "sub": "user@example.com",
  "preferred_username": "user@example.com",
  "email": "user@example.com",
  "groups": ["5f605d68-06bc-4208-b992-bb378eee12c5"],
  "scope": "public-mcp-users",
  "token_use": "access",
  "auth_method": "oauth2",
  "provider": "entra",
  "iat": 1768685007,
  "exp": 1768713807,
  "description": "Generated via sidebar"
}
```

### Token Validation Logic

The Entra provider's `validate_token` method:

1. **Check issuer first**: If `iss == "mcp-auth-server"`, validate as self-signed
2. **Self-signed validation**: Use HS256 with SECRET_KEY
3. **Entra validation**: If not self-signed, use RSA with Microsoft JWKS

This ensures both token types work seamlessly with the same validation endpoint.

## Usage Examples

### CLI with Self-Signed Token

```bash
# Set the token from "Get JWT Token" button
export MCP_TOKEN="eyJhbGciOiJIUzI1NiIs..."

# Use with mcpgw CLI
mcpgw servers list --token "$MCP_TOKEN"
mcpgw tools call context7 resolve-library-id --args '{"libraryName": "react"}'
```

### Python SDK

```python
import requests

token = "eyJhbGciOiJIUzI1NiIs..."
headers = {"Authorization": f"Bearer {token}"}

response = requests.get(
    "https://your-gateway.com/api/servers",
    headers=headers
)
```

## Troubleshooting

### Common Issues

1. **"Token missing 'kid' in header"**
   - Cause: Self-signed tokens don't have `kid`, but validation expected RSA token
   - Fix: Auth server now checks issuer before attempting JWKS validation

2. **"Invalid token issuer"**
   - Cause: Token issuer doesn't match expected value
   - Fix: Ensure `JWT_ISSUER` env var matches on token generation and validation

3. **"Access denied - no scopes configured"**
   - Cause: User's groups don't map to any scopes
   - Fix: Add group mapping in the `mcp_scopes` collection in DocumentDB (via the scope management API)

4. **Groups not appearing in token**
   - Cause: Entra app not configured to emit groups
   - Fix: Configure "Token configuration" in Entra app registration
