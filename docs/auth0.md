# Auth0 Integration for MCP Gateway Registry

This document describes the integration between Auth0 and the MCP Gateway Registry, including setup requirements for group-based authorization.

## Overview

The MCP Gateway Registry supports Auth0 as an OAuth2/OIDC identity provider. Users can authenticate via Auth0 and obtain JWT tokens for programmatic access to the gateway APIs (CLI tools, coding assistants, etc.).

## Architecture

### Authentication Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Browser   │     │  Registry   │     │ Auth Server │     │    Auth0    │
│   (User)    │     │  Frontend   │     │             │     │   Tenant    │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │                   │
       │  1. Click Login   │                   │                   │
       │──────────────────>│                   │                   │
       │                   │                   │                   │
       │  2. Redirect to Auth Server          │                   │
       │<──────────────────│                   │                   │
       │                   │                   │                   │
       │  3. /oauth2/login/auth0              │                   │
       │──────────────────────────────────────>│                   │
       │                   │                   │                   │
       │  4. Redirect to Auth0 /authorize endpoint                │
       │<─────────────────────────────────────────────────────────>│
       │                   │                   │                   │
       │  5. User authenticates with Auth0    │                   │
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

### Group Extraction

User groups are extracted from the Auth0 ID token using a **custom namespaced claim**. Auth0 does not include group memberships in tokens by default -- you must configure an Auth0 Action (or legacy Rule) to add them.

**Claim lookup order:**

1. Custom namespaced claim (default: `https://mcp-gateway/groups`)
2. Fallback: `permissions` claim from Auth0 RBAC

If neither claim contains data, the user will have an empty groups list.

## Auth0 Tenant Setup

### 1. Create an Application

1. Go to **Applications > Applications** in the Auth0 Dashboard
2. Click **Create Application**
3. Select **Regular Web Application**
4. Note the **Domain**, **Client ID**, and **Client Secret**

### 2. Configure Application Settings

In the application settings:

- **Allowed Callback URLs**: `https://<your-auth-server>/oauth2/callback/auth0`
- **Allowed Logout URLs**: `https://<your-registry-url>`
- **Allowed Web Origins**: `https://<your-registry-url>`

### 3. Create an API (Optional, for M2M tokens)

If you need machine-to-machine authentication:

1. Go to **Applications > APIs**
2. Click **Create API**
3. Set the **Identifier** (this becomes your `AUTH0_AUDIENCE`)

### 4. Configure Groups Claim (Required for Authorization)

Auth0 requires an **Action** (or legacy Rule) to add group memberships to the ID token. Without this, users will not have group-based authorization.

#### Using Auth0 Actions (Recommended)

1. Go to **Actions > Flows > Login**
2. Click **Add Action > Build from scratch**
3. Name it (e.g., "Add Groups to Tokens")
4. Add the following code:

```javascript
exports.onExecutePostLogin = async (event, api) => {
  const namespace = "https://mcp-gateway/";

  // Add user's groups (from Auth0 Organizations or Authorization Extension)
  if (event.authorization && event.authorization.roles) {
    api.idToken.setCustomClaim(namespace + "groups", event.authorization.roles);
    api.accessToken.setCustomClaim(namespace + "groups", event.authorization.roles);
  }

  // Alternative: use Auth0 Organizations groups
  if (event.organization) {
    api.idToken.setCustomClaim(namespace + "org_id", event.organization.id);
    api.idToken.setCustomClaim(namespace + "org_name", event.organization.name);
  }
};
```

5. Click **Deploy**
6. Drag the action into the Login flow and click **Apply**

**Note**: The namespace must match your `AUTH0_GROUPS_CLAIM` environment variable. The default is `https://mcp-gateway/groups`. Auth0 requires custom claims to use a URL-style namespace to avoid collisions with standard OIDC claims.

## Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `AUTH_PROVIDER` | Yes | Set to `auth0` | `auth0` |
| `AUTH0_DOMAIN` | Yes | Auth0 tenant domain | `your-tenant.auth0.com` |
| `AUTH0_CLIENT_ID` | Yes | Application client ID | `abc123...` |
| `AUTH0_CLIENT_SECRET` | Yes | Application client secret | `def456...` |
| `AUTH0_AUDIENCE` | No | API identifier for access tokens | `https://api.example.com` |
| `AUTH0_GROUPS_CLAIM` | No | Custom claim name for groups | `https://mcp-gateway/groups` (default) |
| `AUTH0_M2M_CLIENT_ID` | No | M2M application client ID | `ghi789...` |
| `AUTH0_M2M_CLIENT_SECRET` | No | M2M application client secret | `jkl012...` |
| `AUTH0_ENABLED` | Yes | Enable Auth0 provider in YAML | `true` |

### Example `.env` Configuration

```bash
AUTH_PROVIDER=auth0
AUTH0_DOMAIN=your-tenant.auth0.com
AUTH0_CLIENT_ID=your-client-id
AUTH0_CLIENT_SECRET=your-client-secret
AUTH0_AUDIENCE=https://api.your-domain.com
AUTH0_GROUPS_CLAIM=https://mcp-gateway/groups
AUTH0_ENABLED=true
```

## Troubleshooting

### Users have no groups after login

- Verify that the Auth0 Action/Rule is deployed and active in the Login flow
- Check that the `AUTH0_GROUPS_CLAIM` matches the namespace used in the Action
- Inspect the ID token claims in the auth server logs (look for "Auth0 ID token claims")
- Confirm users have roles assigned in Auth0 (under **User Management > Users > Roles**)

### Token validation errors

- Ensure `AUTH0_DOMAIN` does not include `https://` prefix (just the domain)
- Verify the application's **Client ID** and **Client Secret** match the environment variables
- Check that the callback URL in Auth0 matches your deployed auth server URL exactly

### M2M token failures

- Verify an API is created in Auth0 and `AUTH0_AUDIENCE` matches its identifier
- Ensure the M2M application is authorized to access the API (under **Machine to Machine Applications** tab on the API page)
