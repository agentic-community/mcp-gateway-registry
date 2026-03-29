# Auth0 Integration for MCP Gateway Registry

> **⚠️ IMPORTANT DISCLAIMER**
>
> This documentation is a **reference guide based on our testing and development experience**, not an official Auth0 configuration manual. Auth0's interface, features, and best practices evolve over time.
>
> **Always consult the [official Auth0 documentation](https://auth0.com/docs) for:**
> - Current UI layouts and navigation paths
> - Latest security recommendations
> - Production-grade configuration guidance
> - Detailed API references
>
> **Purpose of this guide:**
> - Document the specific configuration steps we used during development
> - Provide a working reference for MCP Gateway Registry integration
> - Share lessons learned and troubleshooting tips
>
> If you encounter differences between this guide and your Auth0 console, refer to Auth0's official documentation as the authoritative source.

This document provides instructions for integrating Auth0 as the authentication provider for the MCP Gateway Registry, including user management and group-based authorization.

## Overview

The MCP Gateway Registry supports Auth0 as an OAuth2/OIDC identity provider. Users authenticate via Auth0 and receive JWT tokens for programmatic access to gateway APIs (CLI tools, coding assistants, etc.).

**Key Concepts:**
- **Users**: People who log in to the registry
- **Roles**: Auth0's term for groups (e.g., `registry-admins`, `registry-users`)
- **Groups**: The MCP Gateway converts Auth0 roles → groups for authorization
- **M2M (Machine-to-Machine)**: Service accounts for CLI tools and scripts (no human login)

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

If neither claim contains data, the user will have an empty groups list and no permissions.

---

## Complete Setup Guide

### Prerequisites

- Auth0 account (free tier works fine for testing)
- MCP Gateway Registry deployed and accessible via HTTPS
- Access to modify nginx configuration and environment variables

### Step 1: Create an Auth0 Application

1. **Log in to Auth0 Dashboard** at https://manage.auth0.com/
2. Navigate to **Applications > Applications** (left sidebar)
3. Click **Create Application**
4. Configure the application:
   - **Name**: `AI Registry` (or your preferred name)
   - **Application Type**: Select **Regular Web Application**
   - Click **Create**

5. **Copy your credentials** from the Settings tab:
   - **Domain**: e.g., `dev-abc123xyz.us.auth0.com` (without `https://`)
   - **Client ID**: Long alphanumeric string
   - **Client Secret**: Click the eye icon to reveal and copy

**Important:** Keep these credentials secure. You'll need them for environment configuration.

### Step 2: Configure Application URLs

Scroll down to **Application URIs** section and configure:

**Allowed Callback URLs:**
```
https://your-registry-domain.com/oauth2/callback/auth0
```

**Allowed Logout URLs:**
```
https://your-registry-domain.com
```

**Allowed Web Origins:**
```
https://your-registry-domain.com
```

**Example for local testing:**
```
http://localhost/oauth2/callback/auth0
http://localhost
```

Click **Save Changes** at the bottom.

### Step 3: Create an Auth0 Action (Required for Groups)

Auth0 Actions are custom code that runs during authentication to add groups to tokens.

1. Navigate to **Actions > Triggers** (left sidebar)
2. Click on the **post-login** trigger box
3. You'll see the Login Flow diagram with Start → Complete
4. Click **Create Action** (bottom right)
5. Configure the Action:
   - **Name**: `Add Groups to Tokens`
   - **Trigger**: `Login / Post Login` (already selected)
   - **Runtime**: `Node 18` (or latest)
   - Click **Create**

6. **Paste this code** in the editor:

```javascript
exports.onExecutePostLogin = async (event, api) => {
  const namespace = "https://mcp-gateway/";

  // Add user's roles as groups
  if (event.authorization && event.authorization.roles) {
    api.idToken.setCustomClaim(namespace + "groups", event.authorization.roles);
    api.accessToken.setCustomClaim(namespace + "groups", event.authorization.roles);
  }

  // Fallback to permissions if no roles
  if (event.authorization && event.authorization.permissions) {
    if (!event.authorization.roles || event.authorization.roles.length === 0) {
      api.idToken.setCustomClaim(namespace + "groups", event.authorization.permissions);
      api.accessToken.setCustomClaim(namespace + "groups", event.authorization.permissions);
    }
  }

  // Optional: Add organization info if using Auth0 Organizations
  if (event.organization) {
    api.idToken.setCustomClaim(namespace + "org_id", event.organization.id);
    api.idToken.setCustomClaim(namespace + "org_name", event.organization.name);
  }
};
```

7. Click **Deploy** (top-right corner)
8. Go back to the Post Login flow (click the back arrow)
9. **Add the Action to the flow**:
   - On the right panel, click the **Custom** tab
   - Find your "Add Groups to Tokens" Action
   - **Drag and drop** it between "Start" and "Complete" in the flow diagram
10. Click **Apply** (top-right)

**Note:** The namespace `https://mcp-gateway/` must match your `AUTH0_GROUPS_CLAIM` environment variable.

### Step 4: Create Roles (Groups)

Auth0 uses "Roles" for authorization. The MCP Gateway maps these to "groups".

1. Navigate to **User Management > Roles** (left sidebar)
2. Click **Create Role**
3. Create the administrator role:
   - **Name**: `registry-admins`
   - **Description**: `Registry administrators with full access`
   - Click **Create**

4. **Optional:** Create additional roles as needed:
   - `registry-users` - Regular users
   - `registry-viewers` - Read-only access
   - `developers` - Developer access

**Important:** Role names must match the groups configured in your `scopes.yml` file.

### Step 5: Create Users

1. Navigate to **User Management > Users** (left sidebar)
2. Click **Create User**
3. Fill in user details:
   - **Email**: User's email address
   - **Password**: Set a strong password (or send password reset email)
   - **Connection**: `Username-Password-Authentication` (default database)
4. Click **Create**

**Repeat** for additional users.

### Step 6: Assign Roles to Users

1. Go to **User Management > Users**
2. Click on a user you just created
3. Go to the **Roles** tab
4. Click **Assign Roles**
5. Select `registry-admins` (or other roles)
6. Click **Assign**

**Verification:** The user should now have roles listed in their profile.

### Step 7: Configure Environment Variables

#### Option A: Update Existing .env File

Edit your `.env` file and update these variables:

```bash
# Authentication Provider
AUTH_PROVIDER=auth0

# Auth0 Configuration
AUTH0_DOMAIN=dev-abc123xyz.us.auth0.com
AUTH0_CLIENT_ID=your-client-id-here
AUTH0_CLIENT_SECRET=your-client-secret-here
AUTH0_GROUPS_CLAIM=https://mcp-gateway/groups
AUTH0_ENABLED=true

# Disable other providers
KEYCLOAK_ENABLED=false
ENTRA_ENABLED=false
COGNITO_ENABLED=false
```

#### Option B: Create Provider-Specific Files

For easy switching between providers:

```bash
# Backup current configuration
cp .env .env.keycloak

# Create Auth0 configuration
cp .env .env.auth0

# Edit .env.auth0 with Auth0 credentials (as shown above)

# Activate Auth0
cp .env.auth0 .env
```

#### Complete Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `AUTH_PROVIDER` | Yes | Set to `auth0` | `auth0` |
| `AUTH0_DOMAIN` | Yes | Auth0 tenant domain (no https://) | `dev-abc123xyz.us.auth0.com` |
| `AUTH0_CLIENT_ID` | Yes | Application client ID | `eYNHy8GXBHH1s60Po9J0SLGcsLGsNPoA` |
| `AUTH0_CLIENT_SECRET` | Yes | Application client secret | `q-9A_nlgypKAOfwLmTvv0k...` |
| `AUTH0_GROUPS_CLAIM` | No | Custom claim name for groups | `https://mcp-gateway/groups` (default) |
| `AUTH0_ENABLED` | Yes | Enable Auth0 provider | `true` |
| `AUTH0_AUDIENCE` | No | API identifier (M2M only) | `https://api.example.com` |
| `AUTH0_M2M_CLIENT_ID` | No | M2M client ID (M2M only) | `xyz789...` |
| `AUTH0_M2M_CLIENT_SECRET` | No | M2M client secret (M2M only) | `abc456...` |

### Step 8: Verify Nginx Configuration

The nginx reverse proxy needs Auth0 route configuration. Check that these location blocks exist in your nginx config file:

**File:** `docker/nginx_rev_proxy_http_and_https.conf`

```nginx
# OAuth2 Auth0 callback endpoint
location /oauth2/callback/auth0 {
    proxy_pass http://auth-server:8888/oauth2/callback/auth0;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $real_scheme;
    proxy_pass_request_headers on;
    proxy_pass_request_body on;
}

# OAuth2 Auth0 login endpoint
location /oauth2/login/auth0 {
    proxy_pass http://auth-server:8888/oauth2/login/auth0;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $real_scheme;
    proxy_pass_request_headers on;
}
```

**If these blocks are missing**, add them to both the HTTP (port 8080) and HTTPS (port 8443) server blocks. Place them after the Google OAuth endpoints and before the Keycloak section.

### Step 9: Restart Services

Restart the registry and auth-server containers to apply the new configuration:

```bash
# If using docker-compose
docker-compose restart registry auth-server

# Or rebuild and restart all services
docker-compose down
docker-compose up -d
```

Wait for services to become healthy:
```bash
docker-compose ps
```

### Step 10: Test Authentication

1. **Open the registry** in your browser: `https://your-registry-domain.com`
2. **Click "Login"** or navigate to the login page
3. **Select "Auth0"** from the provider list
4. You should be **redirected to Auth0 login page**
5. **Enter credentials** for the user you created
6. After successful login, you should be **redirected back to the registry**
7. **Verify your session**:
   - Check that your username appears in the UI
   - Admin users should see admin panels/options

**Check logs if login fails:**
```bash
# Auth server logs
docker-compose logs --tail=50 auth-server

# Registry logs
docker-compose logs --tail=50 registry
```

---

## Machine-to-Machine (M2M) Authentication

M2M authentication allows non-human clients (CLI tools, scripts, cron jobs) to authenticate and access the registry API programmatically.

### When to Use M2M

- CLI tools that need to access the registry without browser login
- Automated scripts (CI/CD pipelines)
- Service-to-service authentication
- Cron jobs that sync or update registry data

### M2M Setup

#### 1. Create an API in Auth0

1. Navigate to **Applications > APIs** in Auth0 Dashboard
2. Click **Create API**
3. Configure the API:
   - **Name**: `MCP Registry API`
   - **Identifier**: `https://api.your-domain.com` (this becomes `AUTH0_AUDIENCE`)
   - **Signing Algorithm**: `RS256`
4. Click **Create**

#### 2. Create an M2M Application

1. Navigate to **Applications > Applications**
2. Click **Create Application**
3. Configure:
   - **Name**: `Registry M2M Client`
   - **Application Type**: Select **Machine to Machine Applications**
4. Select the API you just created (`MCP Registry API`)
5. Click **Authorize**
6. Copy the **Client ID** and **Client Secret**

#### 3. Configure M2M Environment Variables

Add these to your `.env` file:

```bash
AUTH0_AUDIENCE=https://api.your-domain.com
AUTH0_M2M_CLIENT_ID=your-m2m-client-id
AUTH0_M2M_CLIENT_SECRET=your-m2m-client-secret
```

#### 4. Test M2M Authentication

```bash
# Get M2M access token
curl --request POST \
  --url https://your-tenant.auth0.com/oauth/token \
  --header 'content-type: application/json' \
  --data '{
    "client_id": "your-m2m-client-id",
    "client_secret": "your-m2m-client-secret",
    "audience": "https://api.your-domain.com",
    "grant_type": "client_credentials"
  }'

# Use the token to access the registry API
curl -H "Authorization: Bearer <access_token>" \
  https://your-registry-domain.com/api/servers
```

---

## User and Role Management

### Creating Additional Roles

1. Go to **User Management > Roles**
2. Click **Create Role**
3. Enter role name (e.g., `developers`, `viewers`)
4. Click **Create**
5. **Map roles to MCP Gateway groups** in your `scopes.yml` file

### Assigning Roles in Bulk

1. Go to **User Management > Roles**
2. Click on a role (e.g., `registry-users`)
3. Go to the **Users** tab
4. Click **Add Users**
5. Search and select multiple users
6. Click **Assign**

### Removing Roles from Users

1. Go to **User Management > Users**
2. Click on a user
3. Go to the **Roles** tab
4. Click the `...` menu next to a role
5. Click **Remove**

### Creating Users Programmatically

Use the Auth0 Management API to automate user creation:

```bash
# Get Management API token
curl --request POST \
  --url https://your-tenant.auth0.com/oauth/token \
  --header 'content-type: application/json' \
  --data '{
    "client_id": "your-management-api-client-id",
    "client_secret": "your-management-api-client-secret",
    "audience": "https://your-tenant.auth0.com/api/v2/",
    "grant_type": "client_credentials"
  }'

# Create a user
curl --request POST \
  --url https://your-tenant.auth0.com/api/v2/users \
  --header 'authorization: Bearer <management_token>' \
  --header 'content-type: application/json' \
  --data '{
    "email": "newuser@example.com",
    "password": "SecurePassword123!",
    "connection": "Username-Password-Authentication"
  }'
```

### IAM Management (Settings > IAM > Groups/Users)

The MCP Gateway Registry provides a web UI for managing users and roles via **Settings > IAM**. This requires Auth0 Management API access.

**Important:** This is separate from M2M authentication for registry API access. The Management API allows the registry to:
- List users and roles
- Create/delete users
- Assign roles to users
- Manage role (group) definitions

#### Option 1: M2M Application for Management API (Recommended)

Create a dedicated M2M application with Management API permissions:

1. **Navigate to Applications > Applications** in Auth0 Dashboard
2. Click **Create Application**
3. Configure:
   - **Name**: `Registry Management Client`
   - **Application Type**: Select **Machine to Machine Applications**
4. **Select API**: Choose **Auth0 Management API** (this is pre-created by Auth0)
5. Click **Authorize**
6. **Grant Permissions**: Select the following scopes:
   - `read:users`
   - `update:users`
   - `create:users`
   - `delete:users`
   - `read:roles`
   - `update:roles`
   - `create:roles`
   - `delete:roles`
   - `read:users_app_metadata`
   - `update:users_app_metadata`
7. Click **Authorize** to confirm
8. Copy the **Client ID** and **Client Secret**

**Add to .env file:**

```bash
AUTH0_M2M_CLIENT_ID=your-management-client-id
AUTH0_M2M_CLIENT_SECRET=your-management-client-secret
```

#### Option 2: Static Management API Token

Alternatively, use a static token (less secure, expires):

1. Go to **Applications > APIs > Auth0 Management API**
2. Click **API Explorer** tab
3. Click **Create & Authorize Test Application**
4. Copy the generated token

**Add to .env file:**

```bash
AUTH0_MANAGEMENT_API_TOKEN=your-static-token
```

**⚠️ Warning:** Static tokens expire after 24 hours by default. M2M credentials (Option 1) are recommended for production.

#### Testing IAM Management

After configuring Management API access:

1. Restart the registry: `docker-compose restart registry auth-server`
2. Open the web UI: `https://your-registry-domain.com`
3. Navigate to **Settings > IAM > Groups**
4. You should see your Auth0 roles listed (e.g., `registry-admins`, `registry-users`)
5. Navigate to **Settings > IAM > Users**
6. You should see all Auth0 users with their role assignments

**Troubleshooting:**
- If you see an empty list or errors, check auth server logs: `docker-compose logs auth-server | grep Management`
- Verify M2M credentials are correct: `grep AUTH0_M2M .env`
- Ensure Management API permissions are granted in Auth0 Dashboard

---

## Group-to-Scope Mapping

The MCP Gateway uses a `scopes.yml` file to map Auth0 roles to registry permissions.

### Example scopes.yml Configuration

```yaml
group_mappings:
  registry-admins:
    - admin:*
    - servers:*
    - agents:*
    - scopes:manage

  registry-users:
    - servers:read
    - servers:write
    - agents:read
    - tools:*

  registry-viewers:
    - servers:read
    - agents:read
    - tools:read
```

**Location:** This file should be in your registry configuration directory and loaded at startup.

---

## Troubleshooting

### Empty Page or No Redirect to Auth0

**Symptom:** Clicking login shows an empty page at `/oauth2/login/auth0`

**Causes:**
1. Missing nginx configuration for Auth0 routes
2. Auth server not receiving the request

**Solution:**
1. Verify nginx has Auth0 location blocks (Step 8)
2. Restart the registry container: `docker-compose restart registry`
3. Check nginx error logs: `docker-compose exec registry cat /var/log/nginx/error.log`

### Users Have No Groups After Login

**Symptom:** User logs in successfully but has no permissions

**Causes:**
1. Auth0 Action not deployed or not in the flow
2. User has no roles assigned
3. `AUTH0_GROUPS_CLAIM` mismatch

**Solution:**
1. Go to **Actions > Triggers > Post Login** and verify the Action is in the flow
2. Check user has roles: **User Management > Users > [User] > Roles tab**
3. Verify environment variable: `grep AUTH0_GROUPS_CLAIM .env`
4. Check auth server logs: `docker-compose logs auth-server | grep "Auth0 ID token claims"`

### Callback URL Mismatch Error

**Symptom:** Auth0 shows "Callback URL mismatch" error after login

**Solution:**
1. Go to Auth0 Dashboard > Applications > Your App > Settings
2. Verify **Allowed Callback URLs** exactly matches:
   ```
   https://your-domain.com/oauth2/callback/auth0
   ```
3. Click **Save Changes**
4. Try logging in again

### Token Validation Errors

**Symptom:** "Invalid token" or "Token validation failed" errors

**Causes:**
1. `AUTH0_DOMAIN` has `https://` prefix
2. Wrong Client ID or Client Secret
3. Token expired

**Solution:**
1. Verify domain has no protocol:
   ```bash
   # Correct
   AUTH0_DOMAIN=dev-abc123xyz.us.auth0.com

   # Wrong
   AUTH0_DOMAIN=https://dev-abc123xyz.us.auth0.com
   ```
2. Verify credentials match Auth0 Dashboard
3. Check auth server logs for specific error messages

### M2M Token Failures

**Symptom:** M2M authentication returns 401 or 403

**Causes:**
1. M2M application not authorized for the API
2. Wrong audience parameter
3. Missing API in Auth0

**Solution:**
1. Go to Auth0 Dashboard > Applications > APIs > [Your API]
2. Go to the **Machine to Machine Applications** tab
3. Ensure your M2M app is listed and **Authorized**
4. Verify `AUTH0_AUDIENCE` matches the API **Identifier** exactly

### CORS Errors

**Symptom:** Browser console shows CORS errors

**Solution:**
1. Verify **Allowed Web Origins** in Auth0 includes your domain
2. Check nginx is setting correct CORS headers
3. Ensure `AUTH_SERVER_EXTERNAL_URL` matches your public domain

---

## Security Best Practices

### 1. Use Strong Secrets

- Generate strong, random client secrets (Auth0 does this automatically)
- Never commit secrets to version control
- Rotate secrets periodically

### 2. Restrict Callback URLs

Only add legitimate callback URLs to Auth0:
```
# Good - specific domains
https://registry.example.com/oauth2/callback/auth0
https://registry-staging.example.com/oauth2/callback/auth0

# Bad - wildcards allow any subdomain
https://*.example.com/oauth2/callback/auth0
```

### 3. Enable Multi-Factor Authentication (MFA)

1. Go to **Security > Multi-factor Auth** in Auth0 Dashboard
2. Enable **One-time Password** or **SMS**
3. Configure policies (e.g., require MFA for admins)

### 4. Monitor Login Activity

1. Go to **Monitoring > Logs** in Auth0 Dashboard
2. Review failed login attempts
3. Set up alerts for suspicious activity

### 5. Implement Principle of Least Privilege

- Create specific roles with minimal permissions
- Don't assign `registry-admins` to regular users
- Regularly audit user roles

---

## Additional Resources

- **Auth0 Documentation**: https://auth0.com/docs
- **Auth0 Actions**: https://auth0.com/docs/customize/actions
- **Auth0 Roles & Permissions**: https://auth0.com/docs/manage-users/access-control
- **MCP Gateway Registry Docs**: https://github.com/agentic-community/mcp-gateway-registry/docs

---

## Summary Checklist

Use this checklist to verify your Auth0 integration is complete:

- [ ] Auth0 Application created (Regular Web Application)
- [ ] Domain, Client ID, and Client Secret copied
- [ ] Allowed Callback URLs configured
- [ ] Allowed Logout URLs configured
- [ ] Allowed Web Origins configured
- [ ] Auth0 Action created and deployed
- [ ] Action added to Post Login flow
- [ ] Roles created (e.g., `registry-admins`)
- [ ] Users created in Auth0
- [ ] Roles assigned to users
- [ ] Environment variables configured in `.env`
- [ ] Nginx configuration includes Auth0 routes
- [ ] Services restarted to apply configuration
- [ ] Login tested successfully
- [ ] User groups appear correctly in registry
- [ ] Admin permissions verified (if applicable)

Once all items are checked, your Auth0 integration is complete!
