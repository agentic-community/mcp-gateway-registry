# Okta Identity Provider Setup Guide

This guide walks through configuring Okta as the identity provider for the MCP Gateway Registry.

> **⚠️ IMPORTANT DISCLAIMER**
>
> This documentation is a **reference guide based on our testing and development experience**, not an official Okta configuration manual. Okta's interface, features, and best practices evolve over time.
>
> **Always consult the [official Okta documentation](https://developer.okta.com/docs/) for:**
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
> If you encounter differences between this guide and your Okta console, refer to Okta's official documentation as the authoritative source.

## Prerequisites

- An Okta developer account ([sign up free](https://developer.okta.com/signup/))
- Your Okta domain (e.g., `dev-123456.okta.com`)
- Understanding of OAuth2/OIDC flows (see [Okta OAuth2 documentation](https://developer.okta.com/docs/concepts/oauth-openid/))

## Step 1: Create an OAuth2 Web Application

1. In the Okta Admin Console, go to **Applications** → **Applications** → **Create App Integration**
2. Select **OIDC - OpenID Connect** and **Web Application**, then click **Next**
3. Configure the application:
   - **Name**: `MCP Gateway Registry`
   - **Grant types**: Authorization Code, Refresh Token, Client Credentials
   - **Sign-in redirect URIs**: `http://localhost:8888/oauth2/callback/okta` (dev) or `https://your-auth-server-domain/oauth2/callback/okta` (production)
   - **Sign-out redirect URIs**: `http://localhost/logout` (dev) or `https://your-registry-domain/logout` (production)
   - **Controlled access**: Allow everyone in your organization
4. Click **Save** and copy the **Client ID** and **Client Secret** immediately

## Step 2: Configure Groups Claim in ID Tokens

The groups claim is configured on the application's Sign On tab using the legacy configuration. This uses the Okta Org Authorization Server (`/oauth2/v1/*`), which has a built-in `groups` scope.

1. Go to **Applications** → your app → **Sign On** tab
2. Scroll to the **Token claims (OIDC)** section and expand **Show legacy configuration**
3. Under **Group Claims**, click **Edit**
4. Set **Groups claim type** to **Filter**
5. Set the name to `groups`, select **Matches regex**, and enter `.*`
6. Click **Save**

> **Note:** The Org Authorization Server and the "default" custom authorization server are different. Without `OKTA_AUTH_SERVER_ID`, this integration uses the Org Authorization Server, which natively supports the `groups` scope, and claims configured under Security → API → Authorization Servers → default do not apply. With `OKTA_AUTH_SERVER_ID` set, see Step 2a instead.

## Step 2a: Custom Authorization Server (Optional)

You need a custom authorization server in two cases:

- MCP clients present their own Okta access token to the gateway, instead of a token generated in the registry UI. The
  gateway checks an access token's signature, issuer and audience, and Okta's Org Authorization Server issues access
  tokens for Okta's own APIs only, so the gateway refuses them.
- M2M service accounts need custom claims or access policies.

When `OKTA_AUTH_SERVER_ID` is set, the gateway uses that server for the browser login as well as for token validation
and its OAuth discovery metadata. The server therefore has to sign users in, not only issue tokens to clients.

| Feature                           | Org Authorization Server   | Custom Authorization Server        |
|-----------------------------------|----------------------------|------------------------------------|
| Endpoint pattern                  | `/oauth2/v1/*`             | `/oauth2/{authServerId}/v1/*`      |
| Built-in `groups` scope           | Yes                        | No (add it, step 4 below)          |
| Groups claim                      | App's Sign On tab (Step 2) | Server's Claims tab (step 5 below) |
| Access tokens the gateway accepts | No                         | Yes                                |
| Custom access policies            | No                         | Yes                                |

**Setup steps:**

1. Go to **Security** → **API** → **Authorization Servers**. Use the `default` server, or click **Add Authorization
   Server** and give it an audience such as `api://ai-registry`. The audience becomes the `aud` claim of every access
   token the server issues.
2. Copy the server's **Issuer URI**. `OKTA_AUTH_SERVER_ID` is the last segment of its path: `default` for
   `https://dev-123456.okta.com/oauth2/default`, or an ID such as `aus1234567890abcdef` for a server you added. For the
   default server use `default`, not the `aus…` ID the Okta API reports for it: the gateway builds the issuer it expects
   from this value, and the default server's tokens say `iss: https://dev-123456.okta.com/oauth2/default`.
3. On the **Access Policies** tab, add a policy assigned to the web app from Step 1 and to any M2M apps. Add a rule that
   allows the **Authorization Code** grant (and **Client Credentials** for M2M apps) for the groups that use the
   gateway. A server without a policy issues no tokens.
4. On the **Scopes** tab, add a scope named `groups`. The gateway's login requests `openid email profile groups`, and a
   custom server refuses the whole sign-in ("One or more scopes are not configured for the authorization server") when
   one of them is missing.
5. On the **Claims** tab, add a claim named `groups`: **Include in token type** Access Token, **Value type** Groups, *
   *Filter** Matches regex `.*`, **Include in** Any scope. Add a second `groups` claim with the same settings for **ID
   Token**, **Always**. The login reads groups from the ID token, and the gateway maps an MCP client's access-token
   groups to scopes.
6. Set `OKTA_AUTH_SERVER_ID` to the value from step 2, and set `OKTA_M2M_ALLOWED_AUDIENCES` to the server's audience (
   `api://default` for the default server). The gateway accepts an access token only when its `aud` is one of its own
   client IDs or is listed in `OKTA_M2M_ALLOWED_AUDIENCES`. Despite its name, that setting applies to user access tokens
   as well as M2M tokens.

The Group Claims configured in Step 2 apply only to the Org Authorization Server. Once `OKTA_AUTH_SERVER_ID` is set,
groups come from the claims in step 5.

**Groups enrichment for M2M tokens:**

When M2M tokens are issued with empty groups (common with custom authorization servers), the registry enriches them from DocumentDB/MongoDB:

1. M2M token is validated successfully but has no groups claim (or empty array)
2. Registry queries `idp_m2m_clients` collection for the client ID
3. Groups from the database are injected into the authorization context
4. Standard group-to-scope mapping applies

This allows scalable M2M authorization without hardcoding client IDs in authorization server expressions.

## Step 3: Create Groups for Access Control

Okta group names must match the group names in your registry's scope configuration (the `mcp_scopes` collection in DocumentDB, seeded from JSON scope seed files in `scripts/`). The default configuration expects groups like `registry-admins` and `public-mcp-users`.

1. Go to **Directory** → **Groups** → **Add Group**
2. Create groups that match your scope group mappings:
   - `registry-admins` — full admin access to the registry
   - `public-mcp-users` — read-only access to public MCP servers
3. Assign users to groups via each group's **Assign people** tab

### Group-to-Scope Mapping

The registry maps Okta groups to authorization scopes using the scope configuration (the `mcp_scopes` collection in DocumentDB, seeded from JSON scope seed files in `scripts/`). Example mapping:

```yaml
# group mapping in the mcp_scopes collection
groups:
  registry-admins:
    - registry:admin:full
    - mcp:servers:read
    - mcp:servers:write
    - mcp:servers:delete

  public-mcp-users:
    - mcp:servers:read
    - mcp:servers:list
```

**How it works:**

1. User logs in with Okta → ID token contains `groups` claim: `["registry-admins"]`
2. Registry extracts groups from token → queries DocumentDB for group-to-scope mappings
3. Scopes are assigned based on group membership
4. User can access resources matching their scopes

**For M2M tokens:**

1. M2M client authenticates with Client Credentials flow
2. If token has empty `groups` (common with custom auth servers)
3. Registry queries `idp_m2m_clients` collection in DocumentDB for client groups
4. Groups are enriched and mapped to scopes using the same scope configuration (the `mcp_scopes` collection)

## Step 3a: Create and Manage Users

### Creating Users Manually (Okta Console)

1. Go to **Directory** → **People** → **Add Person**
2. Fill in user details:
   - **First name** and **Last name**
   - **Username** (email format)
   - **Primary email**
   - **Password**: Choose activation method
3. Click **Save**
4. Assign to groups:
   - Open the user's profile
   - Go to **Groups** tab
   - Click **Edit** → Select groups → **Save**

### Creating Users via Registry IAM API

If `OKTA_API_TOKEN` is configured, you can create users through the registry:

```bash
# Create a new user
curl -X POST https://your-registry/api/iam/users \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john.doe@example.com",
    "email": "john.doe@example.com",
    "firstName": "John",
    "lastName": "Doe",
    "groups": ["public-mcp-users"]
  }'
```

### Creating M2M Service Accounts

M2M service accounts are OAuth2 clients with Client Credentials grant:

**Via Okta Console:**

1. Go to **Applications** → **Applications** → **Create App Integration**
2. Select **API Services** (not Web Application)
3. **Name**: `ai-agent-3` (or your service name)
4. Click **Save** → Copy **Client ID** and **Client Secret**
5. The application is created but groups are managed separately in the registry

**Via Registry IAM API:**

```bash
# Create M2M account with groups
curl -X POST https://your-registry/api/iam/m2m \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "ai-agent-3",
    "description": "AI Agent for autonomous operations",
    "groups": ["public-mcp-users", "ai-agents"]
  }'

# Response includes client_id and client_secret
{
  "client_id": "0oa9876543210fedcba",
  "client_secret": "secret-value-here",
  "groups": ["public-mcp-users", "ai-agents"],
  "okta_app_id": "0oa9876543210fedcba"
}
```

The M2M account is stored in DocumentDB's `idp_m2m_clients` collection for groups enrichment.

**Testing M2M token:**

```bash
# Get M2M token
curl -X POST https://dev-123456.okta.com/oauth2/v1/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials&scope=openid" \
  -u "CLIENT_ID:CLIENT_SECRET"

# Use token to call registry
curl https://your-registry/api/servers \
  -H "Authorization: Bearer M2M_TOKEN"
```

## Step 4: Create API Token (Optional)

Only required if you need IAM operations (user/group management through the registry).

1. Go to **Security** → **API** → **Tokens** → **Create Token**
2. Name it `MCP Gateway IAM` and copy the token value immediately
3. For least-privilege access, create a custom admin role with only the permissions you need:

| Operation | Required Permission |
|-----------|-------------------|
| List users | `okta.users.read` |
| List groups | `okta.groups.read` |
| Create/delete users | `okta.users.manage` |
| Create/delete groups | `okta.groups.manage` |
| Create service accounts | `okta.apps.manage` |

## Environment Variables

### Core Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `AUTH_PROVIDER` | Yes | Set to `okta` |
| `OKTA_DOMAIN` | Yes | Your Okta org domain (e.g., `dev-123456.okta.com`) |
| `OKTA_CLIENT_ID` | Yes | OAuth2 client ID from Step 1 |
| `OKTA_CLIENT_SECRET` | Yes | OAuth2 client secret from Step 1 |

### Optional Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `OKTA_AUTH_SERVER_ID` | Optional | Custom authorization server from Step 2a: the last segment of its Issuer URI (`default`, or an ID such as `aus1234567890abcdef`). The browser login and token validation both use it. If not set, both use the Org Authorization Server. |
| `OKTA_M2M_ALLOWED_AUDIENCES` | With `OKTA_AUTH_SERVER_ID` | Comma- or space-separated `aud` values the gateway accepts on Okta access tokens besides its own client IDs. Set it to the custom server's audience (for example `api://default`). Applies to user access tokens as well as M2M tokens. Empty accepts only the client IDs. |
| `OKTA_M2M_CLIENT_ID` | Optional | Separate M2M client ID (defaults to `OKTA_CLIENT_ID`) |
| `OKTA_M2M_CLIENT_SECRET` | Optional | Separate M2M client secret (defaults to `OKTA_CLIENT_SECRET`) |
| `OKTA_API_TOKEN` | For IAM | Admin API token from Step 4 (required for user/group management) |

## Example Configuration

### Basic Setup (Org Authorization Server)

```bash
# .env or docker-compose environment
AUTH_PROVIDER=okta
OKTA_DOMAIN=dev-123456.okta.com
OKTA_CLIENT_ID=0oa1234567890abcdef
OKTA_CLIENT_SECRET=your-client-secret-here

# Optional: Admin API token for IAM operations
# OKTA_API_TOKEN=your-api-token-here
```

### Advanced Setup (Custom Authorization Server)

```bash
# .env or docker-compose environment
AUTH_PROVIDER=okta
OKTA_DOMAIN=dev-123456.okta.com
OKTA_CLIENT_ID=0oa1234567890abcdef
OKTA_CLIENT_SECRET=your-client-secret-here

# Custom authorization server (Step 2a): the last segment of its Issuer URI
OKTA_AUTH_SERVER_ID=default
# The server's audience, so the gateway accepts its access tokens
OKTA_M2M_ALLOWED_AUDIENCES=api://default

# Optional: Separate M2M credentials
OKTA_M2M_CLIENT_ID=0oa0987654321fedcba
OKTA_M2M_CLIENT_SECRET=your-m2m-secret-here

# Admin API token for IAM operations
OKTA_API_TOKEN=your-api-token-here
```

### Terraform Configuration

```terraform
# terraform.tfvars
okta_enabled           = true
okta_domain            = "dev-123456.okta.com"
okta_client_id         = "0oa1234567890abcdef"
okta_client_secret     = "your-client-secret-here"
okta_m2m_client_id     = "0oa0987654321fedcba"
okta_m2m_client_secret = "your-m2m-secret-here"
okta_api_token         = "your-api-token-here"
okta_auth_server_id    = "default"  # Optional - custom auth server, last segment of its Issuer URI
okta_m2m_allowed_audiences = "api://default"  # Required with okta_auth_server_id: the server's audience

# Ensure other providers are disabled
entra_enabled = false
```

## Okta Endpoints (Auto-Derived)

The application automatically constructs OAuth2 endpoints based on your configuration:

### Org Authorization Server (default, when `OKTA_AUTH_SERVER_ID` is not set)

| Endpoint | URL Pattern |
|----------|-------------|
| Authorization | `https://{OKTA_DOMAIN}/oauth2/v1/authorize` |
| Token | `https://{OKTA_DOMAIN}/oauth2/v1/token` |
| UserInfo | `https://{OKTA_DOMAIN}/oauth2/v1/userinfo` |
| JWKS | `https://{OKTA_DOMAIN}/oauth2/v1/keys` |
| Logout | `https://{OKTA_DOMAIN}/oauth2/v1/logout` |
| Issuer | `https://{OKTA_DOMAIN}` |

### Custom Authorization Server (when `OKTA_AUTH_SERVER_ID` is set)

| Endpoint | URL Pattern |
|----------|-------------|
| Authorization | `https://{OKTA_DOMAIN}/oauth2/{OKTA_AUTH_SERVER_ID}/v1/authorize` |
| Token | `https://{OKTA_DOMAIN}/oauth2/{OKTA_AUTH_SERVER_ID}/v1/token` |
| UserInfo | `https://{OKTA_DOMAIN}/oauth2/{OKTA_AUTH_SERVER_ID}/v1/userinfo` |
| JWKS | `https://{OKTA_DOMAIN}/oauth2/{OKTA_AUTH_SERVER_ID}/v1/keys` |
| Logout | `https://{OKTA_DOMAIN}/oauth2/{OKTA_AUTH_SERVER_ID}/v1/logout` |
| Issuer | `https://{OKTA_DOMAIN}/oauth2/{OKTA_AUTH_SERVER_ID}` |

**Example with custom auth server:**
- `OKTA_DOMAIN=dev-123456.okta.com`
- `OKTA_AUTH_SERVER_ID=aus1234567890abcdef`
- JWKS URL: `https://dev-123456.okta.com/oauth2/aus1234567890abcdef/v1/keys`

The browser login takes its endpoints from the same table, so the ID token it receives comes from the issuer the gateway
verifies.

## How the Gateway Identifies an Okta User

With per-user egress (`EGRESS_AUTH_ENABLED=true`), the gateway stores each user's connections to upstream services under
an account ID that never changes, and audit records name the same ID as `identity.subject`. Which claim carries that ID
depends on the token:

| Token                                    | Claim the gateway uses      | Why                                                                                                                                                                                        |
|------------------------------------------|-----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ID token from the browser login          | `sub`                       | In an Okta ID token, `sub` is the user ID (`00u…`).                                                                                                                                        |
| User access token an MCP client presents | `uid`                       | In an Okta access token, `sub` is the user's login, often an email address. An admin can edit a login or give it to another user. `uid` is the same user ID the ID token carries in `sub`. |
| Client-credentials (M2M) token           | `sub`, when it equals `cid` | The token has no `uid` because Okta issues it to the client itself.                                                                                                                        |

A user access token without a usable `uid` gets no key, and the gateway refuses to hand it a stored credential rather
than fall back to the login. Audit records still show the login as `identity.username`.

Because the browser login and the access token resolve to the same user ID, a user's MCP clients can use the connections
that user made in the browser. When a login moves from one user to another, the second user gets none of the first
user's connections, even though their access tokens carry the same `sub`.

## Verifying Your Setup

Test the JWKS endpoint:

```bash
curl https://dev-123456.okta.com/oauth2/v1/keys
```

Test client credentials token generation:

```bash
curl -X POST https://dev-123456.okta.com/oauth2/v1/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials&scope=openid" \
  -u "CLIENT_ID:CLIENT_SECRET"
```

### Test the Per-User Identity

This test checks that the gateway keys a user's stored connections on the Okta user ID and not on the login. It needs
the custom authorization server from Step 2a, per-user egress turned on, an MCP server with per-user egress configured (
see [Egress Credential Vault](egress-credential-vault.md); any upstream provider works), and two Okta users in a group
that can use that server.

A token generated in the registry UI does not exercise this path, because the gateway writes the user's key into that
token itself. The test needs access tokens that Okta issues.

1. Sign in to the registry as user A and connect the server, from Connected Accounts or the connect link a tool returns.
2. Get an access token for user A from the custom authorization server (
   `https://{OKTA_DOMAIN}/oauth2/{OKTA_AUTH_SERVER_ID}/v1/authorize`, then `/v1/token`) with the Authorization Code flow
   and PKCE, using the web app from Step 1 or any OIDC app the server's access policy allows. Decode it and check three
   claims: `sub` is the login, `uid` is the user ID, and `aud` is listed in `OKTA_M2M_ALLOWED_AUDIENCES`.
3. Call one of the server's tools through the gateway with `Authorization: Bearer <user A's access token>`. The tool
   runs without asking user A to connect, because the token's `uid` matches the ID token `sub` the connection was stored
   under.
4. In Okta, change user A's username. Create user B with user A's old username, and put user B in the same group. Get a
   new access token for each user.
5. Call the tool with user B's token. It asks user B to connect, although user B's token carries the same `sub` as user
   A's tokens before the rename.
6. Call the tool with user A's new token. It still runs.
7. Check the audit records for those calls. `identity.subject` shows each user's ID (the access token `uid`), and
   `identity.username` shows the login, so two calls with the same username are recorded under two subjects.

## Troubleshooting

**"Permission Required" error after login**
Your Okta groups don't match the group names in the scope configuration (the `mcp_scopes` collection). Create groups in Okta that match (e.g., `registry-admins`) and assign your user to them. See Step 3.

**Groups not appearing in tokens**
Without `OKTA_AUTH_SERVER_ID`, the groups claim must be configured on the app's Sign On tab under "Show legacy
configuration" (Step 2); claims on an authorization server's Claims tab do not apply. With `OKTA_AUTH_SERVER_ID` set, it
is the other way round: add the `groups` claims to that server (Step 2a, step 5). Also verify your user is assigned to
at least one group.

**"One or more scopes are not configured" error**
The login requests the `groups` scope, which only the Org Authorization Server defines on its own. With
`OKTA_AUTH_SERVER_ID` set, add a `groups` scope to that server (Step 2a, step 4).

**Every login fails with "ID token verification failed" after setting `OKTA_AUTH_SERVER_ID`**
The auth-server log names the cause, for example
`id_token issuer 'https://dev-123456.okta.com' is not in the expected issuer allowlist`. Two causes are common:

- `OKTA_AUTH_SERVER_ID` names the default server by its `aus…` ID. Set it to `default`, the last segment of the server's
  Issuer URI.
- The gateway runs a release whose login still used the Org Authorization Server while token validation used the custom
  one. Upgrade, or unset `OKTA_AUTH_SERVER_ID` until you do.

**Can't find Client Secret after app creation**
Regenerate it: App → General tab → Client Credentials → Edit → Regenerate Secret.

**API token permission errors**
Check **Security** → **Administrators** for the role assigned to the token. Create a custom admin role with the specific scopes needed.

**Non-standard domain warning in logs**
The provider validates domains against `*.okta.com`, `*.oktapreview.com`, and `*.okta-emea.com`. Custom domains will log a warning but still work.

**"No matching key found for kid" error**
This means the JWT token was signed by a different authorization server than the one configured. Common causes:
- Token was issued by custom auth server, but `OKTA_AUTH_SERVER_ID` is not set → Set the auth server ID
- Token was issued by org auth server, but `OKTA_AUTH_SERVER_ID` is set → Remove or correct the auth server ID
- Check the token's `iss` claim matches your issuer configuration

Verify JWKS endpoint:
```bash
# For Org Authorization Server
curl https://dev-123456.okta.com/oauth2/v1/keys

# For Custom Authorization Server
curl https://dev-123456.okta.com/oauth2/aus1234567890abcdef/v1/keys
```

**"Audience doesn't match" error for access tokens from a custom authorization server**
A custom authorization server sets `aud` to its own audience (for example `api://default` or `api://ai-registry`), not
to the client ID. The gateway accepts that value only when it is listed in `OKTA_M2M_ALLOWED_AUDIENCES`, for user access
tokens and M2M tokens alike.

**M2M token returns 0 servers despite valid groups**
Check that groups are being mapped to scopes:
1. Verify the `mcp_scopes` collection contains mappings for the M2M client's groups
2. Check auth server logs for group enrichment messages:
   ```
   Groups enriched from MongoDB for client {client_id}: {groups}
   Mapped okta groups {groups} to scopes: {scopes}
   ```
3. If using custom auth server, ensure `groups` claim is configured (see Step 2a)
4. Verify the M2M client exists in `idp_m2m_clients` collection with correct groups

**M2M groups not enriched from database**
The groups enrichment only activates when:
- Token validation succeeds (`valid: true`)
- Token has no groups OR empty groups array
- Token contains a `client_id` claim (M2M tokens)

Check DocumentDB:
```bash
# Connect to mongo container
docker exec -it mcp-mongodb mongosh

# Query M2M clients collection
use mcp_registry_default
db.idp_m2m_clients.find({ client_id: "0oa9876543210fedcba" })
```

Expected document structure:
```json
{
  "client_id": "0oa9876543210fedcba",
  "name": "ai-agent-3",
  "groups": ["public-mcp-users", "ai-agents"],
  "provider": "okta",
  "enabled": true,
  "created_at": "2026-03-15T12:00:00Z"
}
```
