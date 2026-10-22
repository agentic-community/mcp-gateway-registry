# Microsoft Entra ID Setup Guide

This guide provides step-by-step instructions for setting up Microsoft Entra ID (formerly Azure Active Directory) authentication for the MCP Gateway Registry.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Azure Portal Configuration](#azure-portal-configuration)
3. [Environment Configuration](#environment-configuration)
4. [Group Configuration](#group-configuration)
5. [Testing the Setup](#testing-the-setup)
6. [Troubleshooting](#troubleshooting)
7. [Using the IAM API to Manage Groups, Users, and M2M Accounts](#using-the-iam-api-to-manage-groups-users-and-m2m-accounts)
8. [Generating JWT Tokens for M2M Accounts](#generating-jwt-tokens-for-m2m-accounts)

---

## Prerequisites

Before you begin, ensure you have:

- Access to an Azure account with permissions to create App Registrations
- Azure Active Directory (Entra ID) tenant
- Admin rights to configure App Registrations and assign users to groups
- The MCP Gateway Registry codebase

---

## Azure Portal Configuration

### Step 1: Create an App Registration

1. Navigate to the [Azure Portal](https://portal.azure.com)
2. Go to **Azure Active Directory** → **App registrations**
3. Click **New registration**
4. Configure the app registration:
   - **Name**: `mcp-gateway-web` (or your preferred name)
   - **Supported account types**: Select the appropriate option:
     - **Single tenant** (recommended): Only users in your organization
     - **Multi-tenant**: Users from any Azure AD tenant
   - **Redirect URI**:
     - Platform: **Web**
     - URI: `http://localhost/auth/callback` (for local development)
     - For production, use: `https://your-domain.com/auth/callback`
5. Click **Register**

### Step 2: Note Your Application IDs

After creating the app registration, note the following values (you'll need them later):

1. From the app registration **Overview** page:
   - **Application (client) ID**: This is your `ENTRA_CLIENT_ID`
   - **Directory (tenant) ID**: This is your `ENTRA_TENANT_ID`

### Step 3: Create a Client Secret

1. In your app registration, click **Certificates & secrets** in the left menu
2. Click **New client secret**
3. Configure the secret:
   - **Description**: `mcp-gateway-auth` (or your preferred description)
   - **Expires**: Choose an appropriate expiration period (recommended: 24 months)
4. Click **Add**
5. **IMPORTANT**: Copy the **Value** immediately (not the Secret ID)
   - This is your `ENTRA_CLIENT_SECRET`
   - You cannot retrieve this value later - if you lose it, you'll need to create a new secret

### Step 4: Configure Redirect URIs

1. In your app registration, click **Authentication** in the left menu
2. Under **Platform configurations** → **Web**, add redirect URIs:
   - For local development: `http://localhost/auth/callback`
   - For production: `https://your-domain.com/auth/callback`
3. Under **Implicit grant and hybrid flows**, ensure nothing is checked (not needed for authorization code flow)
4. Click **Save**

### Step 5: Add API Permissions

The Entra browser login requests `openid`, `email`, and `profile` by default. Set `ENTRA_BROWSER_USER_READ_ENABLED=true`
in Docker or Terraform, or `auth-server.entra.browserUserReadEnabled: true` in Helm, to append exactly `User.Read` to
the login request. The callback records the requested scopes in the signed login state and uses Graph when claims are
missing or overage-marked, provided group recovery is enabled. The default avoids a new consent requirement; without
`User.Read`, missing claims yield no group-derived scopes. Microsoft lists delegated `User.Read` as the least-privileged
permission for [
`/me/transitiveMemberOf`](https://learn.microsoft.com/en-us/graph/api/user-list-transitivememberof?view=graph-rest-1.0).
Delegated `GroupMember.Read.All` is not needed.

1. Click **API permissions** in the left menu
2. Click **Add a permission** and select **Microsoft Graph**
3. Select **Delegated permissions**
4. Add `User.Read` (it may already appear in the app registration)
5. Click **Add permissions**

Before adding `User.Read` to the login scopes, confirm that users can grant it or grant tenant-wide administrator
consent. Otherwise Entra can refuse a login before the callback runs, including one with inline group claims.

The callback calls Graph as the signed-in user with an access token issued for the Microsoft Graph resource (
`https://graph.microsoft.com` in the public cloud), not an ID token or a token for the registry. Entra access tokens may
identify Graph in their `aud` claim by its application ID (`00000003-0000-0000-c000-000000000000`). Sovereign clouds use
their corresponding Graph resource. No **Application permissions** are needed for this login lookup.

For user-delegated bearer access to the registry, `/validate` starts with an
access token for the registry. When group claims are missing or show overage,
it exchanges that token on behalf of the user for a Graph-audience token with
delegated `User.Read`. It then calls `/me/transitiveMemberOf`. The login
callback's Graph token cannot replace this exchange for a bearer request.
Neither lookup uses Graph **Application permissions**. Entra M2M `roles`
tokens do not use the on-behalf-of exchange.

`ENTRA_GRAPH_BASE_URL` selects the HTTPS Graph fetch endpoint. Leave it unset for a recognized Entra login cloud to use
that cloud's Graph host. To use a private reverse proxy, set it to the proxy's HTTPS origin. The guarded transport pins
DNS answers and blocks metadata and link-local addresses. Only the configured proxy hostname may resolve to a private
address; Microsoft Graph hostnames must remain public. The proxy must present a certificate the auth server trusts and
forward to the selected cloud's Graph service.

`ENTRA_GRAPH_RESOURCE_URL` selects the native Microsoft Graph OAuth audience and its delegated `User.Read` scope. It
does not route network traffic through that hostname. The auth server infers this resource for public, US Government,
and China login origins. An unrecognized HTTPS `ENTRA_LOGIN_BASE_URL` proxy always needs `ENTRA_GRAPH_BASE_URL` for the
Graph fetch endpoint. It also needs an official `ENTRA_GRAPH_RESOURCE_URL` when group recovery or the browser
`User.Read` request is enabled. With both off, the auth server requests no Graph audience. Only
`https://graph.microsoft.com`, `https://graph.microsoft.us`, `https://dod-graph.microsoft.us`, and
`https://microsoftgraph.chinacloudapi.cn` are valid resources. The configured resource must match a known login cloud;
`login.microsoftonline.us` also accepts the DoD resource.

With a proxy, the auth server rewrites Graph pagination links only from the configured native resource to the configured
fetch endpoint. Links to other clouds, hosts, or paths fail without forwarding the token. A login proxy must forward OBO
exchanges to the same Entra cloud that issued the incoming token. The auth server cannot prove which cloud an arbitrary
proxy serves, so the operator must supply the correct native resource and trusted proxy routing.

Entra browser code exchange sends the client secret through the HTTPS-only credentialed OAuth URL guard. If the login
proxy resolves to a private IP, add its exact hostname to `EGRESS_OAUTH_TRUSTED_IDP_HOSTS`; this opt-in does not apply
to Graph's fetch proxy. Leave the list empty for public login hosts.

#### Consent for the login lookup

Consent to delegated `User.Read` depends on your tenant's **User consent for applications** policy. If that policy
permits users to consent to this app and permission, each user can approve the sign-in prompt. Otherwise, ask a tenant
administrator to grant consent under **App registrations** > your app > **API permissions** > **Grant admin consent
for [Your Tenant]**. Check **Enterprise apps** > **Consent and permissions** > **User consent settings** in
the [Microsoft Entra admin center](https://entra.microsoft.com) to see the tenant policy. If the app requires users to
be assigned to it, an administrator must consent even when user consent is allowed.
See [Microsoft's user consent settings](https://learn.microsoft.com/en-us/entra/identity/enterprise-apps/configure-user-consent).

The bearer on-behalf-of exchange needs consent for this app to access Graph
as the user. Consent to `User.Read` during browser login alone does not
guarantee that Entra has granted the consent needed for the exchange. Check
the app registration's delegated Graph permission and the tenant's consent
policy for the bearer flow. Depending on that policy, a tenant administrator
may need to grant consent to the app.

The IAM management UI and API use a separate server-side client credentials flow. Their **Application permissions**
still require administrator consent. Granting delegated `User.Read` for login does not grant those IAM permissions.

#### Application Permissions (Required for IAM Management)

If you plan to use the IAM management UI (Settings -> Users/Groups) or the IAM API to manage users, groups, and M2M accounts, you must also add **Application permissions**. These are used by the server-side client credentials flow to call the Microsoft Graph API.

1. Click **Add a permission**
2. Select **Microsoft Graph**
3. Select **Application permissions**
4. Search for and add the following permissions:

   **Read-only access (minimum for IAM UI to list users and groups):**
   - `User.Read.All` - Read all users' full profiles
   - `Group.Read.All` - Read all groups
   - `GroupMember.Read.All` - Read all group memberships

   **Read-write access (required to create/delete users, groups, and M2M accounts):**
   - `User.ReadWrite.All` - Create, update, and delete users
   - `Group.ReadWrite.All` - Create, update, and delete groups
   - `GroupMember.ReadWrite.All` - Manage group memberships
   - `Application.ReadWrite.All` - Create and manage M2M service principal accounts

5. Click **Add permissions**

### Step 6: Configure Optional Claims

To include email, username, and groups in the ID token:

1. Click **Token configuration** in the left menu
2. Click **Add optional claim**
3. Select **ID** token type
4. Add these claims:
   - `email` - User's email address
   - `preferred_username` - User's UPN (User Principal Name)
   - `groups` - Security group Object IDs
5. Click **Add**
6. When prompted "Turn on the Microsoft Graph email, profile permission", click **Add**

### Step 7: Configure Group Claims

1. Still in **Token configuration**
2. Click **Add groups claim**
3. Select **Security groups**
4. Under "Customize token properties by type":
   - **ID**: Check "Group ID"
   - **Access**: Check "Group ID"
5. Click **Add**

### Step 8: Create Security Groups

Create Azure AD security groups for authorization:

1. Go to **Azure Active Directory** → **Groups**
2. Click **New group**
3. Create an admin group:
   - **Group type**: Security
   - **Group name**: `Mcp-test-admin` (or your preferred name)
   - **Group description**: MCP Gateway administrators
   - **Membership type**: Assigned
4. Click **Create**
5. Repeat for a users group:
   - **Group name**: `mcp-test-users` (or your preferred name)
   - **Group description**: MCP Gateway users

### Step 9: Note Group Object IDs

For each group you created:

1. Click on the group name
2. From the **Overview** page, copy the **Object Id**
3. Note these IDs - you'll need them when you configure scope mappings (in the `mcp_scopes` collection)

### Step 10: Add Users to Groups

1. For each group, click on the group name
2. Click **Members** in the left menu
3. Click **Add members**
4. Search for and select users
5. Click **Select**

### Step 11: Configure App for API Access (Optional)

If you plan to use machine-to-machine (M2M) authentication:

1. Click **Expose an API** in the left menu
2. Click **Add** next to "Application ID URI"
3. Accept the default (`api://{client-id}`) or customize it
4. Click **Save**
5. Click **Add a scope**
6. Configure the scope:
   - **Scope name**: `.default`
   - **Who can consent**: Admins only
   - **Admin consent display name**: Access MCP Gateway
   - **Admin consent description**: Allow the application to access MCP Gateway
   - **State**: Enabled
7. Click **Add scope**

---

## Environment Configuration

### Step 1: Update .env File

1. Copy `.env.example` to `.env` if you haven't already:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and configure Entra ID settings:

```bash
# =============================================================================
# AUTHENTICATION PROVIDER CONFIGURATION
# =============================================================================
# Choose authentication provider: 'cognito', 'keycloak', or 'entra'
AUTH_PROVIDER=entra

# =============================================================================
# MICROSOFT ENTRA ID CONFIGURATION
# =============================================================================

# Azure AD Tenant ID (from Azure Portal → App registration → Overview)
ENTRA_TENANT_ID=12345678-1234-1234-1234-123456789012

# Entra ID Application (client) ID (from Azure Portal → App registration → Overview)
ENTRA_CLIENT_ID=87654321-4321-4321-4321-210987654321

# Entra ID Client Secret (from Azure Portal → App registration → Certificates & secrets)
ENTRA_CLIENT_SECRET=your-secret-value-here

# Enable Entra ID in OAuth2 providers
ENTRA_ENABLED=true

# Azure AD Group Object IDs (from Azure Portal → Groups → Overview)
# IMPORTANT: ENTRA_GROUP_ADMIN_ID is required for admin access to persist across restarts.
# This group ID is added to the registry-admins scope in MongoDB during initialization.
# Users in this Entra group will have full admin access to the MCP Gateway.
ENTRA_GROUP_ADMIN_ID=16c7e67e-e8ae-498c-ba2e-0593c0159e43
ENTRA_GROUP_USERS_ID=62c07ac1-03d0-4924-90c7-a0255f23bd1d

# OAuth scope-advertisement format for coding-assistant discovery (PRM).
# Leave unset (default 'v2') unless your app exposes v1-style api:// scopes and
# you hit AADSTS650053 at login. See docs/entra-scope-format.md.
# ENTRA_SCOPE_FORMAT=v2
# ENTRA_APPLICATION_ID_URI=api://87654321-4321-4321-4321-210987654321
```

> **Coding assistants (Claude Code / Cursor / VS Code):** if you connect an IDE
> to the gateway over OAuth and hit `AADSTS650053: ... asked for scope ... that
> doesn't exist on the resource`, your app exposes v1-style `api://` scopes; set
> `ENTRA_SCOPE_FORMAT=v1` (and `ENTRA_APPLICATION_ID_URI` if you use a custom
> Application ID URI). Full guidance: [Entra v1 vs v2 scope format](entra-scope-format.md).

3. Update other required settings:

```bash
# =============================================================================
# REGISTRY CONFIGURATION
# =============================================================================
# For local development
REGISTRY_URL=http://localhost

# For production with custom domain
# REGISTRY_URL=https://mcpgateway.mycorp.com

# =============================================================================
# AUTH SERVER CONFIGURATION
# =============================================================================
# For local development
AUTH_SERVER_EXTERNAL_URL=http://localhost

# For production with custom domain
# AUTH_SERVER_EXTERNAL_URL=https://mcpgateway.mycorp.com

# =============================================================================
# APPLICATION SECURITY
# =============================================================================
# CRITICAL: CHANGE THIS SECRET KEY IMMEDIATELY!
SECRET_KEY=your-super-secure-random-64-character-string-here
```

---

## Group Configuration

### Configure Scope Mappings

Scope mappings live in the `mcp_scopes` collection in DocumentDB / MongoDB. They are seeded from the JSON scope files in `scripts/` at init time, and can also be managed through the scope management API or the registry UI's IAM Settings page. There is no `scopes.yml` file to edit on a current install.

To map Azure AD groups to MCP Gateway scopes and permissions, add the group Object IDs to the `group_mappings` section of the relevant scope document. The mapping has this shape:

```yaml
group_mappings:
  # Entra ID group mappings (by Azure AD Group Object IDs)
  # Admin group
  "object_id":
  - mcp-registry-admin
  - registry-admins

```

Replace the group Object IDs with your actual group IDs from Azure Portal, then seed the JSON scope files in `scripts/` (or apply the change via the scope management API).

### Group membership at sign-in

The Entra callback uses verified inline `groups` or `roles` claims. If that login requested delegated `User.Read` and
the ID token marks group overage with `hasgroups` or `_claim_names.groups`, or supplies neither `groups` nor `roles`,
the callback calls Microsoft Graph `/me/transitiveMemberOf` with the signed-in user's Graph access token. Graph returns
direct and nested group IDs for scope mappings.
See [Microsoft's transitive membership API](https://learn.microsoft.com/en-us/graph/api/user-list-transitivememberof?view=graph-rest-1.0).

The callback requires a signed Entra ID token and verifies its nonce before reading group claims. It rejects a token
response without an ID token instead of calling Graph userinfo with an unverified bearer assertion.

Without a signed `User.Read` request, the callback does not call Graph. It uses verified inline claims when there is no
overage marker. On overage, it discards unreliable inline groups and keeps only verified app roles.
`ENTRA_GROUP_RECOVERY_ENABLED=false` applies the same rule even after a `User.Read` request. A missing group claim
remains empty.

When recovery runs, the callback preserves verified app roles alongside Graph group IDs and ignores inline group IDs
marked unreliable by overage, even when Graph confirms no memberships.

The Graph lookup cache keeps successful results for `ENTRA_GROUP_CACHE_TTL_SECONDS` (default 300, allowed 30 to 3600)
per auth-server process. It holds at most 512 entries keyed by the Graph fetch URL, native Graph resource, verified
tenant ID (`tid`), and user object ID (`oid`); without both IDs, the callback does not cache the result. Graph failures
never become successful empty memberships. A failed lookup refuses login with HTTP 503 and creates no session; a
confirmed empty membership list creates a session without group-derived scopes unless verified app roles grant access.

A confirmed Graph result, including zero groups, takes precedence over `IDP_USER_GROUP_FALLBACK_ENABLED_PROVIDERS` local
user-group mappings. The session store records this Graph provenance so later session-cookie validations cannot restore
a removed Entra group. If Graph did not run, the configured local fallback retains its existing behavior.

When the session mints a user token, the auth server copies Graph-result provenance from the resolved session into its
signed token. A later `/validate` request keeps a confirmed empty membership authoritative instead of restoring groups
from local fallback. Tokens minted without a resolved session cannot assert that provenance.

The first Graph request asks for up to 999 memberships, Graph's maximum page size. The auth server follows subsequent
`@odata.nextLink` pages without changing their query. If a configured Graph proxy receives a continuation link on the
selected native Graph resource, the auth server rewrites it to the proxy before sending the bearer token. Links to other
hosts or paths fail. Results above 100 pages or 1,000 distinct groups fail rather than granting scopes from a partial
list.

Group-based access in an existing login session reflects the membership snapshot taken at sign-in. Removing someone from
a group may leave their current session authorized until they sign in again or the session expires. A new login can
reuse group IDs until the configured positive cache TTL expires.

### Group membership for user-delegated bearer tokens

For a verified Entra user-delegated registry access token with `scp`, `tid`,
and `oid`, `/validate` recovers missing `groups` claims and overage markers
(`hasgroups` or `_claim_names.groups`). It exchanges the incoming token on
behalf of the user for a Graph-audience `User.Read` token and fetches direct
and nested group IDs from `/me/transitiveMemberOf`. Entra M2M tokens that use
`roles` do not trigger this recovery. The auth server filters recovered IDs
with `filter_session_groups` before scope mapping and the `X-Groups` header.

Bearer recovery is enabled by default. Even a verified token with direct `scp` access needs a working Graph OBO exchange
when its group claims are absent; a consent or Graph outage returns 503 before direct-scope authorization. Grant
delegated Graph `User.Read` consent for the gateway's OBO exchange before rollout. Set
`ENTRA_GROUP_RECOVERY_ENABLED=false` to keep direct-scope bearer requests independent of Graph, at the cost of missing
group-derived scopes for users with absent or overage claims.

`ENTRA_GROUP_RECOVERY_ENABLED` defaults to `true`. Set it to `false` to stop Graph calls from browser and bearer
recovery. Inline verified groups remain usable when there is no overage marker. For a delegated bearer with missing
claims, validation has no group-derived scopes; when overage is marked, it discards unreliable inline groups but keeps
verified app roles. This may restore a 403 for users who need Graph memberships.

Entra reads bearer scopes from signed `scp` or, for older tokens, signed `scope`. Both classify a delegated user token,
so neither can inherit M2M client-group mappings. A token with no groups is authorized by its signed direct scopes.
When the user has inline or Graph-recovered groups, registry scopes come only from the group mappings, and the token's
`scp` values do not add to them. An app registration defines `scp` names and users grant them by consent, so they must
not stand in for the group membership an administrator controls. A successful Graph result, including zero groups,
takes precedence over local `idp_user_groups` fallback.

The scope-derived group mapping set has a 30-second per-process cache. Changes to the registry's group-to-scope mappings
can take up to 30 seconds to affect recovered bearer groups; lookup errors or absent mappings deny validation rather
than forwarding unfiltered groups. An explicit `ALLOWED_IDP_GROUPS` allowlist takes precedence.

If the filtered `X-Groups` value still exceeds 4 KiB, `/validate` refuses the request instead of filling nginx's 16 KiB
auth-subrequest header buffer. Registry API requests also carry an internal token encoding those groups, so the smaller
limit leaves room for both headers. Map only the groups that grant scopes, or set `ALLOWED_IDP_GROUPS` to the exact IDs
the gateway needs.

Successful lookups stay in a per-process cache for `ENTRA_GROUP_CACHE_TTL_SECONDS` (default 300 seconds, allowed 30 to
3600), with at most 512 entries keyed by Graph fetch URL, native Graph resource, and verified `(tid, oid)`. Concurrent
requests with the same exact bearer token share one in-flight OBO exchange; different tokens run their own exchanges
before they can use a successful membership result. A failed consent or configuration exchange, or a Graph 403, is
cached as a denial for 30 seconds per exact incoming token and identity; a new token can retry at once. Transient
failures and Graph 401 responses are retried on the next request. Denials never become empty successful memberships, are
not stored with the token bytes, and occupy at most 512 process-local entries. The OBO exchange and all Graph pages
share an eight-second deadline before nginx's ten-second `/validate` timeout.

Each bearer token's successful Graph exchange has a bounded process-local proof entry. The entry expires with the
positive membership result and never overrides a denial for that token. A browser lookup can fill the membership cache,
but it cannot authorize a bearer token's first OBO exchange.

Graph, consent, pagination, and mapping failures produce a 503 on `/validate`. nginx returns a public 503 only when that
internal response has the auth-server's Graph-outage marker; unrelated auth failures remain 500 and no denied request
reaches the backend. The auth-server log records a finite reason label such as `timeout`, `consent`, `pagination`, or
`transport` without recording group IDs or tokens. Group removals can take up to the configured positive cache TTL to
reach a bearer request. Existing browser sessions reflect the sign-in snapshot until sign-in or expiry.

### Understanding Scope Mappings

- **mcp-registry-admin**: Full administrative access to the registry
  - Can list, register, modify, and toggle services
  - Has unrestricted read and execute access to MCP servers

- **mcp-registry-user**: Limited user access
  - Can list and view specific services
  - Has restricted read access to MCP servers

- **mcp-registry-developer**: Development access
  - Can list, register, and health check services
  - Has restricted read and execute access

- **mcp-registry-operator**: Operations access
  - Can list, health check, and toggle services
  - Has restricted read and execute access

---

## Testing the Setup

### Step 1: Start the Services

1. Build and start the Docker containers:
   ```bash
   docker-compose up -d --build
   ```

2. Check that services are running:
   ```bash
   docker-compose ps
   ```

### Step 2: Test User Authentication

1. Open your browser and navigate to:
   ```
   http://localhost
   ```

2. You should see the MCP Gateway Registry login page

3. Click the **Sign in with Microsoft Entra ID** button

4. You will be redirected to Microsoft's login page

5. Sign in with a user account that belongs to one of your configured groups

6. After successful authentication, you should be redirected back to the registry

### Step 3: Verify User Information

1. Check the auth server logs to verify user information is being received:
   ```bash
   docker-compose logs auth-server | grep "Raw user info"
   ```

2. You should see output similar to:
   ```
   Raw user info from entra: {
     'sub': 'abc123...',
     'email': 'user@yourdomain.onmicrosoft.com',
     'preferred_username': 'user@yourdomain.onmicrosoft.com',
     'groups': ['16c7e67e-...', '62c07ac1-...'],
     'name': 'First Last'
   }
   ```

3. Verify the mapped scopes:
   ```bash
   docker-compose logs auth-server | grep "Mapped user info"
   ```

4. You should see:
   ```
   Mapped user info: {
     'username': 'user@yourdomain.onmicrosoft.com',
     'email': 'user@yourdomain.onmicrosoft.com',
     'name': 'First Last',
     'groups': ['mcp-registry-admin', 'mcp-servers-unrestricted/read', ...]
   }
   ```

### Step 4: Test Authorization

1. Log in with an admin user (member of the admin group)

2. Verify you can access admin functions:
   - Register new services
   - Modify service configurations
   - Toggle services on/off

3. Log in with a regular user (member of the users group)

4. Verify restricted access:
   - Can view services
   - Cannot register or modify services

### Step 5: Test Machine-to-Machine (M2M) Authentication

If you configured API access for M2M authentication:

1. Create a service principal for your AI agent:
   ```bash
   # This is done in Azure Portal → App registrations
   # Create a new app registration for the AI agent
   ```

2. Test M2M token generation:
   ```bash
   curl -X POST "https://login.microsoftonline.com/{tenant-id}/oauth2/v2.0/token" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "grant_type=client_credentials" \
     -d "client_id={agent-client-id}" \
     -d "client_secret={agent-client-secret}" \
     -d "scope=api://{mcp-gateway-client-id}/.default"
   ```

3. Use the access token to call MCP Gateway APIs

---

## Troubleshooting

### Issue: 403 Forbidden when using IAM management UI

**Symptoms:**
```
Client error '403 Forbidden' for url 'https://graph.microsoft.com/v1.0/users?...'
```

**Cause:**
The IAM management features (listing users, creating users/groups, managing M2M accounts) use the OAuth2 client credentials flow to call the Microsoft Graph API. This flow requires **Application permissions**, not Delegated permissions. If only Delegated permissions are configured, the Graph API will return 403 Forbidden.

**Solution:**
1. Go to Azure Portal -> App registrations -> Your app -> **API permissions**
2. Click **Add a permission** -> **Microsoft Graph** -> **Application permissions**
3. Add the required Application permissions (see [Step 5: Application Permissions](#application-permissions-required-for-iam-management))
4. Click **Grant admin consent for [Your Tenant]**
5. Wait 5-10 minutes for Azure AD to propagate changes
6. Restart the registry service

### Issue: Missing email or group claims

If the email claim is absent, check [optional claims](#step-6-configure-optional-claims). For group-based scopes, the
callback uses verified inline `groups` or `roles`. For overage or missing claims, add `User.Read` to the Entra login
scopes and confirm delegated consent before expecting Graph recovery.
Check [group membership at sign-in](#group-membership-at-sign-in) and the signed-in token's Graph audience. Configure
group claims in [Step 7](#step-7-configure-group-claims) if you want inline group IDs. A Graph lookup failure denies
login; a confirmed empty result yields no group-derived scopes.

### Issue: Token validation fails with "Invalid issuer"

**Symptoms:**
```
Token validation failed: Invalid issuer: https://sts.windows.net/{tenant}/
```

**Solution:**
The Entra ID provider supports both v1.0 and v2.0 token formats. This error should not occur with the current implementation. If you see this:

1. Check that `ENTRA_TENANT_ID` in `.env` matches your actual tenant ID
2. Verify the token is being issued by Microsoft Entra ID
3. Check auth server logs for more details

### Issue: User cannot access any resources

**Symptoms:**
User can log in but sees "Access Denied" or "Insufficient Permissions"

**Solution:**
1. Verify the user is added to at least one security group in Azure AD
2. Check that group Object IDs in the `mcp_scopes` collection match the groups in Azure Portal
3. Verify the group mappings include the necessary scopes
4. Check auth server logs to see what groups are being received:
   ```bash
   docker-compose logs auth-server | grep "groups"
   ```

### Issue: Redirect URI mismatch error

**Symptoms:**
```
AADSTS50011: The redirect URI 'http://localhost/auth/callback' does not match the redirect URIs configured for the application
```

**Solution:**
1. Go to Azure Portal → App registrations → Your app → Authentication
2. Verify the redirect URI exactly matches what's in the error message
3. Add any missing redirect URIs
4. Ensure `AUTH_SERVER_EXTERNAL_URL` in `.env` matches the base URL

### Issue: Group overage or missing group claims

Entra may include `hasgroups` or `_claim_names.groups` instead of inline group IDs when a token exceeds its group limit.
A browser login that requests `User.Read` fetches direct and nested memberships through `/me/transitiveMemberOf` for
overage or missing claims. Without that requested scope, login uses inline claims only when there is no overage marker.
It drops overage-marked inline groups and makes no Graph request. Check consent and token audience
in [Step 5](#step-5-add-api-permissions). Graph failures deny the opted-in login; they are not empty membership results.
See [group membership at sign-in](#group-membership-at-sign-in) for cache and session behavior.

For user-delegated bearer requests, `/validate` also recovers missing or
overage group claims using an on-behalf-of Graph `User.Read` token. If it
returns 503, check that the registry token has a delegated `scp` claim and
verified `tid` and `oid`, that Graph consent covers the on-behalf-of exchange,
and that Graph is reachable. The browser login consent prompt alone may not
cover this exchange. The request is not forwarded when recovery fails.

### Issue: Client secret expired

**Symptoms:**
```
AADSTS7000215: Invalid client secret provided
```

**Solution:**
1. Go to Azure Portal → App registrations → Your app → Certificates & secrets
2. Create a new client secret
3. Update `ENTRA_CLIENT_SECRET` in `.env`
4. Restart the services:
   ```bash
   docker-compose restart auth-server
   ```

### Issue: Cannot grant consent

For login, check whether your
tenant's [user consent policy](https://learn.microsoft.com/en-us/entra/identity/enterprise-apps/configure-user-consent)
allows the signed-in user to consent to delegated `User.Read`. If it does not, ask a tenant administrator to grant
consent to the app. Administrator consent is required for the separate IAM **Application permissions** listed
in [Step 5](#application-permissions-required-for-iam-management).

The user-delegated bearer on-behalf-of exchange to Graph also needs consent.
Browser login with `User.Read` does not guarantee that consent for this
exchange exists. Check your tenant's policy and ask an administrator to
grant consent if the policy requires it.

---

## Additional Resources

- [Microsoft Entra ID Documentation](https://learn.microsoft.com/en-us/entra/)
- [OAuth 2.0 Authorization Code Flow](https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-auth-code-flow)
- [Optional Claims Configuration](https://learn.microsoft.com/en-us/entra/identity-platform/optional-claims)
- [Configure Group Claims](https://learn.microsoft.com/en-us/entra/identity-platform/optional-claims#configure-groups-optional-claims)
- [Microsoft Graph Permissions Reference](https://learn.microsoft.com/en-us/graph/permissions-reference)

---

## Production Deployment

### Update Redirect URIs

For production, update redirect URIs:
```
https://your-domain.com/oauth2/callback/entra
```

### Environment Variables

Update production `.env`:
```bash
ENTRA_REDIRECT_URI=https://your-domain.com/oauth2/callback/entra
AUTH_SERVER_EXTERNAL_URL=https://your-domain.com:8888
```

### SSL/TLS Configuration

Ensure your production deployment uses HTTPS for all OAuth flows.

---

## Advanced Configuration

### Custom Claims

To add custom claims to tokens:
1. Go to **Token configuration**
2. Click **Add optional claim**
3. Select token type and claims
4. Configure claim conditions

### Group Filtering

To limit which groups are included in tokens:
1. Go to **Token configuration**
2. Click **Add groups claim**
3. Configure **Groups assigned to the application**

### Enterprise Applications

For advanced management:
1. Go to **Enterprise applications**
2. Find your app registration
3. Configure:
   - User assignment required
   - Visibility settings
   - Provisioning (if needed)

---

## Adding New Users

### Option 1: Add User to Existing Group (Recommended)

**In Azure Portal:**
1. Go to **Microsoft Entra ID** → **Groups**
2. Click on **MCP Registry Admins** (or appropriate group)
3. Click **Members** → **Add members**
4. Search and select the new user
5. Click **Select**

**Access will be immediate** - user can login and see servers/agents.

### Option 2: Create New Group for User

**If you need different permissions:**

1. **Create new group in Azure:**
   - **Group name**: `MCP Registry LOB3 Users`
   - **Members**: Add the new user

2. **Get the group Object ID** from the group overview page

3. **Add the group mapping to the `mcp_scopes` collection** (via a JSON scope seed file in `scripts/` or the scope management API):
```yaml
group_mappings:
  # Add new group mapping
  "new-group-object-id-here":
  - registry-users-lob1  # or whatever permission level needed
```

4. **Apply the change.** Scopes are seeded into the `mcp_scopes` collection, not copied as a file. The auth server picks up DB-backed edits on its next reload (no compose restart required for scope changes).

---

## API Reference

### Token Endpoint
```
POST https://login.microsoftonline.com/{tenant-id}/oauth2/v2.0/token
```

### Authorization Endpoint
```
GET https://login.microsoftonline.com/{tenant-id}/oauth2/v2.0/authorize
```

### User Info Endpoint
```
GET https://graph.microsoft.com/v1.0/me
```

---

## Security Best Practices

1. **Client Secret Management**
   - Store client secrets securely (use Azure Key Vault in production)
   - Rotate secrets regularly (set expiration and create new secrets)
   - Never commit secrets to version control

2. **Token Configuration**
   - Keep token expiration times reasonable (default: 1 hour for access tokens)
   - Use refresh tokens for long-running sessions
   - Implement proper token revocation

3. **Group Management**
   - Use security groups (not distribution lists or Microsoft 365 groups)
   - Apply principle of least privilege
   - Regularly audit group memberships

4. **HTTPS in Production**
   - Always use HTTPS in production environments
   - Configure proper SSL/TLS certificates
   - Update redirect URIs to use HTTPS

5. **Monitoring and Logging**
   - Enable Azure AD audit logs
   - Monitor sign-in logs for suspicious activity
   - Set up alerts for authentication failures

6. **Multi-Factor Authentication**
   - Enable MFA for all users (configured in Azure AD)
   - Use conditional access policies
   - Enforce MFA for admin accounts

---

## Using the IAM API to Manage Groups, Users, and M2M Accounts

The MCP Gateway Registry provides an IAM API for managing groups, human users, and M2M (machine-to-machine) service accounts programmatically. This section covers how to use the `registry_management.py` CLI to perform these operations.

### Prerequisites

Before using the IAM API commands, you need:

1. **An admin access token**: Either a self-signed token from the UI sidebar or a Keycloak/Entra ID token for an admin user
2. **Registry URL**: The URL of your MCP Gateway Registry deployment
3. **Admin group membership**: Your user must be in the `registry-admins` group

Save your token to a file for CLI usage:
```bash
# Save token from UI sidebar to a file
echo "eyJhbGci..." > api/.token
```

### Creating a Group (Scope)

Groups define access permissions for users and M2M accounts. Create a group definition JSON file:

**Example: `cli/examples/public-mcp-users.json`**
```json
{
  "scope_name": "public-mcp-users",
  "description": "Users with access to public MCP servers",
  "server_access": [
    {
      "server": "context7",
      "methods": ["initialize", "tools/list", "tools/call"],
      "tools": ["*"]
    },
    {
      "server": "api",
      "methods": ["initialize", "GET", "POST", "servers", "agents"],
      "tools": []
    },
    {
      "agents": {
        "actions": [
          {"action": "list_agents", "resources": ["/flight-booking"]},
          {"action": "get_agent", "resources": ["/flight-booking"]}
        ]
      }
    }
  ],
  "group_mappings": ["public-mcp-users"],
  "ui_permissions": {
    "list_service": ["all"],
    "list_agents": ["/flight-booking"],
    "get_agent": ["/flight-booking"]
  },
  "create_in_idp": true
}
```

**Import the group:**
```bash
uv run python api/registry_management.py \
  --token-file api/.token \
  --registry-url https://your-registry-url.example.com \
  import-group --file cli/examples/public-mcp-users.json
```

**Key fields in group definition:**

| Field | Description |
|-------|-------------|
| `scope_name` | Unique identifier for the scope/group |
| `description` | Human-readable description |
| `server_access` | Array of server access rules |
| `group_mappings` | List of IdP group names/IDs that map to this scope |
| `ui_permissions` | Permissions for the web UI |
| `create_in_idp` | If `true`, creates corresponding group in Entra ID |

### Creating a Human User

Human users can log in via the web UI using Entra ID authentication.

```bash
uv run python api/registry_management.py \
  --token-file api/.token \
  --registry-url https://your-registry-url.example.com \
  user-create-human \
  --username jsmith \
  --email jsmith@example.com \
  --first-name John \
  --last-name Smith \
  --groups public-mcp-users \
  --password "SecurePassword123!"
```

**Parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--username` | Yes | Username for the account |
| `--email` | Yes | Email address |
| `--first-name` | Yes | User's first name |
| `--last-name` | Yes | User's last name |
| `--groups` | Yes | Comma-separated list of groups |
| `--password` | No | Initial password (auto-generated if not provided) |

### Creating an M2M Service Account

M2M (machine-to-machine) accounts are used for programmatic API access, AI coding assistants, and agent identities.

```bash
uv run python api/registry_management.py \
  --token-file api/.token \
  --registry-url https://your-registry-url.example.com \
  user-create-m2m \
  --name my-ai-agent \
  --groups public-mcp-users \
  --description "AI coding assistant service account"
```

**Output:**
```
Client ID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
Client Secret: xxxxx~xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
Groups: public-mcp-users
Service Principal ID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

IMPORTANT: Save the client secret securely - it cannot be retrieved later.
```

**Parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--name` | Yes | Service account name/client ID |
| `--groups` | Yes | Comma-separated list of groups |
| `--description` | No | Account description |

### Where to Find Parameter Values

| Parameter | Location |
|-----------|----------|
| **Tenant ID** | Azure Portal -> Microsoft Entra ID -> Overview -> Tenant ID |
| **App Client ID** | Azure Portal -> App registrations -> [Your App] -> Application (client) ID |
| **App Client Secret** | Azure Portal -> App registrations -> [Your App] -> Certificates & secrets |
| **Group Object ID** | Azure Portal -> Microsoft Entra ID -> Groups -> [Group Name] -> Object Id |
| **M2M Client ID/Secret** | Output from `user-create-m2m` command |

---

## Generating JWT Tokens for M2M Accounts

M2M accounts use OAuth2 client credentials flow to obtain JWT tokens. These tokens can be used for:

- Agent identities in A2A (Agent-to-Agent) communication
- AI coding assistants (Cursor, VS Code, etc.)
- Programmatic API access
- Automated scripts and CI/CD pipelines

### Method 1: Direct Token Request (curl)

Request a token directly from Microsoft Entra ID:

```bash
curl -X POST "https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id={M2M_CLIENT_ID}" \
  -d "client_secret={M2M_CLIENT_SECRET}" \
  -d "scope=api://{APP_CLIENT_ID}/.default" \
  -d "grant_type=client_credentials"
```

**Example with placeholder values:**
```bash
curl -X POST "https://login.microsoftonline.com/your-tenant-id/oauth2/v2.0/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id=your-m2m-client-id" \
  -d "client_secret=your-m2m-client-secret" \
  -d "scope=api://your-app-client-id/.default" \
  -d "grant_type=client_credentials"
```

**Response:**
```json
{
  "token_type": "Bearer",
  "expires_in": 3599,
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOi..."
}
```

### Method 2: Using the Credentials Provider Script

The `generate_creds.sh` script automates token generation for multiple identities.

**Step 1: Configure identities file**

Create or edit `.oauth-tokens/entra-identities.json`:
```json
[
  {
    "identity_name": "my-ai-agent",
    "tenant_id": "your-tenant-id",
    "client_id": "your-m2m-client-id",
    "client_secret": "your-m2m-client-secret",
    "scope": "api://your-app-client-id/.default"
  }
]
```

**Identities File Structure:**

| Field | Required | Description |
|-------|----------|-------------|
| `identity_name` | Yes | Unique name for this identity (used for output file naming) |
| `tenant_id` | Yes | Azure AD Tenant ID (from Azure Portal -> Microsoft Entra ID -> Overview) |
| `client_id` | Yes | M2M service account Client ID (from `user-create-m2m` output) |
| `client_secret` | Yes | M2M service account Client Secret (from `user-create-m2m` output) |
| `scope` | Yes | OAuth2 scope in format `api://{APP_CLIENT_ID}/.default` |

**Multiple Identities Example:**
```json
[
  {
    "identity_name": "cursor-assistant",
    "tenant_id": "your-tenant-id",
    "client_id": "cursor-m2m-client-id",
    "client_secret": "cursor-m2m-client-secret",
    "scope": "api://your-app-client-id/.default"
  },
  {
    "identity_name": "ci-pipeline",
    "tenant_id": "your-tenant-id",
    "client_id": "cicd-m2m-client-id",
    "client_secret": "cicd-m2m-client-secret",
    "scope": "api://your-app-client-id/.default"
  }
]
```

**Step 2: Set auth provider**

Ensure `AUTH_PROVIDER=entra` is set in your `.env` file.

**Step 3: Run the script**

```bash
./credentials-provider/generate_creds.sh
```

Or with a custom identities file:
```bash
uv run credentials-provider/entra/get_m2m_token.py \
  --identities-file /path/to/my-identities.json \
  --output-dir .oauth-tokens \
  --verbose
```

**Output:**
- Tokens are saved to `.oauth-tokens/{identity_name}.json`
- Each file contains the access token, expiration time, and metadata

### Token Scope Format

The scope for Entra ID M2M tokens follows this format:
```
api://{APP_CLIENT_ID}/.default
```

Where:
- `{APP_CLIENT_ID}` is the Application (client) ID of your MCP Gateway app registration
- `.default` requests all scopes that admin consent has been granted for

### Using Tokens in AI Coding Assistants

Once you have a JWT token, you can use it in AI coding assistants like Cursor or VS Code extensions:

1. **Configure the MCP server connection** with the registry URL
2. **Set the Bearer token** in the authorization header
3. **The token** grants access based on the M2M account's group membership

Example configuration for an AI assistant:
```json
{
  "mcp_registry_url": "https://your-registry-url.example.com",
  "auth_token": "eyJ0eXAiOiJKV1QiLCJhbGciOi..."
}
```

### User-Generated Tokens from the UI

Users can also generate personal JWT tokens from the MCP Gateway Registry web UI:

1. Log in to the registry at `https://your-registry-url.example.com`
2. Navigate to the sidebar
3. Click on "Generate Token" or similar option
4. Copy the generated token

These self-signed tokens:
- Are signed with HS256 using the server's secret key
- Include the user's groups and scopes
- Can be used for programmatic API access
- Work with the same endpoints as M2M tokens

---

## Next Steps

After completing the setup:

1. **Configure Additional Services**: Add more MCP servers to the registry
2. **Set Up Custom Domain**: Configure HTTPS and custom domain names
3. **Configure M2M Authentication**: Set up service principals for AI agents
4. **Implement Monitoring**: Set up observability and alerting
5. **Production Deployment**: Deploy to your production environment

For more information, see:
- [Complete Setup Guide](./complete-setup-guide.md)
- [Observability Documentation](./OBSERVABILITY.md)
- [FAQ](./faq/index.md)
