# How do I pass an M2M token from Entra to the registration gate?

## Question

I have an external registration gate (admission control webhook) that is protected by Microsoft Entra ID. I want the registry to automatically acquire a machine-to-machine (M2M) access token from Entra and pass it as a Bearer token to the gate endpoint. How do I configure this?

## Answer

The registry supports **OAuth2 Client Credentials** authentication for the registration gate. When configured, the registry acquires a fresh access token from your Entra tenant before each gate call and sends it as `Authorization: Bearer <token>` to the gate endpoint.

This uses the standard [OAuth2 Client Credentials flow (RFC 6749 Section 4.4)](https://datatracker.ietf.org/doc/html/rfc6749#section-4.4), which is the M2M authentication pattern where a service (the registry) authenticates with a client ID and secret to obtain an access token, with no user interaction required.

### Step 1: Create an App Registration in Entra

If you do not already have one, create an App Registration in the Azure Portal:

1. Go to **Azure Portal > App registrations > New registration**
2. Name it (e.g., `mcp-registry-gate-client`)
3. Under **Certificates & secrets**, create a new client secret and copy the value
4. Note the **Application (client) ID** and your **Tenant ID** from the Overview page
5. If your gate endpoint requires a specific audience, expose an API under **Expose an API** and note the Application ID URI (e.g., `api://your-app-id`)

### Step 2: Configure the Registry

Set these environment variables on the registry service:

```bash
# Enable the registration gate
REGISTRATION_GATE_ENABLED=true
REGISTRATION_GATE_URL=https://your-gate-endpoint.example.com/check

# Use OAuth2 Client Credentials authentication
REGISTRATION_GATE_AUTH_TYPE=oauth2_client_credentials

# Entra token endpoint (replace {tenant-id} with your Azure AD tenant ID)
REGISTRATION_GATE_OAUTH2_TOKEN_URL=https://login.microsoftonline.com/{tenant-id}/oauth2/v2.0/token

# App registration credentials
REGISTRATION_GATE_OAUTH2_CLIENT_ID=your-client-id
REGISTRATION_GATE_OAUTH2_CLIENT_SECRET=your-client-secret

# Entra scope (use the Application ID URI + /.default)
REGISTRATION_GATE_OAUTH2_SCOPE=api://your-app-id/.default
```

### Step 3: Verify

When the registry starts, it will:

1. Validate that the token URL, client ID, and client secret are all set
2. Attempt a test token acquisition to verify the credentials work
3. Log warnings if there are issues (check `docker compose logs registry`)

When an agent, server, or skill is registered, the registry will:

1. POST to the Entra token endpoint with `grant_type=client_credentials`
2. Receive an access token (JWT signed by Entra)
3. Send `Authorization: Bearer <entra-jwt>` to your gate endpoint
4. If the gate returns 200, the registration proceeds; if 403, it is denied

If token acquisition fails (invalid credentials, network error, timeout), the registration is **blocked immediately** (fail-closed design). The gate endpoint is never called with missing or invalid credentials.

### What the gate endpoint receives

The gate endpoint receives the Entra JWT in the `Authorization` header. You can decode the JWT to verify claims such as:

- `iss`: `https://sts.windows.net/{tenant-id}/` (your Entra tenant)
- `appid`: your client ID
- `aud`: the scope/audience you configured
- `tid`: your tenant ID

### Works with other IdPs too

While this example uses Entra, the same configuration pattern works with any OAuth2 provider that supports the client credentials grant:

- **Okta**: `https://{domain}/oauth2/default/v1/token`
- **Auth0**: `https://{tenant}.auth0.com/oauth/token`
- **Keycloak**: `https://{host}/realms/{realm}/protocol/openid-connect/token`
- **Cognito**: `https://cognito-idp.{region}.amazonaws.com/{user-pool-id}`

## Related documentation

- [Registration Webhooks and Gate](../registration-webhooks.md) for full configuration reference, including the OAuth2 parameter table and examples for multiple IdPs
- [Issue #917](https://github.com/agentic-community/mcp-gateway-registry/issues/917) for the design specification
