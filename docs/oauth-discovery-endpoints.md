# OAuth Discovery Endpoints

The gateway publishes two `.well-known` URLs so that spec-compliant MCP clients
(Claude Code, Claude.ai Custom Connectors, Cursor, and other coding
assistants) can auto-discover how to authenticate without operator-provided
configuration. This page is the operator-facing reference for those endpoints.

The endpoints implement the MCP 2025-06-18 authorization spec, which composes
several IETF specifications:

* RFC 9728 - OAuth 2.0 Protected Resource Metadata
* RFC 8414 - OAuth 2.0 Authorization Server Metadata
* RFC 8707 - Resource Indicators for OAuth 2.0
* RFC 7636 - PKCE (mandatory in OAuth 2.1)

## Endpoint summary

| URL | Spec | Cache-Control | Purpose |
| --- | --- | --- | --- |
| `<gateway>/.well-known/oauth-protected-resource` | RFC 9728 | `no-store` | Tells clients which authorization server protects the gateway and which scopes the gateway recognizes. |
| `<gateway>/.well-known/oauth-authorization-server` | RFC 8414 | `no-store` | Thin passthrough/normalization of the configured IdP's metadata. Smooths over provider-specific quirks (e.g. Cognito's split host). |

Both documents point clients at the authorization server and token endpoint, so
they are served with `Cache-Control: no-store`. This keeps shared/intermediary/
CDN caches from holding (and an attacker from poisoning) a copy that would
redirect clients to a malicious authorization server for the cache lifetime.

Both endpoints are public (no authentication required). They are served by
the registry FastAPI app and routed through nginx's existing `/.well-known/*`
proxy rule (no nginx changes needed for the endpoints themselves).

## How a client uses them

```
Client                                     Gateway                    IdP
  |                                          |                          |
  | GET /<server>/mcp  (no token)            |                          |
  |----------------------------------------->|                          |
  |                                          |                          |
  | 401  WWW-Authenticate: Bearer            |                          |
  |   realm="mcp",                           |                          |
  |   resource_metadata="<gateway>/.well-known/oauth-protected-resource"
  |<-----------------------------------------|                          |
  |                                          |                          |
  | GET /.well-known/oauth-protected-resource|                          |
  |----------------------------------------->|                          |
  | 200 { resource, authorization_servers, scopes_supported, ... }      |
  |<-----------------------------------------|                          |
  |                                          |                          |
  | GET /.well-known/oauth-authorization-server (or directly to the IdP per RFC 8414)
  |--------------------------------------------------------->| or |---->|
  | 200 { issuer, authorization_endpoint, token_endpoint, jwks_uri, ... }
  |<---------------------------------------------------------|         |
  |                                          |                          |
  | OAuth 2.1 + PKCE flow                    |                          |
  |...                                                                  |
  |                                                                     |
  | POST /<server>/mcp                                                  |
  |   Authorization: Bearer <access_token>                              |
  |----------------------------------------->|                          |
  | 200 (validated)                          |                          |
  |<-----------------------------------------|                          |
```

## Required configuration

The gateway derives both documents from existing settings; there is one new
required setting and one optional override.

### `registry_url` (existing)

The canonical public URL of the gateway. Must include scheme + host (and
port/path if non-default). Trailing slashes are normalized away. Used as the
`resource` field in PRM and as the audience that tokens must bind to per
RFC 8707.

```bash
REGISTRY_URL=https://mcpgateway.example.com
```

### `mcp_https_required` (new, defaults to `true`)

Refuses to serve a PRM document advertising an `http://` resource URL. Set to
`false` only in local development where the gateway runs over plaintext.

```bash
MCP_HTTPS_REQUIRED=true   # production default
MCP_HTTPS_REQUIRED=false  # local dev only
```

If `mcp_https_required=true` and `registry_url` is `http://...`, the registry
fails to start with a clear error.

### `mcp_resource_documentation_url` (new, optional)

Override URL for the `resource_documentation` field in the PRM document.
Defaults to `<registry_url>/docs/oauth` (this page).

```bash
MCP_RESOURCE_DOCUMENTATION_URL=https://docs.example.com/internal/mcp-oauth
```

## Provider-specific notes

The gateway picks the active provider via the existing `AUTH_PROVIDER` env
var. All five supported providers populate the AS-metadata document; only
two have notable quirks worth understanding for operators.

### Cognito: split-host rehoming

Cognito's OAuth surface lives on two hosts:

* Issuer + JWKS: `https://cognito-idp.<region>.amazonaws.com/<userPoolId>`
* `/authorize`, `/token`, `/userInfo`, `/logout`: `https://<domain>.auth.<region>.amazoncognito.com`

The gateway's AS-metadata response rehomes the OAuth endpoints onto the
cognito-domain host while keeping `issuer` and `jwks_uri` on the cognito-idp
host. From the client's perspective this is a single, consistent RFC 8414
document.

### Entra ID: v2 only in Phase 1

Phase 1 (this issue) emits Entra v2 metadata
(`https://login.microsoftonline.com/<tenant>/v2.0`). Tokens issued from the
v1 endpoint (`https://sts.windows.net/<tenant>/`) are still recognized by
the validator, but are not advertised in discovery. Verbatim
`api://<app-id>/<scope>` scope handling for v1 deployments is tracked in
sub-issue F (#990).

## Inspecting the endpoints

### From the gateway host

```bash
# PRM document
curl -s https://mcpgateway.example.com/.well-known/oauth-protected-resource | jq .

# AS metadata
curl -s https://mcpgateway.example.com/.well-known/oauth-authorization-server | jq .

# 401 with WWW-Authenticate (no token sent)
curl -i https://mcpgateway.example.com/airegistry-tools/mcp
# HTTP/2 401
# www-authenticate: Bearer realm="mcp", resource_metadata="https://mcpgateway.example.com/.well-known/oauth-protected-resource"
# ...
```

### Validating with a spec-compliant client

`claude mcp add https://mcpgateway.example.com` should drive the full
discovery chain: 401 -> PRM -> AS metadata -> PKCE auth-code flow -> first
tool call. With Keycloak / Auth0 / Okta (which support RFC 7591 Dynamic
Client Registration), no manual paste is required. With Cognito or Entra
public clients, you must pre-register an OAuth client with the redirect URI
the client expects (e.g. `https://claude.ai/api/mcp/auth_callback` for
Claude.ai's Custom Connector UI) and paste the resulting `client_id`/
`client_secret` into the connector's "Advanced settings" panel.

## Common diagnostic checks

| Symptom | Likely cause |
| --- | --- |
| `claude mcp add` fails with "no PRM" | Gateway not advertising PRM; check `<gateway>/.well-known/oauth-protected-resource` returns 200. |
| 401 on `/<server>/mcp` has no `WWW-Authenticate` | nginx `@auth_error` block missing the patch from this PR; redeploy with the new nginx confs and re-render. |
| `resource_metadata` URL in WWW-Authenticate header doesn't match PRM `resource` | Stale `REGISTRY_URL` in the gateway env vs. what nginx was rendered with. Restart the registry app to re-render config. |
| PRM endpoint returns 502 | Gateway can't reach the configured IdP. Check `AUTH_PROVIDER` env vars, network policy, and IdP health. |
| PRM endpoint returns 501 | The configured `AUTH_PROVIDER` lacks an `authorization_server_metadata()` implementation. Should not happen with the five built-in providers. |
| `AADSTS9010010: resource parameter ... doesn't match the requested scopes` (Entra, at IDE login) | The server's connection URL is not a registered Entra App ID URI, so Entra can't match the `resource`. On Entra every server uses a per-server PRM (`resource` = connection URL); register that URL as an `identifierUri` with a `user_impersonation` scope exposed + granted to the IDE client. See [Entra scope format](entra-scope-format.md) and the client-id method doc. |

## Related issues

* Umbrella: [#988](https://github.com/agentic-community/mcp-gateway-registry/issues/988) - Coding-Assistant OAuth Integration
* This issue (Phase 1): [#989](https://github.com/agentic-community/mcp-gateway-registry/issues/989) - PRM + AS metadata + WWW-Authenticate
* Phase 2 (Entra v1): [#990](https://github.com/agentic-community/mcp-gateway-registry/issues/990)
* Phase 3 (RFC 8707 enforcement + token proxy): [#991](https://github.com/agentic-community/mcp-gateway-registry/issues/991)
* Decision: no RFC 7591 DCR server-side: [#995](https://github.com/agentic-community/mcp-gateway-registry/issues/995)
