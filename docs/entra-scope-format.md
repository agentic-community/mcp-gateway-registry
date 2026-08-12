# Entra v1 vs v2 scope format (`ENTRA_SCOPE_FORMAT`)

When the MCP Gateway is protected by Microsoft Entra ID, spec-compliant coding
assistants (Claude Code, Cursor, VS Code) discover which scopes to request by
reading the gateway's OAuth Protected Resource Metadata (PRM) document at
`/.well-known/oauth-protected-resource`. Whatever the PRM lists in
`scopes_supported` is what the client copies onto its `/authorize` request.

Entra is strict about the *form* of a custom (resource) scope on `/authorize`,
and the required form differs between the two Entra token endpoints. If the PRM
advertises the wrong form, Entra rejects the authorization request with:

```
AADSTS650053: The application '<app>' asked for scope '<scope>' that doesn't
exist on the resource '<resource>'.
```

`ENTRA_SCOPE_FORMAT` tells the gateway which form to advertise.

## Which value do I set?

| Your Entra app exposes scopes as… | Set | PRM advertises `mcp.read` as… |
| --- | --- | --- |
| `api://<app-id>/<scope>` (v1 resource scopes) | `ENTRA_SCOPE_FORMAT=v1` | `api://<app-id>/mcp.read` |
| bare fragment scopes (v2, the modern default) | leave unset / `v2` | `mcp.read` |

**If you are unsure, leave it unset.** The default is `v2`, which matches how
most modern Entra app registrations expose scopes and is backward-compatible
with every existing deployment.

Set `ENTRA_SCOPE_FORMAT=v1` **only** if your app registration exposes v1-style
`api://` scopes and you are seeing `AADSTS650053` at login. This is a known
requirement for some field deployments and for the Amazon Bedrock AgentCore
Identity integration, which requires the `api://<app>/<scope>` scope string to
be preserved verbatim.

> Standard OIDC scopes (`openid`, `profile`, `email`, `offline_access`) are
> **always** advertised bare, regardless of `ENTRA_SCOPE_FORMAT`. Entra rejects
> `api://<app-id>/openid` even under v1, so the gateway never prefixes them.

## `ENTRA_APPLICATION_ID_URI`

The v1 prefix defaults to `api://<ENTRA_CLIENT_ID>`. If your app registration
uses a **custom** Application ID URI (Azure Portal → App registrations → your app
→ **Expose an API** → **Application ID URI**), set it explicitly so the
advertised scope prefix matches exactly what Entra expects:

```bash
ENTRA_SCOPE_FORMAT=v1
ENTRA_APPLICATION_ID_URI=api://mcp-gateway   # your custom URI
# PRM then advertises:  api://mcp-gateway/mcp.read
```

`ENTRA_APPLICATION_ID_URI` is also accepted as a valid token **audience** on the
inbound validation path (see below), so setting it keeps the outbound
advertisement and inbound acceptance aligned.

## Audience acceptance is asymmetric by design

The two directions are deliberately not symmetric:

- **Outbound (PRM advertisement).** The gateway advertises exactly **one** scope
  form — the one Entra expects on `/authorize` for the configured version — so
  the client sends the right string in the first place.
- **Inbound (token validation).** An Entra v1-issued access token may carry its
  `aud` claim as **either** the bare client-id GUID (`<app-id>`) **or** the URI
  form (`api://<app-id>`). The gateway accepts both as equivalent, plus any
  configured `ENTRA_APPLICATION_ID_URI` and per-server OBO resource audiences.
  This is a closed allowlist — never a wildcard, and `aud` verification is never
  disabled.

You do not need to configure the inbound side; dual-audience acceptance is
automatic.

## Do not auto-detect

The Entra issuer URL does encode the version (`/v1.0/` vs `/v2.0/`), but the
gateway does **not** auto-detect the scope format from it. Auto-detection would
hide operator intent and make login behavior unpredictable across token sources.
The behavior is explicit config only.

## Configuration surface

| Surface | Setting |
| --- | --- |
| `.env` / Docker Compose | `ENTRA_SCOPE_FORMAT=v1`, `ENTRA_APPLICATION_ID_URI=api://…` |
| Helm | `entra.scopeFormat: "v1"`, `entra.applicationIdUri: "api://…"` (auth-server + registry subcharts; `auth-server.entra.*` at the stack level) |
| Terraform (aws-ecs) | `entra_scope_format = "v1"`, `entra_application_id_uri = "api://…"` |
| CDK (`infra/`) | `entra.scopeFormat`, `entra.applicationIdUri` |

## Per-server App ID URI requirement (coding-assistant / IDE login)

Separate from the v1/v2 scope *format* above: when a spec-compliant coding
assistant (Claude Code) logs in to a gateway server on Entra, it performs RFC
8707 discovery and sends a `resource` parameter. Entra matches that `resource` to
a registered App ID URI (`identifierUris`) **exactly**, and App ID URIs cannot
end in a trailing slash. Because a client canonicalizes the bare gateway origin
with a trailing slash, the gateway-wide PRM's origin `resource` can never match
on Entra (`AADSTS9010010`).

The gateway therefore serves a **per-server PRM** for every server on Entra,
whose `resource` is the server's exact connection URL (e.g.
`https://<gateway>/<server>/mcp`). For each server you expose to a coding
assistant on Entra, register that connection URL as an App ID URI on the gateway
app and expose a `user_impersonation` scope on it (granted to the IDE public
client). `GET /api/egress/obo-identifier-uris` lists the exact URIs to register.
See the [client-id connection method](connection-methods/client-id.md#per-idp-setup)
for step-by-step setup.

This applies only to Entra. Lenient IdPs (Keycloak/Cognito/Okta/Auth0) accept the
bare-origin gateway-wide PRM and need no per-server registration.

## See also

- [Microsoft Entra ID Setup Guide](entra-id-setup.md)
- [Entra v1 vs v2 endpoint comparison](https://learn.microsoft.com/en-us/entra/identity-platform/azure-ad-endpoint-comparison)
- [AADSTS650053 error reference](https://learn.microsoft.com/en-us/entra/identity-platform/reference-error-codes)
