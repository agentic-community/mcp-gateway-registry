# How do I use my custom embeddings endpoint (OpenAI-compatible, e.g. LiteLLM) that needs an auth token from my IdP?

**Short answer**: set `EMBEDDINGS_AUTH_MODE=idp` and point the registry at your IdP's
OAuth2 token endpoint plus a client id/secret. On every embedding call the registry
fetches (and caches) an OAuth2 **client-credentials** bearer token from your IdP and
injects it as the `Authorization: Bearer` header on the OpenAI-compatible embedding
request, instead of using a static `EMBEDDINGS_API_KEY`. It works with any standard
OAuth2 IdP (Keycloak, Microsoft Entra ID, Okta, Auth0, PingFederate). The default is
`static`, so existing deployments are unaffected until you opt in.

## When to use this

Use this when your embedding endpoint is OpenAI-compatible (served through LiteLLM,
a proxy, or a gateway) but is **protected by your IdP** and expects a short-lived
bearer token rather than a fixed API key. The registry handles the token lifecycle
(fetch, cache, refresh-before-expiry) for you.

## Requirement: the endpoint must return the OpenAI embeddings *envelope*

The registry uses LiteLLM's OpenAI-compatible client, so your endpoint must accept
the standard request and return the standard **response envelope**:

```jsonc
// request  (what the registry sends)
{ "input": ["text one", "text two"], "model": "<your-model>" }

// response (what the endpoint MUST return)
{
  "object": "list",
  "data": [ { "object": "embedding", "index": 0, "embedding": [0.12, ...] } ],
  "model": "<your-model>",
  "usage": { "prompt_tokens": 0, "total_tokens": 0 }
}
```

If your endpoint instead returns a **bare array of vectors** (`[[0.12, ...], ...]`)
rather than the envelope, LiteLLM cannot parse it (it reads `data[i].embedding`). For
that case set `EMBEDDINGS_RESPONSE_FORMAT=raw_array`: the registry then calls the
endpoint directly and reads the bare array (it also accepts the same array wrapped
under a top-level `embeddings`, `data`, or `vectors` key). Leave the variable unset
(or `openai`) for standard envelope endpoints.

## Extra request headers

Some endpoints sit behind a gateway that routes or authorizes on a header other than `Authorization`, such as a tenant or module ID. Set one header per slot in `EMBEDDINGS_EXTRA_HEADER_1` through `EMBEDDINGS_EXTRA_HEADER_10`, in `Name: value` form:

```bash
EMBEDDINGS_EXTRA_HEADER_1=X-Tenant-Id: research
EMBEDDINGS_EXTRA_HEADER_2=X-Route: embeddings-v2
```

The registry adds these headers to every embeddings request, with either value of `EMBEDDINGS_RESPONSE_FORMAT`. It skips blank slots, so you can leave gaps. It refuses to start if a slot has no colon, holds an invalid name or value, repeats a header name from another slot, or names `Authorization`, `Content-Type`, or `Accept`, which the registry sets itself. The headers do not go to the IdP token endpoint.

**Warning:** the registry stores these values in clear text and shows them unmasked on the System Config page and in config exports. Never put a credential in a slot. Use `EMBEDDINGS_API_KEY` or `EMBEDDINGS_AUTH_MODE=idp` for credentials.

With `EMBEDDINGS_PROVIDER=sentence-transformers` the registry makes no HTTP call, so it ignores the slots and prints a warning banner at startup.

## Configuration parameters

| Variable | Required | Description |
| --- | --- | --- |
| `EMBEDDINGS_PROVIDER` | Yes | Set to `litellm`. |
| `EMBEDDINGS_MODEL_NAME` | Yes | e.g. `openai/<your-model>`. |
| `EMBEDDINGS_MODEL_DIMENSIONS` | Yes | Vector dimension your endpoint returns (must match your index). |
| `EMBEDDINGS_API_BASE` | Yes | Base URL of the OpenAI-compatible endpoint (e.g. `https://embeddings.example.com/v1`). |
| `EMBEDDINGS_AUTH_MODE` | Yes | `idp` to enable IdP auth (`static` is the default). |
| `EMBEDDINGS_IDP_TOKEN_ENDPOINT` | If `idp` | OAuth2 token endpoint. Must be `https://` (see note on `ALLOW_INSECURE`). |
| `EMBEDDINGS_IDP_CLIENT_ID` | If `idp` | Client-credentials client id. |
| `EMBEDDINGS_IDP_CLIENT_SECRET` | If `idp` | Client secret. Never logged; store as a secret (see below). |
| `EMBEDDINGS_IDP_SCOPE` | Depends | OAuth2 scope. Required for Entra (`api://<app-id>/.default`); usually omit for Keycloak. |
| `EMBEDDINGS_IDP_TIMEOUT_SECONDS` | No | Token request timeout (default `30`). |
| `EMBEDDINGS_IDP_ALLOW_INSECURE` | No | Local dev only: permit an `http://` **loopback** token endpoint. Default `false` (https required). A remote `http://` endpoint is always rejected. |
| `EMBEDDINGS_RESPONSE_FORMAT` | No | Endpoint response shape: `openai` (default, standard envelope) or `raw_array` (endpoint returns a bare `[[float]]` array). Independent of auth mode. |
| `EMBEDDINGS_EXTRA_HEADER_1` to `_10` | No | One extra request header per slot, as `Name: value`. Clear text, never a credential. See [Extra request headers](#extra-request-headers). |

The full cross-surface reference (Docker / Terraform / Helm) is in
[docs/unified-parameter-reference.md](../unified-parameter-reference.md).

## Example: Keycloak

Create a confidential client with **Service accounts (client credentials) enabled**
in your realm, then set:

```bash
EMBEDDINGS_PROVIDER=litellm
EMBEDDINGS_MODEL_NAME=openai/my-embedding-model
EMBEDDINGS_MODEL_DIMENSIONS=384
EMBEDDINGS_API_BASE=https://embeddings.example.com/v1
EMBEDDINGS_AUTH_MODE=idp
EMBEDDINGS_IDP_TOKEN_ENDPOINT=https://<keycloak-host>/realms/<realm>/protocol/openid-connect/token
EMBEDDINGS_IDP_CLIENT_ID=embeddings-client
EMBEDDINGS_IDP_CLIENT_SECRET=<client-secret>
# EMBEDDINGS_IDP_SCOPE is usually not needed for Keycloak client-credentials
```

## Example: Microsoft Entra ID

Register a confidential app (Certificates & secrets -> New client secret; Expose an
API -> Application ID URI `api://<app-id>`), then set:

```bash
EMBEDDINGS_PROVIDER=litellm
EMBEDDINGS_MODEL_NAME=openai/my-embedding-model
EMBEDDINGS_MODEL_DIMENSIONS=384
EMBEDDINGS_API_BASE=https://embeddings.example.com/v1
EMBEDDINGS_AUTH_MODE=idp
EMBEDDINGS_IDP_TOKEN_ENDPOINT=https://login.microsoftonline.com/<tenant-id>/oauth2/v2.0/token
EMBEDDINGS_IDP_CLIENT_ID=<application-client-id>
EMBEDDINGS_IDP_CLIENT_SECRET=<client-secret>
EMBEDDINGS_IDP_SCOPE=api://<application-client-id>/.default   # required for Entra v2 client-credentials
```

## Where the client secret goes per deployment

- **Docker Compose**: set `EMBEDDINGS_IDP_CLIENT_SECRET` in `.env` (or an `extra_env` file).
- **Helm**: reference a pre-created Secret via
  `registry.embeddings.idpClientSecretExistingSecret` (and `...Key`); the non-secret
  fields are `registry.embeddings.authMode` / `idpTokenEndpoint` / `idpClientId` /
  `idpScope` / `idpTimeoutSeconds` / `idpAllowInsecure`.
- **Terraform/ECS**: `embeddings_idp_client_secret` is stored in AWS Secrets Manager
  and injected via `valueFrom`; the rest are plain ECS env vars.

## How the token is handled

- The token is **cached and reused** until it is within 60s of expiry, then refreshed;
  the registry does not fetch a new token on every embedding call.
- Concurrent embedding calls share a single refresh (thread-safe).
- The client secret is never logged; token-request failures are surfaced as an
  actionable error and counted in the `mcpgw_registry_embeddings_idp_token_refresh_total`
  metric (`result="success"|"failure"`).

## Related

- [Why are some of my assets not showing up in semantic search?](fix-missing-search-embeddings.md)
- [How do I switch embedding models?](switching-embedding-models.md)
