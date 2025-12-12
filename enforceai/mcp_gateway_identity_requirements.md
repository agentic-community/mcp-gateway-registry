# EnforceAI Identity Layer Extension Requirements for MCP Gateway & Registry

## 0. Architectural Principle
All authentication methods (OIDC JWT, API key, gateway token) must produce a **normalized IdentityContext** consumed by the gateway’s FGAC engine.

---

## 1. Identity Resolver (NEW CORE COMPONENT)

### Purpose
Central point that determines which authentication method is used for the incoming request and constructs an `IdentityContext`.

### Requirements
- Inspect incoming headers:
  - `Authorization: Bearer <token>`
  - `X-Authorization`
  - `X-API-Key`
  - `X-Gateway-Token`
  - `X-Agent-Id` (required for OIDC-authenticated MCP access)
- Select the appropriate provider:
  - OIDC provider
  - API key provider
  - Gateway token provider
  - Anonymous provider (optional)
- Return IdentityContext:
  ```json
  {
    "principal_id": "string",
    "provider": "oidc | gateway-token | api-key | anonymous",
    "scopes": ["..."],
    "roles": ["..."],
    "metadata": {}
  }
  ```
- Handle malformed/expired credentials.

### Credential Precedence (Decision)
- Canonical external client interface is `Authorization: Bearer <token>` for both OIDC JWTs and gateway-issued tokens.
- `X-Gateway-Token` is accepted as a fallback for constrained clients.
- Requests that present multiple credential headers (e.g., both `Authorization` and `X-Gateway-Token`, or `X-API-Key` plus a token) must be rejected to avoid ambiguity.

### Agent Binding (Decision)
- All MCP access is agent-scoped; `agent_id` is required on the request path.
- `gateway-token` must embed `agent_id`.
- `api-key` records are agent-bound and resolve to `{principal/user_id, agent_id}`.
- For `oidc`, clients must send `X-Agent-Id`, which is validated against the gateway-managed agent registry for that user.

---

## 2. Generic OIDC Provider (MODIFIED / REPLACED COMPONENT)

### Purpose
Replace Cognito/Keycloak-specific validators with a generic OIDC JWT validator.

### Requirements
- Support any OIDC issuer via config, including multi-issuer deployments:
  - Phase 1 uses an issuer map (may contain a single issuer).
  - Each configured issuer must specify:
    - issuer (`iss`)
    - `jwks_uri`
    - `audience`
    - claim mapping for scopes/roles (optional per issuer)
- Validate:
  - `iss`, `aud`, `exp`, `iat`, signature via JWKS
- Extract:
  - `sub` as `principal_id`
  - roles/scopes via configurable claims

### OIDC Configuration Shape (Decision)
Use an issuer map even for single-issuer deployments:

```
OIDC_ISSUERS='{
  "https://example.okta.com/oauth2/default": {
    "jwks_uri": "https://example.okta.com/oauth2/default/v1/keys",
    "audience": ["mcp-gateway"],
    "role_claims": ["groups", "roles"],
    "scope_claims": ["scp", "scope", "permissions"]
  }
}'
```

---

## 3. API Key Provider (NEW COMPONENT)

### Purpose
Support identity without any IdP (API-key-only mode).

### Requirements
- Read key from `X-API-Key`
- Lookup entry in:
  ```text
  api_key → { principal_id, scopes, roles, expires_at?, revoked? }
  ```
- Return IdentityContext
- Support key creation, revocation, rotation.

### API Key Format and Storage (Decision)
- Header format: `X-API-Key: eak_<key_id>.<secret>`
- Store only a verifier hash (no plaintext keys), using a pepper:
  - `secret_hash = HMAC-SHA256(API_KEY_PEPPER, secret)`
- API keys are agent-bound and resolve to `{user_id/principal_id, agent_id}`.
- Effective scopes must be non-elevating:
  - `effective_scopes = api_key.scopes ∩ agent.scopes` (or `agent.scopes` if key scopes are unset).

---

## 4. Gateway Token Provider (NEW COMPONENT)

### Purpose
Issue long-lived tokens for non-OAuth clients (Claude, Cursor, VSCode).

### Requirements
- Issue gateway-signed JWTs containing:
  ```json
  { "principal_id": "...", "scopes": [...], "roles": [...], "provider": "gateway-token" }
  ```
- Validate signature, expiration, revocation
- Maintain a token revocation table
- Prefer asymmetric signing for compatibility (`RS256`) and to avoid sharing signing capability with verifiers.
- Gateway tokens are long-lived (PAT-style) in Phase 1 and must include `exp` (no non-expiring tokens).

### Revocation Semantics (Decision)
- Layered revocation:
  - Agent kill switch (`agent.revoked`) denies all access for that agent.
  - Token-level revocation denies if `jti` is present in the revocation table.
  - Bulk token revocation per agent via `agent.tokens_valid_after` (deny if `token.iat < tokens_valid_after`).

---

## 5. Unified IdentityContext Model (NEW INTERNAL CONTRACT)

### Purpose
Normalize identity across OIDC, API keys, and gateway tokens.

### Requirements
- Fields:
  ```json
  {
    "principal_id": "...",
    "provider": "...",
    "scopes": ["..."],
    "roles": ["..."],
    "metadata": {}
  }
  ```
- Feed directly into FGAC

---

## 6. Scope / Role Mapping Engine (NEW COMPONENT)

### Purpose
Normalize scopes/roles across different identity providers.

### Requirements
- Configurable claim resolution:
  ```
  ROLE_CLAIMS = roles,groups,permissions
  SCOPE_CLAIMS = scopes,permissions
  ```
- Provider-specific overrides:
  - google → `permissions`
  - okta → `groups`
  - auth0 → `permissions`
- Merge scopes from:
  - JWT claims
  - API key metadata
  - Gateway token payload

### Enterprise Policy Catalog (Decision)
- Phase 1 uses `auth_server/scopes.yml` as the authoritative scope catalog (scope definitions and what they allow).
- IdP group/role mappings are legacy/optional and must not override gateway-managed agent scopes.

---

## 7. Config System Extensions (MODIFIED COMPONENT)

### Requirements
Add:

```
AUTH_PROVIDER = oidc | api-key | gateway-token | mixed

# OIDC config
OIDC_ISSUERS='{"https://issuer": {"jwks_uri": "...", "audience": ["mcp-gateway"]}}'

# Gateway token config
GATEWAY_TOKEN_ALG=RS256
GATEWAY_PRIVATE_KEY_PATH=/run/secrets/gateway_private_key.pem
GATEWAY_PUBLIC_KEYS_DIR=/run/secrets/gateway_public_keys
GATEWAY_ACTIVE_KID=...

# API Key config
API_KEY_PEPPER_PATH=/run/secrets/api_key_pepper
ENFORCEAI_DB_PATH=/path/to/sqlite.db

# Identity mapping
ROLE_CLAIMS=roles,groups,custom_roles
SCOPE_CLAIMS=scopes,permissions
```

### Config Delivery (Decision)
- Phase 1 is environment-variable driven.
- Secrets are provided via mounted secret files and referenced via `*_PATH`/directory variables.

### Audit Retention (Decision)
- Audit retention thresholds are configurable (not hard-coded), for example:
  - `ENFORCEAI_AUDIT_RETENTION_DAYS=400`
  - `ENFORCEAI_AUDIT_MAX_DB_BYTES=10737418240`  # 10 GiB

---

## 8. Authentication Middleware Rewrite (MODIFIED COMPONENT)

### Purpose
Introduce identity-agnostic authentication pipeline.

### Required Flow
1. Extract raw credentials  
2. Run Identity Resolver  
3. Produce IdentityContext  
4. Apply FGAC  
5. Route MCP request

---

## 9. Backward Compatibility Layer (OPTIONAL)

### Requirements
- Not required for EnforceAI. Backward compatibility with legacy Cognito/Keycloak-specific modes is out of scope.

---

## Summary Table

| Requirement | Current | Needed |
|------------|---------|--------|
| Any OIDC provider | ❌ | Generic OIDC Validator |
| API-key identity | ❌ | API Key Provider |
| Gateway-issued tokens | ❌ | Token Issuer |
| Unified identity model | ❌ | IdentityContext |
| Flexible role mapping | ❌ | Scope Mapper |
| Identity-agnostic middleware | ❌ | Auth Rewrite |
| Mixed-mode operation | ❌ | Config Enhancements |

---

## Result
Implementing these components transforms MCP Gateway & Registry into a **universal, IdP-agnostic identity gateway** suitable for EnforceAI’s requirements.
