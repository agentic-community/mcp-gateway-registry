# Upstream OAuth Provider Registry (Admin-Managed)
*Status: Accepted*  
*Date: 2025-12-21*

## Context
Gateway-terminated upstream OAuth requires a configured OAuth client (authorization endpoint, token endpoint, client_id, client_secret, scopes, and optional authorize params).

The existing implementation supports provider configuration via environment variables (`ENFORCEAI_UPSTREAM_OAUTH_PROVIDERS`) and secret references (`env`/`file`). This is sufficient for local development but does not satisfy a UI-driven operational model where:
- admins can configure upstream OAuth providers without deploying env changes, and
- client secrets are never exposed to browsers.

## Decision
Implement an EnforceAI admin-managed upstream OAuth provider registry:
- Provider configs are stored in the EnforceAI DB.
- Client secrets are encrypted-at-rest using the existing upstream KEK (`ENFORCEAI_UPSTREAM_KEK_PATH`).
- Client secrets are write-only from an API perspective:
  - create/update endpoints accept a client secret (or secret ref),
  - read/list endpoints never return client secrets (only metadata + `secret_present`).
- Only EnforceAI admins can manage providers, with CSRF enforcement and audit events on all mutations.

Backwards compatibility:
- Keep `ENFORCEAI_UPSTREAM_OAUTH_PROVIDERS` as a dev/bootstrap fallback until the provider registry is fully adopted.
- Runtime provider resolution should prefer the DB registry, with explicit error semantics when a server references an unknown provider id.

## Consequences
Pros:
- Enables UI-driven configuration for upstream OAuth providers.
- Avoids per-server/per-user env var management.
- Keeps secrets out of the browser and out of API read responses.

Cons:
- Adds a new sensitive data class in the EnforceAI DB (must be encrypted and carefully audited).
- Requires additional admin APIs and UI.

## Security Notes
- Provider client secrets must never be logged or returned by APIs.
- Nginx/debug logging must not emit request/response headers that can contain upstream bearer tokens.
- Deletion should consider reference safety (prevent deleting a provider that is referenced by registered servers, or require explicit force semantics).

## Test Requirements
- Unit: provider model validation, encryption-at-rest invariants, write-only secret behavior.
- Integration: admin CRUD endpoints (auth + CSRF), and OAuth start/callback/refresh using DB-backed provider config (offline stub provider).
