# Gateway Tokens — EnforceAI Gateway

## Token Format
```
{
  "iss": "<gateway>",
  "sub": "<user_id>",
  "agent_id": "<agent_id>",
  "scopes": [...],
  "iat": ts,
  "exp": ts
}
```

## Rules
- Only gateway issues tokens
- Tokens MUST embed user_id + agent_id + scopes
- No agent may have implicit permissions
- Signature required (RS256 preferred for compatibility; ES256/HS256 allowed only if explicitly configured)

## Lifetime Policy (Decision)
- Phase 1 gateway tokens are long-lived (PAT-style) and must include `exp`.
- Target max lifetime: up to 365 days, with shorter lifetimes recommended for higher-risk agents.
- Revocation controls (agent kill switch, bulk revoke, `jti` revocation) are mandatory and are the primary mechanism for rapid response.

## Revocation and Storage
- Phase 1 uses the same local SQLite database as the agent registry to store gateway-issued token metadata and revocation state.
- Enforcement must treat revocation as authoritative and default-deny if revocation data is unavailable.
- Storage layer must remain portable to Postgres (see `enforceai/architecture/agent_registry.md`).

## Revocation Semantics (Decision)
- Revocation is layered and agent-scoped:
  - Agent kill switch: if `agent.revoked` is true, deny all access for that agent (applies to gateway tokens and API keys).
  - Token-level revocation: deny if JWT `jti` is present in the token revocation table.
  - Bulk token revocation per agent: maintain `agent.tokens_valid_after` and deny if `token.iat < tokens_valid_after`.
- If revocation/agent registry state cannot be read on the enforcement path, default-deny.

## Algorithm Decision (Phase 1)
- Use `RS256` for gateway tokens to maximize compatibility with standard JWT tooling and JWKS-based key distribution patterns.
- Streamline validation by keeping verification local to the enforcement point:
  - Verify using a locally loaded public key (no network call on the request path).
  - Cache parsed key material in memory and select keys by `kid` to support rotation.

## Effective Scopes (Decision)
- For `gateway-token` requests, effective scopes are:
  - `effective_scopes = token.scopes ∩ agent.scopes`
- Token scopes must never elevate permissions beyond the current agent registry scopes.
- Scope mismatches (token contains scopes not currently on the agent) should be logged and treated as a reduction, not an error, to support immediate scope tightening without forced token rotation.

## Key Management and Rotation (Decision)
- Canonical key material is provided via mounted secret files (not environment variables):
  - Private key: a single PEM file used for signing.
  - Public keys: one PEM file per `kid` (key id) to support rotation.
- Validation must remain local and fast:
  - Public keys are loaded and parsed at startup and cached in memory.
  - Key selection is performed by JWT header `kid`.
- Phase 1 rotation is restart-based:
  - Add new keypair, switch active `kid`, restart the enforcement point.
  - Keep old public keys available for verification until all previously issued tokens have expired.
- Future (optional): publish a JWKS endpoint for public keys to support additional verifiers; never fetch JWKS on the request path.
