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

## Revocation and Storage
- Phase 1 uses the same local SQLite database as the agent registry to store gateway-issued token metadata and revocation state.
- Enforcement must treat revocation as authoritative and default-deny if revocation data is unavailable.
- Storage layer must remain portable to Postgres (see `enforceai/architecture/agent_registry.md`).

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
