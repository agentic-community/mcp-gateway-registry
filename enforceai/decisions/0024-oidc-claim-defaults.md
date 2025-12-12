# Decision 0024 — OIDC Claim Mapping Defaults
*Status: Accepted*  
*Date: 2025-12-12*

## Context
Generic OIDC must work across IdPs with different claim conventions for scopes and roles/groups.
EnforceAI does not allow IdP roles/groups to grant agent permissions, but capturing these claims is useful for audit and metadata.

## Decision
- Provide conservative defaults with per-issuer overrides:
  - Scopes claim precedence (unless overridden): `scp` → `scope` → `permissions`
    - If `scope` is a space-delimited string, split on spaces.
  - Roles/groups claim precedence (unless overridden): `roles` → `groups` → `permissions`
- IdP-derived roles/groups are captured for audit/metadata only and must not grant agent permissions.

## Consequences
- Reduces configuration burden for common IdPs.
- Requires clear documentation of precedence and override behavior.
