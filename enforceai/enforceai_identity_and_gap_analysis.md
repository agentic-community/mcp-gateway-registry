# EnforceAI MCP Gateway — Detailed Requirements & Gap Analysis

## DOCUMENT 1 — Detailed Requirements Specification

### 1. Overview
The EnforceAI MCP Gateway requires a new identity model that supports:
- Authentication via **any OIDC IdP**
- **Agent‑level authorization**, independent of IdP
- **Gateway‑issued agent tokens**
- **API‑key‑only mode** (no IdP)
- Mixed authentication for different agent types

The gateway must combine:
- User identity → from IdP  
- Agent identity → managed internally  
- Agent scopes → enforced by gateway  
- Unified identity model → consumed by FGAC  

---

## 2. Identity Model Requirements

### 2.1 IdentityContext Object
```
IdentityContext {
    user_id: string
    agent_id: string
    provider: "oidc" | "api-key" | "gateway-token"
    scopes: string[]
    user_roles: string[]
    metadata: object
}
```

### 2.2 Identity Component Sources
| Component | Source |
|----------|--------|vg
| user_id | IdP JWT or API-key record |
| agent_id | Gateway registry or header |
| scopes | Gateway-managed agent config |
| roles | Optional, from IdP |

---

## 3. Authentication Requirements

### 3.1 Supported Modes
#### A. OIDC JWT Mode
- Accept tokens from any IdP: Okta, Google, Auth0, Entra, Keycloak, Cognito.
- Use generic OIDC config with multi-issuer support (issuer map): issuer, JWKS, audience, claims.

#### B. Gateway Token Mode
- Gateway issues tokens embedding: user_id, agent_id, agent scopes.

#### C. API Key Mode
- Identity via `X-API-Key`.

#### D. Mixed Mode
Gateway picks the right provider based on the request.

### 3.2 Compatibility Scope (Decision)
- EnforceAI does not require backward compatibility with upstream provider-specific modes or legacy token types.
- Only the EnforceAI-defined authentication modes are supported: OIDC, gateway tokens, API keys (mixed mode via resolver).

---

## 4. Agent Identity Requirements

### 4.1 Agent Registry
Stores:
```
agents[user_id][agent_id] = {
    scopes: [...],
    metadata: {...},
    created_at,
    revoked: bool
}
```

### 4.3 Agent Identity Source (Decision)
- `agent_id` is required for MCP access and must come from the gateway (not the IdP).
- For `gateway-token`, `agent_id` is embedded in the token.
- For `api-key`, API keys are agent-bound and resolve to `{user_id, agent_id}`.
- For `oidc`, callers must provide `X-Agent-Id`, and the enforcement point validates it is owned by the authenticated `user_id`.

### 4.2 Required Metadata
- Agent name  
- Agent type  
- Agent scopes  
- Allowed tools  
- Risk level  
- Token issue history  

---

## 5. Token Requirements

### 5.1 Gateway Tokens
Payload:
```
{
 "iss": "<gateway>",
 "sub": "<user_id>",
 "agent_id": "<agent_id>",
 "scopes": [...],
 "iat": <ts>,
 "exp": <ts>
}
```
- Signature algorithm: `RS256` (selected for compatibility).

### 5.2 IdP JWT Handling
- Validate issuer, audience, signature, expiration.
- Do *not* derive agent permissions from IdP roles.

---

## 6. Authorization Requirements

### 6.1 Agent-Level FGAC
FGAC must evaluate:
- user identity  
- agent identity  
- agent-scoped permissions  

### 6.2 Policy Overlay
```
final_permissions =
  enterprise_policy ∩ user_baseline_scopes ∩ agent_scopes
```

#### Overlay Semantics (Decision)
- Phase 1 runtime enforcement is agent-scoped: effective permissions are computed from agent scopes (and optional allowed-tools) against the enterprise scope catalog.
- Any user baseline constraints are applied at agent provisioning time (agent scopes cannot exceed the gateway-defined baseline), not on the request path.

---

## 7. Audit Requirements
Each request log must include:
- user_id  
- agent_id  
- scopes used  
- tool/action  
- allow/deny result  
- reason  

---

## 8. Operational Requirements
- UI/CLI for generating agent tokens  
- UI/CLI for agent lifecycle management  
- Multi-tenant-safe  
- Stateless scaling  

---

# DOCUMENT 2 — Gap Analysis vs Existing MCP Gateway Registry

## 1. High-Level Gap Summary
| Feature | Today | Gap |
|---------|--------|------|
| Any OIDC IdP | ❌ Cognito/Keycloak only | Add generic OIDC |
| Agent identity | ❌ None | Add agent registry |
| Agent tokens | ❌ None | Add gateway-issued tokens |
| Agent scopes | ❌ None | Add scope overlay |
| Agent-level FGAC | ❌ No | Extend FGAC engine |
| API-key-only mode | ❌ No | Add API-key provider |
| Unified IdentityContext | ❌ No abstraction | Implement identity layer |
| Multi-agent per user | ❌ Not possible | Add agent model |
| Authorization independent of IdP | ❌ No | Move scopes to gateway |
| Agent-aware audit | ❌ No | Add agent_id to logs |
| Mixed auth | ❌ No | Add resolver pipeline |

---

## 2. Detailed Gaps and Fixes

### 2.1 Identity & Authentication Gaps
**Gap:** IdP support limited to Cognito/Keycloak  
**Fix:** Replace with generic OIDC validator.

**Gap:** No API-key auth  
**Fix:** API-key provider + storage.

**Gap:** Cannot interpret multiple token types  
**Fix:** Identity Resolver.

---

### 2.2 Agent Identity Gaps
**Gap:** No agent concept  
**Fix:** Build agent registry.

**Gap:** No differentiation between agents  
**Fix:** Add agent_id header or embed in gateway token.

---

### 2.3 Token Gaps
**Gap:** Gateway cannot issue tokens  
**Fix:** Add gateway token issuer & signing keys.

**Gap:** No long-lived agent tokens  
**Fix:** Add token generation UI/CLI.

---

### 2.4 Authorization Gaps
**Gap:** Authorization tied to IdP scopes  
**Fix:** Use gateway-managed agent scopes.

**Gap:** No agent-level FGAC  
**Fix:** Extend FGAC to evaluate agent identity.

---

### 2.5 Audit Gaps
**Gap:** Logs do not include agent identity  
**Fix:** Include `agent_id`.

---

### 2.6 Operational Gaps
**Gap:** No agent lifecycle flows  
**Fix:** Add CRUD for agents and tokens.

---

# Final Summary
MCP Gateway Registry is a strong foundation but lacks:
- agent identity  
- gateway-issued tokens  
- generic OIDC  
- agent-level FGAC  
- agent-level audit  
- API-key mode  
- unified identity abstraction  

Implementing these components creates a **universal, IdP-agnostic, multi-agent-capable gateway** suitable for EnforceAI.
