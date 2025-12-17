@# MCP Authentication Models
**Gateway-Terminated Authentication: Implementation Guide**

## Scope and Assumptions

This document describes **all common authentication methods used with MCP servers today**, and what **upstream MCP servers must support** in order for an **MCP Gateway** to securely authenticate users on their behalf.

### Assumed Architecture

```
Agent / Client
    |
    |  (authenticates only to Gateway)
    v
MCP Gateway
    |
    |  (trusted upstream connection)
    v
Upstream MCP Servers
```

* The agent never authenticates directly to upstream MCP servers
* The gateway terminates all authentication flows
* The gateway may present a UI to complete auth flows
* Upstream MCP servers trust the gateway
* Gateway may validate tokens, exchange tokens, mint derived credentials, and forward identity context upstream

All MCP transports are in scope:
- HTTP / HTTPS
- WebSocket
- stdio / pipes

---

## Terminology

| Term | Meaning |
|---|---|
| Gateway | MCP-aware proxy that terminates authentication |
| Upstream MCP Server | MCP server behind the gateway |
| Agent | LLM / client speaking MCP |
| Principal | Authenticated user or service |
| Identity Context | User/service identity forwarded upstream |

---

## Authentication Categories

1. No Authentication
2. API Keys
3. OAuth 2.x (Authorization)
4. OAuth + OpenID Connect (OIDC)
5. Provider-Delegated OAuth (Slack, Google, etc.)
6. JWT Bearer Tokens
7. Mutual TLS (mTLS)
8. Gateway-Terminated Authentication (Header Trust Model)

---

## 1. No Authentication

### Description
No auth at any layer. Trust is implicit.

### Gateway Responsibilities
None.

### Upstream MCP Requirements
None.

### Recommendation
Acceptable only for local MCP servers.

---

## 2. API Key Authentication

### Description
Static shared secret identifying a client or tenant.

### Gateway Responsibilities
- Store API keys securely
- Validate key
- Map key to principal

### Upstream MCP Requirements
- Trust gateway
- Accept forwarded identity

Example headers:
```
X-MCP-Principal: service:analytics
X-MCP-Auth-Type: api-key
```

### Recommendation
Service-level access only.

---

## 3. OAuth 2.x (Authorization)

### Description
Bearer tokens for authorization.

### Gateway Responsibilities
- Complete OAuth flows
- Validate tokens
- Enforce scopes
- Map tokens to principals

### Upstream MCP Requirements

**Gateway-validated mode (recommended):**
```
X-MCP-Principal: user:123
X-MCP-Scopes: files.read
X-MCP-Auth-Type: oauth
```

### Recommendation
Gateway-validated OAuth is best practice.

---

## 4. OAuth + OpenID Connect (OIDC)

### Description
Adds explicit user identity.

### Gateway Responsibilities
- Complete OIDC flows
- Validate ID tokens
- Extract identity claims

### Upstream MCP Requirements
```
X-MCP-Principal: user:alice@example.com
X-MCP-Auth-Type: oidc
```

### Recommendation
Use when identity matters.

---

## 5. Provider-Delegated OAuth

### Description
OAuth for third-party APIs (GitHub, Google, Slack).

### Gateway Responsibilities
- Complete provider OAuth
- Store provider tokens
- Call provider APIs or forward tokens

### Upstream MCP Requirements
```
X-MCP-Provider: github
X-MCP-Auth-Type: provider-oauth
```

### Recommendation
Gateway should own provider OAuth.

---

## 6. JWT Bearer Authentication

### Description
Signed JWTs issued by internal or enterprise systems.

### Gateway Responsibilities
- Validate JWTs
- Map claims to principals

### Upstream MCP Requirements
```
X-MCP-Principal: user:42
X-MCP-Auth-Type: jwt
```

---

## 7. Mutual TLS (mTLS)

### Description
Client authenticated via certificate.

### Gateway Responsibilities
- Terminate TLS
- Validate certs
- Extract identity

### Upstream MCP Requirements
- Trust forwarded identity

---

## 8. Gateway-Terminated Authentication (Header Trust Model)

### Description
Upstream MCP servers never authenticate.

### Required Upstream Capabilities
- Trust gateway
- Accept identity via headers, env vars, or handshake metadata
- Enforce authorization based on forwarded identity

### Canonical Headers
| Header | Meaning |
|---|---|
| X-MCP-Principal | Canonical identity |
| X-MCP-Auth-Type | Auth mechanism |
| X-MCP-Scopes | Optional |
| X-MCP-Provider | Optional |
| X-MCP-Claims | Optional |

---

## Best Practice Recommendation

- Gateway-terminated OAuth 2.1 + OIDC
- Gateway owns all auth flows
- Upstream MCP servers trust gateway identity
- Provider OAuth handled at gateway
- API keys only for services

---

## Non-Goals
- Agent-side auth complexity
- MCP-mandated auth mechanisms
- Token format standardization beyond gateway

---

**End of document**
