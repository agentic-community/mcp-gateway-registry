# MCP Gateway & Registry: Architecture Diagrams

This document presents the system architecture at four levels of detail,
each targeting a different audience and depth of understanding.

---

## Level 100: Executive Overview

**What does this system do?**

The MCP Gateway & Registry is a central hub that connects AI tools
(coding assistants, autonomous agents) to the services and data they need,
with built-in security, discovery, and governance.

```
                        +-----------------------------------------+
                        |              CONSUMERS                    |
                        |                                          |
                        |  Human Users       AI Coding Assistants  |
                        |  (Browser UI)      (VS Code, Cursor,     |
                        |                     Claude Code, Kiro)   |
                        |                                          |
                        |  Autonomous Agents                       |
                        |  (Custom agents, A2A protocol)           |
                        +-------------------+---------------------+
                                            |
                                   Secure API Calls
                                            |
                                            v
            +---------------------------------------------------------------+
            |                                                               |
            |                  MCP GATEWAY & REGISTRY                       |
            |                                                               |
            |   +------------------+  +----------------+  +--------------+  |
            |   |   Gateway        |  |   Registry     |  |  Security    |  |
            |   |                  |  |                |  |              |  |
            |   | Routes requests  |  | Catalog of all |  | Auth, scans, |  |
            |   | to the right     |  | available      |  | access       |  |
            |   | backend tool     |  | tools & agents |  | control      |  |
            |   +------------------+  +----------------+  +--------------+  |
            |                                                               |
            |   +------------------+  +----------------+  +--------------+  |
            |   |   Discovery      |  |   Federation   |  |  Monitoring  |  |
            |   |                  |  |                |  |              |  |
            |   | Semantic search  |  | Connect to     |  | Health, logs |  |
            |   | (ask in plain    |  | other orgs'    |  | metrics,     |  |
            |   |  English)        |  | registries     |  | audit trail  |  |
            |   +------------------+  +----------------+  +--------------+  |
            |                                                               |
            +------------------------------+--------------------------------+
                                           |
                              Proxied & Authenticated
                                           |
                                           v
                        +-----------------------------------------+
                        |         BACKEND MCP SERVERS              |
                        |                                          |
                        |  GitHub   Slack   Jira   Databases       |
                        |  Docs     APIs    Custom internal tools  |
                        +-----------------------------------------+
```

**Key value propositions:**

- One secure front door for all AI tool access
- Find the right tool with natural language search
- Connect multiple registries across teams or organizations
- Enterprise identity (SSO) with fine-grained permissions
- Built-in security scanning of every registered server

---

## Level 200: Technical Architecture

**How do the services interact?**

```
                     CLIENTS
    +------------------------------------------+
    | Browser (UI)  | CLI Tools | AI Agents    |
    | OAuth2 PKCE   | JWT Token | M2M OAuth2   |
    +-------+-------+-----+-----+------+------+
            |             |            |
            +-------------+------------+
                          |
                   HTTPS (443)
                          |
                          v
+-------------------------------------------------------------------+
|                     INGRESS LAYER                                   |
|                                                                    |
|  +-------------------------------+   +-------------------------+   |
|  | CDN (optional)                |   | Load Balancer / Ingress |   |
|  | - CloudFront (ECS)            |   |                         |   |
|  | - CloudFront or external (K8s)|   | ECS:  ALB + ACM cert    |   |
|  | - Cache GET, DDoS protection  |   | K8s:  Ingress Controller|   |
|  +-------------------------------+   |       (ALB/Nginx/Istio) |   |
|                                      | Compose: host port bind |   |
|                                      |                         |   |
|                                      | All: TLS termination,   |   |
|                                      |      health routing     |   |
|                                      +-------------------------+   |
+------------------------------------+------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------+
|                     NGINX REVERSE PROXY (Port 80/443)               |
|                                                                    |
|  Step 1: Auth subrequest (blocks until response)                   |
|      |                                                             |
|      v                                                             |
|  +-----------------------+                                         |
|  | GET /validate         |------> AUTH SERVER (8888)               |
|  | (internal subrequest) |<------ Returns: allow/deny              |
|  |                       |        + X-User                         |
|  |                       |        + X-Groups                       |
|  |                       |        + X-Scopes                       |
|  +-----------+-----------+                                         |
|              |                         Supports:                    |
|              | (only if allowed)        Keycloak, Entra ID,         |
|              v                          Okta, Auth0, Cognito,       |
|  Step 2: Route & forward               GitHub, Google              |
|      |                                                             |
|      v                                                             |
|  +---------------------+    +------------------+                   |
|  | Static Routes       |    | Dynamic Routes   |                   |
|  | /api/*  -> Registry |    | /{server-path}/* |                   |
|  | /*      -> Frontend |    |   -> Backend MCP |                   |
|  +---------------------+    | /virtual/{path}  |                   |
|                              |   -> Lua router  |                   |
|              |               +--------+---------+                   |
|              |                        |                             |
|              | proxy_pass             | proxy_pass                  |
|              | (+ injected headers)   | (+ injected headers)       |
+-------------------------------------------------------------------+
               |                        |
               v                        v
+----------------------+     +----------------------+
| REGISTRY API (7860)  |     | BACKEND MCP SERVERS  |
|                      |     | (registered targets) |
| - Server CRUD        |     +----------------------+
| - Agent CRUD         |
| - Skill CRUD         |
| - Virtual servers    |
| - Semantic search    |
| - Federation sync    |
| - Security scanning  |
| - Audit logging      |
| - Health checks      |
| - Webhooks/gates     |
+----------+-----------+
           |
           v
+-------------------------------------------------------------------+
|                     DATA LAYER                                      |
|                                                                    |
|  +----------------------------+    +---------------------------+   |
|  | MongoDB CE / DocumentDB / |    | Embeddings Engine         |   |
|  | MongoDB Atlas (BYOA*)     |    |                           |   |
|  |                            |    |                           |   |
|  | Collections:               |    | sentence-transformers    |   |
|  |  - mcp_servers_*          |    |    (local, free)          |   |
|  |  - mcp_agents_*           |    | OpenAI text-embedding    |   |
|  |  - agent_skills_*         |    | Amazon Bedrock Titan     |   |
|  |  - mcp_scopes_*           |    | Any LiteLLM provider     |   |
|  |  - mcp_embeddings_{dim}   |    |                           |   |
|  |  - mcp_security_scans_*   |    | HNSW vector index        |   |
|  |  - audit_events_*         |    |   (DocumentDB native)    |   |
|  |  - mcp_federation_config  |    +---------------------------+   |
|  |  - backend_sessions       |                                    |
|  +----------------------------+    +---------------------------+   |
|                                    | Metrics (SQLite + OTLP)   |   |
|                                    | Prometheus export (9465)  |   |
|                                    +---------------------------+   |
+-------------------------------------------------------------------+

                     FEDERATION (Peer-to-Peer)

    +----------------+          +----------------+
    | Registry A     |  <---->  | Registry B     |
    | (Team Alpha)   |  sync    | (Team Beta)    |
    +-------+--------+          +-------+--------+
            |                           |
            +---------------------------+
                        |
                        v
                +----------------+
                | Registry C     |
                | (Central Hub)  |
                +----------------+

    Sync modes: All | Allow-list | Tag-filter
    Auth: Static token | OAuth2 JWT
```

**Deployment options:**

| Surface      | Orchestration       | Storage                        | Identity        |
|-------------|--------------------|---------------------------------|----------------|
| Docker Compose | Single host       | MongoDB CE                     | Keycloak (local) |
| Terraform/ECS | AWS Fargate       | AWS DocumentDB                 | Entra/Okta/Cognito |
| Helm/EKS     | Kubernetes         | DocumentDB / MongoDB Atlas*    | Any OIDC provider |

*MongoDB Atlas is bring-your-own-account: customer provides their own Atlas cluster.

---

## Level 300: Internals & Data Flow

**Request lifecycle, concurrency model, and data paths.**

```
REQUEST LIFECYCLE: Tool Call via Gateway
========================================

1. CLIENT REQUEST
   POST /github/mcp  (JSON-RPC: tools/call "search_repos")
   Headers: Authorization: Bearer <jwt>

2. NGINX PROCESSING (sub-millisecond)
   +----------------------------------------------------------------+
   | a) auth_request -> GET /validate                                |
   |    Auth Server validates JWT signature (HS256 self-signed       |
   |    or RS256 from IdP), checks expiry, extracts claims          |
   |    Returns: X-User, X-Groups (JSON array), X-Scopes            |
   |                                                                 |
   | b) Scope enforcement (Lua):                                     |
   |    - Load user scopes from X-Scopes header                     |
   |    - Check server path + method against scope rules            |
   |    - If tools/call: validate tool name against allowed tools   |
   |    - Reject with 403 if not permitted                          |
   |                                                                 |
   | c) Route resolution:                                            |
   |    - Direct server: proxy_pass to registered backend URL       |
   |    - Virtual server: Lua content handler dispatches to         |
   |      multiple backends, aggregates responses                   |
   +----------------------------------------------------------------+

3. MCP SERVER PROXYING
   +----------------------------------------------------------------+
   | - Inject server-specific auth (Bearer token, API key, custom   |
   |   headers from Fernet-encrypted store)                         |
   | - Protocol: HTTP/1.1 with Connection: upgrade for SSE          |
   | - Timeout: configurable per-server (default 30s)               |
   | - Response streamed back to client (chunked transfer)          |
   +----------------------------------------------------------------+

4. METRICS EMISSION (async, non-blocking)
   +----------------------------------------------------------------+
   | - Lua shared_dict buffer accumulates counters                   |
   | - Periodic flush to metrics service (every 10s)                |
   | - OpenTelemetry spans if OTLP configured                       |
   +----------------------------------------------------------------+


VIRTUAL SERVER DISPATCH (Lua)
==============================

    Client: tools/list on /virtual/dev-toolkit
                    |
                    v
    +---------------------------------+
    | Load /etc/nginx/lua/            |
    |   virtual_mappings/{id}.json    |
    |                                 |
    | Mapping:                        |
    |   github.search_repos -> /github|
    |   slack.post_msg -> /slack      |
    |   jira.create -> /jira          |
    +---------------------------------+
                    |
        +-----------+-----------+
        |           |           |
        v           v           v
    /github     /slack      /jira
    tools/list  tools/list  tools/list
        |           |           |
        v           v           v
    [3 tools]   [5 tools]   [4 tools]
        |           |           |
        +-----+-----+-----+----+
              |
              v
    Combined response: 12 tools
    (aliases applied, conflicts resolved)
    Cached for 60 seconds


SEMANTIC SEARCH PIPELINE
=========================

    Query: "tools for managing pull requests"
                    |
                    v
    +-----------------------------------+
    | 1. EMBED QUERY                    |
    |    Provider: sentence-transformers|
    |    Model: all-MiniLM-L6-v2       |
    |    Output: float[384] vector      |
    +-----------------------------------+
                    |
                    v
    +-----------------------------------+
    | 2. VECTOR SEARCH                  |
    |                                   |
    |    DocumentDB (prod):             |
    |      db.mcp_embeddings.aggregate( |
    |        $search: {                 |
    |          vectorSearch: {          |
    |            vector: query_vec,     |
    |            path: "embedding",     |
    |            similarity: "cosine",  |
    |            k: max(n*3, 50),       |
    |            efSearch: 100          |
    |        }})                        |
    |      Latency: < 50ms             |
    |                                   |
    |    MongoDB Atlas (prod, BYOA):    |
    |      Customer-provided cluster    |
    |      Native Atlas vector search   |
    |      Latency: < 50ms             |
    |                                   |
    |    MongoDB CE (dev):              |
    |      Full scan + Python cosine   |
    |      Latency: ~200ms (1000 docs) |
    +-----------------------------------+
                    |
                    v
    +-----------------------------------+
    | 3. KEYWORD BOOST                  |
    |    Token match on:                |
    |      path: +5.0 boost            |
    |      name: +3.0 boost            |
    |      description: +2.0 boost     |
    |      tags: +1.5 boost            |
    |      metadata/tools: +1.0 boost  |
    +-----------------------------------+
                    |
                    v
    +-----------------------------------+
    | 4. RANK & RETURN                  |
    |    score = normalize(cosine) +    |
    |            keyword_boost          |
    |    Clamp to [0.0, 1.0]           |
    |    Sort descending                |
    |    Return top max_results         |
    +-----------------------------------+


FEDERATION SYNC LIFECYCLE
==========================

    Peer A (source)                     Peer B (consumer)
    +--------------+                    +--------------+
    | Servers:     |                    | Local:       |
    |  /github     |   GET /api/       |  /internal   |
    |  /slack      |   servers?        |              |
    |  /internal   |   visibility=     | Federated:   |
    +--------------+   public          |  (none yet)  |
                   |                    +--------------+
                   |  <-- scheduled sync (cron) --|
                   |                              |
                   v                              v
    +-----------------------------------+   +-----------+
    | Auth: Static token in header      |   | Increment |
    | Filter: allow-list=[/github,/slack]|   | generation|
    | Response: server cards (JSON)     |   | counter   |
    +-----------------------------------+   +-----------+
                                                  |
                                                  v
                                        +-------------------+
                                        | Upsert federated  |
                                        | servers with:     |
                                        |   source_registry |
                                        |   generation: N   |
                                        |   read_only: true |
                                        +-------------------+
                                                  |
                                                  v
                                        +-------------------+
                                        | Orphan detection: |
                                        | DELETE WHERE      |
                                        |   source=peer_a   |
                                        |   generation < N  |
                                        +-------------------+


AUTHENTICATION FLOWS (Concurrent Session Model)
=================================================

    BROWSER LOGIN (OAuth2 Authorization Code + PKCE)
    -------------------------------------------------
    Browser -> Auth Server -> IdP (Keycloak/Entra/Okta)
         <- auth_code redirect <-
    Auth Server:
      1. Exchange code for tokens (IdP)
      2. Extract groups from ID token claims
      3. Create server-side session (MongoDB, encrypted)
      4. Set HttpOnly session cookie (session ID only, not payload)
      5. Session TTL: 24h (configurable)

    CLI TOKEN (Self-Signed JWT)
    ----------------------------
    Authenticated user -> POST /v0.1/auth/token
    Auth Server:
      1. Validate session cookie
      2. Sign JWT with SECRET_KEY (HS256)
      3. Include: sub, groups, scopes, exp (24h)
      4. Return token (no server-side state)

    M2M AGENT (Client Credentials)
    --------------------------------
    Agent -> POST /token (client_id + client_secret)
    Auth Server:
      1. Validate credentials against IdP or MongoDB
      2. Resolve groups via enrichment lookup
      3. Issue RS256 access token (default 8h, configurable)
      4. No refresh token (re-authenticate on expiry)

    STATIC API TOKEN (IdP-Independent)
    ------------------------------------
    Client -> Any endpoint (Authorization: Bearer <static-token>)
    Nginx/Auth Server:
      1. HMAC-SHA256 timing-safe compare
      2. Multi-key support: REGISTRY_API_KEYS (JSON)
         { "ci-bot": { "token": "...", "groups": [...] } }
      3. Resolve groups/scopes from key config
      4. No expiry (rotate manually)
```

---

## Level 400: Security Architecture (For Security and Compliance Teams)

**Trust boundaries, attack surface, and defense-in-depth layers.**

```
TRUST BOUNDARIES
=================

    UNTRUSTED                SEMI-TRUSTED              TRUSTED
    (Internet)              (Internal Network)        (Private Subnet)
    +-----------+           +------------------+      +------------------+
    | End users |           | Registered MCP   |      | DocumentDB       |
    | AI agents |           | Servers (backends)|      | Auth Server      |
    | Federated |           |                  |      | Metrics DB       |
    | peers     |           | (May be external |      | Encryption keys  |
    +-----------+           |  or internal)    |      +------------------+
         |                  +------------------+             |
         |                         |                         |
    =====|=========================|=========================|=====
         |     TRUST BOUNDARY 1    |    TRUST BOUNDARY 2     |
         |     (Perimeter)         |    (Service Mesh)       |
         v                         v                         v

    +------------------------------------------------------------------+
    |                    DEFENSE-IN-DEPTH LAYERS                         |
    |                                                                    |
    |  Layer 1: NETWORK                                                  |
    |  +--------------------------------------------------------------+ |
    |  | CloudFront + WAF (optional, via enable_waf flag)             | |
    |  | ALB Security Groups -> Only 443 inbound                      | |
    |  | VPC Private Subnets -> No direct internet access             | |
    |  | NAT Gateway -> Egress-only for backend calls                 | |
    |  | VPC Endpoints -> STS, S3 (no internet routing)               | |
    |  +--------------------------------------------------------------+ |
    |                                                                    |
    |  Layer 2: TRANSPORT                                                |
    |  +--------------------------------------------------------------+ |
    |  | TLS 1.2+ everywhere (ACM certificates, auto-renewal)         | |
    |  | DocumentDB TLS connections (DOCUMENTDB_USE_TLS flag)          | |
    |  +--------------------------------------------------------------+ |
    |                                                                    |
    |  Layer 3: AUTHENTICATION                                           |
    |  +--------------------------------------------------------------+ |
    |  | Every request validated via auth subrequest                   | |
    |  | JWT signature verification (RS256 from IdP, HS256 internal)  | |
    |  | Token expiry enforcement (8h M2M default, configurable)       | |
    |  | Session server-side (no sensitive data in cookies)            | |
    |  | HMAC-SHA256 timing-safe comparison for static tokens         | |
    |  | PKCE required for browser OAuth2 flows                       | |
    |  +--------------------------------------------------------------+ |
    |                                                                    |
    |  Layer 4: AUTHORIZATION                                            |
    |  +--------------------------------------------------------------+ |
    |  | Group-based access (mapped from IdP claims)                   | |
    |  | Fine-grained scopes (server + method + tool level)           | |
    |  | Scope enforcement at Nginx layer (before app code)           | |
    |  | Read-only federated items (cannot modify remote data)        | |
    |  | Registration gate webhook (admission control)                | |
    |  +--------------------------------------------------------------+ |
    |                                                                    |
    |  Layer 5: APPLICATION SECURITY                                     |
    |  +--------------------------------------------------------------+ |
    |  | Input validation (Pydantic models on all endpoints)          | |
    |  | CRLF injection prevention (custom header validation)         | |
    |  | No shell=True subprocess calls                               | |
    |  | Parameterized DB queries (no SQL/NoSQL injection)            | |
    |  | Credential masking in all logs and audit records             | |
    |  | Fernet encryption for federation tokens (AES-128-CBC+HMAC)  | |
    |  | AES-GCM encryption for session id_tokens (HKDF-derived key) | |
    |  +--------------------------------------------------------------+ |
    |                                                                    |
    |  Layer 6: SUPPLY CHAIN SECURITY                                    |
    |  +--------------------------------------------------------------+ |
    |  | Cisco AI Defense MCP Scanner on registration                  | |
    |  |   - YARA pattern matching (known malicious patterns)         | |
    |  |   - LLM-based behavioral analysis                           | |
    |  |   - Heuristic threat detection (data exfiltration, etc.)    | |
    |  | Auto-disable unsafe servers (SECURITY_BLOCK_UNSAFE_SERVERS)  | |
    |  | security-pending tag during scan (no execution allowed)      | |
    |  | Periodic full registry rescan                                 | |
    |  | Agent + Skill scanners (same pipeline)                       | |
    |  +--------------------------------------------------------------+ |
    |                                                                    |
    |  Layer 7: DATA PROTECTION                                          |
    |  +--------------------------------------------------------------+ |
    |  | Encryption at rest: KMS (DocumentDB, EBS, EFS, S3)           | |
    |  | Encryption in transit: TLS 1.2+ (all inter-service)          | |
    |  | Fernet-encrypted token storage (federation, custom headers)  | |
    |  | No PII in logs (credential masking, token redaction)         | |
    |  | TTL-based data expiry (audit logs, sessions, metrics)        | |
    |  | Server-side session store (no token in cookies)              | |
    |  +--------------------------------------------------------------+ |
    |                                                                    |
    |  Layer 8: OPERATIONAL SECURITY                                     |
    |  +--------------------------------------------------------------+ |
    |  | Non-root containers (CIS Docker Benchmark 4.1)               | |
    |  | cap_drop: ALL (selectively restored per service)             | |
    |  | no-new-privileges security option                            | |
    |  | IAM roles for DocumentDB (optional, via DOCUMENTDB_USE_IAM) | |
    |  | Secrets Manager for credential storage                       | |
    |  | Bandit + Ruff security scanning in CI/CD                     | |
    |  +--------------------------------------------------------------+ |
    +------------------------------------------------------------------+


CREDENTIAL FLOW & STORAGE
===========================

    +-------------------+     +-------------------+     +------------------+
    | SECRET_KEY        |     | IdP Signing Keys  |     | Federation Token |
    | (env var)         |     | (RS256 public)    |     | (env or DB)      |
    +--------+----------+     +--------+----------+     +--------+---------+
             |                         |                          |
             v                         v                          v
    +-------------------+     +-------------------+     +------------------+
    | HKDF-SHA256       |     | JWT Verification  |     | Fernet Encrypt   |
    | derive AES-GCM    |     | (PyJWT + JWKS     |     | (AES-128-CBC +   |
    | session key       |     |  endpoint cache)  |     |  HMAC-SHA256)    |
    +--------+----------+     +-------------------+     +--------+---------+
             |                                                    |
             v                                                    v
    +-------------------+                              +------------------+
    | id_token encrypted|                              | Stored in        |
    | via AES-GCM in    |                              | MongoDB:         |
    | MongoDB (opaque   |                              |  federation_*    |
    | to DB admin)      |                              |  custom_headers  |
    +-------------------+                              +------------------+


AUDIT & COMPLIANCE TRAIL
==========================

    Every API call generates:
    +--------------------------------------------------+
    | {                                                 |
    |   "timestamp": "2026-05-21T10:30:00Z",          |
    |   "user_id": "amit@example.com",                |
    |   "operation": "tools/call",                    |
    |   "target_server": "/github",                   |
    |   "tool_name": "search_repos",                  |
    |   "source_ip": "10.0.1.42",                     |
    |   "http_status": 200,                           |
    |   "auth_method": "jwt",                         |
    |   "groups": ["engineering"],                     |
    |   "sensitive_fields": "[REDACTED]"              |
    | }                                                |
    +--------------------------------------------------+
              |
              v
    +-------------------+     +-------------------+
    | MongoDB           |     | File rotation     |
    | mcp_audit_logs_*  |     | (50MB x 5 files)  |
    | TTL: 7 days       |     | JSON structured   |
    | (configurable)    |     |                   |
    +-------------------+     +-------------------+


THREAT MODEL SUMMARY
=====================

    Threat                          | Mitigation
    --------------------------------|------------------------------------------
    Stolen JWT                      | Configurable expiry (8h M2M default),
                                    | no refresh tokens, scope-limited
    --------------------------------|------------------------------------------
    Malicious MCP server            | Cisco AI Defense scan on registration,
    (supply chain)                  | auto-block unsafe, periodic rescan
    --------------------------------|------------------------------------------
    Credential exfiltration         | Fernet encryption at rest, never logged,
    from database                   | masking in audit trail
    --------------------------------|------------------------------------------
    Federation man-in-the-middle    | TLS required, static token or OAuth2,
                                    | tokens encrypted in DB
    --------------------------------|------------------------------------------
    Privilege escalation via        | Scope enforcement at Nginx (before app),
    API manipulation                | Pydantic validation, group-based ACL
    --------------------------------|------------------------------------------
    Container breakout              | Non-root, dropped caps, no-new-privs,
                                    | private subnets, security groups
    --------------------------------|------------------------------------------
    Injection (SQL/NoSQL/Command)   | Parameterized queries, no shell=True,
                                    | CRLF prevention, Pydantic input models
    --------------------------------|------------------------------------------
    DDoS / resource exhaustion      | CloudFront WAF, ALB rate limiting,
                                    | subprocess timeouts, query limits
    --------------------------------|------------------------------------------
    Insider data access             | Server-side encrypted sessions, audit
                                    | logging, KMS encryption, IAM roles
    --------------------------------|------------------------------------------
    Stale federated data            | Generation-based orphan detection,
    (poisoning via abandoned peer)  | cascade cleanup on peer removal
```

---

## Quick Reference

| Level | What it covers                                           |
|-------|----------------------------------------------------------|
| 100   | What the system does, core capabilities, value props     |
| 200   | How services connect, deployment options, data layer     |
| 300   | Request lifecycle, protocol details, search pipeline     |
| 400   | Trust boundaries, threat model, compliance controls      |
