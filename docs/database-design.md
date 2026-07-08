# Database Design - MCP Gateway Registry

## Overview

The MCP Gateway Registry supports three storage backends for data persistence:

1. **File-Based Backend** (Legacy, Backwards Compatible)
   - JSON file storage in the local filesystem
   - Maintained for backwards compatibility
   - Single-node deployments only
   - Application-level vector search

2. **MongoDB CE** (Local Development)
   - MongoDB Community Edition 8.2
   - Docker-based local deployment
   - Application-level vector search
   - Development and testing environments

3. **AWS DocumentDB** (Production, Recommended)
   - MongoDB-compatible managed service
   - Supports clustering configuration
   - Native vector search with HNSW indexes
   - Multi-tenancy support via namespaces
   - Recommended for all production deployments

The default configuration for local development uses **MongoDB CE**, while production deployments use **AWS DocumentDB**.

---

## Quick Architecture Reference

```
Application Services
        │
        ▼
Repository Factory (factory.py)
        │
        ├─> File Backend (legacy)
        │   └─> Local JSON files
        │
        ├─> MongoDB CE (local dev)
        │   └─> Docker container + app-level vector search
        │
        └─> AWS DocumentDB (production)
            └─> Managed service + native vector search
```

---

## Storage Backend Comparison

| Feature | File | MongoDB CE | AWS DocumentDB |
|---------|------|------------|----------------|
| **Use Case** | Legacy/Testing | Local Development | Production |
| **Setup** | None | Docker Compose | Terraform |
| **Scalability** | ~1,000 entities | ~10,000 | Millions |
| **Vector Search** | App-level (Python) | App-level (Python) | HNSW (native) |
| **Query Latency** | 50-100ms | 50-200ms | 10-50ms |
| **Concurrency** | Limited | Good | Excellent |
| **HA/Clustering** | No | Manual | Automatic |
| **Multi-tenancy** | No | Via namespace | Via namespace |
| **Cost** | Free | Free | AWS pricing |
| **Best For** | Quick start | Feature development | Production |

---

## MongoDB CE & DocumentDB Architecture

For detailed information about the MongoDB and DocumentDB backends, see:

**[Storage Architecture: MongoDB CE & AWS DocumentDB](./design/storage-architecture-mongodb-documentdb.md)**

This comprehensive guide covers:
- MongoDB CE local development setup
- AWS DocumentDB production deployment
- Vector search implementation (app-level vs. native)
- Build and run process with `build_and_run.sh`
- Collection schemas and indexes
- Migration strategies
- Performance characteristics

### Quick Summary

**Collections (both MongoDB CE and DocumentDB):**

All collections are suffixed with the configured namespace (e.g., `_default`, `_production`):

1. **mcp_servers_{namespace}** - Server definitions
2. **mcp_agents_{namespace}** - Agent cards
3. **mcp_scopes_{namespace}** - Authorization scopes
4. **mcp_embeddings_1536_{namespace}** - Vector embeddings
5. **mcp_security_scans_{namespace}** - Security scan results
6. **mcp_federation_config_{namespace}** - Federation configuration
7. **oauth_sessions_{namespace}** - Server-side OAuth session records (cookie-based browser login)
8. **mcp_custom_types_{namespace}** - Custom entity type descriptors (admin-defined schemas)
9. **mcp_custom_entities_{namespace}** - Custom entity records (instances of custom types)

**Key Differences:**

| Aspect | MongoDB CE | AWS DocumentDB |
|--------|------------|----------------|
| Vector Search | Python cosine similarity | HNSW index |
| Connection | `mongodb://mongodb:27017` | `mongodb://cluster.docdb.amazonaws.com:27017` |
| Authentication | None (local) | Username/Password or IAM |
| TLS | Disabled | Required |
| Deployment | Docker Compose | Terraform |

---

## Collection Schemas

### 1. MCP Servers

**Collection:** `mcp_servers_{namespace}`

Stores MCP server definitions and metadata.

**Document Structure:**

```json
{
  "_id": "/servers/financial-data",
  "server_name": "Financial Data Server",
  "description": "Provides stock market data and analysis",
  "path": "/servers/financial-data",
  "proxy_pass_url": "http://financial-server:8000",
  "supported_transports": ["stdio", "sse"],
  "auth_type": "oauth",
  "tags": ["finance", "data", "stocks"],
  "num_tools": 15,
  "tool_list": [
    {
      "name": "get_stock_price",
      "description": "Get current stock price",
      "schema": { /* JSON schema */ }
    }
  ],
  "is_enabled": true,
  "registered_at": "2026-01-03T10:00:00Z",
  "updated_at": "2026-01-03T12:30:00Z",
  "ans_metadata": null
}
```

The `ans_metadata` field follows the same structure as in the agents collection (see below). It is `null` when no ANS link is configured.

**Indexes:**

- `path` (unique) - Primary key
- `is_enabled` - Filter active servers
- `tags` - Tag-based filtering
- `server_name` - Text search

---

### 2. A2A Agents

**Collection:** `mcp_agents_{namespace}`

Stores Agent-to-Agent (A2A) agent cards and capabilities.

**Document Structure:**

```json
{
  "_id": "/agents/financial-analyst",
  "protocol_version": "1.0",
  "name": "Financial Analysis Agent",
  "description": "Analyzes financial data and provides insights",
  "path": "/agents/financial-analyst",
  "url": "https://registry.example.com/agents/financial-analyst",
  "version": "2.1.0",
  "capabilities": ["analysis", "reporting", "forecasting"],
  "tags": ["finance", "analysis"],
  "is_enabled": true,
  "visibility": "public",
  "trust_level": "high",
  "registered_at": "2026-01-02T09:00:00Z",
  "updated_at": "2026-01-03T11:00:00Z",
  "ans_metadata": {
    "ans_agent_id": "ans://v1.0.0.agent.example.com",
    "linked_at": "2026-01-02T09:00:00Z",
    "last_verified": "2026-01-02T09:00:00Z",
    "status": "verified",
    "domain": "agent.example.com",
    "organization": null,
    "ans_name": "ans://v1.0.0.agent.example.com",
    "ans_display_name": "Financial Analysis Agent",
    "ans_version": "1.0.0",
    "registered_with_ans_at": "2026-01-01T12:00:00Z",
    "certificate": null,
    "endpoints": [
      { "type": "http", "url": "https://agent.example.com/a2a", "protocol": "A2A", "transports": ["STREAMABLE-HTTP"], "functions": [] }
    ],
    "links": [
      { "rel": "self", "href": "https://api.godaddy.com/v1/agents/uuid" },
      { "rel": "server-certificates", "href": "https://api.godaddy.com/v1/agents/uuid/certificates/server" },
      { "rel": "identity-certificates", "href": "https://api.godaddy.com/v1/agents/uuid/certificates/identity" }
    ],
    "raw_ans_response": {}
  }
}
```

The `ans_metadata` field is `null` when no ANS Agent ID is linked. It is populated when an agent is linked to the GoDaddy Agent Name Service (ANS) for PKI-based identity verification.

**Indexes:**

- `path` (unique) - Primary key
- `is_enabled` - Filter active agents
- `tags` - Tag-based filtering
- `name` - Text search
- `visibility` - Access control
- `ans_metadata.status` - Filter by ANS verification status

---

### 3. Authorization Scopes

**Collection:** `mcp_scopes_{namespace}`

Stores authorization scopes, permission mappings, and UI access control.

**Document Types:**

The scopes collection stores three document types, distinguished by `scope_type`:

#### Server Scope Document

```json
{
  "_id": "scope:admin_access",
  "scope_type": "server_scope",
  "scope_name": "admin_access",
  "server_access": [
    {
      "server": "financial_server",
      "methods": ["GET", "POST", "PUT"],
      "tools": ["analyze_data", "generate_report"]
    }
  ],
  "description": "Full access to financial servers",
  "created_at": "2026-01-01T08:00:00Z",
  "updated_at": "2026-01-03T10:00:00Z"
}
```

#### Group Mapping Document

```json
{
  "_id": "group:finance_team",
  "scope_type": "group_mapping",
  "group_name": "finance_team",
  "group_mappings": ["admin_access", "read_only_access"],
  "created_at": "2026-01-01T08:00:00Z",
  "updated_at": "2026-01-03T10:00:00Z"
}
```

#### UI Scope Document

```json
{
  "_id": "ui:finance_team",
  "scope_type": "ui_scope",
  "scope_name": "finance_team",
  "ui_permissions": {
    "list_service": ["financial_server", "analytics_server"]
  },
  "created_at": "2026-01-01T08:00:00Z",
  "updated_at": "2026-01-03T10:00:00Z"
}
```

**Indexes:**

- `_id` (unique) - Primary key
- `scope_type` - Document type filter
- `scope_name` - Scope lookup
- `group_name` - Group lookup

---

### 4. Vector Embeddings

**Collection:** `mcp_embeddings_{dimensions}_{namespace}`

Example: `mcp_embeddings_1536_default` for 1536-dimensional embeddings

Stores vector embeddings for semantic search across servers and agents.

**Document Structure:**

```json
{
  "_id": "/servers/financial-data",
  "entity_type": "mcp_server",
  "path": "/servers/financial-data",
  "name": "Financial Data Server",
  "description": "Provides stock market data and analysis",
  "tags": ["finance", "data"],
  "is_enabled": true,
  "text_for_embedding": "Financial Data Server. Provides stock market data and analysis. Tools: get_stock_price, analyze_portfolio",
  "embedding": [0.125, -0.342, 0.098, ...],  // 1536 floats
  "embedding_metadata": {
    "model": "amazon.titan-embed-text-v1",
    "provider": "litellm",
    "dimensions": 1536,
    "created_at": "2026-01-03T10:30:00Z"
  },
  "tools": [
    {"name": "get_stock_price", "description": "Get current stock price"}
  ],
  "metadata": { /* full server info */ },
  "indexed_at": "2026-01-03T10:30:00Z"
}
```

**Indexes:**

- `path` (unique) - Primary key
- `entity_type` - Filter by entity type
- `embedding` (vector) - **DocumentDB only:** HNSW vector index for fast similarity search
  ```javascript
  // HNSW index configuration (DocumentDB)
  {
    "type": "hnsw",
    "similarity": "cosine",
    "dimensions": 1536,
    "m": 16,
    "efConstruction": 128
  }
  ```

**Vector Search:**

- **MongoDB CE:** Application-level cosine similarity in Python
- **DocumentDB:** Native HNSW index for sub-100ms queries

---

### 5. Security Scans

**Collection:** `mcp_security_scans_{namespace}`

Stores security vulnerability scan results.

**Document Structure:**

```json
{
  "_id": "scan:financial_server:2026-01-03",
  "server_path": "/servers/financial-data",
  "scan_timestamp": "2026-01-03T14:00:00Z",
  "scan_status": "unsafe",
  "vulnerabilities": [
    {
      "severity": "high",
      "title": "SQL Injection vulnerability",
      "description": "User input not sanitized",
      "cve_id": "CVE-2024-12345",
      "package_name": "db-connector",
      "package_version": "2.1.0",
      "fixed_version": "2.1.5"
    }
  ],
  "risk_score": 0.75,
  "total_vulnerabilities": 2,
  "critical_count": 0,
  "high_count": 1,
  "medium_count": 1,
  "low_count": 0
}
```

**Indexes:**

- `server_path` - Lookup scans by server
- `scan_status` - Filter by status
- `scan_timestamp` (descending) - Get latest scans

---

### 6. Federation Config

**Collection:** `mcp_federation_config_{namespace}`

Stores federation configuration for external registries (Anthropic, ASOR).

**Document Structure:**

```json
{
  "_id": "federation-config",
  "anthropic": {
    "enabled": true,
    "endpoint": "https://registry.modelcontextprotocol.io",
    "sync_on_startup": true,
    "servers": [
      {"name": "weather-service"},
      {"name": "news-aggregator"}
    ]
  },
  "asor": {
    "enabled": false,
    "endpoint": "https://asor-registry.example.com",
    "auth_env_var": "ASOR_AUTH_TOKEN",
    "sync_on_startup": false,
    "agents": []
  },
  "updated_at": "2026-01-03T12:00:00Z"
}
```

**Indexes:**

- `_id` (unique) - Single config per namespace

---

### 7. OAuth Sessions

**Collection:** `oauth_sessions_{namespace}`

Stores the server-side session record for browser-based logins. Introduced
in [PR #1042](https://github.com/agentic-community/mcp-gateway-registry/pull/1042)
to keep the browser cookie payload small (a single signed `session_id`)
regardless of how many groups the user belongs to. The auth-server writes
records on a successful OAuth callback; the registry reads on every
authenticated request.

See [docs/design/session-flow-cookie-based.md](design/session-flow-cookie-based.md)
for the end-to-end browser login flow that uses this collection.

**Document Structure:**

```json
{
  "_id": "ObjectId('...')",
  "session_id": "94780545a5b374024e91881e18ada42dfa17161c2cd03aca5b24b4320556bb4d",
  "username": "alice@example.com",
  "email": "alice@example.com",
  "name": "Alice Example",
  "groups": ["504c300b-d7a5-4731-b5ea-e4999a1e2496"],
  "provider": "entra",
  "auth_method": "oauth2",
  "created_at": "ISODate('2026-05-17T15:29:27.090Z')",
  "expires_at": "ISODate('2026-05-17T23:29:27.090Z')",
  "id_token_encrypted": "BinData(0, '<12-byte nonce || AES-GCM ciphertext>')"
}
```

**Field notes:**

- `session_id` — 64-character hex string (32 bytes from `secrets.token_hex(32)`).
  Primary lookup key; the only thing the browser cookie actually carries.
- `groups` — IdP-provided group claims (Entra group object IDs, Cognito
  group names, Keycloak group paths, etc.). May be empty.
- `provider` — One of `cognito`, `keycloak`, `okta`, `auth0`, `entra`.
  Used at logout time to redirect back to the right IdP.
- `id_token_encrypted` — Optional. The OIDC `id_token` encrypted at rest
  with AES-GCM. The 12-byte random nonce is prepended to the ciphertext.
  Encryption key is derived from `SECRET_KEY` via HKDF-SHA256 with the
  fixed info string `mcp-gateway-session-id-token-encryption`. Used only
  at logout to populate `id_token_hint` for SSO termination at the IdP.
- Username, email, name, and groups are stored in plaintext — they were
  already client-visible in the previous in-cookie payload, so encrypting
  them adds operational cost (debugging, audit) for no real
  threat-model gain. The `id_token` is the only true bearer credential.

**Indexes:**

- `_id` (default) — ObjectId primary key.
- `ux_session_id` (unique) on `session_id` — every session_id resolves to
  at most one record; collision prevention.
- `ttl_expires_at` on `expires_at` with `expireAfterSeconds=0` — MongoDB's
  TTL monitor automatically deletes the document at the time stored in
  `expires_at`. Cleanup runs every ~60 seconds.

**Lifecycle:**

| Event | What happens |
|-------|--------------|
| OAuth callback succeeds | auth-server inserts the record, returns `Set-Cookie: <session_id signed>`. |
| Authenticated request | Registry resolves cookie → `session_id` → `find_one({session_id})`. One indexed PK lookup per request. |
| TTL expiry | MongoDB auto-deletes via the TTL index. Subsequent reads return None → 401 → redirect to `/login`. |
| Logout | Registry calls `delete_session(session_id)` *before* clearing the cookie, closing the cookie-replay window. |
| Suspected leak | Operator drops the entire collection — every active session invalidated at once. See [docs/operations/incident-response.md](operations/incident-response.md). |

**Code references:**

- Writer: [auth_server/session_store.py](https://github.com/agentic-community/mcp-gateway-registry/blob/main/auth_server/session_store.py) (create / resolve / delete + index management).
- Reader: [registry/auth/session_store.py](https://github.com/agentic-community/mcp-gateway-registry/blob/main/registry/auth/session_store.py) (resolve / delete only).
- Crypto: [registry/auth/session_crypto.py](https://github.com/agentic-community/mcp-gateway-registry/blob/main/registry/auth/session_crypto.py) (HKDF-derived AES-GCM cipher, shared by writer and reader).

---

### 8. Custom Entity Types

**Collection:** `mcp_custom_types_{namespace}`

Stores admin-defined custom entity type descriptors (the schemas). Each document
is one type; its `_id` is the URL-safe, immutable type name. See
[Custom Entity Types](custom-entities.md) for the feature overview.

**Document Structure:**

```json
{
  "_id": "llm_prompt",
  "name": "llm_prompt",
  "display_name": "Prompts",
  "description": "Reusable, versioned LLM prompts shared across teams.",
  "fields": [
    {
      "name": "prompt_text",
      "label": "Prompt Text",
      "datatype": "text",
      "enum_values": null,
      "required": true,
      "semantic": true,
      "show_in_list": false
    },
    {
      "name": "role",
      "label": "Role",
      "datatype": "enum",
      "enum_values": ["system", "user", "assistant", "developer"],
      "required": true,
      "semantic": false,
      "show_in_list": true
    }
  ],
  "schema_version": 1,
  "created_by": "admin",
  "created_at": "2026-06-10T12:00:00Z"
}
```

- `_id` / `name` are immutable (the name is also the record path prefix and the
  `entity_type` discriminator). Only `display_name`/`description` are mutable
  (via `PATCH`).
- `datatype` is one of `string`, `text`, `number`, `bool`, `enum`, `date`,
  `array<string>`. `enum_values` is required iff `datatype == enum`.
- Read paths use a short-TTL in-process descriptor cache; write paths read
  authoritatively from this collection.

### 9. Custom Entity Records

**Collection:** `mcp_custom_entities_{namespace}`

Stores records (instances) of custom types. A single collection holds records of
all custom types, discriminated by `entity_type`. The `_id` is the synthetic
`path` (`/{entity_type}/{uuid}`).

**Document Structure:**

```json
{
  "_id": "/llm_prompt/2d687ddf-4d84-47e5-8182-b05359f07813",
  "path": "/llm_prompt/2d687ddf-4d84-47e5-8182-b05359f07813",
  "entity_type": "llm_prompt",
  "name": "Concise Code Reviewer",
  "description": "System prompt for terse, high-signal code review.",
  "visibility": "public",
  "allowed_groups": [],
  "owner": "admin",
  "tags": ["code-review", "engineering"],
  "is_enabled": true,
  "num_stars": 0.0,
  "rating_details": [],
  "created_at": "2026-06-10T12:00:00Z",
  "updated_at": "2026-06-10T12:00:00Z",
  "attributes": {
    "prompt_text": "You are a senior engineer reviewing a PR...",
    "role": "system"
  }
}
```

- The **envelope** (name, description, visibility, allowed_groups, owner, tags,
  is_enabled, ratings, timestamps) is uniform across all custom types.
- The **`attributes`** sub-document holds the per-type, user-defined fields,
  validated against the type's descriptor at write time. User field names live
  only inside `attributes`, never as top-level keys, so they cannot collide with
  envelope keys or Mongo-interpreted keys.
- `owner` is always server-derived from the caller, never the request body.
- Records are visibility-filtered on read (`public` / `private`-owner /
  `group-restricted` via `allowed_groups`). Each record also has a vector
  embedding in the shared embeddings collection (fields marked `semantic` in the
  descriptor contribute to the embedding text).

---

## Vector Search Architecture

### Embedding Generation

**Module:** `registry/embeddings/`

**Supported Providers:**

1. **Sentence Transformers** (Default, Local)
   - Model: `all-MiniLM-L6-v2` (384 dimensions)
   - Runs locally, no API costs
   - Good for development

2. **OpenAI** (Cloud)
   - Model: `text-embedding-ada-002` (1536 dimensions)
   - Requires API key
   - High quality embeddings

3. **Amazon Bedrock Titan** (Cloud)
   - Model: `amazon.titan-embed-text-v1` (1536 dimensions)
   - Uses IAM authentication
   - AWS-native integration

### Search Implementation

**See:** [Storage Architecture: MongoDB CE & AWS DocumentDB](./design/storage-architecture-mongodb-documentdb.md) for detailed search implementation.

**Summary:**

| Backend | Algorithm | Complexity | Latency |
|---------|-----------|------------|---------|
| MongoDB CE | Python cosine similarity | O(n) | 50-200ms |
| DocumentDB | HNSW index | O(log n) | 10-50ms |

### Hybrid Search

Both backends combine:
- **Vector similarity** (semantic matching) - Primary ranking
- **Text matching** (keyword boosting) - Secondary bonus

**Formula:**

```
final_score = vector_score + (text_boost * 0.03)

Where:
  vector_score = cosine_similarity(query_embedding, doc_embedding)  // 0-1
  text_boost = 3.0 (name match) + 2.0 (description match)           // 0-5
```

---

## Configuration

### Environment Variables

**File:** `.env`

```bash
# Storage Backend Selection
# Options:
#   "file" - JSON files (legacy)
#   "mongodb-ce" - MongoDB Community Edition (local dev)
#   "documentdb" - AWS DocumentDB (production)
STORAGE_BACKEND=mongodb-ce

# MongoDB/DocumentDB Connection
DOCUMENTDB_HOST=mongodb                    # Local: "mongodb", Prod: "cluster.docdb.amazonaws.com"
DOCUMENTDB_PORT=27017
DOCUMENTDB_DATABASE=mcp_registry
DOCUMENTDB_NAMESPACE=default               # Multi-tenancy: dev, staging, production

# Authentication (not needed for MongoDB CE)
DOCUMENTDB_USERNAME=admin
DOCUMENTDB_PASSWORD=secure_password

# TLS (MongoDB CE: false, DocumentDB: true)
DOCUMENTDB_USE_TLS=false
DOCUMENTDB_TLS_CA_FILE=global-bundle.pem
DOCUMENTDB_USE_IAM=false

# Replica Set
DOCUMENTDB_REPLICA_SET=rs0
DOCUMENTDB_READ_PREFERENCE=secondaryPreferred

# Embeddings Configuration
EMBEDDINGS_PROVIDER=sentence-transformers  # Or: litellm
EMBEDDINGS_MODEL_NAME=all-MiniLM-L6-v2    # Or: openai/text-embedding-ada-002
EMBEDDINGS_MODEL_DIMENSIONS=384            # Or: 1536
```

### Initialization

**MongoDB CE:**

```bash
# Start MongoDB and initialize
docker compose up -d mongodb
docker compose up mongodb-init

# Verify
docker exec mcp-mongodb mongosh --eval "use mcp_registry; show collections"
```

**AWS DocumentDB:**

```bash
# Deploy with Terraform
cd terraform/aws-ecs
terraform apply

# Collections and indexes created automatically on first application startup
```

---

## Repository Layer

All database operations go through repository interfaces defined in [`registry/repositories/interfaces.py`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/registry/repositories/interfaces.py):

- **ServerRepositoryBase:** Server CRUD operations
- **AgentRepositoryBase:** Agent card CRUD operations
- **ScopeRepositoryBase:** Authorization scope management
- **SecurityScanRepositoryBase:** Vulnerability scan storage
- **FederationConfigRepositoryBase:** Federation configuration
- **SearchRepositoryBase:** Vector search operations

**Factory:** `registry/repositories/factory.py`

The repository factory automatically selects the correct implementation based on `STORAGE_BACKEND`:

```python
if backend in ["documentdb", "mongodb-ce"]:
    from .documentdb.server_repository import DocumentDBServerRepository
    return DocumentDBServerRepository()
else:
    from .file.server_repository import FileServerRepository
    return FileServerRepository()
```

**Key Point:** `mongodb-ce` and `documentdb` use the **same repository code**. The only difference is the connection configuration.

---

## Migration from File Backend

### To MongoDB CE (Local Development)

1. **Update configuration:**
   ```bash
   # In .env
   STORAGE_BACKEND=mongodb-ce
   ```

2. **Start MongoDB:**
   ```bash
   docker compose up -d mongodb
   docker compose up mongodb-init
   ```

3. **Re-register servers and agents:**
   ```bash
   # Use API to register from backup files
   for file in backup/*.json; do
       curl -X POST http://localhost/servers \
           -H "Content-Type: application/json" \
           -d @"$file"
   done
   ```

### To AWS DocumentDB (Production)

1. **Deploy infrastructure:**
   ```bash
   cd terraform/aws-ecs
   terraform apply
   ```

2. **Update configuration:**
   ```bash
   STORAGE_BACKEND=documentdb
   DOCUMENTDB_HOST=<cluster-endpoint>
   DOCUMENTDB_USERNAME=<username>
   DOCUMENTDB_PASSWORD=<password>
   DOCUMENTDB_USE_TLS=true
   ```

3. **Import data:**
   ```bash
   # Use mongodump/mongorestore or API
   mongorestore --host=<cluster> --ssl --db=mcp_registry ./backup
   ```

---

## Performance Considerations

### MongoDB CE (Local Development)

- **Good for:** <10,000 documents
- **Search latency:** 50-200ms (O(n) scan)
- **Indexing:** Fast document insertion
- **Scaling:** Limited to single container resources

### AWS DocumentDB (Production)

- **Good for:** Millions of documents
- **Search latency:** 10-50ms (O(log n) HNSW)
- **Indexing:** Distributed across cluster
- **Scaling:** Horizontal (add read replicas), vertical (instance size)

### Optimization Tips

1. **Use appropriate instance sizes** (DocumentDB)
   - `db.r5.large` for development
   - `db.r5.xlarge` or larger for production

2. **Enable read replicas** for high read throughput

3. **Tune HNSW parameters** (DocumentDB)
   - `m=16, efConstruction=128` balances accuracy and speed
   - Increase for higher accuracy (slower)
   - Decrease for faster search (lower accuracy)

4. **Monitor query patterns** and create additional indexes as needed

---

## See Also

- **[Storage Architecture: MongoDB CE & AWS DocumentDB](./design/storage-architecture-mongodb-documentdb.md)** - Comprehensive guide
- **[Database Abstraction Layer Design](./design/database-abstraction-layer.md)** - Repository pattern details
- **[Embeddings Configuration](./embeddings.md)** - Vector embedding setup
- **[Configuration Guide](./configuration.md)** - Full configuration reference
- [MongoDB Documentation](https://www.mongodb.com/docs/manual/)
- [AWS DocumentDB Documentation](https://docs.aws.amazon.com/documentdb/)
