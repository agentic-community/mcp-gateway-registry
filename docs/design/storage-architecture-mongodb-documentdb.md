# Storage Architecture: MongoDB CE & AWS DocumentDB

**Status:** Current Implementation
**Last Updated:** January 30, 2026
**Target Audience:** Developers, DevOps Engineers, System Architects

## Table of Contents

1. [Overview](#overview)
2. [Storage Backend Options](#storage-backend-options)
3. [MongoDB CE Local Development](#mongodb-ce-local-development)
4. [AWS DocumentDB Production](#aws-documentdb-production)
5. [Vector Search Implementation](#vector-search-implementation)
6. [Build and Run Process](#build-and-run-process)
7. [Repository Architecture](#repository-architecture)
8. [Configuration](#configuration)
9. [Migration Strategy](#migration-strategy)

## Overview

The MCP Gateway Registry supports three storage backends for data persistence:

1. **File-Based Backend** (Legacy) - JSON/YAML files with application-level search
2. **MongoDB CE** (Local Development) - MongoDB Community Edition 8.2 with application-level vector search
3. **AWS DocumentDB** (Production) - MongoDB-compatible service with native vector search

This document focuses on the MongoDB and DocumentDB backends, which provide distributed storage with semantic search capabilities.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   Application Layer                          │
│             (Services, API Endpoints)                        │
└────────────────────┬────────────────────────────────────────┘
                     │ depends on
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Repository Factory Layer                        │
│  get_server_repository()                                    │
│  get_search_repository()                                    │
│  etc.                                                       │
└────────┬──────────────────────┬──────────────┬─────────────┘
         │                      │              │
         │ STORAGE_BACKEND=     │              │
         │ file / mongodb-ce /  │              │
         │ documentdb           │              │
         ▼                      ▼              ▼
┌──────────────────┐  ┌─────────────────┐  ┌──────────────────┐
│ File Backend     │  │ MongoDB CE      │  │ AWS DocumentDB   │
├──────────────────┤  ├─────────────────┤  ├──────────────────┤
│ FileServerRepo   │  │ DocumentDBRepo  │  │ DocumentDBRepo   │
│ App-level search │  │ + App-level     │  │ + Native         │
│                  │  │   vector search │  │   vector search  │
└──────────────────┘  └─────────────────┘  └──────────────────┘
         │                      │                     │
         ▼                      ▼                     ▼
    Local Files        Local MongoDB CE     AWS DocumentDB
   (JSON files)          (Docker)           (Managed Service)
```

---

## Storage Backend Options

### Comparison Matrix

| Aspect | File | MongoDB CE | AWS DocumentDB |
|--------|------|------------|----------------|
| **Use Case** | Dev/Testing | Local Development | Production |
| **Scalability** | ~1000 entities | 10,000s | Millions |
| **Vector Search** | App-level (Python) | App-level (Python) | Native (HNSW) |
| **Setup Complexity** | None | Docker Compose | Terraform/AWS |
| **Concurrency** | Limited | Good | Excellent |
| **HA/Clustering** | No | Manual | Automatic |
| **Cost** | Free | Free | AWS Pricing |
| **Best For** | Quick start | Feature development | Production deployments |

---

## MongoDB CE Local Development

### Architecture

MongoDB Community Edition 8.2 provides a local development environment that mimics production DocumentDB behavior without requiring AWS infrastructure.

#### Key Components

1. **MongoDB 8.2 Container** (`mongo:8.2`)
   - Runs in Docker Compose
   - Configured as replica set (`rs0`) for transaction support
   - No authentication for local development simplicity
   - Bind address: `127.0.0.1,mongodb`

2. **Application-Level Vector Search**
   - Python-based cosine similarity computation
   - Embeddings stored in `mcp_embeddings_1536_default` collection
   - Full scan with in-memory ranking for search queries
   - Isolated in `DocumentDBSearchRepository` class

3. **Collections**
   - `mcp_servers_{namespace}` - Server definitions
   - `mcp_agents_{namespace}` - Agent cards
   - `mcp_scopes_{namespace}` - Authorization scopes
   - `mcp_embeddings_1536_{namespace}` - Vector embeddings
   - `mcp_security_scans_{namespace}` - Security scan results
   - `mcp_federation_config_{namespace}` - Federation settings

### Vector Search Implementation (MongoDB CE)

Since MongoDB CE 8.2 doesn't include the separate `mongot` search component needed for native vector search, we implement semantic search at the application layer.

#### Search Flow

```python
# 1. Query arrives
query = "financial analysis tools"

# 2. Generate query embedding
model = create_embeddings_client(...)
query_embedding = model.encode([query])[0]  # [1536 dimensions]

# 3. Retrieve all embeddings from MongoDB
docs = await collection.find({"entity_type": "mcp_server"}).to_list(length=1000)

# 4. Calculate cosine similarity in Python
for doc in docs:
    doc_embedding = doc["embedding"]  # [1536 dimensions]
    score = cosine_similarity(query_embedding, doc_embedding)
    doc["relevance_score"] = score

# 5. Rank results by similarity
results = sorted(docs, key=lambda x: x["relevance_score"], reverse=True)

# 6. Apply text-based boosting (hybrid search)
# If query appears in name or description, add bonus points
for result in results:
    if query.lower() in result["name"].lower():
        result["relevance_score"] += 0.1  # Name match bonus
    if query.lower() in result["description"].lower():
        result["relevance_score"] += 0.05  # Description match bonus

# 7. Return top-k results
return results[:max_results]
```

#### Code Location

**File:** `registry/repositories/documentdb/search_repository.py`

**Key Methods:**

```python
class DocumentDBSearchRepository:
    async def search(
        self,
        query: str,
        entity_types: Optional[List[str]] = None,
        max_results: int = 10,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Hybrid search with application-level vector similarity."""
        # Lines 240-385: Full implementation

    def _calculate_cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        # Lines 199-220: Pure Python implementation
        # Returns value between 0 and 1 (1 = identical)
```

#### Performance Characteristics

- **Pros:**
  - No dependency on external search services
  - Works identically to production DocumentDB (same code path)
  - Full control over ranking algorithm
  - No additional containers needed

- **Cons:**
  - O(n) full collection scan for every search query
  - All embeddings loaded into memory for comparison
  - Not suitable for >10,000 entities
  - No index optimization (brute force)

- **Optimization:**
  - For local dev with <1,000 entities, performance is acceptable (<100ms)
  - Production workloads should use AWS DocumentDB with native vector search

### Docker Compose Configuration

**File:** `docker-compose.yml` (lines 59-77)

```yaml
mongodb:
  image: mongo:8.2
  container_name: mcp-mongodb
  command: mongod --replSet rs0 --bind_ip 127.0.0.1,mongodb
  ports:
    - "27017:27017"
  volumes:
    - mongodb-data:/data/db
    - mongodb-config:/data/configdb
  healthcheck:
    test: ["CMD", "mongosh", "--eval", "db.adminCommand('ping')"]
    interval: 10s
    timeout: 5s
    retries: 5
    start_period: 20s
  restart: unless-stopped
```

**Key Settings:**

- **Replica Set:** `--replSet rs0` enables transactions (required for DocumentDB compatibility)
- **Bind Address:** `127.0.0.1,mongodb` - listens on localhost and container network
- **No Authentication:** Simplifies local development (not for production!)
- **Healthcheck:** Ensures container is ready before dependent services start

### Initialization

**Script:** `scripts/init-mongodb-ce.py`

**What It Does:**

1. **Initializes Replica Set**
   ```python
   # Initialize rs0 replica set (required for transactions)
   config = {
       "_id": "rs0",
       "members": [{"_id": 0, "host": "mongodb:27017"}]
   }
   client.admin.command("replSetInitiate", config)
   ```

2. **Creates Collections**
   - Server registry (`mcp_servers_default`)
   - Agent registry (`mcp_agents_default`)
   - Authorization scopes (`mcp_scopes_default`)
   - Vector embeddings (`mcp_embeddings_1536_default` - 1536 dimensions for OpenAI/Titan)
   - Security scans (`mcp_security_scans_default`)
   - Federation config (`mcp_federation_config_default`)

3. **Creates Indexes** for query performance
   ```python
   # Server indexes
   await collection.create_index([("path", ASCENDING)], unique=True)
   await collection.create_index([("enabled", ASCENDING)])
   await collection.create_index([("tags", ASCENDING)])

   # Embeddings indexes
   await collection.create_index([("path", ASCENDING)], unique=True)
   await collection.create_index([("entity_type", ASCENDING)])
   ```

4. **Loads OAuth Scopes** from JSON scope seed files in `scripts/`
   ```python
   # Reads JSON scope files in scripts/ (registry-admins.json,
   # mcp-registry-admin.json, mcp-servers-unrestricted-read.json,
   # mcp-servers-unrestricted-execute.json, federation-service.json)
   # and populates the mcp_scopes collection.
   # Includes server scopes and group mappings.
   await _seed_scopes_from_json(db, namespace, scope_seed_files)
   ```

5. **Note on Vector Index**
   - MongoDB CE 8.2 does not include native vector search (requires `mongot` component)
   - Vector search is implemented at application level using Python cosine similarity
   - See `registry/repositories/documentdb/search_repository.py` for implementation

### Environment Variables

**File:** `.env.example` (lines 360-386)

```bash
# Storage backend selection
STORAGE_BACKEND=mongodb-ce  # Use MongoDB CE local instance

# DocumentDB Configuration (reused for MongoDB CE)
DOCUMENTDB_HOST=mongodb  # Docker Compose service name
DOCUMENTDB_PORT=27017
DOCUMENTDB_DATABASE=mcp_registry
DOCUMENTDB_NAMESPACE=default

# No authentication needed for local MongoDB CE
# DOCUMENTDB_USERNAME and DOCUMENTDB_PASSWORD not required
# DOCUMENTDB_USE_TLS=false (disabled for local)
```

---

## AWS DocumentDB Production

### Architecture

AWS DocumentDB is a MongoDB-compatible managed database service optimized for cloud deployments with native vector search support.

#### Key Components

1. **DocumentDB Cluster**
   - Managed by AWS
   - Multi-AZ deployment for redundancy
   - Auto-scaling read replicas
   - Automated backups and point-in-time recovery

2. **Native Vector Search**
   - HNSW (Hierarchical Navigable Small World) algorithm
   - Index-based approximate nearest neighbor (ANN)
   - Sub-100ms query times even with millions of vectors
   - Cosine similarity metric

3. **Network Architecture**
   - Private VPC deployment
   - TLS encryption in transit
   - VPC security groups for access control

### Vector Search Implementation (DocumentDB)

AWS DocumentDB provides native vector search using optimized indexes, eliminating the need for application-level computation.

#### Search Flow

```python
# 1. Query arrives
query = "financial analysis tools"

# 2. Generate query embedding (same as MongoDB CE)
query_embedding = model.encode([query])[0]

# 3. DocumentDB performs indexed vector search with tuned parameters
ef_search = settings.vector_search_ef_search  # Default: 100
k_value = max(max_results * 3, 50)            # At least 50 for small collections
pipeline = [
    {
        "$search": {
            "vectorSearch": {
                "vector": query_embedding,
                "path": "embedding",
                "similarity": "cosine",
                "k": k_value,
                "efSearch": ef_search,
            }
        }
    }
]

# 4. DocumentDB returns sorted results (FAST - uses HNSW index)
# Results are already ranked by vector similarity

# 5. Apply text-based boosting in aggregation pipeline
#    Each query keyword is matched independently against path, name,
#    description, tags, and tool names/descriptions
pipeline.append({
    "$addFields": {
        "text_boost": {
            "$add": [
                # Per keyword: path(+5), name(+3), description(+2), tags(+1.5)
                {"$cond": [{"$regexMatch": {"input": "$path", "regex": keyword, "options": "i"}}, 5.0, 0.0]},
                {"$cond": [{"$regexMatch": {"input": "$name", "regex": keyword, "options": "i"}}, 3.0, 0.0]},
                {"$cond": [{"$regexMatch": {"input": "$description", "regex": keyword, "options": "i"}}, 2.0, 0.0]},
                # ... plus tags (+1.5) and each tool (+1.0)
            ]
        }
    }
})

# 6. Score combination: normalized_vector + (text_boost * 0.1)
# All candidates scored, sorted by hybrid score, then top-3 per entity type

# 7. Execute aggregation pipeline
results = await collection.aggregate(pipeline).to_list(k_value)
```

#### Code Location

**File:** `registry/repositories/documentdb/search_repository.py`

**The same code works for both MongoDB CE and AWS DocumentDB!**

The key difference:
- **MongoDB CE:** No vector index → slow full scan
- **DocumentDB:** HNSW vector index → fast indexed search

The application code is identical, but DocumentDB executes it much faster.

#### Performance Characteristics

- **Pros:**
  - O(log n) indexed search via HNSW
  - Handles millions of vectors efficiently
  - Sub-100ms latency even with large datasets
  - Native database operation (no network round trips)
  - Automatic index optimization

- **Cons:**
  - Requires AWS infrastructure
  - Additional cost for managed service
  - Network latency to AWS region

- **Optimization:**
  - HNSW parameters tuned for accuracy vs. speed tradeoff
  - `m=16, efConstruction=128` provides good balance for index construction
  - `efSearch=100` (configurable) provides near-exact recall for typical deployments
  - Minimum `k=50` ensures small collections are fully covered
  - Can adjust via `VECTOR_SEARCH_EF_SEARCH` environment variable

### Terraform Configuration

**Directory:** `terraform/aws-ecs/`

**Key Resources:**

1. **DocumentDB Cluster** (`modules/documentdb/main.tf`)
   ```hcl
   resource "aws_docdb_cluster" "main" {
     cluster_identifier      = "mcp-registry-${var.environment}"
     engine                  = "docdb"
     master_username         = var.master_username
     master_password         = var.master_password

     # Redundancy
     backup_retention_period = 7
     preferred_backup_window = "03:00-04:00"

     # Security
     storage_encrypted       = true
     kms_key_id             = aws_kms_key.docdb.arn

     # Network
     db_subnet_group_name    = aws_docdb_subnet_group.main.name
     vpc_security_group_ids  = [aws_security_group.docdb.id]
   }
   ```

2. **DocumentDB Instance(s)** (read/write nodes)
   ```hcl
   resource "aws_docdb_cluster_instance" "main" {
     count              = var.instance_count
     identifier         = "mcp-registry-${var.environment}-${count.index}"
     cluster_identifier = aws_docdb_cluster.main.id
     instance_class     = var.instance_class  # db.r5.large, db.r6g.xlarge, etc.
   }
   ```

3. **Vector Search Configuration**
   - Automatically enabled on DocumentDB 5.0+
   - No additional configuration needed
   - HNSW index created via application init script

### Environment Variables (Production)

**File:** `.env` (not in git)

```bash
# Storage backend
STORAGE_BACKEND=documentdb  # Use AWS DocumentDB

# DocumentDB connection
DOCUMENTDB_HOST=mcp-registry-prod.cluster-xxxxx.us-east-1.docdb.amazonaws.com
DOCUMENTDB_PORT=27017
DOCUMENTDB_DATABASE=mcp_registry
DOCUMENTDB_USERNAME=mcp_admin
DOCUMENTDB_PASSWORD=<secure-password-from-secrets-manager>

# Security settings
DOCUMENTDB_USE_TLS=true
DOCUMENTDB_TLS_CA_FILE=/app/global-bundle.pem  # AWS DocumentDB CA bundle
DOCUMENTDB_USE_IAM=false  # Set to true for IAM authentication

# Replica set configuration
DOCUMENTDB_REPLICA_SET=rs0
DOCUMENTDB_READ_PREFERENCE=secondaryPreferred  # Distribute read load

# Namespace for multi-tenancy
DOCUMENTDB_NAMESPACE=production
```

---

## Vector Search Implementation

### Overview

Vector search enables semantic search - finding conceptually similar servers and agents even when exact keywords don't match.

**Example:**

```
Query: "financial analytics"
Matches:
  ✓ "Stock market analysis tools" (85% similarity)
  ✓ "Portfolio management assistant" (78% similarity)
  ✗ "Weather forecast service" (12% similarity)
```

### Embedding Generation

**Module:** `registry/embeddings/`

**Providers:**

1. **Sentence Transformers** (Default, Local)
   - Model: `all-MiniLM-L6-v2`
   - Dimensions: 384
   - Runs locally, no API costs

2. **LiteLLM** (Cloud, via API)
   - OpenAI: `text-embedding-ada-002` (1536 dims)
   - Amazon Bedrock Titan: `amazon.titan-embed-text-v1` (1536 dims)
   - Cohere: `embed-english-v3.0` (1024 dims)

**Configuration:**

```bash
# Local embeddings
EMBEDDINGS_PROVIDER=sentence-transformers
EMBEDDINGS_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDINGS_MODEL_DIMENSIONS=384

# OR Cloud embeddings (OpenAI)
EMBEDDINGS_PROVIDER=litellm
EMBEDDINGS_MODEL_NAME=openai/text-embedding-ada-002
EMBEDDINGS_MODEL_DIMENSIONS=1536
EMBEDDINGS_API_KEY=sk-...

# OR Amazon Bedrock
EMBEDDINGS_PROVIDER=litellm
EMBEDDINGS_MODEL_NAME=bedrock/amazon.titan-embed-text-v1
EMBEDDINGS_MODEL_DIMENSIONS=1536
EMBEDDINGS_AWS_REGION=us-east-1
```

### Embedding Storage

**Collection:** `mcp_embeddings_{dimensions}_{namespace}`

**Document Structure:**

```json
{
  "_id": "/servers/financial-data",
  "entity_type": "mcp_server",
  "path": "/servers/financial-data",
  "name": "Financial Data Server",
  "description": "Provides stock market data and analysis tools",
  "tags": ["finance", "data", "stocks"],
  "is_enabled": true,
  "text_for_embedding": "Financial Data Server. Provides stock market data and analysis tools. Tools: get_stock_price, analyze_portfolio, market_trends",
  "embedding": [0.125, -0.342, 0.098, ...],  // 1536 floats
  "embedding_metadata": {
    "model": "amazon.titan-embed-text-v1",
    "provider": "litellm",
    "dimensions": 1536,
    "created_at": "2026-01-03T10:30:00Z"
  },
  "tools": [
    {"name": "get_stock_price", "description": "Get current stock price"},
    {"name": "analyze_portfolio", "description": "Analyze investment portfolio"}
  ],
  "metadata": { /* full server info */ },
  "indexed_at": "2026-01-03T10:30:00Z"
}
```

### Search Algorithm Comparison

#### MongoDB CE (Application-Level)

```python
# File: registry/repositories/documentdb/search_repository.py

async def search(self, query: str, max_results: int = 10):
    # 1. Generate query embedding
    query_embedding = model.encode([query])[0]

    # 2. Fetch ALL documents (full scan)
    docs = await collection.find({}).to_list(length=10000)

    # 3. Calculate similarity for each document
    for doc in docs:
        doc["score"] = cosine_similarity(query_embedding, doc["embedding"])

    # 4. Sort by score (in Python)
    ranked = sorted(docs, key=lambda x: x["score"], reverse=True)

    # 5. Return top-k
    return ranked[:max_results]
```

**Time Complexity:** O(n) - must compare against every document
**Latency:** ~50-200ms for 1,000 documents
**Scalability:** Limited to ~10,000 documents

#### DocumentDB (Native Vector Search)

```python
# File: registry/repositories/documentdb/search_repository.py

async def search(self, query: str, max_results: int = 10):
    # 1. Generate query embedding
    query_embedding = model.encode([query])[0]

    # 2. DocumentDB HNSW index search with tuned parameters
    ef_search = settings.vector_search_ef_search  # Default: 100
    k_value = max(max_results * 3, 50)
    pipeline = [{
        "$search": {
            "vectorSearch": {
                "vector": query_embedding,
                "path": "embedding",
                "similarity": "cosine",
                "k": k_value,
                "efSearch": ef_search,
            }
        }
    }]

    # 3. DocumentDB returns sorted results (FAST!)
    results = await collection.aggregate(pipeline).to_list(k_value)

    # 4. Score all results, sort by hybrid score, pick top-3 per entity type
    return results
```

**Time Complexity:** O(log n) - HNSW index lookup
**Latency:** ~10-50ms for millions of documents
**Scalability:** Millions of documents

### Search Resilience: Lexical Fallback

Both backends support automatic fallback to lexical-only search when the embedding model is unavailable. This ensures search remains operational even during embedding provider outages, misconfiguration, or API key expiration.

**Behavior when embeddings are unavailable:**

- Servers and agents are indexed without embeddings (empty vectors)
- DocumentDB rejects 0-dimension vectors, so documents are stored without vector data
- Search uses MongoDB aggregation with `$regexMatch` for keyword matching on path, name, description, tags, and tools
- The `_load_error` cache in `SentenceTransformersClient` prevents repeated model download attempts
- API response includes `"search_mode": "lexical-only"` to indicate degraded mode

**Recovery:** Restart the service with correct embedding configuration. The error cache resets on restart and search returns to full hybrid mode.

See [Hybrid Search Architecture](hybrid-search-architecture.md) for detailed fallback flow and scoring.

### Hybrid Search (Text + Vector)

Both backends support hybrid search combining:
- **Vector similarity** (semantic matching)
- **Text matching** (keyword boosting)

**Example:**

```python
# Query: "stock market"
results = [
    {
        "name": "Financial Analysis Server",
        "vector_score": 0.85,         # Cosine similarity
        "normalized_vector": 0.925,   # (0.85 + 1.0) / 2.0
        "text_boost": 3.0,            # "market" found in name
        "boost_contrib": 0.30,        # 3.0 * 0.1
        "final_score": 1.0            # clamped to 1.0
    },
    {
        "name": "Investment Tools",
        "vector_score": 0.75,         # Cosine similarity
        "normalized_vector": 0.875,   # (0.75 + 1.0) / 2.0
        "text_boost": 0.0,            # No keyword match
        "boost_contrib": 0.0,
        "final_score": 0.875
    }
]
```

**Formula:**

```
normalized_vector = (cosine_similarity + 1.0) / 2.0   # Map [-1,1] to [0,1]
boost_contribution = text_boost * 0.1                   # Scale boost down
final_score = clamp(normalized_vector + boost_contribution, 0.0, 1.0)
```

The `0.1` multiplier is consistent across both DocumentDB and MongoDB CE search paths. Semantic relevance is primary (normalized vector score dominates) while keyword matches provide a meaningful boost for exact references.

---

## Build and Run Process

### Local Development with MongoDB CE

**Script:** `build_and_run.sh`

#### Flow

```
1. Load .env file
   └─> Check STORAGE_BACKEND variable

2. If STORAGE_BACKEND = mongodb-ce or documentdb:
   └─> Create empty directories for Docker mounts
   └─> (Data stored in MongoDB, not local files)

3. Start Docker Compose
   ├─> Start MongoDB container
   │   └─> Wait for healthcheck to pass
   ├─> Run mongodb-init container
   │   ├─> Initialize replica set
   │   ├─> Create collections
   │   └─> Create indexes
   └─> Start application services

4. Application startup
   ├─> Repository factory creates DocumentDBRepository instances
   ├─> Search repository initializes
   │   └─> Creates vector index if not exists (DocumentDB only)
   └─> Services load data from MongoDB
```

#### Key Script Logic

**File:** `build_and_run.sh` (lines 180-298)

```bash
# Build and run script always creates mount directories
# and copies JSON files for all backends
# (MongoDB/DocumentDB stores data in database, files are for initial seeding)

# Create mount directories
mkdir -p "$MCPGATEWAY_SERVERS_DIR"
mkdir -p "${HOME}/mcp-gateway/agents"
mkdir -p "${HOME}/mcp-gateway/auth_server"
mkdir -p "${HOME}/mcp-gateway/security_scans"
touch "${HOME}/mcp-gateway/federation.json"

# Copy server definitions from registry/servers/*.json
# Copy agent cards from cli/examples/*agent*.json
# (These provide initial data that can be imported via API)
# Authorization scopes are seeded into the mcp_scopes collection from
# JSON scope files in scripts/ at init time, not copied as a file.
```

### Starting the Stack

```bash
# 1. Ensure .env is configured
cat .env | grep STORAGE_BACKEND
# Should show: STORAGE_BACKEND=mongodb-ce

# 2. Run build and run script
./build_and_run.sh

# What happens:
# - MongoDB container starts
# - Waits for healthcheck (ping succeeds)
# - Runs init script (replica set + collections)
# - Starts application services
# - Registry connects to MongoDB
# - Search repository creates vector index (if DocumentDB)
```

### Verifying MongoDB CE Setup

```bash
# 1. Check MongoDB is running
docker compose ps mongodb
# Should show: Status = healthy

# 2. Check collections were created
docker exec -it mcp-mongodb mongosh --eval "use mcp_registry; show collections"
# Expected output:
#   mcp_agents_default
#   mcp_embeddings_1536_default
#   mcp_federation_config_default
#   mcp_scopes_default
#   mcp_security_scans_default
#   mcp_servers_default

# 3. Check replica set status
docker exec -it mcp-mongodb mongosh --eval "rs.status()"
# Should show: rs0 with 1 member (primary)

# 4. Verify application can connect
curl http://localhost/health
# Should return: {"status": "healthy"}
```

### Data Flow

```
User Action          →  API Endpoint  →  Service Layer  →  Repository
───────────────────────────────────────────────────────────────────────
Register Server      →  POST /servers →  server_service →  DocumentDBServerRepository
                                                           └─> MongoDB: mcp_servers_default

Search "finance"     →  GET /search   →  search_service →  DocumentDBSearchRepository
                                                           └─> MongoDB: mcp_embeddings_1536_default
                                                               └─> Vector search (app-level)

List Agents          →  GET /agents   →  agent_service  →  DocumentDBAgentRepository
                                                           └─> MongoDB: mcp_agents_default
```

---

## Repository Architecture

### Abstract Base Classes

**File:** `registry/repositories/interfaces.py`

All storage backends implement the same interfaces:

```python
class ServerRepositoryBase(ABC):
    @abstractmethod
    async def get(self, path: str) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    async def list_all(self) -> Dict[str, Dict[str, Any]]: ...

    @abstractmethod
    async def create(self, server_info: Dict[str, Any]) -> bool: ...
```

### Factory Pattern

**File:** `registry/repositories/factory.py`

```python
def get_server_repository() -> ServerRepositoryBase:
    backend = settings.storage_backend

    if backend in ["documentdb", "mongodb-ce"]:
        from .documentdb.server_repository import DocumentDBServerRepository
        return DocumentDBServerRepository()
    else:
        from .file.server_repository import FileServerRepository
        return FileServerRepository()
```

**Key Point:** `mongodb-ce` and `documentdb` use the **same repository implementation**. The only difference is:
- **mongodb-ce:** Connects to local MongoDB container
- **documentdb:** Connects to AWS DocumentDB cluster

The repository code is identical - only the connection string changes!

### Implementation Files

**Directory:** `registry/repositories/documentdb/`

```
documentdb/
├── __init__.py
├── client.py                        # MongoDB/DocumentDB client management
├── server_repository.py             # Server CRUD operations
├── agent_repository.py              # Agent CRUD operations
├── scope_repository.py              # Authorization scopes
├── search_repository.py             # Vector search (app-level OR native)
├── security_scan_repository.py      # Security scan results
└── federation_config_repository.py  # Federation configuration
```

### Client Management

**File:** `registry/repositories/documentdb/client.py`

```python
async def get_documentdb_client() -> AsyncIOMotorDatabase:
    """Get DocumentDB/MongoDB database client.

    Works with both:
    - MongoDB CE (local Docker)
    - AWS DocumentDB (production)

    Configuration via environment variables.
    """
    if _client is None:
        connection_string = _build_connection_string()
        # Example (MongoDB CE):
        # mongodb://mongodb:27017/mcp_registry?replicaSet=rs0

        # Example (DocumentDB):
        # mongodb://user:pass@docdb-cluster.us-east-1.docdb.amazonaws.com:27017/
        #   ?tls=true&tlsCAFile=/app/global-bundle.pem&replicaSet=rs0

        motor_client = AsyncIOMotorClient(connection_string)
        _client = motor_client[settings.documentdb_database]

    return _client

def get_collection_name(base_name: str) -> str:
    """Add namespace suffix to collection name."""
    return f"{base_name}_{settings.documentdb_namespace}"
```

---

## Configuration

### Environment Variables

**File:** `.env`

```bash
# ============================================================================
# STORAGE BACKEND CONFIGURATION
# ============================================================================

# Backend selection
STORAGE_BACKEND=mongodb-ce  # Options: file, mongodb-ce, documentdb

# MongoDB/DocumentDB connection
DOCUMENTDB_HOST=mongodb                    # MongoDB CE: "mongodb"
                                          # DocumentDB: "cluster.us-east-1.docdb.amazonaws.com"
DOCUMENTDB_PORT=27017
DOCUMENTDB_DATABASE=mcp_registry
DOCUMENTDB_NAMESPACE=default              # Multi-tenancy: dev, staging, production

# Authentication (not needed for MongoDB CE local)
DOCUMENTDB_USERNAME=admin                 # DocumentDB: actual username
DOCUMENTDB_PASSWORD=secure_password       # DocumentDB: from Secrets Manager

# TLS/Security (MongoDB CE: disabled, DocumentDB: enabled)
DOCUMENTDB_USE_TLS=false                  # MongoDB CE: false
                                          # DocumentDB: true
DOCUMENTDB_TLS_CA_FILE=global-bundle.pem  # DocumentDB CA bundle
DOCUMENTDB_USE_IAM=false                  # DocumentDB: set true for IAM auth

# Replica set configuration
DOCUMENTDB_REPLICA_SET=rs0
DOCUMENTDB_READ_PREFERENCE=secondaryPreferred  # Load balance reads

# ============================================================================
# EMBEDDINGS CONFIGURATION
# ============================================================================

# Embedding provider
EMBEDDINGS_PROVIDER=sentence-transformers  # Options: sentence-transformers, litellm
EMBEDDINGS_MODEL_NAME=all-MiniLM-L6-v2    # Or: openai/text-embedding-ada-002
EMBEDDINGS_MODEL_DIMENSIONS=384           # Or: 1536 for OpenAI/Titan

# For cloud embeddings
EMBEDDINGS_API_KEY=                       # OpenAI: sk-...
                                          # Bedrock: uses IAM (leave empty)
EMBEDDINGS_AWS_REGION=us-east-1           # For Amazon Bedrock
```

### Docker Compose

**File:** `docker-compose.yml`

```yaml
services:
  # MongoDB CE 8.2
  mongodb:
    image: mongo:8.2
    container_name: mcp-mongodb
    command: mongod --replSet rs0 --bind_ip 127.0.0.1,mongodb
    ports:
      - "27017:27017"
    volumes:
      - mongodb-data:/data/db
      - mongodb-config:/data/configdb
    healthcheck:
      test: ["CMD", "mongosh", "--eval", "db.adminCommand('ping')"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 20s
    restart: unless-stopped

  # MongoDB initialization (runs once)
  mongodb-init:
    image: mongo:8.2
    container_name: mcp-mongodb-init
    depends_on:
      mongodb:
        condition: service_healthy
    environment:
      - DOCUMENTDB_HOST=mongodb
      - DOCUMENTDB_PORT=27017
      - DOCUMENTDB_DATABASE=${DOCUMENTDB_DATABASE:-mcp_registry}
      - DOCUMENTDB_NAMESPACE=${DOCUMENTDB_NAMESPACE:-default}
    volumes:
      - ./scripts/init-mongodb.sh:/init-mongodb.sh:ro
    entrypoint: ["/bin/bash", "/init-mongodb.sh"]
    restart: "no"

volumes:
  mongodb-data:
  mongodb-config:
```

### Terraform (DocumentDB)

**File:** `terraform/aws-ecs/modules/documentdb/variables.tf`

```hcl
variable "cluster_name" {
  description = "DocumentDB cluster name"
  type        = string
}

variable "instance_class" {
  description = "Instance class (e.g., db.r5.large)"
  type        = string
  default     = "db.r5.large"
}

variable "instance_count" {
  description = "Number of instances (1 writer + N readers)"
  type        = number
  default     = 3  # 1 writer + 2 readers
}

variable "master_username" {
  description = "Master username"
  type        = string
  sensitive   = true
}

variable "master_password" {
  description = "Master password"
  type        = string
  sensitive   = true
}

variable "backup_retention_period" {
  description = "Backup retention in days"
  type        = number
  default     = 7
}
```

---

## Migration Strategy

### From File Backend to MongoDB CE

**Scenario:** Moving from JSON files to MongoDB for local development

**Steps:**

1. **Export existing data**
   ```bash
   # Servers
   cp ~/mcp-gateway/servers/*.json /tmp/servers-backup/

   # Agents
   cp ~/mcp-gateway/agents/*.json /tmp/agents-backup/
   ```

2. **Update configuration**
   ```bash
   # In .env
   sed -i 's/STORAGE_BACKEND=file/STORAGE_BACKEND=mongodb-ce/' .env
   ```

3. **Start MongoDB**
   ```bash
   docker compose up -d mongodb
   docker compose up mongodb-init
   ```

4. **Import data via API**
   ```bash
   # Re-register servers
   for file in /tmp/servers-backup/*.json; do
       curl -X POST http://localhost/servers \
           -H "Content-Type: application/json" \
           -d @"$file"
   done

   # Re-register agents
   for file in /tmp/agents-backup/*.json; do
       curl -X POST http://localhost/agents \
           -H "Content-Type: application/json" \
           -d @"$file"
   done
   ```

5. **Verify**
   ```bash
   # Check server count
   curl http://localhost/servers | jq 'length'

   # Test search
   curl "http://localhost/search?q=financial" | jq '.servers | length'
   ```

### From MongoDB CE to AWS DocumentDB

**Scenario:** Moving from local development to production

**Steps:**

1. **Export from MongoDB CE**
   ```bash
   # Dump all collections
   docker exec mcp-mongodb mongodump \
       --db=mcp_registry \
       --out=/tmp/mongodb-backup

   # Copy backup from container
   docker cp mcp-mongodb:/tmp/mongodb-backup ./mongodb-backup
   ```

2. **Deploy DocumentDB with Terraform**
   ```bash
   cd terraform/aws-ecs
   terraform apply

   # Get DocumentDB endpoint
   terraform output documentdb_endpoint
   ```

3. **Import to DocumentDB**
   ```bash
   # From bastion host or ECS task
   mongorestore \
       --host=mcp-registry.cluster-xxxxx.us-east-1.docdb.amazonaws.com:27017 \
       --ssl \
       --sslCAFile=/app/global-bundle.pem \
       --username=mcp_admin \
       --password=<password> \
       --db=mcp_registry \
       ./mongodb-backup/mcp_registry
   ```

4. **Update application configuration**
   ```bash
   # In .env
   STORAGE_BACKEND=documentdb
   DOCUMENTDB_HOST=mcp-registry.cluster-xxxxx.us-east-1.docdb.amazonaws.com
   DOCUMENTDB_USERNAME=mcp_admin
   DOCUMENTDB_PASSWORD=<password>
   DOCUMENTDB_USE_TLS=true
   DOCUMENTDB_TLS_CA_FILE=/app/global-bundle.pem
   ```

5. **Deploy application**
   ```bash
   terraform apply
   ```

6. **Verify vector search**
   ```bash
   # Test search endpoint
   curl "https://api.example.com/search?q=financial" | jq '.servers'

   # Should return results with relevance_score
   # DocumentDB will use native HNSW index (faster!)
   ```

---

## Summary

### Architecture Highlights

| Component | MongoDB CE | AWS DocumentDB |
|-----------|------------|----------------|
| **Container** | mongo:8.2 | Managed Service |
| **Connection** | mongodb://mongodb:27017 | mongodb://cluster.docdb.amazonaws.com:27017 |
| **Authentication** | None (local) | Username/Password or IAM |
| **TLS** | Disabled | Required |
| **Vector Search** | App-level (Python) | Native (HNSW) |
| **Latency** | 50-200ms | 10-50ms |
| **Max Scale** | ~10,000 docs | Millions |
| **Cost** | Free | AWS pricing |

### Key Takeaways

1. **Same Code, Different Backends**
   - Identical repository implementation
   - Only connection configuration differs
   - Seamless migration path

2. **Vector Search Strategy**
   - MongoDB CE: Application-level for dev simplicity
   - DocumentDB: Native HNSW for production performance
   - Both use cosine similarity metric

3. **Development Workflow**
   - Local dev: `STORAGE_BACKEND=mongodb-ce`
   - Production: `STORAGE_BACKEND=documentdb`
   - Terraform handles infrastructure

4. **No Terraform Changes**
   - AWS DocumentDB infrastructure deployed via Terraform
   - Local MongoDB CE runs in Docker Compose
   - Terraform only manages AWS resources

---

## See Also

- [Database Abstraction Layer Design](./database-abstraction-layer.md)
- [Embeddings Configuration](../embeddings.md)
- [Configuration Guide](../configuration.md)
- [MongoDB Documentation](https://www.mongodb.com/docs/manual/)
- [AWS DocumentDB Documentation](https://docs.aws.amazon.com/documentdb/)
