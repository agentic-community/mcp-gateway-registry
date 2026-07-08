# Database Abstraction Layer Design

**Status:** Current Implementation
**Last Updated:** January 5, 2026
**Target Audience:** Senior/Staff Engineers, Architecture Review

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Design Patterns](#design-patterns)
3. [Abstract Base Classes](#abstract-base-classes)
4. [Implementations](#implementations)
5. [Factory Pattern](#factory-pattern)
6. [Design Rationale](#design-rationale)
7. [Code Organization](#code-organization)

## Architecture Overview

The MCP Gateway Registry implements a **repository pattern with abstract base classes** to provide a clean separation between business logic and data storage. This design enables seamless switching between file-based storage (legacy, backwards compatible) and DocumentDB/MongoDB (recommended) without modifying application code.

### System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                       Application Layer                          │
│  (Services, API Endpoints, Business Logic)                      │
└────────────────────┬────────────────────────────────────────────┘
                     │ depends on
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              Repository Factory Layer                            │
│  get_server_repository()                                        │
│  get_agent_repository()                                         │
│  get_scope_repository()                                         │
│  get_security_scan_repository()                                 │
│  get_search_repository()                                        │
│  get_federation_config_repository()                             │
└────────┬──────────────────────────────┬────────────────────────┘
         │                              │
         │ creates (based on           │ creates (based on
         │ STORAGE_BACKEND)             │ STORAGE_BACKEND)
         ▼                              ▼
┌──────────────────────┐      ┌──────────────────────────┐
│ File-Based Impl      │      │ DocumentDB Impl          │
├──────────────────────┤      ├──────────────────────────┤
│ FileServerRepo       │      │ DocumentDBServerRepo     │
│ FileAgentRepo        │      │ DocumentDBAgentRepo      │
│ FileScopeRepo        │      │ DocumentDBScopeRepo      │
│ FileSecurityScanRepo │      │ DocumentDBSecurityRepo   │
│ FileSearchRepo       │      │ DocumentDBSearchRepo     │
│ FileFederationRepo   │      │ DocumentDBFederationRepo │
└──────────┬───────────┘      └──────────┬───────────────┘
           │                             │
           ▼                             ▼
    Local File System         DocumentDB/MongoDB Cluster
    (JSON files)               (Collections, Vector Search)
```

### Key Principles

1. **Abstraction**: All repositories implement abstract base classes defining strict contracts
2. **Polymorphism**: Multiple storage backend implementations (file, DocumentDB/MongoDB)
3. **Dependency Injection**: Factory pattern provides single point of repository creation
4. **Consistency**: All implementations provide identical behavior regardless of backend
5. **Testability**: Mock implementations easy to create; reset_repositories() enables test isolation

## Design Patterns

### Repository Pattern

The repository pattern provides a **data access abstraction** layer that:

- Isolates business logic from storage implementation details
- Defines clear, intent-driven interfaces (e.g., `get()`, `create()`, `update()`, `delete()`)
- Makes switching storage backends transparent to consuming code
- Enables comprehensive testing without actual storage dependencies

### Factory Pattern

The factory pattern manages **repository instantiation** by:

- Creating repositories lazily on first access (singleton pattern)
- Selecting implementation based on environment configuration (`STORAGE_BACKEND` setting)
- Centralizing backend selection logic in one place (`factory.py`)
- Providing `reset_repositories()` for test isolation

### Strategy Pattern

Different storage backends are **strategies** implementing the same interface:

- **File Strategy**: YAML/JSON files with application-level search
- **DocumentDB/MongoDB Strategy**: Distributed document database with vector search and aggregation pipelines

## Abstract Base Classes

All repository implementations inherit from abstract base classes defined in [`registry/repositories/interfaces.py`](../../registry/repositories/interfaces.py). These define the contract that ALL implementations must follow.

### ServerRepositoryBase

**Location:** [`registry/repositories/interfaces.py` (lines 14-76)](../../registry/repositories/interfaces.py#L14-L76)

**Purpose:** Data access for MCP server definitions and lifecycle management

**Key Methods:**

```python
# Lifecycle operations
async def get(path: str) -> Optional[Dict[str, Any]]
async def list_all() -> Dict[str, Dict[str, Any]]
async def create(server_info: Dict[str, Any]) -> bool
async def update(path: str, server_info: Dict[str, Any]) -> bool
async def delete(path: str) -> bool

# State management (enabled/disabled)
async def get_state(path: str) -> bool
async def set_state(path: str, enabled: bool) -> bool

# Lifecycle
async def load_all() -> None  # Load/reload from storage at startup
```

**Implementations:**
- `registry/repositories/file/server_repository.py` - File-based
- [`registry/repositories/documentdb/server_repository.py`](../../registry/repositories/documentdb/server_repository.py) - DocumentDB/MongoDB

---

### AgentRepositoryBase

**Location:** [`registry/repositories/interfaces.py` (lines 78-140)](../../registry/repositories/interfaces.py#L78-L140)

**Purpose:** Data access for A2A (Agent-to-Agent) agent definitions

**Key Methods:**

```python
async def get(path: str) -> Optional[AgentCard]
async def list_all() -> List[AgentCard]
async def create(agent: AgentCard) -> AgentCard
async def update(path: str, updates: Dict[str, Any]) -> AgentCard
async def delete(path: str) -> bool

async def get_state(path: str) -> bool
async def set_state(path: str, enabled: bool) -> bool

async def load_all() -> None  # Load/reload from storage
```

**Implementations:**
- `registry/repositories/file/agent_repository.py` - File-based
- [`registry/repositories/documentdb/agent_repository.py`](../../registry/repositories/documentdb/agent_repository.py) - DocumentDB/MongoDB

---

### ScopeRepositoryBase

**Location:** [`registry/repositories/interfaces.py` (lines 142-506)](../../registry/repositories/interfaces.py#L142-L506)

**Purpose:** Data access for authorization scopes (RBAC - Role-Based Access Control)

**Overview:**

The Scope Repository manages a complex authorization model with three primary data structures:

1. **UI-Scopes** - Permissions for UI features grouped by Keycloak group
2. **Server Scopes** - Server access rules (which servers, methods, tools)
3. **Group Mappings** - Mapping between Keycloak groups and scopes

**Key Methods:**

```python
# UI permission queries
async def get_ui_scopes(group_name: str) -> Dict[str, Any]
async def get_group_mappings(keycloak_group: str) -> List[str]

# Server access rules
async def get_server_scopes(scope_name: str) -> List[Dict[str, Any]]

# Group management
async def get_group(group_name: str) -> Dict[str, Any]
async def list_groups() -> Dict[str, Any]
async def group_exists(group_name: str) -> bool
async def create_group(group_name: str, description: str = "") -> bool
async def delete_group(group_name: str, remove_from_mappings: bool = True) -> bool

# Server scope assignment
async def add_server_scope(
    server_path: str,
    scope_name: str,
    methods: List[str],
    tools: Optional[List[str]] = None
) -> bool
async def remove_server_scope(server_path: str, scope_name: str) -> bool
async def remove_server_from_all_scopes(server_path: str) -> bool

# UI visibility management
async def add_server_to_ui_scopes(group_name: str, server_name: str) -> bool
async def remove_server_from_ui_scopes(group_name: str, server_name: str) -> bool

# Scope mapping management
async def add_group_mapping(group_name: str, scope_name: str) -> bool
async def remove_group_mapping(group_name: str, scope_name: str) -> bool
async def get_all_group_mappings() -> Dict[str, List[str]]

# Bulk operations
async def add_server_to_multiple_scopes(
    server_path: str,
    scope_names: List[str],
    methods: List[str],
    tools: List[str]
) -> bool

async def load_all() -> None  # Load/reload from storage
```

**Implementations:**
- `registry/repositories/file/scope_repository.py` - File-based (YAML)
- [`registry/repositories/documentdb/scope_repository.py`](../../registry/repositories/documentdb/scope_repository.py) - DocumentDB/MongoDB

---

### SecurityScanRepositoryBase

**Location:** [`registry/repositories/interfaces.py` (lines 508-598)](../../registry/repositories/interfaces.py#L508-L598)

**Purpose:** Data access for security scanning results (both MCP servers and agents)

**Key Methods:**

```python
# CRUD operations
async def get(server_path: str) -> Optional[Dict[str, Any]]
async def list_all() -> List[Dict[str, Any]]
async def create(scan_result: Dict[str, Any]) -> bool

# Querying
async def get_latest(server_path: str) -> Optional[Dict[str, Any]]
async def query_by_status(status: str) -> List[Dict[str, Any]]

async def load_all() -> None  # Load/reload from storage
```

**Implementations:**
- `registry/repositories/file/security_scan_repository.py` - File-based
- [`registry/repositories/documentdb/security_scan_repository.py`](../../registry/repositories/documentdb/security_scan_repository.py) - DocumentDB/MongoDB

---

### SearchRepositoryBase

**Location:** [`registry/repositories/interfaces.py` (lines 600-645)](../../registry/repositories/interfaces.py#L600-L645)

**Purpose:** Data access for semantic/hybrid search functionality

**Key Methods:**

```python
async def initialize() -> None  # Initialize search service

# Indexing
async def index_server(
    path: str,
    server_info: Dict[str, Any],
    is_enabled: bool = False
) -> None

async def index_agent(
    path: str,
    agent_card: AgentCard,
    is_enabled: bool = False
) -> None

# Query
async def search(
    query: str,
    entity_types: Optional[List[str]] = None,
    max_results: int = 10
) -> Dict[str, List[Dict[str, Any]]]

async def remove_entity(path: str) -> None
```

**Implementations:**
- `registry/repositories/file/search_repository.py` - Application-level vector search (legacy)
- [`registry/repositories/documentdb/search_repository.py`](../../registry/repositories/documentdb/search_repository.py) - DocumentDB/MongoDB Hybrid Search (BM25 + k-NN)

---

### FederationConfigRepositoryBase

**Location:** [`registry/repositories/interfaces.py` (lines 647-709)](../../registry/repositories/interfaces.py#L647-L709)

**Purpose:** Data access for federation configuration (multi-registry federation)

**Key Methods:**

```python
# CRUD operations
async def get_config(config_id: str = "default") -> Optional[FederationConfig]
async def save_config(
    config: FederationConfig,
    config_id: str = "default"
) -> FederationConfig

async def delete_config(config_id: str = "default") -> bool
async def list_configs() -> List[Dict[str, Any]]
```

**Implementations:**
- `registry/repositories/file/federation_config_repository.py` - File-based (JSON)
- [`registry/repositories/documentdb/federation_config_repository.py`](../../registry/repositories/documentdb/federation_config_repository.py) - DocumentDB/MongoDB

---

## Implementations

### File-Based Backend (Legacy, Backwards Compatible)

**Purpose:** Local file system storage for development, testing, and backwards compatibility

**Characteristics:**
- **Simplicity**: JSON/YAML files, no external dependencies
- **Isolation**: No network calls required
- **State Management**: Separate state files for enabled/disabled status
- **Search**: Application-level vector search

**Storage Structure:**
```
registry/
├── servers/
│   ├── server1.json
│   ├── server2.json
│   └── server_state.json          # Persistence for enabled/disabled state
├── agents/
│   ├── agent1_agent.json
│   ├── agent2_agent.json
│   └── agent_state.json
├── config/
│   ├── scopes.yml                 # Authorization scopes (legacy; scopes.yml was removed in v1.24.8, scopes now live in the mcp_scopes collection in DocumentDB)
│   └── federation/
│       └── default.json           # Federation configuration
├── security_scans/
│   ├── scan_result_1.json
│   └── scan_result_2.json
├── models/
│   └── all-MiniLM-L6-v2/          # Embedding model weights
└── service_index.bin              # Vector search index
```

**Implementation Classes:**
- `FileServerRepository`
- `FileAgentRepository`
- `FileScopeRepository`
- `FileSecurityScanRepository`
- `FileSearchRepository`
- `FileFederationConfigRepository`

**Advantages:**
- No infrastructure setup needed
- Good for development and testing
- Human-readable file formats
- Git-friendly for version control

**Limitations:**
- Single-node only (no distributed deployment)
- Limited query capabilities
- File locking issues in concurrent scenarios
- Not suitable for production at scale
- Requires local model file storage for embeddings

---

### DocumentDB/MongoDB Backend (Recommended for Production)

**Purpose:** Distributed document database for production deployments

**Characteristics:**
- **Scalability**: Clustered deployment for redundancy (DocumentDB) or replica sets (MongoDB)
- **Query Capabilities**: Rich aggregation pipelines, complex filtering, projections
- **Vector Search**: Native vector search (DocumentDB) or application-level (MongoDB CE)
- **Collection Management**: Automatic index creation with proper field mappings
- **Async Support**: Full async/await implementation for non-blocking I/O
- **Strong Consistency**: ACID transactions and strong consistency guarantees

**Collection Structure:**
```
DocumentDB/MongoDB Cluster
├── mcp_servers_{namespace}         # Server definitions
├── mcp_agents_{namespace}          # Agent definitions
├── mcp_scopes_{namespace}          # Authorization scopes
├── mcp_security_scans_{namespace}  # Security scan results
├── mcp_embeddings_1536_{namespace} # Vector embeddings
└── mcp_federation_config_{namespace} # Federation configurations
```

**Implementation Classes:**
- [`DocumentDBServerRepository`](../../registry/repositories/documentdb/server_repository.py)
- [`DocumentDBAgentRepository`](../../registry/repositories/documentdb/agent_repository.py)
- [`DocumentDBScopeRepository`](../../registry/repositories/documentdb/scope_repository.py)
- [`DocumentDBSecurityScanRepository`](../../registry/repositories/documentdb/security_scan_repository.py)
- [`DocumentDBSearchRepository`](../../registry/repositories/documentdb/search_repository.py)
- [`DocumentDBFederationConfigRepository`](../../registry/repositories/documentdb/federation_config_repository.py)

**DocumentDB Client:** [`registry/repositories/documentdb/client.py`](../../registry/repositories/documentdb/client.py)

Provides:
- Async MongoDB client singleton
- Connection pooling and authentication
- Index name management with namespace support
- Client lifecycle management (initialization and cleanup)

**Advantages:**
- Distributed, highly available
- Rich query capabilities
- Hybrid search (BM25 + k-NN) for semantic understanding
- Native vector support for embeddings
- Scales to millions of documents
- Recommended for deployment

**Limitations:**
- Requires DocumentDB cluster or MongoDB CE instance
- Higher operational complexity
- More network I/O
- Requires proper index tuning for performance

**Hybrid Search Explanation:**

DocumentDB/MongoDB vector search provides semantic similarity:

1. **BM25 (Best Matching 25)**: Full-text relevance scoring based on term frequency
   - Good for exact keyword matches
   - Fast and traditional IR approach
   - Default web search behavior

2. **k-NN (k-Nearest Neighbors)**: Vector similarity search
   - Semantic understanding
   - Captures meaning, not just keywords
   - Uses neural embeddings (sentence-transformers or Bedrock Titan)

**Default Weighting:**
```python
# Legacy - removed: float = 0.4
# Legacy - removed: float = 0.6
```

This gives 60% weight to semantic search (vector similarity) and 40% to keyword matching, optimized for finding semantically relevant servers/agents.

---

## Factory Pattern

### Factory Implementation

**Location:** [`registry/repositories/factory.py`](../../registry/repositories/factory.py)

The factory provides singleton repository instances with lazy initialization:

```python
def get_server_repository() -> ServerRepositoryBase:
    """Get server repository singleton."""
    global _server_repo

    if _server_repo is not None:
        return _server_repo

    backend = settings.storage_backend
    logger.info(f"Creating server repository with backend: {backend}")

    if backend == "documentdb":
        from .documentdb.server_repository import DocumentDBServerRepository
        _server_repo = DocumentDBServerRepository()
    else:
        from .file.server_repository import FileServerRepository
        _server_repo = FileServerRepository()

    return _server_repo
```

**Similar factory functions exist for:**
- `get_agent_repository()`
- `get_scope_repository()`
- `get_security_scan_repository()`
- `get_search_repository()`
- `get_federation_config_repository()`

### Backend Selection Logic

Backend selection is controlled by the `storage_backend` setting in [`registry/core/config.py`](../../registry/core/config.py):

```python
storage_backend: str = "file"  # Options: "file", "documentdb"
```

**How it works:**

1. **At startup**: Application reads `STORAGE_BACKEND` environment variable (or defaults to "file")
2. **On first access**: Factory function checks backend setting
3. **Instance creation**: Creates appropriate implementation (file vs DocumentDB/MongoDB)
4. **Caching**: Returns cached singleton on subsequent calls

### Dependency Injection Pattern

Services consume repositories through factory functions:

```python
# In any service module
from registry.repositories.factory import get_server_repository

async def list_servers():
    repo = get_server_repository()  # Gets file or DocumentDB/MongoDB impl
    servers = await repo.list_all()
    return servers
```

**Benefits:**
- No hardcoded dependencies
- Easy to swap implementations
- Test isolation via `reset_repositories()`

### Test Isolation

For testing, repositories can be reset:

```python
from registry.repositories.factory import reset_repositories

async def test_server_creation():
    reset_repositories()  # Clear all singletons
    # Now fresh repositories are created
    repo = get_server_repository()
    # ... test code ...
```

---

## Design Rationale

### Why Repository Pattern?

The repository pattern addresses several critical concerns:

1. **Separation of Concerns**: Business logic doesn't know about storage details
   - Controllers/services call `repo.get()`, not filesystem or database directly
   - Easier to test services in isolation

2. **Multiple Storage Backends**: Single interface, multiple implementations
   - Same code works with file or DocumentDB/MongoDB backend
   - Switching backends requires only env var change
   - No code changes needed

3. **Consistency**: All implementations follow same contract
   - Same method signatures and behavior
   - Predictable error handling
   - Consistent data transformations

4. **Abstraction**: Storage complexity hidden behind simple interface
   - `repo.get(path)` is simple whether backed by files or DocumentDB/MongoDB
   - Consumers don't care about implementation details

### Benefits of Repository Abstraction

**For Development:**
- Develop with file backend (fast, no setup)
- Test with mocks (no I/O overhead)
- Easy to add new storage backends

**For Deployment:**
- Production uses DocumentDB/MongoDB (scalable)
- Backwards compatible with file storage
- Can migrate gradually

**For Testing:**
- Mock repositories easily
- Test services without actual storage
- Test data isolation via `reset_repositories()`

**For Maintenance:**
- Storage logic centralized in repository classes
- Changes to storage don't affect business logic
- Clear interfaces make code changes safer

### Interface Consistency

All repositories implement abstract base classes with:

```python
# All repos have these patterns
async def load_all() -> None          # Startup initialization
async def get(id: str) -> Optional    # Single entity retrieval
async def list_all() -> List/Dict     # Bulk retrieval
async def create(entity) -> bool/Entity  # Create new entity
async def update(id, updates) -> bool/Entity  # Modify existing
async def delete(id) -> bool          # Remove entity
```

This consistent interface means:
- Easier onboarding for engineers
- Less mental overhead switching between repositories
- Standardized error handling

### Data Consistency Strategy

**File Backend:**
- Atomic file operations
- Separate state file for enabled/disabled status
- No transactions (file-by-file consistency)
- Best effort on concurrent writes

**DocumentDB/MongoDB Backend:**
- Document-level consistency
- Index operations with refresh flags
- No distributed transactions
- Eventual consistency model
- Suitable for high concurrency

---

## Code Organization

### Directory Structure

```
registry/repositories/
├── __init__.py                          # Package initialization
├── interfaces.py                        # Abstract base classes (6 repos)
├── factory.py                           # Repository factory
│
├── file/                                # File-based implementations
│   ├── __init__.py
│   ├── server_repository.py
│   ├── agent_repository.py
│   ├── scope_repository.py
│   ├── security_scan_repository.py
│   ├── search_repository.py             # Application-level vector search
│   └── federation_config_repository.py
│
└── documentdb/                          # DocumentDB/MongoDB implementations
    ├── __init__.py
    ├── client.py                        # MongoDB client management
    ├── server_repository.py
    ├── agent_repository.py
    ├── scope_repository.py
    ├── security_scan_repository.py
    ├── search_repository.py             # Vector search (native or app-level)
    └── federation_config_repository.py
```

### Naming Conventions

**Abstract Base Classes** (in `interfaces.py`):
- `ServerRepositoryBase`
- `AgentRepositoryBase`
- `ScopeRepositoryBase`
- `SecurityScanRepositoryBase`
- `SearchRepositoryBase`
- `FederationConfigRepositoryBase`

**File Implementations** (in `file/`):
- `FileServerRepository`
- `FileAgentRepository`
- `FileScopeRepository`
- `FileSecurityScanRepository`
- `FileSearchRepository` (application-level vector search)
- `FileFederationConfigRepository`

**DocumentDB/MongoDB Implementations** (in `documentdb/`):
- `DocumentDBServerRepository`
- `DocumentDBAgentRepository`
- `DocumentDBScopeRepository`
- `DocumentDBSecurityScanRepository`
- `DocumentDBSearchRepository`
- `DocumentDBFederationConfigRepository`

### Configuration

Backend selection and DocumentDB/MongoDB settings in [`registry/core/config.py`](../../registry/core/config.py):

```python
# Backend selection
storage_backend: str = "file"  # "file", "mongodb", or "documentdb"

# DocumentDB/MongoDB connection
documentdb_endpoint: str = "localhost"
documentdb_port: int = 27017
documentdb_username: Optional[str] = None
documentdb_password: Optional[str] = None
documentdb_database: str = "mcp_gateway"
documentdb_use_tls: bool = False
documentdb_tls_ca_file: Optional[str] = None

# Multi-tenancy support
documentdb_namespace: str = "default"

# Collection names (with namespace suffix)
documentdb_collection_servers: str = "mcp_servers"
documentdb_collection_agents: str = "mcp_agents"
documentdb_collection_scopes: str = "mcp_scopes"
documentdb_collection_embeddings: str = "mcp_embeddings_1536"
documentdb_collection_security_scans: str = "mcp_security_scans"
documentdb_collection_federation_config: str = "mcp_federation_config"

# Vector search configuration
documentdb_vector_dimension: int = 1536  # For embeddings
```

---

## Advanced Topics

### Namespace Support (Multi-Tenancy)

DocumentDB/MongoDB implementation supports multi-tenancy via namespaces:

```python
# In config.py
documentdb_namespace: str = "default"

# Collection names are automatically namespaced
# mcp_servers_{namespace}
# mcp_agents_{namespace}
# etc.
```

**Use Cases:**
- Multiple registry instances on shared DocumentDB/MongoDB cluster
- Isolated test environments
- Customer separation in SaaS deployments

**Implementation:** [`registry/repositories/documentdb/client.py`](../../registry/repositories/documentdb/client.py)

```python
def get_index_name(base_name: str) -> str:
    """Get full index name with namespace."""
    return f"{base_name}-{settings.documentdb_namespace}"
```

### Error Handling

Both implementations handle errors gracefully:

**File Backend:**
- Missing files → return None or empty list
- Parse errors → log and skip
- I/O errors → raised to caller

**DocumentDB/MongoDB Backend:**
- Connection errors → log and attempt retry
- Index not found → initialize index
- Query errors → log and raise

### Performance Considerations

**File Backend:**
- Fast for small datasets (<1000 entities)
- No network latency
- Vector search: O(n) linear scan (not scalable)
- Not suitable for high concurrency

**DocumentDB/MongoDB Backend:**
- Scalable to millions of entities
- Network latency (typically <100ms)
- BM25 + k-NN: O(log n) with proper indexing
- Built for high concurrency (lock-free reads)

**Recommendations:**
- Use file backend for development only
- Use DocumentDB/MongoDB for production
- Monitor DocumentDB/MongoDB query latency
- Tune index refresh intervals for throughput vs. latency tradeoff

---

## Testing Strategy

### Unit Tests

Mock repositories for unit testing:

```python
class MockServerRepository(ServerRepositoryBase):
    """Mock implementation for testing."""

    def __init__(self):
        self.servers = {}

    async def get(self, path: str) -> Optional[Dict]:
        return self.servers.get(path)

    async def list_all(self) -> Dict:
        return self.servers.copy()

    # ... implement other methods ...
```

### Integration Tests

Use file backend for integration tests:

```python
@pytest.fixture
def reset_repos():
    """Reset repositories between tests."""
    yield
    reset_repositories()

async def test_server_lifecycle(reset_repos):
    repo = get_server_repository()

    # File backend used by default
    server = {"path": "/test", "name": "test_server"}
    await repo.create(server)

    result = await repo.get("/test")
    assert result is not None
```

### End-to-End Tests

Test against MongoDB CE in Docker:

```yaml
# docker-compose.yml
services:
  mongodb:
    image: mongo:8.2
    command: ["--replSet", "rs0", "--bind_ip_all"]
    ports:
      - "27017:27017"
```

Set `STORAGE_BACKEND=mongodb` and run full integration tests.

---

## Summary

The database abstraction layer provides a **clean, extensible architecture** for data storage in the MCP Gateway Registry:

| Aspect | File | DocumentDB/MongoDB |
|--------|------|-------------------|
| **Use Case** | Development, testing | Production |
| **Scalability** | ~1000 entities | Millions |
| **Dependencies** | None | DocumentDB/MongoDB cluster |
| **Query Power** | Basic | Advanced (aggregation pipelines) |
| **Concurrency** | Limited | High |
| **Setup Complexity** | None | Moderate |
| **Cost** | Free | Infrastructure cost |

The **repository pattern** with factory ensures:
- Clean separation of concerns
- Easy backend switching
- Comprehensive testability
- Consistent interfaces
- Future extensibility

All implementations maintain **identical behavior**, making backend selection purely an operational decision rather than a code architecture choice.

