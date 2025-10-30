# Design Document - Test Suite Modernization

## Overview

This design document outlines the technical approach for modernizing the test suite to align with the current architecture, new authentication system (Keycloak/OAuth2), and recent feature additions. The modernization will achieve 80% code coverage, ensure all tests pass, and provide comprehensive testing infrastructure for future development.

## Architecture

### Test Organization Structure

```
tests/
├── conftest.py                              # Shared fixtures and configuration
├── pytest.ini                               # Pytest configuration with markers
├── fixtures/
│   ├── factories.py                         # Test data factories
│   ├── auth_fixtures.py                     # Authentication fixtures
│   └── servers/                             # Sample server configurations
├── unit/                                    # Unit tests (isolated components)
│   ├── auth/
│   ├── servers/
│   ├── services/
│   ├── search/
│   ├── health/
│   ├── core/
│   ├── utils/                               # NEW: Utils tests
│   └── cli/                                 # NEW: CLI tests
├── integration/                             # Integration tests (component interaction)
│   ├── test_server_routes.py
│   ├── test_server_lifecycle.py            # NEW
│   ├── test_authentication_flow.py         # NEW
│   └── test_internal_api.py                # NEW
├── e2e/                                     # End-to-end tests (complete workflows)
│   └── test_complete_workflows.py
└── reporting/                               # Test reporting system
    ├── coverage_analyzer.py
    ├── dashboard_generator.py
    └── report_orchestrator.py
```

### Testing Layers

1. **Unit Tests**: Test individual functions/classes in isolation with mocked dependencies
2. **Integration Tests**: Test multiple components working together with minimal mocking
3. **E2E Tests**: Test complete user workflows from start to finish

## Components and Interfaces

### 1. Test Infrastructure (conftest.py)

#### Authentication Fixtures

```python
@pytest.fixture
def mock_keycloak_user_context() -> dict:
    """Mock user context from Keycloak authentication."""
    return {
        "username": "testuser",
        "is_admin": False,
        "groups": ["mcp-servers-unrestricted"],
        "scopes": [
            "mcp-servers-unrestricted/read",
            "mcp-servers-unrestricted/execute"
        ],
        "accessible_servers": ["currenttime", "mcpgw"],
        "accessible_services": ["all"],
        "ui_permissions": {
            "toggle_service": ["all"],
            "modify_service": ["all"],
            "register_service": ["all"],
            "health_check_service": ["all"]
        }
    }

@pytest.fixture
def mock_admin_user_context() -> dict:
    """Mock admin user context with full permissions."""
    return {
        "username": "admin",
        "is_admin": True,
        "groups": ["mcp-servers-unrestricted", "admins"],
        "scopes": ["mcp-servers-unrestricted/read", "mcp-servers-unrestricted/execute"],
        "accessible_servers": ["all"],
        "accessible_services": ["all"],
        "ui_permissions": {
            "toggle_service": ["all"],
            "modify_service": ["all"],
            "register_service": ["all"],
            "health_check_service": ["all"]
        }
    }

@pytest.fixture
def mock_m2m_token() -> str:
    """Mock M2M JWT token for agent authentication."""
    import jwt
    from datetime import datetime, timedelta
    
    payload = {
        "sub": "agent-test-m2m",
        "scope": "mcp-servers-unrestricted/read mcp-servers-unrestricted/execute",
        "exp": datetime.utcnow() + timedelta(hours=1),
        "iat": datetime.utcnow(),
        "client_id": "agent-test-m2m"
    }
    return jwt.encode(payload, "test-secret", algorithm="HS256")

@pytest.fixture
def mock_enhanced_auth(monkeypatch, mock_keycloak_user_context):
    """Mock enhanced_auth dependency."""
    def mock_auth(session=None, authorization=None):
        return mock_keycloak_user_context
    
    monkeypatch.setattr("registry.auth.dependencies.enhanced_auth", mock_auth)
    return mock_auth
```

#### Test Client Fixtures

```python
@pytest.fixture
def test_client():
    """FastAPI test client with app instance."""
    from fastapi.testclient import TestClient
    from registry.main import app
    return TestClient(app)

@pytest.fixture
def authenticated_client(test_client, mock_enhanced_auth):
    """Test client with authenticated user context."""
    return test_client
```

### 2. Test Data Factories (factories.py)

#### Server Metadata Factory

```python
import factory
from faker import Faker

fake = Faker()

class ServerMetadataFactory(factory.Factory):
    """Factory for generating realistic server metadata."""
    
    class Meta:
        model = dict
    
    name = factory.LazyAttribute(lambda _: fake.slug())
    display_name = factory.LazyAttribute(lambda _: fake.company())
    description = factory.LazyAttribute(lambda _: fake.text(max_nb_chars=200))
    version = "1.0.0"
    enabled = True
    proxy_pass = factory.LazyAttribute(lambda o: f"http://localhost:8000/{o.name}")
    
    @factory.lazy_attribute
    def tools(self):
        return [
            {
                "name": f"{fake.word()}_tool",
                "description": fake.sentence(),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string", "description": fake.sentence()}
                    }
                }
            }
            for _ in range(3)
        ]
```

#### User Context Factory

```python
class UserContextFactory(factory.Factory):
    """Factory for generating user contexts."""
    
    class Meta:
        model = dict
    
    username = factory.LazyAttribute(lambda _: fake.user_name())
    is_admin = False
    groups = factory.List([factory.LazyAttribute(lambda _: f"group-{fake.word()}")])
    scopes = factory.List([factory.LazyAttribute(lambda _: f"scope-{fake.word()}/read")])
    accessible_servers = factory.List([factory.LazyAttribute(lambda _: fake.slug())])
    accessible_services = ["all"]
    
    @factory.lazy_attribute
    def ui_permissions(self):
        return {
            "toggle_service": self.accessible_services,
            "modify_service": self.accessible_services,
            "register_service": self.accessible_services,
            "health_check_service": self.accessible_services
        }
```

### 3. FAISS Service Mocking Strategy

**Problem**: Direct mocking of FAISS internals causes segmentation faults.

**Solution**: Use test doubles and in-memory indices.

```python
@pytest.fixture
def mock_faiss_service(monkeypatch):
    """Mock FAISS service with in-memory index."""
    
    class MockFAISSService:
        def __init__(self):
            self.index_data = {}  # Simple dict instead of real FAISS index
            self.metadata = {}
        
        async def add_or_update_service(self, service_name: str, metadata: dict):
            self.index_data[service_name] = metadata
            self.metadata[service_name] = metadata
        
        async def search(self, query: str, k: int = 5):
            # Simple keyword matching instead of vector search
            results = []
            for name, meta in self.metadata.items():
                if query.lower() in str(meta).lower():
                    results.append({"name": name, "metadata": meta, "score": 0.9})
            return results[:k]
        
        async def remove_service(self, service_name: str):
            self.index_data.pop(service_name, None)
            self.metadata.pop(service_name, None)
    
    mock_service = MockFAISSService()
    monkeypatch.setattr("registry.search.service.FAISSService", lambda: mock_service)
    return mock_service
```

### 4. External Service Mocking

#### Keycloak OAuth Mocking

```python
@pytest.fixture
def mock_keycloak_oauth(monkeypatch):
    """Mock Keycloak OAuth endpoints."""
    
    async def mock_get_providers():
        return [
            {"name": "google", "display_name": "Google"},
            {"name": "github", "display_name": "GitHub"}
        ]
    
    monkeypatch.setattr(
        "registry.auth.routes.get_oauth2_providers",
        mock_get_providers
    )
```

#### MCP Server Mocking

```python
@pytest.fixture
def mock_mcp_server():
    """Mock MCP server responses."""
    
    class MockMCPServer:
        async def list_tools(self):
            return {
                "tools": [
                    {
                        "name": "test_tool",
                        "description": "A test tool",
                        "inputSchema": {"type": "object", "properties": {}}
                    }
                ]
            }
        
        async def call_tool(self, tool_name: str, arguments: dict):
            return {"result": "success", "data": arguments}
    
    return MockMCPServer()
```

## Data Models

### Test Configuration Model

```python
from pydantic import BaseModel

class TestConfig(BaseModel):
    """Configuration for test execution."""
    
    coverage_threshold: float = 80.0
    critical_path_threshold: float = 95.0
    new_feature_threshold: float = 90.0
    test_timeout: int = 60
    parallel_workers: int = 4
    markers: list[str] = ["unit", "integration", "e2e"]
```

### Test Result Model

```python
class TestResult(BaseModel):
    """Model for test execution results."""
    
    total_tests: int
    passed: int
    failed: int
    skipped: int
    duration: float
    coverage_percent: float
    critical_path_coverage: float
```

## Error Handling

### Test Failure Patterns

1. **Import Errors**: Catch and report missing dependencies or modules
2. **Fixture Errors**: Provide clear messages when fixtures fail to initialize
3. **Assertion Errors**: Use descriptive assertion messages with context
4. **Timeout Errors**: Set reasonable timeouts and provide debugging info

### Error Reporting

```python
def pytest_runtest_makereport(item, call):
    """Custom test report with enhanced error information."""
    if call.excinfo is not None:
        # Add context to error reports
        item.add_report_section(
            "call",
            "error_context",
            f"Test: {item.nodeid}\nPhase: {call.when}\nDuration: {call.duration}s"
        )
```

## Testing Strategy

### Phase 1: Infrastructure Setup

1. Update `conftest.py` with new fixtures
2. Create `auth_fixtures.py` for authentication mocks
3. Update `factories.py` with realistic data generators
4. Configure pytest markers in `pytest.ini`

### Phase 2: Critical Path Tests

1. **Authentication Tests**
   - OAuth2 login flow
   - JWT token validation
   - Session management
   - Permission extraction

2. **Server Service Tests**
   - Server registration
   - Permission filtering
   - Server removal with cleanup

3. **Scopes Manager Tests**
   - Add/remove server to scopes
   - Group management
   - YAML file operations

### Phase 3: New Feature Tests

1. **CLI Tools Tests**
   - Command execution
   - M2M authentication
   - Error handling

2. **Internal API Tests**
   - Registration endpoints
   - Management operations
   - Authentication variants

3. **MCP Gateway Tests**
   - Service management tools
   - Health checking
   - Intelligent tool finder

### Phase 4: Integration Tests

1. **Server Lifecycle**
   - Complete registration flow
   - Deletion with cleanup
   - Toggle with health checks

2. **Authentication Flow**
   - OAuth login to session
   - M2M token flow
   - Permission-based access

### Phase 5: E2E Tests

1. **Complete Workflows**
   - New server to first call
   - User permission restrictions
   - Multi-user scenarios

## CI/CD Integration

### GitHub Actions Workflow

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install uv
          uv sync --extra dev
      
      - name: Run tests with coverage
        run: |
          uv run pytest --cov=registry --cov-report=xml --cov-report=html
      
      - name: Check coverage threshold
        run: |
          uv run coverage report --fail-under=80
      
      - name: Upload coverage reports
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

### Coverage Enforcement

- Overall coverage: 80% minimum
- Critical paths (auth, registration, health): 95% minimum
- New features: 90% minimum
- Fail CI/CD if thresholds not met

## Performance Considerations

### Test Execution Speed

- Use pytest-xdist for parallel execution
- Mock external services to avoid network delays
- Use in-memory databases for integration tests
- Set reasonable timeouts (60s max per test)

### Resource Management

- Clean up test data after each test
- Use fixtures with proper teardown
- Avoid leaving background processes running
- Monitor memory usage in long-running tests

## Documentation

### Test Documentation Structure

```
tests/
├── README.md                    # Overview and quick start
├── CONTRIBUTING.md              # How to write tests
├── FIXTURES.md                  # Available fixtures guide
└── TROUBLESHOOTING.md           # Common issues and solutions
```

### Documentation Content

1. **README.md**: Test execution commands, coverage targets, CI/CD integration
2. **CONTRIBUTING.md**: Test writing guidelines, AAA pattern, fixture usage
3. **FIXTURES.md**: Complete list of available fixtures with examples
4. **TROUBLESHOOTING.md**: Common test failures and how to fix them

## Migration Plan

### Backward Compatibility

- Keep old fixtures temporarily with deprecation warnings
- Provide migration guide for existing tests
- Update tests incrementally, one module at a time

### Rollout Strategy

1. Phase 1: Infrastructure (Week 1)
2. Phase 2: Critical paths (Week 2-3)
3. Phase 3: New features (Week 4)
4. Phase 4: Integration (Week 5)
5. Phase 5: E2E (Week 6)

### Success Metrics

- All tests passing
- 80%+ coverage achieved
- CI/CD integration working
- No deprecated fixtures remaining
- Documentation complete
