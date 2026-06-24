# Test Categories and Best Practices

## Test Markers

Tests are organized using pytest markers to enable selective test execution:

### Primary Categories

- **`@pytest.mark.unit`** - Fast, isolated unit tests
  - No external dependencies
  - Mocked services and models
  - Should run in < 1 second each

- **`@pytest.mark.integration`** - Integration tests
  - May interact with services
  - May use real HTTP clients
  - Can take longer to run

- **`@pytest.mark.e2e`** - End-to-end workflow tests
  - Test complete user workflows
  - May involve multiple components
  - Typically slower

### Domain-Specific Markers

- **`@pytest.mark.auth`** - Authentication and authorization tests
- **`@pytest.mark.servers`** - Server management tests
- **`@pytest.mark.search`** - Search and AI functionality tests
- **`@pytest.mark.health`** - Health monitoring tests
- **`@pytest.mark.core`** - Core infrastructure tests

### Special Markers

- **`@pytest.mark.slow`** - Slow-running tests (> 5 seconds)
  - Excluded by default in fast test runs
  - Should be minimized

- **`@pytest.mark.requires_models`** - Tests requiring real ML models
  - Will load actual embeddings models and vector search
  - **WARNING**: These tests can cause OOM on small EC2 instances
  - Should only be used when absolutely necessary
  - Consider if the functionality can be tested with mocks instead

## Default Test Behavior (Memory-Safe)

By default, **ALL** tests use mocked versions of heavy dependencies to prevent OOM crashes:

- **Search service** - Mocked automatically
- **Embeddings models** - Mocked automatically
- **Sentence-transformers** - Mocked automatically
- **PyTorch model loading** - Blocked

This means tests run fast and safely on any EC2 instance size.

## Writing Memory-Safe Tests

### Good Example (Default)

```python
import pytest

@pytest.mark.unit
def test_server_registration(server_service, sample_server):
    """Test server registration with mocked dependencies."""
    # Search service and embeddings are automatically mocked
    server_service.register_server(sample_server)
    assert server_service.is_registered(sample_server["name"])
```

### When You Need Real Models (Use Sparingly)

Only use real models when:
1. Testing the actual ML model functionality
2. Testing embeddings quality or accuracy
3. Integration testing with real vector search

```python
import pytest

@pytest.mark.requires_models  # Mark as requiring real models
@pytest.mark.slow  # Will be slow
@pytest.mark.integration  # Not a unit test
def test_real_embeddings_search(real_search_service):
    """Test search with real embeddings model.

    WARNING: This test loads real ML models and may cause OOM on small instances.
    """
    # This test actually loads sentence-transformers and search dependencies
    await real_search_service.initialize()
    results = await real_search_service.search_services("test query")
    assert len(results) > 0
```

**Running tests that require models:**

```bash
# Run all tests including those requiring models (WARNING: High memory usage)
pytest -m requires_models

# Exclude tests requiring models (safe for EC2)
pytest -m "not requires_models"
```

## Test Fixtures

### Automatically Available (Mocked)

These fixtures are automatically mocked for all tests:

- `mock_search_service` - Mocked vector search service
- `mock_embeddings` - Mocked embeddings client
- `prevent_real_model_loading` - Prevents torch/sentence-transformers loading

### Commonly Used Test Fixtures

- `test_client` - FastAPI TestClient
- `async_client` - Async HTTP client
- `mock_authenticated_user` - Simulates authenticated user
- `server_service` - Server management service
- `health_service` - Health monitoring service
- `sample_server` - Sample server data for testing
- `sample_servers` - Multiple sample servers
- `temp_dir` - Temporary directory for tests

### Settings and Configuration

- `test_settings` - Test configuration with temp directories
- `mock_settings` - Globally mocked settings

## Writing Good Tests

### Unit Test Example

```python
import pytest
from unittest.mock import Mock, AsyncMock

@pytest.mark.unit
@pytest.mark.auth
class TestAuthService:
    """Tests for authentication service."""

    def test_valid_token_verification(self, auth_service):
        """Test that valid tokens are verified correctly."""
        token = "valid-token-12345"
        result = auth_service.verify_token(token)
        assert result is True

    async def test_token_generation(self, auth_service):
        """Test JWT token generation."""
        user_data = {"username": "testuser", "role": "admin"}
        token = await auth_service.generate_token(user_data)
        assert token is not None
        assert len(token) > 50
```

### Integration Test Example

```python
import pytest
from httpx import AsyncClient

@pytest.mark.integration
@pytest.mark.servers
class TestServerRegistration:
    """Integration tests for server registration API."""

    async def test_register_server_endpoint(
        self,
        async_client: AsyncClient,
        sample_server,
        integration_auth_headers
    ):
        """Test server registration via API endpoint."""
        response = await async_client.post(
            "/api/servers",
            json=sample_server,
            headers=integration_auth_headers
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == sample_server["name"]
```

## Test Organization

```
tests/
├── unit/                      # Unit tests (fast, isolated)
│   ├── auth/                  # Authentication tests
│   ├── api/                   # API endpoint tests
│   ├── core/                  # Core functionality tests
│   ├── services/              # Service layer tests
│   └── ...
├── integration/               # Integration tests
│   ├── test_server_routes.py
│   ├── test_search_routes.py
│   └── test_e2e_workflows.py
├── fixtures/                  # Test data factories
│   └── factories.py
├── reports/                   # Generated test reports
└── conftest.py               # Shared fixtures and configuration
```

## Best Practices

### DO

✅ Use markers to categorize tests
✅ Mock heavy dependencies by default
✅ Keep unit tests fast (< 1 second)
✅ Test one thing per test function
✅ Use descriptive test names
✅ Clean up resources in fixtures
✅ Use AAA pattern (Arrange, Act, Assert)

### DON'T

❌ Load real ML models in unit tests
❌ Make network calls in unit tests
❌ Share state between tests
❌ Test implementation details
❌ Write tests longer than 30 lines
❌ Use `time.sleep()` - use mocks instead

### Memory-Safe Testing

✅ Use mocked services by default
✅ Mark tests requiring real models with `@pytest.mark.requires_models`
✅ Run tests serially on EC2 by default
✅ Monitor memory usage during test development

❌ Don't use `-n auto` on small EC2 instances
❌ Don't load real models unless absolutely necessary
❌ Don't skip mocking fixtures without good reason

## Running Tests Efficiently

```bash
# Fast unit tests only (seconds)
python scripts/test.py unit

# Specific domain tests
python scripts/test.py auth
python scripts/test.py servers

# Exclude slow tests
python scripts/test.py fast

# Full test suite (serial, safe)
python scripts/test.py full

# With parallelization (if you have memory)
python scripts/test.py full -n 2

# Exclude tests requiring real models
pytest -m "not requires_models"

# Run only tests requiring models (high memory!)
pytest -m requires_models
```

## Debugging Test Failures

```bash
# Run with verbose output
pytest tests/unit/auth/test_auth_routes.py -v

# Run specific test
pytest tests/unit/auth/test_auth_routes.py::test_login_success -v

# Run with debug output
pytest tests/unit/auth/ -v --log-cli-level=DEBUG

# Stop on first failure
pytest -x

# Show local variables on failure
pytest -l

# Run last failed tests
pytest --lf
```

## Coverage Requirements

- Minimum overall coverage: 80%
- All new code should have tests
- Critical paths should have 100% coverage

```bash
# Generate coverage report
python scripts/test.py coverage

# View coverage report
open htmlcov/index.html
```
