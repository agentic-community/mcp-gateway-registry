# MCP Gateway Registry Tests

This directory contains the complete test infrastructure for the MCP Gateway Registry project.

## Directory Structure

```
tests/
├── conftest.py                          # Root conftest with session fixtures and auto-mocking
├── test_infrastructure.py               # Test to verify infrastructure works
├── fixtures/                            # Test fixtures and utilities
│   ├── __init__.py
│   ├── constants.py                     # Test constants
│   ├── factories.py                     # Factory Boy factories for test data
│   ├── helpers.py                       # Helper functions for tests
│   └── mocks/                          # Mock implementations
│       ├── __init__.py
│       ├── mock_embeddings.py          # Mock embeddings clients
│       ├── mock_http.py                # Mock HTTP clients
│       └── mock_auth.py                # Mock authentication
├── unit/                               # Unit tests
│   ├── __init__.py
│   ├── conftest.py                     # Unit test fixtures
│   ├── core/                           # Core infrastructure tests
│   ├── services/                       # Service layer tests
│   ├── search/                         # Search and vector search tests
│   ├── embeddings/                     # Embeddings tests
│   ├── health/                         # Health monitoring tests
│   ├── auth/                          # Auth tests
│   └── api/                           # API routes tests
├── integration/                        # Integration tests
│   ├── __init__.py
│   └── conftest.py                     # Integration test fixtures
└── auth_server/                        # Auth server tests
    ├── __init__.py
    ├── conftest.py                     # Auth server fixtures
    └── fixtures/                       # Auth-specific fixtures
        ├── __init__.py
        ├── mock_jwt.py                 # JWT utilities
        └── mock_providers.py           # Mock auth providers
```

## Key Features

### Auto-Mocking

The root `conftest.py` automatically mocks heavy dependencies BEFORE they are imported:

- **Vector search**: Mocked to avoid loading native dependencies
- **sentence-transformers**: Mocked to avoid loading ML models
- **litellm**: Mocked for embeddings testing

This ensures tests run fast without downloading or loading large dependencies.

### Test Fixtures

#### Session-Scoped Fixtures

- `event_loop_policy`: Configures async event loop for tests
- `tmp_test_dir`: Session-wide temporary directory

#### Function-Scoped Fixtures

- `test_settings`: Settings instance with temporary directories
- `mock_settings`: Patches global settings with test settings
- `sample_server_info`: Sample server data dictionary
- `sample_agent_card`: Sample agent card data dictionary

### Factory Boy Factories

Create realistic test data with `Factory Boy`:

```python
from tests.fixtures.factories import ServerDetailFactory, AgentCardFactory

# Create a server with defaults
server = ServerDetailFactory()

# Create with custom values
server = ServerDetailFactory(name="custom.server", version="2.0.0")

# Create multiple servers
servers = [ServerDetailFactory() for _ in range(5)]

# Create agent with skills
from tests.fixtures.factories import create_agent_with_skills
agent = create_agent_with_skills(num_skills=5)
```

### Mock Implementations

#### Mock Embeddings Client

```python
from tests.fixtures.mocks.mock_embeddings import MockEmbeddingsClient

client = MockEmbeddingsClient(dimension=384)
embeddings = client.encode(["text 1", "text 2"])
# Returns deterministic embeddings based on text hash
```

#### Mock Authentication

```python
from tests.fixtures.mocks.mock_auth import MockJWTValidator

validator = MockJWTValidator()
token = validator.create_token(
    username="testuser",
    groups=["users"],
    scopes=["read:servers"]
)
payload = validator.validate_token(token)
```

### Test Constants

All test constants are centralized in `fixtures/constants.py`:

```python
from tests.fixtures.constants import (
    TEST_SERVER_NAME_1,
    TEST_AGENT_NAME_1,
    TEST_USER_GROUPS,
    VISIBILITY_PUBLIC,
)
```

### Helper Functions

Common test operations are in `fixtures/helpers.py`:

```python
from tests.fixtures.helpers import (
    create_test_server_file,
    create_test_agent_file,
    create_minimal_server_dict,
    assert_server_equals,
)

# Create server file in temp directory
server_file = create_test_server_file(
    servers_dir=tmp_path / "servers",
    server_name="test.server",
    server_data={"name": "test.server", ...}
)
```

## Running Tests

### Run all tests

```bash
pytest tests/
```

### Run specific test categories

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Auth server tests only
pytest tests/auth_server/

# Tests marked as 'unit'
pytest -m unit

# Tests marked as 'integration'
pytest -m integration
```

### Run with coverage

```bash
pytest tests/ --cov=registry --cov-report=html
```

### Run specific test file

```bash
pytest tests/unit/services/test_server_service.py
```

### Run with verbose output

```bash
pytest tests/ -v
```

## Writing Tests

### Unit Test Example

```python
import pytest
from tests.fixtures.factories import ServerDetailFactory

class TestServerService:
    """Tests for server service."""

    def test_get_server(self, mock_settings):
        """Test retrieving a server."""
        # Arrange
        server = ServerDetailFactory()

        # Act
        # ... test logic

        # Assert
        assert server.name is not None
```

### Integration Test Example

```python
import pytest

class TestServerRoutes:
    """Integration tests for server routes."""

    @pytest.mark.integration
    async def test_list_servers(self, async_test_client):
        """Test listing servers via API."""
        response = await async_test_client.get("/api/v1/servers")
        assert response.status_code == 200
```

### Auth Test Example

```python
import pytest
from tests.auth_server.fixtures.mock_jwt import create_mock_jwt_token

class TestAuthentication:
    """Tests for authentication."""

    def test_token_validation(self, mock_jwt_validator):
        """Test JWT token validation."""
        token = mock_jwt_validator.create_token("testuser")
        payload = mock_jwt_validator.validate_token(token)
        assert payload["username"] == "testuser"
```

## Test Markers

Tests can be marked with pytest markers:

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.auth`: Authentication tests
- `@pytest.mark.slow`: Slow-running tests
- `@pytest.mark.requires_models`: Tests needing real ML models

Markers are automatically applied based on file location:
- Files in `unit/` get `@pytest.mark.unit`
- Files in `integration/` get `@pytest.mark.integration`
- Files in `auth_server/` get `@pytest.mark.auth`

## Troubleshooting

### Import Errors

If you get import errors, ensure you're running pytest from the project root:

```bash
cd /home/ubuntu/mcp-gateway-registry-MAIN
pytest tests/
```

### Search Dependencies Not Mocked

If search dependencies load during tests, ensure `conftest.py` is being loaded:

```bash
pytest tests/ -v --setup-show
```

You should see the auto-mocking messages in the output.

### Async Tests Not Running

Ensure `pytest-asyncio` is installed:

```bash
uv pip install pytest-asyncio
```

## Test Data

Test data is generated using:

1. **Factory Boy** for model instances
2. **Helper functions** for file-based data
3. **Constants** for consistent values

This ensures test data is:
- Realistic
- Consistent
- Easy to maintain
- Fast to generate

## Best Practices

1. **Use fixtures** for common setup
2. **Use factories** for creating test data
3. **Use constants** instead of hardcoding values
4. **Mock external dependencies** (HTTP, databases, etc.)
5. **Test one thing per test** function
6. **Use descriptive test names**
7. **Follow AAA pattern**: Arrange, Act, Assert
8. **Clean up** in fixture teardown if needed

## Coverage Goals

- Minimum coverage: 80%
- Target coverage: 90%+
- Critical paths: 100%

Run coverage report:

```bash
pytest tests/ --cov=registry --cov-report=html
open htmlcov/index.html
```
