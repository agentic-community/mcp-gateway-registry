"""
Root conftest for pytest configuration and shared fixtures.

This module provides session-scoped fixtures and auto-mocking configuration
that applies to all tests.
"""

# =============================================================================
# SSL PATH MOCKING (BEFORE ANY IMPORTS)
# =============================================================================
# This must run FIRST to avoid permission errors when nginx_service is imported

import errno
import os

_original_stat = os.stat


def _patched_stat(path, *args, **kwargs):
    """Patched stat that handles SSL paths gracefully in CI environments."""
    path_str = str(path).lower()
    if "ssl" in path_str or "privkey" in path_str or "fullchain" in path_str:
        # Raise FileNotFoundError with proper errno for SSL paths
        # This simulates missing certs and is properly handled by Path.exists()
        raise FileNotFoundError(errno.ENOENT, "No such file or directory", str(path))
    return _original_stat(path, *args, **kwargs)


# Apply the patch immediately
os.stat = _patched_stat

# =============================================================================
# NOW SAFE TO IMPORT
# =============================================================================

import logging
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.fixtures.mocks.mock_embeddings import (
    create_mock_litellm_module,
    create_mock_st_module,
)
from tests.fixtures.mocks.mock_faiss import create_mock_faiss_module

logger = logging.getLogger(__name__)


# =============================================================================
# ENVIRONMENT SETUP (BEFORE ANY IMPORTS)
# =============================================================================
# Set environment variables for test environment BEFORE any app code imports
# This ensures Settings loads the correct values for tests


def pytest_configure(config):
    """
    Pytest hook that runs BEFORE test collection.

    This runs before any imports happen, ensuring environment variables
    are set before Settings() is created. Also registers custom markers.

    Args:
        config: Pytest config object
    """
    # Set MongoDB connection to localhost for tests
    # (Docker deployments use 'mongodb' hostname from docker-compose.yml)
    os.environ["DOCUMENTDB_HOST"] = "localhost"
    os.environ["DOCUMENTDB_PORT"] = "27017"

    # Keep mongodb-ce as storage backend for integration tests
    os.environ["STORAGE_BACKEND"] = "mongodb-ce"

    # Use directConnection for single-node MongoDB in tests
    # (AWS DocumentDB clusters should NOT use directConnection)
    os.environ["DOCUMENTDB_DIRECT_CONNECTION"] = "true"

    # Disable TLS for local MongoDB in tests
    # (AWS DocumentDB requires TLS, but local MongoDB CE does not)
    os.environ["DOCUMENTDB_USE_TLS"] = "false"

    print(
        "Test environment configured: DOCUMENTDB_HOST=localhost, STORAGE_BACKEND=mongodb-ce, DOCUMENTDB_DIRECT_CONNECTION=true, DOCUMENTDB_USE_TLS=false"
    )

    # Force reload settings if it's already been imported
    # This is needed because Settings() is created at module level
    try:
        import registry.core.config as config_module

        # Recreate the settings object with the new environment variables
        config_module.settings = config_module.Settings()
        print(f"Reloaded settings with documentdb_host={config_module.settings.documentdb_host}")
    except ImportError:
        # Settings hasn't been imported yet, which is fine
        pass

    # Register custom markers
    config.addinivalue_line("markers", "unit: Unit tests that test single components in isolation")
    config.addinivalue_line(
        "markers", "integration: Integration tests that test multiple components together"
    )
    config.addinivalue_line("markers", "requires_models: Tests that require real ML models (slow)")
    config.addinivalue_line("markers", "auth: Authentication and authorization tests")
    config.addinivalue_line("markers", "agents: A2A agent service tests")
    config.addinivalue_line("markers", "servers: MCP server service tests")
    config.addinivalue_line("markers", "api: API route tests")
    config.addinivalue_line("markers", "search: Search functionality tests")
    config.addinivalue_line("markers", "slow: Tests that take a long time to run")


# =============================================================================
# AUTO-MOCKING SETUP (BEFORE IMPORTS)
# =============================================================================
# This section must run BEFORE any registry code imports the real libraries


def _setup_auto_mocking() -> None:
    """
    Set up automatic mocking for heavy dependencies.

    This function mocks FAISS and sentence-transformers BEFORE they are
    imported by the application code, avoiding loading large ML models
    during tests.
    """
    # Mock FAISS
    mock_faiss = create_mock_faiss_module()
    sys.modules["faiss"] = mock_faiss
    logger.info("Auto-mocked: faiss")

    # Mock sentence_transformers
    mock_st = create_mock_st_module()
    sys.modules["sentence_transformers"] = mock_st
    logger.info("Auto-mocked: sentence_transformers")

    # Mock litellm
    mock_litellm = create_mock_litellm_module()
    sys.modules["litellm"] = mock_litellm
    logger.info("Auto-mocked: litellm")


# Execute auto-mocking setup
_setup_auto_mocking()


# Now we can safely import registry modules
from registry.core.config import Settings  # noqa: E402

# =============================================================================
# SESSION-SCOPED FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def event_loop_policy():
    """
    Configure event loop policy for async tests.

    Returns:
        Event loop policy instance
    """
    import asyncio

    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture(scope="session")
def tmp_test_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for test files that persists for the session.

    Yields:
        Path to temporary directory
    """
    temp_dir = tempfile.mkdtemp(prefix="mcp_registry_test_")
    temp_path = Path(temp_dir)
    logger.info(f"Created session temp directory: {temp_path}")

    yield temp_path

    # Cleanup handled by OS temp dir cleanup


# =============================================================================
# FUNCTION-SCOPED FIXTURES
# =============================================================================


@pytest.fixture
def tmp_path(tmp_path_factory) -> Path:
    """
    Create a temporary directory for a single test.

    Args:
        tmp_path_factory: Pytest's tmp_path_factory fixture

    Returns:
        Path to temporary directory
    """
    return tmp_path_factory.mktemp("test")


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """
    Create test settings with temporary directories.

    This fixture provides a Settings instance with all paths pointing to
    temporary directories to avoid conflicts with actual data.

    Args:
        tmp_path: Temporary directory path

    Returns:
        Test Settings instance
    """
    # Create subdirectories
    servers_dir = tmp_path / "servers"
    agents_dir = tmp_path / "agents"
    models_dir = tmp_path / "models"
    logs_dir = tmp_path / "logs"

    servers_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Override settings with test values
    settings = Settings(
        secret_key="test-secret-key-for-testing-only",
        admin_user="testadmin",
        admin_password="testpass",
        session_cookie_name="test_session",
        auth_server_url="http://localhost:8888",
        embeddings_provider="sentence-transformers",
        embeddings_model_name="all-MiniLM-L6-v2",
        embeddings_model_dimensions=384,
        documentdb_host="localhost",  # Use localhost for tests
        documentdb_port=27017,
        documentdb_use_tls=False,  # Disable TLS for local MongoDB in tests
        documentdb_direct_connection=True,  # Use direct connection for single-node MongoDB
    )

    # Patch path properties to use temp directories
    # Save original property descriptors (not computed values) for restoration
    original_servers_dir_prop = type(settings).__dict__.get("servers_dir")
    original_agents_dir_prop = type(settings).__dict__.get("agents_dir")
    original_embeddings_model_dir_prop = type(settings).__dict__.get("embeddings_model_dir")
    original_log_dir_prop = type(settings).__dict__.get("log_dir")

    # Mock the path properties with temp directory values
    type(settings).servers_dir = property(lambda self: servers_dir)
    type(settings).agents_dir = property(lambda self: agents_dir)
    type(settings).embeddings_model_dir = property(lambda self: models_dir)
    type(settings).log_dir = property(lambda self: logs_dir)

    logger.debug(f"Created test settings with temp dirs in {tmp_path}")

    yield settings

    # Restore original property descriptors (not fixed values)
    if original_servers_dir_prop is not None:
        type(settings).servers_dir = original_servers_dir_prop
    if original_agents_dir_prop is not None:
        type(settings).agents_dir = original_agents_dir_prop
    if original_embeddings_model_dir_prop is not None:
        type(settings).embeddings_model_dir = original_embeddings_model_dir_prop
    if original_log_dir_prop is not None:
        type(settings).log_dir = original_log_dir_prop


@pytest.fixture
def mock_settings(test_settings: Settings, monkeypatch):
    """
    Mock the global settings instance with test settings.

    This fixture patches registry.core.config.settings to use test settings
    for the duration of the test.

    Args:
        test_settings: Test settings instance
        monkeypatch: Pytest monkeypatch fixture

    Returns:
        Test settings instance
    """
    monkeypatch.setattr("registry.core.config.settings", test_settings)
    logger.debug("Patched global settings with test settings")
    return test_settings


@pytest.fixture
def mock_scope_repository():
    """
    Mock scope repository to avoid DocumentDB access.

    Returns:
        AsyncMock instance with common scope repository methods
    """
    mock = AsyncMock()
    mock.load_all = AsyncMock()
    mock.get_group_mappings.return_value = []
    mock.list_groups.return_value = {}  # Return empty dict, not list
    mock.get_group.return_value = None
    mock.get_scope_definition.return_value = None
    mock.list_scope_definitions.return_value = []
    return mock


@pytest.fixture
def mock_server_repository():
    """
    Mock server repository to avoid DocumentDB access.

    Returns:
        AsyncMock instance with common server repository methods
    """
    mock = AsyncMock()
    mock.load_all.return_value = {}  # Return empty dict of servers
    mock.list_all.return_value = {}  # Return empty dict of servers, not list
    mock.get.return_value = None
    mock.save.return_value = None
    mock.delete.return_value = None
    mock.delete_with_versions.return_value = 0
    mock.create.return_value = True
    mock.update.return_value = True
    mock.get_state.return_value = False
    mock.set_state.return_value = True
    return mock


@pytest.fixture
def mock_agent_repository():
    """
    Mock agent repository to avoid DocumentDB access.

    Returns:
        AsyncMock instance with common agent repository methods
    """
    mock = AsyncMock()
    mock.load_all.return_value = []
    mock.list_all.return_value = []
    mock.get.return_value = None
    mock.save.return_value = None
    mock.delete.return_value = None
    mock.create.return_value = True
    mock.update.return_value = True
    mock.get_state.return_value = {"enabled": [], "disabled": []}
    mock.save_state.return_value = True
    mock.set_state.return_value = True
    mock.get_all_state.return_value = {}
    return mock


@pytest.fixture
def mock_search_repository():
    """
    Mock search repository to avoid DocumentDB/FAISS access.

    Returns:
        AsyncMock instance with common search repository methods
    """
    mock = AsyncMock()
    mock.initialize.return_value = None
    mock.add_embedding.return_value = None
    mock.search.return_value = []
    mock.hybrid_search.return_value = []
    mock.index_server.return_value = None
    mock.index_agent.return_value = None
    return mock


@pytest.fixture
def mock_federation_config_repository():
    """
    Mock federation config repository to avoid DocumentDB access.

    Returns:
        AsyncMock instance with common federation config methods
    """
    mock = AsyncMock()
    mock.get_config.return_value = None
    mock.save_config.return_value = None
    mock.list_configs.return_value = []
    return mock


@pytest.fixture
def mock_security_scan_repository():
    """
    Mock security scan repository to avoid DocumentDB access.

    Returns:
        AsyncMock instance with common security scan methods
    """
    mock = AsyncMock()
    mock.save_scan.return_value = None
    mock.get_scan.return_value = None
    mock.list_scans.return_value = []
    return mock


@pytest.fixture
def mock_virtual_server_repository():
    """
    Mock virtual server repository to avoid DocumentDB access.

    Returns:
        AsyncMock instance with common virtual server repository methods
    """
    mock = AsyncMock()
    mock.ensure_indexes = AsyncMock()
    mock.get.return_value = None
    mock.list_all.return_value = []
    mock.list_enabled.return_value = []
    mock.create = AsyncMock()
    mock.update = AsyncMock()
    mock.delete.return_value = True
    mock.get_state.return_value = False
    mock.set_state.return_value = True
    return mock


@pytest.fixture
def mock_backend_session_repository():
    """
    Mock backend session repository to avoid DocumentDB access.

    Returns:
        AsyncMock instance with common backend session repository methods
    """
    mock = AsyncMock()
    mock.ensure_indexes = AsyncMock()
    mock.get_backend_session.return_value = None
    mock.store_backend_session = AsyncMock()
    mock.delete_backend_session = AsyncMock()
    mock.create_client_session = AsyncMock()
    mock.validate_client_session.return_value = False
    return mock


@pytest.fixture
def mock_skill_security_scan_repository():
    """
    Mock skill security scan repository to avoid DocumentDB access.

    Returns:
        AsyncMock instance with common skill security scan methods
    """
    mock = AsyncMock()
    mock.create.return_value = True
    mock.get_latest.return_value = None
    mock.get.return_value = None
    mock.list_all.return_value = []
    mock.query_by_status.return_value = []
    mock.load_all.return_value = None
    return mock


@pytest.fixture(autouse=True)
def mock_all_repositories(
    mock_scope_repository,
    mock_server_repository,
    mock_agent_repository,
    mock_search_repository,
    mock_federation_config_repository,
    mock_security_scan_repository,
    mock_virtual_server_repository,
    mock_backend_session_repository,
    mock_skill_security_scan_repository,
):
    """
    Auto-mock all repository factory functions to prevent DocumentDB access.

    This fixture automatically applies to all tests and prevents any
    accidental DocumentDB connections during test execution.

    Args:
        mock_scope_repository: Mock scope repository
        mock_server_repository: Mock server repository
        mock_agent_repository: Mock agent repository
        mock_search_repository: Mock search repository
        mock_federation_config_repository: Mock federation config repository
        mock_security_scan_repository: Mock security scan repository
        mock_virtual_server_repository: Mock virtual server repository
        mock_backend_session_repository: Mock backend session repository

    Yields:
        None
    """
    # Most tests only need registry patches, not auth_server patches
    # Only patch auth_server for auth_server tests (they have their own conftest)
    with (
        patch(
            "registry.repositories.factory.get_scope_repository", return_value=mock_scope_repository
        ),
        patch(
            "registry.repositories.factory.get_server_repository",
            return_value=mock_server_repository,
        ),
        patch(
            "registry.repositories.factory.get_agent_repository", return_value=mock_agent_repository
        ),
        patch(
            "registry.repositories.factory.get_search_repository",
            return_value=mock_search_repository,
        ),
        patch(
            "registry.repositories.factory.get_federation_config_repository",
            return_value=mock_federation_config_repository,
        ),
        patch(
            "registry.repositories.factory.get_security_scan_repository",
            return_value=mock_security_scan_repository,
        ),
        patch(
            "registry.repositories.factory.get_virtual_server_repository",
            return_value=mock_virtual_server_repository,
        ),
        patch(
            "registry.repositories.factory.get_backend_session_repository",
            return_value=mock_backend_session_repository,
        ),
        patch(
            "registry.repositories.factory.get_skill_security_scan_repository",
            return_value=mock_skill_security_scan_repository,
        ),
    ):
        logger.debug("Auto-mocked all repository factory functions")
        yield


@pytest.fixture
def sample_server_info() -> dict[str, Any]:
    """
    Create sample server information for testing.

    Returns:
        Dictionary with sample server data
    """
    return {
        "name": "com.example.test-server",
        "description": "A test MCP server for unit tests",
        "version": "1.0.0",
        "title": "Test Server",
        "repository": {
            "url": "https://github.com/example/test-server",
            "source": "github",
            "id": "test-repo-123",
        },
        "websiteUrl": "https://example.com/test-server",
        "packages": [
            {
                "registryType": "npm",
                "identifier": "@example/test-server",
                "version": "1.0.0",
                "transport": {"type": "stdio", "command": "uvx", "args": ["test-server"]},
                "runtimeHint": "uvx",
            }
        ],
        "_meta": {
            "tools": [
                {
                    "name": "get_data",
                    "description": "Retrieve data from source",
                    "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}},
                }
            ],
            "prompts": [],
            "resources": [],
        },
    }


@pytest.fixture
def sample_agent_card() -> dict[str, Any]:
    """
    Create sample agent card for testing.

    Returns:
        Dictionary with sample agent card data
    """
    return {
        "protocolVersion": "1.0",
        "name": "test-agent",
        "description": "A test agent for unit tests",
        "url": "http://localhost:9000/test-agent",
        "version": "1.0",
        "capabilities": {"streaming": False, "tools": True},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain", "application/json"],
        "skills": [
            {
                "id": "data-retrieval",
                "name": "Data Retrieval",
                "description": "Retrieve data from various sources",
                "tags": ["data", "retrieval"],
                "examples": ["Get customer data", "Fetch order information"],
            }
        ],
        "path": "/agents/test-agent",
        "tags": ["test", "data"],
        "isEnabled": True,
        "numStars": 4.5,
        "license": "MIT",
        "visibility": "public",
        "trustLevel": "unverified",
    }


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers automatically.

    Args:
        config: Pytest config object
        items: List of collected test items
    """
    for item in items:
        # Auto-mark tests based on file location
        if "unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "auth_server/" in str(item.fspath):
            item.add_marker(pytest.mark.auth)


# =============================================================================
# DEPLOYMENT MODE FIXTURES
# =============================================================================


@pytest.fixture
def client_registry_only(mock_settings) -> Generator[Any, None, None]:
    """Test client with registry-only deployment mode."""
    from fastapi.testclient import TestClient
    from registry.core.config import DeploymentMode, RegistryMode

    object.__setattr__(mock_settings, "deployment_mode", DeploymentMode.REGISTRY_ONLY)
    object.__setattr__(mock_settings, "registry_mode", RegistryMode.FULL)

    from registry.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def client_skills_only(mock_settings) -> Generator[Any, None, None]:
    """Test client with skills-only registry mode."""
    from fastapi.testclient import TestClient
    from registry.core.config import DeploymentMode, RegistryMode

    object.__setattr__(mock_settings, "deployment_mode", DeploymentMode.REGISTRY_ONLY)
    object.__setattr__(mock_settings, "registry_mode", RegistryMode.SKILLS_ONLY)

    from registry.main import app

    with TestClient(app) as client:
        yield client
