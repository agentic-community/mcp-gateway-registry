"""
Conftest for integration tests.

Provides fixtures specific to integration tests that involve multiple
components working together.
"""

import logging
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function", autouse=True)
def reset_mongodb_client():
    """Reset MongoDB client singleton before each test to pick up correct settings."""
    from registry.repositories.documentdb import client

    # Clear the global client cache so next test creates a new one with correct settings
    client._client = None
    client._database = None

    yield

    # Cleanup is handled by TestClient teardown


@pytest.fixture(autouse=True)
def mock_security_scanner():
    """Mock security scanner for integration tests to avoid mcp-scanner dependency."""
    from registry.schemas.security import SecurityScanConfig, SecurityScanResult

    mock_service = MagicMock()

    # Return config with scanning disabled to avoid scan during registration
    mock_service.get_scan_config.return_value = SecurityScanConfig(
        enabled=False, scan_on_registration=False, block_unsafe_servers=False
    )

    # If scan is called anyway, return a passing result
    mock_service.scan_server = AsyncMock(
        return_value=SecurityScanResult(
            server_url="http://localhost:9000/mcp",
            server_path="/test-server",
            scan_timestamp="2025-01-01T00:00:00Z",
            is_safe=True,
            critical_issues=0,
            high_severity=0,
            medium_severity=0,
            low_severity=0,
            analyzers_used=["yara"],
            raw_output={},
            scan_failed=False,
        )
    )

    with patch("registry.api.server_routes.security_scanner_service", mock_service):
        yield mock_service


@pytest.fixture
def test_client(mock_settings) -> Generator[TestClient, None, None]:
    """
    Create a FastAPI test client for integration tests.

    Args:
        mock_settings: Test settings fixture

    Yields:
        FastAPI TestClient instance
    """
    from registry.main import app

    with TestClient(app) as client:
        logger.debug("Created FastAPI test client")
        yield client


@pytest.fixture
async def async_test_client(mock_settings):
    """
    Create an async FastAPI test client for integration tests.

    Args:
        mock_settings: Test settings fixture

    Yields:
        Async test client
    """
    from httpx import AsyncClient

    from registry.main import app

    async with AsyncClient(app=app, base_url="http://test") as client:
        logger.debug("Created async FastAPI test client")
        yield client
