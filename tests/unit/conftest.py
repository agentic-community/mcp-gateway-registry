"""
Conftest for unit tests.

Provides fixtures specific to unit tests.
"""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_embeddings_client():
    """
    Create a mock embeddings client for testing.

    Returns:
        Mock embeddings client
    """
    from tests.fixtures.mocks.mock_embeddings import MockEmbeddingsClient

    return MockEmbeddingsClient()


@pytest.fixture
def mock_http_client():
    """
    Create a mock HTTP client for testing.

    Returns:
        Mock HTTP client
    """
    from tests.fixtures.mocks.mock_http import MockAsyncClient

    return MockAsyncClient()


@pytest.fixture
def mock_mcp_client():
    """
    Create a mock MCP client for testing.

    Returns:
        Mock MCP client with common methods
    """
    client = AsyncMock()
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.list_tools = AsyncMock(return_value=[])
    client.call_tool = AsyncMock(return_value={})
    return client
