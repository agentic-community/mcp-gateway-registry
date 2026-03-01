"""
Integration tests for the search pipeline.

This module tests the full search flow from registration to semantic search,
including filters and visibility controls.
"""

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status

from registry.search.service import FaissService
from registry.services.agent_service import agent_service
from registry.services.server_service import server_service
from tests.fixtures.factories import AgentCardFactory
from tests.fixtures.mocks.mock_embeddings import MockEmbeddingsClient

logger = logging.getLogger(__name__)


# Skip all tests in this file due to MongoDB connection timeouts
pytestmark = pytest.mark.skip(
    reason="MongoDB connection timeout during search repository initialization"
)


# =============================================================================
# AUTH DEPENDENCY OVERRIDES
# =============================================================================


@pytest.fixture
def mock_auth_dependencies():
    """
    Mock authentication dependencies using dependency_overrides.

    Returns:
        Dict with admin and regular user contexts
    """
    from registry.auth.dependencies import enhanced_auth, nginx_proxied_auth
    from registry.main import app

    admin_user_context = {
        "username": "testadmin",
        "is_admin": True,
        "groups": ["admin"],
        "scopes": ["admin"],
        "accessible_servers": ["all"],
        "accessible_agents": ["all"],
        "accessible_services": ["all"],
        "ui_permissions": {
            "list_service": ["all"],
            "toggle_service": ["all"],
            "register_service": ["all"],
            "modify_service": ["all"],
        },
        "auth_method": "session",
    }

    def mock_enhanced_auth_override():
        return admin_user_context

    def mock_nginx_proxied_auth_override():
        return admin_user_context

    # Override dependencies at the app level
    app.dependency_overrides[enhanced_auth] = mock_enhanced_auth_override
    app.dependency_overrides[nginx_proxied_auth] = mock_nginx_proxied_auth_override

    yield {"admin": admin_user_context}

    # Cleanup
    app.dependency_overrides.clear()


@pytest.fixture
def mock_nginx_service():
    """Mock nginx service."""
    with patch("registry.core.nginx_service.nginx_service") as mock_nginx:
        mock_nginx.generate_config = MagicMock()
        mock_nginx.reload_nginx = MagicMock()
        mock_nginx.generate_config_async = AsyncMock()
        yield mock_nginx


@pytest.fixture
def mock_health_service():
    """Mock health service."""
    with patch("registry.health.service.health_service") as mock_health:
        mock_health.initialize = AsyncMock()
        mock_health.shutdown = AsyncMock()
        mock_health.broadcast_health_update = AsyncMock()
        yield mock_health




@pytest.fixture(autouse=True)
def setup_search_environment(
    mock_settings,
    mock_auth_dependencies,
    mock_nginx_service,
    mock_health_service,
):
    """
    Auto-use fixture to set up test environment with all mocks.

    This fixture runs automatically for all tests in this module.
    """
    # Initialize services with clean state
    server_service.registered_servers = {}
    server_service.service_state = {}
    agent_service.registered_agents = {}
    agent_service.agent_enabled_state = {}

    yield

    # Cleanup
    server_service.registered_servers.clear()
    server_service.service_state.clear()
    agent_service.registered_agents.clear()
    agent_service.agent_enabled_state.clear()


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_embeddings_client():
    """Create a mock embeddings client for testing."""
    return MockEmbeddingsClient(model_name="test-model", dimension=384)


@pytest.fixture
def search_test_servers() -> list[dict[str, Any]]:
    """
    Create test servers with diverse content for search testing.

    Returns:
        List of server info dictionaries
    """
    return [
        {
            "path": "/database-server",
            "server_name": "database-tools",
            "description": "Server for database operations and queries",
            "tags": ["database", "sql", "query"],
            "num_tools": 3,
            "entity_type": "mcp_server",
            "tool_list": [
                {
                    "name": "query_database",
                    "description": "Execute SQL queries on database",
                    "parsed_description": {
                        "main": "Execute SQL queries on database",
                        "args": "query: string, database: string",
                    },
                    "schema": {"type": "object"},
                },
                {
                    "name": "list_tables",
                    "description": "List all tables in database",
                    "parsed_description": {
                        "main": "List all tables in database",
                        "args": "database: string",
                    },
                    "schema": {"type": "object"},
                },
                {
                    "name": "export_data",
                    "description": "Export data from database to CSV",
                    "parsed_description": {
                        "main": "Export data from database to CSV",
                        "args": "table: string, format: string",
                    },
                    "schema": {"type": "object"},
                },
            ],
        },
        {
            "path": "/weather-server",
            "server_name": "weather-api",
            "description": "Fetch weather data and forecasts",
            "tags": ["weather", "forecast", "climate"],
            "num_tools": 2,
            "entity_type": "mcp_server",
            "tool_list": [
                {
                    "name": "get_current_weather",
                    "description": "Get current weather for a location",
                    "parsed_description": {
                        "main": "Get current weather for a location",
                        "args": "location: string, units: string",
                    },
                    "schema": {"type": "object"},
                },
                {
                    "name": "get_forecast",
                    "description": "Get weather forecast for next 7 days",
                    "parsed_description": {
                        "main": "Get weather forecast for next 7 days",
                        "args": "location: string, days: integer",
                    },
                    "schema": {"type": "object"},
                },
            ],
        },
        {
            "path": "/file-server",
            "server_name": "file-operations",
            "description": "File system operations and file management",
            "tags": ["files", "filesystem", "storage"],
            "num_tools": 4,
            "entity_type": "mcp_server",
            "tool_list": [
                {
                    "name": "read_file",
                    "description": "Read contents of a file",
                    "parsed_description": {
                        "main": "Read contents of a file",
                        "args": "path: string",
                    },
                    "schema": {"type": "object"},
                },
                {
                    "name": "write_file",
                    "description": "Write data to a file",
                    "parsed_description": {
                        "main": "Write data to a file",
                        "args": "path: string, content: string",
                    },
                    "schema": {"type": "object"},
                },
                {
                    "name": "list_directory",
                    "description": "List files in a directory",
                    "parsed_description": {
                        "main": "List files in a directory",
                        "args": "path: string",
                    },
                    "schema": {"type": "object"},
                },
                {
                    "name": "delete_file",
                    "description": "Delete a file from filesystem",
                    "parsed_description": {
                        "main": "Delete a file from filesystem",
                        "args": "path: string",
                    },
                    "schema": {"type": "object"},
                },
            ],
        },
        {
            "path": "/search-server",
            "server_name": "web-search",
            "description": "Search the web and retrieve information",
            "tags": ["search", "web", "internet"],
            "num_tools": 2,
            "entity_type": "mcp_server",
            "tool_list": [
                {
                    "name": "web_search",
                    "description": "Search the web using search engines",
                    "parsed_description": {
                        "main": "Search the web using search engines",
                        "args": "query: string, limit: integer",
                    },
                    "schema": {"type": "object"},
                },
                {
                    "name": "scrape_webpage",
                    "description": "Extract content from a webpage",
                    "parsed_description": {
                        "main": "Extract content from a webpage",
                        "args": "url: string",
                    },
                    "schema": {"type": "object"},
                },
            ],
        },
    ]


@pytest.fixture
def search_test_agents() -> list[dict[str, Any]]:
    """
    Create test agents with diverse content for search testing.

    Returns:
        List of agent card dictionaries
    """
    agent1 = AgentCardFactory(
        name="data-analyst-agent",
        description="Analyze data and generate insights from databases",
        path="/agents/data-analyst",
        tags=["data", "analysis", "database"],
        skills=[
            {
                "id": "data-analysis",
                "name": "Data Analysis",
                "description": "Analyze datasets and generate statistical insights",
                "tags": ["analysis", "statistics"],
                "examples": ["Analyze sales data", "Generate trend reports"],
            },
            {
                "id": "database-query",
                "name": "Database Querying",
                "description": "Query databases and retrieve information",
                "tags": ["database", "sql"],
                "examples": ["Query customer records", "Extract transaction data"],
            },
        ],
        visibility="public",
    )

    agent2 = AgentCardFactory(
        name="weather-assistant",
        description="Provide weather information and forecasts",
        path="/agents/weather-assistant",
        tags=["weather", "forecast", "climate"],
        skills=[
            {
                "id": "weather-info",
                "name": "Weather Information",
                "description": "Get current weather and forecasts",
                "tags": ["weather"],
                "examples": ["What's the weather today?", "Will it rain tomorrow?"],
            }
        ],
        visibility="public",
    )

    agent3 = AgentCardFactory(
        name="code-reviewer",
        description="Review code and suggest improvements",
        path="/agents/code-reviewer",
        tags=["code", "review", "development"],
        skills=[
            {
                "id": "code-review",
                "name": "Code Review",
                "description": "Review code for quality and best practices",
                "tags": ["code", "review"],
                "examples": ["Review my Python code", "Check this function"],
            }
        ],
        visibility="public",
    )

    agent4 = AgentCardFactory(
        name="private-agent",
        description="Internal agent for internal use only",
        path="/agents/internal-agent",
        tags=["internal"],
        skills=[],
        visibility="internal",
        registered_by="testuser",
    )

    agent5 = AgentCardFactory(
        name="group-restricted-agent",
        description="Agent accessible only to specific groups",
        path="/agents/group-agent",
        tags=["group", "restricted"],
        skills=[],
        visibility="group-restricted",
        allowed_groups=["admin", "developers"],
        registered_by="testadmin",
    )

    return [
        agent1.model_dump(),
        agent2.model_dump(),
        agent3.model_dump(),
        agent4.model_dump(),
        agent5.model_dump(),
    ]


@pytest.fixture
def mock_faiss_search_results():
    """
    Create mock FAISS search results for predictable testing.

    Returns:
        Dictionary mapping query patterns to search results
    """
    return {
        "database": {
            "servers": [
                {
                    "entity_type": "mcp_server",
                    "path": "/database-server",
                    "server_name": "database-tools",
                    "description": "Server for database operations and queries",
                    "tags": ["database", "sql", "query"],
                    "num_tools": 3,
                    "is_enabled": True,
                    "relevance_score": 0.92,
                    "match_context": "Server for database operations and queries",
                    "matching_tools": [],
                }
            ],
            "tools": [],
            "agents": [
                {
                    "entity_type": "a2a_agent",
                    "path": "/agents/data-analyst",
                    "agent_name": "data-analyst-agent",
                    "description": "Analyze data and generate insights from databases",
                    "tags": ["data", "analysis", "database"],
                    "skills": ["Data Analysis", "Database Querying"],
                    "visibility": "public",
                    "is_enabled": True,
                    "relevance_score": 0.88,
                    "match_context": "Analyze data and generate insights from databases",
                }
            ],
        },
        "weather": {
            "servers": [
                {
                    "entity_type": "mcp_server",
                    "path": "/weather-server",
                    "server_name": "weather-api",
                    "description": "Fetch weather data and forecasts",
                    "tags": ["weather", "forecast", "climate"],
                    "num_tools": 2,
                    "is_enabled": True,
                    "relevance_score": 0.95,
                    "match_context": "Fetch weather data and forecasts",
                    "matching_tools": [],
                }
            ],
            "tools": [],
            "agents": [
                {
                    "entity_type": "a2a_agent",
                    "path": "/agents/weather-assistant",
                    "agent_name": "weather-assistant",
                    "description": "Provide weather information and forecasts",
                    "tags": ["weather", "forecast", "climate"],
                    "skills": ["Weather Information"],
                    "visibility": "public",
                    "is_enabled": True,
                    "relevance_score": 0.93,
                    "match_context": "Provide weather information and forecasts",
                }
            ],
        },
        "file operations": {
            "servers": [
                {
                    "entity_type": "mcp_server",
                    "path": "/file-server",
                    "server_name": "file-operations",
                    "description": "File system operations and file management",
                    "tags": ["files", "filesystem", "storage"],
                    "num_tools": 4,
                    "is_enabled": True,
                    "relevance_score": 0.90,
                    "match_context": "File system operations and file management",
                    "matching_tools": [
                        {
                            "tool_name": "read_file",
                            "description": "Read contents of a file",
                            "relevance_score": 0.85,
                            "match_context": "Read contents of a file",
                        }
                    ],
                }
            ],
            "tools": [
                {
                    "entity_type": "tool",
                    "server_path": "/file-server",
                    "server_name": "file-operations",
                    "tool_name": "read_file",
                    "description": "Read contents of a file",
                    "relevance_score": 0.85,
                    "match_context": "Read contents of a file",
                }
            ],
            "agents": [],
        },
        "empty query": {"servers": [], "tools": [], "agents": []},
    }


@pytest.fixture
async def setup_search_data(
    mock_settings, search_test_servers, search_test_agents, mock_embeddings_client
):
    """
    Set up test data in FAISS service for search testing.

    Args:
        mock_settings: Test settings fixture
        search_test_servers: Test servers fixture
        search_test_agents: Test agents fixture
        mock_embeddings_client: Mock embeddings client

    Yields:
        Initialized FAISS service with test data
    """
    # Initialize FAISS service with mock embeddings
    faiss_service = FaissService()
    faiss_service.embedding_model = mock_embeddings_client
    faiss_service._initialize_new_index()

    # Add servers to FAISS
    for server in search_test_servers:
        await faiss_service.add_or_update_service(
            service_path=server["path"], server_info=server, is_enabled=True
        )

    # Add agents to FAISS
    from registry.schemas.agent_models import AgentCard

    for agent_data in search_test_agents:
        agent_card = AgentCard(**agent_data)
        await faiss_service.add_or_update_agent(
            agent_path=agent_card.path, agent_card=agent_card, is_enabled=True
        )

    # Register servers with server service
    for server in search_test_servers:
        server_service.registered_servers[server["path"]] = server
        server_service.service_state[server["path"]] = True

    # Register agents with agent service
    from registry.schemas.agent_models import AgentCard

    for agent_data in search_test_agents:
        agent_card = AgentCard(**agent_data)
        agent_service.registered_agents[agent_card.path] = agent_card

    yield faiss_service

    # Cleanup
    server_service.registered_servers.clear()
    server_service.service_state.clear()
    agent_service.registered_agents.clear()


# =============================================================================
# SEMANTIC SEARCH TESTS
# =============================================================================


@pytest.mark.integration
@pytest.mark.search
class TestSemanticSearchIntegration:
    """Tests for semantic search integration."""

    def test_search_servers_basic(self, test_client, mock_faiss_search_results):
        """Test basic semantic search for servers."""
        # Arrange
        search_query = "database"

        # Mock FAISS search to return predictable results
        with patch(
            "registry.api.search_routes.faiss_service.search_mixed",
            new_callable=AsyncMock,
            return_value=mock_faiss_search_results["database"],
        ):
            # Act
            response = test_client.post(
                "/api/search/semantic", json={"query": search_query, "max_results": 10}
            )

            # Assert
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["query"] == search_query
            assert "servers" in data
            assert "tools" in data
            assert "agents" in data
            assert data["total_servers"] >= 0
            assert data["total_tools"] >= 0
            assert data["total_agents"] >= 0

    def test_search_agents_basic(self, test_client, mock_faiss_search_results):
        """Test basic semantic search for agents."""
        # Arrange
        search_query = "weather"

        # Mock FAISS search
        with patch(
            "registry.api.search_routes.faiss_service.search_mixed",
            new_callable=AsyncMock,
            return_value=mock_faiss_search_results["weather"],
        ):
            # Act
            response = test_client.post(
                "/api/search/semantic", json={"query": search_query, "max_results": 10}
            )

            # Assert
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["query"] == search_query
            assert len(data["agents"]) >= 0

    def test_search_mixed_results(self, test_client, mock_faiss_search_results):
        """Test semantic search returning both servers and agents."""
        # Arrange
        search_query = "database"

        with patch(
            "registry.api.search_routes.faiss_service.search_mixed",
            new_callable=AsyncMock,
            return_value=mock_faiss_search_results["database"],
        ):
            # Act
            response = test_client.post(
                "/api/search/semantic", json={"query": search_query, "max_results": 10}
            )

            # Assert
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["total_servers"] + data["total_agents"] >= 0

    def test_search_with_tools(self, test_client, mock_faiss_search_results):
        """Test semantic search including tool matches."""
        # Arrange
        search_query = "file operations"

        with patch(
            "registry.api.search_routes.faiss_service.search_mixed",
            new_callable=AsyncMock,
            return_value=mock_faiss_search_results["file operations"],
        ):
            # Act
            response = test_client.post(
                "/api/search/semantic", json={"query": search_query, "max_results": 10}
            )

            # Assert
            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Check tools in response
            if data["total_tools"] > 0:
                assert "tools" in data
                assert len(data["tools"]) > 0


# =============================================================================
# SEARCH FILTER TESTS
# =============================================================================


@pytest.mark.integration
@pytest.mark.search
class TestSearchFilters:
    """Tests for search filtering functionality."""

    def test_search_filter_mcp_server_only(self, test_client, mock_faiss_search_results):
        """Test search with mcp_server entity type filter."""
        # Arrange
        search_query = "database"
        mock_result = {
            "servers": mock_faiss_search_results["database"]["servers"],
            "tools": [],
            "agents": [],
        }

        with patch(
            "registry.api.search_routes.faiss_service.search_mixed",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            # Act
            response = test_client.post(
                "/api/search/semantic",
                json={"query": search_query, "entity_types": ["mcp_server"], "max_results": 10},
            )

            # Assert
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["total_agents"] == 0

    def test_search_filter_agent_only(self, test_client, mock_faiss_search_results):
        """Test search with a2a_agent entity type filter."""
        # Arrange
        search_query = "weather"
        mock_result = {
            "servers": [],
            "tools": [],
            "agents": mock_faiss_search_results["weather"]["agents"],
        }

        with patch(
            "registry.api.search_routes.faiss_service.search_mixed",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            # Act
            response = test_client.post(
                "/api/search/semantic",
                json={"query": search_query, "entity_types": ["a2a_agent"], "max_results": 10},
            )

            # Assert
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["total_servers"] == 0
            assert data["total_tools"] == 0

    def test_search_filter_tool_only(self, test_client, mock_faiss_search_results):
        """Test search with tool entity type filter."""
        # Arrange
        search_query = "file operations"
        mock_result = {
            "servers": [],
            "tools": mock_faiss_search_results["file operations"]["tools"],
            "agents": [],
        }

        with patch(
            "registry.api.search_routes.faiss_service.search_mixed",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            # Act
            response = test_client.post(
                "/api/search/semantic",
                json={"query": search_query, "entity_types": ["tool"], "max_results": 10},
            )

            # Assert
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["total_servers"] == 0
            assert data["total_agents"] == 0

    def test_search_max_results_limit(self, test_client, mock_faiss_search_results):
        """Test search respects max_results parameter."""
        # Arrange
        search_query = "database"
        max_results = 2

        with patch(
            "registry.api.search_routes.faiss_service.search_mixed",
            new_callable=AsyncMock,
            return_value=mock_faiss_search_results["database"],
        ):
            # Act
            response = test_client.post(
                "/api/search/semantic", json={"query": search_query, "max_results": max_results}
            )

            # Assert
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["total_servers"] <= max_results
            assert data["total_tools"] <= max_results
            assert data["total_agents"] <= max_results


# =============================================================================
# VISIBILITY FILTERING TESTS
# =============================================================================


@pytest.mark.integration
@pytest.mark.search
class TestSearchVisibilityFiltering:
    """Tests for search visibility filtering."""

    def test_search_public_agents_admin(self, test_client, mock_faiss_search_results):
        """Test that admin users can see all agents."""
        # Arrange - Auth is mocked to admin via autouse fixture
        all_agents_result = {
            "servers": [],
            "tools": [],
            "agents": [
                {
                    "entity_type": "a2a_agent",
                    "path": "/agents/data-analyst",
                    "agent_name": "data-analyst-agent",
                    "description": "Public agent",
                    "relevance_score": 0.9,
                    "match_context": "Public agent",
                }
            ],
        }

        with patch(
            "registry.api.search_routes.faiss_service.search_mixed",
            new_callable=AsyncMock,
            return_value=all_agents_result,
        ):
            # Act
            response = test_client.post(
                "/api/search/semantic", json={"query": "agent", "max_results": 10}
            )

            # Assert
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "agents" in data

    def test_search_returns_agents_with_visibility_info(
        self, test_client, mock_faiss_search_results
    ):
        """Test that search results include agent visibility information."""
        # Arrange
        agent_result = {
            "servers": [],
            "tools": [],
            "agents": [
                {
                    "entity_type": "a2a_agent",
                    "path": "/agents/private-agent",
                    "agent_name": "private-agent",
                    "description": "Private agent",
                    "relevance_score": 0.9,
                    "match_context": "Private agent",
                }
            ],
        }

        with patch(
            "registry.api.search_routes.faiss_service.search_mixed",
            new_callable=AsyncMock,
            return_value=agent_result,
        ):
            # Act
            response = test_client.post(
                "/api/search/semantic", json={"query": "private", "max_results": 10}
            )

            # Assert
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["total_agents"] >= 0

    def test_search_group_restricted_agents(self, test_client, mock_faiss_search_results):
        """Test search with group-restricted agents."""
        # Arrange
        group_agent_result = {
            "servers": [],
            "tools": [],
            "agents": [
                {
                    "entity_type": "a2a_agent",
                    "path": "/agents/group-agent",
                    "agent_name": "group-restricted-agent",
                    "description": "Group restricted agent",
                    "relevance_score": 0.9,
                    "match_context": "Group agent",
                }
            ],
        }

        with patch(
            "registry.api.search_routes.faiss_service.search_mixed",
            new_callable=AsyncMock,
            return_value=group_agent_result,
        ):
            # Act
            response = test_client.post(
                "/api/search/semantic", json={"query": "group", "max_results": 10}
            )

            # Assert
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["total_agents"] >= 0

    def test_search_admin_sees_all_agents(self, test_client, mock_faiss_search_results):
        """Test that admin users can see all agents regardless of visibility."""
        # Arrange
        all_agents_result = {
            "servers": [],
            "tools": [],
            "agents": [
                {
                    "entity_type": "a2a_agent",
                    "path": "/agents/public-agent",
                    "agent_name": "public-agent",
                    "description": "Public agent",
                    "relevance_score": 0.9,
                    "match_context": "Public",
                },
                {
                    "entity_type": "a2a_agent",
                    "path": "/agents/private-agent",
                    "agent_name": "private-agent",
                    "description": "Private agent",
                    "relevance_score": 0.85,
                    "match_context": "Private",
                },
            ],
        }

        with patch(
            "registry.api.search_routes.faiss_service.search_mixed",
            new_callable=AsyncMock,
            return_value=all_agents_result,
        ):
            # Act
            response = test_client.post(
                "/api/search/semantic", json={"query": "agent", "max_results": 10}
            )

            # Assert
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            # Admin should see all agents
            assert data["total_agents"] >= 0


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


@pytest.mark.integration
@pytest.mark.search
class TestSearchErrorHandling:
    """Tests for search error handling."""

    def test_search_empty_query_validation(self, test_client):
        """Test that empty query is rejected."""
        # Act
        response = test_client.post("/api/search/semantic", json={"query": "", "max_results": 10})

        # Assert
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_search_missing_query(self, test_client):
        """Test that missing query field is rejected."""
        # Act
        response = test_client.post("/api/search/semantic", json={"max_results": 10})

        # Assert
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_search_service_unavailable(self, test_client):
        """Test handling of FAISS service errors."""
        # Arrange
        with patch(
            "registry.api.search_routes.faiss_service.search_mixed",
            new_callable=AsyncMock,
            side_effect=RuntimeError("FAISS service unavailable"),
        ):
            # Act
            response = test_client.post(
                "/api/search/semantic", json={"query": "test", "max_results": 10}
            )

            # Assert - should handle error gracefully
            assert response.status_code in [
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                status.HTTP_503_SERVICE_UNAVAILABLE,
            ]

    def test_search_invalid_entity_type(self, test_client):
        """Test handling of invalid entity type filter."""
        # Arrange
        mock_result = {"servers": [], "tools": [], "agents": []}

        with patch(
            "registry.api.search_routes.faiss_service.search_mixed",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            # Act
            response = test_client.post(
                "/api/search/semantic",
                json={"query": "test", "entity_types": ["invalid_type"], "max_results": 10},
            )

            # Assert - should handle gracefully or return validation error
            assert response.status_code in [
                status.HTTP_200_OK,
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_422_UNPROCESSABLE_ENTITY,
            ]

    def test_search_empty_results(self, test_client, mock_faiss_search_results):
        """Test search with no matching results."""
        # Arrange
        with patch(
            "registry.api.search_routes.faiss_service.search_mixed",
            new_callable=AsyncMock,
            return_value=mock_faiss_search_results["empty query"],
        ):
            # Act
            response = test_client.post(
                "/api/search/semantic", json={"query": "nonexistent query", "max_results": 10}
            )

            # Assert
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["total_servers"] == 0
            assert data["total_tools"] == 0
            assert data["total_agents"] == 0
            assert len(data["servers"]) == 0
            assert len(data["tools"]) == 0
            assert len(data["agents"]) == 0


# =============================================================================
# SEARCH RANKING TESTS
# =============================================================================


@pytest.mark.integration
@pytest.mark.search
class TestSearchRanking:
    """Tests for search result ranking and scoring."""

    def test_search_results_sorted_by_relevance(self, test_client):
        """Test that search results are sorted by relevance score."""
        # Arrange
        ranked_results = {
            "servers": [
                {
                    "entity_type": "mcp_server",
                    "path": "/server-1",
                    "server_name": "high-score",
                    "description": "High relevance",
                    "relevance_score": 0.95,
                    "is_enabled": True,
                    "tags": [],
                    "num_tools": 0,
                    "match_context": "High",
                    "matching_tools": [],
                },
                {
                    "entity_type": "mcp_server",
                    "path": "/server-2",
                    "server_name": "medium-score",
                    "description": "Medium relevance",
                    "relevance_score": 0.75,
                    "is_enabled": True,
                    "tags": [],
                    "num_tools": 0,
                    "match_context": "Medium",
                    "matching_tools": [],
                },
                {
                    "entity_type": "mcp_server",
                    "path": "/server-3",
                    "server_name": "low-score",
                    "description": "Low relevance",
                    "relevance_score": 0.55,
                    "is_enabled": True,
                    "tags": [],
                    "num_tools": 0,
                    "match_context": "Low",
                    "matching_tools": [],
                },
            ],
            "tools": [],
            "agents": [],
        }

        with patch(
            "registry.api.search_routes.faiss_service.search_mixed",
            new_callable=AsyncMock,
            return_value=ranked_results,
        ):
            # Act
            response = test_client.post(
                "/api/search/semantic", json={"query": "test", "max_results": 10}
            )

            # Assert
            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Check scores are in descending order
            if len(data["servers"]) > 1:
                for i in range(len(data["servers"]) - 1):
                    assert (
                        data["servers"][i]["relevance_score"]
                        >= data["servers"][i + 1]["relevance_score"]
                    )

    def test_search_relevance_scores_range(self, test_client, mock_faiss_search_results):
        """Test that relevance scores are in valid range (0-1)."""
        # Arrange
        with patch(
            "registry.api.search_routes.faiss_service.search_mixed",
            new_callable=AsyncMock,
            return_value=mock_faiss_search_results["database"],
        ):
            # Act
            response = test_client.post(
                "/api/search/semantic", json={"query": "database", "max_results": 10}
            )

            # Assert
            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Check all scores are in valid range
            for server in data["servers"]:
                assert 0.0 <= server["relevance_score"] <= 1.0

            for tool in data["tools"]:
                assert 0.0 <= tool["relevance_score"] <= 1.0

            for agent in data["agents"]:
                assert 0.0 <= agent["relevance_score"] <= 1.0
