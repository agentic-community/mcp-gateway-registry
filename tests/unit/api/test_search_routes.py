"""
Unit tests for registry/api/search_routes.py

Tests all components of the semantic search API including:
- Pydantic model validation
- User access control helper functions
- Semantic search endpoint with various scenarios
- Error handling and edge cases
"""

import logging
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException, Request
from pydantic import ValidationError

from registry.api.search_routes import (
    AgentSearchResult,
    MatchingToolResult,
    SemanticSearchRequest,
    SemanticSearchResponse,
    ServerSearchResult,
    ToolSearchResult,
    _user_can_access_agent,
    _user_can_access_server,
    semantic_search,
)
from tests.fixtures.factories import AgentCardFactory

logger = logging.getLogger(__name__)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_http_request():
    """Mock HTTP request for testing."""
    mock_request = Mock(spec=Request)
    mock_request.state = Mock()
    return mock_request


@pytest.fixture
def mock_search_repo():
    """Mock search repository for testing."""
    mock = AsyncMock()
    yield mock


@pytest.fixture
def mock_server_service():
    """Mock server service for testing."""
    with patch("registry.api.search_routes.server_service") as mock:
        yield mock


@pytest.fixture
def mock_agent_service():
    """Mock agent service for testing."""
    with patch("registry.api.search_routes.agent_service") as mock:
        yield mock


@pytest.fixture(autouse=True)
def mock_server_and_agent_service_db_calls():
    """Mock server_service and agent_service to avoid MongoDB connections in unit tests.

    This is an autouse fixture that automatically patches both services
    for ALL tests in this file to prevent slow MongoDB connection attempts.
    """

    # Mock get_server_info method to return server info based on path
    async def get_server_info(path: str):
        # Return mock server info for known paths
        if "currenttime" in path or "Time Server" in path:
            return {
                "path": path,
                "server_name": "Time Server",
                "description": "Time utilities",
                "tags": ["time"],
                "num_tools": 1,
            }
        elif "restricted" in path:
            return {
                "path": path,
                "server_name": "restricted",
                "description": "Restricted server",
                "tags": [],
                "num_tools": 0,
            }
        elif "mcpgw" in path:
            return {
                "path": path,
                "server_name": "mcpgw",
                "description": "MCP Gateway",
                "tags": ["gateway"],
                "num_tools": 5,
            }
        return None

    # Mock get_agent_info method to return agent info based on path
    async def get_agent_info(path: str):
        # Return mock agent card for known paths
        from registry.schemas.agent_models import AgentCard
        from tests.fixtures.factories import AgentCardFactory

        if "code-reviewer" in path:
            return AgentCardFactory(path=path, name="code-reviewer", visibility="public")
        elif "test-agent" in path:
            return AgentCardFactory(path=path, name="test-agent", visibility="public")
        elif "data-analyst" in path:
            return AgentCardFactory(path=path, name="data-analyst", visibility="public")
        return None

    # Patch both service methods
    with (
        patch(
            "registry.api.search_routes.server_service.get_server_info",
            new=AsyncMock(side_effect=get_server_info),
        ),
        patch(
            "registry.api.search_routes.agent_service.get_agent_info",
            new=AsyncMock(side_effect=get_agent_info),
        ),
    ):
        yield


@pytest.fixture
def admin_user_context() -> dict[str, Any]:
    """Create admin user context for testing."""
    return {
        "username": "admin",
        "is_admin": True,
        "groups": ["mcp-registry-admin"],
        "scopes": ["mcp-servers-unrestricted/read"],
        "accessible_servers": ["*"],
        "accessible_agents": ["all"],
    }


@pytest.fixture
def regular_user_context() -> dict[str, Any]:
    """Create regular user context with specific access."""
    return {
        "username": "regular_user",
        "is_admin": False,
        "groups": ["registry-users-lob1"],
        "scopes": ["registry-users-lob1"],
        "accessible_servers": ["currenttime", "mcpgw"],
        "accessible_agents": ["/agents/code-reviewer", "/agents/test-agent"],
    }


@pytest.fixture
def restricted_user_context() -> dict[str, Any]:
    """Create user context with no access."""
    return {
        "username": "restricted_user",
        "is_admin": False,
        "groups": [],
        "scopes": [],
        "accessible_servers": [],
        "accessible_agents": [],
    }


@pytest.fixture
def user_with_all_servers_context() -> dict[str, Any]:
    """Create user context with 'all' access to servers."""
    return {
        "username": "all_servers_user",
        "is_admin": False,
        "groups": ["registry-users"],
        "scopes": ["registry-users"],
        "accessible_servers": ["all"],
        "accessible_agents": [],
    }


@pytest.fixture
def sample_faiss_search_results() -> dict[str, list[dict[str, Any]]]:
    """Create sample FAISS search results."""
    return {
        "servers": [
            {
                "path": "/servers/currenttime",
                "server_name": "currenttime",
                "description": "Get current time in various timezones",
                "tags": ["time", "timezone"],
                "num_tools": 1,
                "is_enabled": True,
                "relevance_score": 0.95,
                "match_context": "time timezone utilities",
                "matching_tools": [
                    {
                        "tool_name": "get_current_time",
                        "description": "Get current time for timezone",
                        "relevance_score": 0.92,
                        "match_context": "current time timezone",
                    }
                ],
            },
            {
                "path": "/servers/weather",
                "server_name": "weather",
                "description": "Get weather information",
                "tags": ["weather", "forecast"],
                "num_tools": 2,
                "is_enabled": True,
                "relevance_score": 0.75,
                "match_context": "weather data",
                "matching_tools": [],
            },
        ],
        "tools": [
            {
                "server_path": "/servers/currenttime",
                "server_name": "currenttime",
                "tool_name": "get_current_time",
                "description": "Get current time for timezone",
                "relevance_score": 0.92,
                "match_context": "current time timezone",
            },
            {
                "server_path": "/servers/weather",
                "server_name": "weather",
                "tool_name": "get_forecast",
                "description": "Get weather forecast",
                "relevance_score": 0.85,
                "match_context": "weather forecast data",
            },
        ],
        "agents": [
            {
                "path": "/agents/code-reviewer",
                "relevance_score": 0.88,
                "match_context": "code review analysis",
                "agent_card": {
                    "name": "code-reviewer",
                    "description": "Review code for best practices",
                    "tags": ["code", "review"],
                    "skills": [{"name": "Code Review"}],
                    "visibility": "public",
                    "is_enabled": True,
                },
            },
            {
                "path": "/agents/test-agent",
                "relevance_score": 0.82,
                "match_context": "test automation",
                "agent_card": {
                    "name": "test-agent",
                    "description": "Test automation agent",
                    "tags": ["test", "automation"],
                    "skills": [{"name": "Test Generation"}],
                    "visibility": "public",
                    "is_enabled": True,
                },
            },
        ],
    }


# =============================================================================
# TEST: Pydantic Model Validation
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.search
class TestPydanticModels:
    """Tests for Pydantic model validation."""

    def test_matching_tool_result_valid(self):
        """Test MatchingToolResult with valid data."""
        # Arrange & Act
        tool = MatchingToolResult(
            tool_name="test_tool",
            description="A test tool",
            relevance_score=0.85,
            match_context="test context",
        )

        # Assert
        assert tool.tool_name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.relevance_score == 0.85
        assert tool.match_context == "test context"

    def test_matching_tool_result_defaults(self):
        """Test MatchingToolResult with default values."""
        # Arrange & Act
        tool = MatchingToolResult(tool_name="test_tool")

        # Assert
        assert tool.tool_name == "test_tool"
        assert tool.description is None
        assert tool.relevance_score == 0.0
        assert tool.match_context is None

    def test_matching_tool_result_score_validation(self):
        """Test MatchingToolResult score must be between 0 and 1."""
        # Act & Assert - score too high
        with pytest.raises(ValidationError) as exc_info:
            MatchingToolResult(tool_name="test", relevance_score=1.5)
        assert "relevance_score" in str(exc_info.value)

        # Act & Assert - negative score
        with pytest.raises(ValidationError) as exc_info:
            MatchingToolResult(tool_name="test", relevance_score=-0.1)
        assert "relevance_score" in str(exc_info.value)

    def test_server_search_result_valid(self):
        """Test ServerSearchResult with valid data."""
        # Arrange & Act
        server = ServerSearchResult(
            path="/servers/test",
            server_name="test-server",
            description="Test server",
            tags=["test"],
            num_tools=5,
            is_enabled=True,
            relevance_score=0.9,
            match_context="test context",
            matching_tools=[MatchingToolResult(tool_name="tool1", relevance_score=0.8)],
        )

        # Assert
        assert server.path == "/servers/test"
        assert server.server_name == "test-server"
        assert server.num_tools == 5
        assert len(server.matching_tools) == 1

    def test_server_search_result_defaults(self):
        """Test ServerSearchResult with default values."""
        # Arrange & Act
        server = ServerSearchResult(
            path="/servers/test",
            server_name="test-server",
            relevance_score=0.9,
        )

        # Assert
        assert server.tags == []
        assert server.num_tools == 0
        assert server.is_enabled is False
        assert server.matching_tools == []

    def test_tool_search_result_valid(self):
        """Test ToolSearchResult with valid data."""
        # Arrange & Act
        tool = ToolSearchResult(
            server_path="/servers/test",
            server_name="test-server",
            tool_name="test_tool",
            description="Test tool",
            relevance_score=0.85,
            match_context="test context",
        )

        # Assert
        assert tool.server_path == "/servers/test"
        assert tool.server_name == "test-server"
        assert tool.tool_name == "test_tool"

    def test_agent_search_result_valid(self):
        """Test AgentSearchResult with valid data."""
        # Arrange & Act
        agent = AgentSearchResult(
            path="/agents/test",
            relevance_score=0.88,
            match_context="test context",
            agent_card={
                "name": "test-agent",
                "description": "Test agent",
                "tags": ["test"],
                "skills": [{"name": "skill1"}, {"name": "skill2"}],
                "trust_level": "verified",
                "visibility": "public",
                "is_enabled": True,
            },
        )

        # Assert
        assert agent.path == "/agents/test"
        assert agent.agent_card["name"] == "test-agent"
        assert len(agent.agent_card["skills"]) == 2

    def test_agent_search_result_defaults(self):
        """Test AgentSearchResult with default values."""
        # Arrange & Act
        agent = AgentSearchResult(
            path="/agents/test",
            relevance_score=0.8,
            agent_card={
                "name": "test-agent",
                "tags": [],
                "skills": [],
                "is_enabled": False,
            },
        )

        # Assert
        assert agent.agent_card["tags"] == []
        assert agent.agent_card["skills"] == []
        assert agent.agent_card["is_enabled"] is False

    def test_semantic_search_request_valid(self):
        """Test SemanticSearchRequest with valid data."""
        # Arrange & Act
        request = SemanticSearchRequest(
            query="test query",
            entity_types=["mcp_server", "tool"],
            max_results=20,
        )

        # Assert
        assert request.query == "test query"
        assert len(request.entity_types) == 2
        assert request.max_results == 20

    def test_semantic_search_request_defaults(self):
        """Test SemanticSearchRequest with default values."""
        # Arrange & Act
        request = SemanticSearchRequest(query="test query")

        # Assert
        assert request.query == "test query"
        assert request.entity_types is None
        assert request.max_results == 10

    def test_semantic_search_request_query_length_validation(self):
        """Test SemanticSearchRequest query length constraints."""
        # Act & Assert - empty query
        with pytest.raises(ValidationError) as exc_info:
            SemanticSearchRequest(query="")
        assert "query" in str(exc_info.value)

        # Act & Assert - query too long
        with pytest.raises(ValidationError) as exc_info:
            SemanticSearchRequest(query="x" * 513)
        assert "query" in str(exc_info.value)

    def test_semantic_search_request_max_results_validation(self):
        """Test SemanticSearchRequest max_results constraints."""
        # Act & Assert - max_results too low
        with pytest.raises(ValidationError) as exc_info:
            SemanticSearchRequest(query="test", max_results=0)
        assert "max_results" in str(exc_info.value)

        # Act & Assert - max_results too high
        with pytest.raises(ValidationError) as exc_info:
            SemanticSearchRequest(query="test", max_results=51)
        assert "max_results" in str(exc_info.value)

    def test_semantic_search_request_entity_types_validation(self):
        """Test SemanticSearchRequest entity_types must be valid."""
        # Act & Assert - invalid entity type
        with pytest.raises(ValidationError) as exc_info:
            SemanticSearchRequest(query="test", entity_types=["invalid_type"])
        assert "entity_types" in str(exc_info.value)

    def test_semantic_search_response_valid(self):
        """Test SemanticSearchResponse with valid data."""
        # Arrange & Act
        response = SemanticSearchResponse(
            query="test query",
            servers=[
                ServerSearchResult(
                    path="/servers/test",
                    server_name="test",
                    relevance_score=0.9,
                )
            ],
            tools=[],
            agents=[],
            total_servers=1,
            total_tools=0,
            total_agents=0,
        )

        # Assert
        assert response.query == "test query"
        assert len(response.servers) == 1
        assert response.total_servers == 1

    def test_semantic_search_response_defaults(self):
        """Test SemanticSearchResponse with default values."""
        # Arrange & Act
        response = SemanticSearchResponse(query="test query")

        # Assert
        assert response.servers == []
        assert response.tools == []
        assert response.agents == []
        assert response.total_servers == 0
        assert response.total_tools == 0
        assert response.total_agents == 0


# =============================================================================
# TEST: _user_can_access_server Helper Function
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.search
class TestUserCanAccessServer:
    """Tests for _user_can_access_server helper function."""

    @pytest.mark.asyncio
    async def test_admin_user_can_access_any_server(self):
        """Test admin user can access any server."""
        # Arrange
        user_context = {"is_admin": True}

        # Act
        result = await _user_can_access_server("/servers/test", "test-server", user_context)

        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_user_with_all_accessible_servers(self):
        """Test user with 'all' in accessible_servers can access any server."""
        # Arrange
        user_context = {
            "is_admin": False,
            "accessible_servers": ["all"],
        }

        # Act
        result = await _user_can_access_server("/servers/test", "test-server", user_context)

        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_user_with_no_accessible_servers(self):
        """Test user with empty accessible_servers cannot access."""
        # Arrange
        user_context = {
            "is_admin": False,
            "accessible_servers": [],
        }

        # Act
        result = await _user_can_access_server("/servers/test", "test-server", user_context)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_user_with_none_accessible_servers(self):
        """Test user with None accessible_servers cannot access."""
        # Arrange
        user_context = {
            "is_admin": False,
            "accessible_servers": None,
        }

        # Act
        result = await _user_can_access_server("/servers/test", "test-server", user_context)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_user_can_access_via_server_service(self, mock_server_service):
        """Test user can access via server_service path validation."""
        # Arrange
        mock_server_service.user_can_access_server_path = AsyncMock(return_value=True)
        user_context = {
            "is_admin": False,
            "accessible_servers": ["server1"],
        }

        # Act
        result = await _user_can_access_server("/servers/server1", "server1", user_context)

        # Assert
        assert result is True
        mock_server_service.user_can_access_server_path.assert_called_once_with(
            "/servers/server1", ["server1"]
        )

    @pytest.mark.asyncio
    async def test_user_can_access_via_technical_name(self, mock_server_service):
        """Test user can access via technical name match."""
        # Arrange
        # Note: technical_name is extracted as path.strip("/") which gives
        # "servers/currenttime", not "currenttime". Need server_service to handle.
        mock_server_service.user_can_access_server_path = AsyncMock(return_value=True)
        user_context = {
            "is_admin": False,
            "accessible_servers": ["currenttime"],
        }

        # Act
        result = await _user_can_access_server("/servers/currenttime", "Time Server", user_context)

        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_user_can_access_via_server_name(self):
        """Test user can access via server name match."""
        # Arrange
        user_context = {
            "is_admin": False,
            "accessible_servers": ["Time Server"],
        }

        # Act
        result = await _user_can_access_server("/servers/currenttime", "Time Server", user_context)

        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_user_cannot_access_unlisted_server(self):
        """Test user cannot access server not in accessible list."""
        # Arrange
        user_context = {
            "is_admin": False,
            "accessible_servers": ["server1", "server2"],
        }

        # Act
        result = await _user_can_access_server("/servers/server3", "server3", user_context)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_server_service_exception_fallback_to_name_check(self, mock_server_service):
        """Test fallback to name check when server_service raises exception."""
        # Arrange
        mock_server_service.user_can_access_server_path = AsyncMock(
            side_effect=Exception("Service error")
        )
        user_context = {
            "is_admin": False,
            "accessible_servers": ["test-server"],
        }

        # Act
        result = await _user_can_access_server("/servers/test", "test-server", user_context)

        # Assert
        assert result is True


# =============================================================================
# TEST: _user_can_access_agent Helper Function
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.search
class TestUserCanAccessAgent:
    """Tests for _user_can_access_agent helper function."""

    @pytest.mark.asyncio
    async def test_admin_user_can_access_any_agent(self, mock_agent_service):
        """Test admin user can access any agent."""
        # Arrange
        mock_agent = AgentCardFactory(visibility="internal")
        mock_agent_service.get_agent_info = AsyncMock(return_value=mock_agent)
        user_context = {"is_admin": True}

        # Act
        result = await _user_can_access_agent("/agents/test", user_context)

        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_user_without_agent_in_accessible_list(self, mock_agent_service):
        """Test user cannot access agent not in accessible_agents list."""
        # Arrange
        user_context = {
            "is_admin": False,
            "accessible_agents": ["/agents/other"],
        }

        # Act
        result = await _user_can_access_agent("/agents/test", user_context)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_user_with_all_can_access_public_agent(self, mock_agent_service):
        """Test user with 'all' can access public agents."""
        # Arrange
        mock_agent = AgentCardFactory(visibility="public")
        mock_agent_service.get_agent_info = AsyncMock(return_value=mock_agent)
        user_context = {
            "is_admin": False,
            "accessible_agents": ["all"],
        }

        # Act
        result = await _user_can_access_agent("/agents/test", user_context)

        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_agent_not_found_returns_false(self, mock_agent_service):
        """Test returns False when agent not found."""
        # Arrange
        mock_agent_service.get_agent_info = AsyncMock(return_value=None)
        user_context = {
            "is_admin": False,
            "accessible_agents": ["all"],
        }

        # Act
        result = await _user_can_access_agent("/agents/test", user_context)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_public_agent_accessible_to_authorized_user(self, mock_agent_service):
        """Test public agent is accessible to user in accessible list."""
        # Arrange
        mock_agent = AgentCardFactory(visibility="public")
        mock_agent_service.get_agent_info = AsyncMock(return_value=mock_agent)
        user_context = {
            "is_admin": False,
            "accessible_agents": ["/agents/test"],
        }

        # Act
        result = await _user_can_access_agent("/agents/test", user_context)

        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_internal_agent_accessible_to_owner(self, mock_agent_service):
        """Test internal agent is accessible to owner."""
        # Arrange
        mock_agent = AgentCardFactory(visibility="internal", registered_by="testuser")
        mock_agent_service.get_agent_info = AsyncMock(return_value=mock_agent)
        user_context = {
            "is_admin": False,
            "username": "testuser",
            "accessible_agents": ["/agents/test"],
        }

        # Act
        result = await _user_can_access_agent("/agents/test", user_context)

        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_internal_agent_not_accessible_to_others(self, mock_agent_service):
        """Test internal agent is not accessible to non-owners."""
        # Arrange
        mock_agent = AgentCardFactory(visibility="internal", registered_by="owner")
        mock_agent_service.get_agent_info = AsyncMock(return_value=mock_agent)
        user_context = {
            "is_admin": False,
            "username": "otheruser",
            "accessible_agents": ["/agents/test"],
        }

        # Act
        result = await _user_can_access_agent("/agents/test", user_context)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_group_restricted_agent_accessible_to_group_member(self, mock_agent_service):
        """Test group-restricted agent is accessible to group members."""
        # Arrange
        mock_agent = AgentCardFactory(
            visibility="group-restricted",
            allowed_groups=["group1", "group2"],
        )
        mock_agent_service.get_agent_info = AsyncMock(return_value=mock_agent)
        user_context = {
            "is_admin": False,
            "groups": ["group1", "group3"],
            "accessible_agents": ["/agents/test"],
        }

        # Act
        result = await _user_can_access_agent("/agents/test", user_context)

        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_group_restricted_agent_not_accessible_to_non_member(self, mock_agent_service):
        """Test group-restricted agent is not accessible to non-members."""
        # Arrange
        mock_agent = AgentCardFactory(
            visibility="group-restricted",
            allowed_groups=["group1", "group2"],
        )
        mock_agent_service.get_agent_info = AsyncMock(return_value=mock_agent)
        user_context = {
            "is_admin": False,
            "groups": ["group3"],
            "accessible_agents": ["/agents/test"],
        }

        # Act
        result = await _user_can_access_agent("/agents/test", user_context)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_unknown_visibility_returns_false(self, mock_agent_service):
        """Test unknown visibility type returns False."""
        # Arrange
        # Note: AgentCard validates visibility, so we use a Mock instead
        mock_agent = Mock()
        mock_agent.visibility = "unknown"
        mock_agent_service.get_agent_info = AsyncMock(return_value=mock_agent)
        user_context = {
            "is_admin": False,
            "accessible_agents": ["/agents/test"],
        }

        # Act
        result = await _user_can_access_agent("/agents/test", user_context)

        # Assert
        assert result is False


# =============================================================================
# TEST: semantic_search Endpoint - Success Cases
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.search
class TestSemanticSearchSuccess:
    """Tests for successful semantic search endpoint operations."""

    @pytest.mark.asyncio
    async def test_semantic_search_admin_sees_all_results(
        self,
        mock_http_request,
        mock_search_repo,
        mock_agent_service,
        admin_user_context,
        sample_faiss_search_results,
    ):
        """Test admin user sees all search results."""
        # Arrange
        mock_search_repo.search = AsyncMock(return_value=sample_faiss_search_results)
        mock_agent_service.get_agent_info.side_effect = lambda path: AgentCardFactory(
            path=path,
            name=path.split("/")[-1],
            visibility="public",
        )

        request = SemanticSearchRequest(query="test query", max_results=10)

        # Mock agent_service.get_agent_info to be async
        async def get_agent_side_effect(path):
            return AgentCardFactory(
                path=path,
                name=path.split("/")[-1],
                visibility="public",
            )

        mock_agent_service.get_agent_info = AsyncMock(side_effect=get_agent_side_effect)

        # Act
        response = await semantic_search(
            mock_http_request, request, admin_user_context, mock_search_repo
        )

        # Assert
        assert response.query == "test query"
        assert len(response.servers) == 2
        assert len(response.tools) == 2
        assert len(response.agents) == 2
        assert response.total_servers == 2
        assert response.total_tools == 2
        assert response.total_agents == 2

    @pytest.mark.asyncio
    async def test_semantic_search_filters_by_server_access(
        self,
        mock_http_request,
        mock_search_repo,
        mock_agent_service,
        regular_user_context,
        sample_faiss_search_results,
    ):
        """Test search filters servers by user access."""
        # Arrange
        mock_search_repo.search = AsyncMock(return_value=sample_faiss_search_results)

        async def get_agent_side_effect(path):
            return AgentCardFactory(path=path, visibility="public")

        mock_agent_service.get_agent_info = AsyncMock(side_effect=get_agent_side_effect)

        request = SemanticSearchRequest(query="test query")

        # Act
        response = await semantic_search(
            mock_http_request, request, regular_user_context, mock_search_repo
        )

        # Assert
        # User has access to "currenttime" but not "weather"
        assert len(response.servers) == 1
        assert response.servers[0].server_name == "currenttime"
        assert len(response.tools) == 1
        assert response.tools[0].server_name == "currenttime"

    @pytest.mark.asyncio
    async def test_semantic_search_filters_by_agent_access(
        self,
        mock_http_request,
        mock_search_repo,
        mock_agent_service,
        regular_user_context,
        sample_faiss_search_results,
    ):
        """Test search filters agents by user access."""
        # Arrange
        mock_search_repo.search = AsyncMock(return_value=sample_faiss_search_results)

        # Create mock agents with proper model_dump method
        def create_mock_agent(path, name, visibility):
            agent = AgentCardFactory(
                path=path,
                name=name,
                visibility=visibility,
            )
            return agent

        async def get_agent_info_side_effect(path):
            if path == "/agents/code-reviewer":
                return create_mock_agent(path, "code-reviewer", "public")
            elif path == "/agents/test-agent":
                return create_mock_agent(path, "test-agent", "public")
            return None

        mock_agent_service.get_agent_info = AsyncMock(side_effect=get_agent_info_side_effect)

        request = SemanticSearchRequest(query="test query")

        # Act
        response = await semantic_search(
            mock_http_request, request, regular_user_context, mock_search_repo
        )

        # Assert
        # User has access to both agents
        assert len(response.agents) == 2

    @pytest.mark.asyncio
    async def test_semantic_search_restricted_user_sees_nothing(
        self,
        mock_http_request,
        mock_search_repo,
        restricted_user_context,
        sample_faiss_search_results,
    ):
        """Test restricted user sees no results."""
        # Arrange
        mock_search_repo.search = AsyncMock(return_value=sample_faiss_search_results)

        request = SemanticSearchRequest(query="test query")

        # Act
        response = await semantic_search(
            mock_http_request, request, restricted_user_context, mock_search_repo
        )

        # Assert
        assert len(response.servers) == 0
        assert len(response.tools) == 0
        assert len(response.agents) == 0
        assert response.total_servers == 0
        assert response.total_tools == 0
        assert response.total_agents == 0

    @pytest.mark.asyncio
    async def test_semantic_search_empty_results(
        self, mock_http_request, mock_search_repo, admin_user_context
    ):
        """Test search with no results."""
        # Arrange
        mock_search_repo.search = AsyncMock(return_value={"servers": [], "tools": [], "agents": []})

        request = SemanticSearchRequest(query="nonexistent")

        # Act
        response = await semantic_search(
            mock_http_request, request, admin_user_context, mock_search_repo
        )

        # Assert
        assert response.query == "nonexistent"
        assert len(response.servers) == 0
        assert len(response.tools) == 0
        assert len(response.agents) == 0

    @pytest.mark.asyncio
    async def test_semantic_search_with_entity_type_filter(
        self,
        mock_http_request,
        mock_search_repo,
        admin_user_context,
        sample_faiss_search_results,
    ):
        """Test search with entity type filtering."""
        # Arrange
        mock_search_repo.search = AsyncMock(return_value=sample_faiss_search_results)

        request = SemanticSearchRequest(query="test query", entity_types=["mcp_server"])

        # Act
        await semantic_search(mock_http_request, request, admin_user_context, mock_search_repo)

        # Assert
        mock_search_repo.search.assert_called_once_with(
            query="test query",
            entity_types=["mcp_server"],
            max_results=10,
        )

    @pytest.mark.asyncio
    async def test_semantic_search_with_custom_max_results(
        self,
        mock_http_request,
        mock_search_repo,
        admin_user_context,
        sample_faiss_search_results,
    ):
        """Test search with custom max_results."""
        # Arrange
        mock_search_repo.search = AsyncMock(return_value=sample_faiss_search_results)

        request = SemanticSearchRequest(query="test query", max_results=25)

        # Act
        await semantic_search(mock_http_request, request, admin_user_context, mock_search_repo)

        # Assert
        mock_search_repo.search.assert_called_once_with(
            query="test query",
            entity_types=None,
            max_results=25,
        )

    @pytest.mark.asyncio
    async def test_semantic_search_strips_query(
        self, mock_http_request, mock_search_repo, admin_user_context
    ):
        """Test search strips whitespace from query."""
        # Arrange
        mock_search_repo.search = AsyncMock(return_value={"servers": [], "tools": [], "agents": []})

        request = SemanticSearchRequest(query="  test query  ")

        # Act
        response = await semantic_search(
            mock_http_request, request, admin_user_context, mock_search_repo
        )

        # Assert
        assert response.query == "test query"

    @pytest.mark.asyncio
    async def test_semantic_search_server_with_matching_tools(
        self,
        mock_http_request,
        mock_search_repo,
        admin_user_context,
        sample_faiss_search_results,
    ):
        """Test server result includes matching tools."""
        # Arrange
        mock_search_repo.search = AsyncMock(return_value=sample_faiss_search_results)

        request = SemanticSearchRequest(query="time")

        # Act
        response = await semantic_search(
            mock_http_request, request, admin_user_context, mock_search_repo
        )

        # Assert
        currenttime_server = next(s for s in response.servers if s.server_name == "currenttime")
        assert len(currenttime_server.matching_tools) == 1
        assert currenttime_server.matching_tools[0].tool_name == "get_current_time"


# =============================================================================
# TEST: semantic_search Endpoint - Error Handling
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.search
class TestSemanticSearchErrorHandling:
    """Tests for semantic search error handling."""

    @pytest.mark.asyncio
    async def test_semantic_search_value_error_returns_400(
        self, mock_http_request, mock_search_repo, admin_user_context
    ):
        """Test ValueError from search service returns 400."""
        # Arrange
        mock_search_repo.search = AsyncMock(side_effect=ValueError("Invalid search parameters"))

        request = SemanticSearchRequest(query="test")

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await semantic_search(mock_http_request, request, admin_user_context, mock_search_repo)

        assert exc_info.value.status_code == 400
        assert "Invalid search parameters" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_semantic_search_runtime_error_returns_503(
        self, mock_http_request, mock_search_repo, admin_user_context
    ):
        """Test RuntimeError from search service returns 503."""
        # Arrange
        mock_search_repo.search = AsyncMock(side_effect=RuntimeError("Search index not available"))

        request = SemanticSearchRequest(query="test")

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await semantic_search(mock_http_request, request, admin_user_context, mock_search_repo)

        assert exc_info.value.status_code == 503
        assert "temporarily unavailable" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_semantic_search_handles_missing_agent_gracefully(
        self,
        mock_http_request,
        mock_search_repo,
        mock_agent_service,
        admin_user_context,
    ):
        """Test search handles missing agent gracefully."""
        # Arrange
        faiss_results = {
            "servers": [],
            "tools": [],
            "agents": [
                {
                    "path": "/agents/missing",
                    "relevance_score": 0.8,
                    "agent_card": {
                        "name": "missing-agent",
                        "visibility": "public",
                    },
                }
            ],
        }
        mock_search_repo.search = AsyncMock(return_value=faiss_results)
        mock_agent_service.get_agent_info = AsyncMock(return_value=None)

        request = SemanticSearchRequest(query="test")

        # Act
        response = await semantic_search(
            mock_http_request, request, admin_user_context, mock_search_repo
        )

        # Assert
        # Note: Current implementation uses fallback data from FAISS results
        # even when agent_service.get_agent_info returns None, so agent is included
        assert len(response.agents) == 1
        assert response.agents[0].agent_card["name"] == "missing-agent"

    @pytest.mark.asyncio
    async def test_semantic_search_handles_agent_without_path(
        self,
        mock_http_request,
        mock_search_repo,
        admin_user_context,
    ):
        """Test search handles agent result without path."""
        # Arrange
        faiss_results = {
            "servers": [],
            "tools": [],
            "agents": [
                {
                    "path": "",
                    "agent_name": "no-path-agent",
                    "relevance_score": 0.8,
                }
            ],
        }
        mock_search_repo.search = AsyncMock(return_value=faiss_results)

        request = SemanticSearchRequest(query="test")

        # Act
        response = await semantic_search(
            mock_http_request, request, admin_user_context, mock_search_repo
        )

        # Assert
        # Agent should be filtered out since it has no path
        assert len(response.agents) == 0


# =============================================================================
# TEST: semantic_search Endpoint - Agent Field Extraction
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.search
class TestSemanticSearchAgentFieldExtraction:
    """Tests for agent field extraction in search results."""

    @pytest.mark.asyncio
    async def test_semantic_search_extracts_agent_fields_from_card(
        self,
        mock_http_request,
        mock_search_repo,
        mock_agent_service,
        admin_user_context,
    ):
        """Test agent fields are extracted from agent card."""
        # Arrange
        # Create a mock with proper model_dump method
        mock_agent = Mock()
        mock_agent.model_dump.return_value = {
            "name": "Test Agent",
            "description": "Test description",
            "tags": ["tag1", "tag2"],
            "skills": [{"name": "Skill 1"}, {"name": "Skill 2"}],
            "trust_level": "verified",
            "visibility": "public",
            "is_enabled": True,
        }
        mock_agent_service.get_agent_info = AsyncMock(return_value=mock_agent)

        faiss_results = {
            "servers": [],
            "tools": [],
            "agents": [
                {
                    "path": "/agents/test",
                    "relevance_score": 0.9,
                    "match_context": "test context",
                    "agent_card": {"name": "old-name", "visibility": "public"},
                }
            ],
        }
        mock_search_repo.search = AsyncMock(return_value=faiss_results)

        request = SemanticSearchRequest(query="test")

        # Act
        response = await semantic_search(
            mock_http_request, request, admin_user_context, mock_search_repo
        )

        # Assert
        assert len(response.agents) == 1
        agent = response.agents[0]
        # Agent card data comes from the mock agent service's model_dump
        assert agent.agent_card["name"] == "Test Agent"
        assert agent.agent_card["description"] == "Test description"
        assert agent.agent_card["tags"] == ["tag1", "tag2"]
        assert agent.agent_card["skills"] == [{"name": "Skill 1"}, {"name": "Skill 2"}]
        assert agent.agent_card["trust_level"] == "verified"
        assert agent.agent_card["visibility"] == "public"
        assert agent.agent_card["is_enabled"] is True

    @pytest.mark.asyncio
    async def test_semantic_search_handles_skills_as_strings(
        self,
        mock_http_request,
        mock_search_repo,
        mock_agent_service,
        admin_user_context,
    ):
        """Test agent skills are handled when they are strings."""
        # Arrange
        mock_agent = Mock()
        mock_agent.model_dump.return_value = {
            "name": "Test Agent",
            "description": "Test",
            "tags": [],
            "skills": ["Skill 1", "Skill 2"],  # Skills as strings
            "trust_level": "unverified",
            "visibility": "public",
            "is_enabled": True,
        }
        mock_agent_service.get_agent_info = AsyncMock(return_value=mock_agent)

        faiss_results = {
            "servers": [],
            "tools": [],
            "agents": [
                {
                    "path": "/agents/test",
                    "relevance_score": 0.9,
                    "agent_card": {"name": "Test Agent", "visibility": "public"},
                }
            ],
        }
        mock_search_repo.search = AsyncMock(return_value=faiss_results)

        request = SemanticSearchRequest(query="test")

        # Act
        response = await semantic_search(
            mock_http_request, request, admin_user_context, mock_search_repo
        )

        # Assert
        assert len(response.agents) == 1
        assert response.agents[0].agent_card["skills"] == ["Skill 1", "Skill 2"]

    @pytest.mark.asyncio
    async def test_semantic_search_fallback_to_faiss_agent_data(
        self,
        mock_http_request,
        mock_search_repo,
        mock_agent_service,
        admin_user_context,
    ):
        """Test fallback to search data when agent card not found."""
        # Arrange
        mock_agent_service.get_agent_info = AsyncMock(return_value=None)

        faiss_results = {
            "servers": [],
            "tools": [],
            "agents": [
                {
                    "path": "/agents/test",
                    "relevance_score": 0.9,
                    "agent_card": {
                        "name": "Test Agent",
                        "description": "From search",
                        "tags": ["from_card"],
                        "skills": [{"name": "Card Skill"}],
                        "visibility": "public",
                    },
                }
            ],
        }
        mock_search_repo.search = AsyncMock(return_value=faiss_results)

        request = SemanticSearchRequest(query="test")

        # Act
        response = await semantic_search(
            mock_http_request, request, admin_user_context, mock_search_repo
        )

        # Assert
        assert len(response.agents) == 1
        agent = response.agents[0]
        # Should use fallback data from agent_card in FAISS results
        assert agent.agent_card["tags"] == ["from_card"]
        assert agent.agent_card["skills"] == [{"name": "Card Skill"}]

    @pytest.mark.asyncio
    async def test_semantic_search_preserves_skills_structure(
        self,
        mock_http_request,
        mock_search_repo,
        mock_agent_service,
        admin_user_context,
    ):
        """Test skills structure is preserved in agent_card."""
        # Arrange
        mock_agent = Mock()
        mock_agent.model_dump.return_value = {
            "name": "Test Agent",
            "description": "Test",
            "tags": [],
            "skills": [{"name": "Skill 1"}, {"name": None}, {"name": "Skill 2"}],
            "trust_level": "unverified",
            "visibility": "public",
            "is_enabled": True,
        }
        mock_agent_service.get_agent_info = AsyncMock(return_value=mock_agent)

        faiss_results = {
            "servers": [],
            "tools": [],
            "agents": [
                {
                    "path": "/agents/test",
                    "relevance_score": 0.9,
                    "agent_card": {"name": "Test Agent", "visibility": "public"},
                }
            ],
        }
        mock_search_repo.search = AsyncMock(return_value=faiss_results)

        request = SemanticSearchRequest(query="test")

        # Act
        response = await semantic_search(
            mock_http_request, request, admin_user_context, mock_search_repo
        )

        # Assert
        assert len(response.agents) == 1
        # Skills structure is preserved in agent_card
        assert response.agents[0].agent_card["skills"] == [
            {"name": "Skill 1"},
            {"name": None},
            {"name": "Skill 2"},
        ]


# =============================================================================
# TEST: Integration-style Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.search
class TestSemanticSearchIntegration:
    """Integration-style tests for semantic search."""

    @pytest.mark.asyncio
    async def test_semantic_search_full_workflow(
        self,
        mock_http_request,
        mock_search_repo,
        mock_agent_service,
        regular_user_context,
    ):
        """Test complete search workflow with mixed results and filtering."""
        # Arrange
        faiss_results = {
            "servers": [
                {
                    "path": "/servers/currenttime",
                    "server_name": "currenttime",
                    "description": "Time utilities",
                    "tags": ["time"],
                    "num_tools": 1,
                    "is_enabled": True,
                    "relevance_score": 0.95,
                    "match_context": "time",
                    "matching_tools": [
                        {
                            "tool_name": "get_time",
                            "description": "Get time",
                            "relevance_score": 0.9,
                        }
                    ],
                },
                {
                    "path": "/servers/restricted",
                    "server_name": "restricted",
                    "description": "Restricted server",
                    "tags": [],
                    "num_tools": 0,
                    "is_enabled": True,
                    "relevance_score": 0.8,
                    "match_context": "restricted",
                    "matching_tools": [],
                },
            ],
            "tools": [
                {
                    "server_path": "/servers/currenttime",
                    "server_name": "currenttime",
                    "tool_name": "get_time",
                    "description": "Get time",
                    "relevance_score": 0.9,
                },
                {
                    "server_path": "/servers/restricted",
                    "server_name": "restricted",
                    "tool_name": "restricted_tool",
                    "description": "Restricted",
                    "relevance_score": 0.85,
                },
            ],
            "agents": [
                {
                    "path": "/agents/code-reviewer",
                    "relevance_score": 0.88,
                    "agent_card": {"name": "code-reviewer", "visibility": "public"},
                },
                {
                    "path": "/agents/restricted-agent",
                    "relevance_score": 0.82,
                    "agent_card": {"name": "restricted-agent", "visibility": "private"},
                },
            ],
        }
        mock_search_repo.search = AsyncMock(return_value=faiss_results)

        def create_mock_agent(path, name, visibility, registered_by="testuser"):
            agent = AgentCardFactory(
                path=path,
                name=name,
                visibility=visibility,
                registered_by=registered_by,
            )
            return agent

        async def get_agent_side_effect(path):
            if path == "/agents/code-reviewer":
                return create_mock_agent(path, "code-reviewer", "public")
            elif path == "/agents/restricted-agent":
                return create_mock_agent(path, "restricted-agent", "private", "otheruser")
            return None

        mock_agent_service.get_agent_info = AsyncMock(side_effect=get_agent_side_effect)

        request = SemanticSearchRequest(query="test query")

        # Act
        response = await semantic_search(
            mock_http_request, request, regular_user_context, mock_search_repo
        )

        # Assert
        # User has access to "currenttime" but not "restricted"
        assert len(response.servers) == 1
        assert response.servers[0].server_name == "currenttime"

        # Tools filtered by server access
        assert len(response.tools) == 1
        assert response.tools[0].server_name == "currenttime"

        # User has access to "/agents/code-reviewer" but not private agent
        assert len(response.agents) == 1
        assert response.agents[0].agent_card["name"] == "code-reviewer"

        # Totals should match filtered results
        assert response.total_servers == 1
        assert response.total_tools == 1
        assert response.total_agents == 1
