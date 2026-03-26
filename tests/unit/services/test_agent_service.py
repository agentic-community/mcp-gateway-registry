"""
Unit tests for registry.services.agent_service module.

This module tests the AgentService class which manages A2A agent registration,
state management, ratings, and file-based storage operations.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from registry.services.agent_service import AgentService
from tests.fixtures.constants import (
    TEST_AGENT_NAME_1,
    TEST_AGENT_NAME_2,
    TEST_AGENT_PATH_1,
    TEST_AGENT_PATH_2,
    TEST_AGENT_URL_1,
    TEST_AGENT_URL_2,
    TRUST_UNVERIFIED,
    VISIBILITY_PUBLIC,
)
from tests.fixtures.factories import AgentCardFactory

logger = logging.getLogger(__name__)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def agent_service(
    mock_settings,
    mock_agent_repository,
    mock_search_repository,
) -> AgentService:
    """
    Create a fresh AgentService instance with mocked dependencies.

    Args:
        mock_settings: Mocked settings fixture
        mock_agent_repository: Mocked agent repository
        mock_search_repository: Mocked search repository

    Returns:
        AgentService instance with injected mocks
    """
    service = AgentService()
    # Inject mocked singletons
    service._repo = mock_agent_repository
    service._search_repo = mock_search_repository
    return service


@pytest.fixture
def sample_agent_dict() -> dict[str, Any]:
    """
    Create a sample agent dictionary for testing.

    Returns:
        Dictionary with sample agent data
    """
    return {
        "protocol_version": "1.0",
        "name": TEST_AGENT_NAME_1,
        "description": "A test agent for unit tests",
        "url": TEST_AGENT_URL_1,
        "version": "1.0",
        "path": TEST_AGENT_PATH_1,
        "capabilities": {"streaming": False, "tools": True},
        "default_input_modes": ["text/plain"],
        "default_output_modes": ["text/plain"],
        "skills": [
            {
                "id": "skill-1",
                "name": "Data Processing",
                "description": "Process data efficiently",
                "tags": ["data", "processing"],
            }
        ],
        "tags": ["test", "data"],
        "is_enabled": False,
        "num_stars": 0.0,
        "rating_details": [],
        "license": "MIT",
        "visibility": VISIBILITY_PUBLIC,
        "trust_level": TRUST_UNVERIFIED,
    }


@pytest.fixture
def sample_agent_dict_2() -> dict[str, Any]:
    """
    Create a second sample agent dictionary for testing.

    Returns:
        Dictionary with sample agent data
    """
    return {
        "protocol_version": "1.0",
        "name": TEST_AGENT_NAME_2,
        "description": "Another test agent",
        "url": TEST_AGENT_URL_2,
        "version": "2.0",
        "path": TEST_AGENT_PATH_2,
        "capabilities": {"streaming": True},
        "default_input_modes": ["text/plain"],
        "default_output_modes": ["application/json"],
        "skills": [],
        "tags": ["test"],
        "is_enabled": False,
        "num_stars": 4.5,
        "rating_details": [],
        "license": "Apache-2.0",
        "visibility": VISIBILITY_PUBLIC,
        "trust_level": TRUST_UNVERIFIED,
    }


@pytest.fixture
def agent_json_files(
    tmp_path: Path,
    sample_agent_dict: dict[str, Any],
) -> Path:
    """
    Create sample JSON agent files in tmp_path.

    Args:
        tmp_path: Temporary directory path
        sample_agent_dict: Sample agent data

    Returns:
        Path to agents directory with JSON files
    """
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)

    # Create a valid agent file
    agent_file = agents_dir / "test_agent_1_agent.json"
    with open(agent_file, "w") as f:
        json.dump(sample_agent_dict, f, indent=2)

    # Create another valid agent file
    agent_2 = {
        "protocol_version": "1.0",
        "name": TEST_AGENT_NAME_2,
        "description": "Another agent",
        "url": TEST_AGENT_URL_2,
        "version": "1.0",
        "path": TEST_AGENT_PATH_2,
        "capabilities": {},
        "default_input_modes": ["text/plain"],
        "default_output_modes": ["text/plain"],
        "skills": [],
        "tags": [],
        "is_enabled": False,
        "num_stars": 0.0,
        "rating_details": [],
    }
    agent_file_2 = agents_dir / "test_agent_2_agent.json"
    with open(agent_file_2, "w") as f:
        json.dump(agent_2, f, indent=2)

    # Create an invalid agent file (missing required fields)
    invalid_file = agents_dir / "invalid_agent.json"
    with open(invalid_file, "w") as f:
        json.dump({"invalid": "data"}, f)

    # Create a malformed JSON file
    malformed_file = agents_dir / "malformed_agent.json"
    with open(malformed_file, "w") as f:
        f.write("{invalid json")

    # Create agent_state.json (should be excluded from loading)
    state_file = agents_dir / "agent_state.json"
    with open(state_file, "w") as f:
        json.dump({"enabled": [], "disabled": []}, f)

    return agents_dir


@pytest.fixture
def agent_state_file(
    tmp_path: Path,
) -> Path:
    """
    Create an agent state file with test data.

    Args:
        tmp_path: Temporary directory path

    Returns:
        Path to agent_state.json
    """
    state_file = tmp_path / "agent_state.json"
    state_data = {
        "enabled": [TEST_AGENT_PATH_1],
        "disabled": [TEST_AGENT_PATH_2],
    }
    with open(state_file, "w") as f:
        json.dump(state_data, f, indent=2)

    return state_file


# =============================================================================
# TEST: Helper Functions - Path to Filename Conversion
# =============================================================================


@pytest.mark.unit
@pytest.mark.agents
class TestAgentServiceInstantiation:
    """Test AgentService initialization and basic properties."""

    def test_init_creates_empty_registries(
        self,
        agent_service: AgentService,
    ):
        """Test that __init__ creates empty registries."""
        # Assert
        assert agent_service.registered_agents == {}
        assert agent_service.agent_state == {"enabled": [], "disabled": []}

    def test_init_does_not_load_agents(
        self,
        agent_service: AgentService,
    ):
        """Test that __init__ does not automatically load agents."""
        # Assert - should be empty until load_agents_and_state is called
        assert len(agent_service.registered_agents) == 0
        assert len(agent_service.agent_state["enabled"]) == 0
        assert len(agent_service.agent_state["disabled"]) == 0


# =============================================================================
# TEST: Loading Agents and State
# =============================================================================


@pytest.mark.unit
@pytest.mark.agents
@pytest.mark.agents
class TestRegisterAgent:
    """Test agent registration."""

    @pytest.mark.asyncio
    async def test_register_new_agent_successfully(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test registering a new agent."""
        # Arrange
        agent_card = AgentCardFactory(path="/new-agent")
        mock_agent_repository.create.return_value = agent_card
        mock_agent_repository.save_state.return_value = True

        # Act
        result = await agent_service.register_agent(agent_card)

        # Assert
        assert result == agent_card
        mock_agent_repository.create.assert_called_once()
        mock_agent_repository.save_state.assert_called()
        mock_search_repository.index_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_agent_sets_timestamps(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test that registration sets registered_at and updated_at."""
        # Arrange
        agent_card = AgentCardFactory(
            path="/test-agent",
            registered_at=None,
            updated_at=None,
        )

        # Mock create to set timestamps like real repository does
        def mock_create(agent):
            if not agent.registered_at:
                agent.registered_at = datetime.now(UTC)
            if not agent.updated_at:
                agent.updated_at = datetime.now(UTC)
            return agent

        mock_agent_repository.create.side_effect = mock_create

        # Act
        result = await agent_service.register_agent(agent_card)

        # Assert
        assert result.registered_at is not None
        assert result.updated_at is not None
        assert isinstance(result.registered_at, datetime)
        assert isinstance(result.updated_at, datetime)

    @pytest.mark.asyncio
    async def test_register_agent_preserves_existing_timestamps(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test that registration preserves existing timestamps."""
        # Arrange
        original_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        agent_card = AgentCardFactory(
            path="/test-agent",
            registered_at=original_time,
            updated_at=original_time,
        )
        mock_agent_repository.create.return_value = agent_card

        # Act
        result = await agent_service.register_agent(agent_card)

        # Assert
        assert result.registered_at == original_time
        assert result.updated_at == original_time

    @pytest.mark.asyncio
    async def test_register_agent_fails_for_duplicate_path(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test that registering duplicate path raises ValueError."""
        # Arrange
        agent_card = AgentCardFactory(path="/duplicate")
        # Simulate agent already in registry
        agent_service.registered_agents["/duplicate"] = agent_card

        # Act & Assert
        with pytest.raises(ValueError, match="already exists"):
            await agent_service.register_agent(agent_card)

    @pytest.mark.asyncio
    async def test_register_agent_defaults_to_disabled(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test that newly registered agents are disabled by default."""
        # Arrange
        agent_card = AgentCardFactory(path="/new-agent")
        mock_agent_repository.create.return_value = agent_card
        mock_agent_repository.save_state.return_value = True

        # Act
        await agent_service.register_agent(agent_card)

        # Assert
        # Verify the agent was added to disabled list via state persistence
        assert mock_agent_repository.save_state.called
        assert "/new-agent" in agent_service.agent_state["disabled"]


# =============================================================================
# TEST: Get Agent
# =============================================================================


@pytest.mark.unit
@pytest.mark.agents
class TestGetAgent:
    """Test retrieving agents."""

    def test_get_existing_agent(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test getting an existing agent."""
        # Arrange
        agent_card = AgentCardFactory(path="/test-agent")
        agent_service.registered_agents["/test-agent"] = agent_card

        # Act
        result = agent_service.get_agent("/test-agent")

        # Assert
        assert result.path == "/test-agent"
        assert result.name == agent_card.name

    def test_get_agent_not_found(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test getting a non-existent agent raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="not found"):
            agent_service.get_agent("/nonexistent")

    def test_get_agent_handles_trailing_slash(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test getting agent with/without trailing slash."""
        # Arrange
        agent_card = AgentCardFactory(path="/test-agent")
        agent_service.registered_agents["/test-agent"] = agent_card

        # Act - try with trailing slash
        result = agent_service.get_agent("/test-agent/")

        # Assert
        assert result.path == "/test-agent"

    def test_get_agent_with_slash_registered_without(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test getting agent registered with slash when querying without."""
        # Arrange
        agent_card = AgentCardFactory(path="/test-agent")
        # Manually register with trailing slash to test edge case
        agent_service.registered_agents["/test-agent/"] = agent_card

        # Act
        result = agent_service.get_agent("/test-agent")

        # Assert
        # Should find agent despite slash mismatch
        assert result is not None
        assert result.name == agent_card.name


# =============================================================================
# TEST: List Agents
# =============================================================================


@pytest.mark.unit
@pytest.mark.agents
class TestListAgents:
    """Test listing all agents."""

    def test_list_agents_empty(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test listing agents when none are registered."""
        # Arrange
        agent_service.registered_agents = {}

        # Act
        result = agent_service.list_agents()

        # Assert
        assert result == []

    def test_list_agents_returns_all(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test listing all registered agents."""
        # Arrange
        agent_1 = AgentCardFactory(path="/agent-1")
        agent_2 = AgentCardFactory(path="/agent-2")
        agent_service.registered_agents["/agent-1"] = agent_1
        agent_service.registered_agents["/agent-2"] = agent_2

        # Act
        result = agent_service.list_agents()

        # Assert
        assert len(result) == 2
        paths = [a.path for a in result]
        assert "/agent-1" in paths
        assert "/agent-2" in paths

    @pytest.mark.asyncio
    async def test_get_all_agents_alias(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test that get_all_agents is an alias for list_agents."""
        # Arrange
        agent = AgentCardFactory(path="/test")
        agent_service.registered_agents["/test"] = agent
        mock_agent_repository.list_all.return_value = [agent]

        # Act
        list_result = agent_service.list_agents()
        get_all_result = await agent_service.get_all_agents()

        # Assert
        assert len(list_result) == 1
        assert len(get_all_result) == 1
        assert list_result[0].path == get_all_result[0].path


# =============================================================================
# TEST: Update Agent
# =============================================================================


@pytest.mark.unit
@pytest.mark.agents
class TestUpdateAgent:
    """Test updating agent information."""

    @pytest.mark.asyncio
    async def test_update_agent_successfully(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test updating an agent's information."""
        # Arrange
        agent_card = AgentCardFactory(
            path="/test-agent",
            description="Original description",
        )
        # Pre-populate registered_agents
        agent_service.registered_agents["/test-agent"] = agent_card

        updated_card = AgentCardFactory(
            path="/test-agent",
            description="Updated description",
        )
        mock_agent_repository.update.return_value = updated_card

        updates = {"description": "Updated description"}

        # Act
        result = await agent_service.update_agent("/test-agent", updates)

        # Assert
        assert result.description == "Updated description"
        assert result.path == "/test-agent"
        mock_agent_repository.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_agent_updates_timestamp(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test that update sets updated_at timestamp."""
        # Arrange
        original_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        agent_card = AgentCardFactory(
            path="/test-agent",
            updated_at=original_time,
        )
        agent_service.registered_agents["/test-agent"] = agent_card

        updated_card = AgentCardFactory(
            path="/test-agent",
            updated_at=datetime.now(UTC),
        )
        mock_agent_repository.update.return_value = updated_card

        # Act
        result = await agent_service.update_agent("/test-agent", {"description": "New"})

        # Assert
        assert result.updated_at > original_time

    @pytest.mark.asyncio
    async def test_update_agent_not_found(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test updating non-existent agent raises ValueError."""
        # Arrange - don't add to registered_agents

        # Act & Assert
        with pytest.raises(ValueError, match="not found"):
            await agent_service.update_agent("/nonexistent", {"description": "test"})

    @pytest.mark.asyncio
    async def test_update_agent_preserves_path(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test that path cannot be changed via update."""
        # Arrange
        agent_card = AgentCardFactory(path="/original-path")
        agent_service.registered_agents["/original-path"] = agent_card

        updated_card = AgentCardFactory(path="/original-path", description="Updated")
        mock_agent_repository.update.return_value = updated_card

        # Act
        result = await agent_service.update_agent(
            "/original-path",
            {"path": "/new-path", "description": "Updated"},
        )

        # Assert
        # Path should remain unchanged
        assert result.path == "/original-path"

    @pytest.mark.asyncio
    async def test_update_agent_with_invalid_data(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test updating with invalid data raises ValueError."""
        # Arrange
        agent_card = AgentCardFactory(path="/test-agent")
        agent_service.registered_agents["/test-agent"] = agent_card

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid"):
            # Try to set num_stars to invalid value
            await agent_service.update_agent("/test-agent", {"num_stars": 10.0})


# =============================================================================
# TEST: Delete Agent
# =============================================================================


@pytest.mark.unit
@pytest.mark.agents
class TestDeleteAgent:
    """Test agent deletion."""

    @pytest.mark.asyncio
    async def test_delete_agent_successfully(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test deleting an agent."""
        # Arrange
        agent_card = AgentCardFactory(path="/test-agent")
        agent_service.registered_agents["/test-agent"] = agent_card
        agent_service.agent_state["disabled"].append("/test-agent")
        mock_agent_repository.delete.return_value = True

        # Act
        result = await agent_service.delete_agent("/test-agent")

        # Assert
        assert result is True
        assert "/test-agent" not in agent_service.registered_agents
        mock_agent_repository.delete.assert_called_once_with("/test-agent")

    @pytest.mark.asyncio
    async def test_delete_agent_removes_from_state(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test that delete removes agent from state lists."""
        # Arrange
        agent_card = AgentCardFactory(path="/test-agent")
        agent_service.registered_agents["/test-agent"] = agent_card
        agent_service.agent_state["enabled"].append("/test-agent")
        mock_agent_repository.delete.return_value = True

        # Act
        await agent_service.delete_agent("/test-agent")

        # Assert
        assert "/test-agent" not in agent_service.agent_state["enabled"]
        assert "/test-agent" not in agent_service.agent_state["disabled"]

    @pytest.mark.asyncio
    async def test_delete_agent_not_found(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test deleting non-existent agent raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="not found"):
            await agent_service.delete_agent("/nonexistent")

    @pytest.mark.asyncio
    async def test_remove_agent_alias(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test that remove_agent is an alias for delete_agent."""
        # Arrange
        agent_card = AgentCardFactory(path="/test-agent")
        agent_service.registered_agents["/test-agent"] = agent_card
        agent_service.agent_state["disabled"].append("/test-agent")
        mock_agent_repository.delete.return_value = True

        # Act
        result = await agent_service.remove_agent("/test-agent")

        # Assert
        assert result is True
        assert "/test-agent" not in agent_service.registered_agents

    @pytest.mark.asyncio
    async def test_remove_agent_returns_false_for_not_found(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test that remove_agent returns False for non-existent agent."""
        # Act
        result = await agent_service.remove_agent("/nonexistent")

        # Assert
        assert result is False


# =============================================================================
# TEST: Enable/Disable Agent
# =============================================================================


@pytest.mark.unit
@pytest.mark.agents
class TestEnableDisableAgent:
    """Test enabling and disabling agents."""

    @pytest.mark.asyncio
    async def test_enable_agent(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test enabling an agent."""
        # Arrange
        agent_card = AgentCardFactory(path="/test-agent")
        agent_service.registered_agents["/test-agent"] = agent_card
        agent_service.agent_state["disabled"].append("/test-agent")

        # Act
        await agent_service.enable_agent("/test-agent")

        # Assert
        assert "/test-agent" in agent_service.agent_state["enabled"]
        assert "/test-agent" not in agent_service.agent_state["disabled"]
        mock_agent_repository.set_state.assert_called_with("/test-agent", True)

    @pytest.mark.asyncio
    async def test_enable_already_enabled_agent(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test enabling an already enabled agent (idempotent)."""
        # Arrange
        agent_card = AgentCardFactory(path="/test-agent")
        agent_service.registered_agents["/test-agent"] = agent_card
        agent_service.agent_state["enabled"].append("/test-agent")

        # Act - enable again
        await agent_service.enable_agent("/test-agent")

        # Assert
        assert "/test-agent" in agent_service.agent_state["enabled"]
        # Should only appear once
        assert agent_service.agent_state["enabled"].count("/test-agent") == 1

    @pytest.mark.asyncio
    async def test_enable_agent_not_found(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test enabling non-existent agent raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="not found"):
            await agent_service.enable_agent("/nonexistent")

    @pytest.mark.asyncio
    async def test_disable_agent(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test disabling an agent."""
        # Arrange
        agent_card = AgentCardFactory(path="/test-agent")
        agent_service.registered_agents["/test-agent"] = agent_card
        agent_service.agent_state["enabled"].append("/test-agent")

        # Act
        await agent_service.disable_agent("/test-agent")

        # Assert
        assert "/test-agent" in agent_service.agent_state["disabled"]
        assert "/test-agent" not in agent_service.agent_state["enabled"]
        mock_agent_repository.set_state.assert_called_with("/test-agent", False)

    @pytest.mark.asyncio
    async def test_disable_already_disabled_agent(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test disabling an already disabled agent (idempotent)."""
        # Arrange
        agent_card = AgentCardFactory(path="/test-agent")
        agent_service.registered_agents["/test-agent"] = agent_card
        agent_service.agent_state["disabled"].append("/test-agent")

        # Act - disable again (already disabled by default)
        await agent_service.disable_agent("/test-agent")

        # Assert
        assert "/test-agent" in agent_service.agent_state["disabled"]
        # Should only appear once
        assert agent_service.agent_state["disabled"].count("/test-agent") == 1

    @pytest.mark.asyncio
    async def test_disable_agent_not_found(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test disabling non-existent agent raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="not found"):
            await agent_service.disable_agent("/nonexistent")

    @pytest.mark.asyncio
    async def test_toggle_agent_enable(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test toggling agent to enabled state."""
        # Arrange
        agent_card = AgentCardFactory(path="/test-agent")
        agent_service.registered_agents["/test-agent"] = agent_card
        agent_service.agent_state["disabled"].append("/test-agent")

        # Act
        result = await agent_service.toggle_agent("/test-agent", enabled=True)

        # Assert
        assert result is True
        assert "/test-agent" in agent_service.agent_state["enabled"]

    @pytest.mark.asyncio
    async def test_toggle_agent_disable(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test toggling agent to disabled state."""
        # Arrange
        agent_card = AgentCardFactory(path="/test-agent")
        agent_service.registered_agents["/test-agent"] = agent_card
        agent_service.agent_state["enabled"].append("/test-agent")

        # Act
        result = await agent_service.toggle_agent("/test-agent", enabled=False)

        # Assert
        assert result is True
        assert "/test-agent" in agent_service.agent_state["disabled"]

    @pytest.mark.asyncio
    async def test_toggle_agent_not_found(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test toggling non-existent agent returns False."""
        # Act
        result = await agent_service.toggle_agent("/nonexistent", enabled=True)

        # Assert
        assert result is False


# =============================================================================
# TEST: Agent State Queries
# =============================================================================


@pytest.mark.unit
@pytest.mark.agents
class TestAgentStateQueries:
    """Test querying agent state."""

    def test_is_agent_enabled_true(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test checking if agent is enabled."""
        # Arrange
        agent_service.agent_state["enabled"].append("/test-agent")

        # Act
        result = agent_service.is_agent_enabled("/test-agent")

        # Assert
        assert result is True

    def test_is_agent_enabled_false(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test checking if agent is disabled."""
        # Arrange
        agent_service.agent_state["disabled"].append("/test-agent")

        # Act
        result = agent_service.is_agent_enabled("/test-agent")

        # Assert
        assert result is False

    def test_is_agent_enabled_handles_trailing_slash(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test is_agent_enabled with trailing slash."""
        # Arrange
        agent_service.agent_state["enabled"].append("/test-agent")

        # Act
        result = agent_service.is_agent_enabled("/test-agent/")

        # Assert
        assert result is True

    def test_get_enabled_agents(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test getting list of enabled agents."""
        # Arrange
        agent_service.agent_state["enabled"].append("/agent-1")
        agent_service.agent_state["disabled"].append("/agent-2")

        # Act
        result = agent_service.get_enabled_agents()

        # Assert
        assert len(result) == 1
        assert "/agent-1" in result
        assert "/agent-2" not in result

    def test_get_disabled_agents(
        self,
        agent_service: AgentService,
        mock_agent_repository,
        mock_search_repository,
    ):
        """Test getting list of disabled agents."""
        # Arrange
        agent_service.agent_state["enabled"].append("/agent-1")
        agent_service.agent_state["disabled"].append("/agent-2")

        # Act
        result = agent_service.get_disabled_agents()

        # Assert
        assert len(result) == 1
        assert "/agent-2" in result
        assert "/agent-1" not in result
