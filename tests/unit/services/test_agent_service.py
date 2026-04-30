"""
Unit tests for registry.services.agent_service module.

These tests exercise AgentService against an in-memory fake implementation
of AgentRepositoryBase so we test the real service-to-repo contract rather
than MagicMock behavior.
"""

import logging
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from registry.repositories.interfaces import AgentRepositoryBase
from registry.schemas.agent_models import AgentCard
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
# In-memory fake repository
# =============================================================================


class InMemoryAgentRepository(AgentRepositoryBase):
    """In-memory AgentRepositoryBase implementation for tests.

    Stores AgentCard objects keyed by path and a parallel enabled/disabled
    map. Mirrors the persistence contract used by real repository
    implementations.
    """

    def __init__(self) -> None:
        self._agents: dict[str, AgentCard] = {}
        self._enabled: dict[str, bool] = {}

    async def get(self, path: str) -> AgentCard | None:
        return self._agents.get(path)

    async def list_all(self) -> list[AgentCard]:
        return list(self._agents.values())

    async def list_paginated(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> list[AgentCard]:
        return list(self._agents.values())[skip : skip + limit]

    async def create(self, agent: AgentCard) -> AgentCard:
        if agent.path in self._agents:
            raise ValueError(f"Agent path '{agent.path}' already exists")
        if not agent.registered_at:
            agent.registered_at = datetime.now(UTC)
        if not agent.updated_at:
            agent.updated_at = datetime.now(UTC)
        self._agents[agent.path] = agent
        self._enabled.setdefault(agent.path, False)
        return agent

    async def update(self, path: str, updates: dict[str, Any]) -> AgentCard:
        existing = self._agents.get(path)
        if existing is None:
            raise ValueError(f"Agent not found at path: {path}")

        data = existing.model_dump()
        data.update(updates)
        data["path"] = path
        data["updated_at"] = datetime.now(UTC)
        new_agent = AgentCard(**data)
        self._agents[path] = new_agent
        return new_agent

    async def delete(self, path: str) -> bool:
        if path not in self._agents:
            return False
        del self._agents[path]
        self._enabled.pop(path, None)
        return True

    async def get_state(self, path: str) -> bool:
        return self._enabled.get(path, False)

    async def get_all_states(self) -> dict[str, bool]:
        return dict(self._enabled)

    async def set_state(self, path: str, enabled: bool) -> bool:
        if path not in self._agents:
            return False
        self._enabled[path] = enabled
        agent = self._agents[path]
        data = agent.model_dump()
        data["is_enabled"] = enabled
        self._agents[path] = AgentCard(**data)
        return True

    async def load_all(self) -> None:
        return None

    async def count(self) -> int:
        return len(self._agents)

    async def update_field(self, path: str, field: str, value: Any) -> bool:
        agent = self._agents.get(path)
        if agent is None:
            return False
        data = agent.model_dump()
        data[field] = value
        self._agents[path] = AgentCard(**data)
        return True

    async def find_with_filter(
        self, filter_dict: dict[str, Any]
    ) -> dict[str, dict]:
        results: dict[str, dict] = {}
        for path, agent in self._agents.items():
            data = agent.model_dump()
            if all(data.get(k) == v for k, v in filter_dict.items()):
                results[path] = data
        return results


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def fake_repo() -> InMemoryAgentRepository:
    return InMemoryAgentRepository()


@pytest.fixture
def fake_search_repo() -> AsyncMock:
    """Search repository is an integration boundary we don't exercise here."""
    mock = AsyncMock()
    mock.index_agent.return_value = None
    mock.remove_entity.return_value = None
    mock.index_entity.return_value = None
    return mock


@pytest.fixture
def agent_service(
    mock_settings,
    fake_repo: InMemoryAgentRepository,
    fake_search_repo: AsyncMock,
) -> AgentService:
    """AgentService backed by an in-memory repository."""
    service = AgentService()
    service._repo = fake_repo
    service._search_repo = fake_search_repo
    return service


@pytest.fixture
def sample_agent_dict() -> dict[str, Any]:
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
    return {
        "protocol_version": "1.0",
        "name": TEST_AGENT_NAME_2,
        "description": "Another test agent",
        "url": TEST_AGENT_URL_2,
        "version": "1.0",
        "path": TEST_AGENT_PATH_2,
        "capabilities": {"streaming": True, "tools": False},
        "default_input_modes": ["text/plain"],
        "default_output_modes": ["text/plain"],
        "skills": [],
        "tags": ["test"],
        "is_enabled": False,
        "num_stars": 0.0,
        "rating_details": [],
        "license": "Apache-2.0",
        "visibility": VISIBILITY_PUBLIC,
        "trust_level": TRUST_UNVERIFIED,
    }


# =============================================================================
# TEST: Register Agent
# =============================================================================


@pytest.mark.unit
@pytest.mark.agents
class TestRegisterAgent:
    @pytest.mark.asyncio
    async def test_register_new_agent_successfully(
        self,
        agent_service: AgentService,
        fake_repo: InMemoryAgentRepository,
        fake_search_repo: AsyncMock,
    ):
        agent_card = AgentCardFactory(path="/new-agent")

        result = await agent_service.register_agent(agent_card)

        assert result.path == "/new-agent"
        assert await fake_repo.get("/new-agent") is not None
        fake_search_repo.index_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_agent_fails_for_duplicate_path(
        self,
        agent_service: AgentService,
        fake_repo: InMemoryAgentRepository,
    ):
        await fake_repo.create(AgentCardFactory(path="/duplicate"))

        with pytest.raises(ValueError, match="already exists"):
            await agent_service.register_agent(AgentCardFactory(path="/duplicate"))

    @pytest.mark.asyncio
    async def test_register_agent_defaults_to_disabled(
        self,
        agent_service: AgentService,
        fake_repo: InMemoryAgentRepository,
    ):
        agent_card = AgentCardFactory(path="/new-agent")

        await agent_service.register_agent(agent_card)

        assert await fake_repo.get_state("/new-agent") is False


# =============================================================================
# TEST: Get Agent
# =============================================================================


@pytest.mark.unit
@pytest.mark.agents
class TestGetAgent:
    @pytest.mark.asyncio
    async def test_get_existing_agent(
        self,
        agent_service: AgentService,
        fake_repo: InMemoryAgentRepository,
    ):
        agent_card = AgentCardFactory(path="/test-agent")
        await fake_repo.create(agent_card)

        result = await agent_service.get_agent("/test-agent")

        assert result.path == "/test-agent"
        assert result.name == agent_card.name

    @pytest.mark.asyncio
    async def test_get_agent_not_found(
        self,
        agent_service: AgentService,
    ):
        with pytest.raises(ValueError, match="not found"):
            await agent_service.get_agent("/nonexistent")

    @pytest.mark.asyncio
    async def test_get_agent_handles_trailing_slash(
        self,
        agent_service: AgentService,
        fake_repo: InMemoryAgentRepository,
    ):
        await fake_repo.create(AgentCardFactory(path="/test-agent"))

        result = await agent_service.get_agent("/test-agent/")

        assert result.path == "/test-agent"

    @pytest.mark.asyncio
    async def test_get_agent_falls_back_when_query_has_extra_slash(
        self,
        agent_service: AgentService,
        fake_repo: InMemoryAgentRepository,
    ):
        """A query with a trailing slash should still find an agent stored without one."""
        await fake_repo.create(AgentCardFactory(path="/test-agent"))

        result = await agent_service.get_agent("/test-agent/")

        assert result is not None
        assert result.path == "/test-agent"


# =============================================================================
# TEST: List Agents
# =============================================================================


@pytest.mark.unit
@pytest.mark.agents
class TestListAgents:
    @pytest.mark.asyncio
    async def test_list_agents_empty(self, agent_service: AgentService):
        result = await agent_service.list_agents()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_agents_returns_all(
        self,
        agent_service: AgentService,
        fake_repo: InMemoryAgentRepository,
    ):
        await fake_repo.create(AgentCardFactory(path="/agent-1"))
        await fake_repo.create(AgentCardFactory(path="/agent-2"))

        result = await agent_service.list_agents()

        paths = [a.path for a in result]
        assert set(paths) == {"/agent-1", "/agent-2"}

    @pytest.mark.asyncio
    async def test_get_all_agents_alias(
        self,
        agent_service: AgentService,
        fake_repo: InMemoryAgentRepository,
    ):
        await fake_repo.create(AgentCardFactory(path="/test"))

        list_result = await agent_service.list_agents()
        get_all_result = await agent_service.get_all_agents()

        assert len(list_result) == len(get_all_result) == 1
        assert list_result[0].path == get_all_result[0].path


# =============================================================================
# TEST: Update Agent
# =============================================================================


@pytest.mark.unit
@pytest.mark.agents
class TestUpdateAgent:
    @pytest.mark.asyncio
    async def test_update_agent_successfully(
        self,
        agent_service: AgentService,
        fake_repo: InMemoryAgentRepository,
    ):
        await fake_repo.create(
            AgentCardFactory(path="/test-agent", description="Original description")
        )

        result = await agent_service.update_agent(
            "/test-agent", {"description": "Updated description"}
        )

        assert result.description == "Updated description"
        assert result.path == "/test-agent"
        persisted = await fake_repo.get("/test-agent")
        assert persisted.description == "Updated description"

    @pytest.mark.asyncio
    async def test_update_agent_updates_timestamp(
        self,
        agent_service: AgentService,
        fake_repo: InMemoryAgentRepository,
    ):
        original_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        await fake_repo.create(
            AgentCardFactory(path="/test-agent", updated_at=original_time)
        )

        result = await agent_service.update_agent("/test-agent", {"description": "New"})

        assert result.updated_at > original_time

    @pytest.mark.asyncio
    async def test_update_agent_not_found(self, agent_service: AgentService):
        with pytest.raises(ValueError, match="not found"):
            await agent_service.update_agent("/nonexistent", {"description": "test"})

    @pytest.mark.asyncio
    async def test_update_agent_preserves_path(
        self,
        agent_service: AgentService,
        fake_repo: InMemoryAgentRepository,
    ):
        await fake_repo.create(AgentCardFactory(path="/original-path"))

        result = await agent_service.update_agent(
            "/original-path",
            {"path": "/new-path", "description": "Updated"},
        )

        assert result.path == "/original-path"
        assert await fake_repo.get("/new-path") is None

    @pytest.mark.asyncio
    async def test_update_agent_with_invalid_data(
        self,
        agent_service: AgentService,
        fake_repo: InMemoryAgentRepository,
    ):
        await fake_repo.create(AgentCardFactory(path="/test-agent"))

        with pytest.raises(ValueError, match="Invalid"):
            await agent_service.update_agent("/test-agent", {"num_stars": 10.0})


# =============================================================================
# TEST: Delete Agent
# =============================================================================


@pytest.mark.unit
@pytest.mark.agents
class TestDeleteAgent:
    @pytest.mark.asyncio
    async def test_delete_agent_successfully(
        self,
        agent_service: AgentService,
        fake_repo: InMemoryAgentRepository,
    ):
        await fake_repo.create(AgentCardFactory(path="/test-agent"))

        result = await agent_service.delete_agent("/test-agent")

        assert result is True
        assert await fake_repo.get("/test-agent") is None

    @pytest.mark.asyncio
    async def test_delete_agent_not_found(self, agent_service: AgentService):
        with pytest.raises(ValueError, match="not found"):
            await agent_service.delete_agent("/nonexistent")

    @pytest.mark.asyncio
    async def test_remove_agent_alias(
        self,
        agent_service: AgentService,
        fake_repo: InMemoryAgentRepository,
    ):
        await fake_repo.create(AgentCardFactory(path="/test-agent"))

        result = await agent_service.remove_agent("/test-agent")

        assert result is True
        assert await fake_repo.get("/test-agent") is None

    @pytest.mark.asyncio
    async def test_remove_agent_returns_false_for_not_found(
        self, agent_service: AgentService
    ):
        result = await agent_service.remove_agent("/nonexistent")

        assert result is False


# =============================================================================
# TEST: Enable/Disable Agent
# =============================================================================


@pytest.mark.unit
@pytest.mark.agents
class TestEnableDisableAgent:
    @pytest.mark.asyncio
    async def test_enable_agent(
        self,
        agent_service: AgentService,
        fake_repo: InMemoryAgentRepository,
    ):
        await fake_repo.create(AgentCardFactory(path="/test-agent"))

        await agent_service.enable_agent("/test-agent")

        assert await fake_repo.get_state("/test-agent") is True

    @pytest.mark.asyncio
    async def test_enable_already_enabled_agent(
        self,
        agent_service: AgentService,
        fake_repo: InMemoryAgentRepository,
    ):
        await fake_repo.create(AgentCardFactory(path="/test-agent"))
        await fake_repo.set_state("/test-agent", True)

        await agent_service.enable_agent("/test-agent")

        assert await fake_repo.get_state("/test-agent") is True

    @pytest.mark.asyncio
    async def test_enable_agent_not_found(self, agent_service: AgentService):
        with pytest.raises(ValueError, match="not found"):
            await agent_service.enable_agent("/nonexistent")

    @pytest.mark.asyncio
    async def test_disable_agent(
        self,
        agent_service: AgentService,
        fake_repo: InMemoryAgentRepository,
    ):
        await fake_repo.create(AgentCardFactory(path="/test-agent"))
        await fake_repo.set_state("/test-agent", True)

        await agent_service.disable_agent("/test-agent")

        assert await fake_repo.get_state("/test-agent") is False

    @pytest.mark.asyncio
    async def test_disable_already_disabled_agent(
        self,
        agent_service: AgentService,
        fake_repo: InMemoryAgentRepository,
    ):
        await fake_repo.create(AgentCardFactory(path="/test-agent"))

        await agent_service.disable_agent("/test-agent")

        assert await fake_repo.get_state("/test-agent") is False

    @pytest.mark.asyncio
    async def test_disable_agent_not_found(self, agent_service: AgentService):
        with pytest.raises(ValueError, match="not found"):
            await agent_service.disable_agent("/nonexistent")

    @pytest.mark.asyncio
    async def test_toggle_agent_enable(
        self,
        agent_service: AgentService,
        fake_repo: InMemoryAgentRepository,
    ):
        await fake_repo.create(AgentCardFactory(path="/test-agent"))

        result = await agent_service.toggle_agent("/test-agent", enabled=True)

        assert result is True
        assert await fake_repo.get_state("/test-agent") is True

    @pytest.mark.asyncio
    async def test_toggle_agent_disable(
        self,
        agent_service: AgentService,
        fake_repo: InMemoryAgentRepository,
    ):
        await fake_repo.create(AgentCardFactory(path="/test-agent"))
        await fake_repo.set_state("/test-agent", True)

        result = await agent_service.toggle_agent("/test-agent", enabled=False)

        assert result is True
        assert await fake_repo.get_state("/test-agent") is False

    @pytest.mark.asyncio
    async def test_toggle_agent_not_found(self, agent_service: AgentService):
        result = await agent_service.toggle_agent("/nonexistent", enabled=True)
        assert result is False


# =============================================================================
# TEST: Agent State Queries
# =============================================================================


@pytest.mark.unit
@pytest.mark.agents
class TestAgentStateQueries:
    @pytest.mark.asyncio
    async def test_is_agent_enabled_true(
        self,
        agent_service: AgentService,
        fake_repo: InMemoryAgentRepository,
    ):
        await fake_repo.create(AgentCardFactory(path="/test-agent"))
        await fake_repo.set_state("/test-agent", True)

        assert await agent_service.is_agent_enabled("/test-agent") is True

    @pytest.mark.asyncio
    async def test_is_agent_enabled_false(
        self,
        agent_service: AgentService,
        fake_repo: InMemoryAgentRepository,
    ):
        await fake_repo.create(AgentCardFactory(path="/test-agent"))

        assert await agent_service.is_agent_enabled("/test-agent") is False

    @pytest.mark.asyncio
    async def test_is_agent_enabled_handles_trailing_slash(
        self,
        agent_service: AgentService,
        fake_repo: InMemoryAgentRepository,
    ):
        await fake_repo.create(AgentCardFactory(path="/test-agent"))
        await fake_repo.set_state("/test-agent", True)

        assert await agent_service.is_agent_enabled("/test-agent/") is True

    @pytest.mark.asyncio
    async def test_get_enabled_agents(
        self,
        agent_service: AgentService,
        fake_repo: InMemoryAgentRepository,
    ):
        await fake_repo.create(AgentCardFactory(path="/agent-1"))
        await fake_repo.create(AgentCardFactory(path="/agent-2"))
        await fake_repo.set_state("/agent-1", True)

        result = await agent_service.get_enabled_agents()

        assert result == ["/agent-1"]

    @pytest.mark.asyncio
    async def test_get_disabled_agents(
        self,
        agent_service: AgentService,
        fake_repo: InMemoryAgentRepository,
    ):
        await fake_repo.create(AgentCardFactory(path="/agent-1"))
        await fake_repo.create(AgentCardFactory(path="/agent-2"))
        await fake_repo.set_state("/agent-1", True)

        result = await agent_service.get_disabled_agents()

        assert result == ["/agent-2"]
