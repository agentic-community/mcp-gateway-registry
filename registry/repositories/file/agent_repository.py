"""File-based agent repository implementation."""

import json
import logging
from datetime import UTC, datetime
from typing import Any

from ...core.config import settings
from ...schemas.agent_models import AgentCard
from ..interfaces import AgentRepositoryBase

logger = logging.getLogger(__name__)


def _path_to_filename(path: str) -> str:
    """Convert agent path to safe filename."""
    normalized = path.lstrip("/").replace("/", "_")
    if not normalized.endswith("_agent.json"):
        if normalized.endswith(".json"):
            normalized = normalized.replace(".json", "_agent.json")
        else:
            normalized += "_agent.json"
    return normalized


class FileAgentRepository(AgentRepositoryBase):
    """File-based agent repository using JSON files."""

    def __init__(self):
        self.agents_dir = settings.agents_dir
        self.state_file = settings.agent_state_file_path
        self.agents_dir.mkdir(parents=True, exist_ok=True)

    async def get_all(self) -> dict[str, AgentCard]:
        """Load all agents from disk."""
        agents = {}
        agent_files = [
            f for f in self.agents_dir.glob("**/*_agent.json") if f.name != self.state_file.name
        ]

        for file in agent_files:
            try:
                with open(file) as f:
                    data = json.load(f)
                if isinstance(data, dict) and "path" in data and "name" in data:
                    agent = AgentCard(**data)
                    agents[agent.path] = agent
            except Exception as e:
                logger.error(f"Failed to load agent from {file}: {e}")

        return agents

    async def get(self, path: str) -> AgentCard | None:
        """Get agent by path."""
        agents = await self.get_all()
        return agents.get(path)

    async def save(self, agent: AgentCard) -> AgentCard:
        """Save agent to disk."""
        if not agent.registered_at:
            agent.registered_at = datetime.now(UTC)
        agent.updated_at = datetime.now(UTC)

        filename = _path_to_filename(agent.path)
        file_path = self.agents_dir / filename

        with open(file_path, "w") as f:
            json.dump(agent.model_dump(mode="json"), f, indent=2)

        return agent

    async def delete(self, path: str) -> bool:
        """Delete agent from disk."""
        filename = _path_to_filename(path)
        file_path = self.agents_dir / filename

        if file_path.exists():
            file_path.unlink()
            return True
        return False

    async def get_state(self) -> dict[str, list[str]]:
        """Load agent state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                if isinstance(state, dict):
                    return {
                        "enabled": state.get("enabled", []),
                        "disabled": state.get("disabled", []),
                    }
            except Exception as e:
                logger.error(f"Failed to load state: {e}")

        return {"enabled": [], "disabled": []}

    async def save_state(self, state: dict[str, list[str]]) -> None:
        """Save agent state to disk."""
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    async def is_enabled(self, path: str) -> bool:
        """Check if agent is enabled."""
        state = await self.get_state()
        return path in state["enabled"]

    async def set_enabled(self, path: str, enabled: bool) -> None:
        """Set agent enabled state."""
        state = await self.get_state()

        if enabled:
            if path in state["disabled"]:
                state["disabled"].remove(path)
            if path not in state["enabled"]:
                state["enabled"].append(path)
        else:
            if path in state["enabled"]:
                state["enabled"].remove(path)
            if path not in state["disabled"]:
                state["disabled"].append(path)

        await self.save_state(state)

    async def create(self, agent: AgentCard) -> AgentCard:
        """Create a new agent (alias for save)."""
        return await self.save(agent)

    async def update(self, path: str, agent: AgentCard) -> AgentCard | None:
        """Update an existing agent."""
        existing = await self.get(path)
        if not existing:
            return None
        return await self.save(agent)

    async def list_all(self) -> list[AgentCard]:
        """List all agents."""
        agents = await self.get_all()
        return list(agents.values())

    async def load_all(self) -> dict[str, AgentCard]:
        """Load all agents (alias for get_all)."""
        return await self.get_all()

    async def set_state(self, path: str, enabled: bool) -> None:
        """Set agent state (alias for set_enabled)."""
        await self.set_enabled(path, enabled)

    async def count(self) -> int:
        """Get total count of agents.

        Returns:
            Total number of agents in the repository.
        """
        agents = await self.get_all()
        return len(agents)

    async def update_field(
        self,
        path: str,
        field: str,
        value: Any,
    ) -> bool:
        """Update a single field on a document (file-based)."""
        agent = await self.get(path)
        if not agent:
            return False

        data = agent.model_dump(mode="json")

        if value is None:
            parts = field.split(".")
            obj = data
            for part in parts[:-1]:
                obj = obj.get(part, {})
            obj.pop(parts[-1], None)
        else:
            parts = field.split(".")
            obj = data
            for part in parts[:-1]:
                if part not in obj:
                    obj[part] = {}
                obj = obj[part]
            obj[parts[-1]] = value

        updated_agent = AgentCard(**data)
        await self.save(updated_agent)
        return True

    async def find_with_filter(
        self,
        filter_dict: dict[str, Any],
    ) -> dict[str, dict]:
        """Find documents matching a filter (file-based, basic support)."""
        all_agents = await self.get_all()
        results = {}
        for agent_path, agent in all_agents.items():
            data = agent.model_dump(mode="json")
            match = True
            for field_name, condition in filter_dict.items():
                if isinstance(condition, dict) and "$exists" in condition:
                    has_field = data.get(field_name) is not None
                    if condition.get("$ne") is not None:
                        has_field = has_field and data.get(field_name) != condition["$ne"]
                    if condition["$exists"] != has_field:
                        match = False
                        break
            if match:
                results[agent_path] = data
        return results
