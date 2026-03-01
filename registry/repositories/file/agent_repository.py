"""File-based agent repository implementation."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

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

    async def get_all(self) -> Dict[str, AgentCard]:
        """Load all agents from disk."""
        agents = {}
        agent_files = [
            f for f in self.agents_dir.glob("**/*_agent.json") if f.name != self.state_file.name
        ]

        for file in agent_files:
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "path" in data and "name" in data:
                    agent = AgentCard(**data)
                    agents[agent.path] = agent
            except Exception as e:
                logger.error(f"Failed to load agent from {file}: {e}")

        return agents

    async def get(self, path: str) -> Optional[AgentCard]:
        """Get agent by path."""
        agents = await self.get_all()
        return agents.get(path)

    async def save(self, agent: AgentCard) -> AgentCard:
        """Save agent to disk."""
        if not agent.registered_at:
            agent.registered_at = datetime.now(timezone.utc)
        agent.updated_at = datetime.now(timezone.utc)

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

    async def get_state(self) -> Dict[str, List[str]]:
        """Load agent state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                if isinstance(state, dict):
                    return {
                        "enabled": state.get("enabled", []),
                        "disabled": state.get("disabled", []),
                    }
            except Exception as e:
                logger.error(f"Failed to load state: {e}")

        return {"enabled": [], "disabled": []}

    async def save_state(self, state: Dict[str, List[str]]) -> None:
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

    async def update(self, path: str, agent: AgentCard) -> Optional[AgentCard]:
        """Update an existing agent."""
        existing = await self.get(path)
        if not existing:
            return None
        return await self.save(agent)

    async def list_all(self) -> List[AgentCard]:
        """List all agents."""
        agents = await self.get_all()
        return list(agents.values())

    async def load_all(self) -> Dict[str, AgentCard]:
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
