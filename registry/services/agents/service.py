"""
Service for managing A2A agent registration and state.

This module provides CRUD operations for agent cards following the A2A protocol,
with file-based storage and enable/disable state management.

Based on: registry/services/server_service.py
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ...core.config import settings
from ...schemas.agent_models import AgentCard
from .indexing import (
    _index_agent_in_faiss,
)
from .ratings import (
    _apply_rating_update,
)
from .state import (
    _load_state_file,
    _persist_state_to_disk,
)
from .storage import (
    _load_agent_from_file,
    _path_to_filename,
    _save_agent_to_disk,
)

logger = logging.getLogger(__name__)


class AgentService:
    """Service for managing A2A agent registration and state."""

    def __init__(self):
        """Initialize agent service with empty state."""
        self.registered_agents: Dict[str, AgentCard] = {}
        self.agent_state: Dict[str, List[str]] = {"enabled": [], "disabled": []}


    def load_agents_and_state(self) -> None:
        """Load agent cards and persisted state from disk."""
        logger.info(f"Loading agent cards from {settings.agents_dir}...")

        # Create agents directory if it doesn't exist
        settings.agents_dir.mkdir(parents=True, exist_ok=True)

        temp_agents = {}
        # Only load files matching *_agent.json pattern (excludes FAISS metadata files)
        agent_files = list(settings.agents_dir.glob("**/*_agent.json"))

        # Additionally filter out agent_state.json if it somehow matches pattern
        agent_files = [
            f for f in agent_files
            if f.name != settings.agent_state_file_path.name
        ]

        logger.info(f"Found {len(agent_files)} agent files in {settings.agents_dir}")

        for file in agent_files:
            logger.debug(f"Loading agent from {file.relative_to(settings.agents_dir)}")

        if not agent_files:
            logger.warning(
                f"No agent definition files found in {settings.agents_dir}. "
                "Initializing empty agent registry."
            )
            self.registered_agents = {}
        else:
            for agent_file in agent_files:
                agent_data = _load_agent_from_file(agent_file)

                if agent_data:
                    agent_path = agent_data["path"]

                    if agent_path in temp_agents:
                        logger.warning(
                            f"Duplicate agent path in {agent_file}: {agent_path}. "
                            "Overwriting previous definition."
                        )

                    try:
                        # Validate by creating AgentCard instance
                        agent_card = AgentCard(**agent_data)
                        temp_agents[agent_path] = agent_card

                    except Exception as e:
                        logger.error(
                            f"Failed to validate agent card from {agent_file}: {e}"
                        )

            self.registered_agents = temp_agents
            logger.info(f"Successfully loaded {len(self.registered_agents)} agent cards")

        # Load persisted state
        self._load_agent_state()


    def _load_agent_state(self) -> None:
        """Load persisted agent state from disk."""
        state_data = _load_state_file(settings.agent_state_file_path)

        # Initialize state for all registered agents
        for path in self.registered_agents.keys():
            if path in state_data["enabled"]:
                continue
            elif path in state_data["disabled"]:
                continue
            else:
                # New agent not in state file - add to disabled
                state_data["disabled"].append(path)

        self.agent_state = state_data
        logger.info(
            f"Agent state initialized: {len(state_data['enabled'])} enabled, "
            f"{len(state_data['disabled'])} disabled"
        )


    def _persist_state(self) -> None:
        """Persist agent state to disk."""
        _persist_state_to_disk(self.agent_state, settings.agent_state_file_path)


    def register_agent(
        self,
        agent_card: AgentCard,
    ) -> AgentCard:
        """
        Register a new agent.

        Args:
            agent_card: Agent card to register

        Returns:
            Registered agent card

        Raises:
            ValueError: If agent path already exists
        """
        path = agent_card.path

        # Check if path already exists
        if path in self.registered_agents:
            logger.error(f"Agent registration failed: path '{path}' already exists")
            raise ValueError(f"Agent path '{path}' already exists")

        # Set registration metadata
        if not agent_card.registered_at:
            agent_card.registered_at = datetime.now(timezone.utc)
        if not agent_card.updated_at:
            agent_card.updated_at = datetime.now(timezone.utc)

        # Save to disk
        if not _save_agent_to_disk(agent_card, settings.agents_dir):
            raise ValueError(f"Failed to save agent '{agent_card.name}' to disk")

        # Add to in-memory registry and default to disabled
        self.registered_agents[path] = agent_card
        self.agent_state["disabled"].append(path)

        # Persist state
        self._persist_state()

        logger.info(
            f"New agent registered: '{agent_card.name}' at path '{path}' "
            f"(disabled by default)"
        )

        return agent_card


    def get_agent(
        self,
        path: str,
    ) -> AgentCard:
        """
        Get agent card by path.

        Args:
            path: Agent path

        Returns:
            Agent card

        Raises:
            ValueError: If agent not found
        """
        agent = self.registered_agents.get(path)

        if not agent:
            # Try alternate form (with/without trailing slash)
            if path.endswith("/"):
                alternate_path = path.rstrip("/")
            else:
                alternate_path = path + "/"

            agent = self.registered_agents.get(alternate_path)

        if not agent:
            raise ValueError(f"Agent not found at path: {path}")

        return agent


    def list_agents(self) -> List[AgentCard]:
        """
        List all registered agents.

        Returns:
            List of all agent cards
        """
        return list(self.registered_agents.values())

    def update_rating(
        self,
        path: str,
        username: str,
        rating: int,
    ) -> float:
        """
        Log a user rating for an agent. If the user has already rated, update their rating.

        Args:
            path: Agent path
            username: The user who submitted rating
            rating: integer between 1-5
        
        Return:
            Updated average rating

        Raises:
            ValueError: If agent not found
        """
        if path not in self.registered_agents:
            logger.error("Cannot update agent at path '%s': not found", path)
            raise ValueError(f"Agent not found at path: {path}")

        existing_agent = self.registered_agents[path]
        agent_dict = _apply_rating_update(
            existing_agent=existing_agent,
            username=username,
            rating=rating,
        )

        # Validate updated agent
        try:
            updated_agent = AgentCard(**agent_dict)
        except Exception as e:
            logger.error(f"Failed to validate updated agent: {e}")
            raise ValueError(f"Invalid agent update: {e}")

        # Save to disk
        if not _save_agent_to_disk(updated_agent, settings.agents_dir):
            raise ValueError(f"Failed to save updated agent to disk")

        # Update in-memory registry
        self.registered_agents[path] = updated_agent

        logger.info(f"Agent '{updated_agent.name}' ({path}) updated with rating {rating} from user {username}")

        return agent_dict["num_stars"]

    def update_agent(
        self,
        path: str,
        updates: Dict[str, Any],
    ) -> AgentCard:
        """
        Update an existing agent.

        Args:
            path: Agent path
            updates: Dictionary of fields to update

        Returns:
            Updated agent card

        Raises:
            ValueError: If agent not found
        """
        if path not in self.registered_agents:
            logger.error(f"Cannot update agent at path '{path}': not found")
            raise ValueError(f"Agent not found at path: {path}")

        # Get existing agent
        existing_agent = self.registered_agents[path]

        # Merge updates with existing data
        agent_dict = existing_agent.model_dump()
        agent_dict.update(updates)

        # Ensure path is consistent
        agent_dict["path"] = path

        # Update timestamp
        agent_dict["updated_at"] = datetime.now(timezone.utc)

        # Validate updated agent
        try:
            updated_agent = AgentCard(**agent_dict)
        except Exception as e:
            logger.error(f"Failed to validate updated agent: {e}")
            raise ValueError(f"Invalid agent update: {e}")

        # Save to disk
        if not _save_agent_to_disk(updated_agent, settings.agents_dir):
            raise ValueError(f"Failed to save updated agent to disk")

        # Update in-memory registry
        self.registered_agents[path] = updated_agent

        logger.info(f"Agent '{updated_agent.name}' ({path}) updated")

        return updated_agent


    def delete_agent(
        self,
        path: str,
    ) -> bool:
        """
        Delete an agent from registry.

        Args:
            path: Agent path

        Returns:
            True if deleted successfully

        Raises:
            ValueError: If agent not found
        """
        if path not in self.registered_agents:
            logger.error(f"Cannot delete agent at path '{path}': not found")
            raise ValueError(f"Agent not found at path: {path}")

        try:
            # Remove from file system
            filename = _path_to_filename(path)
            file_path = settings.agents_dir / filename

            if file_path.exists():
                file_path.unlink()
                logger.info(f"Removed agent file: {file_path}")
            else:
                logger.warning(f"Agent file not found: {file_path}")

            # Remove from in-memory registry
            agent_name = self.registered_agents[path].name
            del self.registered_agents[path]

            # Remove from state
            if path in self.agent_state["enabled"]:
                self.agent_state["enabled"].remove(path)
            if path in self.agent_state["disabled"]:
                self.agent_state["disabled"].remove(path)

            # Persist updated state
            self._persist_state()

            logger.info(f"Successfully deleted agent '{agent_name}' from path '{path}'")
            return True

        except Exception as e:
            logger.error(f"Failed to delete agent at path '{path}': {e}", exc_info=True)
            raise ValueError(f"Failed to delete agent: {e}")


    def enable_agent(
        self,
        path: str,
    ) -> None:
        """
        Enable an agent.

        Args:
            path: Agent path

        Raises:
            ValueError: If agent not found
        """
        if path not in self.registered_agents:
            raise ValueError(f"Agent not found at path: {path}")

        if path in self.agent_state["enabled"]:
            logger.info(f"Agent '{path}' is already enabled")
            return

        # Move from disabled to enabled
        if path in self.agent_state["disabled"]:
            self.agent_state["disabled"].remove(path)

        self.agent_state["enabled"].append(path)

        # Persist state
        self._persist_state()

        agent_name = self.registered_agents[path].name
        logger.info(f"Enabled agent '{agent_name}' ({path})")


    def disable_agent(
        self,
        path: str,
    ) -> None:
        """
        Disable an agent.

        Args:
            path: Agent path

        Raises:
            ValueError: If agent not found
        """
        if path not in self.registered_agents:
            raise ValueError(f"Agent not found at path: {path}")

        if path in self.agent_state["disabled"]:
            logger.info(f"Agent '{path}' is already disabled")
            return

        # Move from enabled to disabled
        if path in self.agent_state["enabled"]:
            self.agent_state["enabled"].remove(path)

        self.agent_state["disabled"].append(path)

        # Persist state
        self._persist_state()

        agent_name = self.registered_agents[path].name
        logger.info(f"Disabled agent '{agent_name}' ({path})")


    def is_agent_enabled(
        self,
        path: str,
    ) -> bool:
        """
        Check if agent is enabled.

        Args:
            path: Agent path

        Returns:
            True if enabled, False otherwise
        """
        # Try exact match first
        if path in self.agent_state["enabled"]:
            return True

        # Try alternate form (with/without trailing slash)
        if path.endswith("/"):
            alternate_path = path.rstrip("/")
        else:
            alternate_path = path + "/"

        return alternate_path in self.agent_state["enabled"]


    def get_enabled_agents(self) -> List[str]:
        """
        Get list of enabled agent paths.

        Returns:
            List of enabled agent paths
        """
        return list(self.agent_state["enabled"])


    def get_disabled_agents(self) -> List[str]:
        """
        Get list of disabled agent paths.

        Returns:
            List of disabled agent paths
        """
        return list(self.agent_state["disabled"])


    async def index_agent(
        self,
        agent_card: AgentCard,
    ) -> None:
        """
        Add agent to FAISS search index.

        Args:
            agent_card: Agent card to index
        """
        await _index_agent_in_faiss(
            agent_card=agent_card,
            is_enabled=self.is_agent_enabled(agent_card.path),
        )


    def get_agent_info(
        self,
        path: str,
    ) -> Optional[AgentCard]:
        """
        Get agent by path (returns None if not found).

        Args:
            path: Agent path

        Returns:
            Agent card or None if not found
        """
        try:
            return self.get_agent(path)
        except ValueError:
            return None


    def get_all_agents(self) -> List[AgentCard]:
        """
        Get all registered agents.

        Returns:
            List of all agent cards
        """
        return self.list_agents()


    def remove_agent(
        self,
        path: str,
    ) -> bool:
        """
        Remove an agent from registry.

        Args:
            path: Agent path

        Returns:
            True if successful, False otherwise
        """
        try:
            self.delete_agent(path)
            return True
        except ValueError:
            return False


    def toggle_agent(
        self,
        path: str,
        enabled: bool,
    ) -> bool:
        """
        Toggle agent enabled/disabled state.

        Args:
            path: Agent path
            enabled: New enabled state

        Returns:
            True if successful, False otherwise
        """
        try:
            if enabled:
                self.enable_agent(path)
            else:
                self.disable_agent(path)
            return True
        except ValueError:
            return False


# Global service instance
agent_service = AgentService()
