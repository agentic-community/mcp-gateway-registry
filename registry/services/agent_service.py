"""
Service for managing A2A agent registration and state.

This module provides CRUD operations for agent cards following the A2A protocol,
using repository pattern for storage abstraction.

Based on: registry/services/server_service.py
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..repositories.factory import get_agent_repository, get_search_repository
from ..repositories.interfaces import AgentRepositoryBase, SearchRepositoryBase
from ..schemas.agent_models import AgentCard


logger = logging.getLogger(__name__)


class AgentService:
    """Service for managing A2A agent registration and state."""

    def __init__(self):
        """Initialize agent service with repository."""
        self._repo: AgentRepositoryBase = get_agent_repository()
        self._search_repo: SearchRepositoryBase = get_search_repository()
        self.registered_agents: Dict[str, AgentCard] = {}
        self.agent_state: Dict[str, List[str]] = {"enabled": [], "disabled": []}

    async def load_agents_and_state(self) -> None:
        """Load agent cards and persisted state from repository."""
        logger.info("Loading agent cards from repository...")

        # Load agents from storage first (OpenSearch, file, etc.)
        await self._repo.load_all()

        # Now get the list of loaded agents
        agents_list = await self._repo.list_all()
        self.registered_agents = {agent.path: agent for agent in agents_list}
        logger.info(f"Successfully loaded {len(self.registered_agents)} agent cards")

        await self._load_agent_state()

    async def _load_agent_state(self) -> None:
        """Load persisted agent state from repository."""
        state_data = await self._repo.get_state()

        # Initialize state for all registered agents
        for path in self.registered_agents.keys():
            if path not in state_data["enabled"] and path not in state_data["disabled"]:
                state_data["disabled"].append(path)

        self.agent_state = state_data
        await self._repo.save_state(state_data)
        logger.info(
            f"Agent state initialized: {len(state_data['enabled'])} enabled, "
            f"{len(state_data['disabled'])} disabled"
        )

    async def _persist_state(self) -> None:
        """Persist agent state to repository."""
        await self._repo.save_state(self.agent_state)

    async def register_agent(
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

        if path in self.registered_agents:
            logger.error(f"Agent registration failed: path '{path}' already exists")
            raise ValueError(f"Agent path '{path}' already exists")

        # Save to repository
        agent_card = await self._repo.create(agent_card)

        # Add to in-memory registry and default to disabled
        self.registered_agents[path] = agent_card
        self.agent_state["disabled"].append(path)
        await self._persist_state()

        # Index in search backend
        try:
            is_enabled = self.is_agent_enabled(path)
            await self._search_repo.index_agent(path, agent_card, is_enabled)
        except Exception as e:
            logger.error(f"Failed to index agent {path}: {e}")
            # Don't fail the primary operation

        logger.info(
            f"New agent registered: '{agent_card.name}' at path '{path}' (disabled by default)"
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

    async def update_rating(
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
            ValueError: If agent not found or invalid rating
        """
        from . import rating_service

        # Query repository directly instead of using cache
        existing_agent = await self._repo.get(path)
        if not existing_agent:
            logger.error(f"Cannot update agent at path '{path}': not found")
            raise ValueError(f"Agent not found at path: {path}")

        # Validate rating using shared service
        rating_service.validate_rating(rating)

        # Convert to dict for modification
        agent_dict = existing_agent.model_dump()

        # Ensure rating_details is a list
        if "rating_details" not in agent_dict or agent_dict["rating_details"] is None:
            agent_dict["rating_details"] = []

        # Update rating details using shared service
        updated_details, is_new_rating = rating_service.update_rating_details(
            agent_dict["rating_details"], username, rating
        )
        agent_dict["rating_details"] = updated_details

        # Calculate average rating using shared service
        agent_dict["num_stars"] = rating_service.calculate_average_rating(
            agent_dict["rating_details"]
        )

        # Save to repository (this will handle AOSS eventual consistency)
        await self._repo.update(path, agent_dict)

        # Update in-memory registry
        try:
            updated_agent = AgentCard(**agent_dict)
            self.registered_agents[path] = updated_agent
        except Exception as e:
            logger.warning(f"Failed to update in-memory agent cache: {e}")

        logger.info(
            f"Updated rating for agent {path}: user {username} rated {rating}, "
            f"new average: {agent_dict['num_stars']:.2f}"
        )

        return agent_dict["num_stars"]

    async def update_agent(
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

        existing_agent = self.registered_agents[path]
        agent_dict = existing_agent.model_dump()
        agent_dict.update(updates)
        agent_dict["path"] = path
        agent_dict["updated_at"] = datetime.now(timezone.utc)

        try:
            updated_agent = AgentCard(**agent_dict)
        except Exception as e:
            logger.error(f"Failed to validate updated agent: {e}")
            raise ValueError(f"Invalid agent update: {e}")

        # Save to repository
        updated_agent = await self._repo.save(updated_agent)
        self.registered_agents[path] = updated_agent

        # Re-index in search backend
        try:
            is_enabled = self.is_agent_enabled(path)
            await self._search_repo.index_agent(path, updated_agent, is_enabled)
        except Exception as e:
            logger.error(f"Failed to re-index agent {path}: {e}")
            # Don't fail the primary operation

        logger.info(f"Agent '{updated_agent.name}' ({path}) updated")
        return updated_agent

    async def delete_agent(
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
            agent_name = self.registered_agents[path].name

            # Delete from repository
            await self._repo.delete(path)

            # Remove from in-memory registry
            del self.registered_agents[path]

            # Remove from state
            if path in self.agent_state["enabled"]:
                self.agent_state["enabled"].remove(path)
            if path in self.agent_state["disabled"]:
                self.agent_state["disabled"].remove(path)

            await self._persist_state()

            # Remove from search backend
            try:
                await self._search_repo.remove_entity(path)
            except Exception as e:
                logger.error(f"Failed to remove agent {path} from search: {e}")
                # Don't fail the primary operation

            logger.info(f"Successfully deleted agent '{agent_name}' from path '{path}'")
            return True

        except Exception as e:
            logger.error(f"Failed to delete agent at path '{path}': {e}", exc_info=True)
            raise ValueError(f"Failed to delete agent: {e}")

    async def enable_agent(
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

        if path in self.agent_state["disabled"]:
            self.agent_state["disabled"].remove(path)
        self.agent_state["enabled"].append(path)

        await self._persist_state()

        agent_name = self.registered_agents[path].name
        logger.info(f"Enabled agent '{agent_name}' ({path})")

    async def disable_agent(
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

        if path in self.agent_state["enabled"]:
            self.agent_state["enabled"].remove(path)
        self.agent_state["disabled"].append(path)

        await self._persist_state()

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
        Add agent to search index.

        Args:
            agent_card: Agent card to index
        """
        try:
            agent_data = agent_card.model_dump(mode="json")
            await self._search_repo.index_entity(
                entity_path=agent_card.path,
                entity_data=agent_data,
                entity_type="a2a_agent",
                is_enabled=self.is_agent_enabled(agent_card.path),
            )
            logger.info(f"Indexed agent '{agent_card.name}' in search")
        except Exception as e:
            logger.error(f"Failed to index agent: {e}", exc_info=True)

    async def get_agent_info(
        self,
        path: str,
    ) -> Optional[AgentCard]:
        """
        Get agent by path - queries repository directly (returns None if not found).

        Args:
            path: Agent path

        Returns:
            Agent card or None if not found
        """
        return await self._repo.get(path)

    async def get_all_agents(self) -> List[AgentCard]:
        """
        Get all registered agents - queries repository directly.

        Returns:
            List of all agent cards
        """
        # Query repository directly instead of using cache
        return await self._repo.list_all()

    async def remove_agent(
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
            await self.delete_agent(path)
            return True
        except ValueError:
            return False

    async def toggle_agent(
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
                await self.enable_agent(path)
            else:
                await self.disable_agent(path)
            return True
        except ValueError:
            return False


# Global service instance
agent_service = AgentService()
