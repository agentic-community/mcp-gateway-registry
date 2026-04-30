"""DocumentDB-based repository for A2A agent storage."""

import logging
from datetime import datetime
from typing import Any

from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo.errors import DuplicateKeyError

from ...schemas.agent_models import AgentCard
from ..interfaces import AgentRepositoryBase
from .client import get_collection_name, get_documentdb_client

logger = logging.getLogger(__name__)


class DocumentDBAgentRepository(AgentRepositoryBase):
    """DocumentDB implementation of agent repository."""

    def __init__(self):
        self._collection: AsyncIOMotorCollection | None = None
        self._collection_name = get_collection_name("mcp_agents")

    async def _get_collection(self) -> AsyncIOMotorCollection:
        """Get DocumentDB collection."""
        if self._collection is None:
            db = await get_documentdb_client()
            self._collection = db[self._collection_name]
        return self._collection

    async def load_all(self) -> None:
        """Load all agents from DocumentDB."""
        logger.info(f"Loading agents from DocumentDB collection: {self._collection_name}")
        collection = await self._get_collection()

        try:
            count = await collection.count_documents({})
            logger.info(f"Loaded {count} agents from DocumentDB")
        except Exception as e:
            logger.error(f"Error loading agents from DocumentDB: {e}", exc_info=True)

    async def get(
        self,
        path: str,
    ) -> AgentCard | None:
        """Get agent by path."""
        collection = await self._get_collection()

        try:
            agent_doc = await collection.find_one({"_id": path})
            if not agent_doc:
                return None

            agent_doc["path"] = agent_doc.pop("_id")
            return AgentCard(**agent_doc)
        except Exception as e:
            logger.error(f"Error getting agent '{path}' from DocumentDB: {e}", exc_info=True)
            return None

    async def list_all(self) -> list[AgentCard]:
        """List all agents."""
        collection = await self._get_collection()

        try:
            cursor = collection.find({})
            agents = []
            async for doc in cursor:
                path = doc.pop("_id")
                doc["path"] = path
                try:
                    agent_card = AgentCard(**doc)
                    agents.append(agent_card)
                except Exception as e:
                    logger.error(f"Failed to parse agent {path}: {e}")
            return agents
        except Exception as e:
            logger.error(f"Error listing agents from DocumentDB: {e}", exc_info=True)
            return []

    async def list_paginated(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> list[AgentCard]:
        """List agents with DB-level skip/limit pagination."""
        collection = await self._get_collection()

        try:
            cursor = collection.find({}).sort("_id", 1).skip(skip).limit(limit)
            agents = []
            async for doc in cursor:
                path = doc.pop("_id")
                doc["path"] = path
                try:
                    agents.append(AgentCard(**doc))
                except Exception as e:
                    logger.warning(f"Skipping invalid agent document {path}: {e}")
            return agents
        except Exception as e:
            logger.error(f"Error listing paginated agents from DocumentDB: {e}", exc_info=True)
            return []

    async def create(
        self,
        agent: AgentCard,
    ) -> AgentCard:
        """Create a new agent."""
        path = agent.path
        collection = await self._get_collection()

        if not agent.registered_at:
            agent.registered_at = datetime.utcnow()
        if not agent.updated_at:
            agent.updated_at = datetime.utcnow()

        agent_dict = agent.model_dump(mode="json")
        agent_dict["is_enabled"] = False

        try:
            doc = {**agent_dict}
            doc["_id"] = path
            doc.pop("path", None)

            await collection.insert_one(doc)
            logger.info(f"Created agent '{agent.name}' at '{path}'")
            return agent
        except DuplicateKeyError:
            logger.error(f"Agent path '{path}' already exists")
            raise ValueError(f"Agent path '{path}' already exists")
        except Exception as e:
            logger.error(f"Failed to create agent in DocumentDB: {e}", exc_info=True)
            raise ValueError(f"Failed to create agent: {e}")

    async def update(
        self,
        path: str,
        updates: dict[str, Any],
    ) -> AgentCard:
        """Update an existing agent."""
        existing_agent = await self.get(path)
        if not existing_agent:
            logger.error(f"Cannot update agent at '{path}': not found")
            raise ValueError(f"Agent not found at path: {path}")

        collection = await self._get_collection()

        agent_dict = existing_agent.model_dump()
        agent_dict.update(updates)
        agent_dict["updated_at"] = datetime.utcnow()

        try:
            updated_agent = AgentCard(**agent_dict)
        except Exception as e:
            logger.error(f"Failed to validate updated agent: {e}")
            raise ValueError(f"Invalid agent update: {e}")

        update_dict = updated_agent.model_dump(mode="json")
        update_dict.pop("path", None)

        try:
            result = await collection.update_one({"_id": path}, {"$set": update_dict})

            if result.matched_count == 0:
                raise ValueError(f"Agent at '{path}' not found in DocumentDB")

            logger.info(f"Updated agent '{updated_agent.name}' ({path})")
            return updated_agent
        except Exception as e:
            logger.error(f"Failed to update agent in DocumentDB: {e}", exc_info=True)
            raise ValueError(f"Failed to update agent: {e}")

    async def delete(
        self,
        path: str,
    ) -> bool:
        """Delete an agent."""
        collection = await self._get_collection()

        try:
            agent_doc = await collection.find_one({"_id": path})
            if not agent_doc:
                logger.error(f"Agent at '{path}' not found in DocumentDB")
                return False

            agent_name = agent_doc.get("name", "Unknown")

            result = await collection.delete_one({"_id": path})

            if result.deleted_count == 0:
                logger.error(f"Failed to delete agent at '{path}'")
                return False

            logger.info(f"Deleted agent '{agent_name}' from '{path}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete agent from DocumentDB: {e}", exc_info=True)
            return False

    async def get_state(
        self,
        path: str,
    ) -> bool:
        """Get enabled/disabled state for a single agent."""
        agent = await self.get(path)
        if agent:
            return getattr(agent, "is_enabled", False)
        return False

    async def get_all_states(self) -> dict[str, bool]:
        """Get enabled/disabled state for all agents in a single query."""
        collection = await self._get_collection()

        try:
            cursor = collection.find({}, {"_id": 1, "is_enabled": 1})
            states: dict[str, bool] = {}
            async for doc in cursor:
                agent_path = doc.get("_id")
                if agent_path:
                    states[agent_path] = doc.get("is_enabled", False)
            return states
        except Exception as e:
            logger.error(f"Error getting all agent states from DocumentDB: {e}", exc_info=True)
            return {}

    async def set_state(
        self,
        path: str,
        enabled: bool,
    ) -> bool:
        """Set agent enabled/disabled state."""
        collection = await self._get_collection()

        try:
            agent_doc = await collection.find_one({"_id": path})
            if not agent_doc:
                logger.error(f"Agent at '{path}' not found in DocumentDB")
                return False

            agent_name = agent_doc.get("name", path)

            result = await collection.update_one(
                {"_id": path},
                {"$set": {"is_enabled": enabled, "updated_at": datetime.utcnow().isoformat()}},
            )

            if result.matched_count == 0:
                logger.error(f"Agent at '{path}' not found")
                return False

            logger.info(f"Toggled '{agent_name}' ({path}) to {enabled}")
            return True
        except Exception as e:
            logger.error(f"Failed to update agent state in DocumentDB: {e}", exc_info=True)
            return False

    async def save_state(
        self,
        state: dict[str, list[str]],
    ) -> None:
        """Save agent state (compatibility method for file repository interface)."""
        logger.debug(
            f"Updated agent state cache: {len(state['enabled'])} enabled, "
            f"{len(state['disabled'])} disabled"
        )

    async def count(self) -> int:
        """Get total count of agents.

        Returns:
            Total number of agents in the repository.
        """
        collection = await self._get_collection()

        try:
            count = await collection.count_documents({})
            logger.debug(f"DocumentDB COUNT: Found {count} agents")
            return count
        except Exception as e:
            logger.error(f"Error counting agents in DocumentDB: {e}", exc_info=True)
            return 0

    async def update_field(
        self,
        path: str,
        field: str,
        value: Any,
    ) -> bool:
        """Update a single field on a document."""
        collection = await self._get_collection()

        if value is None:
            result = await collection.update_one(
                {"_id": path},
                {"$unset": {field: ""}},
            )
        else:
            result = await collection.update_one(
                {"_id": path},
                {"$set": {field: value}},
            )

        return result.modified_count > 0

    async def find_with_filter(
        self,
        filter_dict: dict[str, Any],
    ) -> dict[str, dict]:
        """Find documents matching a MongoDB-style filter."""
        collection = await self._get_collection()
        cursor = collection.find(filter_dict)
        results = {}
        async for doc in cursor:
            doc_id = doc.pop("_id", None)
            if doc_id:
                results[doc_id] = doc
        return results
