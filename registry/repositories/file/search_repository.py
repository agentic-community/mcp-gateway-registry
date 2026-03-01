"""File-based search repository using FAISS."""

import logging
from typing import Any, Dict, List, Optional

from ...core.config import settings
from ..interfaces import SearchRepositoryBase

logger = logging.getLogger(__name__)


class FaissSearchRepository(SearchRepositoryBase):
    """FAISS-based search repository."""

    def __init__(self):
        # Import FaissService lazily to avoid circular imports
        from ...search.service import faiss_service

        self.faiss_service = faiss_service

    async def index_entity(
        self, entity_path: str, entity_data: Dict[str, Any], entity_type: str, is_enabled: bool
    ) -> None:
        """Add or update entity in FAISS index."""
        await self.faiss_service.add_or_update_entity(
            entity_path=entity_path,
            entity_info=entity_data,
            entity_type=entity_type,
            is_enabled=is_enabled,
        )

    async def remove_entity(self, entity_path: str) -> None:
        """Remove entity from FAISS index."""
        await self.faiss_service.remove_entity(entity_path)

    async def search(
        self,
        query: str,
        entity_types: Optional[List[str]] = None,
        max_results: int = 10,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Search entities using FAISS.

        Args:
            query: Search query text
            entity_types: Optional list of entity types to filter by (e.g., ["mcp_server", "tool", "a2a_agent"])
            max_results: Maximum number of results per entity type

        Returns:
            Dictionary with entity types as keys and lists of results as values
        """
        return await self.faiss_service.search_mixed(
            query=query, entity_types=entity_types, max_results=max_results
        )

    async def rebuild_index(self) -> None:
        """Rebuild FAISS index from scratch."""
        await self.faiss_service.rebuild_index()

    async def initialize(self) -> None:
        """Initialize the search repository."""
        # FAISS service initializes itself
        pass

    async def index_server(
        self, server_path: str, server_data: Dict[str, Any], is_enabled: bool
    ) -> None:
        """Index a server."""
        await self.index_entity(server_path, server_data, "server", is_enabled)

    async def index_agent(
        self, agent_path: str, agent_data: Dict[str, Any], is_enabled: bool
    ) -> None:
        """Index an agent."""
        await self.index_entity(agent_path, agent_data, "agent", is_enabled)
