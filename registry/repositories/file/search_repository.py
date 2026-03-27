"""File-based search repository using FAISS."""

import logging
from typing import Any

from ..interfaces import SearchRepositoryBase

logger = logging.getLogger(__name__)


class FaissSearchRepository(SearchRepositoryBase):
    """FAISS-based search repository."""

    def __init__(self):
        # Import FaissService lazily to avoid circular imports
        from ...search.service import faiss_service

        self.faiss_service = faiss_service

    async def index_entity(
        self, entity_path: str, entity_data: dict[str, Any], entity_type: str, is_enabled: bool
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

    async def search_by_tags(
        self,
        tags: list[str],
        entity_types: list[str] | None = None,
        max_results: int = 10,
    ) -> dict[str, list[dict[str, Any]]]:
        """Search entities by exact tag match from FAISS metadata store."""
        required = {t.lower() for t in tags}
        results: dict[str, list[dict[str, Any]]] = {
            "servers": [],
            "tools": [],
            "agents": [],
            "skills": [],
            "virtual_servers": [],
        }
        for path, metadata in self.faiss_service.metadata_store.items():
            entity_tags = {t.lower() for t in metadata.get("tags", [])}
            if not required.issubset(entity_tags):
                continue
            entity_type = metadata.get("entity_type", "")
            if entity_types and entity_type not in entity_types:
                continue
            entry = {
                "path": path,
                "server_name": metadata.get("server_name", metadata.get("name", "")),
                "description": metadata.get("description", ""),
                "tags": metadata.get("tags", []),
                "is_enabled": metadata.get("is_enabled", False),
                "relevance_score": 1.0,
                "match_context": metadata.get("description", ""),
                "matching_tools": [],
            }
            if entity_type == "mcp_server":
                entry["num_tools"] = metadata.get("num_tools", 0)
                results["servers"].append(entry)
            elif entity_type == "a2a_agent":
                results["agents"].append(entry)
            elif entity_type == "skill":
                entry["skill_name"] = metadata.get("name", "")
                results["skills"].append(entry)
            elif entity_type == "virtual_server":
                results["virtual_servers"].append(entry)
        # Limit each group
        for key in results:
            results[key] = results[key][:max_results]
        return results

    async def get_all_tags(self) -> list[str]:
        """Return a sorted list of all unique tags from the FAISS metadata store."""
        tags_set: set[str] = set()
        for metadata in self.faiss_service.metadata_store.values():
            for tag in metadata.get("tags", []):
                if tag:
                    tags_set.add(tag)
        return sorted(tags_set, key=str.lower)

    async def search(
        self,
        query: str,
        entity_types: list[str] | None = None,
        max_results: int = 10,
    ) -> dict[str, list[dict[str, Any]]]:
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
        # Explicitly initialize the shared FAISS service used by this repository.
        await self.faiss_service.initialize()

    async def index_server(
        self, server_path: str, server_data: dict[str, Any], is_enabled: bool
    ) -> None:
        """Index a server."""
        await self.index_entity(server_path, server_data, "mcp_server", is_enabled)

    async def index_agent(
        self, agent_path: str, agent_data: dict[str, Any], is_enabled: bool
    ) -> None:
        """Index an agent."""
        await self.index_entity(agent_path, agent_data, "a2a_agent", is_enabled)
