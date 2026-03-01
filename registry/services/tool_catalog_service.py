"""
Service layer for the global tool catalog.

Aggregates tools from all enabled, active MCP servers to provide
a browsable catalog for building virtual server configurations.
"""

import logging
from typing import (
    Any,
    Optional,
)

from ..repositories.factory import get_server_repository
from ..repositories.interfaces import ServerRepositoryBase
from ..schemas.virtual_server_models import ToolCatalogEntry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


# Singleton instance
_tool_catalog_service: Optional["ToolCatalogService"] = None


class ToolCatalogService:
    """Service for aggregating tools across all registered backend servers."""

    def __init__(self):
        self._server_repo: ServerRepositoryBase = get_server_repository()

    async def get_tool_catalog(
        self,
        server_path_filter: str | None = None,
        user_scopes: list[str] | None = None,
    ) -> list[ToolCatalogEntry]:
        """Get all tools available across enabled servers.

        Reads tool_list from each server's MongoDB document and returns
        structured catalog entries, filtered by the user's scopes.

        Args:
            server_path_filter: Optional filter to only return tools from
                a specific server path
            user_scopes: User's scopes for access filtering. If None,
                no scope filtering is applied (backwards-compatible).

        Returns:
            List of ToolCatalogEntry objects the user has access to
        """
        catalog: list[ToolCatalogEntry] = []

        # Get all servers
        all_servers = await self._server_repo.list_all()

        # Pre-compute user scope set for efficient lookup
        user_scope_set: set[str] | None = None
        if user_scopes is not None:
            user_scope_set = set(user_scopes)

        for path, server_info in all_servers.items():
            # Skip version documents (contain ":" in path)
            if ":" in path:
                continue

            # Apply server path filter if specified (normalize slashes for comparison)
            if server_path_filter:
                normalized_filter = server_path_filter.strip("/")
                normalized_path = path.strip("/")
                if normalized_path != normalized_filter:
                    continue

            # Check if server is enabled
            is_enabled = await self._server_repo.get_state(path)
            if not is_enabled:
                continue

            # Filter by user's accessible servers if scopes are provided
            if user_scope_set is not None:
                server_required_scopes = server_info.get("required_scopes", [])
                if server_required_scopes and not all(
                    s in user_scope_set for s in server_required_scopes
                ):
                    logger.debug(f"Filtering out server {path}: user lacks required scopes")
                    continue

            server_name = server_info.get("server_name", path)
            tool_list = server_info.get("tool_list", [])

            # Get available versions from other_version_ids
            available_versions = self._get_available_versions(server_info)

            for tool in tool_list:
                tool_name = tool.get("name", "")
                if not tool_name:
                    continue

                catalog.append(
                    ToolCatalogEntry(
                        tool_name=tool_name,
                        server_path=path,
                        server_name=server_name,
                        description=tool.get("description", ""),
                        input_schema=tool.get("inputSchema", {}),
                        available_versions=available_versions,
                    )
                )

        logger.debug(
            f"Tool catalog: {len(catalog)} tools from "
            f"{len(set(e.server_path for e in catalog))} servers"
        )
        return catalog

    def _get_available_versions(
        self,
        server_info: dict[str, Any],
    ) -> list[str]:
        """Extract available versions for a server.

        Args:
            server_info: Server document from repository

        Returns:
            List of version strings
        """
        versions = []

        # Current/active version
        current_version = server_info.get("version")
        if current_version:
            versions.append(current_version)

        # Other versions from linked version documents
        other_version_ids = server_info.get("other_version_ids", [])
        for version_id in other_version_ids:
            # Version IDs are like "/context7:v1.5.0"
            if ":" in version_id:
                version_str = version_id.split(":")[-1]
                if version_str and version_str not in versions:
                    versions.append(version_str)

        return versions


def get_tool_catalog_service() -> ToolCatalogService:
    """Get tool catalog service singleton."""
    global _tool_catalog_service

    if _tool_catalog_service is not None:
        return _tool_catalog_service

    _tool_catalog_service = ToolCatalogService()
    return _tool_catalog_service
