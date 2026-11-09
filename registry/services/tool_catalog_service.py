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

from ..auth.tool_filter import filter_tools_for_user
from ..repositories.factory import get_server_repository
from ..repositories.interfaces import ServerRepositoryBase
from ..schemas.virtual_server_models import ToolCatalogEntry
from ..utils.tool_schema import stored_input_schema
from .visibility import user_can_access_server_from_doc

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
        user_context: dict[str, Any] | None = None,
    ) -> list[ToolCatalogEntry]:
        """Get all tools available across enabled servers.

        Reads tool_list from each server's MongoDB document and returns
        structured catalog entries, filtered by the user's server access.

        Access is enforced in two layers, matching the canonical server and
        semantic-search paths:

        1. Server access — the same scope-based check used by the server
           listing (:func:`registry.services.visibility.user_can_access_server_from_doc`);
           a server's tools are only considered when the caller can access
           that server.
        2. Tool access — the shared per-tool allowlist filter
           (:func:`registry.auth.tool_filter.filter_tools_for_user`); a caller
           with server access but a restricted tool set only sees the tools in
           that set.

        Both layers fail closed, so the catalog cannot disclose tools on
        servers (or individual tools) the caller has no scope to reach.

        The ``accessible_services`` UI-visibility dimension is intentionally
        NOT applied here, consistent with the other API-surface endpoints
        (``/v0.1/servers``): MCP-level server/tool scope is the authoritative
        gate for the catalog, and it is the stricter of the two.

        Args:
            server_path_filter: Optional filter to only return tools from
                a specific server path
            user_context: Authenticated caller's context (``is_admin``,
                ``accessible_servers``, ``username``, ``groups``). When
                ``None`` the caller is treated as unauthenticated and NO
                tools are returned (fail closed). Admins and callers with a
                wildcard server grant see all servers' tools.

        Returns:
            List of ToolCatalogEntry objects the caller may access
        """
        catalog: list[ToolCatalogEntry] = []

        # Fail closed: an unauthenticated / context-less caller sees nothing.
        # Every real request goes through nginx_proxied_auth, which always
        # supplies a user_context; None here means the access dimension is
        # unknown, so we must not disclose any server's tools.
        if user_context is None:
            logger.warning("Tool catalog requested without a user context; returning empty catalog")
            return catalog

        # Get all servers
        all_servers = await self._server_repo.list_all()

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

            server_name = server_info.get("server_name", path)

            # Enforce the canonical scope-based server-access check. This is the
            # SAME helper the server-listing / server-tool paths use, so the
            # catalog cannot expose tools from a server the caller lacks scope
            # for. Fails closed (omits the server) on any access uncertainty.
            if not user_can_access_server_from_doc(path, server_name, user_context):
                logger.debug("Filtering out server %s: caller lacks server access", path)
                continue

            # Apply the canonical per-tool allowlist filter on top of the
            # server-access check, matching the semantic-search path. A caller
            # granted server access but a restricted tool set must not see tool
            # names outside that set. filter_tools_for_user fails closed
            # (returns [] on a missing/empty allowlist) and passes through for
            # admin / wildcard callers.
            raw_tools = server_info.get("tool_list", [])
            tool_list = filter_tools_for_user(
                server_name=server_name,
                tools=raw_tools,
                user_context=user_context,
                endpoint="tool_catalog",
                server_path=path,
            )

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
                        input_schema=stored_input_schema(tool),
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
