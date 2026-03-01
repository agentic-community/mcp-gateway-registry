"""
Service layer for virtual MCP server management.

Handles validation, CRUD operations, tool resolution, and nginx config
regeneration for virtual servers that aggregate tools from multiple backends.
"""

import asyncio
import logging
import re
from datetime import UTC, datetime
from typing import (
    Optional,
)

from ..exceptions import (
    VirtualServerNotFoundError,
    VirtualServerValidationError,
)
from ..repositories.factory import (
    get_search_repository,
    get_server_repository,
    get_virtual_server_repository,
)
from ..repositories.interfaces import (
    ServerRepositoryBase,
    VirtualServerRepositoryBase,
)
from ..schemas.virtual_server_models import (
    CreateVirtualServerRequest,
    ResolvedTool,
    ToolMapping,
    UpdateVirtualServerRequest,
    VirtualServerConfig,
    VirtualServerInfo,
)
from ..services.rating_service import (
    calculate_average_rating,
    update_rating_details,
    validate_rating,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


# Singleton instance
_virtual_server_service: Optional["VirtualServerService"] = None

# Lock to serialize nginx config regeneration across concurrent mutations
_nginx_reload_lock = asyncio.Lock()


def _generate_path_from_name(
    name: str,
) -> str:
    """Generate a virtual server path from a name.

    Converts to lowercase, replaces spaces/special chars with hyphens,
    and prepends /virtual/.

    Args:
        name: Human-readable server name

    Returns:
        Path like /virtual/dev-essentials
    """
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    if not slug:
        slug = "virtual-server"
    return f"/virtual/{slug}"


def _get_unique_backends(
    tool_mappings: list[ToolMapping],
) -> list[str]:
    """Extract unique backend server paths from tool mappings."""
    return list({tm.backend_server_path for tm in tool_mappings})


def _get_effective_tool_name(
    mapping: ToolMapping,
) -> str:
    """Get the effective tool name (alias if set, otherwise original)."""
    return mapping.alias if mapping.alias else mapping.tool_name


class VirtualServerService:
    """Service for managing virtual MCP server configurations."""

    def __init__(self):
        self._repo: VirtualServerRepositoryBase = get_virtual_server_repository()
        self._server_repo: ServerRepositoryBase = get_server_repository()

    async def list_virtual_servers(self) -> list[VirtualServerInfo]:
        """List all virtual servers with summary information.

        Returns:
            List of VirtualServerInfo summaries
        """
        configs = await self._repo.list_all()
        return [self._config_to_info(c) for c in configs]

    async def get_virtual_server(
        self,
        path: str,
    ) -> VirtualServerConfig | None:
        """Get a virtual server by path.

        Args:
            path: Virtual server path

        Returns:
            VirtualServerConfig if found, None otherwise
        """
        return await self._repo.get(path)

    async def create_virtual_server(
        self,
        request: CreateVirtualServerRequest,
        created_by: str | None = None,
    ) -> VirtualServerConfig:
        """Create a new virtual server.

        Validates all backend references and tool mappings before creation.

        Args:
            request: Creation request with server config
            created_by: Username of creator

        Returns:
            Created VirtualServerConfig

        Raises:
            VirtualServerValidationError: If validation fails
            VirtualServerAlreadyExistsError: If path already exists
        """
        # Generate path from name if not provided
        path = request.path
        if not path:
            path = _generate_path_from_name(request.server_name)

        # Ensure path starts with /virtual/
        if not path.startswith("/virtual/"):
            path = f"/virtual/{path.strip('/')}"

        # Validate tool mappings
        if request.tool_mappings:
            await self._validate_tool_mappings(request.tool_mappings)

        # Validate unique tool names/aliases
        self._validate_unique_tool_names(request.tool_mappings)

        now = datetime.now(UTC)
        config = VirtualServerConfig(
            path=path,
            server_name=request.server_name,
            description=request.description,
            tool_mappings=request.tool_mappings,
            required_scopes=request.required_scopes,
            tool_scope_overrides=request.tool_scope_overrides,
            tags=request.tags,
            supported_transports=request.supported_transports,
            is_enabled=False,
            created_by=created_by,
            created_at=now,
            updated_at=now,
        )

        result = await self._repo.create(config)
        logger.info(
            f"Created virtual server '{config.server_name}' at {config.path} "
            f"with {len(config.tool_mappings)} tools"
        )

        await self._trigger_nginx_reload()
        await self._index_for_search(result)
        return result

    async def update_virtual_server(
        self,
        path: str,
        request: UpdateVirtualServerRequest,
    ) -> VirtualServerConfig | None:
        """Update an existing virtual server.

        Args:
            path: Virtual server path
            request: Update request with changed fields

        Returns:
            Updated VirtualServerConfig if found

        Raises:
            VirtualServerNotFoundError: If not found
            VirtualServerValidationError: If validation fails
        """
        existing = await self._repo.get(path)
        if not existing:
            raise VirtualServerNotFoundError(path)

        updates = request.model_dump(exclude_unset=True)

        # Validate tool mappings if being updated
        if "tool_mappings" in updates and updates["tool_mappings"]:
            tool_mappings = [ToolMapping(**tm) for tm in updates["tool_mappings"]]
            await self._validate_tool_mappings(tool_mappings)
            self._validate_unique_tool_names(tool_mappings)

        result = await self._repo.update(path, updates)

        if result:
            await self._trigger_nginx_reload()
            await self._index_for_search(result)
            logger.info(f"Updated virtual server: {path}")

        return result

    async def delete_virtual_server(
        self,
        path: str,
    ) -> bool:
        """Delete a virtual server.

        Args:
            path: Virtual server path

        Returns:
            True if deleted

        Raises:
            VirtualServerNotFoundError: If not found
        """
        existing = await self._repo.get(path)
        if not existing:
            raise VirtualServerNotFoundError(path)

        success = await self._repo.delete(path)

        if success:
            await self._trigger_nginx_reload()
            await self._remove_from_search(path)
            logger.info(f"Deleted virtual server: {path}")
        return success

    async def toggle_virtual_server(
        self,
        path: str,
        enabled: bool,
    ) -> bool:
        """Toggle virtual server enabled/disabled state.

        Args:
            path: Virtual server path
            enabled: New enabled state

        Returns:
            True if toggled successfully

        Raises:
            VirtualServerNotFoundError: If not found
            VirtualServerValidationError: If enabling with no tool mappings
        """
        existing = await self._repo.get(path)
        if not existing:
            raise VirtualServerNotFoundError(path)

        # Validate before enabling
        if enabled and not existing.tool_mappings:
            raise VirtualServerValidationError("Cannot enable virtual server with no tool mappings")

        if enabled:
            # Re-validate tool mappings before enabling
            await self._validate_tool_mappings(existing.tool_mappings)

        success = await self._repo.set_state(path, enabled)

        if success:
            await self._trigger_nginx_reload()
            # Re-index with new enabled state
            updated = await self._repo.get(path)
            if updated:
                await self._index_for_search(updated)
            logger.info(f"Virtual server {path} {'enabled' if enabled else 'disabled'}")

        return success

    async def resolve_tools(
        self,
        path: str,
    ) -> list[ResolvedTool]:
        """Resolve all tools for a virtual server.

        Fetches tool metadata from backend servers and applies
        aliases, version pins, and scope overrides.

        Args:
            path: Virtual server path

        Returns:
            List of resolved tools with full metadata
        """
        config = await self._repo.get(path)
        if not config:
            raise VirtualServerNotFoundError(path)

        return await self._resolve_tool_list(config)

    async def rate_virtual_server(
        self,
        path: str,
        username: str,
        rating: int,
    ) -> dict:
        """Rate a virtual server.

        Args:
            path: Virtual server path
            username: Username submitting the rating
            rating: Rating value (1-5)

        Returns:
            Dict with average_rating and is_new_rating

        Raises:
            VirtualServerNotFoundError: If virtual server not found
            ValueError: If rating is invalid
        """
        validate_rating(rating)

        config = await self._repo.get(path)
        if not config:
            raise VirtualServerNotFoundError(path)

        rating_details = config.rating_details or []
        updated_details, is_new = update_rating_details(
            rating_details,
            username,
            rating,
        )
        average = calculate_average_rating(updated_details)

        success = await self._repo.update_rating(path, average, updated_details)
        if not success:
            raise VirtualServerNotFoundError(path)

        logger.info(
            f"User '{username}' rated virtual server '{path}': {rating} stars "
            f"(new avg: {average:.2f}, new={is_new})"
        )

        return {
            "average_rating": average,
            "is_new_rating": is_new,
            "total_ratings": len(updated_details),
        }

    async def get_virtual_server_rating(
        self,
        path: str,
    ) -> dict:
        """Get rating information for a virtual server.

        Args:
            path: Virtual server path

        Returns:
            Dict with num_stars and rating_details

        Raises:
            VirtualServerNotFoundError: If virtual server not found
        """
        rating_info = await self._repo.get_rating(path)
        if rating_info is None:
            raise VirtualServerNotFoundError(path)

        return rating_info

    async def _validate_tool_mappings(
        self,
        tool_mappings: list[ToolMapping],
    ) -> None:
        """Validate that all tool mappings reference existing backends and tools.

        Args:
            tool_mappings: List of tool mappings to validate

        Raises:
            VirtualServerValidationError: If any validation fails
        """
        errors = []

        for mapping in tool_mappings:
            # Check backend server exists
            server_path = mapping.backend_server_path
            server_info = await self._server_repo.get(server_path)

            if not server_info:
                errors.append(f"Backend server '{server_path}' does not exist")
                continue

            # Check tool exists in backend
            tool_list = server_info.get("tool_list", [])
            tool_names = [t.get("name", "") for t in tool_list]

            if mapping.tool_name not in tool_names:
                errors.append(
                    f"Tool '{mapping.tool_name}' not found in backend "
                    f"server '{server_path}'. Available tools: "
                    f"{', '.join(tool_names[:10])}"
                )

            # Check version exists if pinned
            if mapping.backend_version:
                version_id = f"{server_path}:{mapping.backend_version}"
                version_info = await self._server_repo.get(version_id)
                if not version_info:
                    errors.append(
                        f"Version '{mapping.backend_version}' not found "
                        f"for backend server '{server_path}'"
                    )

        if errors:
            raise VirtualServerValidationError(
                "Tool mapping validation failed:\n" + "\n".join(f"- {e}" for e in errors)
            )

    def _validate_unique_tool_names(
        self,
        tool_mappings: list[ToolMapping],
    ) -> None:
        """Validate that effective tool names are unique within a virtual server.

        Args:
            tool_mappings: List of tool mappings

        Raises:
            VirtualServerValidationError: If duplicate names found
        """
        seen_names: dict[str, str] = {}
        duplicates = []

        for mapping in tool_mappings:
            effective_name = _get_effective_tool_name(mapping)
            if effective_name in seen_names:
                duplicates.append(
                    f"'{effective_name}' (from {mapping.backend_server_path} "
                    f"and {seen_names[effective_name]})"
                )
            else:
                seen_names[effective_name] = mapping.backend_server_path

        if duplicates:
            raise VirtualServerValidationError(
                "Duplicate tool names in virtual server: "
                + ", ".join(duplicates)
                + ". Use aliases to resolve conflicts."
            )

    async def _resolve_tool_list(
        self,
        config: VirtualServerConfig,
    ) -> list[ResolvedTool]:
        """Resolve tool mappings to full tool metadata.

        Args:
            config: Virtual server configuration

        Returns:
            List of ResolvedTool with full metadata from backends
        """
        resolved = []

        # Build scope override lookup
        scope_overrides: dict[str, list[str]] = {}
        for override in config.tool_scope_overrides:
            scope_overrides[override.tool_alias] = override.required_scopes

        for mapping in config.tool_mappings:
            effective_name = _get_effective_tool_name(mapping)

            # Get tool metadata from backend
            server_path = mapping.backend_server_path

            # If version is pinned, look up version-specific server doc
            if mapping.backend_version:
                version_id = f"{server_path}:{mapping.backend_version}"
                server_info = await self._server_repo.get(version_id)
            else:
                server_info = await self._server_repo.get(server_path)

            if not server_info:
                logger.warning(
                    f"Backend server '{server_path}' not found, skipping tool '{mapping.tool_name}'"
                )
                continue

            # Find tool in backend's tool list
            tool_list = server_info.get("tool_list", [])
            tool_meta = None
            for tool in tool_list:
                if tool.get("name") == mapping.tool_name:
                    tool_meta = tool
                    break

            if not tool_meta:
                logger.warning(
                    f"Tool '{mapping.tool_name}' not found in backend '{server_path}', skipping"
                )
                continue

            # Build resolved tool
            description = mapping.description_override or tool_meta.get("description", "")
            input_schema = tool_meta.get("inputSchema", {})
            tool_scopes = scope_overrides.get(effective_name, [])

            resolved.append(
                ResolvedTool(
                    name=effective_name,
                    original_name=mapping.tool_name,
                    backend_server_path=server_path,
                    backend_version=mapping.backend_version,
                    description=description,
                    input_schema=input_schema,
                    required_scopes=tool_scopes,
                )
            )

        return resolved

    def _config_to_info(
        self,
        config: VirtualServerConfig,
    ) -> VirtualServerInfo:
        """Convert a VirtualServerConfig to a lightweight VirtualServerInfo.

        Args:
            config: Full virtual server configuration

        Returns:
            VirtualServerInfo summary
        """
        backend_paths = _get_unique_backends(config.tool_mappings)
        return VirtualServerInfo(
            path=config.path,
            server_name=config.server_name,
            description=config.description,
            tool_count=len(config.tool_mappings),
            backend_count=len(backend_paths),
            backend_paths=backend_paths,
            is_enabled=config.is_enabled,
            tags=config.tags,
            num_stars=config.num_stars,
            rating_details=config.rating_details,
            created_by=config.created_by,
            created_at=config.created_at,
            updated_at=config.updated_at,
        )

    async def _index_for_search(
        self,
        config: VirtualServerConfig,
    ) -> None:
        """Index or update a virtual server in the search index.

        Args:
            config: Virtual server configuration to index
        """
        try:
            search_repo = get_search_repository()
            await search_repo.index_virtual_server(
                path=config.path,
                virtual_server=config,
                is_enabled=config.is_enabled,
            )
        except Exception as e:
            logger.warning(f"Failed to index virtual server '{config.path}' for search: {e}")

    async def _remove_from_search(
        self,
        path: str,
    ) -> None:
        """Remove a virtual server from the search index.

        Args:
            path: Virtual server path to remove
        """
        try:
            search_repo = get_search_repository()
            await search_repo.remove_entity(path)
        except Exception as e:
            logger.warning(f"Failed to remove virtual server '{path}' from search: {e}")

    async def _trigger_nginx_reload(self) -> bool:
        """Trigger nginx configuration regeneration.

        Serializes concurrent nginx reloads using an asyncio.Lock to
        prevent race conditions when multiple mutations happen at once.

        This regenerates the full nginx config including virtual server
        location blocks and mapping files, then reloads nginx.

        Returns:
            True if nginx was successfully reloaded, False otherwise.
            The CRUD operation itself has already succeeded at this point,
            so callers should treat False as a non-fatal warning.
        """
        async with _nginx_reload_lock:
            try:
                from ..core.nginx_service import nginx_service
                from ..services.server_service import server_service

                # Get currently enabled servers for the full config generation
                enabled_paths = await server_service.get_enabled_services()
                enabled_servers = {}
                for path in enabled_paths:
                    server_info = await server_service.get_server_info(path)
                    if server_info:
                        enabled_servers[path] = server_info

                await nginx_service.generate_config_async(enabled_servers)
                logger.info("Nginx configuration regenerated for virtual server change")
                return True
            except Exception as e:
                logger.error(
                    f"Failed to regenerate nginx config after virtual server change: {e}",
                    exc_info=True,
                )
                return False


def get_virtual_server_service() -> VirtualServerService:
    """Get virtual server service singleton."""
    global _virtual_server_service

    if _virtual_server_service is not None:
        return _virtual_server_service

    _virtual_server_service = VirtualServerService()
    return _virtual_server_service
