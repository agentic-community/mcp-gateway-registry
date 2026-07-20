import asyncio
import logging
from typing import Any

from ..core.metrics import ASSET_ID_CONFLICT_TOTAL
from ..exceptions import AssetIdConflictError
from ..repositories.factory import get_server_repository
from ..repositories.interfaces import ServerRepositoryBase
from ..utils.credential_encryption import (
    _migrate_auth_type_to_auth_scheme,
    strip_credentials_from_dict,
)
from ..utils.url_guard import validate_proxy_pass_url, validate_server_path

logger = logging.getLogger(__name__)


def _validate_endpoint_fields(
    server_info: dict[str, Any],
) -> None:
    """Validate optional override endpoint fields at every write path.

    ``mcp_endpoint`` and ``sse_endpoint`` are user-supplied alternate targets
    that are later fetched (health checks, tool discovery, the scanner
    subprocess) and interpolated into generated nginx config, exactly like
    ``proxy_pass_url``. They therefore MUST pass the same canonical guard as
    ``proxy_pass_url`` so an attacker cannot use them to smuggle a
    private/metadata/loopback target or an nginx metacharacter past the check
    on the primary URL field. Empty/unset is allowed (they are optional); a
    present-but-invalid value is rejected with the same error shape as an
    invalid ``proxy_pass_url`` (fail closed).

    Raises:
        UrlValidationError: If a present endpoint field fails scheme, host,
            metacharacter, or private/metadata-IP validation.
    """
    for field_name in ("mcp_endpoint", "sse_endpoint"):
        value = server_info.get(field_name)
        if value:
            validate_proxy_pass_url(value)


class ServerService:
    """Service for managing server registration and state."""

    def __init__(self):
        self._repo: ServerRepositoryBase = get_server_repository()
        from ..repositories.factory import get_search_repository

        self._search_repo = get_search_repository()

    def _prepare_server_dict(
        self,
        server_dict: dict[str, Any],
        include_credentials: bool = False,
    ) -> dict[str, Any]:
        """Apply read-time migration and optionally strip credentials.

        Args:
            server_dict: Raw server dict from storage.
            include_credentials: If True, keep encrypted credentials in the dict.

        Returns:
            Prepared server dict with auth_scheme migrated and credentials
            optionally stripped.
        """
        _migrate_auth_type_to_auth_scheme(server_dict)
        if not include_credentials:
            strip_credentials_from_dict(server_dict)
        return server_dict

    async def load_servers_and_state(self):
        """Load server definitions and persisted state from repository."""
        # Delegate to repository - no longer maintains service-level cache
        await self._repo.load_all()

    async def register_server(
        self,
        server_info: dict[str, Any],
        is_version_registration: bool = False,
    ) -> dict[str, Any]:
        """
        Register a new server or a new version of an existing server.

        If a server with the same path exists:
        - If new server has a different version, register as inactive version
        - If same version or no version specified, return 409 conflict

        Args:
            server_info: Server configuration dict
            is_version_registration: Internal flag, True when called recursively for version

        Returns:
            Dict with 'success', 'message', and optionally 'is_new_version' keys
        """
        path = server_info.get("path")
        new_version = server_info.get("version")

        # Fail-closed URL validation on every registration path (dashboard/API,
        # internal register, CLI, federation import) before anything is
        # persisted. Rejects non-http(s) schemes, nginx metacharacters, and
        # private/metadata targets (unless the operator allowlisted the internal
        # target). Raises UrlValidationError -> HTTP 400.
        proxy_pass_url = server_info.get("proxy_pass_url")
        if proxy_pass_url:
            validate_proxy_pass_url(proxy_pass_url)

        # Alternate/override endpoint fields are fetched and interpolated into
        # config just like proxy_pass_url, so they go through the SAME guard.
        _validate_endpoint_fields(server_info)

        # The path is interpolated into nginx location directives; reject any
        # nginx metacharacters so a crafted path cannot inject config.
        if path:
            validate_server_path(path)

        # Check if server with this path already exists
        existing_server = await self._repo.get(path)

        if existing_server:
            existing_version = existing_server.get("version", "v1.0.0")

            # If new version is specified and different, register as new version
            if new_version and new_version != existing_version:
                logger.info(
                    f"Server {path} exists with version {existing_version}, "
                    f"registering {new_version} as new version"
                )

                # Use add_server_version to create the inactive version document
                try:
                    await self.add_server_version(
                        path=path,
                        version=new_version,
                        proxy_pass_url=server_info.get("proxy_pass_url"),
                        status=server_info.get("status", "stable"),
                        is_default=False,
                    )
                    return {
                        "success": True,
                        "message": f"Registered version {new_version} for existing server {path}",
                        "is_new_version": True,
                        "existing_version": existing_version,
                    }
                except ValueError as e:
                    return {
                        "success": False,
                        "message": str(e),
                        "is_new_version": False,
                    }

            # Same version or no version - conflict
            return {
                "success": False,
                "message": f"Server already exists at path {path} with version {existing_version}",
                "is_new_version": False,
            }

        # Id uniqueness pre-check (#1276): a caller-supplied id must not
        # collide with an existing asset. Return a failure dict (mirrors the
        # path-conflict contract above) so the routes surface a 409.
        asset_id = server_info.get("id")
        if asset_id and await self._repo.find_by_id(asset_id):
            logger.warning(f"Server registration rejected: id '{asset_id}' already exists")
            ASSET_ID_CONFLICT_TOTAL.labels(asset_type="server").inc()
            return {
                "success": False,
                "message": f"Server with id '{asset_id}' already exists",
                "is_new_version": False,
                "error_type": "id_conflict",
            }

        # New server - create it
        # Initialize version metadata for new servers
        if not server_info.get("version"):
            server_info["version"] = "v1.0.0"
        server_info["is_active"] = True

        try:
            result = await self._repo.create(server_info)
        except AssetIdConflictError as e:
            # Lost the insert race after the pre-check (#1276).
            logger.warning(f"Server registration id conflict (race): {e}")
            ASSET_ID_CONFLICT_TOTAL.labels(asset_type="server").inc()
            return {
                "success": False,
                "message": f"Server with id '{e.asset_id}' already exists",
                "is_new_version": False,
                "error_type": "id_conflict",
            }

        if result:
            # Index in search backend
            try:
                is_enabled = await self._repo.get_state(path)
                await self._search_repo.index_server(path, server_info, is_enabled)
            except Exception as e:
                logger.error(f"Failed to index server {path}: {e}")
                # Don't fail the primary operation

            return {
                "success": True,
                "message": f"Server registered at {path}",
                "is_new_version": False,
            }

        return {
            "success": False,
            "message": f"Failed to register server at {path}",
            "is_new_version": False,
        }

    async def update_server(self, path: str, server_info: dict[str, Any]) -> bool:
        """Update an existing server.

        Raises:
            UrlValidationError: If the update sets a proxy_pass_url that fails
                SSRF/scheme validation. Validation only runs when the update
                payload actually carries a proxy_pass_url, so health-status and
                tool-list updates are unaffected.
        """
        # Fail-closed URL validation on edit paths that change the backend URL.
        if server_info.get("proxy_pass_url"):
            validate_proxy_pass_url(server_info["proxy_pass_url"])

        # Alternate/override endpoint fields get the same guard on edit paths.
        _validate_endpoint_fields(server_info)

        result = await self._repo.update(path, server_info)

        if result:
            # Update search index
            try:
                is_enabled = await self._repo.get_state(path)
                await self._search_repo.index_server(path, server_info, is_enabled)
            except Exception as e:
                logger.error(f"Failed to update search index after server update: {e}")

            # Regenerate nginx config if enabled
            if await self._repo.get_state(path):
                from ..core.nginx_service import nginx_reload_scheduler

                nginx_reload_scheduler.mark_dirty()

        return result

    async def toggle_service(self, path: str, enabled: bool) -> bool:
        """Toggle service enabled/disabled state."""
        result = await self._repo.set_state(path, enabled)

        if result:
            from ..core.nginx_service import nginx_reload_scheduler

            nginx_reload_scheduler.mark_dirty()

        return result

    async def get_server_info(
        self,
        path: str,
        include_credentials: bool = False,
    ) -> dict[str, Any] | None:
        """Get server information by path - queries repository directly.

        Args:
            path: Server path (e.g., "/my-server").
            include_credentials: If True, include encrypted credentials in result.
                Set to True only for internal callers like health checks.

        Returns:
            Server info dict, or None if not found.
        """
        result = await self._repo.get(path)
        if result:
            self._prepare_server_dict(result, include_credentials)
        return result

    async def get_all_servers(
        self,
        include_inactive: bool = False,
        include_credentials: bool = False,
        exclude_tool_list: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """
        Get all registered servers.

        Args:
            include_inactive: If True, include inactive server versions (default False)
            include_credentials: If True, include encrypted credentials in result
            exclude_tool_list: If True, omit the heavy tool_list field (DB-side
                projection) for callers that only need metadata. num_tools is
                unaffected.

        Returns:
            Dict of all servers
        """
        # Query repository directly instead of using cache
        all_servers = await self._repo.list_all(exclude_tool_list=exclude_tool_list)

        # Apply read-time migration and credential stripping
        for server_info in all_servers.values():
            self._prepare_server_dict(server_info, include_credentials)

        # Filter out inactive servers (non-default versions) unless requested
        if not include_inactive:
            all_servers = {
                path: server_info
                for path, server_info in all_servers.items()
                if server_info.get("is_active", True)  # Default to True for backward compatibility
            }

        return all_servers

    async def get_servers_paginated(
        self,
        skip: int = 0,
        limit: int = 100,
        exclude_tool_list: bool = False,
    ) -> tuple[dict[str, dict[str, Any]], int]:
        """Get a page of servers with total count.

        Used for unrestricted users (admins) where DB-level pagination
        is correct because no servers are filtered out by access control.

        Note: list_paginated and count are separate DB calls, so total_count
        may be slightly inconsistent if servers are added/removed between calls.
        This is standard for offset-based pagination.

        Args:
            skip: Number of servers to skip.
            limit: Maximum number of servers to return.
            exclude_tool_list: If True, omit the heavy tool_list field for
                callers that only need metadata. num_tools is unaffected.

        Returns:
            Tuple of (servers dict for the requested page, total count of all servers).
        """
        servers = await self._repo.list_paginated(
            skip=skip, limit=limit, exclude_tool_list=exclude_tool_list
        )
        total = await self._repo.count()

        # Apply read-time migration and credential stripping
        for server_info in servers.values():
            self._prepare_server_dict(server_info, include_credentials=False)

        # Filter out inactive servers (non-default versions)
        servers = {
            path: server_info
            for path, server_info in servers.items()
            if server_info.get("is_active", True)
        }

        return servers, total

    async def count_servers(self) -> int:
        """Return the total number of server documents.

        Counts all documents (including inactive version docs), so it may be
        marginally higher than len(get_all_servers()). Intended for cheap
        totals like log/metric context, not exact active-server counts.

        Returns:
            Total document count from the repository.
        """
        return await self._repo.count()

    async def get_filtered_servers(
        self,
        accessible_servers: list[str],
        include_inactive: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """
        Get servers filtered by user's accessible servers list.

        Args:
            accessible_servers: List of server names the user can access
            include_inactive: If True, include inactive server versions (default False)

        Returns:
            Dict of servers the user is authorized to see
        """
        if not accessible_servers:
            logger.debug("User has no accessible servers, returning empty dict")
            return {}

        # Wildcard access — return all servers (non-admin with server: '*')
        if "*" in accessible_servers:
            logger.debug("Wildcard access detected in accessible_servers, returning all servers")
            return await self.get_all_servers(include_inactive=include_inactive)

        # Query repository directly instead of using cache
        all_servers = await self._repo.list_all()

        # Apply read-time migration and credential stripping
        for server_info in all_servers.values():
            self._prepare_server_dict(server_info, include_credentials=False)

        # Filter out inactive servers (non-default versions) unless requested
        if not include_inactive:
            all_servers = {
                path: server_info
                for path, server_info in all_servers.items()
                if server_info.get("is_active", True)  # Default to True for backward compatibility
            }

        logger.info(
            f"DEBUG: get_filtered_servers called with accessible_servers: {accessible_servers}"
        )
        logger.info(f"DEBUG: Available registered servers paths: {list(all_servers.keys())}")

        filtered_servers = {}
        for path, server_info in all_servers.items():
            server_name = server_info.get("server_name", "")
            # Extract technical name from path (remove leading and trailing slashes)
            technical_name = path.strip("/")
            logger.info(
                f"DEBUG: Checking server path='{path}', server_name='{server_name}', technical_name='{technical_name}' against accessible_servers"
            )

            # Check if user has access to this server using technical name
            if technical_name in accessible_servers:
                filtered_servers[path] = server_info
                logger.info(f"DEBUG: ✓ User has access to server: {technical_name} ({server_name})")
            else:
                logger.info(
                    f"DEBUG: ✗ User does not have access to server: {technical_name} ({server_name})"
                )

        logger.info(
            f"Filtered {len(filtered_servers)} servers from {len(all_servers)} total servers"
        )
        return filtered_servers

    async def get_all_servers_with_permissions(
        self,
        accessible_servers: list[str] | None = None,
        exclude_tool_list: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """
        Get servers with optional filtering based on user permissions.

        Args:
            accessible_servers: Optional list of server names the user can access.
                               If None, returns all servers (admin access).
            exclude_tool_list: If True, omit the heavy tool_list field for
                callers that only need metadata. num_tools is unaffected.

        Returns:
            Dict of servers the user is authorized to see
        """
        if accessible_servers is None:
            # Admin access - return all servers
            logger.debug("Admin access - returning all servers")
            return await self.get_all_servers(exclude_tool_list=exclude_tool_list)
        elif "*" in accessible_servers:
            # Wildcard access — return all servers (non-admin with server: '*')
            logger.debug("Wildcard access detected in accessible_servers, returning all servers")
            return await self.get_all_servers(exclude_tool_list=exclude_tool_list)
        else:
            # Filtered access - fetch only the accessible servers via a targeted
            # query instead of scanning the full collection.
            logger.debug(
                f"Filtered access - returning servers accessible to user: {accessible_servers}"
            )

            # Build candidate _id paths covering the slash variants the caller
            # may supply ("currenttime", "/currenttime", "/currenttime/"). The
            # $in query only matches existing docs, so extra candidates are
            # harmless and version docs ({path}:{version}) never match.
            candidate_paths: list[str] = []
            for accessible_server in accessible_servers:
                normalized = accessible_server.strip("/")
                if not normalized:
                    continue
                candidate_paths.extend(
                    [f"/{normalized}", f"/{normalized}/", normalized, f"{normalized}/"]
                )

            filtered_servers = await self._repo.list_by_ids(candidate_paths)

            # Apply read-time migration and credential stripping
            for server_info in filtered_servers.values():
                self._prepare_server_dict(server_info, include_credentials=False)
                if exclude_tool_list:
                    server_info.pop("tool_list", None)

            # Filter out inactive servers (non-default versions) for parity with
            # get_all_servers; defaults to True for backward compatibility.
            filtered_servers = {
                path: server_info
                for path, server_info in filtered_servers.items()
                if server_info.get("is_active", True)
            }

            logger.info(
                f"Filtered access: fetched {len(filtered_servers)} servers "
                f"for {len(accessible_servers)} accessible entries"
            )
            return filtered_servers

    async def user_can_access_server_path(self, path: str, accessible_servers: list[str]) -> bool:
        """
        Check if user can access a specific server by path.

        Args:
            path: Server path to check
            accessible_servers: List of server names the user can access

        Returns:
            True if user can access the server, False otherwise
        """
        server_info = await self.get_server_info(path)
        if not server_info:
            return False

        # Wildcard access — grant access to any existing server (non-admin with server: '*')
        if "*" in accessible_servers:
            return True

        # Extract technical name from path (remove leading and trailing slashes)
        technical_name = path.strip("/")

        # Check with normalized paths - support "currenttime", "/currenttime", "/currenttime/"
        for accessible_server in accessible_servers:
            normalized_accessible = accessible_server.strip("/")
            if technical_name == normalized_accessible:
                return True

        return False

    async def is_service_enabled(self, path: str) -> bool:
        """Check if a service is enabled."""
        return await self._repo.get_state(path)

    async def get_enabled_services(self) -> list[str]:
        """Get list of enabled service paths - queries repository directly.

        Only returns active versions for health checks. Inactive versions
        (those with is_active=False) are skipped since health checks should
        only run on the currently active version of each server.
        """
        all_servers = await self._repo.list_all()
        enabled_paths = []

        # Extract state from list_all() response instead of N+1 queries
        for path, server_info in all_servers.items():
            if not server_info.get("is_enabled", False):
                continue

            # Skip inactive versions - only health check active versions
            # Servers without version_group are single-version (implicitly active)
            # Servers with version_group but is_active=False are inactive versions
            if server_info.get("version_group") and not server_info.get("is_active", True):
                continue

            enabled_paths.append(path)

        return enabled_paths

    async def reload_state_from_disk(self):
        """Reload service state from repository."""
        logger.info("Reloading service state from repository...")

        previous_enabled_services = set(await self.get_enabled_services())

        # Reload from repository
        await self._repo.load_all()

        current_enabled_services = set(await self.get_enabled_services())

        if previous_enabled_services != current_enabled_services:
            logger.info(
                f"Service state changes detected: {len(previous_enabled_services)} -> {len(current_enabled_services)} enabled services"
            )

            from ..core.nginx_service import nginx_reload_scheduler

            nginx_reload_scheduler.mark_dirty()
        else:
            logger.info("No service state changes detected after reload")

    async def update_rating(
        self,
        path: str,
        username: str,
        rating: int,
    ) -> float:
        """
        Log a user rating for a server. If the user has already rated, update their rating.

        Args:
            path: server path
            username: The user who submitted rating
            rating: integer between 1-5

        Return:
            Updated average rating

        Raises:
            ValueError: If server not found or invalid rating
        """
        from . import rating_service

        # Query repository directly instead of using cache
        server_info = await self._repo.get(path)
        if not server_info:
            logger.error(f"Cannot update server at path '{path}': not found")
            raise ValueError(f"Server not found at path: {path}")

        # Validate rating using shared service
        rating_service.validate_rating(rating)

        # Ensure rating_details is a list
        if "rating_details" not in server_info or server_info["rating_details"] is None:
            server_info["rating_details"] = []

        # Update rating details using shared service
        updated_details, is_new_rating = rating_service.update_rating_details(
            server_info["rating_details"], username, rating
        )
        server_info["rating_details"] = updated_details

        # Calculate average rating using shared service
        server_info["num_stars"] = rating_service.calculate_average_rating(
            server_info["rating_details"]
        )

        # Save to repository
        await self._repo.update(path, server_info)

        logger.info(
            f"Updated rating for server {path}: user {username} rated {rating}, "
            f"new average: {server_info['num_stars']:.2f}"
        )
        return server_info["num_stars"]

    async def remove_server(self, path: str) -> bool:
        """Remove a server and all its version documents from the registry.

        Deletes the active document and any inactive version documents
        with IDs matching `{path}:{version}` (e.g., /context7:v2.0.0).

        Removes the search embedding before deleting source documents so a
        failed vector removal does not leave an orphan (issue #1145).

        Args:
            path: Server base path (e.g., "/context7")

        Returns:
            True if at least one document was deleted
        """
        from .search_index_cleanup import remove_from_search_index_with_retry

        if not await remove_from_search_index_with_retry(
            self._search_repo,
            path,
            entity_type="mcp_server",
        ):
            logger.error(
                "Aborting server delete for %s: search index removal failed",
                path,
            )
            return False

        deleted_count = await self._repo.delete_with_versions(path)
        return deleted_count > 0

    async def add_server_version(
        self,
        path: str,
        version: str,
        proxy_pass_url: str,
        status: str = "stable",
        is_default: bool = False,
    ) -> bool:
        """
        Add a new version to an existing server.

        Uses the separate-documents design where each version is a separate document:
        - Active version uses `_id: "{path}"`
        - Inactive versions use `_id: "{path}:{version}"`

        Args:
            path: Server path (e.g., "/context7")
            version: Version identifier (e.g., "v2.0.0")
            proxy_pass_url: Backend URL for this version
            status: Version status (stable, deprecated, beta)
            is_default: Set this as the default version

        Returns:
            True if version added successfully

        Raises:
            ValueError: If server not found or version already exists
            UrlValidationError: If proxy_pass_url fails SSRF/scheme validation
        """
        # Fail-closed URL validation before persisting a new version's backend.
        if proxy_pass_url:
            validate_proxy_pass_url(proxy_pass_url)

        # Get active server document
        active_server = await self._repo.get(path)
        if not active_server:
            raise ValueError(f"Server not found: {path}")

        # Derive version_group from path (e.g., "/context7" -> "context7")
        version_group = path.strip("/").replace("/", "-")

        # Initialize version metadata on active server if first multi-version setup
        if not active_server.get("version_group"):
            active_server["version"] = active_server.get("version", "v1.0.0")
            active_server["is_active"] = True
            active_server["version_group"] = version_group
            active_server["other_version_ids"] = []
            await self._repo.update(path, active_server)

        # Check if version already exists
        new_version_id = f"{path}:{version}"
        existing_inactive = await self._repo.get(new_version_id)
        if existing_inactive:
            raise ValueError(f"Version {version} already exists for server {path}")

        # Check if version matches active version
        if active_server.get("version") == version:
            raise ValueError(f"Version {version} already exists as active version")

        # Create new version document (inactive by default)
        new_version_doc = {
            "path": new_version_id,
            "server_name": active_server.get("server_name"),
            "version": version,
            "proxy_pass_url": proxy_pass_url,
            "status": status,
            "is_active": False,
            "version_group": version_group,
            "active_version_id": path,
            "description": active_server.get("description", ""),
            "tags": active_server.get("tags", []),
            "supported_transports": active_server.get("supported_transports", []),
            "is_enabled": active_server.get("is_enabled", False),
        }

        # Create the new version document
        result = await self._repo.create(new_version_doc)

        if result:
            # Update active server's other_version_ids
            other_versions = active_server.get("other_version_ids", [])
            other_versions.append(new_version_id)
            active_server["other_version_ids"] = other_versions
            await self._repo.update(path, active_server)

            # If is_default, swap this to be the active version
            if is_default:
                await self.set_default_version(path, version)

            # Regenerate nginx config
            await self._regenerate_nginx_config()
            logger.info(f"Added version {version} to server {path}")

        return result

    async def remove_server_version(self, path: str, version: str) -> bool:
        """
        Remove a version from a server.

        Uses separate-documents design: deletes the inactive version document.

        Args:
            path: Server path
            version: Version to remove

        Returns:
            True if version removed successfully

        Raises:
            ValueError: If server not found, version not found, or trying to remove active
        """
        # Get active server document
        active_server = await self._repo.get(path)
        if not active_server:
            raise ValueError(f"Server not found: {path}")

        # Cannot remove active version
        if active_server.get("version") == version:
            raise ValueError(
                f"Cannot remove active version {version}. Set a new active version first."
            )

        # Check if this is a single-version server
        if not active_server.get("version_group"):
            raise ValueError(f"Server {path} has no versions to remove")

        # Find and remove the inactive version document
        version_id = f"{path}:{version}"
        inactive_version = await self._repo.get(version_id)
        if not inactive_version:
            raise ValueError(f"Version {version} not found for server {path}")

        # Delete the inactive version document
        result = await self._repo.delete(version_id)

        if result:
            # Update active server's other_version_ids
            other_versions = active_server.get("other_version_ids", [])
            if version_id in other_versions:
                other_versions.remove(version_id)
                active_server["other_version_ids"] = other_versions
                await self._repo.update(path, active_server)

            await self._regenerate_nginx_config()
            logger.info(f"Removed version {version} from server {path}")

        return result

    async def set_default_version(self, path: str, version: str) -> bool:
        """
        Set the default (active) version for a server.

        Uses separate-documents design: swaps documents by:
        1. Current active becomes inactive with _id: "{path}:{current_version}"
        2. Target inactive becomes active with _id: "{path}"

        Args:
            path: Server path
            version: Version to set as active

        Returns:
            True if swap successful

        Raises:
            ValueError: If server or version not found
        """
        # Get current active document - try with and without trailing slash
        current_active = await self._repo.get(path)
        if not current_active and not path.endswith("/"):
            path_with_slash = path + "/"
            current_active = await self._repo.get(path_with_slash)
            if current_active:
                path = path_with_slash
                logger.debug(f"Normalized path to {path} (added trailing slash)")
        if not current_active:
            raise ValueError(f"Server not found: {path}")

        current_version = current_active.get("version", "v1.0.0")

        # If already active, nothing to do
        if current_version == version:
            logger.info(f"Version {version} is already the active version")
            return True

        # Check if this is a single-version server
        if not current_active.get("version_group"):
            raise ValueError(f"Server {path} has no other versions configured")

        # Find target inactive version
        target_version_id = f"{path}:{version}"
        target_inactive = await self._repo.get(target_version_id)
        if not target_inactive:
            # List available versions
            other_version_ids = current_active.get("other_version_ids", [])
            available = [vid.split(":")[-1] for vid in other_version_ids]
            raise ValueError(f"Version {version} not found. Available: {available}")

        # Prepare new active doc (target becomes active with original path)
        new_active = {**target_inactive}
        new_active["path"] = path
        new_active["is_active"] = True
        new_active.pop("active_version_id", None)
        # Update other_version_ids: remove target, add current
        other_versions = list(current_active.get("other_version_ids", []))
        if target_version_id in other_versions:
            other_versions.remove(target_version_id)
        new_inactive_id = f"{path}:{current_version}"
        other_versions.append(new_inactive_id)
        new_active["other_version_ids"] = other_versions

        # Prepare new inactive doc (current becomes inactive with compound id)
        new_inactive = {**current_active}
        new_inactive["path"] = new_inactive_id
        new_inactive["is_active"] = False
        new_inactive["active_version_id"] = path
        new_inactive.pop("other_version_ids", None)

        # Execute swap: delete old docs, insert new docs
        await self._repo.delete(path)
        await self._repo.delete(target_version_id)
        await self._repo.create(new_active)
        await self._repo.create(new_inactive)

        # Update search index: re-index with new active version
        try:
            is_enabled = new_active.get("is_enabled", False)
            await self._search_repo.index_server(path, new_active, is_enabled)
            logger.info(f"Updated search index for {path} with version {version}")
        except Exception as e:
            logger.error(f"Failed to update search index after version swap: {e}")

        await self._regenerate_nginx_config()
        logger.info(f"Swapped active version from {current_version} to {version} for {path}")

        # Trigger an immediate health check for the newly active version
        try:
            from ..health.service import health_service

            asyncio.create_task(health_service.perform_immediate_health_check(path))
            logger.info(
                f"Triggered background health check for {path} after version swap to {version}"
            )
        except Exception as e:
            logger.error(f"Failed to trigger health check after version swap for {path}: {e}")

        return True

    async def get_server_versions(self, path: str) -> dict[str, Any]:
        """
        Get all versions for a server.

        Uses separate-documents design: queries by version_group.

        Args:
            path: Server path

        Returns:
            Dictionary with version information

        Raises:
            ValueError: If server not found
        """
        # Get active server
        active_server = await self._repo.get(path)
        if not active_server:
            raise ValueError(f"Server not found: {path}")

        # Single-version server (no version_group)
        if not active_server.get("version_group"):
            return {
                "path": path,
                "default_version": active_server.get("version", "v1.0.0"),
                "versions": [
                    {
                        "version": active_server.get("version", "v1.0.0"),
                        "proxy_pass_url": active_server.get("proxy_pass_url"),
                        "status": "stable",
                        "is_default": True,
                    }
                ],
            }

        # Build versions list from active + inactive documents
        versions = []

        # Add active version
        versions.append(
            {
                "version": active_server.get("version"),
                "proxy_pass_url": active_server.get("proxy_pass_url"),
                "status": active_server.get("status", "stable"),
                "is_default": True,
                "description": active_server.get("description"),
            }
        )

        # Add inactive versions
        other_version_ids = active_server.get("other_version_ids", [])
        for version_id in other_version_ids:
            inactive_doc = await self._repo.get(version_id)
            if inactive_doc:
                versions.append(
                    {
                        "version": inactive_doc.get("version"),
                        "proxy_pass_url": inactive_doc.get("proxy_pass_url"),
                        "status": inactive_doc.get("status", "stable"),
                        "is_default": False,
                        "description": inactive_doc.get("description"),
                    }
                )

        return {"path": path, "default_version": active_server.get("version"), "versions": versions}

    async def _regenerate_nginx_config(self) -> None:
        """Signal nginx config needs regeneration (debounced)."""
        from ..core.nginx_service import nginx_reload_scheduler

        nginx_reload_scheduler.mark_dirty()


# Global service instance
server_service = ServerService()
