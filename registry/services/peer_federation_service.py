"""
Service for managing peer registry federation configurations.

This module provides CRUD operations for peer registry connections,
using the repository pattern for storage abstraction. Supports both
MongoDB/DocumentDB and file-based storage backends.

Based on: registry/services/server_service.py and registry/services/agent_service.py
"""

import asyncio
import logging
import time
import uuid
from datetime import UTC, datetime
from threading import Lock as ThreadingLock
from typing import Any, Literal, Optional

from ..core.metrics import PEER_SYNC_DURATION_SECONDS, PEER_SYNC_FAILURES
from ..repositories.factory import (
    get_peer_federation_repository,
    get_search_repository,
    get_security_scan_repository,
)
from ..repositories.interfaces import PeerFederationRepositoryBase
from ..schemas.agent_models import AgentCard
from ..schemas.peer_federation_schema import (
    PeerRegistryConfig,
    PeerSyncStatus,
    SyncHistoryEntry,
    SyncResult,
)
from .agent_service import agent_service
from .federation.peer_registry_client import PeerRegistryClient
from .server_service import server_service

logger = logging.getLogger(__name__)


class PeerFederationService:
    """Service for managing peer registry federation configurations.

    Uses repository pattern for data access, supporting multiple storage backends.
    """

    _instance: Optional["PeerFederationService"] = None
    _lock: ThreadingLock = ThreadingLock()

    def __new__(cls) -> "PeerFederationService":
        """Singleton pattern with thread-safe double-checked locking."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize peer federation service with repository."""
        # Singleton: only initialize once
        if self._initialized:
            return

        self._repo: PeerFederationRepositoryBase | None = None
        self._operation_lock = asyncio.Lock()  # Async-safe lock for operations

        # In-memory caches for quick access (populated from repository)
        self.registered_peers: dict[str, PeerRegistryConfig] = {}
        self.peer_sync_status: dict[str, PeerSyncStatus] = {}

        self._initialized = True

    def _get_repo(self) -> PeerFederationRepositoryBase:
        """Get or create repository instance."""
        if self._repo is None:
            self._repo = get_peer_federation_repository()
        return self._repo

    async def load_peers_and_state(self) -> None:
        """Load peer configs and sync state from repository."""
        logger.info("Loading peer federation data from repository...")

        repo = self._get_repo()
        await repo.load_all()

        # Load peers into cache
        peers = await repo.list_peers()
        self.registered_peers = {peer.peer_id: peer for peer in peers}

        # Load sync statuses into cache
        statuses = await repo.list_sync_statuses()
        self.peer_sync_status = {status.peer_id: status for status in statuses}

        # Initialize sync status for any peers without one
        for peer_id in self.registered_peers.keys():
            if peer_id not in self.peer_sync_status:
                status = PeerSyncStatus(peer_id=peer_id)
                self.peer_sync_status[peer_id] = status
                await repo.update_sync_status(peer_id, status)

        logger.info(
            f"Loaded {len(self.registered_peers)} peers, {len(self.peer_sync_status)} sync statuses"
        )

    # Synchronous wrapper for backward compatibility
    def load_peers_and_state_sync(self) -> None:
        """Synchronous wrapper for load_peers_and_state.

        DEPRECATED: Use async version load_peers_and_state() instead.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create a new task in the running loop
            asyncio.create_task(self.load_peers_and_state())
        else:
            loop.run_until_complete(self.load_peers_and_state())

    async def add_peer(
        self,
        config: PeerRegistryConfig,
    ) -> PeerRegistryConfig:
        """
        Add a new peer registry configuration.

        Args:
            config: Peer registry config to add

        Returns:
            Added peer config

        Raises:
            ValueError: If peer_id already exists or is invalid
        """
        async with self._operation_lock:
            repo = self._get_repo()

            # Create peer via repository (handles validation and timestamps)
            created_peer = await repo.create_peer(config)

            # Update cache
            self.registered_peers[created_peer.peer_id] = created_peer

            # Get/create sync status
            sync_status = await repo.get_sync_status(created_peer.peer_id)
            if sync_status:
                self.peer_sync_status[created_peer.peer_id] = sync_status

            logger.info(
                f"New peer registered: '{created_peer.name}' with peer_id "
                f"'{created_peer.peer_id}' (enabled={created_peer.enabled})"
            )

            return created_peer

    async def get_peer(
        self,
        peer_id: str,
    ) -> PeerRegistryConfig:
        """
        Get peer config by peer_id.

        Args:
            peer_id: Peer identifier

        Returns:
            Peer config

        Raises:
            ValueError: If peer not found
        """
        # Check cache first
        if peer_id in self.registered_peers:
            return self.registered_peers[peer_id]

        # Try repository
        repo = self._get_repo()
        peer_config = await repo.get_peer(peer_id)

        if not peer_config:
            raise ValueError(f"Peer not found: {peer_id}")

        # Update cache
        self.registered_peers[peer_id] = peer_config
        return peer_config

    async def update_peer(
        self,
        peer_id: str,
        updates: dict[str, Any],
    ) -> PeerRegistryConfig:
        """
        Update an existing peer config.

        Args:
            peer_id: Peer identifier
            updates: Dictionary of fields to update

        Returns:
            Updated peer config

        Raises:
            ValueError: If peer not found or invalid
        """
        async with self._operation_lock:
            repo = self._get_repo()

            # Check if token is being updated for audit logging
            is_token_update = "federation_token" in updates
            had_token_before = False

            if is_token_update:
                # Get existing peer to check if it had a token
                try:
                    existing_peer = await repo.get_peer(peer_id)
                    had_token_before = existing_peer and existing_peer.federation_token is not None
                except Exception:
                    pass  # Continue with update even if we can't check existing state

            # Update via repository (handles validation)
            updated_peer = await repo.update_peer(peer_id, updates)

            # Update cache
            self.registered_peers[peer_id] = updated_peer

            # Audit logging for token updates
            if is_token_update:
                logger.info(
                    f"AUDIT: Federation token updated for peer '{peer_id}' "
                    f"(name='{updated_peer.name}'). "
                    f"Previous token existed: {had_token_before}, "
                    f"New token provided: {updated_peer.federation_token is not None}"
                )

            logger.info(f"Peer '{updated_peer.name}' ({peer_id}) updated")
            return updated_peer

    async def remove_peer(
        self,
        peer_id: str,
    ) -> bool:
        """
        Remove a peer from registry.

        Also cleans up all servers and agents synced from this peer
        (paths starting with /{peer_id}/).

        Args:
            peer_id: Peer identifier

        Returns:
            True if deleted successfully

        Raises:
            ValueError: If peer not found
        """
        async with self._operation_lock:
            # Get peer name for logging before deletion
            peer_name = self.registered_peers.get(
                peer_id,
                PeerRegistryConfig(peer_id=peer_id, name="unknown", endpoint="http://unknown"),
            ).name

            # Clean up synced servers from this peer
            servers_deleted = await self._cleanup_synced_servers(peer_id)
            logger.info(f"Deleted {servers_deleted} synced servers from peer '{peer_id}'")

            # Clean up synced agents from this peer
            agents_deleted = await self._cleanup_synced_agents(peer_id)
            logger.info(f"Deleted {agents_deleted} synced agents from peer '{peer_id}'")

            repo = self._get_repo()

            # Delete via repository (handles cascade delete of sync status)
            result = await repo.delete_peer(peer_id)

            if result:
                # Remove from caches
                if peer_id in self.registered_peers:
                    del self.registered_peers[peer_id]
                if peer_id in self.peer_sync_status:
                    del self.peer_sync_status[peer_id]

                logger.info(
                    f"Successfully removed peer '{peer_name}' with peer_id '{peer_id}' "
                    f"(cleaned up {servers_deleted} servers, {agents_deleted} agents)"
                )

            return result

    async def _cleanup_synced_servers(
        self,
        peer_id: str,
    ) -> int:
        """
        Delete all servers synced from a specific peer.

        Args:
            peer_id: Peer identifier

        Returns:
            Number of servers deleted
        """
        deleted_count = 0
        path_prefix = f"/{peer_id}/"

        try:
            # Get all servers from the repository
            all_servers = await server_service.get_all_servers()

            # Find servers with paths starting with the peer prefix
            for path in list(all_servers.keys()):
                if path.startswith(path_prefix):
                    try:
                        success = await server_service.remove_server(path)
                        if success:
                            deleted_count += 1
                            logger.debug(f"Deleted synced server: {path}")
                        else:
                            logger.warning(f"Failed to delete synced server: {path}")
                    except Exception as e:
                        logger.error(f"Error deleting synced server {path}: {e}")

        except Exception as e:
            logger.error(f"Error cleaning up synced servers for peer '{peer_id}': {e}")

        return deleted_count

    async def _cleanup_synced_agents(
        self,
        peer_id: str,
    ) -> int:
        """
        Delete all agents synced from a specific peer.

        Args:
            peer_id: Peer identifier

        Returns:
            Number of agents deleted
        """
        deleted_count = 0
        path_prefix = f"/{peer_id}/"

        try:
            # Get all agents from the repository
            all_agents = await agent_service.get_all_agents()

            # Find agents with paths starting with the peer prefix
            for agent in all_agents:
                if agent.path.startswith(path_prefix):
                    try:
                        success = await agent_service.delete_agent(agent.path)
                        if success:
                            deleted_count += 1
                            logger.debug(f"Deleted synced agent: {agent.path}")
                        else:
                            logger.warning(f"Failed to delete synced agent: {agent.path}")
                    except Exception as e:
                        logger.error(f"Error deleting synced agent {agent.path}: {e}")

        except Exception as e:
            logger.error(f"Error cleaning up synced agents for peer '{peer_id}': {e}")

        return deleted_count

    async def list_peers(
        self,
        enabled: bool | None = None,
    ) -> list[PeerRegistryConfig]:
        """
        List all configured peers with optional filtering.

        Args:
            enabled: If True, return only enabled peers.
                    If False, return only disabled peers.
                    If None, return all peers.

        Returns:
            List of peer configs
        """
        peers = list(self.registered_peers.values())

        if enabled is None:
            return peers

        return [peer for peer in peers if peer.enabled == enabled]

    async def get_peer_by_client_id(
        self,
        client_id: str,
    ) -> PeerRegistryConfig | None:
        """
        Find peer config by Azure AD/Keycloak client_id (from azp claim).

        This enables peer identification during federation requests by matching
        the client_id from the OAuth2 token to a registered peer's expected_client_id.

        Args:
            client_id: The client_id from the token's azp claim

        Returns:
            PeerRegistryConfig if found, None otherwise
        """
        if not client_id:
            return None

        peers = await self.list_peers()
        for peer in peers:
            if peer.expected_client_id == client_id:
                logger.debug(f"Found peer '{peer.peer_id}' for client_id '{client_id}'")
                return peer

        logger.debug(f"No peer found for client_id '{client_id}'")
        return None

    async def get_sync_status(
        self,
        peer_id: str,
    ) -> PeerSyncStatus | None:
        """
        Get sync status for a peer.

        Args:
            peer_id: Peer identifier

        Returns:
            PeerSyncStatus or None if not found
        """
        # Check cache first
        if peer_id in self.peer_sync_status:
            return self.peer_sync_status[peer_id]

        # Try repository
        repo = self._get_repo()
        status = await repo.get_sync_status(peer_id)

        if status:
            self.peer_sync_status[peer_id] = status

        return status

    async def update_sync_status(
        self,
        peer_id: str,
        sync_status: PeerSyncStatus,
    ) -> None:
        """
        Update sync status for a peer.

        Args:
            peer_id: Peer identifier
            sync_status: Updated sync status
        """
        repo = self._get_repo()
        await repo.update_sync_status(peer_id, sync_status)

        # Update cache
        self.peer_sync_status[peer_id] = sync_status

        logger.debug(f"Updated sync status for peer '{peer_id}'")

    async def sync_peer(
        self,
        peer_id: str,
    ) -> SyncResult:
        """
        Sync servers and agents from a single peer.

        Args:
            peer_id: Peer identifier

        Returns:
            SyncResult with sync statistics

        Raises:
            ValueError: If peer not found or disabled
        """
        # Start timing
        start_time = time.time()

        # Generate sync ID
        sync_id = f"sync-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

        # Get peer config
        peer_config = await self.get_peer(peer_id)

        # Check if peer is enabled
        if not peer_config.enabled:
            error_msg = f"Peer '{peer_id}' is disabled. Enable it before syncing."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Get current sync status for incremental sync
        sync_status = await self.get_sync_status(peer_id)
        if not sync_status:
            # Initialize if not exists
            sync_status = PeerSyncStatus(peer_id=peer_id)

        since_generation = sync_status.current_generation

        logger.info(
            f"Starting sync from peer '{peer_id}' ({peer_config.name}) "
            f"with generation {since_generation}"
        )

        # Mark sync as in progress
        sync_status.sync_in_progress = True
        sync_status.last_sync_attempt = datetime.now(UTC)
        await self.update_sync_status(peer_id, sync_status)

        try:
            # Create PeerRegistryClient for this peer
            client = PeerRegistryClient(
                peer_config=peer_config, timeout_seconds=30, retry_attempts=3
            )

            # Fetch servers using client
            servers = client.fetch_servers(since_generation=since_generation)

            # Fetch agents using client
            agents = client.fetch_agents(since_generation=since_generation)

            # Fetch security scans using client
            security_scans = client.fetch_security_scans()

            # Check for fetch failures (None indicates error, not empty result)
            # Fixes issue #561: None was silently converted to [] making auth
            # failures appear as successful syncs with 0 items.
            fetch_errors = []
            if servers is None:
                fetch_errors.append("servers")
                servers = []
            if agents is None:
                fetch_errors.append("agents")
                agents = []
            if security_scans is None:
                fetch_errors.append("security_scans")
                security_scans = []

            # If any fetch failed, raise error to mark sync as failed
            if fetch_errors:
                error_types = ", ".join(fetch_errors)
                raise ValueError(
                    f"Failed to fetch {error_types} from peer '{peer_config.peer_id}'. "
                    f"This typically indicates authentication or network errors. "
                    f"Check peer configuration and logs for details."
                )

            logger.info(
                f"Fetched {len(servers)} servers, {len(agents)} agents, and "
                f"{len(security_scans)} security scans from peer '{peer_id}'"
            )

            # Apply filters based on peer config
            servers = self._filter_servers_by_config(servers, peer_config)
            agents = self._filter_agents_by_config(agents, peer_config)

            logger.info(
                f"After filtering: {len(servers)} servers and {len(agents)} agents "
                f"from peer '{peer_id}'"
            )

            # Store fetched items
            servers_stored = await self._store_synced_servers(peer_id, servers)
            agents_stored = await self._store_synced_agents(peer_id, agents)
            scans_stored = await self._store_synced_security_scans(peer_id, security_scans)

            # Extract paths from fetched items for orphan detection
            fetched_server_paths = [s.get("path", "") for s in servers]
            fetched_agent_paths = [a.get("path", "") for a in agents]

            # Detect orphaned items
            orphaned_servers, orphaned_agents = await self.detect_orphaned_items(
                peer_id, fetched_server_paths, fetched_agent_paths
            )

            # Handle orphaned items (mark by default)
            if orphaned_servers or orphaned_agents:
                await self.handle_orphaned_items(
                    peer_id, orphaned_servers, orphaned_agents, action="mark"
                )

            # Calculate duration
            duration_seconds = time.time() - start_time

            # Update sync status with success
            sync_status.sync_in_progress = False
            sync_status.last_successful_sync = datetime.now(UTC)

            # Only increment generation if items were actually synced
            if servers_stored > 0 or agents_stored > 0 or since_generation == 0:
                sync_status.current_generation += 1

            sync_status.total_servers_synced = servers_stored
            sync_status.total_agents_synced = agents_stored
            sync_status.consecutive_failures = 0
            sync_status.is_healthy = True
            sync_status.last_health_check = datetime.now(UTC)

            # Create history entry
            history_entry = SyncHistoryEntry(
                sync_id=sync_id,
                started_at=sync_status.last_sync_attempt,
                completed_at=datetime.now(UTC),
                success=True,
                servers_synced=servers_stored,
                agents_synced=agents_stored,
                servers_orphaned=len(orphaned_servers),
                agents_orphaned=len(orphaned_agents),
                sync_generation=sync_status.current_generation,
                full_sync=(since_generation == 0),
            )
            sync_status.add_history_entry(history_entry)

            # Persist updated status
            await self.update_sync_status(peer_id, sync_status)

            logger.info(
                f"Successfully synced peer '{peer_id}': "
                f"{servers_stored} servers, {agents_stored} agents, {scans_stored} security scans, "
                f"{len(orphaned_servers)} orphaned servers, {len(orphaned_agents)} orphaned agents "
                f"in {duration_seconds:.2f} seconds"
            )

            # Record success metrics
            PEER_SYNC_DURATION_SECONDS.labels(peer_id=peer_id, success="true").set(duration_seconds)

            return SyncResult(
                success=True,
                peer_id=peer_id,
                servers_synced=servers_stored,
                agents_synced=agents_stored,
                servers_orphaned=len(orphaned_servers),
                agents_orphaned=len(orphaned_agents),
                duration_seconds=duration_seconds,
                new_generation=sync_status.current_generation,
            )

        except Exception as e:
            # Calculate duration even on failure
            duration_seconds = time.time() - start_time

            # Update sync status with failure
            sync_status.sync_in_progress = False
            sync_status.consecutive_failures += 1
            sync_status.is_healthy = False
            sync_status.last_health_check = datetime.now(UTC)

            error_msg = str(e)

            # Create history entry for failure
            history_entry = SyncHistoryEntry(
                sync_id=sync_id,
                started_at=sync_status.last_sync_attempt,
                completed_at=datetime.now(UTC),
                success=False,
                servers_synced=0,
                agents_synced=0,
                servers_orphaned=0,
                agents_orphaned=0,
                error_message=error_msg,
                sync_generation=sync_status.current_generation,
                full_sync=(since_generation == 0),
            )
            sync_status.add_history_entry(history_entry)

            # Persist updated status
            await self.update_sync_status(peer_id, sync_status)

            # Record failure metrics
            # Determine failure type from error message
            failure_type = "unknown"
            if "authentication" in error_msg.lower() or "token" in error_msg.lower():
                failure_type = "auth_error"
            elif "network" in error_msg.lower() or "timeout" in error_msg.lower():
                failure_type = "network_error"
            elif "failed to fetch" in error_msg.lower():
                failure_type = "fetch_error"

            PEER_SYNC_FAILURES.labels(peer_id=peer_id, failure_type=failure_type).inc()
            PEER_SYNC_DURATION_SECONDS.labels(peer_id=peer_id, success="false").set(
                duration_seconds
            )

            logger.error(f"Failed to sync peer '{peer_id}': {error_msg}", exc_info=True)

            return SyncResult(
                success=False,
                peer_id=peer_id,
                servers_synced=0,
                agents_synced=0,
                servers_orphaned=0,
                agents_orphaned=0,
                error_message=error_msg,
                duration_seconds=duration_seconds,
                new_generation=sync_status.current_generation,
            )

    async def sync_all_peers(
        self,
        enabled_only: bool = True,
    ) -> dict[str, SyncResult]:
        """
        Sync all (or enabled) peers.

        Args:
            enabled_only: If True, only sync enabled peers

        Returns:
            Dictionary mapping peer_id to SyncResult
        """
        peers = await self.list_peers(enabled=enabled_only if enabled_only else None)

        logger.info(
            f"Starting sync for {len(peers)} peers ({'enabled only' if enabled_only else 'all'})"
        )

        results = {}

        for peer in peers:
            peer_id = peer.peer_id

            try:
                logger.info(f"Syncing peer '{peer_id}' ({peer.name})...")
                result = await self.sync_peer(peer_id)
                results[peer_id] = result

                if result.success:
                    logger.info(
                        f"Successfully synced '{peer_id}': "
                        f"{result.servers_synced} servers, {result.agents_synced} agents"
                    )
                else:
                    logger.error(f"Failed to sync '{peer_id}': {result.error_message}")

            except Exception as e:
                logger.error(f"Unexpected error syncing peer '{peer_id}': {e}", exc_info=True)
                results[peer_id] = SyncResult(
                    success=False,
                    peer_id=peer_id,
                    servers_synced=0,
                    agents_synced=0,
                    servers_orphaned=0,
                    agents_orphaned=0,
                    error_message=str(e),
                    duration_seconds=0.0,
                    new_generation=0,
                )

        # Summary logging
        successful = sum(1 for r in results.values() if r.success)
        failed = len(results) - successful
        total_servers = sum(r.servers_synced for r in results.values())
        total_agents = sum(r.agents_synced for r in results.values())

        logger.info(
            f"Sync completed: {successful} succeeded, {failed} failed. "
            f"Total: {total_servers} servers, {total_agents} agents"
        )

        return results

    def _filter_servers_by_config(
        self,
        servers: list[dict[str, Any]],
        peer_config: PeerRegistryConfig,
    ) -> list[dict[str, Any]]:
        """
        Filter servers based on peer sync configuration.

        Args:
            servers: List of server data from peer
            peer_config: Peer configuration with sync settings

        Returns:
            Filtered list of servers
        """
        if peer_config.sync_mode == "all":
            return servers

        if peer_config.sync_mode == "whitelist":
            if not peer_config.whitelist_servers:
                logger.debug(
                    f"Peer '{peer_config.peer_id}' has empty whitelist_servers, "
                    "returning empty list"
                )
                return []

            filtered = []
            for server in servers:
                server_path = server.get("path", "")
                if server_path in peer_config.whitelist_servers:
                    filtered.append(server)
                    logger.debug(
                        f"Server '{server_path}' matches whitelist for peer '{peer_config.peer_id}'"
                    )

            logger.info(
                f"Filtered {len(servers)} servers to {len(filtered)} using whitelist "
                f"for peer '{peer_config.peer_id}'"
            )
            return filtered

        if peer_config.sync_mode == "tag_filter":
            if not peer_config.tag_filters:
                logger.debug(
                    f"Peer '{peer_config.peer_id}' has empty tag_filters, returning empty list"
                )
                return []

            filtered = []
            for server in servers:
                if self._matches_tag_filter(server, peer_config.tag_filters):
                    filtered.append(server)
                    logger.debug(
                        f"Server '{server.get('path', '')}' matches tag filter "
                        f"for peer '{peer_config.peer_id}'"
                    )

            logger.info(
                f"Filtered {len(servers)} servers to {len(filtered)} using tag filter "
                f"for peer '{peer_config.peer_id}'"
            )
            return filtered

        logger.warning(
            f"Unknown sync_mode '{peer_config.sync_mode}' for peer "
            f"'{peer_config.peer_id}', returning all servers"
        )
        return servers

    def _filter_agents_by_config(
        self,
        agents: list[dict[str, Any]],
        peer_config: PeerRegistryConfig,
    ) -> list[dict[str, Any]]:
        """
        Filter agents based on peer sync configuration.

        Args:
            agents: List of agent data from peer
            peer_config: Peer configuration with sync settings

        Returns:
            Filtered list of agents
        """
        if peer_config.sync_mode == "all":
            return agents

        if peer_config.sync_mode == "whitelist":
            if not peer_config.whitelist_agents:
                logger.debug(
                    f"Peer '{peer_config.peer_id}' has empty whitelist_agents, returning empty list"
                )
                return []

            filtered = []
            for agent in agents:
                agent_path = agent.get("path", "")
                if agent_path in peer_config.whitelist_agents:
                    filtered.append(agent)
                    logger.debug(
                        f"Agent '{agent_path}' matches whitelist for peer '{peer_config.peer_id}'"
                    )

            logger.info(
                f"Filtered {len(agents)} agents to {len(filtered)} using whitelist "
                f"for peer '{peer_config.peer_id}'"
            )
            return filtered

        if peer_config.sync_mode == "tag_filter":
            if not peer_config.tag_filters:
                logger.debug(
                    f"Peer '{peer_config.peer_id}' has empty tag_filters, returning empty list"
                )
                return []

            filtered = []
            for agent in agents:
                if self._matches_tag_filter(agent, peer_config.tag_filters):
                    filtered.append(agent)
                    logger.debug(
                        f"Agent '{agent.get('path', '')}' matches tag filter "
                        f"for peer '{peer_config.peer_id}'"
                    )

            logger.info(
                f"Filtered {len(agents)} agents to {len(filtered)} using tag filter "
                f"for peer '{peer_config.peer_id}'"
            )
            return filtered

        logger.warning(
            f"Unknown sync_mode '{peer_config.sync_mode}' for peer "
            f"'{peer_config.peer_id}', returning all agents"
        )
        return agents

    def _matches_tag_filter(
        self,
        item: dict[str, Any],
        tag_filters: list[str],
    ) -> bool:
        """
        Check if an item matches any of the tag filters.

        Args:
            item: Server or agent data dict
            tag_filters: List of tag strings to match

        Returns:
            True if item has any matching tag
        """
        # Extract tags from item - could be in "tags" or "categories" field
        item_tags = item.get("tags", [])
        if not isinstance(item_tags, list):
            item_tags = []

        # Also check categories field
        item_categories = item.get("categories", [])
        if not isinstance(item_categories, list):
            item_categories = []

        # Combine both lists
        all_item_tags = item_tags + item_categories

        # Check if any filter matches any tag
        for filter_tag in tag_filters:
            if filter_tag in all_item_tags:
                return True

        return False

    async def detect_orphaned_items(
        self,
        peer_id: str,
        current_server_paths: list[str],
        current_agent_paths: list[str],
    ) -> tuple[list[str], list[str]]:
        """
        Detect items that exist locally but no longer exist in peer.

        Args:
            peer_id: Peer identifier
            current_server_paths: Paths of servers currently in peer
            current_agent_paths: Paths of agents currently in peer

        Returns:
            Tuple of (orphaned_server_paths, orphaned_agent_paths)
        """
        orphaned_servers = []
        orphaned_agents = []

        # Normalize current paths for comparison (ensure leading slash)
        normalized_server_paths = {
            p if p.startswith("/") else f"/{p}" for p in current_server_paths if p
        }
        normalized_agent_paths = {
            p if p.startswith("/") else f"/{p}" for p in current_agent_paths if p
        }

        # Find all local servers with sync_metadata.source_peer_id == peer_id
        all_servers = await server_service.get_all_servers()
        for server in all_servers.values():
            server_dict = server if isinstance(server, dict) else server
            sync_metadata = server_dict.get("sync_metadata") or {}
            path = server_dict.get("path", "")

            if sync_metadata.get("source_peer_id") == peer_id:
                # Extract and normalize original path for comparison
                original_path = sync_metadata.get("original_path", "")
                normalized_original = (
                    original_path if original_path.startswith("/") else f"/{original_path}"
                )

                # Check if normalized original path is in current peer paths
                if normalized_original not in normalized_server_paths:
                    orphaned_servers.append(path)
                    logger.debug(f"Detected orphaned server: {path} (original: {original_path})")

        # Find all local agents with sync_metadata.source_peer_id == peer_id
        all_agents = await agent_service.get_all_agents()
        for agent in all_agents:
            agent_dict = agent.model_dump() if hasattr(agent, "model_dump") else agent
            sync_metadata = agent_dict.get("sync_metadata") or {}
            path = agent_dict.get("path", "")

            if sync_metadata.get("source_peer_id") == peer_id:
                # Extract and normalize original path for comparison
                original_path = sync_metadata.get("original_path", "")
                normalized_original = (
                    original_path if original_path.startswith("/") else f"/{original_path}"
                )

                # Check if normalized original path is in current peer paths
                if normalized_original not in normalized_agent_paths:
                    orphaned_agents.append(path)
                    logger.debug(f"Detected orphaned agent: {path} (original: {original_path})")

        logger.info(
            f"Detected {len(orphaned_servers)} orphaned servers and "
            f"{len(orphaned_agents)} orphaned agents from peer '{peer_id}'"
        )

        return orphaned_servers, orphaned_agents

    async def mark_item_as_orphaned(
        self,
        item_path: str,
        item_type: Literal["server", "agent"],
    ) -> bool:
        """
        Mark a synced item as orphaned.

        Args:
            item_path: Path of the item (prefixed)
            item_type: "server" or "agent"

        Returns:
            True if marked successfully
        """
        try:
            if item_type == "server":
                existing_server = await server_service.get_server_info(item_path)
                if not existing_server:
                    logger.warning(f"Server not found for orphan marking: {item_path}")
                    return False

                # get_server_info returns a dict
                server_dict = existing_server

                # Update sync_metadata
                sync_metadata = server_dict.get("sync_metadata") or {}
                sync_metadata["is_orphaned"] = True
                sync_metadata["orphaned_at"] = datetime.now(UTC).isoformat()

                server_dict["sync_metadata"] = sync_metadata

                # Update server
                success = await server_service.update_server(item_path, server_dict)
                if success:
                    logger.info(f"Marked server as orphaned: {item_path}")
                return success

            elif item_type == "agent":
                existing_agent = await agent_service.get_agent_info(item_path)
                if not existing_agent:
                    logger.warning(f"Agent not found for orphan marking: {item_path}")
                    return False

                # get_agent_info returns a dict
                agent_dict = existing_agent

                # Update sync_metadata
                sync_metadata = agent_dict.get("sync_metadata") or {}
                sync_metadata["is_orphaned"] = True
                sync_metadata["orphaned_at"] = datetime.now(UTC).isoformat()

                agent_dict["sync_metadata"] = sync_metadata

                # Update agent
                updated_agent = await agent_service.update_agent(item_path, agent_dict)
                if updated_agent:
                    logger.info(f"Marked agent as orphaned: {item_path}")
                    return True
                return False

            else:
                logger.error(f"Invalid item_type: {item_type}")
                return False

        except Exception as e:
            logger.error(
                f"Failed to mark item as orphaned: {item_path} ({item_type}): {e}",
                exc_info=True,
            )
            return False

    async def handle_orphaned_items(
        self,
        peer_id: str,
        orphaned_servers: list[str],
        orphaned_agents: list[str],
        action: Literal["mark", "delete"] = "mark",
    ) -> int:
        """
        Handle orphaned items by marking or deleting them.

        Args:
            peer_id: Source peer ID
            orphaned_servers: List of orphaned server paths
            orphaned_agents: List of orphaned agent paths
            action: "mark" to mark as orphaned, "delete" to remove

        Returns:
            Number of items handled
        """
        handled_count = 0

        logger.info(
            f"Handling {len(orphaned_servers)} orphaned servers and "
            f"{len(orphaned_agents)} orphaned agents from peer '{peer_id}' "
            f"(action: {action})"
        )

        # Handle orphaned servers
        for server_path in orphaned_servers:
            try:
                if action == "mark":
                    if await self.mark_item_as_orphaned(server_path, "server"):
                        handled_count += 1
                elif action == "delete":
                    success = await server_service.remove_server(server_path)
                    if success:
                        logger.info(f"Deleted orphaned server: {server_path}")
                        handled_count += 1
                    else:
                        logger.error(f"Failed to delete orphaned server: {server_path}")
            except Exception as e:
                logger.error(
                    f"Failed to handle orphaned server {server_path}: {e}",
                    exc_info=True,
                )

        # Handle orphaned agents
        for agent_path in orphaned_agents:
            try:
                if action == "mark":
                    if await self.mark_item_as_orphaned(agent_path, "agent"):
                        handled_count += 1
                elif action == "delete":
                    success = await agent_service.remove_agent(agent_path)
                    if success:
                        logger.info(f"Deleted orphaned agent: {agent_path}")
                        handled_count += 1
                    else:
                        logger.error(f"Failed to delete orphaned agent: {agent_path}")
            except Exception as e:
                logger.error(f"Failed to handle orphaned agent {agent_path}: {e}", exc_info=True)

        logger.info(
            f"Successfully handled {handled_count}/{len(orphaned_servers) + len(orphaned_agents)} "
            f"orphaned items from peer '{peer_id}'"
        )

        return handled_count

    async def set_local_override(
        self,
        item_path: str,
        item_type: Literal["server", "agent"],
        override: bool = True,
    ) -> bool:
        """
        Set or clear local override flag for a synced item.

        When override=True, sync will skip this item to preserve local changes.

        Args:
            item_path: Path of the item
            item_type: "server" or "agent"
            override: True to set override, False to clear

        Returns:
            True if set successfully
        """
        try:
            if item_type == "server":
                existing_server = await server_service.get_server_info(item_path)
                if not existing_server:
                    logger.warning(f"Server not found for local override: {item_path}")
                    return False

                # get_server_info returns a dict
                server_dict = existing_server

                # Update sync_metadata
                sync_metadata = server_dict.get("sync_metadata") or {}
                sync_metadata["local_overrides"] = override

                server_dict["sync_metadata"] = sync_metadata

                # Update server
                success = await server_service.update_server(item_path, server_dict)
                if success:
                    logger.info(f"Set local override to {override} for server: {item_path}")
                return success

            elif item_type == "agent":
                existing_agent = await agent_service.get_agent_info(item_path)
                if not existing_agent:
                    logger.warning(f"Agent not found for local override: {item_path}")
                    return False

                # get_agent_info returns a dict
                agent_dict = existing_agent

                # Update sync_metadata
                sync_metadata = agent_dict.get("sync_metadata") or {}
                sync_metadata["local_overrides"] = override

                agent_dict["sync_metadata"] = sync_metadata

                # Update agent
                updated_agent = await agent_service.update_agent(item_path, agent_dict)
                if updated_agent:
                    logger.info(f"Set local override to {override} for agent: {item_path}")
                    return True
                return False

            else:
                logger.error(f"Invalid item_type: {item_type}")
                return False

        except Exception as e:
            logger.error(
                f"Failed to set local override for item: {item_path} ({item_type}): {e}",
                exc_info=True,
            )
            return False

    def is_locally_overridden(
        self,
        item: dict[str, Any],
    ) -> bool:
        """
        Check if an item has local override flag set.

        Args:
            item: Server or agent data dict

        Returns:
            True if item has local override
        """
        sync_metadata = item.get("sync_metadata") or {}
        return sync_metadata.get("local_overrides", False)

    async def _index_server_for_search(
        self,
        path: str,
        server_data: dict[str, Any],
    ) -> None:
        """
        Explicitly index a server for search (embeddings).

        This is called after successfully storing a synced server to ensure
        it's indexed for semantic search. The server_service methods should
        do this automatically, but this is a fallback to ensure it happens.

        Args:
            path: Server path (e.g., /peer-registry-lob-1/my-server)
            server_data: Server data dict
        """
        try:
            search_repo = get_search_repository()
            is_enabled = server_data.get("is_enabled", True)
            await search_repo.index_server(path, server_data, is_enabled)
            logger.debug(f"Indexed synced server for search: {path}")
        except Exception as e:
            logger.error(f"Failed to index synced server {path} for search: {e}")

    async def _index_agent_for_search(
        self,
        path: str,
        agent_card: AgentCard,
    ) -> None:
        """
        Explicitly index an agent for search (embeddings).

        This is called after successfully storing a synced agent to ensure
        it's indexed for semantic search. The agent_service methods should
        do this automatically, but this is a fallback to ensure it happens.

        Args:
            path: Agent path (e.g., /peer-registry-lob-1/my-agent)
            agent_card: AgentCard instance
        """
        try:
            search_repo = get_search_repository()
            is_enabled = agent_service.is_agent_enabled(path)
            await search_repo.index_agent(path, agent_card, is_enabled)
            logger.debug(f"Indexed synced agent for search: {path}")
        except Exception as e:
            logger.error(f"Failed to index synced agent {path} for search: {e}")

    async def _store_synced_servers(
        self,
        peer_id: str,
        servers: list[dict[str, Any]],
    ) -> int:
        """
        Store servers fetched from a peer.

        Args:
            peer_id: Source peer identifier
            servers: List of server data dictionaries

        Returns:
            Number of servers stored/updated
        """
        stored_count = 0

        for server in servers:
            try:
                # Extract original path
                original_path = server.get("path", "")

                if not original_path:
                    logger.warning(f"Server missing 'path' field, skipping: {server}")
                    continue

                # Normalize path - ensure it starts with /
                normalized_path = (
                    original_path if original_path.startswith("/") else f"/{original_path}"
                )

                # Prefix path with peer_id to avoid collisions
                # e.g., "/my-server" becomes "/peer-central/my-server"
                prefixed_path = f"/{peer_id}{normalized_path}"

                # Add sync_metadata to track origin
                sync_metadata = {
                    "source_peer_id": peer_id,
                    "synced_at": datetime.now(UTC).isoformat(),
                    "is_federated": True,
                    "original_path": original_path,
                }

                # Create a copy to avoid modifying original
                server_data = server.copy()
                server_data["path"] = prefixed_path
                server_data["sync_metadata"] = sync_metadata

                # Check if server already exists and store
                try:
                    existing_server = await server_service.get_server_info(prefixed_path)
                    if existing_server:
                        # Check if locally overridden - if so, skip update
                        # get_server_info returns a dict
                        if self.is_locally_overridden(existing_server):
                            logger.debug(
                                f"Skipping update for locally overridden server: {prefixed_path}"
                            )
                            continue

                        # Update existing server - returns bool
                        success = await server_service.update_server(prefixed_path, server_data)
                        if success:
                            logger.debug(f"Updated synced server: {prefixed_path}")
                            stored_count += 1
                            # Explicitly index for search (embeddings)
                            await self._index_server_for_search(prefixed_path, server_data)
                        else:
                            logger.error(f"Failed to update server: {prefixed_path}")
                    else:
                        # Register new server - returns dict with 'success' key
                        result = await server_service.register_server(server_data)
                        if result.get("success"):
                            logger.debug(f"Registered synced server: {prefixed_path}")
                            stored_count += 1
                            # Explicitly index for search (embeddings)
                            await self._index_server_for_search(prefixed_path, server_data)
                        else:
                            logger.error(f"Failed to register server: {prefixed_path}")

                except Exception as e:
                    logger.error(f"Failed to store server '{prefixed_path}': {e}", exc_info=True)

            except Exception as e:
                logger.error(
                    f"Failed to process server from peer '{peer_id}': {e}",
                    exc_info=True,
                )

        logger.info(f"Stored {stored_count}/{len(servers)} servers from peer '{peer_id}'")
        return stored_count

    async def _store_synced_agents(
        self,
        peer_id: str,
        agents: list[dict[str, Any]],
    ) -> int:
        """
        Store agents fetched from a peer.

        Args:
            peer_id: Source peer identifier
            agents: List of agent data dictionaries

        Returns:
            Number of agents stored/updated
        """
        stored_count = 0

        for agent in agents:
            try:
                # Extract original path
                original_path = agent.get("path", "")

                if not original_path:
                    logger.warning(f"Agent missing 'path' field, skipping: {agent}")
                    continue

                # Normalize path - ensure it starts with /
                normalized_path = (
                    original_path if original_path.startswith("/") else f"/{original_path}"
                )

                # Prefix path with peer_id to avoid collisions
                # e.g., "/code-reviewer" becomes "/peer-central/code-reviewer"
                prefixed_path = f"/{peer_id}{normalized_path}"

                # Add sync_metadata to track origin
                sync_metadata = {
                    "source_peer_id": peer_id,
                    "synced_at": datetime.now(UTC).isoformat(),
                    "is_federated": True,
                    "original_path": original_path,
                }

                # Create a copy to avoid modifying original
                agent_data = agent.copy()
                agent_data["path"] = prefixed_path
                agent_data["sync_metadata"] = sync_metadata

                # Check if agent already exists and store
                try:
                    existing_agent = await agent_service.get_agent_info(prefixed_path)

                    if existing_agent:
                        # Check if locally overridden - if so, skip update
                        # get_agent_info returns a dict
                        if self.is_locally_overridden(existing_agent):
                            logger.debug(
                                f"Skipping update for locally overridden agent: {prefixed_path}"
                            )
                            continue

                        # Update existing agent - returns AgentCard on success
                        updated_agent = await agent_service.update_agent(prefixed_path, agent_data)
                        if updated_agent:
                            logger.debug(f"Updated synced agent: {prefixed_path}")
                            stored_count += 1
                            # Explicitly index for search (embeddings)
                            await self._index_agent_for_search(prefixed_path, updated_agent)
                        else:
                            logger.error(f"Failed to update agent: {prefixed_path}")
                    else:
                        # Register new agent - create AgentCard instance
                        agent_card = AgentCard(**agent_data)
                        registered_agent = await agent_service.register_agent(agent_card)
                        if registered_agent:
                            logger.debug(f"Registered synced agent: {prefixed_path}")
                            stored_count += 1
                            # Explicitly index for search (embeddings)
                            await self._index_agent_for_search(prefixed_path, registered_agent)
                        else:
                            logger.error(f"Failed to register agent: {prefixed_path}")

                except ValueError as e:
                    # Validation errors
                    logger.error(f"Validation error storing agent '{prefixed_path}': {e}")
                except Exception as e:
                    logger.error(f"Failed to store agent '{prefixed_path}': {e}", exc_info=True)

            except Exception as e:
                logger.error(f"Failed to process agent from peer '{peer_id}': {e}", exc_info=True)

        logger.info(f"Stored {stored_count}/{len(agents)} agents from peer '{peer_id}'")
        return stored_count

    async def _store_synced_security_scans(
        self,
        peer_id: str,
        security_scans: list[dict[str, Any]],
    ) -> int:
        """
        Store security scan results fetched from a peer.

        Args:
            peer_id: Source peer identifier
            security_scans: List of security scan dictionaries

        Returns:
            Number of scans stored/updated
        """
        stored_count = 0

        if not security_scans:
            logger.debug(f"No security scans to store from peer '{peer_id}'")
            return 0

        # Get security scan repository
        scan_repo = get_security_scan_repository()

        for scan in security_scans:
            try:
                # Extract original server path
                original_server_path = scan.get("server_path", "")

                if not original_server_path:
                    logger.warning(f"Security scan missing 'server_path' field, skipping")
                    continue

                # Normalize path - ensure it starts with /
                normalized_path = (
                    original_server_path
                    if original_server_path.startswith("/")
                    else f"/{original_server_path}"
                )

                # Prefix path with peer_id to match synced server paths
                # e.g., "/my-server" becomes "/peer-central/my-server"
                prefixed_path = f"/{peer_id}{normalized_path}"

                # Create a copy to avoid modifying original
                scan_data = scan.copy()
                scan_data["server_path"] = prefixed_path

                # Add sync_metadata to track origin
                scan_data["sync_metadata"] = {
                    "source_peer_id": peer_id,
                    "synced_at": datetime.now(UTC).isoformat(),
                    "is_federated": True,
                    "original_server_path": original_server_path,
                }

                # Store the scan via repository
                try:
                    success = await scan_repo.create(scan_data)
                    if success:
                        logger.debug(f"Stored synced security scan for: {prefixed_path}")
                        stored_count += 1
                    else:
                        logger.error(f"Failed to store security scan for: {prefixed_path}")

                except Exception as e:
                    logger.error(
                        f"Failed to store security scan for '{prefixed_path}': {e}",
                        exc_info=True,
                    )

            except Exception as e:
                logger.error(
                    f"Failed to process security scan from peer '{peer_id}': {e}",
                    exc_info=True,
                )

        logger.info(
            f"Stored {stored_count}/{len(security_scans)} security scans from peer '{peer_id}'"
        )
        return stored_count


# Global service instance
_peer_federation_service: PeerFederationService | None = None


def get_peer_federation_service() -> PeerFederationService:
    """
    Get the global peer federation service instance.

    Returns:
        Singleton PeerFederationService instance
    """
    global _peer_federation_service
    if _peer_federation_service is None:
        _peer_federation_service = PeerFederationService()
    return _peer_federation_service
