"""
Peer registry federation client.

Fetches servers and agents from peer registries using the standard
federation API endpoints with JWT authentication.
"""

import logging
from typing import Any

from ...schemas.peer_federation_schema import PeerRegistryConfig
from .base_client import BaseFederationClient
from .federation_auth import FederationAuthManager

logger = logging.getLogger(__name__)


class PeerRegistryClient(BaseFederationClient):
    """Client for fetching servers and agents from peer registries."""

    def __init__(
        self, peer_config: PeerRegistryConfig, timeout_seconds: int = 30, retry_attempts: int = 3
    ):
        """
        Initialize peer registry client.

        Args:
            peer_config: Configuration for the peer registry
            timeout_seconds: HTTP request timeout
            retry_attempts: Number of retry attempts for failed requests
        """
        super().__init__(peer_config.endpoint, timeout_seconds, retry_attempts)
        self.peer_config = peer_config

        # Per-peer federation static token takes priority over global OAuth2
        self._federation_token = peer_config.federation_token
        self._auth_manager = FederationAuthManager()

        # Validate auth is configured (either per-peer token or global OAuth2)
        if self._federation_token:
            logger.info(f"Using per-peer federation static token for peer '{peer_config.peer_id}'")
        elif not self._auth_manager.is_configured():
            logger.warning(
                f"Federation authentication not configured for peer '{peer_config.peer_id}'. "
                "Set federation_token in peer config, or set FEDERATION_TOKEN_ENDPOINT, "
                "FEDERATION_CLIENT_ID, and FEDERATION_CLIENT_SECRET environment variables."
            )

        logger.info(
            f"Initialized PeerRegistryClient for peer '{peer_config.peer_id}' "
            f"at {peer_config.endpoint}"
        )

    def _get_auth_token(self) -> str | None:
        """Get authentication token for this peer.

        Uses per-peer federation static token if configured,
        otherwise falls back to global OAuth2 FederationAuthManager.

        Returns:
            Bearer token string, or None if auth fails.

        Raises:
            ValueError: If no authentication method is configured.
        """
        # Per-peer federation static token takes priority
        if self._federation_token:
            return self._federation_token

        # Fall back to global OAuth2 auth manager
        return self._auth_manager.get_token()

    def fetch_servers(self, since_generation: int | None = None) -> list[dict[str, Any]] | None:
        """
        Fetch servers from peer registry.

        Args:
            since_generation: Optional generation number for incremental sync.
                            If provided, only returns servers updated since that generation.

        Returns:
            List of server dictionaries or None if fetch fails
        """
        # Build URL
        url = f"{self.endpoint}/api/federation/servers"

        # Get authentication token
        try:
            token = self._get_auth_token()
        except ValueError as e:
            logger.error(f"Cannot fetch servers: {e}")
            return None

        if not token:
            logger.error(
                f"Failed to obtain authentication token for peer '{self.peer_config.peer_id}'"
            )
            return None

        # Build headers
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Build query parameters
        params = {}
        if since_generation is not None:
            params["since_generation"] = since_generation

        # Make request
        logger.info(
            f"Fetching servers from peer '{self.peer_config.peer_id}' "
            f"(since_generation={since_generation})"
        )

        response = self._make_request(url, headers=headers, params=params)

        if not response:
            logger.error(f"Failed to fetch servers from peer '{self.peer_config.peer_id}'")
            return None

        # Extract items from response
        # Expected format: {"items": [...], "sync_generation": N, ...}
        if isinstance(response, dict):
            items = response.get("items", [])
            sync_generation = response.get("sync_generation", 0)
            total_count = response.get("total_count", len(items))

            logger.info(
                f"Successfully fetched {len(items)} servers from peer "
                f"'{self.peer_config.peer_id}' (generation={sync_generation}, "
                f"total={total_count})"
            )
            return items

        elif isinstance(response, list):
            # Handle direct list response
            logger.info(
                f"Successfully fetched {len(response)} servers from peer "
                f"'{self.peer_config.peer_id}' (direct list response)"
            )
            return response

        else:
            logger.error(
                f"Unexpected response format from peer '{self.peer_config.peer_id}': "
                f"{type(response)}"
            )
            return None

    def fetch_security_scans(self) -> list[dict[str, Any]] | None:
        """
        Fetch security scan results from peer registry.

        Security scans are filtered by the peer based on server visibility,
        so only scans for servers visible to this client are returned.

        Returns:
            List of security scan dictionaries or None if fetch fails
        """
        # Build URL
        url = f"{self.endpoint}/api/federation/security-scans"

        # Get authentication token
        try:
            token = self._get_auth_token()
        except ValueError as e:
            logger.error(f"Cannot fetch security scans: {e}")
            return None

        if not token:
            logger.error(
                f"Failed to obtain authentication token for peer '{self.peer_config.peer_id}'"
            )
            return None

        # Build headers
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Make request
        logger.info(f"Fetching security scans from peer '{self.peer_config.peer_id}'")

        response = self._make_request(url, headers=headers)

        if not response:
            logger.error(f"Failed to fetch security scans from peer '{self.peer_config.peer_id}'")
            return None

        # Extract items from response
        # Expected format: {"items": [...], "sync_generation": N, ...}
        if isinstance(response, dict):
            items = response.get("items", [])
            total_count = response.get("total_count", len(items))

            logger.info(
                f"Successfully fetched {len(items)} security scans from peer "
                f"'{self.peer_config.peer_id}' (total={total_count})"
            )
            return items

        elif isinstance(response, list):
            # Handle direct list response
            logger.info(
                f"Successfully fetched {len(response)} security scans from peer "
                f"'{self.peer_config.peer_id}' (direct list response)"
            )
            return response

        else:
            logger.error(
                f"Unexpected response format from peer '{self.peer_config.peer_id}': "
                f"{type(response)}"
            )
            return None

    def fetch_agents(self, since_generation: int | None = None) -> list[dict[str, Any]] | None:
        """
        Fetch agents from peer registry.

        Args:
            since_generation: Optional generation number for incremental sync.
                            If provided, only returns agents updated since that generation.

        Returns:
            List of agent dictionaries or None if fetch fails
        """
        # Build URL
        url = f"{self.endpoint}/api/federation/agents"

        # Get authentication token
        try:
            token = self._get_auth_token()
        except ValueError as e:
            logger.error(f"Cannot fetch agents: {e}")
            return None

        if not token:
            logger.error(
                f"Failed to obtain authentication token for peer '{self.peer_config.peer_id}'"
            )
            return None

        # Build headers
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Build query parameters
        params = {}
        if since_generation is not None:
            params["since_generation"] = since_generation

        # Make request
        logger.info(
            f"Fetching agents from peer '{self.peer_config.peer_id}' "
            f"(since_generation={since_generation})"
        )

        response = self._make_request(url, headers=headers, params=params)

        if not response:
            logger.error(f"Failed to fetch agents from peer '{self.peer_config.peer_id}'")
            return None

        # Extract items from response
        # Expected format: {"items": [...], "sync_generation": N, ...}
        if isinstance(response, dict):
            items = response.get("items", [])
            sync_generation = response.get("sync_generation", 0)
            total_count = response.get("total_count", len(items))

            logger.info(
                f"Successfully fetched {len(items)} agents from peer "
                f"'{self.peer_config.peer_id}' (generation={sync_generation}, "
                f"total={total_count})"
            )
            return items

        elif isinstance(response, list):
            # Handle direct list response
            logger.info(
                f"Successfully fetched {len(response)} agents from peer "
                f"'{self.peer_config.peer_id}' (direct list response)"
            )
            return response

        else:
            logger.error(
                f"Unexpected response format from peer '{self.peer_config.peer_id}': "
                f"{type(response)}"
            )
            return None

    def check_peer_health(self) -> bool:
        """
        Check if peer registry is healthy and reachable.

        Makes a lightweight health check request to the peer's health endpoint.

        Returns:
            True if peer is healthy, False otherwise
        """
        # Try health endpoint first
        health_url = f"{self.endpoint}/health"

        logger.debug(f"Checking health of peer '{self.peer_config.peer_id}'")

        try:
            # Don't need auth for health check
            response = self.client.get(health_url)

            # Accept 2xx status codes
            if 200 <= response.status_code < 300:
                logger.debug(
                    f"Peer '{self.peer_config.peer_id}' is healthy (status={response.status_code})"
                )
                return True

            logger.warning(
                f"Peer '{self.peer_config.peer_id}' health check returned "
                f"status {response.status_code}"
            )
            return False

        except Exception as e:
            logger.error(f"Health check failed for peer '{self.peer_config.peer_id}': {e}")
            return False

    def fetch_server(self, server_name: str, **kwargs) -> dict[str, Any] | None:
        """
        Fetch a single server from peer registry.

        This is required by BaseFederationClient but not used for peer registries
        which typically fetch in bulk via fetch_servers().

        Args:
            server_name: Name/path of the server to fetch
            **kwargs: Additional parameters

        Returns:
            Server data dictionary or None if fetch fails
        """
        # For peer registries, we typically fetch all servers
        # and filter client-side. But we can implement single fetch
        # if the peer API supports it.
        servers = self.fetch_servers()
        if not servers:
            return None

        # Find server by name/path
        for server in servers:
            if server.get("path") == server_name or server.get("server_name") == server_name:
                return server

        logger.warning(f"Server '{server_name}' not found in peer '{self.peer_config.peer_id}'")
        return None

    def fetch_all_servers(self, server_names: list[str], **kwargs) -> list[dict[str, Any]]:
        """
        Fetch multiple servers from peer registry.

        This is required by BaseFederationClient but for peer registries
        we typically fetch all servers and filter client-side.

        Args:
            server_names: List of server names/paths to fetch
            **kwargs: Additional parameters

        Returns:
            List of server data dictionaries
        """
        # Fetch all servers
        all_servers = self.fetch_servers()
        if not all_servers:
            return []

        # Filter to requested servers if specific names provided
        if server_names:
            filtered = []
            for server in all_servers:
                server_id = server.get("path") or server.get("server_name")
                if server_id in server_names:
                    filtered.append(server)
            return filtered

        return all_servers
