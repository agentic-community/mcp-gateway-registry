"""
Anthropic MCP Registry federation client.

Fetches server configurations from Anthropic's MCP Registry API
and transforms them to the gateway's internal format.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from .base_client import BaseFederationClient
from ...schemas.federation_schema import AnthropicServerConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


class AnthropicFederationClient(BaseFederationClient):
    """Client for fetching servers from Anthropic MCP Registry."""

    def __init__(
        self,
        endpoint: str,
        api_version: str = "v0.1",
        timeout_seconds: int = 30,
        retry_attempts: int = 3
    ):
        """
        Initialize Anthropic federation client.

        Args:
            endpoint: Base URL for Anthropic MCP Registry API
            api_version: API version to use (default: v0.1)
            timeout_seconds: HTTP request timeout
            retry_attempts: Number of retry attempts
        """
        super().__init__(endpoint, timeout_seconds, retry_attempts)
        self.api_version = api_version

    def fetch_server(
        self,
        server_name: str,
        server_config: Optional[AnthropicServerConfig] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a single server from Anthropic Registry.

        Args:
            server_name: Server name in Anthropic format (e.g., ai.smithery/github)
            server_config: Optional server configuration with auth details

        Returns:
            Server data dictionary or None if fetch fails
        """
        # Use custom endpoint if provided, otherwise construct from server name
        if server_config and server_config.endpoint:
            url = server_config.endpoint
        else:
            # URL-encode server name (replace / with %2F)
            encoded_name = quote(server_name, safe="")
            url = f"{self.endpoint}/{self.api_version}/servers/{encoded_name}/versions/latest"

        # Build headers
        headers = {"Content-Type": "application/json"}

        # Add authentication if required
        if server_config and server_config.requires_auth:
            auth_value = self._get_auth_value(server_config)
            if auth_value:
                if server_config.auth_type == "bearer" or server_config.auth_type == "oauth":
                    headers["Authorization"] = f"Bearer {auth_value}"
                elif server_config.auth_type == "api-key":
                    headers["X-API-Key"] = auth_value
                else:
                    logger.warning(f"Unknown auth type: {server_config.auth_type}")

        # Make request
        logger.info(f"Fetching server {server_name} from Anthropic Registry")
        response = self._make_request(url, headers=headers)

        if not response:
            logger.error(f"Failed to fetch server {server_name}")
            return None

        # Transform response to internal format
        return self._transform_server_response(response, server_name, server_config)

    def fetch_all_servers(
        self,
        server_configs: List[AnthropicServerConfig]
    ) -> List[Dict[str, Any]]:
        """
        Fetch multiple servers from Anthropic Registry.

        Args:
            server_configs: List of server configurations

        Returns:
            List of server data dictionaries
        """
        servers = []

        for config in server_configs:
            if not config.enabled:
                logger.info(f"Skipping disabled server: {config.name}")
                continue

            server_data = self.fetch_server(config.name, config)
            if server_data:
                servers.append(server_data)
            else:
                logger.warning(f"Failed to fetch server: {config.name}")

        logger.info(f"Successfully fetched {len(servers)}/{len(server_configs)} servers")
        return servers

    def _get_auth_value(
        self,
        server_config: AnthropicServerConfig
    ) -> Optional[str]:
        """
        Get authentication value from environment variable.

        Args:
            server_config: Server configuration with auth details

        Returns:
            Authentication value or None
        """
        if not server_config.auth_env_var:
            logger.warning(f"No auth_env_var specified for {server_config.name}")
            return None

        auth_value = os.getenv(server_config.auth_env_var)
        if not auth_value:
            logger.error(
                f"Environment variable {server_config.auth_env_var} not found for {server_config.name}"
            )
            return None

        return auth_value

    def _transform_server_response(
        self,
        response: Dict[str, Any],
        server_name: str,
        server_config: Optional[AnthropicServerConfig]
    ) -> Dict[str, Any]:
        """
        Transform Anthropic API response to internal gateway format.

        Args:
            response: Raw response from Anthropic API
            server_name: Server name
            server_config: Optional server configuration

        Returns:
            Transformed server data
        """
        # Extract server details from response
        server = response.get("server", {})

        # Get basic info
        description = server.get("description", "")
        version = server.get("version", "1.0.0")
        title = server.get("title", server_name)

        # Extract transport info from packages
        packages = server.get("packages", [])
        transport_type = "streamable-http"
        proxy_url = None

        if packages:
            package = packages[0]
            transport = package.get("transport", {})
            transport_type = transport.get("type", "streamable-http")
            proxy_url = transport.get("url")

        # Extract tags from metadata if available
        tags = []
        metadata = server.get("_meta", {})
        for key, value in metadata.items():
            if isinstance(value, dict):
                internal_tags = value.get("tags", [])
                if internal_tags:
                    tags.extend(internal_tags)

        # Add default tags from server name
        name_parts = server_name.split("/")
        if len(name_parts) > 1:
            tags.extend([name_parts[0], name_parts[1]])
        tags.append("anthropic-registry")
        tags.append("federated")

        # Build auth headers if needed
        auth_headers = []
        if server_config and server_config.requires_auth:
            auth_value = self._get_auth_value(server_config)
            if auth_value:
                if server_config.auth_type == "bearer" or server_config.auth_type == "oauth":
                    auth_headers.append({"Authorization": f"Bearer {auth_value}"})
                elif server_config.auth_type == "api-key":
                    auth_headers.append({"X-API-Key": auth_value})

        # Build transformed server object
        transformed = {
            "source": "anthropic",
            "server_name": server_name,
            "description": description,
            "version": version,
            "title": title,
            "proxy_pass_url": proxy_url,
            "transport_type": transport_type,
            "requires_auth": server_config.requires_auth if server_config else False,
            "auth_headers": auth_headers,
            "tags": list(set(tags)),  # Remove duplicates
            "metadata": {
                "original_response": response,
                "config_metadata": server_config.metadata if server_config else {}
            },
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "is_read_only": True,
            "attribution_label": "Anthropic MCP Registry",
            # Additional fields for compatibility
            "path": f"/{server_name.replace('/', '-')}",
            "is_enabled": True,
            "health_status": "unknown",  # Will be updated by health checks
            "num_tools": 0,  # Will be updated if we can query the server
        }

        return transformed
