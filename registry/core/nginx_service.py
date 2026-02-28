import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from registry.constants import REGISTRY_CONSTANTS, HealthStatus

from .config import settings
from .metrics import NGINX_UPDATES_SKIPPED

logger = logging.getLogger(__name__)


def _ensure_mcp_compliant_schema(input_schema: dict[str, Any]) -> dict[str, Any]:
    """Ensure inputSchema conforms to MCP spec by adding 'type': 'object' if missing.

    The MCP spec requires all tool inputSchema definitions to have "type": "object"
    at the top level. This function ensures backend tool schemas are compliant.

    Args:
        input_schema: The input schema from a backend tool

    Returns:
        MCP-compliant schema with "type": "object" at top level
    """
    if not input_schema:
        return {"type": "object", "properties": {}}

    # If schema already has "type": "object", return as-is
    if input_schema.get("type") == "object":
        return input_schema

    # If schema has "type" but it's not "object", wrap it
    if "type" in input_schema:
        logger.warning(
            f"Tool inputSchema has non-object type '{input_schema.get('type')}'. "
            "Wrapping in object schema to comply with MCP spec."
        )
        return {"type": "object", "properties": {"value": input_schema}}

    # If no "type" field but has "properties", add "type": "object"
    if "properties" in input_schema or "additionalProperties" in input_schema:
        schema_copy = input_schema.copy()
        schema_copy["type"] = "object"
        return schema_copy

    # Default: wrap unknown schema structure
    logger.warning(
        "Tool inputSchema missing 'type' field and has unexpected structure. "
        "Adding 'type': 'object' to comply with MCP spec."
    )
    schema_copy = input_schema.copy()
    schema_copy["type"] = "object"
    return schema_copy


class NginxConfigService:
    """Service for generating Nginx configuration for registered servers."""

    def __init__(self):
        # Determine which template to use based on SSL certificate availability
        ssl_cert_path = Path(REGISTRY_CONSTANTS.SSL_CERT_PATH)
        ssl_key_path = Path(REGISTRY_CONSTANTS.SSL_KEY_PATH)

        # Check if SSL certificates exist
        if ssl_cert_path.exists() and ssl_key_path.exists():
            # Use HTTP + HTTPS template
            if Path(REGISTRY_CONSTANTS.NGINX_TEMPLATE_HTTP_AND_HTTPS).exists():
                self.nginx_template_path = Path(REGISTRY_CONSTANTS.NGINX_TEMPLATE_HTTP_AND_HTTPS)
            else:
                # Fallback for local development
                self.nginx_template_path = Path(
                    REGISTRY_CONSTANTS.NGINX_TEMPLATE_HTTP_AND_HTTPS_LOCAL
                )
        else:
            # Use HTTP-only template
            if Path(REGISTRY_CONSTANTS.NGINX_TEMPLATE_HTTP_ONLY).exists():
                self.nginx_template_path = Path(REGISTRY_CONSTANTS.NGINX_TEMPLATE_HTTP_ONLY)
            else:
                # Fallback for local development
                self.nginx_template_path = Path(REGISTRY_CONSTANTS.NGINX_TEMPLATE_HTTP_ONLY_LOCAL)

    async def get_additional_server_names(self) -> str:
        """Fetch or determine additional server names for nginx gateway configuration.

        Supports multi-platform detection:
        1. User-provided GATEWAY_ADDITIONAL_SERVER_NAMES env var
        2. EC2 private IP detection via metadata service
        3. ECS metadata service detection
        4. EKS/Kubernetes pod detection
        5. Generic hostname command fallback
        6. Backward compatibility with EC2_PUBLIC_DNS env var
        """
        import os
        import subprocess  # nosec B404

        # Priority 1: Check GATEWAY_ADDITIONAL_SERVER_NAMES env var (user-provided)
        gateway_names = os.environ.get("GATEWAY_ADDITIONAL_SERVER_NAMES", "")
        if gateway_names:
            logger.info(f"Using GATEWAY_ADDITIONAL_SERVER_NAMES from environment: {gateway_names}")
            return gateway_names.strip()

        # Priority 2: Try EC2 metadata service for private IP
        try:
            async with httpx.AsyncClient() as client:
                # Get session token for IMDSv2
                token_response = await client.put(
                    "http://169.254.169.254/latest/api/token",
                    headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
                    timeout=2.0,
                )

                if token_response.status_code == 200:
                    token = token_response.text

                    # Try to get private IP from EC2 metadata
                    ip_response = await client.get(
                        "http://169.254.169.254/latest/meta-data/local-ipv4",
                        headers={"X-aws-ec2-metadata-token": token},
                        timeout=2.0,
                    )

                    if ip_response.status_code == 200:
                        private_ip = ip_response.text.strip()
                        logger.info(f"Auto-detected EC2 private IP: {private_ip}")
                        return private_ip

        except (httpx.TimeoutException, httpx.ConnectError):
            logger.debug("EC2 metadata service not available - not running on EC2")
        except Exception as e:
            logger.debug(f"EC2 metadata detection failed: {e}")

        # Priority 3: Try ECS metadata service
        ecs_uri = os.environ.get("ECS_CONTAINER_METADATA_URI") or os.environ.get(
            "ECS_CONTAINER_METADATA_URI_V4"
        )
        if ecs_uri:
            try:
                async with httpx.AsyncClient() as client:
                    metadata_response = await client.get(f"{ecs_uri}", timeout=2.0)
                    if metadata_response.status_code == 200:
                        import json

                        metadata = json.loads(metadata_response.text)
                        # Try to extract IP from ECS metadata
                        if "Networks" in metadata and metadata["Networks"]:
                            private_ip = metadata["Networks"][0].get("IPv4Addresses", [None])[0]
                            if private_ip:
                                logger.info(f"Auto-detected ECS container IP: {private_ip}")
                                return private_ip
            except Exception as e:
                logger.debug(f"ECS metadata detection failed: {e}")

        # Priority 4: Try EKS/Kubernetes detection
        pod_ip = os.environ.get("POD_IP")
        if pod_ip:
            logger.info(f"Auto-detected Kubernetes pod IP: {pod_ip}")
            return pod_ip

        # Priority 5: Try generic hostname command (works on most Linux systems)
        try:
            result = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=2.0)
            if result.returncode == 0:
                ips = result.stdout.strip().split()
                if ips:
                    # Use first IP (usually the private IP on single-interface systems)
                    private_ip = ips[0]
                    logger.info(f"Auto-detected private IP via hostname command: {private_ip}")
                    return private_ip
        except Exception as e:
            logger.debug(f"Generic hostname detection failed: {e}")

        # Priority 6: Backward compatibility with old EC2_PUBLIC_DNS env var
        fallback_dns = os.environ.get("EC2_PUBLIC_DNS", "")
        if fallback_dns:
            logger.info(f"Using EC2_PUBLIC_DNS environment variable (deprecated): {fallback_dns}")
            return fallback_dns

        # No additional server names available
        logger.info(
            "No additional server names available - will use only localhost and mcpgateway.ddns.net"
        )
        return ""

    def generate_config(self, servers: dict[str, dict[str, Any]]) -> bool:
        """Generate Nginx configuration (synchronous version for non-async contexts)."""
        if not settings.nginx_updates_enabled:
            logger.info(
                f"Skipping nginx config generation - "
                f"DEPLOYMENT_MODE={settings.deployment_mode.value}"
            )
            NGINX_UPDATES_SKIPPED.labels(operation="generate_config").inc()
            return True

        try:
            # Check if we're in an async context
            try:
                # If we're already in an event loop, we need to run this differently
                loop = asyncio.get_running_loop()
                # We're in an async context, this won't work
                logger.error(
                    "generate_config called from async context - use generate_config_async instead"
                )
                return False
            except RuntimeError:
                # No running loop, we can use asyncio.run()
                return asyncio.run(self.generate_config_async(servers))
        except Exception as e:
            logger.error(f"Failed to generate Nginx configuration: {e}", exc_info=True)
            return False

    async def generate_config_async(
        self, servers: dict[str, dict[str, Any]], force_base_config: bool = False
    ) -> bool:
        """Generate Nginx configuration with additional server names and dynamic location blocks.

        Args:
            servers: Dictionary of server path -> server info for location blocks
            force_base_config: If True, generate base config even in registry-only mode
                              (used at startup to ensure nginx has valid config)

        In registry-only mode:
        - At startup (force_base_config=True): generates base config with empty location blocks
        - On server changes (force_base_config=False): skips regeneration (no-op)
        """
        if not settings.nginx_updates_enabled and not force_base_config:
            logger.info(
                f"Skipping nginx config generation - "
                f"DEPLOYMENT_MODE={settings.deployment_mode.value}"
            )
            NGINX_UPDATES_SKIPPED.labels(operation="generate_config").inc()
            return True

        try:
            # Read template
            if not self.nginx_template_path.exists():
                logger.warning(f"Nginx template not found at {self.nginx_template_path}")
                return False

            with open(self.nginx_template_path) as f:
                template_content = f.read()

            # Local-dev / Podman compatibility:
            # The default nginx templates protect `/api/` via `auth_request /validate` (JWT validation).
            # The React dashboard, however, uses cookie-based session auth for `/api/servers` and
            # `/api/tokens/generate`. When auth_request is enabled but Keycloak/Cognito isn't fully
            # configured, nginx returns 403/500 and the UI cannot load.
            #
            # Set NGINX_DISABLE_API_AUTH_REQUEST=true to bypass `auth_request` for `/api/` and rely
            # on FastAPI's own auth (session cookie or bearer token validation inside the app).
            import os

            if os.environ.get("NGINX_DISABLE_API_AUTH_REQUEST", "false").lower() in (
                "1",
                "true",
                "yes",
                "on",
            ):
                protected_api_block = """    # Protected API endpoints - require authentication
    location {{ROOT_PATH}}/api/ {
        # Authenticate request via auth server (validates JWT Bearer tokens)
        auth_request /validate;

        # Capture auth server response headers
        auth_request_set $auth_user $upstream_http_x_user;
        auth_request_set $auth_username $upstream_http_x_username;
        auth_request_set $auth_client_id $upstream_http_x_client_id;
        auth_request_set $auth_scopes $upstream_http_x_scopes;
        auth_request_set $auth_method $upstream_http_x_auth_method;

        # Proxy to FastAPI service
        proxy_pass http://127.0.0.1:7860/api/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Forward validated auth context to FastAPI
        proxy_set_header X-User $auth_user;
        proxy_set_header X-Username $auth_username;
        proxy_set_header X-Client-Id $auth_client_id;
        proxy_set_header X-Scopes $auth_scopes;
        proxy_set_header X-Auth-Method $auth_method;

        # Pass through original Authorization header
        proxy_set_header Authorization $http_authorization;

        # Pass all request headers
        proxy_pass_request_headers on;

        # Timeouts
        proxy_connect_timeout 10s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }"""

                unprotected_api_block = """    # API endpoints - FastAPI handles authentication (session cookie / bearer)
    location {{ROOT_PATH}}/api/ {
        # Proxy to FastAPI service
        proxy_pass http://127.0.0.1:7860/api/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Pass through original Authorization header (if present)
        proxy_set_header Authorization $http_authorization;

        # Pass all request headers and cookies
        proxy_pass_request_headers on;

        # Timeouts
        proxy_connect_timeout 10s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }"""

                if protected_api_block in template_content:
                    template_content = template_content.replace(
                        protected_api_block, unprotected_api_block
                    )
                    logger.warning(
                        "NGINX_DISABLE_API_AUTH_REQUEST enabled: bypassing auth_request for /api/"
                    )
                else:
                    logger.warning(
                        "NGINX_DISABLE_API_AUTH_REQUEST enabled but could not find /api/ auth_request block in template"
                    )

            # Generate location blocks for enabled and healthy servers with transport support
            # In registry-only mode, skip MCP server location blocks (use empty list)
            location_blocks = []
            if settings.nginx_updates_enabled:
                # Get health service to check server health
                from ..health.service import health_service

                for path, server_info in servers.items():
                    proxy_pass_url = server_info.get("proxy_pass_url")
                    if proxy_pass_url:
                        # Check if server is healthy (including auth-expired which is still reachable)
                        health_status = health_service.server_health_status.get(
                            path, HealthStatus.UNKNOWN
                        )

                        # Include servers that are healthy or just have expired auth (server is up)
                        if HealthStatus.is_healthy(health_status):
                            # Generate transport-aware location blocks
                            transport_blocks = self._generate_transport_location_blocks(
                                path, server_info
                            )
                            location_blocks.extend(transport_blocks)
                            logger.debug(f"Added location blocks for healthy service: {path}")
                        else:
                            # Add commented out block for unhealthy services
                            commented_block = f"""
#    location {{{{ROOT_PATH}}}}{path}/ {{
#        # Service currently unhealthy (status: {health_status})
#        # Proxy to MCP server
#        proxy_pass {proxy_pass_url};
#        proxy_http_version 1.1;
#        proxy_set_header Host $host;
#        proxy_set_header X-Real-IP $remote_addr;
#        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
#        proxy_set_header X-Forwarded-Proto $scheme;
#    }}"""
                            location_blocks.append(commented_block)
                            logger.debug(
                                f"Added commented location block for unhealthy service {path} (status: {health_status})"
                            )
            else:
                logger.info(
                    "Registry-only mode: generating base nginx config without MCP server location blocks"
                )

            # Fetch additional server names (custom domains/IPs)
            additional_server_names = await self.get_additional_server_names()

            # Get API version from constants
            api_version = REGISTRY_CONSTANTS.ANTHROPIC_API_VERSION

            # Parse Keycloak configuration from KEYCLOAK_URL environment variable
            import os

            auth_provider = os.environ.get("AUTH_PROVIDER", "keycloak").lower()

            # Strip Keycloak location blocks from nginx config when not using Keycloak
            if auth_provider != "keycloak":
                template_content = re.sub(
                    r"    # \{\{KEYCLOAK_LOCATIONS_START\}\}.*?# \{\{KEYCLOAK_LOCATIONS_END\}\}\n?",
                    "",
                    template_content,
                    flags=re.DOTALL,
                )
                logger.info(
                    f"AUTH_PROVIDER is '{auth_provider}', removed Keycloak location blocks from nginx config"
                )
                keycloak_scheme = "http"
                keycloak_host = "keycloak"
                keycloak_port = "8080"
            else:
                keycloak_url = os.environ.get("KEYCLOAK_URL", "http://keycloak:8080")
                try:
                    parsed_keycloak = urlparse(keycloak_url)
                    keycloak_scheme = parsed_keycloak.scheme or "http"
                    keycloak_host = parsed_keycloak.hostname or "keycloak"
                    # Use default port based on scheme if not specified
                    if parsed_keycloak.port:
                        keycloak_port = str(parsed_keycloak.port)
                    else:
                        keycloak_port = "443" if keycloak_scheme == "https" else "8080"

                    # Validate that we can actually resolve the hostname
                    if not keycloak_host or keycloak_host == "keycloak":
                        # If we end up with just 'keycloak', use the full URL's netloc instead
                        keycloak_host = (
                            parsed_keycloak.netloc.split(":")[0]
                            if parsed_keycloak.netloc
                            else "keycloak"
                        )
                        logger.warning(
                            f"Keycloak hostname is 'keycloak', using netloc instead: {keycloak_host}"
                        )

                    logger.info(
                        f"Using Keycloak configuration from KEYCLOAK_URL '{keycloak_url}': "
                        f"{keycloak_scheme}://{keycloak_host}:{keycloak_port}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to parse KEYCLOAK_URL '{keycloak_url}': {e}. Using defaults."
                    )
                    keycloak_scheme = "http"
                    keycloak_host = "keycloak"
                    keycloak_port = "8080"

            # Generate version map for multi-version servers
            # In registry-only mode, skip version map generation (use empty string)
            if settings.nginx_updates_enabled:
                version_map = await self._generate_version_map(servers)
            else:
                version_map = ""

            # Replace placeholders in template
            config_content = template_content.replace("{{VERSION_MAP}}", version_map)
            config_content = config_content.replace(
                "{{LOCATION_BLOCKS}}", "\n".join(location_blocks)
            )
            config_content = config_content.replace(
                "{{ADDITIONAL_SERVER_NAMES}}", additional_server_names
            )
            config_content = config_content.replace("{{ANTHROPIC_API_VERSION}}", api_version)
            config_content = config_content.replace("{{KEYCLOAK_SCHEME}}", keycloak_scheme)
            config_content = config_content.replace("{{KEYCLOAK_HOST}}", keycloak_host)
            config_content = config_content.replace("{{KEYCLOAK_PORT}}", keycloak_port)

            # Generate registry-only block (503 response for MCP proxy requests)
            registry_only_block = self._generate_registry_only_block()
            config_content = config_content.replace("{{REGISTRY_ONLY_BLOCK}}", registry_only_block)

            # Generate virtual server blocks
            try:
                virtual_server_locations = await self._generate_virtual_server_blocks()

                # Get the virtual servers list for backend locations and mappings
                from registry.repositories.factory import get_virtual_server_repository

                virtual_repo = get_virtual_server_repository()
                virtual_servers = await virtual_repo.list_enabled()

                virtual_backend_locations = await self._generate_virtual_backend_locations(
                    virtual_servers
                )

                # Combine virtual server and backend location blocks
                virtual_blocks = virtual_server_locations
                if virtual_backend_locations:
                    virtual_blocks = (
                        virtual_blocks + "\n" + virtual_backend_locations
                        if virtual_blocks
                        else virtual_backend_locations
                    )

                config_content = config_content.replace("{{VIRTUAL_SERVER_BLOCKS}}", virtual_blocks)

                # Write mapping JSON files for Lua router
                await self._write_virtual_server_mappings(virtual_servers)

                logger.info(
                    f"Generated virtual server config with {len(virtual_servers)} virtual servers"
                )
            except Exception as e:
                logger.error(f"Failed to generate virtual server config: {e}", exc_info=True)
                config_content = config_content.replace("{{VIRTUAL_SERVER_BLOCKS}}", "")

            root_path = os.environ.get("ROOT_PATH", "").rstrip("/")
            config_content = config_content.replace("{{ROOT_PATH}}", root_path)

            # Write config file
            with open(settings.nginx_config_path, "w") as f:
                f.write(config_content)

            logger.info(
                f"Generated Nginx configuration with {len(location_blocks)} location blocks and additional server names: {additional_server_names}"
            )

            # Automatically reload nginx after generating config
            # Use force=True when generating base config to ensure nginx picks up changes
            self.reload_nginx(force=force_base_config)

            return True

        except Exception as e:
            logger.error(f"Failed to generate Nginx configuration: {e}", exc_info=True)
            return False

    def reload_nginx(self, force: bool = False) -> bool:
        """Reload Nginx configuration (if running in appropriate environment).

        Args:
            force: If True, reload even in registry-only mode (used after base config generation)

        In registry-only mode, skip reload unless force=True.
        """
        if not settings.nginx_updates_enabled and not force:
            logger.info(f"Skipping nginx reload - DEPLOYMENT_MODE={settings.deployment_mode.value}")
            NGINX_UPDATES_SKIPPED.labels(operation="reload").inc()
            return True

        try:
            import subprocess  # nosec B404

            # Test the configuration first before reloading
            test_result = subprocess.run(["nginx", "-t"], capture_output=True, text=True)
            if test_result.returncode != 0:
                logger.error(f"Nginx configuration test failed: {test_result.stderr}")
                logger.info("Skipping Nginx reload due to configuration errors")
                return False

            result = subprocess.run(["nginx", "-s", "reload"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Nginx configuration reloaded successfully")
                return True
            else:
                logger.error(f"Failed to reload Nginx: {result.stderr}")
                return False
        except FileNotFoundError:
            logger.warning("Nginx not found - skipping reload")
            return False
        except Exception as e:
            logger.error(f"Error reloading Nginx: {e}")
            return False

    def _generate_registry_only_block(self) -> str:
        """
        Generate nginx location block for registry-only mode.

        In registry-only mode, this block returns 503 for paths that look like
        MCP server requests (paths not matching known API prefixes).
        In with-gateway mode, this returns an empty string.

        Returns:
            Nginx location block string or empty string
        """
        if settings.nginx_updates_enabled:
            # with-gateway mode: no blocking needed, MCP servers are proxied
            return ""

        # registry-only mode: block MCP proxy requests with 503
        # This regex matches paths that don't start with known API prefixes
        block = """
    # Registry-only mode: block MCP proxy requests with 503
    # Matches paths that don't start with known API/auth prefixes
    location ~ ^{{ROOT_PATH}}/(?!api/|oauth2/|keycloak/|realms/|resources/|v0\\.1/|health|static/|assets/|_next/|validate).+ {
        default_type application/json;
        return 503 '{"error":"gateway_proxy_disabled","message":"Gateway proxy is disabled in registry-only mode. Connect directly to the MCP server using the proxy_pass_url from server registration.","deployment_mode":"registry-only","hint":"Use GET /api/servers/{path} to retrieve the proxy_pass_url for direct connection."}';
    }"""
        logger.info("Generated registry-only 503 block for MCP proxy requests")
        return block

    async def _generate_version_map(self, servers: dict[str, dict[str, Any]]) -> str:
        """
        Generate nginx map directive for version routing.

        Args:
            servers: Dictionary of server path -> server info

        Returns:
            Nginx map block as string, or empty string if no multi-version servers
        """
        from ..services.server_service import server_service

        map_entries = []

        for path, server_info in servers.items():
            # Check if this server has other versions via other_version_ids
            other_version_ids = server_info.get("other_version_ids", [])

            if not other_version_ids:
                # Single-version server - no map entry needed
                continue

            # Build versions list from active server and other versions
            versions = []

            # Add the current (active) version
            current_version = server_info.get("version", "v1.0.0")
            current_proxy_url = server_info.get("proxy_pass_url", "")
            if current_proxy_url:
                versions.append(
                    {
                        "version": current_version,
                        "proxy_pass_url": current_proxy_url,
                        "is_default": True,
                    }
                )

            # Add other versions by fetching their info
            for version_id in other_version_ids:
                version_info = await server_service.get_server_info(version_id)
                if version_info:
                    versions.append(
                        {
                            "version": version_info.get("version", "unknown"),
                            "proxy_pass_url": version_info.get("proxy_pass_url", ""),
                            "is_default": False,
                        }
                    )

            if len(versions) <= 1:
                # Only one version found, skip
                continue

            # Default backend is the active version's URL
            default_backend = current_proxy_url

            if not default_backend:
                logger.warning(f"No default backend found for {path}, skipping version map")
                continue

            # Escape path for nginx regex
            # Handle paths like /context7, /currenttime/, /ai.smithery-xxx
            escaped_path = re.escape(path.rstrip("/"))

            # Add map entries for this server
            # Entry for no header (empty string after colon)
            map_entries.append(f'    "~^{escaped_path}(/.*)?:$"            "{default_backend}";')
            # Entry for explicit "latest"
            map_entries.append(f'    "~^{escaped_path}(/.*)?:latest$"      "{default_backend}";')

            # Entry for each version
            for v in versions:
                version_str = v.get("version", "")
                backend_url = v.get("proxy_pass_url", "")
                if version_str and backend_url:
                    map_entries.append(
                        f'    "~^{escaped_path}(/.*)?:{re.escape(version_str)}$"  "{backend_url}";'
                    )

            logger.info(f"Generated version map entries for {path} with {len(versions)} versions")

        if not map_entries:
            return ""  # No multi-version servers configured

        return f"""# Version routing map (auto-generated)
# Routes requests based on X-MCP-Server-Version header
map "$uri:$http_x_mcp_server_version" $versioned_backend {{
    default "";

{chr(10).join(map_entries)}
}}

"""

    def _sanitize_path_for_location(
        self,
        path: str,
    ) -> str:
        """Sanitize a server path for use as an nginx internal location name.

        Replaces /, -, and . with underscores.

        Args:
            path: Server path (e.g., '/github')

        Returns:
            Sanitized string (e.g., '_github')
        """
        return re.sub(r"[/\-.]", "_", path)

    @staticmethod
    def _sanitize_for_nginx_comment(
        value: str,
    ) -> str:
        """Sanitize a string for safe interpolation into an nginx comment.

        Strips newlines and carriage returns to prevent header injection
        via multi-line nginx directives.

        Args:
            value: Raw string (e.g., server_name from user input)

        Returns:
            Sanitized single-line string
        """
        return re.sub(r"[\r\n]+", " ", value)

    @staticmethod
    def _sanitize_for_nginx_set(
        value: str,
    ) -> str:
        """Sanitize a string for safe use inside an nginx set directive's double quotes.

        Escapes double quotes and backslashes, and strips newlines.

        Args:
            value: Raw string (e.g., server_id from URL path)

        Returns:
            Escaped string safe for use in: set $var "value";
        """
        sanitized = re.sub(r"[\r\n]+", " ", value)
        sanitized = sanitized.replace("\\", "\\\\")
        sanitized = sanitized.replace('"', '\\"')
        return sanitized

    async def _generate_virtual_server_blocks(self) -> str:
        """Generate nginx location blocks for enabled virtual servers.

        Returns:
            Nginx configuration string with virtual server location blocks
        """
        try:
            from registry.repositories.factory import get_virtual_server_repository

            virtual_repo = get_virtual_server_repository()
            virtual_servers = await virtual_repo.list_enabled()

            if not virtual_servers:
                logger.info("No enabled virtual servers found")
                return ""

            location_blocks = []
            for vs in virtual_servers:
                # Extract server_id from path (e.g., '/virtual/dev-essentials' -> 'dev-essentials')
                server_id = vs.path.replace("/virtual/", "", 1)

                # Sanitize values for safe interpolation into nginx config
                safe_name = self._sanitize_for_nginx_comment(vs.server_name)
                safe_id = self._sanitize_for_nginx_set(server_id)

                block = f"""
    # Virtual MCP Server: {safe_name}
    location {{{{ROOT_PATH}}}}{vs.path} {{
        set $virtual_server_id "{safe_id}";
        auth_request /validate;
        auth_request_set $auth_scopes $upstream_http_x_scopes;
        auth_request_set $auth_user $upstream_http_x_user;
        auth_request_set $auth_username $upstream_http_x_username;
        auth_request_set $auth_method $upstream_http_x_auth_method;
        rewrite_by_lua_file /etc/nginx/lua/capture_body.lua;
        content_by_lua_file /etc/nginx/lua/virtual_router.lua;
    }}"""
                location_blocks.append(block)
                logger.debug(f"Generated virtual server location block for {vs.path}")

            logger.info(f"Generated {len(location_blocks)} virtual server location blocks")
            return "\n".join(location_blocks)

        except Exception as e:
            logger.error(f"Failed to generate virtual server blocks: {e}", exc_info=True)
            return ""

    async def _generate_virtual_backend_locations(
        self,
        virtual_servers: list,
    ) -> str:
        """Generate internal nginx location blocks for virtual server backends.

        Args:
            virtual_servers: List of VirtualServerConfig objects

        Returns:
            Nginx configuration string with internal backend location blocks
        """
        try:
            from registry.repositories.factory import get_server_repository

            server_repo = get_server_repository()

            # Collect unique backend server paths
            backend_paths = set()
            for vs in virtual_servers:
                for tm in vs.tool_mappings:
                    backend_paths.add(tm.backend_server_path)

            if not backend_paths:
                return ""

            location_blocks = []
            for backend_path in sorted(backend_paths):
                sanitized = self._sanitize_path_for_location(backend_path)
                server_info = await server_repo.get(backend_path)

                if not server_info:
                    logger.warning(
                        f"Backend server not found for virtual server mapping: {backend_path}"
                    )
                    continue

                proxy_pass_url = server_info.get("proxy_pass_url", "")
                if not proxy_pass_url:
                    logger.warning(f"No proxy_pass_url for backend server: {backend_path}")
                    continue

                # Determine upstream host from proxy_pass_url
                parsed_url = urlparse(proxy_pass_url)
                upstream_host = parsed_url.netloc

                # Build MCP endpoint URL from the server's mcp_endpoint or proxy_pass_url
                mcp_endpoint = server_info.get("mcp_endpoint", "")
                if mcp_endpoint:
                    mcp_parsed = urlparse(mcp_endpoint)
                    mcp_path = mcp_parsed.path.rstrip("/")
                    # Construct full MCP URL from proxy_pass host + mcp path
                    mcp_proxy_url = f"{parsed_url.scheme}://{parsed_url.netloc}{mcp_path}"
                else:
                    # Fallback: use proxy_pass_url, appending /mcp only if needed
                    bare_url = proxy_pass_url.rstrip("/")
                    # Check if URL already ends with common MCP endpoint paths
                    if bare_url.endswith("/mcp") or bare_url.endswith("/sse"):
                        mcp_proxy_url = bare_url
                    else:
                        mcp_proxy_url = f"{bare_url}/mcp"

                # Use regular internal location (not named @) so proxy_pass
                # can include a URI path for the MCP endpoint
                location_path = f"/_vs_backend{sanitized}"

                block = f"""
    location {location_path} {{
        internal;
        proxy_pass {mcp_proxy_url};
        proxy_http_version 1.1;
        proxy_ssl_server_name on;
        proxy_set_header Host {upstream_host};
        proxy_set_header Authorization $http_authorization;
        proxy_buffering off;
        proxy_set_header Accept "application/json, text/event-stream";
        proxy_set_header Content-Type $content_type;
    }}"""
                location_blocks.append(block)
                logger.debug(
                    f"Generated virtual backend location for {backend_path} -> {location_path}"
                )

            logger.info(f"Generated {len(location_blocks)} virtual backend location blocks")
            return "\n".join(location_blocks)

        except Exception as e:
            logger.error(f"Failed to generate virtual backend locations: {e}", exc_info=True)
            return ""

    async def _write_virtual_server_mappings(
        self,
        virtual_servers: list,
    ) -> None:
        """Write pre-computed mapping JSON files for each virtual server.

        These files are consumed by virtual_router.lua at request time.

        Args:
            virtual_servers: List of VirtualServerConfig objects
        """
        try:
            from registry.repositories.factory import get_server_repository

            server_repo = get_server_repository()

            mappings_dir = Path("/etc/nginx/lua/virtual_mappings")
            mappings_dir.mkdir(parents=True, exist_ok=True)

            for vs in virtual_servers:
                server_id = vs.path.replace("/virtual/", "", 1)

                # Build scope override lookup
                scope_overrides = {}
                for override in vs.tool_scope_overrides:
                    scope_overrides[override.tool_alias] = override.required_scopes

                tools = []
                tool_backend_map = {}

                for tm in vs.tool_mappings:
                    sanitized_backend = self._sanitize_path_for_location(tm.backend_server_path)
                    backend_location = f"/_vs_backend{sanitized_backend}"
                    tool_display_name = tm.alias if tm.alias else tm.tool_name

                    # Get tool metadata from the backend server
                    server_info = await server_repo.get(tm.backend_server_path)
                    description = tm.description_override or ""
                    input_schema: dict[str, Any] = {}

                    if server_info:
                        server_tools = server_info.get("tool_list", [])
                        for st in server_tools:
                            if st.get("name") == tm.tool_name:
                                description = tm.description_override or st.get("description", "")
                                input_schema = st.get("inputSchema", st.get("input_schema", {}))
                                break

                    input_schema = _ensure_mcp_compliant_schema(input_schema)

                    # Per-tool scopes
                    required_scopes = scope_overrides.get(tool_display_name, [])

                    tools.append(
                        {
                            "name": tool_display_name,
                            "original_name": tm.tool_name,
                            "description": description,
                            "inputSchema": input_schema,
                            "backend_location": backend_location,
                            "backend_version": tm.backend_version,
                            "required_scopes": required_scopes,
                        }
                    )

                    tool_backend_map[tool_display_name] = {
                        "backend_location": backend_location,
                        "original_name": tm.tool_name,
                        "backend_version": tm.backend_version,
                    }

                mapping_data = {
                    "server_name": vs.server_name,
                    "required_scopes": vs.required_scopes,
                    "tools": tools,
                    "tool_backend_map": tool_backend_map,
                }

                mapping_path = mappings_dir / f"{server_id}.json"
                with open(mapping_path, "w") as f:
                    json.dump(mapping_data, f, indent=2, default=str)

                logger.debug(f"Wrote virtual server mapping: {mapping_path}")

            logger.info(f"Wrote {len(virtual_servers)} virtual server mapping files")

        except Exception as e:
            logger.error(f"Failed to write virtual server mappings: {e}", exc_info=True)

    def _generate_transport_location_blocks(self, path: str, server_info: dict[str, Any]) -> list:
        """Generate nginx location blocks for different transport types."""
        blocks = []
        proxy_pass_url = server_info.get("proxy_pass_url", "")
        supported_transports = server_info.get("supported_transports", ["streamable-http"])

        # Use the proxy_pass_url exactly as specified in the JSON file
        # Users are responsible for including /mcp, /sse, or any other path in the URL
        proxy_url = proxy_pass_url

        # Determine transport type based on supported_transports
        if not supported_transports:
            # Default to streamable-http if no transports specified
            transport_type = "streamable-http"
            logger.info(
                f"Server {path}: No supported_transports specified, defaulting to streamable-http"
            )
        elif "streamable-http" in supported_transports and "sse" in supported_transports:
            # If both are supported, prefer streamable-http
            transport_type = "streamable-http"
            logger.info(
                f"Server {path}: Both streamable-http and sse supported, preferring streamable-http"
            )
        elif "sse" in supported_transports:
            # SSE only
            transport_type = "sse"
            logger.info(f"Server {path}: Only sse transport supported, using sse")
        elif "streamable-http" in supported_transports:
            # Streamable-http only
            transport_type = "streamable-http"
            logger.info(
                f"Server {path}: Only streamable-http transport supported, using streamable-http"
            )
        else:
            # Default to streamable-http if unknown transport
            transport_type = "streamable-http"
            logger.info(
                f"Server {path}: Unknown transport types {supported_transports}, defaulting to streamable-http"
            )

        # Create a single location block for this server
        # The proxy_pass URL is used exactly as provided in the server configuration
        logger.info(f"Server {path}: Using proxy_pass URL as configured: {proxy_url}")

        block = self._create_location_block(path, proxy_url, transport_type, server_info)
        blocks.append(block)

        return blocks

    def _create_location_block(
        self,
        path: str,
        proxy_pass_url: str,
        transport_type: str,
        server_info: dict[str, Any] | None = None,
    ) -> str:
        """Create a single nginx location block with transport-specific configuration.

        Args:
            path: Server location path
            proxy_pass_url: Default backend URL
            transport_type: Transport type (streamable-http, sse, direct)
            server_info: Full server info dict (for version support)

        Returns:
            Nginx location block as string
        """
        # Check if this server has multiple versions
        # The MongoDB document stores linked version IDs in "other_version_ids"
        has_versions = False
        if server_info:
            other_version_ids = server_info.get("other_version_ids", [])
            has_versions = len(other_version_ids) > 0

        # Extract hostname from proxy_pass_url for external services
        parsed_url = urlparse(proxy_pass_url)
        upstream_host = parsed_url.netloc

        # Determine whether to use upstream hostname or preserve original host
        # For external services (https), use the upstream hostname
        # For internal services (http without dots in hostname), preserve original host
        if parsed_url.scheme == "https" or "." in upstream_host:
            # External service - use upstream hostname
            host_header = upstream_host
            logger.info(f"Using upstream hostname for Host header: {host_header}")
        else:
            # Internal service - preserve original host
            host_header = "$host"
            logger.info("Using original host for Host header: $host")

        # Generate proxy_pass directive based on version routing
        if has_versions:
            # Multi-version server: use map variable with fallback
            proxy_directive = f'''
        # Version routing - use header-based backend selection
        # If X-MCP-Server-Version header matches a version, use that backend
        # Otherwise, use the default backend
        set $backend_url "{proxy_pass_url}";
        if ($versioned_backend != "") {{
            set $backend_url $versioned_backend;
        }}

        # Proxy to selected backend
        proxy_pass $backend_url;'''
            version_headers = """

        # Add version info to response
        add_header X-MCP-Version-Routing "enabled" always;"""
        else:
            # Single-version server: direct proxy_pass (existing behavior)
            proxy_directive = f"""
        # Proxy to MCP server
        proxy_pass {proxy_pass_url};"""
            version_headers = ""

        # Common proxy settings
        common_settings = f"""
        # Use IPv4 resolver (disable IPv6)
        resolver 8.8.8.8 8.8.4.4 valid=10s;
        resolver_timeout 5s;

        # Authenticate request - pass entire request to auth server
        auth_request /validate;

        # Capture auth server response headers for forwarding
        auth_request_set $auth_user $upstream_http_x_user;
        auth_request_set $auth_username $upstream_http_x_username;
        auth_request_set $auth_client_id $upstream_http_x_client_id;
        auth_request_set $auth_scopes $upstream_http_x_scopes;
        auth_request_set $auth_method $upstream_http_x_auth_method;
        auth_request_set $auth_server_name $upstream_http_x_server_name;
        auth_request_set $auth_tool_name $upstream_http_x_tool_name;
{proxy_directive}
        proxy_http_version 1.1;
        proxy_ssl_server_name on;
        proxy_set_header Host {host_header};
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Add original URL for auth server scope validation
        proxy_set_header X-Original-URL $scheme://$host$request_uri;

        # Pass through the original authentication headers
        proxy_set_header Authorization $http_authorization;
        proxy_set_header X-Authorization $http_x_authorization;
        proxy_set_header X-User-Pool-Id $http_x_user_pool_id;
        proxy_set_header X-Client-Id $http_x_client_id;
        proxy_set_header X-Region $http_x_region;


        # Forward auth server response headers to backend
        proxy_set_header X-User $auth_user;
        proxy_set_header X-Username $auth_username;
        proxy_set_header X-Client-Id-Auth $auth_client_id;
        proxy_set_header X-Scopes $auth_scopes;
        proxy_set_header X-Auth-Method $auth_method;
        proxy_set_header X-Server-Name $auth_server_name;
        proxy_set_header X-Tool-Name $auth_tool_name;

        # Pass all original client headers
        proxy_pass_request_headers on;

        # Handle auth errors
        error_page 401 = @auth_error;
        error_page 403 = @forbidden_error;{version_headers}"""

        # Transport-specific settings
        if transport_type == "sse":
            transport_settings = """
        # Capture request body for auth validation using Lua
        rewrite_by_lua_file /etc/nginx/lua/capture_body.lua;
        log_by_lua_file /etc/nginx/lua/emit_metrics.lua;

        # For SSE connections and WebSocket upgrades
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection $http_connection;
        proxy_set_header Upgrade $http_upgrade;
        # Explicitly preserve Accept header for MCP protocol requirements
        proxy_set_header Accept $http_accept;
        chunked_transfer_encoding off;"""

        elif transport_type == "streamable-http":
            transport_settings = """
        # Capture request body for auth validation using Lua
        rewrite_by_lua_file /etc/nginx/lua/capture_body.lua;
        log_by_lua_file /etc/nginx/lua/emit_metrics.lua;

        # HTTP transport configuration
        proxy_buffering off;
        proxy_set_header Connection "";
        # Explicitly preserve Accept header for MCP protocol requirements
        proxy_set_header Accept $http_accept;"""

        else:  # direct
            transport_settings = """
        # Capture request body for auth validation using Lua
        rewrite_by_lua_file /etc/nginx/lua/capture_body.lua;
        log_by_lua_file /etc/nginx/lua/emit_metrics.lua;

        # Generic transport configuration
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection $http_connection;
        proxy_set_header Upgrade $http_upgrade;
        chunked_transfer_encoding off;"""

        # Use the location path exactly as specified in the server configuration
        # Users have full control over the location path format (with or without trailing slash)
        location_path = path
        logger.info(f"Creating location block for {location_path} with {transport_type} transport")

        return f"""
    location {{{{ROOT_PATH}}}}{location_path} {{{transport_settings}{common_settings}
    }}"""


# Global nginx service instance
nginx_service = NginxConfigService()
