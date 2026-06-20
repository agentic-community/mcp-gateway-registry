import asyncio
import hashlib
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from registry.constants import REGISTRY_CONSTANTS, DeploymentType, HealthStatus

from .config import settings
from .metrics import NGINX_CONFIG_WRITES, NGINX_UPDATES_SKIPPED

logger = logging.getLogger(__name__)


# Default mode applied to a fresh nginx config file when no destination
# exists yet. Subsequent writes preserve whatever mode the destination
# currently has so an operator's chmod isn't silently reverted.
DEFAULT_NGINX_CONFIG_MODE: int = 0o644


def _atomic_write_text(
    path: Path,
    content: str,
) -> None:
    """Write content to path atomically (issue #1044).

    Writes to a temporary file in the same directory as ``path`` and uses
    ``os.replace()`` to swap it into place. ``os.replace()`` is atomic on POSIX
    when source and destination are on the same filesystem, so any reader
    (including ``nginx -t``) sees either the old config or the new one - never
    a truncated mid-write file.

    The temp file's mode is set to match the destination's existing mode, or
    ``DEFAULT_NGINX_CONFIG_MODE`` when the destination does not yet exist.
    Without this, ``tempfile.NamedTemporaryFile`` defaults to ``0o600`` and
    silently tightens permissions across atomic writes.

    On any failure the temp file is removed and the destination is left
    unchanged. ``NGINX_CONFIG_WRITES`` is incremented with
    ``status="success"`` or ``status="failure"`` accordingly.

    Args:
        path: Destination path.
        content: Text content to write.

    Raises:
        OSError: If the temp file cannot be created, written, or replaced.
    """
    dest_dir = path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)

    if path.exists():
        target_mode = path.stat().st_mode & 0o777
    else:
        target_mode = DEFAULT_NGINX_CONFIG_MODE

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        dir=dest_dir,
        prefix=f".{path.name}.tmp.",
        delete=False,
        encoding="utf-8",
    )
    tmp_path = Path(tmp.name)
    try:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.chmod(tmp_path, target_mode)
        os.replace(tmp_path, path)
        NGINX_CONFIG_WRITES.labels(status="success").inc()
    except Exception:
        NGINX_CONFIG_WRITES.labels(status="failure").inc()
        try:
            tmp.close()
        except Exception:
            pass
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _cleanup_stale_temp_files(config_path: Path) -> None:
    """Remove leftover ``.{config_name}.tmp.*`` files from a crashed write.

    On a clean write, the temp file is renamed away by ``os.replace()``. If the
    process was killed mid-write (SIGKILL, OOM kill, host reboot), a temp file
    matching ``.{config_name}.tmp.*`` can remain. Container restarts on
    ECS/EKS typically clean this up naturally, but long-lived hosts (local
    dev, EC2 without container ephemerality) need explicit cleanup.

    Best-effort: logs warnings but does not raise on failure.
    """
    dest_dir = config_path.parent
    pattern = f".{config_path.name}.tmp.*"
    try:
        leftovers = list(dest_dir.glob(pattern))
    except OSError as e:
        logger.warning(f"Could not scan {dest_dir} for stale temp files: {e}")
        return

    for stale in leftovers:
        try:
            stale.unlink()
            logger.info(f"Removed stale nginx config temp file: {stale}")
        except OSError as e:
            logger.warning(f"Failed to remove stale temp file {stale}: {e}")


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
        # Contract: every call site that invokes generate_config_async() or
        # reload_nginx() (directly or transitively) MUST acquire this lock for
        # the duration of those calls. The lock prevents:
        #   1. Two writers racing on the nginx config path (lost-update).
        #   2. nginx -t in one writer reading a partial file written by another.
        #   3. Two `nginx -s reload` signals in flight simultaneously.
        # The lock is intentionally coarse-grained because regen + reload is
        # bounded (~150-300ms) and infrequent (tens per minute). See issue
        # #1044 and .scratchpad/issue-1044/lld.md for the full rationale.
        self.reload_lock: asyncio.Lock = asyncio.Lock()

        # Cache for get_additional_server_names (avoids hitting metadata
        # endpoints on every scheduler tick). Invalidated by mark_dirty().
        self._cached_server_names: str | None = None

        # Minimum interval between nginx reload signals. Prevents cascading
        # SIGHUP when many flush_now() calls land in rapid succession (e.g.
        # bulk toggle during stress tests). nginx needs time for worker
        # processes to shut down before accepting another reload.
        self._min_reload_interval_seconds: float = 3.0
        self._last_reload_time: float = 0.0

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
            result = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=2.0)  # nosec B603 B607 - hardcoded command
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

    async def render_config(
        self,
        servers: dict[str, dict[str, Any]],
    ) -> str | None:
        """Render the nginx config string without writing to disk or reloading.

        Returns the rendered config text, or None if nginx updates are disabled.
        Used by NginxReloadScheduler for hash-based change detection.
        """
        if not settings.nginx_updates_enabled:
            return None
        return await self._render_config_impl(servers)

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
            config_content = await self._render_config_impl(servers)
            if config_content is None:
                return False

            # Write virtual server Lua mapping files (side effect, not part of render)
            await self._commit_virtual_server_mappings()

            _atomic_write_text(settings.nginx_config_path, config_content)

            logger.info(
                f"Generated Nginx configuration with location blocks "
                f"and additional server names"
            )

            await asyncio.to_thread(self.reload_nginx, force_base_config)
            return True

        except Exception as e:
            logger.error(f"Failed to generate Nginx configuration: {e}", exc_info=True)
            return False

    async def _render_config_impl(
        self,
        servers: dict[str, dict[str, Any]],
    ) -> str | None:
        """Internal: render the full nginx config content string.

        Returns the rendered string, or None if the template is missing.
        """
        try:
            # Read template
            if not self.nginx_template_path.exists():
                logger.warning(f"Nginx template not found at {self.nginx_template_path}")
                return None

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
        # Mark this as a registry-API request (rewrite phase) so the shared
        # /validate subrequest forwards X-Registry-Api-Auth and /validate mints
        # the registry-UI internal token. Set inside the location (NOT a
        # server-scope default), which the auth_request subrequest would clobber.
        set $registry_api_auth "1";

        # Authenticate request via auth server (validates JWT Bearer tokens)
        auth_request /validate;

        # Capture auth server response headers
        auth_request_set $auth_user $upstream_http_x_user;
        auth_request_set $auth_username $upstream_http_x_username;
        auth_request_set $auth_client_id $upstream_http_x_client_id;
        auth_request_set $auth_scopes $upstream_http_x_scopes;
        auth_request_set $auth_method $upstream_http_x_auth_method;
        # Capture the /validate-minted registry-UI token (binds verified identity).
        # The registry verifies this instead of trusting the forgeable X-* headers.
        auth_request_set $auth_internal_token_registry $upstream_http_x_internal_token_registry;

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
        # The internal token the registry verifies (it ignores the X-* headers above).
        proxy_set_header X-Internal-Token-Registry $auth_internal_token_registry;

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
                    # Local servers don't get nginx routes
                    if server_info.get("deployment") == DeploymentType.LOCAL:
                        logger.debug(f"Skipping local server {path} from nginx config")
                        continue
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

            # Fetch additional server names (cached to avoid per-tick metadata calls)
            if self._cached_server_names is None:
                self._cached_server_names = await self.get_additional_server_names()
            additional_server_names = self._cached_server_names

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

            # Strip PingFederate location blocks from nginx config when not using PingFederate
            if auth_provider != "pingfederate":
                template_content = re.sub(
                    r"    # \{\{PINGFEDERATE_LOCATIONS_START\}\}.*?# \{\{PINGFEDERATE_LOCATIONS_END\}\}\n?",
                    "",
                    template_content,
                    flags=re.DOTALL,
                )
                logger.info(
                    f"AUTH_PROVIDER is '{auth_provider}', removed PingFederate location blocks from nginx config"
                )

            # Parse Keycloak configuration from KEYCLOAK_URL environment variable.
            # This always runs so the Keycloak template placeholders are filled even
            # when another provider is active (the location blocks are stripped above).
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

            # Parse PingFederate configuration, falling back to defaults on any error
            # so a malformed PINGFEDERATE_BASE_URL never breaks config generation.
            pingfederate_url = os.environ.get("PINGFEDERATE_BASE_URL", "http://pingfederate:9032")
            try:
                pf_parsed = urlparse(pingfederate_url)
                pf_scheme = pf_parsed.scheme or "http"
                pf_host = pf_parsed.hostname or "pingfederate"
                pf_port = str(pf_parsed.port or ("443" if pf_scheme == "https" else "9032"))
            except Exception as e:
                logger.warning(
                    f"Failed to parse PINGFEDERATE_BASE_URL '{pingfederate_url}': {e}. Using defaults."
                )
                pf_scheme = "http"
                pf_host = "pingfederate"
                pf_port = "9032"
            config_content = config_content.replace("{{PINGFEDERATE_SCHEME}}", pf_scheme)
            config_content = config_content.replace("{{PINGFEDERATE_HOST}}", pf_host)
            config_content = config_content.replace("{{PINGFEDERATE_PORT}}", pf_port)

            # Parse AUTH_SERVER_URL so nginx templates can reference the
            # auth-server by its actual hostname/FQDN instead of the
            # hard-coded Docker-Compose service name (#553).  Follows the
            # same pattern used for Keycloak and PingFederate above.
            auth_server_url = os.environ.get("AUTH_SERVER_URL", "http://auth-server:8888")
            try:
                parsed_auth = urlparse(auth_server_url)
                auth_host = parsed_auth.hostname or "auth-server"
                if parsed_auth.port:
                    auth_port = str(parsed_auth.port)
                else:
                    auth_scheme = parsed_auth.scheme or "http"
                    auth_port = "443" if auth_scheme == "https" else "8888"

                logger.info(
                    f"Using auth-server configuration from AUTH_SERVER_URL "
                    f"'{auth_server_url}': {auth_host}:{auth_port}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to parse AUTH_SERVER_URL '{auth_server_url}': {e}. "
                    "Using defaults."
                )
                auth_host = "auth-server"
                auth_port = "8888"
            config_content = config_content.replace("{{AUTH_SERVER_HOST}}", auth_host)
            config_content = config_content.replace("{{AUTH_SERVER_PORT}}", auth_port)

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

                logger.info(
                    f"Generated virtual server config with {len(virtual_servers)} virtual servers"
                )
            except Exception as e:
                logger.error(f"Failed to generate virtual server config: {e}", exc_info=True)
                config_content = config_content.replace("{{VIRTUAL_SERVER_BLOCKS}}", "")

            root_path = os.environ.get("ROOT_PATH", "").rstrip("/")
            config_content = config_content.replace("{{ROOT_PATH}}", root_path)

            # MCP 2025-06-18 / RFC 9728 §5.1: WWW-Authenticate on auth-failure 401s
            # must point at the gateway's PRM endpoint. The URL must match the
            # `resource` field returned by /.well-known/oauth-protected-resource
            # byte-for-byte.
            try:
                from registry.auth.oauth_metadata import (
                    build_canonical_resource_url,
                    build_resource_metadata_url,
                )

                resource_metadata_url = build_resource_metadata_url(
                    build_canonical_resource_url(settings.registry_url)
                )
            except ValueError as exc:
                logger.warning(
                    f"Could not derive MCP_RESOURCE_METADATA_URL "
                    f"(registry_url={settings.registry_url!r}): {exc}. "
                    "Substituting empty value; clients will not see WWW-Authenticate."
                )
                resource_metadata_url = ""
            config_content = config_content.replace(
                "{{MCP_RESOURCE_METADATA_URL}}", resource_metadata_url
            )

            return config_content

        except Exception as e:
            logger.error(f"Failed to render Nginx configuration: {e}", exc_info=True)
            return None

    async def _commit_virtual_server_mappings(self) -> None:
        """Write Lua mapping JSON files for virtual servers.

        Separated from _render_config_impl so that rendering is pure (no disk
        side effects) and mappings are only written when config actually changes.
        """
        try:
            from registry.repositories.factory import get_virtual_server_repository

            virtual_repo = get_virtual_server_repository()
            virtual_servers = await virtual_repo.list_enabled()
            await self._write_virtual_server_mappings(virtual_servers)
        except Exception as e:
            logger.error(f"Failed to write virtual server mappings: {e}")

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

        # Rate-limit reload signals. nginx needs time for worker processes to
        # gracefully shut down before accepting another SIGHUP. Without this
        # guard, rapid-fire flush_now() calls (e.g. bulk toggle) can spawn
        # multiple master processes and leave workers in "shutting down" limbo.
        import time as _time

        now = _time.monotonic()
        elapsed = now - self._last_reload_time
        if elapsed < self._min_reload_interval_seconds and not force:
            logger.debug(
                "Skipping nginx reload (%.1fs since last, min interval %.1fs)",
                elapsed,
                self._min_reload_interval_seconds,
            )
            return False

        try:
            import subprocess  # nosec B404

            # Test the configuration first before reloading
            test_result = subprocess.run(["nginx", "-t"], capture_output=True, text=True, timeout=5)  # nosec B603 B607 - hardcoded command
            if test_result.returncode != 0:
                logger.error(f"Nginx configuration test failed: {test_result.stderr}")
                logger.info("Skipping Nginx reload due to configuration errors")
                return False

            result = subprocess.run(["nginx", "-s", "reload"], capture_output=True, text=True, timeout=5)  # nosec B603 B607 - hardcoded command
            if result.returncode == 0:
                self._last_reload_time = _time.monotonic()
                logger.info("Nginx configuration reloaded successfully")
                return True
            # On Fargate the registry container starts uvicorn before nginx
            # (the entrypoint waits for the runtime nginx config to be
            # generated by uvicorn before starting nginx). When the demo
            # servers register during uvicorn startup and call reload_nginx(),
            # nginx is not running yet — the pid file is empty and nginx -s
            # reload exits non-zero with "invalid PID number". The reload is
            # idempotent, so retry briefly to give the entrypoint time to
            # start nginx. Without this, server location blocks are written
            # to disk but never made active until the next reload (which may
            # never come for auto-registered demo servers).
            stderr = result.stderr or ""
            if "invalid PID number" in stderr or ("open()" in stderr and "nginx.pid" in stderr):
                logger.warning(
                    "Nginx not yet started (pid file empty); will retry reload"
                )
                for attempt in range(10):
                    _time.sleep(1.0)
                    retry = subprocess.run(["nginx", "-s", "reload"], capture_output=True, text=True, timeout=5)  # nosec B603 B607 - hardcoded command
                    if retry.returncode == 0:
                        self._last_reload_time = _time.monotonic()
                        logger.info(
                            "Nginx configuration reloaded successfully after %d retry attempts",
                            attempt + 1,
                        )
                        return True
                logger.error(
                    "Nginx still not running after 10 retries; reload abandoned"
                )
                return False
            logger.error(f"Failed to reload Nginx: {stderr}")
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
    def _is_host_resolvable_at_startup(
        hostname: str,
    ) -> bool:
        """Decide whether an upstream host is safe to resolve at nginx config load.

        Nginx resolves literal proxy_pass hosts when it loads the config and
        fails to start ("host not found in upstream") if any cannot be resolved.
        A fully-qualified domain name (contains a dot, e.g. "api.github.com") or
        an IP address is expected to resolve in any environment, so it is safe to
        emit as a literal proxy_pass (resolved once, no per-request DNS cost).

        A bare hostname with no dot (e.g. a docker-compose service name like
        "currenttime-server") only resolves inside the environment that defines
        it. Treat it as NOT safe so the caller defers resolution to request time.

        Args:
            hostname: The upstream hostname (no scheme or port), may be empty.

        Returns:
            True if the host can be safely resolved at config load, else False.
        """
        if not hostname:
            return False
        # A dot indicates an FQDN or an IPv4 literal; both resolve everywhere.
        # IPv6 literals contain colons and are also always resolvable.
        return "." in hostname or ":" in hostname

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

        # Route 401s through @auth_error so the WWW-Authenticate header
        # mandated by RFC 9728 §5.1 is emitted (issue #989).
        error_page 401 = @auth_error;
        error_page 403 = @forbidden_error;
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

                # Decide how to emit proxy_pass based on whether the backend host
                # is safe to resolve at config-load time.
                #
                # Normal external hosts (with a dot, like api.github.com, or an IP)
                # use a literal proxy_pass: nginx resolves them once at startup and
                # caches for the worker's life, so there is no per-request DNS cost.
                #
                # Bare hostnames (no dot, e.g. a docker-compose service name like
                # "currenttime-server") are NOT resolvable in every environment.
                # A literal proxy_pass to such a host makes nginx fail to start with
                # "host not found in upstream", crashing the whole registry
                # container. For those we pass the URL through a variable plus a
                # resolver so nginx resolves at request time instead; an
                # unresolvable backend then degrades to a per-request 502 for only
                # that backend rather than taking the gateway down at boot.
                backend_hostname = parsed_url.hostname or ""
                host_is_resolvable_at_startup = self._is_host_resolvable_at_startup(
                    backend_hostname
                )

                if host_is_resolvable_at_startup:
                    proxy_directive = f"proxy_pass {mcp_proxy_url};"
                else:
                    # sanitized is already underscore-safe (valid nginx var name).
                    backend_var = f"$vs_backend{sanitized}"
                    dns_resolver = os.environ.get("NGINX_DNS_RESOLVER", "8.8.8.8 8.8.4.4")
                    dns_resolver_timeout = os.environ.get("NGINX_DNS_RESOLVER_TIMEOUT", "5")
                    proxy_directive = (
                        f"resolver {dns_resolver} valid=10s;\n"
                        f"        resolver_timeout {dns_resolver_timeout}s;\n"
                        f'        set {backend_var} "{mcp_proxy_url}";\n'
                        f"        proxy_pass {backend_var};"
                    )

                block = f"""
    location {location_path} {{
        internal;
        {proxy_directive}
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

        # Issue #1026 - route MCP traffic through auth_server proxy for tools/list filtering.
        # All MCP POSTs go to auth-server:8888/mcp-proxy/{server_name} instead of the upstream
        # directly. auth_server forwards the request to the original upstream (passed via
        # X-Upstream-Url) and filters `tools/list` responses when MCP_TOOLS_LIST_FILTER_ENABLED
        # is set. All other JSON-RPC methods are passed through unchanged, so the only latency
        # impact is the extra hop. Nginx never inspects the body or flag; auth_server decides.
        # We use the header strategy (X-Upstream-Url) so auth_server does not need a separate
        # MongoDB lookup per request, and version-aware upstream selection stays in nginx.
        mcp_proxy_target = f"{settings.auth_server_url}/mcp-proxy/" + path.strip("/") + "/"
        if has_versions:
            # Multi-version server: use map variable with fallback, then proxy the selected
            # upstream URL to auth_server via X-Upstream-Url so it knows where to forward.
            proxy_directive = f'''
        # Version routing - use header-based backend selection
        # If X-MCP-Server-Version header matches a version, use that backend
        # Otherwise, use the default backend
        set $backend_url "{proxy_pass_url}";
        if ($versioned_backend != "") {{
            set $backend_url $versioned_backend;
        }}

        # Tell auth_server which upstream to forward to after filtering
        proxy_set_header X-Upstream-Url $backend_url;

        # Proxy to auth_server mcp-proxy hop (Issue #1026)
        proxy_pass {mcp_proxy_target};'''
            version_headers = """

        # Add version info to response
        add_header X-MCP-Version-Routing "enabled" always;"""
        else:
            # Single-version server: forward the fixed upstream via X-Upstream-Url header.
            # Set $backend_url (in the rewrite phase) so the /validate subrequest can
            # bind it into the internal token via X-Resolved-Upstream, matching what
            # is forwarded here. Quote the URL so nginx does not interpret braces.
            proxy_directive = f"""
        set $backend_url "{proxy_pass_url}";

        # Tell auth_server which upstream to forward to after filtering
        proxy_set_header X-Upstream-Url $backend_url;

        # Proxy to auth_server mcp-proxy hop (Issue #1026)
        proxy_pass {mcp_proxy_target};"""
            version_headers = ""

        # Common proxy settings
        common_settings = f"""
        # DNS resolver for dynamic proxy_pass upstreams.
        # Default: 8.8.8.8 8.8.4.4 (public DNS).
        # Override with NGINX_DNS_RESOLVER env var for environments where
        # backend servers use internal hostnames (e.g., Kubernetes
        # cluster-local names like *.svc.cluster.local need kube-dns).
        resolver {os.environ.get("NGINX_DNS_RESOLVER", "8.8.8.8 8.8.4.4")} valid=10s;
        resolver_timeout {os.environ.get("NGINX_DNS_RESOLVER_TIMEOUT", "5")}s;

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
        # Capture the /validate-minted internal JWT (binds identity/scopes/upstream).
        # mcp_proxy verifies this instead of trusting the forgeable X-* headers below.
        auth_request_set $auth_internal_token $upstream_http_x_internal_token;
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

        # Forward MCP session ID for streamable-http transport
        proxy_set_header Mcp-Session-Id $http_mcp_session_id;

        # Forward auth server response headers to backend
        proxy_set_header X-User $auth_user;
        proxy_set_header X-Username $auth_username;
        proxy_set_header X-Client-Id-Auth $auth_client_id;
        proxy_set_header X-Scopes $auth_scopes;
        proxy_set_header X-Auth-Method $auth_method;
        proxy_set_header X-Server-Name $auth_server_name;
        proxy_set_header X-Tool-Name $auth_tool_name;
        # The internal JWT mcp_proxy verifies (it ignores the X-* headers above).
        proxy_set_header X-Internal-Token $auth_internal_token;

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


class NginxReloadScheduler:
    """Coalesces multiple nginx reload requests into periodic batched reloads.

    Instead of reloading nginx on every server registration, callers invoke
    mark_dirty() which sets a boolean flag. A background task wakes every
    debounce_seconds, checks if the flag is set (or polls the DB for external
    changes in multi-replica deployments), regenerates the config if the
    rendered output differs from the last-applied version, and reloads nginx
    once.

    See issue #1087 and .scratchpad/lld-nginx-debounced-reload.md.
    """

    def __init__(
        self,
        debounce_seconds: float = 2.0,
        poll_external: bool = True,
    ) -> None:
        self._dirty: bool = False
        self._debounce_seconds = debounce_seconds
        self._poll_external = poll_external
        self._task: asyncio.Task | None = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._last_config_hash: str = ""
        self._flush_lock: asyncio.Lock = asyncio.Lock()

    def mark_dirty(self) -> None:
        """Signal that nginx config needs regeneration. Non-blocking."""
        self._dirty = True
        nginx_service._cached_server_names = None

    def seed_hash(self, config_text: str) -> None:
        """Set the initial config hash after startup generation.

        Prevents a redundant reload on the first scheduler tick.
        """
        self._last_config_hash = hashlib.sha256(config_text.encode()).hexdigest()

    async def start(self) -> None:
        """Start the background flush loop. Call once at app startup."""
        self._task = asyncio.create_task(self._flush_loop())
        logger.info(
            "NginxReloadScheduler started (debounce=%.1fs, poll_external=%s)",
            self._debounce_seconds,
            self._poll_external,
        )

    async def stop(self) -> None:
        """Gracefully stop the flush loop. Performs one final flush if dirty."""
        self._stop_event.set()
        if self._task:
            await self._task

    async def flush_now(self) -> None:
        """Force an immediate regen+reload. Used for toggle/delete where the
        change must be reflected before the HTTP response returns."""
        await self._do_reload_if_changed()

    async def _flush_loop(self) -> None:
        while not self._stop_event.is_set():
            await asyncio.sleep(self._debounce_seconds)
            if self._dirty or self._poll_external:
                await self._do_reload_if_changed()

        if self._dirty:
            await self._do_reload_if_changed()

    async def _do_reload_if_changed(self) -> None:
        async with self._flush_lock:
            self._dirty = False
            try:
                enabled_servers = await _fetch_all_enabled_servers()
                config_text = await nginx_service.render_config(enabled_servers)
                if config_text is None:
                    return

                new_hash = hashlib.sha256(config_text.encode()).hexdigest()
                if new_hash == self._last_config_hash:
                    return

                # Config changed: write virtual server Lua mappings, then nginx config
                await nginx_service._commit_virtual_server_mappings()

                async with nginx_service.reload_lock:
                    _atomic_write_text(settings.nginx_config_path, config_text)
                    reloaded = await asyncio.to_thread(nginx_service.reload_nginx)
                if reloaded:
                    self._last_config_hash = new_hash
                    logger.info(
                        "Debounced nginx reload completed (hash=%s)",
                        new_hash[:12],
                    )
                else:
                    self._dirty = True
            except Exception as e:
                logger.error("Debounced nginx reload failed: %s", e)
                self._dirty = True


async def _fetch_all_enabled_servers() -> dict[str, Any]:
    """Fetch all enabled servers from the DB for nginx config generation."""
    from registry.services.server_service import server_service

    enabled_servers: dict[str, Any] = {}
    for path in await server_service.get_enabled_services():
        info = await server_service.get_server_info(path)
        if info:
            enabled_servers[path] = info
    return enabled_servers


# Module-level singleton
nginx_reload_scheduler = NginxReloadScheduler(
    debounce_seconds=float(os.getenv("NGINX_RELOAD_DEBOUNCE_SECONDS", "5.0")),
    poll_external=os.getenv("NGINX_RELOAD_POLL_EXTERNAL", "true").lower()
    in ("1", "true", "yes", "on"),
)
