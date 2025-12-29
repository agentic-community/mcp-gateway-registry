import logging
import asyncio
import httpx
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.parse import urlparse

from .config import settings
from registry.constants import HealthStatus, REGISTRY_CONSTANTS

logger = logging.getLogger(__name__)


class NginxConfigService:
    """Service for generating Nginx configuration for registered servers."""

    def _build_internal_validate_location_path(
        self,
        *,
        location_path: str,
        server_path_for_auth: str,
        transport_type: str,
    ) -> str:
        digest_source = f"{location_path}|{server_path_for_auth}|{transport_type}".encode(
            "utf-8"
        )
        digest = hashlib.sha256(digest_source).hexdigest()[:12]
        return f"/__enforceai_validate_{digest}"

    def _normalize_gateway_base_path(
        self,
        path: str,
    ) -> str:
        normalized = path.strip()
        if not normalized:
            return "/"

        normalized = "/" + normalized.lstrip("/")
        if normalized != "/":
            normalized = normalized.rstrip("/")

        return normalized or "/"

    def _normalize_endpoint_suffix(
        self,
        endpoint: Optional[str],
        *,
        default: str,
    ) -> str:
        if endpoint is None or not str(endpoint).strip():
            endpoint_value = default
        else:
            endpoint_value = str(endpoint).strip()

        if not endpoint_value.startswith("/"):
            endpoint_value = f"/{endpoint_value}"

        return endpoint_value

    def _build_upstream_endpoint_url(
        self,
        proxy_pass_url: str,
        *,
        endpoint_suffix: str,
    ) -> str:
        proxy_pass_url = str(proxy_pass_url).strip()
        if not proxy_pass_url:
            return proxy_pass_url

        base_url = proxy_pass_url.rstrip("/")

        # If the user already provided a URL that includes a transport endpoint, do not
        # append a second endpoint suffix.
        if (
            base_url.endswith("/mcp")
            or "/mcp/" in base_url
            or base_url.endswith("/sse")
            or "/sse/" in base_url
        ):
            return proxy_pass_url

        return f"{base_url}{endpoint_suffix}"

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
                self.nginx_template_path = Path(REGISTRY_CONSTANTS.NGINX_TEMPLATE_HTTP_AND_HTTPS_LOCAL)
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
        import shutil
        import subprocess

        # Priority 1: Check GATEWAY_ADDITIONAL_SERVER_NAMES env var (user-provided)
        gateway_names = os.environ.get('GATEWAY_ADDITIONAL_SERVER_NAMES', '')
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
                    timeout=2.0
                )

                if token_response.status_code == 200:
                    token = token_response.text

                    # Try to get private IP from EC2 metadata
                    ip_response = await client.get(
                        "http://169.254.169.254/latest/meta-data/local-ipv4",
                        headers={"X-aws-ec2-metadata-token": token},
                        timeout=2.0
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
        ecs_uri = os.environ.get('ECS_CONTAINER_METADATA_URI') or os.environ.get('ECS_CONTAINER_METADATA_URI_V4')
        if ecs_uri:
            try:
                async with httpx.AsyncClient() as client:
                    metadata_response = await client.get(
                        f"{ecs_uri}",
                        timeout=2.0
                    )
                    if metadata_response.status_code == 200:
                        import json
                        metadata = json.loads(metadata_response.text)
                        # Try to extract IP from ECS metadata
                        if 'Networks' in metadata and metadata['Networks']:
                            private_ip = metadata['Networks'][0].get('IPv4Addresses', [None])[0]
                            if private_ip:
                                logger.info(f"Auto-detected ECS container IP: {private_ip}")
                                return private_ip
            except Exception as e:
                logger.debug(f"ECS metadata detection failed: {e}")

        # Priority 4: Try EKS/Kubernetes detection
        pod_ip = os.environ.get('POD_IP')
        if pod_ip:
            logger.info(f"Auto-detected Kubernetes pod IP: {pod_ip}")
            return pod_ip

        # Priority 5: Try generic hostname command (works on most Linux systems)
        try:
            hostname_executable = shutil.which("hostname")
            if not hostname_executable:
                raise FileNotFoundError("hostname not found")
            result = subprocess.run(
                [hostname_executable, "-I"],
                capture_output=True,
                text=True,
                timeout=2.0
            )
            if result.returncode == 0:
                ips = result.stdout.strip().split()
                if ips:
                    # Use first IP (usually the private IP on single-interface systems)
                    private_ip = ips[0]
                    logger.info(f"Auto-detected private IP via hostname command: {private_ip}")
                    return private_ip
        except FileNotFoundError:
            logger.debug("hostname command not available")
        except Exception as e:
            logger.debug(f"Generic hostname detection failed: {e}")

        # Priority 6: Backward compatibility with old EC2_PUBLIC_DNS env var
        fallback_dns = os.environ.get('EC2_PUBLIC_DNS', '')
        if fallback_dns:
            logger.info(f"Using EC2_PUBLIC_DNS environment variable (deprecated): {fallback_dns}")
            return fallback_dns

        # No additional server names available
        logger.info("No additional server names available - will use only localhost and mcpgateway.ddns.net")
        return ""

    def generate_config(self, servers: Dict[str, Dict[str, Any]]) -> bool:
        """Generate Nginx configuration (synchronous version for non-async contexts)."""
        try:
            # Check if we're in an async context
            try:
                # If we're already in an event loop, we need to run this differently
                loop = asyncio.get_running_loop()
                # We're in an async context, this won't work
                logger.error("generate_config called from async context - use generate_config_async instead")
                return False
            except RuntimeError:
                # No running loop, we can use asyncio.run()
                return asyncio.run(self.generate_config_async(servers))
        except Exception as e:
            logger.error(f"Failed to generate Nginx configuration: {e}", exc_info=True)
            return False
        
    async def generate_config_async(self, servers: Dict[str, Dict[str, Any]]) -> bool:
        """Generate Nginx configuration with additional server names and dynamic location blocks."""
        try:
            servers = self._filter_servers_by_egress_allowlist(servers)

            # Read template
            if not self.nginx_template_path.exists():
                logger.warning(f"Nginx template not found at {self.nginx_template_path}")
                return False
                
            with open(self.nginx_template_path, "r") as f:
                template_content = f.read()
            
            # Get health service to check server health
            from ..health.service import health_service
            
            # Generate location blocks for enabled and healthy servers with transport support
            location_blocks = []
            for path, server_info in servers.items():
                proxy_pass_url = server_info.get("proxy_pass_url")
                if proxy_pass_url:
                    # Check if server is healthy (including auth-expired which is still reachable)
                    health_status = health_service.server_health_status.get(path, HealthStatus.UNKNOWN)
                    
                    # Include servers that are healthy or just have expired auth (server is up)
                    if HealthStatus.is_healthy(health_status):
                        # Generate transport-aware location blocks
                        transport_blocks = self._generate_transport_location_blocks(path, server_info)
                        location_blocks.extend(transport_blocks)
                        logger.debug(f"Added location blocks for healthy service: {path}")
                    else:
                        # Add commented out block for unhealthy services
                        commented_block = f"""
#    location {path}/ {{
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
                        logger.debug(f"Added commented location block for unhealthy service {path} (status: {health_status})")
            
            # Fetch additional server names (custom domains/IPs)
            additional_server_names = await self.get_additional_server_names()

            # Get API version from constants
            api_version = REGISTRY_CONSTANTS.ANTHROPIC_API_VERSION

            # Parse Keycloak configuration from KEYCLOAK_URL environment variable
            import os
            keycloak_url = os.environ.get('KEYCLOAK_URL', 'http://keycloak:8080')
            try:
                parsed_keycloak = urlparse(keycloak_url)
                keycloak_scheme = parsed_keycloak.scheme or 'http'
                keycloak_host = parsed_keycloak.hostname or 'keycloak'
                # Use default port based on scheme if not specified
                if parsed_keycloak.port:
                    keycloak_port = str(parsed_keycloak.port)
                else:
                    keycloak_port = '443' if keycloak_scheme == 'https' else '8080'

                # Validate that we can actually resolve the hostname
                if not keycloak_host or keycloak_host == 'keycloak':
                    # If we end up with just 'keycloak', use the full URL's netloc instead
                    keycloak_host = parsed_keycloak.netloc.split(':')[0] if parsed_keycloak.netloc else 'keycloak'
                    logger.warning(f"Keycloak hostname is 'keycloak', using netloc instead: {keycloak_host}")

                logger.info(f"Using Keycloak configuration from KEYCLOAK_URL '{keycloak_url}': {keycloak_scheme}://{keycloak_host}:{keycloak_port}")
            except Exception as e:
                logger.warning(f"Failed to parse KEYCLOAK_URL '{keycloak_url}': {e}. Using defaults.")
                keycloak_scheme = 'http'
                keycloak_host = 'keycloak'
                keycloak_port = '8080'

            # Replace placeholders in template
            config_content = template_content.replace("{{LOCATION_BLOCKS}}", "\n".join(location_blocks))
            config_content = config_content.replace("{{ADDITIONAL_SERVER_NAMES}}", additional_server_names)
            config_content = config_content.replace("{{ANTHROPIC_API_VERSION}}", api_version)
            config_content = config_content.replace("{{KEYCLOAK_SCHEME}}", keycloak_scheme)
            config_content = config_content.replace("{{KEYCLOAK_HOST}}", keycloak_host)
            config_content = config_content.replace("{{KEYCLOAK_PORT}}", keycloak_port)

            # Write config file
            with open(settings.nginx_config_path, "w") as f:
                f.write(config_content)

            logger.info(f"Generated Nginx configuration with {len(location_blocks)} location blocks and additional server names: {additional_server_names}")
            
            # Automatically reload nginx after generating config
            self.reload_nginx()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate Nginx configuration: {e}", exc_info=True)
            return False

    def _filter_servers_by_egress_allowlist(
        self,
        servers: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        import os

        db_path = os.getenv("ENFORCEAI_DB_PATH")
        if db_path is None or not db_path.strip():
            return servers

        try:
            from auth_server.enforceai.egress.allowlist import (
                check_proxy_pass_url,
            )
            from auth_server.enforceai.stores.sqlite.egress_allowlist_store import (
                SqliteEgressAllowlistStore,
            )
        except Exception as exc:
            logger.error(
                f"Egress allowlist enforcement unavailable; refusing to proxy: {exc}"
            )
            return {}

        store = SqliteEgressAllowlistStore(db_path=Path(db_path))
        try:
            entries = store.list_entries(include_expired=False)
        except Exception as exc:
            logger.error(
                f"Egress allowlist store unavailable; refusing to proxy: {exc}"
            )
            return {}

        if not entries:
            logger.error("Egress allowlist is empty; refusing to proxy any servers")
            return {}

        allowed: Dict[str, Dict[str, Any]] = {}
        for path, server_info in servers.items():
            proxy_pass_url = server_info.get("proxy_pass_url")
            if not proxy_pass_url:
                continue

            decision = check_proxy_pass_url(
                proxy_pass_url=str(proxy_pass_url),
                entries=entries,
            )
            if not decision.allowed:
                logger.error(
                    f"Skipping server {path}: proxy_pass_url not allowlisted ({decision.reason})"
                )
                continue

            allowed[path] = server_info

        return allowed
            
    def reload_nginx(self) -> bool:
        """Reload Nginx configuration (if running in appropriate environment)."""
        try:
            import shutil
            import subprocess

            nginx_executable = shutil.which("nginx")
            if not nginx_executable:
                logger.warning("Nginx not found - skipping reload")
                return False

            # Test the configuration first before reloading
            test_result = subprocess.run(
                [nginx_executable, "-t"],
                capture_output=True,
                text=True,
            )
            if test_result.returncode != 0:
                logger.error(f"Nginx configuration test failed: {test_result.stderr}")
                logger.info("Skipping Nginx reload due to configuration errors")
                return False

            result = subprocess.run(
                [nginx_executable, "-s", "reload"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.info("Nginx configuration reloaded successfully")
                return True
            else:
                logger.error(f"Failed to reload Nginx: {result.stderr}")
                return False
        except Exception:
            logger.exception("Unexpected error reloading Nginx")
            return False


    def _generate_transport_location_blocks(self, path: str, server_info: Dict[str, Any]) -> list:
        """Generate nginx location blocks for different transport types."""
        blocks = []
        proxy_pass_url = server_info.get("proxy_pass_url", "")
        supported_transports = server_info.get("supported_transports") or ["streamable-http"]

        if not isinstance(supported_transports, list) or not supported_transports:
            supported_transports = ["streamable-http"]

        gateway_base = self._normalize_gateway_base_path(path)
        upstream_auth = server_info.get("upstream_auth")

        for transport_type in supported_transports:
            if transport_type == "stdio":
                continue

            if transport_type == "streamable-http":
                gateway_location_path = (
                    "/mcp" if gateway_base == "/" else f"{gateway_base}/mcp"
                )
                endpoint_suffix = self._normalize_endpoint_suffix(
                    server_info.get("mcp_endpoint"),
                    default="/mcp",
                )
            elif transport_type == "sse":
                gateway_location_path = (
                    "/sse" if gateway_base == "/" else f"{gateway_base}/sse"
                )
                endpoint_suffix = self._normalize_endpoint_suffix(
                    server_info.get("sse_endpoint"),
                    default="/sse",
                )
            else:
                logger.info(
                    f"Server {path}: Unknown transport '{transport_type}', skipping"
                )
                continue

            proxy_url = self._build_upstream_endpoint_url(
                str(proxy_pass_url),
                endpoint_suffix=endpoint_suffix,
            )
            logger.info(
                f"Server {path}: Routing {gateway_location_path} -> {proxy_url} ({transport_type})"
            )

            blocks.append(
                self._create_location_block(
                    location_path=gateway_location_path,
                    server_path_for_auth=path,
                    proxy_pass_url=proxy_url,
                    transport_type=transport_type,
                    upstream_auth=upstream_auth,
                )
            )

        return blocks

    def _create_location_block(
        self,
        *,
        location_path: str,
        server_path_for_auth: str,
        proxy_pass_url: str,
        transport_type: str,
        upstream_auth: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a single nginx location block with transport-specific configuration."""
        
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

        def _escape_nginx_string(value: str) -> str:
            return value.replace("\\", "\\\\").replace('"', '\\"')

        internal_validate_location = self._build_internal_validate_location_path(
            location_path=location_path,
            server_path_for_auth=server_path_for_auth,
            transport_type=transport_type,
        )

        upstream_auth_type = "none"
        upstream_credential_binding = "service"
        upstream_provider = ""
        upstream_header_name = ""
        upstream_scheme = ""

        if isinstance(upstream_auth, dict):
            upstream_auth_type = str(upstream_auth.get("type") or "none")
            upstream_auth_type = (
                upstream_auth_type.strip().lower().replace("_", "-") or "none"
            )
            upstream_credential_binding = str(
                upstream_auth.get("credential_binding") or "service"
            )
            upstream_provider = str(upstream_auth.get("provider") or "")

            injection = upstream_auth.get("injection")
            if isinstance(injection, dict):
                upstream_header_name = str(injection.get("header_name") or "")
                upstream_scheme = str(injection.get("scheme") or "")

        if not upstream_header_name:
            if upstream_auth_type == "api-key":
                upstream_header_name = "X-API-Key"
            elif upstream_auth_type in {"jwt", "oauth2", "oidc", "provider-oauth"}:
                upstream_header_name = "Authorization"

        if upstream_auth_type in {"jwt", "oauth2", "oidc", "provider-oauth"} and not upstream_scheme:
            upstream_scheme = "Bearer"

        upstream_injection_settings = ""
        if upstream_auth_type == "api-key":
            upstream_injection_settings = (
                f"\n        proxy_set_header {upstream_header_name} $enforceai_upstream_api_key;"
            )
        elif upstream_auth_type in {"jwt", "oauth2", "oidc", "provider-oauth"}:
            upstream_injection_settings = (
                f"\n        proxy_set_header {upstream_header_name} $enforceai_upstream_authorization;"
            )

        # Common proxy settings
        common_settings = f"""
	        # Use IPv4 resolver (disable IPv6)
	        resolver 8.8.8.8 8.8.4.4 valid=10s;
	        resolver_timeout 5s;

	        # Per-server upstream auth context for /validate.
	        set $enforceai_server_path "{_escape_nginx_string(server_path_for_auth)}";
	        set $enforceai_upstream_auth_type "{_escape_nginx_string(upstream_auth_type)}";
	        set $enforceai_upstream_credential_binding "{_escape_nginx_string(upstream_credential_binding)}";
	        set $enforceai_upstream_provider "{_escape_nginx_string(upstream_provider)}";
	        set $enforceai_upstream_header_name "{_escape_nginx_string(upstream_header_name)}";
	        set $enforceai_upstream_scheme "{_escape_nginx_string(upstream_scheme)}";

	        # Preserve inbound auth headers for the auth_request subrequest (/validate).
	        # auth_request can run as an internal subrequest where $http_* vars may not reflect
	        # the original client headers, so we copy them into variables here.
	        set $enforceai_authorization $http_authorization;
	        set $enforceai_x_authorization $http_x_authorization;

	        # Authenticate request - pass entire request to auth server
	        auth_request {internal_validate_location};
	        
	        # Capture auth server response headers for forwarding
	        auth_request_set $auth_user $upstream_http_x_user;
	        auth_request_set $auth_username $upstream_http_x_username;
	        auth_request_set $auth_client_id $upstream_http_x_client_id;
	        auth_request_set $auth_scopes $upstream_http_x_scopes;
	        auth_request_set $auth_method $upstream_http_x_auth_method;
	        auth_request_set $auth_server_name $upstream_http_x_server_name;
	        auth_request_set $auth_tool_name $upstream_http_x_tool_name;
	        auth_request_set $auth_allowed_tools $upstream_http_x_allowed_tools;
	        auth_request_set $mcp_principal $upstream_http_x_mcp_principal;
	        auth_request_set $mcp_auth_type $upstream_http_x_mcp_auth_type;
	        auth_request_set $mcp_scopes $upstream_http_x_mcp_scopes;
	        auth_request_set $mcp_provider $upstream_http_x_mcp_provider;
	        auth_request_set $mcp_claims $upstream_http_x_mcp_claims;
	        auth_request_set $enforceai_upstream_authorization $upstream_http_x_enforceai_upstream_authorization;
	        auth_request_set $enforceai_upstream_api_key $upstream_http_x_enforceai_upstream_api_key;
	        auth_request_set $enforceai_upstream_api_key_header $upstream_http_x_enforceai_upstream_api_key_header;
	        auth_request_set $enforceai_upstream_mode $upstream_http_x_enforceai_upstream_mode;
	        auth_request_set $enforceai_error_code $upstream_http_x_enforceai_error_code;
	        
	        # Proxy to MCP server
	        proxy_pass {proxy_pass_url};
	        proxy_http_version 1.1;
	        proxy_ssl_server_name on;
        proxy_set_header Host {host_header};
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
	        # Add original URL for auth server scope validation
	        proxy_set_header X-Original-URL $scheme://$host$request_uri;

	        # Strip client-supplied identity and upstream injection headers.
	        proxy_set_header X-MCP-Principal "";
	        proxy_set_header X-MCP-Auth-Type "";
	        proxy_set_header X-MCP-Scopes "";
	        proxy_set_header X-MCP-Provider "";
	        proxy_set_header X-MCP-Claims "";
	        proxy_set_header X-EnforceAI-Upstream-Authorization "";
	        proxy_set_header X-EnforceAI-Upstream-Api-Key "";
	        proxy_set_header X-EnforceAI-Upstream-Api-Key-Header "";
	        proxy_set_header X-EnforceAI-Upstream-Mode "";

	        # Strip client-supplied gateway credentials (never forwarded upstream).
	        proxy_set_header Authorization "";
	        proxy_set_header X-Authorization "";
	        proxy_set_header X-API-Key "";
	        proxy_set_header X-Gateway-Token "";
	        proxy_set_header Cookie "";

	        # Forward auth server response headers to backend
	        proxy_set_header X-User $auth_user;
	        proxy_set_header X-Username $auth_username;
	        proxy_set_header X-Client-Id-Auth $auth_client_id;
	        proxy_set_header X-Scopes $auth_scopes;
	        proxy_set_header X-Auth-Method $auth_method;
	        proxy_set_header X-Server-Name $auth_server_name;
	        proxy_set_header X-Tool-Name $auth_tool_name;
	        proxy_set_header X-MCP-Principal $mcp_principal;
	        proxy_set_header X-MCP-Auth-Type $mcp_auth_type;
	        proxy_set_header X-MCP-Scopes $mcp_scopes;
	        proxy_set_header X-MCP-Provider $mcp_provider;
	        proxy_set_header X-MCP-Claims $mcp_claims;{upstream_injection_settings}
	        
	        # Pass all original client headers
	        proxy_pass_request_headers on;
	        
	        # Handle auth errors
	        error_page 401 = @auth_error;
	        error_page 403 = @forbidden_error;
	        error_page 424 = @upstream_credentials_required;"""

        validate_location_block = f"""
    # Internal auth_request endpoint for {location_path} ({transport_type})
    location = {internal_validate_location} {{
        internal;

        proxy_pass http://auth-server:8888/validate;

        # Pass original request info (auth_request uses a subrequest; these vars are expected to reflect the parent request).
        proxy_set_header X-Original-URI $request_uri;
        proxy_set_header X-Original-Method $request_method;
        proxy_set_header X-Original-URL $scheme://$host$request_uri;

        # Per-server upstream auth context as literals (do not rely on nginx variable inheritance into auth_request subrequests).
        proxy_set_header X-EnforceAI-Server-Path "{_escape_nginx_string(server_path_for_auth)}";
        proxy_set_header X-EnforceAI-Upstream-Auth-Type "{_escape_nginx_string(upstream_auth_type)}";
        proxy_set_header X-EnforceAI-Upstream-Credential-Binding "{_escape_nginx_string(upstream_credential_binding)}";
        proxy_set_header X-EnforceAI-Upstream-Provider "{_escape_nginx_string(upstream_provider)}";
        proxy_set_header X-EnforceAI-Upstream-Header-Name "{_escape_nginx_string(upstream_header_name)}";
        proxy_set_header X-EnforceAI-Upstream-Scheme "{_escape_nginx_string(upstream_scheme)}";

        # Forward all original headers (including Authorization and X-Body from Lua).
        proxy_pass_request_headers on;

        # Short timeouts for auth validation
        proxy_connect_timeout 10s;
        proxy_read_timeout 10s;
        proxy_send_timeout 10s;
    }}
"""
        
        # Transport-specific settings
        if transport_type == "sse":
            transport_settings = """
        # Capture request body for auth validation using Lua
        rewrite_by_lua_file /etc/nginx/lua/capture_body.lua;
        # Ensure filtered tools/list responses don't keep stale Content-Length
        header_filter_by_lua_file /etc/nginx/lua/filter_tools_list_headers.lua;
        # Filter tools/list responses based on allowlist from auth server
        body_filter_by_lua_file /etc/nginx/lua/filter_tools_list.lua;

        # Enable buffering for Lua body filter to work correctly.
        # For actual SSE streams, upstream can send X-Accel-Buffering: no to disable.
        proxy_buffering on;
        proxy_buffer_size 16k;
        proxy_buffers 4 16k;
        proxy_cache off;
        proxy_set_header Connection $http_connection;
        proxy_set_header Upgrade $http_upgrade;
        # Explicitly preserve Accept header for MCP protocol requirements
        proxy_set_header Accept $http_accept;
        chunked_transfer_encoding off;"""

        elif transport_type == "streamable-http":
            transport_settings = """
        # MCP streamable-http endpoints are JSON-RPC over HTTP POST. Browsers often do a GET; fail fast with a helpful error.
        if ($request_method !~ ^(POST|OPTIONS)$) {
            add_header Content-Type application/json;
            return 405 '{"error":"Method not allowed. Use HTTP POST with a JSON-RPC body."}';
        }

        # Capture request body for auth validation using Lua
        rewrite_by_lua_file /etc/nginx/lua/capture_body.lua;
        # Ensure filtered tools/list responses don't keep stale Content-Length
        header_filter_by_lua_file /etc/nginx/lua/filter_tools_list_headers.lua;
        # Filter tools/list responses based on allowlist from auth server
        body_filter_by_lua_file /etc/nginx/lua/filter_tools_list.lua;

        # Enable buffering for Lua body filter to work correctly.
        # For actual SSE streams, upstream can send X-Accel-Buffering: no to disable.
        proxy_buffering on;
        proxy_buffer_size 16k;
        proxy_buffers 4 16k;
        proxy_set_header Connection "";
        # FastMCP streamable HTTP requires clients to accept both JSON and SSE frames.
        proxy_set_header Accept "application/json, text/event-stream";"""

        else:  # direct
            transport_settings = """
        # Capture request body for auth validation using Lua
        rewrite_by_lua_file /etc/nginx/lua/capture_body.lua;
        # Ensure filtered tools/list responses don't keep stale Content-Length
        header_filter_by_lua_file /etc/nginx/lua/filter_tools_list_headers.lua;
        # Filter tools/list responses based on allowlist from auth server
        body_filter_by_lua_file /etc/nginx/lua/filter_tools_list.lua;

        # Enable buffering for Lua body filter to work correctly.
        proxy_buffering on;
        proxy_buffer_size 16k;
        proxy_buffers 4 16k;
        proxy_cache off;
        proxy_set_header Connection $http_connection;
        proxy_set_header Upgrade $http_upgrade;
        chunked_transfer_encoding off;"""
        
        # Use the location path exactly as specified in the server configuration
        # Users have full control over the location path format (with or without trailing slash)
        logger.info(f"Creating location block for {location_path} with {transport_type} transport")
        
        return f"""{validate_location_block}
    location {location_path} {{{transport_settings}{common_settings}
    }}"""


# Global nginx service instance
nginx_service = NginxConfigService() 
