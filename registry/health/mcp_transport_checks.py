from __future__ import annotations

import asyncio
import logging
from typing import (
    Dict,
    Optional,
)

import httpx

from registry.constants import (
    HealthStatus,
)

logger = logging.getLogger("registry.health.service")

_SENSITIVE_HEADER_KEYS: set[str] = {
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    "api-key",
    "x-auth-token",
    "x-amz-security-token",
}


def _redact_headers_for_logging(
    headers: Dict[str, str],
) -> Dict[str, str]:
    redacted: Dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in _SENSITIVE_HEADER_KEYS:
            redacted[key] = "***REDACTED***"
        else:
            redacted[key] = value
    return redacted


class McpTransportChecker:
    def _build_headers_for_server(
        self,
        server_info: Dict,
        include_session_id: bool = False
    ) -> Dict[str, str]:
        """
        Build HTTP headers for server requests by merging default headers with server-specific headers.

        Args:
            server_info: Server configuration dictionary
            include_session_id: Whether to generate and include Mcp-Session-Id header

        Returns:
            Merged headers dictionary
        """
        import uuid

        # Start with default headers for MCP endpoints
        headers = {
            'Accept': 'application/json, text/event-stream',
            'Content-Type': 'application/json'
        }

        # Add session ID if requested (required by some MCP servers like Cloudflare)
        if include_session_id:
            session_id = str(uuid.uuid4())
            headers['Mcp-Session-Id'] = session_id
            logger.debug("Generated Mcp-Session-Id for request")

        # Merge server-specific headers if present
        server_headers = server_info.get("headers", [])
        if server_headers and isinstance(server_headers, list):
            for header_dict in server_headers:
                if isinstance(header_dict, dict):
                    headers.update(header_dict)
                    logger.debug("Added server header keys: %s", list(header_dict.keys()))

        return headers


    async def _initialize_mcp_session(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        headers: Dict[str, str]
    ) -> Optional[str]:
        """
        Initialize an MCP session and retrieve the session ID from the server.

        Args:
            client: httpx AsyncClient instance
            endpoint: The MCP endpoint URL
            headers: Headers to send with the request

        Returns:
            Session ID string if successful, None otherwise
        """
        import uuid

        try:
            # Send initialize request without session ID
            # The server will generate and return a session ID in the response header
            init_headers = headers.copy()

            initialize_payload = {
                "jsonrpc": "2.0",
                "id": "0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "mcp-gateway-registry",
                        "version": "1.0.0"
                    }
                }
            }

            response = await client.post(
                endpoint,
                headers=init_headers,
                json=initialize_payload,
                timeout=httpx.Timeout(5.0),
                follow_redirects=True
            )

            # Check if initialize succeeded
            if response.status_code not in [200, 201]:
                logger.warning(
                    "MCP initialize failed for %s: status %s",
                    endpoint,
                    response.status_code,
                )
                logger.debug(
                    "MCP initialize response (truncated): %s",
                    response.text[:200],
                )
                return None

            # Get session ID from response headers (server-generated)
            server_session_id = response.headers.get('Mcp-Session-Id') or response.headers.get('mcp-session-id')
            if server_session_id:
                logger.debug(f"Server returned session ID: {server_session_id}")
                return server_session_id
            else:
                # If server doesn't return a session ID, generate one for stateless servers
                client_session_id = str(uuid.uuid4())
                logger.debug(f"Server did not return session ID, using client-generated: {client_session_id}")
                return client_session_id

        except Exception as e:
            logger.warning(f"MCP initialize failed for {endpoint}: {e}")
            return None


    async def _try_ping_without_auth(self, client: httpx.AsyncClient, endpoint: str) -> bool:
        """
        Try a simple ping without authentication headers.
        Used as fallback when auth fails to determine if server is reachable.

        Args:
            client: httpx AsyncClient instance
            endpoint: The MCP endpoint URL to ping

        Returns:
            bool: True if server responds (indicating it's reachable but auth expired)
        """
        import uuid

        try:
            # Minimal headers without auth but with session ID (required by some servers)
            headers = {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
                'Mcp-Session-Id': str(uuid.uuid4())
            }
            ping_payload = '{ "jsonrpc": "2.0", "id": "0", "method": "ping" }'

            response = await client.post(
                endpoint,
                headers=headers,
                content=ping_payload,
                timeout=httpx.Timeout(5.0),
                follow_redirects=True
            )
            
            # Check if we get any valid response (even auth errors indicate server is up)
            if response.status_code in [200, 400, 401, 403]:
                logger.info(f"Ping without auth succeeded for {endpoint} - server is reachable but auth may have expired")
                return True
            else:
                logger.warning(f"Ping without auth failed for {endpoint}: Status {response.status_code}")
                return False
                
        except Exception as e:
            logger.warning(f"Ping without auth failed for {endpoint}: {type(e).__name__} - {e}")
            return False


    async def _check_server_endpoint_transport_aware(self, client: httpx.AsyncClient, proxy_pass_url: str, server_info: Dict) -> tuple[bool, str]:
        """Check server endpoint using transport-aware logic.
        
        Returns:
            tuple[bool, str]: (is_healthy, status_detail)
        """
        if not proxy_pass_url:
            return False, HealthStatus.UNHEALTHY_MISSING_PROXY_URL
            
        # Get transport information from server_info
        supported_transports = server_info.get("supported_transports", ["streamable-http"])

        # If URL already has transport endpoint, use it directly
        # BUT skip this shortcut for streamable-http to ensure proper POST ping is used
        has_transport_in_url = (proxy_pass_url.endswith('/mcp') or proxy_pass_url.endswith('/sse') or
                                '/mcp/' in proxy_pass_url or '/sse/' in proxy_pass_url)

        if has_transport_in_url and "streamable-http" not in supported_transports:
            logger.info(f"[TRACE] Found transport endpoint in URL: {proxy_pass_url}")
            logger.info(f"[TRACE] URL contains /mcp: {'/mcp' in proxy_pass_url}, URL contains /sse: {'/sse' in proxy_pass_url}")
            try:
                # Build headers including server-specific headers
                headers = self._build_headers_for_server(server_info)
                # For SSE endpoints, use a shorter timeout since they start streaming immediately
                if proxy_pass_url.endswith('/sse') or '/sse/' in proxy_pass_url:
                    logger.info(f"[TRACE] Detected SSE endpoint in URL, using SSE-specific handling")
                    timeout = httpx.Timeout(connect=5.0, read=2.0, write=5.0, pool=5.0)
                    try:
                        response = await client.get(proxy_pass_url, headers=headers, follow_redirects=True, timeout=timeout)
                        if self._is_mcp_endpoint_healthy(response):
                            return True, HealthStatus.HEALTHY
                        else:
                            return False, f"unhealthy: status {response.status_code}"
                    except (httpx.TimeoutException, asyncio.TimeoutError) as e:
                        # For SSE endpoints, timeout while reading streaming response is normal after getting 200 OK
                        logger.debug(f"SSE endpoint {proxy_pass_url} timed out while streaming (expected): {e}")
                        # If we can extract status code from response, check if it was 200
                        if hasattr(e, 'response') and e.response and e.response.status_code == 200:
                            logger.debug(f"SSE endpoint {proxy_pass_url} returned 200 OK before timeout - considering healthy")
                            return True, HealthStatus.HEALTHY
                        # For SSE, timeout after initial connection usually means server is responding
                        return True, HealthStatus.HEALTHY
                    except Exception as e:
                        logger.warning(f"SSE endpoint {proxy_pass_url} failed with exception: {type(e).__name__} - {e}")
                        return False, f"unhealthy: {type(e).__name__}"
                else:
                    logger.info(f"[TRACE] Detected MCP endpoint in URL, using standard HTTP handling")
                    response = await client.get(proxy_pass_url, headers=headers, follow_redirects=True)

                    # Check for auth failures first
                    if response.status_code in [401, 403]:
                        logger.info(f"[TRACE] Auth failure detected ({response.status_code}) for {proxy_pass_url}, trying ping without auth")
                        if await self._try_ping_without_auth(client, proxy_pass_url):
                            return True, HealthStatus.HEALTHY
                        else:
                            return False, f"unhealthy: auth failed and ping without auth failed"

                    if self._is_mcp_endpoint_healthy(response):
                        return True, HealthStatus.HEALTHY
                    else:
                        return False, f"unhealthy: status {response.status_code}"
            except Exception as e:
                logger.warning(f"Health check failed for {proxy_pass_url}: {type(e).__name__} - {e}")
                return False, f"unhealthy: {type(e).__name__}"

        # Skip health checks for stdio transport (as requested)
        if supported_transports == ["stdio"]:
            logger.info(f"[TRACE] Skipping health check for stdio transport: {proxy_pass_url}")
            return True, HealthStatus.UNKNOWN
        
        # Try endpoints based on supported transports, prioritizing streamable-http
        logger.info(f"[TRACE] No transport endpoint in URL: {proxy_pass_url}")
        logger.info(f"[TRACE] Supported transports: {supported_transports}")
        base_url = proxy_pass_url.rstrip('/')
        
        # Try streamable-http first (default preference)
        if "streamable-http" in supported_transports:
            logger.info(f"[TRACE] Trying streamable-http transport")
            # Build base headers without session ID
            headers = self._build_headers_for_server(server_info, include_session_id=False)

            # Only try /mcp endpoint for streamable-http transport
            # Don't append /mcp if URL already has a transport endpoint (/mcp or /sse)
            if base_url.endswith('/mcp') or base_url.endswith('/sse') or '/mcp/' in base_url or '/sse/' in base_url:
                endpoint = f"{base_url}"
            else:
                endpoint = f"{base_url}/mcp"

            try:
                # Step 1: Initialize session to get session ID
                logger.info(f"[TRACE] Initializing MCP session for endpoint: {endpoint}")
                session_id = await self._initialize_mcp_session(client, endpoint, headers)

                # If initialize failed, check if it was due to auth (401/403)
                # Try ping without auth before giving up
                if not session_id:
                    logger.warning(f"Failed to initialize MCP session for {endpoint}, trying ping without auth")
                    if await self._try_ping_without_auth(client, endpoint):
                        return True, HealthStatus.HEALTHY
                    else:
                        return False, "unhealthy: session initialization failed and ping without auth failed"

                # Step 2: Add session ID to headers for ping
                headers['Mcp-Session-Id'] = session_id
                ping_payload = '{ "jsonrpc": "2.0", "id": "0", "method": "ping" }'

                logger.info(f"[TRACE] Sending ping to endpoint: {endpoint}")
                logger.debug(
                    "[TRACE] Headers being sent: %s",
                    _redact_headers_for_logging(headers),
                )
                response = await client.post(endpoint, headers=headers, content=ping_payload, follow_redirects=True)
                logger.info(f"[TRACE] Response status: {response.status_code}")

                # Check for auth failures first
                if response.status_code in [401, 403]:
                    logger.info(f"[TRACE] Auth failure detected ({response.status_code}) for {endpoint}, trying ping without auth")
                    if await self._try_ping_without_auth(client, endpoint):
                        # ============================================================================
                        # TEMPORARY WORKAROUND - TODO: REVERT AFTER CREDENTIALS MANAGER IS IMPLEMENTED
                        # ============================================================================
                        # Issue: https://github.com/agentic-community/mcp-gateway-registry/issues/167
                        #
                        # Temporarily marking servers with auth failures as "healthy" instead of
                        # "healthy-auth-expired" to avoid confusing users when servers are registered
                        # with auth requirements but no credentials manager is in place yet.
                        #
                        # This allows servers like customer-support-assistant (Bedrock AgentCore) to
                        # show as healthy when they respond to ping, even though live tool fetching
                        # requires authentication.
                        #
                        # BEFORE CREDENTIALS MANAGER: Return healthy (current behavior)
                        # AFTER CREDENTIALS MANAGER:  Return healthy-auth-expired (proper behavior)
                        #
                        # When the credentials manager container is implemented (see design doc at
                        # .scratchpad/credentials-manager-design.md), this should be changed back to:
                        #   return True, HealthStatus.HEALTHY_AUTH_EXPIRED
                        # ============================================================================
                        return True, HealthStatus.HEALTHY  # TODO: Change back to HEALTHY_AUTH_EXPIRED
                    else:
                        return False, f"unhealthy: auth failed and ping without auth failed"
                
                # Check normal health status
                if self._is_mcp_endpoint_healthy_streamable(response):
                    logger.info(f"Health check succeeded at {endpoint}")
                    return True, HealthStatus.HEALTHY
                else:
                    logger.warning(
                        "Health check failed for %s: status %s",
                        endpoint,
                        response.status_code,
                    )
                    logger.debug(
                        "Health check response (truncated): %s",
                        response.text[:200],
                    )
                    return False, f"unhealthy: status {response.status_code}"
                    
            except Exception as e:
                logger.warning(f"Health check failed for {endpoint}: {type(e).__name__} - {e}")
                return False, f"unhealthy: {type(e).__name__}"
        
        # Fallback to SSE
        if "sse" in supported_transports:
            logger.info(f"[TRACE] Trying SSE transport")
            try:
                if base_url.endswith('/sse'):
                    sse_endpoint = f"{base_url}"
                else:
                    sse_endpoint = f"{base_url}/sse"
                # Build headers including server-specific headers
                headers = self._build_headers_for_server(server_info)
                # Use shorter timeout for SSE since it starts streaming immediately
                timeout = httpx.Timeout(connect=5.0, read=2.0, write=5.0, pool=5.0)
                response = await client.get(sse_endpoint, headers=headers, follow_redirects=True, timeout=timeout)
                if self._is_mcp_endpoint_healthy(response):
                    return True, HealthStatus.HEALTHY
            except (httpx.TimeoutException, asyncio.TimeoutError) as e:
                # For SSE endpoints, timeout while reading streaming response is normal after getting 200 OK
                logger.info(f"SSE endpoint {sse_endpoint} timed out while streaming (expected): {e}")
                # If we can extract status code from response, check if it was 200
                if hasattr(e, 'response') and e.response and e.response.status_code == 200:
                    logger.info(f"SSE endpoint {sse_endpoint} returned 200 OK before timeout - considering healthy")
                    return True, HealthStatus.HEALTHY
                # For SSE, timeout after initial connection usually means server is responding
                return True, HealthStatus.HEALTHY
            except Exception as e:
                logger.error(f"SSE endpoint {sse_endpoint} failed with exception: {type(e).__name__} - {e}")
                pass
        
        # If no specific transports, try default streamable-http then sse
        if not supported_transports or supported_transports == []:
            logger.info(f"[TRACE] No specific transports defined, trying defaults")
            headers = self._build_headers_for_server(server_info)
            
            # Only try /mcp endpoint for default streamable-http transport
            endpoint = f"{base_url}/mcp"
            ping_payload = '{ "jsonrpc": "2.0", "id": "0", "method": "ping" }'
            
            try:
                logger.info(f"[TRACE] Trying default endpoint: {endpoint}")
                logger.debug(
                    "[TRACE] Headers being sent: %s",
                    _redact_headers_for_logging(headers),
                )
                response = await client.post(endpoint, headers=headers, content=ping_payload, follow_redirects=True)
                logger.info(f"[TRACE] Response status: {response.status_code}")
                if self._is_mcp_endpoint_healthy_streamable(response):
                    logger.info(f"Health check succeeded at {endpoint}")
                    return True, HealthStatus.HEALTHY
                else:
                    logger.warning(
                        "Health check failed for %s: status %s",
                        endpoint,
                        response.status_code,
                    )
                    logger.debug(
                        "Health check response (truncated): %s",
                        response.text[:200],
                    )
                    return False, f"unhealthy: status {response.status_code}"
            except Exception as e:
                logger.warning(f"Health check failed for {endpoint}: {type(e).__name__} - {e}")
                
            try:
                sse_endpoint = f"{base_url}/sse"
                # Build headers including server-specific headers
                headers = self._build_headers_for_server(server_info)
                # Use shorter timeout for SSE since it starts streaming immediately
                timeout = httpx.Timeout(connect=5.0, read=2.0, write=5.0, pool=5.0)
                response = await client.get(sse_endpoint, headers=headers, follow_redirects=True, timeout=timeout)
                if self._is_mcp_endpoint_healthy(response):
                    return True, HealthStatus.HEALTHY
            except (httpx.TimeoutException, asyncio.TimeoutError) as e:
                # For SSE endpoints, timeout while reading streaming response is normal after getting 200 OK
                logger.info(f"SSE endpoint {sse_endpoint} timed out while streaming (expected): {e}")
                # If we can extract status code from response, check if it was 200
                if hasattr(e, 'response') and e.response and e.response.status_code == 200:
                    logger.info(f"SSE endpoint {sse_endpoint} returned 200 OK before timeout - considering healthy")
                    return True, HealthStatus.HEALTHY
                # For SSE, timeout after initial connection usually means server is responding
                return True, "healthy"
            except Exception as e:
                logger.error(f"SSE endpoint {sse_endpoint} failed with exception: {type(e).__name__} - {e}")
                pass
        
        return False, "unhealthy: all transport checks failed"


    def _is_mcp_endpoint_healthy_streamable(self, response) -> bool:
        """
        Determine if a streamable-http MCP endpoint is healthy based on HTTP response.
        
        For streamable-http MCP endpoints, we consider them healthy if:
        1. HTTP 200 OK - Normal successful response
        2. HTTP 400 Bad Request with JSON-RPC error code -32600
        
        Args:
            response: httpx.Response object from the health check request
            
        Returns:
            bool: True if the endpoint is considered healthy, False otherwise
        """
        # HTTP 200 is always healthy
        if response.status_code == 200:
            return True
            
        # HTTP 400 is healthy only if it has JSON-RPC error code -32600
        if response.status_code == 400:
            try:
                # Parse the JSON response
                response_data = response.json()
                
                # Check for error dictionary with code -32600 (standard MCP error)
                if isinstance(response_data.get("error"), dict):
                    error = response_data["error"]
                    if isinstance(error.get("code"), int) and error.get("code") == -32600:
                        return True
                    
                # Check for streamable-http no auth specific query parameter error
                if isinstance(response_data.get("error"), str):
                    error_msg = response_data["error"]
                    if "Missing required query parameter: strata_id or instance_id" in error_msg:
                        return True
                        
            except (ValueError, KeyError, TypeError):
                # If we can't parse JSON or the structure is wrong, treat as unhealthy
                pass
                
        # All other status codes are considered unhealthy
        return False


    def _is_mcp_endpoint_healthy(self, response) -> bool:
        """
        Determine if an MCP endpoint is healthy based on HTTP response.
        
        For MCP endpoints, we consider them healthy if:
        1. HTTP 200 OK - Normal successful response
        2. HTTP 400 Bad Request with specific JSON-RPC error indicating missing session ID
        
        The 400 status with "Missing session ID" error is considered healthy because:
        - It proves the MCP endpoint is reachable and functioning
        - The server is properly validating requests according to MCP protocol
        - It's rejecting our basic GET request because we're not providing a session ID
        - This is expected behavior for a working MCP server when accessed without proper session
        
        Args:
            response: httpx.Response object from the health check request
            
        Returns:
            bool: True if the endpoint is considered healthy, False otherwise
        """
        # HTTP 200 is always healthy
        if response.status_code == 200:
            return True
            
        # HTTP 400 is healthy only if it's the expected MCP session error
        if response.status_code == 400:
            try:
                # Parse the JSON response
                response_data = response.json()
                
                # Check for the specific JSON-RPC error indicating missing session ID
                # This is the expected response from a healthy MCP endpoint when accessed without session
                if (response_data.get("jsonrpc") == "2.0" and 
                    response_data.get("id") == "server-error" and 
                    isinstance(response_data.get("error"), dict)):
                    
                    error = response_data["error"]
                    if (error.get("code") == -32600 and 
                        "Missing session ID" in error.get("message", "")):
                        return True
                        
            except (ValueError, KeyError, TypeError):
                # If we can't parse JSON or the structure is wrong, treat as unhealthy
                pass
                
        # All other status codes (404, 500, etc.) are considered unhealthy
        return False
