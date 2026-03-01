"""Client for MCP Registry API - tool discovery and search."""

import json
import logging
import time
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import aiohttp
from pydantic import (
    BaseModel,
    Field,
)


# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


class MatchingTool(BaseModel):
    """Tool matching result from semantic search.

    Note: inputSchema is NOT included here to avoid duplication.
    Full tool details including inputSchema are in the tools[] array.
    """

    tool_name: str = Field(..., description="Name of the matching tool")
    description: Optional[str] = Field(None, description="Tool description")
    relevance_score: float = Field(0.0, ge=0.0, le=1.0, description="Relevance score")
    match_context: Optional[str] = Field(None, description="Match context")


class ServerSearchResult(BaseModel):
    """MCP Server search result from semantic search."""

    path: str = Field(..., description="Server path in registry")
    server_name: str = Field(..., description="Server name")
    description: Optional[str] = Field(None, description="Server description")
    tags: List[str] = Field(default_factory=list, description="Server tags")
    num_tools: int = Field(0, description="Number of tools")
    is_enabled: bool = Field(False, description="Whether server is enabled")
    relevance_score: float = Field(0.0, ge=0.0, le=1.0, description="Relevance score")
    match_context: Optional[str] = Field(None, description="Match context")
    matching_tools: List[MatchingTool] = Field(
        default_factory=list, description="Tools matching the query"
    )


class ToolSearchResult(BaseModel):
    """Tool search result from semantic search."""

    server_path: str = Field(..., description="Server path in registry")
    server_name: str = Field(..., description="Server name")
    tool_name: str = Field(..., description="Tool name")
    description: Optional[str] = Field(None, description="Tool description")
    inputSchema: Optional[Dict[str, Any]] = Field(None, description="JSON Schema for tool input")
    relevance_score: float = Field(0.0, ge=0.0, le=1.0, description="Relevance score")
    match_context: Optional[str] = Field(None, description="Match context")


class SearchResponse(BaseModel):
    """Response from semantic search API."""

    query: str = Field(..., description="Original query")
    servers: List[ServerSearchResult] = Field(default_factory=list, description="Matching servers")
    tools: List[ToolSearchResult] = Field(default_factory=list, description="Matching tools")
    total_servers: int = Field(0, description="Total matching servers")
    total_tools: int = Field(0, description="Total matching tools")


class RegistryClient:
    """Client for MCP Registry API operations."""

    def __init__(
        self,
        registry_url: str,
        jwt_token: Optional[str] = None,
        keycloak_url: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        realm: str = "mcp-gateway",
    ) -> None:
        """
        Initialize the Registry Client.

        Args:
            registry_url: Base URL of the MCP Registry (e.g., https://mcpgateway.ddns.net)
            jwt_token: Pre-generated JWT token (bypasses M2M auth)
            keycloak_url: Keycloak URL for M2M token generation
            client_id: OAuth client ID
            client_secret: OAuth client secret
            realm: Keycloak realm name
        """
        self.registry_url = registry_url.rstrip("/")
        self.jwt_token = jwt_token
        self.keycloak_url = keycloak_url.rstrip("/") if keycloak_url else None
        self.client_id = client_id
        self.client_secret = client_secret
        self.realm = realm

        # Token caching
        self._cached_token: Optional[str] = None
        self._token_expires_at: float = 0

        if jwt_token:
            logger.info(f"RegistryClient initialized with JWT token for {registry_url}")
        elif keycloak_url:
            logger.info(f"RegistryClient initialized with M2M credentials for {registry_url}")
        else:
            logger.warning("RegistryClient initialized without authentication")

    async def _get_token(self) -> str:
        """
        Get or refresh the authentication token.

        Returns:
            JWT access token

        Raises:
            Exception: If token acquisition fails
        """
        # Use direct JWT token if provided
        if self.jwt_token:
            logger.debug("Using direct JWT token")
            return self.jwt_token

        # Check cached token validity (with 60s safety margin)
        current_time = time.time()
        if self._cached_token and current_time < self._token_expires_at - 60:
            logger.debug("Using cached token")
            return self._cached_token

        # Need to fetch new token from Keycloak
        if not self.keycloak_url or not self.client_id or not self.client_secret:
            raise ValueError("M2M credentials required but not provided")

        token_url = f"{self.keycloak_url}/realms/{self.realm}/protocol/openid-connect/token"
        logger.debug(f"Requesting new token from {token_url}")

        async with aiohttp.ClientSession() as session:
            data = {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            }

            try:
                async with session.post(token_url, data=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Token request failed: {response.status} - {error_text}")
                        raise Exception(f"Failed to get token: {response.status}")

                    token_data = await response.json()
                    self._cached_token = token_data["access_token"]
                    expires_in = token_data.get("expires_in", 300)
                    self._token_expires_at = current_time + expires_in

                    logger.info(f"Token acquired, expires in {expires_in}s")
                    return self._cached_token

            except aiohttp.ClientError as e:
                logger.error(f"Network error getting token: {e}")
                raise Exception(f"Network error: {e}")

    async def search_tools(
        self,
        query: str,
        max_results: int = 10,
        entity_types: Optional[List[str]] = None,
    ) -> SearchResponse:
        """
        Search for MCP tools using semantic search.

        Args:
            query: Natural language search query
            max_results: Maximum number of results to return
            entity_types: Entity types to search (mcp_server, tool, a2a_agent)

        Returns:
            SearchResponse with matching servers and tools

        Raises:
            Exception: If search fails
        """
        logger.info(f"Semantic search: '{query}' (max_results={max_results})")

        token = await self._get_token()
        search_url = f"{self.registry_url}/api/search/semantic"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        body = {
            "query": query,
            "max_results": max_results,
        }

        if entity_types:
            body["entity_types"] = entity_types

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    search_url,
                    headers=headers,
                    json=body,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Search failed: {response.status} - {error_text}")
                        raise Exception(f"Search failed: {response.status} - {error_text}")

                    result = await response.json()
                    logger.info(
                        f"Search returned {result.get('total_servers', 0)} servers, "
                        f"{result.get('total_tools', 0)} tools"
                    )

                    # Log full response for debugging
                    logger.info(
                        f"Full search API response:\n{json.dumps(result, indent=2, default=str)}"
                    )

                    return SearchResponse(**result)

            except aiohttp.ClientError as e:
                logger.error(f"Network error during search: {e}")
                raise Exception(f"Network error: {e}")

    async def get_server_info(
        self,
        server_path: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific MCP server.

        Uses the /api/servers endpoint with query parameter to find the server.

        Args:
            server_path: Path of the server in the registry

        Returns:
            Server information dict or None if not found
        """
        logger.info(f"Getting server info for: {server_path}")

        token = await self._get_token()
        # Normalize path - remove leading/trailing slashes
        clean_path = server_path.strip("/")

        # Use the servers list endpoint with query to find the specific server
        server_url = f"{self.registry_url}/api/servers"

        headers = {
            "Authorization": f"Bearer {token}",
        }

        params = {
            "query": clean_path,
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    server_url,
                    headers=headers,
                    params=params,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Get servers failed: {response.status} - {error_text}")
                        return None

                    result = await response.json()

                    # Find the matching server in the results
                    servers = result if isinstance(result, list) else result.get("servers", [])
                    for server in servers:
                        srv_path = server.get("path", "").strip("/")
                        if srv_path == clean_path:
                            logger.info(f"Got server info for {server_path}")
                            return server

                    logger.warning(f"Server not found in results: {server_path}")
                    return None

            except aiohttp.ClientError as e:
                logger.error(f"Network error getting server info: {e}")
                return None


def _format_tool_result(
    tool: ToolSearchResult,
) -> Dict[str, Any]:
    """
    Format a tool search result for display to the agent.

    The search API returns inputSchema directly, so no additional server lookup is needed.

    Args:
        tool: Tool search result

    Returns:
        Formatted tool information dict
    """
    result = {
        "tool_name": tool.tool_name,
        "server_path": tool.server_path,
        "server_name": tool.server_name,
        "description": tool.description or "No description available",
        "relevance_score": tool.relevance_score,
        "supported_transports": ["streamable_http"],
    }

    # Use inputSchema from search result if available
    if tool.inputSchema:
        result["tool_schema"] = tool.inputSchema

    return result


def _format_server_result(
    server: ServerSearchResult,
) -> Dict[str, Any]:
    """
    Format a server search result for display to the agent.

    Args:
        server: Server search result

    Returns:
        Formatted server information dict
    """
    matching_tools = []
    for t in server.matching_tools:
        tool_info = {
            "tool_name": t.tool_name,
            "description": t.description,
            "relevance_score": t.relevance_score,
        }
        # Note: inputSchema is available in the tools[] array, not matching_tools
        matching_tools.append(tool_info)

    return {
        "server_path": server.path,
        "server_name": server.server_name,
        "description": server.description or "No description available",
        "tags": server.tags,
        "num_tools": server.num_tools,
        "is_enabled": server.is_enabled,
        "relevance_score": server.relevance_score,
        "matching_tools": matching_tools,
    }
