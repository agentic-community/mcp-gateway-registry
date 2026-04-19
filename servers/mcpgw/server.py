"""MCP Gateway Interaction Server (mcpgw).

This MCP server provides tools to interact with the MCP Gateway Registry API.
It acts as a thin protocol adapter, translating MCP tool calls into registry HTTP requests.

All tools require bearer token authentication via the Authorization header.
"""

import logging
import os
from typing import Any

import httpx
from fastmcp import Context, FastMCP
from models import AgentInfo, RegistryStats, ServerInfo, SkillInfo, ToolSearchResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

# Get registry URL from environment (used inside Docker containers)
# This is the internal registry URL that mcpgw uses to communicate with the registry
# Example: http://registry:7860 (Docker), http://registry.namespace.svc.cluster.local:8000 (K8s)
REGISTRY_URL = os.getenv("REGISTRY_BASE_URL", "http://localhost")

# Input validation constants
MAX_QUERY_LENGTH: int = 500
MIN_TOP_N: int = 1
MAX_TOP_N: int = 50

logger.info(f"Registry URL: {REGISTRY_URL}")

# Initialize FastMCP server
mcp = FastMCP("mcpgw")


def _validate_top_n(top_n: int) -> int:
    """Validate top_n parameter is within acceptable bounds.

    Args:
        top_n: Number of results to return

    Returns:
        Validated top_n value

    Raises:
        ValueError: If top_n is out of bounds
    """
    if not isinstance(top_n, int) or top_n < MIN_TOP_N or top_n > MAX_TOP_N:
        raise ValueError(f"top_n must be an integer between {MIN_TOP_N} and {MAX_TOP_N}")
    return top_n


def _validate_query(query: str) -> str:
    """Validate query parameter.

    Args:
        query: Search query string

    Returns:
        Validated and trimmed query

    Raises:
        ValueError: If query is empty or too long
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters")

    return query.strip()


def _extract_bearer_token(ctx: Context | None) -> str:
    """Extract bearer token from FastMCP context via Starlette Request.

    Supports both standard Authorization header and MCP Gateway's X-Authorization header.

    Args:
        ctx: FastMCP context containing request information

    Returns:
        Bearer token string

    Raises:
        ValueError: If token cannot be extracted or is missing
    """
    if not ctx:
        raise ValueError("Authentication required: Context is None")

    try:
        # Access the Starlette Request object from request_context
        if hasattr(ctx, "request_context") and ctx.request_context:
            request = ctx.request_context.request
            if request and hasattr(request, "headers"):
                # Try standard Authorization header first (case-insensitive)
                auth_header = request.headers.get("authorization")

                # If not found, try MCP Gateway's X-Authorization header
                if not auth_header:
                    auth_header = request.headers.get("x-authorization")

                if auth_header and auth_header.lower().startswith("bearer "):
                    token = auth_header.split(" ", 1)[1]
                    logger.debug(f"Successfully extracted token (length: {len(token)})")
                    return token

                raise ValueError(
                    "Authorization or X-Authorization header not found or not a Bearer token"
                )
            else:
                raise ValueError("Request object or headers not found in request_context")
        else:
            raise ValueError("request_context not available in Context")

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Failed to extract token: {e}", exc_info=True)
        raise ValueError(f"Failed to extract bearer token: {e}") from e


@mcp.tool()
async def list_services(ctx: Context | None = None) -> dict[str, Any]:
    """
    List all MCP servers registered in the gateway.

    Returns:
        Dictionary containing services, total_count, enabled_count, and status
    """
    logger.info("list_services called")

    try:
        token = _extract_bearer_token(ctx)
        # Use X-Authorization header for internal registry API calls
        headers = {"X-Authorization": f"Bearer {token}"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{REGISTRY_URL}/api/servers", headers=headers)
            response.raise_for_status()
            data = response.json()

        # Parse response
        if isinstance(data, dict) and "servers" in data:
            servers = data["servers"]
        elif isinstance(data, list):
            servers = data
        else:
            servers = []

        # Validate and convert to ServerInfo models
        services = []
        for s in servers:
            try:
                services.append(ServerInfo(**s).model_dump())
            except Exception as e:
                logger.warning(f"Failed to parse server {s.get('path', 'unknown')}: {e}")
        enabled_count = sum(1 for s in services if s.get("enabled"))

        return {
            "services": services,
            "total_count": len(services),
            "enabled_count": enabled_count,
            "status": "success",
        }

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {
            "services": [],
            "total_count": 0,
            "error": str(e),
            "status": "failed",
        }
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code}")
        return {
            "services": [],
            "total_count": 0,
            "error": f"Registry API error: {e.response.status_code}",
            "status": "failed",
        }
    except Exception as e:
        logger.error(f"Failed to list services: {e}")
        return {
            "services": [],
            "total_count": 0,
            "error": str(e),
            "status": "failed",
        }


@mcp.tool()
async def list_agents(ctx: Context | None = None) -> dict[str, Any]:
    """
    List all agents registered in the gateway.

    Returns:
        Dictionary containing agents, total_count, and status
    """
    logger.info("list_agents called")

    try:
        token = _extract_bearer_token(ctx)
        # Use X-Authorization header for internal registry API calls
        headers = {"X-Authorization": f"Bearer {token}"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{REGISTRY_URL}/api/agents", headers=headers)
            response.raise_for_status()
            data = response.json()

        agents = data.get("agents", []) if isinstance(data, dict) else data
        agent_list = [AgentInfo(**a).model_dump() for a in agents]

        return {
            "agents": agent_list,
            "total_count": len(agent_list),
            "status": "success",
        }

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {
            "agents": [],
            "total_count": 0,
            "error": str(e),
            "status": "failed",
        }
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code}")
        return {
            "agents": [],
            "total_count": 0,
            "error": f"Registry API error: {e.response.status_code}",
            "status": "failed",
        }
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        return {
            "agents": [],
            "total_count": 0,
            "error": str(e),
            "status": "failed",
        }


@mcp.tool()
async def list_skills(ctx: Context | None = None) -> dict[str, Any]:
    """
    List all skills registered in the gateway.

    Returns:
        Dictionary containing skills, total_count, and status
    """
    logger.info("list_skills called")

    try:
        token = _extract_bearer_token(ctx)
        # Use X-Authorization header for internal registry API calls
        headers = {"X-Authorization": f"Bearer {token}"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{REGISTRY_URL}/api/skills", headers=headers)
            response.raise_for_status()
            data = response.json()

        skills = data.get("skills", []) if isinstance(data, dict) else data
        skill_list = [SkillInfo(**s).model_dump() for s in skills]

        return {
            "skills": skill_list,
            "total_count": len(skill_list),
            "status": "success",
        }

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {
            "skills": [],
            "total_count": 0,
            "error": str(e),
            "status": "failed",
        }
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code}")
        return {
            "skills": [],
            "total_count": 0,
            "error": f"Registry API error: {e.response.status_code}",
            "status": "failed",
        }
    except Exception as e:
        logger.error(f"Failed to list skills: {e}")
        return {
            "skills": [],
            "total_count": 0,
            "error": str(e),
            "status": "failed",
        }


@mcp.tool()
async def intelligent_tool_finder(
    query: str,
    top_n: int = 5,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """
    Search for tools using natural language semantic search.

    Args:
        query: Natural language description of what you want to do
        top_n: Number of results to return (default: 5, max: 50)

    Returns:
        Dictionary containing results, query, total_results, and status
    """
    logger.info(f"intelligent_tool_finder called: query={query}, top_n={top_n}")

    try:
        # Validate inputs
        query = _validate_query(query)
        top_n = _validate_top_n(top_n)
        token = _extract_bearer_token(ctx)
        # Use X-Authorization header for internal registry API calls
        headers = {"X-Authorization": f"Bearer {token}"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{REGISTRY_URL}/api/search/semantic",
                headers=headers,
                json={"query": query, "entity_types": ["mcp_server", "tool", "virtual_server"], "max_results": top_n},
            )
            response.raise_for_status()
            data = response.json()

        # Extract servers array from response
        servers = data.get("servers", []) if isinstance(data, dict) else []

        # Flatten matching_tools from all servers into ToolSearchResult objects
        result_list = []
        for server in servers:
            server_path = server.get("path", "")
            server_name = server.get("server_name", "")
            for tool in server.get("matching_tools", []):
                result_list.append(
                    ToolSearchResult(
                        tool_name=tool.get("tool_name", ""),
                        server_name=server_name,
                        description=tool.get("description"),
                        score=tool.get("relevance_score"),
                        path=server_path,
                    ).model_dump()
                )

        # Enforce client-side limit (safety net in case registry returns more)
        result_list = result_list[:top_n]

        return {
            "results": result_list,
            "query": query,
            "total_results": len(result_list),
            "status": "success",
        }

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {
            "results": [],
            "query": query,
            "total_results": 0,
            "error": str(e),
            "status": "failed",
        }
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code}")
        return {
            "results": [],
            "query": query,
            "total_results": 0,
            "error": f"Registry API error: {e.response.status_code}",
            "status": "failed",
        }
    except Exception as e:
        logger.error(f"Failed to search tools: {e}")
        return {
            "results": [],
            "query": query,
            "total_results": 0,
            "error": str(e),
            "status": "failed",
        }


@mcp.tool()
async def healthcheck(ctx: Context | None = None) -> dict[str, Any]:
    """
    Get registry health status and statistics.

    Returns:
        Dictionary containing health stats and status
    """
    logger.info("healthcheck called")

    try:
        token = _extract_bearer_token(ctx)
        # Use X-Authorization header for internal registry API calls
        headers = {"X-Authorization": f"Bearer {token}"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{REGISTRY_URL}/api/servers/health", headers=headers)
            response.raise_for_status()
            data = response.json()

        stats = RegistryStats(**data)
        return {**stats.model_dump(), "status": "success"}

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {
            "health_status": "error",
            "error": str(e),
            "status": "failed",
        }
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code}")
        return {
            "health_status": "error",
            "error": f"Registry API error: {e.response.status_code}",
            "status": "failed",
        }
    except Exception as e:
        logger.error(f"Failed to get health status: {e}")
        return {
            "health_status": "error",
            "error": str(e),
            "status": "failed",
        }


if __name__ == "__main__":
    import os

    logger.info("Starting mcpgw server")

    # Use HTTP transport if PORT is set (Docker container), otherwise stdio
    port = os.environ.get("PORT")
    if port:
        # Use configurable host with secure default (127.0.0.1)
        # Set HOST=0.0.0.0 in environment for Docker deployments
        host = os.environ.get("HOST", "127.0.0.1")
        logger.info(f"Running in HTTP mode on {host}:{port}")
        mcp.run(transport="streamable-http", host=host, port=int(port))
    else:
        logger.info("Running in stdio mode")
        mcp.run(transport="stdio")
