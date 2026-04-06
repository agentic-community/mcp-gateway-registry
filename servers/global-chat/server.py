"""
Global Chat MCP Server — AI Agent Discovery across 15+ protocols.

This server provides tools for discovering AI agents across multiple registries
and protocols (MCP, A2A, agents.txt, ACDP, and more). It queries the Global Chat
directory of 100K+ agents and exposes search, lookup, and validation capabilities.

API documentation: https://global-chat.io
npm package: @global-chat/mcp-server
GitHub: https://github.com/geetchoubey/global-chat
"""

import argparse
import logging
import os
from typing import Annotated, Any

import httpx
from fastmcp import FastMCP
from pydantic import Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

# Constants
API_BASE_URL = "https://global-chat.io/api"
REQUEST_TIMEOUT = 15.0


def _parse_arguments():
    """Parse command line arguments with defaults matching environment variables."""
    parser = argparse.ArgumentParser(description="Global Chat MCP Server")

    parser.add_argument(
        "--port",
        type=str,
        default=os.environ.get("MCP_SERVER_LISTEN_PORT", "9010"),
        help="Port for the MCP server to listen on (default: 9010)",
    )

    parser.add_argument(
        "--transport",
        type=str,
        default=os.environ.get("MCP_TRANSPORT", "streamable-http"),
        choices=["sse", "streamable-http"],
        help="Transport type for the MCP server (default: streamable-http)",
    )

    return parser.parse_args()


# Parse arguments at module level
args = _parse_arguments()

logger.info(f"Parsed arguments - port: {args.port}, transport: {args.transport}")

# Initialize FastMCP server
mcp = FastMCP("GlobalChatServer", host="127.0.0.1", port=int(args.port))
mcp.settings.mount_path = "/global-chat"


@mcp.prompt()
def system_prompt_for_agent(task: str) -> str:
    """
    Generates a system prompt for an AI Agent that wants to discover other agents.

    Args:
        task (str): The task or operation the agent wants to perform.

    Returns:
        str: A formatted system prompt for the AI Agent.
    """
    return f"""
You are an AI agent that needs to discover other AI agents or MCP servers.
You can use the Global Chat discovery tools to search across 15+ agent protocols
and 100K+ registered agents.

The task you need to perform is: {task}

Available tools:
- search_agents: Search for agents by keyword across all protocols
- get_agent: Get detailed information about a specific agent
- list_protocols: List all supported agent protocols
- validate_agents_txt: Validate an agents.txt file at a given URL
"""


async def _api_request(endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Make a request to the Global Chat API.

    Args:
        endpoint: API endpoint path
        params: Query parameters

    Returns:
        API response as dictionary

    Raises:
        Exception: If the API request fails
    """
    url = f"{API_BASE_URL}/{endpoint.lstrip('/')}"
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def search_agents(
    query: Annotated[str, Field(description="Search query for finding agents (e.g. 'code review', 'data analysis')")],
    protocol: Annotated[
        str | None,
        Field(
            default=None,
            description="Filter by protocol: mcp, a2a, agents-txt, acdp, etc.",
        ),
    ] = None,
    limit: Annotated[
        int,
        Field(default=20, ge=1, le=100, description="Maximum number of results to return"),
    ] = 20,
) -> dict[str, Any]:
    """
    Search for AI agents across 15+ protocols and 100K+ registered agents.

    Searches the Global Chat directory which aggregates agents from MCP registries,
    A2A protocol, agents.txt, ACDP, and other agent discovery protocols.

    Args:
        query: Search terms for finding agents
        protocol: Optional protocol filter
        limit: Maximum results (1-100, default 20)

    Returns:
        Dict containing matching agents with their metadata

    Raises:
        Exception: If the search request fails
    """
    try:
        logger.info(f"Searching agents: query='{query}', protocol={protocol}, limit={limit}")
        params: dict[str, Any] = {"q": query, "limit": limit}
        if protocol:
            params["protocol"] = protocol
        result = await _api_request("search", params)
        logger.info(f"Search returned {len(result.get('agents', []))} results")
        return result
    except Exception as e:
        logger.error(f"Error searching agents: {e!s}")
        raise Exception(f"Failed to search agents: {e!s}")


@mcp.tool()
async def get_agent(
    agent_id: Annotated[str, Field(description="The unique identifier of the agent to look up")],
) -> dict[str, Any]:
    """
    Get detailed information about a specific agent.

    Returns full metadata including the agent's protocol, capabilities,
    endpoint URLs, and registry source.

    Args:
        agent_id: The agent's unique identifier

    Returns:
        Dict containing the agent's full metadata

    Raises:
        Exception: If the lookup fails
    """
    try:
        logger.info(f"Looking up agent: {agent_id}")
        result = await _api_request(f"agents/{agent_id}")
        return result
    except Exception as e:
        logger.error(f"Error looking up agent: {e!s}")
        raise Exception(f"Failed to get agent: {e!s}")


@mcp.tool()
async def list_protocols() -> dict[str, Any]:
    """
    List all supported agent discovery protocols.

    Returns the protocols aggregated by Global Chat, including MCP, A2A,
    agents.txt, ACDP, and others, with counts of registered agents per protocol.

    Returns:
        Dict containing protocol names and agent counts

    Raises:
        Exception: If the request fails
    """
    try:
        logger.info("Listing supported protocols")
        result = await _api_request("protocols")
        return result
    except Exception as e:
        logger.error(f"Error listing protocols: {e!s}")
        raise Exception(f"Failed to list protocols: {e!s}")


@mcp.tool()
async def validate_agents_txt(
    url: Annotated[
        str,
        Field(description="URL of the agents.txt file to validate (e.g. 'https://example.com/agents.txt')"),
    ],
) -> dict[str, Any]:
    """
    Validate an agents.txt file at a given URL.

    Checks the agents.txt file for compliance with the agents.txt specification,
    reporting any syntax errors, missing required fields, or format issues.

    Args:
        url: Full URL to the agents.txt file

    Returns:
        Dict containing validation results with errors and warnings

    Raises:
        Exception: If the validation request fails
    """
    try:
        logger.info(f"Validating agents.txt at: {url}")
        result = await _api_request("validate", {"url": url})
        return result
    except Exception as e:
        logger.error(f"Error validating agents.txt: {e!s}")
        raise Exception(f"Failed to validate agents.txt: {e!s}")


@mcp.resource("config://app")
def get_config() -> str:
    """Static configuration data for the Global Chat server."""
    return """
Global Chat MCP Server Configuration:
- Server Name: Global Chat Agent Discovery
- API Base: https://global-chat.io/api
- Available Tools: search_agents, get_agent, list_protocols, validate_agents_txt
- Transport: streamable-http
- Description: Cross-protocol AI agent discovery across 15+ registries
- Protocols: MCP, A2A, agents.txt, ACDP, and 10+ more
- Directory Size: 100,000+ agents across 15+ registries
- Website: https://global-chat.io
- npm: @global-chat/mcp-server
- GitHub: https://github.com/geetchoubey/global-chat
"""


def main():
    endpoint = "/mcp" if args.transport == "streamable-http" else "/sse"
    logger.info(f"Starting Global Chat MCP server on port {args.port} with transport {args.transport}")
    logger.info(f"Server will be available at: http://localhost:{args.port}{endpoint}")
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
