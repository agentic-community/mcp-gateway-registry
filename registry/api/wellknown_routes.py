import logging
from typing import Optional

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

from ..core.config import settings, RegistryMode
from ..services.server_service import server_service
from ..health.service import health_service
from ..constants import HealthStatus

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/mcp-servers")
async def get_wellknown_mcp_servers(
    request: Request, user_context: Optional[dict] = None
) -> JSONResponse:
    """
    Main endpoint handler for /.well-known/mcp-servers
    Returns JSON with all discoverable MCP servers
    """
    # Step 1: Check if discovery is enabled
    if not settings.enable_wellknown_discovery:
        raise HTTPException(status_code=404, detail="Well-known discovery is disabled")

    # Step 1.5: In skills-only mode, return empty server list
    if settings.registry_mode == RegistryMode.SKILLS_ONLY:
        response_data = {
            "version": "1.0",
            "servers": [],
            "registry": {
                "name": "Enterprise MCP Gateway (Skills Only)",
                "description": "Skills-only registry mode - no MCP servers available",
                "version": "1.0.0",
                "contact": {
                    "url": str(request.base_url).rstrip("/"),
                    "support": "mcp-support@company.com",
                },
            },
        }
        headers = {
            "Cache-Control": f"public, max-age={settings.wellknown_cache_ttl}",
            "Content-Type": "application/json",
        }
        logger.info("Returning empty server list - skills-only mode")
        return JSONResponse(content=response_data, headers=headers)

    # Step 2: Get all servers from server_service
    all_servers = await server_service.get_all_servers()

    # Step 3: Filter based on discoverability and permissions
    discoverable_servers = []
    for server_path, server_info in all_servers.items():
        # For now, include all enabled servers
        # TODO: Add discoverability flag to server configs if needed
        if await server_service.is_service_enabled(server_path):
            formatted_server = _format_server_discovery(server_info, request)
            discoverable_servers.append(formatted_server)

    # Step 4: Format response
    response_data = {
        "version": "1.0",
        "servers": discoverable_servers,
        "registry": {
            "name": "Enterprise MCP Gateway",
            "description": "Centralized MCP server registry for enterprise tools",
            "version": "1.0.0",
            "contact": {
                "url": str(request.base_url).rstrip("/"),
                "support": "mcp-support@company.com",
            },
        },
    }

    # Step 5: Return JSONResponse with cache headers
    headers = {
        "Cache-Control": f"public, max-age={settings.wellknown_cache_ttl}",
        "Content-Type": "application/json",
    }

    logger.info(f"Returned {len(discoverable_servers)} servers for well-known discovery")
    return JSONResponse(content=response_data, headers=headers)


def _format_server_discovery(server_info: dict, request: Request) -> dict:
    """Format individual server for discovery response"""
    server_path = server_info.get("path", "")
    server_name = server_info.get("server_name", server_path)
    description = server_info.get("description", "MCP Server")

    # Generate dynamic URL based on request host and server config
    server_url = _get_server_url(server_path, request, server_info)

    # Get transport type from config
    transport_type = _get_transport_type(server_info)

    # Get authentication requirements
    auth_info = _get_authentication_info(server_info)

    # Get first 5 tools as preview
    tools_preview = _get_tools_preview(server_info, max_tools=5)

    # Get actual health status from health service
    health_status = _get_normalized_health_status(server_path)

    return {
        "name": server_name,
        "description": description,
        "url": server_url,
        "transport": transport_type,
        "authentication": auth_info,
        "capabilities": ["tools", "resources"],
        "health_status": health_status,
        "tools_preview": tools_preview,
    }


def _get_server_url(server_path: str, request: Request, server_info: dict = None) -> str:
    """Generate full URL for MCP server based on request host and server config.

    Priority:
    1. If server_info has mcp_endpoint, use it as the full URL
    2. Otherwise, construct URL from request host + server_path + /mcp
    """
    # Check if server has explicit mcp_endpoint configured
    if server_info and server_info.get("mcp_endpoint"):
        return server_info.get("mcp_endpoint")

    # Get host from request headers
    host = request.headers.get("host", "localhost:7860")

    # Get protocol (http/https) from X-Forwarded-Proto or scheme
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)

    # Clean up server path (remove leading and trailing slashes)
    clean_path = server_path.strip("/")

    # Return formatted URL with default /mcp suffix
    return f"{proto}://{host}/{clean_path}/mcp"


def _get_transport_type(server_config: dict) -> str:
    """Determine transport type (sse or streamable-http)"""
    # Check server configuration for transport setting
    # Default to "streamable-http" if not specified
    return server_config.get("transport", "streamable-http")


def _get_authentication_info(server_info: dict) -> dict:
    """Extract authentication requirements for server.

    Reads auth_scheme (the new field). Legacy auth_type is migrated to
    auth_scheme at read time by the service layer, so we only need to
    check auth_scheme here.
    """
    auth_scheme = server_info.get("auth_scheme", "none")
    auth_provider = server_info.get("auth_provider", "default")

    if auth_scheme == "bearer":
        return {
            "type": "oauth2",
            "required": True,
            "authorization_url": "/auth/oauth/authorize",
            "provider": auth_provider,
            "scopes": ["mcp:read", f"{auth_provider}:read"],
        }
    elif auth_scheme == "api_key":
        header_name = server_info.get("auth_header_name", "X-API-Key")
        return {"type": "api-key", "required": True, "header": header_name}
    else:
        return {"type": "none", "required": False}


def _get_tools_preview(server_info: dict, max_tools: int = 5) -> list:
    """Get limited list of tools for discovery preview"""
    # Extract tools from server_info
    tools = server_info.get("tool_list", [])

    # Return first N tools with name and description
    preview_tools = []
    for tool in tools[:max_tools]:
        if isinstance(tool, dict):
            # Try to get description from parsed_description.main first, then fall back to description field
            description = tool.get("parsed_description", {}).get(
                "main", tool.get("description", "No description available")
            )
            preview_tools.append({"name": tool.get("name", "unknown"), "description": description})
        elif isinstance(tool, str):
            # Handle case where tools are just strings
            preview_tools.append({"name": tool, "description": "No description available"})

    return preview_tools


def _get_normalized_health_status(server_path: str) -> str:
    """
    Get normalized health status for a server from health service.

    Normalizes detailed status strings (e.g., "unhealthy: timeout") to simple
    values ("unhealthy") for cleaner client consumption in discovery responses.

    Args:
        server_path: The server path to get health status for

    Returns:
        Normalized health status string: "healthy", "unhealthy", "disabled", or "unknown"
    """
    # Get raw status from health service
    raw_status = health_service.server_health_status.get(server_path, HealthStatus.UNKNOWN)

    # Normalize status to clean values for client consumption
    if isinstance(raw_status, str):
        status_lower = raw_status.lower()
        if "unhealthy" in status_lower or "error" in status_lower:
            return "unhealthy"
        elif "healthy" in status_lower:
            return "healthy"
        elif "disabled" in status_lower:
            return "disabled"
        elif "checking" in status_lower:
            return "unknown"
        else:
            return raw_status

    return str(raw_status) if raw_status else "unknown"
