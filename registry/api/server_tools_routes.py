import logging
from typing import (
    Annotated,
)

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    status,
)

from ..auth.dependencies import (
    enhanced_auth,
    nginx_proxied_auth,
)
from ..services.server_service import (
    server_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/tools/{service_path:path}")
async def get_service_tools(
    service_path: str,
    user_context: Annotated[dict, Depends(enhanced_auth)],
):
    """Get tool list for a service (filtered by permissions)."""
    from ..core.mcp_client import mcp_client_service
    from ..search.service import faiss_service

    if not service_path.startswith("/"):
        service_path = "/" + service_path

    # Handle special case for '/all' to return tools from all accessible servers
    if service_path == "/all":
        all_tools = []
        all_servers_tools = {}

        # Get servers based on user permissions
        if user_context["is_admin"]:
            all_servers = server_service.get_all_servers()
        else:
            all_servers = server_service.get_all_servers_with_permissions(
                user_context["accessible_servers"],
            )

        for path, server_info in all_servers.items():
            # For '/all', we can use cached data to avoid too many MCP calls
            tool_list = server_info.get("tool_list")

            if tool_list is not None and isinstance(tool_list, list):
                # Add server information to each tool
                server_tools = []
                for tool in tool_list:
                    # Create a copy of the tool with server info added
                    tool_with_server = dict(tool)
                    tool_with_server["server_path"] = path
                    tool_with_server["server_name"] = server_info.get("server_name", "Unknown")
                    server_tools.append(tool_with_server)

                all_tools.extend(server_tools)
                all_servers_tools[path] = server_tools

        return {
            "service_path": "all",
            "tools": all_tools,
            "servers": all_servers_tools,
        }

    # Handle specific server case - fetch live tools from MCP server
    server_info = server_service.get_server_info(service_path)
    if not server_info:
        raise HTTPException(status_code=404, detail="Service path not registered")

    # For non-admin users, check if they have access to this specific server
    if not user_context["is_admin"]:
        if not server_service.user_can_access_server_path(
            service_path,
            user_context["accessible_servers"],
        ):
            logger.warning(
                "User %s attempted to access tools for %s without access",
                user_context["username"],
                service_path,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this server",
            )

    # Check if service is enabled and healthy
    is_enabled = server_service.is_service_enabled(service_path)
    if not is_enabled:
        raise HTTPException(status_code=400, detail="Cannot fetch tools from disabled service")

    proxy_pass_url = server_info.get("proxy_pass_url")
    if not proxy_pass_url:
        raise HTTPException(status_code=500, detail="Service has no proxy URL configured")

    logger.info("Fetching live tools for %s from %s", service_path, proxy_pass_url)

    try:
        # Call MCP client to fetch fresh tools using server configuration
        tool_list = await mcp_client_service.get_tools_from_server_with_server_info(
            proxy_pass_url,
            server_info,
        )

        if tool_list is None:
            # If live fetch fails but we have cached tools, use those
            cached_tools = server_info.get("tool_list")
            if cached_tools is not None and isinstance(cached_tools, list):
                logger.warning(
                    "Failed to fetch live tools for %s, using cached tools",
                    service_path,
                )
                return {"service_path": service_path, "tools": cached_tools, "cached": True}
            raise HTTPException(
                status_code=503,
                detail="Failed to fetch tools from MCP server. Service may be unhealthy.",
            )

        # Update the server registry with the fresh tools
        new_tool_count = len(tool_list)
        current_tool_count = server_info.get("num_tools", 0)

        if current_tool_count != new_tool_count or server_info.get("tool_list") != tool_list:
            logger.info("Updating tool list for %s. New count: %s", service_path, new_tool_count)

            # Update server info with fresh tools
            updated_server_info = server_info.copy()
            updated_server_info["tool_list"] = tool_list
            updated_server_info["num_tools"] = new_tool_count

            # Save updated server info
            success = server_service.update_server(service_path, updated_server_info)
            if success:
                logger.info("Successfully updated tool list for %s", service_path)

                # Update FAISS index with new tool data
                await faiss_service.add_or_update_service(service_path, updated_server_info, is_enabled)
                logger.info("Updated FAISS index for %s", service_path)
            else:
                logger.error("Failed to save updated tool list for %s", service_path)

        return {"service_path": service_path, "tools": tool_list, "cached": False}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching tools for %s: %s", service_path, e)
        cached_tools = server_info.get("tool_list")
        if cached_tools is not None and isinstance(cached_tools, list):
            logger.warning(
                "Error fetching live tools for %s, falling back to cached tools: %s",
                service_path,
                e,
            )
            return {"service_path": service_path, "tools": cached_tools, "cached": True}
        raise HTTPException(status_code=500, detail=f"Error fetching tools: {str(e)}") from e


@router.get("/servers/tools/{service_path:path}")
async def get_service_tools_api(
    service_path: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    Get tool list for a service via JWT Bearer Token authentication (External API).

    This endpoint provides the same functionality as GET /tools/{service_path}
    but uses modern JWT Bearer token authentication.
    """
    logger.info(
        "API get tools request from user '%s' for path '%s'",
        user_context.get("username") if user_context else "unknown",
        service_path,
    )

    return await get_service_tools(service_path=service_path, user_context=user_context)

