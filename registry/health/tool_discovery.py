from __future__ import annotations

import asyncio
import logging
from typing import (
    Protocol,
)

logger = logging.getLogger(__name__)


class _HealthServiceProto(Protocol):
    async def broadcast_health_update(
        self,
        service_path: str | None = None,
    ): ...


async def update_tools_background(
    service: _HealthServiceProto,
    service_path: str,
    proxy_pass_url: str,
) -> None:
    """Update tool list in the background without blocking health checks."""
    try:
        logger.info("Starting background tool update for %s", service_path)
        from ..core.mcp_client import mcp_client_service
        from ..services.server_service import server_service

        await asyncio.sleep(0.5)

        server_info = server_service.get_server_info(service_path)
        logger.info("Fetching tools from %s for %s", proxy_pass_url, service_path)
        tool_list = await mcp_client_service.get_tools_from_server_with_server_info(
            proxy_pass_url,
            server_info,
        )
        logger.info(
            "Tool fetch result for %s: %s tools",
            service_path,
            len(tool_list) if tool_list else "None",
        )

        if tool_list is None:
            return

        new_tool_count = len(tool_list)
        current_server_info = server_service.get_server_info(service_path)
        if not current_server_info:
            return

        current_tool_count = current_server_info.get("num_tools", 0)
        current_tool_list = current_server_info.get("tool_list", [])
        if current_tool_count == new_tool_count and current_tool_list:
            return

        updated_server_info = current_server_info.copy()
        updated_server_info["tool_list"] = tool_list
        updated_server_info["num_tools"] = new_tool_count
        server_service.update_server(service_path, updated_server_info)

        try:
            from ..utils.scopes_manager import update_server_scopes

            tool_names = [tool["name"] for tool in tool_list if "name" in tool]
            await update_server_scopes(
                service_path,
                current_server_info.get("server_name", "Unknown"),
                tool_names,
            )
            logger.info(
                "Updated scopes for %s with %s discovered tools",
                service_path,
                len(tool_names),
            )
        except Exception as exc:
            logger.error(
                "Failed to update scopes for %s after tool discovery: %s",
                service_path,
                exc,
            )

        await service.broadcast_health_update(service_path)

    except Exception as exc:
        logger.warning("Failed to fetch tools for %s: %s", service_path, exc)

