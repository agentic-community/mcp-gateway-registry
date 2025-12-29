import logging
from typing import (
    Any,
)

from fastapi import (
    HTTPException,
    status,
)

logger = logging.getLogger(__name__)


async def _refresh_service_impl(
    service_path: str,
    user_context: dict[str, Any],
    server_service_obj: Any,
) -> dict[str, Any]:
    """Refresh service health and tool information (shared by UI + JSON endpoints)."""
    from ..search.service import faiss_service
    from ..health.service import health_service
    from ..core.mcp_client import mcp_client_service
    from ..core.nginx_service import nginx_service
    from ..auth.dependencies import user_has_ui_permission_for_service

    _ = mcp_client_service

    if not service_path.startswith("/"):
        service_path = "/" + service_path

    server_info = server_service_obj.get_server_info(service_path)
    if not server_info:
        raise HTTPException(status_code=404, detail="Service path not registered")

    service_name = server_info["server_name"]

    if not user_has_ui_permission_for_service(
        "health_check_service",
        service_name,
        user_context.get("ui_permissions", {}),
    ):
        logger.warning(
            "User %s attempted to refresh service %s without health_check_service permission",
            user_context.get("username"),
            service_name,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You do not have permission to refresh {service_name}",
        )

    if not user_context.get("is_admin", False):
        if not server_service_obj.user_can_access_server_path(
            service_path,
            user_context.get("accessible_servers", []),
        ):
            logger.warning(
                "User %s attempted to refresh service %s without access",
                user_context.get("username"),
                service_path,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this server",
            )

    is_enabled = server_service_obj.is_service_enabled(service_path)
    if not is_enabled:
        raise HTTPException(status_code=400, detail="Cannot refresh disabled service")

    proxy_pass_url = server_info.get("proxy_pass_url")
    if not proxy_pass_url:
        raise HTTPException(status_code=500, detail="Service has no proxy URL configured")

    logger.info(
        "Refreshing service %s at %s by user '%s'",
        service_path,
        proxy_pass_url,
        user_context.get("username"),
    )

    try:
        status_str, last_checked_dt = await health_service.perform_immediate_health_check(
            service_path,
        )
        last_checked_iso = last_checked_dt.isoformat() if last_checked_dt else None
        logger.info(
            "Manual refresh health check for %s completed. Status: %s",
            service_path,
            status_str,
        )

        logger.info(
            "Regenerating Nginx config after manual refresh for %s...",
            service_path,
        )
        enabled_servers = {
            path: server_service_obj.get_server_info(path)
            for path in server_service_obj.get_enabled_services()
        }
        await nginx_service.generate_config_async(enabled_servers)

    except Exception as e:
        logger.error("ERROR during manual refresh check for %s: %s", service_path, e)
        await health_service.broadcast_health_update(service_path)
        raise HTTPException(status_code=500, detail=f"Refresh check failed: {e}") from e

    await faiss_service.add_or_update_service(service_path, server_info, is_enabled)
    await health_service.broadcast_health_update(service_path)

    logger.info("Service '%s' refreshed by user '%s'", service_path, user_context.get("username"))
    return {
        "message": f"Service {service_path} refreshed successfully",
        "service_path": service_path,
        "status": status_str,
        "last_checked_iso": last_checked_iso,
        "num_tools": server_info.get("num_tools", 0),
    }
