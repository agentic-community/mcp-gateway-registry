import logging
from typing import (
    Annotated,
)

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Response,
    status,
)
from fastapi.responses import (
    JSONResponse,
)

from ..auth.dependencies import (
    nginx_proxied_auth,
)
from ..services.server_service import (
    server_service,
)
from .server_refresh_common import (
    _refresh_service_impl,
)
from .server_routes_common import (
    ServerCreateRequest,
    ServerUpdateRequest,
    _apply_remove_side_effects,
    _enforce_proxy_pass_url_allowlist,
    _enforce_upstream_oauth_provider_configured,
    _normalize_server_path,
    _normalize_upstream_auth_payload,
    _require_admin_user_context,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/servers")
async def get_servers_json(
    query: str | None = None,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Get servers data as JSON for the React frontend and external API."""
    logger.debug("[GET_SERVERS_DEBUG] Received user_context: %s", user_context)
    logger.debug("[GET_SERVERS_DEBUG] user_context type: %s", type(user_context))
    if user_context:
        logger.debug("[GET_SERVERS_DEBUG] Username: %s", user_context.get("username", "NOT PRESENT"))
        logger.debug("[GET_SERVERS_DEBUG] Scopes: %s", user_context.get("scopes", "NOT PRESENT"))
        logger.debug("[GET_SERVERS_DEBUG] Auth method: %s", user_context.get("auth_method", "NOT PRESENT"))

    service_data = []
    search_query = query.lower() if query else ""

    if user_context["is_admin"]:
        all_servers = server_service.get_all_servers()
    else:
        all_servers = server_service.get_all_servers_with_permissions(
            user_context["accessible_servers"],
        )

    sorted_server_paths = sorted(
        all_servers.keys(),
        key=lambda p: all_servers[p]["server_name"],
    )

    accessible_services = user_context.get("accessible_services", [])

    for path in sorted_server_paths:
        server_info = all_servers[path]
        server_name = server_info["server_name"]
        technical_name = path.strip("/")

        if "all" not in accessible_services and technical_name not in accessible_services:
            continue

        searchable_text = (
            f"{server_name.lower()} {server_info.get('description', '').lower()} "
            f"{' '.join(server_info.get('tags', []))}"
        )
        if search_query and search_query not in searchable_text:
            continue

        from ..health.service import health_service

        health_data = health_service._get_service_health_data(path)

        service_data.append(
            {
                "display_name": server_name,
                "path": path,
                "description": server_info.get("description", ""),
                "proxy_pass_url": server_info.get("proxy_pass_url", ""),
                "is_enabled": server_service.is_service_enabled(path),
                "tags": server_info.get("tags", []),
                "num_tools": server_info.get("num_tools", 0),
                "num_stars": server_info.get("num_stars", 0),
                "is_python": server_info.get("is_python", False),
                "license": server_info.get("license", "N/A"),
                "health_status": health_data["status"],
                "last_checked_iso": health_data["last_checked_iso"],
                "supported_transports": server_info.get("supported_transports", []),
                "upstream_auth": server_info.get("upstream_auth"),
                "upstream_credential_status": server_info.get("upstream_credential_status"),
            }
        )

    return {"servers": service_data}


@router.post("/servers")
async def create_server_json(
    payload: ServerCreateRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Register a new server via JSON (used by the React UI)."""
    from ..core.nginx_service import nginx_service
    from ..health.service import health_service
    from ..search.service import faiss_service

    ui_permissions = user_context.get("ui_permissions", {})
    register_permissions = ui_permissions.get("register_service", [])
    if not register_permissions:
        logger.warning(
            "User %s attempted to register service without register_service permission",
            user_context.get("username"),
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to register new services",
        )

    proxy_pass_url = payload.proxy_pass_url.strip()
    _enforce_proxy_pass_url_allowlist(proxy_pass_url=proxy_pass_url)

    path = _normalize_server_path(raw_path=payload.path)

    upstream_auth_payload = _normalize_upstream_auth_payload(
        upstream_auth=payload.upstream_auth,
        auth_type=None,
        auth_provider=None,
        headers=None,
    )
    _enforce_upstream_oauth_provider_configured(upstream_auth=upstream_auth_payload)

    server_entry = {
        "server_name": payload.name.strip(),
        "description": (payload.description or "").strip(),
        "path": path,
        "proxy_pass_url": proxy_pass_url,
        "tags": payload.tags or [],
        "num_tools": 0,
        "num_stars": 0,
        "is_python": False,
        "license": "N/A",
        "tool_list": [],
        "supported_transports": ["streamable-http"],
        "auth_type": "none",
        "upstream_auth": upstream_auth_payload,
    }

    existing_server = server_service.get_server_info(path)
    if existing_server and not payload.overwrite:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Service with path '{path}' already exists",
        )

    success = (
        server_service.update_server(path, server_entry)
        if existing_server and payload.overwrite
        else server_service.register_server(server_entry)
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Service with path '{path}' already exists or failed to save",
        )

    is_enabled = server_service.is_service_enabled(path)
    await faiss_service.add_or_update_service(path, server_entry, is_enabled)

    enabled_servers = {
        server_path: server_service.get_server_info(server_path)
        for server_path in server_service.get_enabled_services()
    }
    await nginx_service.generate_config_async(enabled_servers)

    await health_service.broadcast_health_update(path)

    health_data = health_service._get_service_health_data(path)

    return JSONResponse(
        status_code=201,
        content={
            "display_name": server_entry["server_name"],
            "path": path,
            "proxy_pass_url": proxy_pass_url,
            "description": server_entry.get("description", ""),
            "tags": server_entry.get("tags", []),
            "is_enabled": is_enabled,
            "health_status": health_data["status"],
            "last_checked_iso": health_data["last_checked_iso"],
            "num_tools": 0,
            "num_stars": 0,
            "is_python": False,
            "license": "N/A",
            "supported_transports": server_entry.get("supported_transports", []),
            "upstream_auth": server_entry.get("upstream_auth"),
        },
    )


@router.get("/servers/{service_path}")
async def get_server_json(
    service_path: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Get a single server's details as JSON (used by the React UI)."""
    from ..health.service import health_service

    path = _normalize_server_path(raw_path=service_path)

    server_info = server_service.get_server_info(path)
    if not server_info:
        raise HTTPException(status_code=404, detail="Service path not registered")

    if not user_context.get("is_admin", False):
        if not server_service.user_can_access_server_path(
            path,
            user_context.get("accessible_servers", []),
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this server",
            )

    health_data = health_service._get_service_health_data(path)
    is_enabled = server_service.is_service_enabled(path)

    return {
        "display_name": server_info.get("server_name", ""),
        "path": path,
        "proxy_pass_url": server_info.get("proxy_pass_url", ""),
        "description": server_info.get("description", ""),
        "tags": server_info.get("tags", []),
        "is_enabled": is_enabled,
        "health_status": health_data["status"],
        "last_checked_iso": health_data["last_checked_iso"],
        "num_tools": server_info.get("num_tools", 0),
        "num_stars": server_info.get("num_stars", 0),
        "is_python": server_info.get("is_python", False),
        "license": server_info.get("license", "N/A"),
        "supported_transports": server_info.get("supported_transports", []),
        "upstream_auth": server_info.get("upstream_auth"),
        "upstream_credential_status": server_info.get("upstream_credential_status"),
        "tools": server_info.get("tool_list", []),
        "metadata": server_info.get("metadata", {}),
    }


@router.put("/servers/{service_path}")
async def update_server_json(
    service_path: str,
    payload: ServerUpdateRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Update a server via JSON (used by the React UI)."""
    from ..auth.dependencies import user_has_ui_permission_for_service
    from ..core.nginx_service import nginx_service
    from ..health.service import health_service
    from ..search.service import faiss_service

    path = _normalize_server_path(raw_path=service_path)

    server_info = server_service.get_server_info(path)
    if not server_info:
        raise HTTPException(status_code=404, detail="Service path not registered")

    service_name = server_info.get("server_name", "")
    if not user_has_ui_permission_for_service(
        "modify_service",
        service_name,
        user_context.get("ui_permissions", {}),
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You do not have permission to modify {service_name}",
        )

    if not user_context.get("is_admin", False):
        if not server_service.user_can_access_server_path(
            path,
            user_context.get("accessible_servers", []),
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to edit this server",
            )

    if payload.proxy_pass_url is not None:
        _enforce_proxy_pass_url_allowlist(proxy_pass_url=payload.proxy_pass_url.strip())

    updated_server_entry = dict(server_info)
    if payload.name is not None:
        updated_server_entry["server_name"] = payload.name.strip()
    if payload.proxy_pass_url is not None:
        updated_server_entry["proxy_pass_url"] = payload.proxy_pass_url.strip()
    if payload.description is not None:
        updated_server_entry["description"] = payload.description
    if payload.tags is not None:
        updated_server_entry["tags"] = payload.tags
    if payload.upstream_auth is not None:
        upstream_auth_payload = _normalize_upstream_auth_payload(
            upstream_auth=payload.upstream_auth,
            auth_type=None,
            auth_provider=None,
            headers=None,
        )
        _enforce_upstream_oauth_provider_configured(upstream_auth=upstream_auth_payload)
        updated_server_entry["upstream_auth"] = upstream_auth_payload

    updated_server_entry["path"] = path

    success = server_service.update_server(path, updated_server_entry)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save updated server data")

    enabled_changed = False
    is_enabled = server_service.is_service_enabled(path)
    if payload.enabled is not None and payload.enabled != is_enabled:
        enabled_changed = True
        is_enabled = payload.enabled
        toggle_success = server_service.toggle_service(path, is_enabled)
        if not toggle_success:
            raise HTTPException(status_code=500, detail="Failed to toggle service")

        if is_enabled:
            await health_service.perform_immediate_health_check(path)
        else:
            logger.info("Service %s toggled OFF via JSON API", path)

    await faiss_service.add_or_update_service(path, updated_server_entry, is_enabled)

    enabled_servers = {
        server_path: server_service.get_server_info(server_path)
        for server_path in server_service.get_enabled_services()
    }
    await nginx_service.generate_config_async(enabled_servers)

    if enabled_changed:
        await health_service.broadcast_health_update(path)

    health_data = health_service._get_service_health_data(path)

    return {
        "display_name": updated_server_entry.get("server_name", ""),
        "path": path,
        "proxy_pass_url": updated_server_entry.get("proxy_pass_url", ""),
        "description": updated_server_entry.get("description", ""),
        "tags": updated_server_entry.get("tags", []),
        "is_enabled": is_enabled,
        "health_status": health_data["status"],
        "last_checked_iso": health_data["last_checked_iso"],
        "num_tools": updated_server_entry.get("num_tools", 0),
        "num_stars": updated_server_entry.get("num_stars", 0),
        "is_python": updated_server_entry.get("is_python", False),
        "license": updated_server_entry.get("license", "N/A"),
        "supported_transports": updated_server_entry.get("supported_transports", []),
        "upstream_auth": updated_server_entry.get("upstream_auth"),
        "upstream_credential_status": updated_server_entry.get("upstream_credential_status"),
    }


@router.delete("/servers/{service_path}", status_code=204)
async def delete_server_json(
    service_path: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Delete a server via JSON (admin-only)."""
    from ..core.nginx_service import nginx_service
    from ..health.service import health_service
    from ..search.service import faiss_service
    from ..utils.scopes_manager import remove_server_scopes

    _require_admin_user_context(user_context)

    path = _normalize_server_path(raw_path=service_path)

    server_info = server_service.get_server_info(path)
    if not server_info:
        raise HTTPException(status_code=404, detail="Service path not registered")

    success = server_service.remove_server(path)
    if not success:
        raise HTTPException(status_code=500, detail="Service removal failed")

    await _apply_remove_side_effects(
        service_path=path,
        server_service=server_service,
        faiss_service=faiss_service,
        health_service=health_service,
        nginx_service=nginx_service,
        remove_server_scopes=remove_server_scopes,
        scopes_error_log_level="warning",
        logger=logger,
    )

    return Response(status_code=204)


@router.post("/servers/{service_path}/refresh")
async def refresh_server_json(
    service_path: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Refresh a server via JSON (used by the React UI)."""
    return await _refresh_service_impl(
        service_path=service_path,
        user_context=user_context,
        server_service_obj=server_service,
    )
