import logging
from typing import (
    Annotated,
)

from fastapi import (
    APIRouter,
    Cookie,
    Depends,
    Form,
    HTTPException,
    Request,
    status,
)
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
)
from fastapi.templating import (
    Jinja2Templates,
)

from ..auth.dependencies import (
    enhanced_auth,
)
from ..core.config import (
    settings,
)
from ..services.server_service import (
    server_service,
)
from .server_refresh_common import (
    _refresh_service_impl,
)
from .server_routes_common import (
    _enforce_proxy_pass_url_allowlist,
)

logger = logging.getLogger(__name__)

router = APIRouter()

templates = Jinja2Templates(directory=settings.templates_dir)


@router.get("/", response_class=HTMLResponse)
async def read_root(
    request: Request,
    query: str | None = None,
    session: Annotated[str | None, Cookie(alias=settings.session_cookie_name)] = None,
):
    """Main dashboard page showing services based on user permissions."""
    if not session:
        logger.info("No session cookie at root route, redirecting to login")
        return RedirectResponse(url="/login", status_code=302)

    try:
        user_context = enhanced_auth(session)
    except HTTPException as e:
        logger.info("Authentication failed at root route: %s, redirecting to login", e.detail)
        return RedirectResponse(url="/login", status_code=302)

    from ..auth.dependencies import user_has_ui_permission_for_service

    def can_perform_action(permission: str, service_name: str) -> bool:
        return user_has_ui_permission_for_service(
            permission,
            service_name,
            user_context.get("ui_permissions", {}),
        )

    service_data = []
    search_query = query.lower() if query else ""

    if user_context["is_admin"]:
        all_servers = server_service.get_all_servers()
        logger.info(
            "Admin user %s accessing all %s servers",
            user_context["username"],
            len(all_servers),
        )
    else:
        all_servers = server_service.get_all_servers_with_permissions(
            user_context["accessible_servers"],
        )
        logger.info(
            "User %s accessing %s of %s total servers",
            user_context["username"],
            len(all_servers),
            len(server_service.get_all_servers()),
        )

    sorted_server_paths = sorted(
        all_servers.keys(),
        key=lambda p: all_servers[p]["server_name"],
    )

    accessible_services = user_context.get("accessible_services", [])
    logger.info(
        "DEBUG: User %s accessible_services: %s",
        user_context["username"],
        accessible_services,
    )
    logger.info(
        "DEBUG: User %s ui_permissions: %s",
        user_context["username"],
        user_context.get("ui_permissions", {}),
    )
    logger.info("DEBUG: User %s scopes: %s", user_context["username"], user_context.get("scopes", []))

    from ..health.service import health_service

    for path in sorted_server_paths:
        server_info = all_servers[path]
        server_name = server_info["server_name"]

        if "all" not in accessible_services and server_name not in accessible_services:
            logger.debug(
                "Filtering out service '%s' - user doesn't have list_service permission",
                server_name,
            )
            continue

        searchable_text = (
            f"{server_name.lower()} {server_info.get('description', '').lower()} "
            f"{' '.join(server_info.get('tags', []))}"
        )
        if search_query and search_query not in searchable_text:
            continue

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
            }
        )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "services": service_data,
            "username": user_context["username"],
            "user_context": user_context,
            "can_perform_action": can_perform_action,
        },
    )


@router.post("/toggle/{service_path:path}")
async def toggle_service_route(
    request: Request,
    service_path: str,
    enabled: Annotated[str | None, Form()] = None,
    user_context: Annotated[dict, Depends(enhanced_auth)] = None,
):
    """Toggle a service on/off (requires toggle_service UI permission)."""
    from ..search.service import faiss_service
    from ..health.service import health_service
    from ..core.nginx_service import nginx_service
    from ..auth.dependencies import user_has_ui_permission_for_service

    _ = request

    if not service_path.startswith("/"):
        service_path = "/" + service_path

    server_info = server_service.get_server_info(service_path)
    if not server_info:
        raise HTTPException(status_code=404, detail="Service path not registered")

    service_name = server_info["server_name"]

    if not user_has_ui_permission_for_service(
        "toggle_service",
        service_name,
        user_context.get("ui_permissions", {}),
    ):
        logger.warning(
            "User %s attempted to toggle service %s without toggle_service permission",
            user_context["username"],
            service_name,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You do not have permission to toggle {service_name}",
        )

    if not user_context["is_admin"]:
        if not server_service.user_can_access_server_path(
            service_path,
            user_context["accessible_servers"],
        ):
            logger.warning(
                "User %s attempted to toggle service %s without access",
                user_context["username"],
                service_path,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this server",
            )

    new_state = enabled == "on"
    success = server_service.toggle_service(service_path, new_state)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to toggle service")

    logger.info(
        "Toggled '%s' (%s) to %s by user '%s'",
        server_info["server_name"],
        service_path,
        new_state,
        user_context["username"],
    )

    status_str = "disabled"
    last_checked_iso = None
    if new_state:
        logger.info("Performing immediate health check for %s upon toggle ON...", service_path)
        try:
            status_str, last_checked_dt = await health_service.perform_immediate_health_check(
                service_path,
            )
            last_checked_iso = last_checked_dt.isoformat() if last_checked_dt else None
            logger.info("Immediate health check for %s completed. Status: %s", service_path, status_str)
        except Exception as e:
            logger.error("ERROR during immediate health check for %s: %s", service_path, e)
            status_str = f"error: immediate check failed ({type(e).__name__})"
    else:
        logger.info("Service %s toggled OFF. Status set to disabled.", service_path)

    await faiss_service.add_or_update_service(service_path, server_info, new_state)

    enabled_servers = {
        path: server_service.get_server_info(path)
        for path in server_service.get_enabled_services()
    }
    await nginx_service.generate_config_async(enabled_servers)

    await health_service.broadcast_health_update(service_path)

    return JSONResponse(
        status_code=200,
        content={
            "message": f"Toggle request for {service_path} processed.",
            "service_path": service_path,
            "new_enabled_state": new_state,
            "status": status_str,
            "last_checked_iso": last_checked_iso,
            "num_tools": server_info.get("num_tools", 0),
        },
    )


@router.post("/register")
async def register_service(
    name: Annotated[str, Form()],
    description: Annotated[str, Form()],
    path: Annotated[str, Form()],
    proxy_pass_url: Annotated[str, Form()],
    tags: Annotated[str, Form()] = "",
    num_tools: Annotated[int, Form()] = 0,
    num_stars: Annotated[int, Form()] = 0,
    is_python: Annotated[bool, Form()] = False,
    license_str: Annotated[str, Form(alias="license")] = "N/A",
    user_context: Annotated[dict, Depends(enhanced_auth)] = None,
):
    """Register a new service (requires register_service UI permission)."""
    from ..search.service import faiss_service
    from ..health.service import health_service
    from ..core.nginx_service import nginx_service

    ui_permissions = user_context.get("ui_permissions", {})
    register_permissions = ui_permissions.get("register_service", [])

    if not register_permissions:
        logger.warning(
            "User %s attempted to register service without register_service permission",
            user_context["username"],
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to register new services",
        )

    logger.info("Service registration request from user '%s'", user_context["username"])
    logger.info("Name: %s, Path: %s, URL: %s", name, path, proxy_pass_url)

    _enforce_proxy_pass_url_allowlist(proxy_pass_url=proxy_pass_url)

    if not path.startswith("/"):
        path = "/" + path

    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

    server_entry = {
        "server_name": name,
        "description": description,
        "path": path,
        "proxy_pass_url": proxy_pass_url,
        "tags": tag_list,
        "num_tools": num_tools,
        "num_stars": num_stars,
        "is_python": is_python,
        "license": license_str,
        "tool_list": [],
    }

    success = server_service.register_server(server_entry)
    if not success:
        return JSONResponse(
            status_code=400,
            content={"error": f"Service with path '{path}' already exists or failed to save"},
        )

    is_enabled = server_service.is_service_enabled(path)
    await faiss_service.add_or_update_service(path, server_entry, is_enabled)

    enabled_servers = {
        server_path: server_service.get_server_info(server_path)
        for server_path in server_service.get_enabled_services()
    }
    await nginx_service.generate_config_async(enabled_servers)

    await health_service.broadcast_health_update(path)

    logger.info(
        "New service registered: '%s' at path '%s' by user '%s'",
        name,
        path,
        user_context["username"],
    )

    return JSONResponse(
        status_code=201,
        content={
            "message": "Service registered successfully",
            "service": server_entry,
        },
    )


@router.get("/edit/{service_path:path}", response_class=HTMLResponse)
async def edit_server_form(
    request: Request,
    service_path: str,
    user_context: Annotated[dict, Depends(enhanced_auth)],
):
    """Show edit form for a service (requires modify_service UI permission)."""
    from ..auth.dependencies import user_has_ui_permission_for_service

    if not service_path.startswith("/"):
        service_path = "/" + service_path

    server_info = server_service.get_server_info(service_path)
    if not server_info:
        raise HTTPException(status_code=404, detail="Service path not found")

    service_name = server_info["server_name"]

    if not user_has_ui_permission_for_service(
        "modify_service",
        service_name,
        user_context.get("ui_permissions", {}),
    ):
        logger.warning(
            "User %s attempted to access edit form for %s without modify_service permission",
            user_context["username"],
            service_name,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You do not have permission to modify {service_name}",
        )

    if not user_context["is_admin"]:
        if not server_service.user_can_access_server_path(
            service_path,
            user_context["accessible_servers"],
        ):
            logger.warning(
                "User %s attempted to edit service %s without access",
                user_context["username"],
                service_path,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to edit this server",
            )

    return templates.TemplateResponse(
        "edit_server.html",
        {
            "request": request,
            "server": server_info,
            "username": user_context["username"],
            "user_context": user_context,
        },
    )


@router.post("/edit/{service_path:path}")
async def edit_server_submit(
    request: Request,
    service_path: str,
    name: Annotated[str, Form()] = "",
    description: Annotated[str, Form()] = "",
    proxy_pass_url: Annotated[str, Form()] = "",
    tags: Annotated[str, Form()] = "",
    num_tools: Annotated[int, Form()] = 0,
    num_stars: Annotated[int, Form()] = 0,
    is_python: Annotated[bool, Form()] = False,
    license_str: Annotated[str, Form(alias="license")] = "N/A",
    user_context: Annotated[dict, Depends(enhanced_auth)] = None,
):
    """Process edit form submission."""
    from ..search.service import faiss_service
    from ..core.nginx_service import nginx_service
    from ..auth.dependencies import user_has_ui_permission_for_service

    _ = request

    if not service_path.startswith("/"):
        service_path = "/" + service_path

    server_info = server_service.get_server_info(service_path)
    if not server_info:
        raise HTTPException(status_code=404, detail="Service path not found")

    service_name = server_info["server_name"]
    if not user_has_ui_permission_for_service(
        "modify_service",
        service_name,
        user_context.get("ui_permissions", {}),
    ):
        logger.warning(
            "User %s attempted to edit service %s without modify_service permission",
            user_context["username"],
            service_name,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You do not have permission to modify {service_name}",
        )

    if not user_context["is_admin"]:
        if not server_service.user_can_access_server_path(
            service_path,
            user_context["accessible_servers"],
        ):
            logger.warning(
                "User %s attempted to edit service %s without access",
                user_context["username"],
                service_path,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to edit this server",
            )

    _enforce_proxy_pass_url_allowlist(proxy_pass_url=proxy_pass_url)

    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

    updated_server_entry = {
        "server_name": name,
        "description": description,
        "path": service_path,
        "proxy_pass_url": proxy_pass_url,
        "tags": tag_list,
        "num_tools": num_tools,
        "num_stars": num_stars,
        "is_python": bool(is_python),
        "license": license_str,
        "tool_list": [],
    }

    success = server_service.update_server(service_path, updated_server_entry)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save updated server data")

    is_enabled = server_service.is_service_enabled(service_path)
    await faiss_service.add_or_update_service(service_path, updated_server_entry, is_enabled)

    enabled_servers = {
        path: server_service.get_server_info(path)
        for path in server_service.get_enabled_services()
    }
    await nginx_service.generate_config_async(enabled_servers)

    logger.info(
        "Server '%s' (%s) updated by user '%s'",
        name,
        service_path,
        user_context["username"],
    )

    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/server_details/{service_path:path}")
async def get_server_details(
    service_path: str,
    user_context: Annotated[dict, Depends(enhanced_auth)],
):
    """Get server details by path, or all servers if path is 'all' (filtered by permissions)."""
    if not service_path.startswith("/"):
        service_path = "/" + service_path

    if service_path == "/all":
        if user_context["is_admin"]:
            return server_service.get_all_servers()
        return server_service.get_all_servers_with_permissions(user_context["accessible_servers"])

    server_info = server_service.get_server_info(service_path)
    if not server_info:
        raise HTTPException(status_code=404, detail="Service path not registered")

    if not user_context["is_admin"]:
        if not server_service.user_can_access_server_path(
            service_path,
            user_context["accessible_servers"],
        ):
            logger.warning(
                "User %s attempted to access server details for %s without access",
                user_context["username"],
                service_path,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this server",
            )

    return server_info


@router.post("/refresh/{service_path:path}")
async def refresh_service(
    service_path: str,
    user_context: Annotated[dict, Depends(enhanced_auth)],
):
    """Refresh service health and tool information (requires health_check_service permission)."""
    return await _refresh_service_impl(
        service_path=service_path,
        user_context=user_context,
        server_service_obj=server_service,
    )
