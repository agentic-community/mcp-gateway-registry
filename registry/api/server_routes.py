import logging
from typing import (
    Annotated,
    Any,
    Optional,
)

from fastapi import (
    APIRouter,
    Cookie,
    Depends,
    Form,
    HTTPException,
    Response,
    Request,
    status,
)
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from ..core.config import settings
from ..auth.dependencies import web_auth, api_auth, enhanced_auth, nginx_proxied_auth
from ..services.server_service import server_service
from .server_internal_routes import (
    router as internal_router,
)
from .server_external_routes import (
    router as external_router,
)
from .server_json_routes import (
    router as server_json_router,
)
from .server_groups_routes import (
    router as groups_router,
)
from .server_tools_routes import (
    router as tools_router,
)
from .server_tokens_routes import (
    router as tokens_router,
)
from .server_routes_common import (
    _apply_remove_side_effects,
    _apply_toggle_side_effects,
    _build_server_entry_from_form,
    _enforce_proxy_pass_url_allowlist,
    _enforce_upstream_oauth_provider_configured,
    _normalize_server_path,
    _normalize_upstream_auth_payload,
    _require_can_modify_servers,
    _require_admin_user_context,
)

logger = logging.getLogger(__name__)

router = APIRouter()
router.include_router(internal_router)
router.include_router(external_router)
router.include_router(server_json_router)
router.include_router(groups_router)
router.include_router(tools_router)
router.include_router(tokens_router)

# Templates
templates = Jinja2Templates(directory=settings.templates_dir)
@router.get("/", response_class=HTMLResponse)
async def read_root(
    request: Request,
    query: str | None = None,
    session: Annotated[str | None, Cookie(alias=settings.session_cookie_name)] = None,
):
    """Main dashboard page showing services based on user permissions."""
    # Check authentication first and redirect if not authenticated
    if not session:
        logger.info("No session cookie at root route, redirecting to login")
        return RedirectResponse(url="/login", status_code=302)
    
    try:
        # Get user context
        user_context = enhanced_auth(session)
    except HTTPException as e:
        logger.info(f"Authentication failed at root route: {e.detail}, redirecting to login")
        return RedirectResponse(url="/login", status_code=302)
        
    from ..auth.dependencies import user_has_ui_permission_for_service
    
    # Helper function for templates
    def can_perform_action(permission: str, service_name: str) -> bool:
        """Check if user has UI permission for a specific service"""
        return user_has_ui_permission_for_service(permission, service_name, user_context.get('ui_permissions', {}))
    
    service_data = []
    search_query = query.lower() if query else ""
    
    # Get servers based on user permissions
    if user_context['is_admin']:
        # Admin users see all servers
        all_servers = server_service.get_all_servers()
        logger.info(f"Admin user {user_context['username']} accessing all {len(all_servers)} servers")
    else:
        # Filtered users see only accessible servers
        all_servers = server_service.get_all_servers_with_permissions(user_context['accessible_servers'])
        logger.info(f"User {user_context['username']} accessing {len(all_servers)} of {len(server_service.get_all_servers())} total servers")
    
    sorted_server_paths = sorted(
        all_servers.keys(), 
        key=lambda p: all_servers[p]["server_name"]
    )
    
    # Filter services based on UI permissions
    accessible_services = user_context.get('accessible_services', [])
    logger.info(f"DEBUG: User {user_context['username']} accessible_services: {accessible_services}")
    logger.info(f"DEBUG: User {user_context['username']} ui_permissions: {user_context.get('ui_permissions', {})}")
    logger.info(f"DEBUG: User {user_context['username']} scopes: {user_context.get('scopes', [])}")
    
    for path in sorted_server_paths:
        server_info = all_servers[path]
        server_name = server_info["server_name"]
        
        # Check if user can list this service
        if 'all' not in accessible_services and server_name not in accessible_services:
            logger.debug(f"Filtering out service '{server_name}' - user doesn't have list_service permission")
            continue
        
        # Include description and tags in search
        searchable_text = f"{server_name.lower()} {server_info.get('description', '').lower()} {' '.join(server_info.get('tags', []))}"
        if not search_query or search_query in searchable_text:
            # Get real health status from health service
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
                    "last_checked_iso": health_data["last_checked_iso"]
                }
            )
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request, 
            "services": service_data, 
            "username": user_context['username'],
            "user_context": user_context,  # Pass full user context to template
            "can_perform_action": can_perform_action  # Helper function for permission checks
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
    
    if not service_path.startswith("/"):
        service_path = "/" + service_path
        
    server_info = server_service.get_server_info(service_path)
    if not server_info:
        raise HTTPException(status_code=404, detail="Service path not registered")
    
    service_name = server_info["server_name"]
    
    # Check if user has toggle_service permission for this specific service
    if not user_has_ui_permission_for_service('toggle_service', service_name, user_context.get('ui_permissions', {})):
        logger.warning(f"User {user_context['username']} attempted to toggle service {service_name} without toggle_service permission")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail=f"You do not have permission to toggle {service_name}"
        )

    # For non-admin users, check if they have access to this specific server
    if not user_context['is_admin']:
        if not server_service.user_can_access_server_path(service_path, user_context['accessible_servers']):
            logger.warning(f"User {user_context['username']} attempted to toggle service {service_path} without access")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, 
                detail="You do not have access to this server"
            )

    new_state = enabled == "on"
    success = server_service.toggle_service(service_path, new_state)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to toggle service")
    
    server_name = server_info["server_name"]
    logger.info(f"Toggled '{server_name}' ({service_path}) to {new_state} by user '{user_context['username']}'")

    # If enabling, perform immediate health check
    status = "disabled"
    last_checked_iso = None
    if new_state:
        logger.info(f"Performing immediate health check for {service_path} upon toggle ON...")
        try:
            status, last_checked_dt = await health_service.perform_immediate_health_check(service_path)
            last_checked_iso = last_checked_dt.isoformat() if last_checked_dt else None
            logger.info(f"Immediate health check for {service_path} completed. Status: {status}")
        except Exception as e:
            logger.error(f"ERROR during immediate health check for {service_path}: {e}")
            status = f"error: immediate check failed ({type(e).__name__})"
    else:
        # When disabling, set status to disabled
        status = "disabled"
        logger.info(f"Service {service_path} toggled OFF. Status set to disabled.")

    # Update FAISS metadata with new enabled state
    await faiss_service.add_or_update_service(service_path, server_info, new_state)
    
    # Regenerate Nginx configuration
    enabled_servers = {
        path: server_service.get_server_info(path) 
        for path in server_service.get_enabled_services()
    }
    await nginx_service.generate_config_async(enabled_servers)
    
    # Broadcast health status update to WebSocket clients
    await health_service.broadcast_health_update(service_path)
    
    return JSONResponse(
        status_code=200,
        content={
            "message": f"Toggle request for {service_path} processed.",
            "service_path": service_path,
            "new_enabled_state": new_state,
            "status": status,
            "last_checked_iso": last_checked_iso,
            "num_tools": server_info.get("num_tools", 0)
        }
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
    from ..auth.dependencies import user_has_ui_permission_for_service
    
    # Check if user has register_service permission for any service
    ui_permissions = user_context.get('ui_permissions', {})
    register_permissions = ui_permissions.get('register_service', [])
    
    if not register_permissions:
        logger.warning(f"User {user_context['username']} attempted to register service without register_service permission")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="You do not have permission to register new services"
        )
    
    logger.info(f"Service registration request from user '{user_context['username']}'")
    logger.info(f"Name: {name}, Path: {path}, URL: {proxy_pass_url}")

    _enforce_proxy_pass_url_allowlist(proxy_pass_url=proxy_pass_url)

    # Ensure path starts with a slash
    if not path.startswith("/"):
        path = "/" + path

    # Process tags
    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

    # Create server entry
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
        "tool_list": []
    }

    # Register the server
    success = server_service.register_server(server_entry)
    
    if not success:
        return JSONResponse(
            status_code=400,
            content={"error": f"Service with path '{path}' already exists or failed to save"},
        )

    # Add to FAISS index with current enabled state
    is_enabled = server_service.is_service_enabled(path)
    await faiss_service.add_or_update_service(path, server_entry, is_enabled)
    
    # Regenerate Nginx configuration
    enabled_servers = {
        server_path: server_service.get_server_info(server_path) 
        for server_path in server_service.get_enabled_services()
    }
    await nginx_service.generate_config_async(enabled_servers)
    
    # Broadcast health status update to WebSocket clients
    await health_service.broadcast_health_update(path)
    
    logger.info(f"New service registered: '{name}' at path '{path}' by user '{user_context['username']}'")

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
    user_context: Annotated[dict, Depends(enhanced_auth)]
):
    """Show edit form for a service (requires modify_service UI permission)."""
    from ..auth.dependencies import user_has_ui_permission_for_service
    
    if not service_path.startswith('/'):
        service_path = '/' + service_path

    server_info = server_service.get_server_info(service_path)
    if not server_info:
        raise HTTPException(status_code=404, detail="Service path not found")
    
    service_name = server_info["server_name"]
    
    # Check if user has modify_service permission for this specific service
    if not user_has_ui_permission_for_service('modify_service', service_name, user_context.get('ui_permissions', {})):
        logger.warning(f"User {user_context['username']} attempted to access edit form for {service_name} without modify_service permission")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail=f"You do not have permission to modify {service_name}"
        )
    
    # For non-admin users, check if they have access to this specific server
    if not user_context['is_admin']:
        if not server_service.user_can_access_server_path(service_path, user_context['accessible_servers']):
            logger.warning(f"User {user_context['username']} attempted to edit service {service_path} without access")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, 
                detail="You do not have access to edit this server"
            )
    
    return templates.TemplateResponse(
        "edit_server.html", 
        {
            "request": request, 
            "server": server_info, 
            "username": user_context['username'],
            "user_context": user_context
        }
    )


@router.post("/edit/{service_path:path}")
async def edit_server_submit(
    service_path: str, 
    name: Annotated[str, Form()], 
    proxy_pass_url: Annotated[str, Form()], 
    user_context: Annotated[dict, Depends(enhanced_auth)], 
    description: Annotated[str, Form()] = "", 
    tags: Annotated[str, Form()] = "", 
    num_tools: Annotated[int, Form()] = 0, 
    num_stars: Annotated[int, Form()] = 0, 
    is_python: Annotated[bool | None, Form()] = False,  
    license_str: Annotated[str, Form(alias="license")] = "N/A", 
):
    """Handle server edit form submission (requires modify_service UI permission)."""
    from ..search.service import faiss_service
    from ..core.nginx_service import nginx_service
    from ..auth.dependencies import user_has_ui_permission_for_service
    
    if not service_path.startswith('/'):
        service_path = '/' + service_path

    _enforce_proxy_pass_url_allowlist(proxy_pass_url=proxy_pass_url)

    # Check if the server exists and get service name
    server_info = server_service.get_server_info(service_path)
    if not server_info:
        raise HTTPException(status_code=404, detail="Service path not found")
    
    service_name = server_info["server_name"]
    
    # Check if user has modify_service permission for this specific service
    if not user_has_ui_permission_for_service('modify_service', service_name, user_context.get('ui_permissions', {})):
        logger.warning(f"User {user_context['username']} attempted to edit service {service_name} without modify_service permission")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail=f"You do not have permission to modify {service_name}"
        )


    # For non-admin users, check if they have access to this specific server
    if not user_context['is_admin']:
        if not server_service.user_can_access_server_path(service_path, user_context['accessible_servers']):
            logger.warning(f"User {user_context['username']} attempted to edit service {service_path} without access")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, 
                detail="You do not have access to edit this server"
            )

    # Process tags
    tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]

    # Prepare updated server data
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
        "tool_list": []  # Keep existing or initialize
    }

    # Update server
    success = server_service.update_server(service_path, updated_server_entry)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save updated server data")

    # Update FAISS metadata (keep current enabled state)
    is_enabled = server_service.is_service_enabled(service_path)
    await faiss_service.add_or_update_service(service_path, updated_server_entry, is_enabled)
    
    # Regenerate Nginx configuration
    enabled_servers = {
        path: server_service.get_server_info(path) 
        for path in server_service.get_enabled_services()
    }
    await nginx_service.generate_config_async(enabled_servers)
    
    logger.info(f"Server '{name}' ({service_path}) updated by user '{user_context['username']}'")

    # Redirect back to the main page
    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/server_details/{service_path:path}")
async def get_server_details(
    service_path: str,
    user_context: Annotated[dict, Depends(enhanced_auth)]
):
    """Get server details by path, or all servers if path is 'all' (filtered by permissions)."""
    # Normalize the path to ensure it starts with '/'
    if not service_path.startswith('/'):
        service_path = '/' + service_path
    
    # Special case: if path is 'all' or '/all', return details for all accessible servers
    if service_path == '/all':
        if user_context['is_admin']:
            return server_service.get_all_servers()
        else:
            return server_service.get_all_servers_with_permissions(user_context['accessible_servers'])
    
    # Regular case: return details for a specific server
    server_info = server_service.get_server_info(service_path)
    if not server_info:
        raise HTTPException(status_code=404, detail="Service path not registered")
    
    # For non-admin users, check if they have access to this specific server
    if not user_context['is_admin']:
        if not server_service.user_can_access_server_path(service_path, user_context['accessible_servers']):
            logger.warning(f"User {user_context['username']} attempted to access server details for {service_path} without access")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, 
                detail="You do not have access to this server"
            )
    
    return server_info


@router.post("/refresh/{service_path:path}")
async def refresh_service(
    service_path: str, 
    user_context: Annotated[dict, Depends(enhanced_auth)]
):
    """Refresh service health and tool information (requires health_check_service permission)."""
    from .server_refresh_common import (
        _refresh_service_impl,
    )

    return await _refresh_service_impl(
        service_path=service_path,
        user_context=user_context,
        server_service_obj=server_service,
    )
