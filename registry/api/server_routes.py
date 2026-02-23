import asyncio
import json
import logging
import os
from typing import Annotated

import httpx
from fastapi import APIRouter, Cookie, Depends, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from ..audit import set_audit_action
from ..auth.dependencies import enhanced_auth, nginx_proxied_auth
from ..auth.internal import validate_internal_auth
from ..core.config import settings
from ..services.security_scanner import security_scanner_service
from ..services.server_service import server_service

logger = logging.getLogger(__name__)

router = APIRouter()


class RatingRequest(BaseModel):
    rating: int


# Templates
templates = Jinja2Templates(directory=settings.templates_dir)


async def _perform_security_scan_on_registration(
    path: str,
    proxy_pass_url: str,
    server_entry: dict,
    headers_list: list | None = None,
) -> None:
    """Perform security scan on newly registered server.

    Handles the complete security scan workflow including:
    - Running the security scan with configured analyzers
    - Adding security-pending tag if scan fails
    - Disabling server if configured and scan fails
    - Updating FAISS and regenerating Nginx config if server disabled

    All scan failures are non-fatal and will be logged but not raised.

    Args:
        path: Server path (e.g., /mcpgw)
        proxy_pass_url: URL to scan
        server_entry: Server metadata dictionary
        headers_list: Optional headers for authenticated endpoints
    """
    scan_config = security_scanner_service.get_scan_config()
    if not (scan_config.enabled and scan_config.scan_on_registration):
        return

    logger.info(f"Running security scan for newly registered server: {path}")

    try:
        # Prepare headers if needed (for authenticated endpoints)
        headers_json = None
        if headers_list:
            headers_json = json.dumps(headers_list)

        # Run the security scan
        scan_result = await security_scanner_service.scan_server(
            server_url=proxy_pass_url,
            server_path=path,
            analyzers=scan_config.analyzers,
            api_key=scan_config.llm_api_key,
            headers=headers_json,
            timeout=scan_config.scan_timeout_seconds,
        )

        # Handle unsafe servers
        if not scan_result.is_safe:
            logger.warning(
                f"Server {path} failed security scan. "
                f"Critical: {scan_result.critical_issues}, High: {scan_result.high_severity}"
            )

            # Add security-pending tag if configured
            if scan_config.add_security_pending_tag:
                current_tags = server_entry.get("tags", [])
                if "security-pending" not in current_tags:
                    current_tags.append("security-pending")
                    server_entry["tags"] = current_tags
                    await server_service.update_server(path, server_entry)
                    logger.info(f"Added 'security-pending' tag to {path}")

            # Disable server if configured
            if scan_config.block_unsafe_servers:
                from ..core.nginx_service import nginx_service
                from ..repositories.factory import get_search_repository

                await server_service.toggle_service(path, False)
                logger.warning(f"Disabled server {path} due to failed security scan")

                # Update search index with disabled state
                search_repo = get_search_repository()
                await search_repo.index_server(path, server_entry, is_enabled=False)

                # Regenerate Nginx config to remove disabled server
                enabled_servers = {}

                for server_path in await server_service.get_enabled_services():
                    server_info = await server_service.get_server_info(server_path)

                    if server_info:
                        enabled_servers[server_path] = server_info
                await nginx_service.generate_config_async(enabled_servers)
        else:
            logger.info(f"Server {path} passed security scan")

    except Exception as e:
        logger.error(f"Security scan failed for {path}: {e}")
        # Non-fatal error - server is registered but scan failed


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
        return user_has_ui_permission_for_service(
            permission, service_name, user_context.get("ui_permissions", {})
        )

    service_data = []
    search_query = query.lower() if query else ""

    # Get servers based on user permissions
    if user_context["is_admin"]:
        # Admin users see all servers
        all_servers = await server_service.get_all_servers()
        logger.info(
            f"Admin user {user_context['username']} accessing all {len(all_servers)} servers"
        )
    else:
        # Filtered users see only accessible servers
        all_servers = await server_service.get_all_servers_with_permissions(
            user_context["accessible_servers"]
        )
        all_servers_count = await server_service.get_all_servers()
        logger.info(
            f"User {user_context['username']} accessing {len(all_servers)} of {len(all_servers_count)} total servers"
        )

    sorted_server_paths = sorted(all_servers.keys(), key=lambda p: all_servers[p]["server_name"])

    # Filter services based on UI permissions
    accessible_services = user_context.get("accessible_services", [])
    # Normalize accessible_services by stripping slashes for comparison
    normalized_accessible_services = [s.strip("/") for s in accessible_services]
    logger.info(
        f"DEBUG: User {user_context['username']} accessible_services: {accessible_services}"
    )
    logger.info(
        f"DEBUG: User {user_context['username']} ui_permissions: {user_context.get('ui_permissions', {})}"
    )
    logger.info(f"DEBUG: User {user_context['username']} scopes: {user_context.get('scopes', [])}")

    for path in sorted_server_paths:
        server_info = all_servers[path]
        server_name = server_info["server_name"]
        # Extract technical name from path (remove leading and trailing slashes)
        technical_name = path.strip("/")

        # Check if user can list this service using technical name
        if (
            "all" not in accessible_services
            and technical_name not in normalized_accessible_services
        ):
            logger.debug(
                f"Filtering out service '{server_name}' (path={path}) - user doesn't have list_service permission"
            )
            continue

        # Include description and tags in search
        searchable_text = f"{server_name.lower()} {server_info.get('description', '').lower()} {' '.join(server_info.get('tags', []))}"
        if not search_query or search_query in searchable_text:
            # Get real health status from health service
            from ..health.service import health_service

            health_data = health_service._get_service_health_data(path, server_info)

            # Normalize health status to enum values only (strip error messages)
            raw_status = health_data["status"]
            if isinstance(raw_status, str):
                if "unhealthy" in raw_status.lower():
                    normalized_status = "unhealthy"
                elif "healthy" in raw_status.lower():
                    normalized_status = "healthy"
                elif "disabled" in raw_status.lower():
                    normalized_status = "disabled"
                elif "checking" in raw_status.lower():
                    normalized_status = "unknown"
                elif "error" in raw_status.lower():
                    normalized_status = "unhealthy"
                else:
                    normalized_status = raw_status
            else:
                normalized_status = raw_status

            service_data.append(
                {
                    "display_name": server_name,
                    "path": path,
                    "description": server_info.get("description", ""),
                    "proxy_pass_url": server_info.get("proxy_pass_url", ""),
                    "is_enabled": await server_service.is_service_enabled(path),
                    "tags": server_info.get("tags", []),
                    "num_tools": server_info.get("num_tools", 0),
                    "num_stars": server_info.get("num_stars", 0),
                    "is_python": server_info.get("is_python", False),
                    "license": server_info.get("license", "N/A"),
                    "health_status": normalized_status,
                    "last_checked_iso": health_data["last_checked_iso"],
                    "mcp_endpoint": server_info.get("mcp_endpoint"),
                }
            )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "services": service_data,
            "username": user_context["username"],
            "user_context": user_context,  # Pass full user context to template
            "can_perform_action": can_perform_action,  # Helper function for permission checks
        },
    )


@router.get("/servers")
async def get_servers_json(
    request: Request,
    query: str | None = None,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """Get servers data as JSON for React frontend and external API (supports both session cookies and Bearer tokens)."""
    # Set audit action for server list
    set_audit_action(request, "list", "server", description="List all servers")

    # CRITICAL DIAGNOSTIC: Log user_context received by endpoint
    logger.debug(f"[GET_SERVERS_DEBUG] Received user_context: {user_context}")
    logger.debug(f"[GET_SERVERS_DEBUG] user_context type: {type(user_context)}")
    if user_context:
        logger.debug(f"[GET_SERVERS_DEBUG] Username: {user_context.get('username', 'NOT PRESENT')}")
        logger.debug(f"[GET_SERVERS_DEBUG] Scopes: {user_context.get('scopes', 'NOT PRESENT')}")
        logger.debug(
            f"[GET_SERVERS_DEBUG] Auth method: {user_context.get('auth_method', 'NOT PRESENT')}"
        )

    service_data = []
    search_query = query.lower() if query else ""

    # Get servers based on user permissions (same logic as root route)
    if user_context["is_admin"]:
        all_servers = await server_service.get_all_servers()
    else:
        all_servers = await server_service.get_all_servers_with_permissions(
            user_context["accessible_servers"]
        )

    sorted_server_paths = sorted(all_servers.keys(), key=lambda p: all_servers[p]["server_name"])

    # Filter services based on UI permissions (same logic as root route)
    accessible_services = user_context.get("accessible_services", [])
    # Normalize accessible_services by stripping slashes for comparison
    normalized_accessible_services = [s.strip("/") for s in accessible_services]

    for path in sorted_server_paths:
        server_info = all_servers[path]
        server_name = server_info["server_name"]
        # Extract technical name from path (remove leading and trailing slashes)
        technical_name = path.strip("/")

        # Check if user can list this service using technical name
        if (
            "all" not in accessible_services
            and technical_name not in normalized_accessible_services
        ):
            continue

        # Include description and tags in search
        searchable_text = f"{server_name.lower()} {server_info.get('description', '').lower()} {' '.join(server_info.get('tags', []))}"
        if not search_query or search_query in searchable_text:
            # Get real health status from health service
            from ..health.service import health_service

            health_data = health_service._get_service_health_data(path, server_info)

            # Normalize health status to enum values only (strip error messages)
            raw_status = health_data["status"]
            if isinstance(raw_status, str):
                if "unhealthy" in raw_status.lower():
                    normalized_status = "unhealthy"
                elif "healthy" in raw_status.lower():
                    normalized_status = "healthy"
                elif "disabled" in raw_status.lower():
                    normalized_status = "disabled"
                elif "checking" in raw_status.lower():
                    normalized_status = "unknown"
                elif "error" in raw_status.lower():
                    normalized_status = "unhealthy"
                else:
                    normalized_status = raw_status
            else:
                normalized_status = raw_status

            # Build versions list if this server has other versions
            versions = []
            current_version = server_info.get("version", "v1.0.0")
            current_status = server_info.get("status", "stable")

            # Add current (active) version first
            versions.append(
                {
                    "version": current_version,
                    "proxy_pass_url": server_info.get("proxy_pass_url", ""),
                    "status": current_status,
                    "is_default": True,
                }
            )

            # Add other versions if they exist
            other_version_ids = server_info.get("other_version_ids", [])
            for version_id in other_version_ids:
                version_info = await server_service.get_server_info(version_id)
                if version_info:
                    versions.append(
                        {
                            "version": version_info.get("version", "unknown"),
                            "proxy_pass_url": version_info.get("proxy_pass_url", ""),
                            "status": version_info.get("status", "stable"),
                            "is_default": False,
                        }
                    )

            service_data.append(
                {
                    "display_name": server_name,
                    "path": path,
                    "description": server_info.get("description", ""),
                    "proxy_pass_url": server_info.get("proxy_pass_url", ""),
                    "is_enabled": await server_service.is_service_enabled(path),
                    "tags": server_info.get("tags", []),
                    "num_tools": server_info.get("num_tools", 0),
                    "num_stars": server_info.get("num_stars", 0),
                    "is_python": server_info.get("is_python", False),
                    "license": server_info.get("license", "N/A"),
                    "health_status": normalized_status,
                    "last_checked_iso": health_data["last_checked_iso"],
                    "mcp_endpoint": server_info.get("mcp_endpoint"),
                    "metadata": server_info.get("metadata", {}),
                    "version": current_version,
                    "versions": versions if len(versions) > 1 else None,
                    "default_version": current_version,
                    "mcp_server_version": server_info.get("mcp_server_version"),
                    "mcp_server_version_previous": server_info.get("mcp_server_version_previous"),
                    "mcp_server_version_updated_at": server_info.get(
                        "mcp_server_version_updated_at"
                    ),
                    "sync_metadata": server_info.get("sync_metadata"),
                }
            )

    return {"servers": service_data}


@router.post("/toggle/{service_path:path}")
async def toggle_service_route(
    request: Request,
    service_path: str,
    enabled: Annotated[str | None, Form()] = None,
    user_context: Annotated[dict, Depends(enhanced_auth)] = None,
):
    """Toggle a service on/off (requires toggle_service UI permission)."""
    from ..auth.dependencies import user_has_ui_permission_for_service
    from ..core.nginx_service import nginx_service
    from ..health.service import health_service
    from ..search.service import faiss_service

    if not service_path.startswith("/"):
        service_path = "/" + service_path

    server_info = await server_service.get_server_info(service_path)
    if not server_info:
        raise HTTPException(status_code=404, detail="Service path not registered")

    service_name = server_info["server_name"]

    # Check if user has toggle_service permission for this specific service
    if not user_has_ui_permission_for_service(
        "toggle_service", service_name, user_context.get("ui_permissions", {})
    ):
        logger.warning(
            f"User {user_context['username']} attempted to toggle service {service_name} without toggle_service permission"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You do not have permission to toggle {service_name}",
        )

    # For non-admin users, check if they have access to this specific server
    if not user_context["is_admin"]:
        if not await server_service.user_can_access_server_path(
            service_path, user_context["accessible_servers"]
        ):
            logger.warning(
                f"User {user_context['username']} attempted to toggle service {service_path} without access"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this server",
            )

    new_state = enabled == "on"
    success = await server_service.toggle_service(service_path, new_state)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to toggle service")

    server_name = server_info["server_name"]
    logger.info(
        f"Toggled '{server_name}' ({service_path}) to {new_state} by user '{user_context['username']}'"
    )

    # If enabling, perform immediate health check
    status = "disabled"
    last_checked_iso = None
    if new_state:
        logger.info(f"Performing immediate health check for {service_path} upon toggle ON...")
        try:
            (
                status,
                last_checked_dt,
            ) = await health_service.perform_immediate_health_check(service_path)
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
    enabled_servers = {}

    for path in await server_service.get_enabled_services():
        server_info = await server_service.get_server_info(path)

        if server_info:
            enabled_servers[path] = server_info
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
    mcp_endpoint: Annotated[str | None, Form()] = None,
    sse_endpoint: Annotated[str | None, Form()] = None,
    metadata: Annotated[str | None, Form()] = None,
    visibility: Annotated[str, Form()] = "public",
    allowed_groups: Annotated[str | None, Form()] = None,
    user_context: Annotated[dict, Depends(enhanced_auth)] = None,
):
    """Register a new service (requires register_service UI permission)."""
    from ..core.nginx_service import nginx_service
    from ..health.service import health_service
    from ..search.service import faiss_service

    # Check if user has register_service permission for any service
    ui_permissions = user_context.get("ui_permissions", {})
    register_permissions = ui_permissions.get("register_service", [])

    if not register_permissions:
        logger.warning(
            f"User {user_context['username']} attempted to register service without register_service permission"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to register new services",
        )

    logger.info(f"Service registration request from user '{user_context['username']}'")
    logger.info(f"Name: {name}, Path: {path}, URL: {proxy_pass_url}")

    # Ensure path starts with a slash
    if not path.startswith("/"):
        path = "/" + path

    # Process tags
    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

    # Validate visibility value
    valid_visibility = ["public", "group-restricted", "internal"]
    if visibility not in valid_visibility:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid visibility value. Must be one of: {', '.join(valid_visibility)}",
        )

    # Process allowed_groups (comma-separated string to list)
    allowed_groups_list = []
    if allowed_groups:
        allowed_groups_list = [g.strip() for g in allowed_groups.split(",") if g.strip()]

    # Validate group-restricted requires allowed_groups
    if visibility == "group-restricted" and not allowed_groups_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="group-restricted visibility requires at least one allowed_group",
        )

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
        "tool_list": [],
        "visibility": visibility,
        "allowed_groups": allowed_groups_list,
    }

    # Add custom endpoint fields if provided
    if mcp_endpoint:
        server_entry["mcp_endpoint"] = mcp_endpoint
    if sse_endpoint:
        server_entry["sse_endpoint"] = sse_endpoint

    # Add metadata if provided (expects JSON string)
    if metadata:
        try:
            import json

            server_entry["metadata"] = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON in metadata field",
            )

    # Register the server (or new version if path exists with different version)
    result = await server_service.register_server(server_entry)

    if not result["success"]:
        # Check if it's a version conflict (same path, same version)
        logger.warning(f"Server registration failed for path '{path}': {result['message']}")
        return JSONResponse(
            status_code=409,
            content={
                "error": "Service registration failed",
                "detail": "Check server logs for more information",
            },
        )

    # Handle new version registration vs new server
    if result.get("is_new_version"):
        logger.info(
            f"New version registered: '{name}' version '{server_entry.get('version')}' "
            f"at path '{path}' by user '{user_context['username']}'"
        )
        return JSONResponse(
            status_code=201,
            content={
                "message": f"Service '{name}' version registered successfully",
                "service": server_entry,
                "is_new_version": True,
                "existing_version": result.get("existing_version"),
            },
        )

    # New server - proceed with full setup
    # Add to FAISS index with current enabled state
    is_enabled = await server_service.is_service_enabled(path)
    await faiss_service.add_or_update_service(path, server_entry, is_enabled)

    # Regenerate Nginx configuration
    enabled_servers = {}

    for server_path in await server_service.get_enabled_services():
        server_info = await server_service.get_server_info(server_path)

        if server_info:
            enabled_servers[server_path] = server_info
    await nginx_service.generate_config_async(enabled_servers)

    # Broadcast health status update to WebSocket clients
    await health_service.broadcast_health_update(path)

    # Security scanning if enabled
    await _perform_security_scan_on_registration(path, proxy_pass_url, server_entry)

    logger.info(
        f"New service registered: '{name}' at path '{path}' by user '{user_context['username']}'"
    )

    return JSONResponse(
        status_code=201,
        content={
            "message": "Service registered successfully",
            "service": server_entry,
        },
    )


@router.post("/internal/register")
async def internal_register_service(
    request: Request,
    caller: Annotated[str, Depends(validate_internal_auth)],
    name: Annotated[str, Form()],
    description: Annotated[str, Form()],
    path: Annotated[str, Form()],
    proxy_pass_url: Annotated[str, Form()],
    tags: Annotated[str, Form()] = "",
    num_tools: Annotated[int, Form()] = 0,
    num_stars: Annotated[int, Form()] = 0,
    is_python: Annotated[bool, Form()] = False,
    license_str: Annotated[str, Form(alias="license")] = "N/A",
    overwrite: Annotated[bool, Form()] = True,
    auth_provider: Annotated[str | None, Form()] = None,
    auth_type: Annotated[str | None, Form()] = None,
    supported_transports: Annotated[str | None, Form()] = None,
    headers: Annotated[str | None, Form()] = None,
    tool_list_json: Annotated[str | None, Form()] = None,
    visibility: Annotated[str, Form()] = "public",
    allowed_groups: Annotated[str | None, Form()] = None,
):
    """Internal service registration endpoint for mcpgw-server (requires admin authentication)."""
    logger.warning(
        "INTERNAL REGISTER: Function called - starting execution"
    )  # TODO: replace with debug

    from ..core.nginx_service import nginx_service
    from ..health.service import health_service
    from ..search.service import faiss_service

    logger.warning(
        f"INTERNAL REGISTER: Request parameters - name={name}, path={path}, proxy_pass_url={proxy_pass_url}"
    )  # TODO: replace with debug

    logger.info(f"Internal service registration request from caller '{caller}'")

    # Validate path format
    if not path.startswith("/"):
        path = "/" + path
    logger.warning(f"INTERNAL REGISTER: Validated path: {path}")  # TODO: replace with debug

    # Process tags
    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
    logger.warning(f"INTERNAL REGISTER: Processed tags: {tag_list}")  # TODO: replace with debug

    # Process supported_transports
    if supported_transports:
        try:
            transports_list = (
                json.loads(supported_transports)
                if supported_transports.startswith("[")
                else [t.strip() for t in supported_transports.split(",")]
            )
        except Exception as e:
            logger.warning(
                f"INTERNAL REGISTER: Failed to parse supported_transports, using default: {e}"
            )
            transports_list = ["streamable-http"]
    else:
        transports_list = ["streamable-http"]

    # Process headers
    headers_list = []
    if headers:
        try:
            headers_list = json.loads(headers) if isinstance(headers, str) else headers
        except Exception as e:
            logger.warning(f"INTERNAL REGISTER: Failed to parse headers: {e}")

    # Process tool_list
    tool_list = []
    if tool_list_json:
        try:
            tool_list = (
                json.loads(tool_list_json) if isinstance(tool_list_json, str) else tool_list_json
            )
        except Exception as e:
            logger.warning(f"INTERNAL REGISTER: Failed to parse tool_list_json: {e}")

    # Process allowed_groups (comma-separated string to list)
    allowed_groups_list = []
    if allowed_groups:
        allowed_groups_list = [g.strip() for g in allowed_groups.split(",") if g.strip()]

    # Validate visibility value
    valid_visibility = ["public", "group-restricted", "internal"]
    if visibility not in valid_visibility:
        visibility = "public"  # Default to public for internal registration

    # Create server entry
    server_entry = {
        "server_name": name,
        "description": description,
        "path": path,
        "proxy_pass_url": proxy_pass_url,
        "supported_transports": transports_list,
        "auth_type": auth_type if auth_type else "none",
        "tags": tag_list,
        "num_tools": num_tools,
        "num_stars": num_stars,
        "is_python": is_python,
        "license": license_str,
        "tool_list": tool_list,
        "visibility": visibility,
        "allowed_groups": allowed_groups_list,
    }

    # Add optional fields if provided
    if auth_provider:
        server_entry["auth_provider"] = auth_provider
    if headers_list:
        server_entry["headers"] = headers_list

    logger.warning(
        f"INTERNAL REGISTER: Created server entry: {server_entry}"
    )  # TODO: replace with debug
    logger.warning(
        f"INTERNAL REGISTER: Overwrite parameter: {overwrite}"
    )  # TODO: replace with debug

    # Check if server exists and handle overwrite logic
    existing_server = await server_service.get_server_info(path)
    if existing_server and not overwrite:
        logger.warning(
            f"INTERNAL REGISTER: Server exists and overwrite=False for path {path}"
        )  # TODO: replace with debug
        return JSONResponse(
            status_code=409,  # Conflict status code for existing resource
            content={
                "error": "Service registration failed",
                "reason": f"A service with path '{path}' already exists",
                "suggestion": "Set overwrite=true or use the remove command first",
            },
        )

    # Register the server (this will overwrite if server exists and overwrite=True)
    logger.warning(
        "INTERNAL REGISTER: Calling server_service.register_server"
    )  # TODO: replace with debug
    if existing_server and overwrite:
        logger.warning(
            f"INTERNAL REGISTER: Overwriting existing server at path {path}"
        )  # TODO: replace with debug
        success = await server_service.update_server(path, server_entry)
        is_new_version = False
    else:
        result = await server_service.register_server(server_entry)
        success = result["success"]
        is_new_version = result.get("is_new_version", False)

    if not success:
        logger.warning(
            f"INTERNAL REGISTER: Registration failed for path {path}: "
            f"{result.get('message', 'unknown error')}"
        )
        return JSONResponse(
            status_code=409,  # Conflict status code for existing resource
            content={
                "error": "Service registration failed",
                "detail": "Check server logs for more information",
            },
        )

    logger.warning(
        "INTERNAL REGISTER: Auto-enabling newly registered server"
    )  # TODO: replace with debug

    # Automatically enable the newly registered server BEFORE FAISS indexing
    try:
        toggle_success = await server_service.toggle_service(path, True)
        if toggle_success:
            logger.info(f"Successfully auto-enabled server {path} after registration")
        else:
            logger.warning(f"Failed to auto-enable server {path} after registration")
    except Exception as e:
        logger.error(f"Error auto-enabling server {path}: {e}")
        # Non-fatal error - server is registered but not enabled

    logger.warning(
        "INTERNAL REGISTER: Server registered successfully, adding to FAISS index"
    )  # TODO: replace with debug

    # Add to FAISS index with current enabled state (should be True after auto-enable)
    is_enabled = await server_service.is_service_enabled(path)
    await faiss_service.add_or_update_service(path, server_entry, is_enabled)

    logger.warning(
        "INTERNAL REGISTER: Regenerating Nginx configuration"
    )  # TODO: replace with debug

    # Regenerate Nginx configuration
    enabled_servers = {}

    for server_path in await server_service.get_enabled_services():
        server_info = await server_service.get_server_info(server_path)

        if server_info:
            enabled_servers[server_path] = server_info
    await nginx_service.generate_config_async(enabled_servers)

    logger.warning(
        "INTERNAL REGISTER: Broadcasting health status update"
    )  # TODO: replace with debug

    # Broadcast health status update to WebSocket clients
    await health_service.broadcast_health_update(path)

    logger.warning(
        "INTERNAL REGISTER: Updating scopes.yml for new server"
    )  # TODO: replace with debug

    # Update scopes.yml with the new server's tools
    from ..services.scope_service import update_server_scopes

    # Get the tool list from the server entry
    tool_names = []
    if "tool_list" in server_entry and server_entry["tool_list"]:
        tool_names = [tool["name"] for tool in server_entry["tool_list"] if "name" in tool]

    # Update scopes and reload auth server
    try:
        await update_server_scopes(path, name, tool_names)
        logger.info(f"Successfully updated scopes for server {path} with {len(tool_names)} tools")
    except Exception as e:
        logger.error(f"Failed to update scopes for server {path}: {e}")
        # Non-fatal error - server is registered but scopes not updated

    # Security scanning if enabled
    await _perform_security_scan_on_registration(path, proxy_pass_url, server_entry, headers_list)

    logger.warning(
        "INTERNAL REGISTER: Registration complete, returning success response"
    )  # TODO: replace with debug
    logger.info(
        f"New service registered via internal endpoint: '{name}' at path '{path}' by caller '{caller}'"
    )

    return JSONResponse(
        status_code=201,
        content={
            "message": "Service registered successfully",
            "service": server_entry,
        },
    )


@router.post("/internal/remove")
async def internal_remove_service(
    request: Request,
    caller: Annotated[str, Depends(validate_internal_auth)],
    service_path: Annotated[str, Form()],
):
    """Internal service removal endpoint for mcpgw-server (requires admin authentication)."""
    from ..core.nginx_service import nginx_service
    from ..health.service import health_service
    from ..search.service import faiss_service

    logger.warning(
        "INTERNAL REMOVE: Function called - starting execution"
    )  # TODO: replace with debug

    logger.info(
        f"Internal service removal request from caller '{caller}' for service '{service_path}'"
    )

    # Validate path format
    if not service_path.startswith("/"):
        service_path = "/" + service_path

    logger.warning(
        f"INTERNAL REMOVE: Normalized service path: {service_path}"
    )  # TODO: replace with debug

    # Check if server exists
    server_info = await server_service.get_server_info(service_path)
    if not server_info:
        logger.warning(
            f"INTERNAL REMOVE: Service not found at path '{service_path}'"
        )  # TODO: replace with debug
        return JSONResponse(
            status_code=404,
            content={
                "error": "Service not found",
                "reason": f"No service registered at path '{service_path}'",
                "suggestion": "Check the service path and ensure it is registered",
            },
        )

    logger.warning(
        "INTERNAL REMOVE: Service found, proceeding with removal"
    )  # TODO: replace with debug

    # Remove the server
    success = await server_service.remove_server(service_path)

    if not success:
        logger.warning(
            f"INTERNAL REMOVE: Failed to remove service at path '{service_path}'"
        )  # TODO: replace with debug
        return JSONResponse(
            status_code=500,
            content={
                "error": "Service removal failed",
                "reason": f"Failed to remove service at path '{service_path}'",
                "suggestion": "Check server logs for detailed error information",
            },
        )

    logger.warning(
        "INTERNAL REMOVE: Service removed successfully, updating FAISS index"
    )  # TODO: replace with debug

    # Remove from FAISS index
    await faiss_service.remove_service(service_path)

    logger.warning("INTERNAL REMOVE: Regenerating Nginx configuration")  # TODO: replace with debug

    # Regenerate Nginx configuration
    enabled_servers = {}

    for server_path in await server_service.get_enabled_services():
        server_info = await server_service.get_server_info(server_path)

        if server_info:
            enabled_servers[server_path] = server_info
    await nginx_service.generate_config_async(enabled_servers)

    logger.warning("INTERNAL REMOVE: Broadcasting health status update")  # TODO: replace with debug

    # Broadcast health status update to WebSocket clients
    await health_service.broadcast_health_update(service_path)

    logger.warning("INTERNAL REMOVE: Removing server from scopes.yml")  # TODO: replace with debug

    # Remove server from scopes.yml and reload auth server
    from ..services.scope_service import remove_server_scopes

    try:
        await remove_server_scopes(service_path)
        logger.info(f"Successfully removed server {service_path} from scopes")
    except Exception as e:
        logger.error(f"Failed to remove server {service_path} from scopes: {e}")
        # Non-fatal error - server is removed but scopes not updated

    logger.warning(
        "INTERNAL REMOVE: Removal complete, returning success response"
    )  # TODO: replace with debug
    logger.info(f"Service removed via internal endpoint: '{service_path}' by caller '{caller}'")

    return JSONResponse(
        status_code=200,
        content={
            "message": "Service removed successfully",
            "service_path": service_path,
        },
    )


@router.post("/internal/toggle")
async def internal_toggle_service(
    request: Request,
    caller: Annotated[str, Depends(validate_internal_auth)],
    service_path: Annotated[str, Form()],
):
    """Internal service toggle endpoint for mcpgw-server (requires admin authentication)."""
    from ..core.nginx_service import nginx_service
    from ..health.service import health_service
    from ..search.service import faiss_service

    logger.warning(
        "INTERNAL TOGGLE: Function called - starting execution"
    )  # TODO: replace with debug

    # Ensure service_path starts with /
    if not service_path.startswith("/"):
        service_path = "/" + service_path

    # Check if server exists
    server_info = await server_service.get_server_info(service_path)
    if not server_info:
        logger.warning(
            f"INTERNAL TOGGLE: Service not found at path '{service_path}'"
        )  # TODO: replace with debug
        return JSONResponse(
            status_code=404,
            content={
                "error": "Service not found",
                "reason": f"No service registered at path '{service_path}'",
                "suggestion": "Check the service path and ensure it is registered",
            },
        )

    logger.warning(
        "INTERNAL TOGGLE: Service found, proceeding with toggle"
    )  # TODO: replace with debug

    # Get current state and toggle it
    current_state = await server_service.is_service_enabled(service_path)
    new_state = not current_state
    success = await server_service.toggle_service(service_path, new_state)

    if not success:
        logger.warning(
            f"INTERNAL TOGGLE: Failed to toggle service at path '{service_path}'"
        )  # TODO: replace with debug
        return JSONResponse(
            status_code=500,
            content={
                "error": "Service toggle failed",
                "reason": f"Failed to toggle service at path '{service_path}'",
                "suggestion": "Check server logs for detailed error information",
            },
        )

    server_name = server_info["server_name"]
    logger.info(f"Toggled '{server_name}' ({service_path}) to {new_state} by caller '{caller}'")

    # If enabling, perform immediate health check
    status_result = "disabled"
    last_checked_iso = None
    if new_state:
        logger.info(f"Performing immediate health check for {service_path} upon toggle ON...")
        try:
            (
                status_result,
                last_checked_dt,
            ) = await health_service.perform_immediate_health_check(service_path)
            last_checked_iso = last_checked_dt.isoformat() if last_checked_dt else None
            logger.info(
                f"Immediate health check for {service_path} completed. Status: {status_result}"
            )
        except Exception as e:
            logger.error(f"ERROR during immediate health check for {service_path}: {e}")
            status_result = f"error: immediate check failed ({type(e).__name__})"
    else:
        # When disabling, set status to disabled
        status_result = "disabled"
        logger.info(f"Service {service_path} toggled OFF. Status set to disabled.")

    # Update FAISS metadata with new enabled state
    await faiss_service.add_or_update_service(service_path, server_info, new_state)

    # Regenerate Nginx configuration
    enabled_servers = {}

    for path in await server_service.get_enabled_services():
        server_info = await server_service.get_server_info(path)

        if server_info:
            enabled_servers[path] = server_info
    await nginx_service.generate_config_async(enabled_servers)

    # Broadcast health status update to WebSocket clients
    await health_service.broadcast_health_update(service_path)

    logger.warning(
        "INTERNAL TOGGLE: Toggle complete, returning success response"
    )  # TODO: replace with debug
    return JSONResponse(
        status_code=200,
        content={
            "message": "Service toggled successfully",
            "service_path": service_path,
            "new_enabled_state": new_state,
            "status": status_result,
            "last_checked_iso": last_checked_iso,
            "num_tools": server_info.get("num_tools", 0),
        },
    )


@router.post("/internal/healthcheck")
async def internal_healthcheck(
    request: Request,
    caller: Annotated[str, Depends(validate_internal_auth)],
):
    """Internal health check endpoint for mcpgw-server (requires admin authentication)."""
    from ..health.service import health_service

    logger.warning(
        "INTERNAL HEALTHCHECK: Function called - starting execution"
    )  # TODO: replace with debug

    logger.info(f"Internal healthcheck request from caller '{caller}'")

    # Get health status for all servers
    try:
        health_data = await health_service.get_all_health_status()
        logger.info(f"Retrieved health status for {len(health_data)} servers")

        return JSONResponse(status_code=200, content=health_data)

    except Exception as e:
        logger.error(f"Failed to retrieve health status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve health status: {str(e)}")


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

    server_info = await server_service.get_server_info(service_path)
    if not server_info:
        raise HTTPException(status_code=404, detail="Service path not found")

    service_name = server_info["server_name"]

    # Check if user has modify_service permission for this specific service
    if not user_has_ui_permission_for_service(
        "modify_service", service_name, user_context.get("ui_permissions", {})
    ):
        logger.warning(
            f"User {user_context['username']} attempted to access edit form for {service_name} without modify_service permission"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You do not have permission to modify {service_name}",
        )

    # For non-admin users, check if they have access to this specific server
    if not user_context["is_admin"]:
        if not await server_service.user_can_access_server_path(
            service_path, user_context["accessible_servers"]
        ):
            logger.warning(
                f"User {user_context['username']} attempted to edit service {service_path} without access"
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
    mcp_endpoint: Annotated[str | None, Form()] = None,
    metadata: Annotated[str | None, Form()] = None,
    visibility: Annotated[str, Form()] = "public",
    allowed_groups: Annotated[str | None, Form()] = None,
):
    """Handle server edit form submission (requires modify_service UI permission)."""
    from ..auth.dependencies import user_has_ui_permission_for_service
    from ..core.nginx_service import nginx_service
    from ..search.service import faiss_service

    if not service_path.startswith("/"):
        service_path = "/" + service_path

    # Check if the server exists and get service name
    server_info = await server_service.get_server_info(service_path)
    if not server_info:
        raise HTTPException(status_code=404, detail="Service path not found")

    service_name = server_info["server_name"]

    # Check if user has modify_service permission for this specific service
    if not user_has_ui_permission_for_service(
        "modify_service", service_name, user_context.get("ui_permissions", {})
    ):
        logger.warning(
            f"User {user_context['username']} attempted to edit service {service_name} without modify_service permission"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You do not have permission to modify {service_name}",
        )

    # For non-admin users, check if they have access to this specific server
    if not user_context["is_admin"]:
        if not await server_service.user_can_access_server_path(
            service_path, user_context["accessible_servers"]
        ):
            logger.warning(
                f"User {user_context['username']} attempted to edit service {service_path} without access"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to edit this server",
            )

    # Process tags
    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

    # Validate visibility value
    valid_visibility = ["public", "group-restricted", "internal"]
    if visibility not in valid_visibility:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid visibility value. Must be one of: {', '.join(valid_visibility)}",
        )

    # Process allowed_groups (comma-separated string to list)
    allowed_groups_list = []
    if allowed_groups:
        allowed_groups_list = [g.strip() for g in allowed_groups.split(",") if g.strip()]

    # Validate group-restricted requires allowed_groups
    if visibility == "group-restricted" and not allowed_groups_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="group-restricted visibility requires at least one allowed_group",
        )

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
        "tool_list": [],  # Keep existing or initialize
        "visibility": visibility,
        "allowed_groups": allowed_groups_list,
    }

    # Add optional mcp_endpoint if provided
    if mcp_endpoint:
        updated_server_entry["mcp_endpoint"] = mcp_endpoint

    # Parse and add metadata if provided
    if metadata:
        try:
            import json

            updated_server_entry["metadata"] = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON in metadata field",
            )

    # Update server
    success = await server_service.update_server(service_path, updated_server_entry)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to save updated server data")

    # Update FAISS metadata (keep current enabled state)
    is_enabled = await server_service.is_service_enabled(service_path)
    await faiss_service.add_or_update_service(service_path, updated_server_entry, is_enabled)

    # Regenerate Nginx configuration
    enabled_servers = {}

    for path in await server_service.get_enabled_services():
        server_info = await server_service.get_server_info(path)

        if server_info:
            enabled_servers[path] = server_info
    await nginx_service.generate_config_async(enabled_servers)

    logger.info(f"Server '{name}' ({service_path}) updated by user '{user_context['username']}'")

    # Redirect back to the main page
    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/tokens", response_class=HTMLResponse)
async def token_generation_page(
    request: Request, user_context: Annotated[dict, Depends(enhanced_auth)]
):
    """Show token generation page for authenticated users."""
    return templates.TemplateResponse(
        "token_generation.html",
        {
            "request": request,
            "username": user_context["username"],
            "user_context": user_context,
            "user_scopes": user_context["scopes"],
            "available_scopes": user_context["scopes"],  # For the UI to show what's available
        },
    )


@router.get("/server_details/{service_path:path}")
async def get_server_details(
    request: Request, service_path: str, user_context: Annotated[dict, Depends(enhanced_auth)]
):
    """Get server details by path, or all servers if path is 'all' (filtered by permissions)."""
    # Normalize the path to ensure it starts with '/'
    if not service_path.startswith("/"):
        service_path = "/" + service_path

    # Set audit action for server read
    if service_path == "/all":
        set_audit_action(request, "list", "server", description="List all server details")
    else:
        set_audit_action(
            request,
            "read",
            "server",
            resource_id=service_path,
            description=f"Read server details for {service_path}",
        )

    # Special case: if path is 'all' or '/all', return details for all accessible servers
    if service_path == "/all":
        if user_context["is_admin"]:
            return await server_service.get_all_servers()
        else:
            return await server_service.get_all_servers_with_permissions(
                user_context["accessible_servers"]
            )

    # Regular case: return details for a specific server
    server_info = await server_service.get_server_info(service_path)
    if not server_info:
        raise HTTPException(status_code=404, detail="Service path not registered")

    # For non-admin users, check if they have access to this specific server
    if not user_context["is_admin"]:
        if not await server_service.user_can_access_server_path(
            service_path, user_context["accessible_servers"]
        ):
            logger.warning(
                f"User {user_context['username']} attempted to access server details for {service_path} without access"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this server",
            )

    # Build versions list if this server has version routing enabled
    versions = []
    current_version = server_info.get("version", "v1.0.0")
    current_status = server_info.get("status", "stable")

    # Add current (active) version first
    versions.append(
        {
            "version": current_version,
            "proxy_pass_url": server_info.get("proxy_pass_url", ""),
            "status": current_status,
            "is_default": True,
        }
    )

    # Add other versions if they exist
    other_version_ids = server_info.get("other_version_ids", [])
    for version_id in other_version_ids:
        version_info = await server_service.get_server_info(version_id)
        if version_info:
            versions.append(
                {
                    "version": version_info.get("version", "unknown"),
                    "proxy_pass_url": version_info.get("proxy_pass_url", ""),
                    "status": version_info.get("status", "stable"),
                    "is_default": False,
                }
            )

    # Add versions to response if there are multiple versions
    if len(versions) > 1 or server_info.get("version_group"):
        server_info["versions"] = versions
        server_info["default_version"] = current_version

    return server_info


@router.get("/tools/{service_path:path}")
async def get_service_tools(
    service_path: str, user_context: Annotated[dict, Depends(enhanced_auth)]
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
            all_servers = await server_service.get_all_servers()
        else:
            all_servers = await server_service.get_all_servers_with_permissions(
                user_context["accessible_servers"]
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

        return {"service_path": "all", "tools": all_tools, "servers": all_servers_tools}

    # Handle specific server case - fetch live tools from MCP server
    server_info = await server_service.get_server_info(service_path)
    if not server_info:
        raise HTTPException(status_code=404, detail="Service path not registered")

    # For non-admin users, check if they have access to this specific server
    if not user_context["is_admin"]:
        if not await server_service.user_can_access_server_path(
            service_path, user_context["accessible_servers"]
        ):
            logger.warning(
                f"User {user_context['username']} attempted to access tools for {service_path} without access"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this server",
            )

    # Check if service is enabled and healthy
    is_enabled = await server_service.is_service_enabled(service_path)
    if not is_enabled:
        raise HTTPException(status_code=400, detail="Cannot fetch tools from disabled service")

    proxy_pass_url = server_info.get("proxy_pass_url")
    if not proxy_pass_url:
        raise HTTPException(status_code=500, detail="Service has no proxy URL configured")

    logger.info(f"Fetching live tools for {service_path} from {proxy_pass_url}")

    try:
        # Call MCP client to fetch fresh tools using server configuration
        tool_list = await mcp_client_service.get_tools_from_server_with_server_info(
            proxy_pass_url, server_info
        )

        if tool_list is None:
            # If live fetch fails but we have cached tools, use those
            cached_tools = server_info.get("tool_list")
            if cached_tools is not None and isinstance(cached_tools, list):
                logger.warning(f"Failed to fetch live tools for {service_path}, using cached tools")
                return {
                    "service_path": service_path,
                    "tools": cached_tools,
                    "cached": True,
                }
            raise HTTPException(
                status_code=503,
                detail="Failed to fetch tools from MCP server. Service may be unhealthy.",
            )

        # Update the server registry with the fresh tools
        new_tool_count = len(tool_list)
        current_tool_count = server_info.get("num_tools", 0)

        if current_tool_count != new_tool_count or server_info.get("tool_list") != tool_list:
            logger.info(f"Updating tool list for {service_path}. New count: {new_tool_count}")

            # Update server info with fresh tools
            updated_server_info = server_info.copy()
            updated_server_info["tool_list"] = tool_list
            updated_server_info["num_tools"] = new_tool_count

            # Save updated server info
            success = await server_service.update_server(service_path, updated_server_info)
            if success:
                logger.info(f"Successfully updated tool list for {service_path}")

                # Update FAISS index with new tool data
                await faiss_service.add_or_update_service(
                    service_path, updated_server_info, is_enabled
                )
                logger.info(f"Updated FAISS index for {service_path}")
            else:
                logger.error(f"Failed to save updated tool list for {service_path}")

        return {"service_path": service_path, "tools": tool_list, "cached": False}

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error fetching tools for {service_path}: {e}")
        # Try to return cached tools if available
        cached_tools = server_info.get("tool_list")
        if cached_tools is not None and isinstance(cached_tools, list):
            logger.warning(
                f"Error fetching live tools for {service_path}, falling back to cached tools: {e}"
            )
            return {"service_path": service_path, "tools": cached_tools, "cached": True}
        raise HTTPException(status_code=500, detail=f"Error fetching tools: {str(e)}")


@router.post("/refresh/{service_path:path}")
async def refresh_service(service_path: str, user_context: Annotated[dict, Depends(enhanced_auth)]):
    """Refresh service health and tool information (requires health_check_service permission)."""
    from ..auth.dependencies import user_has_ui_permission_for_service
    from ..core.nginx_service import nginx_service
    from ..health.service import health_service
    from ..search.service import faiss_service

    if not service_path.startswith("/"):
        service_path = "/" + service_path

    server_info = await server_service.get_server_info(service_path)
    if not server_info:
        raise HTTPException(status_code=404, detail="Service path not registered")

    service_name = server_info["server_name"]

    # Check if user has health_check_service permission for this specific service
    if not user_has_ui_permission_for_service(
        "health_check_service", service_name, user_context.get("ui_permissions", {})
    ):
        logger.warning(
            f"User {user_context['username']} attempted to refresh service {service_name} without health_check_service permission"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You do not have permission to refresh {service_name}",
        )

    # For non-admin users, check if they have access to this specific server
    if not user_context["is_admin"]:
        if not await server_service.user_can_access_server_path(
            service_path, user_context["accessible_servers"]
        ):
            logger.warning(
                f"User {user_context['username']} attempted to refresh service {service_path} without access"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this server",
            )

    # Check if service is enabled
    is_enabled = await server_service.is_service_enabled(service_path)
    if not is_enabled:
        raise HTTPException(status_code=400, detail="Cannot refresh disabled service")

    proxy_pass_url = server_info.get("proxy_pass_url")
    if not proxy_pass_url:
        raise HTTPException(status_code=500, detail="Service has no proxy URL configured")

    logger.info(
        f"Refreshing service {service_path} at {proxy_pass_url} by user '{user_context['username']}'"
    )

    try:
        # Perform immediate health check
        status, last_checked_dt = await health_service.perform_immediate_health_check(service_path)
        last_checked_iso = last_checked_dt.isoformat() if last_checked_dt else None
        logger.info(f"Manual refresh health check for {service_path} completed. Status: {status}")

        # Regenerate Nginx config after manual refresh
        logger.info(f"Regenerating Nginx config after manual refresh for {service_path}...")
        enabled_servers = {}

        for path in await server_service.get_enabled_services():
            path_server_info = await server_service.get_server_info(path)

            if path_server_info:
                enabled_servers[path] = path_server_info
        await nginx_service.generate_config_async(enabled_servers)

    except Exception as e:
        logger.error(f"ERROR during manual refresh check for {service_path}: {e}")
        # Still broadcast the error state
        await health_service.broadcast_health_update(service_path)
        raise HTTPException(status_code=500, detail=f"Refresh check failed: {e}")

    # Update FAISS index
    await faiss_service.add_or_update_service(service_path, server_info, is_enabled)

    # Broadcast the updated status
    await health_service.broadcast_health_update(service_path)

    logger.info(f"Service '{service_path}' refreshed by user '{user_context['username']}'")
    return {
        "message": f"Service {service_path} refreshed successfully",
        "service_path": service_path,
        "status": status,
        "last_checked_iso": last_checked_iso,
        "num_tools": server_info.get("num_tools", 0),
    }


async def _add_server_to_groups_impl(
    server_name: str,
    group_names: str,
) -> JSONResponse:
    """
    Internal implementation for adding server to groups.

    This function contains the business logic for adding a server to groups
    and can be called from both Basic Auth and JWT endpoints.
    """
    from ..services.scope_service import add_server_to_groups

    # Parse group names from comma-separated string
    groups = [group.strip() for group in group_names.split(",") if group.strip()]
    if not groups:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid group names provided",
        )

    # Convert server name to path format
    server_path = f"/{server_name}" if not server_name.startswith("/") else server_name

    try:
        success = await add_server_to_groups(server_path, groups)

        if success:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Server successfully added to groups",
                    "server_path": server_path,
                    "groups": groups,
                },
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to add server to groups",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding server {server_path} to groups {groups}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@router.post("/internal/add-to-groups")
async def internal_add_server_to_groups(
    request: Request,
    caller: Annotated[str, Depends(validate_internal_auth)],
    server_name: Annotated[str, Form()],
    group_names: Annotated[str, Form()],  # Comma-separated list
):
    """Internal endpoint to add a server to specific scopes groups (requires admin authentication)."""
    logger.info(f"Adding server to groups via internal endpoint by caller '{caller}'")

    # Call the shared implementation
    return await _add_server_to_groups_impl(server_name, group_names)


async def _remove_server_from_groups_impl(
    server_name: str,
    group_names: str,
) -> JSONResponse:
    """
    Internal implementation for removing server from groups.

    This function contains the business logic for removing a server from groups
    and can be called from both Basic Auth and JWT endpoints.
    """
    from ..services.scope_service import remove_server_from_groups

    # Parse group names from comma-separated string
    groups = [group.strip() for group in group_names.split(",") if group.strip()]
    if not groups:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid group names provided",
        )

    # Convert server name to path format
    server_path = f"/{server_name}" if not server_name.startswith("/") else server_name

    try:
        success = await remove_server_from_groups(server_path, groups)

        if success:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Server successfully removed from groups",
                    "server_path": server_path,
                    "groups": groups,
                },
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to remove server from groups",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing server {server_path} from groups {groups}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@router.post("/internal/remove-from-groups")
async def internal_remove_server_from_groups(
    request: Request,
    caller: Annotated[str, Depends(validate_internal_auth)],
    server_name: Annotated[str, Form()],
    group_names: Annotated[str, Form()],  # Comma-separated list
):
    """Internal endpoint to remove a server from specific scopes groups (requires admin authentication)."""
    logger.info(f"Removing server from groups via internal endpoint by caller '{caller}'")

    # Call the shared implementation
    return await _remove_server_from_groups_impl(server_name, group_names)


@router.get("/internal/list")
async def internal_list_services(
    request: Request,
    caller: Annotated[str, Depends(validate_internal_auth)],
):
    """Internal service listing endpoint for mcpgw-server (requires admin authentication)."""
    logger.warning(
        "INTERNAL LIST: Function called - starting execution"
    )  # TODO: replace with debug

    logger.info(f"Internal service list request from caller '{caller}'")

    # Get all servers (admin access - no permission filtering)
    all_servers = await server_service.get_all_servers()

    logger.warning(f"INTERNAL LIST: Found {len(all_servers)} servers")  # TODO: replace with debug

    # Transform the data to include enabled status and health information
    services = []
    for service_path, server_info in all_servers.items():
        from ..health.service import health_service

        # Get real health status from health service
        health_data = health_service._get_service_health_data(service_path)

        service_data = {
            "server_name": server_info.get("server_name", "Unknown"),
            "path": service_path,
            "description": server_info.get("description", ""),
            "proxy_pass_url": server_info.get("proxy_pass_url", ""),
            "is_enabled": await server_service.is_service_enabled(service_path),
            "tags": server_info.get("tags", []),
            "num_tools": server_info.get("num_tools", 0),
            "num_stars": server_info.get("num_stars", 0),
            "is_python": server_info.get("is_python", False),
            "license": server_info.get("license", "N/A"),
            "health_status": health_data["status"],
            "last_checked_iso": health_data["last_checked_iso"],
            "tool_list": server_info.get("tool_list", []),
        }
        services.append(service_data)

    logger.warning(f"INTERNAL LIST: Returning {len(services)} services")  # TODO: replace with debug
    logger.info(
        f"Internal service list completed for caller '{caller}' - returned {len(services)} services"
    )

    return JSONResponse(
        status_code=200,
        content={"services": services, "total_count": len(services)},
    )


@router.post("/internal/create-group")
async def internal_create_group(
    request: Request,
    caller: Annotated[str, Depends(validate_internal_auth)],
    group_name: Annotated[str, Form()],
    description: Annotated[str, Form()] = "",
    create_in_idp: Annotated[bool, Form()] = True,
):
    """Internal endpoint to create a new group in both IdP and scopes.yml (requires admin authentication)."""
    logger.info(f"Creating group '{group_name}' via internal endpoint by caller '{caller}'")

    # Call the shared implementation
    return await _create_group_impl(group_name, description, create_in_idp)


@router.post("/internal/delete-group")
async def internal_delete_group(
    request: Request,
    caller: Annotated[str, Depends(validate_internal_auth)],
    group_name: Annotated[str, Form()],
    delete_from_keycloak: Annotated[bool, Form()] = True,
    force: Annotated[bool, Form()] = False,
):
    """Internal endpoint to delete a group from both Keycloak and scopes (requires admin authentication)."""
    logger.info(f"Deleting group '{group_name}' via internal endpoint by caller '{caller}'")

    # Call the shared implementation
    return await _delete_group_impl(group_name, delete_from_idp=delete_from_keycloak, force=force)


async def _list_groups_impl(
    include_idp: bool = True,
    include_scopes: bool = True,
) -> JSONResponse:
    """
    Internal implementation for listing groups.

    This function contains the business logic for listing groups
    and can be called from both Basic Auth and JWT endpoints.
    Uses the IAMManager abstraction to support any identity provider.
    """
    from ..services.scope_service import list_groups
    from ..utils.iam_manager import get_iam_manager

    try:
        result = {
            "keycloak_groups": [],
            "scopes_groups": {},
            "synchronized": [],
            "keycloak_only": [],
            "scopes_only": [],
        }

        # Get groups from identity provider (Keycloak, Entra ID, etc.)
        idp_group_names = set()
        if include_idp:
            try:
                iam = get_iam_manager()
                idp_groups = await iam.list_groups()
                result["keycloak_groups"] = [
                    {
                        "name": group.get("name"),
                        "id": group.get("id"),
                        "path": group.get("path", ""),
                    }
                    for group in idp_groups
                ]
                idp_group_names = {group.get("name") for group in idp_groups}
                logger.info(f"Found {len(idp_groups)} groups in identity provider")
            except Exception as e:
                logger.error(f"Failed to list identity provider groups: {e}")
                result["keycloak_error"] = str(e)

        # Get groups from scopes (file or OpenSearch based on STORAGE_BACKEND)
        scopes_group_names = set()
        if include_scopes:
            try:
                scopes_data = await list_groups()
                result["scopes_groups"] = scopes_data.get("groups", {})
                scopes_group_names = set(scopes_data.get("groups", {}).keys())
                logger.info(f"Found {len(scopes_group_names)} groups in scopes")
            except Exception as e:
                logger.error(f"Failed to list scopes groups: {e}")
                result["scopes_error"] = str(e)

        # Find synchronized and out-of-sync groups
        if include_idp and include_scopes:
            result["synchronized"] = list(idp_group_names & scopes_group_names)
            result["keycloak_only"] = list(idp_group_names - scopes_group_names)
            result["scopes_only"] = list(scopes_group_names - idp_group_names)

        return JSONResponse(status_code=200, content=result)

    except Exception as e:
        logger.error(f"Error listing groups: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@router.get("/internal/list-groups")
async def internal_list_groups(
    request: Request,
    caller: Annotated[str, Depends(validate_internal_auth)],
    include_keycloak: bool = True,
    include_scopes: bool = True,
):
    """Internal endpoint to list groups from Keycloak and/or scopes (requires admin authentication)."""
    logger.info(f"Listing groups via internal endpoint by caller '{caller}'")

    # Call the shared implementation
    return await _list_groups_impl(include_idp=include_keycloak, include_scopes=include_scopes)


@router.post("/tokens/generate")
async def generate_user_token(
    request: Request, user_context: Annotated[dict, Depends(enhanced_auth)]
):
    """
    Generate a JWT token for the authenticated user.

    Request body should contain:
    {
        "requested_scopes": ["scope1", "scope2"],  // Optional, defaults to user's current scopes
        "expires_in_hours": 8,                     // Optional, defaults to 8 hours
        "description": "Token for automation"      // Optional description
    }

    Returns:
        Generated JWT token with expiration info

    Raises:
        HTTPException: If request fails or user lacks permissions
    """
    try:
        # Parse request body
        try:
            body = await request.json()
        except Exception as e:
            logger.warning(f"Invalid JSON in token generation request: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON in request body")

        requested_scopes = body.get("requested_scopes", [])
        expires_in_hours = body.get("expires_in_hours", 8)
        description = body.get("description", "")

        # Validate expires_in_hours
        if not isinstance(expires_in_hours, int) or expires_in_hours <= 0 or expires_in_hours > 24:
            raise HTTPException(
                status_code=400,
                detail="expires_in_hours must be an integer between 1 and 24",
            )

        # Validate requested_scopes
        if requested_scopes and not isinstance(requested_scopes, list):
            raise HTTPException(
                status_code=400, detail="requested_scopes must be a list of strings"
            )

        # Get full session data to include stored OAuth tokens
        from ..auth.dependencies import get_user_session_data

        try:
            session_cookie = request.cookies.get(settings.session_cookie_name)
            logger.info(f"Session cookie present: {bool(session_cookie)}")
            session_data = get_user_session_data(session_cookie)
            logger.info(
                f"Session data extracted: auth_method={session_data.get('auth_method')}, "
                f"provider={session_data.get('provider')}, "
                f"has_access_token={bool(session_data.get('access_token'))}, "
                f"has_refresh_token={bool(session_data.get('refresh_token'))}"
            )
        except Exception as e:
            logger.warning(f"Could not get session data for tokens: {e}")
            session_data = {}

        # Prepare request to auth server
        # Include user identity info for self-signed JWT generation
        auth_request = {
            "user_context": {
                "username": user_context["username"],
                "email": user_context.get("email", session_data.get("email", "")),
                "scopes": user_context["scopes"],
                "groups": user_context["groups"],
                "provider": user_context.get("provider", session_data.get("provider")),
                "auth_method": user_context.get("auth_method", session_data.get("auth_method")),
            },
            "requested_scopes": requested_scopes,
            "expires_in_hours": expires_in_hours,
            "description": description,
        }

        # Call auth server internal API (no authentication needed since both are trusted internal services)
        async with httpx.AsyncClient() as client:
            headers = {"Content-Type": "application/json"}

            auth_server_url = settings.auth_server_url
            response = await client.post(
                f"{auth_server_url}/internal/tokens",
                json=auth_request,
                headers=headers,
                timeout=10.0,
            )

            if response.status_code == 200:
                token_data = response.json()
                logger.info(f"Successfully generated token for user '{user_context['username']}'")

                # Format response to match expected structure (including refresh token)
                formatted_response = {
                    "success": True,
                    "tokens": {
                        "access_token": token_data.get("access_token"),
                        "refresh_token": token_data.get("refresh_token"),
                        "expires_in": token_data.get("expires_in"),
                        "refresh_expires_in": token_data.get("refresh_expires_in"),
                        "token_type": token_data.get("token_type", "Bearer"),
                        "scope": token_data.get("scope", ""),
                    },
                    "keycloak_url": getattr(settings, "keycloak_url", None)
                    or "http://keycloak:8080",
                    "realm": getattr(settings, "keycloak_realm", None) or "mcp-gateway",
                    "client_id": "user-generated",
                    # Legacy fields for backward compatibility
                    "token_data": token_data,
                    "user_scopes": user_context["scopes"],
                    "requested_scopes": requested_scopes or user_context["scopes"],
                }

                return formatted_response
            else:
                error_detail = "Unknown error"
                try:
                    error_response = response.json()
                    error_detail = error_response.get("detail", "Unknown error")
                except:
                    error_detail = response.text

                logger.warning(f"Auth server returned error {response.status_code}: {error_detail}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Token generation failed: {error_detail}",
                )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error generating token for user '{user_context['username']}': {e}"
        )
        raise HTTPException(status_code=500, detail="Internal error generating token")


# ============================================================================
# NEW API: /api/servers/* endpoints with JWT Bearer Token Authentication
# ============================================================================
# These are the modern, JWT-authenticated equivalents of the /api/internal/*
# endpoints. They use Depends(nginx_proxied_auth) for authentication and
# support fine-grained permission checks via user context.
#
# Architecture:
# - Both /api/internal/* and /api/servers/* call the same internal functions
# - No code duplication; external API simply wraps existing endpoints
# - User context from JWT is passed through for audit logging
#
# Migration Path:
# Phase 1 (Now): Both endpoints work identically with same business logic
# Phase 2 (Future): Clients migrate to /api/servers/*
# Phase 3 (Future): /api/internal/* deprecated with sunset headers
# Phase 4 (Future): /api/internal/* removed in major version


@router.post("/servers/register")
async def register_service_api(
    request: Request,
    name: Annotated[str, Form()],
    description: Annotated[str, Form()],
    path: Annotated[str, Form()],
    proxy_pass_url: Annotated[str, Form()],
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    tags: Annotated[str, Form()] = "",
    num_tools: Annotated[int, Form()] = 0,
    num_stars: Annotated[int, Form()] = 0,
    is_python: Annotated[bool, Form()] = False,
    license_str: Annotated[str, Form(alias="license")] = "N/A",
    overwrite: Annotated[bool, Form()] = True,
    auth_provider: Annotated[str | None, Form()] = None,
    auth_type: Annotated[str | None, Form()] = None,
    supported_transports: Annotated[str | None, Form()] = None,
    headers: Annotated[str | None, Form()] = None,
    tool_list_json: Annotated[str | None, Form()] = None,
    mcp_endpoint: Annotated[str | None, Form()] = None,
    sse_endpoint: Annotated[str | None, Form()] = None,
    metadata: Annotated[str | None, Form()] = None,
    version: Annotated[str | None, Form()] = None,
    status: Annotated[str | None, Form()] = None,
):
    """
    Register a service via JWT Bearer Token authentication (External API).

    This endpoint provides the same functionality as POST /api/internal/register
    but uses modern JWT Bearer token authentication via nginx headers, making it
    suitable for external service-to-service communication.

    **Authentication:** JWT Bearer token (via nginx X-User header)
    **Authorization:** Requires valid JWT token from auth system

    **Request body (form data):**
    - `name` (required): Service name
    - `description` (required): Service description
    - `path` (required): Service path (e.g., /myservice)
    - `proxy_pass_url` (required): Proxy URL (e.g., http://localhost:8000)
    - `tags` (optional): Comma-separated tags
    - `num_tools` (optional): Number of tools
    - `num_stars` (optional): Star rating
    - `is_python` (optional): Is Python server (boolean)
    - `license` (optional): License name
    - `overwrite` (optional): Overwrite if exists (boolean, default true)
    - `auth_provider` (optional): Auth provider name
    - `auth_type` (optional): Auth type (e.g., oauth, basic)
    - `supported_transports` (optional): JSON array of transports
    - `headers` (optional): JSON object of headers
    - `tool_list_json` (optional): JSON array of tool definitions
    - `mcp_endpoint` (optional): Full URL for custom MCP endpoint (overrides /mcp suffix)
    - `sse_endpoint` (optional): Full URL for custom SSE endpoint (overrides /sse suffix)
    - `version` (optional): Server version (e.g., v1.0.0, v2.0.0)
    - `status` (optional): Version status (stable, beta, deprecated)

    **Response:**
    - `201 Created`: Service registered successfully
    - `400 Bad Request`: Invalid input data
    - `401 Unauthorized`: Missing or invalid JWT token
    - `409 Conflict`: Service already exists with same version (different version auto-creates new version)
    - `500 Internal Server Error`: Server error

    **Example:**
    ```bash
    curl -X POST https://registry.example.com/api/servers/register \\
      -H "Authorization: Bearer $JWT_TOKEN" \\
      -F "name=My Service" \\
      -F "description=My MCP Service" \\
      -F "path=/myservice" \\
      -F "proxy_pass_url=http://localhost:8000"
    ```
    """
    # Set audit action for server registration
    set_audit_action(
        request, "create", "server", resource_id=path, description=f"Register server {name}"
    )

    logger.info(
        f"API register service request from user '{user_context.get('username')}' for service '{name}'"
    )

    # Implementation extracted from internal_register_service to avoid duplicating auth logic
    # Auth is already validated by nginx_proxied_auth dependency
    from ..health.service import health_service
    from ..search.service import faiss_service

    # Validate path format
    if not path.startswith("/"):
        path = "/" + path
    logger.warning(f"SERVERS REGISTER: Validated path: {path}")

    # Process tags
    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []

    # Process supported_transports
    if supported_transports:
        try:
            transports_list = (
                json.loads(supported_transports)
                if supported_transports.startswith("[")
                else [t.strip() for t in supported_transports.split(",")]
            )
        except Exception as e:
            logger.warning(
                f"SERVERS REGISTER: Failed to parse supported_transports, using default: {e}"
            )
            transports_list = ["streamable-http"]
    else:
        transports_list = ["streamable-http"]

    # Process headers
    headers_list = []
    if headers:
        try:
            headers_list = json.loads(headers) if isinstance(headers, str) else headers
        except Exception as e:
            logger.warning(f"SERVERS REGISTER: Failed to parse headers: {e}")

    # Process tool_list
    tool_list = []
    if tool_list_json:
        try:
            tool_list = (
                json.loads(tool_list_json) if isinstance(tool_list_json, str) else tool_list_json
            )
        except Exception as e:
            logger.warning(f"SERVERS REGISTER: Failed to parse tool_list_json: {e}")

    # Create server entry
    server_entry = {
        "server_name": name,
        "description": description,
        "path": path,
        "proxy_pass_url": proxy_pass_url,
        "supported_transports": transports_list,
        "auth_type": auth_type if auth_type else "none",
        "tags": tag_list,
        "num_tools": num_tools,
        "num_stars": num_stars,
        "is_python": is_python,
        "license": license_str,
        "tool_list": tool_list,
    }

    # Add optional fields if provided
    if auth_provider:
        server_entry["auth_provider"] = auth_provider
    if headers_list:
        server_entry["headers"] = headers_list
    if mcp_endpoint:
        server_entry["mcp_endpoint"] = mcp_endpoint
    if sse_endpoint:
        server_entry["sse_endpoint"] = sse_endpoint
    if version:
        server_entry["version"] = version
    if status:
        server_entry["status"] = status
    if metadata:
        try:
            server_entry["metadata"] = (
                json.loads(metadata) if isinstance(metadata, str) else metadata
            )
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Invalid metadata",
                    "reason": "metadata must be valid JSON",
                    "detail": "Provide metadata as a JSON string",
                },
            )

    # Check if server exists and handle overwrite/version logic
    existing_server = await server_service.get_server_info(path)

    # If server exists with a different version, register_server will auto-create new version
    # Only reject if overwrite=False AND it's the same version (or no version specified)
    if existing_server and not overwrite:
        existing_version = existing_server.get("version", "v1.0.0")
        new_version = version
        # If versions are different, let register_server handle it as a new version
        if not new_version or new_version == existing_version:
            logger.warning(
                f"SERVERS REGISTER: Server exists with same version and overwrite=False for path {path}"
            )
            return JSONResponse(
                status_code=409,
                content={
                    "error": "Service registration failed",
                    "reason": f"A service with path '{path}' already exists with version {existing_version}",
                    "detail": "Use overwrite=true to replace, or specify a different version",
                },
            )

    try:
        # Register service (use update_server if overwriting, otherwise register_server)
        if existing_server and overwrite:
            logger.info(
                f"Overwriting existing server at path {path} by user {user_context.get('username')}"
            )
            success = await server_service.update_server(path, server_entry)
            is_new_version = False
        else:
            result = await server_service.register_server(server_entry)
            success = result["success"]
            is_new_version = result.get("is_new_version", False)

        if not success:
            logger.error(
                f"Service registration failed for {path}: {result.get('message', 'unknown error')}"
            )
            return JSONResponse(
                status_code=409,
                content={
                    "error": "Service registration failed",
                    "detail": "Check server logs for more information",
                },
            )

        if is_new_version:
            logger.info(f"New version registered for {path} by user {user_context.get('username')}")
        else:
            logger.info(
                f"Service registered successfully via API: {path} by user {user_context.get('username')}"
            )

        # Security scanning if enabled
        await _perform_security_scan_on_registration(
            path, proxy_pass_url, server_entry, headers_list
        )

        # Trigger async tasks for health check and FAISS sync
        asyncio.create_task(health_service.perform_immediate_health_check(path))
        asyncio.create_task(faiss_service.save_data())

        return JSONResponse(
            status_code=201,
            content={
                "path": path,
                "name": name,
                "message": f"Service '{name}' registered successfully at path '{path}'",
            },
        )

    except Exception as e:
        logger.error(f"Service registration failed for {path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Service registration failed: {str(e)}")


@router.post("/servers/toggle")
async def toggle_service_api(
    request: Request,
    path: Annotated[str, Form()],
    new_state: Annotated[bool, Form()],
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    Toggle a service's enabled/disabled state via JWT authentication (External API).

    This endpoint provides the same functionality as POST /api/internal/toggle
    but uses modern JWT Bearer token authentication.

    **Authentication:** JWT Bearer token (via nginx X-User header)
    **Authorization:** Requires valid JWT token from auth system

    **Request body (form data):**
    - `path` (required): Service path
    - `new_state` (required): New state (true=enabled, false=disabled)

    **Response:**
    Returns the updated service status.

    **Example:**
    ```bash
    curl -X POST https://registry.example.com/api/servers/toggle \\
      -H "Authorization: Bearer $JWT_TOKEN" \\
      -F "path=/myservice" \\
      -F "new_state=true"
    ```
    """
    from ..core.nginx_service import nginx_service
    from ..health.service import health_service
    from ..search.service import faiss_service

    # Set audit action for server toggle
    set_audit_action(
        request, "toggle", "server", resource_id=path, description=f"Toggle server to {new_state}"
    )

    logger.info(
        f"API toggle service request from user '{user_context.get('username')}' for path '{path}' to {new_state}"
    )

    # Normalize path
    if not path.startswith("/"):
        path = "/" + path

    # Check if server exists
    server_info = await server_service.get_server_info(path)
    if not server_info:
        raise HTTPException(status_code=404, detail="Service path not registered")

    # Toggle the service
    success = await server_service.toggle_service(path, new_state)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to toggle service")

    logger.info(
        f"Toggled '{server_info['server_name']}' ({path}) to {new_state} by user '{user_context.get('username')}'"
    )

    # If enabling, perform immediate health check
    status = "disabled"
    last_checked_iso = None
    if new_state:
        logger.info(f"Performing immediate health check for {path} upon toggle ON...")
        try:
            (
                status,
                last_checked_dt,
            ) = await health_service.perform_immediate_health_check(path)
            last_checked_iso = last_checked_dt.isoformat() if last_checked_dt else None
            logger.info(f"Immediate health check for {path} completed. Status: {status}")
        except Exception as e:
            logger.error(f"ERROR during immediate health check for {path}: {e}")
            status = f"error: immediate check failed ({type(e).__name__})"
    else:
        # When disabling, set status to disabled
        status = "disabled"
        logger.info(f"Service {path} toggled OFF. Status set to disabled.")

    # Update FAISS metadata with new enabled state
    await faiss_service.add_or_update_service(path, server_info, new_state)

    # Regenerate Nginx configuration
    enabled_servers = {}

    for server_path in await server_service.get_enabled_services():
        server_info = await server_service.get_server_info(server_path)

        if server_info:
            enabled_servers[server_path] = server_info
    await nginx_service.generate_config_async(enabled_servers)

    # Broadcast health status update to WebSocket clients
    await health_service.broadcast_health_update(path)

    return JSONResponse(
        status_code=200,
        content={
            "message": f"Toggle request for {path} processed.",
            "service_path": path,
            "new_enabled_state": new_state,
            "status": status,
            "last_checked_iso": last_checked_iso,
            "num_tools": server_info.get("num_tools", 0),
        },
    )


@router.post("/servers/remove")
async def remove_service_api(
    request: Request,
    path: Annotated[str, Form()],
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    Remove a service via JWT Bearer Token authentication (External API).

    This endpoint provides the same functionality as POST /api/internal/remove
    but uses modern JWT Bearer token authentication.

    **Authentication:** JWT Bearer token (via nginx X-User header)
    **Authorization:** Requires valid JWT token from auth system

    **Request body (form data):**
    - `path` (required): Service path to remove

    **Response:**
    Returns confirmation of removal.

    **Example:**
    ```bash
    curl -X POST https://registry.example.com/api/servers/remove \\
      -H "Authorization: Bearer $JWT_TOKEN" \\
      -F "path=/myservice"
    ```
    """
    from ..core.nginx_service import nginx_service
    from ..health.service import health_service
    from ..search.service import faiss_service
    from ..services.scope_service import remove_server_scopes

    # Set audit action for server removal
    set_audit_action(
        request, "delete", "server", resource_id=path, description=f"Remove server at {path}"
    )

    logger.info(
        f"API remove service request from user '{user_context.get('username')}' for path '{path}'"
    )

    # Normalize path
    if not path.startswith("/"):
        path = "/" + path

    # Check if server exists
    server_info = await server_service.get_server_info(path)
    if not server_info:
        logger.warning(f"Service not found at path '{path}'")
        return JSONResponse(
            status_code=404,
            content={
                "error": "Service not found",
                "reason": f"No service registered at path '{path}'",
                "suggestion": "Check the service path and ensure it is registered",
            },
        )

    # Block deletion of federated (read-only) servers from peer registries
    sync_metadata = server_info.get("sync_metadata", {})
    if sync_metadata.get("is_federated") or sync_metadata.get("is_read_only"):
        source_peer = sync_metadata.get("source_peer_id", "unknown peer registry")
        logger.warning(
            f"User {user_context.get('username')} attempted to delete federated server {path} "
            f"from {source_peer}"
        )
        return JSONResponse(
            status_code=403,
            content={
                "error": "Cannot delete federated server",
                "reason": f"Server '{path}' is synced from {source_peer} and cannot be deleted locally",
                "suggestion": "Delete this server from its source registry, or remove the peer federation",
            },
        )

    # Fine-grained delete permission check (gateway already validated api.servers access)
    if not user_context.get("is_admin", False):
        ui_permissions = user_context.get("ui_permissions", {})
        delete_service_perms = ui_permissions.get("delete_service", [])
        server_name = path.strip("/")
        if "all" not in delete_service_perms and server_name not in delete_service_perms:
            logger.warning(f"User {user_context.get('username')} denied delete for server {path}")
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Permission denied",
                    "reason": f"User does not have delete_service permission for '{path}'",
                },
            )

    # Remove the server
    success = await server_service.remove_server(path)

    if not success:
        logger.warning(f"Failed to remove service at path '{path}'")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Service removal failed",
                "reason": f"Failed to remove service at path '{path}'",
                "suggestion": "Check server logs for detailed error information",
            },
        )

    logger.info(f"Service removed successfully: {path} by user {user_context.get('username')}")

    # Remove from FAISS index
    await faiss_service.remove_service(path)

    # Regenerate Nginx configuration
    enabled_servers = {}

    for server_path in await server_service.get_enabled_services():
        server_info = await server_service.get_server_info(server_path)

        if server_info:
            enabled_servers[server_path] = server_info
    await nginx_service.generate_config_async(enabled_servers)

    # Broadcast health status update to WebSocket clients
    await health_service.broadcast_health_update(path)

    # Remove server from scopes.yml and reload auth server
    try:
        await remove_server_scopes(path)
        logger.info(f"Successfully removed server {path} from scopes")
    except Exception as e:
        logger.warning(f"Failed to remove server {path} from scopes: {e}")

    return JSONResponse(
        status_code=200,
        content={"message": "Service removed successfully", "path": path},
    )


# IMPORTANT: Specific routes with path suffixes (/health, /rate, /rating, /toggle)
# must come BEFORE catch-all /servers/ routes to prevent FastAPI from matching them incorrectly


@router.get("/servers/health")
async def healthcheck_api(
    request: Request,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    Get health status for all registered services via JWT authentication (External API).

    This endpoint provides the same functionality as GET /api/internal/healthcheck
    but uses modern JWT Bearer token authentication.

    **Authentication:** JWT Bearer token (via nginx X-User header)
    **Authorization:** Requires valid JWT token from auth system

    **Response:**
    Returns health status for all services.

    **Example:**
    ```bash
    curl -X GET https://registry.example.com/api/servers/health \\
      -H "Authorization: Bearer $JWT_TOKEN"
    ```
    """
    from ..health.service import health_service

    logger.info(
        f"API healthcheck request from user '{user_context.get('username') if user_context else 'unknown'}'"
    )

    # Get health status for all servers using JWT authentication
    try:
        health_data = await health_service.get_all_health_status()
        logger.info(f"Retrieved health status for {len(health_data)} servers")

        return JSONResponse(status_code=200, content=health_data)

    except Exception as e:
        logger.error(f"Failed to retrieve health status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve health status: {str(e)}")


@router.post("/servers/groups/add")
async def add_server_to_groups_api(
    request: Request,
    server_name: Annotated[str, Form()],
    group_names: Annotated[str, Form()],
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    Add a service to scope groups via JWT authentication (External API).

    This endpoint provides the same functionality as POST /api/internal/add-to-groups
    but uses modern JWT Bearer token authentication.

    **Authentication:** JWT Bearer token (via nginx X-User header)
    **Authorization:** Requires valid JWT token from auth system

    **Request body (form data):**
    - `server_name` (required): Service name
    - `group_names` (required): Comma-separated list of group names

    **Response:**
    Returns confirmation of group assignment.

    **Example:**
    ```bash
    curl -X POST https://registry.example.com/api/servers/groups/add \\
      -H "Authorization: Bearer $JWT_TOKEN" \\
      -F "server_name=myservice" \\
      -F "group_names=admin,developers"
    ```
    """
    logger.info(
        f"API add to groups request from user '{user_context.get('username')}' for server '{server_name}'"
    )

    # Call the shared implementation
    return await _add_server_to_groups_impl(server_name, group_names)


@router.post("/servers/groups/remove")
async def remove_server_from_groups_api(
    request: Request,
    server_name: Annotated[str, Form()],
    group_names: Annotated[str, Form()],
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    Remove a service from scope groups via JWT authentication (External API).

    This endpoint provides the same functionality as POST /api/internal/remove-from-groups
    but uses modern JWT Bearer token authentication.

    **Authentication:** JWT Bearer token (via nginx X-User header)
    **Authorization:** Requires valid JWT token from auth system

    **Request body (form data):**
    - `server_name` (required): Service name
    - `group_names` (required): Comma-separated list of group names to remove

    **Response:**
    Returns confirmation of removal from groups.

    **Example:**
    ```bash
    curl -X POST https://registry.example.com/api/servers/groups/remove \\
      -H "Authorization: Bearer $JWT_TOKEN" \\
      -F "server_name=myservice" \\
      -F "group_names=developers"
    ```
    """
    logger.info(
        f"API remove from groups request from user '{user_context.get('username')}' for server '{server_name}'"
    )

    # Call the shared implementation
    return await _remove_server_from_groups_impl(server_name, group_names)


async def _create_group_impl(
    group_name: str,
    description: str = "",
    create_in_idp: bool = True,
) -> JSONResponse:
    """
    Internal implementation for group creation.

    This function contains the business logic for creating a group
    and can be called from both Basic Auth and JWT endpoints.
    """
    from ..services.scope_service import create_group
    from ..utils.iam_manager import get_iam_manager

    # Validate group name
    if not group_name or not group_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Group name is required"
        )

    try:
        # Create in IdP first if requested
        idp_created = False
        if create_in_idp:
            try:
                iam_manager = get_iam_manager()
                # Check if group already exists in IdP
                existing_groups = await iam_manager.list_groups()
                group_exists = any(
                    g.get("name", "").lower() == group_name.lower() for g in existing_groups
                )
                if group_exists:
                    logger.warning(f"Group '{group_name}' already exists in IdP")
                else:
                    await iam_manager.create_group(group_name, description)
                    idp_created = True
                    logger.info(f"Group '{group_name}' created in IdP")
            except Exception as e:
                logger.error(f"Failed to create group in IdP: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to create group in IdP: {str(e)}",
                )

        # Create in scopes (file or OpenSearch based on STORAGE_BACKEND)
        scopes_success = await create_group(group_name, description)

        if scopes_success:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Group successfully created",
                    "group_name": group_name,
                    "created_in_idp": idp_created,
                    "created_in_scopes": True,
                },
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create group in scopes (may already exist)",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating group '{group_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@router.post("/servers/groups/create")
async def create_group_api(
    request: Request,
    group_name: Annotated[str, Form()],
    description: Annotated[str, Form()] = "",
    create_in_idp: Annotated[bool, Form()] = True,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    Create a new scope group via JWT authentication (External API).

    This endpoint provides the same functionality as POST /api/internal/create-group
    but uses modern JWT Bearer token authentication.

    **Authentication:** JWT Bearer token (via nginx X-User header)
    **Authorization:** Requires valid JWT token from auth system

    **Request body (form data):**
    - `group_name` (required): Name of the new group
    - `description` (optional): Group description
    - `create_in_idp` (optional): Whether to create in IdP (default: true)

    **Response:**
    Returns confirmation of group creation.

    **Example:**
    ```bash
    curl -X POST https://registry.example.com/api/servers/groups/create \\
      -H "Authorization: Bearer $JWT_TOKEN" \\
      -F "group_name=new-team" \\
      -F "description=Team for new project" \\
      -F "create_in_idp=true"
    ```
    """
    logger.info(
        f"API create group request from user '{user_context.get('username')}' for group '{group_name}'"
    )

    # Call the shared implementation
    return await _create_group_impl(group_name, description, create_in_idp)


async def _delete_group_impl(
    group_name: str,
    delete_from_idp: bool = True,
    force: bool = False,
) -> JSONResponse:
    """
    Internal implementation for group deletion.

    This function contains the business logic for deleting a group
    and can be called from both Basic Auth and JWT endpoints.
    Uses the IAMManager abstraction to support any identity provider.
    """
    from ..services.scope_service import delete_group
    from ..utils.iam_manager import get_iam_manager

    # Validate group name
    if not group_name or not group_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Group name is required"
        )

    # Prevent deletion of system groups unless force=True
    if not force:
        system_groups = [
            "UI-Scopes",
            "group_mappings",
            "mcp-registry-admin",
            "mcp-registry-user",
            "mcp-registry-developer",
            "mcp-registry-operator",
        ]

        if group_name in system_groups:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Cannot delete system group '{group_name}'. Use force=true to override.",
            )

    try:
        # Delete from scopes (file or OpenSearch)
        scopes_success = await delete_group(group_name, remove_from_mappings=True)

        if not scopes_success:
            logger.warning(f"Group '{group_name}' not found in scopes or deletion failed")

        # Delete from identity provider if requested
        idp_deleted = False
        if delete_from_idp:
            try:
                iam = get_iam_manager()
                if await iam.group_exists(group_name):
                    await iam.delete_group(group_name)
                    idp_deleted = True
                    logger.info(f"Group '{group_name}' deleted from identity provider")
                else:
                    logger.warning(f"Group '{group_name}' not found in identity provider")
            except Exception as e:
                logger.error(f"Failed to delete group from identity provider: {e}")
                # Continue anyway - scopes deletion might have succeeded

        if scopes_success or idp_deleted:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Group deletion completed",
                    "group_name": group_name,
                    "deleted_from_keycloak": idp_deleted,
                    "deleted_from_scopes": scopes_success,
                },
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Group '{group_name}' not found in either identity provider or scopes",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting group '{group_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@router.post("/servers/groups/delete")
async def delete_group_api(
    request: Request,
    group_name: Annotated[str, Form()],
    delete_from_keycloak: Annotated[bool, Form()] = True,
    force: Annotated[bool, Form()] = False,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    Delete a scope group via JWT authentication (External API).

    This endpoint provides the same functionality as POST /api/internal/delete-group
    but uses modern JWT Bearer token authentication.

    **Authentication:** JWT Bearer token (via nginx X-User header)
    **Authorization:** Requires valid JWT token from auth system

    **Request body (form data):**
    - `group_name` (required): Name of the group to delete
    - `delete_from_keycloak` (optional): Whether to delete from Keycloak (default: true)
    - `force` (optional): Force deletion of system groups (default: false)

    **Response:**
    Returns confirmation of group deletion.

    **Example:**
    ```bash
    curl -X POST https://registry.example.com/api/servers/groups/delete \\
      -H "Authorization: Bearer $JWT_TOKEN" \\
      -F "group_name=old-team" \\
      -F "delete_from_keycloak=true" \\
      -F "force=false"
    ```
    """
    logger.info(
        f"API delete group request from user '{user_context.get('username')}' for group '{group_name}'"
    )

    # Call the shared implementation
    return await _delete_group_impl(group_name, delete_from_idp=delete_from_keycloak, force=force)


@router.get("/servers/groups")
async def list_groups_api(
    request: Request,
    include_keycloak: bool = True,
    include_scopes: bool = True,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    List all scope groups via JWT Bearer Token authentication (External API).

    This endpoint provides the same functionality as GET /api/internal/list-groups
    but uses modern JWT Bearer token authentication.

    **Authentication:** JWT Bearer token (via nginx X-User header)
    **Authorization:** Requires valid JWT token from auth system

    **Response:**
    Returns a list of all groups and their synchronization status.

    **Example:**
    ```bash
    curl -X GET https://registry.example.com/api/servers/groups \\
      -H "Authorization: Bearer $JWT_TOKEN"
    ```
    """
    logger.info(
        f"API list groups request from user '{user_context.get('username') if user_context else 'unknown'}'"
    )

    # Call the shared implementation
    return await _list_groups_impl(include_idp=include_keycloak, include_scopes=include_scopes)


@router.get("/servers/groups/{group_name}")
async def get_group_api(
    group_name: str,
    request: Request,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    Get full details of a specific group via JWT Bearer Token authentication (External API).

    This endpoint retrieves complete information about a group including server_access,
    group_mappings, and ui_permissions from the scopes storage backend.

    **Authentication:** JWT Bearer token (via nginx X-User header)
    **Authorization:** Requires valid JWT token from auth system

    **Response:**
    Returns complete group definition including:
    - scope_name: Name of the scope/group
    - scope_type: Type of scope (e.g., "server_scope")
    - description: Description of the group
    - server_access: List of server access definitions
    - group_mappings: List of group mappings
    - ui_permissions: UI permissions configuration
    - created_at: Creation timestamp
    - updated_at: Last update timestamp

    **Example:**
    ```bash
    curl -X GET https://registry.example.com/api/servers/groups/currenttime-users \\
      -H "Authorization: Bearer $JWT_TOKEN"
    ```
    """
    from ..services.scope_service import get_group

    logger.info(
        f"API get group request from user '{user_context.get('username') if user_context else 'unknown'}' "
        f"for group '{group_name}'"
    )

    try:
        group_data = await get_group(group_name)

        if not group_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Group '{group_name}' not found",
            )

        return JSONResponse(status_code=200, content=group_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting group {group_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@router.post("/servers/groups/import")
async def import_group_definition(
    request: Request,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    Import a complete group definition via JSON (External API).

    This endpoint accepts a complete group definition including all three document types
    (server_scope, group_mapping, ui_scope) and creates/updates them in the storage backend.

    **Authentication:** JWT Bearer token (via nginx X-User header)
    **Authorization:** Requires valid JWT token from auth system

    **Request Body:**
    ```json
    {
      "scope_name": "group-name",
      "scope_type": "server_scope",
      "description": "Group description",
      "server_access": [
        {
          "server": "currenttime",
          "methods": ["initialize", "tools/list", "tools/call"],
          "tools": ["current_time_by_timezone"]
        }
      ],
      "group_mappings": ["group-name", "other-group"],
      "ui_permissions": {
        "list_service": ["currenttime", "mcpgw"],
        "list_agents": ["/code-reviewer"],
        "health_check_service": ["currenttime"]
      },
      "create_in_idp": true
    }
    ```

    **Required Fields:**
    - `scope_name`: Name of the scope/group

    **Optional Fields:**
    - `scope_type`: Type of scope (default: "server_scope")
    - `description`: Description of the group
    - `server_access`: List of server access definitions
    - `group_mappings`: List of group names this group maps to
    - `ui_permissions`: Dictionary of UI permissions
    - `create_in_idp`: Whether to create the group in IdP (default: true)

    **Example:**
    ```bash
    curl -X POST https://registry.example.com/api/servers/groups/import \\
      -H "Authorization: Bearer $JWT_TOKEN" \\
      -H "Content-Type: application/json" \\
      -d @group-definition.json
    ```
    """
    from ..services.scope_service import import_group
    from ..utils.iam_manager import get_iam_manager

    try:
        # Parse request body
        body = await request.json()

        # Required field
        scope_name = body.get("scope_name")
        if not scope_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="scope_name is required",
            )

        # Optional fields
        scope_type = body.get("scope_type", "server_scope")
        description = body.get("description", "")
        server_access = body.get("server_access")
        group_mappings = body.get("group_mappings")
        ui_permissions = body.get("ui_permissions")
        # Support both create_in_idp (new) and create_in_keycloak (legacy)
        create_in_idp = body.get("create_in_idp", body.get("create_in_keycloak", True))

        logger.info(
            f"API import group request from user '{user_context.get('username') if user_context else 'unknown'}' "
            f"for group '{scope_name}'"
        )

        # Import group definition
        success = await import_group(
            scope_name=scope_name,
            scope_type=scope_type,
            description=description,
            server_access=server_access,
            group_mappings=group_mappings,
            ui_permissions=ui_permissions,
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to import group {scope_name}",
            )

        # Create in IdP if requested
        idp_created = False
        idp_group_id = None
        if create_in_idp:
            try:
                iam_manager = get_iam_manager()
                result = await iam_manager.create_group(scope_name, description)
                if result:
                    idp_created = True
                    idp_group_id = result.get("id")
                    logger.info(f"Created group {scope_name} in IdP with ID: {idp_group_id}")

                    # For Entra ID only: add group ID (GUID) to group_mappings
                    # Entra returns GUIDs in tokens, while Keycloak returns group names
                    # This ensures token group claims match scope group_mappings
                    auth_provider = os.environ.get("AUTH_PROVIDER", "keycloak").lower()
                    if auth_provider == "entra" and idp_group_id:
                        from ..services.scope_service import (
                            add_group_mapping_to_scope,
                        )

                        mapping_success = await add_group_mapping_to_scope(scope_name, idp_group_id)
                        if mapping_success:
                            logger.info(
                                f"Added Entra group ID {idp_group_id} to scope "
                                f"{scope_name} group_mappings"
                            )
                        else:
                            logger.warning(f"Failed to add Entra group ID to scope {scope_name}")
                else:
                    logger.warning(
                        f"Failed to create group {scope_name} in IdP (may already exist)"
                    )
            except Exception as e:
                logger.error(f"Error creating IdP group {scope_name}: {e}")

        # Trigger auth server reload
        from ..services.scope_service import trigger_auth_server_reload

        reload_success = await trigger_auth_server_reload()

        return JSONResponse(
            status_code=200,
            content={
                "message": f"Group {scope_name} imported successfully",
                "group_name": scope_name,
                "idp_created": idp_created,
                "auth_server_reloaded": reload_success,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error importing group definition: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@router.post("/servers/{path:path}/rate")
async def rate_server(
    request: Request,
    path: str,
    rating_request: RatingRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
):
    """Save integer ratings to server."""
    # Set audit action for server rating
    set_audit_action(
        request,
        "rate",
        "server",
        resource_id=path,
        description=f"Rate server with {rating_request.rating}",
    )

    if not path.startswith("/"):
        path = "/" + path

    server_info = await server_service.get_server_info(path)
    # Try with trailing slash if not found (path normalization)
    if not server_info and not path.endswith("/"):
        path_with_slash = path + "/"
        server_info = await server_service.get_server_info(path_with_slash)
        if server_info:
            path = path_with_slash
    if not server_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Server not found at path '{path}'",
        )

    # Use the actual path from the server (handles trailing slash normalization)
    actual_path = server_info.get("path", path)

    # For non-admin users, check if they have access to this specific server
    if not user_context["is_admin"]:
        if not await server_service.user_can_access_server_path(
            actual_path, user_context["accessible_servers"]
        ):
            logger.warning(
                f"User {user_context['username']} attempted to rate server {actual_path} without permission"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this server",
            )

    try:
        avg_rating = await server_service.update_rating(
            actual_path, user_context["username"], rating_request.rating
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error updating rating: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save rating",
        )

    return {
        "message": "Rating added successfully",
        "average_rating": avg_rating,
    }


@router.get("/servers/{path:path}/rating")
async def get_server_rating(
    path: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
):
    """Get server rating information."""
    if not path.startswith("/"):
        path = "/" + path

    server_info = await server_service.get_server_info(path)
    # Try with trailing slash if not found (path normalization)
    if not server_info and not path.endswith("/"):
        path_with_slash = path + "/"
        server_info = await server_service.get_server_info(path_with_slash)
        if server_info:
            path = path_with_slash
    if not server_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Server not found at path '{path}'",
        )

    # For non-admin users, check if they have access to this specific server
    if not user_context["is_admin"]:
        if not await server_service.user_can_access_server_path(
            path, user_context["accessible_servers"]
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this server",
            )

    return {
        "num_stars": server_info.get("num_stars", 0.0),
        "rating_details": server_info.get("rating_details", []),
    }


@router.get("/servers/{path:path}/security-scan")
async def get_server_security_scan(
    path: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
):
    """
    Get security scan results for a server.

    Returns the latest security scan results for the specified server,
    including threat analysis, severity levels, and detailed findings.

    **Authentication:** JWT Bearer token or session cookie
    **Authorization:** Requires admin privileges or access to the server

    **Path Parameters:**
    - `path` (required): Server path (e.g., /cloudflare-docs)

    **Response:**
    Returns security scan results with analysis_results and tool_results.

    **Example:**
    ```bash
    curl -X GET http://localhost/api/servers/cloudflare-docs/security-scan \\
      --cookie-jar .cookies --cookie .cookies
    ```
    """
    if not path.startswith("/"):
        path = "/" + path

    # Check if server exists
    server_info = await server_service.get_server_info(path)
    if not server_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Server not found at path '{path}'",
        )

    # Check user permissions
    if not user_context["is_admin"]:
        if not await server_service.user_can_access_server_path(
            path, user_context["accessible_servers"]
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this server",
            )

    # Get scan results
    scan_result = await security_scanner_service.get_scan_result(path)
    if not scan_result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No security scan results found for server '{path}'. "
            "The server may not have been scanned yet.",
        )

    return scan_result


@router.post("/servers/{path:path}/rescan")
async def rescan_server(
    path: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
):
    """
    Trigger a manual security scan for a server.

    Initiates a new security scan for the specified server and returns
    the results. This endpoint is useful for re-scanning servers after
    updates or for on-demand security assessments.

    **Authentication:** JWT Bearer token or session cookie
    **Authorization:** Requires admin privileges

    **Path Parameters:**
    - `path` (required): Server path (e.g., /cloudflare-docs)

    **Response:**
    Returns the newly generated security scan results.

    **Example:**
    ```bash
    curl -X POST http://localhost/api/servers/cloudflare-docs/rescan \\
      --cookie-jar .cookies --cookie .cookies
    ```
    """
    # Only admins can trigger manual scans
    if not user_context["is_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can trigger security scans",
        )

    if not path.startswith("/"):
        path = "/" + path

    # Check if server exists
    server_info = await server_service.get_server_info(path)
    if not server_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Server not found at path '{path}'",
        )

    # Get server URL from server info
    server_url = server_info.get("proxy_pass_url")
    if not server_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Server '{path}' does not have a proxy_pass_url configured",
        )

    logger.info(
        f"Manual security scan requested by user '{user_context.get('username')}' "
        f"for server '{path}' at URL '{server_url}'"
    )

    try:
        # Trigger security scan
        scan_result = await security_scanner_service.scan_server(
            server_url=server_url,
            server_path=path,
            analyzers=None,
            api_key=None,
            headers=None,
            timeout=None,
        )

        # Return the scan result data
        return {
            "server_url": scan_result.server_url,
            "server_path": path,
            "scan_timestamp": scan_result.scan_timestamp,
            "is_safe": scan_result.is_safe,
            "critical_issues": scan_result.critical_issues,
            "high_severity": scan_result.high_severity,
            "medium_severity": scan_result.medium_severity,
            "low_severity": scan_result.low_severity,
            "analyzers_used": scan_result.analyzers_used,
            "scan_failed": scan_result.scan_failed,
            "error_message": scan_result.error_message,
            "raw_output": scan_result.raw_output,
        }
    except Exception as e:
        logger.exception(f"Failed to scan server '{path}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to scan server: {str(e)}",
        )


@router.get("/servers/tools/{service_path:path}")
async def get_service_tools_api(
    service_path: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    Get tool list for a service via JWT Bearer Token authentication (External API).

    This endpoint provides the same functionality as GET /tools/{service_path}
    but uses modern JWT Bearer token authentication.

    **Authentication:** JWT Bearer token (via nginx X-User header)
    **Authorization:** Requires valid JWT token from auth system

    **Path Parameters:**
    - `service_path` (required): Service path (e.g., /myservice or /all for all services)

    **Response:**
    Returns the list of tools available on the service, filtered by user permissions.

    **Example:**
    ```bash
    curl -X GET https://registry.example.com/api/servers/tools/myservice \\
      -H "Authorization: Bearer $JWT_TOKEN"

    # Get tools from all accessible services
    curl -X GET https://registry.example.com/api/servers/tools/all \\
      -H "Authorization: Bearer $JWT_TOKEN"
    ```
    """
    logger.info(
        f"API get tools request from user '{user_context.get('username') if user_context else 'unknown'}' for path '{service_path}'"
    )

    # Call the existing get_service_tools function
    return await get_service_tools(service_path=service_path, user_context=user_context)


# ============================================================================
# Server Version Management Endpoints
# ============================================================================


class SetDefaultVersion(BaseModel):
    """Request model for setting default version."""

    version: str


@router.delete("/servers/{service_path:path}/versions/{version}")
async def remove_server_version(
    service_path: str,
    version: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    Remove a version from a server.

    Args:
        service_path: Server path (URL encoded)
        version: Version to remove

    Returns:
        Success message
    """
    decoded_path = "/" + service_path if not service_path.startswith("/") else service_path

    try:
        result = await server_service.remove_server_version(path=decoded_path, version=version)

        if result:
            return {
                "status": "success",
                "message": f"Version {version} removed from {decoded_path}",
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to remove version"
            )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.put("/servers/{service_path:path}/versions/default")
async def set_default_version(
    service_path: str,
    version_data: SetDefaultVersion,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    Set the default (latest) version for a server.

    Args:
        service_path: Server path (URL encoded)
        version_data: Contains version to set as default

    Returns:
        Success message
    """
    decoded_path = "/" + service_path if not service_path.startswith("/") else service_path

    try:
        result = await server_service.set_default_version(
            path=decoded_path, version=version_data.version
        )

        if result:
            return {
                "status": "success",
                "message": f"Default version set to {version_data.version} for {decoded_path}",
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to set default version",
            )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/servers/{service_path:path}/versions")
async def get_server_versions(
    service_path: str,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
):
    """
    Get all versions for a server.

    Args:
        service_path: Server path (URL encoded)

    Returns:
        Version information
    """
    decoded_path = "/" + service_path if not service_path.startswith("/") else service_path

    try:
        return await server_service.get_server_versions(decoded_path)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
