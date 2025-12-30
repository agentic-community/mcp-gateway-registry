from __future__ import annotations

import logging
from dataclasses import (
    dataclass,
)
from typing import (
    Any,
    Callable,
)

from fastapi import (
    HTTPException,
    status,
)
from fastapi.responses import (
    JSONResponse,
)

from .server_routes_common import (
    _apply_remove_side_effects,
    _apply_toggle_side_effects,
    _build_server_entry_from_form,
    _enforce_proxy_pass_url_allowlist,
)

RequireUserContextFn = Callable[[dict | None], dict]
CreateTaskFn = Callable[[Any], Any]


@dataclass(frozen=True, slots=True)
class RegisterResult:
    path: str
    server_entry: dict
    existing_server: dict | None
    conflict: bool
    success: bool


def _normalize_service_path(
    service_path: str,
) -> str:
    if not service_path.startswith("/"):
        return "/" + service_path
    return service_path


def _register_core(
    *,
    name: str,
    description: str,
    path: str,
    proxy_pass_url: str,
    tags: str,
    num_tools: int,
    num_stars: int,
    is_python: bool,
    license_str: str,
    overwrite: bool,
    auth_provider: str | None,
    auth_type: str | None,
    upstream_auth: str | None,
    supported_transports: str | None,
    headers: str | None,
    tool_list_json: str | None,
    server_service_obj: Any,
    logger: logging.Logger,
) -> RegisterResult:
    normalized_path = _normalize_service_path(path)
    _enforce_proxy_pass_url_allowlist(proxy_pass_url=proxy_pass_url)

    normalized_path, server_entry = _build_server_entry_from_form(
        name=name,
        description=description,
        path=normalized_path,
        proxy_pass_url=proxy_pass_url,
        tags=tags,
        num_tools=num_tools,
        num_stars=num_stars,
        is_python=is_python,
        license_str=license_str,
        auth_provider=auth_provider,
        auth_type=auth_type,
        upstream_auth=upstream_auth,
        supported_transports=supported_transports,
        headers=headers,
        tool_list_json=tool_list_json,
        logger=logger,
    )

    existing_server = server_service_obj.get_server_info(normalized_path)
    if existing_server and not overwrite:
        return RegisterResult(
            path=normalized_path,
            server_entry=server_entry,
            existing_server=existing_server,
            conflict=True,
            success=False,
        )

    if existing_server and overwrite:
        success = server_service_obj.update_server(normalized_path, server_entry)
    else:
        success = server_service_obj.register_server(server_entry)

    return RegisterResult(
        path=normalized_path,
        server_entry=server_entry,
        existing_server=existing_server,
        conflict=False,
        success=bool(success),
    )


async def _register_service_internal(
    *,
    name: str,
    description: str,
    path: str,
    proxy_pass_url: str,
    tags: str,
    num_tools: int,
    num_stars: int,
    is_python: bool,
    license_str: str,
    overwrite: bool,
    auth_provider: str | None,
    auth_type: str | None,
    upstream_auth: str | None,
    supported_transports: str | None,
    headers: str | None,
    tool_list_json: str | None,
    user_context: dict | None,
    require_user_context: RequireUserContextFn,
    server_service_obj: Any,
    logger: logging.Logger,
) -> JSONResponse:
    from ..core.nginx_service import nginx_service
    from ..health.service import health_service
    from ..search.service import faiss_service

    user_context = require_user_context(user_context)
    username = user_context.get("username", "unknown")
    logger.info(f"Internal service registration request from admin user '{username}'")

    result = _register_core(
        name=name,
        description=description,
        path=path,
        proxy_pass_url=proxy_pass_url,
        tags=tags,
        num_tools=num_tools,
        num_stars=num_stars,
        is_python=is_python,
        license_str=license_str,
        overwrite=overwrite,
        auth_provider=auth_provider,
        auth_type=auth_type,
        upstream_auth=upstream_auth,
        supported_transports=supported_transports,
        headers=headers,
        tool_list_json=tool_list_json,
        server_service_obj=server_service_obj,
        logger=logger,
    )

    if result.conflict:
        return JSONResponse(
            status_code=409,
            content={
                "error": "Service registration failed",
                "reason": f"A service with path '{result.path}' already exists",
                "suggestion": "Set overwrite=true or use the remove command first",
            },
        )

    if not result.success:
        return JSONResponse(
            status_code=409,
            content={
                "error": "Service registration failed",
                "reason": f"Failed to register service at path '{result.path}'",
                "suggestion": "Check server logs for detailed error information",
            },
        )

    # Automatically enable the newly registered server BEFORE FAISS indexing
    try:
        toggle_success = server_service_obj.toggle_service(result.path, True)
        if toggle_success:
            logger.info(f"Successfully auto-enabled server {result.path} after registration")
        else:
            logger.warning(f"Failed to auto-enable server {result.path} after registration")
    except Exception as exc:
        logger.error(f"Error auto-enabling server {result.path}: {exc}")
        # Non-fatal error - server is registered but not enabled

    # Add to FAISS index with current enabled state (should be True after auto-enable)
    is_enabled = server_service_obj.is_service_enabled(result.path)
    await faiss_service.add_or_update_service(result.path, result.server_entry, is_enabled)

    # Regenerate Nginx configuration
    enabled_servers = {
        server_path: server_service_obj.get_server_info(server_path)
        for server_path in server_service_obj.get_enabled_services()
    }
    await nginx_service.generate_config_async(enabled_servers)

    # Broadcast health status update to WebSocket clients
    await health_service.broadcast_health_update(result.path)

    # Update scopes.yml with the new server's tools
    from ..utils.scopes_manager import update_server_scopes

    tool_names = []
    if "tool_list" in result.server_entry and result.server_entry["tool_list"]:
        tool_names = [
            tool["name"] for tool in result.server_entry["tool_list"] if "name" in tool
        ]

    try:
        await update_server_scopes(result.path, name, tool_names)
        logger.info(
            f"Successfully updated scopes for server {result.path} with {len(tool_names)} tools"
        )
    except Exception as exc:
        logger.error(f"Failed to update scopes for server {result.path}: {exc}")
        # Non-fatal error - server is registered but scopes not updated

    logger.info(
        f"New service registered via internal endpoint: '{name}' at path '{result.path}' by admin '{username}'"
    )

    return JSONResponse(
        status_code=201,
        content={
            "message": "Service registered successfully",
            "service": result.server_entry,
        },
    )


async def _register_service_external(
    *,
    name: str,
    description: str,
    path: str,
    proxy_pass_url: str,
    tags: str,
    num_tools: int,
    num_stars: int,
    is_python: bool,
    license_str: str,
    overwrite: bool,
    auth_provider: str | None,
    auth_type: str | None,
    upstream_auth: str | None,
    supported_transports: str | None,
    headers: str | None,
    tool_list_json: str | None,
    user_context: dict | None,
    require_user_context: RequireUserContextFn,
    server_service_obj: Any,
    create_task: CreateTaskFn,
    logger: logging.Logger,
) -> JSONResponse:
    from ..core.nginx_service import nginx_service
    from ..health.service import health_service
    from ..search.service import faiss_service

    require_user_context(user_context)

    result = _register_core(
        name=name,
        description=description,
        path=path,
        proxy_pass_url=proxy_pass_url,
        tags=tags,
        num_tools=num_tools,
        num_stars=num_stars,
        is_python=is_python,
        license_str=license_str,
        overwrite=overwrite,
        auth_provider=auth_provider,
        auth_type=auth_type,
        upstream_auth=upstream_auth,
        supported_transports=supported_transports,
        headers=headers,
        tool_list_json=tool_list_json,
        server_service_obj=server_service_obj,
        logger=logger,
    )

    if result.conflict:
        return JSONResponse(
            status_code=409,
            content={
                "error": "Service registration failed",
                "reason": f"A service with path '{result.path}' already exists",
                "detail": "Use overwrite=true to replace existing service",
            },
        )

    if not result.success:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Service registration failed",
                "reason": f"Failed to register service at path '{result.path}'",
                "detail": "Check server logs for more information",
            },
        )

    # Automatically enable the newly registered server BEFORE FAISS indexing
    try:
        toggle_success = server_service_obj.toggle_service(result.path, True)
        if toggle_success:
            logger.info("Successfully auto-enabled server %s after registration", result.path)
        else:
            logger.warning("Failed to auto-enable server %s after registration", result.path)
    except Exception as exc:
        logger.error("Error auto-enabling server %s: %s", result.path, exc)
        # Non-fatal error - server is registered but not enabled

    is_enabled = server_service_obj.is_service_enabled(result.path)
    await faiss_service.add_or_update_service(
        result.path,
        result.server_entry,
        is_enabled,
    )

    enabled_servers = {
        server_path: server_service_obj.get_server_info(server_path)
        for server_path in server_service_obj.get_enabled_services()
    }
    await nginx_service.generate_config_async(enabled_servers)
    await health_service.broadcast_health_update(result.path)

    from ..utils.scopes_manager import update_server_scopes

    tool_names = []
    if "tool_list" in result.server_entry and result.server_entry["tool_list"]:
        tool_names = [
            tool["name"] for tool in result.server_entry["tool_list"] if "name" in tool
        ]

    try:
        await update_server_scopes(result.path, name, tool_names)
        logger.info(
            "Successfully updated scopes for server %s with %s tools",
            result.path,
            len(tool_names),
        )
    except Exception as exc:
        logger.error("Failed to update scopes for server %s: %s", result.path, exc)
        # Non-fatal error - server is registered but scopes not updated

    create_task(health_service.perform_immediate_health_check(result.path))
    create_task(faiss_service.save_data())

    return JSONResponse(
        status_code=201,
        content={
            "path": result.path,
            "name": name,
            "message": f"Service '{name}' registered successfully at path '{result.path}'",
        },
    )


async def _toggle_service_common(
    *,
    path: str,
    desired_state: bool,
    server_info: dict,
    server_service_obj: Any,
    logger: logging.Logger,
) -> tuple[str, str | None]:
    from ..core.nginx_service import nginx_service
    from ..health.service import health_service
    from ..search.service import faiss_service

    status_str, last_checked_iso = await _apply_toggle_side_effects(
        service_path=path,
        server_info=server_info,
        new_state=desired_state,
        server_service=server_service_obj,
        faiss_service=faiss_service,
        health_service=health_service,
        nginx_service=nginx_service,
        logger=logger,
    )

    return status_str, last_checked_iso


async def _toggle_service_internal(
    *,
    service_path: str,
    user_context: dict | None,
    require_user_context: RequireUserContextFn,
    server_service_obj: Any,
    logger: logging.Logger,
) -> JSONResponse:
    user_context = require_user_context(user_context)
    username = user_context.get("username", "unknown")

    normalized_path = _normalize_service_path(service_path)
    server_info = server_service_obj.get_server_info(normalized_path)
    if not server_info:
        return JSONResponse(
            status_code=404,
            content={
                "error": "Service not found",
                "reason": f"No service registered at path '{normalized_path}'",
                "suggestion": "Check the service path and ensure it is registered",
            },
        )

    current_state = server_service_obj.is_service_enabled(normalized_path)
    desired_state = not current_state
    success = server_service_obj.toggle_service(normalized_path, desired_state)
    if not success:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Service toggle failed",
                "reason": f"Failed to toggle service at path '{normalized_path}'",
                "suggestion": "Check server logs for detailed error information",
            },
        )

    server_name = server_info["server_name"]
    logger.info(
        f"Toggled '{server_name}' ({normalized_path}) to {desired_state} by admin '{username}'"
    )

    status_str, last_checked_iso = await _toggle_service_common(
        path=normalized_path,
        desired_state=desired_state,
        server_info=server_info,
        server_service_obj=server_service_obj,
        logger=logger,
    )

    return JSONResponse(
        status_code=200,
        content={
            "message": "Service toggled successfully",
            "service_path": normalized_path,
            "new_enabled_state": desired_state,
            "status": status_str,
            "last_checked_iso": last_checked_iso,
            "num_tools": server_info.get("num_tools", 0),
        },
    )


async def _toggle_service_external(
    *,
    path: str | None,
    service_path: str | None,
    new_state: bool | None,
    user_context: dict | None,
    require_user_context: RequireUserContextFn,
    server_service_obj: Any,
    logger: logging.Logger,
) -> JSONResponse:
    require_user_context(user_context)

    raw_path = path or service_path
    if raw_path is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="path is required",
        )

    normalized_path = _normalize_service_path(raw_path)
    server_info = server_service_obj.get_server_info(normalized_path)
    if not server_info:
        raise HTTPException(status_code=404, detail="Service path not registered")

    desired_state = new_state
    if desired_state is None:
        desired_state = not server_service_obj.is_service_enabled(normalized_path)

    success = server_service_obj.toggle_service(normalized_path, desired_state)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to toggle service")

    status_str, last_checked_iso = await _toggle_service_common(
        path=normalized_path,
        desired_state=desired_state,
        server_info=server_info,
        server_service_obj=server_service_obj,
        logger=logger,
    )

    return JSONResponse(
        status_code=200,
        content={
            "message": f"Toggle request for {normalized_path} processed.",
            "path": normalized_path,
            "is_enabled": desired_state,
            "service_path": normalized_path,
            "new_enabled_state": desired_state,
            "status": status_str,
            "last_checked_iso": last_checked_iso,
            "num_tools": server_info.get("num_tools", 0),
        },
    )


async def _remove_service_internal(
    *,
    service_path: str,
    user_context: dict | None,
    require_user_context: RequireUserContextFn,
    server_service_obj: Any,
    logger: logging.Logger,
) -> JSONResponse:
    from ..core.nginx_service import nginx_service
    from ..health.service import health_service
    from ..search.service import faiss_service
    from ..utils.scopes_manager import remove_server_scopes

    user_context = require_user_context(user_context)
    username = user_context.get("username", "unknown")

    normalized_path = _normalize_service_path(service_path)
    logger.info(
        f"Internal service removal request from admin user '{username}' for service '{normalized_path}'"
    )

    server_info = server_service_obj.get_server_info(normalized_path)
    if not server_info:
        return JSONResponse(
            status_code=404,
            content={
                "error": "Service not found",
                "reason": f"No service registered at path '{normalized_path}'",
                "suggestion": "Check the service path and ensure it is registered",
            },
        )

    success = server_service_obj.remove_server(normalized_path)
    if not success:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Service removal failed",
                "reason": f"Failed to remove service at path '{normalized_path}'",
                "suggestion": "Check server logs for detailed error information",
            },
        )

    await _apply_remove_side_effects(
        service_path=normalized_path,
        server_service=server_service_obj,
        faiss_service=faiss_service,
        health_service=health_service,
        nginx_service=nginx_service,
        remove_server_scopes=remove_server_scopes,
        scopes_error_log_level="error",
        logger=logger,
    )

    logger.info(
        f"Service removed via internal endpoint: '{normalized_path}' by admin '{username}'"
    )

    return JSONResponse(
        status_code=200,
        content={
            "message": "Service removed successfully",
            "service_path": normalized_path,
        },
    )


async def _remove_service_external(
    *,
    path: str,
    user_context: dict | None,
    require_user_context: RequireUserContextFn,
    server_service_obj: Any,
    logger: logging.Logger,
) -> JSONResponse:
    from ..core.nginx_service import nginx_service
    from ..health.service import health_service
    from ..search.service import faiss_service
    from ..utils.scopes_manager import remove_server_scopes

    require_user_context(user_context)
    normalized_path = _normalize_service_path(path)

    server_info = server_service_obj.get_server_info(normalized_path)
    if not server_info:
        logger.warning("Service not found at path '%s'", normalized_path)
        return JSONResponse(
            status_code=404,
            content={
                "error": "Service not found",
                "reason": f"No service registered at path '{normalized_path}'",
                "suggestion": "Check the service path and ensure it is registered",
            },
        )

    success = server_service_obj.remove_server(normalized_path)
    if not success:
        logger.warning("Failed to remove service at path '%s'", normalized_path)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Service removal failed",
                "reason": f"Failed to remove service at path '{normalized_path}'",
                "suggestion": "Check server logs for detailed error information",
            },
        )

    await _apply_remove_side_effects(
        service_path=normalized_path,
        server_service=server_service_obj,
        faiss_service=faiss_service,
        health_service=health_service,
        nginx_service=nginx_service,
        remove_server_scopes=remove_server_scopes,
        scopes_error_log_level="warning",
        logger=logger,
    )

    return JSONResponse(
        status_code=200,
        content={
            "message": "Service removed successfully",
            "path": normalized_path,
        },
    )
