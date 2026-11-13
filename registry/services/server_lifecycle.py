"""Shared steps for registering and removing an MCP server (issue #2022).

``clean_up_removed_server`` is the body of ``POST /api/servers/remove`` after
it deletes the record, so the public route and OpenAPI MCP servers clean up
search, nginx, scopes and webhooks the same way.

``register_and_enable_managed_server`` registers a server the registry itself
owns (an OpenAPI MCP server), runs the same registration gate as
``/api/servers/register``, enables it, waits for its nginx route and its tool
list, and starts the same security scan.
"""

import asyncio
import logging

from ..utils.credential_encryption import strip_credentials_from_dict
from .openapi_errors import OpenApiServerError
from .registration_gate_service import check_registration_gate
from .server_service import server_service
from .webhook_service import send_registration_webhook

logger = logging.getLogger(__name__)


async def _remove_search_entry(
    path: str,
) -> None:
    try:
        from ..repositories.factory import get_search_repository

        await get_search_repository().remove_entity(path)
    except Exception as e:
        logger.warning(f"Failed to remove search index for {path}: {e}")


async def _flush_nginx() -> None:
    from ..core.nginx_service import nginx_reload_scheduler

    nginx_reload_scheduler.mark_dirty()
    await nginx_reload_scheduler.flush_now()


async def _remove_scopes(
    path: str,
) -> None:
    from .scope_service import remove_server_scopes

    try:
        await remove_server_scopes(path)
        logger.info(f"Successfully removed server {path} from scopes")
    except Exception as e:
        logger.warning(f"Failed to remove server {path} from scopes: {e}")


async def clean_up_removed_server(
    path: str,
    server_info: dict,
    performed_by: str | None,
) -> None:
    """Remove everything derived from a server record that was just deleted.

    Removes its search entry and nginx location (flushed before returning) and
    its scopes, tells WebSocket clients, and sends the deletion webhook.
    """
    from ..health.service import health_service

    logger.info(f"Service removed successfully: {path} by user {performed_by}")
    await _remove_search_entry(path)
    await _flush_nginx()
    await health_service.broadcast_health_update(path)
    await _remove_scopes(path)
    asyncio.create_task(
        send_registration_webhook(
            event_type="deletion",
            registration_type="server",
            card_data=strip_credentials_from_dict(server_info),
            performed_by=performed_by,
        )
    )


async def remove_server_everywhere(
    path: str,
    server_info: dict,
    performed_by: str | None,
) -> bool:
    """Delete a server record, then everything derived from it.

    Returns:
        False when the record could not be removed; nothing else ran.
    """
    if not await server_service.remove_server(path):
        return False
    await clean_up_removed_server(path, server_info, performed_by)
    return True


async def _check_gate(
    server_entry: dict,
    source_api: str,
    raw_headers: list[tuple[bytes, bytes]],
) -> None:
    gate_result = await check_registration_gate(
        asset_type="server",
        operation="register",
        source_api=source_api,
        registration_payload=server_entry,
        raw_headers=raw_headers,
    )
    if not gate_result.allowed:
        logger.warning(
            f"Registration gate denied server '{server_entry['path']}': {gate_result.error_message}"
        )
        raise OpenApiServerError(
            403,
            "registration_denied",
            f"Registration denied by policy gate: {gate_result.error_message}",
        )


def _start_security_scan(
    server_entry: dict,
) -> None:
    # The scan helper lives in the API module and several tests patch it there,
    # so it is imported from there rather than moved.
    from ..api.server_routes import _perform_security_scan_on_registration

    asyncio.create_task(
        _perform_security_scan_on_registration(
            server_entry["path"], server_entry["proxy_pass_url"], server_entry
        )
    )


async def register_and_enable_managed_server(
    server_entry: dict,
    performed_by: str,
    source_api: str,
    raw_headers: list[tuple[bytes, bytes]],
) -> None:
    """Register, enable and route a registry-owned server.

    Raises:
        OpenApiServerError: 403 when the registration gate denies it, 409 when
            the registry refuses the record.
    """
    path = server_entry["path"]
    await _check_gate(server_entry, source_api, raw_headers)
    result = await server_service.register_server(server_entry)
    if not result.get("success"):
        raise OpenApiServerError(
            409, "duplicate_name", f"the registry refused path {path}: {result.get('message')}"
        )
    if not await server_service.toggle_service(path, True):
        raise OpenApiServerError(500, "enable_failed", f"registered {path} but could not enable it")
    from ..repositories.factory import get_search_repository

    await get_search_repository().index_server(path, server_entry, True)
    await _flush_nginx()
    # The same scan and webhook as POST /api/servers/register. The caller runs
    # the health check and tool refresh itself, after pushing the hub.
    from ..api.server_routes import start_post_registration_tasks

    start_post_registration_tasks(
        path, server_entry["proxy_pass_url"], server_entry, performed_by, run_health_check=False
    )
    logger.info(f"Managed server {path} registered and enabled by {performed_by}")


async def refresh_managed_server(
    path: str,
    rescan: bool,
) -> None:
    """Probe a managed server, refresh its tool list, and optionally rescan it."""
    from ..health.service import health_service

    await health_service.perform_immediate_health_check(path)
    await health_service.refresh_tools_now(path)
    if not rescan:
        return
    server_info = await server_service.get_server_info(path)
    if server_info:
        _start_security_scan(server_info)
