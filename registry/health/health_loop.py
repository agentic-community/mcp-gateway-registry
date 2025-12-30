from __future__ import annotations

import asyncio
import logging
from datetime import (
    datetime,
    timezone,
)
from typing import (
    Dict,
    Protocol,
)

import httpx

from ..core.config import (
    settings,
)
from registry.constants import (
    HealthStatus,
)

logger = logging.getLogger(__name__)


class _HealthServiceProto(Protocol):
    server_health_status: Dict[str, str]
    server_last_check_time: Dict[str, datetime]

    async def _check_server_endpoint_transport_aware(
        self,
        client: httpx.AsyncClient,
        proxy_pass_url: str,
        server_info: Dict,
    ) -> tuple[bool, str]: ...

    async def _update_tools_background(
        self,
        service_path: str,
        proxy_pass_url: str,
    ): ...

    async def broadcast_health_update(
        self,
        service_path: str | None = None,
    ): ...


async def run_health_checks(
    service: _HealthServiceProto,
) -> None:
    """Background task to run periodic health checks."""
    logger.info("Starting periodic health checks...")

    while True:
        try:
            await perform_health_checks(service)
            await asyncio.sleep(settings.health_check_interval_seconds)
        except asyncio.CancelledError:
            logger.info("Health check task cancelled")
            break
        except Exception as exc:
            logger.error("Error in health check loop: %s", exc, exc_info=True)
            await asyncio.sleep(60)


async def perform_health_checks(
    service: _HealthServiceProto,
) -> None:
    """Perform health checks on all enabled services."""
    from ..core.nginx_service import nginx_service
    from ..services.server_service import server_service

    enabled_services = server_service.get_enabled_services()
    if not enabled_services:
        return

    if len(enabled_services) > 1:
        logger.debug("Performing health checks on %s enabled services", len(enabled_services))

    status_changed = False

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(settings.health_check_timeout_seconds),
    ) as client:
        check_tasks = []
        for service_path in enabled_services:
            server_info = server_service.get_server_info(service_path)
            if server_info and server_info.get("proxy_pass_url"):
                check_tasks.append(check_single_service(service, client, service_path, server_info))

        if check_tasks:
            results = await asyncio.gather(*check_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, bool) and result:
                    status_changed = True
                    break

    if not status_changed:
        return

    await service.broadcast_health_update()

    try:
        enabled_servers = {
            path: server_service.get_server_info(path)
            for path in server_service.get_enabled_services()
        }
        await nginx_service.generate_config_async(enabled_servers)
        logger.info("Nginx configuration regenerated due to health status changes")
    except Exception as exc:
        logger.error("Failed to regenerate nginx configuration after health status change: %s", exc)


async def check_single_service(
    service: _HealthServiceProto,
    client: httpx.AsyncClient,
    service_path: str,
    server_info: Dict,
    previous_status_override: str | None = None,
) -> bool:
    """Check a single service and return True if status changed."""
    proxy_pass_url = server_info.get("proxy_pass_url")
    previous_status = previous_status_override or service.server_health_status.get(
        service_path,
        HealthStatus.UNKNOWN,
    )
    new_status = previous_status

    try:
        is_healthy, status_detail = await service._check_server_endpoint_transport_aware(
            client,
            proxy_pass_url,
            server_info,
        )

        if is_healthy:
            new_status = status_detail

            should_fetch_tools = False
            if status_detail == HealthStatus.HEALTHY:
                if previous_status == HealthStatus.UNKNOWN:
                    should_fetch_tools = True
                    logger.info("First health check for %s - will fetch tools", service_path)
                elif previous_status != HealthStatus.HEALTHY:
                    should_fetch_tools = True
                    logger.info("Service %s transitioned to healthy - will fetch tools", service_path)
                else:
                    current_tool_list = server_info.get("tool_list", [])
                    if not current_tool_list:
                        should_fetch_tools = True
                        logger.info(
                            "Service %s is healthy but has no tools - will fetch tools",
                            service_path,
                        )

            if should_fetch_tools:
                asyncio.create_task(service._update_tools_background(service_path, proxy_pass_url))
        else:
            new_status = status_detail

    except httpx.TimeoutException:
        new_status = HealthStatus.UNHEALTHY_TIMEOUT
    except httpx.ConnectError:
        new_status = HealthStatus.UNHEALTHY_CONNECTION_ERROR
    except Exception as exc:
        new_status = f"error: {type(exc).__name__}"

    service.server_health_status[service_path] = new_status
    service.server_last_check_time[service_path] = datetime.now(timezone.utc)

    return previous_status != new_status
