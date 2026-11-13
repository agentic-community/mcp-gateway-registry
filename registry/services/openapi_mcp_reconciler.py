"""Keep the openapi-mcp container in step with the registry (issue #2022).

The container keeps every hub in memory only. This loop compares each stored
server's generation with the generation the container reports, pushes any
server that is stale or missing, and deletes any hub the registry does not
know. It runs a full pass at startup and every
OPENAPI_MCP_RECONCILE_INTERVAL_SECONDS, and polls the container's /health
every HEALTH_POLL_SECONDS so a container restart is repaired within seconds.

Every registry replica runs the loop. Pushes carry the generation and the
container ignores anything not newer, so two replicas pushing at once is safe.
"""

import asyncio
import logging
import time
from dataclasses import dataclass

from ..core.config import settings
from ..repositories.factory import get_openapi_server_repository
from . import (
    openapi_mcp_client,
    server_lifecycle,
)
from .openapi_errors import OpenApiServerError
from .openapi_server_service import push_server

logger = logging.getLogger(__name__)


HEALTH_POLL_SECONDS: float = 10.0


@dataclass
class ReconcileResult:
    """What one pass did."""

    reachable: bool
    pushed: int = 0
    deleted: int = 0
    failed: int = 0


_task: asyncio.Task | None = None
_container_reachable: bool = False
_stored_servers: int = 0
_stored_targets: int = 0


def container_reachable() -> bool:
    """Whether the last status or health call to the container worked."""
    return _container_reachable


def stored_server_count() -> int:
    """Stored servers as of the last reconcile pass (read by the metrics gauge)."""
    return _stored_servers


def stored_target_count() -> int:
    """Stored targets as of the last reconcile pass (read by the metrics gauge)."""
    return _stored_targets


async def _push_stale(
    stale: list,
    result: ReconcileResult,
) -> None:
    for server in stale:
        try:
            await push_server(server, trigger="reconcile")
            result.pushed += 1
            await server_lifecycle.refresh_managed_server(server.path, rescan=False)
        except OpenApiServerError as e:
            result.failed += 1
            logger.error(
                f"Reconcile push of '{server.slug}' generation {server.generation} failed: {e.code}"
            )
        except Exception:
            result.failed += 1
            logger.exception(f"Reconcile push of '{server.slug}' failed")


async def _delete_extra(
    extra: list[str],
    result: ReconcileResult,
) -> None:
    for slug in extra:
        try:
            await openapi_mcp_client.delete_server(slug)
            result.deleted += 1
        except OpenApiServerError as e:
            result.failed += 1
            logger.warning(f"Reconcile could not delete stray hub '{slug}': {e.code}")


async def reconcile_once() -> ReconcileResult:
    """One full comparison of stored and running state."""
    global _container_reachable
    try:
        running = (await openapi_mcp_client.status()).get("servers", {})
    except OpenApiServerError:
        _container_reachable = False
        logger.warning("openapi-mcp container unreachable; reconcile will retry")
        return ReconcileResult(reachable=False)
    _container_reachable = True
    stored = {s.slug: s for s in await get_openapi_server_repository().list_all()}
    _remember_counts(list(stored.values()))
    stale = [
        s
        for slug, s in stored.items()
        if (running.get(slug) or {}).get("generation") != s.generation
    ]
    extra = [slug for slug in running if slug not in stored]
    result = ReconcileResult(reachable=True)
    await _push_stale(stale, result)
    await _delete_extra(extra, result)
    if stale or extra:
        logger.info(
            f"Reconcile pass: pushed={result.pushed} deleted={result.deleted} failed={result.failed}"
        )
    _record_drift(stale, extra, running)
    return result


def _remember_counts(
    servers: list,
) -> None:
    global _stored_servers, _stored_targets
    _stored_servers = len(servers)
    _stored_targets = sum(len(s.targets) for s in servers)


def _record_drift(
    stale: list,
    extra: list[str],
    running: dict,
) -> None:
    try:
        from ..observability.meters import openapi_mcp_reconcile_drift_total
    except Exception:  # noqa: BLE001 - metrics are optional
        return
    missing = sum(1 for s in stale if s.slug not in running)
    if missing:
        openapi_mcp_reconcile_drift_total.add(missing, {"kind": "missing"})
    if len(stale) - missing:
        openapi_mcp_reconcile_drift_total.add(len(stale) - missing, {"kind": "stale"})
    if extra:
        openapi_mcp_reconcile_drift_total.add(len(extra), {"kind": "extra"})


async def _health_changed(
    last_started_at: float | None,
) -> tuple[bool, float | None]:
    """Poll /health. True when the container restarted or came back after failing."""
    global _container_reachable
    body = await openapi_mcp_client.health()
    if body is None:
        _container_reachable = False
        return False, last_started_at
    started_at = body.get("started_at")
    changed = (not _container_reachable) or (started_at != last_started_at)
    _container_reachable = True
    return changed, started_at


async def _loop() -> None:
    interval = settings.openapi_mcp_reconcile_interval_seconds
    last_full = 0.0
    last_started_at: float | None = None
    while True:
        try:
            changed, last_started_at = await _health_changed(last_started_at)
            if changed or time.monotonic() - last_full >= interval:
                await reconcile_once()
                last_full = time.monotonic()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("OpenAPI MCP reconcile loop iteration failed")
        await asyncio.sleep(HEALTH_POLL_SECONDS)


async def start() -> None:
    """Start the loop. A container outage never blocks registry startup."""
    global _task
    if _task is None or _task.done():
        _task = asyncio.create_task(_loop())
        logger.info("OpenAPI MCP reconciler started")


async def stop() -> None:
    global _task
    if _task is None:
        return
    _task.cancel()
    try:
        await _task
    except asyncio.CancelledError:
        pass
    _task = None
    logger.info("OpenAPI MCP reconciler stopped")


async def warn_if_disabled_with_records() -> None:
    """With the flag off, name any stored servers that now have no backend. Deletes nothing."""
    try:
        servers = await get_openapi_server_repository().list_all()
    except Exception:
        logger.debug("Could not read openapi_servers while the feature is off", exc_info=True)
        return
    if servers:
        slugs = ", ".join(sorted(s.slug for s in servers))
        logger.warning(
            f"OPENAPI_MCP_ENABLED is false; these OpenAPI MCP servers have no backend: {slugs}. "
            "Turn the flag back on, or delete them with the flag on."
        )
