"""
Federation server reconciliation service.

Detects and removes servers from mcp_servers_default that are no longer
present in the federation configuration. Called after config saves,
manual syncs, and startup syncs.

IMPORTANT: This module should only be called from authenticated contexts
(route handlers with user_context or startup code). It does not perform
its own authorization checks.

If reconciliation fails after a config save, stale servers will be
cleaned up on next startup (reconciliation always runs at startup).
"""

import logging
import time
from typing import (
    Any,
    Dict,
    Optional,
    Set,
)

from ..schemas.federation_schema import FederationConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# OTel metric names
RECONCILIATION_REMOVED_METRIC: str = "mcp_federation_reconciliation_removed_total"
RECONCILIATION_DURATION_METRIC: str = "mcp_federation_reconciliation_duration_seconds"


def _config_server_names_to_paths(
    config: FederationConfig,
) -> Set[str]:
    """Convert federation config server names to expected DB paths.

    Anthropic server names like 'io.github.jgador/websharp'
    are stored with paths like '/io.github.jgador-websharp'.

    Args:
        config: The current federation configuration

    Returns:
        Set of expected server paths
    """
    expected_paths = set()
    for server in config.anthropic.servers:
        path = f"/{server.name.replace('/', '-')}"
        expected_paths.add(path)
    return expected_paths


def _record_reconciliation_metrics(
    removed_count: int,
    elapsed_seconds: float,
) -> None:
    """Record OTel metrics for reconciliation.

    Args:
        removed_count: Number of servers removed
        elapsed_seconds: Time taken for reconciliation
    """
    try:
        from ..otel.instruments import get_instruments

        instruments = get_instruments()
        if instruments:
            # Record removed count (counter)
            counter = instruments.get(RECONCILIATION_REMOVED_METRIC)
            if counter:
                counter.add(removed_count, {"source": "anthropic"})

            # Record duration (histogram)
            histogram = instruments.get(RECONCILIATION_DURATION_METRIC)
            if histogram:
                histogram.record(elapsed_seconds, {"source": "anthropic"})
    except Exception as e:
        logger.debug(f"Failed to record reconciliation metrics: {e}")


async def reconcile_anthropic_servers(
    config: FederationConfig,
    server_service: Any,
    server_repo: Any,
    nginx_service: Optional[Any] = None,
    dry_run: bool = False,
    skip_nginx_regen: bool = False,
    audit_username: Optional[str] = None,
) -> Dict[str, Any]:
    """Reconcile Anthropic federated servers against the current config.

    Removes servers from mcp_servers_default that have source="anthropic"
    but are no longer listed in the federation config.

    Args:
        config: Current federation configuration
        server_service: ServerService instance for remove_server()
        server_repo: Server repository for list_by_source()
        nginx_service: Optional NginxService for config regeneration
        dry_run: If True, compute delta but do not delete anything
        skip_nginx_regen: If True, skip nginx config regeneration
            (useful during startup when nginx is regenerated separately)
        audit_username: Username to include in audit log entry
            (None for startup/system-triggered reconciliation)

    Returns:
        Dictionary with reconciliation results:
        - removed: List of removed server names
        - removed_count: Number of servers removed
        - expected_count: Number of servers in config
        - actual_count: Number of servers found in DB
        - dry_run: Whether this was a dry run
        - errors: List of errors (if any)
    """
    start_time = time.time()

    # Step 1: Get expected paths from config
    expected_paths = _config_server_names_to_paths(config)

    # If anthropic is disabled entirely, all anthropic servers are stale
    if not config.anthropic.enabled:
        expected_paths = set()

    logger.info(
        f"Reconciliation: {len(expected_paths)} servers expected "
        f"from Anthropic federation config"
    )

    # Step 2: Get actual Anthropic servers in DB
    actual_servers = await server_repo.list_by_source("anthropic")
    actual_paths = set(actual_servers.keys())

    logger.info(
        f"Reconciliation: {len(actual_paths)} servers found "
        f"in mcp_servers_default with source='anthropic'"
    )

    # Step 3: Compute stale servers (in DB but not in config)
    stale_paths = actual_paths - expected_paths

    if not stale_paths:
        logger.debug("Reconciliation: no stale servers found")
        return {
            "removed": [],
            "removed_count": 0,
            "expected_count": len(expected_paths),
            "actual_count": len(actual_paths),
            "dry_run": dry_run,
        }

    stale_names = [
        actual_servers[p].get("server_name", p) for p in sorted(stale_paths)
    ]
    logger.info(
        f"Reconciliation: {len(stale_paths)} stale servers to remove: "
        f"{stale_names}"
    )

    # Dry run: return what would be removed without deleting
    if dry_run:
        logger.info("Reconciliation: dry_run=True, skipping actual removal")
        return {
            "removed": stale_names,
            "removed_count": len(stale_names),
            "expected_count": len(expected_paths),
            "actual_count": len(actual_paths),
            "dry_run": True,
        }

    # Step 4: Remove stale servers
    removed = []
    errors = []
    for path in sorted(stale_paths):
        try:
            server_name = actual_servers[path].get("server_name", path)
            success = await server_service.remove_server(path)
            if success:
                removed.append(server_name)
                logger.info(
                    f"Reconciliation: removed stale server "
                    f"'{server_name}' ({path})"
                )
            else:
                errors.append(f"Failed to remove {server_name} ({path})")
                logger.warning(
                    f"Reconciliation: failed to remove server "
                    f"'{server_name}' ({path})"
                )
        except Exception as e:
            errors.append(f"Error removing {path}: {e}")
            logger.error(f"Reconciliation: error removing server {path}: {e}")

    # Step 5: Regenerate nginx config if any servers were removed
    if removed and nginx_service and not skip_nginx_regen:
        try:
            all_servers = await server_repo.list_all()
            enabled_servers = {
                p: info
                for p, info in all_servers.items()
                if info.get("is_enabled", False)
            }
            await nginx_service.generate_config_async(enabled_servers)
            logger.info("Reconciliation: nginx config regenerated")
        except Exception as e:
            logger.error(
                f"Reconciliation: failed to regenerate nginx config: {e}"
            )

    elapsed = time.time() - start_time

    # Step 6: Record OTel metrics
    _record_reconciliation_metrics(len(removed), elapsed)

    # Step 7: Audit trail summary
    triggered_by = audit_username or "system"
    logger.info(
        f"Reconciliation complete: removed {len(removed)} stale servers "
        f"in {elapsed:.1f} seconds "
        f"(triggered_by={triggered_by}, "
        f"expected={len(expected_paths)}, "
        f"actual_in_db={len(actual_paths)}, "
        f"stale={len(stale_paths)}, "
        f"errors={len(errors)})"
    )

    return {
        "removed": removed,
        "removed_count": len(removed),
        "expected_count": len(expected_paths),
        "actual_count": len(actual_paths),
        "dry_run": False,
        "errors": errors,
    }
