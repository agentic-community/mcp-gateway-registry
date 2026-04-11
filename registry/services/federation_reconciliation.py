"""
Federation server reconciliation service.

Detects and removes servers from mcp_servers_default that are no longer
present in the federation configuration. Called after config saves,
manual syncs, and startup syncs.

Supports reconciliation for:
- Anthropic MCP Registry (servers)
- AWS Agent Registry (servers, agents, skills)

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
) -> set[str]:
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
    nginx_service: Any | None = None,
    dry_run: bool = False,
    skip_nginx_regen: bool = False,
    audit_username: str | None = None,
) -> dict[str, Any]:
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
        f"Reconciliation: {len(expected_paths)} servers expected from Anthropic federation config"
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

    stale_names = [actual_servers[p].get("server_name", p) for p in sorted(stale_paths)]
    logger.info(f"Reconciliation: {len(stale_paths)} stale servers to remove: {stale_names}")

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
                logger.info(f"Reconciliation: removed stale server '{server_name}' ({path})")
            else:
                errors.append(f"Failed to remove {server_name} ({path})")
                logger.warning(f"Reconciliation: failed to remove server '{server_name}' ({path})")
        except Exception as e:
            errors.append(f"Error removing {path}: {e}")
            logger.error(f"Reconciliation: error removing server {path}: {e}")

    # Step 5: Regenerate nginx config if any servers were removed
    if removed and nginx_service and not skip_nginx_regen:
        try:
            all_servers = await server_repo.list_all()
            enabled_servers = {
                p: info for p, info in all_servers.items() if info.get("is_enabled", False)
            }
            await nginx_service.generate_config_async(enabled_servers)
            logger.info("Reconciliation: nginx config regenerated")
        except Exception as e:
            logger.error(f"Reconciliation: failed to regenerate nginx config: {e}")

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


def _build_expected_agentcore_paths(
    config: FederationConfig,
    synced_paths: dict[str, set[str]],
) -> dict[str, set[str]]:
    """Build sets of expected paths from synced data.

    If agentcore federation is disabled, returns empty sets (all records are stale).

    Args:
        config: Current federation configuration
        synced_paths: Dict with "servers", "agents", "skills" keys
            containing paths that were just synced

    Returns:
        Dict with "servers", "agents", "skills" keys containing expected path sets
    """
    if not config.aws_registry.enabled:
        return {"servers": set(), "agents": set(), "skills": set()}

    return {
        "servers": synced_paths.get("servers", set()),
        "agents": synced_paths.get("agents", set()),
        "skills": synced_paths.get("skills", set()),
    }


async def _reconcile_agentcore_servers(
    expected_paths: set[str],
    server_service: Any,
    server_repo: Any,
) -> dict[str, Any]:
    """Reconcile AgentCore federated servers.

    Args:
        expected_paths: Paths that should exist after sync
        server_service: ServerService instance
        server_repo: Server repository

    Returns:
        Dict with removed list and error list
    """
    removed: list[str] = []
    errors: list[str] = []

    actual_servers = await server_repo.list_by_source("agentcore")
    actual_paths = set(actual_servers.keys())
    stale_paths = actual_paths - expected_paths

    for path in sorted(stale_paths):
        try:
            server_name = actual_servers[path].get("server_name", path)
            success = await server_service.remove_server(path)
            if success:
                removed.append(server_name)
                logger.info(f"AgentCore reconciliation: removed stale server '{server_name}'")
            else:
                errors.append(f"Failed to remove server {server_name} ({path})")
        except Exception as e:
            errors.append(f"Error removing server {path}: {e}")
            logger.error(f"AgentCore reconciliation: error removing server {path}: {e}")

    return {"removed": removed, "errors": errors}


async def _reconcile_agentcore_agents(
    expected_paths: set[str],
    agent_repo: Any,
) -> dict[str, Any]:
    """Reconcile AgentCore federated agents.

    Finds agents with 'agentcore' tag and path starting with /agents/agentcore-,
    then removes those not in expected_paths.

    Args:
        expected_paths: Paths that should exist after sync
        agent_repo: Agent repository

    Returns:
        Dict with removed list and error list
    """
    removed: list[str] = []
    errors: list[str] = []

    all_agents = await agent_repo.list_all()
    agentcore_agents = [
        a for a in all_agents
        if "agentcore" in (a.tags or []) and str(a.path).startswith("/agents/agentcore-")
    ]

    for agent in agentcore_agents:
        if agent.path not in expected_paths:
            try:
                success = await agent_repo.delete(agent.path)
                if success:
                    removed.append(agent.name)
                    logger.info(
                        f"AgentCore reconciliation: removed stale agent '{agent.name}'"
                    )
                else:
                    errors.append(f"Failed to remove agent {agent.name} ({agent.path})")
            except Exception as e:
                errors.append(f"Error removing agent {agent.path}: {e}")
                logger.error(
                    f"AgentCore reconciliation: error removing agent {agent.path}: {e}"
                )

    return {"removed": removed, "errors": errors}


async def _reconcile_agentcore_skills(
    expected_paths: set[str],
    skill_repo: Any,
) -> dict[str, Any]:
    """Reconcile AgentCore federated skills.

    Finds skills with 'agentcore' tag and path starting with /skills/agentcore-,
    then removes those not in expected_paths.

    Args:
        expected_paths: Paths that should exist after sync
        skill_repo: Skill repository

    Returns:
        Dict with removed list and error list
    """
    removed: list[str] = []
    errors: list[str] = []

    all_skills = await skill_repo.list_all()
    agentcore_skills = [
        s for s in all_skills
        if "agentcore" in (s.tags or []) and str(s.path).startswith("/skills/agentcore-")
    ]

    for skill in agentcore_skills:
        if skill.path not in expected_paths:
            try:
                success = await skill_repo.delete(skill.path)
                if success:
                    removed.append(skill.name)
                    logger.info(
                        f"AgentCore reconciliation: removed stale skill '{skill.name}'"
                    )
                else:
                    errors.append(f"Failed to remove skill {skill.name} ({skill.path})")
            except Exception as e:
                errors.append(f"Error removing skill {skill.path}: {e}")
                logger.error(
                    f"AgentCore reconciliation: error removing skill {skill.path}: {e}"
                )

    return {"removed": removed, "errors": errors}


async def reconcile_agentcore_records(
    config: FederationConfig,
    server_service: Any,
    server_repo: Any,
    agent_repo: Any,
    skill_repo: Any,
    synced_paths: dict[str, set[str]] | None = None,
    dry_run: bool = False,
    audit_username: str | None = None,
) -> dict[str, Any]:
    """Reconcile AgentCore federated records against the current config.

    Removes servers, agents, and skills that have source/tag "agentcore"
    but were not part of the latest sync.

    Args:
        config: Current federation configuration
        server_service: ServerService instance for remove_server()
        server_repo: Server repository for list_by_source()
        agent_repo: Agent repository for list_all() and delete()
        skill_repo: Skill repository for list_all() and delete()
        synced_paths: Dict with "servers", "agents", "skills" keys containing
            paths that were just synced. If None, uses empty sets (removes all).
        dry_run: If True, compute delta but do not delete anything
        audit_username: Username for audit trail

    Returns:
        Dictionary with reconciliation results per item type
    """
    start_time = time.time()

    if synced_paths is None:
        synced_paths = {"servers": set(), "agents": set(), "skills": set()}

    expected = _build_expected_agentcore_paths(config, synced_paths)

    logger.info(
        f"AgentCore reconciliation: expecting "
        f"{len(expected['servers'])} servers, "
        f"{len(expected['agents'])} agents, "
        f"{len(expected['skills'])} skills"
    )

    if dry_run:
        logger.info("AgentCore reconciliation: dry_run=True, skipping actual removal")
        return {
            "dry_run": True,
            "servers": {"removed": [], "errors": []},
            "agents": {"removed": [], "errors": []},
            "skills": {"removed": [], "errors": []},
        }

    # Reconcile each type
    server_result = await _reconcile_agentcore_servers(
        expected["servers"], server_service, server_repo
    )
    agent_result = await _reconcile_agentcore_agents(
        expected["agents"], agent_repo
    )
    skill_result = await _reconcile_agentcore_skills(
        expected["skills"], skill_repo
    )

    elapsed = time.time() - start_time
    total_removed = (
        len(server_result["removed"])
        + len(agent_result["removed"])
        + len(skill_result["removed"])
    )

    # Record metrics
    try:
        from ..otel.instruments import get_instruments

        instruments = get_instruments()
        if instruments:
            counter = instruments.get(RECONCILIATION_REMOVED_METRIC)
            if counter:
                counter.add(len(server_result["removed"]), {"source": "agentcore", "item_type": "server"})
                counter.add(len(agent_result["removed"]), {"source": "agentcore", "item_type": "agent"})
                counter.add(len(skill_result["removed"]), {"source": "agentcore", "item_type": "skill"})

            histogram = instruments.get(RECONCILIATION_DURATION_METRIC)
            if histogram:
                histogram.record(elapsed, {"source": "agentcore"})
    except Exception as e:
        logger.debug(f"Failed to record AgentCore reconciliation metrics: {e}")

    triggered_by = audit_username or "system"
    logger.info(
        f"AgentCore reconciliation complete: removed {total_removed} stale records "
        f"in {elapsed:.1f} seconds (triggered_by={triggered_by})"
    )

    return {
        "dry_run": False,
        "servers": server_result,
        "agents": agent_result,
        "skills": skill_result,
        "total_removed": total_removed,
        "elapsed_seconds": round(elapsed, 2),
    }
