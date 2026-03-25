# registry/services/ans_service.py

import logging
import time
from datetime import (
    datetime,
    timezone,
)
from typing import Any

import httpx

from registry.core.config import settings
from registry.repositories.factory import (
    get_agent_repository,
    get_server_repository,
)
from registry.schemas.ans_models import (
    ANSIntegrationMetrics,
    ANSMetadata,
    ANSSyncStats,
)
from registry.services.ans_client import (
    ANS_STATUS_ERROR,
    ANS_STATUS_NOT_FOUND,
    verify_ans_agent,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

# In-memory sync history (last 20 entries)
_sync_history: list[dict] = []
MAX_SYNC_HISTORY: int = 20


def _store_sync_history(
    stats: ANSSyncStats,
) -> None:
    """Store sync result in history for admin visibility."""
    global _sync_history
    _sync_history.append({
        "completed_at": datetime.now(timezone.utc).isoformat(),
        **stats.model_dump(),
    })
    if len(_sync_history) > MAX_SYNC_HISTORY:
        _sync_history = _sync_history[-MAX_SYNC_HISTORY:]


async def _sync_asset_type(
    repo: Any,
    asset_type: str,
    stats: ANSSyncStats,
) -> None:
    """Sync ANS status for one asset type (agent or server).

    Args:
        repo: Repository instance (agent or server)
        asset_type: "agent" or "server" for logging
        stats: Mutable stats object to update
    """
    if hasattr(repo, "find_with_filter"):
        linked_assets = await repo.find_with_filter(
            {"ans_metadata": {"$exists": True, "$ne": None}}
        )
    else:
        all_assets = await repo.list_all()
        linked_assets = {}
        for asset in all_assets:
            asset_dict = asset.model_dump() if hasattr(asset, "model_dump") else asset
            asset_path = asset_dict.get("path", "")
            if asset_dict.get("ans_metadata"):
                linked_assets[asset_path] = asset_dict

    for path, asset_data in linked_assets.items():
        ans_meta = asset_data.get("ans_metadata", {})
        stats.total += 1
        ans_agent_id = ans_meta.get("ans_agent_id", "")

        try:
            result = await verify_ans_agent(ans_agent_id)
            now = datetime.now(timezone.utc)
            if result is None:
                await repo.update_field(path, "ans_metadata.status", ANS_STATUS_NOT_FOUND)
                await repo.update_field(path, "ans_metadata.last_verified", now.isoformat())
            else:
                metadata_dict = result.model_dump(mode="json")
                metadata_dict["linked_at"] = ans_meta.get("linked_at")
                await repo.update_field(path, "ans_metadata", metadata_dict)
            stats.updated += 1

        except Exception as e:
            stats.errors += 1
            logger.error(f"ANS sync error for {asset_type} {path}: {e}")


async def link_ans_to_agent(
    agent_path: str,
    ans_agent_id: str,
    username: str | None = None,
) -> dict:
    """Link an ANS Agent ID to an AI Registry agent.

    Args:
        agent_path: Agent path in the registry
        ans_agent_id: ANS Agent ID to link
        username: Authenticated user's username (for ownership check)

    Returns:
        Dict with success, message, and ans_metadata
    """
    repo = get_agent_repository()

    agent = await repo.get(agent_path)
    if not agent:
        return {"success": False, "message": f"Agent not found: {agent_path}"}

    registered_by = getattr(agent, "registered_by", None)
    if username and registered_by and username != registered_by:
        return {"success": False, "message": "Not authorized: you are not the owner of this agent"}

    try:
        ans_metadata = await verify_ans_agent(ans_agent_id)
    except httpx.TimeoutException:
        return {"success": False, "message": "ANS API timed out"}
    except httpx.HTTPStatusError as e:
        return {"success": False, "message": f"ANS API error: {e.response.status_code}"}
    except RuntimeError as e:
        return {"success": False, "message": str(e)}

    if ans_metadata is None:
        return {"success": False, "message": f"ANS Agent ID not found: {ans_agent_id}"}

    metadata_dict = ans_metadata.model_dump(mode="json")
    await repo.update_field(agent_path, "ans_metadata", metadata_dict)

    logger.info(f"ANS ID linked to agent {agent_path}: {ans_agent_id} (status: {ans_metadata.status})")
    return {
        "success": True,
        "message": f"ANS Agent ID linked and verified (status: {ans_metadata.status})",
        "ans_metadata": metadata_dict,
    }


async def link_ans_to_server(
    server_path: str,
    ans_agent_id: str,
    username: str | None = None,
) -> dict:
    """Link an ANS Agent ID to an MCP server.

    Args:
        server_path: Server path in the registry
        ans_agent_id: ANS Agent ID to link
        username: Authenticated user's username (for ownership check)

    Returns:
        Dict with success, message, and ans_metadata
    """
    repo = get_server_repository()

    server = await repo.get(server_path)
    if not server:
        return {"success": False, "message": f"Server not found: {server_path}"}

    registered_by = getattr(server, "registered_by", None)
    if username and registered_by and username != registered_by:
        return {"success": False, "message": "Not authorized: you are not the owner of this server"}

    try:
        ans_metadata = await verify_ans_agent(ans_agent_id)
    except httpx.TimeoutException:
        return {"success": False, "message": "ANS API timed out"}
    except httpx.HTTPStatusError as e:
        return {"success": False, "message": f"ANS API error: {e.response.status_code}"}
    except RuntimeError as e:
        return {"success": False, "message": str(e)}

    if ans_metadata is None:
        return {"success": False, "message": f"ANS Agent ID not found: {ans_agent_id}"}

    metadata_dict = ans_metadata.model_dump(mode="json")
    await repo.update_field(server_path, "ans_metadata", metadata_dict)

    logger.info(f"ANS ID linked to server {server_path}: {ans_agent_id} (status: {ans_metadata.status})")
    return {
        "success": True,
        "message": f"ANS Agent ID linked and verified (status: {ans_metadata.status})",
        "ans_metadata": metadata_dict,
    }


async def unlink_ans_from_agent(
    agent_path: str,
    username: str | None = None,
) -> dict:
    """Remove ANS link from an agent.

    Args:
        agent_path: Agent path in the registry
        username: Authenticated user's username (for ownership check)

    Returns:
        Dict with success and message
    """
    repo = get_agent_repository()
    agent = await repo.get(agent_path)
    if not agent:
        return {"success": False, "message": f"Agent not found: {agent_path}"}

    registered_by = getattr(agent, "registered_by", None)
    if username and registered_by and username != registered_by:
        return {"success": False, "message": "Not authorized: you are not the owner of this agent"}

    await repo.update_field(agent_path, "ans_metadata", None)
    logger.info(f"ANS link removed from agent: {agent_path}")
    return {"success": True, "message": "ANS link removed"}


async def unlink_ans_from_server(
    server_path: str,
    username: str | None = None,
) -> dict:
    """Remove ANS link from a server.

    Args:
        server_path: Server path in the registry
        username: Authenticated user's username (for ownership check)

    Returns:
        Dict with success and message
    """
    repo = get_server_repository()
    server = await repo.get(server_path)
    if not server:
        return {"success": False, "message": f"Server not found: {server_path}"}

    registered_by = getattr(server, "registered_by", None)
    if username and registered_by and username != registered_by:
        return {"success": False, "message": "Not authorized: you are not the owner of this server"}

    await repo.update_field(server_path, "ans_metadata", None)
    logger.info(f"ANS link removed from server: {server_path}")
    return {"success": True, "message": "ANS link removed"}


async def sync_all_ans_status() -> ANSSyncStats:
    """Sync ANS verification status for all linked assets.

    Returns:
        Sync statistics
    """
    start_time = time.time()
    stats = ANSSyncStats()

    agent_repo = get_agent_repository()
    server_repo = get_server_repository()

    await _sync_asset_type(agent_repo, "agent", stats)
    await _sync_asset_type(server_repo, "server", stats)

    elapsed = time.time() - start_time
    stats.duration_seconds = round(elapsed, 2)

    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    if minutes > 0:
        logger.info(f"ANS sync completed in {minutes} minutes and {seconds:.1f} seconds: {stats.model_dump()}")
    else:
        logger.info(f"ANS sync completed in {seconds:.1f} seconds: {stats.model_dump()}")

    _store_sync_history(stats)

    return stats


def get_sync_history() -> list[dict]:
    """Get recent sync history entries."""
    return list(_sync_history)


async def get_ans_metrics() -> ANSIntegrationMetrics:
    """Get ANS integration metrics for admin dashboard.

    Returns:
        ANS integration metrics
    """
    agent_repo = get_agent_repository()
    server_repo = get_server_repository()

    metrics = ANSIntegrationMetrics()

    agents = await agent_repo.list_all()
    for agent in agents:
        agent_dict = agent.model_dump() if hasattr(agent, "model_dump") else agent
        ans_meta = agent_dict.get("ans_metadata")
        if ans_meta:
            metrics.total_linked += 1
            status = ans_meta.get("status", "pending")
            metrics.by_status[status] = metrics.by_status.get(status, 0) + 1
            metrics.by_asset_type["agent"] = metrics.by_asset_type.get("agent", 0) + 1

    servers = await server_repo.list_all()
    for server in servers:
        server_dict = server.model_dump() if hasattr(server, "model_dump") else server
        ans_meta = server_dict.get("ans_metadata")
        if ans_meta:
            metrics.total_linked += 1
            status = ans_meta.get("status", "pending")
            metrics.by_status[status] = metrics.by_status.get(status, 0) + 1
            metrics.by_asset_type["server"] = metrics.by_asset_type.get("server", 0) + 1

    return metrics
