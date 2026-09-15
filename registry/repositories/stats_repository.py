"""
Stats repository for tracking usage counters (semantic search, etc.).

Stores counters at three granularities:
- hourly: resets every hour
- daily: resets every 24 hours
- forever: never resets

Primary storage is the mcp_stats_{namespace} MongoDB collection.
Falls back to a local JSON file ({data_dir}/.stats.json) on error.
"""

import fcntl
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..core.config import settings

logger = logging.getLogger(__name__)


async def increment_search_counter() -> None:
    """Increment semantic search counter across all three time windows.

    Fail-silent: never impacts search operation.
    """
    try:
        await _increment_mongodb()
    except Exception as e:
        logger.debug(f"[stats] Failed to increment search counter: {e}")


async def get_search_count() -> int:
    """Get lifetime (forever) semantic search count.

    Returns:
        Cumulative search count, or 0 on failure.
    """
    try:
        return await _get_count_mongodb()
    except Exception as e:
        logger.debug(f"[stats] Failed to get search count: {e}")
        return 0


async def get_search_counts() -> dict[str, int]:
    """Get search counts for all three time windows.

    Returns:
        Dict with keys: total, last_24h, last_1h (all default to 0 on failure).
    """
    try:
        return await _get_counts_mongodb()
    except Exception as e:
        logger.debug(f"[stats] Failed to get search counts: {e}")
        return {"total": 0, "last_24h": 0, "last_1h": 0}


async def increment_auth_path_counters(counts: dict[str, int]) -> None:
    """Add per-auth-path request counts to the daily window.

    The auth-server flushes its in-process tally here periodically; the registry
    reads it back for the telemetry heartbeat.

    Raises on write failure, unlike increment_search_counter above. That one has
    no observer, so swallowing the error costs nothing. This one is the write
    behind mcpgw_registry_auth_path_flush_total: a caller that cannot tell a
    successful flush from a failed one cannot report flush health, and a
    silently failing write looks exactly like no traffic. The fail-silent
    boundary lives in exactly one place instead -- flush_once() in
    auth_server/auth_path_stats.py -- which records the outcome, logs at
    WARNING, and never re-raises into request handling.

    Args:
        counts: Auth path -> requests observed since the last flush.
    """
    if not counts:
        return
    await _increment_auth_path_mongodb(counts)


async def get_auth_path_counts() -> dict[str, Any]:
    """Get per-auth-path request counts for the current daily window.

    Returns:
        Dict with keys: counts (auth path -> request count) and window_hours
        (hours elapsed since the daily window was last reset). Empty counts and
        0.0 hours when nothing has been recorded, and on failure.
    """
    try:
        return await _get_auth_path_counts_mongodb()
    except Exception as e:
        logger.debug(f"[stats] Failed to get auth path counts: {e}")
        return {"counts": {}, "window_hours": 0.0}


async def _ensure_counters_document(collection, now: datetime) -> None:
    """Create the singleton counters document if it does not exist yet."""
    await collection.update_one(
        {"_id": "counters"},
        {
            "$setOnInsert": {
                "hourly": {"semantic_search_ctr": 0},
                "daily": {"semantic_search_ctr": 0, "auth_path": {}},
                "forever": {"semantic_search_ctr": 0},
                "hourly_reset_at": now,
                "daily_reset_at": now,
            }
        },
        upsert=True,
    )


async def _reset_stale_windows(collection, now: datetime) -> None:
    """Zero each counter window whose reset timestamp has gone stale.

    One conditional update_one per window, and the ``<window>_reset_at``
    predicate in the filter *is* the concurrency control -- that is the whole
    point of this shape. MongoDB matches and applies the update as a single
    atomic document-level operation, so exactly one writer wins each window
    boundary and no ``$set`` of zeros can land between another writer's read
    and its ``$inc``. The read-modify-write this replaced lost any increment
    that arrived while the roll was in flight.

    A missing or null reset timestamp does not match a date ``$lt``, so a
    half-written document is left alone rather than repeatedly zeroed.
    """
    await collection.update_one(
        {"_id": "counters", "hourly_reset_at": {"$lt": now - timedelta(hours=1)}},
        {"$set": {"hourly.semantic_search_ctr": 0, "hourly_reset_at": now}},
    )
    await collection.update_one(
        {"_id": "counters", "daily_reset_at": {"$lt": now - timedelta(hours=24)}},
        {
            "$set": {
                "daily.semantic_search_ctr": 0,
                "daily.auth_path": {},
                "daily_reset_at": now,
            }
        },
    )


async def _increment_mongodb() -> None:
    """Atomic increment in MongoDB with inline staleness reset."""
    from .documentdb.client import get_collection_name, get_documentdb_client

    db = await get_documentdb_client()
    collection_name = get_collection_name("mcp_stats")
    collection = db[collection_name]

    now = datetime.now(UTC)
    await _ensure_counters_document(collection, now)
    await _reset_stale_windows(collection, now)

    # Atomic increment on all three windows
    await collection.update_one(
        {"_id": "counters"},
        {
            "$inc": {
                "hourly.semantic_search_ctr": 1,
                "daily.semantic_search_ctr": 1,
                "forever.semantic_search_ctr": 1,
            }
        },
    )


async def _get_count_mongodb() -> int:
    """Read forever.semantic_search_ctr from MongoDB."""
    from .documentdb.client import get_collection_name, get_documentdb_client

    db = await get_documentdb_client()
    collection_name = get_collection_name("mcp_stats")
    collection = db[collection_name]

    doc = await collection.find_one({"_id": "counters"})
    if doc:
        return doc.get("forever", {}).get("semantic_search_ctr", 0)
    return 0


async def _get_counts_mongodb() -> dict[str, int]:
    """Read all three time-window counters from MongoDB."""
    from .documentdb.client import get_collection_name, get_documentdb_client

    db = await get_documentdb_client()
    collection_name = get_collection_name("mcp_stats")
    collection = db[collection_name]

    doc = await collection.find_one({"_id": "counters"})
    if doc:
        return {
            "total": doc.get("forever", {}).get("semantic_search_ctr", 0),
            "last_24h": doc.get("daily", {}).get("semantic_search_ctr", 0),
            "last_1h": doc.get("hourly", {}).get("semantic_search_ctr", 0),
        }
    return {"total": 0, "last_24h": 0, "last_1h": 0}


async def _increment_auth_path_mongodb(counts: dict[str, int]) -> None:
    """Batch $inc of the daily.auth_path.<path> counters in MongoDB."""
    from .documentdb.client import get_collection_name, get_documentdb_client

    db = await get_documentdb_client()
    collection_name = get_collection_name("mcp_stats")
    collection = db[collection_name]

    # Paths arrive allowlisted, but they originate as a metric label: a "." or
    # "$" would address a field path of the sender's choosing, not a counter.
    increments = {
        f"daily.auth_path.{path}": n
        for path, n in counts.items()
        if "." not in path and "$" not in path
    }
    if not increments:
        return

    now = datetime.now(UTC)
    await _ensure_counters_document(collection, now)
    # Roll before adding: incrementing first would have the reset immediately
    # zero the counts just written.
    await _reset_stale_windows(collection, now)
    await collection.update_one({"_id": "counters"}, {"$inc": increments})


async def _get_auth_path_counts_mongodb() -> dict[str, Any]:
    """Read daily.auth_path and the age of the daily window from MongoDB."""
    from .documentdb.client import get_collection_name, get_documentdb_client

    db = await get_documentdb_client()
    collection_name = get_collection_name("mcp_stats")
    collection = db[collection_name]

    doc = await collection.find_one({"_id": "counters"})
    if not doc:
        return {"counts": {}, "window_hours": 0.0}

    counts = doc.get("daily", {}).get("auth_path") or {}
    daily_reset = doc.get("daily_reset_at")
    if not counts or not daily_reset:
        return {"counts": {}, "window_hours": 0.0}

    # MongoDB returns naive datetimes; make them UTC-aware for comparison
    if daily_reset.tzinfo is None:
        daily_reset = daily_reset.replace(tzinfo=UTC)
    return {
        "counts": counts,
        "window_hours": (datetime.now(UTC) - daily_reset).total_seconds() / 3600,
    }


# Auth-path counts are DocumentDB/MongoDB only, with no file-backed mirror below:
# the registry and auth-server share no data volume, so a .stats.json written by
# one is never read by the other.
def _get_stats_file() -> Path:
    """Get path to file-based stats storage."""
    return settings.data_dir / ".stats.json"


def _read_file_stats() -> dict:
    """Read stats from file."""
    stats_file = _get_stats_file()
    if stats_file.exists():
        return json.loads(stats_file.read_text())
    return {
        "hourly": {"semantic_search_ctr": 0},
        "daily": {"semantic_search_ctr": 0},
        "forever": {"semantic_search_ctr": 0},
        "hourly_reset_at": datetime.now(UTC).isoformat(),
        "daily_reset_at": datetime.now(UTC).isoformat(),
    }


def _write_file_stats(stats: dict) -> None:
    """Write stats to file."""
    stats_file = _get_stats_file()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    stats_file.write_text(json.dumps(stats, default=str))


def _increment_file() -> None:
    """Increment counter in file-based storage with staleness reset.

    Uses file locking (fcntl.flock) to prevent lost updates from
    concurrent processes.
    """
    stats_file = _get_stats_file()
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    # Open file for read+write, create if missing
    with open(stats_file, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            content = f.read()
            stats = json.loads(content) if content.strip() else _read_file_stats()

            now = datetime.now(UTC)

            # Check hourly staleness
            hourly_reset = stats.get("hourly_reset_at", "")
            if hourly_reset:
                try:
                    reset_time = datetime.fromisoformat(hourly_reset.replace("Z", "+00:00"))
                    if (now - reset_time) > timedelta(hours=1):
                        stats["hourly"] = {"semantic_search_ctr": 0}
                        stats["hourly_reset_at"] = now.isoformat()
                except (ValueError, TypeError):
                    stats["hourly_reset_at"] = now.isoformat()

            # Check daily staleness
            daily_reset = stats.get("daily_reset_at", "")
            if daily_reset:
                try:
                    reset_time = datetime.fromisoformat(daily_reset.replace("Z", "+00:00"))
                    if (now - reset_time) > timedelta(hours=24):
                        stats["daily"] = {"semantic_search_ctr": 0}
                        stats["daily_reset_at"] = now.isoformat()
                except (ValueError, TypeError):
                    stats["daily_reset_at"] = now.isoformat()

            # Increment all three
            for window in ("hourly", "daily", "forever"):
                if window not in stats:
                    stats[window] = {"semantic_search_ctr": 0}
                stats[window]["semantic_search_ctr"] = (
                    stats[window].get("semantic_search_ctr", 0) + 1
                )

            # Write back while holding lock
            f.seek(0)
            f.truncate()
            f.write(json.dumps(stats, default=str))
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _get_count_file() -> int:
    """Read forever.semantic_search_ctr from file."""
    stats = _read_file_stats()
    return stats.get("forever", {}).get("semantic_search_ctr", 0)


def _get_counts_file() -> dict[str, int]:
    """Read all three time-window counters from file."""
    stats = _read_file_stats()
    return {
        "total": stats.get("forever", {}).get("semantic_search_ctr", 0),
        "last_24h": stats.get("daily", {}).get("semantic_search_ctr", 0),
        "last_1h": stats.get("hourly", {}).get("semantic_search_ctr", 0),
    }
