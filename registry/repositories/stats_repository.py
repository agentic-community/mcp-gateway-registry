"""
Stats repository for tracking usage counters (semantic search, etc.).

Stores counters at three granularities:
- hourly: resets every hour
- daily: resets every 24 hours
- forever: never resets

MongoDB storage uses the mcp_stats_{namespace} collection.
File-based storage uses {data_dir}/.stats.json.
"""

import fcntl
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..core.config import settings

logger = logging.getLogger(__name__)


async def increment_search_counter() -> None:
    """Increment semantic search counter across all three time windows.

    Fail-silent: never impacts search operation.
    """
    try:
        if settings.storage_backend in ("mongodb-ce", "documentdb"):
            await _increment_mongodb()
        else:
            _increment_file()
    except Exception as e:
        logger.debug(f"[stats] Failed to increment search counter: {e}")


async def get_search_count() -> int:
    """Get lifetime (forever) semantic search count.

    Returns:
        Cumulative search count, or 0 on failure.
    """
    try:
        if settings.storage_backend in ("mongodb-ce", "documentdb"):
            return await _get_count_mongodb()
        else:
            return _get_count_file()
    except Exception as e:
        logger.debug(f"[stats] Failed to get search count: {e}")
        return 0


async def get_search_counts() -> dict[str, int]:
    """Get search counts for all three time windows.

    Returns:
        Dict with keys: total, last_24h, last_1h (all default to 0 on failure).
    """
    try:
        if settings.storage_backend in ("mongodb-ce", "documentdb"):
            return await _get_counts_mongodb()
        else:
            return _get_counts_file()
    except Exception as e:
        logger.debug(f"[stats] Failed to get search counts: {e}")
        return {"total": 0, "last_24h": 0, "last_1h": 0}


async def _increment_mongodb() -> None:
    """Atomic increment in MongoDB with inline staleness reset."""
    from .documentdb.client import get_collection_name, get_documentdb_client

    db = await get_documentdb_client()
    collection_name = get_collection_name("mcp_stats")
    collection = db[collection_name]

    now = datetime.now(UTC)

    # Ensure document exists
    await collection.update_one(
        {"_id": "counters"},
        {
            "$setOnInsert": {
                "hourly": {"semantic_search_ctr": 0},
                "daily": {"semantic_search_ctr": 0},
                "forever": {"semantic_search_ctr": 0},
                "hourly_reset_at": now,
                "daily_reset_at": now,
            }
        },
        upsert=True,
    )

    # Check staleness and reset if needed
    doc = await collection.find_one({"_id": "counters"})
    if doc:
        updates = {}
        hourly_reset = doc.get("hourly_reset_at")
        daily_reset = doc.get("daily_reset_at")

        # MongoDB returns naive datetimes; make them UTC-aware for comparison
        if hourly_reset and hourly_reset.tzinfo is None:
            hourly_reset = hourly_reset.replace(tzinfo=UTC)
        if daily_reset and daily_reset.tzinfo is None:
            daily_reset = daily_reset.replace(tzinfo=UTC)

        if hourly_reset and (now - hourly_reset) > timedelta(hours=1):
            updates["hourly.semantic_search_ctr"] = 0
            updates["hourly_reset_at"] = now

        if daily_reset and (now - daily_reset) > timedelta(hours=24):
            updates["daily.semantic_search_ctr"] = 0
            updates["daily_reset_at"] = now

        if updates:
            await collection.update_one({"_id": "counters"}, {"$set": updates})

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
