"""Rescan stored metadata and reindex it for search (issue #1811).

Records written before metadata scanning existed have no verdict, so their
metadata is withheld from model-facing reads, and their search entries may
still hold text a scan would now flag. A record whose verdict was produced by
an older rule set, or whose scan failed, should be rescanned too.

``rescan_metadata`` walks the server, agent and skill collections and, for
each record:

1. recomputes ``metadata_scan`` when it is missing, malformed, stale, failed or
   from another rule set. The write is a compare-and-set that only lands if
   the metadata is still what was scanned, and it never touches
   ``updated_at``. On a miss the record is read again and retried;
2. reindexes it with ``skip_if_unchanged=True``. Indexing scans the metadata
   again, so flagged text never enters the embedding text or
   ``metadata_text``. After indexing, the record is read again: if it changed
   meanwhile, the newer version is indexed instead, and if it was deleted, its
   index entry is removed. So a snapshot taken by the rescan can never leave
   older content in the index than a concurrent writer's own reindex.

A record that keeps changing for ``MAX_ATTEMPTS`` rounds is skipped and
counted; the writer that is changing it reindexes it itself.

Idempotence: a second run writes no verdicts for records whose verdict is
current, and ``skip_if_unchanged`` makes no embedding call and no index write
for entries that are already current. Records whose scan fails are retried on
every run. Resumable: after an interruption, run it again. An unexpected error
(a database outage, say) stops the run with ``MetadataRescanError`` rather
than being counted per record. Exposed as
``POST /api/admin/metadata-security/rescan`` (admin only).
"""

import logging
import time
from typing import Any

from ..repositories.documentdb.client import get_collection_name, get_documentdb_client
from ..repositories.documentdb.skill_repository import document_to_skill_card
from ..repositories.factory import get_search_repository
from ..schemas.agent_models import AgentCard
from ..utils.metadata_security import EntityType, needs_rescan, scan_metadata

logger = logging.getLogger(__name__)

ALL_ENTITY_TYPES: tuple[EntityType, ...] = ("server", "agent", "skill")

# Rounds of re-read and retry for a record that changes while it is processed.
MAX_ATTEMPTS: int = 3

# Source collection for each entity type (names as passed to get_collection_name).
_COLLECTION_BY_TYPE: dict[EntityType, str] = {
    "server": "mcp_servers",
    "agent": "mcp_agents",
    "skill": "agent_skills",
}

# Errors that mean one record's data cannot be indexed (bad stored data); they
# are counted and the run continues. pydantic's ValidationError is a ValueError.
_RECORD_ERRORS: tuple[type[Exception], ...] = (ValueError, TypeError, KeyError)


class MetadataRescanError(RuntimeError):
    """The rescan stopped early. It is safe to run again; it resumes."""


def _empty_counts() -> dict[str, int]:
    return {
        "examined": 0,
        "rescanned": 0,
        "flagged": 0,
        "failed": 0,
        "changed_during_scan": 0,
        "deleted_during_scan": 0,
        "skipped_still_changing": 0,
        "reindexed": 0,
        "reindex_failed": 0,
    }


def _indexed_content(
    doc: dict[str, Any],
) -> dict[str, Any]:
    """The part of a record that index content is built from (all but the verdict)."""
    return {k: v for k, v in doc.items() if k != "metadata_scan"}


def _count_verdict(
    verdict: dict[str, Any],
    counts: dict[str, int],
) -> None:
    counts["rescanned"] += 1
    if verdict["status"] == "flagged":
        counts["flagged"] += 1
    elif verdict["status"] == "failed":
        counts["failed"] += 1


async def _refresh_verdict(
    collection: Any,
    doc: dict[str, Any],
    entity_type: EntityType,
    counts: dict[str, int],
) -> dict[str, Any] | None:
    """Make sure the record carries a current verdict.

    Returns:
        The record as now stored, or None when it was deleted or kept changing.
    """
    for _ in range(MAX_ATTEMPTS):
        if not needs_rescan(doc.get("metadata"), doc.get("metadata_scan"), entity_type):
            return doc
        verdict = scan_metadata(doc.get("metadata"), entity_type).model_dump(mode="json")
        result = await collection.update_one(
            {"_id": doc["_id"], "metadata": doc.get("metadata")},
            {"$set": {"metadata_scan": verdict}},
        )
        if result.matched_count:
            _count_verdict(verdict, counts)
            return {**doc, "metadata_scan": verdict}
        counts["changed_during_scan"] += 1
        doc = await collection.find_one({"_id": doc["_id"]})
        if doc is None:
            counts["deleted_during_scan"] += 1
            return None
    counts["skipped_still_changing"] += 1
    return None


async def _index_once(
    doc: dict[str, Any],
    entity_type: EntityType,
) -> bool:
    """Index one record snapshot; True only if it was stored with an embedding."""
    search_repo = get_search_repository()
    path = doc["_id"]
    is_enabled = doc.get("is_enabled", True)
    if entity_type == "server":
        return await search_repo.index_server(
            path, {**doc, "path": path}, is_enabled=is_enabled, skip_if_unchanged=True
        )
    if entity_type == "agent":
        card = AgentCard(**{**doc, "path": path})
        return await search_repo.index_agent(
            path, card, is_enabled=is_enabled, skip_if_unchanged=True
        )
    skill = document_to_skill_card(doc)
    return await search_repo.index_skill(path, skill, is_enabled=is_enabled, skip_if_unchanged=True)


async def _reindex_until_current(
    collection: Any,
    doc: dict[str, Any],
    entity_type: EntityType,
    counts: dict[str, int],
) -> None:
    """Index the record, then confirm nothing changed underneath; repeat if it did."""
    for _ in range(MAX_ATTEMPTS):
        if entity_type == "server" and doc.get("is_active") is False:
            return  # Inactive server versions are not indexed for search.
        stored = await _index_once(doc, entity_type)
        current = await collection.find_one({"_id": doc["_id"]})
        if current is None:
            # Deleted while we indexed: our write may have recreated its entry.
            await get_search_repository().remove_entity(doc["_id"])
            counts["deleted_during_scan"] += 1
            return
        if _indexed_content(current) == _indexed_content(doc):
            counts["reindexed" if stored else "reindex_failed"] += 1
            return
        counts["changed_during_scan"] += 1
        doc = current
    counts["skipped_still_changing"] += 1


async def _process_record(
    collection: Any,
    doc: dict[str, Any],
    entity_type: EntityType,
    dry_run: bool,
    counts: dict[str, int],
) -> None:
    """Rescan and reindex one record."""
    counts["examined"] += 1
    if dry_run:
        if needs_rescan(doc.get("metadata"), doc.get("metadata_scan"), entity_type):
            counts["rescanned"] += 1
        return
    current = await _refresh_verdict(collection, doc, entity_type, counts)
    if current is None:
        return
    try:
        await _reindex_until_current(collection, current, entity_type, counts)
    except _RECORD_ERRORS as e:
        logger.error(f"Reindex failed for {entity_type} {doc.get('_id')}: {type(e).__name__}")
        counts["reindex_failed"] += 1


async def _process_collection(
    collection: Any,
    entity_type: EntityType,
    dry_run: bool,
    results: dict[str, Any],
) -> None:
    counts = _empty_counts()
    results[entity_type] = counts
    async for doc in collection.find({}):
        await _process_record(collection, doc, entity_type, dry_run, counts)
    logger.info(f"Metadata rescan {entity_type}: {counts}")


async def rescan_metadata(
    entity_types: list[EntityType] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Rescan and reindex the metadata of every server, agent and skill.

    Args:
        entity_types: Which kinds of record to process; all three by default.
        dry_run: Count the records that need a new verdict without writing anything.

    Returns:
        Per-type counts: examined, rescanned (or, on a dry run, would be),
        flagged, failed, changed_during_scan, deleted_during_scan,
        skipped_still_changing, reindexed and reindex_failed, plus dry_run and
        elapsed_seconds.

    Raises:
        MetadataRescanError: If the run stopped on an unexpected error; the
            message carries the counts reached. Running again resumes.
    """
    types = list(entity_types or ALL_ENTITY_TYPES)
    logger.warning(f"Starting metadata rescan over all records: types={types} dry_run={dry_run}")
    started = time.perf_counter()
    db = await get_documentdb_client()
    results: dict[str, Any] = {}
    for entity_type in types:
        collection = db[get_collection_name(_COLLECTION_BY_TYPE[entity_type])]
        try:
            await _process_collection(collection, entity_type, dry_run, results)
        except Exception as e:
            logger.exception(f"Metadata rescan stopped during {entity_type}: {results}")
            raise MetadataRescanError(
                f"Metadata rescan stopped during {entity_type} records "
                f"({type(e).__name__}) after {results}. Records already processed "
                "are kept; run the rescan again to resume."
            ) from e
    results["dry_run"] = dry_run
    results["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    logger.info(f"Metadata rescan finished in {results['elapsed_seconds']}s")
    return results
