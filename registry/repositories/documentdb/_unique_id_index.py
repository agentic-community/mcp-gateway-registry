"""Shared helpers for the unique ``id`` index on asset collections.

Each server / agent / skill document carries a user-facing ``id`` (an
arbitrary non-empty string: UUID, ARN, peer-registry id, ...). To honor
a caller-supplied ``id`` while keeping it unique per asset type (#1276),
each collection gets a unique *partial* index on ``id`` plus an indexed
``find_by_id`` lookup.

Mirrors ``_identity_url_sidecar.py``. **Ordering matters:** run
``backfill_missing_id`` *before* ``ensure_unique_id_index`` so the unique
index never fails to build on legacy rows. (Reverse of the sidecar's
order, safe there only because that index is sparse, not unique.)
"""

import logging
from typing import Any
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorCollection

logger = logging.getLogger(__name__)

ID_INDEX_NAME = "id_idx"


async def ensure_unique_id_index(
    collection: AsyncIOMotorCollection,
    collection_name: str,
) -> None:
    """Create a unique partial index on ``id`` (tolerant of engines that reject it)."""
    try:
        await collection.create_index(
            "id",
            name=ID_INDEX_NAME,
            unique=True,
            partialFilterExpression={"id": {"$exists": True}},
        )
    except Exception as exc:
        # The unique index is the only true race guard; the service-layer
        # pre-check is racy. Surface a failed build loudly (ERROR + a dedicated
        # counter) so ops can alert on "running without the DB uniqueness
        # guarantee" instead of it being a silent warning.
        from ...core.metrics import ASSET_ID_INDEX_BUILD_FAILED_TOTAL

        logger.error(
            "Could not create unique id index on %s: %s. "
            "Falling back to the racy service-layer id dedup; "
            "id uniqueness is NOT enforced at the database level.",
            collection_name,
            exc,
        )
        ASSET_ID_INDEX_BUILD_FAILED_TOTAL.labels(collection=collection_name).inc()


async def backfill_missing_id(
    collection: AsyncIOMotorCollection,
    collection_name: str,
) -> None:
    """Assign a uuid4 to any legacy document missing a non-empty ``id``.

    One-shot, run before the unique index is built. Mirrors
    backfill_normalized_identity_url. Expected to touch ~0 rows.
    """
    try:
        cursor = collection.find(
            {"$or": [{"id": {"$exists": False}}, {"id": None}, {"id": ""}]},
            projection={"_id": 1},
        )
        updated = 0
        async for doc in cursor:
            await collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"id": str(uuid4())}},
            )
            updated += 1
        if updated:
            logger.info("Backfilled id on %d %s documents", updated, collection_name)
    except Exception as exc:
        logger.warning("id backfill failed on %s: %s", collection_name, exc)


async def find_doc_by_id(
    collection: AsyncIOMotorCollection,
    asset_id: str,
) -> dict[str, Any] | None:
    """Indexed lookup by ``id``.

    Returns the document with ``_id`` remapped to ``path``, or None.
    """
    doc = await collection.find_one({"id": asset_id})
    if doc is None:
        return None
    doc["path"] = doc.pop("_id")
    return doc
