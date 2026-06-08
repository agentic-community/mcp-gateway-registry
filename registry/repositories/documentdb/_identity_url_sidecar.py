"""Shared helpers for the ``_identity_url_normalized`` sidecar field.

The DocumentDB server / agent / skill repositories each store a
sidecar field next to their user-facing identity URL
(``proxy_pass_url`` / ``url`` / ``skill_md_url``). The sidecar holds
the normalized form of that URL so registration deduplication can
look up by indexed ``$eq`` instead of scanning and normalizing
client-side.

The three repos differ only in the entity type they pass to
:func:`derive_normalized_identity_url`. These helpers wrap the
shared work — populating the sidecar on writes, ensuring the index,
backfilling legacy rows — so the per-repo glue stays thin.
"""

import logging
from typing import Any

from motor.motor_asyncio import AsyncIOMotorCollection

from ...utils.url_normalize import (
    IDENTITY_URL_FIELD_BY_ENTITY,
    NORMALIZED_IDENTITY_URL_FIELD,
    derive_normalized_identity_url,
)

logger = logging.getLogger(__name__)


def populate_normalized_identity_url(
    doc: dict[str, Any],
    entity_type: str,
) -> None:
    """Set the ``_identity_url_normalized`` sidecar on ``doc`` in place.

    Removes the sidecar entirely when the source URL is missing or
    unparseable so the sparse index doesn't keep a stale value.
    """
    normalized = derive_normalized_identity_url(doc, entity_type)
    if normalized is None:
        doc.pop(NORMALIZED_IDENTITY_URL_FIELD, None)
    else:
        doc[NORMALIZED_IDENTITY_URL_FIELD] = normalized


async def ensure_normalized_identity_url_index(
    collection: AsyncIOMotorCollection,
    collection_name: str,
) -> None:
    """Create the sparse index on the sidecar field if missing.

    Tolerates DocumentDB engines that reject sparse indexes — the
    dedup path still works without the index, it just scans more
    documents.
    """
    try:
        await collection.create_index(NORMALIZED_IDENTITY_URL_FIELD, sparse=True)
    except Exception as exc:
        logger.warning(
            "Could not create %s index on %s: %s",
            NORMALIZED_IDENTITY_URL_FIELD,
            collection_name,
            exc,
        )


async def backfill_normalized_identity_url(
    collection: AsyncIOMotorCollection,
    collection_name: str,
    entity_type: str,
) -> None:
    """One-shot backfill of the sidecar for legacy documents.

    Scans documents that have the source URL but no sidecar and
    populates the sidecar in place. Errors are logged and swallowed
    — dedup is advisory, so a partially backfilled collection still
    works for newly-written rows.
    """
    source_field = IDENTITY_URL_FIELD_BY_ENTITY.get(entity_type)
    if source_field is None:
        return
    try:
        cursor = collection.find(
            {
                source_field: {"$exists": True, "$ne": None},
                NORMALIZED_IDENTITY_URL_FIELD: {"$exists": False},
            },
            projection={"_id": 1, source_field: 1},
        )
        updated = 0
        async for doc in cursor:
            normalized = derive_normalized_identity_url(doc, entity_type)
            if normalized is None:
                continue
            await collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {NORMALIZED_IDENTITY_URL_FIELD: normalized}},
            )
            updated += 1
        if updated:
            logger.info(
                "Backfilled %s on %d %s document(s)",
                NORMALIZED_IDENTITY_URL_FIELD,
                updated,
                entity_type,
            )
    except Exception as exc:
        logger.warning(
            "Backfill of %s on %s failed: %s",
            NORMALIZED_IDENTITY_URL_FIELD,
            collection_name,
            exc,
        )


async def find_by_normalized_identity_url(
    collection: AsyncIOMotorCollection,
    identity_url: str,
) -> dict[str, Any] | None:
    """Indexed ``$eq`` lookup against the sidecar field.

    Returns the matching document with ``_id`` re-keyed as ``path``
    (matching the rest of the repository convention), or None when
    no match exists.

    Projects out ``_identity_url_normalized`` server-side as
    defense-in-depth: the sidecar is internal registry bookkeeping,
    and even though Pydantic response models would ignore it today,
    no upstream caller of this helper should ever see it. Keeping
    that contract enforced at the data-access boundary means a
    future raw-doc API surface can't accidentally leak it.
    """
    if not identity_url:
        return None
    full_doc = await collection.find_one(
        {NORMALIZED_IDENTITY_URL_FIELD: identity_url},
        projection={NORMALIZED_IDENTITY_URL_FIELD: 0},
    )
    if full_doc is None:
        return None
    full_doc["path"] = full_doc.pop("_id")
    return full_doc
