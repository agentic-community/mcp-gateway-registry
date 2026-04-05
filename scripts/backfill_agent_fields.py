"""One-time backfill: normalize supported_protocol and trust_level on existing agents."""

import logging

from pymongo import MongoClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

MONGODB_URI = "mongodb://localhost:27017"
DB_NAME = "mcp_registry"
COLLECTION = "mcp_agents_default"


def _backfill_supported_protocol(
    collection,
) -> None:
    """Set supported_protocol='other' on agents that don't have the field."""
    result = collection.update_many(
        {"supported_protocol": {"$exists": False}},
        {"$set": {"supported_protocol": "other"}},
    )
    logger.info(
        f"supported_protocol backfill: {result.modified_count} agents updated"
    )


def _backfill_trust_level(
    collection,
) -> None:
    """Update trust_level from 'unverified' to 'community' for consistency."""
    result = collection.update_many(
        {"trust_level": "unverified"},
        {"$set": {"trust_level": "community"}},
    )
    logger.info(
        f"trust_level backfill: {result.modified_count} agents updated"
    )


def _backfill_visibility(
    collection,
) -> None:
    """Update visibility from 'internal' to 'public' for consistency."""
    result = collection.update_many(
        {"visibility": "internal"},
        {"$set": {"visibility": "public"}},
    )
    logger.info(
        f"visibility backfill: {result.modified_count} agents updated"
    )


def backfill_agent_fields() -> None:
    """Run all backfill operations."""
    client = MongoClient(MONGODB_URI, directConnection=True)
    db = client[DB_NAME]
    collection = db[COLLECTION]

    _backfill_supported_protocol(collection)
    _backfill_trust_level(collection)
    _backfill_visibility(collection)

    logger.info("Backfill complete")


if __name__ == "__main__":
    backfill_agent_fields()
