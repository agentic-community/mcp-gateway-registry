"""DocumentDB/MongoDB Groups Enrichment for M2M Tokens.

This module provides functionality to enrich M2M tokens with groups from DocumentDB/MongoDB
when the IdP token has empty groups claim. This solves the authorization problem
for M2M clients across all identity providers (Keycloak, Okta, Entra).

Works with both:
- AWS DocumentDB (with IAM auth or username/password)
- MongoDB Community Edition (local or cloud)
"""

import logging

from motor.motor_asyncio import AsyncIOMotorDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


_mongodb_database: AsyncIOMotorDatabase | None = None


async def _get_mongodb() -> AsyncIOMotorDatabase:
    """Get MongoDB/DocumentDB database connection singleton.

    This uses the same connection logic as the registry to ensure compatibility
    with both MongoDB Community Edition and AWS DocumentDB.

    Returns:
        MongoDB/DocumentDB database instance

    Raises:
        ValueError: If database connection parameters not configured
    """
    global _mongodb_client, _mongodb_database

    if _mongodb_database is not None:
        return _mongodb_database

    try:
        # Use the registry's DocumentDB client for compatibility
        # This handles both MongoDB CE and AWS DocumentDB with proper auth mechanisms
        import sys
        from pathlib import Path

        # Add registry path to sys.path if not already there
        registry_path = Path(__file__).parent.parent / "registry"
        if str(registry_path) not in sys.path:
            sys.path.insert(0, str(registry_path.parent))

        from registry.repositories.documentdb.client import get_documentdb_client

        _mongodb_database = await get_documentdb_client()
        logger.info("✓ Connected to DocumentDB/MongoDB for groups enrichment")

        return _mongodb_database

    except Exception as e:
        logger.error(f"Failed to connect to DocumentDB/MongoDB: {e}")
        raise ValueError(f"Database connection failed: {e}")


async def enrich_groups_from_mongodb(
    client_id: str,
    current_groups: list[str],
) -> list[str]:
    """Enrich groups from DocumentDB/MongoDB if current groups are empty.

    This function checks if an M2M client has groups defined in the database
    and returns them if the current groups list is empty. This provides
    a fallback authorization mechanism for M2M tokens.

    Works with both AWS DocumentDB and MongoDB Community Edition.

    Args:
        client_id: Client ID from the JWT token
        current_groups: Current groups from JWT token

    Returns:
        Enriched groups list (either from MongoDB or original)
    """
    # If groups already exist in token (non-empty array), use them
    if current_groups:
        logger.debug(f"Client {client_id} has groups in token: {current_groups}")
        return current_groups

    logger.info(f"Client {client_id} has no groups in token, querying database")

    # Try to fetch groups from DocumentDB/MongoDB
    try:
        db = await _get_mongodb()
        collection = db["idp_m2m_clients"]

        doc = await collection.find_one({"client_id": client_id})

        if doc:
            db_groups = doc.get("groups", [])
            if db_groups:
                logger.info(f"Enriched groups for client {client_id} from database: {db_groups}")
                return db_groups
            else:
                logger.debug(f"Client {client_id} found in database but has no groups")
        else:
            logger.debug(f"Client {client_id} not found in groups database")

    except Exception as e:
        logger.warning(f"Failed to query database for groups enrichment: {e}")
        # Don't fail token validation if database is unavailable

    # Return original empty groups if no enrichment possible
    return current_groups


def should_enrich_groups(validation_result: dict) -> bool:
    """Check if groups should be enriched from MongoDB.

    Args:
        validation_result: Token validation result dictionary

    Returns:
        True if groups enrichment should be attempted
    """
    # Only enrich if:
    # 1. Token is valid
    # 2. Groups list is empty (not present or empty array)
    # 3. Has a client_id
    is_valid = validation_result.get("valid", False)
    groups = validation_result.get("groups", [])
    client_id = validation_result.get("client_id")

    return is_valid and not groups and client_id is not None
