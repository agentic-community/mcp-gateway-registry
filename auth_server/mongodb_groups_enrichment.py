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
        # Count only: group names are organizational PII.
        logger.debug(f"Client {client_id} has {len(current_groups)} groups in token")
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
                logger.info(
                    f"Enriched {len(db_groups)} groups for client {client_id} from database"
                )
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


async def enrich_user_groups_from_mongodb(
    username: str,
    current_groups: list[str],
    provider: str,
) -> list[str]:
    """Enrich user groups from DocumentDB/MongoDB if current groups are empty.

    Mirrors enrich_groups_from_mongodb but reads the idp_user_groups
    collection and looks up by username. Used as a fallback when an IdP's
    user JWT does not carry a groups claim (e.g., PingFederate without the
    custom ATM groups attribute).

    Works with both AWS DocumentDB and MongoDB Community Edition.

    Args:
        username: Username (sub or preferred_username) from the JWT token
        current_groups: Current groups from JWT token
        provider: IdP name the token came from (e.g., "pingfederate"), used
            for logging and future per-provider scoping; the lookup itself
            is by username only.

    Returns:
        Enriched groups list (either from MongoDB or original)
    """
    # If groups already exist in token (non-empty array), use them
    if current_groups:
        # Count only: group names are organizational PII.
        logger.debug(f"User {username} has {len(current_groups)} groups in token")
        return current_groups

    logger.info(
        f"User {username} (provider={provider}) has no groups in token, querying database"
    )

    # Try to fetch groups from DocumentDB/MongoDB
    try:
        db = await _get_mongodb()
        collection = db["idp_user_groups"]

        doc = await collection.find_one({"username": username})

        if doc:
            db_groups = doc.get("groups", [])
            if db_groups:
                logger.info(
                    f"Enriched {len(db_groups)} groups for user {username} from database"
                )
                return db_groups
            else:
                logger.debug(f"User {username} found in database but has no groups")
        else:
            logger.debug(f"User {username} not found in idp_user_groups database")

    except Exception as e:
        logger.warning(f"Failed to query database for user groups enrichment: {e}")
        # Don't fail token validation if database is unavailable

    # Return original empty groups if no enrichment possible
    return current_groups


def should_enrich_user_groups(
    username: str,
    current_groups: list[str],
    provider: str | None,
    enabled_providers: list[str],
) -> bool:
    """Check if user groups should be enriched from MongoDB.

    Args:
        username: Username from validated token
        current_groups: Current groups from validated token
        provider: IdP that issued the token (e.g., "pingfederate"); may be None
        enabled_providers: Lowercase list of providers for which user-group
            fallback is enabled. Compared case-insensitively.

    Returns:
        True iff provider is enabled, current groups are empty, and username
        is non-empty.
    """
    if not provider:
        return False

    if provider.lower() not in enabled_providers:
        return False

    if current_groups:
        return False

    if not username:
        return False

    return True
