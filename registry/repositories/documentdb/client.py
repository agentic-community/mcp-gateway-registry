"""DocumentDB client singleton with IAM authentication support."""

import logging

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from ...core.config import settings
from ...utils.mongodb_connection import build_client_options, build_connection_string, build_tls_kwargs

logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient | None = None
_database: AsyncIOMotorDatabase | None = None


async def get_documentdb_client() -> AsyncIOMotorDatabase:
    """Get DocumentDB database client singleton."""
    global _client, _database

    if _database is not None:
        return _database

    connection_string = build_connection_string()
    logger.info(
        f"Connecting to {settings.storage_backend} "
        f"(host: {settings.documentdb_host})"
    )

    _client = AsyncIOMotorClient(
        connection_string,
        **build_client_options(),
        **build_tls_kwargs(),
    )
    _database = _client[settings.documentdb_database]

    server_info = await _client.server_info()
    logger.info(f"Connected to DocumentDB/MongoDB {server_info.get('version', 'unknown')}")

    return _database


async def close_documentdb_client() -> None:
    """Close DocumentDB client."""
    global _client, _database
    if _client is not None:
        _client.close()
        _client = None
        _database = None


def get_collection_name(
    base_name: str,
) -> str:
    """Get full collection name with namespace."""
    return f"{base_name}_{settings.documentdb_namespace}"
