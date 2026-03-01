"""DocumentDB client singleton with IAM authentication support."""

import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from ...core.config import settings


logger = logging.getLogger(__name__)

_client: Optional[AsyncIOMotorClient] = None
_database: Optional[AsyncIOMotorDatabase] = None


async def get_documentdb_client() -> AsyncIOMotorDatabase:
    """Get DocumentDB database client singleton."""
    global _client, _database

    if _database is not None:
        return _database

    # Build connection parameters
    if settings.documentdb_use_iam:
        # IAM authentication for DocumentDB
        import boto3

        session = boto3.Session()
        credentials = session.get_credentials()

        if not credentials:
            raise ValueError("AWS credentials not found for DocumentDB IAM auth")

        # DocumentDB connection string with IAM auth
        connection_string = (
            f"mongodb://{credentials.access_key}:{credentials.secret_key}@"
            f"{settings.documentdb_host}:{settings.documentdb_port}/"
            f"{settings.documentdb_database}?"
            f"authSource=$external&authMechanism=MONGODB-AWS"
        )

        logger.info(
            f"Using AWS IAM authentication (MONGODB-AWS) for {settings.storage_backend} "
            f"(host: {settings.documentdb_host})"
        )

    else:
        # Username/password authentication
        if settings.documentdb_username and settings.documentdb_password:
            # Choose auth mechanism based on storage backend
            # - MongoDB CE 8.2+: Use SCRAM-SHA-256 (stronger, modern authentication)
            # - AWS DocumentDB v5.0: Only supports SCRAM-SHA-1
            if settings.storage_backend == "mongodb-ce":
                auth_mechanism = "SCRAM-SHA-256"
            else:
                # AWS DocumentDB (storage_backend="documentdb")
                auth_mechanism = "SCRAM-SHA-1"

            connection_string = (
                f"mongodb://{settings.documentdb_username}:{settings.documentdb_password}@"
                f"{settings.documentdb_host}:{settings.documentdb_port}/"
                f"{settings.documentdb_database}?authMechanism={auth_mechanism}&authSource=admin"
            )

            logger.info(
                f"Using username/password authentication ({auth_mechanism}) for "
                f"{settings.storage_backend} (host: {settings.documentdb_host})"
            )
        else:
            # No authentication (local development)
            connection_string = (
                f"mongodb://{settings.documentdb_host}:{settings.documentdb_port}/"
                f"{settings.documentdb_database}"
            )
            logger.info(
                f"Using no authentication for {settings.storage_backend} "
                f"(host: {settings.documentdb_host})"
            )

    # Prepare TLS options
    tls_options = {}
    if settings.documentdb_use_tls:
        tls_options["tls"] = True
        if settings.documentdb_tls_ca_file:
            tls_options["tlsCAFile"] = settings.documentdb_tls_ca_file
            logger.info(f"Using TLS CA file: {settings.documentdb_tls_ca_file}")

    # Create client with TLS options
    # IMPORTANT: DocumentDB does not support retryable writes
    # Use directConnection only for single-node MongoDB (tests), not for DocumentDB clusters
    client_options = {"retryWrites": False}
    if settings.documentdb_direct_connection:
        client_options["directConnection"] = True

    _client = AsyncIOMotorClient(connection_string, **client_options, **tls_options)
    _database = _client[settings.documentdb_database]

    # Verify connection
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
