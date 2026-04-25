"""Shared MongoDB connection string builder.

Provides a single source of truth for building MongoDB/DocumentDB connection
strings and TLS options, used by both the async motor client and the
synchronous MongoDBLogHandler.
"""

from typing import Any


def build_connection_string() -> str:
    """Build a MongoDB/DocumentDB connection string from registry settings.

    Handles three authentication modes:
    - IAM (MONGODB-AWS) for DocumentDB with AWS credentials
    - Username/password (SCRAM-SHA-256 for MongoDB CE, SCRAM-SHA-1 for DocumentDB)
    - No authentication (local development)
    """
    from ..core.config import settings

    if settings.documentdb_use_iam:
        import boto3

        session = boto3.Session()
        credentials = session.get_credentials()
        if not credentials:
            raise ValueError("AWS credentials not found for DocumentDB IAM auth")
        return (
            f"mongodb://{credentials.access_key}:{credentials.secret_key}@"
            f"{settings.documentdb_host}:{settings.documentdb_port}/"
            f"{settings.documentdb_database}?"
            f"authSource=$external&authMechanism=MONGODB-AWS"
        )

    if settings.documentdb_username and settings.documentdb_password:
        if settings.storage_backend == "mongodb-ce":
            auth_mechanism = "SCRAM-SHA-256"
        else:
            auth_mechanism = "SCRAM-SHA-1"
        return (
            f"mongodb://{settings.documentdb_username}:{settings.documentdb_password}@"
            f"{settings.documentdb_host}:{settings.documentdb_port}/"
            f"{settings.documentdb_database}?authMechanism={auth_mechanism}&authSource=admin"
        )

    return (
        f"mongodb://{settings.documentdb_host}:{settings.documentdb_port}/"
        f"{settings.documentdb_database}"
    )


def build_tls_kwargs() -> dict[str, Any]:
    """Build TLS keyword arguments for MongoDB client."""
    from ..core.config import settings

    kwargs: dict[str, Any] = {}
    if settings.documentdb_use_tls:
        kwargs["tls"] = True
        if settings.documentdb_tls_ca_file:
            kwargs["tlsCAFile"] = settings.documentdb_tls_ca_file
    return kwargs


def build_client_options() -> dict[str, Any]:
    """Build common client options for MongoDB connections."""
    from ..core.config import settings

    options: dict[str, Any] = {"retryWrites": False}
    if settings.documentdb_direct_connection:
        options["directConnection"] = True
    return options
