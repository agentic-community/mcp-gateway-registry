"""Shared MongoDB connection string builder.

Provides a single source of truth for building MongoDB/DocumentDB connection
strings and TLS options, used by both the async motor client and the
synchronous MongoDBLogHandler.
"""

from typing import Any


def build_connection_string() -> str:
    """Build a MongoDB/DocumentDB connection string from registry settings.

    If ``mongodb_connection_string`` is set, it is returned verbatim and all
    other auth/host branches are skipped. Otherwise, handles three modes:
    - IAM (MONGODB-AWS) for DocumentDB with AWS credentials
    - Username/password. AWS DocumentDB is SCRAM-SHA-1 (v5.0 limitation);
      every other MongoDB variant (CE, self-managed, Atlas, etc.) uses
      SCRAM-SHA-256.
    - No authentication (local development)
    """
    from ..core.config import settings

    if settings.mongodb_connection_string:
        return settings.mongodb_connection_string

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
        # AWS DocumentDB v5.0 only supports SCRAM-SHA-1. All other
        # MongoDB-compatible backends (mongodb-ce / mongodb / mongodb-atlas
        # and any future alias) support SCRAM-SHA-256.
        if settings.storage_backend == "documentdb":
            auth_mechanism = "SCRAM-SHA-1"
        else:
            auth_mechanism = "SCRAM-SHA-256"
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
    """Build TLS keyword arguments for MongoDB client.

    When ``mongodb_connection_string`` is set, the URI owns TLS configuration
    (e.g. ``mongodb+srv://`` implies TLS automatically), so we return an
    empty dict.
    """
    from ..core.config import settings

    if settings.mongodb_connection_string:
        return {}

    kwargs: dict[str, Any] = {}
    if settings.documentdb_use_tls:
        kwargs["tls"] = True
        if settings.documentdb_tls_ca_file:
            kwargs["tlsCAFile"] = settings.documentdb_tls_ca_file
    return kwargs


def build_client_options() -> dict[str, Any]:
    """Build common client options for MongoDB connections.

    When ``mongodb_connection_string`` is set, the URI owns all options
    (retryWrites, directConnection, replicaSet, etc.), so we return an
    empty dict and let the caller-supplied URI decide.
    """
    from ..core.config import settings

    if settings.mongodb_connection_string:
        return {}

    options: dict[str, Any] = {"retryWrites": False}
    if settings.documentdb_direct_connection:
        options["directConnection"] = True
    return options
