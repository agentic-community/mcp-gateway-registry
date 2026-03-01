#!/usr/bin/env python3
"""
Initialize DocumentDB collections and indexes for MCP Gateway Registry.

This script creates all necessary vector indexes and standard indexes for
the MCP Gateway Registry DocumentDB backend.

Usage:
    # Using environment variables
    export DOCUMENTDB_HOST=your-cluster.docdb.amazonaws.com
    export DOCUMENTDB_USERNAME=admin
    export DOCUMENTDB_PASSWORD=yourpassword
    uv run python scripts/init-documentdb-indexes.py

    # Using command-line arguments
    uv run python scripts/init-documentdb-indexes.py --host your-cluster.docdb.amazonaws.com
    uv run python scripts/init-documentdb-indexes.py --use-iam --host your-cluster.docdb.amazonaws.com

    # With namespace
    uv run python scripts/init-documentdb-indexes.py --namespace tenant-a

    # Recreate indexes
    uv run python scripts/init-documentdb-indexes.py --recreate

Requires:
    - motor (AsyncIOMotorClient)
    - boto3 (for IAM authentication)
    - DocumentDB connection details via environment variables or command-line
"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient


# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


# Collection names
COLLECTION_SERVERS = "mcp_servers"
COLLECTION_AGENTS = "mcp_agents"
COLLECTION_SCOPES = "mcp_scopes"
COLLECTION_EMBEDDINGS = "mcp_embeddings_1536"
COLLECTION_SECURITY_SCANS = "mcp_security_scans"
COLLECTION_FEDERATION_CONFIG = "mcp_federation_config"
COLLECTION_AUDIT_EVENTS = "audit_events"


async def _get_documentdb_connection_string(
    host: str,
    port: int,
    database: str,
    username: Optional[str],
    password: Optional[str],
    use_iam: bool,
    use_tls: bool,
    tls_ca_file: Optional[str],
    storage_backend: str = "documentdb",
) -> str:
    """Build DocumentDB connection string with appropriate auth mechanism.

    Args:
        storage_backend: Either 'documentdb' (uses SCRAM-SHA-1) or 'mongodb-ce' (uses SCRAM-SHA-256)
    """
    if use_iam:
        import boto3

        session = boto3.Session()
        credentials = session.get_credentials()

        if not credentials:
            raise ValueError("AWS credentials not found for DocumentDB IAM auth")

        connection_string = (
            f"mongodb://{credentials.access_key}:{credentials.secret_key}@"
            f"{host}:{port}/{database}?"
            f"tls=true&authSource=$external&authMechanism=MONGODB-AWS"
        )

        if tls_ca_file:
            connection_string += f"&tlsCAFile={tls_ca_file}"

        logger.info(f"Using AWS IAM authentication for DocumentDB (host: {host})")

    else:
        if username and password:
            # Choose auth mechanism based on storage backend
            # - MongoDB CE 8.2+: Use SCRAM-SHA-256 (stronger, modern authentication)
            # - AWS DocumentDB v5.0: Only supports SCRAM-SHA-1
            if storage_backend == "mongodb-ce":
                auth_mechanism = "SCRAM-SHA-256"
            else:
                # AWS DocumentDB (storage_backend="documentdb")
                auth_mechanism = "SCRAM-SHA-1"

            connection_string = (
                f"mongodb://{username}:{password}@"
                f"{host}:{port}/{database}?"
                f"authMechanism={auth_mechanism}&authSource=admin&"
                f"tls={str(use_tls).lower()}"
            )

            if use_tls and tls_ca_file:
                connection_string += f"&tlsCAFile={tls_ca_file}"

            logger.info(
                f"Using username/password authentication ({auth_mechanism}) for "
                f"{storage_backend} (host: {host})"
            )
        else:
            connection_string = f"mongodb://{host}:{port}/{database}"
            logger.info(f"Using no authentication for DocumentDB (host: {host})")

    return connection_string


async def _create_vector_index(
    collection,
    collection_name: str,
    recreate: bool,
) -> None:
    """Create vector index for embeddings collection.

    Note: DocumentDB Elastic does not support vector indexes.
    This will be skipped for DocumentDB deployments.
    """
    index_name = "embedding_vector_idx"

    try:
        await collection.create_index(
            [("embedding", "vector")],
            name=index_name,
            vectorOptions={
                "type": "hnsw",
                "similarity": "cosine",
                "dimensions": 1536,
                "m": 16,
                "efConstruction": 128,
            },
        )
        logger.info(f"Created vector index '{index_name}' on {collection_name}")
    except Exception as e:
        # Debug logging
        logger.info(f"DEBUG: Caught exception in vector index creation")
        logger.info(f"DEBUG: Exception type: {type(e).__name__}")
        logger.info(f"DEBUG: Exception str: {str(e)}")
        logger.info(f"DEBUG: Exception repr: {repr(e)}")

        # Check if index already exists with different options (error code 85)
        if (
            "'code': 85" in str(e) or "code': 85" in str(e)
        ) or "already exists with different options" in str(e).lower():
            if recreate:
                logger.info(f"Vector index exists with different options. Recreating...")

                # List all indexes to see what's there
                logger.info(f"Listing all indexes on {collection_name}...")
                indexes = await collection.list_indexes().to_list(None)
                for idx in indexes:
                    logger.info(
                        f"  Found index: name='{idx.get('name')}', key={idx.get('key', {})}"
                    )

                # Drop ALL non-_id indexes to ensure clean slate
                dropped_count = 0
                for idx in indexes:
                    idx_name = idx.get("name")
                    if idx_name and idx_name != "_id_":
                        try:
                            await collection.drop_index(idx_name)
                            logger.info(f"Dropped index '{idx_name}' from {collection_name}")
                            dropped_count += 1
                        except Exception as drop_err:
                            logger.warning(f"Failed to drop index '{idx_name}': {drop_err}")

                logger.info(f"Dropped {dropped_count} indexes from {collection_name}")

                # Now try to create again
                try:
                    await collection.create_index(
                        [("embedding", "vector")],
                        name=index_name,
                        vectorOptions={
                            "type": "hnsw",
                            "similarity": "cosine",
                            "dimensions": 1536,
                            "m": 16,
                            "efConstruction": 128,
                        },
                    )
                    logger.info(
                        f"Created vector index '{index_name}' on {collection_name} after dropping {dropped_count} old indexes"
                    )
                except Exception as create_err:
                    logger.error(
                        f"Failed to create vector index after dropping all indexes: {create_err}",
                        exc_info=True,
                    )
                    raise
            else:
                logger.info(
                    f"Vector index already exists on {collection_name} (recreate=False, skipping)"
                )
        # DocumentDB Elastic doesn't support vector indexes (error code 303)
        elif "vectorOptions" in str(e) or "not supported" in str(e):
            logger.warning(
                f"Vector indexes not supported (DocumentDB Elastic limitation). "
                f"Skipping vector index creation for {collection_name}. "
                f"Vector search will use fallback implementation."
            )
        else:
            logger.error(f"Failed to create vector index on {collection_name}: {e}", exc_info=True)
            raise


async def _create_embeddings_indexes(
    collection,
    collection_name: str,
    recreate: bool,
) -> None:
    """Create all indexes for embeddings collection."""
    await _create_vector_index(collection, collection_name, recreate)

    indexes = [
        ("name", 1),
        ("path", 1),
        ("entity_type", 1),
    ]

    for field, order in indexes:
        index_name = f"{field}_idx"
        unique = field == "path"

        if recreate:
            try:
                await collection.drop_index(index_name)
                logger.info(f"Dropped existing index '{index_name}' from {collection_name}")
            except Exception as e:
                logger.debug(f"No existing index '{index_name}' to drop: {e}")

        try:
            await collection.create_index(
                [(field, order)],
                name=index_name,
                unique=unique,
            )
            logger.info(
                f"Created {'unique ' if unique else ''}index '{index_name}' on {collection_name}"
            )
        except Exception as e:
            logger.error(f"Failed to create index '{index_name}' on {collection_name}: {e}")


async def _create_servers_indexes(
    collection,
    collection_name: str,
    recreate: bool,
) -> None:
    """Create all indexes for servers collection."""
    indexes = [
        ("server_name", 1, False),
        ("is_enabled", 1, False),
        ("version", 1, False),
        ("tags", 1, False),
    ]

    for field, order, unique in indexes:
        index_name = f"{field}_idx"

        if recreate:
            try:
                await collection.drop_index(index_name)
                logger.info(f"Dropped existing index '{index_name}' from {collection_name}")
            except Exception as e:
                logger.debug(f"No existing index '{index_name}' to drop: {e}")

        try:
            await collection.create_index(
                [(field, order)],
                name=index_name,
                unique=unique,
            )
            logger.info(
                f"Created {'unique ' if unique else ''}index '{index_name}' on {collection_name}"
            )
        except Exception as e:
            logger.error(f"Failed to create index '{index_name}' on {collection_name}: {e}")


async def _create_agents_indexes(
    collection,
    collection_name: str,
    recreate: bool,
) -> None:
    """Create all indexes for agents collection."""
    indexes = [
        ("name", 1, False),
        ("is_enabled", 1, False),
        ("version", 1, False),
        ("tags", 1, False),
    ]

    for field, order, unique in indexes:
        index_name = f"{field}_idx"

        if recreate:
            try:
                await collection.drop_index(index_name)
                logger.info(f"Dropped existing index '{index_name}' from {collection_name}")
            except Exception as e:
                logger.debug(f"No existing index '{index_name}' to drop: {e}")

        try:
            await collection.create_index(
                [(field, order)],
                name=index_name,
                unique=unique,
            )
            logger.info(
                f"Created {'unique ' if unique else ''}index '{index_name}' on {collection_name}"
            )
        except Exception as e:
            logger.error(f"Failed to create index '{index_name}' on {collection_name}: {e}")


async def _create_scopes_indexes(
    collection,
    collection_name: str,
    recreate: bool,
) -> None:
    """Create all indexes for scopes collection."""
    indexes = [
        ("name", 1, False),
    ]

    for field, order, unique in indexes:
        index_name = f"{field}_idx"

        if recreate:
            try:
                await collection.drop_index(index_name)
                logger.info(f"Dropped existing index '{index_name}' from {collection_name}")
            except Exception as e:
                logger.debug(f"No existing index '{index_name}' to drop: {e}")

        try:
            await collection.create_index(
                [(field, order)],
                name=index_name,
                unique=unique,
            )
            logger.info(
                f"Created {'unique ' if unique else ''}index '{index_name}' on {collection_name}"
            )
        except Exception as e:
            logger.error(f"Failed to create index '{index_name}' on {collection_name}: {e}")


async def _load_default_scopes(
    db,
    namespace: str,
    entra_group_id: Optional[str] = None,
) -> None:
    """Load default admin scope from JSON file into scopes collection.

    Args:
        db: Database connection
        namespace: Collection namespace
        entra_group_id: Optional Entra ID Group Object ID to add to group_mappings.
                        Required when using Microsoft Entra ID as the auth provider.
    """
    collection_name = f"{COLLECTION_SCOPES}_{namespace}"
    collection = db[collection_name]

    # Find the registry-admins.json file in the same directory as this script
    script_dir = Path(__file__).parent
    admin_scope_file = script_dir / "registry-admins.json"

    if not admin_scope_file.exists():
        logger.warning(f"Default admin scope file not found: {admin_scope_file}")
        return

    try:
        with open(admin_scope_file, "r") as f:
            admin_scope = json.load(f)

        logger.info(f"Loading default admin scope from {admin_scope_file}")

        # Add Entra ID Group Object ID if provided
        if entra_group_id:
            if entra_group_id not in admin_scope.get("group_mappings", []):
                admin_scope["group_mappings"].append(entra_group_id)
                logger.info(f"Added Entra ID Group Object ID: {entra_group_id}")

        # Upsert the admin scope document
        result = await collection.update_one(
            {"_id": admin_scope["_id"]}, {"$set": admin_scope}, upsert=True
        )

        if result.upserted_id:
            logger.info(f"Inserted admin scope: {admin_scope['_id']}")
        elif result.modified_count > 0:
            logger.info(f"Updated admin scope: {admin_scope['_id']}")
        else:
            logger.info(f"Admin scope already up-to-date: {admin_scope['_id']}")

        logger.info(f"Admin scope group_mappings: {admin_scope.get('group_mappings', [])}")

    except Exception as e:
        logger.error(f"Failed to load default admin scope: {e}", exc_info=True)


async def _create_security_scans_indexes(
    collection,
    collection_name: str,
    recreate: bool,
) -> None:
    """Create all indexes for security scans collection."""
    indexes = [
        ("entity_path", 1, False),
        ("entity_type", 1, False),
        ("scan_status", 1, False),
        ("scanned_at", 1, False),
    ]

    for field, order, unique in indexes:
        index_name = f"{field}_idx"

        if recreate:
            try:
                await collection.drop_index(index_name)
                logger.info(f"Dropped existing index '{index_name}' from {collection_name}")
            except Exception as e:
                logger.debug(f"No existing index '{index_name}' to drop: {e}")

        try:
            await collection.create_index(
                [(field, order)],
                name=index_name,
                unique=unique,
            )
            logger.info(
                f"Created {'unique ' if unique else ''}index '{index_name}' on {collection_name}"
            )
        except Exception as e:
            logger.error(f"Failed to create index '{index_name}' on {collection_name}: {e}")


async def _create_federation_config_indexes(
    collection,
    collection_name: str,
    recreate: bool,
) -> None:
    """Create all indexes for federation config collection."""
    # No additional indexes needed - _id is automatically indexed
    logger.info(f"No additional indexes to create for {collection_name} (_id is auto-indexed)")


async def _create_audit_events_indexes(
    collection,
    collection_name: str,
    recreate: bool,
) -> None:
    """Create all indexes for audit events collection including TTL index.

    Indexes support:
    - Query by username + time range
    - Query by operation + time range
    - Query by resource type + time range
    - Composite unique lookup by (request_id, log_type)
    - TTL-based automatic expiration (default 7 days)
    """
    # Standard query indexes (compound with timestamp for range queries)
    indexes = [
        (("identity.username", 1), ("timestamp", 1)),
        (("action.operation", 1), ("timestamp", 1)),
        (("action.resource_type", 1), ("timestamp", 1)),
    ]

    # Single-field index for MCP server name distinct/filter queries
    single_field_indexes = [
        ("mcp_server.name", 1),
    ]

    for fields in indexes:
        index_spec = [(f[0], f[1]) for f in fields]
        index_name = "_".join(f[0].replace(".", "_") for f in fields) + "_idx"

        if recreate:
            try:
                await collection.drop_index(index_name)
                logger.info(f"Dropped existing index '{index_name}' from {collection_name}")
            except Exception as e:
                logger.debug(f"No existing index '{index_name}' to drop: {e}")

        try:
            await collection.create_index(
                index_spec,
                name=index_name,
            )
            logger.info(f"Created index '{index_name}' on {collection_name}")
        except Exception as e:
            logger.error(f"Failed to create index '{index_name}' on {collection_name}: {e}")

    # Create single-field indexes for distinct/filter queries
    for field, order in single_field_indexes:
        index_name = field.replace(".", "_") + "_idx"

        if recreate:
            try:
                await collection.drop_index(index_name)
                logger.info(f"Dropped existing index '{index_name}' from {collection_name}")
            except Exception as e:
                logger.debug(f"No existing index '{index_name}' to drop: {e}")

        try:
            await collection.create_index(
                [(field, order)],
                name=index_name,
            )
            logger.info(f"Created index '{index_name}' on {collection_name}")
        except Exception as e:
            logger.error(f"Failed to create index '{index_name}' on {collection_name}: {e}")

    # Composite unique index on (request_id, log_type)
    # Allows both MCPServerAccessRecord and RegistryApiAccessRecord
    # to coexist for the same request_id while preventing true duplicates
    composite_index_name = "request_id_log_type_idx"
    old_index_name = "request_id_idx"

    # Always try to drop the old single-field index (migration from previous versions)
    try:
        await collection.drop_index(old_index_name)
        logger.info(f"Dropped old single-field index '{old_index_name}' from {collection_name}")
    except Exception as e:
        logger.debug(f"No old index '{old_index_name}' to drop: {e}")

    if recreate:
        try:
            await collection.drop_index(composite_index_name)
            logger.info(f"Dropped existing index '{composite_index_name}' from {collection_name}")
        except Exception as e:
            logger.debug(f"No existing index '{composite_index_name}' to drop: {e}")

    try:
        await collection.create_index(
            [("request_id", 1), ("log_type", 1)],
            name=composite_index_name,
            unique=True,
        )
        logger.info(f"Created composite unique index '{composite_index_name}' on {collection_name}")
    except Exception as e:
        logger.error(f"Failed to create index '{composite_index_name}' on {collection_name}: {e}")

    # TTL index for automatic expiration
    # Default 7 days (604800 seconds), configurable via AUDIT_LOG_MONGODB_TTL_DAYS
    ttl_index_name = "timestamp_ttl"
    ttl_days = int(os.getenv("AUDIT_LOG_MONGODB_TTL_DAYS", "7"))
    ttl_seconds = ttl_days * 24 * 60 * 60

    if recreate:
        try:
            await collection.drop_index(ttl_index_name)
            logger.info(f"Dropped existing TTL index '{ttl_index_name}' from {collection_name}")
        except Exception as e:
            logger.debug(f"No existing TTL index '{ttl_index_name}' to drop: {e}")

    try:
        await collection.create_index(
            [("timestamp", 1)],
            name=ttl_index_name,
            expireAfterSeconds=ttl_seconds,
        )
        logger.info(
            f"Created TTL index '{ttl_index_name}' on {collection_name} "
            f"(expireAfterSeconds={ttl_seconds}, {ttl_days} days)"
        )
    except Exception as e:
        logger.error(f"Failed to create TTL index on {collection_name}: {e}")


async def _print_collection_summary(
    db,
    namespace: str,
) -> None:
    """Print summary of all collections and their indexes."""
    logger.info("=" * 80)
    logger.info("DOCUMENTDB COLLECTIONS AND INDEXES SUMMARY")
    logger.info("=" * 80)

    collection_names = [
        f"{COLLECTION_SERVERS}_{namespace}",
        f"{COLLECTION_AGENTS}_{namespace}",
        f"{COLLECTION_SCOPES}_{namespace}",
        f"{COLLECTION_EMBEDDINGS}_{namespace}",
        f"{COLLECTION_SECURITY_SCANS}_{namespace}",
        f"{COLLECTION_FEDERATION_CONFIG}_{namespace}",
        f"{COLLECTION_AUDIT_EVENTS}_{namespace}",
    ]

    for coll_name in collection_names:
        try:
            collection = db[coll_name]

            # Get document count
            count = await collection.count_documents({})

            # Get indexes
            indexes = await collection.list_indexes().to_list(None)

            logger.info(f"\nCollection: {coll_name}")
            logger.info(f"  Documents: {count}")
            logger.info(f"  Indexes ({len(indexes)}):")

            for idx in indexes:
                idx_name = idx.get("name")
                if "vectorOptions" in idx:
                    vector_opts = idx["vectorOptions"]
                    logger.info(
                        f"    - {idx_name} (VECTOR: {vector_opts.get('type')}, "
                        f"dims={vector_opts.get('dimensions')}, "
                        f"similarity={vector_opts.get('similarity')})"
                    )
                else:
                    keys = idx.get("key", {})
                    unique = " UNIQUE" if idx.get("unique", False) else ""
                    logger.info(f"    - {idx_name} on {keys}{unique}")

        except Exception as e:
            logger.error(f"Error getting info for {coll_name}: {e}")

    logger.info("=" * 80)


async def _initialize_collections(
    db,
    namespace: str,
    recreate: bool,
    entra_group_id: Optional[str] = None,
) -> None:
    """Initialize all collections and indexes.

    Args:
        db: Database connection
        namespace: Collection namespace
        recreate: Whether to recreate existing indexes
        entra_group_id: Optional Entra ID Group Object ID for admin scope
    """
    collection_configs = [
        (COLLECTION_SERVERS, _create_servers_indexes),
        (COLLECTION_AGENTS, _create_agents_indexes),
        (COLLECTION_SCOPES, _create_scopes_indexes),
        (COLLECTION_EMBEDDINGS, _create_embeddings_indexes),
        (COLLECTION_SECURITY_SCANS, _create_security_scans_indexes),
        (COLLECTION_FEDERATION_CONFIG, _create_federation_config_indexes),
        (COLLECTION_AUDIT_EVENTS, _create_audit_events_indexes),
    ]

    for base_name, create_indexes_func in collection_configs:
        collection_name = f"{base_name}_{namespace}"
        collection = db[collection_name]

        logger.info(f"Creating indexes for collection: {collection_name}")

        # Create collection first (DocumentDB Elastic requires explicit collection creation)
        try:
            # Check if collection exists
            existing_collections = await db.list_collection_names()
            if collection_name not in existing_collections:
                logger.info(f"Creating collection: {collection_name}")
                await db.create_collection(collection_name)
                logger.info(f"Collection {collection_name} created successfully")
            else:
                logger.info(f"Collection {collection_name} already exists")
        except Exception as e:
            logger.warning(f"Could not create collection {collection_name}: {e}")

        try:
            await create_indexes_func(collection, collection_name, recreate)
            logger.info(f"Successfully created indexes for {collection_name}")
        except Exception as e:
            logger.error(f"Failed to create indexes for {collection_name}: {e}", exc_info=True)
            # Don't raise - continue with other collections
            continue

    # Load default admin scope after scopes collection is initialized
    logger.info("Loading default admin scope...")
    await _load_default_scopes(db, namespace, entra_group_id)


async def main():
    """Main initialization function."""
    parser = argparse.ArgumentParser(
        description="Initialize DocumentDB collections and indexes for MCP Gateway Registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Using environment variables
    export DOCUMENTDB_HOST=your-cluster.docdb.amazonaws.com
    uv run python scripts/init-documentdb-indexes.py

    # Using command-line arguments
    uv run python scripts/init-documentdb-indexes.py --host your-cluster.docdb.amazonaws.com

    # With IAM authentication
    uv run python scripts/init-documentdb-indexes.py --use-iam --host your-cluster.docdb.amazonaws.com

    # With namespace
    uv run python scripts/init-documentdb-indexes.py --namespace tenant-a
""",
    )

    parser.add_argument(
        "--host",
        default=os.getenv("DOCUMENTDB_HOST", "localhost"),
        help="DocumentDB host (default: from DOCUMENTDB_HOST env var or 'localhost')",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("DOCUMENTDB_PORT", "27017")),
        help="DocumentDB port (default: from DOCUMENTDB_PORT env var or 27017)",
    )
    parser.add_argument(
        "--database",
        default=os.getenv("DOCUMENTDB_DATABASE", "mcp_registry"),
        help="Database name (default: from DOCUMENTDB_DATABASE env var or 'mcp_registry')",
    )
    parser.add_argument(
        "--username",
        default=os.getenv("DOCUMENTDB_USERNAME"),
        help="DocumentDB username (default: from DOCUMENTDB_USERNAME env var)",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("DOCUMENTDB_PASSWORD"),
        help="DocumentDB password (default: from DOCUMENTDB_PASSWORD env var)",
    )
    parser.add_argument(
        "--use-iam",
        action="store_true",
        default=os.getenv("DOCUMENTDB_USE_IAM", "false").lower() == "true",
        help="Use AWS IAM authentication (default: from DOCUMENTDB_USE_IAM env var or false)",
    )
    parser.add_argument(
        "--use-tls",
        action="store_true",
        default=os.getenv("DOCUMENTDB_USE_TLS", "true").lower() == "true",
        help="Use TLS for connection (default: from DOCUMENTDB_USE_TLS env var or true)",
    )
    parser.add_argument(
        "--tls-ca-file",
        default=os.getenv("DOCUMENTDB_TLS_CA_FILE", "global-bundle.pem"),
        help="TLS CA file path (default: from DOCUMENTDB_TLS_CA_FILE env var or 'global-bundle.pem')",
    )
    parser.add_argument(
        "--namespace",
        default=os.getenv("DOCUMENTDB_NAMESPACE", "default"),
        help="Namespace for collection names (default: from DOCUMENTDB_NAMESPACE env var or 'default')",
    )
    parser.add_argument(
        "--storage-backend",
        default=os.getenv("STORAGE_BACKEND", "documentdb"),
        choices=["documentdb", "mongodb-ce"],
        help="Storage backend type: 'documentdb' (uses SCRAM-SHA-1) or 'mongodb-ce' (uses SCRAM-SHA-256) (default: from STORAGE_BACKEND env var or 'documentdb')",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        default=True,
        help="Drop and recreate indexes if they exist (default: True)",
    )
    parser.add_argument(
        "--no-recreate",
        dest="recreate",
        action="store_false",
        help="Do not recreate existing indexes",
    )
    parser.add_argument(
        "--entra-group-id",
        default=os.getenv("ENTRA_ADMIN_GROUP_ID"),
        help=(
            "Entra ID Group Object ID for the admin group. Required when using "
            "Microsoft Entra ID as the auth provider. Get this from: Azure Portal -> "
            "Groups -> [group name] -> Object Id (default: from ENTRA_ADMIN_GROUP_ID env var)"
        ),
    )

    args = parser.parse_args()

    logger.info("Initializing DocumentDB collections and indexes")
    logger.info(f"Host: {args.host}:{args.port}")
    logger.info(f"Database: {args.database}")
    logger.info(f"Namespace: {args.namespace}")
    logger.info(f"Storage backend: {args.storage_backend}")
    logger.info(f"Recreate indexes: {args.recreate}")
    logger.info(f"Use IAM: {args.use_iam}")
    logger.info(f"Use TLS: {args.use_tls}")
    logger.info(f"Entra Group ID: {args.entra_group_id or '<not set>'}")

    try:
        connection_string = await _get_documentdb_connection_string(
            host=args.host,
            port=args.port,
            database=args.database,
            username=args.username,
            password=args.password,
            use_iam=args.use_iam,
            use_tls=args.use_tls,
            tls_ca_file=args.tls_ca_file if args.use_tls else None,
            storage_backend=args.storage_backend,
        )

        # IMPORTANT: DocumentDB does not support retryable writes
        client = AsyncIOMotorClient(connection_string, retryWrites=False)
        db = client[args.database]

        server_info = await client.server_info()
        logger.info(f"Connected to DocumentDB/MongoDB {server_info.get('version', 'unknown')}")

        await _initialize_collections(
            db,
            args.namespace,
            args.recreate,
            args.entra_group_id,
        )

        logger.info(f"DocumentDB initialization complete for namespace '{args.namespace}'")

        # Print summary of collections and indexes
        await _print_collection_summary(db, args.namespace)

        client.close()

    except Exception as e:
        logger.error(f"Failed to initialize DocumentDB: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
