#!/usr/bin/env python3
"""
Load scopes from YAML file into DocumentDB.

This script reads scopes.yml and loads the scope definitions into the
DocumentDB scopes collection.

Usage:
    python load-scopes.py --scopes-file /app/config/scopes.yml
"""

import argparse
import asyncio
import logging
import os
import yaml
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient


# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


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
            connection_string = f"mongodb://{host}:{port}/{database}?tls={str(use_tls).lower()}"

            if use_tls and tls_ca_file:
                connection_string += f"&tlsCAFile={tls_ca_file}"

            logger.info(f"Using no authentication for DocumentDB (host: {host})")

    return connection_string


async def load_scopes_from_yaml(
    scopes_file: str,
    db,
    namespace: str,
    clear_existing: bool = False,
) -> None:
    """Load scopes from YAML file into DocumentDB."""
    logger.info(f"Loading scopes from {scopes_file}")

    # Debug: Check if file exists
    import os

    logger.info(f"DEBUG: Current working directory: {os.getcwd()}")
    logger.info(f"DEBUG: File exists check: {os.path.exists(scopes_file)}")
    logger.info(f"DEBUG: File is absolute path: {os.path.isabs(scopes_file)}")
    if os.path.exists("/app/auth_server"):
        logger.info(f"DEBUG: /app/auth_server exists, contents: {os.listdir('/app/auth_server')}")
    else:
        logger.info(f"DEBUG: /app/auth_server does NOT exist")

    # Read YAML file
    with open(scopes_file, "r") as f:
        scopes_data = yaml.safe_load(f)

    if not scopes_data:
        logger.error("Scopes file is empty or invalid")
        return

    collection_name = f"mcp_scopes_{namespace}"
    collection = db[collection_name]

    # Clear existing scopes if requested
    if clear_existing:
        logger.info(f"Clearing existing scopes from {collection_name}")
        result = await collection.delete_many({})
        logger.info(f"Deleted {result.deleted_count} existing scope documents")

    # Extract group mappings and UI scopes
    group_mappings = scopes_data.get("group_mappings", {})
    ui_scopes = scopes_data.get("UI-Scopes", {})

    # Process each scope group
    scope_groups = []
    for key, value in scopes_data.items():
        # Skip the top-level keys
        if key in ["group_mappings", "UI-Scopes"]:
            continue

        # This is a scope group
        scope_name = key
        server_access = value if isinstance(value, list) else []

        # Build the scope document
        scope_doc = {
            "_id": scope_name,
            "group_mappings": [],
            "server_access": server_access,
            "ui_permissions": {},
        }

        # Add group mappings for this scope
        for keycloak_group, scope_names in group_mappings.items():
            if scope_name in scope_names:
                scope_doc["group_mappings"].append(keycloak_group)

        # Add UI permissions for this scope
        if scope_name in ui_scopes:
            scope_doc["ui_permissions"] = ui_scopes[scope_name]

        scope_groups.append(scope_doc)

    # Insert scopes into DocumentDB
    if scope_groups:
        logger.info(f"Inserting {len(scope_groups)} scope groups into {collection_name}")

        for scope_doc in scope_groups:
            try:
                # Use update_one with upsert to avoid duplicate key errors
                result = await collection.update_one(
                    {"_id": scope_doc["_id"]}, {"$set": scope_doc}, upsert=True
                )

                if result.upserted_id:
                    logger.info(f"Inserted scope: {scope_doc['_id']}")
                elif result.modified_count > 0:
                    logger.info(f"Updated scope: {scope_doc['_id']}")
                else:
                    logger.debug(f"No changes for scope: {scope_doc['_id']}")

            except Exception as e:
                logger.error(f"Failed to insert scope {scope_doc['_id']}: {e}")

        logger.info(f"Successfully loaded {len(scope_groups)} scopes")

        # Print summary
        logger.info("=" * 80)
        logger.info("SCOPES SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total scopes loaded: {len(scope_groups)}")
        logger.info("\nScope groups:")
        for scope_doc in scope_groups:
            logger.info(f"  - {scope_doc['_id']}")
            logger.info(f"      Keycloak groups: {scope_doc['group_mappings']}")
            logger.info(f"      Server access rules: {len(scope_doc['server_access'])} rules")
            logger.info(f"      UI permissions: {len(scope_doc['ui_permissions'])} permissions")
        logger.info("=" * 80)
    else:
        logger.warning("No scope groups found in YAML file")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Load scopes from YAML file into DocumentDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Using environment variables
    export DOCUMENTDB_HOST=your-cluster.docdb.amazonaws.com
    python load-scopes.py --scopes-file /app/config/scopes.yml

    # Clear existing scopes before loading
    python load-scopes.py --scopes-file /app/config/scopes.yml --clear-existing
""",
    )

    parser.add_argument(
        "--scopes-file",
        required=True,
        help="Path to scopes YAML file",
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
        "--clear-existing",
        action="store_true",
        help="Clear existing scopes before loading new ones",
    )

    args = parser.parse_args()

    # Get storage backend from environment variable
    storage_backend = os.getenv("STORAGE_BACKEND", "documentdb")

    logger.info("Loading scopes into DocumentDB")
    logger.info(f"Host: {args.host}:{args.port}")
    logger.info(f"Database: {args.database}")
    logger.info(f"Namespace: {args.namespace}")
    logger.info(f"Storage backend: {storage_backend}")
    logger.info(f"Scopes file: {args.scopes_file}")
    logger.info(f"Clear existing: {args.clear_existing}")

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
            storage_backend=storage_backend,
        )

        # IMPORTANT: DocumentDB does not support retryable writes
        client = AsyncIOMotorClient(connection_string, retryWrites=False)
        db = client[args.database]

        server_info = await client.server_info()
        logger.info(f"Connected to DocumentDB/MongoDB {server_info.get('version', 'unknown')}")

        await load_scopes_from_yaml(
            scopes_file=args.scopes_file,
            db=db,
            namespace=args.namespace,
            clear_existing=args.clear_existing,
        )

        logger.info("Scopes loading complete")

        client.close()

    except Exception as e:
        logger.error(f"Failed to load scopes: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
