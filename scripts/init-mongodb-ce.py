#!/usr/bin/env python3
"""
Initialize MongoDB CE for local development.

This script:
1. Initializes replica set (rs0)
2. Creates collections and indexes
3. Loads scopes from scopes.yml

Usage:
    python init-mongodb-ce.py
"""

import asyncio
import logging
import os
import sys
import time
import yaml
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING
from pymongo.errors import ServerSelectionTimeoutError, OperationFailure


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


def _get_config_from_env() -> dict:
    """Get MongoDB CE configuration from environment variables."""
    return {
        "host": os.getenv("DOCUMENTDB_HOST", "mongodb"),
        "port": int(os.getenv("DOCUMENTDB_PORT", "27017")),
        "database": os.getenv("DOCUMENTDB_DATABASE", "mcp_registry"),
        "namespace": os.getenv("DOCUMENTDB_NAMESPACE", "default"),
        "scopes_file": os.getenv("SCOPES_FILE", "/app/config/scopes.yml"),
        "username": os.getenv("DOCUMENTDB_USERNAME", ""),
        "password": os.getenv("DOCUMENTDB_PASSWORD", ""),
        "replicaset": os.getenv("DOCUMENTDB_REPLICA_SET", "rs0"),
    }


def _initialize_replica_set(
    host: str,
    port: int,
    username: str,
    password: str,
) -> None:
    """Initialize MongoDB replica set using pymongo (synchronous)."""
    from pymongo import MongoClient

    logger.info("Initializing MongoDB replica set...")

    try:
        # Connect without replica set for initialization
        client = MongoClient(
            f"mongodb://{username}:{password}@{host}:{port}/?authMechanism=SCRAM-SHA-256&authSource=admin",
            serverSelectionTimeoutMS=5000,
            directConnection=True,
        )

        # Check if already initialized
        try:
            status = client.admin.command("replSetGetStatus")
            logger.info("Replica set already initialized")
            client.close()
            return
        except OperationFailure as e:
            if "no replset config has been received" in str(e).lower():
                # Not initialized, proceed
                pass
            else:
                raise

        # Initialize replica set
        config = {
            "_id": "rs0",
            "members": [
                {"_id": 0, "host": f"{host}:{port}"}
            ]
        }

        result = client.admin.command("replSetInitiate", config)
        logger.info(f"Replica set initialized: {result}")
        client.close()

        # Wait for replica set to elect primary
        logger.info("Waiting for replica set to elect primary...")
        time.sleep(10)

    except Exception as e:
        logger.error(f"Error initializing replica set: {e}")
        raise


async def _create_standard_indexes(
    collection,
    collection_name: str,
    namespace: str,
) -> None:
    """Create standard indexes for collections."""
    full_name = f"{collection_name}_{namespace}"

    if collection_name == COLLECTION_SERVERS:
        # Note: path is stored as _id, so no separate path index needed
        await collection.create_index([("enabled", ASCENDING)])
        await collection.create_index([("tags", ASCENDING)])
        await collection.create_index([("manifest.serverInfo.name", ASCENDING)])
        logger.info(f"Created indexes for {full_name}")

    elif collection_name == COLLECTION_AGENTS:
        # Note: path is stored as _id, so no separate path index needed
        await collection.create_index([("enabled", ASCENDING)])
        await collection.create_index([("tags", ASCENDING)])
        await collection.create_index([("card.name", ASCENDING)])
        logger.info(f"Created indexes for {full_name}")

    elif collection_name == COLLECTION_SCOPES:
        # No additional indexes needed - scopes use _id as primary key
        # group_mappings is an array, not indexed
        logger.info(f"Created indexes for {full_name}")

    elif collection_name == COLLECTION_EMBEDDINGS:
        # Note: path is stored as _id, so no separate path index needed
        await collection.create_index([("entity_type", ASCENDING)])
        logger.info(f"Created indexes for {full_name} (vector search via app code)")

    elif collection_name == COLLECTION_SECURITY_SCANS:
        await collection.create_index([("server_path", ASCENDING)])
        await collection.create_index([("scan_status", ASCENDING)])
        await collection.create_index([("scanned_at", ASCENDING)])
        logger.info(f"Created indexes for {full_name}")

    elif collection_name == COLLECTION_FEDERATION_CONFIG:
        await collection.create_index([("registry_name", ASCENDING)], unique=True)
        await collection.create_index([("enabled", ASCENDING)])
        logger.info(f"Created indexes for {full_name}")


async def _load_scopes_from_yaml(
    db,
    namespace: str,
    scopes_file: str,
) -> None:
    """Load scopes from YAML file into MongoDB.

    Uses same logic as load-scopes.py to parse scopes.yml structure:
    - group_mappings: Keycloak group to scope name mappings
    - UI-Scopes: UI permissions for each scope
    - Individual scope entries: Server access lists for each scope
    """
    logger.info(f"Loading scopes from {scopes_file}")

    # Check if file exists
    if not os.path.exists(scopes_file):
        logger.warning(f"Scopes file not found: {scopes_file}")
        logger.warning("Scopes will not be loaded. You can load them later using load-scopes.py")
        return

    # Read YAML file
    with open(scopes_file, "r") as f:
        scopes_data = yaml.safe_load(f)

    if not scopes_data:
        logger.warning(f"No scopes data found in {scopes_file}")
        return

    collection_name = f"{COLLECTION_SCOPES}_{namespace}"
    collection = db[collection_name]

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

    # Insert scopes into MongoDB
    if scope_groups:
        logger.info(f"Inserting {len(scope_groups)} scope groups into {collection_name}")

        # Clear existing scopes first
        await collection.delete_many({})

        for scope_doc in scope_groups:
            try:
                # Use update_one with upsert to avoid duplicate key errors
                result = await collection.update_one(
                    {"_id": scope_doc["_id"]},
                    {"$set": scope_doc},
                    upsert=True
                )

                if result.upserted_id:
                    logger.info(f"Inserted scope: {scope_doc['_id']}")
                elif result.modified_count > 0:
                    logger.info(f"Updated scope: {scope_doc['_id']}")

            except Exception as e:
                logger.error(f"Failed to insert scope {scope_doc['_id']}: {e}")

        logger.info(f"Successfully loaded {len(scope_groups)} scopes")
    else:
        logger.warning("No scope groups found to insert")


async def _initialize_mongodb_ce() -> None:
    """Main initialization function."""
    config = _get_config_from_env()

    logger.info("=" * 60)
    logger.info("MongoDB CE Initialization for MCP Gateway")
    logger.info("=" * 60)
    logger.info(f"Host: {config['host']}:{config['port']}")
    logger.info(f"Database: {config['database']}")
    logger.info(f"Namespace: {config['namespace']}")
    logger.info(f"Scopes file: {config['scopes_file']}")
    logger.info("")

    # Wait for MongoDB to be ready
    logger.info("Waiting for MongoDB to be ready...")
    time.sleep(10)

    # Initialize replica set (synchronous)
    _initialize_replica_set(config["host"], config["port"], config["username"], config["password"])

    # Connect with motor for async operations
    connection_string = f"mongodb://{config['username']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}?replicaSet={config['replicaset']}&authMechanism=SCRAM-SHA-256&authSource=admin"
    try:
        client = AsyncIOMotorClient(
            connection_string,
            serverSelectionTimeoutMS=10000,
        )

        # Verify connection
        await client.admin.command("ping")
        logger.info("Connected to MongoDB successfully")

        db = client[config["database"]]
        namespace = config["namespace"]

        # Create collections and indexes
        logger.info("Creating collections and indexes...")

        collections = [
            COLLECTION_SERVERS,
            COLLECTION_AGENTS,
            COLLECTION_SCOPES,
            COLLECTION_EMBEDDINGS,
            COLLECTION_SECURITY_SCANS,
            COLLECTION_FEDERATION_CONFIG,
        ]

        for coll_name in collections:
            full_name = f"{coll_name}_{namespace}"

            # Check if collection already exists
            existing_collections = await db.list_collection_names()

            if full_name in existing_collections:
                logger.info(f"Collection {full_name} already exists, skipping creation")
            else:
                logger.info(f"Creating collection: {full_name}")
                await db.create_collection(full_name)

            # Create indexes (idempotent - MongoDB handles duplicates)
            collection = db[full_name]
            await _create_standard_indexes(collection, coll_name, namespace)

        # Load scopes
        await _load_scopes_from_yaml(db, namespace, config["scopes_file"])

        logger.info("")
        logger.info("=" * 60)
        logger.info("MongoDB CE Initialization Complete!")
        logger.info("=" * 60)
        logger.info("Collections created:")
        for coll_name in collections:
            if coll_name == COLLECTION_EMBEDDINGS:
                logger.info(f"  - {coll_name}_{namespace} (with vector search)")
            else:
                logger.info(f"  - {coll_name}_{namespace}")
        logger.info("")
        logger.info("To use MongoDB CE:")
        logger.info("  export STORAGE_BACKEND=mongodb-ce")
        logger.info("  docker-compose up registry")
        logger.info("")
        logger.info("Or for AWS DocumentDB:")
        logger.info("  export STORAGE_BACKEND=documentdb")
        logger.info("  docker-compose up registry")
        logger.info("=" * 60)

        client.close()

    except ServerSelectionTimeoutError as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        logger.error("Make sure MongoDB is running and accessible")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        raise


def main() -> None:
    """Entry point."""
    asyncio.run(_initialize_mongodb_ce())


if __name__ == "__main__":
    main()
