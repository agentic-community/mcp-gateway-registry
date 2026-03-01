#!/usr/bin/env python3
"""
Initialize MongoDB CE for local development.

This script:
1. Initializes replica set (rs0)
2. Creates collections and indexes
3. Loads default admin scope from registry-admins.json

Usage:
    python init-mongodb-ce.py
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
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
COLLECTION_AUDIT_EVENTS = "audit_events"
COLLECTION_SKILLS = "agent_skills"


def _get_config_from_env() -> dict:
    """Get MongoDB CE configuration from environment variables."""
    return {
        "host": os.getenv("DOCUMENTDB_HOST", "mongodb"),
        "port": int(os.getenv("DOCUMENTDB_PORT", "27017")),
        "database": os.getenv("DOCUMENTDB_DATABASE", "mcp_registry"),
        "namespace": os.getenv("DOCUMENTDB_NAMESPACE", "default"),
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
        # Use auth only if username is provided (MongoDB CE runs without auth by default)
        if username and password:
            connection_uri = f"mongodb://{username}:{password}@{host}:{port}/?authMechanism=SCRAM-SHA-256&authSource=admin"
        else:
            connection_uri = f"mongodb://{host}:{port}/"
            logger.info("Connecting without authentication (MongoDB CE no-auth mode)")

        client = MongoClient(
            connection_uri,
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
        config = {"_id": "rs0", "members": [{"_id": 0, "host": f"{host}:{port}"}]}

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

    elif collection_name == COLLECTION_AUDIT_EVENTS:
        # Indexes for audit event queries (Requirements 6.2)
        # Note: timestamp index is created as TTL index below, so we use compound indexes here
        await collection.create_index([("identity.username", ASCENDING), ("timestamp", ASCENDING)])
        await collection.create_index([("action.operation", ASCENDING), ("timestamp", ASCENDING)])
        await collection.create_index(
            [("action.resource_type", ASCENDING), ("timestamp", ASCENDING)]
        )

        # Index for MCP server name distinct/filter queries
        await collection.create_index([("mcp_server.name", ASCENDING)])

        # Migration: drop old single-field request_id index if it exists
        # Try both auto-generated name and explicit name variants
        for old_index_name in ("request_id_1", "request_id_idx"):
            try:
                await collection.drop_index(old_index_name)
                logger.info(f"Dropped old single-field index '{old_index_name}' from {full_name}")
            except Exception:
                logger.debug(f"No old index '{old_index_name}' to drop from {full_name}")

        # Composite unique index on (request_id, log_type)
        # Allows both MCPServerAccessRecord and RegistryApiAccessRecord
        # to coexist for the same request_id while preventing true duplicates
        await collection.create_index(
            [("request_id", ASCENDING), ("log_type", ASCENDING)],
            name="request_id_log_type_idx",
            unique=True,
        )

        # TTL index for automatic expiration (Requirements 6.3)
        # This also serves as the timestamp index for sorting
        # Default 7 days (604800 seconds), configurable via AUDIT_LOG_MONGODB_TTL_DAYS
        ttl_days = int(os.getenv("AUDIT_LOG_MONGODB_TTL_DAYS", "7"))
        ttl_seconds = ttl_days * 24 * 60 * 60
        try:
            await collection.create_index(
                [("timestamp", ASCENDING)], expireAfterSeconds=ttl_seconds, name="timestamp_ttl"
            )
        except OperationFailure as e:
            if e.code == 85:  # IndexOptionsConflict
                logger.info(f"TTL index options changed for {full_name}, recreating index...")
                await collection.drop_index("timestamp_ttl")
                await collection.create_index(
                    [("timestamp", ASCENDING)], expireAfterSeconds=ttl_seconds, name="timestamp_ttl"
                )
            else:
                raise
        logger.info(f"Created indexes for {full_name} (TTL: {ttl_days} days)")

    elif collection_name == COLLECTION_SKILLS:
        # Note: path is stored as _id, so no separate path index needed
        await collection.create_index([("name", ASCENDING)], unique=True)
        await collection.create_index([("tags", ASCENDING)])
        await collection.create_index([("visibility", ASCENDING)])
        await collection.create_index([("is_enabled", ASCENDING)])
        await collection.create_index([("registry_name", ASCENDING)])
        await collection.create_index([("owner", ASCENDING)])
        logger.info(f"Created indexes for {full_name}")


async def _load_default_scopes(
    db,
    namespace: str,
) -> None:
    """Load default scopes from JSON files into scopes collection.

    This loads all scope JSON files from the scripts directory:
    - registry-admins.json: Bootstrap admin scope with full permissions
    - mcp-registry-admin.json: MCP registry admin scope (Keycloak group)
    - mcp-servers-unrestricted-read.json: Read-only access to all servers
    - mcp-servers-unrestricted-execute.json: Full CRUD access to all servers
    """
    collection_name = f"{COLLECTION_SCOPES}_{namespace}"
    collection = db[collection_name]

    # Find scope files in the same directory as this script
    script_dir = Path(__file__).parent

    # List of scope files to load (order matters - base scopes first)
    scope_files = [
        "registry-admins.json",
        "mcp-registry-admin.json",
        "mcp-servers-unrestricted-read.json",
        "mcp-servers-unrestricted-execute.json",
    ]

    loaded_count = 0
    for scope_filename in scope_files:
        scope_file = script_dir / scope_filename

        if not scope_file.exists():
            logger.warning(f"Scope file not found: {scope_file}")
            continue

        try:
            with open(scope_file, "r") as f:
                scope_data = json.load(f)

            logger.info(f"Loading scope from {scope_filename}")

            # For registry-admins scope, add Entra admin group ID from env if configured
            if scope_data["_id"] == "registry-admins":
                entra_admin_group_id = os.getenv("ENTRA_GROUP_ADMIN_ID", "").strip()
                if entra_admin_group_id:
                    group_mappings = scope_data.get("group_mappings", [])
                    if entra_admin_group_id not in group_mappings:
                        group_mappings.append(entra_admin_group_id)
                        scope_data["group_mappings"] = group_mappings
                        logger.info(f"  Added Entra admin group ID: {entra_admin_group_id}")

            # Upsert the scope document
            result = await collection.update_one(
                {"_id": scope_data["_id"]}, {"$set": scope_data}, upsert=True
            )

            if result.upserted_id:
                logger.info(f"Inserted scope: {scope_data['_id']}")
                loaded_count += 1
            elif result.modified_count > 0:
                logger.info(f"Updated scope: {scope_data['_id']}")
                loaded_count += 1
            else:
                logger.info(f"Scope already up-to-date: {scope_data['_id']}")

            if "group_mappings" in scope_data:
                logger.info(f"  group_mappings: {scope_data.get('group_mappings', [])}")

        except Exception as e:
            logger.error(f"Failed to load scope from {scope_filename}: {e}", exc_info=True)

    logger.info(f"Loaded {loaded_count} scopes into {collection_name}")


async def _initialize_mongodb_ce() -> None:
    """Main initialization function."""
    config = _get_config_from_env()

    logger.info("=" * 60)
    logger.info("MongoDB CE Initialization for MCP Gateway")
    logger.info("=" * 60)
    logger.info(f"Host: {config['host']}:{config['port']}")
    logger.info(f"Database: {config['database']}")
    logger.info(f"Namespace: {config['namespace']}")
    logger.info("")

    # Wait for MongoDB to be ready
    logger.info("Waiting for MongoDB to be ready...")
    time.sleep(10)

    # Initialize replica set (synchronous)
    _initialize_replica_set(config["host"], config["port"], config["username"], config["password"])

    # Connect with motor for async operations
    # Use auth only if username is provided (MongoDB CE runs without auth by default)
    if config["username"] and config["password"]:
        connection_string = f"mongodb://{config['username']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}?replicaSet={config['replicaset']}&authMechanism=SCRAM-SHA-256&authSource=admin"
    else:
        connection_string = f"mongodb://{config['host']}:{config['port']}/{config['database']}?replicaSet={config['replicaset']}"
        logger.info("Using no-auth connection for async client")

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
            COLLECTION_AUDIT_EVENTS,
            COLLECTION_SKILLS,
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

        # Load default admin scope
        await _load_default_scopes(db, namespace)

        logger.info("")
        logger.info("=" * 60)
        logger.info("MongoDB CE Initialization Complete!")
        logger.info("=" * 60)
        logger.info("Collections created:")
        for coll_name in collections:
            if coll_name == COLLECTION_EMBEDDINGS:
                logger.info(f"  - {coll_name}_{namespace} (with vector search)")
            elif coll_name == COLLECTION_AUDIT_EVENTS:
                ttl_days = int(os.getenv("AUDIT_LOG_MONGODB_TTL_DAYS", "7"))
                logger.info(f"  - {coll_name}_{namespace} (TTL: {ttl_days} days)")
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
