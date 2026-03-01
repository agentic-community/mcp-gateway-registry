#!/usr/bin/env python3
"""
Migration script to add is_active field to existing servers.

This script ensures all existing servers have the is_active field set to True,
which is required for the server version routing feature. Existing servers
without this field are treated as active (default behavior).

Usage:
    # Dry run (default) - show what would be updated
    uv run python scripts/migrate-servers-add-is-active.py

    # Actually apply changes
    uv run python scripts/migrate-servers-add-is-active.py --apply

    # With specific DocumentDB settings
    uv run python scripts/migrate-servers-add-is-active.py --host your-cluster.docdb.amazonaws.com

    # Using file-based storage
    uv run python scripts/migrate-servers-add-is-active.py --storage file --servers-dir /path/to/servers

Requires:
    - motor (AsyncIOMotorClient) for DocumentDB
    - boto3 (for IAM authentication if using DocumentDB)
"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
)


# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


# Constants
SERVERS_COLLECTION = "servers"


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Migrate servers to add is_active field",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run (default) - show what would be updated
    uv run python scripts/migrate-servers-add-is-active.py

    # Actually apply changes
    uv run python scripts/migrate-servers-add-is-active.py --apply

    # With DocumentDB
    uv run python scripts/migrate-servers-add-is-active.py --host your-cluster.docdb.amazonaws.com

    # Using file-based storage
    uv run python scripts/migrate-servers-add-is-active.py --storage file --servers-dir ./data/servers
""",
    )

    parser.add_argument(
        "--apply", action="store_true", help="Actually apply changes (default is dry run)"
    )

    parser.add_argument(
        "--storage",
        type=str,
        choices=["documentdb", "mongodb-ce", "file"],
        default=os.getenv("MCP_STORAGE_BACKEND", "documentdb"),
        help="Storage backend type (default: from MCP_STORAGE_BACKEND env or documentdb)",
    )

    parser.add_argument(
        "--host", type=str, default=os.getenv("DOCUMENTDB_HOST"), help="DocumentDB/MongoDB host"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("DOCUMENTDB_PORT", "27017")),
        help="DocumentDB/MongoDB port (default: 27017)",
    )

    parser.add_argument(
        "--database",
        type=str,
        default=os.getenv("DOCUMENTDB_DATABASE", "mcp_registry"),
        help="Database name (default: mcp_registry)",
    )

    parser.add_argument(
        "--namespace",
        type=str,
        default=os.getenv("DOCUMENTDB_NAMESPACE"),
        help="Namespace prefix for collections",
    )

    parser.add_argument(
        "--servers-dir",
        type=str,
        default=os.getenv("MCP_SERVERS_DIR"),
        help="Directory for server JSON files (file storage)",
    )

    parser.add_argument(
        "--use-iam", action="store_true", help="Use IAM authentication for DocumentDB"
    )

    return parser.parse_args()


async def _migrate_documentdb(args: argparse.Namespace, dry_run: bool) -> Dict[str, Any]:
    """
    Migrate servers in DocumentDB to add is_active field.

    Args:
        args: Parsed command-line arguments
        dry_run: If True, only report what would be done

    Returns:
        Migration summary
    """
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
    except ImportError:
        logger.error("motor package required for DocumentDB migration")
        logger.error("Install with: uv add motor")
        return {"error": "motor not installed"}

    # Build connection string
    host = args.host
    port = args.port
    database = args.database

    if not host:
        logger.error("DocumentDB host required. Set via --host or DOCUMENTDB_HOST env var")
        return {"error": "host required"}

    if args.use_iam:
        try:
            import boto3

            session = boto3.Session()
            credentials = session.get_credentials()
            token = session.client("rds").generate_db_auth_token(
                DBHostname=host, Port=port, DBUsername="admin", Region=session.region_name
            )
            connection_string = f"mongodb://admin:{token}@{host}:{port}/?authMechanism=MONGODB-AWS&authSource=$external&tls=true&tlsCAFile=global-bundle.pem"
        except Exception as e:
            logger.error(f"Failed to get IAM credentials: {e}")
            return {"error": str(e)}
    else:
        username = os.getenv("DOCUMENTDB_USERNAME")
        password = os.getenv("DOCUMENTDB_PASSWORD")
        if username and password:
            connection_string = f"mongodb://{username}:{password}@{host}:{port}/"
        else:
            connection_string = f"mongodb://{host}:{port}/"

    # Handle MongoDB CE with directConnection
    if args.storage == "mongodb-ce":
        connection_string += "?directConnection=true"

    logger.info(f"Connecting to {args.storage} at {host}:{port}")

    client = AsyncIOMotorClient(connection_string)
    db = client[database]

    # Get collection name with namespace
    collection_name = SERVERS_COLLECTION
    if args.namespace:
        collection_name = f"{args.namespace}_{SERVERS_COLLECTION}"

    collection = db[collection_name]

    # Find servers without is_active field
    query = {"is_active": {"$exists": False}}
    servers_to_update: List[Dict[str, Any]] = []

    async for server in collection.find(query):
        servers_to_update.append(
            {"_id": server["_id"], "server_name": server.get("server_name", "unknown")}
        )

    logger.info(f"Found {len(servers_to_update)} servers without is_active field")

    if dry_run:
        logger.info("DRY RUN - No changes will be made")
        for server in servers_to_update:
            logger.info(f"  Would update: {server['_id']} ({server['server_name']})")
    else:
        if servers_to_update:
            result = await collection.update_many(query, {"$set": {"is_active": True}})
            logger.info(f"Updated {result.modified_count} servers with is_active=True")
        else:
            logger.info("No servers need updating")

    client.close()

    return {
        "storage": args.storage,
        "servers_found": len(servers_to_update),
        "servers_updated": 0 if dry_run else len(servers_to_update),
        "dry_run": dry_run,
    }


async def _migrate_file_storage(args: argparse.Namespace, dry_run: bool) -> Dict[str, Any]:
    """
    Migrate servers in file storage to add is_active field.

    Args:
        args: Parsed command-line arguments
        dry_run: If True, only report what would be done

    Returns:
        Migration summary
    """
    servers_dir = args.servers_dir
    if not servers_dir:
        servers_dir = os.getenv("MCP_SERVERS_DIR", "./data/servers")

    servers_path = Path(servers_dir)
    if not servers_path.exists():
        logger.error(f"Servers directory not found: {servers_path}")
        return {"error": f"directory not found: {servers_path}"}

    logger.info(f"Scanning servers directory: {servers_path}")

    servers_to_update: List[Dict[str, Any]] = []
    updated_count = 0

    for json_file in servers_path.glob("*.json"):
        if json_file.name == "_state.json":
            continue

        try:
            with open(json_file, "r") as f:
                server_data = json.load(f)

            # Check if is_active field is missing
            if "is_active" not in server_data:
                servers_to_update.append(
                    {
                        "file": str(json_file),
                        "server_name": server_data.get("server_name", "unknown"),
                        "path": server_data.get("path", "unknown"),
                    }
                )

                if not dry_run:
                    server_data["is_active"] = True
                    with open(json_file, "w") as f:
                        json.dump(server_data, f, indent=2)
                    updated_count += 1

        except json.JSONDecodeError as e:
            logger.warning(f"Skipping invalid JSON file {json_file}: {e}")
        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")

    logger.info(f"Found {len(servers_to_update)} servers without is_active field")

    if dry_run:
        logger.info("DRY RUN - No changes will be made")
        for server in servers_to_update:
            logger.info(f"  Would update: {server['file']} ({server['server_name']})")
    else:
        logger.info(f"Updated {updated_count} server files with is_active=True")

    return {
        "storage": "file",
        "servers_found": len(servers_to_update),
        "servers_updated": updated_count,
        "dry_run": dry_run,
    }


async def main() -> None:
    """Main entry point for the migration script."""
    args = _parse_args()
    dry_run = not args.apply

    logger.info("=" * 60)
    logger.info("Server Migration: Add is_active Field")
    logger.info("=" * 60)
    logger.info(f"Storage backend: {args.storage}")
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'APPLY CHANGES'}")
    logger.info("=" * 60)

    if args.storage in ["documentdb", "mongodb-ce"]:
        result = await _migrate_documentdb(args, dry_run)
    elif args.storage == "file":
        result = await _migrate_file_storage(args, dry_run)
    else:
        logger.error(f"Unknown storage backend: {args.storage}")
        result = {"error": f"unknown storage: {args.storage}"}

    logger.info("=" * 60)
    logger.info("Migration Summary:")
    logger.info(f"  Storage: {result.get('storage', 'unknown')}")
    logger.info(f"  Servers found: {result.get('servers_found', 0)}")
    logger.info(f"  Servers updated: {result.get('servers_updated', 0)}")
    if result.get("dry_run"):
        logger.info("  Note: This was a dry run. Use --apply to make changes.")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
