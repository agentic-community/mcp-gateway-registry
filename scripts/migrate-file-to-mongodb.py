#!/usr/bin/env python3
"""
Migrate file-based storage to MongoDB.

This script reads server and agent JSON files from the file-based storage
and imports them into MongoDB.

Usage:
    # Run migration from host machine (connects to localhost:27017)
    python scripts/migrate-file-to-mongodb.py --servers-dir ~/mcp-gateway/servers --agents-dir ~/mcp-gateway/agents

    # Run with custom host/port
    python scripts/migrate-file-to-mongodb.py --host localhost --port 27017

    # Dry run to see what would be migrated
    python scripts/migrate-file-to-mongodb.py --dry-run
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorClient


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


def _get_config_from_env(
    host_override: Optional[str] = None,
    port_override: Optional[int] = None,
) -> dict:
    """Get MongoDB configuration from environment variables or overrides.

    Args:
        host_override: Override host (ignores DOCUMENTDB_HOST env var)
        port_override: Override port (ignores DOCUMENTDB_PORT env var)
    """
    return {
        "host": host_override or os.getenv("DOCUMENTDB_HOST", "localhost"),
        "port": port_override or int(os.getenv("DOCUMENTDB_PORT", "27017")),
        "database": os.getenv("DOCUMENTDB_DATABASE", "mcp_registry"),
        "namespace": os.getenv("DOCUMENTDB_NAMESPACE", "default"),
        "username": os.getenv("DOCUMENTDB_USERNAME", ""),
        "password": os.getenv("DOCUMENTDB_PASSWORD", ""),
        "replicaset": os.getenv("DOCUMENTDB_REPLICA_SET", "rs0"),
    }


async def _get_mongodb_client(
    config: dict,
    direct_connection: bool = True,
) -> AsyncIOMotorClient:
    """Create MongoDB async client.

    Args:
        config: MongoDB connection configuration
        direct_connection: Use directConnection=true for single-node replica sets
    """
    if config["username"] and config["password"]:
        connection_string = (
            f"mongodb://{config['username']}:{config['password']}@"
            f"{config['host']}:{config['port']}/{config['database']}?"
            f"authMechanism=SCRAM-SHA-256&authSource=admin"
        )
    else:
        connection_string = (
            f"mongodb://{config['host']}:{config['port']}/{config['database']}"
        )
        logger.info("Using no-auth connection for MongoDB")

    # Add directConnection for single-node replica set
    if direct_connection:
        separator = "&" if "?" in connection_string else "?"
        connection_string += f"{separator}directConnection=true"
        logger.info("Using directConnection=true for single-node MongoDB")

    client = AsyncIOMotorClient(
        connection_string,
        serverSelectionTimeoutMS=10000,
    )

    # Verify connection
    await client.admin.command("ping")
    logger.info(f"Connected to MongoDB at {config['host']}:{config['port']}")

    return client


def _load_server_json(filepath: Path) -> Optional[Dict[str, Any]]:
    """Load and transform a server JSON file."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        # Skip non-server files
        if "server_name" not in data and "path" not in data:
            logger.debug(f"Skipping {filepath.name} - not a server config")
            return None

        # Ensure path is set
        if "path" not in data:
            # Extract path from filename (e.g., currenttime.json -> /currenttime)
            stem = filepath.stem
            if stem.endswith("_"):
                stem = stem[:-1]
            data["path"] = f"/{stem}"

        # Normalize path
        path = data["path"]
        if not path.startswith("/"):
            path = f"/{path}"
        if path.endswith("/"):
            path = path[:-1]
        data["path"] = path

        # Add default fields if missing
        now = datetime.now(timezone.utc).isoformat()
        data.setdefault("is_enabled", True)
        data.setdefault("registered_at", now)
        data.setdefault("updated_at", now)

        logger.info(f"Loaded server: {data.get('server_name', 'unknown')} at {data['path']}")
        return data

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filepath}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None


def _load_agent_json(filepath: Path) -> Optional[Dict[str, Any]]:
    """Load and transform an agent JSON file."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        # Check for agent card structure
        if "card" in data:
            # Agent with card wrapper
            card = data.get("card", {})
            agent_data = {
                "card": card,
                "path": data.get("path") or f"/agents/{card.get('name', filepath.stem)}",
                "is_enabled": data.get("is_enabled", True),
                "registered_at": data.get("registered_at", datetime.now(timezone.utc).isoformat()),
                "updated_at": data.get("updated_at", datetime.now(timezone.utc).isoformat()),
            }
        elif "name" in data:
            # Flat agent structure
            agent_name = data.get("name", filepath.stem)
            agent_data = {
                "card": data,
                "path": f"/agents/{agent_name}",
                "is_enabled": data.get("is_enabled", True),
                "registered_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        else:
            logger.debug(f"Skipping {filepath.name} - not an agent config")
            return None

        # Normalize path
        path = agent_data["path"]
        if not path.startswith("/"):
            path = f"/{path}"
        agent_data["path"] = path

        logger.info(f"Loaded agent: {agent_data.get('card', {}).get('name', 'unknown')} at {agent_data['path']}")
        return agent_data

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filepath}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None


async def _migrate_servers(
    db,
    servers_dir: Path,
    namespace: str,
    dry_run: bool = False,
) -> int:
    """Migrate servers from file storage to MongoDB."""
    collection_name = f"mcp_servers_{namespace}"
    collection = db[collection_name]

    # Find all JSON files (exclude non-server files)
    exclude_files = {"server_state.json", "service_index_metadata.json"}
    json_files = [
        f for f in servers_dir.glob("*.json")
        if f.name not in exclude_files and not f.name.endswith(".faiss")
    ]

    if not json_files:
        logger.warning(f"No server JSON files found in {servers_dir}")
        return 0

    logger.info(f"Found {len(json_files)} potential server files")

    imported = 0
    skipped = 0

    for filepath in json_files:
        server_data = _load_server_json(filepath)
        if not server_data:
            skipped += 1
            continue

        path = server_data["path"]

        if dry_run:
            logger.info(f"[DRY RUN] Would import server: {server_data.get('server_name')} at {path}")
            imported += 1
            continue

        # Check if server already exists
        existing = await collection.find_one({"_id": path})
        if existing:
            logger.info(f"Server already exists at {path}, updating...")
            # Update existing document
            doc = {**server_data}
            doc.pop("path", None)
            doc["updated_at"] = datetime.now(timezone.utc).isoformat()
            await collection.update_one({"_id": path}, {"$set": doc})
        else:
            # Create new document
            doc = {**server_data}
            doc["_id"] = doc.pop("path")
            await collection.insert_one(doc)

        imported += 1

    logger.info(f"Servers: imported={imported}, skipped={skipped}")
    return imported


async def _migrate_agents(
    db,
    agents_dir: Path,
    namespace: str,
    dry_run: bool = False,
) -> int:
    """Migrate agents from file storage to MongoDB."""
    collection_name = f"mcp_agents_{namespace}"
    collection = db[collection_name]

    # Find all JSON files
    json_files = list(agents_dir.glob("*.json"))

    if not json_files:
        logger.warning(f"No agent JSON files found in {agents_dir}")
        return 0

    logger.info(f"Found {len(json_files)} potential agent files")

    imported = 0
    skipped = 0

    for filepath in json_files:
        agent_data = _load_agent_json(filepath)
        if not agent_data:
            skipped += 1
            continue

        path = agent_data["path"]

        if dry_run:
            logger.info(f"[DRY RUN] Would import agent at {path}")
            imported += 1
            continue

        # Check if agent already exists
        existing = await collection.find_one({"_id": path})
        if existing:
            logger.info(f"Agent already exists at {path}, updating...")
            doc = {**agent_data}
            doc.pop("path", None)
            doc["updated_at"] = datetime.now(timezone.utc).isoformat()
            await collection.update_one({"_id": path}, {"$set": doc})
        else:
            # Create new document
            doc = {**agent_data}
            doc["_id"] = doc.pop("path")
            await collection.insert_one(doc)

        imported += 1

    logger.info(f"Agents: imported={imported}, skipped={skipped}")
    return imported


async def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(
        description="Migrate file-based storage to MongoDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--servers-dir",
        type=Path,
        default=Path.home() / "mcp-gateway" / "servers",
        help="Directory containing server JSON files",
    )
    parser.add_argument(
        "--agents-dir",
        type=Path,
        default=Path.home() / "mcp-gateway" / "agents",
        help="Directory containing agent JSON files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes",
    )
    parser.add_argument(
        "--servers-only",
        action="store_true",
        help="Only migrate servers",
    )
    parser.add_argument(
        "--agents-only",
        action="store_true",
        help="Only migrate agents",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="MongoDB host (default: localhost, overrides DOCUMENTDB_HOST env var)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=27017,
        help="MongoDB port (default: 27017, overrides DOCUMENTDB_PORT env var)",
    )

    args = parser.parse_args()

    config = _get_config_from_env(
        host_override=args.host,
        port_override=args.port,
    )

    logger.info("=" * 60)
    logger.info("File to MongoDB Migration")
    logger.info("=" * 60)
    logger.info(f"MongoDB: {config['host']}:{config['port']}/{config['database']}")
    logger.info(f"Namespace: {config['namespace']}")
    logger.info(f"Servers dir: {args.servers_dir}")
    logger.info(f"Agents dir: {args.agents_dir}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("")

    try:
        client = await _get_mongodb_client(config)
        db = client[config["database"]]

        total_imported = 0

        if not args.agents_only:
            if args.servers_dir.exists():
                count = await _migrate_servers(
                    db, args.servers_dir, config["namespace"], args.dry_run
                )
                total_imported += count
            else:
                logger.warning(f"Servers directory not found: {args.servers_dir}")

        if not args.servers_only:
            if args.agents_dir.exists():
                count = await _migrate_agents(
                    db, args.agents_dir, config["namespace"], args.dry_run
                )
                total_imported += count
            else:
                logger.warning(f"Agents directory not found: {args.agents_dir}")

        logger.info("")
        logger.info("=" * 60)
        if args.dry_run:
            logger.info(f"DRY RUN complete. Would import {total_imported} items.")
        else:
            logger.info(f"Migration complete. Imported {total_imported} items.")
        logger.info("=" * 60)

        client.close()
        return 0

    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
