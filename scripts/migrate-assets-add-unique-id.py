#!/usr/bin/env python3
"""
Migration script to add a unique ``id`` index to asset collections (#1276).

Prepares the servers, agents, and skills collections for caller-supplied
asset ids by:
  1. Detecting pre-existing duplicate ``id`` values (a unique index cannot be
     built while duplicates exist).
  2. Backfilling a uuid4 ``id`` onto any legacy document missing a non-empty id.
  3. Building the unique partial index ``id_idx`` on ``id``.

Ordering matters: backfill runs BEFORE the index build so the build never
fails on legacy rows. Duplicate detection runs FIRST; with --apply, the script
refuses (exits non-zero) if any collection has duplicate ids, so a partial
unique-index build never fails mid-deploy. Run this before serving traffic so
the lazy _get_collection() path finds id_idx already present.

Usage:
    # Dry run (default) - report duplicates, missing ids, and planned index
    uv run python scripts/migrate-assets-add-unique-id.py

    # Actually apply (backfill + build index); refuses if duplicates exist
    uv run python scripts/migrate-assets-add-unique-id.py --apply

    # With specific DocumentDB settings
    uv run python scripts/migrate-assets-add-unique-id.py --host your-cluster.docdb.amazonaws.com

Requires:
    - motor (AsyncIOMotorClient)
    - boto3 (only for --use-iam)
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Any
from uuid import uuid4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

# Base collection names (namespace prefix applied at runtime).
ASSET_COLLECTIONS = ["servers", "agents", "skills"]
ID_INDEX_NAME = "id_idx"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add a unique id index to asset collections (#1276)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--apply", action="store_true", help="Actually apply changes (default is dry run)"
    )
    parser.add_argument(
        "--storage",
        type=str,
        choices=["documentdb", "mongodb-ce"],
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
        "--use-iam", action="store_true", help="Use IAM authentication for DocumentDB"
    )
    return parser.parse_args()


def _build_connection_string(args: argparse.Namespace) -> str | None:
    host, port = args.host, args.port
    override = os.getenv("MONGODB_CONNECTION_STRING", "")
    if override:
        logger.info("Using MONGODB_CONNECTION_STRING override")
        return override
    if not host:
        logger.error(
            "DocumentDB host required. Set via --host, DOCUMENTDB_HOST, "
            "or MONGODB_CONNECTION_STRING env var"
        )
        return None
    if args.use_iam:
        try:
            import boto3

            session = boto3.Session()
            token = session.client("rds").generate_db_auth_token(
                DBHostname=host, Port=port, DBUsername="admin", Region=session.region_name
            )
            cs = (
                f"mongodb://admin:{token}@{host}:{port}/"
                "?authMechanism=MONGODB-AWS&authSource=$external"
                "&tls=true&tlsCAFile=global-bundle.pem"
            )
        except Exception as e:
            logger.error(f"Failed to get IAM credentials: {e}")
            return None
    else:
        username = os.getenv("DOCUMENTDB_USERNAME")
        password = os.getenv("DOCUMENTDB_PASSWORD")
        if username and password:
            cs = f"mongodb://{username}:{password}@{host}:{port}/"
        else:
            cs = f"mongodb://{host}:{port}/"
    if args.storage == "mongodb-ce":
        cs += "?directConnection=true"
    return cs


async def _find_duplicate_ids(collection: Any) -> list[dict[str, Any]]:
    """Return [{id, count}] for any id shared by more than one document."""
    pipeline = [
        {"$match": {"id": {"$exists": True, "$nin": [None, ""]}}},
        {"$group": {"_id": "$id", "n": {"$sum": 1}}},
        {"$match": {"n": {"$gt": 1}}},
    ]
    dupes: list[dict[str, Any]] = []
    async for doc in collection.aggregate(pipeline):
        dupes.append({"id": doc["_id"], "count": doc["n"]})
    return dupes


async def _count_missing_ids(collection: Any) -> int:
    return await collection.count_documents(
        {"$or": [{"id": {"$exists": False}}, {"id": None}, {"id": ""}]}
    )


async def _backfill_missing_ids(collection: Any) -> int:
    query = {"$or": [{"id": {"$exists": False}}, {"id": None}, {"id": ""}]}
    updated = 0
    async for doc in collection.find(query, projection={"_id": 1}):
        await collection.update_one({"_id": doc["_id"]}, {"$set": {"id": str(uuid4())}})
        updated += 1
    return updated


async def _process_collection(collection: Any, name: str, dry_run: bool) -> dict[str, Any]:
    logger.info(f"--- Collection: {name} ---")
    result: dict[str, Any] = {
        "collection": name,
        "duplicates": [],
        "missing_ids": 0,
        "backfilled": 0,
        "index_built": False,
    }

    # 1. Duplicate detection (the safety gate).
    dupes = await _find_duplicate_ids(collection)
    result["duplicates"] = dupes
    if dupes:
        logger.warning(
            f"  {len(dupes)} DUPLICATE id(s) found in '{name}' - a unique index "
            f"cannot be built until these are resolved:"
        )
        for d in dupes:
            logger.warning(f"    id={d['id']!r} appears {d['count']} times")

    # 2. Missing ids (would be backfilled).
    missing = await _count_missing_ids(collection)
    result["missing_ids"] = missing
    logger.info(f"  {missing} document(s) missing a non-empty id")

    if dry_run:
        logger.info(
            f"  DRY RUN - would backfill {missing} id(s) and build unique index "
            f"'{ID_INDEX_NAME}'" + (" (BLOCKED by duplicates)" if dupes else "")
        )
        return result

    # --apply path: refuse if duplicates exist (caller must resolve first).
    if dupes:
        logger.error(
            f"  REFUSING to build unique index on '{name}': resolve the "
            f"duplicate id(s) above first, then re-run --apply."
        )
        result["error"] = "duplicates_present"
        return result

    # 3. Backfill before building the unique index.
    backfilled = await _backfill_missing_ids(collection)
    result["backfilled"] = backfilled
    if backfilled:
        logger.info(f"  Backfilled id on {backfilled} document(s)")

    # 4. Build the unique partial index.
    try:
        await collection.create_index(
            "id",
            name=ID_INDEX_NAME,
            unique=True,
            partialFilterExpression={"id": {"$exists": True}},
        )
        result["index_built"] = True
        logger.info(f"  Built unique index '{ID_INDEX_NAME}' on '{name}'")
    except Exception as e:
        logger.error(f"  Failed to build unique index on '{name}': {e}")
        result["error"] = str(e)
    return result


async def _run(args: argparse.Namespace) -> int:
    dry_run = not args.apply
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
    except ImportError:
        logger.error("motor package required. Install with: uv add motor")
        return 1

    connection_string = _build_connection_string(args)
    if connection_string is None:
        return 1

    client = AsyncIOMotorClient(connection_string)
    db = client[args.database]

    results: list[dict[str, Any]] = []
    had_duplicates = False
    try:
        for base in ASSET_COLLECTIONS:
            name = f"{args.namespace}_{base}" if args.namespace else base
            res = await _process_collection(db[name], name, dry_run)
            results.append(res)
            if res.get("duplicates"):
                had_duplicates = True
    finally:
        client.close()

    logger.info("=== Summary ===")
    for r in results:
        logger.info(
            f"  {r['collection']}: duplicates={len(r['duplicates'])} "
            f"missing={r['missing_ids']} backfilled={r['backfilled']} "
            f"index_built={r['index_built']}"
        )
    logger.info(f"  mode={'DRY RUN' if dry_run else 'APPLY'}")

    if args.apply and had_duplicates:
        logger.error(
            "One or more collections had duplicate ids; unique index NOT built "
            "for those. Resolve duplicates and re-run --apply."
        )
        return 2
    return 0


def main() -> int:
    args = _parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
