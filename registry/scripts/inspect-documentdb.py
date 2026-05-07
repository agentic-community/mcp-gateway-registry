#!/usr/bin/env python3
"""
Inspect DocumentDB collections and indexes.

Usage:
    python inspect-documentdb.py
"""

import asyncio
import json
import os
import sys

from motor.motor_asyncio import AsyncIOMotorClient


async def inspect_documentdb():
    """Inspect DocumentDB collections and indexes."""
    # Get connection details from environment
    host = os.getenv("DOCUMENTDB_HOST")
    port = int(os.getenv("DOCUMENTDB_PORT", "27017"))
    username = os.getenv("DOCUMENTDB_USERNAME")
    password = os.getenv("DOCUMENTDB_PASSWORD")
    database = os.getenv("DOCUMENTDB_DATABASE", "mcp_registry")
    use_tls = os.getenv("DOCUMENTDB_USE_TLS", "true").lower() == "true"
    ca_file = os.getenv("DOCUMENTDB_TLS_CA_FILE", "/app/certs/global-bundle.pem")

    print("=" * 80)
    print("DocumentDB Inspection")
    print("=" * 80)
    print(f"Host: {host}:{port}")
    print(f"Database: {database}")
    print(f"TLS: {use_tls}")
    if use_tls:
        print(f"CA File: {ca_file}")
    print("=" * 80)
    print()

    # Build connection string with appropriate auth mechanism.
    # Intentionally inline (not imported from registry.core.config) because
    # this script is invoked standalone from the container shell and we want
    # it to run without the full app package on sys.path. Keep in sync with
    # registry/core/config.py MONGODB_BACKENDS and utils/mongodb_connection.py.
    # AWS DocumentDB v5.0 only supports SCRAM-SHA-1; every other
    # MongoDB-compatible backend uses SCRAM-SHA-256.
    storage_backend = os.getenv("STORAGE_BACKEND", "documentdb")
    if storage_backend == "documentdb":
        auth_mechanism = "SCRAM-SHA-1"
    else:
        auth_mechanism = "SCRAM-SHA-256"

    if username and password:
        connection_string = (
            f"mongodb://{username}:{password}@{host}:{port}/{database}?"
            f"authMechanism={auth_mechanism}&authSource=admin"
        )
    else:
        connection_string = f"mongodb://{host}:{port}/{database}"

    # TLS options
    tls_options = {}
    if use_tls:
        tls_options["tls"] = True
        if ca_file:
            tls_options["tlsCAFile"] = ca_file

    # Connect to DocumentDB
    print("Connecting to DocumentDB...")
    # IMPORTANT: DocumentDB does not support retryable writes
    client = AsyncIOMotorClient(connection_string, retryWrites=False, **tls_options)
    db = client[database]

    try:
        # Test connection
        server_info = await client.server_info()
        print(f"Connected to MongoDB/DocumentDB version: {server_info.get('version')}")
        print()

        # List all collections
        collections = await db.list_collection_names()
        print(f"Collections ({len(collections)}):")
        print("-" * 80)
        for coll_name in sorted(collections):
            print(f"  - {coll_name}")
        print()

        # Inspect each collection
        for coll_name in sorted(collections):
            print("=" * 80)
            print(f"Collection: {coll_name}")
            print("=" * 80)

            collection = db[coll_name]

            # Count documents
            count = await collection.count_documents({})
            print(f"Document count: {count}")
            print()

            # List indexes
            indexes = await collection.list_indexes().to_list(None)
            print(f"Indexes ({len(indexes)}):")
            print("-" * 80)

            for idx in indexes:
                idx_name = idx.get("name")
                print(f"\nIndex: {idx_name}")

                # Check if it's a vector index
                if "vectorOptions" in idx:
                    vector_opts = idx["vectorOptions"]
                    print("  Type: Vector Index (HNSW)")
                    print(f"  Dimensions: {vector_opts.get('dimensions')}")
                    print(f"  Similarity: {vector_opts.get('similarity')}")
                    print(f"  Vector Type: {vector_opts.get('type')}")
                else:
                    print("  Type: Standard Index")
                    if "key" in idx:
                        print(f"  Keys: {idx['key']}")

                if "unique" in idx and idx["unique"]:
                    print("  Unique: True")

                if "sparse" in idx and idx["sparse"]:
                    print("  Sparse: True")

            print()

            # Show sample document (if any exist)
            if count > 0:
                print("Sample document:")
                print("-" * 80)
                sample = await collection.find_one({})
                if sample:
                    # Remove _id for cleaner display
                    sample.pop("_id", None)
                    print(json.dumps(sample, indent=2, default=str))
                print()

        print("=" * 80)
        print("Inspection complete!")
        print("=" * 80)

    finally:
        client.close()


if __name__ == "__main__":
    try:
        asyncio.run(inspect_documentdb())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
