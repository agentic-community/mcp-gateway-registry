#!/usr/bin/env python3
"""
Manage DocumentDB/MongoDB collections and documents.

This script is designed to run inside an ECS task or locally with proper network access.

Usage:
    # List all collections
    python manage-documentdb.py list

    # Inspect specific collection
    python manage-documentdb.py inspect --collection mcp_servers_default

    # Count documents in collection
    python manage-documentdb.py count --collection mcp_servers_default

    # Search documents in collection
    python manage-documentdb.py search --collection mcp_servers_default --limit 5

    # Show sample document from collection
    python manage-documentdb.py sample --collection mcp_servers_default

    # Query with filter
    python manage-documentdb.py query --collection mcp_servers_default --filter '{"enabled": true}'

    # Drop a collection (with confirmation)
    python manage-documentdb.py drop --collection mcp_scopes_default --confirm
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

from motor.motor_asyncio import AsyncIOMotorClient


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


async def _get_client(
    host: str,
    port: int,
    database: str,
    username: Optional[str],
    password: Optional[str],
    use_iam: bool,
    use_tls: bool,
    tls_ca_file: Optional[str],
) -> AsyncIOMotorClient:
    """Create DocumentDB async client."""
    # Get storage backend from environment variable
    storage_backend = os.getenv("STORAGE_BACKEND", "documentdb")

    connection_string = await _get_documentdb_connection_string(
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
        use_iam=use_iam,
        use_tls=use_tls,
        tls_ca_file=tls_ca_file,
        storage_backend=storage_backend,
    )

    # DocumentDB does not support retryable writes
    client = AsyncIOMotorClient(connection_string, retryWrites=False)

    return client


async def list_collections(
    host: str,
    port: int,
    database: str,
    username: Optional[str],
    password: Optional[str],
    use_iam: bool,
    use_tls: bool,
    tls_ca_file: Optional[str],
) -> int:
    """List all collections in the DocumentDB database."""
    try:
        client = await _get_client(
            host, port, database, username, password, use_iam, use_tls, tls_ca_file
        )

        db = client[database]

        # Verify connection
        server_info = await client.server_info()
        logger.info(f"Connected to DocumentDB/MongoDB {server_info.get('version', 'unknown')}")

        # Get all collection names
        collection_names = await db.list_collection_names()

        if not collection_names:
            logger.info(f"No collections found in database '{database}'")
            client.close()
            return 0

        # Sort by name
        collection_names.sort()

        print("\n" + "=" * 100)
        print(f"Found {len(collection_names)} collections in database '{database}'")
        print("=" * 100)

        # Get document counts for each collection
        for coll_name in collection_names:
            collection = db[coll_name]
            doc_count = await collection.count_documents({})

            print(f"\nCollection: {coll_name}")
            print(f"  Documents: {doc_count}")

            # Get estimated size (if available)
            try:
                stats = await db.command("collStats", coll_name)
                size_bytes = stats.get("size", 0)
                size_mb = size_bytes / (1024 * 1024)
                print(f"  Size: {size_mb:.2f} MB")
            except Exception as e:
                logger.warning(f"Could not retrieve collection stats for {coll_name}: {e}")

        print("\n" + "=" * 100)

        client.close()
        return 0

    except Exception as e:
        logger.error(f"Failed to list collections: {e}", exc_info=True)
        return 1


async def inspect_collection(
    host: str,
    port: int,
    database: str,
    collection_name: str,
    username: Optional[str],
    password: Optional[str],
    use_iam: bool,
    use_tls: bool,
    tls_ca_file: Optional[str],
) -> int:
    """Inspect a specific collection (schema and stats)."""
    try:
        client = await _get_client(
            host, port, database, username, password, use_iam, use_tls, tls_ca_file
        )

        db = client[database]
        collection = db[collection_name]

        # Check if collection exists
        collection_names = await db.list_collection_names()
        if collection_name not in collection_names:
            logger.error(f"Collection '{collection_name}' does not exist")
            client.close()
            return 1

        # Get document count
        doc_count = await collection.count_documents({})

        print("\n" + "=" * 100)
        print(f"Collection: {collection_name}")
        print("=" * 100)

        print(f"\nDocument Count: {doc_count}")

        # Get collection stats
        try:
            stats = await db.command("collStats", collection_name)
            print("\n--- Collection Statistics ---")
            print(f"Size: {stats.get('size', 0) / (1024 * 1024):.2f} MB")
            print(f"Storage Size: {stats.get('storageSize', 0) / (1024 * 1024):.2f} MB")
            print(f"Total Index Size: {stats.get('totalIndexSize', 0) / (1024 * 1024):.2f} MB")
            print(f"Average Object Size: {stats.get('avgObjSize', 0)} bytes")
        except Exception as e:
            logger.warning(f"Could not get collection stats: {e}")

        # Get indexes
        try:
            indexes = await collection.list_indexes().to_list(length=None)
            print("\n--- Indexes ---")
            for idx in indexes:
                print(f"\nIndex: {idx.get('name', 'unknown')}")
                print(f"  Keys: {json.dumps(idx.get('key', {}), indent=4)}")
                if idx.get("unique"):
                    print(f"  Unique: True")
        except Exception as e:
            logger.warning(f"Could not get indexes: {e}")

        # Get sample document to infer schema
        try:
            sample_doc = await collection.find_one({})
            if sample_doc:
                print("\n--- Sample Document Schema ---")
                print(json.dumps(_get_schema(sample_doc), indent=2))
        except Exception as e:
            logger.warning(f"Could not get sample document: {e}")

        print("\n" + "=" * 100)

        client.close()
        return 0

    except Exception as e:
        logger.error(f"Failed to inspect collection: {e}", exc_info=True)
        return 1


async def count_documents(
    host: str,
    port: int,
    database: str,
    collection_name: str,
    username: Optional[str],
    password: Optional[str],
    use_iam: bool,
    use_tls: bool,
    tls_ca_file: Optional[str],
) -> int:
    """Count documents in a collection."""
    try:
        client = await _get_client(
            host, port, database, username, password, use_iam, use_tls, tls_ca_file
        )

        db = client[database]
        collection = db[collection_name]

        # Get document count
        doc_count = await collection.count_documents({})

        print("\n" + "=" * 100)
        print(f"Collection: {collection_name}")
        print(f"Document Count: {doc_count}")
        print("=" * 100)

        client.close()
        return 0

    except Exception as e:
        logger.error(f"Failed to count documents: {e}", exc_info=True)
        return 1


async def search_documents(
    host: str,
    port: int,
    database: str,
    collection_name: str,
    limit: int,
    username: Optional[str],
    password: Optional[str],
    use_iam: bool,
    use_tls: bool,
    tls_ca_file: Optional[str],
) -> int:
    """Search/list documents in a collection."""
    try:
        client = await _get_client(
            host, port, database, username, password, use_iam, use_tls, tls_ca_file
        )

        db = client[database]
        collection = db[collection_name]

        # Get documents
        cursor = collection.find({}).limit(limit)
        documents = await cursor.to_list(length=limit)

        print("\n" + "=" * 100)
        print(f"Collection: {collection_name}")
        print(f"Showing {len(documents)} documents (limit: {limit})")
        print("=" * 100)

        for i, doc in enumerate(documents, 1):
            print(f"\n--- Document {i} ---")
            print(json.dumps(doc, indent=2, default=str))

        print("\n" + "=" * 100)

        client.close()
        return 0

    except Exception as e:
        logger.error(f"Failed to search documents: {e}", exc_info=True)
        return 1


async def sample_document(
    host: str,
    port: int,
    database: str,
    collection_name: str,
    username: Optional[str],
    password: Optional[str],
    use_iam: bool,
    use_tls: bool,
    tls_ca_file: Optional[str],
) -> int:
    """Show a sample document from a collection."""
    try:
        client = await _get_client(
            host, port, database, username, password, use_iam, use_tls, tls_ca_file
        )

        db = client[database]
        collection = db[collection_name]

        # Get one sample document
        sample_doc = await collection.find_one({})

        print("\n" + "=" * 100)
        print(f"Collection: {collection_name}")
        print("Sample Document:")
        print("=" * 100)

        if sample_doc:
            print(json.dumps(sample_doc, indent=2, default=str))
        else:
            print("No documents found in collection")

        print("\n" + "=" * 100)

        client.close()
        return 0

    except Exception as e:
        logger.error(f"Failed to get sample document: {e}", exc_info=True)
        return 1


async def query_documents(
    host: str,
    port: int,
    database: str,
    collection_name: str,
    filter_json: str,
    limit: int,
    username: Optional[str],
    password: Optional[str],
    use_iam: bool,
    use_tls: bool,
    tls_ca_file: Optional[str],
) -> int:
    """Query documents with a filter."""
    try:
        # Parse filter JSON
        filter_dict = json.loads(filter_json)

        client = await _get_client(
            host, port, database, username, password, use_iam, use_tls, tls_ca_file
        )

        db = client[database]
        collection = db[collection_name]

        # Get documents matching filter
        cursor = collection.find(filter_dict).limit(limit)
        documents = await cursor.to_list(length=limit)

        print("\n" + "=" * 100)
        print(f"Collection: {collection_name}")
        print(f"Filter: {filter_json}")
        print(f"Found {len(documents)} documents (limit: {limit})")
        print("=" * 100)

        for i, doc in enumerate(documents, 1):
            print(f"\n--- Document {i} ---")
            print(json.dumps(doc, indent=2, default=str))

        print("\n" + "=" * 100)

        client.close()
        return 0

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON filter: {e}")
        return 1
    except Exception as e:
        logger.error(f"Failed to query documents: {e}", exc_info=True)
        return 1


async def drop_collection(
    host: str,
    port: int,
    database: str,
    collection_name: str,
    confirm: bool,
    username: Optional[str],
    password: Optional[str],
    use_iam: bool,
    use_tls: bool,
    tls_ca_file: Optional[str],
) -> int:
    """Drop a collection from the database."""
    if not confirm:
        logger.error(
            "Drop operation requires --confirm flag. "
            "This will permanently delete all documents in the collection."
        )
        return 1

    try:
        client = await _get_client(
            host, port, database, username, password, use_iam, use_tls, tls_ca_file
        )

        db = client[database]

        # Check if collection exists
        collection_names = await db.list_collection_names()
        if collection_name not in collection_names:
            logger.error(f"Collection '{collection_name}' does not exist")
            client.close()
            return 1

        # Get document count before dropping
        collection = db[collection_name]
        doc_count = await collection.count_documents({})

        print("\n" + "=" * 100)
        print(f"Dropping collection: {collection_name}")
        print(f"Documents to be deleted: {doc_count}")
        print("=" * 100)

        # Drop the collection
        await db.drop_collection(collection_name)

        logger.info(f"Successfully dropped collection '{collection_name}'")
        print(f"\nCollection '{collection_name}' has been dropped.")
        print("=" * 100)

        client.close()
        return 0

    except Exception as e:
        logger.error(f"Failed to drop collection: {e}", exc_info=True)
        return 1


def _get_schema(doc: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
    """Infer schema from a document."""
    schema = {}

    for key, value in doc.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            schema.update(_get_schema(value, full_key))
        elif isinstance(value, list):
            if value and isinstance(value[0], dict):
                schema[full_key] = f"array[object]"
            else:
                schema[full_key] = f"array[{type(value[0]).__name__ if value else 'unknown'}]"
        else:
            schema[full_key] = type(value).__name__

    return schema


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Manage DocumentDB/MongoDB collections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List all collections
    python manage-documentdb.py list

    # Inspect a collection
    python manage-documentdb.py inspect --collection mcp_servers_default

    # Count documents
    python manage-documentdb.py count --collection mcp_servers_default

    # Search documents
    python manage-documentdb.py search --collection mcp_servers_default --limit 5

    # Sample document
    python manage-documentdb.py sample --collection mcp_servers_default

    # Query with filter
    python manage-documentdb.py query --collection mcp_servers_default --filter '{"enabled": true}'
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # List command
    subparsers.add_parser("list", help="List all collections")

    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a collection")
    inspect_parser.add_argument("--collection", required=True, help="Collection name")

    # Count command
    count_parser = subparsers.add_parser("count", help="Count documents in collection")
    count_parser.add_argument("--collection", required=True, help="Collection name")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("--collection", required=True, help="Collection name")
    search_parser.add_argument(
        "--limit", type=int, default=10, help="Number of documents to return"
    )

    # Sample command
    sample_parser = subparsers.add_parser("sample", help="Show sample document")
    sample_parser.add_argument("--collection", required=True, help="Collection name")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query with filter")
    query_parser.add_argument("--collection", required=True, help="Collection name")
    query_parser.add_argument("--filter", required=True, help="MongoDB filter as JSON")
    query_parser.add_argument("--limit", type=int, default=10, help="Number of documents to return")

    # Drop command
    drop_parser = subparsers.add_parser("drop", help="Drop a collection")
    drop_parser.add_argument("--collection", required=True, help="Collection name to drop")
    drop_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm the drop operation (required)",
    )

    # Common arguments
    parser.add_argument(
        "--host",
        default=os.getenv("DOCUMENTDB_HOST", "localhost"),
        help="DocumentDB host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("DOCUMENTDB_PORT", "27017")),
        help="DocumentDB port",
    )
    parser.add_argument(
        "--database",
        default=os.getenv("DOCUMENTDB_DATABASE", "mcp_registry"),
        help="Database name",
    )
    parser.add_argument(
        "--username",
        default=os.getenv("DOCUMENTDB_USERNAME"),
        help="DocumentDB username",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("DOCUMENTDB_PASSWORD"),
        help="DocumentDB password",
    )
    parser.add_argument(
        "--use-iam",
        action="store_true",
        default=os.getenv("DOCUMENTDB_USE_IAM", "false").lower() == "true",
        help="Use AWS IAM authentication",
    )
    parser.add_argument(
        "--use-tls",
        action="store_true",
        default=os.getenv("DOCUMENTDB_USE_TLS", "true").lower() == "true",
        help="Use TLS for connection",
    )
    parser.add_argument(
        "--tls-ca-file",
        default=os.getenv("DOCUMENTDB_TLS_CA_FILE", "/app/global-bundle.pem"),
        help="TLS CA file path",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    logger.info(f"Executing command: {args.command}")
    logger.info(f"Host: {args.host}:{args.port}")
    logger.info(f"Database: {args.database}")

    try:
        if args.command == "list":
            exit_code = await list_collections(
                args.host,
                args.port,
                args.database,
                args.username,
                args.password,
                args.use_iam,
                args.use_tls,
                args.tls_ca_file,
            )
        elif args.command == "inspect":
            exit_code = await inspect_collection(
                args.host,
                args.port,
                args.database,
                args.collection,
                args.username,
                args.password,
                args.use_iam,
                args.use_tls,
                args.tls_ca_file,
            )
        elif args.command == "count":
            exit_code = await count_documents(
                args.host,
                args.port,
                args.database,
                args.collection,
                args.username,
                args.password,
                args.use_iam,
                args.use_tls,
                args.tls_ca_file,
            )
        elif args.command == "search":
            exit_code = await search_documents(
                args.host,
                args.port,
                args.database,
                args.collection,
                args.limit,
                args.username,
                args.password,
                args.use_iam,
                args.use_tls,
                args.tls_ca_file,
            )
        elif args.command == "sample":
            exit_code = await sample_document(
                args.host,
                args.port,
                args.database,
                args.collection,
                args.username,
                args.password,
                args.use_iam,
                args.use_tls,
                args.tls_ca_file,
            )
        elif args.command == "query":
            exit_code = await query_documents(
                args.host,
                args.port,
                args.database,
                args.collection,
                args.filter,
                args.limit,
                args.username,
                args.password,
                args.use_iam,
                args.use_tls,
                args.tls_ca_file,
            )
        elif args.command == "drop":
            exit_code = await drop_collection(
                args.host,
                args.port,
                args.database,
                args.collection,
                args.confirm,
                args.username,
                args.password,
                args.use_iam,
                args.use_tls,
                args.tls_ca_file,
            )
        else:
            logger.error(f"Unknown command: {args.command}")
            exit_code = 1

        return exit_code

    except Exception as e:
        logger.error(f"Command failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
