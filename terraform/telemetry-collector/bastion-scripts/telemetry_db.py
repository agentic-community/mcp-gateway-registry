#!/usr/bin/env python3
"""
Manage telemetry data in DocumentDB.

Provides export (CSV dump) and purge (delete all) operations for the
telemetry collector's startup_events and heartbeat_events collections.

Reads connection details from ~/bastion.env and credentials from
AWS Secrets Manager.

Usage:
    python3 telemetry_db.py export
    python3 telemetry_db.py export --output /tmp/metrics.csv
    python3 telemetry_db.py export --collection startup_events
    python3 telemetry_db.py purge
    python3 telemetry_db.py purge --collection heartbeat_events
    python3 telemetry_db.py purge --confirm
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = "registry_metrics.csv"
CA_BUNDLE_PATH = os.path.expanduser("~/global-bundle.pem")
BASTION_ENV_PATH = os.path.expanduser("~/bastion.env")

COLLECTIONS = ["startup_events", "heartbeat_events"]

# Column order for startup events
STARTUP_COLUMNS = [
    "event",
    "registry_id",
    "v",
    "py",
    "os",
    "arch",
    "cloud",
    "compute",
    "mode",
    "registry_mode",
    "storage",
    "auth",
    "federation",
    "search_queries_total",
    "search_queries_24h",
    "search_queries_1h",
    "ts",
    "stored_at",
    "source_ip_hash",
]

# Column order for heartbeat events
HEARTBEAT_COLUMNS = [
    "event",
    "registry_id",
    "v",
    "cloud",
    "compute",
    "servers_count",
    "agents_count",
    "skills_count",
    "peers_count",
    "search_backend",
    "embeddings_provider",
    "uptime_hours",
    "search_queries_total",
    "search_queries_24h",
    "search_queries_1h",
    "ts",
    "stored_at",
    "source_ip_hash",
]

# Union of all columns for the combined CSV
ALL_COLUMNS = [
    "event",
    "registry_id",
    "v",
    "py",
    "os",
    "arch",
    "cloud",
    "compute",
    "mode",
    "registry_mode",
    "storage",
    "auth",
    "federation",
    "servers_count",
    "agents_count",
    "skills_count",
    "peers_count",
    "search_backend",
    "embeddings_provider",
    "uptime_hours",
    "search_queries_total",
    "search_queries_24h",
    "search_queries_1h",
    "ts",
    "stored_at",
    "source_ip_hash",
]


# ---------------------------------------------------------------------------
# Private helpers — connection, credentials, mongosh wrappers
# ---------------------------------------------------------------------------


def _load_bastion_env() -> Dict[str, str]:
    """Load connection variables from ~/bastion.env.

    Returns:
        Dict with DOCDB_ENDPOINT, SECRET_ARN, AWS_REGION.

    Raises:
        SystemExit: If bastion.env is missing or incomplete.
    """
    if not os.path.exists(BASTION_ENV_PATH):
        logger.error(f"Bastion env file not found: {BASTION_ENV_PATH}")
        logger.error("Run setup-bastion.sh first to configure the bastion host.")
        sys.exit(1)

    env = {}
    with open(BASTION_ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip().strip('"')

    required_keys = ["DOCDB_ENDPOINT", "SECRET_ARN", "AWS_REGION"]
    for key in required_keys:
        if key not in env:
            logger.error(f"Missing {key} in {BASTION_ENV_PATH}")
            sys.exit(1)

    return env


def _get_credentials(
    secret_arn: str,
    aws_region: str,
) -> Dict[str, str]:
    """Fetch DocumentDB credentials from AWS Secrets Manager.

    Args:
        secret_arn: ARN of the secret in Secrets Manager.
        aws_region: AWS region for the Secrets Manager call.

    Returns:
        Dict with username, password, database.

    Raises:
        SystemExit: If credentials cannot be retrieved.
    """
    try:
        result = subprocess.run(  # nosec B603 B607 - hardcoded command
            [
                "aws", "secretsmanager", "get-secret-value",
                "--secret-id", secret_arn,
                "--region", aws_region,
                "--query", "SecretString",
                "--output", "text",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        # Parse secret and extract only needed fields — never log raw output
        parsed = json.loads(result.stdout.strip())
        username = parsed["username"]
        password = parsed["password"]
        database = parsed.get("database", "telemetry")
        # Clear raw secret from memory
        del parsed
        return {
            "username": username,
            "password": password,
            "database": database,
        }
    except subprocess.CalledProcessError:
        logger.error("Failed to get secret from Secrets Manager (check ARN and permissions)")
        sys.exit(1)
    except (json.JSONDecodeError, KeyError):
        logger.error("Failed to parse secret (unexpected format)")
        sys.exit(1)


def _run_mongosh(
    endpoint: str,
    username: str,
    password: str,
    database: str,
    eval_script: str,
    timeout: int = 120,
) -> Optional[str]:
    """Run a mongosh eval command and return stdout.

    Args:
        endpoint: DocumentDB cluster endpoint.
        username: Database username.
        password: Database password.
        database: Database name.
        eval_script: JavaScript to evaluate.
        timeout: Command timeout in seconds.

    Returns:
        Stdout string on success, None on failure.
    """
    conn_string = f"mongodb://{username}@{endpoint}:27017/{database}"

    try:
        result = subprocess.run(  # nosec B603 B607 - hardcoded command
            [
                "mongosh", conn_string,
                "--tls",
                "--tlsCAFile", CA_BUNDLE_PATH,
                "--retryWrites", "false",
                "--authenticationMechanism", "SCRAM-SHA-1",
                "--password", password,
                "--quiet",
                "--eval", eval_script,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        logger.error("mongosh command failed (check connection and credentials)")
        return None
    except subprocess.TimeoutExpired:
        logger.error("mongosh command timed out")
        return None


def _get_collection_count(
    endpoint: str,
    username: str,
    password: str,
    database: str,
    collection: str,
) -> int:
    """Get document count for a collection.

    Args:
        endpoint: DocumentDB cluster endpoint.
        username: Database username.
        password: Database password.
        database: Database name.
        collection: Collection name to count.

    Returns:
        Number of documents in the collection.
    """
    eval_script = f"print(db.{collection}.countDocuments({{}}));"
    output = _run_mongosh(endpoint, username, password, database, eval_script, timeout=30)

    if output is None:
        logger.error(f"Failed to count documents in {collection}")
        return 0

    try:
        return int(output)
    except ValueError:
        logger.error(f"Unexpected count output for {collection}: {output[:80]}")
        return 0


def _fetch_documents(
    endpoint: str,
    username: str,
    password: str,
    database: str,
    collection: str,
) -> List[dict]:
    """Fetch all documents from a DocumentDB collection.

    Args:
        endpoint: DocumentDB cluster endpoint.
        username: Database username.
        password: Database password.
        database: Database name.
        collection: Collection name to query.

    Returns:
        List of document dicts.
    """
    eval_script = (
        f"db.{collection}.find({{}}, {{_id:0}})"
        f".sort({{ts:1}}).forEach(d => print(JSON.stringify(d)));"
    )
    output = _run_mongosh(endpoint, username, password, database, eval_script)

    if output is None:
        logger.error(f"Failed to fetch documents from {collection}")
        return []

    documents = []
    for line in output.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            documents.append(json.loads(line))
        except json.JSONDecodeError:
            logger.debug(f"Skipping non-JSON line: {line[:80]}")

    return documents


def _delete_documents(
    endpoint: str,
    username: str,
    password: str,
    database: str,
    collection: str,
) -> int:
    """Delete all documents from a DocumentDB collection.

    Args:
        endpoint: DocumentDB cluster endpoint.
        username: Database username.
        password: Database password.
        database: Database name.
        collection: Collection name to purge.

    Returns:
        Number of documents deleted.
    """
    eval_script = (
        f"var r = db.{collection}.deleteMany({{}});"
        f"print(JSON.stringify({{deletedCount: r.deletedCount}}));"
    )
    output = _run_mongosh(endpoint, username, password, database, eval_script)

    if output is None:
        logger.error(f"Failed to delete documents from {collection}")
        return 0

    try:
        parsed = json.loads(output)
        return parsed.get("deletedCount", 0)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse delete result for {collection}")
        return 0


def _write_csv(
    documents: List[dict],
    columns: List[str],
    output_path: str,
) -> int:
    """Write documents to a CSV file.

    Args:
        documents: List of document dicts.
        columns: Column names for the CSV header.
        output_path: Output file path.

    Returns:
        Number of rows written.
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()

        for doc in documents:
            # Flatten nested $date objects from BSON extended JSON
            for key in ("stored_at", "ts"):
                val = doc.get(key)
                if isinstance(val, dict) and "$date" in val:
                    doc[key] = val["$date"]

            writer.writerow(doc)

    return len(documents)


def _resolve_collections(
    collection_arg: str,
) -> List[str]:
    """Resolve the --collection argument to a list of collection names.

    Args:
        collection_arg: "all", "startup_events", or "heartbeat_events".

    Returns:
        List of collection name strings.
    """
    if collection_arg == "all":
        return list(COLLECTIONS)
    return [collection_arg]


def _connect(args: argparse.Namespace) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load bastion env and fetch credentials.

    Args:
        args: Parsed CLI arguments (uses args.debug).

    Returns:
        Tuple of (env_dict, credentials_dict).
    """
    env = _load_bastion_env()
    logger.info(f"DocumentDB endpoint: {env['DOCDB_ENDPOINT']}")

    creds = _get_credentials(env["SECRET_ARN"], env["AWS_REGION"])
    logger.info("Using configured database for telemetry DocumentDB connection")

    return env, creds


# ---------------------------------------------------------------------------
# Public subcommand handlers
# ---------------------------------------------------------------------------


def cmd_export(args: argparse.Namespace) -> None:
    """Handle the 'export' subcommand — dump telemetry data to CSV.

    Args:
        args: Parsed CLI arguments.
    """
    env, creds = _connect(args)
    target_collections = _resolve_collections(args.collection)

    start_time = time.time()
    all_documents = []

    for collection in target_collections:
        logger.info(f"Fetching {collection}...")
        docs = _fetch_documents(
            endpoint=env["DOCDB_ENDPOINT"],
            username=creds["username"],
            password=creds["password"],
            database=creds["database"],
            collection=collection,
        )
        logger.info(f"  Found {len(docs)} documents")
        all_documents.extend(docs)

    if not all_documents:
        logger.warning("No documents found. CSV not created.")
        return

    # Determine columns based on collection
    if args.collection == "startup_events":
        columns = STARTUP_COLUMNS
    elif args.collection == "heartbeat_events":
        columns = HEARTBEAT_COLUMNS
    else:
        columns = ALL_COLUMNS

    rows_written = _write_csv(all_documents, columns, args.output)

    elapsed = time.time() - start_time
    logger.info(f"Exported {rows_written} rows to {args.output} in {elapsed:.1f}s")


def cmd_purge(args: argparse.Namespace) -> None:
    """Handle the 'purge' subcommand — delete telemetry data from DocumentDB.

    Args:
        args: Parsed CLI arguments.
    """
    env, creds = _connect(args)
    target_collections = _resolve_collections(args.collection)

    # Show counts before deletion
    total_count = 0
    for collection in target_collections:
        count = _get_collection_count(
            endpoint=env["DOCDB_ENDPOINT"],
            username=creds["username"],
            password=creds["password"],
            database=creds["database"],
            collection=collection,
        )
        logger.info(f"  {collection}: {count} documents")
        total_count += count

    if total_count == 0:
        logger.info("No documents to delete.")
        return

    # Confirm deletion
    if not args.confirm:
        answer = input(
            f"\nDelete {total_count} documents from {', '.join(target_collections)}? [y/N] "
        )
        if answer.lower() != "y":
            logger.info("Aborted.")
            return

    # Delete documents
    start_time = time.time()
    total_deleted = 0

    for collection in target_collections:
        logger.info(f"Purging {collection}...")
        deleted = _delete_documents(
            endpoint=env["DOCDB_ENDPOINT"],
            username=creds["username"],
            password=creds["password"],
            database=creds["database"],
            collection=collection,
        )
        logger.info(f"  Deleted {deleted} documents from {collection}")
        total_deleted += deleted

    elapsed = time.time() - start_time
    logger.info(f"Purged {total_deleted} total documents in {elapsed:.1f}s")


def main():
    """Parse arguments and dispatch to the appropriate subcommand."""
    parser = argparse.ArgumentParser(
        description="Manage telemetry data in DocumentDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 telemetry_db.py export
    python3 telemetry_db.py export --output /tmp/metrics.csv
    python3 telemetry_db.py export --collection startup_events
    python3 telemetry_db.py purge
    python3 telemetry_db.py purge --collection heartbeat_events
    python3 telemetry_db.py purge --confirm
""",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- export subcommand ---
    export_parser = subparsers.add_parser(
        "export",
        help="Export telemetry data to CSV",
    )
    export_parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV file path (default: {DEFAULT_OUTPUT})",
    )
    export_parser.add_argument(
        "--collection",
        choices=["all", "startup_events", "heartbeat_events"],
        default="all",
        help="Which collection to export (default: all)",
    )

    # --- purge subcommand ---
    purge_parser = subparsers.add_parser(
        "purge",
        help="Delete all telemetry data from DocumentDB",
    )
    purge_parser.add_argument(
        "--collection",
        choices=["all", "startup_events", "heartbeat_events"],
        default="all",
        help="Which collection to purge (default: all)",
    )
    purge_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip interactive confirmation prompt",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.command == "export":
        cmd_export(args)
    elif args.command == "purge":
        cmd_purge(args)


if __name__ == "__main__":
    main()
