#!/usr/bin/env python3
"""
Export telemetry data from DocumentDB to CSV.

Reads connection details from ~/bastion.env and AWS Secrets Manager,
then dumps startup_events and heartbeat_events into a single CSV file.

Usage:
    python3 export_csv.py
    python3 export_csv.py --output /tmp/registry_metrics.csv
    python3 export_csv.py --collection startup_events
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = "registry_metrics.csv"
CA_BUNDLE_PATH = os.path.expanduser("~/global-bundle.pem")
BASTION_ENV_PATH = os.path.expanduser("~/bastion.env")

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


def _load_bastion_env() -> dict[str, str]:
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
) -> dict[str, str]:
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
    except subprocess.CalledProcessError as e:
        logger.error("Failed to get secret from Secrets Manager (check ARN and permissions)")
        sys.exit(1)
    except (json.JSONDecodeError, KeyError) as e:
        logger.error("Failed to parse secret (unexpected format)")
        sys.exit(1)


def _fetch_documents(
    endpoint: str,
    username: str,
    password: str,
    database: str,
    collection: str,
) -> list[dict]:
    """Fetch all documents from a DocumentDB collection using mongosh.

    Args:
        endpoint: DocumentDB cluster endpoint.
        username: Database username.
        password: Database password.
        database: Database name.
        collection: Collection name to query.

    Returns:
        List of document dicts.
    """
    conn_string = f"mongodb://{username}@{endpoint}:27017/{database}"

    eval_script = (
        f"db.{collection}.find({{}}, {{_id:0}})"
        f".sort({{ts:1}}).forEach(d => print(JSON.stringify(d)));"
    )

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
            timeout=120,
        )
    except subprocess.CalledProcessError:
        logger.error(f"mongosh failed for {collection} (check connection and credentials)")
        return []
    except subprocess.TimeoutExpired:
        logger.error(f"mongosh timed out for {collection}")
        return []

    documents = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            documents.append(json.loads(line))
        except json.JSONDecodeError:
            logger.debug(f"Skipping non-JSON line: {line[:80]}")

    return documents


def _write_csv(
    documents: list[dict],
    columns: list[str],
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


def main():
    """Parse arguments and export telemetry data to CSV."""
    parser = argparse.ArgumentParser(
        description="Export telemetry data from DocumentDB to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 export_csv.py
    python3 export_csv.py --output /tmp/registry_metrics.csv
    python3 export_csv.py --collection startup_events
    python3 export_csv.py --collection heartbeat_events
""",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--collection",
        choices=["all", "startup_events", "heartbeat_events"],
        default="all",
        help="Which collection to export (default: all)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load connection details
    env = _load_bastion_env()
    logger.info(f"DocumentDB endpoint: {env['DOCDB_ENDPOINT']}")

    # Get credentials
    creds = _get_credentials(env["SECRET_ARN"], env["AWS_REGION"])
    logger.info(f"Database: {creds['database']}, User: {creds['username'][:3]}***")

    start_time = time.time()
    all_documents = []

    # Fetch startup events
    if args.collection in ("all", "startup_events"):
        logger.info("Fetching startup_events...")
        startup_docs = _fetch_documents(
            endpoint=env["DOCDB_ENDPOINT"],
            username=creds["username"],
            password=creds["password"],
            database=creds["database"],
            collection="startup_events",
        )
        logger.info(f"  Found {len(startup_docs)} startup events")
        all_documents.extend(startup_docs)

    # Fetch heartbeat events
    if args.collection in ("all", "heartbeat_events"):
        logger.info("Fetching heartbeat_events...")
        heartbeat_docs = _fetch_documents(
            endpoint=env["DOCDB_ENDPOINT"],
            username=creds["username"],
            password=creds["password"],
            database=creds["database"],
            collection="heartbeat_events",
        )
        logger.info(f"  Found {len(heartbeat_docs)} heartbeat events")
        all_documents.extend(heartbeat_docs)

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

    # Write CSV
    rows_written = _write_csv(all_documents, columns, args.output)

    elapsed = time.time() - start_time
    logger.info(f"Exported {rows_written} rows to {args.output} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
