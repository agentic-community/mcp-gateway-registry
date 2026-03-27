#!/usr/bin/env python3
"""
Purge telemetry data from DocumentDB.

Reads connection details from ~/bastion.env and AWS Secrets Manager,
then deletes documents from startup_events and/or heartbeat_events.

Usage:
    python3 purge_data.py                          # interactive confirmation
    python3 purge_data.py --collection startup_events
    python3 purge_data.py --confirm                # skip interactive prompt
"""

import argparse
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

CA_BUNDLE_PATH = os.path.expanduser("~/global-bundle.pem")
BASTION_ENV_PATH = os.path.expanduser("~/bastion.env")

COLLECTIONS = ["startup_events", "heartbeat_events"]


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
    except subprocess.CalledProcessError:
        logger.error("Failed to get secret from Secrets Manager (check ARN and permissions)")
        sys.exit(1)
    except (json.JSONDecodeError, KeyError):
        logger.error("Failed to parse secret (unexpected format)")
        sys.exit(1)


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
    conn_string = f"mongodb://{username}@{endpoint}:27017/{database}"

    eval_script = f"print(db.{collection}.countDocuments({{}}));"

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
            timeout=30,
        )
        return int(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        logger.error(f"Failed to count documents in {collection}")
        return 0


def _delete_collection(
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
    conn_string = f"mongodb://{username}@{endpoint}:27017/{database}"

    eval_script = (
        f"var r = db.{collection}.deleteMany({{}});"
        f"print(JSON.stringify({{deletedCount: r.deletedCount}}));"
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
        parsed = json.loads(result.stdout.strip())
        return parsed.get("deletedCount", 0)
    except subprocess.CalledProcessError:
        logger.error(f"mongosh failed for {collection} (check connection and credentials)")
        return 0
    except (json.JSONDecodeError, subprocess.TimeoutExpired):
        logger.error(f"Failed to parse delete result for {collection}")
        return 0


def main():
    """Parse arguments and purge telemetry data from DocumentDB."""
    parser = argparse.ArgumentParser(
        description="Purge telemetry data from DocumentDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 purge_data.py
    python3 purge_data.py --collection startup_events
    python3 purge_data.py --collection heartbeat_events
    python3 purge_data.py --confirm
""",
    )
    parser.add_argument(
        "--collection",
        choices=["all", "startup_events", "heartbeat_events"],
        default="all",
        help="Which collection to purge (default: all)",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip interactive confirmation prompt",
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
    logger.info(f"Database: {creds['database']}")

    # Determine which collections to purge
    if args.collection == "all":
        target_collections = COLLECTIONS
    else:
        target_collections = [args.collection]

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
        deleted = _delete_collection(
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


if __name__ == "__main__":
    main()
