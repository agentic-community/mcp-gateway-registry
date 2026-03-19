"""
One-time Lambda function to create DocumentDB indexes.

Runs in the VPC, connects to DocumentDB, creates all required indexes.
After successful execution, this Lambda can be deleted.
"""

import json
import logging
import os
from urllib.parse import quote_plus

import boto3
import pymongo
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

secretsmanager = boto3.client("secretsmanager")

DOCUMENTDB_SECRET_ARN = os.environ["DOCUMENTDB_SECRET_ARN"]
DOCUMENTDB_ENDPOINT = os.environ["DOCUMENTDB_ENDPOINT"]


def _get_credentials() -> dict:
    """Get DocumentDB credentials from Secrets Manager."""
    try:
        response = secretsmanager.get_secret_value(SecretId=DOCUMENTDB_SECRET_ARN)
        return json.loads(response["SecretString"])
    except ClientError as e:
        logger.error(f"Failed to retrieve DocumentDB credentials: {e}")
        raise


def _connect() -> pymongo.database.Database:
    """Connect to DocumentDB and return database handle."""
    credentials = _get_credentials()
    username = quote_plus(credentials["username"])
    password = quote_plus(credentials["password"])
    db_name = credentials.get("database", "telemetry")

    connection_string = (
        f"mongodb://{username}:{password}@"
        f"{DOCUMENTDB_ENDPOINT}/{db_name}?"
        f"authMechanism=SCRAM-SHA-1&authSource=admin"
        f"&tls=true&retryWrites=false"
    )

    logger.info(f"Connecting to DocumentDB at {DOCUMENTDB_ENDPOINT}")
    client = pymongo.MongoClient(connection_string)
    server_info = client.server_info()
    logger.info(f"Connected to DocumentDB version {server_info.get('version')}")

    return client[db_name]


def lambda_handler(event, context):
    """Lambda handler for index creation."""
    logger.info("Starting DocumentDB index creation")

    results = {
        "startup_events_indexes": [],
        "heartbeat_events_indexes": [],
        "errors": [],
    }

    try:
        db = _connect()

        # Define indexes per collection
        index_specs = {
            "startup_events": [
                ({"keys": [("received_at", 1)], "kwargs": {"expireAfterSeconds": 31536000}}, "TTL"),
                ({"keys": [("instance_id", 1)], "kwargs": {}}, "query"),
                ({"keys": [("v", 1), ("received_at", -1)], "kwargs": {}}, "query"),
            ],
            "heartbeat_events": [
                ({"keys": [("received_at", 1)], "kwargs": {"expireAfterSeconds": 31536000}}, "TTL"),
                ({"keys": [("instance_id", 1)], "kwargs": {}}, "query"),
            ],
        }

        for collection_name, indexes in index_specs.items():
            collection = db[collection_name]
            for spec, idx_type in indexes:
                try:
                    name = collection.create_index(spec["keys"], **spec["kwargs"])
                    results[f"{collection_name}_indexes"].append(
                        {"name": name, "type": idx_type, "status": "created"}
                    )
                    logger.info(f"Created {idx_type} index on {collection_name}: {name}")
                except Exception as e:
                    msg = f"Failed to create index on {collection_name}: {e}"
                    logger.error(msg)
                    results["errors"].append(msg)

        # Verify
        for coll_name in ["startup_events", "heartbeat_events"]:
            indexes = db[coll_name].index_information()
            count = db[coll_name].count_documents({})
            logger.info(f"{coll_name}: {len(indexes)} indexes, {count} documents")

        return {
            "statusCode": 200,
            "body": json.dumps({"message": "Index creation completed", "results": results}),
        }

    except Exception as e:
        logger.error(f"Lambda execution failed: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"message": "Index creation failed", "error": str(e)}),
        }
