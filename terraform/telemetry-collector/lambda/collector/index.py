"""
AWS Lambda handler for telemetry collector.

Privacy-first design:
- Always returns 204 (no information leakage)
- Hashes source IP for rate limiting (no storage)
- Fail-silent: all errors caught and logged
- TLS-only DocumentDB connection

Architecture:
- API Gateway HTTP API → Lambda → DynamoDB (rate limiting) → DocumentDB (storage)
"""

import hashlib
import hmac
import json
import logging
import os
from datetime import UTC, datetime
from urllib.parse import quote_plus

import boto3
import pymongo
from botocore.exceptions import ClientError
from pydantic import ValidationError
from schemas import HeartbeatEvent, StartupEvent

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# HMAC signing key — must match the key in registry/core/telemetry.py.
# This is NOT a secret. It prevents casual abuse (random curl requests)
# by requiring callers to compute a valid HMAC over the request body.
TELEMETRY_SIGNING_KEY = "mcp-registry-telemetry-v1-a7f3b9c2e1d4"

# AWS clients (lazy-init for testability without credentials)
dynamodb = None
secretsmanager = None


def _init_aws_clients():
    global dynamodb, secretsmanager
    if dynamodb is None:
        dynamodb = boto3.resource("dynamodb")
        secretsmanager = boto3.client("secretsmanager")


# Environment variables (required — Lambda will fail fast if misconfigured)
RATE_LIMIT_TABLE = os.environ["RATE_LIMIT_TABLE"]
DOCUMENTDB_SECRET_ARN = os.environ["DOCUMENTDB_SECRET_ARN"]
DOCUMENTDB_ENDPOINT = os.environ["DOCUMENTDB_ENDPOINT"]

# Rate limiting constants
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 10

# Globals for connection pooling (reused across warm Lambda invocations)
_mongo_client: pymongo.MongoClient | None = None
_mongo_database = None  # pymongo Database instance
_credentials: dict | None = None


def _get_credentials() -> dict:
    """Get DocumentDB credentials from Secrets Manager (cached)."""
    global _credentials
    _init_aws_clients()

    if _credentials is not None:
        return _credentials

    try:
        response = secretsmanager.get_secret_value(SecretId=DOCUMENTDB_SECRET_ARN)
        _credentials = json.loads(response["SecretString"])
        logger.info("Retrieved DocumentDB credentials from Secrets Manager")
        return _credentials
    except ClientError as e:
        logger.error(f"Failed to retrieve DocumentDB credentials: {e}")
        raise


def _get_database():
    """Get DocumentDB database client (singleton, reused across invocations)."""
    global _mongo_client, _mongo_database

    if _mongo_database is not None:
        return _mongo_database

    credentials = _get_credentials()
    username = quote_plus(credentials["username"])
    password = quote_plus(credentials["password"])
    db_name = credentials.get("database", "telemetry")

    connection_string = (
        f"mongodb://{username}:{password}@"
        f"{DOCUMENTDB_ENDPOINT}/{db_name}?"
        f"authMechanism=SCRAM-SHA-1&authSource=admin"
        f"&tls=true&tlsAllowInvalidCertificates=true&retryWrites=false"
        f"&directConnection=true"
        f"&connectTimeoutMS=10000&serverSelectionTimeoutMS=10000"
    )

    logger.info(f"Connecting to DocumentDB at {DOCUMENTDB_ENDPOINT}")
    _mongo_client = pymongo.MongoClient(connection_string)
    _mongo_database = _mongo_client[db_name]

    # Verify connection
    _mongo_client.server_info()
    logger.info("Connected to DocumentDB")

    return _mongo_database


def _verify_signature(body: str, signature: str) -> bool:
    """Verify HMAC-SHA256 signature of the request body.

    Args:
        body: The raw request body string.
        signature: The hex-encoded HMAC signature from the header.

    Returns:
        True if the signature is valid, False otherwise.
    """
    if not signature:
        return False

    expected = hmac.new(
        TELEMETRY_SIGNING_KEY.encode(),
        body.encode(),
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected, signature)


def _hash_ip(ip: str) -> str:
    """Hash IP address (SHA-256) for privacy-preserving rate limiting."""
    return hashlib.sha256(ip.encode()).hexdigest()


def _check_rate_limit(ip_hash: str) -> bool:
    """Check rate limit using DynamoDB atomic counter. Returns True if allowed."""
    _init_aws_clients()
    try:
        table = dynamodb.Table(RATE_LIMIT_TABLE)
        now = int(datetime.now(UTC).timestamp())
        window_start = now - RATE_LIMIT_WINDOW_SECONDS

        # First, try to reset expired entries and set count to 1
        try:
            table.update_item(
                Key={"ip_hash": ip_hash},
                UpdateExpression="SET request_count = :one, expiry_time = :expiry, last_request = :now",
                ExpressionAttributeValues={
                    ":one": 1,
                    ":expiry": now + RATE_LIMIT_WINDOW_SECONDS,
                    ":now": now,
                    ":window_start": window_start,
                },
                ConditionExpression="attribute_not_exists(last_request) OR last_request < :window_start",
            )
            return True  # Window expired or new entry — allowed
        except ClientError as e:
            if e.response["Error"]["Code"] != "ConditionalCheckFailedException":
                raise
            # Item exists and window hasn't expired — increment

        # Increment within active window
        table.update_item(
            Key={"ip_hash": ip_hash},
            UpdateExpression="ADD request_count :inc SET last_request = :now",
            ExpressionAttributeValues={
                ":inc": 1,
                ":now": now,
                ":max": RATE_LIMIT_MAX_REQUESTS,
            },
            ConditionExpression="request_count < :max",
        )
        return True

    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            logger.warning(f"Rate limit exceeded for IP hash: {ip_hash[:8]}...")
            return False
        logger.error(f"Rate limit check failed: {e}")
        return True  # Fail-open for telemetry


def _store_event(event_type: str, payload: dict) -> None:
    """Store validated telemetry event in DocumentDB."""
    db = _get_database()
    collection = db[f"{event_type}_events"]

    # Convert ts string to BSON datetime for consistent querying
    if "ts" in payload and isinstance(payload["ts"], str):
        try:
            payload["ts"] = datetime.fromisoformat(payload["ts"].replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass  # Keep as string if parsing fails

    document = {
        **payload,
        "received_at": datetime.now(UTC),
    }

    result = collection.insert_one(document)
    logger.info(
        f"Stored {event_type} event: registry_id={payload.get('registry_id', 'unknown')} "
        f"doc_id={result.inserted_id}"
    )


def lambda_handler(event: dict, context: dict) -> dict:
    """
    Lambda handler for telemetry collector.

    Always returns 204 No Content (privacy-first: no information leakage).
    """
    try:
        # Rate limit by hashed IP
        source_ip = event.get("requestContext", {}).get("http", {}).get("sourceIp", "unknown")
        if not _check_rate_limit(_hash_ip(source_ip)):
            return {"statusCode": 204}

        # Verify HMAC signature (reject unsigned or forged requests)
        headers = event.get("headers", {})
        signature = headers.get("x-telemetry-signature", "")
        raw_body = event.get("body", "")
        if not _verify_signature(raw_body, signature):
            logger.warning(f"Invalid or missing signature from {_hash_ip(source_ip)[:8]}...")
            return {"statusCode": 204}

        # Parse body
        body = event.get("body", "{}")
        if isinstance(body, str):
            try:
                payload = json.loads(body)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
                return {"statusCode": 204}
        else:
            payload = body

        # Validate by event type
        event_type = payload.get("event")

        if event_type == "startup":
            try:
                validated = StartupEvent(**payload)
                logger.info(f"Validated startup event: v={validated.v} storage={validated.storage}")
            except ValidationError as e:
                logger.error(f"Startup validation failed: {e}")
                return {"statusCode": 204}

        elif event_type == "heartbeat":
            try:
                validated = HeartbeatEvent(**payload)
                logger.info(
                    f"Validated heartbeat event: v={validated.v} servers={validated.servers_count}"
                )
            except ValidationError as e:
                logger.error(f"Heartbeat validation failed: {e}")
                return {"statusCode": 204}

        else:
            logger.error(f"Unknown event type: {event_type}")
            return {"statusCode": 204}

        # Store in DocumentDB
        try:
            _store_event(event_type, validated.model_dump())
        except Exception as e:
            logger.error(f"Failed to store event: {e}")

        return {"statusCode": 204}

    except Exception as e:
        logger.exception(f"Unexpected error in lambda_handler: {e}")
        return {"statusCode": 204}
