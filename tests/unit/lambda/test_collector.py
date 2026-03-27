"""
Unit tests for telemetry collector Lambda function.

Tests validation, rate limiting, storage, and fail-silent behavior.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError
from pydantic import ValidationError

# Add Lambda collector to path for imports
lambda_path = (
    Path(__file__).parent.parent.parent.parent
    / "terraform"
    / "telemetry-collector"
    / "lambda"
    / "collector"
)
sys.path.insert(0, str(lambda_path))

from index import (  # noqa: E402
    _check_rate_limit,
    _get_credentials,
    _get_database,
    _hash_ip,
    _store_event,
    lambda_handler,
)
from schemas import HeartbeatEvent, StartupEvent  # noqa: E402


# Reset global singletons between tests
@pytest.fixture(autouse=True)
def _reset_globals():
    """Reset module-level singletons before each test."""
    import index

    index._mongo_client = None
    index._mongo_database = None
    index._credentials = None
    yield


class TestSchemas:
    """Test Pydantic validation schemas."""

    def test_startup_event_valid(self):
        payload = {
            "event": "startup",
            "schema_version": "1",
            "instance_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "v": "1.0.16",
            "py": "3.12",
            "os": "linux",
            "arch": "x86_64",
            "mode": "with-gateway",
            "registry_mode": "full",
            "storage": "documentdb",
            "auth": "keycloak",
            "federation": True,
            "ts": "2026-03-18T00:00:00Z",
        }
        event = StartupEvent(**payload)
        assert event.event == "startup"
        assert event.v == "1.0.16"
        assert event.storage == "documentdb"

    def test_startup_event_invalid_event_type(self):
        payload = {
            "event": "heartbeat",
            "schema_version": "1",
            "instance_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "v": "1.0.16",
            "py": "3.12",
            "os": "linux",
            "arch": "x86_64",
            "mode": "with-gateway",
            "registry_mode": "full",
            "storage": "documentdb",
            "auth": "keycloak",
            "federation": True,
            "ts": "2026-03-18T00:00:00Z",
        }
        with pytest.raises(ValidationError):
            StartupEvent(**payload)

    def test_startup_event_missing_required_field(self):
        payload = {
            "event": "startup",
            "schema_version": "1",
            "instance_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "py": "3.12",
            "os": "linux",
            "arch": "x86_64",
            "mode": "with-gateway",
            "registry_mode": "full",
            "storage": "documentdb",
            "auth": "keycloak",
            "federation": True,
            "ts": "2026-03-18T00:00:00Z",
        }
        with pytest.raises(ValidationError):
            StartupEvent(**payload)

    def test_heartbeat_event_valid(self):
        payload = {
            "event": "heartbeat",
            "schema_version": "1",
            "instance_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "v": "1.0.16",
            "servers_count": 15,
            "agents_count": 8,
            "skills_count": 23,
            "peers_count": 2,
            "search_backend": "documentdb",
            "embeddings_provider": "sentence-transformers",
            "uptime_hours": 48,
            "ts": "2026-03-18T12:00:00Z",
        }
        event = HeartbeatEvent(**payload)
        assert event.event == "heartbeat"
        assert event.servers_count == 15

    def test_heartbeat_event_negative_count(self):
        payload = {
            "event": "heartbeat",
            "schema_version": "1",
            "instance_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "v": "1.0.16",
            "servers_count": -5,
            "agents_count": 8,
            "skills_count": 23,
            "peers_count": 2,
            "search_backend": "documentdb",
            "embeddings_provider": "sentence-transformers",
            "uptime_hours": 48,
            "ts": "2026-03-18T12:00:00Z",
        }
        with pytest.raises(ValidationError):
            HeartbeatEvent(**payload)


class TestIPHashing:
    """Test IP hashing for privacy-preserving rate limiting."""

    def test_hash_ip_consistent(self):
        hash1 = _hash_ip("192.168.1.100")
        hash2 = _hash_ip("192.168.1.100")
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_hash_ip_different_ips(self):
        assert _hash_ip("192.168.1.100") != _hash_ip("192.168.1.101")


class TestRateLimiting:
    """Test rate limiting logic with DynamoDB."""

    @patch("index.dynamodb")
    def test_rate_limit_allows_new_entry(self, mock_dynamodb):
        """First request in a new window succeeds (reset path)."""
        mock_table = MagicMock()
        mock_dynamodb.Table.return_value = mock_table
        # First update_item succeeds (window expired or new entry)
        mock_table.update_item.return_value = {}

        assert _check_rate_limit("abc123") is True

    @patch("index.dynamodb")
    def test_rate_limit_allows_within_window(self, mock_dynamodb):
        """Request within active window under limit succeeds."""
        mock_table = MagicMock()
        mock_dynamodb.Table.return_value = mock_table
        # First call: ConditionalCheckFailed (window still active)
        # Second call: succeeds (under limit)
        mock_table.update_item.side_effect = [
            ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "update_item"),
            {},
        ]

        assert _check_rate_limit("abc123") is True

    @patch("index.dynamodb")
    def test_rate_limit_blocks_request(self, mock_dynamodb):
        """Request over limit is blocked."""
        mock_table = MagicMock()
        mock_dynamodb.Table.return_value = mock_table
        # First call: ConditionalCheckFailed (window still active)
        # Second call: ConditionalCheckFailed (over limit)
        mock_table.update_item.side_effect = [
            ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "update_item"),
            ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "update_item"),
        ]

        assert _check_rate_limit("abc123") is False

    @patch("index.dynamodb")
    def test_rate_limit_fails_open_on_error(self, mock_dynamodb):
        """DynamoDB error fails open (allows request)."""
        mock_table = MagicMock()
        mock_dynamodb.Table.return_value = mock_table
        mock_table.update_item.side_effect = ClientError(
            {"Error": {"Code": "InternalServerError"}}, "update_item"
        )

        assert _check_rate_limit("abc123") is True


class TestDocumentDBConnection:
    """Test DocumentDB connection and credential retrieval."""

    @patch("index._init_aws_clients")
    @patch("index.secretsmanager")
    def test_get_credentials(self, mock_sm, _mock_init):
        mock_sm.get_secret_value.return_value = {
            "SecretString": json.dumps(
                {
                    "username": "telemetry_admin",
                    "password": "test_password",
                    "database": "telemetry",
                }
            )
        }
        creds = _get_credentials()
        assert creds["username"] == "telemetry_admin"
        assert creds["database"] == "telemetry"

    @patch("index.pymongo.MongoClient")
    @patch("index._get_credentials")
    def test_get_database(self, mock_creds, mock_client_cls):
        mock_creds.return_value = {
            "username": "admin",
            "password": "pass",
            "database": "telemetry",
        }
        mock_client = MagicMock()
        mock_client.server_info.return_value = {"version": "5.0.0"}
        mock_client.__getitem__ = MagicMock(return_value="mock_db")
        mock_client_cls.return_value = mock_client

        db = _get_database()
        assert db == "mock_db"
        mock_client_cls.assert_called_once()


class TestEventStorage:
    """Test event storage in DocumentDB."""

    @patch("index._get_database")
    def test_store_startup_event(self, mock_get_db):
        mock_collection = MagicMock()
        mock_collection.insert_one.return_value = MagicMock(inserted_id="123")
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        mock_get_db.return_value = mock_db

        _store_event("startup", {"event": "startup", "instance_id": "test-id", "v": "1.0.0"})

        mock_collection.insert_one.assert_called_once()
        call_args = mock_collection.insert_one.call_args[0][0]
        assert call_args["event"] == "startup"
        assert "received_at" in call_args


class TestLambdaHandler:
    """Test Lambda handler function."""

    @patch("index._store_event")
    @patch("index._verify_signature", return_value=True)
    @patch("index._check_rate_limit")
    @patch("index._hash_ip")
    def test_valid_startup_event(self, mock_hash, mock_rate, mock_verify, mock_store):
        mock_hash.return_value = "abc123"
        mock_rate.return_value = True

        event = {
            "requestContext": {"http": {"sourceIp": "1.2.3.4"}},
            "headers": {"x-telemetry-signature": "valid"},
            "body": json.dumps(
                {
                    "event": "startup",
                    "schema_version": "1",
                    "instance_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "v": "1.0.16",
                    "py": "3.12",
                    "os": "linux",
                    "arch": "x86_64",
                    "mode": "with-gateway",
                    "registry_mode": "full",
                    "storage": "file",
                    "auth": "keycloak",
                    "federation": False,
                    "ts": "2026-03-18T00:00:00Z",
                }
            ),
        }

        response = lambda_handler(event, {})
        assert response["statusCode"] == 204
        mock_store.assert_called_once()

    @patch("index._store_event")
    @patch("index._verify_signature", return_value=True)
    @patch("index._check_rate_limit")
    @patch("index._hash_ip")
    def test_valid_heartbeat_event(self, mock_hash, mock_rate, mock_verify, mock_store):
        mock_hash.return_value = "abc123"
        mock_rate.return_value = True

        event = {
            "requestContext": {"http": {"sourceIp": "1.2.3.4"}},
            "headers": {"x-telemetry-signature": "valid"},
            "body": json.dumps(
                {
                    "event": "heartbeat",
                    "schema_version": "1",
                    "instance_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "v": "1.0.16",
                    "servers_count": 10,
                    "agents_count": 5,
                    "skills_count": 20,
                    "peers_count": 1,
                    "search_backend": "faiss",
                    "embeddings_provider": "sentence-transformers",
                    "uptime_hours": 24,
                    "ts": "2026-03-18T12:00:00Z",
                }
            ),
        }

        response = lambda_handler(event, {})
        assert response["statusCode"] == 204
        mock_store.assert_called_once()

    @patch("index._check_rate_limit")
    @patch("index._hash_ip")
    def test_rate_limited_returns_204(self, mock_hash, mock_rate):
        mock_hash.return_value = "abc123"
        mock_rate.return_value = False

        event = {
            "requestContext": {"http": {"sourceIp": "1.2.3.4"}},
            "body": json.dumps({"event": "startup"}),
        }

        assert lambda_handler(event, {})["statusCode"] == 204

    @patch("index._hash_ip")
    def test_invalid_json_returns_204(self, mock_hash):
        mock_hash.return_value = "abc123"

        event = {
            "requestContext": {"http": {"sourceIp": "1.2.3.4"}},
            "body": "invalid json",
        }

        assert lambda_handler(event, {})["statusCode"] == 204

    @patch("index._check_rate_limit")
    @patch("index._hash_ip")
    def test_unknown_event_type_returns_204(self, mock_hash, mock_rate):
        mock_hash.return_value = "abc123"
        mock_rate.return_value = True

        event = {
            "requestContext": {"http": {"sourceIp": "1.2.3.4"}},
            "body": json.dumps({"event": "unknown_type"}),
        }

        assert lambda_handler(event, {})["statusCode"] == 204

    @patch("index._store_event", side_effect=Exception("DB down"))
    @patch("index._verify_signature", return_value=True)
    @patch("index._check_rate_limit", return_value=True)
    @patch("index._hash_ip", return_value="abc123")
    def test_storage_failure_returns_204(self, mock_hash, mock_rate, mock_verify, mock_store):
        event = {
            "requestContext": {"http": {"sourceIp": "1.2.3.4"}},
            "headers": {"x-telemetry-signature": "valid"},
            "body": json.dumps(
                {
                    "event": "startup",
                    "schema_version": "1",
                    "instance_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "v": "1.0.16",
                    "py": "3.12",
                    "os": "linux",
                    "arch": "x86_64",
                    "mode": "with-gateway",
                    "registry_mode": "full",
                    "storage": "file",
                    "auth": "keycloak",
                    "federation": False,
                    "ts": "2026-03-18T00:00:00Z",
                }
            ),
        }

        assert lambda_handler(event, {})["statusCode"] == 204
