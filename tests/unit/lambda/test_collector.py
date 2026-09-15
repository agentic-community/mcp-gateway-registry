"""
Unit tests for telemetry collector Lambda function.

Tests validation, rate limiting, storage, and fail-silent behavior.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import boto3  # noqa: F401  (see the sys.path note below)
import pymongo  # noqa: F401  (see the sys.path note below)
import pytest
from botocore.exceptions import ClientError
from pydantic import ValidationError

# Add the Lambda collector to the path for imports. That directory is a
# deployment bundle carrying vendored copies of pydantic, pymongo and boto3, so
# two precautions keep it from leaking into the rest of the session: the venv's
# copies of the shadowed packages are imported above first, so index.py's
# imports are sys.modules hits, and the path comes back off as soon as the two
# modules under test are loaded. Without either one, a vendored pymongo ends up
# in sys.modules and every later `import motor` dies on
# ModuleNotFoundError: No module named 'pymongo.cursor_shared'.
lambda_path = (
    Path(__file__).parent.parent.parent.parent
    / "terraform"
    / "telemetry-collector"
    / "lambda"
    / "collector"
)
sys.path.insert(0, str(lambda_path))
try:
    from index import (  # noqa: E402
        _check_rate_limit,
        _get_credentials,
        _get_database,
        _hash_ip,
        _store_event,
        lambda_handler,
    )
    from schemas import HeartbeatEvent, StartupEvent  # noqa: E402
finally:
    sys.path.remove(str(lambda_path))


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

    # ---- Schema v3 cloud_detection_method backwards-compat + validator ----

    @staticmethod
    def _v2_startup_payload() -> dict:
        """A minimal pre-v3 startup payload (no cloud_detection_method)."""
        return {
            "event": "startup",
            "schema_version": "2",
            "v": "1.0.22",
            "py": "3.12",
            "os": "linux",
            "arch": "x86_64",
            "cloud": "aws",
            "compute": "ecs",
            "mode": "with-gateway",
            "registry_mode": "full",
            "storage": "documentdb",
            "auth": "keycloak",
            "federation": True,
            "ts": "2026-03-18T00:00:00Z",
        }

    @staticmethod
    def _v2_heartbeat_payload() -> dict:
        return {
            "event": "heartbeat",
            "schema_version": "2",
            "v": "1.0.22",
            "cloud": "aws",
            "compute": "ecs",
            "servers_count": 15,
            "agents_count": 8,
            "skills_count": 23,
            "peers_count": 2,
            "search_backend": "documentdb",
            "embeddings_provider": "sentence-transformers",
            "uptime_hours": 48,
            "ts": "2026-03-18T12:00:00Z",
        }

    def test_startup_accepts_pre_v3_payload_without_detection_method(self):
        """Pre-v3 clients must still validate after the v3 schema change."""
        event = StartupEvent(**self._v2_startup_payload())
        assert event.cloud_detection_method is None

    def test_heartbeat_accepts_pre_v3_payload_without_detection_method(self):
        event = HeartbeatEvent(**self._v2_heartbeat_payload())
        assert event.cloud_detection_method is None

    def test_startup_accepts_v3_payload_with_imds_method(self):
        payload = self._v2_startup_payload()
        payload["schema_version"] = "3"
        payload["cloud_detection_method"] = "imds"
        event = StartupEvent(**payload)
        assert event.cloud_detection_method == "imds"

    def test_startup_rejects_ecs_meta_with_non_aws_cloud(self):
        payload = self._v2_startup_payload()
        payload["schema_version"] = "3"
        payload["cloud"] = "gcp"
        payload["cloud_detection_method"] = "ecs_meta"
        with pytest.raises(ValidationError):
            StartupEvent(**payload)

    def test_startup_rejects_unknown_method_with_known_cloud(self):
        payload = self._v2_startup_payload()
        payload["schema_version"] = "3"
        payload["cloud"] = "aws"
        payload["cloud_detection_method"] = "unknown"
        with pytest.raises(ValidationError):
            StartupEvent(**payload)

    def test_startup_rejects_unknown_cloud_with_non_unknown_method(self):
        payload = self._v2_startup_payload()
        payload["schema_version"] = "3"
        payload["cloud"] = "unknown"
        payload["cloud_detection_method"] = "imds"
        with pytest.raises(ValidationError):
            StartupEvent(**payload)

    def test_heartbeat_validator_enforces_same_rules(self):
        payload = self._v2_heartbeat_payload()
        payload["schema_version"] = "3"
        payload["cloud"] = "azure"
        payload["cloud_detection_method"] = "ecs_meta"
        with pytest.raises(ValidationError):
            HeartbeatEvent(**payload)

    def test_invalid_detection_method_rejected_by_pattern(self):
        payload = self._v2_startup_payload()
        payload["schema_version"] = "3"
        payload["cloud_detection_method"] = "wild-guess"
        with pytest.raises(ValidationError):
            StartupEvent(**payload)

    # ---- Schema v4 deployment-shape fields on heartbeat events ----

    def test_heartbeat_accepts_v4_payload_with_deployment_shape(self):
        """Schema v4 heartbeat carries auth/arch/os/py/mode/registry_mode/storage/federation."""
        payload = self._v2_heartbeat_payload()
        payload["schema_version"] = "4"
        payload["py"] = "3.12"
        payload["os"] = "linux"
        payload["arch"] = "x86_64"
        payload["mode"] = "with-gateway"
        payload["registry_mode"] = "full"
        payload["storage"] = "documentdb"
        payload["auth"] = "keycloak"
        payload["federation"] = True
        event = HeartbeatEvent(**payload)
        assert event.auth == "keycloak"
        assert event.arch == "x86_64"
        assert event.federation is True
        assert event.mode == "with-gateway"

    def test_heartbeat_v4_fields_are_optional(self):
        """Pre-v4 clients omit the new fields; the schema must still accept them."""
        event = HeartbeatEvent(**self._v2_heartbeat_payload())
        assert event.auth is None
        assert event.arch is None
        assert event.federation is None

    def test_heartbeat_rejects_invalid_auth_storage(self):
        """v4 fields are still validated when present (storage pattern, etc)."""
        payload = self._v2_heartbeat_payload()
        payload["schema_version"] = "4"
        payload["storage"] = "redis"  # not in the allowed pattern
        with pytest.raises(ValidationError):
            HeartbeatEvent(**payload)

    # ---- Schema v5 internal/workshop deployment classification (issue #1216) ----

    def test_startup_accepts_v5_internal_classification(self):
        """Schema v5 startup carries internal_only_deployment + internal_deployment_type."""
        payload = self._v2_startup_payload()
        payload["schema_version"] = "5"
        payload["internal_only_deployment"] = True
        payload["internal_deployment_type"] = "workshop"
        event = StartupEvent(**payload)
        assert event.internal_only_deployment is True
        assert event.internal_deployment_type == "workshop"

    def test_heartbeat_accepts_v5_internal_classification(self):
        """Schema v5 heartbeat carries the same classification fields."""
        payload = self._v2_heartbeat_payload()
        payload["schema_version"] = "5"
        payload["internal_only_deployment"] = True
        payload["internal_deployment_type"] = "dev"
        event = HeartbeatEvent(**payload)
        assert event.internal_only_deployment is True
        assert event.internal_deployment_type == "dev"

    def test_v5_classification_fields_are_optional(self):
        """Pre-v5 clients omit these fields; both schemas must still accept them."""
        startup = StartupEvent(**self._v2_startup_payload())
        heartbeat = HeartbeatEvent(**self._v2_heartbeat_payload())
        assert startup.internal_only_deployment is None
        assert startup.internal_deployment_type is None
        assert heartbeat.internal_only_deployment is None
        assert heartbeat.internal_deployment_type is None

    def test_rejects_invalid_internal_deployment_type(self):
        """internal_deployment_type is pattern-validated when present."""
        payload = self._v2_startup_payload()
        payload["schema_version"] = "5"
        payload["internal_deployment_type"] = "production"  # not in the allowed set
        with pytest.raises(ValidationError):
            StartupEvent(**payload)

    # ---- Schema v6 observed auth-path mix (issue #1753) ----

    @staticmethod
    def _v6_auth_path_fields() -> dict:
        return {
            "schema_version": "6",
            "auth_path_share_24h": {"session_cookie": 62, "keycloak": 31, "unknown": 7},
            "auth_path_volume_bucket_24h": "100-999",
            "auth_path_window_hours": 21,
        }

    def test_auth_path_fields_survive_model_dump(self):
        """The stored document must carry the mix and the schema version.

        index.py persists validated.model_dump(), and Pydantic drops undeclared
        keys. Before these fields were declared, a working flush and a dead one
        looked identical from inside the repo: the fleet view was empty either
        way.
        """
        payload = self._v2_heartbeat_payload()
        payload.update(self._v6_auth_path_fields())

        dumped = HeartbeatEvent(**payload).model_dump()

        for key, value in self._v6_auth_path_fields().items():
            assert key in dumped, f"{key} dropped before storage"
            assert dumped[key] == value, f"{key} altered: {dumped[key]!r}"

    def test_pre_v6_payload_still_validates(self):
        """The change is additive: an old client must not start failing."""
        event = HeartbeatEvent(**self._v2_heartbeat_payload())

        assert event.schema_version == "2"
        assert event.auth_path_share_24h is None
        assert event.auth_path_volume_bucket_24h is None
        assert event.auth_path_window_hours is None

    def test_unknown_share_key_dropped_not_rejected(self):
        """An unrecognised auth path is dropped, and the rest of the heartbeat lands.

        A client that learns a new auth path before this collector is redeployed
        would otherwise lose its whole payload, and the field must not become a
        free-text channel either.
        """
        payload = self._v2_heartbeat_payload()
        payload.update(self._v6_auth_path_fields())
        payload["auth_path_share_24h"] = {"session_cookie": 60, "quantum_handshake": 40}

        event = HeartbeatEvent(**payload)

        assert event.auth_path_share_24h == {"session_cookie": 60}
        assert event.servers_count == 15

    @pytest.mark.parametrize("share", [101, -1, 1000])
    def test_out_of_range_share_value_rejected(self, share):
        """A share outside 0..100 is not a percentage; reject the payload."""
        payload = self._v2_heartbeat_payload()
        payload.update(self._v6_auth_path_fields())
        payload["auth_path_share_24h"] = {"session_cookie": share}

        with pytest.raises(ValidationError):
            HeartbeatEvent(**payload)

    @pytest.mark.parametrize("bucket", ["42-99", "1k", "10k", "100-999 ", "", "10000+"])
    def test_invalid_volume_bucket_rejected(self, bucket):
        """Only the exact bucket strings the registry emits are accepted."""
        payload = self._v2_heartbeat_payload()
        payload.update(self._v6_auth_path_fields())
        payload["auth_path_volume_bucket_24h"] = bucket

        with pytest.raises(ValidationError):
            HeartbeatEvent(**payload)

    @pytest.mark.parametrize("hours", [49, 200, -1])
    def test_out_of_range_window_hours_rejected(self, hours):
        """The le=48 bound is why the registry clamps the window it reports.

        An unclamped window would cost the entire heartbeat, not one field, so
        the bound has to stay observable from this side too.
        """
        payload = self._v2_heartbeat_payload()
        payload.update(self._v6_auth_path_fields())
        payload["auth_path_window_hours"] = hours

        with pytest.raises(ValidationError):
            HeartbeatEvent(**payload)

    def test_null_auth_path_fields_accepted(self):
        """A window with no traffic sends all three as null, and that must validate."""
        payload = self._v2_heartbeat_payload()
        payload["schema_version"] = "6"
        payload["auth_path_share_24h"] = None
        payload["auth_path_volume_bucket_24h"] = None
        payload["auth_path_window_hours"] = None

        dumped = HeartbeatEvent(**payload).model_dump()

        assert dumped["schema_version"] == "6"
        assert dumped["auth_path_share_24h"] is None
        assert dumped["auth_path_volume_bucket_24h"] is None
        assert dumped["auth_path_window_hours"] is None


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
                    "search_backend": "documentdb",
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
