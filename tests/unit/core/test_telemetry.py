"""Unit tests for telemetry module."""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from registry.core.telemetry import (
    STARTUP_LOCK_INTERVAL_SECONDS,
    TELEMETRY_TIMEOUT_SECONDS,
    TelemetryScheduler,
    _acquire_telemetry_lock,
    _build_heartbeat_payload,
    _build_startup_payload,
    _derive_embeddings_backend_kind,
    _get_heartbeat_interval_minutes,
    _get_heartbeat_lock_interval_seconds,
    _get_or_create_instance_id,
    _get_registry_id,
    _initialize_telemetry_collection,
    _is_heartbeat_enabled,
    _is_telemetry_enabled,
    _send_telemetry,
    send_startup_ping,
    start_heartbeat_scheduler,
)


class TestTelemetryEnabled:
    """Tests for telemetry enabled/disabled checks."""

    def test_telemetry_enabled_by_default(self, monkeypatch):
        """Test telemetry is enabled by default."""
        monkeypatch.delenv("MCP_TELEMETRY_DISABLED", raising=False)
        with patch("registry.core.telemetry.settings") as mock_settings:
            mock_settings.telemetry_enabled = True
            assert _is_telemetry_enabled() is True

    def test_telemetry_disabled_via_env_var(self, monkeypatch):
        """Test telemetry can be disabled via env var."""
        monkeypatch.setenv("MCP_TELEMETRY_DISABLED", "1")
        assert _is_telemetry_enabled() is False

    def test_telemetry_disabled_via_env_var_true(self, monkeypatch):
        """Test telemetry can be disabled via env var with 'true'."""
        monkeypatch.setenv("MCP_TELEMETRY_DISABLED", "true")
        assert _is_telemetry_enabled() is False

    def test_telemetry_disabled_via_env_var_yes(self, monkeypatch):
        """Test telemetry can be disabled via env var with 'yes'."""
        monkeypatch.setenv("MCP_TELEMETRY_DISABLED", "yes")
        assert _is_telemetry_enabled() is False

    def test_heartbeat_enabled_by_default(self, monkeypatch):
        """Test heartbeat is enabled by default (opt-out model)."""
        monkeypatch.delenv("MCP_TELEMETRY_DISABLED", raising=False)
        monkeypatch.delenv("MCP_TELEMETRY_OPT_OUT", raising=False)
        with patch("registry.core.telemetry.settings") as mock_settings:
            mock_settings.telemetry_enabled = True
            mock_settings.telemetry_opt_out = False
            assert _is_heartbeat_enabled() is True

    def test_heartbeat_disabled_via_opt_out_env_var(self, monkeypatch):
        """Test heartbeat can be disabled via MCP_TELEMETRY_OPT_OUT=1."""
        monkeypatch.delenv("MCP_TELEMETRY_DISABLED", raising=False)
        monkeypatch.setenv("MCP_TELEMETRY_OPT_OUT", "1")
        with patch("registry.core.telemetry.settings") as mock_settings:
            mock_settings.telemetry_enabled = True
            assert _is_heartbeat_enabled() is False

    def test_heartbeat_disabled_via_opt_out_true(self, monkeypatch):
        """Test heartbeat can be disabled via MCP_TELEMETRY_OPT_OUT=true."""
        monkeypatch.delenv("MCP_TELEMETRY_DISABLED", raising=False)
        monkeypatch.setenv("MCP_TELEMETRY_OPT_OUT", "true")
        with patch("registry.core.telemetry.settings") as mock_settings:
            mock_settings.telemetry_enabled = True
            assert _is_heartbeat_enabled() is False

    def test_heartbeat_disabled_via_opt_out_yes(self, monkeypatch):
        """Test heartbeat can be disabled via MCP_TELEMETRY_OPT_OUT=yes."""
        monkeypatch.delenv("MCP_TELEMETRY_DISABLED", raising=False)
        monkeypatch.setenv("MCP_TELEMETRY_OPT_OUT", "yes")
        with patch("registry.core.telemetry.settings") as mock_settings:
            mock_settings.telemetry_enabled = True
            assert _is_heartbeat_enabled() is False

    def test_heartbeat_disabled_when_telemetry_disabled(self, monkeypatch):
        """Test heartbeat is disabled when all telemetry is disabled."""
        monkeypatch.setenv("MCP_TELEMETRY_DISABLED", "1")
        monkeypatch.delenv("MCP_TELEMETRY_OPT_OUT", raising=False)
        assert _is_heartbeat_enabled() is False


class TestGetRegistryIdFallback:
    """Tests for _get_registry_id fallback to instance_id."""

    @pytest.mark.asyncio
    async def test_returns_card_id_when_available(self):
        """Registry card UUID takes precedence over instance_id."""
        mock_card = MagicMock()
        mock_card.id = "card-uuid-1234"

        mock_repo = MagicMock()
        mock_repo.get = AsyncMock(return_value=mock_card)

        with patch(
            "registry.repositories.factory.get_registry_card_repository",
            return_value=mock_repo,
        ):
            result = await _get_registry_id()
            assert result == "card-uuid-1234"

    @pytest.mark.asyncio
    async def test_falls_back_to_instance_id_when_no_card(self):
        """Falls back to telemetry instance_id when card is None."""
        mock_repo = MagicMock()
        mock_repo.get = AsyncMock(return_value=None)

        with (
            patch(
                "registry.repositories.factory.get_registry_card_repository",
                return_value=mock_repo,
            ),
            patch(
                "registry.core.telemetry._get_or_create_instance_id",
                new_callable=AsyncMock,
                return_value="instance-uuid-5678",
            ),
        ):
            result = await _get_registry_id()
            assert result == "instance-uuid-5678"

    @pytest.mark.asyncio
    async def test_falls_back_on_exception(self):
        """Falls back to instance_id when card repo throws."""
        with (
            patch(
                "registry.repositories.factory.get_registry_card_repository",
                side_effect=Exception("DB error"),
            ),
            patch(
                "registry.core.telemetry._get_or_create_instance_id",
                new_callable=AsyncMock,
                return_value="instance-uuid-fallback",
            ),
        ):
            result = await _get_registry_id()
            assert result == "instance-uuid-fallback"

    @pytest.mark.asyncio
    async def test_never_returns_none(self):
        """Verify _get_registry_id never returns None."""
        mock_repo = MagicMock()
        mock_repo.get = AsyncMock(return_value=None)

        with (
            patch(
                "registry.repositories.factory.get_registry_card_repository",
                return_value=mock_repo,
            ),
            patch(
                "registry.core.telemetry._get_or_create_instance_id",
                new_callable=AsyncMock,
                return_value="some-uuid",
            ),
        ):
            result = await _get_registry_id()
            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 0


class TestPayloadBuilding:
    """Tests for payload construction."""

    @pytest.mark.asyncio
    async def test_build_startup_payload_structure(self):
        """Test startup payload has correct fields."""
        with (
            patch("registry.core.telemetry.settings") as mock_settings,
            patch(
                "registry.repositories.stats_repository.get_search_counts",
                new_callable=AsyncMock,
                return_value={"total": 42, "last_24h": 5, "last_1h": 1},
            ),
        ):
            mock_settings.deployment_mode.value = "with-gateway"
            mock_settings.registry_mode.value = "full"
            mock_settings.storage_backend = "file"
            mock_settings.auth_provider = "cognito"
            mock_settings.federation_static_token_auth_enabled = False

            payload = await _build_startup_payload()

            # Required fields
            assert "event" in payload
            assert payload["event"] == "startup"
            assert "v" in payload  # Version
            assert "py" in payload  # Python version
            assert "os" in payload
            assert "arch" in payload
            assert "mode" in payload
            assert "registry_mode" in payload
            assert "storage" in payload
            assert "auth" in payload
            assert "federation" in payload
            assert "search_queries_total" in payload
            assert payload["search_queries_total"] == 42
            assert "ts" in payload

    @pytest.mark.asyncio
    async def test_no_pii_in_startup_payload(self):
        """Test startup payload contains no PII."""
        with (
            patch("registry.core.telemetry.settings") as mock_settings,
            patch(
                "registry.repositories.stats_repository.get_search_counts",
                new_callable=AsyncMock,
                return_value={"total": 0, "last_24h": 0, "last_1h": 0},
            ),
            patch(
                "registry.core.telemetry._get_registry_id",
                new_callable=AsyncMock,
                return_value="test-registry-id",
            ),
        ):
            mock_settings.deployment_mode.value = "with-gateway"
            mock_settings.registry_mode.value = "full"
            mock_settings.storage_backend = "file"
            mock_settings.auth_provider = "cognito"
            mock_settings.federation_static_token_auth_enabled = False
            mock_settings.embeddings_provider = "sentence-transformers"
            mock_settings.embeddings_model_name = "all-MiniLM-L6-v2"

            payload = await _build_startup_payload()
            payload_str = json.dumps(payload)

            # Should not contain hostnames, IPs
            assert "localhost" not in payload_str
            assert "127.0.0.1" not in payload_str

    @pytest.mark.asyncio
    async def test_build_heartbeat_payload_structure(self):
        """Test heartbeat payload has correct fields."""
        with (
            patch(
                "registry.api.system_routes.get_server_start_time",
                return_value=datetime.now(UTC),
            ),
            patch("registry.repositories.factory.get_server_repository") as mock_server_repo,
            patch("registry.repositories.factory.get_agent_repository") as mock_agent_repo,
            patch("registry.repositories.factory.get_skill_repository") as mock_skill_repo,
            patch("registry.repositories.factory.get_peer_federation_repository") as mock_peer_repo,
            patch("registry.core.telemetry.settings") as mock_settings,
            patch(
                "registry.repositories.stats_repository.get_search_counts",
                new_callable=AsyncMock,
                return_value={"total": 99, "last_24h": 10, "last_1h": 2},
            ),
        ):
            mock_settings.storage_backend = "file"
            mock_settings.embeddings_provider = "sentence-transformers"

            # Mock repository methods
            mock_server_repo_instance = MagicMock()
            mock_server_repo_instance.list_all = AsyncMock(return_value=[])
            mock_server_repo.return_value = mock_server_repo_instance

            mock_agent_repo_instance = MagicMock()
            mock_agent_repo_instance.list_all = AsyncMock(return_value=[])
            mock_agent_repo.return_value = mock_agent_repo_instance

            mock_skill_repo_instance = MagicMock()
            mock_skill_repo_instance.list_all = AsyncMock(return_value=[])
            mock_skill_repo.return_value = mock_skill_repo_instance

            mock_peer_repo_instance = MagicMock()
            mock_peer_repo_instance.list_peers = AsyncMock(return_value=[])
            mock_peer_repo.return_value = mock_peer_repo_instance

            payload = await _build_heartbeat_payload()

            # Required fields
            assert "event" in payload
            assert payload["event"] == "heartbeat"
            assert "v" in payload
            assert "servers_count" in payload
            assert "agents_count" in payload
            assert "skills_count" in payload
            assert "peers_count" in payload
            assert "search_backend" in payload
            assert "embeddings_provider" in payload
            assert "uptime_hours" in payload
            assert "search_queries_total" in payload
            assert payload["search_queries_total"] == 99
            assert "search_queries_24h" in payload
            assert "search_queries_1h" in payload
            assert "ts" in payload

    @pytest.mark.asyncio
    async def test_heartbeat_payload_search_backend_detection(self):
        """Test heartbeat payload correctly detects search backend."""
        with (
            patch("registry.api.system_routes.get_server_start_time", return_value=None),
            patch("registry.repositories.factory.get_server_repository") as mock_server_repo,
            patch("registry.repositories.factory.get_agent_repository") as mock_agent_repo,
            patch("registry.repositories.factory.get_skill_repository") as mock_skill_repo,
            patch("registry.repositories.factory.get_peer_federation_repository") as mock_peer_repo,
            patch("registry.core.telemetry.settings") as mock_settings,
            patch(
                "registry.repositories.stats_repository.get_search_counts",
                new_callable=AsyncMock,
                return_value={"total": 0, "last_24h": 0, "last_1h": 0},
            ),
        ):
            # Test DocumentDB backend
            mock_settings.storage_backend = "documentdb"
            mock_settings.embeddings_provider = "litellm"

            # Mock repository methods
            for repo in [
                mock_server_repo,
                mock_agent_repo,
                mock_skill_repo,
                mock_peer_repo,
            ]:
                repo_instance = MagicMock()
                if repo == mock_peer_repo:
                    repo_instance.list_peers = AsyncMock(return_value=[])
                else:
                    repo_instance.list_all = AsyncMock(return_value=[])
                repo.return_value = repo_instance

            payload = await _build_heartbeat_payload()
            assert payload["search_backend"] == "documentdb"

            # Test file backend (FAISS)
            mock_settings.storage_backend = "file"
            payload = await _build_heartbeat_payload()
            assert payload["search_backend"] == "faiss"


class TestInstanceID:
    """Tests for instance ID management."""

    @pytest.mark.asyncio
    async def test_instance_id_persistence_file_based(self, tmp_path, monkeypatch):
        """Test instance ID is stable across calls with file-based storage."""
        with patch("registry.core.telemetry.settings") as mock_settings:
            mock_settings.storage_backend = "file"
            mock_settings.data_dir = tmp_path

            # First call creates new ID
            id1 = await _get_or_create_instance_id()
            assert id1

            # Second call returns same ID
            id2 = await _get_or_create_instance_id()
            assert id1 == id2

    @pytest.mark.asyncio
    async def test_instance_id_file_creation(self, tmp_path):
        """Test instance ID file is created correctly."""
        with patch("registry.core.telemetry.settings") as mock_settings:
            mock_settings.storage_backend = "file"
            mock_settings.data_dir = tmp_path

            instance_id = await _get_or_create_instance_id()

            # Check file exists
            telemetry_file = tmp_path / ".telemetry_id"
            assert telemetry_file.exists()

            # Check file content
            file_content = telemetry_file.read_text().strip()
            assert file_content == instance_id


class TestLockAcquisition:
    """Tests for distributed lock mechanism."""

    @pytest.mark.asyncio
    async def test_acquire_lock_file_based_always_succeeds(self):
        """Test lock always succeeds for file-based storage."""
        with patch("registry.core.telemetry.settings") as mock_settings:
            mock_settings.storage_backend = "file"

            result = await _acquire_telemetry_lock("startup", 60)
            assert result is True

    @pytest.mark.asyncio
    async def test_acquire_lock_mongodb_success(self):
        """Test lock acquisition succeeds when not recently sent."""
        with patch("registry.core.telemetry.settings") as mock_settings:
            mock_settings.storage_backend = "mongodb-ce"

            # Mock MongoDB client
            with patch(
                "registry.repositories.documentdb.client.get_documentdb_client"
            ) as mock_get_client:
                mock_db = MagicMock()
                mock_collection = MagicMock()
                mock_db.__getitem__.return_value = mock_collection

                # find_one_and_update returns document (lock acquired)
                mock_collection.find_one_and_update = AsyncMock(
                    return_value={"_id": "telemetry_config"}
                )

                mock_get_client.return_value = mock_db

                result = await _acquire_telemetry_lock("startup", 60)
                assert result is True

    @pytest.mark.asyncio
    async def test_acquire_lock_mongodb_failure(self):
        """Test lock acquisition fails when recently sent."""
        with patch("registry.core.telemetry.settings") as mock_settings:
            mock_settings.storage_backend = "mongodb-ce"

            # Mock MongoDB client
            with patch(
                "registry.repositories.documentdb.client.get_documentdb_client"
            ) as mock_get_client:
                mock_db = MagicMock()
                mock_collection = MagicMock()
                mock_db.__getitem__.return_value = mock_collection

                # find_one_and_update returns None (lock not acquired)
                mock_collection.find_one_and_update = AsyncMock(return_value=None)

                mock_get_client.return_value = mock_db

                result = await _acquire_telemetry_lock("startup", 60)
                assert result is False


class TestSendTelemetry:
    """Tests for telemetry HTTP transmission."""

    @pytest.mark.asyncio
    async def test_send_telemetry_success(self, monkeypatch):
        """Test successful telemetry send."""
        monkeypatch.delenv("MCP_TELEMETRY_DISABLED", raising=False)

        with (
            patch("registry.core.telemetry.settings") as mock_settings,
            patch("registry.core.telemetry._get_or_create_instance_id") as mock_get_id,
            patch("registry.core.telemetry.httpx.AsyncClient") as mock_client_class,
        ):
            mock_settings.telemetry_debug = False
            mock_settings.telemetry_endpoint = "https://telemetry.example.com/v1/collect"
            mock_get_id.return_value = "test-uuid"

            # Mock successful HTTP response
            mock_response = MagicMock()
            mock_response.status_code = 204
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            payload = {"event": "startup", "v": "1.0.0"}
            await _send_telemetry(payload)

            # Verify HTTP call was made
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_telemetry_timeout(self, monkeypatch):
        """Test telemetry send handles timeout gracefully."""
        monkeypatch.delenv("MCP_TELEMETRY_DISABLED", raising=False)

        with (
            patch("registry.core.telemetry.settings") as mock_settings,
            patch("registry.core.telemetry._get_or_create_instance_id") as mock_get_id,
            patch("registry.core.telemetry.httpx.AsyncClient") as mock_client_class,
        ):
            mock_settings.telemetry_debug = False
            mock_settings.telemetry_endpoint = "https://telemetry.example.com/v1/collect"
            mock_get_id.return_value = "test-uuid"

            # Mock timeout exception
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_client_class.return_value = mock_client

            payload = {"event": "startup", "v": "1.0.0"}
            # Should not raise exception
            await _send_telemetry(payload)

    @pytest.mark.asyncio
    async def test_send_telemetry_debug_mode(self, monkeypatch, caplog):
        """Test debug mode logs payload instead of sending."""
        monkeypatch.delenv("MCP_TELEMETRY_DISABLED", raising=False)

        with (
            patch("registry.core.telemetry.settings") as mock_settings,
            patch("registry.core.telemetry._get_or_create_instance_id") as mock_get_id,
            patch("registry.core.telemetry.httpx.AsyncClient") as mock_client_class,
        ):
            mock_settings.telemetry_debug = True
            mock_get_id.return_value = "test-uuid"

            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            payload = {"event": "startup", "v": "1.0.0"}
            await _send_telemetry(payload)

            # HTTP client should not be called in debug mode
            mock_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_telemetry_retry_logic(self, monkeypatch):
        """Test telemetry retries once on failure."""
        monkeypatch.delenv("MCP_TELEMETRY_DISABLED", raising=False)

        with (
            patch("registry.core.telemetry.settings") as mock_settings,
            patch("registry.core.telemetry._get_or_create_instance_id") as mock_get_id,
            patch("registry.core.telemetry.httpx.AsyncClient") as mock_client_class,
            patch("registry.core.telemetry.asyncio.sleep") as mock_sleep,
        ):
            mock_settings.telemetry_debug = False
            mock_settings.telemetry_endpoint = "https://telemetry.example.com/v1/collect"
            mock_get_id.return_value = "test-uuid"

            # Mock exception on first call, success on second
            mock_response_success = MagicMock()
            mock_response_success.status_code = 204

            call_count = 0

            async def post_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception("Network error")
                return mock_response_success

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.post = AsyncMock(side_effect=post_side_effect)
            mock_client_class.return_value = mock_client

            payload = {"event": "startup", "v": "1.0.0"}
            await _send_telemetry(payload)

            # Should retry once and succeed
            assert call_count == 2
            mock_sleep.assert_called_once_with(1.0)


class TestScheduler:
    """Tests for TelemetryScheduler lifecycle."""

    @pytest.mark.asyncio
    async def test_scheduler_start_stop(self):
        """Test scheduler starts and stops cleanly."""
        scheduler = TelemetryScheduler()

        # Start scheduler
        await scheduler.start()
        assert scheduler._running is True
        assert scheduler._task is not None

        # Stop scheduler
        await scheduler.stop()
        assert scheduler._running is False
        assert scheduler._task is None

    @pytest.mark.asyncio
    async def test_scheduler_prevents_double_start(self):
        """Test scheduler prevents double start."""
        scheduler = TelemetryScheduler()

        await scheduler.start()
        first_task = scheduler._task

        # Try to start again
        await scheduler.start()
        second_task = scheduler._task

        # Should be same task
        assert first_task is second_task

        await scheduler.stop()


class TestInitialization:
    """Tests for telemetry initialization."""

    @pytest.mark.asyncio
    async def test_initialize_telemetry_file_based(self):
        """Test initialization with file-based storage does nothing."""
        with patch("registry.core.telemetry.settings") as mock_settings:
            mock_settings.storage_backend = "file"

            # Should not raise exception
            await _initialize_telemetry_collection()

    @pytest.mark.asyncio
    async def test_initialize_telemetry_creates_collection(self):
        """Test initialization creates MongoDB collection."""
        with patch("registry.core.telemetry.settings") as mock_settings:
            mock_settings.storage_backend = "mongodb-ce"

            with patch(
                "registry.repositories.documentdb.client.get_documentdb_client"
            ) as mock_get_client:
                mock_db = MagicMock()
                mock_collection = MagicMock()

                # Mock collection does not exist
                mock_db.list_collection_names = AsyncMock(return_value=[])
                mock_db.create_collection = AsyncMock()
                mock_db.__getitem__.return_value = mock_collection
                mock_collection.find_one = AsyncMock(return_value=None)
                mock_collection.insert_one = AsyncMock()

                mock_get_client.return_value = mock_db

                await _initialize_telemetry_collection()

                # Should create collection
                mock_db.create_collection.assert_called_once_with("_telemetry_state")


class TestPublicAPI:
    """Tests for public API functions."""

    @pytest.mark.asyncio
    async def test_send_startup_ping_disabled(self, monkeypatch, caplog):
        """Test startup ping skips when telemetry disabled."""
        import logging

        monkeypatch.setenv("MCP_TELEMETRY_DISABLED", "1")

        # Set logging level to capture INFO messages
        caplog.set_level(logging.INFO, logger="registry.core.telemetry")

        with patch("registry.core.telemetry.settings") as mock_settings:
            mock_settings.telemetry_enabled = False

            await send_startup_ping()

            # Should log disabled message
            assert "Telemetry is disabled" in caplog.text

    @pytest.mark.asyncio
    async def test_heartbeat_scheduler_starts_by_default(self, monkeypatch):
        """Test heartbeat scheduler starts by default (opt-out model)."""
        monkeypatch.delenv("MCP_TELEMETRY_DISABLED", raising=False)
        monkeypatch.delenv("MCP_TELEMETRY_OPT_OUT", raising=False)

        with patch("registry.core.telemetry.settings") as mock_settings:
            mock_settings.telemetry_enabled = True
            mock_settings.telemetry_opt_out = False
            mock_settings.telemetry_heartbeat_interval_minutes = 1440

            await start_heartbeat_scheduler()

            from registry.core.telemetry import _telemetry_scheduler

            # Scheduler should be started
            assert _telemetry_scheduler is not None

            # Clean up
            from registry.core.telemetry import stop_heartbeat_scheduler

            await stop_heartbeat_scheduler()

    @pytest.mark.asyncio
    async def test_heartbeat_scheduler_not_started_when_opted_out(self, monkeypatch):
        """Test heartbeat scheduler does not start when opted out."""
        monkeypatch.delenv("MCP_TELEMETRY_DISABLED", raising=False)
        monkeypatch.setenv("MCP_TELEMETRY_OPT_OUT", "1")

        with patch("registry.core.telemetry.settings") as mock_settings:
            mock_settings.telemetry_enabled = True

            await start_heartbeat_scheduler()

            from registry.core.telemetry import _telemetry_scheduler

            assert _telemetry_scheduler is None


class TestRepositoryFailures:
    """Tests for graceful error handling in repository calls."""

    @pytest.mark.asyncio
    async def test_heartbeat_repository_failure_logging(self, caplog):
        """Test repository failures log warnings with details."""
        with (
            patch("registry.api.system_routes.get_server_start_time", return_value=None),
            patch("registry.repositories.factory.get_server_repository") as mock_server_repo,
            patch("registry.repositories.factory.get_agent_repository") as mock_agent_repo,
            patch("registry.repositories.factory.get_skill_repository") as mock_skill_repo,
            patch("registry.repositories.factory.get_peer_federation_repository") as mock_peer_repo,
            patch("registry.core.telemetry.settings") as mock_settings,
            patch(
                "registry.repositories.stats_repository.get_search_counts",
                new_callable=AsyncMock,
                return_value={"total": 0, "last_24h": 0, "last_1h": 0},
            ),
        ):
            mock_settings.storage_backend = "file"
            mock_settings.embeddings_provider = "sentence-transformers"

            # Mock server repo to raise exception
            mock_server_repo_instance = MagicMock()
            mock_server_repo_instance.list_all = AsyncMock(side_effect=Exception("Database error"))
            mock_server_repo.return_value = mock_server_repo_instance

            # Other repos succeed
            mock_agent_repo_instance = MagicMock()
            mock_agent_repo_instance.list_all = AsyncMock(return_value=[])
            mock_agent_repo.return_value = mock_agent_repo_instance

            mock_skill_repo_instance = MagicMock()
            mock_skill_repo_instance.list_all = AsyncMock(return_value=[])
            mock_skill_repo.return_value = mock_skill_repo_instance

            mock_peer_repo_instance = MagicMock()
            mock_peer_repo_instance.list_peers = AsyncMock(return_value=[])
            mock_peer_repo.return_value = mock_peer_repo_instance

            payload = await _build_heartbeat_payload()

            # Should still return payload with zero server count
            assert payload["servers_count"] == 0
            # Should log warning
            assert "[telemetry] Failed to get server count" in caplog.text


class TestConstants:
    """Tests for telemetry constants and configurable intervals."""

    def test_telemetry_constants(self):
        """Test telemetry constants have expected values."""
        assert STARTUP_LOCK_INTERVAL_SECONDS == 60
        assert TELEMETRY_TIMEOUT_SECONDS == 5

    def test_heartbeat_interval_from_settings(self):
        """Test heartbeat interval reads from settings."""
        with patch("registry.core.telemetry.settings") as mock_settings:
            mock_settings.telemetry_heartbeat_interval_minutes = 1440
            assert _get_heartbeat_interval_minutes() == 1440

    def test_heartbeat_lock_interval_matches(self):
        """Test heartbeat lock interval = interval minutes * 60."""
        with patch("registry.core.telemetry.settings") as mock_settings:
            mock_settings.telemetry_heartbeat_interval_minutes = 1440
            assert _get_heartbeat_lock_interval_seconds() == 1440 * 60

    def test_custom_heartbeat_interval(self):
        """Test custom heartbeat interval is respected."""
        with patch("registry.core.telemetry.settings") as mock_settings:
            mock_settings.telemetry_heartbeat_interval_minutes = 5
            assert _get_heartbeat_interval_minutes() == 5
            assert _get_heartbeat_lock_interval_seconds() == 300


class TestDeriveEmbeddingsBackendKind:
    """Tests for _derive_embeddings_backend_kind() (issue #934)."""

    # sentence-transformers always wins regardless of model_name
    @pytest.mark.parametrize(
        "model_name",
        ["all-MiniLM-L6-v2", "BAAI/bge-large-en-v1.5", "", None],
    )
    def test_sentence_transformers_provider(self, model_name):
        assert (
            _derive_embeddings_backend_kind("sentence-transformers", model_name)
            == "sentence-transformers"
        )

    # Known prefixes map to the right backend kind
    @pytest.mark.parametrize(
        "model_name,expected",
        [
            # Bedrock
            ("bedrock/amazon.titan-embed-text-v2:0", "bedrock"),
            ("amazon.titan-embed-text-v2:0", "bedrock"),
            ("amazon-titan-v1", "bedrock"),
            # OpenAI
            ("openai/text-embedding-3-small", "openai"),
            ("text-embedding-3-large", "openai"),
            ("text-embedding-ada-002", "openai"),
            # Azure OpenAI
            ("azure/my-deployment-name", "azure-openai"),
            # Voyage
            ("voyage-large-2", "voyage"),
            ("voyage/voyage-3", "voyage"),
            # Cohere
            ("embed-english-v3.0", "cohere"),
            ("embed-multilingual-v3.0", "cohere"),
            ("cohere/embed-v3", "cohere"),
        ],
    )
    def test_known_prefixes_with_litellm(self, model_name, expected):
        assert _derive_embeddings_backend_kind("litellm", model_name) == expected

    def test_case_insensitive_matching(self):
        assert _derive_embeddings_backend_kind("litellm", "BEDROCK/AMAZON.TITAN-V1") == "bedrock"

    def test_whitespace_trimmed(self):
        assert (
            _derive_embeddings_backend_kind("litellm", "  bedrock/amazon.titan-v1  ") == "bedrock"
        )

    def test_litellm_unrecognized_model_falls_back_to_other(self):
        assert _derive_embeddings_backend_kind("litellm", "my-custom-proxy/model-x") == "other"

    def test_litellm_empty_model_is_unknown(self):
        assert _derive_embeddings_backend_kind("litellm", "") == "unknown"

    def test_litellm_none_model_is_unknown(self):
        assert _derive_embeddings_backend_kind("litellm", None) == "unknown"

    def test_unknown_provider_with_known_model_prefix_wins(self):
        # Model prefix wins even when provider is unrecognized
        assert _derive_embeddings_backend_kind("weird", "amazon.titan-v1") == "bedrock"

    def test_unknown_provider_and_unknown_model_is_unknown(self):
        assert _derive_embeddings_backend_kind("weird", "foo-bar") == "unknown"

    def test_return_value_always_in_allowlist(self):
        """The returned string must always be one of the 8 valid values."""
        allowlist = {
            "sentence-transformers",
            "bedrock",
            "openai",
            "azure-openai",
            "voyage",
            "cohere",
            "other",
            "unknown",
        }
        # Spot check a few representative inputs
        for provider, model in [
            ("sentence-transformers", "all-MiniLM-L6-v2"),
            ("litellm", "bedrock/amazon.titan-v1"),
            ("litellm", ""),
            ("litellm", "unknown-vendor/model"),
            ("weird", "foo"),
        ]:
            assert _derive_embeddings_backend_kind(provider, model) in allowlist


class TestEmbeddingsTelemetryFields:
    """Tests for the new embeddings fields in telemetry payloads (issue #934)."""

    @pytest.mark.asyncio
    async def test_startup_includes_embeddings_fields(self):
        """Startup payload must include embeddings_provider and embeddings_backend_kind."""
        with (
            patch("registry.core.telemetry.settings") as mock_settings,
            patch(
                "registry.repositories.stats_repository.get_search_counts",
                new_callable=AsyncMock,
                return_value={"total": 0, "last_24h": 0, "last_1h": 0},
            ),
            patch(
                "registry.core.telemetry._get_registry_id",
                new_callable=AsyncMock,
                return_value="test-registry-id",
            ),
        ):
            mock_settings.deployment_mode.value = "with-gateway"
            mock_settings.registry_mode.value = "full"
            mock_settings.storage_backend = "mongodb-ce"
            mock_settings.auth_provider = "keycloak"
            mock_settings.federation_static_token_auth_enabled = False
            mock_settings.embeddings_provider = "litellm"
            mock_settings.embeddings_model_name = "bedrock/amazon.titan-embed-text-v2:0"

            payload = await _build_startup_payload()

            assert payload["embeddings_provider"] == "litellm"
            assert payload["embeddings_backend_kind"] == "bedrock"
            assert payload["schema_version"] == "3"

    @pytest.mark.asyncio
    async def test_startup_payload_omits_raw_model_name_and_dimensions(self):
        """The raw model name and dimensions must NEVER appear in the payload."""
        with (
            patch("registry.core.telemetry.settings") as mock_settings,
            patch(
                "registry.repositories.stats_repository.get_search_counts",
                new_callable=AsyncMock,
                return_value={"total": 0, "last_24h": 0, "last_1h": 0},
            ),
            patch(
                "registry.core.telemetry._get_registry_id",
                new_callable=AsyncMock,
                return_value="test-registry-id",
            ),
        ):
            mock_settings.deployment_mode.value = "with-gateway"
            mock_settings.registry_mode.value = "full"
            mock_settings.storage_backend = "file"
            mock_settings.auth_provider = "keycloak"
            mock_settings.federation_static_token_auth_enabled = False
            mock_settings.embeddings_provider = "litellm"
            mock_settings.embeddings_model_name = "bedrock/amazon.titan-embed-text-v2:0"
            mock_settings.embeddings_model_dimensions = 1024

            payload = await _build_startup_payload()

            # Privacy assertions: these fields must NEVER be sent
            assert "embeddings_model_name" not in payload
            assert "embeddings_model_dimensions" not in payload
            assert "embeddings_api_key" not in payload
            assert "embeddings_secret_key" not in payload
            assert "embeddings_api_base" not in payload
            assert "embeddings_aws_region" not in payload

            # And the raw model name must not appear anywhere in the serialized JSON
            payload_json = json.dumps(payload)
            assert "titan-embed-text-v2" not in payload_json

    @pytest.mark.asyncio
    async def test_heartbeat_includes_backend_kind_and_keeps_provider(self):
        """Heartbeat must keep embeddings_provider AND include embeddings_backend_kind."""
        with (
            patch(
                "registry.api.system_routes.get_server_start_time",
                return_value=datetime.now(UTC),
            ),
            patch("registry.repositories.factory.get_server_repository") as mock_server_repo,
            patch("registry.repositories.factory.get_agent_repository") as mock_agent_repo,
            patch("registry.repositories.factory.get_skill_repository") as mock_skill_repo,
            patch("registry.repositories.factory.get_peer_federation_repository") as mock_peer_repo,
            patch("registry.core.telemetry.settings") as mock_settings,
            patch(
                "registry.repositories.stats_repository.get_search_counts",
                new_callable=AsyncMock,
                return_value={"total": 0, "last_24h": 0, "last_1h": 0},
            ),
            patch(
                "registry.core.telemetry._get_registry_id",
                new_callable=AsyncMock,
                return_value="test-registry-id",
            ),
        ):
            mock_settings.storage_backend = "documentdb"
            mock_settings.embeddings_provider = "sentence-transformers"
            mock_settings.embeddings_model_name = "all-MiniLM-L6-v2"
            mock_settings.embeddings_model_dimensions = 384

            # Mock repository methods
            for repo_mock in (mock_server_repo, mock_agent_repo, mock_skill_repo):
                instance = MagicMock()
                instance.list_all = AsyncMock(return_value=[])
                repo_mock.return_value = instance
            peer_instance = MagicMock()
            peer_instance.list_peers = AsyncMock(return_value=[])
            mock_peer_repo.return_value = peer_instance

            payload = await _build_heartbeat_payload()

            assert payload["embeddings_provider"] == "sentence-transformers"
            assert payload["embeddings_backend_kind"] == "sentence-transformers"
            assert payload["schema_version"] == "3"

            # Privacy assertions
            assert "embeddings_model_name" not in payload
            assert "embeddings_model_dimensions" not in payload
            payload_json = json.dumps(payload)
            assert "MiniLM" not in payload_json


class TestStorageBackendAliasRouting:
    """Telemetry branching must treat every MONGODB_BACKENDS alias identically.

    Added for issue #954: mongodb and mongodb-atlas are aliases for mongodb-ce.
    The telemetry module branches in four places on storage_backend; if any
    one of them was missed during the MONGODB_BACKENDS rollout, the
    corresponding alias would behave wrongly. These tests lock in the new
    routing by running the same assertion with each alias.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("alias", ["mongodb-ce", "mongodb", "mongodb-atlas", "documentdb"])
    async def test_acquire_lock_hits_mongodb_branch_for_each_alias(self, alias):
        """Every MongoDB-compatible alias must take the MongoDB lock path."""
        with patch("registry.core.telemetry.settings") as mock_settings:
            mock_settings.storage_backend = alias

            with patch(
                "registry.repositories.documentdb.client.get_documentdb_client"
            ) as mock_get_client:
                mock_db = MagicMock()
                mock_collection = MagicMock()
                mock_db.__getitem__.return_value = mock_collection
                mock_collection.find_one_and_update = AsyncMock(
                    return_value={"_id": "telemetry_config"}
                )
                mock_get_client.return_value = mock_db

                result = await _acquire_telemetry_lock("startup", 60)

                assert result is True
                # The MongoDB branch must have been taken: get_documentdb_client called
                mock_get_client.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("alias", ["mongodb-ce", "mongodb", "mongodb-atlas", "documentdb"])
    async def test_initialize_collection_runs_for_each_alias(self, alias):
        """Every MongoDB-compatible alias must trigger _telemetry_state creation."""
        with patch("registry.core.telemetry.settings") as mock_settings:
            mock_settings.storage_backend = alias

            with patch(
                "registry.repositories.documentdb.client.get_documentdb_client"
            ) as mock_get_client:
                mock_db = MagicMock()
                mock_collection = MagicMock()
                mock_db.list_collection_names = AsyncMock(return_value=[])
                mock_db.create_collection = AsyncMock()
                mock_db.__getitem__.return_value = mock_collection
                mock_collection.find_one = AsyncMock(return_value=None)
                mock_collection.insert_one = AsyncMock()
                mock_get_client.return_value = mock_db

                await _initialize_telemetry_collection()

                # Every alias must reach the create_collection call
                mock_db.create_collection.assert_called_once_with("_telemetry_state")
