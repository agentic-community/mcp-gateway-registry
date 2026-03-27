"""
End-to-end integration tests for telemetry opt-in/opt-out and detailed information mode.

Tests cover:
- Opt-out: MCP_TELEMETRY_DISABLED=1 suppresses all telemetry
- Default state: startup ping sent, heartbeat NOT started (opt-in off)
- Opt-in (detailed mode): startup ping + heartbeat scheduler both active
- Debug mode: payloads logged, no network call made

Live AWS tests (require deployed collector + AWS credentials) are marked
with @pytest.mark.live and skipped in CI. Run manually with:
    uv run pytest tests/integration/test_telemetry_e2e.py -v -s -m live --no-cov
"""

import asyncio
import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_settings(
    storage_backend: str = "file",
    telemetry_enabled: bool = True,
    telemetry_opt_in: bool = False,
    telemetry_debug: bool = False,
    telemetry_endpoint: str = "https://telemetry.mcpgateway.io/v1/collect",
    embeddings_provider: str = "sentence-transformers",
    deployment_mode: str = "with-gateway",
    registry_mode: str = "full",
    auth_provider: str = "none",
    federation_static_token_auth_enabled: bool = False,
):
    """Return a configured MagicMock for settings."""
    mock = MagicMock()
    mock.storage_backend = storage_backend
    mock.telemetry_enabled = telemetry_enabled
    mock.telemetry_opt_in = telemetry_opt_in
    mock.telemetry_debug = telemetry_debug
    mock.telemetry_endpoint = telemetry_endpoint
    mock.embeddings_provider = embeddings_provider
    mock.deployment_mode.value = deployment_mode
    mock.registry_mode.value = registry_mode
    mock.auth_provider = auth_provider
    mock.federation_static_token_auth_enabled = federation_static_token_auth_enabled
    return mock


def _mock_repo_factory():
    """Return mock repository that returns empty lists."""
    repo = MagicMock()
    repo.list_all = AsyncMock(return_value=[])
    repo.list_peers = AsyncMock(return_value=[])
    return repo


# ---------------------------------------------------------------------------
# Class 1: Opt-out behaviour
# ---------------------------------------------------------------------------


class TestOptOut:
    """Verify that MCP_TELEMETRY_DISABLED=1 suppresses all telemetry."""

    async def test_startup_ping_not_sent_when_disabled(self, monkeypatch):
        """No HTTP request made when telemetry is disabled."""
        monkeypatch.setenv("MCP_TELEMETRY_DISABLED", "1")
        monkeypatch.delenv("MCP_TELEMETRY_OPT_IN", raising=False)

        from registry.core.telemetry import _is_telemetry_enabled, send_startup_ping

        assert _is_telemetry_enabled() is False

        with patch("registry.core.telemetry.settings", _mock_settings()):
            with patch("registry.core.telemetry._send_telemetry") as mock_send:
                await send_startup_ping()
                mock_send.assert_not_called()

    async def test_heartbeat_not_started_when_disabled(self, monkeypatch):
        """Heartbeat scheduler does not start when telemetry is disabled."""
        monkeypatch.setenv("MCP_TELEMETRY_DISABLED", "1")
        monkeypatch.setenv("MCP_TELEMETRY_OPT_IN", "1")

        from registry.core.telemetry import _is_opt_in_enabled, start_heartbeat_scheduler

        assert _is_opt_in_enabled() is False

        with patch("registry.core.telemetry.settings", _mock_settings(telemetry_enabled=False)):
            with patch("registry.core.telemetry._send_telemetry") as mock_send:
                await start_heartbeat_scheduler()
                mock_send.assert_not_called()

    async def test_disabled_via_env_var_true_string(self, monkeypatch):
        """MCP_TELEMETRY_DISABLED=true also disables telemetry."""
        monkeypatch.setenv("MCP_TELEMETRY_DISABLED", "true")
        from registry.core.telemetry import _is_telemetry_enabled

        assert _is_telemetry_enabled() is False

    async def test_disabled_via_env_var_yes_string(self, monkeypatch):
        """MCP_TELEMETRY_DISABLED=yes also disables telemetry."""
        monkeypatch.setenv("MCP_TELEMETRY_DISABLED", "yes")
        from registry.core.telemetry import _is_telemetry_enabled

        assert _is_telemetry_enabled() is False

    async def test_enabled_by_default_no_env_var(self, monkeypatch):
        """Telemetry is enabled when no disable env var is set."""
        monkeypatch.delenv("MCP_TELEMETRY_DISABLED", raising=False)
        with patch("registry.core.telemetry.settings", _mock_settings(telemetry_enabled=True)):
            from registry.core.telemetry import _is_telemetry_enabled

            assert _is_telemetry_enabled() is True


# ---------------------------------------------------------------------------
# Class 2: Default state (no opt-in)
# ---------------------------------------------------------------------------


class TestDefaultState:
    """Without MCP_TELEMETRY_OPT_IN, only the startup ping fires."""

    async def test_startup_ping_sent_by_default(self, monkeypatch):
        """Startup ping is sent when telemetry is enabled (default)."""
        monkeypatch.delenv("MCP_TELEMETRY_DISABLED", raising=False)
        monkeypatch.delenv("MCP_TELEMETRY_OPT_IN", raising=False)

        captured = []

        async def fake_send(payload):
            captured.append(payload)

        with (
            patch("registry.core.telemetry.settings", _mock_settings()),
            patch("registry.core.telemetry._send_telemetry", side_effect=fake_send),
            patch(
                "registry.core.telemetry._acquire_telemetry_lock",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "registry.core.telemetry._get_or_create_instance_id",
                new=AsyncMock(return_value="test-instance-id"),
            ),
        ):
            from registry.core.telemetry import send_startup_ping

            await send_startup_ping()

        assert len(captured) == 1
        assert captured[0]["event"] == "startup"

    async def test_opt_in_disabled_by_default(self, monkeypatch):
        """opt-in is off unless MCP_TELEMETRY_OPT_IN=1 is set."""
        monkeypatch.delenv("MCP_TELEMETRY_OPT_IN", raising=False)
        with patch(
            "registry.core.telemetry.settings",
            _mock_settings(telemetry_opt_in=False),
        ):
            from registry.core.telemetry import _is_opt_in_enabled

            assert _is_opt_in_enabled() is False

    async def test_heartbeat_not_sent_without_opt_in(self, monkeypatch):
        """Heartbeat scheduler exits immediately when opt-in is off."""
        monkeypatch.delenv("MCP_TELEMETRY_DISABLED", raising=False)
        monkeypatch.delenv("MCP_TELEMETRY_OPT_IN", raising=False)

        with patch(
            "registry.core.telemetry.settings",
            _mock_settings(telemetry_opt_in=False),
        ):
            with patch("registry.core.telemetry._send_telemetry") as mock_send:
                from registry.core.telemetry import start_heartbeat_scheduler

                await start_heartbeat_scheduler()
                mock_send.assert_not_called()

    async def test_startup_payload_fields(self, monkeypatch):
        """Startup payload contains all required schema fields."""
        monkeypatch.delenv("MCP_TELEMETRY_DISABLED", raising=False)

        with (
            patch("registry.core.telemetry.settings", _mock_settings()),
            patch(
                "registry.repositories.stats_repository.get_search_counts",
                new_callable=AsyncMock,
                return_value={"total": 0, "last_24h": 0, "last_1h": 0},
            ),
        ):
            from registry.core.telemetry import _build_startup_payload

            payload = await _build_startup_payload()

        required_fields = {
            "event",
            "schema_version",
            "v",
            "py",
            "os",
            "arch",
            "mode",
            "registry_mode",
            "storage",
            "auth",
            "federation",
            "search_queries_total",
            "ts",
        }
        assert required_fields.issubset(payload.keys())
        assert payload["event"] == "startup"


# ---------------------------------------------------------------------------
# Class 3: Opt-in / detailed information mode
# ---------------------------------------------------------------------------


class TestOptIn:
    """MCP_TELEMETRY_OPT_IN=1 enables the daily heartbeat with aggregate counts."""

    async def test_opt_in_enables_heartbeat(self, monkeypatch):
        """opt-in flag enables the heartbeat path."""
        monkeypatch.delenv("MCP_TELEMETRY_DISABLED", raising=False)
        monkeypatch.setenv("MCP_TELEMETRY_OPT_IN", "1")

        with patch(
            "registry.core.telemetry.settings",
            _mock_settings(telemetry_opt_in=True),
        ):
            from registry.core.telemetry import _is_opt_in_enabled

            assert _is_opt_in_enabled() is True

    async def test_heartbeat_payload_fields(self, monkeypatch):
        """Heartbeat (detailed mode) payload contains all required schema fields."""
        monkeypatch.delenv("MCP_TELEMETRY_DISABLED", raising=False)
        monkeypatch.setenv("MCP_TELEMETRY_OPT_IN", "1")

        repo = _mock_repo_factory()

        with (
            patch("registry.core.telemetry.settings", _mock_settings()),
            patch(
                "registry.api.system_routes.get_server_start_time",
                return_value=datetime.now(UTC),
            ),
            patch("registry.repositories.factory.get_server_repository", return_value=repo),
            patch("registry.repositories.factory.get_agent_repository", return_value=repo),
            patch("registry.repositories.factory.get_skill_repository", return_value=repo),
            patch(
                "registry.repositories.factory.get_peer_federation_repository",
                return_value=repo,
            ),
            patch(
                "registry.repositories.stats_repository.get_search_counts",
                new_callable=AsyncMock,
                return_value={"total": 0, "last_24h": 0, "last_1h": 0},
            ),
        ):
            from registry.core.telemetry import _build_heartbeat_payload

            payload = await _build_heartbeat_payload()

        required_fields = {
            "event",
            "schema_version",
            "v",
            "servers_count",
            "agents_count",
            "skills_count",
            "peers_count",
            "search_backend",
            "embeddings_provider",
            "uptime_hours",
            "search_queries_total",
            "ts",
        }
        assert required_fields.issubset(payload.keys())
        assert payload["event"] == "heartbeat"
        assert isinstance(payload["servers_count"], int)
        assert isinstance(payload["agents_count"], int)
        assert isinstance(payload["uptime_hours"], int)

    async def test_heartbeat_payload_search_backend_file(self, monkeypatch):
        """File storage maps to 'faiss' search backend in heartbeat payload."""
        repo = _mock_repo_factory()

        with (
            patch("registry.core.telemetry.settings", _mock_settings(storage_backend="file")),
            patch("registry.api.system_routes.get_server_start_time", return_value=None),
            patch("registry.repositories.factory.get_server_repository", return_value=repo),
            patch("registry.repositories.factory.get_agent_repository", return_value=repo),
            patch("registry.repositories.factory.get_skill_repository", return_value=repo),
            patch(
                "registry.repositories.factory.get_peer_federation_repository",
                return_value=repo,
            ),
            patch(
                "registry.repositories.stats_repository.get_search_counts",
                new_callable=AsyncMock,
                return_value={"total": 0, "last_24h": 0, "last_1h": 0},
            ),
        ):
            from registry.core.telemetry import _build_heartbeat_payload

            payload = await _build_heartbeat_payload()

        assert payload["search_backend"] == "faiss"

    async def test_heartbeat_payload_search_backend_documentdb(self, monkeypatch):
        """DocumentDB storage maps to 'documentdb' search backend."""
        repo = _mock_repo_factory()

        with (
            patch(
                "registry.core.telemetry.settings",
                _mock_settings(storage_backend="documentdb"),
            ),
            patch("registry.api.system_routes.get_server_start_time", return_value=None),
            patch("registry.repositories.factory.get_server_repository", return_value=repo),
            patch("registry.repositories.factory.get_agent_repository", return_value=repo),
            patch("registry.repositories.factory.get_skill_repository", return_value=repo),
            patch(
                "registry.repositories.factory.get_peer_federation_repository",
                return_value=repo,
            ),
            patch(
                "registry.repositories.stats_repository.get_search_counts",
                new_callable=AsyncMock,
                return_value={"total": 0, "last_24h": 0, "last_1h": 0},
            ),
        ):
            from registry.core.telemetry import _build_heartbeat_payload

            payload = await _build_heartbeat_payload()

        assert payload["search_backend"] == "documentdb"

    async def test_both_startup_and_heartbeat_sent_when_opted_in(self, monkeypatch):
        """When opted in, startup ping fires AND heartbeat scheduler starts."""
        monkeypatch.delenv("MCP_TELEMETRY_DISABLED", raising=False)
        monkeypatch.setenv("MCP_TELEMETRY_OPT_IN", "1")

        events_sent = []

        async def fake_send(payload):
            events_sent.append(payload["event"])

        repo = _mock_repo_factory()

        with (
            patch("registry.core.telemetry.settings", _mock_settings(telemetry_opt_in=True)),
            patch("registry.core.telemetry._send_telemetry", side_effect=fake_send),
            patch(
                "registry.core.telemetry._acquire_telemetry_lock",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "registry.core.telemetry._get_or_create_instance_id",
                new=AsyncMock(return_value="test-instance-id"),
            ),
            patch(
                "registry.api.system_routes.get_server_start_time",
                return_value=datetime.now(UTC),
            ),
            patch("registry.repositories.factory.get_server_repository", return_value=repo),
            patch("registry.repositories.factory.get_agent_repository", return_value=repo),
            patch("registry.repositories.factory.get_skill_repository", return_value=repo),
            patch(
                "registry.repositories.factory.get_peer_federation_repository",
                return_value=repo,
            ),
            patch(
                "registry.repositories.stats_repository.get_search_counts",
                new_callable=AsyncMock,
                return_value={"total": 0, "last_24h": 0, "last_1h": 0},
            ),
        ):
            from registry.core.telemetry import (
                send_startup_ping,
                start_heartbeat_scheduler,
                stop_heartbeat_scheduler,
            )

            await send_startup_ping()
            await start_heartbeat_scheduler()
            await asyncio.sleep(0.5)
            await stop_heartbeat_scheduler()

        assert "startup" in events_sent
        assert "heartbeat" in events_sent

    async def test_heartbeat_not_sent_twice_within_lock_window(self, monkeypatch):
        """Lock mechanism prevents sending heartbeat twice within the 24-hour window."""
        monkeypatch.delenv("MCP_TELEMETRY_DISABLED", raising=False)
        monkeypatch.setenv("MCP_TELEMETRY_OPT_IN", "1")

        events_sent = []

        async def fake_send(payload):
            events_sent.append(payload["event"])

        repo = _mock_repo_factory()
        lock_results = iter([True, False])

        async def mock_lock(*args, **kwargs):
            return next(lock_results)

        with (
            patch("registry.core.telemetry.settings", _mock_settings(telemetry_opt_in=True)),
            patch("registry.core.telemetry._send_telemetry", side_effect=fake_send),
            patch("registry.core.telemetry._acquire_telemetry_lock", side_effect=mock_lock),
            patch(
                "registry.core.telemetry._get_or_create_instance_id",
                new=AsyncMock(return_value="test-instance-id"),
            ),
            patch(
                "registry.api.system_routes.get_server_start_time",
                return_value=datetime.now(UTC),
            ),
            patch("registry.repositories.factory.get_server_repository", return_value=repo),
            patch("registry.repositories.factory.get_agent_repository", return_value=repo),
            patch("registry.repositories.factory.get_skill_repository", return_value=repo),
            patch(
                "registry.repositories.factory.get_peer_federation_repository",
                return_value=repo,
            ),
            patch(
                "registry.repositories.stats_repository.get_search_counts",
                new_callable=AsyncMock,
                return_value={"total": 0, "last_24h": 0, "last_1h": 0},
            ),
        ):
            from registry.core.telemetry import TelemetryScheduler

            scheduler = TelemetryScheduler()
            await scheduler._send_heartbeat()
            await scheduler._send_heartbeat()

        assert events_sent.count("heartbeat") == 1


# ---------------------------------------------------------------------------
# Class 4: Debug mode
# ---------------------------------------------------------------------------


class TestDebugMode:
    """MCP_TELEMETRY_DEBUG=1 logs payloads without making network calls."""

    async def test_debug_mode_logs_not_sends(self, monkeypatch, caplog):
        """In debug mode, payload is logged and no HTTP call is made."""
        monkeypatch.delenv("MCP_TELEMETRY_DISABLED", raising=False)

        with (
            patch("registry.core.telemetry.settings", _mock_settings(telemetry_debug=True)),
            patch(
                "registry.core.telemetry._get_or_create_instance_id",
                new=AsyncMock(return_value="debug-instance"),
            ),
        ):
            with patch("httpx.AsyncClient") as mock_http:
                from registry.core.telemetry import _send_telemetry

                with caplog.at_level(logging.INFO, logger="registry.core.telemetry"):
                    await _send_telemetry({"event": "startup", "schema_version": "1"})

                mock_http.assert_not_called()

        assert "Debug mode" in caplog.text
        assert "startup" in caplog.text

    async def test_debug_mode_shows_full_payload(self, monkeypatch, caplog):
        """Debug mode logs the complete payload as formatted JSON."""
        with (
            patch("registry.core.telemetry.settings", _mock_settings(telemetry_debug=True)),
            patch(
                "registry.core.telemetry._get_or_create_instance_id",
                new=AsyncMock(return_value="debug-instance"),
            ),
        ):
            from registry.core.telemetry import _send_telemetry

            with caplog.at_level(logging.INFO, logger="registry.core.telemetry"):
                await _send_telemetry(
                    {"event": "heartbeat", "schema_version": "1", "servers_count": 42}
                )

        assert "heartbeat" in caplog.text
        assert "42" in caplog.text


# ---------------------------------------------------------------------------
# Live AWS tests — skipped in CI, run manually with -m live
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.skip(
    reason="Requires live AWS infrastructure — run manually with: pytest -m live --no-cov"
)
class TestLiveCollector:
    """Live tests against the deployed AWS collector. See DEMO-GUIDE.md."""

    pass
