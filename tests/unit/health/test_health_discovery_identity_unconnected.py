"""A designated discovery identity with no vaulted token must be named, not probed.

The registry already detected this state and logged it. What it did not do was put
it anywhere an operator looks, and probing anyway actively hid it: the server needs
a credential to answer, so the probe failed, the tool fetch returned nothing, and
``num_tools`` KEPT ITS PREVIOUS VALUE. One deployment therefore read "active, 45
tools" for five hours after the vault lost the token, while the only evidence was
472 identical INFO lines.
"""

from unittest.mock import AsyncMock, patch

import pytest

from registry.health.service import HealthMonitoringService

_UNCONNECTED = "unhealthy: discovery identity designated but not connected"


def _designated(**overrides) -> dict:
    """A server whose discovery identity is designated (per-user, no static scheme)."""
    info = {
        "deployment": "remote",
        "proxy_pass_url": "https://api.example.test/mcp/",
        "auth_scheme": "none",
        "oauth_discovery": {
            "enabled": True,
            "auth_method": "oauth2",
            "user_id": "a811baaf",
            "oauth": {"provider": "github", "client_id": "Ov23x"},
        },
    }
    info.update(overrides)
    return info


@pytest.mark.unit
@pytest.mark.asyncio
class TestUnconnectedDiscoveryIdentityIsNamed:
    async def test_named_when_no_token_resolves(self):
        service = HealthMonitoringService()
        client = AsyncMock()
        # with_bearer resolving nothing is exactly the production state: the
        # designation is in Mongo, the token is not in the vault.
        with patch(
            "registry.health.service._with_backend_oauth",
            AsyncMock(side_effect=lambda info: info),
        ):
            await service._check_single_service(client, "/gh", _designated())

        assert service.server_health_status["/gh"] == _UNCONNECTED

    async def test_does_not_probe(self):
        """Probing is what produced the misleading healthy-with-stale-tools state."""
        service = HealthMonitoringService()
        client = AsyncMock()
        with patch(
            "registry.health.service._with_backend_oauth",
            AsyncMock(side_effect=lambda info: info),
        ):
            await service._check_single_service(client, "/gh", _designated())

        client.get.assert_not_called()
        client.post.assert_not_called()

    async def test_resolved_token_probes_as_normal(self):
        """A connected identity must be unaffected."""
        service = HealthMonitoringService()
        client = AsyncMock()

        async def _resolve(info):
            # Import the real key rather than hardcoding it: a renamed constant must
            # fail this test, not silently make it assert nothing.
            from registry.core.backend_oauth import RESOLVED_BEARER_KEY

            return {**info, RESOLVED_BEARER_KEY: "borrowed-token"}

        with patch("registry.health.service._with_backend_oauth", AsyncMock(side_effect=_resolve)):
            await service._check_single_service(client, "/gh", _designated())

        assert service.server_health_status["/gh"] != _UNCONNECTED

    async def test_static_credential_wins_and_is_not_marked_unhealthy(self):
        """The false positive this gate nearly shipped.

        resolve_discovery_bearer bows out when an explicit static scheme exists,
        because that operator-chosen credential wins. So a server with BOTH a
        designation and a stored bearer resolves no discovery token and is still
        perfectly healthy. Gating on the designation alone would mark every such
        server unhealthy and manufacture an outage.
        """
        service = HealthMonitoringService()
        client = AsyncMock()
        info = _designated(auth_scheme="bearer", auth_credential_encrypted="ENC")

        with patch(
            "registry.health.service._with_backend_oauth",
            AsyncMock(side_effect=lambda i: i),
        ):
            await service._check_single_service(client, "/gh", info)

        assert service.server_health_status["/gh"] != _UNCONNECTED

    async def test_no_discovery_config_is_unaffected(self):
        service = HealthMonitoringService()
        client = AsyncMock()
        info = _designated()
        info.pop("oauth_discovery")

        with patch(
            "registry.health.service._with_backend_oauth",
            AsyncMock(side_effect=lambda i: i),
        ):
            await service._check_single_service(client, "/gh", info)

        assert service.server_health_status["/gh"] != _UNCONNECTED

    async def test_discovery_disabled_is_unaffected(self):
        service = HealthMonitoringService()
        client = AsyncMock()
        info = _designated()
        info["oauth_discovery"]["enabled"] = False

        with patch(
            "registry.health.service._with_backend_oauth",
            AsyncMock(side_effect=lambda i: i),
        ):
            await service._check_single_service(client, "/gh", info)

        assert service.server_health_status["/gh"] != _UNCONNECTED


@pytest.mark.unit
class TestUnconnectedWarningIsRateLimited:
    """The health loop revisits every ~38s and this state never self-heals."""

    def setup_method(self):
        from registry.core import backend_oauth

        backend_oauth._unconnected_warned_at.clear()

    def test_first_call_warns_immediately(self, caplog):
        from registry.core import backend_oauth

        with caplog.at_level("WARNING", logger="registry.core.backend_oauth"):
            backend_oauth._warn_discovery_unconnected("/gh")

        assert sum("designated but not connected" in r.message for r in caplog.records) == 1

    def test_repeat_calls_are_suppressed(self, caplog):
        from registry.core import backend_oauth

        with caplog.at_level("WARNING", logger="registry.core.backend_oauth"):
            for _ in range(50):
                backend_oauth._warn_discovery_unconnected("/gh")

        warnings = [r for r in caplog.records if "designated but not connected" in r.message]
        assert len(warnings) == 1, f"expected 1 warning for 50 cycles, got {len(warnings)}"

    def test_each_server_warns_once(self, caplog):
        """Rate limiting is per server, so one noisy server cannot mask another."""
        from registry.core import backend_oauth

        with caplog.at_level("WARNING", logger="registry.core.backend_oauth"):
            for _ in range(10):
                backend_oauth._warn_discovery_unconnected("/gh")
                backend_oauth._warn_discovery_unconnected("/slack")

        paths = [r.message for r in caplog.records if "designated but not connected" in r.message]
        assert len(paths) == 2
        assert any("/gh" in m for m in paths) and any("/slack" in m for m in paths)

    def test_warns_again_after_the_interval(self, caplog, monkeypatch):
        from registry.core import backend_oauth

        monkeypatch.setattr(backend_oauth, "_UNCONNECTED_WARN_INTERVAL_SECONDS", 0)
        with caplog.at_level("WARNING", logger="registry.core.backend_oauth"):
            backend_oauth._warn_discovery_unconnected("/gh")
            backend_oauth._warn_discovery_unconnected("/gh")

        warnings = [r for r in caplog.records if "designated but not connected" in r.message]
        assert len(warnings) == 2, "a long-lived process must eventually re-warn"

    def test_it_is_a_warning_not_info(self, caplog):
        """At INFO it sat below the level anyone watches."""
        from registry.core import backend_oauth

        with caplog.at_level("INFO", logger="registry.core.backend_oauth"):
            backend_oauth._warn_discovery_unconnected("/gh")

        record = next(r for r in caplog.records if "designated but not connected" in r.message)
        assert record.levelname == "WARNING"
