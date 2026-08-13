"""Tests for per-server ingress audience acceptance in auth_server (issue #990).

``_obo_extra_audiences`` supplies the per-server resource URL(s) that
``EntraIdProvider.validate_token`` accepts as the token ``aud`` for the server
being accessed. The critical invariant (regression guard for the review finding):

    On Entra, a plain (non-egress) server advertises a per-server-resource PRM,
    so the client obtains a token audienced to that per-server resource. That
    audience MUST be accepted at /validate even when egress auth is DISABLED
    (the default) -- otherwise Entra plain-server IDE login mints the correct
    token and is then rejected with a 401. Acceptance is gated on whether a
    per-server PRM is advertised (mirrors registry.server_needs_per_server_prm),
    NOT on egress_auth_enabled.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.auth]


def _settings(*, auth_provider: str, egress_enabled: bool, registry_url: str = "") -> MagicMock:
    s = MagicMock()
    s.auth_provider = auth_provider
    s.egress_auth_enabled = egress_enabled
    s.registry_url = registry_url
    return s


class TestServerAdvertisesPerServerPrm:
    """_server_advertises_per_server_prm mirrors the registry gate on the aud side."""

    def _call(self, settings_obj, env=None):
        from auth_server import server as auth_server

        with (
            patch.object(auth_server, "settings", settings_obj),
            patch.dict("os.environ", env or {}, clear=False),
        ):
            return auth_server._server_advertises_per_server_prm()

    def test_entra_true_even_with_egress_disabled(self):
        """The #990 fix: Entra advertises a per-server PRM for EVERY server, so
        its per-server aud must be accepted regardless of egress."""
        assert self._call(_settings(auth_provider="entra", egress_enabled=False)) is True

    def test_entra_true_with_egress_enabled(self):
        assert self._call(_settings(auth_provider="entra", egress_enabled=True)) is True

    def test_non_entra_egress_enabled_true(self):
        """obo_exchange / oauth_user ingress on any provider needs it -> egress on."""
        assert self._call(_settings(auth_provider="keycloak", egress_enabled=True)) is True

    def test_non_entra_egress_disabled_false(self):
        """Plain server on a lenient IdP with egress off uses the global PRM."""
        assert self._call(_settings(auth_provider="keycloak", egress_enabled=False)) is False

    def test_auth_provider_env_override(self):
        # AUTH_PROVIDER env takes precedence over settings.auth_provider.
        s = _settings(auth_provider="keycloak", egress_enabled=False)
        assert self._call(s, env={"AUTH_PROVIDER": "entra"}) is True


class TestOboExtraAudiences:
    """_obo_extra_audiences returns path-bound per-server resource audiences."""

    def _call(self, server_path, settings_obj, env=None):
        from auth_server import server as auth_server

        base_env = {"AUTH_SERVER_EXTERNAL_URL": "", "REGISTRY_URL": ""}
        base_env.update(env or {})
        with (
            patch.object(auth_server, "settings", settings_obj),
            patch.dict("os.environ", base_env, clear=False),
        ):
            return auth_server._obo_extra_audiences(server_path)

    def test_entra_plain_server_egress_off_returns_audience(self):
        """REGRESSION GUARD (#990): the egress-off Entra plain-server case. The
        per-server resource aud MUST be returned so /validate accepts the token
        the client obtained. Before the fix this returned [] and login 401'd."""
        s = _settings(auth_provider="entra", egress_enabled=False)
        auds = self._call(
            "/plain/mcp", s, env={"AUTH_SERVER_EXTERNAL_URL": "https://gw.example.com"}
        )
        assert "https://gw.example.com/plain/mcp" in auds

    def test_empty_when_not_advertising_per_server_prm(self):
        """Plain server, lenient IdP, egress off -> no per-server PRM -> []."""
        s = _settings(auth_provider="keycloak", egress_enabled=False)
        auds = self._call(
            "/plain/mcp", s, env={"AUTH_SERVER_EXTERNAL_URL": "https://gw.example.com"}
        )
        assert auds == []

    def test_empty_without_server_context(self):
        s = _settings(auth_provider="entra", egress_enabled=False)
        assert self._call(None, s, env={"AUTH_SERVER_EXTERNAL_URL": "https://gw.example.com"}) == []

    def test_accepts_both_mcp_and_bare_forms(self):
        s = _settings(auth_provider="entra", egress_enabled=False)
        auds = self._call("/plain", s, env={"AUTH_SERVER_EXTERNAL_URL": "https://gw.example.com"})
        assert "https://gw.example.com/plain/mcp" in auds
        assert "https://gw.example.com/plain" in auds

    def test_builds_audience_for_each_distinct_base_url(self):
        """When public (external) and internal URLs differ, BOTH per-server
        resources are offered so validation is robust to which URL the registry
        rendered the PRM from (the AUTH_SERVER_EXTERNAL_URL fallback trap)."""
        s = _settings(
            auth_provider="entra",
            egress_enabled=False,
            registry_url="http://registry.internal:8000",
        )
        auds = self._call(
            "/plain/mcp", s, env={"AUTH_SERVER_EXTERNAL_URL": "https://gw.example.com"}
        )
        # public form (what Entra actually minted) is present...
        assert "https://gw.example.com/plain/mcp" in auds
        # ...and the internal-URL form too, so a misconfig doesn't 401.
        assert "http://registry.internal:8000/plain/mcp" in auds

    def test_strips_trailing_mcp_before_building(self):
        s = _settings(auth_provider="entra", egress_enabled=False)
        auds = self._call(
            "plain/mcp", s, env={"AUTH_SERVER_EXTERNAL_URL": "https://gw.example.com"}
        )
        # No doubled /mcp/mcp.
        assert "https://gw.example.com/plain/mcp/mcp" not in auds
        assert "https://gw.example.com/plain/mcp" in auds
