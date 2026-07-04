"""Unit tests for the egress server-path + config helpers
(registry/egress_auth/as_facade.py).
"""

import pytest

from registry.egress_auth import as_facade


@pytest.mark.unit
class TestServerConfigHelpers:
    def _server(self, **over) -> dict:
        base = {
            "egress_auth_mode": "oauth_user",
            "egress_oauth": {"provider": "github", "client_id": "Iv1.x"},
        }
        base.update(over)
        return base

    def test_egress_configured_true_for_valid_github(self):
        assert as_facade.is_server_egress_configured(self._server()) is True

    def test_none_server_not_configured(self):
        assert as_facade.is_server_egress_configured(None) is False

    def test_mode_none_not_configured(self):
        assert as_facade.is_server_egress_configured(self._server(egress_auth_mode="none")) is False

    def test_missing_egress_oauth_not_configured(self):
        assert as_facade.is_server_egress_configured(self._server(egress_oauth=None)) is False

    def test_unresolvable_provider_not_configured(self):
        bad = self._server(egress_oauth={"provider": "does-not-exist"})
        assert as_facade.is_server_egress_configured(bad) is False


@pytest.mark.unit
class TestNormalizeServerPath:
    def test_adds_leading_slash(self):
        assert as_facade._normalize_server_path("github") == "/github"

    def test_keeps_existing_leading_slash(self):
        assert as_facade._normalize_server_path("/github") == "/github"
