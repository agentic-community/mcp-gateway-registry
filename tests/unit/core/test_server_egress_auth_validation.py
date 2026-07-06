"""Validation tests for ServerInfo's per-mode egress auth config.

Covers ServerInfo._validate_egress_auth (@model_validator):
- mode=none: no egress_oauth required.
- mode=oauth_user: requires egress_oauth.provider (3LO).
- mode=obo_exchange: requires target_audience; rejects same-app audience.
- invalid mode string rejected.
"""

import pytest
from pydantic import ValidationError

from registry.core import schemas
from registry.core.schemas import EgressOAuthConfig, ServerInfo


def _server(**egress):
    """Build a minimal remote ServerInfo with the given egress overrides."""
    return ServerInfo(
        server_name="s",
        path="/s",
        proxy_pass_url="http://upstream.test",
        **egress,
    )


@pytest.mark.unit
@pytest.mark.core
class TestServerEgressAuthValidation:
    def test_mode_none_needs_no_egress_oauth(self):
        s = _server(egress_auth_mode="none")
        assert s.egress_auth_mode == "none"
        assert s.egress_oauth is None

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValidationError, match="invalid egress_auth_mode"):
            _server(egress_auth_mode="bogus")

    def test_oauth_user_requires_provider(self):
        with pytest.raises(ValidationError, match="requires egress_oauth.provider"):
            _server(
                egress_auth_mode="oauth_user",
                egress_oauth=EgressOAuthConfig(provider=None),
            )

    def test_oauth_user_with_provider_accepted(self):
        s = _server(
            egress_auth_mode="oauth_user",
            egress_oauth=EgressOAuthConfig(provider="github"),
        )
        assert s.egress_oauth.provider == "github"

    def test_oauth_user_requires_egress_oauth_present(self):
        with pytest.raises(ValidationError, match="requires egress_oauth config"):
            _server(egress_auth_mode="oauth_user", egress_oauth=None)

    def test_obo_exchange_requires_target_audience(self):
        with pytest.raises(ValidationError, match="requires egress_oauth.target_audience"):
            _server(
                egress_auth_mode="obo_exchange",
                egress_oauth=EgressOAuthConfig(target_audience=None),
            )

    def test_obo_exchange_valid_target_accepted(self):
        s = _server(
            egress_auth_mode="obo_exchange",
            egress_oauth=EgressOAuthConfig(
                target_audience="api://outlook-mcp-server",
                scopes=["api://outlook-mcp-server/.default"],
            ),
        )
        assert s.egress_oauth.target_audience == "api://outlook-mcp-server"
        # obo_exchange does NOT require a provider.
        assert s.egress_oauth.provider is None

    def test_obo_exchange_rejects_gateway_own_client_id(self, monkeypatch):
        # Gateway configured for Entra with a known client id.
        from registry.core.config import settings

        monkeypatch.setattr(settings, "auth_provider", "entra", raising=False)
        monkeypatch.setattr(settings, "entra_client_id", "gw-client-123", raising=False)
        with pytest.raises(ValidationError, match="must differ from the gateway's own"):
            _server(
                egress_auth_mode="obo_exchange",
                egress_oauth=EgressOAuthConfig(target_audience="gw-client-123"),
            )

    def test_obo_exchange_rejects_gateway_own_app_id_uri(self, monkeypatch):
        # The api://<client_id> App ID URI form must also be rejected.
        from registry.core.config import settings

        monkeypatch.setattr(settings, "auth_provider", "entra", raising=False)
        monkeypatch.setattr(settings, "entra_client_id", "gw-client-123", raising=False)
        with pytest.raises(ValidationError, match="must differ from the gateway's own"):
            _server(
                egress_auth_mode="obo_exchange",
                egress_oauth=EgressOAuthConfig(target_audience="api://gw-client-123"),
            )

    def test_obo_exchange_allows_distinct_target_when_gateway_id_known(self, monkeypatch):
        from registry.core.config import settings

        monkeypatch.setattr(settings, "auth_provider", "entra", raising=False)
        monkeypatch.setattr(settings, "entra_client_id", "gw-client-123", raising=False)
        s = _server(
            egress_auth_mode="obo_exchange",
            egress_oauth=EgressOAuthConfig(target_audience="api://outlook-mcp-server"),
        )
        assert s.egress_oauth.target_audience == "api://outlook-mcp-server"


@pytest.mark.unit
@pytest.mark.core
class TestGatewayOwnAudienceHelper:
    def test_no_gateway_id_configured_means_no_match(self, monkeypatch):
        from registry.core.config import settings

        # Unknown provider -> helper returns "" -> never flags same-app.
        monkeypatch.setattr(settings, "auth_provider", "cognito", raising=False)
        assert schemas._is_gateway_own_audience("anything") is False

    def test_match_is_case_insensitive(self, monkeypatch):
        from registry.core.config import settings

        monkeypatch.setattr(settings, "auth_provider", "keycloak", raising=False)
        monkeypatch.setattr(settings, "keycloak_client_id", "MCP-Gateway", raising=False)
        assert schemas._is_gateway_own_audience("mcp-gateway") is True

    def test_gateway_own_resource_url_is_same_app(self, monkeypatch):
        # The gateway's own public resource URL (registry_url) is an audience of
        # the gateway app; target_audience equal to it must be flagged same-app,
        # with or without a trailing slash.
        from registry.core.config import settings

        monkeypatch.setattr(settings, "auth_provider", "entra", raising=False)
        monkeypatch.setattr(settings, "entra_client_id", "gw-client-123", raising=False)
        monkeypatch.setattr(settings, "registry_url", "https://gw.example.com", raising=False)
        assert schemas._is_gateway_own_audience("https://gw.example.com") is True
        assert schemas._is_gateway_own_audience("https://gw.example.com/") is True


@pytest.mark.unit
@pytest.mark.core
class TestForbiddenOboAudience:
    """target_audience must not be a shared first-party IdP resource (confused deputy)."""

    @pytest.mark.parametrize(
        "aud",
        [
            "https://graph.microsoft.com",
            "https://graph.microsoft.com/",
            "https://GRAPH.microsoft.com",  # case-insensitive
            "00000003-0000-0000-c000-000000000000",  # Graph app id GUID
            "https://management.azure.com",
            "https://vault.azure.net",
        ],
    )
    def test_forbidden_audiences_rejected(self, aud):
        assert schemas._is_forbidden_obo_audience(aud) is True

    def test_internal_app_audience_allowed(self):
        assert schemas._is_forbidden_obo_audience("api://outlook-mcp-server") is False

    def test_obo_registration_rejects_graph_target(self, monkeypatch):
        from registry.core.config import settings

        # No gateway id set, so the same-app check is inert -- the forbidden-audience
        # check is what must reject Graph.
        monkeypatch.setattr(settings, "auth_provider", "entra", raising=False)
        monkeypatch.setattr(settings, "entra_client_id", "", raising=False)
        with pytest.raises(ValidationError, match="shared first-party IdP resource"):
            _server(
                egress_auth_mode="obo_exchange",
                egress_oauth=EgressOAuthConfig(target_audience="https://graph.microsoft.com"),
            )


@pytest.mark.unit
@pytest.mark.core
class TestServerPathValidation:
    """ServerInfo.path must be a safe slug (nginx-injection defense at the source)."""

    @pytest.mark.parametrize(
        "path",
        ["/atlassian", "/currenttime/", "/a/b-c_d.e", "/x", "bare/segment"],
    )
    def test_valid_paths_accepted(self, path):
        s = ServerInfo(server_name="s", path=path, proxy_pass_url="http://u.test")
        assert s.path == path

    @pytest.mark.parametrize(
        "path",
        [
            '/x" ; return 200 "pwned"; #',  # nginx directive injection
            "/x\ny",  # newline
            "/x y",  # whitespace
            "/x{}",  # braces
            "/x;drop",  # semicolon
            "/x$var",  # nginx variable sigil
            "",  # empty
        ],
    )
    def test_hostile_paths_rejected(self, path):
        with pytest.raises(ValidationError, match="invalid server path"):
            ServerInfo(server_name="s", path=path, proxy_pass_url="http://u.test")
