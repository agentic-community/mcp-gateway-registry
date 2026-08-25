"""Registration-time SSRF/scheme validation for POST /servers/{path}/egress-auth.

The 'custom' provider's ``custom_authorize_url``/``custom_token_url`` are
registrant-supplied and become an outbound token POST (carrying the operator
client_secret) and a browser 302. The configure route -- the sole write path
for a server's ``egress_oauth`` -- must reject at registration time any URL that
is non-https, points at a literal private/metadata IP, or uses a disallowed
scheme, so a config that would exfiltrate the secret to an internal target can
never be persisted. Built-in providers ignore the custom URLs and are unaffected.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import registry.api.egress_auth_routes as routes


class _StubServerService:
    def __init__(self, server):
        self._server = server
        self.updated_with = None

    async def get_server_info(self, path, include_credentials=False):
        return dict(self._server)

    async def update_server(self, path, server):
        self.updated_with = server
        return True


@pytest.fixture
def make_client(monkeypatch):
    def _build(server=None):
        monkeypatch.setattr(routes.settings, "egress_auth_enabled", True)
        svc = _StubServerService(server or {"path": "/gh", "egress_oauth": None})
        monkeypatch.setattr(routes, "server_service", svc)
        # Deterministic encryption stub so a persisted secret is non-empty.
        monkeypatch.setattr(routes, "encrypt_credential", lambda s: f"enc:{s}")

        app = FastAPI()
        app.include_router(routes.router)
        # Admin principal; CSRF satisfied (both are sibling dependencies).
        app.dependency_overrides[routes.nginx_proxied_auth] = lambda: {
            "username": "admin",
            "is_admin": True,
            "auth_method": "keycloak",
        }
        app.dependency_overrides[routes.verify_csrf_token_flexible] = lambda: None
        client = TestClient(app)
        client._svc = svc
        return client

    return _build


def _body(**over):
    base = {
        "egress_auth_mode": "oauth_user",
        "egress_provider": "custom",
        "client_id": "cid",
        "client_secret": "supersecret",
        "scopes": ["read"],
        "custom_authorize_url": "https://idp.example.com/authorize",
        "custom_token_url": "https://idp.example.com/token",
        "custom_scope_separator": " ",
        "custom_token_auth_style": "post_body",
    }
    base.update(over)
    return base


@pytest.mark.unit
class TestConfigureEgressUrlValidation:
    def test_valid_custom_https_urls_accepted(self, make_client):
        client = make_client()
        resp = client.post("/servers/gh/egress-auth", json=_body())
        assert resp.status_code == 200, resp.text
        # The config was persisted with the supplied URLs.
        eo = client._svc.updated_with["egress_oauth"]
        assert eo["custom_token_url"] == "https://idp.example.com/token"

    @pytest.mark.parametrize(
        "field",
        ["custom_authorize_url", "custom_token_url"],
    )
    def test_metadata_ip_rejected(self, make_client, field):
        client = make_client()
        resp = client.post(
            "/servers/gh/egress-auth",
            json=_body(**{field: "http://169.254.169.254/latest/meta-data/"}),
        )
        assert resp.status_code == 400
        assert field in resp.json()["detail"]
        # Nothing persisted.
        assert client._svc.updated_with is None

    @pytest.mark.parametrize(
        "field",
        ["custom_authorize_url", "custom_token_url"],
    )
    def test_http_scheme_rejected(self, make_client, field):
        # http:// would send the client_secret in cleartext to any observer.
        client = make_client()
        resp = client.post(
            "/servers/gh/egress-auth",
            json=_body(**{field: "http://idp.example.com/token"}),
        )
        assert resp.status_code == 400
        assert field in resp.json()["detail"]

    def test_loopback_rejected(self, make_client):
        client = make_client()
        resp = client.post(
            "/servers/gh/egress-auth",
            json=_body(custom_token_url="https://127.0.0.1/token"),
        )
        assert resp.status_code == 400
        assert "custom_token_url" in resp.json()["detail"]

    def test_rfc1918_rejected(self, make_client):
        client = make_client()
        resp = client.post(
            "/servers/gh/egress-auth",
            json=_body(custom_authorize_url="https://10.0.0.5/authorize"),
        )
        assert resp.status_code == 400
        assert "custom_authorize_url" in resp.json()["detail"]

    def test_token_url_ignores_proxy_cidr_allowlist(self, make_client, monkeypatch):
        monkeypatch.setattr(routes.settings, "ssrf_allowed_cidrs", "10.0.0.0/8")
        client = make_client()
        resp = client.post(
            "/servers/gh/egress-auth",
            json=_body(custom_token_url="https://10.0.0.5/token"),
        )
        assert resp.status_code == 400
        assert "custom_token_url" in resp.json()["detail"]
        assert client._svc.updated_with is None

    def test_non_http_scheme_rejected(self, make_client):
        client = make_client()
        resp = client.post(
            "/servers/gh/egress-auth",
            json=_body(custom_token_url="file:///etc/passwd"),
        )
        assert resp.status_code == 400
        assert "custom_token_url" in resp.json()["detail"]

    def test_builtin_provider_ignores_custom_url_fields(self, make_client):
        # A built-in provider has hardcoded https endpoints; a stray (even unsafe)
        # custom_* value must not be validated or used -- the built-in wins.
        client = make_client()
        resp = client.post(
            "/servers/gh/egress-auth",
            json=_body(
                egress_provider="github",
                custom_authorize_url="http://169.254.169.254/",
                custom_token_url="http://169.254.169.254/",
            ),
        )
        assert resp.status_code == 200, resp.text
        assert client._svc.updated_with["egress_oauth"]["provider"] == "github"

    def test_custom_resource_accepted_persisted_and_exposed(self, make_client):
        # RFC 8707 resource indicator: a valid absolute https URI is stored and
        # echoed back in the non-secret config view.
        res = "https://mcp.atlassian.com/v1/mcp/authv2"
        client = make_client()
        resp = client.post("/servers/gh/egress-auth", json=_body(custom_resource=res))
        assert resp.status_code == 200, resp.text
        assert client._svc.updated_with["egress_oauth"]["custom_resource"] == res
        assert resp.json()["custom_resource"] == res

    def test_custom_resource_http_scheme_rejected(self, make_client):
        client = make_client()
        resp = client.post(
            "/servers/gh/egress-auth",
            json=_body(custom_resource="http://mcp.atlassian.com/v1/mcp/authv2"),
        )
        assert resp.status_code == 400
        assert "custom_resource" in resp.json()["detail"]
        assert client._svc.updated_with is None

    def test_custom_resource_with_fragment_rejected(self, make_client):
        client = make_client()
        resp = client.post(
            "/servers/gh/egress-auth",
            json=_body(custom_resource="https://mcp.atlassian.com/v1/mcp/authv2#frag"),
        )
        assert resp.status_code == 400
        assert "custom_resource" in resp.json()["detail"]

    def test_builtin_provider_ignores_custom_resource(self, make_client):
        # custom_resource is only validated/used for the 'custom' provider.
        client = make_client()
        resp = client.post(
            "/servers/gh/egress-auth",
            json=_body(egress_provider="github", custom_resource="http://bad#frag"),
        )
        assert resp.status_code == 200, resp.text

    def test_blank_secret_on_edit_keeps_stored_one(self, make_client):
        # Editing a confidential registration with a blank secret must preserve
        # the previously stored encrypted secret, not wipe it.
        client = make_client(
            server={
                "path": "/gh",
                "egress_oauth": {"client_secret_encrypted": "enc:oldsecret"},
            }
        )
        resp = client.post("/servers/gh/egress-auth", json=_body(client_secret=None))
        assert resp.status_code == 200, resp.text
        eo = client._svc.updated_with["egress_oauth"]
        assert eo["client_secret_encrypted"] == "enc:oldsecret"

    def test_view_echoes_token_auth_style_and_separator(self, make_client):
        # The non-secret config view must round-trip custom_token_auth_style
        # (and the scope separator), or a UI read-modify-write silently resets
        # the style to post_body -- which bricks a public-client config.
        client = make_client()
        resp = client.post(
            "/servers/gh/egress-auth",
            json=_body(custom_token_auth_style="basic_header", custom_scope_separator=","),
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["custom_token_auth_style"] == "basic_header"
        assert resp.json()["custom_scope_separator"] == ","


@pytest.mark.unit
class TestConfigureOperatorCredential:
    def test_accepts_stored_bearer_backend_credential(self, make_client):
        client = make_client(
            server={
                "path": "/gh",
                "auth_scheme": "bearer",
                "auth_credential_encrypted": "enc:service-token",
                "egress_oauth": {"provider": "github"},
            }
        )
        resp = client.post(
            "/servers/gh/egress-auth",
            json={"egress_auth_mode": "operator_credential"},
        )
        assert resp.status_code == 200, resp.text
        assert client._svc.updated_with["egress_auth_mode"] == "operator_credential"
        assert client._svc.updated_with["egress_oauth"] is None

    def test_rejects_missing_backend_credential(self, make_client):
        client = make_client(server={"path": "/gh", "auth_scheme": "bearer", "egress_oauth": None})
        resp = client.post(
            "/servers/gh/egress-auth",
            json={"egress_auth_mode": "operator_credential"},
        )
        assert resp.status_code == 400
        assert "stored Backend Authentication credential" in resp.json()["detail"]
        assert client._svc.updated_with is None

    def test_rejects_none_backend_auth_scheme(self, make_client):
        client = make_client(
            server={
                "path": "/gh",
                "auth_scheme": "none",
                "auth_credential_encrypted": "enc:unused",
                "egress_oauth": None,
            }
        )
        resp = client.post(
            "/servers/gh/egress-auth",
            json={"egress_auth_mode": "operator_credential"},
        )
        assert resp.status_code == 400
        assert "requires Backend Authentication scheme" in resp.json()["detail"]


@pytest.mark.unit
class TestConfigureModeDispatch:
    """Mode parsing and the non-oauth_user branches of the configure route."""

    def test_unknown_server_is_404(self, make_client, monkeypatch):
        client = make_client()

        async def _missing(path, include_credentials=False):
            return None

        monkeypatch.setattr(client._svc, "get_server_info", _missing)
        resp = client.post(
            "/servers/gh/egress-auth",
            json={"egress_auth_mode": "none"},
        )
        assert resp.status_code == 404
        assert client._svc.updated_with is None

    def test_invalid_mode_is_400(self, make_client):
        client = make_client()
        resp = client.post(
            "/servers/gh/egress-auth",
            json={"egress_auth_mode": "not-a-real-mode"},
        )
        assert resp.status_code == 400
        assert client._svc.updated_with is None

    def test_none_resets_mode_and_clears_oauth_config(self, make_client):
        client = make_client(
            server={
                "path": "/gh",
                "egress_auth_mode": "oauth_user",
                "egress_oauth": {"provider": "github", "client_id": "Iv1.x"},
            }
        )
        resp = client.post("/servers/gh/egress-auth", json={"egress_auth_mode": "none"})
        assert resp.status_code == 200, resp.text
        assert client._svc.updated_with["egress_auth_mode"] == "none"
        assert client._svc.updated_with["egress_oauth"] is None

    def test_update_failure_is_500(self, make_client, monkeypatch):
        client = make_client()

        async def _fail(path, server):
            return False

        monkeypatch.setattr(client._svc, "update_server", _fail)
        resp = client.post("/servers/gh/egress-auth", json={"egress_auth_mode": "none"})
        assert resp.status_code == 500

    def test_obo_requires_target_audience(self, make_client):
        client = make_client()
        resp = client.post(
            "/servers/gh/egress-auth",
            json={"egress_auth_mode": "obo_exchange", "target_audience": "   "},
        )
        assert resp.status_code == 400
        assert "target_audience" in resp.json()["detail"]
        assert client._svc.updated_with is None

    def test_obo_stores_target_and_scopes(self, make_client):
        client = make_client()
        resp = client.post(
            "/servers/gh/egress-auth",
            json={
                "egress_auth_mode": "obo_exchange",
                "target_audience": "api://backend-api",
                "scopes": ["read"],
            },
        )
        assert resp.status_code == 200, resp.text
        assert client._svc.updated_with["egress_auth_mode"] == "obo_exchange"
        eo = client._svc.updated_with["egress_oauth"]
        assert eo["target_audience"] == "api://backend-api"
        assert eo["scopes"] == ["read"]


@pytest.mark.unit
class TestConfigurePublicClient:
    """token_endpoint_auth_method=none (custom provider): a public client has
    no secret by design -- the configure route must accept a secretless config,
    require a client_id, and DROP any previously stored secret."""

    def _public_body(self, **over):
        base = _body(
            client_secret=None,
            custom_token_auth_style="none",
            custom_authorize_url="https://app.datadoghq.com/oauth2/v1/authorize",
            custom_token_url="https://app.datadoghq.com/api/v2/oauth2/token",
            custom_resource="https://mcp.datadoghq.com/api/unstable/mcp-server/mcp",
        )
        base.update(over)
        return base

    def test_public_client_config_without_secret_succeeds(self, make_client):
        client = make_client()
        resp = client.post("/servers/gh/egress-auth", json=self._public_body())
        assert resp.status_code == 200, resp.text
        eo = client._svc.updated_with["egress_oauth"]
        assert eo["custom_token_auth_style"] == "none"
        assert eo["client_secret_encrypted"] is None
        assert resp.json()["custom_token_auth_style"] == "none"

    def test_public_client_requires_client_id(self, make_client):
        client = make_client()
        resp = client.post("/servers/gh/egress-auth", json=self._public_body(client_id="  "))
        assert resp.status_code == 400
        assert "client_id required" in resp.json()["detail"]
        assert client._svc.updated_with is None

    def test_switch_to_public_client_drops_stored_secret(self, make_client):
        # A registration previously configured confidential must not carry the
        # stale encrypted secret into the public-client config.
        client = make_client(
            server={
                "path": "/gh",
                "egress_oauth": {"client_secret_encrypted": "enc:oldsecret"},
            }
        )
        resp = client.post("/servers/gh/egress-auth", json=self._public_body())
        assert resp.status_code == 200, resp.text
        assert client._svc.updated_with["egress_oauth"]["client_secret_encrypted"] is None

    def test_supplied_secret_ignored_for_public_client(self, make_client):
        # An operator pasting a secret alongside style 'none' must not have it
        # stored -- there is no request the engine could ever place it in.
        client = make_client()
        resp = client.post(
            "/servers/gh/egress-auth", json=self._public_body(client_secret="pasted-anyway")
        )
        assert resp.status_code == 200, resp.text
        assert client._svc.updated_with["egress_oauth"]["client_secret_encrypted"] is None

    def test_confidential_style_still_requires_secret(self, make_client):
        # post_body/basic_header without a secret (and no stored prior) -> 400.
        client = make_client()
        resp = client.post(
            "/servers/gh/egress-auth",
            json=_body(client_secret=None, custom_token_auth_style="post_body"),
        )
        assert resp.status_code == 400
        assert "client_secret required" in resp.json()["detail"]

    def test_builtin_provider_cannot_go_secretless_via_style(self, make_client):
        # custom_token_auth_style is a custom-provider knob; a built-in provider
        # posting style 'none' stays confidential and still requires a secret.
        client = make_client()
        resp = client.post(
            "/servers/gh/egress-auth",
            json=_body(
                egress_provider="github", client_secret=None, custom_token_auth_style="none"
            ),
        )
        assert resp.status_code == 400
        assert "client_secret required" in resp.json()["detail"]
