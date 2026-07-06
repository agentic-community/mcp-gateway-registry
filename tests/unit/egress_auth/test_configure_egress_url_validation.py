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
