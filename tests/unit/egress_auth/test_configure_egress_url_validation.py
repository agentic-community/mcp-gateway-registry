"""Registration-time SSRF/scheme validation for POST /servers/{path}/egress-auth.

The 'custom' provider's ``custom_authorize_url``/``custom_token_url`` are
registrant-supplied and become an outbound token POST (carrying the operator
client_secret) and a browser 302. The configure route -- the sole write path
for a server's ``egress_oauth`` -- must reject at registration time any URL that
is non-https, points at a literal private/metadata IP, or uses a disallowed
scheme, so a config that would exfiltrate the secret to an internal target can
never be persisted. Built-in providers ignore the custom URLs and are unaffected.
"""

import socket
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import registry.api.egress_auth_routes as routes
import registry.core.config as core_config
from registry.egress_auth import providers
from registry.egress_auth.schemas import TokenEndpointAuthStyle
from registry.services.server_service import ServerService
from tests.unit.egress_auth.dcr_fixtures import (
    DCR_TEST_DEFAULT_SCOPES,
    DCR_TEST_PROVIDER,
    make_dcr_test_config,
)


class _StubServerService:
    def __init__(self, server):
        self._server = server
        self.updated_with = None
        self.updated_fields = None

    async def get_server_info(self, path, include_credentials=False):
        return dict(self._server)

    async def update_server(self, path, server, *, updated_fields=None):
        self.updated_with = server
        self.updated_fields = updated_fields
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

    def test_edit_without_secret_keeps_prior(self, make_client):
        # A non-DCR provider edit that omits client_secret reuses the prior
        # encrypted secret (edit-in-place must not wipe the stored credential).
        client = make_client(
            {"path": "/gh", "egress_oauth": {"client_secret_encrypted": "enc:old"}}
        )
        resp = client.post(
            "/servers/gh/egress-auth",
            json={
                "egress_auth_mode": "oauth_user",
                "egress_provider": "github",
                "client_id": "cid",
                "scopes": ["repo"],
            },
        )
        assert resp.status_code == 200, resp.text
        assert client._svc.updated_with["egress_oauth"]["client_secret_encrypted"] == "enc:old"

    def test_non_dcr_without_secret_or_prior_rejected(self, make_client):
        # A non-DCR provider with neither a supplied nor a stored secret is a 400
        # (confidential clients must have a secret).
        client = make_client({"path": "/gh", "egress_oauth": None})
        resp = client.post(
            "/servers/gh/egress-auth",
            json={
                "egress_auth_mode": "oauth_user",
                "egress_provider": "github",
                "client_id": "cid",
                "scopes": ["repo"],
            },
        )
        assert resp.status_code == 400
        assert "client_secret required" in resp.json()["detail"]
        assert client._svc.updated_with is None

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


def _obo_body(**over):
    """A minimal obo_exchange configure body (provider/client fields unused)."""
    base = {
        "egress_auth_mode": "obo_exchange",
        "target_audience": "api://outlook-mcp-server",
        "scopes": [],
    }
    base.update(over)
    return base


@pytest.mark.unit
class TestConfigureWritesOnlyEgressFields:
    """The save is field-scoped, so it cannot revert what another writer stored meanwhile.

    The route makes outbound provider calls between reading the server and writing
    it back. A full-document write would put the stale read of every other field
    (the health checker's ``tool_list`` and ``mcp_server_version``) back.
    """

    EGRESS_FIELDS = ["egress_auth_mode", "egress_list_tools_before_consent", "egress_oauth"]

    # Fields the health checker, the registry and other routes own.
    STORED = {
        "path": "/gh",
        "server_name": "GitHub",
        "tool_list": [{"name": "stale-tool"}],
        "mcp_server_version": "1.0.0",
        "tags": ["a"],
        "egress_auth_mode": "none",
        "egress_oauth": None,
    }

    @pytest.mark.parametrize(
        "body",
        [
            pytest.param({"egress_auth_mode": "none"}, id="none"),
            pytest.param(_body(), id="oauth_user"),
            pytest.param(_obo_body(), id="obo_exchange"),
            pytest.param(
                {"egress_auth_mode": "pat", "egress_provider": "github", "scopes": []},
                id="pat",
            ),
            pytest.param(_body(list_tools_before_consent=True), id="list_tools_before_consent"),
        ],
    )
    def test_save_names_exactly_the_egress_fields(self, make_client, body):
        client = make_client(dict(self.STORED))
        resp = client.post("/servers/gh/egress-auth", json=body)
        assert resp.status_code == 200, resp.text
        assert client._svc.updated_fields == self.EGRESS_FIELDS

    @pytest.mark.parametrize(
        "body",
        [
            pytest.param({"egress_auth_mode": "none"}, id="none"),
            pytest.param(_body(), id="oauth_user"),
            pytest.param(_obo_body(), id="obo_exchange"),
            pytest.param(
                {"egress_auth_mode": "pat", "egress_provider": "github", "scopes": []},
                id="pat",
            ),
        ],
    )
    def test_every_field_the_route_changes_is_in_the_scope(self, make_client, body):
        # Guards a future edit that assigns another server key without naming it:
        # a field-scoped write would silently discard that change.
        stored = dict(self.STORED)
        client = make_client(stored)
        resp = client.post("/servers/gh/egress-auth", json=body)
        assert resp.status_code == 200, resp.text
        written = client._svc.updated_with
        changed = {k for k in written if k not in stored or written[k] != stored[k]}
        assert changed <= set(client._svc.updated_fields)
        assert "egress_auth_mode" in changed or written["egress_auth_mode"] == "none"

    def test_a_failed_save_writes_nothing(self, make_client):
        client = make_client(dict(self.STORED))
        resp = client.post("/servers/gh/egress-auth", json=_obo_body(target_audience="   "))
        assert resp.status_code == 400
        assert client._svc.updated_fields is None


@pytest.mark.unit
class TestProxiedServerEgressSaveWriteSet:
    """What the repository is asked to write for a proxied server's egress save.

    The route names only the egress fields, but the real ``update_server`` extends
    the write for a proxied server: it re-pins the proxy target and clears a
    refresh auto-disable. This runs the real service over a repository mock so the
    full write set is pinned, not only what the route asked for.
    """

    STORED = {
        "path": "/gh",
        "server_name": "GitHub",
        "is_proxied": True,
        "proxy_pass_url": "https://ok.example/",
        "proxy_disabled_reason": "was auto-disabled",
        "deployment": "remote",
        "tool_list": [{"name": "t"}],
        "egress_auth_mode": "none",
        "egress_oauth": None,
    }

    @pytest.fixture
    def repo(self):
        repo = AsyncMock()
        repo.get.return_value = dict(self.STORED)
        repo.update.return_value = True
        repo.get_state.return_value = False
        return repo

    @pytest.fixture
    def client(self, monkeypatch, repo):
        monkeypatch.setattr(routes.settings, "egress_auth_enabled", True)
        monkeypatch.setattr(core_config.settings, "gateway_proxy_allow_private_targets", False)
        with (
            patch("registry.services.server_service.get_server_repository", return_value=repo),
            patch(
                "registry.repositories.factory.get_search_repository",
                return_value=AsyncMock(),
            ),
        ):
            monkeypatch.setattr(routes, "server_service", ServerService())
        app = FastAPI()
        app.include_router(routes.router)
        app.dependency_overrides[routes.nginx_proxied_auth] = lambda: {
            "username": "admin",
            "is_admin": True,
            "auth_method": "keycloak",
        }
        app.dependency_overrides[routes.verify_csrf_token_flexible] = lambda: None
        return TestClient(app)

    def test_save_writes_the_egress_and_re_pinned_proxy_fields_only(self, client, repo):
        addrinfo = [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 443))
        ]
        with patch("socket.getaddrinfo", return_value=addrinfo):
            resp = client.post("/servers/gh/egress-auth", json={"egress_auth_mode": "none"})
        assert resp.status_code == 200, resp.text

        repo.update.assert_awaited_once()
        kwargs = repo.update.call_args.kwargs
        # Exactly the route's fields, then the three the service adds for a proxied server.
        assert kwargs["updated_fields"] == [
            "egress_auth_mode",
            "egress_list_tools_before_consent",
            "egress_oauth",
            "proxy_resolved_ips",
            "proxy_target_host",
            "proxy_disabled_reason",
        ]
        written = repo.update.call_args.args[1]
        assert written["proxy_resolved_ips"] == ["93.184.216.34"]
        assert written["proxy_target_host"] == "ok.example"
        # A prior refresh auto-disable is cleared by the save.
        assert written["proxy_disabled_reason"] is None
        # No upstream header field is cleared: the target did not change.
        assert "custom_headers_encrypted" not in kwargs["updated_fields"]
        assert "custom_header_names" not in kwargs["updated_fields"]


@pytest.mark.unit
class TestConfigureOboAudience:
    """obo_exchange: the live write path is the only enforcement point (the
    ServerInfo model validator is never instantiated in prod). It must enforce
    the always-on target-audience/scope floor and fail closed -- a config that
    would let auth_server exchange the user's token for an arbitrary third-party
    audience must never be persisted."""

    def test_valid_internal_audience_accepted(self, make_client):
        client = make_client()
        resp = client.post("/servers/gh/egress-auth", json=_obo_body())
        assert resp.status_code == 200, resp.text
        eo = client._svc.updated_with["egress_oauth"]
        assert eo["target_audience"] == "api://outlook-mcp-server"

    def test_https_host_audience_rejected(self, make_client):
        # A shared first-party resource is addressable by https:// host URL; a
        # delegated token for it would be exfiltrated to the server's upstream.
        client = make_client()
        resp = client.post(
            "/servers/gh/egress-auth",
            json=_obo_body(target_audience="https://graph.microsoft.com"),
        )
        assert resp.status_code == 400
        assert "not an allowed obo_exchange target" in resp.json()["detail"]
        assert client._svc.updated_with is None

    def test_bare_guid_audience_rejected(self, make_client):
        # A bare GUID is how a first-party API is directly addressable.
        client = make_client()
        resp = client.post(
            "/servers/gh/egress-auth",
            json=_obo_body(target_audience="00000003-0000-0000-c000-000000000000"),
        )
        assert resp.status_code == 400
        assert client._svc.updated_with is None

    def test_empty_audience_rejected(self, make_client):
        client = make_client()
        resp = client.post("/servers/gh/egress-auth", json=_obo_body(target_audience="   "))
        assert resp.status_code == 400
        assert "requires egress_oauth.target_audience" in resp.json()["detail"]
        assert client._svc.updated_with is None

    def test_malformed_scheme_audience_rejected(self, make_client):
        client = make_client()
        resp = client.post("/servers/gh/egress-auth", json=_obo_body(target_audience="api://"))
        assert resp.status_code == 400
        assert "malformed" in resp.json()["detail"]
        assert client._svc.updated_with is None

    def test_scope_for_other_resource_rejected(self, make_client):
        # An unbound scope for a different resource defeats the target check --
        # the exchange engine sends scopes verbatim, ignoring target_audience.
        client = make_client()
        resp = client.post(
            "/servers/gh/egress-auth",
            json=_obo_body(
                target_audience="api://outlook-mcp-server",
                scopes=["https://graph.microsoft.com/.default"],
            ),
        )
        assert resp.status_code == 400
        assert "grants against a resource other than" in resp.json()["detail"]
        assert client._svc.updated_with is None

    def test_target_scoped_scope_accepted(self, make_client):
        client = make_client()
        resp = client.post(
            "/servers/gh/egress-auth",
            json=_obo_body(
                target_audience="api://outlook-mcp-server",
                scopes=["api://outlook-mcp-server/.default"],
            ),
        )
        assert resp.status_code == 200, resp.text
        assert client._svc.updated_with["egress_oauth"]["scopes"] == [
            "api://outlook-mcp-server/.default"
        ]

    def test_gateway_own_audience_rejected(self, make_client, monkeypatch):
        monkeypatch.setattr(routes.settings, "auth_provider", "entra", raising=False)
        monkeypatch.setattr(routes.settings, "entra_client_id", "gw-client-123", raising=False)
        client = make_client()
        resp = client.post(
            "/servers/gh/egress-auth", json=_obo_body(target_audience="gw-client-123")
        )
        assert resp.status_code == 400
        assert "must differ from the gateway's own" in resp.json()["detail"]
        assert client._svc.updated_with is None

    def test_operator_allowlist_enforced_positively(self, make_client, monkeypatch):
        # When EGRESS_OBO_ALLOWED_AUDIENCES is set it is authoritative: a target
        # outside it is rejected even if otherwise well-shaped.
        monkeypatch.setattr(
            routes.settings,
            "egress_obo_allowed_audiences",
            "api://pinned-mcp-server",
            raising=False,
        )
        client = make_client()
        resp = client.post(
            "/servers/gh/egress-auth",
            json=_obo_body(target_audience="api://outlook-mcp-server"),
        )
        assert resp.status_code == 400
        assert client._svc.updated_with is None

    def test_operator_allowlisted_audience_accepted(self, make_client, monkeypatch):
        monkeypatch.setattr(
            routes.settings,
            "egress_obo_allowed_audiences",
            "api://pinned-mcp-server",
            raising=False,
        )
        client = make_client()
        resp = client.post(
            "/servers/gh/egress-auth",
            json=_obo_body(target_audience="api://pinned-mcp-server"),
        )
        assert resp.status_code == 200, resp.text
        assert (
            client._svc.updated_with["egress_oauth"]["target_audience"] == "api://pinned-mcp-server"
        )


_PRM_SCOPES = [*DCR_TEST_DEFAULT_SCOPES, "write:work", "search:work"]


def _prm_doc(scopes=None):
    """Protected-resource metadata for the test-local dcr-test provider."""
    return {
        "resource": "https://mcp.example.com/v2/mcp",
        "authorization_servers": ["https://auth.example.com/TENANT"],
        "scopes_supported": _PRM_SCOPES if scopes is None else scopes,
    }


def _stub_prm(monkeypatch, doc=None):
    async def fake_prm(cfg):
        return _prm_doc() if doc is None else doc

    monkeypatch.setattr(routes.oauth_engine, "fetch_protected_resource_metadata", fake_prm)


def _forbid_dcr(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("DCR must not run when the stored client is kept")

    monkeypatch.setattr(routes.oauth_engine, "register_dcr_client", boom)


def _record_dcr(monkeypatch, client_id="NEWID", client_secret=None):
    """Stub registration; return the dict that records each call's arguments."""
    calls: dict = {"count": 0}

    async def fake_register(cfg, redirect_uri, scopes, prm=None):
        calls["count"] += 1
        calls["redirect_uri"] = redirect_uri
        calls["scopes"] = scopes
        calls["is_none_style"] = cfg.token_endpoint_auth_style == TokenEndpointAuthStyle.NONE
        return client_id, client_secret

    monkeypatch.setattr(routes.oauth_engine, "register_dcr_client", fake_register)
    return calls


def _dcr_body(**over):
    # No operator client_id/secret: the gateway must DCR-register its own.
    base = {
        "egress_auth_mode": "oauth_user",
        "egress_provider": DCR_TEST_PROVIDER,
        "client_id": "",
        "scopes": ["read:work", "offline_access"],
    }
    base.update(over)
    return base


def _confidential_dcr_provider(monkeypatch):
    """Replace dcr-test with a confidential (client_secret_post) DCR provider."""
    cfg = make_dcr_test_config(token_endpoint_auth_style=TokenEndpointAuthStyle.POST_BODY)
    monkeypatch.setitem(providers.PROVIDER_REGISTRY, DCR_TEST_PROVIDER, cfg)


@pytest.mark.unit
@pytest.mark.usefixtures("dcr_provider")
class TestConfigureEgressDcr:
    """DCR-at-registration for a ``requires_dcr`` provider."""

    @pytest.fixture(autouse=True)
    def _prm(self, monkeypatch):
        """Stub the PRM fetch so no test needs the network."""
        _stub_prm(monkeypatch)

    def test_dcr_registers_none_style_client_and_persists_client_id(self, make_client, monkeypatch):
        calls = _record_dcr(monkeypatch, client_id="DCRID")
        client = make_client({"path": "/atl", "egress_oauth": None})
        resp = client.post("/servers/atl/egress-auth", json=_dcr_body())
        assert resp.status_code == 200, resp.text
        eo = client._svc.updated_with["egress_oauth"]
        # Registered client_id persisted; no secret required for a NONE-style client.
        assert eo["client_id"] == "DCRID"
        assert eo.get("client_secret_encrypted") is None
        # DCR used the gateway callback as redirect and the requested scopes.
        assert calls["redirect_uri"] == routes._callback_url()
        # The RESOLVED scopes, i.e. with the provider's required read:account
        # unioned in: registering the raw operator list would mint a client whose
        # grant can never complete consent.
        assert calls["scopes"] == ["read:work", "offline_access", "read:account"]
        assert calls["is_none_style"] is True

    def test_dcr_records_what_the_client_was_registered_for(self, make_client, monkeypatch):
        _record_dcr(monkeypatch, client_id="DCRID")
        client = make_client({"path": "/atl", "egress_oauth": None})
        resp = client.post("/servers/atl/egress-auth", json=_dcr_body())
        assert resp.status_code == 200, resp.text
        eo = client._svc.updated_with["egress_oauth"]
        assert eo["dcr_provider"] == DCR_TEST_PROVIDER
        assert eo["dcr_redirect_uri"] == routes._callback_url()
        assert eo["dcr_scopes"] == ["read:work", "offline_access", "read:account"]

    def test_dcr_failure_returns_502(self, make_client, monkeypatch):
        async def fail(cfg, redirect_uri, scopes, prm=None):
            raise routes.oauth_engine.OAuthEngineError("registration rejected (http 400)")

        monkeypatch.setattr(routes.oauth_engine, "register_dcr_client", fail)
        client = make_client({"path": "/atl", "egress_oauth": None})
        resp = client.post("/servers/atl/egress-auth", json=_dcr_body())
        assert resp.status_code == 502
        assert resp.json()["detail"] == (
            "dynamic client registration failed: registration rejected (http 400)"
        )
        assert client._svc.updated_with is None

    def test_operator_client_id_with_secret_on_confidential_provider(
        self, make_client, monkeypatch
    ):
        # An explicit operator client_id keeps the old behaviour: used as-is, no
        # registration, and a supplied secret is encrypted. Only meaningful for a
        # CONFIDENTIAL requires_dcr provider; a NONE-style one drops the secret
        # (see test_dcr_none_style_drops_supplied_secret).
        _confidential_dcr_provider(monkeypatch)
        _forbid_dcr(monkeypatch)
        client = make_client(
            {"path": "/atl", "egress_oauth": {"client_id": "OLD", "provider": DCR_TEST_PROVIDER}}
        )
        resp = client.post(
            "/servers/atl/egress-auth",
            json=_dcr_body(client_id="OPERATOR", client_secret="rotated"),
        )
        assert resp.status_code == 200, resp.text
        eo = client._svc.updated_with["egress_oauth"]
        assert eo["client_id"] == "OPERATOR"
        assert eo["client_secret_encrypted"] == "enc:rotated"

    def test_dcr_none_style_drops_supplied_secret(self, make_client, monkeypatch):
        # A NONE-style DCR client proves possession with PKCE, so no secret is kept
        # even when one is supplied (or handed back by the AS): storing a
        # credential the token leg never reads is needless exposure.
        _forbid_dcr(monkeypatch)
        client = make_client(
            {
                "path": "/atl",
                "egress_oauth": {"client_id": "EXISTING", "provider": DCR_TEST_PROVIDER},
            }
        )
        resp = client.post("/servers/atl/egress-auth", json=_dcr_body(client_secret="rotated"))
        assert resp.status_code == 200, resp.text
        eo = client._svc.updated_with["egress_oauth"]
        assert eo["client_id"] == "EXISTING"
        assert eo["client_secret_encrypted"] is None

    def test_dcr_none_style_discards_secret_returned_by_as(self, make_client, monkeypatch):
        # Some authorization servers (Atlassian's among them) return a
        # client_secret even for token_endpoint_auth_method=none. A NONE-style
        # provider must not persist it.
        _record_dcr(monkeypatch, client_id="DCRID", client_secret="secret-we-did-not-ask-for")
        client = make_client({"path": "/atl", "egress_oauth": None})
        resp = client.post("/servers/atl/egress-auth", json=_dcr_body())
        assert resp.status_code == 200, resp.text
        eo = client._svc.updated_with["egress_oauth"]
        assert eo["client_id"] == "DCRID"
        assert eo["client_secret_encrypted"] is None

    def test_dcr_confidential_client_without_secret_rejected(self, make_client, monkeypatch):
        # A requires_dcr provider that is NOT NONE-style and yields no secret is a
        # provider/config error, so 502.
        _confidential_dcr_provider(monkeypatch)
        _record_dcr(monkeypatch, client_id="DCRID", client_secret=None)
        client = make_client({"path": "/atl", "egress_oauth": None})
        resp = client.post("/servers/atl/egress-auth", json=_dcr_body())
        assert resp.status_code == 502
        assert "no client_secret" in resp.json()["detail"]
        assert client._svc.updated_with is None

    def test_dcr_no_client_secret_required(self, make_client, monkeypatch):
        # A public DCR client is valid with no secret: the "client_secret
        # required" gate must not fire for a requires_dcr provider.
        _record_dcr(monkeypatch)
        client = make_client({"path": "/atl", "egress_oauth": None})
        resp = client.post("/servers/atl/egress-auth", json=_dcr_body())
        assert resp.status_code == 200, resp.text


@pytest.mark.unit
@pytest.mark.usefixtures("dcr_provider")
class TestDcrClientReuse:
    """When a stored DCR client is kept and when a new one is registered."""

    @pytest.fixture(autouse=True)
    def _prm(self, monkeypatch):
        _stub_prm(monkeypatch)

    def _stored(self, **over):
        """A config the gateway registered, for the current callback URL."""
        eo = {
            "provider": DCR_TEST_PROVIDER,
            "client_id": "STORED",
            "scopes": ["read:work", "offline_access", "read:account"],
            "dcr_provider": DCR_TEST_PROVIDER,
            "dcr_redirect_uri": routes._callback_url(),
            "dcr_scopes": ["read:work", "offline_access", "read:account"],
        }
        eo.update(over)
        return {"path": "/atl", "egress_oauth": eo}

    def test_plain_resave_reuses_the_stored_client(self, make_client, monkeypatch):
        _forbid_dcr(monkeypatch)
        client = make_client(self._stored())
        resp = client.post("/servers/atl/egress-auth", json=_dcr_body())
        assert resp.status_code == 200, resp.text
        eo = client._svc.updated_with["egress_oauth"]
        assert eo["client_id"] == "STORED"
        # The registration metadata is carried forward for the next save.
        assert eo["dcr_redirect_uri"] == routes._callback_url()
        assert eo["dcr_scopes"] == ["read:work", "offline_access", "read:account"]

    def test_provider_switch_registers_a_new_client(self, make_client, monkeypatch):
        # A server moving from another provider must not keep that provider's
        # client_id.
        calls = _record_dcr(monkeypatch)
        client = make_client(
            {
                "path": "/atl",
                "egress_oauth": {
                    "provider": "github",
                    "client_id": "GITHUB-APP",
                    "client_secret_encrypted": "enc:gh",
                },
            }
        )
        resp = client.post("/servers/atl/egress-auth", json=_dcr_body())
        assert resp.status_code == 200, resp.text
        assert calls["count"] == 1
        eo = client._svc.updated_with["egress_oauth"]
        assert eo["client_id"] == "NEWID"
        assert eo["client_secret_encrypted"] is None

    def test_registered_for_another_provider_registers_a_new_client(self, make_client, monkeypatch):
        calls = _record_dcr(monkeypatch)
        client = make_client(self._stored(dcr_provider="other-dcr-provider"))
        resp = client.post("/servers/atl/egress-auth", json=_dcr_body())
        assert resp.status_code == 200, resp.text
        assert calls["count"] == 1
        assert client._svc.updated_with["egress_oauth"]["client_id"] == "NEWID"

    def test_callback_url_change_registers_a_new_client(self, make_client, monkeypatch):
        calls = _record_dcr(monkeypatch)
        client = make_client(
            self._stored(dcr_redirect_uri="https://old-gateway.example.com/oauth2/egress/callback")
        )
        resp = client.post("/servers/atl/egress-auth", json=_dcr_body())
        assert resp.status_code == 200, resp.text
        assert calls["count"] == 1
        eo = client._svc.updated_with["egress_oauth"]
        assert eo["client_id"] == "NEWID"
        assert eo["dcr_redirect_uri"] == routes._callback_url()

    def test_scope_growth_registers_a_new_client(self, make_client, monkeypatch):
        # An authorization server may limit a client to its registered scope, so
        # a request for more than that needs a new registration.
        calls = _record_dcr(monkeypatch)
        client = make_client(self._stored())
        resp = client.post(
            "/servers/atl/egress-auth",
            json=_dcr_body(scopes=["read:work", "offline_access", "write:work"]),
        )
        assert resp.status_code == 200, resp.text
        assert calls["count"] == 1
        eo = client._svc.updated_with["egress_oauth"]
        assert eo["client_id"] == "NEWID"
        assert eo["dcr_scopes"] == ["read:work", "offline_access", "write:work", "read:account"]

    def test_scope_subset_reuses_the_stored_client(self, make_client, monkeypatch):
        _forbid_dcr(monkeypatch)
        client = make_client(self._stored())
        resp = client.post("/servers/atl/egress-auth", json=_dcr_body(scopes=["read:work"]))
        assert resp.status_code == 200, resp.text
        eo = client._svc.updated_with["egress_oauth"]
        assert eo["client_id"] == "STORED"
        assert eo["scopes"] == ["read:work", "read:account"]
        # The registered scope set is what the client can ask for, so it is kept.
        assert eo["dcr_scopes"] == ["read:work", "offline_access", "read:account"]

    def test_re_register_registers_a_new_client(self, make_client, monkeypatch):
        calls = _record_dcr(monkeypatch)
        client = make_client(self._stored())
        resp = client.post("/servers/atl/egress-auth", json=_dcr_body(re_register=True))
        assert resp.status_code == 200, resp.text
        assert calls["count"] == 1
        assert client._svc.updated_with["egress_oauth"]["client_id"] == "NEWID"

    def test_re_register_with_explicit_client_id_is_rejected(self, make_client, monkeypatch):
        _forbid_dcr(monkeypatch)
        client = make_client(self._stored())
        resp = client.post(
            "/servers/atl/egress-auth",
            json=_dcr_body(client_id="OPERATOR", re_register=True),
        )
        assert resp.status_code == 400
        assert resp.json()["detail"] == (
            "re_register cannot be combined with a client_id that differs from the stored one"
        )
        assert client._svc.updated_with is None

    # The admin UI sends the stored client_id back on every save, so an echoed
    # id must behave exactly like a save with no client_id.

    def test_echoed_stored_client_id_reuses_when_nothing_changed(self, make_client, monkeypatch):
        _forbid_dcr(monkeypatch)
        client = make_client(self._stored())
        resp = client.post("/servers/atl/egress-auth", json=_dcr_body(client_id="STORED"))
        assert resp.status_code == 200, resp.text
        eo = client._svc.updated_with["egress_oauth"]
        assert eo["client_id"] == "STORED"
        assert eo["dcr_redirect_uri"] == routes._callback_url()
        assert eo["dcr_scopes"] == ["read:work", "offline_access", "read:account"]

    def test_echoed_stored_client_id_with_grown_scopes_registers(self, make_client, monkeypatch):
        calls = _record_dcr(monkeypatch)
        client = make_client(self._stored())
        resp = client.post(
            "/servers/atl/egress-auth",
            json=_dcr_body(
                client_id="STORED", scopes=["read:work", "offline_access", "write:work"]
            ),
        )
        assert resp.status_code == 200, resp.text
        assert calls["count"] == 1
        eo = client._svc.updated_with["egress_oauth"]
        assert eo["client_id"] == "NEWID"
        assert eo["dcr_scopes"] == ["read:work", "offline_access", "write:work", "read:account"]

    def test_echoed_stored_client_id_after_callback_change_registers(
        self, make_client, monkeypatch
    ):
        calls = _record_dcr(monkeypatch)
        client = make_client(
            self._stored(dcr_redirect_uri="https://old-gateway.example.com/oauth2/egress/callback")
        )
        resp = client.post("/servers/atl/egress-auth", json=_dcr_body(client_id="STORED"))
        assert resp.status_code == 200, resp.text
        assert calls["count"] == 1
        eo = client._svc.updated_with["egress_oauth"]
        assert eo["client_id"] == "NEWID"
        assert eo["dcr_redirect_uri"] == routes._callback_url()

    def test_echoed_stored_client_id_for_another_provider_registers(self, make_client, monkeypatch):
        calls = _record_dcr(monkeypatch)
        client = make_client(self._stored(dcr_provider="other-dcr-provider"))
        resp = client.post("/servers/atl/egress-auth", json=_dcr_body(client_id="STORED"))
        assert resp.status_code == 200, resp.text
        assert calls["count"] == 1
        assert client._svc.updated_with["egress_oauth"]["client_id"] == "NEWID"

    def test_re_register_with_echoed_stored_client_id_registers(self, make_client, monkeypatch):
        # Not a conflict: the echoed id is not an override.
        calls = _record_dcr(monkeypatch)
        client = make_client(self._stored())
        resp = client.post(
            "/servers/atl/egress-auth",
            json=_dcr_body(client_id="STORED", re_register=True),
        )
        assert resp.status_code == 200, resp.text
        assert calls["count"] == 1
        assert client._svc.updated_with["egress_oauth"]["client_id"] == "NEWID"

    def test_different_explicit_client_id_stays_an_override(self, make_client, monkeypatch):
        # Even with a grown scope set and a changed callback, an id that differs
        # from the stored one is the operator's own client and is used as-is.
        _forbid_dcr(monkeypatch)
        client = make_client(
            self._stored(dcr_redirect_uri="https://old-gateway.example.com/oauth2/egress/callback")
        )
        resp = client.post(
            "/servers/atl/egress-auth",
            json=_dcr_body(client_id="OPERATOR", scopes=["read:work", "write:work"]),
        )
        assert resp.status_code == 200, resp.text
        eo = client._svc.updated_with["egress_oauth"]
        assert eo["client_id"] == "OPERATOR"
        # The stored client's registration metadata does not describe another id.
        assert "dcr_redirect_uri" not in eo

    def test_legacy_config_without_metadata_is_reused_for_the_same_provider(
        self, make_client, monkeypatch
    ):
        # Saved before the gateway recorded registration metadata: client_id only.
        _forbid_dcr(monkeypatch)
        client = make_client(
            {"path": "/atl", "egress_oauth": {"provider": DCR_TEST_PROVIDER, "client_id": "LEGACY"}}
        )
        resp = client.post("/servers/atl/egress-auth", json=_dcr_body())
        assert resp.status_code == 200, resp.text
        eo = client._svc.updated_with["egress_oauth"]
        assert eo["client_id"] == "LEGACY"
        assert "dcr_redirect_uri" not in eo

    def test_echoed_legacy_client_id_of_another_provider_registers_a_new_client(
        self, make_client, monkeypatch
    ):
        # A legacy config (client_id only, no dcr_* metadata) whose stored provider is
        # not the requested DCR provider. The admin UI echoes its client_id back, so
        # the id equals the stored one and is not an operator override: the provider
        # check still applies and the other provider's client must not be kept.
        calls = _record_dcr(monkeypatch)
        client = make_client(
            {"path": "/atl", "egress_oauth": {"provider": "custom", "client_id": "LEGACY"}}
        )
        resp = client.post("/servers/atl/egress-auth", json=_dcr_body(client_id="LEGACY"))
        assert resp.status_code == 200, resp.text
        assert calls["count"] == 1
        eo = client._svc.updated_with["egress_oauth"]
        assert eo["client_id"] == "NEWID"
        assert eo["dcr_provider"] == DCR_TEST_PROVIDER

    def test_legacy_config_for_another_provider_registers_a_new_client(
        self, make_client, monkeypatch
    ):
        calls = _record_dcr(monkeypatch)
        client = make_client(
            {"path": "/atl", "egress_oauth": {"provider": "custom", "client_id": "LEGACY"}}
        )
        resp = client.post("/servers/atl/egress-auth", json=_dcr_body())
        assert resp.status_code == 200, resp.text
        assert calls["count"] == 1
        assert client._svc.updated_with["egress_oauth"]["client_id"] == "NEWID"


@pytest.mark.unit
@pytest.mark.usefixtures("dcr_provider")
class TestConfigureEgressDefaultScopes:
    """Provider default_scopes applied when the operator supplies no explicit scopes."""

    @pytest.fixture(autouse=True)
    def _prm(self, monkeypatch):
        _stub_prm(monkeypatch)
        _forbid_dcr(monkeypatch)

    def _body(self, **over):
        base = {
            "egress_auth_mode": "oauth_user",
            "egress_provider": DCR_TEST_PROVIDER,
            "client_id": "",
            "scopes": [],  # no explicit scopes
        }
        base.update(over)
        return base

    def _client(self, make_client):
        # A legacy stored client, so these tests stay about scopes, not reuse.
        return make_client(
            {
                "path": "/atl",
                "egress_oauth": {"client_id": "EXISTING", "provider": DCR_TEST_PROVIDER},
            }
        )

    def test_default_scopes_applied_when_none_supplied(self, make_client):
        client = self._client(make_client)
        resp = client.post("/servers/atl/egress-auth", json=self._body())
        assert resp.status_code == 200, resp.text
        assert client._svc.updated_with["egress_oauth"]["scopes"] == DCR_TEST_DEFAULT_SCOPES

    def test_explicit_scopes_win_over_defaults(self, make_client):
        # read:account is appended regardless (see the required_scopes tests), so
        # it is included here to keep this test about defaults only.
        explicit = ["read:me", "offline_access", "read:account"]
        client = self._client(make_client)
        resp = client.post("/servers/atl/egress-auth", json=self._body(scopes=explicit))
        assert resp.status_code == 200, resp.text
        assert client._svc.updated_with["egress_oauth"]["scopes"] == explicit

    def test_required_scope_appended_to_explicit_scopes(self, make_client):
        # The case a plain default cannot cover: the operator supplied a list
        # without the mandatory scope, so it must be unioned in.
        client = self._client(make_client)
        resp = client.post("/servers/atl/egress-auth", json=self._body(scopes=["read:work"]))
        assert resp.status_code == 200, resp.text
        scopes = client._svc.updated_with["egress_oauth"]["scopes"]
        assert scopes == ["read:work", "read:account"], scopes

    def test_required_scope_not_duplicated(self, make_client):
        # Already-correct lists must be left untouched, not grow a duplicate.
        client = self._client(make_client)
        resp = client.post(
            "/servers/atl/egress-auth",
            json=self._body(scopes=["read:account", "read:work"]),
        )
        assert resp.status_code == 200, resp.text
        scopes = client._svc.updated_with["egress_oauth"]["scopes"]
        assert scopes == ["read:account", "read:work"], scopes

    def test_provider_without_required_scopes_unaffected(self, make_client):
        # A provider that declares no required_scopes must not gain any.
        client = make_client()
        resp = client.post(
            "/servers/gh/egress-auth",
            json={
                "egress_auth_mode": "oauth_user",
                "egress_provider": "github",
                "client_id": "cid",
                "client_secret": "s",
                "scopes": ["repo"],
            },
        )
        assert resp.status_code == 200, resp.text
        assert client._svc.updated_with["egress_oauth"]["scopes"] == ["repo"]

    def test_provider_without_defaults_unaffected(self, make_client):
        # A provider with no default_scopes (e.g. github) is not changed.
        client = make_client()
        resp = client.post(
            "/servers/gh/egress-auth",
            json={
                "egress_auth_mode": "oauth_user",
                "egress_provider": "github",
                "client_id": "cid",
                "client_secret": "s",
                "scopes": [],
            },
        )
        assert resp.status_code == 200, resp.text
        assert client._svc.updated_with["egress_oauth"]["scopes"] == []


@pytest.mark.unit
@pytest.mark.usefixtures("dcr_provider")
class TestConfigureScopeValidation:
    """Scope validation against PRM scopes_supported for requires_dcr providers."""

    def _body(self, scopes):
        return {
            "egress_auth_mode": "oauth_user",
            "egress_provider": DCR_TEST_PROVIDER,
            "scopes": scopes,
        }

    def _legacy_client(self, make_client):
        return make_client(
            {
                "path": "/atl",
                "egress_oauth": {"client_id": "EXISTING", "provider": DCR_TEST_PROVIDER},
            }
        )

    def test_valid_scopes_pass_validation(self, make_client, monkeypatch):
        _stub_prm(monkeypatch)
        _forbid_dcr(monkeypatch)
        client = self._legacy_client(make_client)
        resp = client.post("/servers/atl/egress-auth", json=self._body(_PRM_SCOPES))
        assert resp.status_code == 200, resp.text

    def test_unsupported_scope_rejected_with_400(self, make_client, monkeypatch):
        _stub_prm(monkeypatch)
        _forbid_dcr(monkeypatch)
        client = self._legacy_client(make_client)
        resp = client.post(
            "/servers/atl/egress-auth", json=self._body(["read:work", "read:legacy"])
        )
        assert resp.status_code == 400, resp.text
        assert "read:legacy" in resp.json()["detail"]

    def test_validation_skipped_when_prm_has_no_scopes_supported(self, make_client, monkeypatch):
        # A PRM that does not advertise scopes_supported skips validation rather
        # than failing closed on a provider that does not declare its scope list.
        _stub_prm(monkeypatch, doc={})
        _forbid_dcr(monkeypatch)
        client = self._legacy_client(make_client)
        resp = client.post("/servers/atl/egress-auth", json=self._body(["anything:goes"]))
        assert resp.status_code == 200, resp.text

    def test_scope_rejection_happens_before_dcr(self, make_client, monkeypatch):
        # Scope validation runs before registration, so a bad scope config never
        # leaves an orphaned client at the AS. No stored client here, so the
        # registration branch is the one that would run.
        _stub_prm(monkeypatch)
        calls = _record_dcr(monkeypatch)
        client = make_client({"path": "/atl", "egress_oauth": None})
        resp = client.post("/servers/atl/egress-auth", json=self._body(["read:legacy"]))
        assert resp.status_code == 400
        assert calls["count"] == 0

    def test_dcr_receives_the_resolved_scopes(self, make_client, monkeypatch):
        # The registration request carries the scopes actually persisted (the
        # provider defaults once substituted in), not the empty list posted.
        _stub_prm(monkeypatch)
        calls = _record_dcr(monkeypatch)
        client = make_client({"path": "/atl", "egress_oauth": None})
        resp = client.post("/servers/atl/egress-auth", json=self._body([]))
        assert resp.status_code == 200, resp.text
        assert calls["scopes"] == client._svc.updated_with["egress_oauth"]["scopes"]
        assert calls["scopes"] == DCR_TEST_DEFAULT_SCOPES

    def test_unusable_prm_is_fatal_when_registering(self, make_client, monkeypatch):
        # With no client to keep, the PRM is the only source of the registration
        # endpoint, so a failure is a 502 with the engine's fixed message.
        async def boom_prm(cfg):
            raise routes.oauth_engine.OAuthEngineError("discovery endpoint unreachable")

        monkeypatch.setattr(routes.oauth_engine, "fetch_protected_resource_metadata", boom_prm)
        client = make_client({"path": "/atl", "egress_oauth": None})
        resp = client.post("/servers/atl/egress-auth", json=self._body([]))
        assert resp.status_code == 502, resp.text
        assert resp.json()["detail"] == (
            "could not fetch protected-resource metadata: discovery endpoint unreachable"
        )

    def test_unusable_prm_degrades_to_unvalidated_when_reusing(self, make_client, monkeypatch):
        # Re-saving an already-registered server must not depend on the provider's
        # metadata endpoint: validation is skipped and the edit succeeds.
        async def boom_prm(cfg):
            raise routes.oauth_engine.OAuthEngineError("discovery endpoint unreachable")

        monkeypatch.setattr(routes.oauth_engine, "fetch_protected_resource_metadata", boom_prm)
        _forbid_dcr(monkeypatch)
        client = self._legacy_client(make_client)
        resp = client.post("/servers/atl/egress-auth", json=self._body(["read:work"]))
        assert resp.status_code == 200, resp.text
        assert client._svc.updated_with["egress_oauth"]["client_id"] == "EXISTING"


@pytest.mark.unit
@pytest.mark.usefixtures("dcr_provider")
class TestSharedProviderConfigBuilder:
    """build_oauth_provider_config is shared with backend-auth discovery, which
    cannot register a DCR client, so a requires_dcr provider must be refused
    there rather than persisted without a usable client."""

    def test_dcr_provider_refused_without_dcr_support(self):
        with pytest.raises(HTTPException) as exc_info:
            routes.build_oauth_provider_config(
                provider=DCR_TEST_PROVIDER, client_id="", client_secret=None, scopes=[]
            )
        assert exc_info.value.status_code == 400
        assert "supported only for per-user egress" in exc_info.value.detail

    def test_dcr_provider_returned_for_caller_to_complete(self):
        eo = routes.build_oauth_provider_config(
            provider=DCR_TEST_PROVIDER,
            client_id="",
            client_secret=None,
            scopes=["read:work"],
            dcr_supported=True,
        )
        assert eo["scopes"] == ["read:work", "read:account"]
        assert "client_secret_encrypted" not in eo


_ROVO_V2_SCOPES = [
    "read:me",
    "read:account",
    "offline_access",
    "read:jira:agent-interface",
    "write:jira:agent-interface",
    "search:jira:agent-interface",
]


@pytest.mark.unit
class TestAtlassianProviders:
    """The classic ``atlassian`` row is unchanged; ``atlassian-rovo`` uses DCR."""

    def test_rovo_registers_a_client_with_the_v2_default_scopes(self, make_client, monkeypatch):
        _stub_prm(monkeypatch, doc={"scopes_supported": _ROVO_V2_SCOPES})
        calls = _record_dcr(monkeypatch, client_id="ROVOID")
        client = make_client({"path": "/rovo", "egress_oauth": None})
        resp = client.post(
            "/servers/rovo/egress-auth",
            json={"egress_auth_mode": "oauth_user", "egress_provider": "atlassian-rovo"},
        )
        assert resp.status_code == 200, resp.text
        eo = client._svc.updated_with["egress_oauth"]
        assert eo["client_id"] == "ROVOID"
        assert eo["client_secret_encrypted"] is None
        assert eo["scopes"] == _ROVO_V2_SCOPES
        assert eo["dcr_provider"] == "atlassian-rovo"
        assert calls["scopes"] == _ROVO_V2_SCOPES
        assert calls["is_none_style"] is True

    def test_rovo_rejects_a_classic_scope(self, make_client, monkeypatch):
        # v2 has no read:jira-work; the classic scope list must fail at save time.
        _stub_prm(monkeypatch, doc={"scopes_supported": _ROVO_V2_SCOPES})
        calls = _record_dcr(monkeypatch)
        client = make_client({"path": "/rovo", "egress_oauth": None})
        resp = client.post(
            "/servers/rovo/egress-auth",
            json={
                "egress_auth_mode": "oauth_user",
                "egress_provider": "atlassian-rovo",
                "scopes": ["read:jira-work"],
            },
        )
        assert resp.status_code == 400, resp.text
        assert "read:jira-work" in resp.json()["detail"]
        assert calls["count"] == 0

    def test_switch_from_classic_atlassian_to_rovo_registers_a_new_client(
        self, make_client, monkeypatch
    ):
        _stub_prm(monkeypatch, doc={"scopes_supported": _ROVO_V2_SCOPES})
        calls = _record_dcr(monkeypatch, client_id="ROVOID")
        client = make_client(
            {
                "path": "/rovo",
                "egress_oauth": {
                    "provider": "atlassian",
                    "client_id": "CLASSIC-3LO",
                    "client_secret_encrypted": "enc:classic",
                },
            }
        )
        resp = client.post(
            "/servers/rovo/egress-auth",
            json={"egress_auth_mode": "oauth_user", "egress_provider": "atlassian-rovo"},
        )
        assert resp.status_code == 200, resp.text
        assert calls["count"] == 1
        eo = client._svc.updated_with["egress_oauth"]
        assert eo["client_id"] == "ROVOID"
        assert eo["client_secret_encrypted"] is None

    def test_classic_atlassian_resave_keeps_its_static_client(self, make_client, monkeypatch):
        # An existing classic 3LO config still saves after the upgrade: no DCR, no
        # metadata fetch, and the stored secret is kept.
        _forbid_dcr(monkeypatch)

        async def no_prm(cfg):
            raise AssertionError("a classic provider has no protected-resource metadata")

        monkeypatch.setattr(routes.oauth_engine, "fetch_protected_resource_metadata", no_prm)
        client = make_client(
            {
                "path": "/jira",
                "egress_oauth": {
                    "provider": "atlassian",
                    "client_id": "CLASSIC-3LO",
                    "client_secret_encrypted": "enc:classic",
                },
            }
        )
        resp = client.post(
            "/servers/jira/egress-auth",
            json={
                "egress_auth_mode": "oauth_user",
                "egress_provider": "atlassian",
                "client_id": "CLASSIC-3LO",
                "scopes": ["read:jira-work", "read:jira-user", "offline_access"],
            },
        )
        assert resp.status_code == 200, resp.text
        eo = client._svc.updated_with["egress_oauth"]
        assert eo["client_id"] == "CLASSIC-3LO"
        assert eo["client_secret_encrypted"] == "enc:classic"
        assert eo["scopes"] == ["read:jira-work", "read:jira-user", "offline_access"]

    def test_rovo_discovery_identity_is_refused(self):
        # Backend-auth discovery cannot register a client, so the shared builder
        # refuses a DCR provider when called without dcr_supported.
        with pytest.raises(HTTPException) as exc_info:
            routes.build_oauth_provider_config(
                provider="atlassian-rovo", client_id="", client_secret=None, scopes=[]
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == (
            "provider 'atlassian-rovo' uses dynamic client registration, which is "
            "supported only for per-user egress (oauth_user)"
        )
