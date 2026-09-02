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
from registry.egress_auth.schemas import TokenEndpointAuthStyle


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


@pytest.mark.unit
class TestConfigureEgressDcr:
    """DCR-at-registration for a ``requires_dcr`` provider (Atlassian authv2)."""

    @pytest.fixture(autouse=True)
    def _stub_prm(self, monkeypatch):
        """Stub PRM fetch for all DCR tests so they don't need a real network call.

        Returns a PRM with scopes_supported matching the DCR body's scopes so
        validation passes without any network activity. ``read:account`` is
        included because the atlassian recipe declares it in required_scopes and
        so appends it to every request -- the real authv2 metadata advertises it
        too, so omitting it here would only be an artefact of the stub.
        """

        async def fake_prm(cfg):
            return {"scopes_supported": ["read:jira-work", "offline_access", "read:account"]}

        monkeypatch.setattr(routes.oauth_engine, "fetch_protected_resource_metadata", fake_prm)

    def _dcr_body(self, **over):
        # No operator client_id/secret: the gateway must DCR-register its own.
        base = {
            "egress_auth_mode": "oauth_user",
            "egress_provider": "atlassian",
            "client_id": "",
            "scopes": ["read:jira-work", "offline_access"],
        }
        base.update(over)
        return base

    def test_dcr_registers_none_style_client_and_persists_client_id(self, make_client, monkeypatch):
        calls: dict = {}

        async def fake_register(cfg, redirect_uri, scopes, prm=None):
            calls["redirect_uri"] = redirect_uri
            calls["scopes"] = scopes
            calls["is_none_style"] = cfg.token_endpoint_auth_style == TokenEndpointAuthStyle.NONE
            return "DCRID", None  # NONE style -> no secret

        monkeypatch.setattr(routes.oauth_engine, "register_dcr_client", fake_register)
        client = make_client({"path": "/atl", "egress_oauth": None})
        resp = client.post("/servers/atl/egress-auth", json=self._dcr_body())
        assert resp.status_code == 200, resp.text
        eo = client._svc.updated_with["egress_oauth"]
        # Registered client_id persisted; no secret required for a NONE-style client.
        assert eo["client_id"] == "DCRID"
        assert eo.get("client_secret_encrypted") is None
        # DCR used the gateway callback as redirect and the requested scopes.
        assert calls["redirect_uri"].endswith("/oauth2/egress/callback")
        # The RESOLVED scopes, i.e. with the recipe's required read:account unioned
        # in -- registering the raw operator list would mint a client whose grant
        # can never complete consent.
        assert calls["scopes"] == ["read:jira-work", "offline_access", "read:account"]
        assert calls["is_none_style"] is True

    def test_dcr_skipped_when_client_id_already_present(self, make_client, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("DCR must not run when a client_id already exists")

        monkeypatch.setattr(routes.oauth_engine, "register_dcr_client", boom)
        client = make_client(
            {"path": "/atl", "egress_oauth": {"client_id": "EXISTING", "provider": "atlassian"}}
        )
        resp = client.post("/servers/atl/egress-auth", json=self._dcr_body())
        assert resp.status_code == 200, resp.text
        assert client._svc.updated_with["egress_oauth"]["client_id"] == "EXISTING"

    def test_dcr_failure_returns_502(self, make_client, monkeypatch):
        async def fail(cfg, redirect_uri, scopes, prm=None):
            raise routes.oauth_engine.OAuthEngineError("registration_endpoint missing")

        monkeypatch.setattr(routes.oauth_engine, "register_dcr_client", fail)
        client = make_client({"path": "/atl", "egress_oauth": None})
        resp = client.post("/servers/atl/egress-auth", json=self._dcr_body())
        assert resp.status_code == 502
        assert "dynamic client registration failed" in resp.json()["detail"]
        assert client._svc.updated_with is None

    def test_dcr_reuse_client_id_rotates_supplied_secret(self, make_client, monkeypatch):
        # Reusing an existing client_id while supplying a client_secret re-encrypts
        # the new secret. Only meaningful for a CONFIDENTIAL requires_dcr provider;
        # the built-in atlassian recipe is NONE-style and drops the secret instead
        # (see test_dcr_none_style_drops_supplied_secret).
        from registry.egress_auth.schemas import OAuthProviderConfig

        confidential = OAuthProviderConfig(
            name="atlassian",
            display_name="Atlassian",
            authorize_url="https://auth.atlassian.com/authorize",
            token_url="https://auth.atlassian.com/oauth/token",
            requires_dcr=True,
            # Default token_endpoint_auth_style is POST_BODY (confidential client).
        )
        monkeypatch.setattr(routes, "resolve_provider", lambda eo: confidential)

        def boom(*a, **k):
            raise AssertionError("DCR must not run when a client_id already exists")

        monkeypatch.setattr(routes.oauth_engine, "register_dcr_client", boom)
        client = make_client(
            {"path": "/atl", "egress_oauth": {"client_id": "EXISTING", "provider": "atlassian"}}
        )
        resp = client.post("/servers/atl/egress-auth", json=self._dcr_body(client_secret="rotated"))
        assert resp.status_code == 200, resp.text
        eo = client._svc.updated_with["egress_oauth"]
        assert eo["client_id"] == "EXISTING"
        assert eo["client_secret_encrypted"] == "enc:rotated"

    def test_dcr_none_style_drops_supplied_secret(self, make_client, monkeypatch):
        # A NONE-style DCR client proves possession with PKCE, so no secret is kept
        # even when one is supplied (or, as with Atlassian's authv2 DCR endpoint,
        # handed back by the AS): storing a credential the token leg never reads is
        # needless exposure.
        def boom(*a, **k):
            raise AssertionError("DCR must not run when a client_id already exists")

        monkeypatch.setattr(routes.oauth_engine, "register_dcr_client", boom)
        client = make_client(
            {"path": "/atl", "egress_oauth": {"client_id": "EXISTING", "provider": "atlassian"}}
        )
        resp = client.post("/servers/atl/egress-auth", json=self._dcr_body(client_secret="rotated"))
        assert resp.status_code == 200, resp.text
        eo = client._svc.updated_with["egress_oauth"]
        assert eo["client_id"] == "EXISTING"
        assert eo["client_secret_encrypted"] is None

    def test_dcr_none_style_discards_secret_returned_by_as(self, make_client, monkeypatch):
        # Atlassian's authv2 DCR returns a client_secret even when we register
        # token_endpoint_auth_method=none (verified live: HTTP 201, auth method
        # echoed as "none", client_secret present with client_secret_expires_at 0).
        # The NONE-style recipe must not persist it.
        async def fake_register(cfg, redirect_uri, scopes, prm=None):
            return "DCRID", "secret-we-did-not-ask-for"

        monkeypatch.setattr(routes.oauth_engine, "register_dcr_client", fake_register)
        client = make_client({"path": "/atl", "egress_oauth": None})
        resp = client.post("/servers/atl/egress-auth", json=self._dcr_body())
        assert resp.status_code == 200, resp.text
        eo = client._svc.updated_with["egress_oauth"]
        assert eo["client_id"] == "DCRID"
        assert eo["client_secret_encrypted"] is None

    def test_dcr_confidential_client_without_secret_rejected(self, make_client, monkeypatch):
        # A requires_dcr provider that is NOT NONE-style and yields no secret is a
        # provider/config error -> 502 (no such built-in today; force via resolve).
        from registry.egress_auth.schemas import OAuthProviderConfig

        confidential = OAuthProviderConfig(
            name="atlassian",
            display_name="Atlassian",
            authorize_url="https://auth.atlassian.com/authorize",
            token_url="https://auth.atlassian.com/oauth/token",
            requires_dcr=True,
            # Default token_endpoint_auth_style is POST_BODY (confidential client).
        )
        monkeypatch.setattr(routes, "resolve_provider", lambda eo: confidential)

        async def fake_register(cfg, redirect_uri, scopes, prm=None):
            return "DCRID", None  # confidential client but AS returned no secret

        monkeypatch.setattr(routes.oauth_engine, "register_dcr_client", fake_register)
        client = make_client({"path": "/atl", "egress_oauth": None})
        resp = client.post("/servers/atl/egress-auth", json=self._dcr_body())
        assert resp.status_code == 502
        assert "no client_secret" in resp.json()["detail"]
        assert client._svc.updated_with is None

    def test_dcr_no_client_secret_required(self, make_client, monkeypatch):
        # A public DCR client is valid with no secret -- the old "client_secret
        # required" gate must not fire for a requires_dcr provider.
        async def fake_register(cfg, redirect_uri, scopes, prm=None):
            return "DCRID", None

        monkeypatch.setattr(routes.oauth_engine, "register_dcr_client", fake_register)
        client = make_client({"path": "/atl", "egress_oauth": None})
        resp = client.post("/servers/atl/egress-auth", json=self._dcr_body())
        assert resp.status_code == 200, resp.text


@pytest.mark.unit
class TestConfigureEgressDefaultScopes:
    """Provider default_scopes applied when the operator supplies no explicit scopes."""

    _ATL_DEFAULTS = [
        "read:me",
        "read:account",
        "offline_access",
        "read:jira-work",
        "write:jira-work",
    ]

    def _body(self, **over):
        base = {
            "egress_auth_mode": "oauth_user",
            "egress_provider": "atlassian",
            "client_id": "EXISTING",
            "scopes": [],  # no explicit scopes
        }
        base.update(over)
        return base

    def test_default_scopes_applied_when_none_supplied(self, make_client, monkeypatch):
        # An atlassian server configured with no scopes gets the provider defaults.
        async def fake_prm(cfg):
            return {"scopes_supported": self._ATL_DEFAULTS}

        monkeypatch.setattr(routes.oauth_engine, "fetch_protected_resource_metadata", fake_prm)
        client = make_client(
            {"path": "/atl", "egress_oauth": {"client_id": "EXISTING", "provider": "atlassian"}}
        )
        resp = client.post("/servers/atl/egress-auth", json=self._body())
        assert resp.status_code == 200, resp.text
        eo = client._svc.updated_with["egress_oauth"]
        assert eo["scopes"] == self._ATL_DEFAULTS

    def test_explicit_scopes_win_over_defaults(self, make_client, monkeypatch):
        # When the operator supplies scopes they must not be replaced by defaults.
        # read:account is appended regardless -- see the required_scopes tests --
        # so it is included here to keep this test about defaults only.
        explicit = ["read:me", "offline_access", "read:account"]

        async def fake_prm(cfg):
            return {"scopes_supported": self._ATL_DEFAULTS}

        monkeypatch.setattr(routes.oauth_engine, "fetch_protected_resource_metadata", fake_prm)
        client = make_client(
            {"path": "/atl", "egress_oauth": {"client_id": "EXISTING", "provider": "atlassian"}}
        )
        resp = client.post("/servers/atl/egress-auth", json=self._body(scopes=explicit))
        assert resp.status_code == 200, resp.text
        assert client._svc.updated_with["egress_oauth"]["scopes"] == explicit

    def test_required_scope_appended_to_explicit_scopes(self, make_client, monkeypatch):
        # THE CASE A PLAIN DEFAULT CANNOT COVER. Atlassian authv2 rejects any
        # authorize request without read:account, and only after the user submits
        # consent, with an opaque invalid_request. Defaults do not help here
        # because the operator supplied a list, so the scope must be unioned in.
        async def fake_prm(cfg):
            return {"scopes_supported": self._ATL_DEFAULTS}

        monkeypatch.setattr(routes.oauth_engine, "fetch_protected_resource_metadata", fake_prm)
        client = make_client(
            {"path": "/atl", "egress_oauth": {"client_id": "EXISTING", "provider": "atlassian"}}
        )
        resp = client.post(
            "/servers/atl/egress-auth",
            json=self._body(scopes=["read:jira-work"]),  # no read:account
        )
        assert resp.status_code == 200, resp.text
        scopes = client._svc.updated_with["egress_oauth"]["scopes"]
        assert scopes == ["read:jira-work", "read:account"], scopes

    def test_required_scope_not_duplicated(self, make_client, monkeypatch):
        # Already-correct lists must be left untouched, not grow a duplicate.
        async def fake_prm(cfg):
            return {"scopes_supported": self._ATL_DEFAULTS}

        monkeypatch.setattr(routes.oauth_engine, "fetch_protected_resource_metadata", fake_prm)
        client = make_client(
            {"path": "/atl", "egress_oauth": {"client_id": "EXISTING", "provider": "atlassian"}}
        )
        resp = client.post(
            "/servers/atl/egress-auth",
            json=self._body(scopes=["read:account", "read:jira-work"]),
        )
        assert resp.status_code == 200, resp.text
        scopes = client._svc.updated_with["egress_oauth"]["scopes"]
        assert scopes == ["read:account", "read:jira-work"], scopes
        assert scopes.count("read:account") == 1

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
class TestConfigureScopeValidation:
    """Scope validation against PRM scopes_supported for requires_dcr providers."""

    _SUPPORTED = [
        "read:me",
        "read:account",
        "offline_access",
        "read:jira-work",
        "write:jira-work",
        "search:confluence",
        "read:confluence-user",
        "read:page:confluence",
        "write:page:confluence",
        "read:comment:confluence",
        "write:comment:confluence",
        "read:space:confluence",
        "read:hierarchical-content:confluence",
    ]

    def _body(self, scopes=None):
        return {
            "egress_auth_mode": "oauth_user",
            "egress_provider": "atlassian",
            "client_id": "EXISTING",
            "scopes": scopes if scopes is not None else self._SUPPORTED[:3],
        }

    def test_valid_scopes_pass_validation(self, make_client, monkeypatch):
        # The known-good 13-scope set must not be rejected.
        async def fake_prm(cfg):
            return {"scopes_supported": self._SUPPORTED}

        monkeypatch.setattr(routes.oauth_engine, "fetch_protected_resource_metadata", fake_prm)
        client = make_client(
            {"path": "/atl", "egress_oauth": {"client_id": "EXISTING", "provider": "atlassian"}}
        )
        resp = client.post("/servers/atl/egress-auth", json=self._body(scopes=self._SUPPORTED))
        assert resp.status_code == 200, resp.text

    def test_unsupported_scope_rejected_with_400(self, make_client, monkeypatch):
        # A classic 3LO scope not in scopes_supported triggers a 400 with the
        # offending scope named.
        async def fake_prm(cfg):
            return {"scopes_supported": self._SUPPORTED}

        monkeypatch.setattr(routes.oauth_engine, "fetch_protected_resource_metadata", fake_prm)
        client = make_client(
            {"path": "/atl", "egress_oauth": {"client_id": "EXISTING", "provider": "atlassian"}}
        )
        resp = client.post(
            "/servers/atl/egress-auth",
            json=self._body(
                scopes=["read:jira-work", "read:jira-user"]
            ),  # read:jira-user unsupported
        )
        assert resp.status_code == 400, resp.text
        assert "read:jira-user" in resp.json()["detail"]

    def test_validation_skipped_when_prm_has_no_scopes_supported(self, make_client, monkeypatch):
        # If the PRM does not advertise scopes_supported, skip validation (do not
        # fail closed on a provider that does not declare its scope list).
        async def fake_prm(cfg):
            return {}  # no scopes_supported key

        monkeypatch.setattr(routes.oauth_engine, "fetch_protected_resource_metadata", fake_prm)
        client = make_client(
            {"path": "/atl", "egress_oauth": {"client_id": "EXISTING", "provider": "atlassian"}}
        )
        resp = client.post(
            "/servers/atl/egress-auth",
            json=self._body(scopes=["anything:goes"]),
        )
        assert resp.status_code == 200, resp.text

    def test_scope_rejection_happens_before_dcr(self, make_client, monkeypatch):
        # Scope validation runs before DCR registration; a bad scope config must
        # NOT trigger a DCR call so no orphaned registration is left at the AS.
        # NOTE: client_id is deliberately omitted. With one supplied, reuse_cid is
        # truthy and the registration branch is never entered, so the assertion
        # would hold trivially and prove nothing about ordering.
        dcr_called = {"called": False}

        async def fake_prm(cfg):
            return {"scopes_supported": self._SUPPORTED}

        async def dcr_must_not_run(*a, **k):
            dcr_called["called"] = True
            return "DCRID", None

        monkeypatch.setattr(routes.oauth_engine, "fetch_protected_resource_metadata", fake_prm)
        monkeypatch.setattr(routes.oauth_engine, "register_dcr_client", dcr_must_not_run)
        client = make_client({"path": "/atl", "egress_oauth": None})
        resp = client.post(
            "/servers/atl/egress-auth",
            json={
                "egress_auth_mode": "oauth_user",
                "egress_provider": "atlassian",
                "scopes": ["read:jira-user"],  # unsupported
            },
        )
        assert resp.status_code == 400
        assert dcr_called["called"] is False

    def test_dcr_receives_the_resolved_scopes(self, make_client, monkeypatch):
        # The registration request must carry the scopes actually persisted --
        # i.e. the provider defaults once they have been substituted in -- not the
        # empty list the operator posted.
        seen = {}

        async def fake_prm(cfg):
            return {"scopes_supported": self._SUPPORTED}

        async def fake_register(cfg, redirect_uri, scopes, prm=None):
            seen["scopes"] = scopes
            return "DCRID", None

        monkeypatch.setattr(routes.oauth_engine, "fetch_protected_resource_metadata", fake_prm)
        monkeypatch.setattr(routes.oauth_engine, "register_dcr_client", fake_register)
        client = make_client({"path": "/atl", "egress_oauth": None})
        resp = client.post(
            "/servers/atl/egress-auth",
            json={
                "egress_auth_mode": "oauth_user",
                "egress_provider": "atlassian",
                "scopes": [],
            },
        )
        assert resp.status_code == 200, resp.text
        assert seen["scopes"] == client._svc.updated_with["egress_oauth"]["scopes"]
        assert seen["scopes"], "DCR must not be told the operator's empty scope list"

    def test_unreachable_prm_is_fatal_when_registering(self, make_client, monkeypatch):
        # With no client_id to reuse, the PRM is the only source of the
        # registration endpoint, so an unreachable one must fail loudly (502)
        # rather than persist a half-configured server.
        async def boom_prm(cfg):
            raise routes.oauth_engine.OAuthEngineError("connect timeout")

        monkeypatch.setattr(routes.oauth_engine, "fetch_protected_resource_metadata", boom_prm)
        client = make_client({"path": "/atl", "egress_oauth": None})
        resp = client.post(
            "/servers/atl/egress-auth",
            json={
                "egress_auth_mode": "oauth_user",
                "egress_provider": "atlassian",
                "scopes": [],
            },
        )
        assert resp.status_code == 502, resp.text
        assert "protected-resource metadata" in resp.json()["detail"]

    def test_unreachable_prm_degrades_to_unvalidated_when_reusing(self, make_client, monkeypatch):
        # Re-saving an already-registered server must not depend on the provider's
        # metadata endpoint being reachable: validation is skipped and the edit
        # succeeds. A bad scope still fails later at the consent screen.
        async def boom_prm(cfg):
            raise routes.oauth_engine.OAuthEngineError("connect timeout")

        monkeypatch.setattr(routes.oauth_engine, "fetch_protected_resource_metadata", boom_prm)
        client = make_client(
            {"path": "/atl", "egress_oauth": {"client_id": "EXISTING", "provider": "atlassian"}}
        )
        resp = client.post("/servers/atl/egress-auth", json=self._body(scopes=["read:jira-work"]))
        assert resp.status_code == 200, resp.text
        assert client._svc.updated_with["egress_oauth"]["client_id"] == "EXISTING"
