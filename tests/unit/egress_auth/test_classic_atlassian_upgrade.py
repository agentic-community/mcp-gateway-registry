"""An existing classic ``atlassian`` egress config keeps working after the upgrade.

Adding the ``atlassian-rovo`` provider must not change what a deployment that
configured the classic ``atlassian`` provider (a 3LO app: confidential client,
``api.atlassian.com`` audience) gets from refresh, consent and a re-save. These
tests start from the stored shape such a deployment has, with no ``dcr_*``
fields, and drive the real service, engine and configure route. Only the HTTP
transport is replaced, so the request that would go to Atlassian is inspected and
any discovery or registration call (which also goes through that transport) is
caught.
"""

from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import registry.api.egress_auth_routes as routes
from registry.egress_auth import oauth_engine
from registry.egress_auth.browser_binding import new_browser_binding
from registry.egress_auth.schemas import StoredToken
from registry.egress_auth.service import EgressAuthService
from registry.secrets import keys
from registry.utils.credential_encryption import decrypt_credential, encrypt_credential
from tests.unit.egress_auth.test_service import _InMemoryStore

TOKEN_URL = "https://auth.atlassian.com/oauth/token"
AUTHORIZE_URL = "https://auth.atlassian.com/authorize"
CLASSIC_SCOPES = [
    "read:me",
    "offline_access",
    "read:jira-work",
    "write:jira-work",
    "read:jira-user",
]
OPERATOR_CLIENT_ID = "operator-3lo-client-id"
OPERATOR_SECRET = "operator-3lo-client-secret"  # nosec B105 - test value, not a credential
UPSTREAM = "https://mcp.atlassian.com"
SERVER_PATH = "/jira"
_BROWSER = new_browser_binding()


def _stored_config() -> dict:
    """The egress_oauth a pre-upgrade deployment has: operator app, classic scopes."""
    return {
        "provider": "atlassian",
        "client_id": OPERATOR_CLIENT_ID,
        "client_secret_encrypted": encrypt_credential(OPERATOR_SECRET),
        "scopes": list(CLASSIC_SCOPES),
    }


class _RecordingTransport:
    """Stands in for the guarded HTTP client and records every request."""

    def __init__(self) -> None:
        self.gets: list[str] = []
        self.posts: list[dict] = []

    def __call__(self, *, profile, verify=True):
        del profile, verify
        return self

    async def get(self, url, headers=None, timeout=None):
        self.gets.append(url)
        raise AssertionError(f"unexpected discovery GET {url}")

    async def post(self, url, data=None, json=None, headers=None, timeout=None):
        self.posts.append({"url": url, "data": data, "json": json, "headers": headers})
        if json is not None:
            raise AssertionError(f"unexpected registration POST {url}")
        return _TokenReply()


class _TokenReply:
    status_code = 200

    def json(self):
        return {"access_token": "at_refreshed", "refresh_token": "rt_rotated", "expires_in": 3600}


@pytest.fixture
def transport(monkeypatch):
    recorder = _RecordingTransport()
    monkeypatch.setattr(oauth_engine, "shared_guarded_async_client", recorder)
    return recorder


@pytest.fixture
def svc():
    return EgressAuthService(
        secret_store=_InMemoryStore(),
        callback_base_url="https://gw.example",
        refresh_skew_seconds=300,
        state_ttl_seconds=600,
    )


@pytest.mark.unit
class TestClassicAtlassianRefreshAndConsent:
    async def test_refresh_sends_the_client_secret_to_the_classic_token_endpoint(
        self, svc, transport
    ):
        await svc._store.put_token(
            "oauth2",
            "alice",
            "atlassian",
            SERVER_PATH,
            StoredToken(
                access_token="old",
                refresh_token="rt_old",
                expires_at="2000-01-01T00:00:00+00:00",
                client_id=OPERATOR_CLIENT_ID,
                bound_upstreams=[UPSTREAM],
            ),
            purpose=keys.EGRESS_PURPOSE,
        )

        token = await svc.get_valid_token(
            "oauth2",
            "alice",
            SERVER_PATH,
            _stored_config(),
            requested_upstream=UPSTREAM,
            purpose=keys.EGRESS_PURPOSE,
        )

        assert token == "at_refreshed"
        assert transport.gets == []
        (request,) = transport.posts
        assert request["url"] == TOKEN_URL
        # The confidential, form-body style: client_id and client_secret in the
        # body, no Basic header, and the refresh token the user connected with.
        assert request["data"]["grant_type"] == "refresh_token"
        assert request["data"]["refresh_token"] == "rt_old"
        assert request["data"]["client_id"] == OPERATOR_CLIENT_ID
        assert request["data"]["client_secret"] == OPERATOR_SECRET
        assert "Authorization" not in request["headers"]

    async def test_consent_url_needs_no_registration_and_keeps_the_audience(self, svc, transport):
        url = await svc.build_consent_url(
            auth_method="oauth2",
            user_id="alice",
            client_id_audit=OPERATOR_CLIENT_ID,
            session_id="sess-1",
            server_path=SERVER_PATH,
            egress_oauth=_stored_config(),
            server={"proxy_pass_url": UPSTREAM},
            browser_binding=_BROWSER.digest,
        )

        parsed = urlparse(url)
        assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == AUTHORIZE_URL
        query = parse_qs(parsed.query)
        assert query["audience"] == ["api.atlassian.com"]
        assert query["prompt"] == ["consent"]
        assert query["client_id"] == [OPERATOR_CLIENT_ID]
        assert query["scope"] == [" ".join(CLASSIC_SCOPES)]
        assert query["code_challenge_method"] == ["S256"]
        assert "resource" not in query
        # No discovery, registration or token request was needed to build it.
        assert transport.gets == []
        assert transport.posts == []


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
    def _build(egress_oauth: dict):
        monkeypatch.setattr(routes.settings, "egress_auth_enabled", True)
        svc = _StubServerService(
            {
                "path": SERVER_PATH,
                "egress_auth_mode": "oauth_user",
                "egress_oauth": egress_oauth,
            }
        )
        monkeypatch.setattr(routes, "server_service", svc)
        app = FastAPI()
        app.include_router(routes.router)
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


def _resave_body(**over) -> dict:
    """What the admin UI sends when an operator re-saves: the stored client_id echoed."""
    body = {
        "egress_auth_mode": "oauth_user",
        "egress_provider": "atlassian",
        "client_id": OPERATOR_CLIENT_ID,
        "scopes": list(CLASSIC_SCOPES),
    }
    body.update(over)
    return body


@pytest.mark.unit
class TestClassicAtlassianResave:
    def _forbid_dcr(self, monkeypatch):
        async def _fail(*_args, **_kwargs):
            raise AssertionError("a classic atlassian save must not use DCR")

        monkeypatch.setattr(oauth_engine, "register_dcr_client", _fail)
        monkeypatch.setattr(oauth_engine, "fetch_protected_resource_metadata", _fail)

    def test_resave_keeps_the_operator_client_and_secret(self, make_client, transport, monkeypatch):
        self._forbid_dcr(monkeypatch)
        stored = _stored_config()
        client = make_client(stored)

        resp = client.post(f"/servers{SERVER_PATH}/egress-auth", json=_resave_body())

        assert resp.status_code == 200, resp.text
        eo = client._svc.updated_with["egress_oauth"]
        assert eo["provider"] == "atlassian"
        assert eo["client_id"] == OPERATOR_CLIENT_ID
        # A blank client_secret keeps the stored one.
        assert decrypt_credential(eo["client_secret_encrypted"]) == OPERATOR_SECRET
        # The classic scopes are accepted as sent: no 400 from a scope check.
        assert eo["scopes"] == CLASSIC_SCOPES
        assert not [key for key in eo if key.startswith("dcr_")]
        assert transport.gets == []
        assert transport.posts == []

    def test_resave_with_a_rotated_secret_stores_the_new_one(
        self, make_client, transport, monkeypatch
    ):
        self._forbid_dcr(monkeypatch)
        client = make_client(_stored_config())

        resp = client.post(
            f"/servers{SERVER_PATH}/egress-auth", json=_resave_body(client_secret="rotated")
        )

        assert resp.status_code == 200, resp.text
        eo = client._svc.updated_with["egress_oauth"]
        assert eo["client_id"] == OPERATOR_CLIENT_ID
        assert decrypt_credential(eo["client_secret_encrypted"]) == "rotated"
        assert transport.gets == []
        assert transport.posts == []
