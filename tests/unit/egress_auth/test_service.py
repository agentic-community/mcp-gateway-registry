"""EgressAuthService orchestration tests.

Uses an in-memory SecretStore and stubs the OAuth engine's token calls. Covers
the full consent->store->vend->refresh->disconnect cycle plus the canonical
auth_method and the callback security guards (TTL, single-use replay,
account-swap).
"""

import pytest

from registry.egress_auth import oauth_engine, service, state_codec
from registry.egress_auth.schemas import StoredToken
from registry.egress_auth.service import (
    EgressAuthService,
    canonical_auth_method,
    is_per_user_auth_method,
)
from registry.secrets.interfaces import SecretStoreBase
from registry.utils.credential_encryption import encrypt_credential


class _InMemoryStore(SecretStoreBase):
    """In-process SecretStore for orchestration tests (no external backend)."""

    def __init__(self) -> None:
        self._data: dict[tuple[str, str, str, str], StoredToken] = {}

    async def put_token(self, auth_method, user_id, provider, server_path, token):
        self._data[(auth_method, user_id, provider, server_path)] = token

    async def get_token(self, auth_method, user_id, provider, server_path):
        return self._data.get((auth_method, user_id, provider, server_path))

    async def delete_token(self, auth_method, user_id, provider, server_path):
        self._data.pop((auth_method, user_id, provider, server_path), None)

    async def list_for_user(self, auth_method, user_id):
        return [
            (provider, server_path, token)
            for (am, uid, provider, server_path), token in self._data.items()
            if am == auth_method and uid == user_id
        ]

EGRESS_OAUTH = {
    "provider": "github",
    "client_id": "Iv1.testclient",
    "client_secret_encrypted": None,  # filled in fixture
    "scopes": ["repo", "read:user"],
}


@pytest.fixture(autouse=True)
def _reset_cipher():
    state_codec.reset_cipher_for_tests()
    yield
    state_codec.reset_cipher_for_tests()


@pytest.fixture
def egress_oauth():
    cfg = dict(EGRESS_OAUTH)
    cfg["client_secret_encrypted"] = encrypt_credential("ghs_testsecret")
    return cfg


@pytest.fixture
def svc():
    store = _InMemoryStore()
    return EgressAuthService(
        secret_store=store,
        callback_base_url="https://gw.example",
        refresh_skew_seconds=300,
        state_ttl_seconds=600,
    )


def _stub_exchange(monkeypatch, **token_over):
    async def fake_post(cfg, data, headers):
        return {
            "access_token": token_over.get("access_token", "at_new"),
            "refresh_token": token_over.get("refresh_token", "rt_new"),
            "token_type": "Bearer",
            "expires_in": token_over.get("expires_in", 3600),
            "scope": "repo read:user",
        }

    monkeypatch.setattr(oauth_engine, "_post_token", fake_post)


@pytest.mark.unit
class TestCanonicalAuthMethod:
    def test_cookie_path_maps_to_oauth2(self):
        vr = {"method": "session_cookie", "data": {"auth_method": "oauth2"}}
        assert canonical_auth_method(vr) == "oauth2"

    def test_cookie_path_defaults_oauth2_when_missing(self):
        assert canonical_auth_method({"method": "session_cookie", "data": {}}) == "oauth2"

    @pytest.mark.parametrize(
        "method",
        ["keycloak", "entra", "cognito", "okta", "auth0", "pingfederate", "jwt", "boto3"],
    )
    def test_idp_provider_methods_canonicalize_to_oauth2(self, method):
        # A bearer issued directly by the per-user IdP (notably a DCR client like
        # Claude Code / Codex presents) reports the provider name as `method`. It
        # MUST fold into the same `oauth2` bucket the cookie-consent path wrote,
        # else the DCR vend misses the vault and the user loops on consent.
        assert canonical_auth_method({"method": method}) == "oauth2"

    @pytest.mark.parametrize("method", ["network-trusted", "federation-static", "future-unknown"])
    def test_non_per_user_methods_pass_through_raw(self, method):
        # Non-per-user (and unknown) methods are NOT folded -- they pass through so
        # is_per_user_auth_method() still rejects them at vend time.
        assert canonical_auth_method({"method": method}) == method

    @pytest.mark.parametrize(
        "method,expected",
        [
            ("oauth2", True),
            ("okta", True),
            ("self_signed", True),
            ("network-trusted", False),
            ("federation-static", False),
            ("", False),
            ("future-unknown", False),  # fail-closed
        ],
    )
    def test_per_user_classification(self, method, expected):
        assert is_per_user_auth_method(method) is expected


@pytest.mark.unit
class TestConsentAndCallback:
    def test_build_consent_url(self, svc, egress_oauth):
        url = svc.build_consent_url(
            auth_method="oauth2",
            user_id="alice",
            client_id_audit="Iv1.testclient",
            session_id="sess-1",
            server_path="/github-mcp",
            egress_oauth=egress_oauth,
        )
        assert url.startswith("https://github.com/login/oauth/authorize?")
        assert "code_challenge=" in url and "state=" in url

    async def test_full_consent_store_then_vend(self, svc, egress_oauth, monkeypatch):
        _stub_exchange(monkeypatch)
        url = svc.build_consent_url(
            "oauth2", "alice", "Iv1.testclient", "sess-1", "/github-mcp", egress_oauth
        )
        state_blob = _extract_state(url)

        conn = await svc.handle_callback(
            code="the-code",
            state_blob=state_blob,
            egress_oauth=egress_oauth,
            current_user_id="alice",
            current_auth_method="oauth2",
        )
        assert conn.provider == "github" and conn.server_path == "/github-mcp"

        # vend hits (same canonical key the consent wrote under)
        token = await svc.get_valid_token("oauth2", "alice", "/github-mcp", egress_oauth)
        assert token == "at_new"

    async def test_replay_is_rejected(self, svc, egress_oauth, monkeypatch):
        _stub_exchange(monkeypatch)
        url = svc.build_consent_url(
            "oauth2", "alice", "Iv1.testclient", "sess-1", "/github-mcp", egress_oauth
        )
        state_blob = _extract_state(url)
        await svc.handle_callback("c", state_blob, egress_oauth, "alice", "oauth2")
        with pytest.raises(service.EgressAuthError, match="replay"):
            await svc.handle_callback("c", state_blob, egress_oauth, "alice", "oauth2")

    async def test_account_swap_rejected(self, svc, egress_oauth, monkeypatch):
        _stub_exchange(monkeypatch)
        url = svc.build_consent_url(
            "oauth2", "alice", "Iv1.testclient", "sess-1", "/github-mcp", egress_oauth
        )
        state_blob = _extract_state(url)
        # different user finishes the callback
        with pytest.raises(service.EgressAuthError, match="user mismatch"):
            await svc.handle_callback("c", state_blob, egress_oauth, "mallory", "oauth2")

    async def test_same_user_new_session_accepted(self, svc, egress_oauth, monkeypatch):
        # account-swap guard binds to (user_id, auth_method), NOT session_id, so a
        # fresh session for the same principal must still complete.
        _stub_exchange(monkeypatch)
        url = svc.build_consent_url(
            "oauth2", "alice", "Iv1.testclient", "sess-OLD", "/github-mcp", egress_oauth
        )
        state_blob = _extract_state(url)
        conn = await svc.handle_callback("c", state_blob, egress_oauth, "alice", "oauth2")
        assert conn.provider == "github"

    async def test_tampered_state_rejected(self, svc, egress_oauth):
        with pytest.raises(service.EgressAuthError, match="invalid state"):
            await svc.handle_callback("c", "garbage-state", egress_oauth, "alice", "oauth2")


@pytest.mark.unit
class TestVendRefreshDisconnect:
    async def test_vend_miss_returns_none(self, svc, egress_oauth):
        assert await svc.get_valid_token("oauth2", "nobody", "/github-mcp", egress_oauth) is None

    async def test_non_per_user_never_vends(self, svc, egress_oauth, monkeypatch):
        _stub_exchange(monkeypatch)
        # write a token under a network-trusted bucket directly, then confirm
        # get_valid_token refuses it (denylist) even though the entry exists.
        await svc._store.put_token(
            "network-trusted", "alice", "github", "/github-mcp", StoredToken(access_token="x")
        )
        assert (
            await svc.get_valid_token("network-trusted", "alice", "/github-mcp", egress_oauth)
            is None
        )

    async def test_near_expiry_triggers_refresh(self, svc, egress_oauth, monkeypatch):
        # seed an already-expired token, then vend -> single-flight refresh fires.
        await svc._store.put_token(
            "oauth2",
            "alice",
            "github",
            "/github-mcp",
            StoredToken(
                access_token="old",
                refresh_token="rt_old",
                expires_at="2000-01-01T00:00:00+00:00",
                client_id="Iv1.testclient",
            ),
        )
        _stub_exchange(monkeypatch, access_token="at_refreshed")
        token = await svc.get_valid_token("oauth2", "alice", "/github-mcp", egress_oauth)
        assert token == "at_refreshed"

    async def test_dead_refresh_marks_failed_and_then_misses(self, svc, egress_oauth, monkeypatch):
        await svc._store.put_token(
            "oauth2",
            "alice",
            "github",
            "/github-mcp",
            StoredToken(
                access_token="old",
                refresh_token="rt_dead",
                expires_at="2000-01-01T00:00:00+00:00",
                client_id="Iv1.testclient",
            ),
        )

        async def dead_post(cfg, data, headers):
            raise oauth_engine.DeadRefreshTokenError("invalid_grant")

        monkeypatch.setattr(oauth_engine, "_post_token", dead_post)
        assert await svc.get_valid_token("oauth2", "alice", "/github-mcp", egress_oauth) is None
        # entry is now refresh_failed -> still a miss (consent needed), no retry storm
        stored = await svc._store.get_token("oauth2", "alice", "github", "/github-mcp")
        assert stored.status == "refresh_failed"
        assert await svc.get_valid_token("oauth2", "alice", "/github-mcp", egress_oauth) is None

    async def test_rotated_client_id_forces_reconsent(self, svc, egress_oauth):
        await svc._store.put_token(
            "oauth2",
            "alice",
            "github",
            "/github-mcp",
            StoredToken(access_token="a", client_id="OLD-client-id"),
        )
        # configured client_id is Iv1.testclient != OLD-client-id -> no vend
        assert await svc.get_valid_token("oauth2", "alice", "/github-mcp", egress_oauth) is None

    async def test_list_and_disconnect(self, svc, egress_oauth, monkeypatch):
        _stub_exchange(monkeypatch)
        url = svc.build_consent_url(
            "oauth2", "alice", "Iv1.testclient", "s", "/github-mcp", egress_oauth
        )
        await svc.handle_callback("c", _extract_state(url), egress_oauth, "alice", "oauth2")

        conns = await svc.list_connections("oauth2", "alice")
        assert [(c.provider, c.server_path) for c in conns] == [("github", "/github-mcp")]
        # tokens never leak into the connection view
        assert not hasattr(conns[0], "access_token")

        await svc.disconnect("oauth2", "alice", "github", "/github-mcp")
        assert await svc.list_connections("oauth2", "alice") == []

    async def test_list_for_non_per_user_is_empty(self, svc):
        assert await svc.list_connections("network-trusted", "alice") == []


def _extract_state(authorize_url: str) -> str:
    from urllib.parse import parse_qs, urlparse

    return parse_qs(urlparse(authorize_url).query)["state"][0]
