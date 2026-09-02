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
from registry.secrets import keys
from registry.secrets.interfaces import SecretStoreBase
from registry.utils.credential_encryption import encrypt_credential


class _InMemoryStore(SecretStoreBase):
    """In-process SecretStore for orchestration tests (no external backend)."""

    def __init__(self) -> None:
        self._data: dict[tuple[str, str, str, str, str], StoredToken] = {}

    async def put_token(self, auth_method, user_id, provider, server_path, token, *, purpose):
        self._data[(purpose, auth_method, user_id, provider, server_path)] = token

    async def get_token(self, auth_method, user_id, provider, server_path, *, purpose):
        return self._data.get((purpose, auth_method, user_id, provider, server_path))

    async def delete_token(self, auth_method, user_id, provider, server_path, *, purpose):
        self._data.pop((purpose, auth_method, user_id, provider, server_path), None)

    async def list_for_user(self, auth_method, user_id):
        # Egress space only, mirroring the real backends: the identity the registry
        # borrowed is not one of the user's own connections.
        return [
            (provider, server_path, token)
            for (purpose, am, uid, provider, server_path), token in self._data.items()
            if am == auth_method and uid == user_id and purpose == keys.EGRESS_PURPOSE
        ]


EGRESS_OAUTH = {
    "provider": "github",
    "client_id": "Iv1.testclient",
    "client_secret_encrypted": None,  # filled in fixture
    "scopes": ["repo", "read:user"],
}

# A registered upstream base for these orchestration tests: the consent write
# binds the credential to this set and the vend must request the same base.
BOUND = "https://api.example.com"
BOUND_DD = "https://mcp.example.net"


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
def egress_oauth_public():
    """A custom public-client config (token_endpoint_auth_method=none): no secret."""
    return {
        "provider": "custom",
        "client_id": "dcr-public-client-id",
        "client_secret_encrypted": None,
        "scopes": [],
        "custom_authorize_url": "https://app.datadoghq.com/oauth2/v1/authorize",
        "custom_token_url": "https://app.datadoghq.com/api/v2/oauth2/token",
        "custom_token_auth_style": "none",
        "custom_resource": "https://mcp.datadoghq.com/api/unstable/mcp-server/mcp",
    }


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
            bound_upstreams=[BOUND],
        )
        assert conn.provider == "github" and conn.server_path == "/github-mcp"

        # vend hits (same canonical key the consent wrote under)
        token = await svc.get_valid_token(
            "oauth2",
            "alice",
            "/github-mcp",
            egress_oauth,
            requested_upstream=BOUND,
            purpose=keys.EGRESS_PURPOSE,
        )
        assert token == "at_new"

    async def test_cross_purpose_consents_coexist(self, svc, egress_oauth, monkeypatch):
        """A discovery consent and the same admin's own egress consent must BOTH stand.

        `purpose` is part of the vault address, so these two do not share an entry. That
        turns what used to be a refused write into the behaviour operators actually want:
        one person can hold their own runtime connection to a server AND be the
        designated discovery identity for it, at the same time, with neither evicting
        nor re-scoping the other.

        Both directions of the old damage are covered by construction -- there is no
        shared entry to evict, and no second write to re-scope the first.
        """
        _stub_exchange(monkeypatch)
        for session, purpose in (("sess-1", "egress"), ("sess-2", "discovery")):
            url = svc.build_consent_url(
                "oauth2",
                "alice",
                "Iv1.testclient",
                session,
                "/github-mcp",
                egress_oauth,
                purpose=purpose,
            )
            conn = await svc.handle_callback(
                "c",
                _extract_state(url),
                egress_oauth,
                "alice",
                "oauth2",
                bound_upstreams=[BOUND],
            )
            assert conn.server_path == "/github-mcp"

        # Two entries, not one: the second consent did not overwrite the first.
        for purpose in (keys.EGRESS_PURPOSE, keys.DISCOVERY_PURPOSE):
            assert (
                await svc._store.get_token(
                    "oauth2", "alice", "github", "/github-mcp", purpose=purpose
                )
                is not None
            )
        # And the user's own connection list shows only their own connection.
        conns = await svc.list_connections("oauth2", "alice")
        assert len(conns) == 1

    async def test_same_purpose_re_consent_survives_a_rotated_client(
        self, svc, egress_oauth, monkeypatch
    ):
        """The guard must NOT key on client_id.

        A client_id difference is also the signature of an ordinary rotated-provider
        -app re-consent, which `get_valid_token` deliberately forces by returning None
        on its client-id binding check. Refusing it would deadlock that documented

        recovery: the vend sends the user to re-consent, the re-consent is refused, and
        the callback renders a generic "try again" that can never succeed. This is the
        regression guard for that loop.
        """
        _stub_exchange(monkeypatch)
        url = svc.build_consent_url(
            "oauth2", "alice", "Iv1.testclient", "sess-1", "/github-mcp", egress_oauth
        )
        await svc.handle_callback(
            "c1", _extract_state(url), egress_oauth, "alice", "oauth2", bound_upstreams=[BOUND]
        )

        rotated = {**egress_oauth, "client_id": "Iv1.rotatedclient"}
        url2 = svc.build_consent_url(
            "oauth2", "alice", "Iv1.rotatedclient", "sess-2", "/github-mcp", rotated
        )
        conn = await svc.handle_callback(
            "c2", _extract_state(url2), rotated, "alice", "oauth2", bound_upstreams=[BOUND]
        )
        assert conn.server_path == "/github-mcp"
        # And the rotated credential is what a subsequent vend returns.
        assert (
            await svc.get_valid_token(
                "oauth2",
                "alice",
                "/github-mcp",
                rotated,
                requested_upstream=BOUND,
                purpose=keys.EGRESS_PURPOSE,
            )
            == "at_new"
        )

    async def test_re_consent_with_the_same_client_still_works(
        self, svc, egress_oauth, monkeypatch
    ):
        """The guard must not block ordinary re-consent (expiry, revoked grant)."""
        _stub_exchange(monkeypatch)
        for session in ("sess-1", "sess-2"):
            url = svc.build_consent_url(
                "oauth2", "alice", "Iv1.testclient", session, "/github-mcp", egress_oauth
            )
            conn = await svc.handle_callback(
                "c", _extract_state(url), egress_oauth, "alice", "oauth2", bound_upstreams=[BOUND]
            )
            assert conn.server_path == "/github-mcp"

    @pytest.mark.parametrize(
        "written_as,read_as", [("egress", "discovery"), ("discovery", "egress")]
    )
    async def test_vend_cannot_reach_the_other_purpose(
        self, svc, egress_oauth, monkeypatch, written_as, read_as
    ):
        """A reader must not be able to reach the other purpose's credential.

        Not "is refused after reading it" -- the two live at different vault addresses,
        so the wrong-purpose read finds nothing at all. Both directions matter: an
        egress entry read as discovery would put the registry's health / discovery /
        scan calls on a token the user consented for themselves, and a discovery entry
        read as egress would hand the registry's designated credential to that user's
        runtime hop.
        """
        _stub_exchange(monkeypatch)
        url = svc.build_consent_url(
            "oauth2",
            "alice",
            "Iv1.testclient",
            "sess-1",
            "/github-mcp",
            egress_oauth,
            purpose=written_as,
        )
        await svc.handle_callback(
            "c", _extract_state(url), egress_oauth, "alice", "oauth2", bound_upstreams=[BOUND]
        )

        async def _vend(purpose):
            return await svc.get_valid_token(
                "oauth2",
                "alice",
                "/github-mcp",
                egress_oauth,
                requested_upstream=BOUND,
                purpose=purpose,
            )

        # Same principal, provider, server and client -- only the purpose differs.
        assert await _vend(read_as) is None
        # ...and the matching purpose still vends, so this is a boundary, not a block.
        assert await _vend(written_as) == "at_new"

    @pytest.mark.parametrize("purpose", ["egress", "discovery"])
    async def test_refresh_keeps_an_entry_in_its_own_purpose(
        self, svc, egress_oauth, monkeypatch, purpose
    ):
        """A refresh must not move an entry between purposes.

        Regression guard for a real defect: while `purpose` was a mutable field on the
        stored payload, the refresh rebuilt the token via `model_copy` and did not carry
        it over, so the model default silently reclassified a refreshed discovery entry
        as egress. That inverted the separation -- the end-user runtime hop would then
        accept the registry's borrowed credential -- and it was reachable on the
        ordinary borrow within one token lifetime.

        Now the address carries the purpose and the payload does not, so the write-back
        lands where it was read from. This exercises the REAL service and store rather
        than a stubbed egress service, which is why the original defect survived: every
        borrow test substituted a fake that never refreshed.
        """
        _stub_exchange(monkeypatch)
        url = svc.build_consent_url(
            "oauth2",
            "alice",
            "Iv1.testclient",
            "sess-1",
            "/github-mcp",
            egress_oauth,
            purpose=purpose,
        )
        await svc.handle_callback(
            "c", _extract_state(url), egress_oauth, "alice", "oauth2", bound_upstreams=[BOUND]
        )

        # Force the stored entry near expiry so the next vend actually refreshes.
        stored = await svc._store.get_token(
            "oauth2", "alice", "github", "/github-mcp", purpose=purpose
        )
        await svc._store.put_token(
            "oauth2",
            "alice",
            "github",
            "/github-mcp",
            stored.model_copy(update={"expires_at": "2000-01-01T00:00:00+00:00"}),
            purpose=purpose,
        )
        _stub_exchange(monkeypatch, access_token="at_refreshed")

        async def _vend(as_purpose):
            return await svc.get_valid_token(
                "oauth2",
                "alice",
                "/github-mcp",
                egress_oauth,
                requested_upstream=BOUND,
                purpose=as_purpose,
            )

        assert await _vend(purpose) == "at_refreshed"
        # The refresh wrote back to the same space, and created nothing in the other.
        other = "discovery" if purpose == "egress" else "egress"
        assert await _vend(other) is None
        assert (
            await svc._store.get_token("oauth2", "alice", "github", "/github-mcp", purpose=other)
            is None
        )

    async def test_replay_is_rejected(self, svc, egress_oauth, monkeypatch):
        _stub_exchange(monkeypatch)
        url = svc.build_consent_url(
            "oauth2", "alice", "Iv1.testclient", "sess-1", "/github-mcp", egress_oauth
        )
        state_blob = _extract_state(url)
        await svc.handle_callback(
            "c", state_blob, egress_oauth, "alice", "oauth2", bound_upstreams=[BOUND]
        )
        with pytest.raises(service.EgressAuthError, match="replay"):
            await svc.handle_callback(
                "c", state_blob, egress_oauth, "alice", "oauth2", bound_upstreams=[BOUND]
            )

    async def test_account_swap_rejected(self, svc, egress_oauth, monkeypatch):
        _stub_exchange(monkeypatch)
        url = svc.build_consent_url(
            "oauth2", "alice", "Iv1.testclient", "sess-1", "/github-mcp", egress_oauth
        )
        state_blob = _extract_state(url)
        # different user finishes the callback
        with pytest.raises(service.EgressAuthError, match="user mismatch"):
            await svc.handle_callback(
                "c", state_blob, egress_oauth, "mallory", "oauth2", bound_upstreams=[BOUND]
            )

    async def test_same_user_new_session_accepted(self, svc, egress_oauth, monkeypatch):
        # account-swap guard binds to (user_id, auth_method), NOT session_id, so a
        # fresh session for the same principal must still complete.
        _stub_exchange(monkeypatch)
        url = svc.build_consent_url(
            "oauth2", "alice", "Iv1.testclient", "sess-OLD", "/github-mcp", egress_oauth
        )
        state_blob = _extract_state(url)
        conn = await svc.handle_callback(
            "c", state_blob, egress_oauth, "alice", "oauth2", bound_upstreams=[BOUND]
        )
        assert conn.provider == "github"

    async def test_tampered_state_rejected(self, svc, egress_oauth):
        with pytest.raises(service.EgressAuthError, match="invalid state"):
            await svc.handle_callback(
                "c", "garbage-state", egress_oauth, "alice", "oauth2", bound_upstreams=[BOUND]
            )

    async def test_confidential_provider_still_requires_secret(self, svc, monkeypatch):
        # A confidential (github) config whose stored secret is missing must
        # fail closed at the callback -- NOT silently proceed secretless.
        _stub_exchange(monkeypatch)
        secretless = dict(EGRESS_OAUTH)
        secretless["client_secret_encrypted"] = None
        url = svc.build_consent_url(
            "oauth2", "alice", "Iv1.testclient", "sess-1", "/github-mcp", secretless
        )
        with pytest.raises(service.EgressAuthError, match="client_secret_encrypted missing"):
            await svc.handle_callback(
                "c", _extract_state(url), secretless, "alice", "oauth2", bound_upstreams=[BOUND]
            )


@pytest.mark.unit
class TestPublicClientFlow:
    """Public client (custom provider, token_endpoint_auth_method=none): the
    whole consent->exchange->vend->refresh cycle works with NO stored secret,
    and no ``client_secret`` ever reaches the token endpoint."""

    async def test_consent_exchange_vend_without_secret(
        self, svc, egress_oauth_public, monkeypatch
    ):
        captured: dict = {}

        async def fake_post(cfg, data, headers):
            captured.update(data)
            return {
                "access_token": "at_public",
                "refresh_token": "rt_public",
                "token_type": "Bearer",
                "expires_in": 3600,
            }

        monkeypatch.setattr(oauth_engine, "_post_token", fake_post)
        url = svc.build_consent_url(
            "oauth2",
            "alice",
            "dcr-public-client-id",
            "sess-1",
            "/datadog-user",
            egress_oauth_public,
        )
        conn = await svc.handle_callback(
            "the-code",
            _extract_state(url),
            egress_oauth_public,
            "alice",
            "oauth2",
            bound_upstreams=[BOUND_DD],
        )
        assert conn.provider == "custom" and conn.server_path == "/datadog-user"
        assert "client_secret" not in captured
        assert captured["client_id"] == "dcr-public-client-id"
        assert captured["code_verifier"]  # PKCE verifier always sent
        assert captured["resource"] == egress_oauth_public["custom_resource"]

        token = await svc.get_valid_token(
            "oauth2",
            "alice",
            "/datadog-user",
            egress_oauth_public,
            requested_upstream=BOUND_DD,
            purpose=keys.EGRESS_PURPOSE,
        )
        assert token == "at_public"

    async def test_refresh_without_secret(self, svc, egress_oauth_public, monkeypatch):
        captured: dict = {}

        async def fake_post(cfg, data, headers):
            captured.update(data)
            return {"access_token": "at_refreshed", "expires_in": 3600}

        monkeypatch.setattr(oauth_engine, "_post_token", fake_post)
        await svc._store.put_token(
            "oauth2",
            "alice",
            "custom",
            "/datadog-user",
            StoredToken(
                access_token="old",
                refresh_token="rt_old",
                expires_at="2000-01-01T00:00:00+00:00",
                client_id="dcr-public-client-id",
                bound_upstreams=[BOUND_DD],
            ),
            purpose=keys.EGRESS_PURPOSE,
        )
        token = await svc.get_valid_token(
            "oauth2",
            "alice",
            "/datadog-user",
            egress_oauth_public,
            requested_upstream=BOUND_DD,
            purpose=keys.EGRESS_PURPOSE,
        )
        assert token == "at_refreshed"
        assert captured["grant_type"] == "refresh_token"
        assert "client_secret" not in captured
        assert captured["resource"] == egress_oauth_public["custom_resource"]

    def test_build_consent_url_unregistered_dcr_raises(self, svc):
        # A requires_dcr provider (atlassian) whose client_id has not been
        # registered yet must fail with a clear, actionable error rather than a
        # KeyError/500 when someone starts consent before the config-time DCR ran.
        eo = {"provider": "atlassian", "scopes": ["read:me"]}  # no client_id yet
        with pytest.raises(service.EgressAuthError, match="not registered yet"):
            svc.build_consent_url("oauth2", "alice", "aud", "sess", "/atlassian", eo)


@pytest.mark.unit
class TestVendRefreshDisconnect:
    async def test_vend_miss_returns_none(self, svc, egress_oauth):
        assert (
            await svc.get_valid_token(
                "oauth2",
                "nobody",
                "/github-mcp",
                egress_oauth,
                requested_upstream=BOUND,
                purpose=keys.EGRESS_PURPOSE,
            )
            is None
        )

    async def test_non_per_user_never_vends(self, svc, egress_oauth, monkeypatch):
        _stub_exchange(monkeypatch)
        # write a token under a network-trusted bucket directly, then confirm
        # get_valid_token refuses it (denylist) even though the entry exists.
        await svc._store.put_token(
            "network-trusted",
            "alice",
            "github",
            "/github-mcp",
            StoredToken(access_token="x"),
            purpose=keys.EGRESS_PURPOSE,
        )
        assert (
            await svc.get_valid_token(
                "network-trusted",
                "alice",
                "/github-mcp",
                egress_oauth,
                requested_upstream=BOUND,
                purpose=keys.EGRESS_PURPOSE,
            )
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
                bound_upstreams=[BOUND],
            ),
            purpose=keys.EGRESS_PURPOSE,
        )
        _stub_exchange(monkeypatch, access_token="at_refreshed")
        token = await svc.get_valid_token(
            "oauth2",
            "alice",
            "/github-mcp",
            egress_oauth,
            requested_upstream=BOUND,
            purpose=keys.EGRESS_PURPOSE,
        )
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
                bound_upstreams=[BOUND],
            ),
            purpose=keys.EGRESS_PURPOSE,
        )

        async def dead_post(cfg, data, headers):
            raise oauth_engine.DeadRefreshTokenError("invalid_grant")

        monkeypatch.setattr(oauth_engine, "_post_token", dead_post)
        assert (
            await svc.get_valid_token(
                "oauth2",
                "alice",
                "/github-mcp",
                egress_oauth,
                requested_upstream=BOUND,
                purpose=keys.EGRESS_PURPOSE,
            )
            is None
        )
        # entry is now refresh_failed -> still a miss (consent needed), no retry storm
        stored = await svc._store.get_token(
            "oauth2", "alice", "github", "/github-mcp", purpose=keys.EGRESS_PURPOSE
        )
        assert stored.status == "refresh_failed"
        assert (
            await svc.get_valid_token(
                "oauth2",
                "alice",
                "/github-mcp",
                egress_oauth,
                requested_upstream=BOUND,
                purpose=keys.EGRESS_PURPOSE,
            )
            is None
        )

    async def test_rotated_client_id_forces_reconsent(self, svc, egress_oauth):
        await svc._store.put_token(
            "oauth2",
            "alice",
            "github",
            "/github-mcp",
            StoredToken(access_token="a", client_id="OLD-client-id", bound_upstreams=[BOUND]),
            purpose=keys.EGRESS_PURPOSE,
        )
        # configured client_id is Iv1.testclient != OLD-client-id -> no vend
        assert (
            await svc.get_valid_token(
                "oauth2",
                "alice",
                "/github-mcp",
                egress_oauth,
                requested_upstream=BOUND,
                purpose=keys.EGRESS_PURPOSE,
            )
            is None
        )

    async def test_list_and_disconnect(self, svc, egress_oauth, monkeypatch):
        _stub_exchange(monkeypatch)
        url = svc.build_consent_url(
            "oauth2", "alice", "Iv1.testclient", "s", "/github-mcp", egress_oauth
        )
        await svc.handle_callback(
            "c", _extract_state(url), egress_oauth, "alice", "oauth2", bound_upstreams=[BOUND]
        )

        conns = await svc.list_connections("oauth2", "alice")
        assert [(c.provider, c.server_path) for c in conns] == [("github", "/github-mcp")]
        # tokens never leak into the connection view
        assert not hasattr(conns[0], "access_token")

        await svc.disconnect(
            "oauth2", "alice", "github", "/github-mcp", purpose=keys.EGRESS_PURPOSE
        )
        assert await svc.list_connections("oauth2", "alice") == []

    @pytest.mark.parametrize("purpose", ["egress", "discovery"])
    async def test_disconnect_only_removes_its_own_purpose(
        self, svc, egress_oauth, monkeypatch, purpose
    ):
        """Revoking one purpose must reach that purpose, and leave the other standing.

        Both halves matter. `disconnect` was pinned to egress, so the discovery space had
        write and read paths but no delete path: turning discovery off left a live
        delegated token for a real person that no UI surfaces -- discovery entries are
        deliberately absent from Connected Accounts, so the user could not revoke it
        either. And the user's own Connected Accounts action must not be able to revoke a
        discovery designation, which is an owner/admin concern.
        """
        _stub_exchange(monkeypatch)
        for session, p in (("sess-1", "egress"), ("sess-2", "discovery")):
            url = svc.build_consent_url(
                "oauth2", "alice", "Iv1.testclient", session, "/github-mcp", egress_oauth, purpose=p
            )
            await svc.handle_callback(
                "c", _extract_state(url), egress_oauth, "alice", "oauth2", bound_upstreams=[BOUND]
            )

        await svc.disconnect("oauth2", "alice", "github", "/github-mcp", purpose=purpose)

        other = "discovery" if purpose == "egress" else "egress"
        assert (
            await svc._store.get_token("oauth2", "alice", "github", "/github-mcp", purpose=purpose)
            is None
        ), "the targeted purpose was not revoked"
        assert (
            await svc._store.get_token("oauth2", "alice", "github", "/github-mcp", purpose=other)
            is not None
        ), "revoking one purpose destroyed the other"

    async def test_list_for_non_per_user_is_empty(self, svc):
        assert await svc.list_connections("network-trusted", "alice") == []


def _extract_state(authorize_url: str) -> str:
    from urllib.parse import parse_qs, urlparse

    return parse_qs(urlparse(authorize_url).query)["state"][0]
