"""Backend OAuth (client_credentials) token cache tests.

Covers the registry-self backend-auth resolver: caching per server path,
config-fingerprint invalidation, single-flight acquisition, and fail-open-to-None
behavior on misconfig / token-endpoint failure.
"""

import asyncio
import pathlib
import re
from datetime import UTC, datetime

import pytest

from registry.core import backend_oauth
from registry.egress_auth import oauth_engine
from registry.egress_auth.schemas import StoredToken


def _server_info(**overrides) -> dict:
    bo = {
        "token_url": "https://idp.example/token",
        "client_id": "cid",
        "client_secret_encrypted": "enc-secret",
        "scopes": ["api:read"],
        "token_auth_style": "post_body",
    }
    bo.update(overrides.pop("backend_oauth", {}))
    si = {"auth_scheme": "oauth", "service_path": "/example", "backend_oauth": bo}
    si.update(overrides)
    return si


@pytest.fixture(autouse=True)
def _clear_cache():
    backend_oauth._cache.clear()
    backend_oauth._locks.clear()
    yield
    backend_oauth._cache.clear()
    backend_oauth._locks.clear()


@pytest.fixture(autouse=True)
def _stub_decrypt(monkeypatch):
    # The resolver decrypts client_secret_encrypted; stub it to a known plaintext.
    monkeypatch.setattr(backend_oauth, "decrypt_credential", lambda c: "plain-secret")


def _token(access="tok", expires_at=None) -> StoredToken:
    return StoredToken(
        access_token=access,
        token_type="Bearer",
        expires_at=expires_at,
        scopes=[],
        status="active",
        client_id="cid",
    )


@pytest.mark.unit
class TestFreshness:
    """The expiry branch of _is_fresh, which no other test reaches.

    Every other test leaves `expires_at=None`, so the whole
    `expires_at_epoch is not None` arm -- the skew cap, the floor, and every
    expiry-driven re-acquire -- was previously unexecuted.
    """

    def _entry(self, *, lifetime: float, age: float) -> backend_oauth._CacheEntry:
        now = datetime.now(UTC).timestamp()
        acquired = now - age
        return backend_oauth._CacheEntry(
            access_token="t",
            expires_at_epoch=acquired + lifetime,
            fingerprint="fp",
            acquired_epoch=acquired,
        )

    def test_long_lived_token_uses_the_configured_skew(self, monkeypatch):
        monkeypatch.setattr(
            backend_oauth.settings, "egress_token_refresh_skew_seconds", 300, raising=False
        )
        # 3600s token, 100s old -> 3500s left, far beyond the 300s skew.
        assert backend_oauth._is_fresh(self._entry(lifetime=3600, age=100)) is True
        # 3600s token, 3400s old -> 200s left, inside the 300s skew.
        assert backend_oauth._is_fresh(self._entry(lifetime=3600, age=3400)) is False

    def test_short_lived_token_is_still_cacheable(self, monkeypatch):
        """A 200s token must not be permanently un-fresh under a 300s skew.

        Applying the skew verbatim would make every such token stale the instant it
        was stored -- one token-endpoint POST per server per health cycle, forever.
        The cap at half-lifetime keeps it usable for ~100s.
        """
        monkeypatch.setattr(
            backend_oauth.settings, "egress_token_refresh_skew_seconds", 300, raising=False
        )
        assert backend_oauth._is_fresh(self._entry(lifetime=200, age=10)) is True
        assert backend_oauth._is_fresh(self._entry(lifetime=200, age=150)) is False

    def test_very_short_token_keeps_a_floor_so_it_cannot_expire_in_flight(self, monkeypatch):
        """The cap must not let the margin collapse toward zero.

        With only the half-lifetime cap, a 2s token reads as fresh with 1s left and
        can expire between this check and the upstream reading the header.
        """
        monkeypatch.setattr(
            backend_oauth.settings, "egress_token_refresh_skew_seconds", 300, raising=False
        )
        assert backend_oauth._is_fresh(self._entry(lifetime=2, age=0)) is False

    def test_no_expiry_falls_back_to_the_short_default_ttl(self):
        now = datetime.now(UTC).timestamp()
        fresh = backend_oauth._CacheEntry(
            access_token="t", expires_at_epoch=None, fingerprint="fp", acquired_epoch=now - 5
        )
        stale = backend_oauth._CacheEntry(
            access_token="t",
            expires_at_epoch=None,
            fingerprint="fp",
            acquired_epoch=now - (backend_oauth._DEFAULT_TTL_SECONDS + 1),
        )
        assert backend_oauth._is_fresh(fresh) is True
        assert backend_oauth._is_fresh(stale) is False


@pytest.mark.unit
class TestSingleFlight:
    async def test_concurrent_resolves_mint_once(self, monkeypatch):
        """Ten simultaneous callers must produce ONE token-endpoint request.

        The whole point of the per-key lock plus the double-check inside it. Without
        both, a health cycle that fans out across servers stampedes the IdP. A
        sequential cache-hit test cannot detect that.
        """
        calls = 0
        started = asyncio.Event()

        async def fake_grant(*a, **k):
            nonlocal calls
            calls += 1
            started.set()
            # Yield so every waiter is parked on the lock before this one resolves.
            await asyncio.sleep(0.01)
            return _token(access="CC")

        monkeypatch.setattr(oauth_engine, "client_credentials_token", fake_grant)
        results = await asyncio.gather(
            *(backend_oauth.resolve_bearer(_server_info()) for _ in range(10))
        )
        assert results == ["CC"] * 10
        assert calls == 1


@pytest.mark.unit
class TestDestinationBinding:
    """Repointing proxy_pass_url must force a re-mint, not reuse the cached token.

    The per-user vault binds its entries to the upstreams registered at consent time
    and refuses to vend elsewhere. Tiers 1 and 3 cache in-process keyed on the server
    PATH, so without the upstream in the fingerprint the path stays the same, the
    fingerprint matches, and the next health cycle ships a token minted for the old
    host to the newly registered one. The SSRF guard only excludes private and
    metadata addresses, so any public host passes it.
    """

    async def test_tier1_remints_when_the_upstream_is_repointed(self, monkeypatch):
        calls = []

        async def fake_grant(*a, **k):
            calls.append(1)
            return _token(access=f"CC{len(calls)}")

        monkeypatch.setattr(oauth_engine, "client_credentials_token", fake_grant)
        si = _server_info(proxy_pass_url="https://original.example.com/mcp")
        assert await backend_oauth.resolve_bearer(si) == "CC1"
        # Same path, same OAuth config -- only the destination moved.
        moved = _server_info(proxy_pass_url="https://attacker-registered.example.com/mcp")
        assert await backend_oauth.resolve_bearer(moved) == "CC2"
        assert len(calls) == 2

    async def test_tier3_remints_when_the_upstream_is_repointed(self, monkeypatch, _entra_gateway):
        """Matters more here: the token is the gateway's OWN app-only credential for
        target_audience, which the server owner never held."""
        calls = []

        async def fake_grant(*a, **k):
            calls.append(1)
            return _token(access=f"OBO{len(calls)}")

        monkeypatch.setattr(oauth_engine, "client_credentials_token", fake_grant)
        si = _obo_server_info(proxy_pass_url="https://original.example.com/mcp")
        assert await backend_oauth.resolve_obo_discovery_bearer(si) == "OBO1"
        moved = _obo_server_info(proxy_pass_url="https://attacker-registered.example.com/mcp")
        assert await backend_oauth.resolve_obo_discovery_bearer(moved) == "OBO2"
        assert len(calls) == 2


@pytest.mark.unit
class TestResolveBearer:
    async def test_non_oauth_scheme_returns_none_without_calling_engine(self, monkeypatch):
        called = False

        async def fake_grant(*a, **k):
            nonlocal called
            called = True
            return _token()

        monkeypatch.setattr(oauth_engine, "client_credentials_token", fake_grant)
        assert await backend_oauth.resolve_bearer({"auth_scheme": "bearer"}) is None
        assert called is False

    async def test_works_with_the_egress_feature_disabled(self, monkeypatch):
        """Tier 1 must NOT be gated on EGRESS_AUTH_ENABLED.

        The gate belongs only where the vault does. This tier reads the per-server
        `backend_oauth` config and posts to that server's own token endpoint -- no
        vault, no user, no consent -- so it has to keep working with the egress
        feature off, including in registry-only deployments. Gating it would silently
        break OAuth 2.0 backend discovery for everyone who never enabled egress.
        """
        monkeypatch.setattr(backend_oauth.settings, "egress_auth_enabled", False, raising=False)

        async def fake_grant(*a, **k):
            return _token(access="CC")

        monkeypatch.setattr(oauth_engine, "client_credentials_token", fake_grant)
        assert await backend_oauth.resolve_bearer(_server_info()) == "CC"

    async def test_missing_config_returns_none(self, monkeypatch):
        monkeypatch.setattr(oauth_engine, "client_credentials_token", lambda *a, **k: _token())
        si = _server_info(backend_oauth={"token_url": "", "client_id": ""})
        assert await backend_oauth.resolve_bearer(si) is None

    async def test_acquires_and_returns_token(self, monkeypatch):
        async def fake_grant(cfg, client_id, secret, scopes):
            assert client_id == "cid"
            assert secret == "plain-secret"
            assert scopes == ["api:read"]
            return _token(access="fresh")

        monkeypatch.setattr(oauth_engine, "client_credentials_token", fake_grant)
        assert await backend_oauth.resolve_bearer(_server_info()) == "fresh"

    async def test_second_call_uses_cache(self, monkeypatch):
        calls = {"n": 0}

        async def fake_grant(*a, **k):
            calls["n"] += 1
            # No expiry hint -> default short TTL, but well within it for two calls.
            return _token(access=f"t{calls['n']}")

        monkeypatch.setattr(oauth_engine, "client_credentials_token", fake_grant)
        first = await backend_oauth.resolve_bearer(_server_info())
        second = await backend_oauth.resolve_bearer(_server_info())
        assert first == second == "t1"
        assert calls["n"] == 1

    async def test_config_change_invalidates_cache(self, monkeypatch):
        calls = {"n": 0}

        async def fake_grant(*a, **k):
            calls["n"] += 1
            return _token(access=f"t{calls['n']}")

        monkeypatch.setattr(oauth_engine, "client_credentials_token", fake_grant)
        await backend_oauth.resolve_bearer(_server_info())
        # Different client_id -> different fingerprint -> re-acquire.
        changed = await backend_oauth.resolve_bearer(
            _server_info(backend_oauth={"client_id": "cid2"})
        )
        assert changed == "t2"
        assert calls["n"] == 2

    async def test_engine_failure_returns_none(self, monkeypatch):
        async def fake_grant(*a, **k):
            raise oauth_engine.OAuthEngineError("token endpoint unreachable")

        monkeypatch.setattr(oauth_engine, "client_credentials_token", fake_grant)
        assert await backend_oauth.resolve_bearer(_server_info()) is None

    async def test_invalidate_forces_reacquire(self, monkeypatch):
        calls = {"n": 0}

        async def fake_grant(*a, **k):
            calls["n"] += 1
            return _token(access=f"t{calls['n']}")

        monkeypatch.setattr(oauth_engine, "client_credentials_token", fake_grant)
        await backend_oauth.resolve_bearer(_server_info())
        backend_oauth.invalidate("/example")
        await backend_oauth.resolve_bearer(_server_info())
        assert calls["n"] == 2


@pytest.mark.unit
class TestWithBearer:
    async def test_stashes_token_under_key(self, monkeypatch):
        monkeypatch.setattr(
            oauth_engine, "client_credentials_token", lambda *a, **k: _token(access="X")
        )

        async def fake_grant(*a, **k):
            return _token(access="X")

        monkeypatch.setattr(oauth_engine, "client_credentials_token", fake_grant)
        si = await backend_oauth.with_bearer(_server_info())
        assert si[backend_oauth.RESOLVED_BEARER_KEY] == "X"

    async def test_noop_for_non_oauth(self):
        si = {"auth_scheme": "bearer"}
        assert await backend_oauth.with_bearer(si) is si

    async def test_client_credentials_failure_does_not_fall_through(self, monkeypatch):
        """A tier-1 failure must yield NO credential, not a borrowed human token.

        auth_scheme == 'oauth' means the operator chose a machine grant. Falling
        through to the vault borrow on a transient token-endpoint error silently
        substitutes a different principal at a different privilege class, which is
        harder to notice than an outright discovery failure.
        """

        async def boom(*a, **k):
            raise oauth_engine.OAuthEngineError("token endpoint 500")

        monkeypatch.setattr(oauth_engine, "client_credentials_token", boom)
        borrowed = False

        async def fake_borrow(server_info):
            nonlocal borrowed
            borrowed = True
            return "BORROWED"

        monkeypatch.setattr(backend_oauth, "resolve_discovery_bearer", fake_borrow)
        si = _server_info()
        si["oauth_discovery"] = {"enabled": True, "oauth": {"provider": "github"}}
        out = await backend_oauth.with_bearer(si)
        assert backend_oauth.RESOLVED_BEARER_KEY not in out
        assert borrowed is False


def _disc_server_info(**overrides) -> dict:
    si = {
        "auth_scheme": "none",
        "service_path": "/tableau-hosted",
        "path": "/tableau-hosted",
        # Registered upstream: the discovery borrow binds the vaulted token to
        # this server's own endpoint base (destination binding).
        "proxy_pass_url": "https://tableau.example.com/mcp",
        # Self-contained backend-auth discovery config (no top-level egress_oauth).
        "oauth_discovery": {
            "enabled": True,
            "oauth": {"provider": "custom", "client_id": "https://x/.well-known/oauth-client"},
            "auth_method": "oauth2",
            "user_id": "u-1",
        },
    }
    si.update(overrides)
    return si


class _FakeEgressSvc:
    def __init__(self, token=None, raises=False):
        self._token = token
        self._raises = raises
        self.calls = []

    async def get_valid_token(
        self,
        *,
        auth_method,
        user_id,
        server_path,
        egress_oauth,
        requested_upstream,
        purpose,
    ):
        # `purpose` is recorded so a test can assert the borrow reads the DISCOVERY
        # address space. It selects which space is read, so asking for the wrong one
        # cannot return the user's own runtime credential -- it finds nothing.
        self.calls.append((auth_method, user_id, server_path, requested_upstream, purpose))
        if self._raises:
            raise RuntimeError("vault down")
        return self._token


@pytest.mark.unit
class TestDiscoveryBorrow:
    @pytest.fixture(autouse=True)
    def _egress_on(self, monkeypatch):
        """Discovery REQUIRES the egress feature: it borrows from the per-user vault
        and the consent that fills that vault is served by the egress facade."""
        monkeypatch.setattr(backend_oauth.settings, "egress_auth_enabled", True, raising=False)

    def _patch_svc(self, monkeypatch, svc):
        import registry.egress_auth.factory as factory

        monkeypatch.setattr(factory, "get_egress_auth_service", lambda: svc)

    async def test_egress_disabled_returns_none_without_calling(self, monkeypatch):
        """The hard requirement, enforced at the resolver.

        Without EGRESS_AUTH_ENABLED there is no vend path to borrow through and no
        mounted consent route to populate the vault, so the borrow must not even be
        attempted -- and the resolver must not depend on the router for that.
        """
        monkeypatch.setattr(backend_oauth.settings, "egress_auth_enabled", False, raising=False)
        svc = _FakeEgressSvc(token="BORROWED")
        self._patch_svc(monkeypatch, svc)
        assert await backend_oauth.resolve_discovery_bearer(_disc_server_info()) is None
        assert svc.calls == []

    async def test_borrows_designated_principal_token(self, monkeypatch):
        svc = _FakeEgressSvc(token="BORROWED")
        self._patch_svc(monkeypatch, svc)
        assert await backend_oauth.resolve_discovery_bearer(_disc_server_info()) == "BORROWED"
        assert svc.calls == [
            ("oauth2", "u-1", "/tableau-hosted", "https://tableau.example.com", "discovery")
        ]

    @pytest.mark.parametrize("scheme", ["bearer", "api_key"])
    async def test_explicit_static_scheme_bows_out(self, monkeypatch, scheme):
        """A stored scan token is the operator's chosen discovery credential.

        The borrow must not fire when auth_scheme is set, because the resolved
        bearer takes precedence in the sync header builders -- so shadowing here
        silently DROPS the static credential rather than merely competing with it.
        Same invariant the obo tier already enforced.
        """
        svc = _FakeEgressSvc(token="BORROWED")
        self._patch_svc(monkeypatch, svc)
        si = _disc_server_info(auth_scheme=scheme)
        assert await backend_oauth.resolve_discovery_bearer(si) is None
        assert svc.calls == []

    async def test_with_bearer_does_not_shadow_static_scheme(self, monkeypatch):
        """End-to-end: with_bearer stashes nothing, so the builder uses the static one."""
        svc = _FakeEgressSvc(token="BORROWED")
        self._patch_svc(monkeypatch, svc)
        out = await backend_oauth.with_bearer(_disc_server_info(auth_scheme="bearer"))
        assert backend_oauth.RESOLVED_BEARER_KEY not in out

    async def test_disabled_returns_none_without_calling(self, monkeypatch):
        svc = _FakeEgressSvc(token="X")
        self._patch_svc(monkeypatch, svc)
        si = _disc_server_info(
            oauth_discovery={"enabled": False, "auth_method": "oauth2", "user_id": "u"}
        )
        assert await backend_oauth.resolve_discovery_bearer(si) is None
        assert svc.calls == []

    async def test_missing_oauth_config_returns_none(self, monkeypatch):
        svc = _FakeEgressSvc(token="X")
        self._patch_svc(monkeypatch, svc)
        # oauth_discovery present + enabled but without its own oauth provider config.
        si = _disc_server_info(
            oauth_discovery={"enabled": True, "auth_method": "oauth2", "user_id": "u-1"}
        )
        assert await backend_oauth.resolve_discovery_bearer(si) is None
        assert svc.calls == []

    async def test_missing_principal_returns_none(self, monkeypatch):
        svc = _FakeEgressSvc(token="X")
        self._patch_svc(monkeypatch, svc)
        si = _disc_server_info(
            oauth_discovery={
                "enabled": True,
                "oauth": {"provider": "custom", "client_id": "x"},
                "auth_method": "",
                "user_id": "",
            }
        )
        assert await backend_oauth.resolve_discovery_bearer(si) is None
        assert svc.calls == []

    async def test_vault_miss_returns_none(self, monkeypatch):
        # Identity not connected / refresh dead -> get_valid_token returns None.
        self._patch_svc(monkeypatch, _FakeEgressSvc(token=None))
        assert await backend_oauth.resolve_discovery_bearer(_disc_server_info()) is None

    async def test_vault_exception_degrades_to_none(self, monkeypatch):
        self._patch_svc(monkeypatch, _FakeEgressSvc(raises=True))
        assert await backend_oauth.resolve_discovery_bearer(_disc_server_info()) is None

    async def test_with_bearer_stashes_borrowed_token(self, monkeypatch):
        self._patch_svc(monkeypatch, _FakeEgressSvc(token="BORROWED"))
        si = await backend_oauth.with_bearer(_disc_server_info())
        assert si[backend_oauth.RESOLVED_BEARER_KEY] == "BORROWED"

    async def test_client_credentials_precedence_over_discovery(self, monkeypatch):
        # auth_scheme 'oauth' resolves via client_credentials; discovery not consulted.
        disc_svc = _FakeEgressSvc(token="BORROWED")
        self._patch_svc(monkeypatch, disc_svc)

        async def fake_cc(server_info):
            return "CC-TOKEN"

        monkeypatch.setattr(backend_oauth, "resolve_bearer", fake_cc)
        si = _disc_server_info(auth_scheme="oauth")
        out = await backend_oauth.with_bearer(si)
        assert out[backend_oauth.RESOLVED_BEARER_KEY] == "CC-TOKEN"
        assert disc_svc.calls == []


def _obo_server_info(**overrides) -> dict:
    si = {
        "auth_scheme": "none",
        "service_path": "/obo-echo",
        "path": "/obo-echo",
        "egress_auth_mode": "obo_exchange",
        "egress_oauth": {"target_audience": "api://internal-mcp", "scopes": []},
    }
    si.update(overrides)
    return si


@pytest.fixture
def _entra_gateway(monkeypatch):
    """Configure the gateway's own Entra IdP client + egress feature on."""
    for attr, val in (
        ("egress_auth_enabled", True),
        ("auth_provider", "entra"),
        ("entra_client_id", "gw-client"),
        ("entra_client_secret", "gw-secret"),
        ("entra_tenant_id", "tenant-1"),
        ("entra_login_base_url", "https://login.microsoftonline.com"),
        # No operator allowlist -> shape heuristic accepts api:// targets.
        ("egress_obo_allowed_audiences", ""),
    ):
        monkeypatch.setattr(backend_oauth.settings, attr, val, raising=False)


@pytest.mark.unit
class TestResolveOboDiscoveryBearer:
    async def test_mints_cc_token_audienced_to_target(self, monkeypatch, _entra_gateway):
        captured = {}

        async def fake_grant(cfg, client_id, secret, scopes):
            captured["token_url"] = cfg.token_url
            captured["client_id"] = client_id
            captured["secret"] = secret
            captured["scopes"] = scopes
            return _token(access="OBO-DISC")

        monkeypatch.setattr(oauth_engine, "client_credentials_token", fake_grant)
        assert await backend_oauth.resolve_obo_discovery_bearer(_obo_server_info()) == "OBO-DISC"
        assert captured["client_id"] == "gw-client"
        assert captured["secret"] == "gw-secret"
        assert captured["scopes"] == ["api://internal-mcp/.default"]
        assert (
            captured["token_url"] == "https://login.microsoftonline.com/tenant-1/oauth2/v2.0/token"
        )

    @pytest.mark.parametrize(
        "configured,expected_host",
        [
            ("https://login.microsoftonline.us", "https://login.microsoftonline.us"),
            (
                "https://login.partner.microsoftonline.cn",
                "https://login.partner.microsoftonline.cn",
            ),
            # Trailing slash must not double up in the token URL.
            ("https://login.microsoftonline.us/", "https://login.microsoftonline.us"),
            # An operator who blanks the override falls back to commercial, not "".
            ("", "https://login.microsoftonline.com"),
        ],
    )
    async def test_login_base_url_selects_the_token_endpoint(
        self, monkeypatch, _entra_gateway, configured, expected_host
    ):
        monkeypatch.setattr(
            backend_oauth.settings, "entra_login_base_url", configured, raising=False
        )
        captured = {}

        async def fake_grant(cfg, client_id, secret, scopes):
            captured["token_url"] = cfg.token_url
            return _token(access="OBO-DISC")

        monkeypatch.setattr(oauth_engine, "client_credentials_token", fake_grant)
        assert await backend_oauth.resolve_obo_discovery_bearer(_obo_server_info()) == "OBO-DISC"
        assert captured["token_url"] == f"{expected_host}/tenant-1/oauth2/v2.0/token"

    async def test_egress_disabled_returns_none(self, monkeypatch, _entra_gateway):
        monkeypatch.setattr(backend_oauth.settings, "egress_auth_enabled", False, raising=False)
        called = False

        async def fake_grant(*a, **k):
            nonlocal called
            called = True
            return _token()

        monkeypatch.setattr(oauth_engine, "client_credentials_token", fake_grant)
        assert await backend_oauth.resolve_obo_discovery_bearer(_obo_server_info()) is None
        assert called is False

    async def test_non_obo_mode_returns_none(self, monkeypatch, _entra_gateway):
        called = False

        async def fake_grant(*a, **k):
            nonlocal called
            called = True
            return _token()

        monkeypatch.setattr(oauth_engine, "client_credentials_token", fake_grant)
        si = _obo_server_info(egress_auth_mode="oauth_user")
        assert await backend_oauth.resolve_obo_discovery_bearer(si) is None
        assert called is False

    async def test_missing_target_returns_none(self, monkeypatch, _entra_gateway):
        monkeypatch.setattr(oauth_engine, "client_credentials_token", lambda *a, **k: _token())
        si = _obo_server_info(egress_oauth={"target_audience": "", "scopes": []})
        assert await backend_oauth.resolve_obo_discovery_bearer(si) is None

    async def test_disallowed_first_party_target_returns_none(self, monkeypatch, _entra_gateway):
        called = False

        async def fake_grant(*a, **k):
            nonlocal called
            called = True
            return _token()

        monkeypatch.setattr(oauth_engine, "client_credentials_token", fake_grant)
        # A shared first-party resource URL is rejected by the audience control.
        si = _obo_server_info(
            egress_oauth={"target_audience": "https://graph.microsoft.com", "scopes": []}
        )
        assert await backend_oauth.resolve_obo_discovery_bearer(si) is None
        assert called is False

    async def test_gateway_own_audience_refused(self, monkeypatch, _entra_gateway):
        """Refuse minting a token audienced to the gateway's OWN app -- reflection.

        This is the clause that matters most for client_credentials specifically.
        A same-app OBO *exchange* is rejected by Entra at runtime, so the runtime
        grant has an IdP-side backstop. `api://<our-own-client-id>/.default` has
        none: Entra issues it happily. That token would then be sent upstream as
        `Authorization: Bearer` to a third-party MCP server, in the exact form this
        gateway's own ingress accepts -- a confused deputy.

        Regression guard: tier 3 previously re-ran only _is_disallowed_obo_audience
        (the first-party floor), which does NOT cover the gateway's own URI. It now
        runs the full _validate_obo_egress_config the registration path uses.
        """
        called = False

        async def fake_grant(*a, **k):
            nonlocal called
            called = True
            return _token()

        monkeypatch.setattr(oauth_engine, "client_credentials_token", fake_grant)
        # _entra_gateway sets entra_client_id == "gw-client".
        si = _obo_server_info(egress_oauth={"target_audience": "api://gw-client", "scopes": []})
        assert await backend_oauth.resolve_obo_discovery_bearer(si) is None
        assert called is False

    async def test_gateway_client_unconfigured_returns_none(self, monkeypatch, _entra_gateway):
        monkeypatch.setattr(backend_oauth.settings, "entra_client_secret", "", raising=False)
        called = False

        async def fake_grant(*a, **k):
            nonlocal called
            called = True
            return _token()

        monkeypatch.setattr(oauth_engine, "client_credentials_token", fake_grant)
        assert await backend_oauth.resolve_obo_discovery_bearer(_obo_server_info()) is None
        assert called is False

    async def test_second_call_uses_cache(self, monkeypatch, _entra_gateway):
        calls = {"n": 0}

        async def fake_grant(*a, **k):
            calls["n"] += 1
            return _token(access=f"t{calls['n']}")

        monkeypatch.setattr(oauth_engine, "client_credentials_token", fake_grant)
        first = await backend_oauth.resolve_obo_discovery_bearer(_obo_server_info())
        second = await backend_oauth.resolve_obo_discovery_bearer(_obo_server_info())
        assert first == second == "t1"
        assert calls["n"] == 1

    async def test_target_change_reacquires(self, monkeypatch, _entra_gateway):
        calls = {"n": 0}

        async def fake_grant(*a, **k):
            calls["n"] += 1
            return _token(access=f"t{calls['n']}")

        monkeypatch.setattr(oauth_engine, "client_credentials_token", fake_grant)
        await backend_oauth.resolve_obo_discovery_bearer(_obo_server_info())
        changed = await backend_oauth.resolve_obo_discovery_bearer(
            _obo_server_info(egress_oauth={"target_audience": "api://other-mcp", "scopes": []})
        )
        assert changed == "t2"
        assert calls["n"] == 2

    async def test_engine_failure_returns_none(self, monkeypatch, _entra_gateway):
        async def fake_grant(*a, **k):
            raise oauth_engine.OAuthEngineError("token endpoint unreachable")

        monkeypatch.setattr(oauth_engine, "client_credentials_token", fake_grant)
        assert await backend_oauth.resolve_obo_discovery_bearer(_obo_server_info()) is None

    async def test_keycloak_sends_no_default_scope(self, monkeypatch, _entra_gateway):
        # Keycloak binds audience via a server-side mapper (follow-on): no .default
        # scope is sent, and the token endpoint is the realm token URL.
        for attr, val in (
            ("auth_provider", "keycloak"),
            ("keycloak_url", "https://kc.example.com"),
            ("keycloak_realm", "mcp-gateway"),
            ("keycloak_client_id", "kc-client"),
            ("keycloak_client_secret", "kc-secret"),
        ):
            monkeypatch.setattr(backend_oauth.settings, attr, val, raising=False)
        captured = {}

        async def fake_grant(cfg, client_id, secret, scopes):
            captured["token_url"] = cfg.token_url
            captured["scopes"] = scopes
            return _token(access="KC")

        monkeypatch.setattr(oauth_engine, "client_credentials_token", fake_grant)
        assert await backend_oauth.resolve_obo_discovery_bearer(_obo_server_info()) == "KC"
        assert captured["scopes"] == []
        assert (
            captured["token_url"]
            == "https://kc.example.com/realms/mcp-gateway/protocol/openid-connect/token"
        )

    async def test_with_bearer_uses_obo_discovery_for_obo_server(self, monkeypatch, _entra_gateway):
        # Pure obo server: no auth_scheme=oauth, no oauth_discovery -> falls through
        # to the machine client_credentials path.
        async def fake_grant(*a, **k):
            return _token(access="OBO-DISC")

        monkeypatch.setattr(oauth_engine, "client_credentials_token", fake_grant)
        si = await backend_oauth.with_bearer(_obo_server_info())
        assert si[backend_oauth.RESOLVED_BEARER_KEY] == "OBO-DISC"

    async def test_discovery_borrow_precedence_over_obo(self, monkeypatch, _entra_gateway):
        # A server configured with BOTH a connected discovery identity and
        # obo_exchange resolves via the borrowed vault token; the machine CC path
        # is not consulted.
        import registry.egress_auth.factory as factory

        monkeypatch.setattr(
            factory, "get_egress_auth_service", lambda: _FakeEgressSvc(token="BORROWED")
        )

        async def fail_grant(*a, **k):
            raise AssertionError("obo discovery CC must not run when the borrow succeeds")

        monkeypatch.setattr(oauth_engine, "client_credentials_token", fail_grant)
        si = _obo_server_info(
            proxy_pass_url="https://internal.example.com/mcp",
            oauth_discovery={
                "enabled": True,
                "oauth": {"provider": "custom", "client_id": "x"},
                "auth_method": "oauth2",
                "user_id": "u-1",
            },
        )
        out = await backend_oauth.with_bearer(si)
        assert out[backend_oauth.RESOLVED_BEARER_KEY] == "BORROWED"

    async def test_explicit_auth_scheme_bows_out(self, monkeypatch, _entra_gateway):
        # An operator-configured backend discovery credential (e.g. a bearer scan
        # token) must win; the derived machine token is only a fallback.
        called = False

        async def fake_grant(*a, **k):
            nonlocal called
            called = True
            return _token()

        monkeypatch.setattr(oauth_engine, "client_credentials_token", fake_grant)
        si = _obo_server_info(auth_scheme="bearer")
        assert await backend_oauth.resolve_obo_discovery_bearer(si) is None
        assert called is False

    async def test_with_bearer_does_not_shadow_static_scheme(self, monkeypatch, _entra_gateway):
        # obo server + a static bearer: with_bearer must NOT stash a resolved
        # bearer, so the sync header builder uses the operator's static credential.
        async def fail_grant(*a, **k):
            raise AssertionError("obo discovery CC must not run when a scheme is set")

        monkeypatch.setattr(oauth_engine, "client_credentials_token", fail_grant)
        si = _obo_server_info(auth_scheme="bearer", auth_credential_encrypted="enc")
        out = await backend_oauth.with_bearer(si)
        assert backend_oauth.RESOLVED_BEARER_KEY not in out


@pytest.mark.unit
class TestDiscoveryCredentialStaysOutOfRuntime:
    """The borrowed identity is for DISCOVERY only: list tools, health, scan -- never execute.

    Using the designated user's vaulted credential to enumerate a server's tools is the
    intended behaviour; that is the whole point of the borrow. Using it to EXECUTE a tool
    on behalf of some other caller is not, and would turn a one-time consent into a
    standing proxy for that person's access.

    Today that holds because tool execution never enters this Python app at all -- nginx
    proxies `tools/call` straight to the upstream, forwarding the caller's own
    Authorization header. Nothing enforces it, though, so a future execute path added
    here could reach the resolver by reusing a discovery helper. These tests pin the
    boundary at the only two places it can be observed in source.
    """

    RESOLVER_NAMES = ("with_bearer", "RESOLVED_BEARER_KEY", "_backend_oauth_bearer")

    def _repo_root(self) -> pathlib.Path:
        return pathlib.Path(__file__).resolve().parents[3]

    def test_runtime_data_plane_never_references_the_resolver(self):
        """The auth server and the generated nginx config are the runtime path.

        Neither may mention the resolver or the key it stashes. A hit here means a
        resolved discovery credential became reachable from end-user traffic.
        """
        root = self._repo_root()
        targets = [*(root / "auth_server").rglob("*.py"), root / "registry/core/nginx_service.py"]
        offenders = []
        for path in targets:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for name in self.RESOLVER_NAMES:
                if name in text:
                    offenders.append(f"{path.relative_to(root)}: {name}")
        assert not offenders, (
            "the runtime data plane must not reach the discovery resolver; found: "
            + "; ".join(offenders)
        )

    def test_resolver_consumers_are_only_discovery_call_sites(self):
        """`with_bearer` may only be called from health, tool discovery, and the scanner.

        All three are the registry acting as itself. If a new caller appears, it must be
        reviewed against this boundary rather than inheriting the credential silently.
        """
        root = self._repo_root()
        allowed = {
            "registry/core/backend_oauth.py",  # defines it
            "registry/core/mcp_client.py",  # tool discovery + connection check
            "registry/health/service.py",  # health checks
            "registry/api/server_routes.py",  # _build_scan_auth_headers
        }
        callers = set()
        for path in (root / "registry").rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if re.search(r"(?<!def )with_bearer\s*\(", text) or "RESOLVED_BEARER_KEY" in text:
                callers.add(str(path.relative_to(root)))
        assert callers <= allowed, "unreviewed consumer of the discovery credential: " + ", ".join(
            sorted(callers - allowed)
        )

    def test_tool_execution_has_no_python_implementation_to_leak_into(self):
        """Execute is proxied by nginx, so there is no server-side call path to audit.

        If this fails, someone added tool invocation to the app and it MUST be checked
        against the boundary above -- the discovery credential must not be attached to it.
        """
        root = self._repo_root()
        hits = []
        for sub in ("registry", "auth_server"):
            for path in (root / sub).rglob("*.py"):
                text = path.read_text(encoding="utf-8", errors="ignore")
                if re.search(r"\b(async )?def (call_tool|invoke_tool|execute_tool)\b", text):
                    hits.append(str(path.relative_to(root)))
        assert not hits, "tool execution implemented in: " + ", ".join(hits)
