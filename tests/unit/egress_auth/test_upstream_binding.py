"""Credential destination tests: exact registered outbound URLs."""

import pytest

from registry.egress_auth import oauth_engine
from registry.egress_auth.schemas import StoredToken
from registry.egress_auth.service import EgressAuthService
from registry.egress_auth.upstream_binding import (
    registered_destinations,
    selected_upstream,
)
from registry.secrets import keys
from registry.secrets.interfaces import SecretStoreBase
from registry.utils.credential_encryption import encrypt_credential

# Registered upstream and an independently hosted version.
REGISTERED = "https://api.example.com/mcp"
NEW_BASE = "https://new.example.example"
NEW_VERSION_BASE = "https://v2.example.net"


class _InMemoryStore(SecretStoreBase):
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


@pytest.fixture
def svc():
    return EgressAuthService(secret_store=_InMemoryStore(), callback_base_url="https://gw.example")


@pytest.fixture
def egress_oauth():
    return {
        "provider": "github",
        "client_id": "Iv1.testclient",
        "client_secret_encrypted": encrypt_credential("ghs_testsecret"),
        "scopes": ["repo"],
    }


async def _seed(svc, *, bound, client_id="Iv1.testclient", **over):
    """Store a github credential bound to exact approved destinations."""
    token = StoredToken(
        access_token="gho_secret",
        client_id=client_id,
        expires_at="2999-01-01T00:00:00+00:00",
        bound_upstreams=bound,
        **over,
    )
    await svc._store.put_token(
        "oauth2", "alice", "github", "/github", token, purpose=keys.EGRESS_PURPOSE
    )


@pytest.mark.unit
class TestBoundUpstreamsHelper:
    async def test_existing_versions_bind_each_exact_destination(self, monkeypatch):
        active = {
            "proxy_pass_url": REGISTERED,
            "other_version_ids": ["/github:v2"],
        }
        inactive = {"path": "/github:v2", "proxy_pass_url": NEW_VERSION_BASE + "/peer/mcp"}

        class Repo:
            async def get(self, path):
                return inactive if path == "/github:v2" else None

        monkeypatch.setattr("registry.repositories.factory.get_server_repository", lambda: Repo())
        assert await registered_destinations(active, "/github") == sorted(
            {REGISTERED, inactive["proxy_pass_url"]}
        )

    async def test_foreign_or_missing_version_is_never_approved(self, monkeypatch):
        # A version document that does not point back at this server (or no longer
        # exists) is not one of its destinations. It is never approved, and it does
        # not stop the user approving the server's real destinations.
        active = {"proxy_pass_url": REGISTERED, "other_version_ids": ["/github:v2", "/github:v3"]}
        foreign = {
            "path": "/github:v2",
            "proxy_pass_url": NEW_VERSION_BASE,
            "active_version_id": "/other",
        }

        class Repo:
            async def get(self, path):
                return foreign if path == "/github:v2" else None

        monkeypatch.setattr("registry.repositories.factory.get_server_repository", lambda: Repo())
        assert await registered_destinations(active, "/github") == [REGISTERED]

    async def test_active_version_without_endpoint_cannot_be_approved(self):
        with pytest.raises(ValueError):
            await registered_destinations({"other_version_ids": []}, "/github")

    def test_virtual_explicit_endpoint_uses_registered_proxy_host(self):
        server = {
            "proxy_pass_url": "https://backend.example/base",
            "mcp_endpoint": "https://public.example/peer/mcp",
        }
        assert selected_upstream(server, True) == "https://backend.example/peer/mcp"


@pytest.mark.unit
class TestRetargetIsRefused:
    async def test_repointed_proxy_pass_url_refuses_vend(self, svc, egress_oauth):
        # Credential consented against the registered host.
        await _seed(svc, bound=[REGISTERED])
        # Admin repoints proxy_pass_url at a host they control; the live server
        # record and the minted upstream claim now BOTH read the new host,
        # so the route's live registered-set cross-check passes. The stored
        # binding is the only anchor -- and it refuses.
        assert (
            await svc.get_valid_token(
                "oauth2",
                "alice",
                "/github",
                egress_oauth,
                requested_upstream=NEW_BASE,
                purpose=keys.EGRESS_PURPOSE,
            )
            is None
        )

    async def test_registered_upstream_still_vends(self, svc, egress_oauth):
        await _seed(svc, bound=[REGISTERED])
        assert (
            await svc.get_valid_token(
                "oauth2",
                "alice",
                "/github",
                egress_oauth,
                requested_upstream=REGISTERED,
                purpose=keys.EGRESS_PURPOSE,
            )
            == "gho_secret"
        )

    async def test_added_version_needs_reconnect_but_old_route_works(self, svc, egress_oauth):
        # Consent happened before a new version's URL was added, so the binding
        # holds only the old URL. The old route still vends; the new URL is a
        # miss (one reconnect) -- never a silent vend to the just-added host.
        await _seed(svc, bound=[REGISTERED])
        assert (
            await svc.get_valid_token(
                "oauth2",
                "alice",
                "/github",
                egress_oauth,
                requested_upstream=REGISTERED,
                purpose=keys.EGRESS_PURPOSE,
            )
            == "gho_secret"
        )
        assert (
            await svc.get_valid_token(
                "oauth2",
                "alice",
                "/github",
                egress_oauth,
                requested_upstream=NEW_VERSION_BASE + "/mcp",
                purpose=keys.EGRESS_PURPOSE,
            )
            is None
        )

    async def test_legacy_entry_without_binding_is_a_miss(self, svc, egress_oauth):
        # Pre-upgrade credential: empty bound set never matches -> one forced
        # reconnect, regardless of the requested upstream.
        await _seed(svc, bound=[])
        assert (
            await svc.get_valid_token(
                "oauth2",
                "alice",
                "/github",
                egress_oauth,
                requested_upstream=REGISTERED,
                purpose=keys.EGRESS_PURPOSE,
            )
            is None
        )

    async def test_refresh_preserves_binding(self, svc, egress_oauth, monkeypatch):
        # A near-expiry vend refreshes; the fresh token the engine builds is
        # server-agnostic (empty binding), so the service MUST carry the binding
        # across, or the very next vend would wrongly miss.
        await svc._store.put_token(
            "oauth2",
            "alice",
            "github",
            "/github",
            StoredToken(
                access_token="old",
                refresh_token="rt_old",
                expires_at="2000-01-01T00:00:00+00:00",
                client_id="Iv1.testclient",
                bound_upstreams=[REGISTERED],
            ),
            purpose=keys.EGRESS_PURPOSE,
        )

        async def fake_post(cfg, data, headers):
            return {"access_token": "at_refreshed", "expires_in": 3600, "scope": "repo"}

        monkeypatch.setattr(oauth_engine, "_post_token", fake_post)
        assert (
            await svc.get_valid_token(
                "oauth2",
                "alice",
                "/github",
                egress_oauth,
                requested_upstream=REGISTERED,
                purpose=keys.EGRESS_PURPOSE,
            )
            == "at_refreshed"
        )
        stored = await svc._store.get_token(
            "oauth2", "alice", "github", "/github", purpose=keys.EGRESS_PURPOSE
        )
        assert stored.bound_upstreams == [REGISTERED]
        # And the refreshed credential still vends to the bound host next time.
        assert (
            await svc.get_valid_token(
                "oauth2",
                "alice",
                "/github",
                egress_oauth,
                requested_upstream=REGISTERED,
                purpose=keys.EGRESS_PURPOSE,
            )
            == "at_refreshed"
        )

    async def test_pat_retarget_is_a_miss(self, svc):
        from datetime import UTC, datetime, timedelta

        await svc._store.put_token(
            "oauth2",
            "alice",
            "github",
            "/github",
            StoredToken(
                access_token="ghp_x",
                expires_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
                bound_upstreams=[REGISTERED],
            ),
            purpose=keys.EGRESS_PURPOSE,
        )
        # Registered host vends; retargeted host misses.
        assert (
            await svc.get_pat(
                "oauth2",
                "alice",
                "github",
                "/github",
                requested_upstream=REGISTERED,
            )
            == "ghp_x"
        )
        assert (
            await svc.get_pat("oauth2", "alice", "github", "/github", requested_upstream=NEW_BASE)
            is None
        )


class _ReplacingStore(_InMemoryStore):
    """Serves ``first`` on the first read, then ``replacement`` -- a concurrent
    re-consent overwriting the vault entry between the vend's reads."""

    def __init__(self, first: StoredToken, replacement: StoredToken) -> None:
        super().__init__()
        self._reads = 0
        self._first = first
        self._replacement = replacement

    async def get_token(self, auth_method, user_id, provider, server_path, *, purpose):
        self._reads += 1
        return self._first if self._reads == 1 else self._replacement


class _BusyLease:
    async def acquire(self, key, holder, ttl):
        return False

    async def release(self, key, holder):
        return None


@pytest.mark.unit
class TestRefreshRereadIsRechecked:
    """The refresh path re-reads the vault; every re-read must pass the same
    destination binding as the first read, or a credential re-consented for
    another destination is injected at this request's upstream."""

    @staticmethod
    def _tokens():
        stale = StoredToken(
            access_token="old",
            refresh_token="rt_old",
            expires_at="2000-01-01T00:00:00+00:00",
            client_id="Iv1.testclient",
            bound_upstreams=[REGISTERED],
        )
        elsewhere = StoredToken(
            access_token="bound_elsewhere",
            refresh_token="rt_elsewhere",
            expires_at="2999-01-01T00:00:00+00:00",
            client_id="Iv1.testclient",
            bound_upstreams=[NEW_VERSION_BASE + "/mcp"],
        )
        return stale, elsewhere

    async def test_lease_holder_refuses_replaced_entry(self, egress_oauth, monkeypatch):
        stale, elsewhere = self._tokens()
        svc = EgressAuthService(
            secret_store=_ReplacingStore(stale, elsewhere), callback_base_url="https://gw.example"
        )

        async def refresh_must_not_run(cfg, data, headers):
            raise AssertionError("refreshed a credential bound to another destination")

        monkeypatch.setattr(oauth_engine, "_post_token", refresh_must_not_run)
        assert (
            await svc.get_valid_token(
                "oauth2",
                "alice",
                "/github",
                egress_oauth,
                requested_upstream=REGISTERED,
                purpose=keys.EGRESS_PURPOSE,
            )
            is None
        )

    async def test_lease_waiter_refuses_replaced_entry(self, egress_oauth):
        stale, elsewhere = self._tokens()
        svc = EgressAuthService(
            secret_store=_ReplacingStore(stale, elsewhere),
            callback_base_url="https://gw.example",
            lease_manager=_BusyLease(),
        )
        assert (
            await svc.get_valid_token(
                "oauth2",
                "alice",
                "/github",
                egress_oauth,
                requested_upstream=REGISTERED,
                purpose=keys.EGRESS_PURPOSE,
            )
            is None
        )


@pytest.mark.unit
class TestTokenEndpointBinding:
    """A repointed custom_token_url would POST refresh_token + client_secret to a
    new endpoint on the next refresh; the token-endpoint binding refuses first."""

    def _custom_oauth(self, token_url):
        return {
            "provider": "custom",
            "client_id": "dcr-public-client-id",
            "client_secret_encrypted": None,
            "scopes": [],
            "custom_authorize_url": "https://idp.example/authorize",
            "custom_token_url": token_url,
            "custom_token_auth_style": "none",
        }

    async def test_repointed_custom_token_url_refuses_vend(self, svc):
        consented_token_url = "https://idp.example/token"
        await svc._store.put_token(
            "oauth2",
            "alice",
            "custom",
            "/custom",
            StoredToken(
                access_token="at",
                client_id="dcr-public-client-id",
                expires_at="2999-01-01T00:00:00+00:00",
                bound_upstreams=[REGISTERED],
                bound_token_url=consented_token_url,
            ),
            purpose=keys.EGRESS_PURPOSE,
        )
        # Upstream binding satisfied; only the token endpoint moved.
        repointed = self._custom_oauth("https://new.example/token")
        assert (
            await svc.get_valid_token(
                "oauth2",
                "alice",
                "/custom",
                repointed,
                requested_upstream=REGISTERED,
                purpose=keys.EGRESS_PURPOSE,
            )
            is None
        )

    async def test_unchanged_custom_token_url_still_vends(self, svc):
        token_url = "https://idp.example/token"
        await svc._store.put_token(
            "oauth2",
            "alice",
            "custom",
            "/custom",
            StoredToken(
                access_token="at",
                client_id="dcr-public-client-id",
                expires_at="2999-01-01T00:00:00+00:00",
                bound_upstreams=[REGISTERED],
                bound_token_url=token_url,
            ),
            purpose=keys.EGRESS_PURPOSE,
        )
        assert (
            await svc.get_valid_token(
                "oauth2",
                "alice",
                "/custom",
                self._custom_oauth(token_url),
                requested_upstream=REGISTERED,
                purpose=keys.EGRESS_PURPOSE,
            )
            == "at"
        )
