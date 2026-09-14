"""Destination-binding tests for the egress vend path.

A vaulted credential is bound at write time to the destinations registered for the
server -- its upstream set, and for a custom provider its OAuth token endpoint. The
vend requires the request's destination to be a member of that bound set, so the
credential is only ever released to a destination the user consented to: if
``proxy_pass_url`` (or ``custom_token_url``) changes, the destination is no longer in
the bound set and the vend fail-closes to re-consent instead of releasing the
credential to it.

See registry/egress_auth/upstream_binding.py and EgressAuthService.get_valid_token
/ get_pat.
"""

import pytest

from registry.egress_auth import oauth_engine
from registry.egress_auth.schemas import StoredToken
from registry.egress_auth.service import EgressAuthService
from registry.egress_auth.upstream_binding import (
    base_url,
    bound_upstreams,
    registered_upstreams,
)
from registry.secrets.interfaces import SecretStoreBase
from registry.utils.credential_encryption import encrypt_credential

# Registered ("consented") upstream and a retarget.
REGISTERED = "https://api.example.com/mcp"
REGISTERED_BASE = "https://api.example.com"
NEW_BASE = "https://new.example.example"
NEW_VERSION_BASE = "https://v2.example.net"


class _InMemoryStore(SecretStoreBase):
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
    """Store a github credential bound to ``bound`` (list of base URLs)."""
    token = StoredToken(
        access_token="gho_secret",
        client_id=client_id,
        expires_at="2999-01-01T00:00:00+00:00",
        bound_upstreams=bound,
        **over,
    )
    await svc._store.put_token("oauth2", "alice", "github", "/github", token)


@pytest.mark.unit
class TestBoundUpstreamsHelper:
    def test_base_url_strips_path_and_lowercases(self):
        assert base_url("HTTPS://Api.Example.com:443/v3/mcp") == "https://api.example.com:443"

    def test_registered_set_is_proxy_pass_plus_versions(self):
        server = {
            "proxy_pass_url": REGISTERED,
            "versions": [{"proxy_pass_url": NEW_VERSION_BASE + "/mcp"}],
        }
        assert registered_upstreams(server) == {REGISTERED_BASE, NEW_VERSION_BASE}

    def test_bound_is_sorted_deterministic(self):
        server = {"proxy_pass_url": REGISTERED, "versions": [{"proxy_pass_url": NEW_BASE}]}
        assert bound_upstreams(server) == sorted([REGISTERED_BASE, NEW_BASE])

    def test_no_upstream_binds_empty(self):
        assert bound_upstreams({}) == []


@pytest.mark.unit
class TestRetargetIsRefused:
    async def test_repointed_proxy_pass_url_refuses_vend(self, svc, egress_oauth):
        # Credential consented against the registered host.
        await _seed(svc, bound=[REGISTERED_BASE])
        # Admin repoints proxy_pass_url at a host they control; the live server
        # record and the minted upstream claim now BOTH read the new host,
        # so the route's live registered-set cross-check passes. The stored
        # binding is the only anchor -- and it refuses.
        assert (
            await svc.get_valid_token(
                "oauth2", "alice", "/github", egress_oauth, requested_upstream=NEW_BASE
            )
            is None
        )

    async def test_registered_upstream_still_vends(self, svc, egress_oauth):
        await _seed(svc, bound=[REGISTERED_BASE])
        assert (
            await svc.get_valid_token(
                "oauth2", "alice", "/github", egress_oauth, requested_upstream=REGISTERED_BASE
            )
            == "gho_secret"
        )

    async def test_added_version_needs_reconnect_but_old_route_works(self, svc, egress_oauth):
        # Consent happened before a new version's base was added, so the binding
        # holds only the old base. The old route still vends; the new base is a
        # miss (one reconnect) -- never a silent vend to the just-added host.
        await _seed(svc, bound=[REGISTERED_BASE])
        assert (
            await svc.get_valid_token(
                "oauth2", "alice", "/github", egress_oauth, requested_upstream=REGISTERED_BASE
            )
            == "gho_secret"
        )
        assert (
            await svc.get_valid_token(
                "oauth2", "alice", "/github", egress_oauth, requested_upstream=NEW_VERSION_BASE
            )
            is None
        )

    async def test_legacy_entry_without_binding_is_a_miss(self, svc, egress_oauth):
        # Pre-upgrade credential: empty bound set never matches -> one forced
        # reconnect, regardless of the requested upstream.
        await _seed(svc, bound=[])
        assert (
            await svc.get_valid_token(
                "oauth2", "alice", "/github", egress_oauth, requested_upstream=REGISTERED_BASE
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
                bound_upstreams=[REGISTERED_BASE],
            ),
        )

        async def fake_post(cfg, data, headers):
            return {"access_token": "at_refreshed", "expires_in": 3600, "scope": "repo"}

        monkeypatch.setattr(oauth_engine, "_post_token", fake_post)
        assert (
            await svc.get_valid_token(
                "oauth2", "alice", "/github", egress_oauth, requested_upstream=REGISTERED_BASE
            )
            == "at_refreshed"
        )
        stored = await svc._store.get_token("oauth2", "alice", "github", "/github")
        assert stored.bound_upstreams == [REGISTERED_BASE]
        # And the refreshed credential still vends to the bound host next time.
        assert (
            await svc.get_valid_token(
                "oauth2", "alice", "/github", egress_oauth, requested_upstream=REGISTERED_BASE
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
                bound_upstreams=[REGISTERED_BASE],
            ),
        )
        # Registered host vends; retargeted host misses.
        assert (
            await svc.get_pat(
                "oauth2", "alice", "github", "/github", requested_upstream=REGISTERED_BASE
            )
            == "ghp_x"
        )
        assert (
            await svc.get_pat("oauth2", "alice", "github", "/github", requested_upstream=NEW_BASE)
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
                bound_upstreams=[REGISTERED_BASE],
                bound_token_url=consented_token_url,
            ),
        )
        # Upstream binding satisfied; only the token endpoint moved.
        repointed = self._custom_oauth("https://new.example/token")
        assert (
            await svc.get_valid_token(
                "oauth2", "alice", "/custom", repointed, requested_upstream=REGISTERED_BASE
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
                bound_upstreams=[REGISTERED_BASE],
                bound_token_url=token_url,
            ),
        )
        assert (
            await svc.get_valid_token(
                "oauth2",
                "alice",
                "/custom",
                self._custom_oauth(token_url),
                requested_upstream=REGISTERED_BASE,
            )
            == "at"
        )
