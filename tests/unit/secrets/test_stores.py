"""Behavioral tests for both SecretStore backends.

Each backend is exercised through the same async contract (put/get/delete/
list_for_user) plus the cross-store round-trip guard: an Auth0-style user_id,
a Keycloak username with a space, and a multi-segment server_path must store
and read back identically on every backend.

- secrets-manager: moto-mocked AWS Secrets Manager.
- openbao: an in-memory fake hvac KV v2 client (the store logic, not hvac, is
  what we test here; a dockerized OpenBao is exercised in integration tests).
"""

import pytest

from registry.egress_auth.schemas import StoredToken
from registry.secrets.openbao.store import OpenBaoStore
from registry.secrets.secrets_manager.store import SecretsManagerStore

# (auth_method, user_id, provider, server_path) tuples that stress the encoding.
HARD_KEYS = [
    ("oauth2", "auth0|abc123", "github", "/github-mcp/mcp"),
    ("keycloak", "alice smith", "slack", "/slack-mcp"),
    ("okta", "café", "google", "/g/v1/mcp"),
]


def _token(access: str = "gho_test_access") -> StoredToken:
    return StoredToken(
        access_token=access,
        refresh_token="rt_test",
        expires_at="2026-06-19T00:00:00+00:00",
        scopes=["repo", "read:user"],
        client_id="Iv1.testclient",
    )


# --------------------------------------------------------------------------- #
# In-memory fake hvac KV v2 client (just enough for OpenBaoStore)
# --------------------------------------------------------------------------- #


class InvalidPath(Exception):
    """Mirrors hvac.exceptions.InvalidPath (matched by class name in the store)."""


_InvalidPath = InvalidPath  # alias used below


class _FakeKvV2:
    def __init__(self) -> None:
        self._data: dict[str, dict] = {}

    def create_or_update_secret(self, path, secret, mount_point):
        self._data[path] = dict(secret)

    def read_secret_version(self, path, mount_point, raise_on_deleted_version=False):
        if path not in self._data:
            raise _InvalidPath(path)
        return {"data": {"data": self._data[path]}}

    def delete_metadata_and_all_versions(self, path, mount_point):
        # delete the entry and anything beneath it
        for k in [k for k in self._data if k == path or k.startswith(path + "/")]:
            del self._data[k]

    def list_secrets(self, path, mount_point):
        prefix = path.rstrip("/") + "/"
        children: set[str] = set()
        for k in self._data:
            if k.startswith(prefix):
                rest = k[len(prefix) :]
                top = rest.split("/", 1)[0]
                # a deeper entry exists -> directory; mark with trailing slash
                children.add(top + "/" if "/" in rest else top)
        if not children:
            raise _InvalidPath(path)
        return {"data": {"keys": sorted(children)}}


class _FakeSecrets:
    def __init__(self) -> None:
        self.kv = type("KV", (), {"v2": _FakeKvV2()})()


class _FakeHvacClient:
    def __init__(self) -> None:
        self.secrets = _FakeSecrets()


# --------------------------------------------------------------------------- #
# Backend fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def secrets_manager_store():
    moto = pytest.importorskip("moto")
    boto3 = pytest.importorskip("boto3")
    with moto.mock_aws():
        client = boto3.client("secretsmanager", region_name="us-east-1")
        yield SecretsManagerStore(client=client, prefix="mcp/egress")


@pytest.fixture
def openbao_store():
    return OpenBaoStore(client=_FakeHvacClient(), mount_point="secret", prefix="mcp/egress")


@pytest.fixture(params=["secrets_manager", "openbao"])
def store(request, secrets_manager_store, openbao_store):
    return {
        "secrets_manager": secrets_manager_store,
        "openbao": openbao_store,
    }[request.param]


# --------------------------------------------------------------------------- #
# Contract tests (run against every backend)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestSecretStoreContract:
    async def test_get_miss_returns_none(self, store):
        assert await store.get_token("oauth2", "nobody", "github", "/x") is None

    async def test_put_then_get_roundtrip(self, store):
        tok = _token()
        await store.put_token("oauth2", "alice", "github", "/github-mcp", tok)
        got = await store.get_token("oauth2", "alice", "github", "/github-mcp")
        assert got is not None
        assert got.access_token == tok.access_token
        assert got.refresh_token == tok.refresh_token
        assert got.scopes == tok.scopes

    async def test_delete_is_idempotent(self, store):
        # deleting a missing entry must not raise
        await store.delete_token("oauth2", "ghost", "github", "/x")
        await store.put_token("oauth2", "alice", "github", "/github-mcp", _token())
        await store.delete_token("oauth2", "alice", "github", "/github-mcp")
        assert await store.get_token("oauth2", "alice", "github", "/github-mcp") is None
        await store.delete_token("oauth2", "alice", "github", "/github-mcp")  # again, no raise

    async def test_list_for_user_returns_connections_without_tokens_leaking(self, store):
        await store.put_token("oauth2", "alice", "github", "/github-mcp", _token("a1"))
        await store.put_token("oauth2", "alice", "slack", "/slack-mcp", _token("a2"))
        await store.put_token("oauth2", "bob", "github", "/github-mcp", _token("b1"))

        conns = await store.list_for_user("oauth2", "alice")
        pairs = sorted((p, s) for p, s, _ in conns)
        assert pairs == [("github", "/github-mcp"), ("slack", "/slack-mcp")]
        # bob's entry is not in alice's list
        assert all(s_tok.access_token != "b1" for _, _, s_tok in conns)

    async def test_auth_method_namespacing_prevents_cross_vend(self, store):
        # A network-trusted static-key caller named "alice" must NOT read
        # the real oauth2 user alice's token.
        await store.put_token("oauth2", "alice", "github", "/github-mcp", _token("real"))
        await store.put_token(
            "network-trusted", "alice", "github", "/github-mcp", _token("staticbot")
        )
        real = await store.get_token("oauth2", "alice", "github", "/github-mcp")
        bot = await store.get_token("network-trusted", "alice", "github", "/github-mcp")
        assert real.access_token == "real"
        assert bot.access_token == "staticbot"  # distinct buckets

    @pytest.mark.parametrize("auth_method,user_id,provider,server_path", HARD_KEYS)
    async def test_hard_keys_roundtrip(self, store, auth_method, user_id, provider, server_path):
        tok = _token(f"tok_{user_id}")
        await store.put_token(auth_method, user_id, provider, server_path, tok)
        got = await store.get_token(auth_method, user_id, provider, server_path)
        assert got is not None and got.access_token == tok.access_token
        conns = await store.list_for_user(auth_method, user_id)
        assert (provider, server_path) in [(p, s) for p, s, _ in conns]


# --------------------------------------------------------------------------- #
# OpenBao re-authentication on token expiry (regression)
# --------------------------------------------------------------------------- #


class _Forbidden(Exception):
    """Mirrors hvac.exceptions.Forbidden (matched by class name in the store)."""


@pytest.mark.unit
class TestOpenBaoReauth:
    """Regression: the registry logs into OpenBao once with a short-lived (e.g.
    1h) token and hvac does not auto-renew. When the token lapses, reads fail
    with Forbidden/permission-denied. The store MUST re-authenticate and retry
    once (using the factory-supplied reauthenticate callback) instead of bubbling
    a 500 forever until the process restarts."""

    def _store_that_forbids_until_reauth(self):
        """Return (store, state) where the fake client's first call raises
        Forbidden; reauthenticate() flips a flag so subsequent calls succeed."""
        kv = _FakeKvV2()
        client = _FakeHvacClient()
        client.secrets.kv.v2 = kv
        state = {"authed": False, "logins": 0}

        # Wrap read/write so they fail with Forbidden until authed=True.
        real_read = kv.read_secret_version
        real_write = kv.create_or_update_secret

        def read(path, mount_point, raise_on_deleted_version=False):
            if not state["authed"]:
                raise _Forbidden("permission denied")
            return real_read(path, mount_point, raise_on_deleted_version)

        def write(path, secret, mount_point):
            if not state["authed"]:
                raise _Forbidden("permission denied")
            return real_write(path, secret, mount_point)

        kv.read_secret_version = read
        kv.create_or_update_secret = write

        def reauth():
            state["logins"] += 1
            state["authed"] = True

        store = OpenBaoStore(
            client=client, mount_point="secret", prefix="mcp/egress", reauthenticate=reauth
        )
        return store, state

    async def test_get_reauthenticates_and_retries_once(self):
        store, state = self._store_that_forbids_until_reauth()
        # token gets written only after reauth flips authed=True
        await store.put_token("oauth2", "alice", "github", "/github-mcp", _token("a1"))
        assert state["logins"] == 1  # the put triggered one re-auth
        got = await store.get_token("oauth2", "alice", "github", "/github-mcp")
        assert got is not None and got.access_token == "a1"

    async def test_persistent_forbidden_eventually_raises(self):
        # If re-auth does NOT fix it (real policy gap), the second attempt fails
        # and the store surfaces a SecretStoreError rather than looping.
        kv = _FakeKvV2()
        client = _FakeHvacClient()
        client.secrets.kv.v2 = kv

        def always_forbid(*a, **k):
            raise _Forbidden("permission denied")

        kv.read_secret_version = always_forbid
        logins = {"n": 0}

        def reauth():
            logins["n"] += 1

        store = OpenBaoStore(
            client=client, mount_point="secret", prefix="mcp/egress", reauthenticate=reauth
        )
        from registry.secrets.interfaces import SecretStoreError

        with pytest.raises(SecretStoreError):
            await store.get_token("oauth2", "alice", "github", "/github-mcp")
        assert logins["n"] == 1  # retried exactly once
