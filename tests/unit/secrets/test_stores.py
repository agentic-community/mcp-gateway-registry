"""Behavioral tests for both SecretStore backends.

Each backend is exercised through the same async contract (put/get/delete/
list_for_user) plus the cross-store round-trip guard: an Auth0-style user_id,
a Keycloak username with a space, and a multi-segment server_path must store
and read back identically on every backend.

- secrets-manager: moto-mocked AWS Secrets Manager.
- openbao: an in-memory fake hvac KV v2 client (the store logic, not hvac, is
  what we test here; a dockerized OpenBao is exercised in integration tests).
"""

import json

import pytest

from registry.egress_auth.schemas import StoredToken
from registry.secrets import keys
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


# --------------------------------------------------------------------------- #
# Overflow generation-cleanup race (issue found in PR #1520 review)
# --------------------------------------------------------------------------- #


class _RaceOnceClient:
    """Wrap a Secrets Manager client and, on the FIRST read of an overflow shard,
    simulate a concurrent same-principal commit that already moved the layout on:
    rewrite the root to the given ``new_root`` and delete the old shards, all
    before the caller's shard read returns. This reproduces the reader-vs-cleanup
    race (a reader holding the old manifest fetches a shard that cleanup just
    removed). A correct reader re-reads the root -- which now holds ``new_root`` --
    and succeeds. Exactly one read is sabotaged.
    """

    def __init__(self, client, root_name: str, new_root: dict) -> None:
        self._client = client
        self._root_name = root_name
        self._new_root = new_root
        self._sabotaged = False

    def get_secret_value(self, **kwargs):
        secret_id = kwargs.get("SecretId", "")
        if "/overflow/" in secret_id and not self._sabotaged:
            self._sabotaged = True
            # Flip the root to the new (post-commit) layout, then delete every old
            # shard -- the concurrent cleanup that races our in-flight read.
            self._client.put_secret_value(
                SecretId=self._root_name, SecretString=json.dumps(self._new_root)
            )
            token = None
            while True:
                page = self._client.list_secrets(
                    Filters=[{"Key": "name", "Values": [self._root_name + "/overflow/"]}],
                    **({"NextToken": token} if token else {}),
                )
                for entry in page.get("SecretList", []):
                    self._client.delete_secret(
                        SecretId=entry["Name"], ForceDeleteWithoutRecovery=True
                    )
                token = page.get("NextToken")
                if not token:
                    break
        return self._client.get_secret_value(**kwargs)

    def __getattr__(self, name):
        return getattr(self._client, name)


@pytest.mark.unit
class TestOverflowStaleReadRetry:
    def _big_token(self, marker: str):
        # ~4 KiB each: two together exceed the 6 KiB target (forcing overflow to
        # shards), but either one alone fits comfortably in a single shard.
        return _token(marker + "x" * 4000)

    async def _sharded_store(self):
        moto = pytest.importorskip("moto")
        boto3 = pytest.importorskip("boto3")
        ctx = moto.mock_aws()
        ctx.start()
        client = boto3.client("secretsmanager", region_name="us-east-1")
        # 6 KiB target: one ~4 KiB token fits per shard, two overflow.
        store = SecretsManagerStore(client=client, prefix="mcp/egress", target_payload_bytes=6144)
        tok_a = self._big_token("a")
        tok_b = self._big_token("b")
        await store.put_token("oauth2", "alice", "github", "/github", tok_a)
        await store.put_token("oauth2", "alice", "slack", "/slack", tok_b)
        # The principal name is encoded; derive the real root secret id from the store.
        root_name = store._secret_name("oauth2", "alice")
        # Confirm we are actually in the sharded layout.
        root = client.get_secret_value(SecretId=root_name)["SecretString"]
        assert "_egress" in root, "test setup did not reach sharded layout"
        # The layout a concurrent commit would leave behind (compacted inline map).
        new_root = {
            keys.map_key("github", "/github"): tok_a.model_dump(),
            keys.map_key("slack", "/slack"): tok_b.model_dump(),
        }
        return store, client, root_name, new_root, ctx

    async def test_get_token_recovers_from_shard_cleanup_race(self):
        store, client, root_name, new_root, ctx = await self._sharded_store()
        try:
            store._client = _RaceOnceClient(client, root_name, new_root)
            # The first shard read is sabotaged; the retry must re-read and succeed.
            got = await store.get_token("oauth2", "alice", "github", "/github")
            assert got is not None and got.access_token.startswith("a")
        finally:
            ctx.stop()

    async def test_list_recovers_from_shard_cleanup_race(self):
        store, client, root_name, new_root, ctx = await self._sharded_store()
        try:
            store._client = _RaceOnceClient(client, root_name, new_root)
            pairs = sorted((p, s) for p, s, _ in await store.list_for_user("oauth2", "alice"))
            assert pairs == [("github", "/github"), ("slack", "/slack")]
        finally:
            ctx.stop()


# --------------------------------------------------------------------------- #
# OpenBao transient-unavailability retry (HA leader election / pod restart)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestOpenBaoTransientRetry:
    """Regression: a Vault/OpenBao HA cluster briefly rejects requests while it
    (re-)elects a leader after a pod restart (eviction / rollout / spot reclaim).
    A consent's token WRITE that lands in that window used to bubble a hard
    SecretStoreError -> the user saw "Connected" (or a 500) but no token was
    vaulted -> a silent "0 tools". The store MUST ride out the blip with a
    bounded backoff retry. Backoff is shrunk to 0 here so the test is instant."""

    @pytest.fixture(autouse=True)
    def _no_backoff(self, monkeypatch):
        import registry.secrets.openbao.store as store_mod

        monkeypatch.setattr(store_mod, "_TRANSIENT_BACKOFF_BASE", 0.0)

    def _store(self):
        kv = _FakeKvV2()
        client = _FakeHvacClient()
        client.secrets.kv.v2 = kv
        return OpenBaoStore(client=client, mount_point="secret", prefix="mcp/egress"), kv

    async def test_write_retries_through_transient_then_persists(self):
        store, kv = self._store()
        real_write = kv.create_or_update_secret
        calls = {"n": 0}

        def flaky_write(path, secret, mount_point):
            calls["n"] += 1
            if calls["n"] < 3:  # fail twice (leader election), then succeed
                raise Exception("local node not active but active cluster node not found")
            return real_write(path, secret, mount_point)

        kv.create_or_update_secret = flaky_write
        await store.put_token("oauth2", "alice", "github", "/github-mcp", _token("a1"))
        assert calls["n"] == 3
        got = await store.get_token("oauth2", "alice", "github", "/github-mcp")
        assert got is not None and got.access_token == "a1"

    async def test_read_retries_through_connection_refused(self):
        store, kv = self._store()
        await store.put_token("oauth2", "alice", "github", "/github-mcp", _token("a1"))
        real_read = kv.read_secret_version
        calls = {"n": 0}

        def flaky_read(path, mount_point, raise_on_deleted_version=False):
            calls["n"] += 1
            if calls["n"] < 2:
                raise Exception(
                    "HTTPConnectionPool: Max retries exceeded ... [Errno 111] Connection refused"
                )
            return real_read(path, mount_point, raise_on_deleted_version)

        kv.read_secret_version = flaky_read
        got = await store.get_token("oauth2", "alice", "github", "/github-mcp")
        assert got is not None and got.access_token == "a1"

    async def test_persistent_transient_raises_after_bounded_retries(self):
        import registry.secrets.openbao.store as store_mod
        from registry.secrets.interfaces import SecretStoreError

        store, kv = self._store()
        calls = {"n": 0}

        def always_down(*a, **k):
            calls["n"] += 1
            raise Exception("connection refused")

        kv.create_or_update_secret = always_down
        with pytest.raises(SecretStoreError):
            await store.put_token("oauth2", "alice", "github", "/github-mcp", _token())
        # initial attempt + N bounded retries, then give up (no infinite loop)
        assert calls["n"] == store_mod._TRANSIENT_RETRIES + 1

    async def test_non_transient_error_is_not_retried(self):
        store, kv = self._store()
        calls = {"n": 0}

        def bad_write(*a, **k):
            calls["n"] += 1
            raise Exception("malformed data: not retryable")

        kv.create_or_update_secret = bad_write
        from registry.secrets.interfaces import SecretStoreError

        with pytest.raises(SecretStoreError):
            await store.put_token("oauth2", "alice", "github", "/github-mcp", _token())
        assert calls["n"] == 1  # raised immediately, no retry

    async def test_write_retries_on_transient_error_type(self):
        # The other transient tests match by MESSAGE (e.g. "local node not
        # active"); this one matches by EXCEPTION TYPE NAME. hvac/urllib3 raise
        # typed errors (ConnectionError, ReadTimeout, ...) whose str() may carry
        # no recognizable phrase, so _is_transient_error must classify them by
        # class name alone -- otherwise a leader-election blip surfacing as a bare
        # ConnectionError would fall straight through as a hard failure.
        store, kv = self._store()
        real_write = kv.create_or_update_secret
        calls = {"n": 0}

        def flaky_write(path, secret, mount_point):
            calls["n"] += 1
            if calls["n"] < 2:  # a bare typed transport error, no telltale text
                raise ConnectionError()
            return real_write(path, secret, mount_point)

        kv.create_or_update_secret = flaky_write
        await store.put_token("oauth2", "alice", "github", "/github-mcp", _token("a1"))
        assert calls["n"] == 2
        got = await store.get_token("oauth2", "alice", "github", "/github-mcp")
        assert got is not None and got.access_token == "a1"


@pytest.mark.unit
class TestTransientRetryBudget:
    """The retry budget is exported so out-of-process callers (the auth-server
    egress vend timeout) can size themselves relative to it instead of hardcoding
    a coupled magic number."""

    def test_budget_matches_backoff_sum(self):
        import registry.secrets.openbao.store as store_mod

        expected = sum(
            store_mod._TRANSIENT_BACKOFF_BASE * (2**i) for i in range(store_mod._TRANSIENT_RETRIES)
        )
        assert store_mod.transient_retry_budget_seconds() == expected

    def test_budget_is_current_default(self):
        # 0.5 + 1 + 2 + 4 = 7.5s with today's constants; a guard against silent
        # drift that would decouple it from the auth-server vend timeout.
        from registry.secrets.openbao.store import transient_retry_budget_seconds

        assert transient_retry_budget_seconds() == pytest.approx(7.5)
