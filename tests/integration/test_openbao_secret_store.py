"""Integration tests for OpenBaoStore against a real OpenBao instance.

The unit suite (tests/unit/secrets/test_stores.py) runs OpenBaoStore against an
in-memory fake hvac client -- good for the store's own logic, but it does not
exercise real KV v2 semantics, the hvac wire calls, or auth. This suite runs the
full SecretStore contract against a live OpenBao, and SKIPS cleanly when none is
reachable (so it never breaks CI).

To run locally:

    docker run -d --rm --name mcp-openbao-dev -p 8200:8200 \
        -e BAO_DEV_ROOT_TOKEN_ID=dev-root-token \
        -e BAO_DEV_LISTEN_ADDRESS=0.0.0.0:8200 \
        openbao/openbao:latest server -dev

    export OPENBAO_TEST_ADDR=http://127.0.0.1:8200
    export OPENBAO_TEST_TOKEN=dev-root-token
    uv run pytest tests/integration/test_openbao_secret_store.py -v

Dev mode auto-mounts KV v2 at ``secret/``. The test writes under a unique
``mcp-egress-test/<runid>`` prefix and cleans it up in teardown.
"""

import os

import pytest

from registry.egress_auth.schemas import StoredToken

hvac = pytest.importorskip("hvac")

OPENBAO_ADDR = os.environ.get("OPENBAO_TEST_ADDR", "http://127.0.0.1:8200")
OPENBAO_TOKEN = os.environ.get("OPENBAO_TEST_TOKEN", "dev-root-token")
OPENBAO_MOUNT = os.environ.get("OPENBAO_TEST_KV_MOUNT", "secret")


def _openbao_reachable() -> bool:
    try:
        client = hvac.Client(url=OPENBAO_ADDR, token=OPENBAO_TOKEN, timeout=2)
        return bool(client.is_authenticated())
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _openbao_reachable(),
        reason=(
            "No reachable OpenBao at OPENBAO_TEST_ADDR. Start one with: "
            "docker run -d --rm --name mcp-openbao-dev -p 8200:8200 "
            "-e BAO_DEV_ROOT_TOKEN_ID=dev-root-token "
            "-e BAO_DEV_LISTEN_ADDRESS=0.0.0.0:8200 openbao/openbao:latest server -dev"
        ),
    ),
]

# Unique prefix per process so parallel/repeat runs don't collide; PID is enough
# (Date/random are avoided per project test conventions).
_TEST_PREFIX = f"mcp-egress-test/{os.getpid()}"


def _token(access: str = "gho_real_access") -> StoredToken:
    return StoredToken(
        access_token=access,
        refresh_token="rt_real",
        expires_at="2026-06-19T00:00:00+00:00",
        scopes=["repo", "read:user"],
        client_id="Iv1.realclient",
    )


@pytest.fixture
def store():
    from registry.secrets.openbao.store import OpenBaoStore

    client = hvac.Client(url=OPENBAO_ADDR, token=OPENBAO_TOKEN)
    s = OpenBaoStore(client=client, mount_point=OPENBAO_MOUNT, prefix=_TEST_PREFIX)
    yield s
    # Teardown: best-effort delete everything written under the test prefix.
    try:
        principals = client.secrets.kv.v2.list_secrets(path=_TEST_PREFIX, mount_point=OPENBAO_MOUNT)
        for am in principals.get("data", {}).get("keys", []):
            client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=f"{_TEST_PREFIX}/{am.rstrip('/')}", mount_point=OPENBAO_MOUNT
            )
    except Exception:
        pass


@pytest.mark.asyncio
class TestOpenBaoStoreIntegration:
    async def test_get_miss_returns_none(self, store):
        assert await store.get_token("oauth2", "ghost", "github", "/x") is None

    async def test_put_get_delete_roundtrip(self, store):
        await store.put_token("oauth2", "alice", "github", "/github-mcp", _token())
        got = await store.get_token("oauth2", "alice", "github", "/github-mcp")
        assert got is not None and got.access_token == "gho_real_access"
        assert got.refresh_token == "rt_real" and got.scopes == ["repo", "read:user"]

        await store.delete_token("oauth2", "alice", "github", "/github-mcp")
        assert await store.get_token("oauth2", "alice", "github", "/github-mcp") is None
        # idempotent
        await store.delete_token("oauth2", "alice", "github", "/github-mcp")

    async def test_list_for_user_enumerates_only_that_principal(self, store):
        await store.put_token("oauth2", "bob", "github", "/github-mcp", _token("b-gh"))
        await store.put_token("oauth2", "bob", "slack", "/slack-mcp", _token("b-sl"))
        await store.put_token("oauth2", "carol", "github", "/github-mcp", _token("c-gh"))

        conns = await store.list_for_user("oauth2", "bob")
        assert sorted((p, s) for p, s, _ in conns) == [
            ("github", "/github-mcp"),
            ("slack", "/slack-mcp"),
        ]
        assert all(t.access_token != "c-gh" for _, _, t in conns)

    async def test_auth_method_namespacing_isolates_static_key_from_real_user(self, store):
        # On a REAL store: an operator-named static key "dave" must not read
        # the oauth2 user "dave"'s token.
        await store.put_token("oauth2", "dave", "github", "/github-mcp", _token("real"))
        await store.put_token(
            "network-trusted", "dave", "github", "/github-mcp", _token("staticbot")
        )
        real = await store.get_token("oauth2", "dave", "github", "/github-mcp")
        bot = await store.get_token("network-trusted", "dave", "github", "/github-mcp")
        assert real.access_token == "real"
        assert bot.access_token == "staticbot"

    @pytest.mark.parametrize(
        "auth_method,user_id,provider,server_path",
        [
            ("oauth2", "auth0|abc123", "github", "/github-mcp/mcp"),  # Auth0 "|" + multi-seg
            ("keycloak", "alice smith", "slack", "/slack-mcp"),  # space
            ("okta", "café", "google", "/g/v1/mcp"),  # non-ASCII NFC
        ],
    )
    async def test_hard_keys_roundtrip_on_real_kv(
        self, store, auth_method, user_id, provider, server_path
    ):
        # Proves the canonicalization produces valid OpenBao paths for the
        # characters that would otherwise break KV path segments.
        await store.put_token(auth_method, user_id, provider, server_path, _token("hk"))
        got = await store.get_token(auth_method, user_id, provider, server_path)
        assert got is not None and got.access_token == "hk"
        conns = await store.list_for_user(auth_method, user_id)
        assert (provider, server_path) in [(p, s) for p, s, _ in conns]

    async def test_overwrite_same_key_updates_in_place(self, store):
        await store.put_token("oauth2", "erin", "github", "/github-mcp", _token("v1"))
        await store.put_token("oauth2", "erin", "github", "/github-mcp", _token("v2"))
        got = await store.get_token("oauth2", "erin", "github", "/github-mcp")
        assert got.access_token == "v2"
        # still exactly one connection, not two
        conns = await store.list_for_user("oauth2", "erin")
        assert len(conns) == 1
