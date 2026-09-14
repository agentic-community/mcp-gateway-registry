"""Encrypted-persistence tests for the egress credential vault (issue #1665).

Covers the application-layer AEAD codec and its integration with BOTH secret
store backends:
- encrypted round-trip (PAT + 3LO access/refresh) through put/get/list/delete;
- the persisted representation is ciphertext, never the plaintext token;
- cross-user and cross-server/provider ciphertext substitution fails closed;
- a missing/wrong key fails closed (never returns/overwrites plaintext);
- legacy plaintext entries are recognized on read and re-encrypted in place
  (read-repair) via both get_token and list_for_user.

The in-memory fake hvac client and the moto Secrets Manager fixture mirror
``tests/unit/secrets/test_stores.py`` so these run without live backends.
"""

import json

import pytest

from registry.egress_auth.schemas import StoredToken
from registry.secrets import keys
from registry.secrets.credential_codec import CredentialCodec, build_credential_codec
from registry.secrets.interfaces import SecretStoreError
from registry.secrets.openbao.store import OpenBaoStore
from registry.secrets.secrets_manager.store import SecretsManagerStore

_KEY = "unit-test-egress-root-key-abcdefghijklmnop"  # >= 32 bytes, non-placeholder
_ADDR = ("oauth2", "auth0|abc123", "github", "/github-mcp/mcp")


def _token(access: str = "gho_super_secret_pat") -> StoredToken:
    return StoredToken(
        access_token=access,
        refresh_token="rt_super_secret_refresh",
        expires_at="2026-06-19T00:00:00+00:00",
        scopes=["repo", "read:user"],
        client_id="Iv1.testclient",
    )


# --------------------------------------------------------------------------- #
# Codec unit tests
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestCredentialCodec:
    def test_disabled_codec_is_passthrough(self):
        codec = build_credential_codec("")
        assert codec.enabled is False
        doc = codec.encode(*_ADDR, _token())
        assert doc == _token().model_dump()
        assert codec.needs_migration(doc) is False

    def test_encode_produces_versioned_envelope_without_plaintext(self):
        codec = build_credential_codec(_KEY)
        doc = codec.encode(*_ADDR, _token())
        assert doc["_encrypted"] is True
        assert doc["version"] == 1
        assert doc["algorithm"] == "AES-256-GCM"
        assert doc["key_id"] == "v1"
        blob = json.dumps(doc)
        assert "gho_super_secret_pat" not in blob
        assert "rt_super_secret_refresh" not in blob

    def test_roundtrip(self):
        codec = build_credential_codec(_KEY)
        doc = codec.encode(*_ADDR, _token())
        got = codec.decode(*_ADDR, doc)
        assert got.access_token == "gho_super_secret_pat"
        assert got.refresh_token == "rt_super_secret_refresh"
        assert got.scopes == ["repo", "read:user"]

    def test_fresh_nonce_per_encryption(self):
        codec = build_credential_codec(_KEY)
        a = codec.encode(*_ADDR, _token())
        b = codec.encode(*_ADDR, _token())
        assert a["nonce"] != b["nonce"]
        assert a["ciphertext"] != b["ciphertext"]

    def test_cross_user_substitution_fails(self):
        codec = build_credential_codec(_KEY)
        doc = codec.encode(*_ADDR, _token())
        with pytest.raises(SecretStoreError, match="failed authentication"):
            codec.decode("oauth2", "auth0|other", "github", "/github-mcp/mcp", doc)

    def test_cross_server_substitution_fails(self):
        codec = build_credential_codec(_KEY)
        doc = codec.encode(*_ADDR, _token())
        with pytest.raises(SecretStoreError, match="failed authentication"):
            codec.decode("oauth2", "auth0|abc123", "github", "/other-mcp/mcp", doc)

    def test_cross_provider_substitution_fails(self):
        codec = build_credential_codec(_KEY)
        doc = codec.encode(*_ADDR, _token())
        with pytest.raises(SecretStoreError, match="failed authentication"):
            codec.decode("oauth2", "auth0|abc123", "slack", "/github-mcp/mcp", doc)

    def test_wrong_key_fails(self):
        doc = build_credential_codec(_KEY).encode(*_ADDR, _token())
        with pytest.raises(SecretStoreError, match="failed authentication"):
            build_credential_codec("a-different-root-key-abcdefghijklmnop").decode(*_ADDR, doc)

    def test_envelope_without_key_fails_closed(self):
        doc = build_credential_codec(_KEY).encode(*_ADDR, _token())
        with pytest.raises(SecretStoreError, match="not set"):
            build_credential_codec("").decode(*_ADDR, doc)

    def test_unsupported_envelope_fails(self):
        codec = build_credential_codec(_KEY)
        doc = codec.encode(*_ADDR, _token())
        doc["version"] = 999
        with pytest.raises(SecretStoreError, match="Unsupported"):
            codec.decode(*_ADDR, doc)

    def test_legacy_plaintext_decoded_and_flagged_for_migration(self):
        codec = build_credential_codec(_KEY)
        legacy = _token().model_dump()
        assert codec.needs_migration(legacy) is True
        got = codec.decode(*_ADDR, legacy)
        assert got.access_token == "gho_super_secret_pat"

    def test_tamper_detected(self):
        codec = build_credential_codec(_KEY)
        doc = codec.encode(*_ADDR, _token())
        # Flip a byte of the ciphertext.
        import base64

        ct = bytearray(base64.b64decode(doc["ciphertext"]))
        ct[0] ^= 0x01
        doc["ciphertext"] = base64.b64encode(bytes(ct)).decode("ascii")
        with pytest.raises(SecretStoreError, match="failed authentication"):
            codec.decode(*_ADDR, doc)

    def test_error_messages_never_leak_plaintext(self):
        codec = build_credential_codec(_KEY)
        doc = codec.encode(*_ADDR, _token())
        with pytest.raises(SecretStoreError) as exc:
            codec.decode("oauth2", "auth0|other", "github", "/github-mcp/mcp", doc)
        assert "gho_super_secret_pat" not in str(exc.value)
        assert _KEY not in str(exc.value)

    def test_strict_mode_rejects_legacy_plaintext(self):
        codec = build_credential_codec(_KEY, require_encrypted=True)
        assert codec.require_encrypted is True
        with pytest.raises(SecretStoreError, match="REQUIRE_ENCRYPTED"):
            codec.decode(*_ADDR, _token().model_dump())

    def test_strict_mode_still_decodes_envelope(self):
        codec = build_credential_codec(_KEY, require_encrypted=True)
        doc = codec.encode(*_ADDR, _token())
        assert codec.decode(*_ADDR, doc).access_token == "gho_super_secret_pat"

    def test_strict_mode_inert_without_key(self):
        # require_encrypted with no key must not silently enable strictness (a
        # disabled codec cannot decrypt anything anyway) -- it stays passthrough.
        codec = build_credential_codec("", require_encrypted=True)
        assert codec.enabled is False
        assert codec.require_encrypted is False
        assert codec.decode(*_ADDR, _token().model_dump()).access_token == "gho_super_secret_pat"


# --------------------------------------------------------------------------- #
# Store-backed encryption + read-repair
# --------------------------------------------------------------------------- #

# Reuse the in-memory fake hvac client from the sibling store tests.
from tests.unit.secrets.test_stores import _FakeHvacClient  # noqa: E402


async def _drain(store):
    """Await any fire-and-forget read-repair migrations the store scheduled."""
    import asyncio

    await asyncio.gather(*list(store._repair_tasks))


def _openbao(client, *, encrypted: bool):
    codec = build_credential_codec(_KEY) if encrypted else CredentialCodec(root_key=None)
    return OpenBaoStore(client=client, mount_point="secret", prefix="mcp/egress", codec=codec)


@pytest.fixture
def secrets_manager_client():
    moto = pytest.importorskip("moto")
    boto3 = pytest.importorskip("boto3")
    with moto.mock_aws():
        yield boto3.client("secretsmanager", region_name="us-east-1")


def _sm(client, *, encrypted: bool):
    codec = build_credential_codec(_KEY) if encrypted else CredentialCodec(root_key=None)
    return SecretsManagerStore(client=client, prefix="mcp/egress", codec=codec)


@pytest.mark.unit
class TestOpenBaoEncryption:
    async def test_encrypted_roundtrip_and_ciphertext_at_rest(self):
        client = _FakeHvacClient()
        store = _openbao(client, encrypted=True)
        await store.put_token(*_ADDR, _token())

        # The one persisted KV entry must be a ciphertext envelope, not the token.
        stored = list(client.secrets.kv.v2._data.values())
        assert len(stored) == 1
        assert stored[0].get("_encrypted") is True
        assert "gho_super_secret_pat" not in json.dumps(stored[0])
        assert "rt_super_secret_refresh" not in json.dumps(stored[0])

        got = await store.get_token(*_ADDR)
        assert got.access_token == "gho_super_secret_pat"
        assert got.refresh_token == "rt_super_secret_refresh"

    async def test_read_repair_on_get(self):
        client = _FakeHvacClient()
        # Seed a legacy plaintext entry with a disabled-codec store.
        await _openbao(client, encrypted=False).put_token(*_ADDR, _token())
        plaintext_doc = next(iter(client.secrets.kv.v2._data.values()))
        assert "_encrypted" not in plaintext_doc
        enc = _openbao(client, encrypted=True)

        got = await enc.get_token(*_ADDR)
        assert got.access_token == "gho_super_secret_pat"
        await _drain(enc)

        # Read-repair rewrote the entry as ciphertext in place.
        repaired = next(iter(client.secrets.kv.v2._data.values()))
        assert repaired.get("_encrypted") is True

    async def test_read_repair_does_not_clobber_concurrent_refresh(self):
        # Regression: a legacy read schedules a repair capturing the OLD token;
        # a concurrent refresh then writes a NEW token. Compare-and-set must make
        # the repair skip, NOT roll the credential back to the stale token.
        client = _FakeHvacClient()
        await _openbao(client, encrypted=False).put_token(*_ADDR, _token("old_access"))
        enc = _openbao(client, encrypted=True)
        got = await enc.get_token(*_ADDR)  # schedules repair with expected=plaintext(old)
        assert got.access_token == "old_access"
        # Refresh commits a new token (an envelope) before the repair task runs.
        await enc.put_token(*_ADDR, _token("new_access"))
        await _drain(enc)
        final = await enc.get_token(*_ADDR)
        assert final.access_token == "new_access"  # not rolled back

    async def test_read_repair_on_list(self):
        client = _FakeHvacClient()
        await _openbao(client, encrypted=False).put_token(*_ADDR, _token())
        enc = _openbao(client, encrypted=True)
        conns = await enc.list_for_user(_ADDR[0], _ADDR[1])
        assert [(p, s) for p, s, _ in conns] == [(_ADDR[2], _ADDR[3])]
        await _drain(enc)
        assert next(iter(client.secrets.kv.v2._data.values())).get("_encrypted") is True

    async def test_missing_key_fails_closed_on_encrypted_entry(self):
        client = _FakeHvacClient()
        await _openbao(client, encrypted=True).put_token(*_ADDR, _token())
        # A replica without the key must NOT return plaintext.
        with pytest.raises(SecretStoreError):
            await _openbao(client, encrypted=False).get_token(*_ADDR)


@pytest.mark.unit
class TestSecretsManagerEncryption:
    async def test_encrypted_roundtrip_and_ciphertext_at_rest(self, secrets_manager_client):
        store = _sm(secrets_manager_client, encrypted=True)
        await store.put_token(*_ADDR, _token())

        name = f"mcp/egress/{keys.user_principal(_ADDR[0], _ADDR[1])}"
        raw = secrets_manager_client.get_secret_value(SecretId=name)["SecretString"]
        assert "gho_super_secret_pat" not in raw
        assert "rt_super_secret_refresh" not in raw
        assert "_encrypted" in raw

        got = await store.get_token(*_ADDR)
        assert got.access_token == "gho_super_secret_pat"
        assert got.refresh_token == "rt_super_secret_refresh"

    async def test_read_repair_on_get(self, secrets_manager_client):
        await _sm(secrets_manager_client, encrypted=False).put_token(*_ADDR, _token())
        name = f"mcp/egress/{keys.user_principal(_ADDR[0], _ADDR[1])}"
        before = secrets_manager_client.get_secret_value(SecretId=name)["SecretString"]
        assert "gho_super_secret_pat" in before  # plaintext at rest

        enc = _sm(secrets_manager_client, encrypted=True)
        got = await enc.get_token(*_ADDR)
        assert got.access_token == "gho_super_secret_pat"
        await _drain(enc)

        after = secrets_manager_client.get_secret_value(SecretId=name)["SecretString"]
        assert "gho_super_secret_pat" not in after

    async def test_read_repair_on_list(self, secrets_manager_client):
        await _sm(secrets_manager_client, encrypted=False).put_token(*_ADDR, _token())
        enc = _sm(secrets_manager_client, encrypted=True)
        conns = await enc.list_for_user(_ADDR[0], _ADDR[1])
        assert [(p, s) for p, s, _ in conns] == [(_ADDR[2], _ADDR[3])]
        await _drain(enc)
        name = f"mcp/egress/{keys.user_principal(_ADDR[0], _ADDR[1])}"
        after = secrets_manager_client.get_secret_value(SecretId=name)["SecretString"]
        assert "gho_super_secret_pat" not in after

    async def test_read_repair_does_not_clobber_concurrent_refresh(self, secrets_manager_client):
        await _sm(secrets_manager_client, encrypted=False).put_token(*_ADDR, _token("old_access"))
        enc = _sm(secrets_manager_client, encrypted=True)
        got = await enc.get_token(*_ADDR)  # schedules compare-and-set repair (expected=old)
        assert got.access_token == "old_access"
        await enc.put_token(*_ADDR, _token("new_access"))  # concurrent refresh
        await _drain(enc)
        final = await enc.get_token(*_ADDR)
        assert final.access_token == "new_access"  # not rolled back

    async def test_missing_key_fails_closed_on_encrypted_entry(self, secrets_manager_client):
        await _sm(secrets_manager_client, encrypted=True).put_token(*_ADDR, _token())
        with pytest.raises(SecretStoreError):
            await _sm(secrets_manager_client, encrypted=False).get_token(*_ADDR)
