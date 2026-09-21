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
from registry.secrets.credential_codec import (
    _AAD_PREFIX,
    CredentialCodec,
    build_credential_codec,
)
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
        doc = codec.encode(*_ADDR, _token(), purpose=keys.EGRESS_PURPOSE)
        assert doc == _token().model_dump()
        assert codec.needs_migration(doc) is False

    def test_encode_produces_versioned_envelope_without_plaintext(self):
        codec = build_credential_codec(_KEY)
        doc = codec.encode(*_ADDR, _token(), purpose=keys.EGRESS_PURPOSE)
        assert doc["_encrypted"] is True
        assert doc["version"] == 1
        assert doc["algorithm"] == "AES-256-GCM"
        assert doc["key_id"] == "v1"
        blob = json.dumps(doc)
        assert "gho_super_secret_pat" not in blob
        assert "rt_super_secret_refresh" not in blob

    def test_roundtrip(self):
        codec = build_credential_codec(_KEY)
        doc = codec.encode(*_ADDR, _token(), purpose=keys.EGRESS_PURPOSE)
        got = codec.decode(*_ADDR, doc, purpose=keys.EGRESS_PURPOSE)
        assert got.access_token == "gho_super_secret_pat"
        assert got.refresh_token == "rt_super_secret_refresh"
        assert got.scopes == ["repo", "read:user"]

    def test_fresh_nonce_per_encryption(self):
        codec = build_credential_codec(_KEY)
        a = codec.encode(*_ADDR, _token(), purpose=keys.EGRESS_PURPOSE)
        b = codec.encode(*_ADDR, _token(), purpose=keys.EGRESS_PURPOSE)
        assert a["nonce"] != b["nonce"]
        assert a["ciphertext"] != b["ciphertext"]

    def test_cross_user_substitution_fails(self):
        codec = build_credential_codec(_KEY)
        doc = codec.encode(*_ADDR, _token(), purpose=keys.EGRESS_PURPOSE)
        with pytest.raises(SecretStoreError, match="failed authentication"):
            codec.decode(
                "oauth2",
                "auth0|other",
                "github",
                "/github-mcp/mcp",
                doc,
                purpose=keys.EGRESS_PURPOSE,
            )

    def test_cross_server_substitution_fails(self):
        codec = build_credential_codec(_KEY)
        doc = codec.encode(*_ADDR, _token(), purpose=keys.EGRESS_PURPOSE)
        with pytest.raises(SecretStoreError, match="failed authentication"):
            codec.decode(
                "oauth2",
                "auth0|abc123",
                "github",
                "/other-mcp/mcp",
                doc,
                purpose=keys.EGRESS_PURPOSE,
            )

    def test_cross_provider_substitution_fails(self):
        codec = build_credential_codec(_KEY)
        doc = codec.encode(*_ADDR, _token(), purpose=keys.EGRESS_PURPOSE)
        with pytest.raises(SecretStoreError, match="failed authentication"):
            codec.decode(
                "oauth2",
                "auth0|abc123",
                "slack",
                "/github-mcp/mcp",
                doc,
                purpose=keys.EGRESS_PURPOSE,
            )

    def test_wrong_key_fails(self):
        doc = build_credential_codec(_KEY).encode(*_ADDR, _token(), purpose=keys.EGRESS_PURPOSE)
        with pytest.raises(SecretStoreError, match="failed authentication"):
            build_credential_codec("a-different-root-key-abcdefghijklmnop").decode(
                *_ADDR, doc, purpose=keys.EGRESS_PURPOSE
            )

    def test_envelope_without_key_fails_closed(self):
        doc = build_credential_codec(_KEY).encode(*_ADDR, _token(), purpose=keys.EGRESS_PURPOSE)
        with pytest.raises(SecretStoreError, match="not set"):
            build_credential_codec("").decode(*_ADDR, doc, purpose=keys.EGRESS_PURPOSE)

    def test_unsupported_envelope_fails(self):
        codec = build_credential_codec(_KEY)
        doc = codec.encode(*_ADDR, _token(), purpose=keys.EGRESS_PURPOSE)
        doc["version"] = 999
        with pytest.raises(SecretStoreError, match="Unsupported"):
            codec.decode(*_ADDR, doc, purpose=keys.EGRESS_PURPOSE)

    def test_legacy_plaintext_decoded_and_flagged_for_migration(self):
        codec = build_credential_codec(_KEY)
        legacy = _token().model_dump()
        assert codec.needs_migration(legacy) is True
        got = codec.decode(*_ADDR, legacy, purpose=keys.EGRESS_PURPOSE)
        assert got.access_token == "gho_super_secret_pat"

    def test_tamper_detected(self):
        codec = build_credential_codec(_KEY)
        doc = codec.encode(*_ADDR, _token(), purpose=keys.EGRESS_PURPOSE)
        # Flip a byte of the ciphertext.
        import base64

        ct = bytearray(base64.b64decode(doc["ciphertext"]))
        ct[0] ^= 0x01
        doc["ciphertext"] = base64.b64encode(bytes(ct)).decode("ascii")
        with pytest.raises(SecretStoreError, match="failed authentication"):
            codec.decode(*_ADDR, doc, purpose=keys.EGRESS_PURPOSE)

    def test_error_messages_never_leak_plaintext(self):
        codec = build_credential_codec(_KEY)
        doc = codec.encode(*_ADDR, _token(), purpose=keys.EGRESS_PURPOSE)
        with pytest.raises(SecretStoreError) as exc:
            codec.decode(
                "oauth2",
                "auth0|other",
                "github",
                "/github-mcp/mcp",
                doc,
                purpose=keys.EGRESS_PURPOSE,
            )
        assert "gho_super_secret_pat" not in str(exc.value)
        assert _KEY not in str(exc.value)

    def test_strict_mode_rejects_legacy_plaintext(self):
        codec = build_credential_codec(_KEY, require_encrypted=True)
        assert codec.require_encrypted is True
        with pytest.raises(SecretStoreError, match="REQUIRE_ENCRYPTED"):
            codec.decode(*_ADDR, _token().model_dump(), purpose=keys.EGRESS_PURPOSE)

    def test_strict_mode_still_decodes_envelope(self):
        codec = build_credential_codec(_KEY, require_encrypted=True)
        doc = codec.encode(*_ADDR, _token(), purpose=keys.EGRESS_PURPOSE)
        assert (
            codec.decode(*_ADDR, doc, purpose=keys.EGRESS_PURPOSE).access_token
            == "gho_super_secret_pat"
        )

    def test_strict_mode_inert_without_key(self):
        # require_encrypted with no key must not silently enable strictness (a
        # disabled codec cannot decrypt anything anyway) -- it stays passthrough.
        codec = build_credential_codec("", require_encrypted=True)
        assert codec.enabled is False
        assert codec.require_encrypted is False
        assert (
            codec.decode(*_ADDR, _token().model_dump(), purpose=keys.EGRESS_PURPOSE).access_token
            == "gho_super_secret_pat"
        )


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
        await store.put_token(*_ADDR, _token(), purpose=keys.EGRESS_PURPOSE)

        # The one persisted KV entry must be a ciphertext envelope, not the token.
        stored = list(client.secrets.kv.v2._data.values())
        assert len(stored) == 1
        assert stored[0].get("_encrypted") is True
        assert "gho_super_secret_pat" not in json.dumps(stored[0])
        assert "rt_super_secret_refresh" not in json.dumps(stored[0])

        got = await store.get_token(*_ADDR, purpose=keys.EGRESS_PURPOSE)
        assert got.access_token == "gho_super_secret_pat"
        assert got.refresh_token == "rt_super_secret_refresh"

    async def test_read_repair_on_get(self):
        client = _FakeHvacClient()
        # Seed a legacy plaintext entry with a disabled-codec store.
        await _openbao(client, encrypted=False).put_token(
            *_ADDR, _token(), purpose=keys.EGRESS_PURPOSE
        )
        plaintext_doc = next(iter(client.secrets.kv.v2._data.values()))
        assert "_encrypted" not in plaintext_doc
        enc = _openbao(client, encrypted=True)

        got = await enc.get_token(*_ADDR, purpose=keys.EGRESS_PURPOSE)
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
        await _openbao(client, encrypted=False).put_token(
            *_ADDR, _token("old_access"), purpose=keys.EGRESS_PURPOSE
        )
        enc = _openbao(client, encrypted=True)
        got = await enc.get_token(
            *_ADDR, purpose=keys.EGRESS_PURPOSE
        )  # schedules repair with expected=plaintext(old)
        assert got.access_token == "old_access"
        # Refresh commits a new token (an envelope) before the repair task runs.
        await enc.put_token(*_ADDR, _token("new_access"), purpose=keys.EGRESS_PURPOSE)
        await _drain(enc)
        final = await enc.get_token(*_ADDR, purpose=keys.EGRESS_PURPOSE)
        assert final.access_token == "new_access"  # not rolled back

    async def test_read_repair_on_list(self):
        client = _FakeHvacClient()
        await _openbao(client, encrypted=False).put_token(
            *_ADDR, _token(), purpose=keys.EGRESS_PURPOSE
        )
        enc = _openbao(client, encrypted=True)
        conns = await enc.list_for_user(_ADDR[0], _ADDR[1])
        assert [(p, s) for p, s, _ in conns] == [(_ADDR[2], _ADDR[3])]
        await _drain(enc)
        assert next(iter(client.secrets.kv.v2._data.values())).get("_encrypted") is True

    async def test_missing_key_fails_closed_on_encrypted_entry(self):
        client = _FakeHvacClient()
        await _openbao(client, encrypted=True).put_token(
            *_ADDR, _token(), purpose=keys.EGRESS_PURPOSE
        )
        # A replica without the key must NOT return plaintext.
        with pytest.raises(SecretStoreError):
            await _openbao(client, encrypted=False).get_token(*_ADDR, purpose=keys.EGRESS_PURPOSE)


@pytest.mark.unit
class TestSecretsManagerEncryption:
    async def test_encrypted_roundtrip_and_ciphertext_at_rest(self, secrets_manager_client):
        store = _sm(secrets_manager_client, encrypted=True)
        await store.put_token(*_ADDR, _token(), purpose=keys.EGRESS_PURPOSE)

        name = f"mcp/egress/{keys.user_principal(_ADDR[0], _ADDR[1])}"
        raw = secrets_manager_client.get_secret_value(SecretId=name)["SecretString"]
        assert "gho_super_secret_pat" not in raw
        assert "rt_super_secret_refresh" not in raw
        assert "_encrypted" in raw

        got = await store.get_token(*_ADDR, purpose=keys.EGRESS_PURPOSE)
        assert got.access_token == "gho_super_secret_pat"
        assert got.refresh_token == "rt_super_secret_refresh"

    async def test_read_repair_on_get(self, secrets_manager_client):
        await _sm(secrets_manager_client, encrypted=False).put_token(
            *_ADDR, _token(), purpose=keys.EGRESS_PURPOSE
        )
        name = f"mcp/egress/{keys.user_principal(_ADDR[0], _ADDR[1])}"
        before = secrets_manager_client.get_secret_value(SecretId=name)["SecretString"]
        assert "gho_super_secret_pat" in before  # plaintext at rest

        enc = _sm(secrets_manager_client, encrypted=True)
        got = await enc.get_token(*_ADDR, purpose=keys.EGRESS_PURPOSE)
        assert got.access_token == "gho_super_secret_pat"
        await _drain(enc)

        after = secrets_manager_client.get_secret_value(SecretId=name)["SecretString"]
        assert "gho_super_secret_pat" not in after

    async def test_read_repair_on_list(self, secrets_manager_client):
        await _sm(secrets_manager_client, encrypted=False).put_token(
            *_ADDR, _token(), purpose=keys.EGRESS_PURPOSE
        )
        enc = _sm(secrets_manager_client, encrypted=True)
        conns = await enc.list_for_user(_ADDR[0], _ADDR[1])
        assert [(p, s) for p, s, _ in conns] == [(_ADDR[2], _ADDR[3])]
        await _drain(enc)
        name = f"mcp/egress/{keys.user_principal(_ADDR[0], _ADDR[1])}"
        after = secrets_manager_client.get_secret_value(SecretId=name)["SecretString"]
        assert "gho_super_secret_pat" not in after

    async def test_read_repair_does_not_clobber_concurrent_refresh(self, secrets_manager_client):
        await _sm(secrets_manager_client, encrypted=False).put_token(
            *_ADDR, _token("old_access"), purpose=keys.EGRESS_PURPOSE
        )
        enc = _sm(secrets_manager_client, encrypted=True)
        got = await enc.get_token(
            *_ADDR, purpose=keys.EGRESS_PURPOSE
        )  # schedules compare-and-set repair (expected=old)
        assert got.access_token == "old_access"
        await enc.put_token(
            *_ADDR, _token("new_access"), purpose=keys.EGRESS_PURPOSE
        )  # concurrent refresh
        await _drain(enc)
        final = await enc.get_token(*_ADDR, purpose=keys.EGRESS_PURPOSE)
        assert final.access_token == "new_access"  # not rolled back

    async def test_missing_key_fails_closed_on_encrypted_entry(self, secrets_manager_client):
        await _sm(secrets_manager_client, encrypted=True).put_token(
            *_ADDR, _token(), purpose=keys.EGRESS_PURPOSE
        )
        with pytest.raises(SecretStoreError):
            await _sm(secrets_manager_client, encrypted=False).get_token(
                *_ADDR, purpose=keys.EGRESS_PURPOSE
            )


@pytest.mark.unit
class TestPurposeIsCryptographicallyBound:
    """The two purposes must be separated by the AEAD tag, not only by the storage path.

    This codec's threat model explicitly includes a write-capable attacker on the backend
    -- that is what `require_encrypted` defends against ("cannot downgrade an envelope to
    plaintext or inject a plaintext token"). Under exactly that model, if `purpose` were
    absent from the associated data, such an attacker could COPY a user's egress
    ciphertext to the discovery address and the registry would then borrow the user's own
    runtime credential for its headless calls. That is the precise separation the purpose
    namespace exists to create, so path-only enforcement is not enough.
    """

    def test_egress_ciphertext_cannot_be_read_at_the_discovery_address(self):
        codec = build_credential_codec(_KEY)
        doc = codec.encode(*_ADDR, _token(), purpose=keys.EGRESS_PURPOSE)
        with pytest.raises(SecretStoreError):
            codec.decode(*_ADDR, doc, purpose=keys.DISCOVERY_PURPOSE)

    def test_discovery_ciphertext_cannot_be_read_at_the_egress_address(self):
        """The reverse direction matters too: it would hand the registry's designated
        credential to that user's runtime hop."""
        codec = build_credential_codec(_KEY)
        doc = codec.encode(*_ADDR, _token(), purpose=keys.DISCOVERY_PURPOSE)
        with pytest.raises(SecretStoreError):
            codec.decode(*_ADDR, doc, purpose=keys.EGRESS_PURPOSE)

    @pytest.mark.parametrize("purpose", ["egress", "discovery"])
    def test_round_trip_within_a_purpose(self, purpose):
        codec = build_credential_codec(_KEY)
        doc = codec.encode(*_ADDR, _token(), purpose=purpose)
        assert codec.decode(*_ADDR, doc, purpose=purpose).access_token == "gho_super_secret_pat"

    def test_egress_aad_is_byte_identical_to_the_pre_purpose_form(self):
        """The no-migration guarantee, at the crypto layer.

        Every envelope written before purposes existed used an AAD with no purpose
        segment, and all of them are egress. If the egress AAD ever gains one, every
        already-vaulted credential fails authentication and every user must re-consent.
        Do not "tidy" this into unconditionally appending the segment.
        """
        pre_purpose = _AAD_PREFIX + (
            f"1|AES-256-GCM|v1|"
            f"{keys.encode_segment(_ADDR[0])}|{keys.encode_segment(_ADDR[1])}|"
            f"{keys.encode_segment(_ADDR[2])}|{keys.encode_segment(_ADDR[3])}"
        ).encode("ascii")
        assert CredentialCodec._aad(1, "AES-256-GCM", "v1", *_ADDR, keys.EGRESS_PURPOSE) == (
            pre_purpose
        )
        # ...and a non-egress purpose appends exactly one encoded segment.
        assert CredentialCodec._aad(1, "AES-256-GCM", "v1", *_ADDR, keys.DISCOVERY_PURPOSE) == (
            pre_purpose + b"|" + keys.encode_segment(keys.DISCOVERY_PURPOSE).encode("ascii")
        )


@pytest.mark.unit
class TestReadRepairHonoursPurpose:
    """Read-repair must rewrite the entry at the address it was READ from.

    Encryption is OFF by default, so the common upgrade path is: entries exist in
    plaintext, an operator sets a key, and each read lazily re-encrypts. If the repair
    path hardcodes one purpose, a discovery entry is never re-encrypted -- a delegated
    HUMAN token stays plaintext at rest indefinitely and nothing reports it. Worse on the
    one-document-per-principal backend, where it also takes the mutation lease on the
    USER'S EGRESS document before discovering there is nothing to compare, contending
    with that user's own writes once per health cycle (tier 2 is uncached).
    """

    @pytest.mark.parametrize("purpose", ["egress", "discovery"])
    async def test_secrets_manager_repairs_in_place(self, secrets_manager_client, purpose):
        await _sm(secrets_manager_client, encrypted=False).put_token(
            *_ADDR, _token(), purpose=purpose
        )
        name = (
            f"{keys.namespaced_prefix('mcp/egress', purpose)}/"
            f"{keys.user_principal(_ADDR[0], _ADDR[1])}"
        )
        assert (
            "gho_super_secret_pat"
            in (secrets_manager_client.get_secret_value(SecretId=name)["SecretString"])
        ), "precondition: plaintext at rest"

        enc = _sm(secrets_manager_client, encrypted=True)
        assert (await enc.get_token(*_ADDR, purpose=purpose)).access_token == (
            "gho_super_secret_pat"
        )
        await _drain(enc)

        after = secrets_manager_client.get_secret_value(SecretId=name)["SecretString"]
        assert "gho_super_secret_pat" not in after, f"{purpose} entry was not re-encrypted in place"
        # ...and it must still be READABLE. Encrypting under the wrong purpose would bind
        # an AAD the reader never reproduces, silently destroying the credential: the
        # plaintext is gone and the tag can never authenticate again.
        assert (await enc.get_token(*_ADDR, purpose=purpose)).access_token == (
            "gho_super_secret_pat"
        ), f"{purpose} entry was re-encrypted under the wrong AAD and is now unreadable"

    async def test_secrets_manager_repair_does_not_touch_the_other_purpose(
        self, secrets_manager_client
    ):
        """Repairing discovery must not read, lease, or rewrite the egress document."""
        plain = _sm(secrets_manager_client, encrypted=False)
        await plain.put_token(*_ADDR, _token("egress-token"), purpose=keys.EGRESS_PURPOSE)
        await plain.put_token(*_ADDR, _token("discovery-token"), purpose=keys.DISCOVERY_PURPOSE)

        egress_name = f"mcp/egress/{keys.user_principal(_ADDR[0], _ADDR[1])}"
        before = secrets_manager_client.get_secret_value(SecretId=egress_name)["SecretString"]

        enc = _sm(secrets_manager_client, encrypted=True)
        await enc.get_token(*_ADDR, purpose=keys.DISCOVERY_PURPOSE)
        await _drain(enc)

        after = secrets_manager_client.get_secret_value(SecretId=egress_name)["SecretString"]
        assert after == before, "a discovery repair rewrote the user's egress document"

    @pytest.mark.parametrize("purpose", ["egress", "discovery"])
    async def test_openbao_repairs_in_place(self, purpose):
        client = _FakeHvacClient()
        await _openbao(client, encrypted=False).put_token(*_ADDR, _token(), purpose=purpose)
        enc = _openbao(client, encrypted=True)
        assert (await enc.get_token(*_ADDR, purpose=purpose)).access_token == (
            "gho_super_secret_pat"
        )
        await _drain(enc)
        repaired = next(iter(client.secrets.kv.v2._data.values()))
        assert repaired.get("_encrypted") is True, f"{purpose} entry was not re-encrypted in place"
