"""Unit tests for auth_server/session_store.py and the shared session_crypto helpers.

The collection layer is mocked with an in-memory dict, so these tests exercise:
  - the encrypt/decrypt round-trip (real AES-GCM)
  - the create/resolve/delete control flow
  - TTL-style expiry checks at read time
  - naive vs aware datetime handling
  - failure-mode degradation (store outage → None)

The Mongo client itself is never reached.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from cryptography.exceptions import InvalidTag

from auth_server import session_store
from registry.auth import session_crypto

pytestmark = [pytest.mark.unit, pytest.mark.auth]


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch):
    """Reset module-level singletons so each test starts clean."""
    monkeypatch.setattr(session_store, "_collection", None)
    monkeypatch.setattr(session_store, "_indexes_created", False)
    # Reset the shared cipher singleton too so SECRET_KEY changes are honored.
    monkeypatch.setattr(session_crypto, "_aesgcm", None)


@pytest.fixture
def fake_collection(monkeypatch):
    """Replace session_store._get_collection with an in-memory mock."""

    storage: dict[str, dict[str, Any]] = {}

    collection = MagicMock()

    async def insert_one(doc):
        storage[doc["session_id"]] = doc
        result = MagicMock()
        result.inserted_id = doc["session_id"]
        return result

    async def find_one(query):
        return storage.get(query["session_id"])

    async def delete_one(query):
        existed = query["session_id"] in storage
        storage.pop(query["session_id"], None)
        result = MagicMock()
        result.deleted_count = 1 if existed else 0
        return result

    collection.insert_one = AsyncMock(side_effect=insert_one)
    collection.find_one = AsyncMock(side_effect=find_one)
    collection.delete_one = AsyncMock(side_effect=delete_one)

    async def _fake_get_collection():
        return collection

    monkeypatch.setattr(session_store, "_get_collection", _fake_get_collection)
    return storage


# ---------------------------------------------------------------------------
# session_crypto round-trips
# ---------------------------------------------------------------------------


class TestSessionCrypto:
    def test_encrypt_decrypt_round_trip(self):
        token = "eyJhbGciOiJSUzI1NiIs.payload.sig"
        blob = session_crypto.encrypt_id_token(token)
        assert blob[: session_crypto.NONCE_BYTES] != b"\x00" * session_crypto.NONCE_BYTES
        assert session_crypto.decrypt_id_token(blob) == token

    def test_each_encrypt_produces_unique_nonce(self):
        token = "abc.def.ghi"
        a = session_crypto.encrypt_id_token(token)
        b = session_crypto.encrypt_id_token(token)
        # Same plaintext under random nonces must produce distinct ciphertext.
        assert a != b
        assert a[: session_crypto.NONCE_BYTES] != b[: session_crypto.NONCE_BYTES]
        assert session_crypto.decrypt_id_token(a) == token
        assert session_crypto.decrypt_id_token(b) == token

    def test_decrypt_fails_when_secret_key_rotated(self, monkeypatch):
        token = "rotated-token"
        blob = session_crypto.encrypt_id_token(token)
        # Rotate SECRET_KEY → cipher singleton must be re-derived.
        monkeypatch.setenv("SECRET_KEY", "a-completely-different-secret-key-for-rotation")
        monkeypatch.setattr(session_crypto, "_aesgcm", None)
        with pytest.raises(InvalidTag):
            session_crypto.decrypt_id_token(blob)

    def test_missing_secret_key_raises(self, monkeypatch):
        monkeypatch.delenv("SECRET_KEY", raising=False)
        monkeypatch.setattr(session_crypto, "_aesgcm", None)
        with pytest.raises(RuntimeError, match="SECRET_KEY"):
            session_crypto.get_aesgcm()


# ---------------------------------------------------------------------------
# create / resolve / delete
# ---------------------------------------------------------------------------


class TestCreateSession:
    @pytest.mark.asyncio
    async def test_create_returns_session_id_and_persists_document(self, fake_collection):
        sid = await session_store.create_session(
            username="alice",
            email="alice@example.com",
            name="Alice",
            groups=["g1", "g2"],
            provider="entra",
            auth_method="oauth2",
            max_age_seconds=60,
        )
        assert isinstance(sid, str)
        assert len(sid) == 64  # 32 bytes hex
        doc = fake_collection[sid]
        assert doc["username"] == "alice"
        assert doc["groups"] == ["g1", "g2"]
        assert doc["provider"] == "entra"
        assert "id_token_encrypted" not in doc

    @pytest.mark.asyncio
    async def test_graph_confirmed_empty_membership_persists_authority(self, fake_collection):
        sid = await session_store.create_session(
            username="alice",
            email=None,
            name=None,
            groups=[],
            provider="entra",
            auth_method="oauth2",
            max_age_seconds=60,
            entra_graph_groups_resolved=True,
        )

        assert fake_collection[sid]["entra_graph_groups_resolved"] is True
        resolved = await session_store.resolve_session(sid)
        assert resolved["groups"] == []
        assert resolved["entra_graph_groups_resolved"] is True

    @pytest.mark.asyncio
    async def test_create_with_id_token_stores_encrypted_blob(self, fake_collection):
        sid = await session_store.create_session(
            username="bob",
            email=None,
            name=None,
            groups=[],
            provider="keycloak",
            auth_method="oauth2",
            max_age_seconds=60,
            id_token="header.payload.signature",
        )
        doc = fake_collection[sid]
        assert "id_token_encrypted" in doc
        assert isinstance(doc["id_token_encrypted"], bytes)
        # Plaintext id_token must NEVER appear in the document.
        assert "id_token" not in doc
        assert b"header.payload.signature" not in doc["id_token_encrypted"]

    @pytest.mark.asyncio
    async def test_session_ids_are_unique_across_calls(self, fake_collection):
        sids = set()
        for _ in range(50):
            sid = await session_store.create_session(
                username="u",
                email=None,
                name=None,
                groups=[],
                provider="keycloak",
                auth_method="oauth2",
                max_age_seconds=60,
            )
            sids.add(sid)
        assert len(sids) == 50


class TestResolveSession:
    @pytest.mark.asyncio
    async def test_resolve_returns_hydrated_record(self, fake_collection):
        sid = await session_store.create_session(
            username="alice",
            email="alice@example.com",
            name="Alice",
            groups=["g1"],
            provider="entra",
            auth_method="oauth2",
            max_age_seconds=60,
            id_token="id.token.value",
        )
        result = await session_store.resolve_session(sid)
        assert result is not None
        assert result["username"] == "alice"
        assert result["email"] == "alice@example.com"
        assert result["groups"] == ["g1"]
        assert result["id_token"] == "id.token.value"
        assert result["session_id"] == sid

    @pytest.mark.asyncio
    async def test_resolve_returns_none_for_empty_session_id(self, fake_collection):
        assert await session_store.resolve_session("") is None
        assert await session_store.resolve_session(None) is None  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_resolve_returns_none_for_missing_record(self, fake_collection):
        assert await session_store.resolve_session("nonexistent") is None

    @pytest.mark.asyncio
    async def test_resolve_returns_none_for_expired_record(self, fake_collection):
        sid = await session_store.create_session(
            username="charlie",
            email=None,
            name=None,
            groups=[],
            provider="keycloak",
            auth_method="oauth2",
            max_age_seconds=60,
        )
        # Force the record's expires_at into the past.
        fake_collection[sid]["expires_at"] = datetime.now(UTC) - timedelta(seconds=1)
        assert await session_store.resolve_session(sid) is None

    @pytest.mark.asyncio
    async def test_resolve_handles_naive_datetime_from_documentdb(self, fake_collection):
        """DocumentDB returns naive datetimes; resolver must coerce to UTC."""
        sid = await session_store.create_session(
            username="dora",
            email=None,
            name=None,
            groups=[],
            provider="keycloak",
            auth_method="oauth2",
            max_age_seconds=60,
        )
        # Replace expires_at with a naive datetime well in the future.
        fake_collection[sid]["expires_at"] = (datetime.now(UTC) + timedelta(hours=1)).replace(
            tzinfo=None
        )
        result = await session_store.resolve_session(sid)
        assert result is not None
        assert result["username"] == "dora"

    @pytest.mark.asyncio
    async def test_resolve_returns_none_when_collection_raises(self, monkeypatch):
        async def boom():
            raise RuntimeError("mongo down")

        monkeypatch.setattr(session_store, "_get_collection", boom)
        assert await session_store.resolve_session("any") is None

    @pytest.mark.asyncio
    async def test_resolve_drops_id_token_silently_when_decryption_fails(
        self, fake_collection, monkeypatch
    ):
        sid = await session_store.create_session(
            username="emi",
            email=None,
            name=None,
            groups=[],
            provider="keycloak",
            auth_method="oauth2",
            max_age_seconds=60,
            id_token="originally.valid.token",
        )
        # Corrupt the stored ciphertext so decryption fails.
        fake_collection[sid]["id_token_encrypted"] = b"\x00" * 32

        result = await session_store.resolve_session(sid)
        assert result is not None
        # User identity still resolves; id_token is just absent.
        assert result["username"] == "emi"
        assert "id_token" not in result


class TestDeleteSession:
    @pytest.mark.asyncio
    async def test_delete_existing_returns_true(self, fake_collection):
        sid = await session_store.create_session(
            username="alice",
            email=None,
            name=None,
            groups=[],
            provider="entra",
            auth_method="oauth2",
            max_age_seconds=60,
        )
        assert await session_store.delete_session(sid) is True
        assert sid not in fake_collection

    @pytest.mark.asyncio
    async def test_delete_missing_returns_false(self, fake_collection):
        assert await session_store.delete_session("not-a-real-id") is False

    @pytest.mark.asyncio
    async def test_delete_empty_returns_false_without_db_call(self, monkeypatch):
        called = False

        async def boom():
            nonlocal called
            called = True
            raise AssertionError("should not be called")

        monkeypatch.setattr(session_store, "_get_collection", boom)
        assert await session_store.delete_session("") is False
        assert called is False

    @pytest.mark.asyncio
    async def test_delete_returns_false_when_collection_raises(self, monkeypatch):
        async def boom():
            raise RuntimeError("mongo down")

        monkeypatch.setattr(session_store, "_get_collection", boom)
        assert await session_store.delete_session("any") is False
