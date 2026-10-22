"""Unit tests for registry.services.patch_key_service (wire-platform-v1 2.1).

The Motor collection is mocked, so the service logic (hash storage, one-time
plaintext, active-filter query shape, revoke irreversibility, quota) is
exercised without a live MongoDB.
"""

import logging
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from registry.schemas.patch_key import (
    PATCH_KEY_STATUS_ACTIVE,
    PATCH_KEY_STATUS_REVOKED,
    PatchKeyCreated,
)
from registry.services.patch_key_service import (
    PATCH_KEY_ENTROPY_BYTES,
    PATCH_KEY_PREFIX,
    PatchKeyAlreadyRevoked,
    PatchKeyNotFound,
    PatchKeyQuotaExceeded,
    PatchKeyService,
    generate_patch_key,
    hash_patch_key,
)

logger = logging.getLogger(__name__)


def _make_collection_mock() -> MagicMock:
    """Return a MagicMock that mimics an AsyncIOMotorCollection."""
    collection = MagicMock()
    collection.insert_one = AsyncMock()
    collection.find_one = AsyncMock()
    collection.update_one = AsyncMock()
    collection.count_documents = AsyncMock(return_value=0)
    collection.create_index = AsyncMock()
    cursor = MagicMock()
    cursor.sort = MagicMock(return_value=cursor)
    cursor.to_list = AsyncMock(return_value=[])
    collection.find = MagicMock(return_value=cursor)
    return collection


@pytest.fixture
def mock_collection() -> MagicMock:
    return _make_collection_mock()


@pytest.fixture
def mock_db(mock_collection: MagicMock) -> MagicMock:
    db = MagicMock()
    db.__getitem__ = MagicMock(return_value=mock_collection)
    return db


@pytest.fixture
def service(mock_db: MagicMock) -> PatchKeyService:
    return PatchKeyService(mock_db)


@pytest.mark.unit
class TestKeyShape:
    """Plaintext key shape and hashing."""

    def test_prefix_and_entropy(self):
        key = generate_patch_key()
        assert key.startswith(PATCH_KEY_PREFIX)
        # token_urlsafe(32) -> 43 chars after the prefix; >= 32 bytes entropy.
        assert len(key) == len(PATCH_KEY_PREFIX) + 43
        assert PATCH_KEY_ENTROPY_BYTES >= 32

    def test_keys_are_unique(self):
        assert len({generate_patch_key() for _ in range(100)}) == 100

    def test_hash_is_sha256_hex(self):
        key = generate_patch_key()
        digest = hash_patch_key(key)
        assert len(digest) == 64
        int(digest, 16)  # parses as hex
        import hashlib

        assert digest == hashlib.sha256(key.encode()).hexdigest()


@pytest.mark.unit
class TestMintKey:
    """Mint behavior: quota, storage shape, one-time plaintext."""

    @pytest.mark.asyncio
    async def test_mint_stores_hash_not_plaintext(self, service, mock_collection):
        created = await service.mint_key(username="alice", groups=["devs"], name="ci-runner")

        assert isinstance(created, PatchKeyCreated)
        assert created.key.startswith(PATCH_KEY_PREFIX)
        stored = mock_collection.insert_one.call_args[0][0]
        assert stored["key_hash"] == hash_patch_key(created.key)
        assert created.key not in stored.values()
        assert created.key not in str(stored)
        assert stored["status"] == PATCH_KEY_STATUS_ACTIVE
        assert stored["username"] == "alice"
        assert stored["groups"] == ["devs"]
        assert stored["last_used_at"] is None
        assert stored["revoked_at"] is None
        # display prefix is short and does not reveal the key body
        assert stored["key_prefix"] == created.key[:12]
        assert len(stored["key_prefix"]) < len(created.key) - 8

    @pytest.mark.asyncio
    async def test_mint_enforces_active_quota(self, service, mock_collection):
        mock_collection.count_documents.return_value = 20
        with pytest.raises(PatchKeyQuotaExceeded):
            await service.mint_key(username="alice", groups=[], name="one-too-many")
        mock_collection.insert_one.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_quota_counts_only_active(self, service, mock_collection):
        # count_documents is queried with the active-status filter, so revoked
        # keys do not consume quota.
        mock_collection.count_documents.return_value = 19
        await service.mint_key(username="alice", groups=[], name="ok")
        query = mock_collection.count_documents.call_args[0][0]
        assert query == {"username": "alice", "status": PATCH_KEY_STATUS_ACTIVE}


@pytest.mark.unit
class TestVerifyKey:
    """Verification: active filter inside the query, fail-closed on DB error."""

    @pytest.mark.asyncio
    async def test_verify_hits_db_with_hash_and_active_filter(self, service, mock_collection):
        doc = {
            "key_id": "k1",
            "key_hash": "h" * 64,
            "key_prefix": "wgk-abc",
            "name": "ci",
            "username": "alice",
            "email": "a@example.com",
            "groups": ["devs"],
            "provider": "oauth2",
            "status": PATCH_KEY_STATUS_ACTIVE,
            "created_at": datetime.utcnow(),
            "last_used_at": None,
            "revoked_at": None,
        }
        mock_collection.find_one.return_value = doc
        info = await service.verify_key("wgk-real-key-value")
        query = mock_collection.find_one.call_args[0][0]
        assert query["key_hash"] == hash_patch_key("wgk-real-key-value")
        assert query["status"] == PATCH_KEY_STATUS_ACTIVE
        assert info is not None
        assert info.key_id == "k1"
        assert info.username == "alice"
        assert info.groups == ["devs"]

    @pytest.mark.asyncio
    async def test_verify_miss_returns_none(self, service, mock_collection):
        mock_collection.find_one.return_value = None
        assert await service.verify_key("wgk-nope") is None

    @pytest.mark.asyncio
    async def test_verify_db_error_fails_closed(self, service, mock_collection):
        mock_collection.find_one.side_effect = RuntimeError("db down")
        assert await service.verify_key("wgk-anything") is None

    @pytest.mark.asyncio
    async def test_touch_last_used_best_effort(self, service, mock_collection):
        mock_collection.update_one.side_effect = RuntimeError("db down")
        # must not raise
        await service.touch_last_used("k1")


@pytest.mark.unit
class TestRevokeKey:
    """Revocation: ownership-scoped, one-way, immediate."""

    def _active_doc(self, username: str = "alice") -> dict:
        return {
            "key_id": "k1",
            "key_hash": "h" * 64,
            "key_prefix": "wgk-abc",
            "name": "ci",
            "username": username,
            "groups": ["devs"],
            "status": PATCH_KEY_STATUS_ACTIVE,
            "created_at": datetime.utcnow(),
        }

    @pytest.mark.asyncio
    async def test_revoke_updates_status_via_filtered_update(self, service, mock_collection):
        doc = self._active_doc()
        mock_collection.find_one.side_effect = [
            doc,
            {**doc, "status": PATCH_KEY_STATUS_REVOKED, "revoked_at": datetime.utcnow()},
        ]
        info = await service.revoke_key(key_id="k1", username="alice")
        assert info.status == PATCH_KEY_STATUS_REVOKED
        assert info.revoked_at is not None
        update_args = mock_collection.update_one.call_args[0]
        # the update itself re-checks active status -> a concurrent double
        # revoke cannot resurrect anything
        assert update_args[0]["status"] == PATCH_KEY_STATUS_ACTIVE
        assert update_args[1]["$set"]["status"] == PATCH_KEY_STATUS_REVOKED

    @pytest.mark.asyncio
    async def test_revoke_other_users_key_is_not_found(self, service, mock_collection):
        # ownership filter is part of the lookup query
        mock_collection.find_one.return_value = None
        with pytest.raises(PatchKeyNotFound):
            await service.revoke_key(key_id="k1", username="mallory")
        query = mock_collection.find_one.call_args[0][0]
        assert query == {"key_id": "k1", "username": "mallory"}

    @pytest.mark.asyncio
    async def test_revoke_already_revoked_conflicts(self, service, mock_collection):
        mock_collection.find_one.return_value = {
            **self._active_doc(),
            "status": PATCH_KEY_STATUS_REVOKED,
        }
        with pytest.raises(PatchKeyAlreadyRevoked):
            await service.revoke_key(key_id="k1", username="alice")
        mock_collection.update_one.assert_not_awaited()


@pytest.mark.unit
class TestListKeys:
    """Listing is scoped to the calling user and drops the hash."""

    @pytest.mark.asyncio
    async def test_list_scopes_to_user_and_strips_hash(self, service, mock_collection):
        doc = {
            "key_id": "k1",
            "key_hash": "secret-hash-value",
            "key_prefix": "wgk-abc",
            "name": "ci",
            "username": "alice",
            "groups": [],
            "status": PATCH_KEY_STATUS_ACTIVE,
            "created_at": datetime.utcnow(),
        }
        mock_collection.find.return_value.to_list.return_value = [doc]
        items = await service.list_keys_for_user("alice")
        query = mock_collection.find.call_args[0][0]
        assert query == {"username": "alice"}
        assert len(items) == 1
        assert "secret-hash-value" not in items[0].model_dump()
