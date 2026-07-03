"""Logic tests for EgressOperationalRepository (replay guard + refresh lease).

Uses an in-memory fake collection that mimics the Mongo methods the repo calls
(insert_one with unique _id, find_one_and_update upsert, delete_one). The real
Mongo path is covered by integration; here we assert the single-use and
lease-ownership semantics.
"""

from datetime import UTC, datetime, timedelta

import pytest
from pymongo.errors import DuplicateKeyError

from registry.repositories.documentdb.egress_operational_repository import (
    EgressOperationalRepository,
)


class _FakeCollection:
    def __init__(self):
        self._docs: dict[str, dict] = {}

    async def insert_one(self, doc):
        _id = doc["_id"]
        if _id in self._docs:
            raise DuplicateKeyError("dup")
        self._docs[_id] = dict(doc)

    async def find_one_and_delete(self, filt):
        _id = filt["_id"]
        return self._docs.pop(_id, None)

    async def find_one_and_update(self, filt, update, upsert=False, return_document=None):
        _id = filt["_id"]
        doc = self._docs.get(_id)
        # Evaluate the $or free/expired condition.
        now = datetime.now(UTC)
        free_or_expired = True
        if doc is not None:
            holder_absent = "holder" not in doc
            expired = doc.get("expires_at_dt") is not None and doc["expires_at_dt"] < now
            free_or_expired = holder_absent or expired
        if doc is None and not upsert:
            return None
        if not free_or_expired:
            return doc  # lease held by someone else; AFTER returns existing holder
        merged = dict(doc or {"_id": _id})
        merged.update(update["$set"])
        self._docs[_id] = merged
        return merged

    async def delete_one(self, filt):
        _id = filt["_id"]
        doc = self._docs.get(_id)
        if doc and all(doc.get(k) == v for k, v in filt.items() if k != "_id"):
            del self._docs[_id]

    async def create_index(self, *a, **k):
        return "idx"


@pytest.fixture
def repo(monkeypatch):
    r = EgressOperationalRepository()
    col = _FakeCollection()

    async def _get_col():
        return col

    monkeypatch.setattr(r, "_get_collection", _get_col)
    r._collection = col
    r._indexes_created = True
    return r


@pytest.mark.unit
class TestReplayGuard:
    async def test_first_use_succeeds_replay_fails(self, repo):
        assert await repo.consume_nonce("n1", 600) is True
        assert await repo.consume_nonce("n1", 600) is False  # replay

    async def test_distinct_nonces_independent(self, repo):
        assert await repo.consume_nonce("a", 600) is True
        assert await repo.consume_nonce("b", 600) is True


@pytest.mark.unit
class TestRefreshLease:
    async def test_acquire_then_second_holder_blocked(self, repo):
        key = "oauth2|alice|github|/github-mcp"
        assert await repo.acquire_lease(key, "holder-1", 30) is True
        # a different holder cannot acquire while held + unexpired
        assert await repo.acquire_lease(key, "holder-2", 30) is False

    async def test_release_lets_next_acquire(self, repo):
        key = "k"
        assert await repo.acquire_lease(key, "h1", 30) is True
        await repo.release_lease(key, "h1")
        assert await repo.acquire_lease(key, "h2", 30) is True

    async def test_release_by_non_holder_is_noop(self, repo):
        key = "k"
        await repo.acquire_lease(key, "h1", 30)
        await repo.release_lease(key, "h2")  # wrong holder: must not release
        assert await repo.acquire_lease(key, "h3", 30) is False  # still held by h1

    async def test_expired_lease_reacquirable(self, repo, monkeypatch):
        key = "k"
        assert await repo.acquire_lease(key, "h1", 30) is True
        # force the stored lease into the past
        repo._collection._docs[f"lease:{key}"]["expires_at_dt"] = datetime.now(UTC) - timedelta(
            seconds=1
        )
        assert await repo.acquire_lease(key, "h2", 30) is True  # h1's lease expired
