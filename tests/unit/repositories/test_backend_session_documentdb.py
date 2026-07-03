"""Unit tests for DocumentDBBackendSessionRepository.validate_client_session.

The Mongo collection is mocked (no live DB) so these tests focus on the
repository's own logic: that client-session validation binds the lookup to the
authenticated owner so a guessed/stolen Mcp-Session-Id for another user is
rejected (session-hijacking fix). The owner match must happen inside the same
atomic find_one_and_update that refreshes the TTL, so there is no check/use gap.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pymongo.errors import DuplicateKeyError

from registry.repositories.documentdb.backend_session_repository import (
    DocumentDBBackendSessionRepository,
)


@pytest.fixture
def collection() -> MagicMock:
    coll = MagicMock()
    coll.find_one_and_update = AsyncMock()
    coll.replace_one = AsyncMock()
    coll.delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))
    return coll


@pytest.fixture
def repo(collection):
    r = DocumentDBBackendSessionRepository()
    with patch.object(r, "_get_collection", new=AsyncMock(return_value=collection)):
        yield r


@pytest.mark.unit
class TestValidateClientSessionOwnership:
    """validate_client_session must enforce the stored owner when user_id given."""

    async def test_owner_match_includes_user_id_in_query(self, repo, collection):
        """When user_id is supplied, the lookup filter pins both _id and user_id."""
        collection.find_one_and_update.return_value = {"_id": "client:vs-abc", "user_id": "alice"}

        result = await repo.validate_client_session("vs-abc", user_id="alice")

        assert result is True
        query = collection.find_one_and_update.call_args[0][0]
        assert query["_id"] == "client:vs-abc"
        assert query["user_id"] == "alice"

    async def test_wrong_owner_returns_false(self, repo, collection):
        """A session owned by another user is treated as not found.

        The atomic filtered query returns None because the document's stored
        user_id does not match the attacker's identity, so the attacker cannot
        hijack the victim's session even with the correct session ID.
        """
        collection.find_one_and_update.return_value = None

        result = await repo.validate_client_session("vs-victim", user_id="attacker")

        assert result is False
        query = collection.find_one_and_update.call_args[0][0]
        assert query["user_id"] == "attacker"

    async def test_owner_match_is_atomic_with_ttl_bump(self, repo, collection):
        """Owner check and last_used_at refresh happen in one find_one_and_update."""
        collection.find_one_and_update.return_value = {"_id": "client:vs-abc"}

        await repo.validate_client_session("vs-abc", user_id="alice")

        # Exactly one DB round-trip; the $set refreshes the TTL field.
        collection.find_one_and_update.assert_awaited_once()
        update = collection.find_one_and_update.call_args[0][1]
        assert "last_used_at" in update["$set"]

    async def test_no_user_id_does_not_filter_on_owner(self, repo, collection):
        """Legacy behavior: omitting user_id validates by existence only."""
        collection.find_one_and_update.return_value = {"_id": "client:vs-abc"}

        result = await repo.validate_client_session("vs-abc")

        assert result is True
        query = collection.find_one_and_update.call_args[0][0]
        assert "user_id" not in query

    async def test_virtual_server_path_added_to_filter(self, repo, collection):
        """When virtual_server_path is supplied, the lookup filters on it too.

        Binds the session to the virtual server it was minted for, so a session
        for /virtual/a cannot be replayed against /virtual/b (issue #2).
        """
        collection.find_one_and_update.return_value = {"_id": "client:vs-abc"}

        result = await repo.validate_client_session(
            "vs-abc", user_id="alice", virtual_server_path="/virtual/a"
        )

        assert result is True
        query = collection.find_one_and_update.call_args[0][0]
        assert query["virtual_server_path"] == "/virtual/a"

    async def test_wrong_virtual_server_path_returns_false(self, repo, collection):
        """A session minted for a different virtual server is rejected."""
        collection.find_one_and_update.return_value = None

        result = await repo.validate_client_session(
            "vs-abc", user_id="alice", virtual_server_path="/virtual/b"
        )

        assert result is False
        query = collection.find_one_and_update.call_args[0][0]
        assert query["virtual_server_path"] == "/virtual/b"


@pytest.mark.unit
class TestGetBackendSessionOwnership:
    """get_backend_session must enforce the stored owner when user_id given.

    Defense in depth: even though every in-router path to a backend session
    already passes the owner-bound client-session gate, the backend-session
    read is bound to the owner here too, so a single missed gate (or a future
    handler reaching the backend lookup early) cannot leak another user's live
    backend session ID.
    """

    async def test_owner_match_includes_user_id_in_query(self, repo, collection):
        collection.find_one_and_update.return_value = {
            "_id": "vs-abc:/_vs_backend_x_",
            "backend_session_id": "be-1",
        }

        result = await repo.get_backend_session("vs-abc", "/_vs_backend_x_", user_id="alice")

        assert result == "be-1"
        query = collection.find_one_and_update.call_args[0][0]
        assert query["_id"] == "vs-abc:/_vs_backend_x_"
        assert query["user_id"] == "alice"

    async def test_wrong_owner_returns_none(self, repo, collection):
        """A backend session owned by another user is treated as not found."""
        collection.find_one_and_update.return_value = None

        result = await repo.get_backend_session("vs-victim", "/_vs_backend_x_", user_id="attacker")

        assert result is None
        query = collection.find_one_and_update.call_args[0][0]
        assert query["user_id"] == "attacker"

    async def test_no_user_id_does_not_filter_on_owner(self, repo, collection):
        """Legacy behavior: omitting user_id looks up by compound key only."""
        collection.find_one_and_update.return_value = {
            "_id": "vs-abc:/_vs_backend_x_",
            "backend_session_id": "be-1",
        }

        result = await repo.get_backend_session("vs-abc", "/_vs_backend_x_")

        assert result == "be-1"
        query = collection.find_one_and_update.call_args[0][0]
        assert "user_id" not in query


@pytest.mark.unit
class TestStoreBackendSessionOwnership:
    """store_backend_session must not overwrite another user's session document."""

    async def test_upsert_filter_pins_owner(self, repo, collection):
        """The upsert filter includes user_id so it only matches the owner's doc."""
        await repo.store_backend_session(
            client_session_id="vs-abc",
            backend_key="/_vs_backend_x_",
            backend_session_id="be-1",
            user_id="alice",
            virtual_server_path="/virtual/a",
        )

        filt = collection.replace_one.call_args[0][0]
        assert filt["_id"] == "vs-abc:/_vs_backend_x_"
        assert filt["user_id"] == "alice"

    async def test_owner_collision_is_refused_not_raised(self, repo, collection):
        """A duplicate-key from a differently-owned _id is swallowed, not propagated.

        client_session_id is owner-namespaced so this cannot happen on the
        legitimate path, but if it ever did, refusing to overwrite (rather than
        crashing or clobbering) is the safe outcome.
        """
        collection.replace_one.side_effect = DuplicateKeyError("dup _id, other owner")

        # Must not raise.
        await repo.store_backend_session(
            client_session_id="vs-abc",
            backend_key="/_vs_backend_x_",
            backend_session_id="be-1",
            user_id="attacker",
            virtual_server_path="/virtual/a",
        )


@pytest.mark.unit
class TestDeleteBackendSessionOwnership:
    """delete_backend_session must scope deletion to the owner when given."""

    async def test_delete_filter_pins_owner(self, repo, collection):
        await repo.delete_backend_session("vs-abc", "/_vs_backend_x_", user_id="alice")

        filt = collection.delete_one.call_args[0][0]
        assert filt["_id"] == "vs-abc:/_vs_backend_x_"
        assert filt["user_id"] == "alice"

    async def test_delete_without_owner_uses_id_only(self, repo, collection):
        await repo.delete_backend_session("vs-abc", "/_vs_backend_x_")

        filt = collection.delete_one.call_args[0][0]
        assert "user_id" not in filt
