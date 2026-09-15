"""Unit tests for the stats repository counter document (issue #1753).

The auth-server flushes its per-auth-path tally into the shared ``mcp_stats``
counters document and the registry reads it back for the telemetry heartbeat.
Two properties carry the weight here:

* The window reset is one **conditional** ``update_one`` per window. The
  ``<window>_reset_at`` predicate in the filter *is* the concurrency control, so
  a ``$set`` of zeros can no longer land between another writer's read and its
  ``$inc``. That was a live lost-update bug on ``semantic_search_ctr`` before
  this change.
* ``increment_auth_path_counters`` propagates write failures while
  ``increment_search_counter`` stays fail-silent. The asymmetry is deliberate:
  the auth-path write is what ``mcpgw_registry_auth_path_flush_total`` reports
  on, and a caller that cannot tell success from failure cannot report health.

The fake collection below models MongoDB's document-level atomicity: the filter
match and the modification happen in one un-awaited step, and the only
interleaving point is *between* operations.
"""

import asyncio
import contextvars
import copy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from registry.repositories.stats_repository import (
    _reset_stale_windows,
    get_auth_path_counts,
    increment_auth_path_counters,
    increment_search_counter,
)

CLIENT_TARGET = "registry.repositories.documentdb.client.get_documentdb_client"


def _get_path(doc: dict, path: str):
    """Read a dotted Mongo field path, returning None when any level is absent."""
    node = doc
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _set_path(doc: dict, path: str, value) -> None:
    """Write a dotted Mongo field path, creating intermediate documents."""
    parts = path.split(".")
    node = doc
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


class FakeCollection:
    """In-memory stand-in for the ``mcp_stats`` collection.

    ``update_one`` awaits exactly once on entry and then matches and modifies
    without awaiting again. That is the guarantee real MongoDB gives for a
    single-document update, and it is what makes the conditional-reset shape
    safe: concurrent callers can only be interleaved between operations.
    """

    def __init__(self, doc: dict | None = None, fail_on_update: Exception | None = None):
        self.docs: dict = {}
        if doc is not None:
            self.docs[doc["_id"]] = copy.deepcopy(doc)
        self.fail_on_update = fail_on_update
        self.calls: list[SimpleNamespace] = []

    @property
    def doc(self) -> dict:
        return self.docs["counters"]

    def _match(self, filt: dict) -> dict | None:
        doc = self.docs.get(filt["_id"])
        if doc is None:
            return None
        for key, condition in filt.items():
            if key == "_id":
                continue
            actual = _get_path(doc, key)
            if isinstance(condition, dict):
                if set(condition) != {"$lt"}:
                    raise NotImplementedError(f"unsupported operator: {condition}")
                # A missing or null field never satisfies a date $lt in MongoDB,
                # so a half-written document is left alone rather than zeroed.
                if actual is None or not actual < condition["$lt"]:
                    return None
            elif actual != condition:
                return None
        return doc

    async def find_one(self, filt: dict) -> dict | None:
        await asyncio.sleep(0)
        doc = self._match(filt)
        return copy.deepcopy(doc) if doc is not None else None

    async def update_one(self, filt: dict, update: dict, upsert: bool = False):
        await asyncio.sleep(0)
        if self.fail_on_update is not None:
            raise self.fail_on_update

        doc = self._match(filt)
        if doc is None and upsert:
            doc = {"_id": filt["_id"]}
            for path, value in update.get("$setOnInsert", {}).items():
                _set_path(doc, path, copy.deepcopy(value))
            self.docs[doc["_id"]] = doc
            self._record(filt, update, upsert, matched=False)
            return SimpleNamespace(matched_count=0, modified_count=0, upserted_id=doc["_id"])

        self._record(filt, update, upsert, matched=doc is not None)
        if doc is None:
            return SimpleNamespace(matched_count=0, modified_count=0, upserted_id=None)

        for path, value in update.get("$set", {}).items():
            _set_path(doc, path, copy.deepcopy(value))
        for path, delta in update.get("$inc", {}).items():
            _set_path(doc, path, (_get_path(doc, path) or 0) + delta)
        return SimpleNamespace(matched_count=1, modified_count=1, upserted_id=None)

    def _record(self, filt: dict, update: dict, upsert: bool, matched: bool) -> None:
        self.calls.append(
            SimpleNamespace(
                filter=copy.deepcopy(filt),
                update=copy.deepcopy(update),
                upsert=upsert,
                matched=matched,
            )
        )

    def resets(self, window: str) -> list[SimpleNamespace]:
        """Calls whose filter carries the conditional predicate for ``window``."""
        return [c for c in self.calls if f"{window}_reset_at" in c.filter]

    def increments(self) -> list[dict]:
        return [c.update["$inc"] for c in self.calls if "$inc" in c.update]


class FakeDB:
    """Database handle that hands out the same collection for any name."""

    def __init__(self, collection: FakeCollection):
        self._collection = collection

    def __getitem__(self, _name: str) -> FakeCollection:
        return self._collection


class PausingView:
    """Collection view that freezes its caller after ``pause_after`` operations.

    Suspending one writer part-way through its sequence and letting the other
    run to completion is how a lost update is reached: two writers that both
    decided the window is stale, with an ``$inc`` landing between them.
    """

    def __init__(self, collection: FakeCollection, pause_after: int, resume: asyncio.Event):
        self._collection = collection
        self._pause_after = pause_after
        self._resume = resume
        self.ops = 0

    async def find_one(self, filt: dict):
        return await self._call(self._collection.find_one(filt))

    async def update_one(self, filt: dict, update: dict, upsert: bool = False):
        return await self._call(self._collection.update_one(filt, update, upsert=upsert))

    async def _call(self, awaitable):
        self.ops += 1
        result = await awaitable
        if self.ops == self._pause_after:
            await self._resume.wait()
        return result


def _counters_doc(
    now: datetime,
    hourly_age: timedelta = timedelta(0),
    daily_age: timedelta = timedelta(0),
    search: int = 0,
    auth_path: dict | None = None,
) -> dict:
    return {
        "_id": "counters",
        "hourly": {"semantic_search_ctr": search},
        "daily": {
            "semantic_search_ctr": search,
            "auth_path": dict(auth_path or {}),
        },
        "forever": {"semantic_search_ctr": search},
        "hourly_reset_at": now - hourly_age,
        "daily_reset_at": now - daily_age,
    }


def _patch_client(collection: FakeCollection):
    return patch(CLIENT_TARGET, new_callable=AsyncMock, return_value=FakeDB(collection))


@pytest.mark.unit
class TestWindowReset:
    """Tests for _reset_stale_windows(): the filter is the concurrency control."""

    @pytest.mark.asyncio
    async def test_reset_is_one_conditional_update_per_window(self):
        """Exactly two conditional writes, with the staleness predicate in the filter.

        Asserting the filter shape and not just the ``$set`` payload is the
        point: the read-modify-write this replaced produced an identical
        ``$set`` and still lost increments.
        """
        now = datetime.now(UTC)
        collection = FakeCollection(_counters_doc(now))

        await _reset_stale_windows(collection, now)

        assert len(collection.calls) == 2
        assert collection.calls[0].filter == {
            "_id": "counters",
            "hourly_reset_at": {"$lt": now - timedelta(hours=1)},
        }
        assert collection.calls[1].filter == {
            "_id": "counters",
            "daily_reset_at": {"$lt": now - timedelta(hours=24)},
        }
        # Nothing was stale, so neither conditional write matched.
        assert [c.matched for c in collection.calls] == [False, False]

    @pytest.mark.asyncio
    async def test_daily_reset_zeroes_search_and_auth_path_in_one_set(self):
        """Both counter families zero in the same ``$set``.

        Two separate writes could be interrupted between them, leaving the two
        windows rolled to different boundaries and a mix computed against a
        denominator from the previous day.
        """
        now = datetime.now(UTC)
        collection = FakeCollection(
            _counters_doc(now, daily_age=timedelta(hours=25), search=9, auth_path={"jwt": 4})
        )

        await _reset_stale_windows(collection, now)

        daily_calls = collection.resets("daily")
        assert len(daily_calls) == 1
        assert daily_calls[0].matched is True
        assert daily_calls[0].update == {
            "$set": {
                "daily.semantic_search_ctr": 0,
                "daily.auth_path": {},
                "daily_reset_at": now,
            }
        }
        assert collection.doc["daily"] == {"semantic_search_ctr": 0, "auth_path": {}}
        assert collection.doc["daily_reset_at"] == now

    @pytest.mark.asyncio
    async def test_stale_hourly_reset_leaves_daily_and_forever(self):
        """A stale hourly window rolls alone; daily and forever keep their totals."""
        now = datetime.now(UTC)
        collection = FakeCollection(
            _counters_doc(now, hourly_age=timedelta(hours=2), search=9, auth_path={"jwt": 4})
        )

        with _patch_client(collection):
            await increment_search_counter()

        assert collection.doc["hourly"]["semantic_search_ctr"] == 1
        assert collection.doc["daily"]["semantic_search_ctr"] == 10
        assert collection.doc["forever"]["semantic_search_ctr"] == 10
        assert collection.doc["daily"]["auth_path"] == {"jwt": 4}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("freeze_flush_after_op", [1, 2, 3, 4])
    async def test_concurrent_reset_and_increment_loses_nothing(self, freeze_flush_after_op):
        """A search increment and an auth-path flush racing the same stale daily
        boundary must both survive, wherever the flush is suspended.

        Both writers roll the window before adding, so both contend for the same
        24-hour boundary. The flush is frozen after each of its operations in
        turn while the search writer runs to completion, which enumerates the
        interleavings that matter: with the conditional shape exactly one
        ``update_one`` matches ``daily_reset_at: {$lt: cutoff}``, it matches and
        applies as one atomic document operation, and the loser's filter no
        longer matches -- so no ``$set`` of zeros can land after an ``$inc``.

        Why this fails against the ``find_one`` -> compare -> ``$set``
        implementation it replaced: freezing the flush after its second
        operation puts the pause between its read and its write. Both writers
        have then read a stale document and both have decided to roll. The
        search writer writes its zeros, increments, and finishes; the flush
        resumes and issues its own unconditional ``$set`` of zeros, which erases
        the increment that landed in between. The old code had no way to notice
        the window had already rolled, because staleness was evaluated in the
        application from a value read one round trip earlier instead of in the
        filter at write time. Verified against a transplanted copy of the old
        implementation: it loses the search counter at this pause point.
        """
        now = datetime.now(UTC)
        collection = FakeCollection(_counters_doc(now, daily_age=timedelta(hours=25), search=3))
        search_done = asyncio.Event()
        flush_view = PausingView(collection, freeze_flush_after_op, search_done)
        writer: contextvars.ContextVar[str] = contextvars.ContextVar("writer")

        async def get_client():
            return FakeDB(flush_view if writer.get() == "flush" else collection)

        async def search():
            writer.set("search")
            try:
                await increment_search_counter()
            finally:
                search_done.set()

        async def flush():
            writer.set("flush")
            await increment_auth_path_counters({"session_cookie": 5, "keycloak": 2})

        with patch(CLIENT_TARGET, new=get_client):
            await asyncio.gather(search(), flush())

        # Both writers' contributions survive: the roll happened once, before
        # either increment, instead of one writer's zeros erasing the other's.
        assert collection.doc["daily"]["semantic_search_ctr"] == 1
        assert collection.doc["daily"]["auth_path"] == {"session_cookie": 5, "keycloak": 2}
        # The untouched windows still carry their pre-existing totals plus the search.
        assert collection.doc["forever"]["semantic_search_ctr"] == 4
        assert collection.doc["hourly"]["semantic_search_ctr"] == 4
        # Exactly one conditional write won the boundary.
        assert [c.matched for c in collection.resets("daily")].count(True) == 1


@pytest.mark.unit
class TestIncrementAuthPathCounters:
    """Tests for the auth-path flush write."""

    @pytest.mark.asyncio
    async def test_increment_auth_path_propagates_write_failure(self):
        """A failed write raises, so the flush can record outcome="error".

        Deliberately unlike increment_search_counter: this write is what the
        flush-health counter reports on, and a silently failing write looks
        exactly like no traffic.
        """
        collection = FakeCollection(fail_on_update=RuntimeError("no primary available"))

        with _patch_client(collection), pytest.raises(RuntimeError, match="no primary"):
            await increment_auth_path_counters({"session_cookie": 3})

    @pytest.mark.asyncio
    async def test_increment_auth_path_propagates_connection_failure(self):
        """A failure reaching the database propagates too, not only a failed write."""
        with (
            patch(CLIENT_TARGET, new_callable=AsyncMock, side_effect=OSError("connection refused")),
            pytest.raises(OSError, match="connection refused"),
        ):
            await increment_auth_path_counters({"session_cookie": 3})

    @pytest.mark.asyncio
    async def test_increment_search_counter_still_fail_silent(self):
        """The neighbouring contract is unchanged: search increments never raise."""
        collection = FakeCollection(fail_on_update=RuntimeError("no primary available"))

        with _patch_client(collection):
            await increment_search_counter()  # must not raise

        with patch(CLIENT_TARGET, new_callable=AsyncMock, side_effect=OSError("refused")):
            await increment_search_counter()  # must not raise

    @pytest.mark.asyncio
    async def test_increment_auth_path_rejects_dotted_and_dollar_keys(self):
        """Paths arrive as metric labels, so a "." or "$" must not reach a field path.

        ``{"a.b": 1}`` as a ``$inc`` key addresses ``daily.auth_path.a.b`` --
        a field path of the sender's choosing rather than a counter.
        """
        now = datetime.now(UTC)
        collection = FakeCollection(_counters_doc(now))

        with _patch_client(collection):
            await increment_auth_path_counters(
                {"session_cookie": 2, "a.b": 5, "$set": 7, "x.$y": 9}
            )

        assert collection.increments() == [{"daily.auth_path.session_cookie": 2}]
        assert collection.doc["daily"]["auth_path"] == {"session_cookie": 2}

    @pytest.mark.asyncio
    async def test_increment_auth_path_all_keys_rejected_writes_nothing(self):
        """A wholly hostile batch performs no counter write at all."""
        now = datetime.now(UTC)
        collection = FakeCollection(_counters_doc(now))

        with _patch_client(collection):
            await increment_auth_path_counters({"a.b": 5, "$inc": 7})

        assert collection.increments() == []
        assert collection.doc["daily"]["auth_path"] == {}

    @pytest.mark.asyncio
    async def test_increment_auth_path_noop_on_empty(self):
        """An empty tally touches the database not at all."""
        client = AsyncMock()
        with patch(CLIENT_TARGET, new=client):
            await increment_auth_path_counters({})

        client.assert_not_awaited()


@pytest.mark.unit
class TestGetAuthPathCounts:
    """Tests for the heartbeat's read side."""

    @pytest.mark.asyncio
    async def test_get_auth_path_counts_fail_silent_returns_zeros(self):
        """A failed read returns empty counts so the heartbeat still goes out.

        Downstream, empty counts render all three fields as null, which is the
        honest reading: nothing was measured.
        """
        with patch(CLIENT_TARGET, new_callable=AsyncMock, side_effect=OSError("refused")):
            result = await get_auth_path_counts()

        assert result == {"counts": {}, "window_hours": 0.0}

    @pytest.mark.asyncio
    async def test_absent_subdocument_reads_as_empty(self):
        """A pre-existing document without daily.auth_path needs no migration.

        It reads as empty counts, and the first flush grows the subdocument
        without disturbing the search counter beside it.
        """
        now = datetime.now(UTC)
        legacy = {
            "_id": "counters",
            "hourly": {"semantic_search_ctr": 2},
            "daily": {"semantic_search_ctr": 5},
            "forever": {"semantic_search_ctr": 11},
            "hourly_reset_at": now,
            "daily_reset_at": now,
        }
        collection = FakeCollection(legacy)

        with _patch_client(collection):
            assert await get_auth_path_counts() == {"counts": {}, "window_hours": 0.0}
            await increment_auth_path_counters({"keycloak": 3})

        assert collection.doc["daily"]["auth_path"] == {"keycloak": 3}
        assert collection.doc["daily"]["semantic_search_ctr"] == 5

    @pytest.mark.asyncio
    async def test_missing_document_reads_as_empty(self):
        """No counters document yet also reads as unmeasured, not as an error."""
        collection = FakeCollection()

        with _patch_client(collection):
            assert await get_auth_path_counts() == {"counts": {}, "window_hours": 0.0}

    @pytest.mark.asyncio
    async def test_window_hours_measured_from_daily_reset(self):
        """The reported window is the real age of the daily window, in hours."""
        now = datetime.now(UTC)
        collection = FakeCollection(
            _counters_doc(now, daily_age=timedelta(minutes=40), auth_path={"session_cookie": 4})
        )

        with _patch_client(collection):
            result = await get_auth_path_counts()

        assert result["counts"] == {"session_cookie": 4}
        assert 0.6 < result["window_hours"] < 0.75

    @pytest.mark.asyncio
    async def test_naive_reset_timestamp_read_as_utc(self):
        """MongoDB hands back naive datetimes; comparing them must not blow up.

        A TypeError here is swallowed by the fail-silent wrapper, so the mix
        would go permanently missing while every other signal looked healthy.
        """
        now = datetime.now(UTC)
        doc = _counters_doc(now, daily_age=timedelta(hours=6), auth_path={"boto3": 2})
        doc["daily_reset_at"] = doc["daily_reset_at"].replace(tzinfo=None)
        collection = FakeCollection(doc)

        with _patch_client(collection):
            result = await get_auth_path_counts()

        assert result["counts"] == {"boto3": 2}
        assert 5.9 < result["window_hours"] < 6.1
