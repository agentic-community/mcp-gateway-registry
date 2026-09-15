"""Auth-path count accumulation and its periodic flush (issue #1753).

The auth-server's own OTel meter is per-process, so the per-path ``/validate``
mix cannot reach the registry's telemetry heartbeat directly. ``auth_path_stats``
carries it through the one channel both deployables share: an in-memory tally on
the hot path, flushed once an interval into the shared ``mcp_stats`` document.

Three properties are load-bearing and each has a distinct failure mode.

``record`` runs inline in the metrics middleware's ``finally``, on the endpoint
nginx calls via ``auth_request``. It must never raise -- an exception there denies
a request -- and it must drop anything outside :data:`KNOWN_AUTH_PATHS`, which is
what bounds both the accumulator and the label cardinality downstream.

``flush_once`` must batch: one database write per interval regardless of request
volume, which is what keeps the write off the hot path. It drains before writing,
so a failed write costs one interval instead of double-counting it into the next.

``flush_once`` is the only fail-silent boundary in the feature, and a flush that
fails quietly looks exactly like a deployment with no traffic. So the failure
must land at WARNING and on ``auth_path_flush_total{outcome="error"}``, carrying
the exception *type* only -- a driver error's message names the connection host
and its whole topology.
"""

import asyncio
import logging
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from auth_server import auth_path_stats as aps

pytestmark = pytest.mark.unit

# flush_once imports this lazily, inside the function, so the auth-server's import
# graph does not drag in the registry's repository stack. Patch it at its source.
REPO_TARGET = "registry.repositories.stats_repository.increment_auth_path_counters"


@pytest.fixture(autouse=True)
def _reset_accumulator():
    """The accumulator is module-level mutable state, and flush_once rebinds it."""
    aps._counts = defaultdict(int)
    yield
    aps._counts = defaultdict(int)


@pytest.fixture
def repo():
    with patch(REPO_TARGET, new_callable=AsyncMock) as mock:
        yield mock


@pytest.fixture
def flush_counter():
    with patch.object(aps, "auth_path_flush_total") as counter:
        yield counter


def _outcomes(flush_counter) -> list[str]:
    return [call[0][1]["outcome"] for call in flush_counter.add.call_args_list]


class TestRecord:
    """One dict increment on the hot path: bounded, and it cannot raise."""

    def test_record_accumulates_known_paths(self):
        for method in ("session_cookie", "session_cookie", "keycloak", "unknown", "session_cookie"):
            aps.record(method)

        assert dict(aps._counts) == {"session_cookie": 3, "keycloak": 1, "unknown": 1}

    def test_record_drops_unknown_path(self):
        """An unrecognized value would mint a key here, in the stats document, and
        in the telemetry payload. `method` arrives from a response header, so the
        closed set is the only thing bounding all three.
        """
        for rejected in (
            "",  # server.py writes X-Auth-Method as `... or ""`
            "   ",
            "SESSION_COOKIE",  # the set is case-sensitive
            "evil\nname",
            "bearer",
            "x" * 500,
        ):
            aps.record(rejected)

        # A known path recorded alongside them still lands, so this is a filter and
        # not a disabled accumulator.
        aps.record("jwt")
        assert dict(aps._counts) == {"jwt": 1}

    @pytest.mark.parametrize("bad", [None, 123, 4.5, True, "x" * 100_000])
    def test_record_never_raises(self, bad):
        """record() runs in the middleware's finally on every /validate request.

        A raise there propagates out of dispatch and denies the request nginx was
        authorizing, so a bad value must be dropped, not surfaced.
        """
        aps.record(bad)

        # `True` hashes equal to 1 and neither is in the set, so nothing is stored.
        assert dict(aps._counts) == {}


class TestFlushOnce:
    """One write per interval, drained before the write, failures visible."""

    @pytest.mark.asyncio
    async def test_flush_batches_one_write_per_interval(self, repo, flush_counter):
        """500 requests, one write. This is what keeps the database off the hot path."""
        for _ in range(300):
            aps.record("session_cookie")
        for _ in range(150):
            aps.record("keycloak")
        for _ in range(50):
            aps.record("unknown")

        await aps.flush_once()

        repo.assert_awaited_once_with({"session_cookie": 300, "keycloak": 150, "unknown": 50})
        assert _outcomes(flush_counter) == ["ok"]

    @pytest.mark.asyncio
    async def test_flush_drains_before_write(self, flush_counter):
        """The swap happens before the await, so a request arriving mid-write lands
        in the next interval instead of being written twice or dropped.
        """
        aps.record("jwt")
        written: dict[str, int] = {}

        async def write(counts):
            written.update(counts)
            # A /validate request completing while the write is in flight.
            aps.record("okta")

        with patch(REPO_TARGET, new=write):
            await aps.flush_once()

        assert written == {"jwt": 1}
        assert dict(aps._counts) == {"okta": 1}
        assert _outcomes(flush_counter) == ["ok"]

    @pytest.mark.asyncio
    async def test_flush_failure_records_error_and_does_not_raise(
        self, repo, flush_counter, caplog
    ):
        """A failing flush must be distinguishable from a deployment with no traffic.

        At DEBUG the two are identical in the logs, which is the state this counter
        and this log level exist to prevent.
        """
        aps.record("jwt")
        repo.side_effect = RuntimeError("connection refused")

        with caplog.at_level(logging.DEBUG, logger=aps.logger.name):
            await aps.flush_once()  # must not propagate: the loop has to survive

        assert _outcomes(flush_counter) == ["error"]
        failures = [r for r in caplog.records if "flush failed" in r.getMessage()]
        assert len(failures) == 1
        assert failures[0].levelno == logging.WARNING

    @pytest.mark.asyncio
    async def test_flush_failure_does_not_double_count(self, flush_counter):
        """A failed write loses its interval rather than folding it into the next.

        For a 24-hour share, losing a minute beats reporting a minute twice.
        """
        aps.record("jwt")
        aps.record("jwt")

        with patch(REPO_TARGET, new_callable=AsyncMock) as failing:
            failing.side_effect = RuntimeError("boom")
            await aps.flush_once()

        assert dict(aps._counts) == {}

        aps.record("okta")
        with patch(REPO_TARGET, new_callable=AsyncMock) as succeeding:
            await aps.flush_once()

        succeeding.assert_awaited_once_with({"okta": 1})
        assert _outcomes(flush_counter) == ["error", "ok"]

    @pytest.mark.asyncio
    async def test_flush_log_omits_exception_message(self, repo, caplog):
        """A pymongo failure's message carries the connection string and topology.

        The handler logs type(e).__name__ for exactly this reason.
        """
        secret = (
            "mongodb://admin:sup3rs3cr3t@docdb.cluster-abc123.us-east-1.docdb.amazonaws.com:27017"
        )
        aps.record("session_cookie")
        repo.side_effect = RuntimeError(
            f"ServerSelectionTimeoutError: {secret}, Topology description: <TopologyDescription>"
        )

        with caplog.at_level(logging.DEBUG, logger=aps.logger.name):
            await aps.flush_once()

        assert secret not in caplog.text
        assert "mongodb://" not in caplog.text
        assert "sup3rs3cr3t" not in caplog.text
        assert "docdb.amazonaws.com" not in caplog.text
        # The type still reaches the operator, which is what makes it diagnosable.
        assert "RuntimeError" in caplog.text

    @pytest.mark.asyncio
    async def test_empty_interval_writes_nothing(self, repo, flush_counter):
        """No traffic is not an outcome: neither a write nor a counter increment.

        An `ok` per idle minute would drown the signal the error rate is read
        against, and a write per minute would be pure churn.
        """
        await aps.flush_once()

        repo.assert_not_awaited()
        flush_counter.add.assert_not_called()


class TestFlushLoop:
    """The loop ticks on the interval and dies on cancellation, not before."""

    @pytest.mark.asyncio
    async def test_flush_loop_ticks_then_stops_on_cancel(self, monkeypatch):
        """CancelledError must propagate: the lifespan awaits this task on shutdown
        and then flushes once more, which is how a graceful stop keeps the tail of
        the interval. A swallowed cancellation hangs shutdown instead.
        """
        delays: list[float] = []
        flushes = 0
        ticked = asyncio.Event()

        async def fake_sleep(delay):
            delays.append(delay)
            await asyncio.sleep(0)  # a real cancellation point, no wall-clock wait

        async def fake_flush():
            nonlocal flushes
            flushes += 1
            ticked.set()

        # Replace only the module's own asyncio reference, so the stub cannot leak
        # into asyncio.wait_for below.
        monkeypatch.setattr(aps, "asyncio", SimpleNamespace(sleep=fake_sleep))
        monkeypatch.setattr(aps, "flush_once", fake_flush)

        task = asyncio.create_task(aps.flush_loop())
        await asyncio.wait_for(ticked.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert flushes >= 1
        # Sleep first, then flush: a fresh process reports its first interval a
        # minute in, never a partial one.
        assert delays[0] == aps._FLUSH_INTERVAL_SECONDS == 60
