"""model.encode() must not run on the event loop.

Issue #1751. ``_embed_texts()`` called ``model.encode()`` directly inside an
``async def``. encode() is synchronous and CPU-bound, so it blocked the loop for
its whole duration and stalled every other request on the worker, including
health checks and auth. Measured at ~16ms per query on all-MiniLM-L6-v2, with a
10ms heartbeat observed returning after 50ms.

The offload is bounded on purpose. Torch parallelises inside a single encode, so
letting the default executor grow to min(32, cpu_count + 4) would put dozens of
encodes on the same cores and make every concurrent search slower. The semaphore
keeps the loop free without trading latency for thrash.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from registry.repositories.documentdb import search_repository
from registry.repositories.documentdb.search_repository import DocumentDBSearchRepository


class _FakeVector:
    """Stands in for the numpy row that encode() returns."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


def _repo_with_model(encode_impl) -> DocumentDBSearchRepository:
    instance = DocumentDBSearchRepository.__new__(DocumentDBSearchRepository)
    instance._embedding_unavailable = False
    model = MagicMock()
    model.encode = MagicMock(side_effect=encode_impl)
    instance._get_embedding_model = AsyncMock(return_value=model)
    return instance


@pytest.fixture(autouse=True)
def _fresh_semaphore():
    """Reset the module semaphore so each test binds it to its own loop."""
    search_repository._encode_semaphore = None
    yield
    search_repository._encode_semaphore = None


class TestEncodeIsOffloaded:
    """The loop must stay responsive while encoding."""

    @pytest.mark.asyncio
    async def test_encode_runs_off_the_event_loop_thread(self) -> None:
        """encode() must execute on a worker thread, not the loop thread."""
        import threading

        loop_thread = threading.current_thread()
        seen: dict = {}

        def encode(texts):
            seen["thread"] = threading.current_thread()
            return [_FakeVector([0.1, 0.2])]

        repo = _repo_with_model(encode)
        result = await repo._embed_texts(["hello"], context="test")

        assert result == [[0.1, 0.2]]
        assert seen["thread"] is not loop_thread, (
            f"encode ran on the loop thread {seen['thread'].name}, expected a worker"
        )

    @pytest.mark.asyncio
    async def test_a_slow_encode_does_not_stall_the_loop(self) -> None:
        """A blocking encode must not delay a concurrent coroutine.

        The old code would hold the loop for the whole sleep, pushing the
        heartbeat far past its 10ms schedule.
        """

        def encode(texts):
            time.sleep(0.15)
            return [_FakeVector([0.3])]

        repo = _repo_with_model(encode)
        ticks: list[float] = []

        async def heartbeat() -> None:
            last = time.perf_counter()
            for _ in range(10):
                await asyncio.sleep(0.01)
                now = time.perf_counter()
                ticks.append(now - last)
                last = now

        await asyncio.gather(repo._embed_texts(["hello"], context="test"), heartbeat())

        worst = max(ticks)
        assert worst < 0.10, f"loop stalled for {worst * 1000:.0f}ms during encode"

    @pytest.mark.asyncio
    async def test_failure_still_latches_and_returns_none(self) -> None:
        """The offload must not swallow the unavailable latch."""

        def encode(texts):
            raise RuntimeError("model gone")

        repo = _repo_with_model(encode)
        result = await repo._embed_texts(["hello"], context="test")

        assert result is None
        assert repo._embedding_unavailable is True

    @pytest.mark.asyncio
    async def test_latch_short_circuits_without_touching_the_model(self) -> None:
        repo = _repo_with_model(lambda texts: [_FakeVector([0.0])])
        repo._embedding_unavailable = True

        assert await repo._embed_texts(["hello"], context="test") is None
        repo._get_embedding_model.assert_not_awaited()


class TestEncodeConcurrencyIsBounded:
    """Unbounded threads would make every concurrent search slower."""

    @pytest.mark.asyncio
    async def test_concurrent_encodes_do_not_all_run_at_once(self) -> None:
        import threading

        limit = search_repository._ENCODE_CONCURRENCY
        in_flight = 0
        peak = 0
        lock = threading.Lock()

        def encode(texts):
            nonlocal in_flight, peak
            with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            time.sleep(0.05)
            with lock:
                in_flight -= 1
            return [_FakeVector([0.5])]

        repo = _repo_with_model(encode)
        await asyncio.gather(
            *(repo._embed_texts([f"q{i}"], context="test") for i in range(limit + 4))
        )

        assert peak <= limit, f"{peak} concurrent encodes exceeded the cap of {limit}"

    def test_the_cap_is_configurable_and_at_least_one(self) -> None:
        assert search_repository._ENCODE_CONCURRENCY >= 1
