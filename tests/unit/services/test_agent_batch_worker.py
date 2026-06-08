"""Tests for registry.services.agent_batch_worker (issue #956).

Covers the worker's health snapshot, the disabled-via-config short-circuit, and
the per-job run loop including the succeeded/failed/partial finalize decision
and mid-job stop behavior. The repository, processor, and service are mocked.
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from registry.schemas.agent_models import (
    AgentBatchItemResult,
    AgentBatchJob,
    AgentBatchJobState,
    BatchItemOp,
)
from registry.services.agent_batch_worker import (
    AgentBatchWorker,
    get_agent_batch_worker,
)


def _job(total=2, next_index=0):
    items = [{"op": "delete", "path": f"/agents/{i}"} for i in range(total)]
    return AgentBatchJob(
        job_id="job-1",
        state=AgentBatchJobState.running,
        submitted_by="alice",
        submitted_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        submitted_body_hash="h",
        total=total,
        next_index=next_index,
        items=items,
    )


def _result(index, status):
    return AgentBatchItemResult(
        index=index, op=BatchItemOp.delete, path=f"/agents/{index}", status=status
    )


@pytest.mark.unit
class TestHealthAndStart:
    def test_health_snapshot_shape(self, monkeypatch):
        from registry.services import agent_batch_worker as mod

        monkeypatch.setattr(mod.settings, "batch_worker_enabled", True)
        w = AgentBatchWorker()
        h = w.health()
        assert h["enabled"] is True
        assert h["running"] is False
        assert h["current_job_id"] is None
        assert h["last_heartbeat"] is None
        assert h["worker_id"] == w._worker_id

    @pytest.mark.asyncio
    async def test_start_noop_when_disabled(self, monkeypatch):
        from registry.services import agent_batch_worker as mod

        monkeypatch.setattr(mod.settings, "batch_worker_enabled", False)
        w = AgentBatchWorker()
        await w.start()
        assert w._running is False
        assert w._task is None

    def test_singleton_returns_same_instance(self):
        assert get_agent_batch_worker() is get_agent_batch_worker()

    @pytest.mark.asyncio
    async def test_start_then_stop_lifecycle(self, monkeypatch):
        from registry.services import agent_batch_worker as mod

        monkeypatch.setattr(mod.settings, "batch_worker_enabled", True)
        repo = AsyncMock()
        repo.claim_next_queued.return_value = None  # idle loop
        with patch.object(mod, "agent_batch_service") as svc:
            svc.ensure_ready = AsyncMock()
            svc.repo.return_value = repo
            w = AgentBatchWorker()
            await w.start()
            assert w._running is True
            assert w._task is not None
            await w.stop()
        assert w._running is False

    @pytest.mark.asyncio
    async def test_stop_when_never_started_is_safe(self):
        w = AgentBatchWorker()
        await w.stop()  # no task; should not raise
        assert w._running is False


@pytest.mark.unit
@pytest.mark.asyncio
class TestRunJob:
    async def _run(self, job, item_statuses, fresh_after):
        """Drive _run_job with mocked repo/processor; return the finalize state."""
        repo = AsyncMock()
        repo.record_item_result = AsyncMock()
        repo.get.return_value = fresh_after
        repo.finalize = AsyncMock()

        results = iter([_result(i, s) for i, s in enumerate(item_statuses)])

        with (
            patch("registry.services.agent_batch_worker.agent_batch_service") as mock_service,
            patch(
                "registry.services.agent_batch_worker.process_item",
                new=AsyncMock(side_effect=lambda **kw: next(results)),
            ),
        ):
            mock_service.repo.return_value = repo
            w = AgentBatchWorker()
            w._running = True
            await w._run_job(job)
        return repo

    async def test_all_succeed_finalizes_succeeded(self):
        job = _job(total=2)
        fresh = _job(total=2)
        fresh.succeeded, fresh.failed = 2, 0
        repo = await self._run(job, [201, 204], fresh)
        repo.finalize.assert_awaited_once_with("job-1", AgentBatchJobState.succeeded)
        assert repo.record_item_result.await_count == 2

    async def test_all_fail_finalizes_failed(self):
        job = _job(total=2)
        fresh = _job(total=2)
        fresh.succeeded, fresh.failed = 0, 2
        repo = await self._run(job, [404, 403], fresh)
        repo.finalize.assert_awaited_once_with("job-1", AgentBatchJobState.failed)

    async def test_mixed_finalizes_partial(self):
        job = _job(total=2)
        fresh = _job(total=2)
        fresh.succeeded, fresh.failed = 1, 1
        repo = await self._run(job, [201, 500], fresh)
        repo.finalize.assert_awaited_once_with("job-1", AgentBatchJobState.partial)

    async def test_record_item_result_deltas(self):
        """A <400 status records succeeded_delta=1; >=400 records failed_delta=1."""
        job = _job(total=2)
        fresh = _job(total=2)
        fresh.succeeded, fresh.failed = 1, 1
        repo = await self._run(job, [200, 422], fresh)
        calls = repo.record_item_result.await_args_list
        assert calls[0].kwargs["succeeded_delta"] == 1
        assert calls[0].kwargs["failed_delta"] == 0
        assert calls[0].kwargs["next_index"] == 1
        assert calls[1].kwargs["succeeded_delta"] == 0
        assert calls[1].kwargs["failed_delta"] == 1
        assert calls[1].kwargs["next_index"] == 2

    async def test_vanished_job_skips_finalize(self):
        job = _job(total=1)
        repo = await self._run(job, [201], fresh_after=None)
        repo.finalize.assert_not_called()

    async def test_resume_from_next_index_skips_done_items(self):
        job = _job(total=3, next_index=2)
        fresh = _job(total=3)
        fresh.succeeded, fresh.failed = 3, 0
        repo = await self._run(job, [204], fresh)  # only one item left to run
        assert repo.record_item_result.await_count == 1
        assert repo.record_item_result.await_args_list[0].kwargs["next_index"] == 3

    async def test_lease_lost_mid_job_skips_finalize(self, monkeypatch):
        """If the heartbeat detects a lost lease, the job is abandoned, not finalized."""
        from registry.services import agent_batch_worker as mod

        monkeypatch.setattr(mod.settings, "batch_worker_lease_heartbeat_seconds", 0.0)
        repo = AsyncMock()
        repo.renew_lease.return_value = False  # another worker stole the lease
        repo.get.return_value = _job(total=2)

        async def slow_item(**kw):
            await asyncio.sleep(0.05)  # give the heartbeat time to fire and lose
            return _result(0, 201)

        with (
            patch("registry.services.agent_batch_worker.agent_batch_service") as svc,
            patch(
                "registry.services.agent_batch_worker.process_item",
                new=AsyncMock(side_effect=slow_item),
            ),
        ):
            svc.repo.return_value = repo
            w = AgentBatchWorker()
            w._running = True
            await w._run_job(_job(total=2))
        repo.finalize.assert_not_called()
        assert w._lease_lost is True

    async def test_loop_claims_runs_then_idles(self, monkeypatch):
        """_loop claims one job, runs it, then sleeps when the queue drains."""
        from registry.services import agent_batch_worker as mod

        monkeypatch.setattr(mod.settings, "batch_worker_poll_interval_seconds", 0.0)
        repo = AsyncMock()
        # First poll yields a job, then the loop stops before the next claim.
        repo.claim_next_queued.side_effect = [_job(total=0), None]

        w = AgentBatchWorker()
        w._running = True

        async def fake_run_job(job):
            # Stop after the first job so the loop exits cleanly.
            w._running = False

        with (
            patch.object(mod, "agent_batch_service") as svc,
            patch.object(w, "_run_job", side_effect=fake_run_job),
        ):
            svc.repo.return_value = repo
            await w._loop()
        assert w._current_job_id is None
        assert w._last_heartbeat is not None

    async def test_loop_survives_job_exception(self, monkeypatch):
        """An exception while running a job is logged and the loop continues."""
        from registry.services import agent_batch_worker as mod

        monkeypatch.setattr(mod.settings, "batch_worker_poll_interval_seconds", 0.0)
        repo = AsyncMock()
        repo.claim_next_queued.side_effect = [_job(total=0), None]
        w = AgentBatchWorker()
        w._running = True

        call_count = {"n": 0}

        async def boom(job):
            call_count["n"] += 1
            w._running = False
            raise RuntimeError("processing blew up")

        with (
            patch.object(mod, "agent_batch_service") as svc,
            patch.object(w, "_run_job", side_effect=boom),
        ):
            svc.repo.return_value = repo
            await w._loop()
        assert call_count["n"] == 1
        assert w._current_job_id is None

    async def test_stop_mid_job_leaves_running_no_finalize(self):
        """If _running flips false before items finish, the job is left running."""
        job = _job(total=2)
        repo = AsyncMock()
        with (
            patch("registry.services.agent_batch_worker.agent_batch_service") as mock_service,
            patch(
                "registry.services.agent_batch_worker.process_item",
                new=AsyncMock(),
            ) as mock_process,
        ):
            mock_service.repo.return_value = repo
            w = AgentBatchWorker()
            w._running = False  # already stopping
            await w._run_job(job)
        mock_process.assert_not_called()
        repo.finalize.assert_not_called()
