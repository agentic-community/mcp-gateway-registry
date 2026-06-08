"""
In-process worker that drains the agent batch job queue (issue #956).

Follows the singleton + lifespan pattern of ans_sync_scheduler. Job ownership
uses a time-bounded lease: claim_next_queued takes a lease, a heartbeat renews
it while the job runs, and a crashed worker's job becomes reclaimable by any
worker once the lease lapses. This makes recovery continuous (not tied to
restarts) and lets any number of replicas run with BATCH_WORKER_ENABLED=true.
"""

import asyncio
import logging
import socket
import uuid
from datetime import UTC, datetime

from ..core.config import settings
from ..schemas.agent_models import AgentBatchJob, AgentBatchJobState
from .agent_batch_item_processor import process_item
from .agent_batch_service import agent_batch_service

logger = logging.getLogger(__name__)


class AgentBatchWorker:
    """Polls MongoDB for queued batch jobs and runs them item by item."""

    def __init__(self):
        self._task: asyncio.Task | None = None
        self._running: bool = False
        self._last_heartbeat: datetime | None = None
        self._current_job_id: str | None = None
        # Stable per-process identity recorded as claimed_by on leased jobs.
        self._worker_id: str = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        # Set by the heartbeat when a renewal fails (lease lost to another worker).
        self._lease_lost: bool = False

    def health(self) -> dict:
        """Expose worker liveness for the /health endpoint."""
        return {
            "enabled": settings.batch_worker_enabled,
            "running": self._running,
            "worker_id": self._worker_id,
            "current_job_id": self._current_job_id,
            "last_heartbeat": self._last_heartbeat.isoformat() if self._last_heartbeat else None,
        }

    async def start(self) -> None:
        """Start the worker loop unless disabled via BATCH_WORKER_ENABLED=false."""
        if not settings.batch_worker_enabled:
            logger.info("Agent batch worker disabled via BATCH_WORKER_ENABLED=false")
            return
        await agent_batch_service.ensure_ready()
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Agent batch worker started (worker_id=%s)", self._worker_id)

    async def stop(self) -> None:
        """Stop the worker loop and cancel the in-flight task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Agent batch worker stopped")

    async def _loop(self) -> None:
        """Claim and run queued jobs until stopped."""
        poll = settings.batch_worker_poll_interval_seconds
        while self._running:
            self._last_heartbeat = datetime.now(UTC)
            try:
                job = await agent_batch_service.repo().claim_next_queued(self._worker_id)
                if job is None:
                    await asyncio.sleep(poll)
                    continue
                self._current_job_id = job.job_id
                await self._run_job(job)
                self._current_job_id = None
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("batch_worker_loop_error")
                self._current_job_id = None
                await asyncio.sleep(poll)

    async def _heartbeat(self, job_id: str) -> None:
        """Renew the lease on the running job until cancelled or ownership lost.

        Insurance for items that legitimately take longer than the lease TTL:
        record_item_result already renews on every completed item, but a single
        slow item would otherwise let the lease lapse mid-processing.
        """
        interval = settings.batch_worker_lease_heartbeat_seconds
        repo = agent_batch_service.repo()
        while True:
            await asyncio.sleep(interval)
            renewed = await repo.renew_lease(job_id, self._worker_id)
            if not renewed:
                logger.warning(
                    "batch_lease_lost job_id=%s worker_id=%s - another worker "
                    "reclaimed the job; abandoning",
                    job_id,
                    self._worker_id,
                )
                self._lease_lost = True
                return
            self._last_heartbeat = datetime.now(UTC)

    async def _process_items(self, job: AgentBatchJob, repo) -> None:
        """Run each remaining item, recording its result and advancing next_index."""
        for idx in range(job.next_index, job.total):
            if not self._running or self._lease_lost:
                logger.info(f"Worker stopping mid-job {job.job_id} at index {idx}")
                return
            item = job.items[idx]
            result = await process_item(
                index=idx,
                item=item,
                submitted_by=job.submitted_by,
                is_admin=job.submitter_is_admin,
                ui_permissions=job.submitter_ui_permissions,
            )
            succeeded_delta = 1 if result.status < 400 else 0
            failed_delta = 1 - succeeded_delta
            await repo.record_item_result(
                job_id=job.job_id,
                result=result,
                succeeded_delta=succeeded_delta,
                failed_delta=failed_delta,
                next_index=idx + 1,
            )
            logger.info(
                "batch_item_done job_id=%s index=%d op=%s status=%d request_id=%s",
                job.job_id,
                idx,
                result.op.value,
                result.status,
                job.request_id,
            )

    async def _run_job(self, job: AgentBatchJob) -> None:
        """Process all remaining items in a claimed job, then finalize it."""
        start = datetime.now(UTC)
        logger.info(f"Running batch job {job.job_id} ({job.total} items, resume@{job.next_index})")
        repo = agent_batch_service.repo()
        self._lease_lost = False
        heartbeat = asyncio.create_task(self._heartbeat(job.job_id))
        try:
            await self._process_items(job, repo)
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass

        # If the lease was lost mid-job, another worker now owns it; do not finalize.
        if self._lease_lost:
            return
        # If we stopped before finishing all items, leave the job running; its
        # lease will lapse and another worker (or this one on restart) reclaims it.
        if not self._running:
            return

        fresh = await repo.get(job.job_id)
        if fresh is None:
            logger.warning(f"Batch job {job.job_id} vanished before finalize")
            return
        if fresh.failed == 0:
            final = AgentBatchJobState.succeeded
        elif fresh.succeeded == 0:
            final = AgentBatchJobState.failed
        else:
            final = AgentBatchJobState.partial
        await repo.finalize(job.job_id, final)
        duration = (datetime.now(UTC) - start).total_seconds()
        logger.info(
            "batch_finalized job_id=%s state=%s succeeded=%d failed=%d duration_s=%.2f request_id=%s",
            job.job_id,
            final.value,
            fresh.succeeded,
            fresh.failed,
            duration,
            job.request_id,
        )


_worker: AgentBatchWorker | None = None


def get_agent_batch_worker() -> AgentBatchWorker:
    """Get the global agent batch worker singleton."""
    global _worker
    if _worker is None:
        _worker = AgentBatchWorker()
    return _worker
