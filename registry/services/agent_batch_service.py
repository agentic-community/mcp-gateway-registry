"""
Service layer for agent batch jobs (issue #956).

Handles job submission (with idempotency and per-user concurrency limits) and
lookup. The heavy lifting of running a job lives in agent_batch_worker; this
service owns creation and retrieval only.
"""

import logging
import uuid
from datetime import UTC, datetime

from ..core.config import settings
from ..repositories.documentdb.agent_batch_job_repository import (
    AgentBatchJobRepository,
)
from ..schemas.agent_models import (
    AgentBatchJob,
    AgentBatchJobState,
    AgentBatchRequest,
)

logger = logging.getLogger(__name__)


class ConcurrentJobLimitError(Exception):
    """Raised when a submitter already has the maximum active jobs."""


class AgentBatchService:
    """Create and look up agent batch jobs."""

    def __init__(self):
        self._repo = AgentBatchJobRepository()

    async def ensure_ready(self) -> None:
        """Ensure required indexes exist before serving traffic."""
        await self._repo.ensure_indexes()

    async def submit(
        self,
        req: AgentBatchRequest,
        submitted_by: str,
        submitted_body_hash: str,
        submitter_is_admin: bool,
        submitter_ui_permissions: dict[str, list[str]],
        request_id: str | None = None,
    ) -> tuple[AgentBatchJob, bool]:
        """Create a job, or return an existing idempotent one.

        Returns:
            (job, was_replay). was_replay is True when an existing job matching
            the (submitter, idempotency_key) pair was returned unchanged.
        """
        if req.idempotency_key:
            existing = await self._repo.find_by_idempotency(submitted_by, req.idempotency_key)
            if existing:
                if existing.submitted_body_hash != submitted_body_hash:
                    logger.warning(
                        "Idempotent replay with divergent body: job_id=%s user=%s key=%s "
                        "old_hash=%s new_hash=%s - returning original job, new body ignored",
                        existing.job_id,
                        submitted_by,
                        req.idempotency_key,
                        existing.submitted_body_hash,
                        submitted_body_hash,
                    )
                return existing, True

        active = await self._repo.count_active_for_user(submitted_by)
        if active >= settings.batch_max_concurrent_jobs_per_user:
            raise ConcurrentJobLimitError(
                f"submitter has {active} active jobs "
                f"(max {settings.batch_max_concurrent_jobs_per_user})"
            )

        now = datetime.now(UTC)
        job = AgentBatchJob(
            job_id=uuid.uuid4().hex,
            state=AgentBatchJobState.queued,
            submitted_by=submitted_by,
            submitted_at=now,
            updated_at=now,
            request_id=request_id,
            idempotency_key=req.idempotency_key,
            submitted_body_hash=submitted_body_hash,
            submitter_is_admin=submitter_is_admin,
            submitter_ui_permissions=submitter_ui_permissions,
            total=len(req.items),
            items=req.items,
        )
        await self._repo.insert(job)
        logger.info(
            "batch_submitted job_id=%s user=%s total=%d request_id=%s",
            job.job_id,
            submitted_by,
            job.total,
            request_id,
        )
        return job, False

    async def get(self, job_id: str) -> AgentBatchJob | None:
        """Fetch a job by id."""
        return await self._repo.get(job_id)

    def repo(self) -> AgentBatchJobRepository:
        """Expose the repository for the worker."""
        return self._repo


agent_batch_service = AgentBatchService()
