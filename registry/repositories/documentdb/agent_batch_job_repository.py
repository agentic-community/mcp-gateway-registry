"""
DocumentDB (MongoDB) implementation for agent batch jobs (issue #956).

Stores asynchronous agent batch jobs in the mcp_agent_batch_jobs collection.
A TTL index on updated_at reaps finished jobs after the configured retention
window; anchoring on updated_at (not submitted_at) guarantees an in-flight job
is never reaped mid-run, because every per-item checkpoint refreshes the clock.

Job ownership uses a time-bounded lease (claimed_by + lease_expires_at) folded
into the atomic claim query, so any number of workers can cooperatively drain
the queue and a crashed worker's job becomes reclaimable once its lease lapses.
"""

import logging
from datetime import UTC, datetime, timedelta

from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo import ASCENDING, ReturnDocument

from ...core.config import settings
from ...schemas.agent_models import (
    AgentBatchItemResult,
    AgentBatchJob,
    AgentBatchJobState,
)
from .client import get_collection_name, get_documentdb_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


def _doc_to_job(doc: dict) -> AgentBatchJob:
    """Convert a raw Mongo document into an AgentBatchJob."""
    doc = dict(doc)
    doc["job_id"] = doc.pop("_id")
    return AgentBatchJob(**doc)


class AgentBatchJobRepository:
    """MongoDB-backed store for agent batch jobs."""

    def __init__(self):
        self._collection: AsyncIOMotorCollection | None = None
        self._collection_name = get_collection_name("mcp_agent_batch_jobs")
        self._indexes_created = False

    async def _get_collection(self) -> AsyncIOMotorCollection:
        """Get the batch jobs collection, creating the handle lazily."""
        if self._collection is None:
            db = await get_documentdb_client()
            self._collection = db[self._collection_name]
        return self._collection

    async def ensure_indexes(self) -> None:
        """Create required indexes if not present (idempotent)."""
        if self._indexes_created:
            return

        col = await self._get_collection()
        try:
            await col.create_index(
                [("updated_at", ASCENDING)],
                expireAfterSeconds=settings.batch_job_retention_days * 86400,
                name="ttl_updated_at",
            )
            await col.create_index(
                [("submitted_by", ASCENDING), ("idempotency_key", ASCENDING)],
                unique=True,
                partialFilterExpression={"idempotency_key": {"$type": "string"}},
                name="uniq_idempotency_key_per_user",
            )
            await col.create_index([("state", ASCENDING)], name="idx_state")
            self._indexes_created = True
            logger.info(
                f"Created indexes for {self._collection_name} collection "
                f"(TTL={settings.batch_job_retention_days * 86400}s)"
            )
        except Exception as e:
            logger.warning(f"Could not create indexes for {self._collection_name}: {e}")

    async def insert(self, job: AgentBatchJob) -> None:
        """Insert a new batch job document."""
        col = await self._get_collection()
        doc = job.model_dump(mode="json")
        doc["_id"] = doc.pop("job_id")
        await col.insert_one(doc)

    async def get(self, job_id: str) -> AgentBatchJob | None:
        """Fetch a batch job by id, or None if it does not exist."""
        col = await self._get_collection()
        doc = await col.find_one({"_id": job_id})
        return _doc_to_job(doc) if doc else None

    async def find_by_idempotency(
        self,
        submitted_by: str,
        idempotency_key: str,
    ) -> AgentBatchJob | None:
        """Find an existing job for a (submitter, idempotency_key) pair."""
        col = await self._get_collection()
        doc = await col.find_one(
            {
                "submitted_by": submitted_by,
                "idempotency_key": idempotency_key,
            }
        )
        return _doc_to_job(doc) if doc else None

    async def count_active_for_user(self, submitted_by: str) -> int:
        """Count queued or running jobs for a submitter."""
        col = await self._get_collection()
        return await col.count_documents(
            {
                "submitted_by": submitted_by,
                "state": {"$in": ["queued", "running"]},
            }
        )

    async def claim_next_queued(self, claimed_by: str) -> AgentBatchJob | None:
        """Atomically claim one job, taking a time-bounded lease, and return it.

        Matches two kinds of jobs: freshly queued ones, and running ones whose
        lease has expired (the previous owner died or stalled). find_one_and_update
        is atomic, so any number of replicas racing on the same job is safe - only
        one wins, and the loser's filter no longer matches once the winner pushes
        lease_expires_at into the future. This makes recovery continuous instead of
        tied to worker restarts, and unblocks multi-worker operation.
        """
        col = await self._get_collection()
        now = datetime.now(UTC)
        lease_expires_at = now + timedelta(seconds=settings.batch_worker_lease_ttl_seconds)
        doc = await col.find_one_and_update(
            {
                "$or": [
                    {"state": "queued"},
                    {"state": "running", "lease_expires_at": {"$lt": now.isoformat()}},
                ]
            },
            {
                "$set": {
                    "state": "running",
                    "claimed_by": claimed_by,
                    "lease_expires_at": lease_expires_at.isoformat(),
                    "updated_at": now.isoformat(),
                }
            },
            sort=[("submitted_at", ASCENDING)],
            return_document=ReturnDocument.AFTER,
        )
        return _doc_to_job(doc) if doc else None

    async def renew_lease(self, job_id: str, claimed_by: str) -> bool:
        """Push the lease forward while still owning the job.

        The claimed_by guard means a worker that has already lost ownership (its
        lease lapsed and another worker reclaimed the job) renews nothing and
        learns it should stop. Returns True if the lease was renewed.
        """
        col = await self._get_collection()
        now = datetime.now(UTC)
        lease_expires_at = now + timedelta(seconds=settings.batch_worker_lease_ttl_seconds)
        result = await col.update_one(
            {"_id": job_id, "claimed_by": claimed_by, "state": "running"},
            {
                "$set": {
                    "lease_expires_at": lease_expires_at.isoformat(),
                    "updated_at": now.isoformat(),
                }
            },
        )
        return result.modified_count == 1

    async def record_item_result(
        self,
        job_id: str,
        result: AgentBatchItemResult,
        succeeded_delta: int,
        failed_delta: int,
        next_index: int,
    ) -> None:
        """Append a per-item result and advance the resume pointer atomically.

        Also renews the lease: every completed item pushes lease_expires_at
        forward, so as long as items finish faster than the lease TTL the worker
        keeps ownership without a separate heartbeat.
        """
        col = await self._get_collection()
        now = datetime.now(UTC)
        lease_expires_at = now + timedelta(seconds=settings.batch_worker_lease_ttl_seconds)
        await col.update_one(
            {"_id": job_id},
            {
                "$push": {"results": result.model_dump(mode="json")},
                "$inc": {"succeeded": succeeded_delta, "failed": failed_delta},
                "$set": {
                    "next_index": next_index,
                    "lease_expires_at": lease_expires_at.isoformat(),
                    "updated_at": now.isoformat(),
                },
            },
        )

    async def finalize(self, job_id: str, final_state: AgentBatchJobState) -> None:
        """Set the terminal state on a finished job and release its lease."""
        col = await self._get_collection()
        await col.update_one(
            {"_id": job_id},
            {
                "$set": {
                    "state": final_state.value,
                    "claimed_by": None,
                    "lease_expires_at": None,
                    "updated_at": datetime.now(UTC).isoformat(),
                }
            },
        )
