"""Integration tests for AgentBatchJobRepository against a real MongoDB (issue #956).

Exercises insert/get, idempotency lookup, active-job counting, the atomic
lease-based claim_next_queued transition (including expired-lease reclaim),
lease renewal, per-item result recording, and finalize. Requires a running
MongoDB (the test harness points DOCUMENTDB_HOST at localhost); skipped
automatically if unreachable.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from registry.repositories.documentdb.agent_batch_job_repository import (
    AgentBatchJobRepository,
)
from registry.schemas.agent_models import (
    AgentBatchItemResult,
    AgentBatchJob,
    AgentBatchJobState,
    BatchItemOp,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


def _job(submitted_by="alice", idempotency_key=None, state=AgentBatchJobState.queued, total=2):
    return AgentBatchJob(
        job_id=uuid.uuid4().hex,
        state=state,
        submitted_by=submitted_by,
        submitted_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        submitted_body_hash="hash",
        idempotency_key=idempotency_key,
        total=total,
        items=[{"op": "delete", "path": f"/agents/{i}"} for i in range(total)],
    )


@pytest.fixture
async def repo():
    """Repository backed by a real collection; cleans up jobs it created."""
    r = AgentBatchJobRepository()
    try:
        col = await r._get_collection()
        await col.database.command("ping")
    except Exception as e:
        pytest.skip(f"MongoDB not reachable: {e}")
    created: list[str] = []
    r._created = created  # track for cleanup
    yield r
    col = await r._get_collection()
    if created:
        await col.delete_many({"_id": {"$in": created}})


async def _insert(repo, job):
    await repo.insert(job)
    repo._created.append(job.job_id)
    return job


class TestInsertAndGet:
    async def test_insert_then_get_roundtrips(self, repo):
        job = await _insert(repo, _job())
        fetched = await repo.get(job.job_id)
        assert fetched is not None
        assert fetched.job_id == job.job_id
        assert fetched.submitted_by == "alice"
        assert fetched.total == 2

    async def test_get_missing_returns_none(self, repo):
        assert await repo.get("does-not-exist") is None


class TestIdempotency:
    async def test_find_by_idempotency_matches(self, repo):
        key = f"key-{uuid.uuid4().hex}"
        job = await _insert(repo, _job(idempotency_key=key))
        found = await repo.find_by_idempotency("alice", key)
        assert found is not None
        assert found.job_id == job.job_id

    async def test_find_by_idempotency_scoped_to_user(self, repo):
        key = f"key-{uuid.uuid4().hex}"
        await _insert(repo, _job(submitted_by="alice", idempotency_key=key))
        # Different user, same key -> no match.
        assert await repo.find_by_idempotency("bob", key) is None


class TestActiveCount:
    async def test_counts_queued_and_running(self, repo):
        user = f"u-{uuid.uuid4().hex}"
        await _insert(repo, _job(submitted_by=user, state=AgentBatchJobState.queued))
        await _insert(repo, _job(submitted_by=user, state=AgentBatchJobState.running))
        await _insert(repo, _job(submitted_by=user, state=AgentBatchJobState.succeeded))
        assert await repo.count_active_for_user(user) == 2


class TestClaimNextQueued:
    async def test_claim_transitions_to_running_with_lease(self, repo):
        user = f"u-{uuid.uuid4().hex}"
        await _insert(repo, _job(submitted_by=user, state=AgentBatchJobState.queued))
        claimed = await repo.claim_next_queued("worker-a")
        assert claimed is not None
        assert claimed.state == AgentBatchJobState.running
        assert claimed.claimed_by == "worker-a"
        assert claimed.lease_expires_at is not None
        refetched = await repo.get(claimed.job_id)
        assert refetched.state == AgentBatchJobState.running

    async def test_claim_is_atomic_single_winner(self, repo):
        """Two concurrent claims on one queued job yield distinct or null results."""
        import asyncio

        user = f"u-{uuid.uuid4().hex}"
        await _insert(repo, _job(submitted_by=user, state=AgentBatchJobState.queued))
        # Drain any other queued jobs first to isolate; then race two claims.
        results = await asyncio.gather(
            repo.claim_next_queued("worker-a"), repo.claim_next_queued("worker-b")
        )
        claimed_ids = [r.job_id for r in results if r is not None]
        # No job_id is claimed twice.
        assert len(claimed_ids) == len(set(claimed_ids))
        for jid in claimed_ids:
            repo._created.append(jid)

    async def test_running_job_with_live_lease_not_reclaimed(self, repo):
        """A running job whose lease is still valid is invisible to a new claim."""
        user = f"u-{uuid.uuid4().hex}"
        job = await _insert(repo, _job(submitted_by=user, state=AgentBatchJobState.queued))
        # worker-a claims it, taking a fresh (live) lease.
        await repo.claim_next_queued("worker-a")
        # worker-b tries to claim; the only job for this user is running with a
        # live lease, so it must not match the expired-lease reclaim branch.
        claimed = await repo.claim_next_queued("worker-b")
        if claimed is not None and claimed.job_id == job.job_id:
            raise AssertionError("live-lease job was wrongly reclaimed")
        if claimed is not None:
            repo._created.append(claimed.job_id)
        # The original job is still owned by worker-a.
        refetched = await repo.get(job.job_id)
        assert refetched.claimed_by == "worker-a"

    async def test_expired_lease_is_reclaimed(self, repo):
        """A running job whose lease has lapsed is reclaimed by the next worker."""
        user = f"u-{uuid.uuid4().hex}"
        job = await _insert(repo, _job(submitted_by=user, state=AgentBatchJobState.queued))
        await repo.claim_next_queued("worker-a")
        # Force the lease into the past, simulating a dead worker.
        col = await repo._get_collection()
        past = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
        await col.update_one({"_id": job.job_id}, {"$set": {"lease_expires_at": past}})
        reclaimed = await repo.claim_next_queued("worker-b")
        assert reclaimed is not None
        assert reclaimed.job_id == job.job_id
        assert reclaimed.claimed_by == "worker-b"


class TestRenewLease:
    async def test_renew_extends_lease_for_owner(self, repo):
        user = f"u-{uuid.uuid4().hex}"
        job = await _insert(repo, _job(submitted_by=user, state=AgentBatchJobState.queued))
        claimed = await repo.claim_next_queued("worker-a")
        before = claimed.lease_expires_at
        ok = await repo.renew_lease(job.job_id, "worker-a")
        assert ok is True
        refetched = await repo.get(job.job_id)
        assert refetched.lease_expires_at >= before

    async def test_renew_rejected_for_non_owner(self, repo):
        user = f"u-{uuid.uuid4().hex}"
        job = await _insert(repo, _job(submitted_by=user, state=AgentBatchJobState.queued))
        await repo.claim_next_queued("worker-a")
        ok = await repo.renew_lease(job.job_id, "worker-b")
        assert ok is False


class TestRecordAndFinalize:
    async def test_record_item_result_appends_and_increments(self, repo):
        job = await _insert(repo, _job(total=2))
        result = AgentBatchItemResult(index=0, op=BatchItemOp.delete, path="/agents/0", status=204)
        await repo.record_item_result(
            job_id=job.job_id,
            result=result,
            succeeded_delta=1,
            failed_delta=0,
            next_index=1,
        )
        fetched = await repo.get(job.job_id)
        assert fetched.succeeded == 1
        assert fetched.failed == 0
        assert fetched.next_index == 1
        assert len(fetched.results) == 1
        assert fetched.results[0].status == 204

    async def test_finalize_sets_terminal_state_and_releases_lease(self, repo):
        job = await _insert(repo, _job(state=AgentBatchJobState.queued))
        await repo.claim_next_queued("worker-a")
        await repo.finalize(job.job_id, AgentBatchJobState.partial)
        fetched = await repo.get(job.job_id)
        assert fetched.state == AgentBatchJobState.partial
        assert fetched.claimed_by is None
        assert fetched.lease_expires_at is None
