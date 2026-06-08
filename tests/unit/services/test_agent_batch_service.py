"""Tests for registry.services.agent_batch_service (issue #956).

Covers job submission with idempotency replay, divergent-body detection, the
per-user concurrency limit, and lookup. The repository is mocked so these stay
unit-level (no MongoDB required).
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from registry.schemas.agent_models import (
    AgentBatchJob,
    AgentBatchJobState,
    AgentBatchRequest,
)
from registry.services.agent_batch_service import (
    AgentBatchService,
    ConcurrentJobLimitError,
)


def _request(idempotency_key=None):
    return AgentBatchRequest(
        idempotency_key=idempotency_key,
        items=[{"op": "delete", "path": "/agents/x"}],
    )


def _existing_job(**overrides):
    base = {
        "job_id": "existing123",
        "state": AgentBatchJobState.queued,
        "submitted_by": "alice",
        "submitted_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
        "submitted_body_hash": "hash-abc",
        "total": 1,
        "items": [{"op": "delete", "path": "/agents/x"}],
    }
    base.update(overrides)
    return AgentBatchJob(**base)


@pytest.fixture
def service():
    svc = AgentBatchService()
    svc._repo = AsyncMock()
    return svc


@pytest.mark.unit
@pytest.mark.asyncio
class TestSubmit:
    async def test_creates_new_job_when_no_idempotency_key(self, service):
        service._repo.count_active_for_user.return_value = 0
        job, replayed = await service.submit(
            _request(),
            submitted_by="alice",
            submitted_body_hash="hash-abc",
            submitter_is_admin=False,
            submitter_ui_permissions={},
        )
        assert replayed is False
        assert job.submitted_by == "alice"
        assert job.state == AgentBatchJobState.queued
        assert job.total == 1
        service._repo.insert.assert_awaited_once()
        # No idempotency lookup when no key supplied.
        service._repo.find_by_idempotency.assert_not_called()

    async def test_idempotent_replay_returns_existing(self, service):
        existing = _existing_job()
        service._repo.find_by_idempotency.return_value = existing
        job, replayed = await service.submit(
            _request(idempotency_key="key-1"),
            submitted_by="alice",
            submitted_body_hash="hash-abc",
            submitter_is_admin=False,
            submitter_ui_permissions={},
        )
        assert replayed is True
        assert job is existing
        service._repo.insert.assert_not_called()
        service._repo.count_active_for_user.assert_not_called()

    async def test_divergent_body_replay_still_returns_original(self, service):
        existing = _existing_job(submitted_body_hash="OLD-hash")
        service._repo.find_by_idempotency.return_value = existing
        job, replayed = await service.submit(
            _request(idempotency_key="key-1"),
            submitted_by="alice",
            submitted_body_hash="NEW-hash",
            submitter_is_admin=False,
            submitter_ui_permissions={},
        )
        # Original job returned unchanged; new body ignored.
        assert replayed is True
        assert job is existing
        service._repo.insert.assert_not_called()

    async def test_no_existing_job_creates_with_idempotency_key(self, service):
        service._repo.find_by_idempotency.return_value = None
        service._repo.count_active_for_user.return_value = 0
        job, replayed = await service.submit(
            _request(idempotency_key="key-1"),
            submitted_by="alice",
            submitted_body_hash="hash-abc",
            submitter_is_admin=False,
            submitter_ui_permissions={},
        )
        assert replayed is False
        assert job.idempotency_key == "key-1"
        service._repo.insert.assert_awaited_once()

    async def test_concurrency_limit_raises(self, service, monkeypatch):
        from registry.services import agent_batch_service as mod

        monkeypatch.setattr(mod.settings, "batch_max_concurrent_jobs_per_user", 3)
        service._repo.count_active_for_user.return_value = 3
        with pytest.raises(ConcurrentJobLimitError) as exc:
            await service.submit(
                _request(),
                submitted_by="alice",
                submitted_body_hash="hash-abc",
                submitter_is_admin=False,
                submitter_ui_permissions={},
            )
        assert "active jobs" in str(exc.value)
        service._repo.insert.assert_not_called()

    async def test_at_limit_boundary_allows_when_below(self, service, monkeypatch):
        from registry.services import agent_batch_service as mod

        monkeypatch.setattr(mod.settings, "batch_max_concurrent_jobs_per_user", 3)
        service._repo.count_active_for_user.return_value = 2
        job, replayed = await service.submit(
            _request(),
            submitted_by="alice",
            submitted_body_hash="hash-abc",
            submitter_is_admin=False,
            submitter_ui_permissions={},
        )
        assert replayed is False
        service._repo.insert.assert_awaited_once()

    async def test_authorization_snapshot_persisted_on_job(self, service):
        service._repo.count_active_for_user.return_value = 0
        perms = {"publish_agent": ["all"], "modify_service": ["/agents/x"]}
        job, _ = await service.submit(
            _request(),
            submitted_by="alice",
            submitted_body_hash="hash-abc",
            submitter_is_admin=True,
            submitter_ui_permissions=perms,
            request_id="req-99",
        )
        assert job.submitter_is_admin is True
        assert job.submitter_ui_permissions == perms
        assert job.request_id == "req-99"


@pytest.mark.unit
@pytest.mark.asyncio
class TestGetAndRepo:
    async def test_get_delegates_to_repo(self, service):
        existing = _existing_job()
        service._repo.get.return_value = existing
        result = await service.get("existing123")
        assert result is existing
        service._repo.get.assert_awaited_once_with("existing123")

    async def test_get_missing_returns_none(self, service):
        service._repo.get.return_value = None
        assert await service.get("nope") is None

    async def test_repo_accessor_returns_repository(self, service):
        assert service.repo() is service._repo

    async def test_ensure_ready_creates_indexes(self, service):
        await service.ensure_ready()
        service._repo.ensure_indexes.assert_awaited_once()
