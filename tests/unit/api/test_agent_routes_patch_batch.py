"""Route-level tests for the PATCH and batch endpoints (issue #956).

Exercises agent_routes.patch_agent, submit_agent_batch, and get_agent_batch_job
through a FastAPI TestClient with the auth dependency overridden. Service-layer
collaborators (agent_service, agent_batch_service, validator, gate,
webhook) are mocked so these stay unit-level.
"""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from registry.api.agent_routes import router
from registry.schemas.agent_models import (
    AgentBatchJob,
    AgentBatchJobState,
)
from tests.fixtures.factories import AgentCardFactory


@pytest.fixture
def user_context() -> dict[str, Any]:
    return {
        "username": "alice",
        "groups": ["g"],
        "scopes": ["write:agents"],
        "auth_method": "session",
        "provider": "local",
        "accessible_agents": ["all"],
        "ui_permissions": {
            "publish_agent": ["all"],
            "modify_service": ["all"],
        },
        "is_admin": False,
    }


@pytest.fixture
def client(user_context):
    app = FastAPI()
    app.include_router(router)
    from registry.api.agent_routes import nginx_proxied_auth
    from registry.auth.csrf import verify_csrf_token_flexible

    app.dependency_overrides[nginx_proxied_auth] = lambda: user_context
    app.dependency_overrides[verify_csrf_token_flexible] = lambda: None
    yield TestClient(app)
    app.dependency_overrides.clear()


def _agent(registered_by="alice", updated_at=None, sync_metadata=None):
    return AgentCardFactory(
        name="existing-agent",
        path="/agents/existing",
        registered_by=registered_by,
        updated_at=updated_at or datetime(2026, 1, 1, tzinfo=UTC),
        sync_metadata=sync_metadata,
    )


# =============================================================================
# PATCH /api/agents/{path}
# =============================================================================


@pytest.mark.unit
class TestPatchAgent:
    def test_patch_not_found_returns_404(self, client):
        with patch("registry.api.agent_routes.agent_service") as svc:
            svc.get_agent_info = AsyncMock(return_value=None)
            r = client.patch("/agents/existing", json={"description": "new"})
        assert r.status_code == 404

    def test_patch_federated_returns_403(self, client):
        agent = _agent(sync_metadata={"is_federated": True, "source_peer_id": "peer-1"})
        with patch("registry.api.agent_routes.agent_service") as svc:
            svc.get_agent_info = AsyncMock(return_value=agent)
            r = client.patch("/agents/existing", json={"description": "new"})
        assert r.status_code == 403
        assert "peer-1" in r.json()["detail"]

    def test_patch_non_owner_non_admin_returns_403(self, client):
        agent = _agent(registered_by="bob")
        with (
            patch("registry.api.agent_routes.agent_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
        ):
            svc.get_agent_info = AsyncMock(return_value=agent)
            r = client.patch("/agents/existing", json={"description": "new"})
        assert r.status_code == 403
        assert "only patch agents you registered" in r.json()["detail"]

    def test_patch_empty_body_returns_400(self, client):
        agent = _agent()
        with (
            patch("registry.api.agent_routes.agent_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
        ):
            svc.get_agent_info = AsyncMock(return_value=agent)
            r = client.patch("/agents/existing", json={})
        assert r.status_code == 400
        assert "Empty patch body" in r.json()["detail"]

    def test_patch_registrant_only_field_returns_422(self, client):
        """AgentCardPatch rejects registrant-only fields before the handler runs."""
        agent = _agent()
        with patch("registry.api.agent_routes.agent_service") as svc:
            svc.get_agent_info = AsyncMock(return_value=agent)
            r = client.patch("/agents/existing", json={"registered_by": "mallory"})
        assert r.status_code == 422

    def test_patch_if_match_mismatch_returns_412(self, client):
        agent = _agent(updated_at=datetime(2026, 1, 1, tzinfo=UTC))
        with (
            patch("registry.api.agent_routes.agent_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
        ):
            svc.get_agent_info = AsyncMock(return_value=agent)
            r = client.patch(
                "/agents/existing",
                json={"description": "new"},
                headers={"If-Match": 'W/"1"'},
            )
        assert r.status_code == 412

    def test_patch_if_match_malformed_returns_400(self, client):
        agent = _agent()
        with (
            patch("registry.api.agent_routes.agent_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
        ):
            svc.get_agent_info = AsyncMock(return_value=agent)
            r = client.patch(
                "/agents/existing",
                json={"description": "new"},
                headers={"If-Match": "not-an-etag"},
            )
        assert r.status_code == 400

    def test_patch_success_returns_200_with_etag(self, client):
        agent = _agent(updated_at=datetime(2026, 1, 1, tzinfo=UTC))
        updated = _agent(updated_at=datetime(2026, 2, 2, tzinfo=UTC))
        validation = MagicMock(is_valid=True, errors=[])
        gate = MagicMock(allowed=True, error_message=None)
        with (
            patch("registry.api.agent_routes.agent_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
            patch("registry.utils.agent_validator.agent_validator") as validator,
            patch(
                "registry.api.agent_routes.check_registration_gate",
                AsyncMock(return_value=gate),
            ),
            patch("registry.api.agent_routes.send_registration_webhook", AsyncMock()),
        ):
            svc.get_agent_info = AsyncMock(side_effect=[agent, updated])
            svc.update_agent = AsyncMock()
            validator.validate_agent_card = AsyncMock(return_value=validation)
            r = client.patch("/agents/existing", json={"description": "new desc"})
        assert r.status_code == 200
        expected_ms = int(datetime(2026, 2, 2, tzinfo=UTC).timestamp() * 1000)
        assert r.headers["ETag"] == f'W/"{expected_ms}"'

    def test_patch_if_match_match_succeeds(self, client):
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        agent = _agent(updated_at=ts)
        updated = _agent(updated_at=datetime(2026, 2, 2, tzinfo=UTC))
        validation = MagicMock(is_valid=True, errors=[])
        gate = MagicMock(allowed=True, error_message=None)
        etag = f'W/"{int(ts.timestamp() * 1000)}"'
        with (
            patch("registry.api.agent_routes.agent_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
            patch("registry.utils.agent_validator.agent_validator") as validator,
            patch(
                "registry.api.agent_routes.check_registration_gate",
                AsyncMock(return_value=gate),
            ),
            patch("registry.api.agent_routes.send_registration_webhook", AsyncMock()),
        ):
            svc.get_agent_info = AsyncMock(side_effect=[agent, updated])
            svc.update_agent = AsyncMock()
            validator.validate_agent_card = AsyncMock(return_value=validation)
            r = client.patch(
                "/agents/existing",
                json={"description": "new desc"},
                headers={"If-Match": etag},
            )
        assert r.status_code == 200


# =============================================================================
# POST /api/agents/batch  and  GET /api/agents/batch/{job_id}
# =============================================================================


def _job(submitted_by="alice", state=AgentBatchJobState.queued):
    return AgentBatchJob(
        job_id="job-abc",
        state=state,
        submitted_by=submitted_by,
        submitted_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        submitted_body_hash="h",
        total=1,
        items=[{"op": "delete", "path": "/agents/x"}],
    )


@pytest.mark.unit
class TestSubmitBatch:
    def test_submit_returns_202_with_status_url(self, client):
        with patch("registry.api.agent_routes.agent_batch_service") as svc:
            svc.submit = AsyncMock(return_value=(_job(), False))
            r = client.post(
                "/agents/batch",
                json={"items": [{"op": "delete", "path": "/agents/x"}]},
            )
        assert r.status_code == 202
        body = r.json()
        assert body["job_id"] == "job-abc"
        assert body["status_url"] == "/api/agents/batch/job-abc"
        assert "X-Idempotent-Replay" not in r.headers

    def test_submit_replay_sets_header(self, client):
        with patch("registry.api.agent_routes.agent_batch_service") as svc:
            svc.submit = AsyncMock(return_value=(_job(), True))
            r = client.post(
                "/agents/batch",
                json={
                    "idempotency_key": "k1",
                    "items": [{"op": "delete", "path": "/agents/x"}],
                },
            )
        assert r.status_code == 202
        assert r.headers["X-Idempotent-Replay"] == "true"

    def test_submit_empty_items_returns_422(self, client):
        r = client.post("/agents/batch", json={"items": []})
        assert r.status_code == 422

    def test_submit_too_many_items_returns_413(self, client, monkeypatch):
        from registry.api import agent_routes as mod

        monkeypatch.setattr(mod.settings, "batch_max_operations_per_job", 2)
        items = [{"op": "delete", "path": f"/agents/{i}"} for i in range(3)]
        r = client.post("/agents/batch", json={"items": items})
        assert r.status_code == 413
        assert "max 2 items" in r.json()["detail"]

    def test_submit_concurrency_limit_returns_429(self, client):
        from registry.services.agent_batch_service import ConcurrentJobLimitError

        with patch("registry.api.agent_routes.agent_batch_service") as svc:
            svc.submit = AsyncMock(side_effect=ConcurrentJobLimitError("too many"))
            r = client.post(
                "/agents/batch",
                json={"items": [{"op": "delete", "path": "/agents/x"}]},
            )
        assert r.status_code == 429
        assert "too many" in r.json()["detail"]


@pytest.mark.unit
class TestGetBatchJob:
    def test_get_job_returns_state(self, client):
        with patch("registry.api.agent_routes.agent_batch_service") as svc:
            svc.get = AsyncMock(return_value=_job(state=AgentBatchJobState.running))
            r = client.get("/agents/batch/job-abc")
        assert r.status_code == 200
        assert r.json()["state"] == "running"
        assert r.json()["job_id"] == "job-abc"

    def test_get_unknown_job_returns_404(self, client):
        with patch("registry.api.agent_routes.agent_batch_service") as svc:
            svc.get = AsyncMock(return_value=None)
            r = client.get("/agents/batch/nope")
        assert r.status_code == 404

    def test_get_other_users_job_returns_403(self, client):
        with patch("registry.api.agent_routes.agent_batch_service") as svc:
            svc.get = AsyncMock(return_value=_job(submitted_by="bob"))
            r = client.get("/agents/batch/job-abc")
        assert r.status_code == 403
        assert "only view batch jobs you submitted" in r.json()["detail"]
