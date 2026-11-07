"""POST /api/admin/metadata-security/rescan (issue #1811)."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from registry.api import metadata_security_routes as routes
from registry.auth.csrf import verify_csrf_token_flexible

pytestmark = pytest.mark.unit


def _client(is_admin: bool) -> TestClient:
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    app.dependency_overrides[routes.nginx_proxied_auth] = lambda: {
        "username": "u",
        "is_admin": is_admin,
    }
    app.dependency_overrides[verify_csrf_token_flexible] = lambda: None
    return TestClient(app)


def test_non_admin_is_refused():
    with patch.object(routes, "rescan_metadata", AsyncMock()) as rescan:
        response = _client(is_admin=False).post("/api/admin/metadata-security/rescan", json={})

    assert response.status_code == 403
    rescan.assert_not_called()


def test_admin_runs_the_rescan_with_the_requested_scope():
    result = {"server": {"examined": 1}, "dry_run": True, "elapsed_seconds": 0.1}
    with patch.object(routes, "rescan_metadata", AsyncMock(return_value=result)) as rescan:
        response = _client(is_admin=True).post(
            "/api/admin/metadata-security/rescan",
            json={"entity_types": ["server"], "dry_run": True},
        )

    assert response.status_code == 200
    assert response.json() == result
    rescan.assert_awaited_once_with(["server"], dry_run=True)


@pytest.mark.parametrize(
    "body",
    [{"entity_types": ["custom"]}, {"entity_types": []}, {"force": True}],
    ids=["unknown-type", "empty-list", "unknown-field"],
)
def test_invalid_body_is_rejected(body):
    with patch.object(routes, "rescan_metadata", AsyncMock()) as rescan:
        response = _client(is_admin=True).post("/api/admin/metadata-security/rescan", json=body)

    assert response.status_code == 422
    rescan.assert_not_called()


def test_stopped_rescan_is_a_503_with_the_resume_hint():
    error = routes.MetadataRescanError("stopped after {...}; run the rescan again to resume")
    with patch.object(routes, "rescan_metadata", AsyncMock(side_effect=error)):
        response = _client(is_admin=True).post("/api/admin/metadata-security/rescan", json={})

    assert response.status_code == 503
    assert "run the rescan again" in response.json()["detail"]
