"""Unit tests for registry/api/custom_entity_routes.py.

Generic record CRUD over a custom type. A minimal FastAPI app mounts the
router directly (production registers it behind a feature flag), with auth
overridden and the service patched. Covers path-param NoSQL guards (422 for
bad type/uuid), the list/get/create/update/delete happy paths, and the
domain-error -> HTTP status mapping (404 unknown-type / not-found, 409 cap,
400 validation).
"""

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from registry.api import custom_entity_routes
from registry.api.custom_entity_routes import router as custom_entity_router
from registry.auth.dependencies import nginx_proxied_auth
from registry.schemas.custom_entity_models import CustomEntityRecord
from registry.services.custom_entity_errors import (
    CustomEntityNotFoundError,
    CustomEntityValidationError,
    CustomTypeRecordCapError,
    UnknownCustomTypeError,
)

logger = logging.getLogger(__name__)

USER_CTX: dict[str, Any] = {"username": "bob", "is_admin": False, "groups": []}
VALID_UUID: str = str(uuid4())
TYPE: str = "workflow"


def _record(name: str = "r") -> CustomEntityRecord:
    rec = CustomEntityRecord(entity_type=TYPE, name=name, owner="bob")
    rec.path = f"/{TYPE}/{VALID_UUID}"
    return rec


def _make_client(user_context: dict | None) -> TestClient:
    app = FastAPI()
    app.include_router(custom_entity_router, prefix="/api")
    app.dependency_overrides[nginx_proxied_auth] = lambda: user_context
    return TestClient(app)


@pytest.fixture
def service() -> MagicMock:
    svc = MagicMock()
    svc.list_records = AsyncMock(return_value=([_record()], 1))
    svc.get_record = AsyncMock(return_value=_record())
    svc.create_record = AsyncMock(return_value=_record())
    svc.update_record = AsyncMock(return_value=_record("updated"))
    svc.delete_record = AsyncMock(return_value=None)
    return svc


@pytest.fixture
def patched_service(service):
    with patch.object(custom_entity_routes, "_get_service", return_value=service):
        yield service


@pytest.mark.unit
class TestList:
    def test_list_ok(self, patched_service):
        client = _make_client(USER_CTX)
        resp = client.get(f"/api/custom/{TYPE}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_count"] == 1
        assert body["records"][0]["entity_type"] == TYPE

    def test_unknown_type_404(self, patched_service):
        patched_service.list_records = AsyncMock(side_effect=UnknownCustomTypeError(TYPE))
        client = _make_client(USER_CTX)
        resp = client.get(f"/api/custom/{TYPE}")
        assert resp.status_code == 404

    def test_bad_type_pattern_422(self, patched_service):
        client = _make_client(USER_CTX)
        resp = client.get("/api/custom/Bad Type")
        assert resp.status_code == 422


@pytest.mark.unit
class TestGet:
    def test_get_ok(self, patched_service):
        client = _make_client(USER_CTX)
        resp = client.get(f"/api/custom/{TYPE}/{VALID_UUID}")
        assert resp.status_code == 200

    def test_not_found_404(self, patched_service):
        patched_service.get_record = AsyncMock(
            side_effect=CustomEntityNotFoundError(f"/{TYPE}/{VALID_UUID}")
        )
        client = _make_client(USER_CTX)
        resp = client.get(f"/api/custom/{TYPE}/{VALID_UUID}")
        assert resp.status_code == 404

    def test_bad_uuid_422(self, patched_service):
        client = _make_client(USER_CTX)
        resp = client.get(f"/api/custom/{TYPE}/not-a-uuid")
        assert resp.status_code == 422


@pytest.mark.unit
class TestCreate:
    def test_create_ok_owner_is_server_derived(self, patched_service):
        client = _make_client(USER_CTX)
        resp = client.post(f"/api/custom/{TYPE}", json={"name": "x", "owner": "hacker"})
        assert resp.status_code == 201
        # owner is passed from user_context, never the body.
        _, kwargs = patched_service.create_record.call_args
        assert kwargs["owner"] == "bob"

    def test_unknown_type_404(self, patched_service):
        patched_service.create_record = AsyncMock(side_effect=UnknownCustomTypeError(TYPE))
        client = _make_client(USER_CTX)
        resp = client.post(f"/api/custom/{TYPE}", json={"name": "x"})
        assert resp.status_code == 404

    def test_record_cap_409(self, patched_service):
        patched_service.create_record = AsyncMock(side_effect=CustomTypeRecordCapError(TYPE, 100))
        client = _make_client(USER_CTX)
        resp = client.post(f"/api/custom/{TYPE}", json={"name": "x"})
        assert resp.status_code == 409

    def test_validation_400(self, patched_service):
        patched_service.create_record = AsyncMock(
            side_effect=CustomEntityValidationError(errors=[{"field": "a", "message": "b"}])
        )
        client = _make_client(USER_CTX)
        resp = client.post(f"/api/custom/{TYPE}", json={"name": "x"})
        assert resp.status_code == 400


@pytest.mark.unit
class TestUpdateDelete:
    def test_update_ok(self, patched_service):
        client = _make_client(USER_CTX)
        resp = client.put(f"/api/custom/{TYPE}/{VALID_UUID}", json={"name": "updated"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "updated"

    def test_update_not_found_404(self, patched_service):
        patched_service.update_record = AsyncMock(
            side_effect=CustomEntityNotFoundError(f"/{TYPE}/{VALID_UUID}")
        )
        client = _make_client(USER_CTX)
        resp = client.put(f"/api/custom/{TYPE}/{VALID_UUID}", json={"name": "x"})
        assert resp.status_code == 404

    def test_delete_ok(self, patched_service):
        client = _make_client(USER_CTX)
        resp = client.delete(f"/api/custom/{TYPE}/{VALID_UUID}")
        assert resp.status_code == 204
        patched_service.delete_record.assert_awaited_once()

    def test_delete_not_found_404(self, patched_service):
        patched_service.delete_record = AsyncMock(
            side_effect=CustomEntityNotFoundError(f"/{TYPE}/{VALID_UUID}")
        )
        client = _make_client(USER_CTX)
        resp = client.delete(f"/api/custom/{TYPE}/{VALID_UUID}")
        assert resp.status_code == 404
