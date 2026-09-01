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

TYPE: str = "workflow"
VALID_UUID: str = str(uuid4())

# A non-admin who holds the full per-type scope set for TYPE. Used for the
# happy-path tests below (the gate passes, then service behavior is exercised).
FULL_SCOPES: dict[str, list[str]] = {
    f"list_{TYPE}_entity": ["all"],
    f"create_{TYPE}_entity": ["all"],
    f"modify_{TYPE}_entity": ["all"],
    f"delete_{TYPE}_entity": ["all"],
}
USER_CTX: dict[str, Any] = {
    "username": "bob",
    "is_admin": False,
    "groups": [],
    "ui_permissions": FULL_SCOPES,
}
# Admin bypasses every gate via is_admin, regardless of ui_permissions.
ADMIN_CTX: dict[str, Any] = {
    "username": "admin",
    "is_admin": True,
    "groups": [],
    "ui_permissions": {},
}
# A non-admin with NO per-type scopes; every gate should deny (404/403).
NO_SCOPE_CTX: dict[str, Any] = {
    "username": "eve",
    "is_admin": False,
    "groups": [],
    "ui_permissions": {},
}


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
    svc.update_record_upstream_headers = AsyncMock(return_value=_record("rotated"))
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

    def test_rotate_headers_ok(self, patched_service):
        client = _make_client(USER_CTX)
        resp = client.patch(
            f"/api/custom/{TYPE}/{VALID_UUID}/upstream-headers",
            json={"custom_headers": [{"name": "X-Api-Key", "value": "sk"}]},
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "rotated"
        # The plaintext header list is forwarded to the service positionally.
        args = patched_service.update_record_upstream_headers.call_args.args
        assert [{"name": "X-Api-Key", "value": "sk"}] in args
        # And the encrypted field is never echoed back to the client.
        assert "custom_headers_encrypted" not in resp.json()

    def test_rotate_headers_empty_clears(self, patched_service):
        client = _make_client(USER_CTX)
        resp = client.patch(
            f"/api/custom/{TYPE}/{VALID_UUID}/upstream-headers", json={"custom_headers": []}
        )
        assert resp.status_code == 200

    def test_rotate_headers_validation_400(self, patched_service):
        patched_service.update_record_upstream_headers = AsyncMock(
            side_effect=CustomEntityValidationError("custom_headers", "managed by the gateway")
        )
        client = _make_client(USER_CTX)
        resp = client.patch(
            f"/api/custom/{TYPE}/{VALID_UUID}/upstream-headers",
            json={"custom_headers": [{"name": "Cookie", "value": "x"}]},
        )
        assert resp.status_code == 400

    def test_rotate_headers_unknown_type_404(self, patched_service):
        patched_service.update_record_upstream_headers = AsyncMock(
            side_effect=UnknownCustomTypeError(TYPE)
        )
        client = _make_client(USER_CTX)
        resp = client.patch(
            f"/api/custom/{TYPE}/{VALID_UUID}/upstream-headers",
            json={"custom_headers": [{"name": "X-Api-Key", "value": "sk"}]},
        )
        assert resp.status_code == 404

    def test_rotate_headers_not_found_404(self, patched_service):
        patched_service.update_record_upstream_headers = AsyncMock(
            side_effect=CustomEntityNotFoundError(f"/{TYPE}/{VALID_UUID}")
        )
        client = _make_client(USER_CTX)
        resp = client.patch(
            f"/api/custom/{TYPE}/{VALID_UUID}/upstream-headers",
            json={"custom_headers": [{"name": "X-Api-Key", "value": "sk"}]},
        )
        assert resp.status_code == 404

    def test_rotate_headers_no_scope_403(self, patched_service):
        client = _make_client(NO_SCOPE_CTX)
        resp = client.patch(
            f"/api/custom/{TYPE}/{VALID_UUID}/upstream-headers", json={"custom_headers": []}
        )
        assert resp.status_code == 403
        patched_service.update_record_upstream_headers.assert_not_awaited()

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


@pytest.mark.unit
class TestPerTypeScopeGates:
    """Per-type authorization gates: read hides the type (404), mutate 403.

    Each gate is checked for a non-holder (deny), a scope-holder (pass), and an
    admin (catch-all pass). Admin bypasses via is_admin even with empty
    ui_permissions.
    """

    # ── Read gate: no list scope hides the type entirely (404) ──
    def test_list_no_scope_404_hides_type(self, patched_service):
        client = _make_client(NO_SCOPE_CTX)
        resp = client.get(f"/api/custom/{TYPE}")
        assert resp.status_code == 404
        patched_service.list_records.assert_not_awaited()

    def test_list_holder_ok(self, patched_service):
        client = _make_client(USER_CTX)
        assert client.get(f"/api/custom/{TYPE}").status_code == 200

    def test_list_admin_ok(self, patched_service):
        client = _make_client(ADMIN_CTX)
        assert client.get(f"/api/custom/{TYPE}").status_code == 200

    def test_get_no_scope_404(self, patched_service):
        client = _make_client(NO_SCOPE_CTX)
        resp = client.get(f"/api/custom/{TYPE}/{VALID_UUID}")
        assert resp.status_code == 404
        patched_service.get_record.assert_not_awaited()

    def test_get_admin_ok(self, patched_service):
        client = _make_client(ADMIN_CTX)
        assert client.get(f"/api/custom/{TYPE}/{VALID_UUID}").status_code == 200

    # ── Create gate: the PRIMARY GAP being closed (403 for non-holder) ──
    def test_create_no_scope_403(self, patched_service):
        client = _make_client(NO_SCOPE_CTX)
        resp = client.post(f"/api/custom/{TYPE}", json={"name": "x"})
        assert resp.status_code == 403
        patched_service.create_record.assert_not_awaited()

    def test_create_holder_ok(self, patched_service):
        client = _make_client(USER_CTX)
        assert client.post(f"/api/custom/{TYPE}", json={"name": "x"}).status_code == 201

    def test_create_admin_ok(self, patched_service):
        client = _make_client(ADMIN_CTX)
        assert client.post(f"/api/custom/{TYPE}", json={"name": "x"}).status_code == 201

    def test_create_list_only_scope_still_403(self, patched_service):
        # Holding only the read scope must NOT permit create.
        ctx = {**NO_SCOPE_CTX, "ui_permissions": {f"list_{TYPE}_entity": ["all"]}}
        client = _make_client(ctx)
        resp = client.post(f"/api/custom/{TYPE}", json={"name": "x"})
        assert resp.status_code == 403

    # ── Update / delete gates ──
    def test_update_no_scope_403(self, patched_service):
        client = _make_client(NO_SCOPE_CTX)
        resp = client.put(f"/api/custom/{TYPE}/{VALID_UUID}", json={"name": "x"})
        assert resp.status_code == 403
        patched_service.update_record.assert_not_awaited()

    def test_delete_no_scope_403(self, patched_service):
        client = _make_client(NO_SCOPE_CTX)
        resp = client.delete(f"/api/custom/{TYPE}/{VALID_UUID}")
        assert resp.status_code == 403
        patched_service.delete_record.assert_not_awaited()

    # ── Rating routes are view-gated (require list scope) ──
    def test_rate_no_scope_404(self, patched_service):
        patched_service.update_rating = AsyncMock(return_value=4.0)
        client = _make_client(NO_SCOPE_CTX)
        resp = client.post(f"/api/custom/{TYPE}/{VALID_UUID}/rate", json={"rating": 4})
        assert resp.status_code == 404

    def test_get_rating_no_scope_404(self, patched_service):
        patched_service.get_rating = AsyncMock(return_value={"num_stars": 0})
        client = _make_client(NO_SCOPE_CTX)
        resp = client.get(f"/api/custom/{TYPE}/{VALID_UUID}/rating")
        assert resp.status_code == 404
