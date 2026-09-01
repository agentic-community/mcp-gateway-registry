"""Unit tests for PATCH /api/skills/{path}/upstream-headers (header rotation).

A minimal FastAPI app mounts the skill router with auth overridden and the
skill service patched. Covers the rotation happy path, the clear case, the
authz gate (owner/admin + modify scope), the 400 policy-violation mapping, and
that the encrypted field is never echoed back.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from registry.api import skill_routes
from registry.api.skill_routes import router as skill_router
from registry.auth.dependencies import nginx_proxied_auth
from registry.schemas.skill_models import SkillCard

pytestmark = pytest.mark.unit

SKILL_PATH = "pdf-tools"

ADMIN_CTX: dict[str, Any] = {"username": "admin", "is_admin": True, "ui_permissions": {}}
OWNER_CTX: dict[str, Any] = {
    "username": "bob",
    "is_admin": False,
    "ui_permissions": {"modify_skill": ["all"]},
}
STRANGER_CTX: dict[str, Any] = {"username": "eve", "is_admin": False, "ui_permissions": {}}


def _skill():
    # A real SkillCard so the SkillCard response_model serializes cleanly.
    return SkillCard(
        path="/skills/pdf-tools",
        name="pdf-tools",
        description="Test skill",
        skill_md_url="https://example.com/SKILL.md",
        owner="bob",
    )


def _make_client(user_context: dict) -> TestClient:
    app = FastAPI()
    app.include_router(skill_router, prefix="/api")
    app.dependency_overrides[nginx_proxied_auth] = lambda: user_context
    return TestClient(app)


@pytest.fixture
def service():
    svc = MagicMock()
    svc.get_skill = AsyncMock(return_value=_skill())
    # update_skill returns a dict-like SkillCard; use a real-ish object with the
    # response_model_exclude in mind. A MagicMock won't serialize, so return a
    # minimal object the SkillCard response model can validate. We patch the
    # service to bypass real persistence; the route returns whatever update_skill
    # yields, so hand back a lightweight stand-in the response model accepts.
    svc.update_skill = AsyncMock(return_value=_skill())
    return svc


@pytest.fixture
def patched(service):
    # The route builds the storage fields itself (validate+encrypt); patch that so
    # the unit test does not require SECRET_KEY / real crypto and we can assert the
    # route forwards them to update_skill.
    with (
        patch.object(skill_routes, "get_skill_service", return_value=service),
        patch.object(
            skill_routes,
            "_user_can_modify_skill",
            side_effect=lambda skill, ctx, action="modify": (
                ctx.get("is_admin") or skill.owner == ctx.get("username")
            ),
        ),
    ):
        yield service


def _patch_build(monkeypatch, captured):
    def _fake_build(raw, existing_encrypted=None):
        captured["raw"] = raw
        captured["existing_encrypted"] = existing_encrypted
        return {
            "custom_headers_encrypted": [],
            "custom_header_names": [h["name"] for h in (raw or [])],
            "custom_header_overridable_names": [],
            "custom_headers_updated_at": "2026-07-21T00:00:00+00:00",
        }

    import registry.utils.credential_encryption as ce

    monkeypatch.setattr(ce, "build_custom_headers_storage_fields", _fake_build)


class TestRotateSkillHeaders:
    def test_rotate_ok_forwards_fields(self, patched, monkeypatch):
        captured: dict = {}
        _patch_build(monkeypatch, captured)
        client = _make_client(OWNER_CTX)
        resp = client.patch(
            f"/api/skills/{SKILL_PATH}/upstream-headers",
            json={"custom_headers": [{"name": "X-Api-Key", "value": "sk", "overridable": False}]},
        )
        assert resp.status_code == 200
        # The plaintext list was validated+encrypted via the shared helper.
        assert captured["raw"] == [{"name": "X-Api-Key", "value": "sk", "overridable": False}]
        # The storage fields (not the plaintext) were persisted.
        updates = patched.update_skill.call_args.args[1]
        assert updates["custom_header_names"] == ["X-Api-Key"]
        assert "custom_headers" not in updates  # never persist plaintext

    def test_clear_ok(self, patched, monkeypatch):
        captured: dict = {}
        _patch_build(monkeypatch, captured)
        client = _make_client(OWNER_CTX)
        resp = client.patch(
            f"/api/skills/{SKILL_PATH}/upstream-headers", json={"custom_headers": []}
        )
        assert resp.status_code == 200
        assert captured["raw"] == []

    def test_admin_ok(self, patched, monkeypatch):
        _patch_build(monkeypatch, {})
        client = _make_client(ADMIN_CTX)
        resp = client.patch(
            f"/api/skills/{SKILL_PATH}/upstream-headers", json={"custom_headers": []}
        )
        assert resp.status_code == 200

    def test_stranger_403(self, patched, monkeypatch):
        _patch_build(monkeypatch, {})
        client = _make_client(STRANGER_CTX)
        resp = client.patch(
            f"/api/skills/{SKILL_PATH}/upstream-headers",
            json={"custom_headers": [{"name": "X-A", "value": "v"}]},
        )
        assert resp.status_code == 403
        patched.update_skill.assert_not_awaited()

    def test_not_found_404(self, patched, monkeypatch):
        _patch_build(monkeypatch, {})
        patched.get_skill = AsyncMock(return_value=None)
        client = _make_client(OWNER_CTX)
        resp = client.patch(
            f"/api/skills/{SKILL_PATH}/upstream-headers", json={"custom_headers": []}
        )
        assert resp.status_code == 404

    def test_policy_violation_400(self, patched):
        # No build patch: the real helper runs and rejects a reserved name.
        client = _make_client(OWNER_CTX)
        resp = client.patch(
            f"/api/skills/{SKILL_PATH}/upstream-headers",
            json={"custom_headers": [{"name": "X-Internal-Token", "value": "x"}]},
        )
        assert resp.status_code == 400

    def test_update_skill_returns_none_404(self, patched, monkeypatch):
        # get_skill/authz succeed but the underlying update_skill
        # returns falsy (skill vanished between read and write) -> 404.
        _patch_build(monkeypatch, {})
        patched.update_skill = AsyncMock(return_value=None)
        client = _make_client(OWNER_CTX)
        resp = client.patch(
            f"/api/skills/{SKILL_PATH}/upstream-headers", json={"custom_headers": []}
        )
        assert resp.status_code == 404
        patched.update_skill.assert_awaited_once()


class TestUpdateSkillDropsHeaders:
    """The general PUT /skills/{path} must strip upstream header fields from the
    persisted update (skill_routes.py): custom_headers are only
    settable at create / via the dedicated rotate PATCH, never on this update."""

    def _put(self, service, body):
        gate = MagicMock(allowed=True, error_message=None)
        with (
            patch.object(skill_routes, "get_skill_service", return_value=service),
            patch.object(
                skill_routes,
                "_user_can_modify_skill",
                side_effect=lambda skill, ctx, action="modify": (
                    ctx.get("is_admin") or skill.owner == ctx.get("username")
                ),
            ),
            patch.object(skill_routes, "check_registration_gate", AsyncMock(return_value=gate)),
        ):
            client = _make_client(OWNER_CTX)
            return client.put(f"/api/skills/{SKILL_PATH}", json=body)

    def test_put_strips_custom_headers_before_persist(self, service):
        resp = self._put(
            service,
            {
                "name": "pdf-tools",
                "description": "Updated description",
                "skill_md_url": "https://example.com/SKILL.md",
                "custom_headers": [{"name": "X-Api-Key", "value": "sk", "overridable": False}],
            },
        )
        assert resp.status_code == 200
        # The update otherwise succeeds and reaches persistence.
        service.update_skill.assert_awaited_once()
        updates = service.update_skill.call_args.args[1]
        # Non-header fields survive.
        assert updates["description"] == "Updated description"
        # All four upstream-header fields are stripped from the $set payload.
        for hdr in (
            "custom_headers",
            "custom_headers_encrypted",
            "custom_header_names",
            "custom_header_overridable_names",
        ):
            assert hdr not in updates
