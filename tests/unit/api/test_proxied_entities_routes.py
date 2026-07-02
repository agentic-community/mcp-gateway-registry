"""Unit tests for the /api/iam/proxied-entities scope-editor endpoint (Task #11).

Verifies admin-gating and that the listing reuses the same list_proxied +
resolve_proxy_target gating as the render feed (federated/disabled/targetless
rows excluded), and emits the canonical authz_key the scope editor must write.
"""

import os
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-definitely-long-enough-32b")

from fastapi import HTTPException  # noqa: E402

from registry.api.proxied_entities_routes import _require_admin, list_proxied_entities  # noqa: E402

pytestmark = pytest.mark.unit


class TestRequireAdmin:
    def test_unauthenticated_401(self):
        with pytest.raises(HTTPException) as e:
            _require_admin(None)
        assert e.value.status_code == 401

    def test_non_admin_403(self):
        with pytest.raises(HTTPException) as e:
            _require_admin({"is_admin": False})
        assert e.value.status_code == 403

    def test_admin_passes(self):
        _require_admin({"is_admin": True})  # no raise


class TestListProxiedEntities:
    async def _call(self, agent_rows=None, skill_rows=None, custom_rows=None):
        agent_repo = AsyncMock()
        agent_repo.list_proxied.return_value = agent_rows or []
        skill_repo = AsyncMock()
        skill_repo.list_proxied.return_value = skill_rows or []
        custom_repo = AsyncMock()
        custom_repo.list_proxied.return_value = custom_rows or []
        with (
            patch(
                "registry.api.proxied_entities_routes.get_agent_repository",
                return_value=agent_repo,
            ),
            patch(
                "registry.api.proxied_entities_routes.get_skill_repository",
                return_value=skill_repo,
            ),
            patch(
                "registry.api.proxied_entities_routes.get_custom_entity_repository",
                return_value=custom_repo,
            ),
        ):
            return await list_proxied_entities(user_context={"is_admin": True})

    async def test_emits_canonical_authz_key(self):
        out = await self._call(
            skill_rows=[
                {
                    "path": "/skills/proxy-demo",
                    "is_proxied": True,
                    "proxy_target_url": "https://ok/",
                    "name": "Proxy Demo",
                }
            ]
        )
        assert out["count"] == 1
        item = out["proxied_entities"][0]
        assert item["entity_type"] == "skill"
        assert item["authz_key"] == "skill/skills/proxy-demo"  # canonical, not bare
        assert item["name"] == "Proxy Demo"

    async def test_custom_row_uses_its_own_type_token(self):
        out = await self._call(
            custom_rows=[
                {
                    "path": "/workflow/w1",
                    "entity_type": "workflow",
                    "is_proxied": True,
                    "proxy_target_url": "https://ok/",
                }
            ]
        )
        item = out["proxied_entities"][0]
        assert item["entity_type"] == "workflow"
        assert item["authz_key"] == "workflow/workflow/w1"
        assert item["name"] == "/workflow/w1"  # falls back to path

    async def test_federated_and_targetless_excluded(self):
        out = await self._call(
            skill_rows=[
                {
                    "path": "/skills/fed",
                    "is_proxied": True,
                    "proxy_target_url": "https://x/",
                    "sync_metadata": {"is_federated": True},
                },
                {"path": "/skills/none", "is_proxied": True},  # no target
                {
                    "path": "/skills/ok",
                    "is_proxied": True,
                    "proxy_target_url": "https://ok/",
                },
            ]
        )
        assert [i["path"] for i in out["proxied_entities"]] == ["/skills/ok"]

    async def test_one_repo_error_does_not_break_others(self):
        boom = AsyncMock()
        boom.list_proxied.side_effect = RuntimeError("db down")
        good = AsyncMock()
        good.list_proxied.return_value = [
            {"path": "/skills/ok", "is_proxied": True, "proxy_target_url": "https://ok/"}
        ]
        empty = AsyncMock()
        empty.list_proxied.return_value = []
        with (
            patch("registry.api.proxied_entities_routes.get_agent_repository", return_value=boom),
            patch("registry.api.proxied_entities_routes.get_skill_repository", return_value=good),
            patch(
                "registry.api.proxied_entities_routes.get_custom_entity_repository",
                return_value=empty,
            ),
        ):
            out = await list_proxied_entities(user_context={"is_admin": True})
        assert [i["path"] for i in out["proxied_entities"]] == ["/skills/ok"]

    async def test_non_admin_rejected_before_any_query(self):
        with pytest.raises(HTTPException) as e:
            await list_proxied_entities(user_context={"is_admin": False})
        assert e.value.status_code == 403
