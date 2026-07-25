"""Unit tests for scripts/backfill-skill-list-scope.py.

The backfill grants list_skills:["all"] to mcp-registry-admin and triggers a
reload. A dry run must not write; an apply must merge the scope and reload. The
module is loaded by path because the filename uses hyphens (not importable as a
package).
"""

import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "backfill-skill-list-scope.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("backfill_skill_list_scope", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
class TestBackfill:
    @pytest.mark.asyncio
    async def test_apply_grants_list_skills_to_admin(self):
        module = _load_module()
        repo = MagicMock()
        repo.merge_ui_permissions = AsyncMock(return_value=True)
        reload = AsyncMock(return_value=True)

        with (
            patch("registry.repositories.factory.get_scope_repository", return_value=repo),
            patch("registry.services.scope_service.trigger_auth_server_reload", reload),
        ):
            result = await module._run_backfill(dry_run=False)

        assert result == {"granted": True}
        repo.merge_ui_permissions.assert_awaited_once_with(
            "mcp-registry-admin", {"list_skills": ["all"]}
        )
        reload.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_dry_run_does_not_write(self):
        module = _load_module()
        repo = MagicMock()
        repo.merge_ui_permissions = AsyncMock(return_value=True)
        reload = AsyncMock(return_value=True)

        with (
            patch("registry.repositories.factory.get_scope_repository", return_value=repo),
            patch("registry.services.scope_service.trigger_auth_server_reload", reload),
        ):
            result = await module._run_backfill(dry_run=True)

        assert result == {"granted": False}
        repo.merge_ui_permissions.assert_not_awaited()
        reload.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_apply_reports_failure_when_group_missing(self):
        module = _load_module()
        repo = MagicMock()
        repo.merge_ui_permissions = AsyncMock(return_value=False)  # group not found
        reload = AsyncMock(return_value=True)

        with (
            patch("registry.repositories.factory.get_scope_repository", return_value=repo),
            patch("registry.services.scope_service.trigger_auth_server_reload", reload),
        ):
            result = await module._run_backfill(dry_run=False)

        assert result == {"granted": False}
        # No reload if nothing was granted.
        reload.assert_not_awaited()
