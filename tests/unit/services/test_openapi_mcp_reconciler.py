"""Tests for the OpenAPI MCP reconciler (issue #2022)."""

from unittest.mock import AsyncMock

import pytest

from registry.schemas.openapi_server_models import OpenApiServer
from registry.services import openapi_mcp_reconciler as reconciler
from registry.services import openapi_spec_checks
from registry.services.openapi_errors import container_unavailable


def _server(
    slug: str,
    generation: int,
) -> OpenApiServer:
    now = openapi_spec_checks.now_utc()
    return OpenApiServer(
        slug=slug,
        name=slug,
        path=f"/{slug}",
        generation=generation,
        created_by="admin",
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def env(monkeypatch):
    stored: list[OpenApiServer] = []
    repo = type("Repo", (), {"list_all": AsyncMock(side_effect=lambda: list(stored))})()
    mocks = {
        "status": AsyncMock(return_value={"servers": {}}),
        "delete": AsyncMock(),
        "push": AsyncMock(),
        "refresh": AsyncMock(),
        "health": AsyncMock(return_value={"status": "ok", "started_at": 100.0}),
    }
    monkeypatch.setattr(reconciler, "get_openapi_server_repository", lambda: repo)
    monkeypatch.setattr(reconciler.openapi_mcp_client, "status", mocks["status"])
    monkeypatch.setattr(reconciler.openapi_mcp_client, "delete_server", mocks["delete"])
    monkeypatch.setattr(reconciler.openapi_mcp_client, "health", mocks["health"])
    monkeypatch.setattr(reconciler, "push_server", mocks["push"])
    monkeypatch.setattr(reconciler.server_lifecycle, "refresh_managed_server", mocks["refresh"])
    monkeypatch.setattr(reconciler, "_container_reachable", False)
    return stored, mocks


class TestReconcileOnce:
    async def test_missing_server_is_pushed(self, env):
        stored, mocks = env
        stored.append(_server("weather", 3))

        result = await reconciler.reconcile_once()

        assert (result.pushed, result.deleted) == (1, 0)
        assert mocks["push"].call_args.kwargs["trigger"] == "reconcile"
        mocks["refresh"].assert_awaited_once()

    async def test_stale_generation_is_pushed_and_current_is_left(self, env):
        stored, mocks = env
        stored.extend([_server("old", 3), _server("current", 2)])
        mocks["status"].return_value = {
            "servers": {"old": {"generation": 2}, "current": {"generation": 2}}
        }

        result = await reconciler.reconcile_once()

        assert result.pushed == 1
        assert mocks["push"].call_args.args[0].slug == "old"

    async def test_extra_hub_is_deleted(self, env):
        _stored, mocks = env
        mocks["status"].return_value = {"servers": {"orphan": {"generation": 1}}}

        result = await reconciler.reconcile_once()

        assert result.deleted == 1
        mocks["delete"].assert_awaited_once_with("orphan")

    async def test_container_down_is_reported_not_raised(self, env):
        stored, mocks = env
        stored.append(_server("weather", 1))
        mocks["status"].side_effect = container_unavailable("down")

        result = await reconciler.reconcile_once()

        assert result.reachable is False
        mocks["push"].assert_not_awaited()
        assert reconciler.container_reachable() is False

    async def test_one_failed_push_does_not_stop_the_rest(self, env):
        stored, mocks = env
        stored.extend([_server("a", 1), _server("b", 1)])
        mocks["push"].side_effect = [container_unavailable("x"), None]

        result = await reconciler.reconcile_once()

        assert (result.pushed, result.failed) == (1, 1)

    async def test_counts_feed_the_gauge(self, env):
        stored, _mocks = env
        stored.extend([_server("a", 1), _server("b", 1)])

        await reconciler.reconcile_once()

        assert reconciler.stored_server_count() == 2


class TestRestartDetection:
    async def test_started_at_change_triggers_pass(self, env):
        _stored, mocks = env
        reconciler._container_reachable = True

        changed, started = await reconciler._health_changed(50.0)

        assert (changed, started) == (True, 100.0)

    async def test_same_started_at_is_quiet(self, env):
        reconciler._container_reachable = True

        changed, _started = await reconciler._health_changed(100.0)

        assert changed is False

    async def test_unreachable_health_marks_down(self, env):
        _stored, mocks = env
        mocks["health"].return_value = None

        changed, _started = await reconciler._health_changed(100.0)

        assert changed is False
        assert reconciler.container_reachable() is False
