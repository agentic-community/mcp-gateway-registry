"""End-to-end unit tests for the per-entity ``/check-duplicates`` endpoints.

Each route is exercised through a FastAPI ``TestClient`` with the
auth dependency and the four repository factories patched out. The
``DuplicateCheckService`` itself is NOT mocked — we want these tests
to confirm the wiring (request schema → service → response envelope)
end-to-end, including visibility filtering and the
``collision`` / ``advisory_matches`` partitioning.
"""

from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _admin_user_context() -> dict[str, Any]:
    return {
        "username": "admin",
        "groups": ["admin"],
        "is_admin": True,
        "accessible_servers": ["all"],
        "accessible_agents": ["all"],
        "ui_permissions": {
            "register_service": ["all"],
            "publish_agent": ["all"],
            "publish_skill": ["all"],
        },
    }


def _stub_repos(
    *,
    server_match: dict | None = None,
    agent_match: dict | None = None,
    skill_match: dict | None = None,
    search_results: dict | None = None,
):
    """Build mocks for the four repository factories.

    Returns a dict of (factory_name -> mock_repo) so the caller can
    patch the right factory in the right module.
    """
    server_repository = MagicMock()
    server_repository.find_by_identity_url = AsyncMock(return_value=server_match)
    agent_repository = MagicMock()
    agent_repository.find_by_identity_url = AsyncMock(return_value=agent_match)
    skill_repository = MagicMock()
    skill_repository.find_by_identity_url = AsyncMock(return_value=skill_match)
    search_repository = MagicMock()
    search_repository.search = AsyncMock(return_value=search_results or {})
    return {
        "get_server_repository": lambda: server_repository,
        "get_agent_repository": lambda: agent_repository,
        "get_skill_repository": lambda: skill_repository,
        "get_search_repository": lambda: search_repository,
    }


@pytest.fixture(autouse=True)
def _allow_all_visibility(monkeypatch):
    """Default: every visibility check passes. Tests can override.

    The agent helper is synchronous (it reads visibility from the
    candidate dict); server and skill helpers are coroutines.
    """

    async def _allow_async(*_args, **_kwargs) -> bool:
        return True

    def _allow_sync(*_args, **_kwargs) -> bool:
        return True

    monkeypatch.setattr(
        "registry.services.duplicate_check_service.user_can_access_server",
        _allow_async,
    )
    monkeypatch.setattr(
        "registry.services.duplicate_check_service.user_can_access_agent_from_doc",
        _allow_sync,
    )
    monkeypatch.setattr(
        "registry.services.duplicate_check_service.user_can_access_skill",
        _allow_async,
    )


@contextmanager
def _client_for_router(router_module: str, factories: dict, user_context: dict):
    """Yield a TestClient with auth + factories patched for one router.

    The repository factories are patched in three places:
      1. The route module itself, in case it imports any of them.
      2. The duplicate-check service module, which resolves the three
         entity repositories via the same factories at construction.
      3. The semantic-search-service module, which resolves the search
         repository via ``get_search_repository`` at construction.

    The DuplicateCheckService module singleton is reset before
    yielding (so each test sees a fresh service against the patched
    factories) and again on cleanup (so unrelated modules don't
    inherit a stale instance). Cleanup runs even if the test body
    raises.
    """
    from importlib import import_module

    from registry.services.duplicate_check_service import (
        reset_duplicate_check_service,
    )

    reset_duplicate_check_service()

    module = import_module(router_module)
    app = FastAPI()
    app.include_router(module.router)
    app.dependency_overrides[module.nginx_proxied_auth] = lambda: user_context

    patchers = []
    for factory_name, factory_fn in factories.items():
        if hasattr(module, factory_name):
            patchers.append(patch.object(module, factory_name, factory_fn))
        try:
            dedup_mod = import_module("registry.services.duplicate_check_service")
            if hasattr(dedup_mod, factory_name):
                patchers.append(patch.object(dedup_mod, factory_name, factory_fn))
        except ImportError:
            pass
        try:
            sem_mod = import_module("registry.services.semantic_search_service")
            if hasattr(sem_mod, factory_name):
                patchers.append(patch.object(sem_mod, factory_name, factory_fn))
        except ImportError:
            pass
    for patcher in patchers:
        patcher.start()

    try:
        yield TestClient(app)
    finally:
        for patcher in patchers:
            patcher.stop()
        app.dependency_overrides.clear()
        reset_duplicate_check_service()


class TestServerCheckDuplicatesEndpoint:
    def test_no_collision_no_advisory_returns_empty_envelope(self) -> None:
        factories = _stub_repos()
        with _client_for_router(
            "registry.api.server_routes",
            factories,
            _admin_user_context(),
        ) as client:
            response = client.post(
                "/servers/check-duplicates",
                json={
                    "name": "My Server",
                    "description": "Some server",
                    "proxy_pass_url": "https://api.example.com/mcp",
                },
            )
            assert response.status_code == 200
            body = response.json()
            assert body["collision_with"] == []
            assert body["advisory_matches"] == []
            assert body["similarity_search_available"] is True

    def test_url_collision_populates_collision_with(self) -> None:
        factories = _stub_repos(
            server_match={
                "path": "/foo",
                "server_name": "Foo",
                "registered_by": "team-a",
                "registered_at": "2026-04-12T10:30:00Z",
            }
        )
        with _client_for_router(
            "registry.api.server_routes",
            factories,
            _admin_user_context(),
        ) as client:
            response = client.post(
                "/servers/check-duplicates",
                json={
                    "name": "New Server",
                    "proxy_pass_url": "https://api.example.com/mcp",
                },
            )
            assert response.status_code == 200
            body = response.json()
            assert len(body["collision_with"]) >= 1
            assert body["collision_with"][0]["path"] == "/foo"
            assert body["collision_with"][0]["name"] == "Foo"
            assert body["collision_with"][0]["owner"] == "team-a"
            assert body["collision_with"][0]["match_reason"] == "exact URL match"
            assert body["advisory_matches"] == []

    def test_advisory_matches_only_when_no_url_collision(self) -> None:
        factories = _stub_repos(
            search_results={
                "servers": [
                    {
                        "path": "/sim",
                        "server_name": "Similar",
                        "relevance_score": 0.85,
                    }
                ]
            }
        )
        with _client_for_router(
            "registry.api.server_routes",
            factories,
            _admin_user_context(),
        ) as client:
            response = client.post(
                "/servers/check-duplicates",
                json={
                    "name": "My Server",
                    "description": "A server that does X",
                },
            )
            assert response.status_code == 200
            body = response.json()
            assert body["collision_with"] == []
            assert len(body["advisory_matches"]) == 1
            assert body["advisory_matches"][0]["path"] == "/sim"

    def test_both_collision_and_advisory_can_populate(self) -> None:
        factories = _stub_repos(
            server_match={"path": "/exact", "server_name": "Exact"},
            search_results={
                "servers": [
                    {
                        "path": "/sim",
                        "server_name": "Similar",
                        "relevance_score": 0.85,
                    }
                ]
            },
        )
        with _client_for_router(
            "registry.api.server_routes",
            factories,
            _admin_user_context(),
        ) as client:
            response = client.post(
                "/servers/check-duplicates",
                json={
                    "name": "New Server",
                    "description": "Server",
                    "proxy_pass_url": "https://api.example.com/mcp",
                },
            )
            assert response.status_code == 200
            body = response.json()
            assert len(body["collision_with"]) >= 1
            assert body["collision_with"][0]["path"] == "/exact"
            assert [match["path"] for match in body["advisory_matches"]] == ["/sim"]

    def test_blank_name_returns_422(self) -> None:
        factories = _stub_repos()
        with _client_for_router(
            "registry.api.server_routes",
            factories,
            _admin_user_context(),
        ) as client:
            response = client.post(
                "/servers/check-duplicates",
                json={"name": "   "},
            )
            assert response.status_code == 422

    def test_unauthenticated_caller_gets_403(self) -> None:
        factories = _stub_repos()
        # No register_service permission.
        unauthorized = {"username": "alice", "ui_permissions": {}}
        with _client_for_router(
            "registry.api.server_routes",
            factories,
            unauthorized,
        ) as client:
            response = client.post(
                "/servers/check-duplicates",
                json={"name": "Anything"},
            )
            assert response.status_code == 403

    def test_search_unavailable_returns_available_false(self) -> None:
        server_repository = MagicMock()
        server_repository.find_by_identity_url = AsyncMock(return_value=None)
        agent_repository = MagicMock()
        agent_repository.find_by_identity_url = AsyncMock(return_value=None)
        skill_repository = MagicMock()
        skill_repository.find_by_identity_url = AsyncMock(return_value=None)
        search_repository = MagicMock()
        search_repository.search = AsyncMock(side_effect=RuntimeError("search unavailable"))
        factories = {
            "get_server_repository": lambda: server_repository,
            "get_agent_repository": lambda: agent_repository,
            "get_skill_repository": lambda: skill_repository,
            "get_search_repository": lambda: search_repository,
        }
        with _client_for_router(
            "registry.api.server_routes",
            factories,
            _admin_user_context(),
        ) as client:
            response = client.post(
                "/servers/check-duplicates",
                json={"name": "X", "description": "Y"},
            )
            assert response.status_code == 200
            body = response.json()
            assert body["similarity_search_available"] is False
            assert body["advisory_matches"] == []


class TestAgentCheckDuplicatesEndpoint:
    def test_url_collision(self) -> None:
        factories = _stub_repos(
            agent_match={
                "path": "/agents/foo",
                "name": "Agent Foo",
                "registered_by": "alice",
            }
        )
        with _client_for_router(
            "registry.api.agent_routes",
            factories,
            _admin_user_context(),
        ) as client:
            response = client.post(
                "/agents/check-duplicates",
                json={
                    "name": "Agent Bar",
                    "url": "https://api.example.com/agent",
                },
            )
            assert response.status_code == 200
            body = response.json()
            assert len(body["collision_with"]) >= 1
            assert body["collision_with"][0]["path"] == "/agents/foo"
            assert body["collision_with"][0]["owner"] == "alice"

    def test_no_publish_agent_permission_returns_403(self) -> None:
        factories = _stub_repos()
        no_perm = {
            "username": "bob",
            "ui_permissions": {"publish_agent": []},
        }
        with _client_for_router(
            "registry.api.agent_routes",
            factories,
            no_perm,
        ) as client:
            response = client.post(
                "/agents/check-duplicates",
                json={"name": "Anything"},
            )
            assert response.status_code == 403


class TestSkillCheckDuplicatesEndpoint:
    def test_url_collision(self) -> None:
        factories = _stub_repos(
            skill_match={
                "path": "/skills/foo",
                "skill_name": "Foo Skill",
                "owner": "bob",
            }
        )
        with _client_for_router(
            "registry.api.skill_routes",
            factories,
            _admin_user_context(),
        ) as client:
            response = client.post(
                "/skills/check-duplicates",
                json={
                    "name": "My Skill",
                    "skill_md_url": "https://github.com/org/repo/blob/main/SKILL.md",
                },
            )
            assert response.status_code == 200
            body = response.json()
            assert len(body["collision_with"]) >= 1
            assert body["collision_with"][0]["path"] == "/skills/foo"
            assert body["collision_with"][0]["name"] == "Foo Skill"

    def test_skill_advisory_only(self) -> None:
        factories = _stub_repos(
            search_results={
                "skills": [
                    {
                        "path": "/skills/sim",
                        "skill_name": "Similar Skill",
                        "relevance_score": 0.88,
                    }
                ]
            }
        )
        with _client_for_router(
            "registry.api.skill_routes",
            factories,
            _admin_user_context(),
        ) as client:
            response = client.post(
                "/skills/check-duplicates",
                json={
                    "name": "My Skill",
                    "description": "Does X",
                },
            )
            assert response.status_code == 200
            body = response.json()
            assert body["collision_with"] == []
            assert len(body["advisory_matches"]) == 1
            assert body["advisory_matches"][0]["path"] == "/skills/sim"
