"""Route-level checks that JSON-body write paths answer 422 for oversized metadata.

Issue #1811. The form-based server routes are covered in test_server_routes.py.
This module covers the JSON-body routes, where the limit lives on the request
model: server PUT and PATCH, agent register and PUT, and skill register and
PUT. Each test asserts the 422 names the broken limit and that the service
layer is never reached, so nothing is persisted.
"""

from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from registry.api import (
    agent_routes,
    server_routes,
    skill_routes,
)
from registry.auth.csrf import verify_csrf_token_flexible

_USER_CONTEXT: dict[str, Any] = {
    "username": "alice",
    "groups": ["g"],
    "scopes": ["modify_service"],
    "auth_method": "session",
    "accessible_servers": ["all"],
    "accessible_services": ["all"],
    "accessible_agents": ["all"],
    "ui_permissions": {"modify_service": ["all"], "publish_skill": ["all"]},
    "is_admin": True,
}


def _metadata_of_size(total_bytes: int) -> dict[str, str]:
    return {"k": "x" * (total_bytes - 8)}


def _metadata_of_depth(depth: int) -> dict[str, Any]:
    doc: dict[str, Any] = {"leaf": "value"}
    for _ in range(depth - 1):
        doc = {"n": doc}
    return doc


_OVER_LIMITS = [
    pytest.param(_metadata_of_size(10241), "10241 bytes", id="one-byte-over"),
    pytest.param(_metadata_of_depth(9), "nests 9 levels deep", id="nine-levels"),
]

# A skill is measured as stored ({"author", "version", "extra": {...}}), so the
# reported size and depth include the wrapper; match on the limit instead.
_SKILL_OVER_LIMITS = [
    pytest.param(_metadata_of_size(10241), "over the limit of 10240 bytes", id="one-byte-over"),
    pytest.param(_metadata_of_depth(9), "over the limit of 8", id="nine-levels"),
]


def _client(router, auth_dependency) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[auth_dependency] = lambda: _USER_CONTEXT
    app.dependency_overrides[verify_csrf_token_flexible] = lambda: None
    return TestClient(app)


def _error_text(response) -> str:
    """Flatten a FastAPI 422 body into one string for matching."""
    return " ".join(str(err.get("msg", "")) for err in response.json()["detail"])


@pytest.fixture
def server_client():
    return _client(server_routes.router, server_routes.nginx_proxied_auth)


@pytest.fixture
def agent_client():
    return _client(agent_routes.router, agent_routes.nginx_proxied_auth)


@pytest.fixture
def skill_client():
    return _client(skill_routes.router, skill_routes.nginx_proxied_auth)


@pytest.mark.unit
@pytest.mark.api
class TestServerJsonRoutes:
    _PUT_BODY = {
        "server_name": "Existing Server",
        "description": "new description",
        "proxy_pass_url": "http://upstream:9000",
    }

    @pytest.mark.parametrize(("metadata", "message"), _OVER_LIMITS)
    def test_put_rejects_metadata_over_the_limit(self, server_client, metadata, message):
        with patch.object(server_routes, "server_service") as svc:
            response = server_client.put(
                "/servers/test-server", json={**self._PUT_BODY, "metadata": metadata}
            )

        assert response.status_code == 422
        assert message in _error_text(response)
        svc.update_server.assert_not_called()

    @pytest.mark.parametrize(("metadata", "message"), _OVER_LIMITS)
    def test_patch_rejects_metadata_over_the_limit(self, server_client, metadata, message):
        with patch.object(server_routes, "server_service") as svc:
            response = server_client.patch("/servers/test-server", json={"metadata": metadata})

        assert response.status_code == 422
        assert message in _error_text(response)
        svc.update_server.assert_not_called()


@pytest.mark.unit
@pytest.mark.api
class TestAgentJsonRoutes:
    _BODY = {
        "name": "limits-agent",
        "url": "https://agent.example.net",
        "supportedProtocol": "a2a",
    }

    @pytest.mark.parametrize(("metadata", "message"), _OVER_LIMITS)
    def test_register_rejects_metadata_over_the_limit(self, agent_client, metadata, message):
        with patch.object(agent_routes, "agent_service") as svc:
            response = agent_client.post(
                "/agents/register", json={**self._BODY, "metadata": metadata}
            )

        assert response.status_code == 422
        assert message in _error_text(response)
        svc.register_agent.assert_not_called()

    @pytest.mark.parametrize(("metadata", "message"), _OVER_LIMITS)
    def test_put_rejects_metadata_over_the_limit(self, agent_client, metadata, message):
        with patch.object(agent_routes, "agent_service") as svc:
            response = agent_client.put(
                "/agents/limits-agent", json={**self._BODY, "metadata": metadata}
            )

        assert response.status_code == 422
        assert message in _error_text(response)
        svc.update_agent.assert_not_called()


@pytest.mark.unit
@pytest.mark.api
class TestSkillJsonRoutes:
    _BODY = {
        "name": "limits-skill",
        "description": "Metadata limits test",
        "skill_md_url": "https://github.com/org/skills/SKILL.md",
    }

    @pytest.mark.parametrize(("metadata", "message"), _SKILL_OVER_LIMITS)
    def test_register_rejects_metadata_over_the_limit(self, skill_client, metadata, message):
        with patch.object(skill_routes, "get_skill_service") as get_svc:
            response = skill_client.post("/skills", json={**self._BODY, "metadata": metadata})

        assert response.status_code == 422
        assert message in _error_text(response)
        get_svc.assert_not_called()

    @pytest.mark.parametrize(("metadata", "message"), _SKILL_OVER_LIMITS)
    def test_put_rejects_metadata_over_the_limit(self, skill_client, metadata, message):
        with patch.object(skill_routes, "get_skill_service") as get_svc:
            response = skill_client.put(
                "/skills/limits-skill", json={**self._BODY, "metadata": metadata}
            )

        assert response.status_code == 422
        assert message in _error_text(response)
        get_svc.assert_not_called()
