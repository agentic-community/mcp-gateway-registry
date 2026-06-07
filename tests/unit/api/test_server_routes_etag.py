"""ETag/If-Match concurrency tests for the server PUT/PATCH endpoints (issue #1164).

Mirrors ``test_agent_routes_etag.py`` style but exercises the routes via a
TestClient so we cover the full If-Match handshake (parse + compare + emit).
"""

import re
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from registry.api.server_routes import router

_WEAK_ETAG_RE = re.compile(r'^W/"\d+"$')


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def user_context() -> dict[str, Any]:
    return {
        "username": "alice",
        "groups": ["g"],
        "scopes": ["modify_service"],
        "auth_method": "session",
        "accessible_servers": ["all"],
        "accessible_services": ["all"],
        "ui_permissions": {
            "modify_service": ["all"],
        },
        "is_admin": False,
    }


@pytest.fixture
def client(user_context):
    app = FastAPI()
    app.include_router(router)

    from registry.api.server_routes import nginx_proxied_auth
    from registry.auth.csrf import verify_csrf_token_flexible

    app.dependency_overrides[nginx_proxied_auth] = lambda: user_context
    app.dependency_overrides[verify_csrf_token_flexible] = lambda: None
    yield TestClient(app)
    app.dependency_overrides.clear()


def _existing(updated_at: datetime, registered_by: str = "alice") -> dict[str, Any]:
    return {
        "server_name": "Etag Server",
        "description": "old description",
        "path": "/test-server",
        "tags": [],
        "license": "MIT",
        "deployment": "remote",
        "proxy_pass_url": "http://upstream:9000",
        "registered_by": registered_by,
        "registered_at": updated_at.isoformat(),
        "updated_at": updated_at.isoformat(),
        "is_enabled": True,
        "version": "v1.0.0",
        "auth_scheme": "none",
    }


def _etag_for(ts: datetime) -> str:
    return f'W/"{int(ts.timestamp() * 1000)}"'


def _valid_put_body(**overrides: Any) -> dict[str, Any]:
    body = {
        "server_name": "Etag Server",
        "description": "new description",
        "proxy_pass_url": "http://upstream:9000",
        "tags": [],
        "license": "MIT",
        "visibility": "public",
    }
    body.update(overrides)
    return body


def _gate_allow() -> MagicMock:
    return MagicMock(allowed=True, error_message=None)


def _patches():
    """Common patches used across tests."""
    return {
        "ui_perm": patch(
            "registry.auth.dependencies.user_has_ui_permission_for_service",
            return_value=True,
        ),
        "gate": patch(
            "registry.api.server_routes.check_registration_gate",
            AsyncMock(return_value=_gate_allow()),
        ),
        "webhook": patch("registry.api.server_routes.send_registration_webhook", AsyncMock()),
    }


# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.servers
class TestServerEtag:
    """ETag/If-Match concurrency tests for server PUT/PATCH."""

    @pytest.mark.skip(
        reason="GET /servers/{path} does not currently emit an ETag header. "
        "PUT/PATCH responses do — see test_etag_format_is_weak below."
    )
    def test_get_server_returns_etag_header(self, client):
        pass

    def test_put_with_matching_if_match_succeeds(self, client):
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        existing = _existing(updated_at=ts)
        fresh = {
            **existing,
            "updated_at": datetime(2026, 2, 2, tzinfo=UTC).isoformat(),
        }
        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
            patch(
                "registry.api.server_routes.check_registration_gate",
                AsyncMock(return_value=_gate_allow()),
            ),
            patch("registry.api.server_routes.send_registration_webhook", AsyncMock()),
        ):
            svc.get_server_info = AsyncMock(side_effect=[existing, fresh])
            svc.update_server = AsyncMock(return_value=True)

            response = client.put(
                "/servers/test-server",
                json=_valid_put_body(),
                headers={"If-Match": _etag_for(ts)},
            )

        assert response.status_code == 200

    def test_put_with_stale_if_match_returns_412(self, client):
        ts_current = datetime(2026, 6, 1, tzinfo=UTC)
        ts_stale = datetime(2025, 1, 1, tzinfo=UTC)
        existing = _existing(updated_at=ts_current)
        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
        ):
            svc.get_server_info = AsyncMock(return_value=existing)
            response = client.put(
                "/servers/test-server",
                json=_valid_put_body(),
                headers={"If-Match": _etag_for(ts_stale)},
            )

        assert response.status_code == 412

    def test_patch_with_matching_if_match_succeeds(self, client):
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        existing = _existing(updated_at=ts)
        fresh = {
            **existing,
            "updated_at": datetime(2026, 2, 2, tzinfo=UTC).isoformat(),
        }
        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
            patch(
                "registry.api.server_routes.check_registration_gate",
                AsyncMock(return_value=_gate_allow()),
            ),
            patch("registry.api.server_routes.send_registration_webhook", AsyncMock()),
        ):
            svc.get_server_info = AsyncMock(side_effect=[existing, fresh])
            svc.update_server = AsyncMock(return_value=True)

            response = client.patch(
                "/servers/test-server",
                json={"description": "patched"},
                headers={"If-Match": _etag_for(ts)},
            )

        assert response.status_code == 200

    def test_patch_with_stale_if_match_returns_412(self, client):
        ts_current = datetime(2026, 6, 1, tzinfo=UTC)
        ts_stale = datetime(2025, 1, 1, tzinfo=UTC)
        existing = _existing(updated_at=ts_current)
        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
        ):
            svc.get_server_info = AsyncMock(return_value=existing)
            response = client.patch(
                "/servers/test-server",
                json={"description": "patched"},
                headers={"If-Match": _etag_for(ts_stale)},
            )

        assert response.status_code == 412

    def test_etag_changes_on_each_update(self, client):
        """After PUT, the new ETag in the response differs from the prior one."""
        ts_old = datetime(2026, 1, 1, tzinfo=UTC)
        ts_new = datetime(2026, 6, 6, tzinfo=UTC)
        existing = _existing(updated_at=ts_old)
        fresh = {**existing, "updated_at": ts_new.isoformat()}

        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
            patch(
                "registry.api.server_routes.check_registration_gate",
                AsyncMock(return_value=_gate_allow()),
            ),
            patch("registry.api.server_routes.send_registration_webhook", AsyncMock()),
        ):
            svc.get_server_info = AsyncMock(side_effect=[existing, fresh])
            svc.update_server = AsyncMock(return_value=True)

            response = client.put("/servers/test-server", json=_valid_put_body())

        assert response.status_code == 200
        assert response.headers["ETag"] == _etag_for(ts_new)
        assert response.headers["ETag"] != _etag_for(ts_old)

    def test_etag_format_is_weak(self, client):
        ts = datetime(2026, 7, 7, tzinfo=UTC)
        existing = _existing(updated_at=ts)
        fresh = {**existing}

        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
            patch(
                "registry.api.server_routes.check_registration_gate",
                AsyncMock(return_value=_gate_allow()),
            ),
            patch("registry.api.server_routes.send_registration_webhook", AsyncMock()),
        ):
            svc.get_server_info = AsyncMock(side_effect=[existing, fresh])
            svc.update_server = AsyncMock(return_value=True)

            response = client.patch("/servers/test-server", json={"description": "x"})

        assert response.status_code == 200
        etag = response.headers["ETag"]
        assert _WEAK_ETAG_RE.match(etag), f"ETag {etag!r} is not weak form"

    def test_malformed_if_match_returns_400(self, client):
        existing = _existing(updated_at=datetime(2026, 1, 1, tzinfo=UTC))
        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
        ):
            svc.get_server_info = AsyncMock(return_value=existing)
            response = client.put(
                "/servers/test-server",
                json=_valid_put_body(),
                headers={"If-Match": "garbage"},
            )

        assert response.status_code == 400
        assert "Malformed If-Match" in response.json()["detail"]

    def test_strong_form_if_match_returns_400(self, client):
        existing = _existing(updated_at=datetime(2026, 1, 1, tzinfo=UTC))
        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
        ):
            svc.get_server_info = AsyncMock(return_value=existing)
            response = client.patch(
                "/servers/test-server",
                json={"description": "x"},
                headers={"If-Match": '"1700000000000"'},
            )

        assert response.status_code == 400
        assert "Strong ETag" in response.json()["detail"]
