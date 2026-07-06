"""Route-level unit tests for PATCH /servers/{path:path} (issue #1164).

Exercises ``patch_server_endpoint`` via a FastAPI TestClient with the auth
dependency overridden. Service-layer collaborators are mocked so these stay
unit-level. Only fields explicitly supplied in the patch body are changed.
"""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from registry.api.server_routes import router

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


def _existing(
    registered_by: str = "alice",
    updated_at: datetime | None = None,
    sync_metadata: dict[str, Any] | None = None,
    deployment: str = "remote",
    proxy_pass_url: str = "http://upstream:9000",
    auth_credential_encrypted: str | None = None,
    description: str = "old description",
    tags: list[str] | None = None,
    license: str = "MIT",
) -> dict[str, Any]:
    ts = updated_at or datetime(2026, 1, 1, tzinfo=UTC)
    info: dict[str, Any] = {
        "server_name": "Existing Server",
        "description": description,
        "path": "/test-server",
        "tags": list(tags) if tags is not None else ["old"],
        "license": license,
        "deployment": deployment,
        "registered_by": registered_by,
        "registered_at": ts.isoformat(),
        "updated_at": ts.isoformat(),
        "is_enabled": True,
        "version": "v1.0.0",
        "auth_scheme": "none",
    }
    if deployment == "remote":
        info["proxy_pass_url"] = proxy_pass_url
    if sync_metadata is not None:
        info["sync_metadata"] = sync_metadata
    if auth_credential_encrypted is not None:
        info["auth_credential_encrypted"] = auth_credential_encrypted
    return info


def _gate_allow() -> MagicMock:
    return MagicMock(allowed=True, error_message=None)


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.servers
class TestServerPatch:
    """Tests for PATCH /servers/{path:path}."""

    def test_patch_partial_update_happy_path(self, client):
        existing = _existing(
            description="old description",
            tags=["preserve-me"],
            license="Apache-2.0",
            proxy_pass_url="http://keep:9000",
        )
        # Fresh reflects only the description change.
        fresh = {**existing, "description": "patched"}

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
            )

        assert response.status_code == 200
        data = response.json()
        assert data["description"] == "patched"
        assert data["tags"] == ["preserve-me"]
        assert data["license"] == "Apache-2.0"
        assert data["proxy_pass_url"] == "http://keep:9000"

    def test_patch_audit_action_set_with_existing_server_name(self, client):
        existing = _existing()
        existing["server_name"] = "Original Name"
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
            patch("registry.api.server_routes.set_audit_action") as mock_audit,
        ):
            svc.get_server_info = AsyncMock(side_effect=[existing, existing])
            svc.update_server = AsyncMock(return_value=True)

            client.patch(
                "/servers/test-server",
                json={"server_name": "Different Name"},
            )

        mock_audit.assert_called_once()
        assert "Original Name" in mock_audit.call_args.kwargs["description"]
        assert "Different Name" not in mock_audit.call_args.kwargs["description"]

    def test_patch_audit_metadata_had_if_match_false_when_absent(self, client):
        existing = _existing()
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
            patch("registry.api.server_routes.set_audit_action") as mock_audit,
        ):
            svc.get_server_info = AsyncMock(side_effect=[existing, existing])
            svc.update_server = AsyncMock(return_value=True)

            client.patch("/servers/test-server", json={"description": "x"})

        assert mock_audit.call_args.kwargs["metadata"] == {"had_if_match": False}

    def test_patch_audit_metadata_had_if_match_true_when_present(self, client):
        ts_ms = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp() * 1000)
        if_match = f'W/"{ts_ms}"'
        existing = _existing()
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
            patch("registry.api.server_routes.set_audit_action") as mock_audit,
        ):
            svc.get_server_info = AsyncMock(side_effect=[existing, existing])
            svc.update_server = AsyncMock(return_value=True)

            client.patch(
                "/servers/test-server",
                json={"description": "x"},
                headers={"If-Match": if_match},
            )

        assert mock_audit.call_args.kwargs["metadata"] == {"had_if_match": True}

    def test_patch_400_empty_body(self, client):
        existing = _existing()
        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=True,
            ),
        ):
            svc.get_server_info = AsyncMock(return_value=existing)
            response = client.patch("/servers/test-server", json={})

        assert response.status_code == 400
        assert "Empty patch body" in response.json()["detail"]

    def test_patch_404_when_server_not_found(self, client):
        with patch("registry.api.server_routes.server_service") as svc:
            svc.get_server_info = AsyncMock(return_value=None)
            response = client.patch("/servers/nope", json={"description": "x"})

        assert response.status_code == 404

    def test_patch_403_when_not_owner(self, client):
        existing = _existing(registered_by="bob")
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
            )

        assert response.status_code == 403
        assert "only patch servers you registered" in response.json()["detail"]

    def test_patch_403_when_missing_modify_service(self, client):
        existing = _existing()
        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.auth.dependencies.user_has_ui_permission_for_service",
                return_value=False,
            ),
        ):
            svc.get_server_info = AsyncMock(return_value=existing)
            response = client.patch(
                "/servers/test-server",
                json={"description": "x"},
            )

        assert response.status_code == 403
        assert "permission" in response.json()["detail"].lower()

    def test_patch_federated_guard_blocks(self, client):
        existing = _existing(
            sync_metadata={"is_federated": True, "source_peer_id": "peer-1"},
        )
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
            )

        assert response.status_code == 403
        assert "peer-1" in response.json()["detail"]

    def test_patch_422_credential_fields_rejected(self, client):
        existing = _existing()
        with patch("registry.api.server_routes.server_service") as svc:
            svc.get_server_info = AsyncMock(return_value=existing)
            response = client.patch(
                "/servers/test-server",
                json={"auth_credential": "Bearer leaked"},
            )

        assert response.status_code == 422

    def test_patch_422_registrant_only_field_rejected(self, client):
        existing = _existing()
        with patch("registry.api.server_routes.server_service") as svc:
            svc.get_server_info = AsyncMock(return_value=existing)
            response = client.patch(
                "/servers/test-server",
                json={"registered_by": "not-allowed"},
            )

        # ServerCardPatch has extra="forbid", so registrant-only fields are
        # rejected at parse time before the model_validator runs. The error
        # body still mentions the offending field name.
        assert response.status_code == 422
        body = response.json()
        assert "registered_by" in str(body)

    def test_patch_credentials_preserved(self, client):
        existing = _existing(auth_credential_encrypted="ENC::keepme")
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
            svc.get_server_info = AsyncMock(side_effect=[existing, existing])
            svc.update_server = AsyncMock(return_value=True)

            response = client.patch(
                "/servers/test-server",
                json={"description": "patched"},
            )

        assert response.status_code == 200
        merged = svc.update_server.call_args[0][1]
        assert merged["auth_credential_encrypted"] == "ENC::keepme"

    def test_patch_proxy_url_change_triggers_scan(self, client):
        existing = _existing(proxy_pass_url="http://old:9000")
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
            patch(
                "registry.api.server_routes._perform_security_scan_on_registration",
                new_callable=AsyncMock,
            ) as mock_scan,
        ):
            svc.get_server_info = AsyncMock(side_effect=[existing, existing])
            svc.update_server = AsyncMock(return_value=True)

            response = client.patch(
                "/servers/test-server",
                json={"proxy_pass_url": "http://new:9000"},
            )

        assert response.status_code == 200
        mock_scan.assert_called_once()
        scan_args = mock_scan.call_args[0]
        assert scan_args[0] == "/test-server"
        assert scan_args[1] == "http://new:9000"

    def test_patch_no_proxy_url_change_no_scan(self, client):
        existing = _existing(proxy_pass_url="http://upstream:9000")
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
            patch(
                "registry.api.server_routes._perform_security_scan_on_registration",
                new_callable=AsyncMock,
            ) as mock_scan,
        ):
            svc.get_server_info = AsyncMock(side_effect=[existing, existing])
            svc.update_server = AsyncMock(return_value=True)

            response = client.patch(
                "/servers/test-server",
                json={"description": "just text"},
            )

        assert response.status_code == 200
        mock_scan.assert_not_called()

    def test_patch_etag_header_set_on_response(self, client):
        existing = _existing(updated_at=datetime(2026, 1, 1, tzinfo=UTC))
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
                json={"description": "x"},
            )

        assert response.status_code == 200
        etag = response.headers.get("ETag")
        assert etag is not None
        assert etag.startswith('W/"')
        assert etag.endswith('"')

    def test_patch_visibility_change_allowed_for_owner(self, client):
        existing = _existing()
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
            svc.get_server_info = AsyncMock(side_effect=[existing, existing])
            svc.update_server = AsyncMock(return_value=True)

            response = client.patch(
                "/servers/test-server",
                json={"visibility": "public"},
            )

        assert response.status_code == 200
        merged = svc.update_server.call_args[0][1]
        assert merged["visibility"] == "public"

    def test_patch_calls_server_service_update_server(self, client):
        existing = _existing()
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
            svc.get_server_info = AsyncMock(side_effect=[existing, existing])
            svc.update_server = AsyncMock(return_value=True)

            client.patch(
                "/servers/test-server",
                json={"description": "patched"},
            )

        svc.update_server.assert_called_once()
        called_path, called_dict = svc.update_server.call_args[0]
        assert called_path == "/test-server"
        assert called_dict["description"] == "patched"


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.servers
class TestServerPatchLifecycleStatusPermission:
    """PATCH status field is gated by change_lifecycle_status (Issue #1330)."""

    def test_status_change_denied_without_permission(self, client):
        """A non-admin without change_lifecycle_status gets 403 on a status change."""
        existing = _existing()
        existing["status"] = "draft"

        with (
            patch("registry.api.server_routes.server_service") as svc,
            # modify_service passes; change_lifecycle_status denied.
            patch(
                "registry.api.server_routes.user_can_change_lifecycle_status",
                return_value=False,
            ),
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
            svc.get_server_info = AsyncMock(side_effect=[existing, existing])
            svc.update_server = AsyncMock(return_value=True)

            response = client.patch("/servers/test-server", json={"status": "active"})

        assert response.status_code == 403
        assert "lifecycle status" in response.json()["detail"]

    def test_status_change_allowed_with_permission(self, client):
        """change_lifecycle_status permits the status change."""
        existing = _existing()
        existing["status"] = "draft"
        fresh = {**existing, "status": "active"}

        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.api.server_routes.user_can_change_lifecycle_status",
                return_value=True,
            ),
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

            response = client.patch("/servers/test-server", json={"status": "active"})

        assert response.status_code == 200
        assert response.json()["status"] == "active"

    def test_non_status_patch_does_not_require_permission(self, client):
        """A metadata-only patch never invokes the lifecycle status gate."""
        existing = _existing(description="old description")
        existing["status"] = "draft"
        fresh = {**existing, "description": "patched"}

        with (
            patch("registry.api.server_routes.server_service") as svc,
            patch(
                "registry.api.server_routes.user_can_change_lifecycle_status",
                return_value=False,
            ) as gate,
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
                "/servers/test-server", json={"description": "patched"}
            )

        assert response.status_code == 200
        gate.assert_not_called()
