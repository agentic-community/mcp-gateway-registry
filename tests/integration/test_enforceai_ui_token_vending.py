from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from auth_server import server as auth_server_module
from auth_server.enforceai.auth.dependency import (
    clear_enforceai_dependency_caches,
)
from auth_server.enforceai.tokens.ui_session import (
    verify_enforceai_ui_session_token,
)
from registry.auth.dependencies import (
    create_session_cookie,
)
from registry.core.config import (
    settings,
)
from registry.main import (
    app as registry_app,
)


def _reset_auth_server_caches() -> None:
    clear_enforceai_dependency_caches()
    auth_server_module._load_enforceai_runtime.cache_clear()


@pytest.mark.integration
class TestEnforceAIUITokenVending:
    def test_registry_token_endpoint_requires_csrf(
        self,
        test_client: TestClient,
    ) -> None:
        cookie_value = create_session_cookie("testuser")
        response = test_client.post(
            "/api/auth/enforceai/token",
            cookies={settings.session_cookie_name: cookie_value},
        )

        assert response.status_code == 403
        assert response.json()["detail"] == "Missing X-CSRF-Token"

    def test_registry_token_endpoint_mints_verifiable_token(
        self,
        test_client: TestClient,
    ) -> None:
        cookie_value = create_session_cookie("testuser")

        csrf = test_client.get(
            "/api/auth/csrf",
            cookies={settings.session_cookie_name: cookie_value},
        )
        assert csrf.status_code == 200
        csrf_token = csrf.json()["csrf_token"]

        response = test_client.post(
            "/api/auth/enforceai/token",
            cookies={settings.session_cookie_name: cookie_value},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert response.status_code == 200

        payload: dict[str, Any] = response.json()
        assert "access_token" in payload
        assert "expires_at" in payload

        claims = verify_enforceai_ui_session_token(
            payload["access_token"],
            secret_key=settings.secret_key,
        )
        assert claims.sub.startswith("local|")
        assert claims.sid

    def test_vended_token_can_call_auth_server_management_api(
        self,
        enforceai_env,
        enforceai_oidc_issuers_env_json: str,
        enforceai_sqlite_db_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        enforceai_env(
            {
                "ENFORCEAI_DB_PATH": str(enforceai_sqlite_db_path),
                "ENFORCEAI_AUTH_PROVIDER": "oidc",
                "OIDC_ISSUERS": enforceai_oidc_issuers_env_json,
            }
        )
        _reset_auth_server_caches()

        monkeypatch.setattr(settings, "enforceai_db_path", enforceai_sqlite_db_path)
        monkeypatch.setattr(
            auth_server_module.app.state,
            "session_secret_key",
            settings.secret_key,
            raising=False,
        )

        registry_client = TestClient(registry_app)
        cookie_value = create_session_cookie("testuser")

        csrf = registry_client.get(
            "/api/auth/csrf",
            cookies={settings.session_cookie_name: cookie_value},
        )
        assert csrf.status_code == 200
        csrf_token = csrf.json()["csrf_token"]

        token_response = registry_client.post(
            "/api/auth/enforceai/token",
            cookies={settings.session_cookie_name: cookie_value},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert token_response.status_code == 200
        access_token = token_response.json()["access_token"]

        auth_client = TestClient(auth_server_module.app)
        response = auth_client.get(
            "/enforceai/agents",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert response.status_code == 200

