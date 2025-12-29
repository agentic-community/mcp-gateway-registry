from __future__ import annotations

from fastapi import (
    HTTPException,
)
from fastapi.testclient import (
    TestClient,
)
import pytest

import auth_server.routes.validate_legacy as validate_legacy_module
import auth_server.server as auth_server_module


class TestValidateHttpExceptionPassthrough:
    def test_provider_http_exception_is_not_wrapped(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("ENFORCEAI_DB_PATH", raising=False)

        def _raise_http_exception() -> None:
            raise HTTPException(
                status_code=400,
                detail="Missing X-User-Pool-Id header",
                headers={"Connection": "close"},
            )

        monkeypatch.setattr(
            validate_legacy_module,
            "get_auth_provider",
            _raise_http_exception,
        )

        client = TestClient(auth_server_module.app)
        response = client.get(
            "/validate",
            headers={"Authorization": "Bearer token"},
        )

        assert response.status_code == 400
        assert response.json() == {"detail": "Missing X-User-Pool-Id header"}

