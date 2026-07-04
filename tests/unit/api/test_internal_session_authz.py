"""Authorization regression tests for /api/internal/sessions/* endpoints.

These virtual-server session endpoints are reachable both by the nginx Lua
router (via the internal /_internal/sessions/ subrequest, which injects the
shared SECRET_KEY as X-Internal-Secret) AND, because FastAPI serves them under
the public /api/ proxy location, by any authenticated user and by anything that
reaches the app port directly. The validate_internal_session_secret dependency
is the real gate: a request without the matching X-Internal-Secret header is
rejected with 403.
"""

import pytest
from fastapi import status
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """TestClient over the internal router with SECRET_KEY configured."""
    monkeypatch.setenv("SECRET_KEY", "test-internal-secret")
    from fastapi import FastAPI

    from registry.api.internal_routes import router

    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


@pytest.mark.unit
class TestInternalSessionSecretGate:
    """Requests without the internal shared secret are rejected."""

    def test_create_client_session_without_secret_forbidden(self, client) -> None:
        response = client.post(
            "/api/internal/sessions/client",
            json={"user_id": "attacker", "virtual_server_path": "/virtual/x"},
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_validate_client_session_without_secret_forbidden(self, client) -> None:
        response = client.get("/api/internal/sessions/client/vs-abc")
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_get_backend_session_without_secret_forbidden(self, client) -> None:
        response = client.get("/api/internal/sessions/backend/vs-abc:loc")
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_put_backend_session_without_secret_forbidden(self, client) -> None:
        response = client.put(
            "/api/internal/sessions/backend/vs-abc:loc",
            json={
                "backend_session_id": "poisoned",
                "user_id": "attacker",
                "virtual_server_path": "/virtual/x",
            },
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_delete_backend_session_without_secret_forbidden(self, client) -> None:
        response = client.delete("/api/internal/sessions/backend/vs-abc:loc")
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_wrong_secret_forbidden(self, client) -> None:
        response = client.post(
            "/api/internal/sessions/client",
            json={"user_id": "attacker", "virtual_server_path": "/virtual/x"},
            headers={"X-Internal-Secret": "wrong-secret"},
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_correct_secret_passes_gate(self, client, monkeypatch) -> None:
        """With the correct secret the gate passes (reaching the handler).

        The backend session repository is mocked so the handler completes
        without real DB I/O; reaching it at all proves the request got PAST the
        403 authorization gate rather than being rejected.
        """
        from unittest.mock import AsyncMock

        import registry.api.internal_routes as internal_routes

        mock_repo = AsyncMock()
        mock_repo.create_client_session = AsyncMock(return_value=None)
        monkeypatch.setattr(internal_routes, "get_backend_session_repository", lambda: mock_repo)

        response = client.post(
            "/api/internal/sessions/client",
            json={"user_id": "router", "virtual_server_path": "/virtual/x"},
            headers={"X-Internal-Secret": "test-internal-secret"},
        )
        assert response.status_code != status.HTTP_403_FORBIDDEN
        assert response.status_code == status.HTTP_201_CREATED


@pytest.mark.unit
class TestOwnershipParamRequired:
    """The ownership-bound endpoints require user_id so the check cannot be
    silently skipped.

    Ownership binding is the session-hijacking control. If user_id were an
    optional query param defaulting to None, any caller that forgot it would
    silently fall back to existence-only validation (the original vulnerable
    behavior) with no error. Making it a required param turns that mistake into
    a loud 422 at request time. The Lua router always supplies it, so this is
    no behavior change on the live path.
    """

    _SECRET = {"X-Internal-Secret": "test-internal-secret"}

    @pytest.fixture
    def mock_repo(self, monkeypatch):
        from unittest.mock import AsyncMock

        import registry.api.internal_routes as internal_routes

        repo = AsyncMock()
        repo.validate_client_session = AsyncMock(return_value=True)
        repo.get_backend_session = AsyncMock(return_value="be-1")
        repo.delete_backend_session = AsyncMock(return_value=None)
        monkeypatch.setattr(internal_routes, "get_backend_session_repository", lambda: repo)
        return repo

    def test_validate_client_session_missing_user_id_is_422(self, client, mock_repo) -> None:
        response = client.get(
            "/api/internal/sessions/client/vs-abc",
            headers=self._SECRET,
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        mock_repo.validate_client_session.assert_not_awaited()

    def test_validate_client_session_with_user_id_passes(self, client, mock_repo) -> None:
        response = client.get(
            "/api/internal/sessions/client/vs-abc",
            params={"user_id": "alice", "virtual_server_path": "/virtual/a"},
            headers=self._SECRET,
        )
        assert response.status_code == status.HTTP_200_OK
        mock_repo.validate_client_session.assert_awaited_once_with(
            "vs-abc", user_id="alice", virtual_server_path="/virtual/a"
        )

    def test_get_backend_session_missing_user_id_is_422(self, client, mock_repo) -> None:
        response = client.get(
            "/api/internal/sessions/backend/vs-abc:loc",
            headers=self._SECRET,
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        mock_repo.get_backend_session.assert_not_awaited()

    def test_get_backend_session_with_user_id_passes(self, client, mock_repo) -> None:
        response = client.get(
            "/api/internal/sessions/backend/vs-abc:loc",
            params={"user_id": "alice"},
            headers=self._SECRET,
        )
        assert response.status_code == status.HTTP_200_OK
        mock_repo.get_backend_session.assert_awaited_once_with(
            client_session_id="vs-abc", backend_key="loc", user_id="alice"
        )

    def test_delete_backend_session_missing_user_id_is_422(self, client, mock_repo) -> None:
        response = client.delete(
            "/api/internal/sessions/backend/vs-abc:loc",
            headers=self._SECRET,
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        mock_repo.delete_backend_session.assert_not_awaited()

    def test_delete_backend_session_with_user_id_passes(self, client, mock_repo) -> None:
        response = client.delete(
            "/api/internal/sessions/backend/vs-abc:loc",
            params={"user_id": "alice"},
            headers=self._SECRET,
        )
        assert response.status_code == status.HTTP_200_OK
        mock_repo.delete_backend_session.assert_awaited_once_with(
            client_session_id="vs-abc", backend_key="loc", user_id="alice"
        )
