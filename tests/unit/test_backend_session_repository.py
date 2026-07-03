"""Unit tests for backend session Pydantic models and internal API routes."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from registry.schemas.backend_session_models import (
    BackendSessionDocument,
    ClientSessionDocument,
    CreateClientSessionRequest,
    CreateClientSessionResponse,
    GetBackendSessionResponse,
    StoreSessionRequest,
)


class TestBackendSessionDocument:
    """Tests for BackendSessionDocument model."""

    def test_valid_document(self):
        """Test creating a valid backend session document."""
        doc = BackendSessionDocument(
            client_session_id="vs-abc123",
            backend_key="/_vs_backend_weather_",
            backend_session_id="backend-sess-xyz",
            user_id="admin",
            virtual_server_path="/virtual/my-server",
        )
        assert doc.client_session_id == "vs-abc123"
        assert doc.backend_key == "/_vs_backend_weather_"
        assert doc.backend_session_id == "backend-sess-xyz"
        assert doc.user_id == "admin"
        assert doc.virtual_server_path == "/virtual/my-server"
        assert doc.created_at is not None
        assert doc.last_used_at is not None

    def test_default_timestamps(self):
        """Test that created_at and last_used_at are set by default."""
        doc = BackendSessionDocument(
            client_session_id="vs-abc123",
            backend_key="/_vs_backend_weather_",
            backend_session_id="backend-sess-xyz",
            user_id="admin",
            virtual_server_path="/virtual/my-server",
        )
        assert isinstance(doc.created_at, datetime)
        assert isinstance(doc.last_used_at, datetime)
        assert doc.created_at.tzinfo is not None

    def test_custom_timestamps(self):
        """Test providing custom timestamps."""
        now = datetime.now(UTC)
        doc = BackendSessionDocument(
            client_session_id="vs-abc123",
            backend_key="/_vs_backend_weather_",
            backend_session_id="backend-sess-xyz",
            user_id="admin",
            virtual_server_path="/virtual/my-server",
            created_at=now,
            last_used_at=now,
        )
        assert doc.created_at == now
        assert doc.last_used_at == now

    def test_requires_client_session_id(self):
        """Test that client_session_id is required."""
        with pytest.raises(ValidationError):
            BackendSessionDocument(
                backend_key="/_vs_backend_weather_",
                backend_session_id="backend-sess-xyz",
                user_id="admin",
                virtual_server_path="/virtual/my-server",
            )

    def test_requires_backend_session_id(self):
        """Test that backend_session_id is required."""
        with pytest.raises(ValidationError):
            BackendSessionDocument(
                client_session_id="vs-abc123",
                backend_key="/_vs_backend_weather_",
                user_id="admin",
                virtual_server_path="/virtual/my-server",
            )

    def test_serialization_roundtrip(self):
        """Test JSON serialization and deserialization."""
        doc = BackendSessionDocument(
            client_session_id="vs-abc123",
            backend_key="/_vs_backend_weather_",
            backend_session_id="backend-sess-xyz",
            user_id="admin",
            virtual_server_path="/virtual/my-server",
        )
        json_data = doc.model_dump(mode="json")
        restored = BackendSessionDocument(**json_data)
        assert restored.client_session_id == doc.client_session_id
        assert restored.backend_session_id == doc.backend_session_id


class TestClientSessionDocument:
    """Tests for ClientSessionDocument model."""

    def test_valid_document(self):
        """Test creating a valid client session document."""
        doc = ClientSessionDocument(
            client_session_id="vs-abc123",
            user_id="admin",
            virtual_server_path="/virtual/my-server",
        )
        assert doc.client_session_id == "vs-abc123"
        assert doc.user_id == "admin"
        assert doc.virtual_server_path == "/virtual/my-server"
        assert doc.created_at is not None
        assert doc.last_used_at is not None

    def test_requires_client_session_id(self):
        """Test that client_session_id is required."""
        with pytest.raises(ValidationError):
            ClientSessionDocument(
                user_id="admin",
                virtual_server_path="/virtual/my-server",
            )

    def test_serialization_roundtrip(self):
        """Test JSON serialization and deserialization."""
        doc = ClientSessionDocument(
            client_session_id="vs-abc123",
            user_id="admin",
            virtual_server_path="/virtual/my-server",
        )
        json_data = doc.model_dump(mode="json")
        restored = ClientSessionDocument(**json_data)
        assert restored.client_session_id == doc.client_session_id
        assert restored.user_id == doc.user_id


class TestStoreSessionRequest:
    """Tests for StoreSessionRequest model."""

    def test_valid_request(self):
        """Test creating a valid store session request."""
        req = StoreSessionRequest(
            backend_session_id="backend-sess-xyz",
            client_session_id="vs-abc123",
            user_id="admin",
            virtual_server_path="/virtual/my-server",
        )
        assert req.backend_session_id == "backend-sess-xyz"
        assert req.client_session_id == "vs-abc123"

    def test_requires_user_id(self):
        """user_id is required: a session must have a concrete owner.

        Defaulting it (previously "anonymous") would silently store a
        wrong-owner document if a caller forgot to set it -- the same
        silent-fallback footgun removed from the ownership-bound endpoints.
        """
        with pytest.raises(ValidationError):
            StoreSessionRequest(
                backend_session_id="backend-sess-xyz",
                client_session_id="vs-abc123",
            )

    def test_default_virtual_server_path(self):
        """Test that virtual_server_path defaults to empty string."""
        req = StoreSessionRequest(
            backend_session_id="backend-sess-xyz",
            client_session_id="vs-abc123",
            user_id="admin",
        )
        assert req.virtual_server_path == ""

    def test_requires_backend_session_id(self):
        """Test that backend_session_id is required."""
        with pytest.raises(ValidationError):
            StoreSessionRequest(
                client_session_id="vs-abc123",
                user_id="admin",
            )

    def test_requires_client_session_id(self):
        """Test that client_session_id is required."""
        with pytest.raises(ValidationError):
            StoreSessionRequest(
                backend_session_id="backend-sess-xyz",
                user_id="admin",
            )


class TestCreateClientSessionRequest:
    """Tests for CreateClientSessionRequest model."""

    def test_valid_request(self):
        """Test creating a valid create client session request."""
        req = CreateClientSessionRequest(
            user_id="admin",
            virtual_server_path="/virtual/my-server",
        )
        assert req.user_id == "admin"
        assert req.virtual_server_path == "/virtual/my-server"

    def test_requires_user_id(self):
        """user_id is required: refuse to mint a session with no concrete owner."""
        with pytest.raises(ValidationError):
            CreateClientSessionRequest()

    def test_default_virtual_server_path(self):
        """virtual_server_path still defaults to empty string."""
        req = CreateClientSessionRequest(user_id="admin")
        assert req.virtual_server_path == ""


class TestCreateClientSessionResponse:
    """Tests for CreateClientSessionResponse model."""

    def test_valid_response(self):
        """Test creating a valid response."""
        resp = CreateClientSessionResponse(
            client_session_id="vs-abc123",
        )
        assert resp.client_session_id == "vs-abc123"

    def test_requires_client_session_id(self):
        """Test that client_session_id is required."""
        with pytest.raises(ValidationError):
            CreateClientSessionResponse()


class TestGetBackendSessionResponse:
    """Tests for GetBackendSessionResponse model."""

    def test_valid_response(self):
        """Test creating a valid response."""
        resp = GetBackendSessionResponse(
            backend_session_id="backend-sess-xyz",
        )
        assert resp.backend_session_id == "backend-sess-xyz"

    def test_requires_backend_session_id(self):
        """Test that backend_session_id is required."""
        with pytest.raises(ValidationError):
            GetBackendSessionResponse()


class TestBackendSessionInternalAPI:
    """Tests for internal API routes using mock repository."""

    @pytest.fixture
    def mock_repo(self):
        """Create a mock backend session repository."""
        mock = AsyncMock()
        mock.create_client_session = AsyncMock()
        mock.validate_client_session = AsyncMock(return_value=True)
        mock.get_backend_session = AsyncMock(return_value="backend-sess-xyz")
        mock.store_backend_session = AsyncMock()
        mock.delete_backend_session = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_create_client_session_generates_vs_id(self, mock_repo):
        """Test that create_client_session generates vs-<uuid> ID."""
        with patch(
            "registry.api.internal_routes.get_backend_session_repository",
            return_value=mock_repo,
        ):
            from registry.api.internal_routes import create_client_session

            request = CreateClientSessionRequest(
                user_id="admin",
                virtual_server_path="/virtual/my-server",
            )
            response = await create_client_session(request)

            assert response.client_session_id.startswith("vs-")
            assert len(response.client_session_id) > 3
            mock_repo.create_client_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_client_session_found(self, mock_repo):
        """Test validate returns 200 for existing session."""
        mock_repo.validate_client_session.return_value = True

        with patch(
            "registry.api.internal_routes.get_backend_session_repository",
            return_value=mock_repo,
        ):
            from registry.api.internal_routes import validate_client_session

            result = await validate_client_session("vs-abc123", user_id="admin")
            assert result == {"status": "valid"}

    @pytest.mark.asyncio
    async def test_validate_client_session_not_found(self, mock_repo):
        """Test validate raises 404 for missing session."""
        mock_repo.validate_client_session.return_value = False

        with patch(
            "registry.api.internal_routes.get_backend_session_repository",
            return_value=mock_repo,
        ):
            from fastapi import HTTPException

            from registry.api.internal_routes import validate_client_session

            with pytest.raises(HTTPException) as exc_info:
                await validate_client_session("vs-nonexistent", user_id="admin")
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_validate_client_session_forwards_owner(self, mock_repo):
        """The authenticated user identity is forwarded to the repo for ownership check.

        The router supplies the validated user via the user_id query param; the
        endpoint must pass it through so the repository can reject sessions owned
        by a different user (session-hijacking fix).
        """
        mock_repo.validate_client_session.return_value = True

        with patch(
            "registry.api.internal_routes.get_backend_session_repository",
            return_value=mock_repo,
        ):
            from registry.api.internal_routes import validate_client_session

            result = await validate_client_session(
                "vs-abc123", user_id="alice", virtual_server_path="/virtual/a"
            )
            assert result == {"status": "valid"}
            mock_repo.validate_client_session.assert_awaited_once_with(
                "vs-abc123", user_id="alice", virtual_server_path="/virtual/a"
            )

    @pytest.mark.asyncio
    async def test_validate_client_session_wrong_owner_404(self, mock_repo):
        """A session that does not belong to the caller is rejected with 404.

        The repo returns False when the stored owner does not match the
        authenticated user, so the attacker presenting a victim's session ID
        gets the same 404 as a non-existent session.
        """
        mock_repo.validate_client_session.return_value = False

        with patch(
            "registry.api.internal_routes.get_backend_session_repository",
            return_value=mock_repo,
        ):
            from fastapi import HTTPException

            from registry.api.internal_routes import validate_client_session

            with pytest.raises(HTTPException) as exc_info:
                await validate_client_session("vs-victim", user_id="attacker")
            assert exc_info.value.status_code == 404
            mock_repo.validate_client_session.assert_awaited_once_with(
                "vs-victim", user_id="attacker", virtual_server_path=None
            )

    @pytest.mark.asyncio
    async def test_get_backend_session_found(self, mock_repo):
        """Test get returns backend session ID."""
        mock_repo.get_backend_session.return_value = "backend-sess-xyz"

        with patch(
            "registry.api.internal_routes.get_backend_session_repository",
            return_value=mock_repo,
        ):
            from registry.api.internal_routes import get_backend_session

            result = await get_backend_session("vs-abc123:/_vs_backend_weather_", user_id="admin")
            assert result.backend_session_id == "backend-sess-xyz"

    @pytest.mark.asyncio
    async def test_get_backend_session_forwards_owner(self, mock_repo):
        """The authenticated user is forwarded to the repo for the owner check.

        Defense in depth: the backend-session lookup is bound to the caller so
        a single missed client-session gate cannot leak another user's live
        backend session ID.
        """
        mock_repo.get_backend_session.return_value = "backend-sess-xyz"

        with patch(
            "registry.api.internal_routes.get_backend_session_repository",
            return_value=mock_repo,
        ):
            from registry.api.internal_routes import get_backend_session

            result = await get_backend_session("vs-abc123:/_vs_backend_weather_", user_id="alice")
            assert result.backend_session_id == "backend-sess-xyz"
            mock_repo.get_backend_session.assert_awaited_once_with(
                client_session_id="vs-abc123",
                backend_key="/_vs_backend_weather_",
                user_id="alice",
            )

    @pytest.mark.asyncio
    async def test_get_backend_session_not_found(self, mock_repo):
        """Test get raises 404 for missing session."""
        mock_repo.get_backend_session.return_value = None

        with patch(
            "registry.api.internal_routes.get_backend_session_repository",
            return_value=mock_repo,
        ):
            from fastapi import HTTPException

            from registry.api.internal_routes import get_backend_session

            with pytest.raises(HTTPException) as exc_info:
                await get_backend_session("vs-abc123:/_vs_backend_weather_", user_id="admin")
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_backend_session_invalid_key(self, mock_repo):
        """Test get raises 400 for invalid session key format."""
        with patch(
            "registry.api.internal_routes.get_backend_session_repository",
            return_value=mock_repo,
        ):
            from fastapi import HTTPException

            from registry.api.internal_routes import get_backend_session

            with pytest.raises(HTTPException) as exc_info:
                await get_backend_session("no-colon-in-key", user_id="admin")
            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_store_backend_session(self, mock_repo):
        """Test store session calls repository correctly."""
        with patch(
            "registry.api.internal_routes.get_backend_session_repository",
            return_value=mock_repo,
        ):
            from registry.api.internal_routes import store_backend_session

            request = StoreSessionRequest(
                backend_session_id="backend-sess-xyz",
                client_session_id="vs-abc123",
                user_id="admin",
                virtual_server_path="/virtual/my-server",
            )
            result = await store_backend_session(
                "vs-abc123:/_vs_backend_weather_",
                request,
            )
            assert result == {"status": "stored"}
            mock_repo.store_backend_session.assert_called_once_with(
                client_session_id="vs-abc123",
                backend_key="/_vs_backend_weather_",
                backend_session_id="backend-sess-xyz",
                user_id="admin",
                virtual_server_path="/virtual/my-server",
            )

    @pytest.mark.asyncio
    async def test_delete_backend_session(self, mock_repo):
        """Test delete session calls repository correctly."""
        with patch(
            "registry.api.internal_routes.get_backend_session_repository",
            return_value=mock_repo,
        ):
            from registry.api.internal_routes import delete_backend_session

            result = await delete_backend_session(
                "vs-abc123:/_vs_backend_weather_", user_id="alice"
            )
            assert result == {"status": "deleted"}
            mock_repo.delete_backend_session.assert_called_once_with(
                client_session_id="vs-abc123",
                backend_key="/_vs_backend_weather_",
                user_id="alice",
            )

    @pytest.mark.asyncio
    async def test_repo_unavailable_returns_503(self):
        """Test that 503 is returned when repo is None."""
        with patch(
            "registry.api.internal_routes.get_backend_session_repository",
            return_value=None,
        ):
            from fastapi import HTTPException

            from registry.api.internal_routes import create_client_session

            request = CreateClientSessionRequest(user_id="admin")
            with pytest.raises(HTTPException) as exc_info:
                await create_client_session(request)
            assert exc_info.value.status_code == 503
