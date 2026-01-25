"""
Unit tests for registry.api.health_routes module.

This module tests the health check endpoints including:
- EmbeddingHealthStatus model
- HealthResponse model
- health_check() endpoint
- embedding_health() endpoint
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from registry.api.health_routes import (
    EmbeddingHealthStatus,
    HealthResponse,
    record_embedding_load_time,
    get_embedding_load_time,
)


logger = logging.getLogger(__name__)


# =============================================================================
# TESTS: Pydantic Models
# =============================================================================


@pytest.mark.unit
class TestEmbeddingHealthStatus:
    """Tests for EmbeddingHealthStatus Pydantic model."""

    def test_model_creation(self):
        """Test creating EmbeddingHealthStatus with all fields."""
        # Arrange & Act
        status = EmbeddingHealthStatus(
            provider="fastembed",
            model="BAAI/bge-small-en-v1.5",
            dimensions=384,
            fallback_mode=True,
            load_time_ms=1523.45,
        )

        # Assert
        assert status.provider == "fastembed"
        assert status.model == "BAAI/bge-small-en-v1.5"
        assert status.dimensions == 384
        assert status.fallback_mode is True
        assert status.load_time_ms == 1523.45

    def test_model_creation_without_load_time(self):
        """Test creating EmbeddingHealthStatus without optional load_time_ms."""
        # Arrange & Act
        status = EmbeddingHealthStatus(
            provider="sentence-transformers",
            model="all-MiniLM-L6-v2",
            dimensions=384,
            fallback_mode=False,
        )

        # Assert
        assert status.provider == "sentence-transformers"
        assert status.model == "all-MiniLM-L6-v2"
        assert status.dimensions == 384
        assert status.fallback_mode is False
        assert status.load_time_ms is None

    def test_model_serialization(self):
        """Test that model serializes to correct JSON format."""
        # Arrange
        status = EmbeddingHealthStatus(
            provider="fastembed",
            model="BAAI/bge-small-en-v1.5",
            dimensions=384,
            fallback_mode=True,
            load_time_ms=1523.45,
        )

        # Act
        json_dict = status.model_dump()

        # Assert
        assert json_dict == {
            "provider": "fastembed",
            "model": "BAAI/bge-small-en-v1.5",
            "dimensions": 384,
            "fallback_mode": True,
            "load_time_ms": 1523.45,
        }


@pytest.mark.unit
class TestHealthResponse:
    """Tests for HealthResponse Pydantic model."""

    def test_model_creation_without_embeddings(self):
        """Test creating HealthResponse without embeddings."""
        # Arrange & Act
        response = HealthResponse(
            status="healthy",
        )

        # Assert
        assert response.status == "healthy"
        assert response.service == "mcp-gateway-registry"
        assert response.embeddings is None

    def test_model_creation_with_embeddings(self):
        """Test creating HealthResponse with embeddings."""
        # Arrange
        embeddings = EmbeddingHealthStatus(
            provider="fastembed",
            model="BAAI/bge-small-en-v1.5",
            dimensions=384,
            fallback_mode=True,
            load_time_ms=1523.45,
        )

        # Act
        response = HealthResponse(
            status="healthy",
            embeddings=embeddings,
        )

        # Assert
        assert response.status == "healthy"
        assert response.service == "mcp-gateway-registry"
        assert response.embeddings is not None
        assert response.embeddings.fallback_mode is True

    def test_model_serialization(self):
        """Test that model serializes to correct JSON format."""
        # Arrange
        embeddings = EmbeddingHealthStatus(
            provider="fastembed",
            model="BAAI/bge-small-en-v1.5",
            dimensions=384,
            fallback_mode=True,
            load_time_ms=1523.45,
        )
        response = HealthResponse(
            status="healthy",
            embeddings=embeddings,
        )

        # Act
        json_dict = response.model_dump()

        # Assert
        assert json_dict == {
            "status": "healthy",
            "service": "mcp-gateway-registry",
            "embeddings": {
                "provider": "fastembed",
                "model": "BAAI/bge-small-en-v1.5",
                "dimensions": 384,
                "fallback_mode": True,
                "load_time_ms": 1523.45,
            },
        }


# =============================================================================
# TESTS: Load Time Recording
# =============================================================================


@pytest.mark.unit
class TestLoadTimeRecording:
    """Tests for embedding load time recording."""

    def test_record_and_get_load_time(self):
        """Test recording and retrieving load time."""
        # Arrange
        load_time = 1523.45

        # Act
        record_embedding_load_time(load_time)
        result = get_embedding_load_time()

        # Assert
        assert result == load_time

    def test_get_load_time_returns_latest(self):
        """Test that get_embedding_load_time returns the latest recorded time."""
        # Arrange
        record_embedding_load_time(1000.0)
        record_embedding_load_time(2000.0)
        record_embedding_load_time(3000.0)

        # Act
        result = get_embedding_load_time()

        # Assert
        assert result == 3000.0


# =============================================================================
# TESTS: Health Check Endpoint
# =============================================================================


@pytest.mark.unit
class TestHealthCheckEndpoint:
    """Tests for health_check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_without_embedding_model(self):
        """Test health check when embedding model is not loaded."""
        # Arrange
        from registry.api.health_routes import health_check

        with patch("registry.api.health_routes.faiss_service") as mock_service:
            mock_service.embedding_model = None

            # Act
            response = await health_check()

            # Assert
            assert response.status == "healthy"
            assert response.embeddings is None

    @pytest.mark.asyncio
    async def test_health_check_with_primary_provider(self):
        """Test health check with primary embedding provider."""
        # Arrange
        from registry.api.health_routes import health_check

        mock_model = MagicMock()
        mock_model.get_embedding_dimension.return_value = 384

        with patch("registry.api.health_routes.faiss_service") as mock_service:
            mock_service.embedding_model = mock_model

            with patch("registry.api.health_routes.settings") as mock_settings:
                mock_settings.embeddings_provider = "sentence-transformers"
                mock_settings.embeddings_model_name = "all-MiniLM-L6-v2"
                mock_settings.embeddings_model_dimensions = 384

                # Act
                response = await health_check()

                # Assert
                assert response.status == "healthy"
                assert response.embeddings is not None
                assert response.embeddings.provider == "sentence-transformers"
                assert response.embeddings.model == "all-MiniLM-L6-v2"
                assert response.embeddings.dimensions == 384
                assert response.embeddings.fallback_mode is False

    @pytest.mark.asyncio
    async def test_health_check_with_fallback_mode(self):
        """Test health check when running in fallback mode."""
        # Arrange
        from registry.api.health_routes import health_check
        from registry.embeddings import FastEmbedClient

        mock_model = MagicMock(spec=FastEmbedClient)
        mock_model.get_embedding_dimension.return_value = 384

        with patch("registry.api.health_routes.faiss_service") as mock_service:
            mock_service.embedding_model = mock_model

            with patch("registry.api.health_routes.FastEmbedClient", FastEmbedClient):
                with patch("registry.api.health_routes.settings") as mock_settings:
                    mock_settings.embeddings_provider = "sentence-transformers"
                    mock_settings.embeddings_model_name = "all-MiniLM-L6-v2"
                    mock_settings.embeddings_fallback_model = "BAAI/bge-small-en-v1.5"
                    mock_settings.embeddings_model_dimensions = 384

                    # Act
                    response = await health_check()

                    # Assert
                    assert response.status == "healthy"
                    assert response.embeddings is not None
                    assert response.embeddings.provider == "fastembed"
                    assert response.embeddings.model == "BAAI/bge-small-en-v1.5"
                    assert response.embeddings.fallback_mode is True

    @pytest.mark.asyncio
    async def test_health_check_includes_load_time(self):
        """Test that health check includes load time if recorded."""
        # Arrange
        from registry.api.health_routes import health_check

        record_embedding_load_time(1500.0)

        mock_model = MagicMock()
        mock_model.get_embedding_dimension.return_value = 384

        with patch("registry.api.health_routes.faiss_service") as mock_service:
            mock_service.embedding_model = mock_model

            with patch("registry.api.health_routes.settings") as mock_settings:
                mock_settings.embeddings_provider = "sentence-transformers"
                mock_settings.embeddings_model_name = "all-MiniLM-L6-v2"
                mock_settings.embeddings_model_dimensions = 384

                # Act
                response = await health_check()

                # Assert
                assert response.embeddings is not None
                assert response.embeddings.load_time_ms == 1500.0
