"""
Unit tests for registry.embeddings.client module.

This module tests the embeddings client abstraction including:
- EmbeddingsClient abstract base class
- SentenceTransformersClient implementation
- LiteLLMClient implementation
- create_embeddings_client() factory function
"""

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from registry.embeddings.client import (
    EmbeddingsClient,
    LiteLLMClient,
    SentenceTransformersClient,
    create_embeddings_client,
)

logger = logging.getLogger(__name__)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_sentence_transformer():
    """
    Create a mock Sentence Transformer model.

    Returns:
        Mock SentenceTransformer instance
    """
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    mock_model.get_sentence_embedding_dimension.return_value = 384
    return mock_model


@pytest.fixture
def mock_litellm_response():
    """
    Create a mock LiteLLM embedding response.

    Returns:
        Mock response dictionary
    """
    return {
        "data": [
            {"embedding": [0.1, 0.2, 0.3, 0.4], "index": 0},
            {"embedding": [0.5, 0.6, 0.7, 0.8], "index": 1},
        ]
    }


@pytest.fixture
def temp_model_dir(tmp_path: Path) -> Path:
    """
    Create a temporary model directory with mock model files.

    Args:
        tmp_path: Pytest temporary path fixture

    Returns:
        Path to temporary model directory
    """
    model_dir = tmp_path / "models" / "test-model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create a dummy file to make the directory non-empty
    (model_dir / "config.json").write_text('{"model_type": "test"}')

    return model_dir


@pytest.fixture
def empty_model_dir(tmp_path: Path) -> Path:
    """
    Create an empty model directory.

    Args:
        tmp_path: Pytest temporary path fixture

    Returns:
        Path to empty directory
    """
    model_dir = tmp_path / "models" / "empty-model"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


# =============================================================================
# TESTS: EmbeddingsClient Abstract Base Class
# =============================================================================


@pytest.mark.unit
@pytest.mark.search
class TestEmbeddingsClient:
    """Tests for EmbeddingsClient abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that EmbeddingsClient cannot be instantiated directly."""
        # Arrange & Act & Assert
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            EmbeddingsClient()

    def test_abstract_encode_method(self):
        """Test that encode method is abstract and must be implemented."""

        # Arrange
        class IncompleteClient(EmbeddingsClient):
            def get_embedding_dimension(self) -> int:
                return 384

        # Act & Assert
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteClient()

    def test_abstract_get_embedding_dimension_method(self):
        """Test that get_embedding_dimension method is abstract."""

        # Arrange
        class IncompleteClient(EmbeddingsClient):
            def encode(self, texts: list[str]) -> np.ndarray:
                return np.array([])

        # Act & Assert
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteClient()

    def test_concrete_implementation_works(self):
        """Test that concrete implementation can be instantiated."""

        # Arrange
        class ConcreteClient(EmbeddingsClient):
            def encode(self, texts: list[str]) -> np.ndarray:
                return np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

            def get_embedding_dimension(self) -> int:
                return 3

        # Act
        client = ConcreteClient()

        # Assert
        assert isinstance(client, EmbeddingsClient)
        assert client.get_embedding_dimension() == 3


# =============================================================================
# TESTS: SentenceTransformersClient
# =============================================================================


@pytest.mark.unit
@pytest.mark.search
class TestSentenceTransformersClient:
    """Tests for SentenceTransformersClient implementation."""

    def test_initialization(self):
        """Test SentenceTransformersClient initialization."""
        # Arrange
        model_name = "all-MiniLM-L6-v2"
        model_dir = Path("/tmp/models")
        cache_dir = Path("/tmp/cache")

        # Act
        client = SentenceTransformersClient(
            model_name=model_name,
            model_dir=model_dir,
            cache_dir=cache_dir,
        )

        # Assert
        assert client.model_name == model_name
        assert client.model_dir == model_dir
        assert client.cache_dir == cache_dir
        assert client._model is None
        assert client._dimension is None

    def test_initialization_minimal(self):
        """Test SentenceTransformersClient with minimal parameters."""
        # Arrange
        model_name = "all-MiniLM-L6-v2"

        # Act
        client = SentenceTransformersClient(model_name=model_name)

        # Assert
        assert client.model_name == model_name
        assert client.model_dir is None
        assert client.cache_dir is None

    def test_load_model_from_huggingface(self, mock_sentence_transformer):
        """Test loading model from Hugging Face Hub."""
        # Arrange
        with patch("sentence_transformers.SentenceTransformer") as mock_st_class:
            mock_st_class.return_value = mock_sentence_transformer
            client = SentenceTransformersClient(model_name="all-MiniLM-L6-v2")

            # Act
            client._load_model()

            # Assert
            mock_st_class.assert_called_once_with("all-MiniLM-L6-v2")
            assert client._model == mock_sentence_transformer
            assert client._dimension == 384

    def test_load_model_from_local_directory(self, mock_sentence_transformer, temp_model_dir):
        """Test loading model from local directory."""
        # Arrange
        with patch("sentence_transformers.SentenceTransformer") as mock_st_class:
            mock_st_class.return_value = mock_sentence_transformer
            client = SentenceTransformersClient(
                model_name="all-MiniLM-L6-v2",
                model_dir=temp_model_dir,
            )

            # Act
            client._load_model()

            # Assert
            mock_st_class.assert_called_once_with(str(temp_model_dir))
            assert client._model == mock_sentence_transformer
            assert client._dimension == 384

    def test_load_model_empty_local_directory(self, mock_sentence_transformer, empty_model_dir):
        """Test loading model when local directory exists but is empty."""
        # Arrange
        with patch("sentence_transformers.SentenceTransformer") as mock_st_class:
            mock_st_class.return_value = mock_sentence_transformer
            client = SentenceTransformersClient(
                model_name="all-MiniLM-L6-v2",
                model_dir=empty_model_dir,
            )

            # Act
            client._load_model()

            # Assert
            # Should fall back to downloading from Hugging Face
            mock_st_class.assert_called_once_with("all-MiniLM-L6-v2")
            assert client._model == mock_sentence_transformer

    def test_load_model_with_cache_dir(self, mock_sentence_transformer, tmp_path):
        """Test loading model with custom cache directory."""
        # Arrange
        cache_dir = tmp_path / "cache"
        with patch("sentence_transformers.SentenceTransformer") as mock_st_class:
            mock_st_class.return_value = mock_sentence_transformer
            client = SentenceTransformersClient(
                model_name="all-MiniLM-L6-v2",
                cache_dir=cache_dir,
            )

            # Act
            client._load_model()

            # Assert
            assert cache_dir.exists()
            assert client._model == mock_sentence_transformer

    def test_load_model_restores_environment_variable(self, mock_sentence_transformer, tmp_path):
        """Test that loading model restores original SENTENCE_TRANSFORMERS_HOME."""
        # Arrange
        original_value = "/original/path"
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = original_value
        cache_dir = tmp_path / "cache"

        with patch("sentence_transformers.SentenceTransformer") as mock_st_class:
            mock_st_class.return_value = mock_sentence_transformer
            client = SentenceTransformersClient(
                model_name="all-MiniLM-L6-v2",
                cache_dir=cache_dir,
            )

            # Act
            client._load_model()

            # Assert
            assert os.environ.get("SENTENCE_TRANSFORMERS_HOME") == original_value

        # Cleanup
        del os.environ["SENTENCE_TRANSFORMERS_HOME"]

    def test_load_model_removes_environment_variable_if_not_set(
        self, mock_sentence_transformer, tmp_path
    ):
        """Test that loading model removes env var if it wasn't set originally."""
        # Arrange
        if "SENTENCE_TRANSFORMERS_HOME" in os.environ:
            del os.environ["SENTENCE_TRANSFORMERS_HOME"]
        cache_dir = tmp_path / "cache"

        with patch("sentence_transformers.SentenceTransformer") as mock_st_class:
            mock_st_class.return_value = mock_sentence_transformer
            client = SentenceTransformersClient(
                model_name="all-MiniLM-L6-v2",
                cache_dir=cache_dir,
            )

            # Act
            client._load_model()

            # Assert
            assert "SENTENCE_TRANSFORMERS_HOME" not in os.environ

    def test_load_model_only_once(self, mock_sentence_transformer):
        """Test that model is only loaded once, not on subsequent calls."""
        # Arrange
        with patch("sentence_transformers.SentenceTransformer") as mock_st_class:
            mock_st_class.return_value = mock_sentence_transformer
            client = SentenceTransformersClient(model_name="all-MiniLM-L6-v2")

            # Act
            client._load_model()
            client._load_model()
            client._load_model()

            # Assert
            # Should only be called once
            assert mock_st_class.call_count == 1

    def test_load_model_failure(self):
        """Test handling of model loading failure."""
        # Arrange
        with patch("sentence_transformers.SentenceTransformer") as mock_st_class:
            mock_st_class.side_effect = Exception("Model not found")
            client = SentenceTransformersClient(model_name="invalid-model")

            # Act & Assert
            with pytest.raises(RuntimeError, match="Failed to load SentenceTransformer model"):
                client._load_model()

    def test_encode_single_text(self, mock_sentence_transformer):
        """Test encoding a single text."""
        # Arrange
        with patch("sentence_transformers.SentenceTransformer") as mock_st_class:
            mock_st_class.return_value = mock_sentence_transformer
            mock_sentence_transformer.encode.return_value = np.array(
                [[0.1, 0.2, 0.3]], dtype=np.float32
            )
            client = SentenceTransformersClient(model_name="all-MiniLM-L6-v2")

            # Act
            result = client.encode(["test text"])

            # Assert
            assert isinstance(result, np.ndarray)
            assert result.shape == (1, 3)
            assert result.dtype == np.float32
            mock_sentence_transformer.encode.assert_called_once_with(["test text"])

    def test_encode_multiple_texts(self, mock_sentence_transformer):
        """Test encoding multiple texts."""
        # Arrange
        with patch("sentence_transformers.SentenceTransformer") as mock_st_class:
            mock_st_class.return_value = mock_sentence_transformer
            mock_sentence_transformer.encode.return_value = np.array(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32
            )
            client = SentenceTransformersClient(model_name="all-MiniLM-L6-v2")
            texts = ["first text", "second text"]

            # Act
            result = client.encode(texts)

            # Assert
            assert isinstance(result, np.ndarray)
            assert result.shape == (2, 3)
            assert result.dtype == np.float32
            mock_sentence_transformer.encode.assert_called_once_with(texts)

    def test_encode_lazy_loads_model(self, mock_sentence_transformer):
        """Test that encode lazy loads the model if not already loaded."""
        # Arrange
        with patch("sentence_transformers.SentenceTransformer") as mock_st_class:
            mock_st_class.return_value = mock_sentence_transformer
            client = SentenceTransformersClient(model_name="all-MiniLM-L6-v2")
            assert client._model is None

            # Act
            client.encode(["test"])

            # Assert
            assert client._model is not None
            mock_st_class.assert_called_once()

    def test_encode_failure(self, mock_sentence_transformer):
        """Test handling of encoding failure."""
        # Arrange
        with patch("sentence_transformers.SentenceTransformer") as mock_st_class:
            mock_st_class.return_value = mock_sentence_transformer
            mock_sentence_transformer.encode.side_effect = Exception("Encoding error")
            client = SentenceTransformersClient(model_name="all-MiniLM-L6-v2")

            # Act & Assert
            with pytest.raises(RuntimeError, match="Failed to encode texts"):
                client.encode(["test"])

    def test_get_embedding_dimension(self, mock_sentence_transformer):
        """Test getting embedding dimension."""
        # Arrange
        with patch("sentence_transformers.SentenceTransformer") as mock_st_class:
            mock_st_class.return_value = mock_sentence_transformer
            client = SentenceTransformersClient(model_name="all-MiniLM-L6-v2")

            # Act
            dimension = client.get_embedding_dimension()

            # Assert
            assert dimension == 384
            mock_sentence_transformer.get_sentence_embedding_dimension.assert_called_once()

    def test_get_embedding_dimension_lazy_loads_model(self, mock_sentence_transformer):
        """Test that get_embedding_dimension lazy loads model if needed."""
        # Arrange
        with patch("sentence_transformers.SentenceTransformer") as mock_st_class:
            mock_st_class.return_value = mock_sentence_transformer
            client = SentenceTransformersClient(model_name="all-MiniLM-L6-v2")
            assert client._dimension is None

            # Act
            dimension = client.get_embedding_dimension()

            # Assert
            assert dimension == 384
            assert client._dimension == 384
            mock_st_class.assert_called_once()

    def test_get_embedding_dimension_cached(self, mock_sentence_transformer):
        """Test that dimension is cached after first load."""
        # Arrange
        with patch("sentence_transformers.SentenceTransformer") as mock_st_class:
            mock_st_class.return_value = mock_sentence_transformer
            client = SentenceTransformersClient(model_name="all-MiniLM-L6-v2")
            client._load_model()

            # Act
            dimension1 = client.get_embedding_dimension()
            dimension2 = client.get_embedding_dimension()

            # Assert
            assert dimension1 == 384
            assert dimension2 == 384
            # Should only load model once
            assert mock_st_class.call_count == 1


# =============================================================================
# TESTS: LiteLLMClient
# =============================================================================


@pytest.mark.unit
@pytest.mark.search
class TestLiteLLMClient:
    """Tests for LiteLLMClient implementation."""

    def test_initialization_minimal(self):
        """Test LiteLLMClient initialization with minimal parameters."""
        # Arrange & Act
        client = LiteLLMClient(model_name="openai/text-embedding-3-small")

        # Assert
        assert client.model_name == "openai/text-embedding-3-small"
        assert client.api_key is None
        assert client.api_base is None
        assert client.aws_region is None
        assert client._embedding_dimension is None
        assert client._validated_dimension is None

    def test_initialization_with_all_parameters(self):
        """Test LiteLLMClient initialization with all parameters."""
        # Arrange & Act
        client = LiteLLMClient(
            model_name="openai/text-embedding-3-small",
            api_key="test-api-key",
            api_base="https://api.test.com",
            aws_region="us-west-2",
            embedding_dimension=1536,
        )

        # Assert
        assert client.model_name == "openai/text-embedding-3-small"
        assert client.api_key == "test-api-key"
        assert client.api_base == "https://api.test.com"
        assert client.aws_region == "us-west-2"
        assert client._embedding_dimension == 1536

    def test_initialization_sets_aws_region_env_var(self):
        """Test that AWS region is set as environment variable."""
        # Arrange
        original_value = os.environ.get("AWS_REGION_NAME")

        try:
            # Act
            LiteLLMClient(
                model_name="bedrock/amazon.titan-embed-text-v1",
                aws_region="us-east-1",
            )

            # Assert
            assert os.environ.get("AWS_REGION_NAME") == "us-east-1"
        finally:
            # Cleanup
            if original_value:
                os.environ["AWS_REGION_NAME"] = original_value
            elif "AWS_REGION_NAME" in os.environ:
                del os.environ["AWS_REGION_NAME"]

    def test_set_api_key_env_openai(self):
        """Test setting OpenAI API key environment variable."""
        # Arrange
        original_value = os.environ.get("OPENAI_API_KEY")

        try:
            client = LiteLLMClient(
                model_name="openai/text-embedding-3-small",
                api_key="test-openai-key",
            )

            # Act
            client._set_api_key_env()

            # Assert
            assert os.environ.get("OPENAI_API_KEY") == "test-openai-key"
        finally:
            # Cleanup
            if original_value:
                os.environ["OPENAI_API_KEY"] = original_value
            elif "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]

    def test_set_api_key_env_cohere(self):
        """Test setting Cohere API key environment variable."""
        # Arrange
        original_value = os.environ.get("COHERE_API_KEY")

        try:
            client = LiteLLMClient(
                model_name="cohere/embed-english-v3.0",
                api_key="test-cohere-key",
            )

            # Act
            client._set_api_key_env()

            # Assert
            assert os.environ.get("COHERE_API_KEY") == "test-cohere-key"
        finally:
            # Cleanup
            if original_value:
                os.environ["COHERE_API_KEY"] = original_value
            elif "COHERE_API_KEY" in os.environ:
                del os.environ["COHERE_API_KEY"]

    def test_set_api_key_env_azure(self):
        """Test setting Azure API key environment variable."""
        # Arrange
        original_value = os.environ.get("AZURE_API_KEY")

        try:
            client = LiteLLMClient(
                model_name="azure/deployment-name",
                api_key="test-azure-key",
            )

            # Act
            client._set_api_key_env()

            # Assert
            assert os.environ.get("AZURE_API_KEY") == "test-azure-key"
        finally:
            # Cleanup
            if original_value:
                os.environ["AZURE_API_KEY"] = original_value
            elif "AZURE_API_KEY" in os.environ:
                del os.environ["AZURE_API_KEY"]

    def test_set_api_key_env_bedrock_skips(self):
        """Test that Bedrock does not set API key (uses AWS credential chain)."""
        # Arrange
        client = LiteLLMClient(
            model_name="bedrock/amazon.titan-embed-text-v1",
            api_key="should-not-be-used",
        )

        # Act
        client._set_api_key_env()

        # Assert
        # No BEDROCK_API_KEY should be set
        assert "BEDROCK_API_KEY" not in os.environ

    def test_encode_single_text(self, mock_litellm_response):
        """Test encoding a single text with LiteLLM."""
        # Arrange
        with patch("litellm.embedding") as mock_embedding:
            mock_embedding.return_value = mock_litellm_response
            client = LiteLLMClient(model_name="openai/text-embedding-3-small")

            # Act
            result = client.encode(["test text"])

            # Assert
            assert isinstance(result, np.ndarray)
            assert result.shape == (2, 4)  # 2 embeddings from mock response
            assert result.dtype == np.float32
            mock_embedding.assert_called_once_with(
                model="openai/text-embedding-3-small",
                input=["test text"],
            )

    def test_encode_multiple_texts(self, mock_litellm_response):
        """Test encoding multiple texts with LiteLLM."""
        # Arrange
        with patch("litellm.embedding") as mock_embedding:
            mock_embedding.return_value = mock_litellm_response
            client = LiteLLMClient(model_name="openai/text-embedding-3-small")
            texts = ["first text", "second text"]

            # Act
            result = client.encode(texts)

            # Assert
            assert isinstance(result, np.ndarray)
            assert result.dtype == np.float32
            mock_embedding.assert_called_once_with(
                model="openai/text-embedding-3-small",
                input=texts,
            )

    def test_encode_with_api_base(self, mock_litellm_response):
        """Test encoding with custom API base URL."""
        # Arrange
        with patch("litellm.embedding") as mock_embedding:
            mock_embedding.return_value = mock_litellm_response
            client = LiteLLMClient(
                model_name="openai/text-embedding-3-small",
                api_base="https://custom.api.com",
            )

            # Act
            client.encode(["test"])

            # Assert
            mock_embedding.assert_called_once_with(
                model="openai/text-embedding-3-small",
                input=["test"],
                api_base="https://custom.api.com",
            )

    def test_encode_validates_dimension(self, mock_litellm_response):
        """Test that encode validates embedding dimension on first call."""
        # Arrange
        with patch("litellm.embedding") as mock_embedding:
            mock_embedding.return_value = mock_litellm_response
            client = LiteLLMClient(
                model_name="openai/text-embedding-3-small",
                embedding_dimension=4,  # Matches mock response
            )

            # Act
            client.encode(["test"])

            # Assert
            assert client._validated_dimension == 4

    def test_encode_warns_on_dimension_mismatch(self, mock_litellm_response, caplog):
        """Test warning when dimension doesn't match expected."""
        # Arrange
        with patch("litellm.embedding") as mock_embedding:
            mock_embedding.return_value = mock_litellm_response
            client = LiteLLMClient(
                model_name="openai/text-embedding-3-small",
                embedding_dimension=1536,  # Doesn't match mock response (4)
            )

            # Act
            with caplog.at_level(logging.WARNING):
                client.encode(["test"])

            # Assert
            assert "Embedding dimension mismatch" in caplog.text

    def test_encode_caches_validated_dimension(self, mock_litellm_response):
        """Test that validated dimension is cached after first call."""
        # Arrange
        with patch("litellm.embedding") as mock_embedding:
            mock_embedding.return_value = mock_litellm_response
            client = LiteLLMClient(model_name="openai/text-embedding-3-small")

            # Act
            client.encode(["first"])
            first_dimension = client._validated_dimension

            client.encode(["second"])
            second_dimension = client._validated_dimension

            # Assert
            assert first_dimension == 4
            assert second_dimension == 4

    def test_encode_handles_api_error(self):
        """Test handling of API errors during encoding."""
        # Arrange
        with patch("litellm.embedding") as mock_embedding:
            mock_embedding.side_effect = Exception("API error")
            client = LiteLLMClient(model_name="openai/text-embedding-3-small")

            # Act & Assert
            with pytest.raises(RuntimeError, match="Failed to generate embeddings via LiteLLM"):
                client.encode(["test"])

    def test_get_embedding_dimension_from_validated(self, mock_litellm_response):
        """Test getting dimension from validated dimension (after encode)."""
        # Arrange
        with patch("litellm.embedding") as mock_embedding:
            mock_embedding.return_value = mock_litellm_response
            client = LiteLLMClient(model_name="openai/text-embedding-3-small")
            client.encode(["test"])  # Validates dimension

            # Act
            dimension = client.get_embedding_dimension()

            # Assert
            assert dimension == 4

    def test_get_embedding_dimension_from_config(self):
        """Test getting dimension from configured value."""
        # Arrange
        client = LiteLLMClient(
            model_name="openai/text-embedding-3-small",
            embedding_dimension=1536,
        )

        # Act
        dimension = client.get_embedding_dimension()

        # Assert
        assert dimension == 1536

    def test_get_embedding_dimension_makes_test_call(self, mock_litellm_response):
        """Test that dimension is determined via test call if not known."""
        # Arrange
        with patch("litellm.embedding") as mock_embedding:
            mock_embedding.return_value = mock_litellm_response
            client = LiteLLMClient(model_name="openai/text-embedding-3-small")

            # Act
            dimension = client.get_embedding_dimension()

            # Assert
            assert dimension == 4
            mock_embedding.assert_called_once_with(
                model="openai/text-embedding-3-small",
                input=["test"],
            )

    def test_get_embedding_dimension_test_call_failure(self):
        """Test error handling when test call fails."""
        # Arrange
        with patch("litellm.embedding") as mock_embedding:
            mock_embedding.side_effect = Exception("API error")
            client = LiteLLMClient(model_name="openai/text-embedding-3-small")

            # Act & Assert
            with pytest.raises(RuntimeError, match="Failed to determine embedding dimension"):
                client.get_embedding_dimension()


# =============================================================================
# TESTS: create_embeddings_client Factory Function
# =============================================================================


@pytest.mark.unit
@pytest.mark.search
class TestCreateEmbeddingsClient:
    """Tests for create_embeddings_client factory function."""

    def test_create_sentence_transformers_client(self, mock_sentence_transformer):
        """Test creating SentenceTransformersClient via factory."""
        # Arrange
        with patch("sentence_transformers.SentenceTransformer") as mock_st_class:
            mock_st_class.return_value = mock_sentence_transformer

            # Act
            client = create_embeddings_client(
                provider="sentence-transformers",
                model_name="all-MiniLM-L6-v2",
            )

            # Assert
            assert isinstance(client, SentenceTransformersClient)
            assert client.model_name == "all-MiniLM-L6-v2"

    def test_create_sentence_transformers_client_case_insensitive(self, mock_sentence_transformer):
        """Test that provider name is case-insensitive."""
        # Arrange
        with patch("sentence_transformers.SentenceTransformer") as mock_st_class:
            mock_st_class.return_value = mock_sentence_transformer

            # Act
            client = create_embeddings_client(
                provider="SENTENCE-TRANSFORMERS",
                model_name="all-MiniLM-L6-v2",
            )

            # Assert
            assert isinstance(client, SentenceTransformersClient)

    def test_create_sentence_transformers_client_with_dirs(
        self, mock_sentence_transformer, tmp_path
    ):
        """Test creating SentenceTransformersClient with directories."""
        # Arrange
        with patch("sentence_transformers.SentenceTransformer") as mock_st_class:
            mock_st_class.return_value = mock_sentence_transformer
            model_dir = tmp_path / "models"
            cache_dir = tmp_path / "cache"

            # Act
            client = create_embeddings_client(
                provider="sentence-transformers",
                model_name="all-MiniLM-L6-v2",
                model_dir=model_dir,
                cache_dir=cache_dir,
            )

            # Assert
            assert isinstance(client, SentenceTransformersClient)
            assert client.model_dir == model_dir
            assert client.cache_dir == cache_dir

    def test_create_litellm_client(self):
        """Test creating LiteLLMClient via factory."""
        # Arrange & Act
        client = create_embeddings_client(
            provider="litellm",
            model_name="openai/text-embedding-3-small",
        )

        # Assert
        assert isinstance(client, LiteLLMClient)
        assert client.model_name == "openai/text-embedding-3-small"

    def test_create_litellm_client_case_insensitive(self):
        """Test that provider name is case-insensitive for LiteLLM."""
        # Arrange & Act
        client = create_embeddings_client(
            provider="LITELLM",
            model_name="openai/text-embedding-3-small",
        )

        # Assert
        assert isinstance(client, LiteLLMClient)

    def test_create_litellm_client_with_parameters(self):
        """Test creating LiteLLMClient with all parameters."""
        # Arrange & Act
        client = create_embeddings_client(
            provider="litellm",
            model_name="bedrock/amazon.titan-embed-text-v1",
            api_key="test-key",
            api_base="https://api.test.com",
            aws_region="us-west-2",
            embedding_dimension=1536,
        )

        # Assert
        assert isinstance(client, LiteLLMClient)
        assert client.model_name == "bedrock/amazon.titan-embed-text-v1"
        assert client.api_key == "test-key"
        assert client.api_base == "https://api.test.com"
        assert client.aws_region == "us-west-2"
        assert client._embedding_dimension == 1536

    def test_create_litellm_client_requires_provider_prefix(self):
        """Test that LiteLLM requires provider prefix in model name."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Invalid model name for LiteLLM provider"):
            create_embeddings_client(
                provider="litellm",
                model_name="text-embedding-3-small",  # Missing "openai/" prefix
            )

    def test_create_litellm_client_error_message_helpful(self):
        """Test that error message provides helpful examples."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError) as exc_info:
            create_embeddings_client(
                provider="litellm",
                model_name="all-MiniLM-L6-v2",
            )

        error_message = str(exc_info.value)
        assert "openai/text-embedding-3-small" in error_message
        assert "bedrock/amazon.titan-embed-text-v1" in error_message
        assert "cohere/embed-english-v3.0" in error_message
        assert "EMBEDDINGS_PROVIDER=sentence-transformers" in error_message

    def test_create_unsupported_provider(self):
        """Test error with unsupported provider."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Unsupported embeddings provider: invalid"):
            create_embeddings_client(
                provider="invalid",
                model_name="some-model",
            )

    def test_create_unsupported_provider_lists_supported(self):
        """Test that error message lists supported providers."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError) as exc_info:
            create_embeddings_client(
                provider="invalid",
                model_name="some-model",
            )

        error_message = str(exc_info.value)
        assert "sentence-transformers" in error_message
        assert "litellm" in error_message
