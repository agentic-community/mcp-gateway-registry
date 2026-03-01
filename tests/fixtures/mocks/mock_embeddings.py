"""
Mock embeddings implementation for testing.

This module provides mock implementations of embedding models to avoid
loading large ML models during tests.
"""

import hashlib
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class MockEmbeddingsClient:
    """
    Mock embeddings client that generates deterministic embeddings from text.

    This mock generates embeddings based on text hash to ensure consistent
    results across test runs without requiring real ML models.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        """
        Initialize mock embeddings client.

        Args:
            model_name: Name of the model (for logging)
            dimension: Dimension of the embeddings to generate
        """
        self.model_name = model_name
        self.dimension = dimension
        logger.debug(f"Created MockEmbeddingsClient: {model_name}, dim={dimension}")

    def encode(
        self,
        texts: str | list[str],
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Generate mock embeddings for input texts.

        Creates deterministic embeddings based on text hash to ensure
        consistency in tests.

        Args:
            texts: Single text string or list of texts
            normalize_embeddings: Whether to normalize the embeddings
            show_progress_bar: Whether to show progress (ignored)
            **kwargs: Additional arguments (ignored)

        Returns:
            Array of embeddings (shape: [n, dimension])
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            # Generate deterministic embedding from text hash
            embedding = self._generate_embedding(text)
            embeddings.append(embedding)

        result = np.array(embeddings, dtype=np.float32)

        if normalize_embeddings:
            # L2 normalization
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            result = result / norms

        logger.debug(f"Generated {len(texts)} mock embeddings, shape={result.shape}")
        return result

    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate a deterministic embedding from text.

        Uses hash of the text to seed random generation for consistency.

        Args:
            text: Input text

        Returns:
            Embedding vector (shape: [dimension])
        """
        # Use hash of text as seed for reproducibility
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        seed = int(text_hash[:8], 16)

        # Generate deterministic "embedding"
        rng = np.random.RandomState(seed)
        embedding = rng.randn(self.dimension).astype(np.float32)

        # Normalize to make it more realistic
        embedding = embedding / np.linalg.norm(embedding)

        return embedding


class MockSentenceTransformer:
    """
    Mock SentenceTransformer class for testing.

    Mimics the interface of sentence_transformers.SentenceTransformer.
    """

    def __init__(self, model_name_or_path: str, **kwargs: Any):
        """
        Initialize mock sentence transformer.

        Args:
            model_name_or_path: Model name or path
            **kwargs: Additional arguments (ignored)
        """
        self.model_name = model_name_or_path
        self.dimension = 384  # Default dimension for MiniLM
        self._client = MockEmbeddingsClient(model_name_or_path, self.dimension)
        logger.debug(f"Created MockSentenceTransformer: {model_name_or_path}")

    def encode(self, sentences: str | list[str], **kwargs: Any) -> np.ndarray:
        """
        Encode sentences to embeddings.

        Args:
            sentences: Single sentence or list of sentences
            **kwargs: Additional arguments passed to client

        Returns:
            Array of embeddings
        """
        return self._client.encode(sentences, **kwargs)

    def get_sentence_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.dimension


def create_mock_st_module() -> Any:
    """
    Create a mock sentence_transformers module for testing.

    Returns:
        Mock sentence_transformers module object
    """

    class MockSTModule:
        """Mock sentence_transformers module."""

        SentenceTransformer = MockSentenceTransformer

    return MockSTModule()


def create_mock_litellm_module() -> Any:
    """
    Create a mock litellm module for testing.

    Returns:
        Mock litellm module object
    """

    class MockLiteLLMModule:
        """Mock litellm module."""

        class MockEmbedding:
            """Mock embedding response."""

            def __init__(self, embedding: list[float]):
                self.embedding = embedding

        class MockEmbeddingResponse:
            """Mock embedding API response."""

            def __init__(self, embeddings: list[list[float]]):
                self.data = [{"embedding": emb, "index": i} for i, emb in enumerate(embeddings)]

        @staticmethod
        def embedding(
            model: str, input: str | list[str], **kwargs: Any
        ) -> "MockLiteLLMModule.MockEmbeddingResponse":
            """
            Mock LiteLLM embedding function.

            Args:
                model: Model name
                input: Text or list of texts
                **kwargs: Additional arguments

            Returns:
                Mock embedding response
            """
            if isinstance(input, str):
                input = [input]

            client = MockEmbeddingsClient(model, dimension=1024)
            embeddings_array = client.encode(input)
            embeddings = [emb.tolist() for emb in embeddings_array]

            logger.debug(f"Mock LiteLLM generated {len(embeddings)} embeddings")
            return MockLiteLLMModule.MockEmbeddingResponse(embeddings)

    return MockLiteLLMModule()
