"""Embeddings module for vendor-agnostic embeddings generation."""

from .client import (
    EmbeddingsClient,
    LiteLLMClient,
    SentenceTransformersClient,
    create_embeddings_client,
)
from .token_provider import (
    EmbeddingsTokenProvider,
)

__all__ = [
    "EmbeddingsClient",
    "SentenceTransformersClient",
    "LiteLLMClient",
    "EmbeddingsTokenProvider",
    "create_embeddings_client",
]
