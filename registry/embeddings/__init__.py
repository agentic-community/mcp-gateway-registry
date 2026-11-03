"""Embeddings module for vendor-agnostic embeddings generation."""

from .client import (
    EmbeddingsClient,
    LiteLLMClient,
    RemoteEmbeddingsClient,
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
    "RemoteEmbeddingsClient",
    "EmbeddingsTokenProvider",
    "create_embeddings_client",
]
