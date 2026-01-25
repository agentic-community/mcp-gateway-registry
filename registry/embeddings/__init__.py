"""Embeddings module for vendor-agnostic embeddings generation."""

from .client import (
    EmbeddingsClient,
    SentenceTransformersClient,
    LiteLLMClient,
    FastEmbedClient,
    create_embeddings_client,
    FALLBACK_MODEL_NAME,
    FALLBACK_MODEL_DIMENSIONS,
)

__all__ = [
    "EmbeddingsClient",
    "SentenceTransformersClient",
    "LiteLLMClient",
    "FastEmbedClient",
    "create_embeddings_client",
    "FALLBACK_MODEL_NAME",
    "FALLBACK_MODEL_DIMENSIONS",
]
