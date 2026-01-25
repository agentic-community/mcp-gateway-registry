"""
Health check routes with embedding status for operational visibility.

This module provides health check endpoints that include embedding model status,
enabling operators to monitor whether the system is running with the primary
embedding provider or has fallen back to the pre-baked FastEmbed model.
"""

import logging
import time
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from ..core.config import settings
from ..embeddings import FastEmbedClient
from ..search.service import faiss_service


logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


class EmbeddingHealthStatus(BaseModel):
    """Embedding service health status.

    Attributes:
        provider: The embedding provider currently in use
        model: The model name being used for embeddings
        dimensions: The dimension of the embedding vectors
        fallback_mode: True if running with fallback FastEmbed model
        load_time_ms: Time taken to load the model (if available)
    """

    provider: str
    model: str
    dimensions: int
    fallback_mode: bool
    load_time_ms: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response.

    Attributes:
        status: Overall health status ('healthy' or 'unhealthy')
        service: Name of the service
        embeddings: Optional embedding service health status
    """

    status: str
    service: str = "mcp-gateway-registry"
    embeddings: Optional[EmbeddingHealthStatus] = None


# Store load time when embedding model is loaded
_embedding_load_time_ms: Optional[float] = None


def record_embedding_load_time(load_time_ms: float) -> None:
    """Record the time taken to load the embedding model.

    Args:
        load_time_ms: Load time in milliseconds
    """
    global _embedding_load_time_ms
    _embedding_load_time_ms = load_time_ms
    logger.info(f"Recorded embedding model load time: {load_time_ms:.2f}ms")


def get_embedding_load_time() -> Optional[float]:
    """Get the recorded embedding model load time.

    Returns:
        Load time in milliseconds, or None if not recorded
    """
    return _embedding_load_time_ms


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check with embedding status",
    description="Simple health check for load balancers and monitoring. "
    "Includes embedding model status when available.",
)
async def health_check() -> HealthResponse:
    """Health check endpoint with embedding status.

    Returns health status including embedding model information.
    When the system falls back to FastEmbed, the fallback_mode field is True.

    Returns:
        HealthResponse with status and optional embedding details

    Example response when running in fallback mode:
        {
            "status": "healthy",
            "service": "mcp-gateway-registry",
            "embeddings": {
                "provider": "fastembed",
                "model": "BAAI/bge-small-en-v1.5",
                "dimensions": 384,
                "fallback_mode": true,
                "load_time_ms": 1523.45
            }
        }
    """
    embeddings_status = None

    # Add embedding status if model is loaded
    if faiss_service.embedding_model is not None:
        is_fallback = isinstance(faiss_service.embedding_model, FastEmbedClient)

        try:
            embedding_dim = faiss_service.embedding_model.get_embedding_dimension()
        except Exception:
            embedding_dim = settings.embeddings_model_dimensions

        embeddings_status = EmbeddingHealthStatus(
            provider="fastembed" if is_fallback else settings.embeddings_provider,
            model=(
                settings.embeddings_fallback_model
                if is_fallback
                else settings.embeddings_model_name
            ),
            dimensions=embedding_dim,
            fallback_mode=is_fallback,
            load_time_ms=get_embedding_load_time(),
        )

    return HealthResponse(
        status="healthy",
        embeddings=embeddings_status,
    )


@router.get(
    "/health/embeddings",
    response_model=Optional[EmbeddingHealthStatus],
    summary="Embedding model health status",
    description="Detailed health status of the embedding model.",
)
async def embedding_health() -> Optional[EmbeddingHealthStatus]:
    """Get detailed embedding model health status.

    Returns:
        EmbeddingHealthStatus if model is loaded, None otherwise
    """
    if faiss_service.embedding_model is None:
        return None

    is_fallback = isinstance(faiss_service.embedding_model, FastEmbedClient)

    try:
        embedding_dim = faiss_service.embedding_model.get_embedding_dimension()
    except Exception:
        embedding_dim = settings.embeddings_model_dimensions

    return EmbeddingHealthStatus(
        provider="fastembed" if is_fallback else settings.embeddings_provider,
        model=(
            settings.embeddings_fallback_model
            if is_fallback
            else settings.embeddings_model_name
        ),
        dimensions=embedding_dim,
        fallback_mode=is_fallback,
        load_time_ms=get_embedding_load_time(),
    )
