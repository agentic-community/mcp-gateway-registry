"""Admin routes for managing search embeddings.

Provides endpoints to detect documents missing from the search index
and re-index them on demand.
"""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from registry.auth.dependencies import nginx_proxied_auth
from registry.repositories.factory import get_search_repository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/embeddings", tags=["Embeddings Admin"])

MAX_REINDEX_BATCH: int = 100


def _require_admin(user_context: dict) -> None:
    """Verify user has admin permissions."""
    if not user_context.get("is_admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator permissions are required for this operation",
        )


class MissingEmbeddingEntry(BaseModel):
    """A document that exists in source but has no embedding."""

    path: str
    entity_type: str
    name: str
    is_enabled: bool = True


class MissingEmbeddingsResponse(BaseModel):
    """Response for the missing embeddings scan."""

    missing: list[MissingEmbeddingEntry]
    total_missing: int
    total_indexed: int
    total_source: int


class ReindexRequest(BaseModel):
    """Request body for re-indexing specific paths."""

    paths: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_REINDEX_BATCH,
        description=f"Paths to re-index (max {MAX_REINDEX_BATCH})",
    )


class ReindexDetailEntry(BaseModel):
    """Per-path result of re-indexing."""

    path: str
    entity_type: str
    status: str
    error: str | None = None


class ReindexResponse(BaseModel):
    """Response for the re-index operation."""

    success: int
    failed: int
    total: int
    details: list[ReindexDetailEntry]


def _get_search_repo():
    """Get the search repository instance."""
    return get_search_repository()


@router.get(
    "/missing",
    response_model=MissingEmbeddingsResponse,
)
async def get_missing_embeddings(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
) -> dict[str, Any]:
    """Scan for documents missing from the search index.

    Compares source collections (servers, agents, skills) against the
    embeddings collection and returns documents that have no embedding.
    """
    _require_admin(user_context)

    search_repo = _get_search_repo()
    result = await search_repo.find_missing_embeddings()

    logger.info(
        "Missing embeddings scan: %d missing out of %d source documents",
        result["total_missing"],
        result["total_source"],
    )

    return result


@router.post(
    "/reindex",
    response_model=ReindexResponse,
)
async def reindex_embeddings(
    request: ReindexRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
) -> dict[str, Any]:
    """Re-index specific documents by generating embeddings from source data.

    Reads each document from its source collection and runs it through
    the embedding pipeline. Useful for fixing documents that failed
    embedding during initial registration.
    """
    _require_admin(user_context)

    search_repo = _get_search_repo()
    result = await search_repo.reindex_paths(request.paths)

    logger.info(
        "Reindex completed: %d success, %d failed",
        result["success"],
        result["failed"],
    )

    return result
