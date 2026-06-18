"""Admin routes for managing search embeddings.

Provides endpoints to detect documents missing from the search index,
orphaned stale embeddings (issue #1145), and re-index on demand.
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


class StaleEmbeddingEntry(BaseModel):
    """An embedding-index document with no matching source registry record."""

    path: str
    entity_type: str
    name: str
    is_enabled: bool = True


class StaleEmbeddingsResponse(BaseModel):
    """Response for the stale embeddings scan."""

    stale: list[StaleEmbeddingEntry]
    total_stale: int
    total_indexed: int
    total_source: int


class StaleCleanupRequest(BaseModel):
    """Request body for removing orphaned embedding documents."""

    paths: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_REINDEX_BATCH,
        description=f"Embedding paths to remove (max {MAX_REINDEX_BATCH})",
    )


class StaleCleanupDetailEntry(BaseModel):
    """Per-path result of stale embedding removal.

    ``status`` is ``removed`` (a stale embedding existed and was deleted),
    ``not_found`` (no-op: nothing indexed at this path), or ``failed``.
    """

    path: str
    status: str
    error: str | None = None


class StaleCleanupResponse(BaseModel):
    """Response for the stale embedding cleanup operation.

    Counts are reported separately so admins can tell a real cleanup from a
    no-op: ``removed`` paths actually had an orphaned embedding deleted, while
    ``not_found`` paths matched nothing (e.g. a typo or an already-clean path).
    """

    removed: int
    not_found: int
    failed: int
    total: int
    details: list[StaleCleanupDetailEntry]


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


@router.get(
    "/stale",
    response_model=StaleEmbeddingsResponse,
)
async def get_stale_embeddings(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
) -> dict[str, Any]:
    """Scan for orphaned embeddings with no source registry document.

    Compares the embeddings collection against source collections
    (servers, agents, skills, virtual servers) and returns index entries
    whose source entity was deleted without removing the vector (issue #1145).
    """
    _require_admin(user_context)

    search_repo = _get_search_repo()
    find_stale = getattr(search_repo, "find_stale_embeddings", None)
    if find_stale is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Stale embedding scan is not supported by the current search backend",
        )

    result = await find_stale()

    logger.info(
        "Stale embeddings scan: %d stale out of %d indexed documents",
        result["total_stale"],
        result["total_indexed"],
    )

    return result


@router.post(
    "/stale/cleanup",
    response_model=StaleCleanupResponse,
)
async def cleanup_stale_embeddings(
    request: StaleCleanupRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)] = None,
) -> dict[str, Any]:
    """Remove orphaned embedding documents from the search index."""
    _require_admin(user_context)

    search_repo = _get_search_repo()
    remove_stale = getattr(search_repo, "remove_stale_embeddings", None)
    if remove_stale is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Stale embedding cleanup is not supported by the current search backend",
        )

    result = await remove_stale(request.paths)

    logger.info(
        "Stale embedding cleanup: %d removed, %d not_found, %d failed",
        result["removed"],
        result["not_found"],
        result["failed"],
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
