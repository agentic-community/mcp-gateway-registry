"""Admin API routes for querying centralized application logs.

All endpoints require admin access.
"""

import json
import logging
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..auth.dependencies import enhanced_auth
from ..repositories.app_log_repository import AppLogRepository
from ..repositories.factory import get_app_log_repository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/logs", tags=["Application Logs"])


def _require_admin(
    user_context: dict[str, Any] = Depends(enhanced_auth),
) -> dict[str, Any]:
    """Dependency that requires admin access."""
    if not user_context.get("is_admin", False):
        logger.warning(
            f"Non-admin user '{user_context.get('username', 'unknown')}' "
            "attempted to access application logs API"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user_context


def _get_repo() -> AppLogRepository:
    """Get the application log repository or raise 503."""
    repo = get_app_log_repository()
    if repo is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Application log storage not available (requires MongoDB backend)",
        )
    return repo


class LogEntry(BaseModel):
    """Single application log entry."""

    timestamp: datetime
    hostname: str
    service: str
    level: str
    logger: str = ""
    filename: str = ""
    lineno: int = 0
    message: str = ""


class LogQueryResponse(BaseModel):
    """Paginated response for log queries."""

    entries: list[LogEntry]
    total_count: int
    limit: int
    offset: int
    has_next: bool


class LogMetadataResponse(BaseModel):
    """Available filter values for log queries."""

    services: list[str] = Field(default_factory=list)
    hostnames: list[str] = Field(default_factory=list)
    levels: list[str] = Field(
        default_factory=lambda: ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )


@router.get(
    "",
    response_model=LogQueryResponse,
    summary="Query application logs",
    description="Query centralized application logs with filtering, pagination, and time range support.",
)
async def query_logs(
    user_context: Annotated[dict, Depends(_require_admin)],
    service: Annotated[str | None, Query(description="Filter by service name")] = None,
    level: Annotated[str | None, Query(description="Filter by log level")] = None,
    hostname: Annotated[str | None, Query(description="Filter by pod/hostname")] = None,
    start: Annotated[datetime | None, Query(description="Start of time range (ISO 8601)")] = None,
    end: Annotated[datetime | None, Query(description="End of time range (ISO 8601)")] = None,
    search: Annotated[str | None, Query(description="Substring search in message")] = None,
    limit: Annotated[int, Query(ge=1, le=1000, description="Max entries to return")] = 100,
    offset: Annotated[int, Query(ge=0, description="Number of entries to skip")] = 0,
) -> LogQueryResponse:
    repo = _get_repo()

    entries, total = await repo.query(
        service=service,
        level=level,
        hostname=hostname,
        start=start,
        end=end,
        search=search,
        skip=offset,
        limit=limit,
    )

    return LogQueryResponse(
        entries=[LogEntry(**e) for e in entries],
        total_count=total,
        limit=limit,
        offset=offset,
        has_next=(offset + limit) < total,
    )


@router.get(
    "/export",
    summary="Export application logs as JSONL",
    description="Stream application logs as newline-delimited JSON for download.",
    response_class=StreamingResponse,
)
async def export_logs(
    user_context: Annotated[dict, Depends(_require_admin)],
    service: Annotated[str | None, Query(description="Filter by service name")] = None,
    level: Annotated[str | None, Query(description="Filter by log level")] = None,
    hostname: Annotated[str | None, Query(description="Filter by pod/hostname")] = None,
    start: Annotated[datetime | None, Query(description="Start of time range (ISO 8601)")] = None,
    end: Annotated[datetime | None, Query(description="End of time range (ISO 8601)")] = None,
    search: Annotated[str | None, Query(description="Substring search in message")] = None,
    limit: Annotated[int, Query(ge=1, le=50000, description="Max entries to export")] = 10000,
) -> StreamingResponse:
    repo = _get_repo()

    entries, _ = await repo.query(
        service=service,
        level=level,
        hostname=hostname,
        start=start,
        end=end,
        search=search,
        skip=0,
        limit=limit,
    )

    def _generate():
        for entry in entries:
            if "timestamp" in entry and hasattr(entry["timestamp"], "isoformat"):
                entry["timestamp"] = entry["timestamp"].isoformat()
            yield json.dumps(entry, default=str) + "\n"

    return StreamingResponse(
        _generate(),
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": "attachment; filename=application-logs.jsonl",
        },
    )


@router.get(
    "/metadata",
    response_model=LogMetadataResponse,
    summary="Get log filter metadata",
    description="Returns available service names, hostnames, and log levels for building filter UIs.",
)
async def get_log_metadata(
    user_context: Annotated[dict, Depends(_require_admin)],
) -> LogMetadataResponse:
    repo = _get_repo()

    services = await repo.get_distinct_services()
    hostnames = await repo.get_distinct_hostnames()

    return LogMetadataResponse(
        services=services,
        hostnames=hostnames,
    )
