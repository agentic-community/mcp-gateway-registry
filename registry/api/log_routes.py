"""Admin API routes for querying centralized application logs.

All endpoints require admin access.
"""

import json
import logging
import re
import time
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..auth.dependencies import nginx_proxied_auth
from ..repositories.app_log_repository import AppLogRepository
from ..repositories.factory import get_app_log_repository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/logs", tags=["Application Logs"])

LEVEL_MAP = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}

RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 10
_rate_limit_cache: dict[str, list[float]] = {}
MAX_SEARCH_LENGTH = 200


def _check_rate_limit(user_id: str) -> bool:
    """Allow up to RATE_LIMIT_MAX_REQUESTS per user per window."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    if user_id not in _rate_limit_cache:
        _rate_limit_cache[user_id] = []

    _rate_limit_cache[user_id] = [t for t in _rate_limit_cache[user_id] if t > window_start]

    if len(_rate_limit_cache[user_id]) >= RATE_LIMIT_MAX_REQUESTS:
        return False

    _rate_limit_cache[user_id].append(now)
    return True


def _sanitize_search(search: str | None) -> str | None:
    """Escape regex metacharacters for safe MongoDB $regex use."""
    if not search:
        return None
    return re.escape(search[:MAX_SEARCH_LENGTH])


def _require_admin(
    user_context: dict[str, Any] = Depends(nginx_proxied_auth),
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
    level_no: int = 0
    logger: str = ""
    filename: str = ""
    lineno: int = 0
    process: int = 0
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
    level: Annotated[
        str | None, Query(description="Minimum log level: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    ] = None,
    hostname: Annotated[str | None, Query(description="Filter by pod/hostname")] = None,
    start: Annotated[datetime | None, Query(description="Start of time range (ISO 8601)")] = None,
    end: Annotated[datetime | None, Query(description="End of time range (ISO 8601)")] = None,
    search: Annotated[
        str | None, Query(description="Substring search in message (max 200 chars)")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=10000, description="Max entries to return")] = 100,
    offset: Annotated[int, Query(ge=0, description="Number of entries to skip")] = 0,
) -> LogQueryResponse:
    username = user_context.get("username", "unknown")
    if not _check_rate_limit(username):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
        )

    repo = _get_repo()

    level_no = LEVEL_MAP.get(level.upper()) if level else None
    sanitized_search = _sanitize_search(search)

    entries, total = await repo.query(
        service=service,
        level_no=level_no,
        hostname=hostname,
        start=start,
        end=end,
        search=sanitized_search,
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
    level: Annotated[str | None, Query(description="Minimum log level")] = None,
    hostname: Annotated[str | None, Query(description="Filter by pod/hostname")] = None,
    start: Annotated[datetime | None, Query(description="Start of time range (ISO 8601)")] = None,
    end: Annotated[datetime | None, Query(description="End of time range (ISO 8601)")] = None,
    search: Annotated[
        str | None, Query(description="Substring search in message (max 200 chars)")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=50000, description="Max entries to export")] = 10000,
) -> StreamingResponse:
    username = user_context.get("username", "unknown")
    if not _check_rate_limit(username):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
        )

    repo = _get_repo()

    level_no = LEVEL_MAP.get(level.upper()) if level else None
    sanitized_search = _sanitize_search(search)

    entries, _ = await repo.query(
        service=service,
        level_no=level_no,
        hostname=hostname,
        start=start,
        end=end,
        search=sanitized_search,
        skip=0,
        limit=limit,
    )

    def _generate():
        for entry in entries:
            if "timestamp" in entry and hasattr(entry["timestamp"], "isoformat"):
                entry["timestamp"] = entry["timestamp"].isoformat()
            yield json.dumps(entry, default=str) + "\n"

    svc_label = service or "all"
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    filename = f"logs-{svc_label}-{ts}.jsonl"

    return StreamingResponse(
        _generate(),
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
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
