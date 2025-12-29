from __future__ import annotations

import csv
import io
import json
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from typing import (
    Iterator,
    Optional,
)

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
)
from fastapi.responses import (
    StreamingResponse,
)

from ..auth.dependency import (
    EnforceAIManagementContext,
    get_enforceai_management_context,
    get_enforceai_stores,
)
from ..db.data_layer import (
    EnforceAIStores,
)
from ..models.audit import (
    AuditEventsQueryResult,
    DEFAULT_AUDIT_PAGE_SIZE,
    DEFAULT_AUDIT_WINDOW_SECONDS,
    MAX_AUDIT_PAGE_SIZE,
)
from .management_common import (
    _emit_management_audit_event,
    _get_request_id,
    _require_admin,
)

MAX_EXPORT_EVENTS: int = 10000

router = APIRouter()


def _normalize_audit_window(
    *,
    since: Optional[datetime],
    until: Optional[datetime],
) -> tuple[datetime, datetime, datetime]:
    now = datetime.now(timezone.utc)

    if until is None:
        until = now
    elif until.tzinfo is None:
        until = until.replace(tzinfo=timezone.utc)

    if since is None:
        since = until - timedelta(seconds=DEFAULT_AUDIT_WINDOW_SECONDS)
    elif since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)

    if since > until:
        raise HTTPException(
            status_code=400,
            detail="'since' must be before 'until'",
        )

    return now, since, until


@router.get("/audit/events", response_model=AuditEventsQueryResult)
async def list_audit_events(
    request: Request,
    since: Optional[datetime] = Query(
        None,
        description="Filter events after this time (ISO 8601)",
    ),
    until: Optional[datetime] = Query(
        None,
        description="Filter events before this time (ISO 8601)",
    ),
    limit: int = Query(
        DEFAULT_AUDIT_PAGE_SIZE,
        ge=1,
        le=MAX_AUDIT_PAGE_SIZE,
        description="Maximum number of events to return",
    ),
    cursor: Optional[str] = Query(
        None,
        description="Pagination cursor from previous response",
    ),
    agent_id: Optional[str] = Query(
        None,
        description="Filter by agent ID",
    ),
    action: Optional[list[str]] = Query(
        None,
        description="Filter by action names",
    ),
    outcome: Optional[list[str]] = Query(
        None,
        description="Filter by outcome values (allow, deny)",
    ),
    request_id: Optional[str] = Query(
        None,
        description="Filter by exact request ID match",
    ),
    server: Optional[str] = Query(
        None,
        description="Filter by server (from event details)",
    ),
    tool: Optional[str] = Query(
        None,
        description="Filter by tool (from event details)",
    ),
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> AuditEventsQueryResult:
    """
    List audit events for the authenticated user.

    Events are returned in descending order by time (most recent first).
    Use the `cursor` parameter from the response to fetch the next page.

    Time window defaults to the last 60 minutes if not specified.
    """
    _, since_norm, until_norm = _normalize_audit_window(
        since=since,
        until=until,
    )

    return stores.audit_store.query_events(
        user_id=context.user_id,
        agent_id=agent_id,
        actions=action,
        outcomes=outcome,
        request_id=request_id,
        server=server,
        tool=tool,
        since=since_norm,
        until=until_norm,
        limit=limit,
        cursor=cursor,
    )


@router.get("/admin/audit/events", response_model=AuditEventsQueryResult)
async def list_admin_audit_events(
    request: Request,
    user_id: Optional[str] = Query(
        None,
        description="Filter events by user ID (admin only, omit to query all users)",
    ),
    since: Optional[datetime] = Query(
        None,
        description="Filter events after this time (ISO 8601)",
    ),
    until: Optional[datetime] = Query(
        None,
        description="Filter events before this time (ISO 8601)",
    ),
    limit: int = Query(
        DEFAULT_AUDIT_PAGE_SIZE,
        ge=1,
        le=MAX_AUDIT_PAGE_SIZE,
        description=f"Maximum events to return (max {MAX_AUDIT_PAGE_SIZE})",
    ),
    cursor: Optional[str] = Query(
        None,
        description="Pagination cursor from previous response",
    ),
    agent_id: Optional[str] = Query(
        None,
        description="Filter by agent ID",
    ),
    action: Optional[list[str]] = Query(
        None,
        description="Filter by action(s)",
    ),
    outcome: Optional[list[str]] = Query(
        None,
        description="Filter by outcome(s) (allow, deny)",
    ),
    request_id: Optional[str] = Query(
        None,
        description="Filter by request ID",
    ),
    server: Optional[str] = Query(
        None,
        description="Filter by server (from event details)",
    ),
    tool: Optional[str] = Query(
        None,
        description="Filter by tool (from event details)",
    ),
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> AuditEventsQueryResult:
    """
    List audit events across all users (admin only).

    Events are returned in descending order by time (most recent first).
    Use the `cursor` parameter from the response to fetch the next page.

    Time window defaults to the last 60 minutes if not specified.
    """
    _require_admin(context)
    request_id_header = _get_request_id(request)

    _, since_norm, until_norm = _normalize_audit_window(
        since=since,
        until=until,
    )

    result = stores.audit_store.query_events(
        user_id=user_id,  # None means all users
        agent_id=agent_id,
        actions=action,
        outcomes=outcome,
        request_id=request_id,
        server=server,
        tool=tool,
        since=since_norm,
        until=until_norm,
        limit=limit,
        cursor=cursor,
    )

    _emit_management_audit_event(
        stores=stores,
        action="admin/audit/query",
        outcome="allow",
        user_id=context.user_id,
        agent_id=context.actor_agent_id,
        request_id=request_id_header,
        details={
            "target_user_id": user_id,
            "count": len(result.items),
            "has_more": result.next_cursor is not None,
        },
    )

    return result


@router.get("/admin/audit/events/export")
async def export_admin_audit_events(
    request: Request,
    user_id: Optional[str] = Query(
        None,
        description="Filter events by user ID (admin only, omit to query all users)",
    ),
    since: Optional[datetime] = Query(
        None,
        description="Filter events after this time (ISO 8601)",
    ),
    until: Optional[datetime] = Query(
        None,
        description="Filter events before this time (ISO 8601)",
    ),
    agent_id: Optional[str] = Query(
        None,
        description="Filter by agent ID",
    ),
    action: Optional[list[str]] = Query(
        None,
        description="Filter by action(s)",
    ),
    outcome: Optional[list[str]] = Query(
        None,
        description="Filter by outcome(s) (allow, deny)",
    ),
    request_id: Optional[str] = Query(
        None,
        description="Filter by request ID",
    ),
    server: Optional[str] = Query(
        None,
        description="Filter by server (from event details)",
    ),
    tool: Optional[str] = Query(
        None,
        description="Filter by tool (from event details)",
    ),
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> StreamingResponse:
    """
    Export audit events as CSV (admin only).

    Returns up to 10,000 events matching the filters. If more than 10,000 events
    match, returns HTTP 413 and requires narrowing filters.
    Time window defaults to the last 60 minutes if not specified.
    """
    _require_admin(context)
    request_id_header = _get_request_id(request)

    now, since_norm, until_norm = _normalize_audit_window(
        since=since,
        until=until,
    )

    result = stores.audit_store.query_events(
        user_id=user_id,
        agent_id=agent_id,
        actions=action,
        outcomes=outcome,
        request_id=request_id,
        server=server,
        tool=tool,
        since=since_norm,
        until=until_norm,
        limit=MAX_EXPORT_EVENTS,
        cursor=None,
        max_limit=MAX_EXPORT_EVENTS,
    )

    if result.next_cursor is not None:
        _emit_management_audit_event(
            stores=stores,
            action="admin/audit/export",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id_header,
            details={
                "target_user_id": user_id,
                "reason": "too_many_events",
                "limit": MAX_EXPORT_EVENTS,
            },
        )
        raise HTTPException(
            status_code=413,
            detail=f"Too many events to export (>{MAX_EXPORT_EVENTS}). Narrow filters and try again.",
        )

    def _csv_stream() -> Iterator[str]:
        output = io.StringIO()
        writer = csv.writer(output)

        header = [
            "event_id",
            "occurred_at",
            "user_id",
            "agent_id",
            "action",
            "outcome",
            "request_id",
            "server",
            "tool",
            "reason",
            "matched_scope",
            "provider",
            "details_json",
        ]
        writer.writerow(header)
        yield output.getvalue()
        output.seek(0)
        output.truncate(0)

        for event in result.items:
            details = event.details or {}
            writer.writerow(
                [
                    event.event_id,
                    event.occurred_at.isoformat(),
                    event.user_id,
                    event.agent_id,
                    event.action,
                    event.outcome,
                    event.request_id or "",
                    str(details.get("server") or ""),
                    str(details.get("tool") or ""),
                    str(details.get("reason") or ""),
                    str(details.get("matched_scope") or ""),
                    str(details.get("provider") or ""),
                    json.dumps(
                        event.details,
                        separators=(",", ":"),
                        sort_keys=True,
                        default=str,
                    )
                    if event.details
                    else "",
                ]
            )
            yield output.getvalue()
            output.seek(0)
            output.truncate(0)

    export_timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f"audit_events_{export_timestamp}.csv"

    _emit_management_audit_event(
        stores=stores,
        action="admin/audit/export",
        outcome="allow",
        user_id=context.user_id,
        agent_id=context.actor_agent_id,
        request_id=request_id_header,
        details={
            "target_user_id": user_id,
            "count": len(result.items),
            "filename": filename,
        },
    )

    return StreamingResponse(
        _csv_stream(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

