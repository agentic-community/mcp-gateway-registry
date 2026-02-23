"""
Audit API routes for querying and exporting audit logs.

This module provides REST endpoints for administrators to query,
view, and export audit events from MongoDB storage.

All endpoints require admin access (is_admin=True in user context).
"""

from __future__ import annotations

import csv
import io
import logging
import re
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..auth.dependencies import enhanced_auth
from ..repositories.audit_repository import DocumentDBAuditRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audit", tags=["Audit Logs"])

# Singleton repository instance
_audit_repository: Optional[DocumentDBAuditRepository] = None


def get_audit_repository() -> DocumentDBAuditRepository:
    """Get or create the audit repository singleton."""
    global _audit_repository
    if _audit_repository is None:
        _audit_repository = DocumentDBAuditRepository()
    return _audit_repository


def require_admin(user_context: Dict[str, Any] = Depends(enhanced_auth)) -> Dict[str, Any]:
    """
    Dependency that requires admin access for audit endpoints.
    
    Args:
        user_context: User context from enhanced_auth dependency
        
    Returns:
        The user context if admin access is granted
        
    Raises:
        HTTPException: 403 Forbidden if user is not an admin
    """
    if not user_context.get("is_admin", False):
        logger.warning(
            f"Non-admin user '{user_context.get('username', 'unknown')}' "
            "attempted to access audit API"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user_context


# Response models
class AuditEventSummary(BaseModel):
    """Summary of an audit event for list responses."""
    
    timestamp: datetime
    request_id: str
    log_type: str = "registry_api_access"
    username: str
    auth_method: str
    is_admin: bool
    method: str
    path: str
    status_code: int
    duration_ms: float
    operation: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None


class AuditEventsResponse(BaseModel):
    """Response model for paginated audit events."""
    
    total: int = Field(description="Total number of matching events")
    limit: int = Field(description="Maximum events per page")
    offset: int = Field(description="Number of events skipped")
    events: List[Dict[str, Any]] = Field(description="List of audit events")


class AuditEventDetail(BaseModel):
    """Full audit event detail."""
    
    event: Dict[str, Any] = Field(description="Complete audit event record")


def _build_query(
    stream: str,
    from_time: Optional[datetime],
    to_time: Optional[datetime],
    username: Optional[str],
    operation: Optional[str],
    resource_type: Optional[str],
    resource_id: Optional[str],
    status_min: Optional[int],
    status_max: Optional[int],
    auth_decision: Optional[str],
) -> Dict[str, Any]:
    """
    Build MongoDB query from filter parameters.
    
    Args:
        stream: Log stream type (registry_api or mcp_access)
        from_time: Start of time range filter
        to_time: End of time range filter
        username: Filter by username
        operation: Filter by operation type
        resource_type: Filter by resource type
        resource_id: Filter by resource ID
        status_min: Minimum HTTP status code
        status_max: Maximum HTTP status code
        auth_decision: Filter by authorization decision
        
    Returns:
        MongoDB query dictionary
    """
    # Map stream parameter to log_type
    log_type_map = {
        "registry_api": "registry_api_access",
        "mcp_access": "mcp_server_access",
    }
    query: Dict[str, Any] = {"log_type": log_type_map.get(stream, stream)}
    
    # Time range filter
    if from_time or to_time:
        query["timestamp"] = {}
        if from_time:
            query["timestamp"]["$gte"] = from_time
        if to_time:
            query["timestamp"]["$lte"] = to_time
    
    # Identity filters - use case-insensitive regex for partial matching
    if username:
        # Escape special regex characters in the username
        escaped_username = re.escape(username)
        query["identity.username"] = {"$regex": escaped_username, "$options": "i"}
    
    # Action filters - different fields per stream
    if stream == "mcp_access":
        # MCP records use mcp_request.method and mcp_server.name
        if operation:
            query["mcp_request.method"] = operation
        if resource_type:
            escaped_resource = re.escape(resource_type)
            query["mcp_server.name"] = {"$regex": escaped_resource, "$options": "i"}
    else:
        # Registry API records use action.* fields
        if operation:
            query["action.operation"] = operation
        if resource_type:
            query["action.resource_type"] = resource_type
        if resource_id:
            query["action.resource_id"] = resource_id
    
    # Response status filter
    # For registry_api: use numeric response.status_code
    # For mcp_access: use string mcp_response.status ("success" or "error")
    if status_min is not None or status_max is not None:
        if stream == "mcp_access":
            # Map numeric ranges to MCP status strings
            # 2xx (200-299) -> success, 4xx/5xx (400-599) -> error
            if status_min is not None and status_min >= 200 and (status_max is None or status_max < 400):
                # 2xx range = success
                query["mcp_response.status"] = "success"
            elif status_min is not None and status_min >= 400:
                # 4xx/5xx range = error
                query["mcp_response.status"] = "error"
            # If "All Errors" (400-599), also map to error
            elif status_min == 400 and status_max == 599:
                query["mcp_response.status"] = "error"
        else:
            # Registry API uses numeric status codes
            query["response.status_code"] = {}
            if status_min is not None:
                query["response.status_code"]["$gte"] = status_min
            if status_max is not None:
                query["response.status_code"]["$lte"] = status_max
    
    # Authorization filter
    if auth_decision:
        query["authorization.decision"] = auth_decision
    
    return query


@router.get("/events", response_model=AuditEventsResponse)
async def get_audit_events(
    user_context: Annotated[Dict[str, Any], Depends(require_admin)],
    stream: str = Query(
        "registry_api",
        pattern="^(registry_api|mcp_access)$",
        description="Log stream type",
    ),
    from_time: Optional[datetime] = Query(
        None,
        alias="from",
        description="Start of time range (ISO 8601)",
    ),
    to_time: Optional[datetime] = Query(
        None,
        alias="to",
        description="End of time range (ISO 8601)",
    ),
    username: Optional[str] = Query(
        None,
        description="Filter by username",
    ),
    operation: Optional[str] = Query(
        None,
        description="Filter by operation type",
    ),
    resource_type: Optional[str] = Query(
        None,
        description="Filter by resource type",
    ),
    resource_id: Optional[str] = Query(
        None,
        description="Filter by resource ID",
    ),
    status_min: Optional[int] = Query(
        None,
        ge=100,
        le=599,
        description="Minimum HTTP status code",
    ),
    status_max: Optional[int] = Query(
        None,
        ge=100,
        le=599,
        description="Maximum HTTP status code",
    ),
    auth_decision: Optional[str] = Query(
        None,
        pattern="^(ALLOW|DENY|NOT_REQUIRED)$",
        description="Filter by authorization decision",
    ),
    limit: int = Query(
        50,
        ge=1,
        le=500,
        description="Maximum events per page",
    ),
    offset: int = Query(
        0,
        ge=0,
        description="Number of events to skip",
    ),
    sort_order: int = Query(
        -1,
        ge=-1,
        le=1,
        description="Sort order: -1 for descending (newest first), 1 for ascending (oldest first)",
    ),
) -> AuditEventsResponse:
    """
    Query recent audit events from MongoDB.
    
    Returns paginated audit events matching the specified filters.
    All filters are optional and can be combined.
    
    Requires admin access.
    """
    logger.info(
        f"Admin '{user_context.get('username')}' querying audit events: "
        f"stream={stream}, limit={limit}, offset={offset}"
    )
    
    query = _build_query(
        stream=stream,
        from_time=from_time,
        to_time=to_time,
        username=username,
        operation=operation,
        resource_type=resource_type,
        resource_id=resource_id,
        status_min=status_min,
        status_max=status_max,
        auth_decision=auth_decision,
    )
    
    repository = get_audit_repository()
    
    try:
        # Get total count for pagination
        total = await repository.count(query)
        
        # Get events
        events = await repository.find(
            query=query,
            limit=limit,
            offset=offset,
            sort_field="timestamp",
            sort_order=sort_order,
        )
        
        logger.debug(f"Found {len(events)} audit events (total: {total})")
        
        return AuditEventsResponse(
            total=total,
            limit=limit,
            offset=offset,
            events=events,
        )
    except Exception as e:
        logger.error(f"Error querying audit events: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to query audit events",
        )


@router.get("/events/{request_id}")
async def get_audit_event(
    request_id: str,
    user_context: Annotated[Dict[str, Any], Depends(require_admin)],
    log_type: Optional[str] = Query(
        default=None,
        description="Filter by log type: registry_api_access or mcp_server_access",
    ),
) -> Dict[str, Any]:
    """
    Get audit events by request_id.

    Returns all audit event records matching the request_id,
    optionally filtered by log_type. A single request may have
    multiple audit events (e.g., MCP server access + registry API access).

    Requires admin access.
    """
    logger.info(
        f"Admin '{user_context.get('username')}' retrieving audit events: "
        f"request_id={request_id}, log_type={log_type}"
    )

    repository = get_audit_repository()

    try:
        query: Dict[str, Any] = {"request_id": request_id}
        if log_type is not None:
            query["log_type"] = log_type

        events = await repository.find(query, limit=10)

        if not events:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Event not found",
            )

        return {
            "request_id": request_id,
            "events": events,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving audit events: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audit events",
        )


def _generate_jsonl(events: List[Dict[str, Any]]):
    """Generate JSONL output from events."""
    import json
    for event in events:
        # Convert datetime objects to ISO format strings
        if "timestamp" in event and isinstance(event["timestamp"], datetime):
            event["timestamp"] = event["timestamp"].isoformat()
        yield json.dumps(event) + "\n"


def _generate_csv(events: List[Dict[str, Any]]):
    """Generate CSV output from events."""
    if not events:
        yield ""
        return
    
    output = io.StringIO()
    
    # Define CSV columns (flattened structure)
    fieldnames = [
        "timestamp",
        "request_id",
        "log_type",
        "username",
        "auth_method",
        "is_admin",
        "method",
        "path",
        "status_code",
        "duration_ms",
        "operation",
        "resource_type",
        "resource_id",
        "auth_decision",
    ]
    
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    for event in events:
        # Flatten nested structure
        row = {
            "timestamp": event.get("timestamp", ""),
            "request_id": event.get("request_id", ""),
            "log_type": event.get("log_type", ""),
            "username": event.get("identity", {}).get("username", ""),
            "auth_method": event.get("identity", {}).get("auth_method", ""),
            "is_admin": event.get("identity", {}).get("is_admin", False),
            "method": event.get("request", {}).get("method", ""),
            "path": event.get("request", {}).get("path", ""),
            "status_code": event.get("response", {}).get("status_code", ""),
            "duration_ms": event.get("response", {}).get("duration_ms", ""),
            "operation": event.get("action", {}).get("operation", "") if event.get("action") else "",
            "resource_type": event.get("action", {}).get("resource_type", "") if event.get("action") else "",
            "resource_id": event.get("action", {}).get("resource_id", "") if event.get("action") else "",
            "auth_decision": event.get("authorization", {}).get("decision", "") if event.get("authorization") else "",
        }
        
        # Convert datetime to string if needed
        if isinstance(row["timestamp"], datetime):
            row["timestamp"] = row["timestamp"].isoformat()
        
        writer.writerow(row)
    
    yield output.getvalue()


@router.get("/export")
async def export_audit_events(
    user_context: Annotated[Dict[str, Any], Depends(require_admin)],
    format: str = Query(
        "jsonl",
        pattern="^(jsonl|csv)$",
        description="Export format: jsonl or csv",
    ),
    stream: str = Query(
        "registry_api",
        pattern="^(registry_api|mcp_access)$",
        description="Log stream type",
    ),
    from_time: Optional[datetime] = Query(
        None,
        alias="from",
        description="Start of time range (ISO 8601)",
    ),
    to_time: Optional[datetime] = Query(
        None,
        alias="to",
        description="End of time range (ISO 8601)",
    ),
    username: Optional[str] = Query(
        None,
        description="Filter by username",
    ),
    operation: Optional[str] = Query(
        None,
        description="Filter by operation type",
    ),
    resource_type: Optional[str] = Query(
        None,
        description="Filter by resource type",
    ),
    resource_id: Optional[str] = Query(
        None,
        description="Filter by resource ID",
    ),
    status_min: Optional[int] = Query(
        None,
        ge=100,
        le=599,
        description="Minimum HTTP status code",
    ),
    status_max: Optional[int] = Query(
        None,
        ge=100,
        le=599,
        description="Maximum HTTP status code",
    ),
    auth_decision: Optional[str] = Query(
        None,
        pattern="^(ALLOW|DENY|NOT_REQUIRED)$",
        description="Filter by authorization decision",
    ),
    limit: int = Query(
        10000,
        ge=1,
        le=100000,
        description="Maximum events to export",
    ),
) -> StreamingResponse:
    """
    Export filtered audit events as JSONL or CSV file.
    
    Returns a downloadable file containing audit events matching
    the specified filters.
    
    Requires admin access.
    """
    logger.info(
        f"Admin '{user_context.get('username')}' exporting audit events: "
        f"format={format}, stream={stream}, limit={limit}"
    )
    
    query = _build_query(
        stream=stream,
        from_time=from_time,
        to_time=to_time,
        username=username,
        operation=operation,
        resource_type=resource_type,
        resource_id=resource_id,
        status_min=status_min,
        status_max=status_max,
        auth_decision=auth_decision,
    )
    
    repository = get_audit_repository()
    
    try:
        # Get events for export (no offset, just limit)
        events = await repository.find(
            query=query,
            limit=limit,
            offset=0,
            sort_field="timestamp",
            sort_order=-1,
        )
        
        # Generate timestamp for filename
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        filename = f"audit-export-{timestamp}.{format}"
        
        if format == "jsonl":
            return StreamingResponse(
                _generate_jsonl(events),
                media_type="application/x-ndjson",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}",
                },
            )
        else:  # csv
            return StreamingResponse(
                _generate_csv(events),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}",
                },
            )
    except Exception as e:
        logger.error(f"Error exporting audit events: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export audit events",
        )
