"""
Audit API routes for querying and exporting audit logs.

This module provides REST endpoints for administrators to query,
view, and export audit events from MongoDB storage.

All endpoints require admin access (is_admin=True in user context).
"""

from __future__ import annotations

import asyncio
import csv
import io
import logging
import re
import time
from datetime import datetime, timedelta, timezone
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


class AuditFilterOptions(BaseModel):
    """Available filter values for audit log dropdowns."""

    usernames: List[str] = Field(
        default_factory=list,
        description="Distinct usernames found in audit events",
    )
    server_names: List[str] = Field(
        default_factory=list,
        description="Distinct MCP server names (MCP stream only)",
    )


class UsageSummaryItem(BaseModel):
    """A single row in a usage summary."""

    name: str = Field(description="Username, server name, or category")
    count: int = Field(description="Number of events")


class TimeSeriesBucket(BaseModel):
    """A single time bucket for the activity chart."""

    period: str = Field(description="Time period label (e.g., '2026-02-28')")
    count: int = Field(description="Number of events in this period")


class StatusDistribution(BaseModel):
    """Status code distribution."""

    status_2xx: int = Field(default=0, description="2xx success count")
    status_4xx: int = Field(default=0, description="4xx client error count")
    status_5xx: int = Field(default=0, description="5xx server error count")


class UserActivityItem(BaseModel):
    """Per-user activity breakdown showing top operations."""

    username: str = Field(description="Username")
    total: int = Field(description="Total requests by this user")
    operations: List[UsageSummaryItem] = Field(
        default_factory=list,
        description="Top operations for this user",
    )


class AuditStatisticsResponse(BaseModel):
    """Aggregated audit statistics."""

    total_events: int = Field(description="Total events in time range")
    top_users: List[UsageSummaryItem] = Field(
        default_factory=list,
        description="Top 10 users by event count",
    )
    top_servers: List[UsageSummaryItem] = Field(
        default_factory=list,
        description="Top 10 MCP servers (MCP stream only)",
    )
    top_operations: List[UsageSummaryItem] = Field(
        default_factory=list,
        description="Top 10 operations by event count",
    )
    activity_timeline: List[TimeSeriesBucket] = Field(
        default_factory=list,
        description="Daily event counts for the time range",
    )
    status_distribution: StatusDistribution = Field(
        default_factory=StatusDistribution,
        description="Distribution of HTTP status codes",
    )
    user_activity: List[UserActivityItem] = Field(
        default_factory=list,
        description="Per-user breakdown of top operations",
    )


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
            if (
                status_min is not None
                and status_min >= 200
                and (status_max is None or status_max < 400)
            ):
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


@router.get("/filter-options", response_model=AuditFilterOptions)
async def get_filter_options(
    user_context: Annotated[Dict[str, Any], Depends(require_admin)],
    stream: str = Query(
        "registry_api",
        pattern="^(registry_api|mcp_access)$",
        description="Log stream type",
    ),
) -> AuditFilterOptions:
    """Get distinct filter values for audit log dropdowns. Requires admin access."""
    start_time = time.time()

    log_type_map = {
        "registry_api": "registry_api_access",
        "mcp_access": "mcp_server_access",
    }
    log_type = log_type_map.get(stream, stream)
    query = {"log_type": log_type}

    repository = get_audit_repository()

    usernames = await repository.distinct("identity.username", query)

    server_names: List[str] = []
    if stream == "mcp_access":
        server_names = await repository.distinct("mcp_server.name", query)

    elapsed = time.time() - start_time
    logger.info(
        f"Filter options fetched in {elapsed:.2f}s (stream={stream}, "
        f"usernames={len(usernames)}, servers={len(server_names)})"
    )

    return AuditFilterOptions(
        usernames=usernames,
        server_names=server_names,
    )


@router.get("/statistics", response_model=AuditStatisticsResponse)
async def get_statistics(
    user_context: Annotated[Dict[str, Any], Depends(require_admin)],
    stream: str = Query(
        "registry_api",
        pattern="^(registry_api|mcp_access)$",
        description="Log stream type",
    ),
    days: int = Query(
        7,
        ge=1,
        le=30,
        description="Number of days to include in statistics",
    ),
    username: Optional[str] = Query(
        None,
        description="Filter statistics to a specific username",
    ),
) -> AuditStatisticsResponse:
    """Get aggregated audit statistics for the dashboard. Requires admin access."""
    start_time = time.time()

    log_type_map = {
        "registry_api": "registry_api_access",
        "mcp_access": "mcp_server_access",
    }
    log_type = log_type_map.get(stream, stream)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    base_match: Dict[str, Any] = {"log_type": log_type, "timestamp": {"$gte": cutoff}}

    if username:
        escaped_username = re.escape(username)
        base_match["identity.username"] = {"$regex": f"^{escaped_username}$", "$options": "i"}

    repository = get_audit_repository()

    # Build all pipelines upfront
    op_field = "$mcp_request.method" if stream == "mcp_access" else "$action.operation"

    # Status distribution pipeline differs by stream
    if stream == "mcp_access":
        status_pipeline: List[Dict[str, Any]] = [
            {"$match": base_match},
            {"$group": {"_id": "$mcp_response.status", "count": {"$sum": 1}}},
        ]
    else:
        status_pipeline = [
            {"$match": base_match},
            {
                "$project": {
                    "bucket": {
                        "$switch": {
                            "branches": [
                                {
                                    "case": {
                                        "$and": [
                                            {"$gte": ["$response.status_code", 200]},
                                            {"$lt": ["$response.status_code", 300]},
                                        ]
                                    },
                                    "then": "2xx",
                                },
                                {
                                    "case": {
                                        "$and": [
                                            {"$gte": ["$response.status_code", 400]},
                                            {"$lt": ["$response.status_code", 500]},
                                        ]
                                    },
                                    "then": "4xx",
                                },
                                {
                                    "case": {"$gte": ["$response.status_code", 500]},
                                    "then": "5xx",
                                },
                            ],
                            "default": "other",
                        }
                    }
                }
            },
            {"$group": {"_id": "$bucket", "count": {"$sum": 1}}},
        ]

    # Run ALL pipelines concurrently with asyncio.gather()
    # Note: audit data is bounded by TTL (default 7 days), so collection size is naturally limited
    tasks = [
        repository.count(base_match),
        repository.aggregate(
            [
                {"$match": base_match},
                {"$group": {"_id": "$identity.username", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 10},
            ]
        ),
        repository.aggregate(
            [
                {"$match": base_match},
                {"$group": {"_id": op_field, "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 10},
            ]
        ),
        repository.aggregate(
            [
                {"$match": base_match},
                {
                    "$group": {
                        "_id": {
                            "$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}
                        },
                        "count": {"$sum": 1},
                    }
                },
                {"$sort": {"_id": 1}},
            ]
        ),
        repository.aggregate(status_pipeline),
        # Per-user activity breakdown: group by (username, operation), then re-group by username
        repository.aggregate(
            [
                {"$match": base_match},
                {
                    "$group": {
                        "_id": {
                            "user": "$identity.username",
                            "op": op_field,
                        },
                        "count": {"$sum": 1},
                    }
                },
                {"$sort": {"count": -1}},
                {
                    "$group": {
                        "_id": "$_id.user",
                        "total": {"$sum": "$count"},
                        "operations": {
                            "$push": {"name": "$_id.op", "count": "$count"}
                        },
                    }
                },
                {"$sort": {"total": -1}},
                {"$limit": 10},
            ]
        ),
    ]

    # Conditionally add MCP server aggregation
    if stream == "mcp_access":
        tasks.append(
            repository.aggregate(
                [
                    {"$match": base_match},
                    {"$group": {"_id": "$mcp_server.name", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1}},
                    {"$limit": 10},
                ]
            )
        )

    results = await asyncio.gather(*tasks)

    # Unpack results
    total_events = results[0]
    top_users_raw = results[1]
    top_ops_raw = results[2]
    timeline_raw = results[3]
    status_raw = results[4]
    user_activity_raw = results[5]
    top_servers_raw = results[6] if stream == "mcp_access" else []

    # Transform results
    top_users = [
        UsageSummaryItem(name=r["_id"] or "unknown", count=r["count"])
        for r in top_users_raw
        if r.get("_id")
    ]

    top_servers = (
        [
            UsageSummaryItem(name=r["_id"] or "unknown", count=r["count"])
            for r in top_servers_raw
            if r.get("_id")
        ]
        if top_servers_raw
        else []
    )

    top_operations = [
        UsageSummaryItem(name=r["_id"] or "unknown", count=r["count"])
        for r in top_ops_raw
        if r.get("_id")
    ]

    activity_timeline = [
        TimeSeriesBucket(period=r["_id"], count=r["count"]) for r in timeline_raw
    ]

    status_dist = StatusDistribution()
    if stream == "mcp_access":
        for r in status_raw:
            if r["_id"] == "success":
                status_dist.status_2xx = r["count"]
            elif r["_id"] == "error":
                status_dist.status_5xx = r["count"]
    else:
        for r in status_raw:
            if r.get("_id") == "2xx":
                status_dist.status_2xx = r["count"]
            elif r.get("_id") == "4xx":
                status_dist.status_4xx = r["count"]
            elif r.get("_id") == "5xx":
                status_dist.status_5xx = r["count"]

    # Transform per-user activity breakdown
    if user_activity_raw:
        logger.debug(f"Raw user_activity sample: {user_activity_raw[0]}")
    user_activity = []
    for r in user_activity_raw:
        if not r.get("_id"):
            continue
        ops = []
        for op in (r.get("operations") or [])[:5]:
            op_name = op.get("name") or op.get("_id", {}).get("op") if isinstance(op, dict) else None
            op_count = op.get("count", 0) if isinstance(op, dict) else 0
            if op_name:
                ops.append(UsageSummaryItem(name=str(op_name), count=op_count))
        user_activity.append(
            UserActivityItem(
                username=r["_id"] or "unknown",
                total=r.get("total", 0),
                operations=ops,
            )
        )

    elapsed = time.time() - start_time
    logger.info(
        f"Audit statistics computed in {elapsed:.2f}s (stream={stream}, days={days})"
    )

    return AuditStatisticsResponse(
        total_events=total_events,
        top_users=top_users,
        top_servers=top_servers,
        top_operations=top_operations,
        activity_timeline=activity_timeline,
        status_distribution=status_dist,
        user_activity=user_activity,
    )


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
            "operation": event.get("action", {}).get("operation", "")
            if event.get("action")
            else "",
            "resource_type": event.get("action", {}).get("resource_type", "")
            if event.get("action")
            else "",
            "resource_id": event.get("action", {}).get("resource_id", "")
            if event.get("action")
            else "",
            "auth_decision": event.get("authorization", {}).get("decision", "")
            if event.get("authorization")
            else "",
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
