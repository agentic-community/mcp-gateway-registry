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
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..auth.dependencies import enhanced_auth
from ..core.config import settings
from ..repositories.audit_repository import DocumentDBAuditRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audit", tags=["Audit Logs"])

# Singleton repository instance
_audit_repository: DocumentDBAuditRepository | None = None

# Credential type that marks non-interactive (agent / programmatic) callers.
# The web UI authenticates with a session cookie; agents and service accounts
# send a Bearer token. credential_type is set on every audit record, so it is
# the reliable signal for the human-vs-agent split. Note: a human using a CLI
# with a bearer token is counted as agent traffic by this definition, which is
# acceptable for a gateway whose purpose is non-interactive agent access.
AGENT_CREDENTIAL_TYPE: str = "bearer_token"
HUMAN_CREDENTIAL_TYPE: str = "session_cookie"
# Username assigned to unauthenticated traffic
ANONYMOUS_USERNAME: str = "anonymous"

# Executive summary cache: this endpoint runs many aggregations per call, so a
# short TTL cache (keyed by the days window) shields the database from repeated
# page loads / refreshes. Mirrors the /api/stats caching precedent.
EXEC_SUMMARY_CACHE_TTL_SECONDS: int = 30
_exec_summary_cache: dict[int, tuple[datetime, ExecutiveSummaryResponse]] = {}


def _window_match(
    start: datetime,
    end: datetime,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a MongoDB match filter for a timestamp window.

    Args:
        start: Inclusive start of the window
        end: Inclusive end of the window
        extra: Additional match conditions merged into the filter

    Returns:
        MongoDB match dictionary spanning [start, end]
    """
    match: dict[str, Any] = {"timestamp": {"$gte": start, "$lte": end}}
    if extra:
        match.update(extra)
    return match


async def _count_distinct_usernames(
    repository: DocumentDBAuditRepository,
    match: dict[str, Any],
) -> int:
    """
    Count distinct non-anonymous usernames matching a filter.

    The repository's distinct() already drops falsy values; this helper
    additionally drops the anonymous username so only real identities count.

    Args:
        repository: Audit repository instance
        match: MongoDB match filter scoping the distinct values

    Returns:
        Number of distinct non-anonymous usernames
    """
    usernames = await repository.distinct("identity.username", match)
    return len([u for u in usernames if u and u != ANONYMOUS_USERNAME])


def _percentage(
    part: int,
    total: int,
) -> float:
    """
    Compute a one-decimal percentage of part over total.

    Args:
        part: Numerator value
        total: Denominator value

    Returns:
        Percentage rounded to one decimal, 0.0 when total is zero
    """
    if total <= 0:
        return 0.0
    return round(part / total * 100, 1)


def _non_anonymous_match(
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    """
    Build a window match that excludes anonymous and empty usernames.

    Args:
        start: Inclusive start of the window
        end: Inclusive end of the window

    Returns:
        MongoDB match dictionary scoped to real (non-anonymous) identities
    """
    return _window_match(
        start,
        end,
        {"identity.username": {"$nin": [ANONYMOUS_USERNAME, None, ""]}},
    )


def get_audit_repository() -> DocumentDBAuditRepository:
    """Get or create the audit repository singleton."""
    global _audit_repository
    if _audit_repository is None:
        _audit_repository = DocumentDBAuditRepository()
    return _audit_repository


def require_admin(user_context: dict[str, Any] = Depends(enhanced_auth)) -> dict[str, Any]:
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
    operation: str | None = None
    resource_type: str | None = None
    resource_id: str | None = None


class AuditEventsResponse(BaseModel):
    """Response model for paginated audit events."""

    total: int = Field(description="Total number of matching events")
    limit: int = Field(description="Maximum events per page")
    offset: int = Field(description="Number of events skipped")
    events: list[dict[str, Any]] = Field(description="List of audit events")


class AuditEventDetail(BaseModel):
    """Full audit event detail."""

    event: dict[str, Any] = Field(description="Complete audit event record")


class AuditFilterOptions(BaseModel):
    """Available filter values for audit log dropdowns."""

    usernames: list[str] = Field(
        default_factory=list,
        description="Distinct usernames found in audit events",
    )
    server_names: list[str] = Field(
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
    operations: list[UsageSummaryItem] = Field(
        default_factory=list,
        description="Top operations for this user",
    )


class AuditStatisticsResponse(BaseModel):
    """Aggregated audit statistics."""

    total_events: int = Field(description="Total events in time range")
    top_users: list[UsageSummaryItem] = Field(
        default_factory=list,
        description="Top 10 users by event count",
    )
    top_servers: list[UsageSummaryItem] = Field(
        default_factory=list,
        description="Top 10 MCP servers (MCP stream only)",
    )
    top_operations: list[UsageSummaryItem] = Field(
        default_factory=list,
        description="Top 10 operations by event count",
    )
    activity_timeline: list[TimeSeriesBucket] = Field(
        default_factory=list,
        description="Daily event counts for the time range",
    )
    activity_timeline_prior: list[TimeSeriesBucket] = Field(
        default_factory=list,
        description=(
            "Daily event counts for the prior window of equal length, "
            "for week-over-week overlay"
        ),
    )
    status_distribution: StatusDistribution = Field(
        default_factory=StatusDistribution,
        description="Distribution of HTTP status codes",
    )
    user_activity: list[UserActivityItem] = Field(
        default_factory=list,
        description="Per-user breakdown of top operations",
    )


class GovernanceScope(BaseModel):
    """Scope of assets currently governed by the gateway."""

    mcp_servers_governed: int = Field(
        default=0,
        description="Distinct MCP servers accessed in the current window",
    )
    tools_under_policy: int = Field(
        default=0,
        description="Distinct MCP tools invoked in the current window",
    )
    identities_active: int = Field(
        default=0,
        description="Distinct non-anonymous identities active in the current window",
    )


class RegisteredAssets(BaseModel):
    """Total registered inventory in the registry catalog (not activity-scoped)."""

    servers: int = Field(default=0, description="Total registered MCP servers")
    tools: int = Field(default=0, description="Total tools exposed across all servers")
    agents: int = Field(default=0, description="Total registered agents")
    skills: int = Field(default=0, description="Total registered skills")
    custom_entities: int = Field(default=0, description="Total custom entity records")


class ActiveUsers(BaseModel):
    """Active identity counts over rolling windows."""

    dau: int = Field(default=0, description="Distinct non-anonymous usernames, last 1 day")
    wau: int = Field(default=0, description="Distinct non-anonymous usernames, last 7 days")
    mau: int = Field(default=0, description="Distinct non-anonymous usernames, last 30 days")
    wau_available: bool = Field(
        default=True,
        description="True when retention covers the 7-day window; WAU is honest only then",
    )
    mau_available: bool = Field(
        default=False,
        description="True when retention covers the 30-day window; MAU is honest only then",
    )


class ActiveAgents(BaseModel):
    """Active agent (bearer-token caller) counts over rolling windows."""

    daa: int = Field(default=0, description="Distinct agent identities, last 1 day")
    waa: int = Field(default=0, description="Distinct agent identities, last 7 days")
    maa: int = Field(default=0, description="Distinct agent identities, last 30 days")
    waa_available: bool = Field(
        default=True,
        description="True when retention covers the 7-day window; WAA is honest only then",
    )
    maa_available: bool = Field(
        default=False,
        description="True when retention covers the 30-day window; MAA is honest only then",
    )


class TrafficSplit(BaseModel):
    """Human vs agent traffic split for the current window."""

    human_events: int = Field(default=0, description="Authenticated, non-agent events")
    agent_events: int = Field(default=0, description="Agent/service-account events (jwt_bearer)")
    human_pct: float = Field(
        default=0.0,
        description="Human share over (human+agent), one decimal, 0 when denominator 0",
    )
    agent_pct: float = Field(
        default=0.0,
        description="Agent share over (human+agent), one decimal, 0 when denominator 0",
    )


class AdoptionMomentum(BaseModel):
    """Week-over-week adoption momentum metrics."""

    events_current: int = Field(
        default=0,
        description="Non-anonymous events in the current window",
    )
    events_prior: int = Field(
        default=0,
        description="Non-anonymous events in the prior window",
    )
    events_wow_pct: float | None = Field(
        default=None,
        description="Week-over-week event change percent, None when prior is zero",
    )
    active_identities_current: int = Field(
        default=0,
        description="Distinct non-anonymous identities in the current window",
    )
    active_identities_prior: int = Field(
        default=0,
        description="Distinct non-anonymous identities in the prior window",
    )
    active_agents_current: int = Field(
        default=0,
        description="Distinct agent identities in the current window",
    )
    active_agents_prior: int = Field(
        default=0,
        description="Distinct agent identities in the prior window",
    )
    has_prior_data: bool = Field(
        default=False,
        description="True when the prior window has at least one event",
    )


class ExecutiveSummaryResponse(BaseModel):
    """Global, cross-stream executive summary of gateway activity."""

    window_days: int = Field(description="Length of the comparison window in days")
    retention_days: int = Field(
        default=0,
        description="Configured audit retention (TTL) in days; gates window-N metric availability",
    )
    governance: GovernanceScope = Field(default_factory=GovernanceScope)
    registered_assets: RegisteredAssets = Field(default_factory=RegisteredAssets)
    active_users: ActiveUsers = Field(default_factory=ActiveUsers)
    active_agents: ActiveAgents = Field(default_factory=ActiveAgents)
    traffic_split: TrafficSplit = Field(default_factory=TrafficSplit)
    momentum: AdoptionMomentum = Field(default_factory=AdoptionMomentum)


def _build_query(
    stream: str,
    from_time: datetime | None,
    to_time: datetime | None,
    username: str | None,
    operation: str | None,
    resource_type: str | None,
    resource_id: str | None,
    status_min: int | None,
    status_max: int | None,
    auth_decision: str | None,
) -> dict[str, Any]:
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
        "token_mint": "token_mint",
    }
    query: dict[str, Any] = {"log_type": log_type_map.get(stream, stream)}

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
        if stream == "token_mint":
            query["username_hash"] = {"$regex": escaped_username, "$options": "i"}
        else:
            query["identity.username"] = {"$regex": escaped_username, "$options": "i"}

    # Action filters - different fields per stream
    if stream == "token_mint":
        if operation:
            query["token_kind"] = operation
        if resource_type:
            query["resource_type"] = resource_type
        if resource_id:
            query["resource_id"] = resource_id
    elif stream == "mcp_access":
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
    user_context: Annotated[dict[str, Any], Depends(require_admin)],
    stream: str = Query(
        "registry_api",
        pattern="^(registry_api|mcp_access|token_mint)$",
        description="Log stream type",
    ),
) -> AuditFilterOptions:
    """Get distinct filter values for audit log dropdowns. Requires admin access."""
    start_time = time.time()

    log_type_map = {
        "registry_api": "registry_api_access",
        "mcp_access": "mcp_server_access",
        "token_mint": "token_mint",
    }
    log_type = log_type_map.get(stream, stream)
    query = {"log_type": log_type}

    repository = get_audit_repository()

    username_field = "username_hash" if stream == "token_mint" else "identity.username"
    usernames = await repository.distinct(username_field, query)

    server_names: list[str] = []
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
    user_context: Annotated[dict[str, Any], Depends(require_admin)],
    stream: str = Query(
        "registry_api",
        pattern="^(registry_api|mcp_access|token_mint)$",
        description="Log stream type",
    ),
    days: int = Query(
        7,
        ge=1,
        le=30,
        description="Number of days to include in statistics",
    ),
    username: str | None = Query(
        None,
        description="Filter statistics to a specific username",
    ),
) -> AuditStatisticsResponse:
    """Get aggregated audit statistics for the dashboard. Requires admin access."""
    start_time = time.time()

    log_type_map = {
        "registry_api": "registry_api_access",
        "mcp_access": "mcp_server_access",
        "token_mint": "token_mint",
    }
    log_type = log_type_map.get(stream, stream)
    now = datetime.now(UTC)
    cutoff = now - timedelta(days=days)
    base_match: dict[str, Any] = {"log_type": log_type, "timestamp": {"$gte": cutoff}}

    # Prior window of equal length, ending where the current window starts
    prior_start = now - timedelta(days=days * 2)
    prior_match: dict[str, Any] = {
        "log_type": log_type,
        "timestamp": {"$gte": prior_start, "$lt": cutoff},
    }

    if username:
        escaped_username = re.escape(username)
        if stream == "token_mint":
            username_filter = {"$regex": escaped_username, "$options": "i"}
            base_match["username_hash"] = username_filter
            prior_match["username_hash"] = username_filter
        else:
            username_filter = {"$regex": f"^{escaped_username}$", "$options": "i"}
            base_match["identity.username"] = username_filter
            prior_match["identity.username"] = username_filter

    repository = get_audit_repository()

    # Build all pipelines upfront
    user_field = "$username_hash" if stream == "token_mint" else "$identity.username"
    op_field = (
        "$token_kind" if stream == "token_mint"
        else "$mcp_request.method" if stream == "mcp_access"
        else "$action.operation"
    )

    # Status distribution pipeline differs by stream
    if stream == "mcp_access":
        status_pipeline: list[dict[str, Any]] = [
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
                {"$group": {"_id": user_field, "count": {"$sum": 1}}},
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
                        "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
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
                            "user": user_field,
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
                        "operations": {"$push": {"name": "$_id.op", "count": "$count"}},
                    }
                },
                {"$sort": {"total": -1}},
                {"$limit": 10},
            ]
        ),
        # Prior-window daily timeline for week-over-week overlay
        repository.aggregate(
            [
                {"$match": prior_match},
                {
                    "$group": {
                        "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
                        "count": {"$sum": 1},
                    }
                },
                {"$sort": {"_id": 1}},
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
    timeline_prior_raw = results[6]
    top_servers_raw = results[7] if stream == "mcp_access" else []

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

    activity_timeline = [TimeSeriesBucket(period=r["_id"], count=r["count"]) for r in timeline_raw]

    activity_timeline_prior = [
        TimeSeriesBucket(period=r["_id"], count=r["count"]) for r in timeline_prior_raw
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
            op_name = (
                op.get("name") or op.get("_id", {}).get("op") if isinstance(op, dict) else None
            )
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
    logger.info(f"Audit statistics computed in {elapsed:.2f}s (stream={stream}, days={days})")

    return AuditStatisticsResponse(
        total_events=total_events,
        top_users=top_users,
        top_servers=top_servers,
        top_operations=top_operations,
        activity_timeline=activity_timeline,
        activity_timeline_prior=activity_timeline_prior,
        status_distribution=status_dist,
        user_activity=user_activity,
    )


def _build_traffic_split(
    human_events: int,
    agent_events: int,
) -> TrafficSplit:
    """
    Build the human vs agent traffic split with percentages.

    Args:
        human_events: Count of authenticated non-agent events
        agent_events: Count of agent (jwt_bearer) events

    Returns:
        TrafficSplit with counts and one-decimal percentages
    """
    total = human_events + agent_events
    return TrafficSplit(
        human_events=human_events,
        agent_events=agent_events,
        human_pct=_percentage(human_events, total),
        agent_pct=_percentage(agent_events, total),
    )


def _build_momentum(
    events_current: int,
    events_prior: int,
    identities_current: int,
    identities_prior: int,
    agents_current: int,
    agents_prior: int,
    prior_window_retained: bool,
) -> AdoptionMomentum:
    """
    Build the week-over-week adoption momentum block.

    Args:
        events_current: Non-anonymous events in the current window
        events_prior: Non-anonymous events in the prior window
        identities_current: Distinct identities in the current window
        identities_prior: Distinct identities in the prior window
        agents_current: Distinct agents in the current window
        agents_prior: Distinct agents in the prior window
        prior_window_retained: True when retention covers the full prior window,
            so a zero prior count means genuinely no traffic rather than expired data

    Returns:
        AdoptionMomentum with computed week-over-week change
    """
    # Only trust the prior window when retention covers it AND it has events.
    # Without retention coverage a zero prior count is expired data, not real zero.
    has_prior_data = prior_window_retained and events_prior > 0
    wow_pct = None
    if has_prior_data:
        wow_pct = round((events_current - events_prior) / events_prior * 100, 1)
    return AdoptionMomentum(
        events_current=events_current,
        events_prior=events_prior,
        events_wow_pct=wow_pct,
        active_identities_current=identities_current,
        active_identities_prior=identities_prior,
        active_agents_current=agents_current,
        active_agents_prior=agents_prior,
        has_prior_data=has_prior_data,
    )


async def _get_registered_assets() -> RegisteredAssets:
    """
    Gather total registered inventory counts from the asset repositories.

    All counts run concurrently and each uses an efficient count/aggregation
    (no per-document loads), so this stays O(1) round-trips rather than N+1.

    Returns:
        RegisteredAssets with servers, tools, agents, skills, and custom
        entity totals; zeros for any asset type whose count fails.
    """
    from registry.repositories.factory import (
        get_agent_repository,
        get_custom_entity_repository,
        get_server_repository,
        get_skill_repository,
    )

    server_repo = get_server_repository()
    agent_repo = get_agent_repository()
    skill_repo = get_skill_repository()
    custom_entity_repo = get_custom_entity_repository()

    try:
        servers, tools, agents, skills, custom_entities = await asyncio.gather(
            server_repo.count(exclude_versions=True),
            server_repo.count_tools(),
            agent_repo.count(),
            skill_repo.count(),
            custom_entity_repo.count_all(),
        )
    except Exception:
        logger.exception("Failed to gather registered asset counts")
        return RegisteredAssets()

    return RegisteredAssets(
        servers=servers,
        tools=tools,
        agents=agents,
        skills=skills,
        custom_entities=custom_entities,
    )


async def _compute_executive_summary(
    days: int,
) -> ExecutiveSummaryResponse:
    """
    Compute the global, cross-stream executive summary.

    Runs all audit aggregations and inventory counts. Caching is handled by the
    route wrapper, so this always computes fresh.

    Args:
        days: Length of the comparison window in days

    Returns:
        The computed ExecutiveSummaryResponse
    """
    start_time = time.time()
    repository = get_audit_repository()

    # Retention (TTL) gates which rolling-window metrics are honest. A window of
    # N days is only meaningful when audit events are retained for at least N days.
    retention_days = settings.audit_log_mongodb_ttl_days

    now = datetime.now(UTC)
    cur_start = now - timedelta(days=days)
    prior_start = now - timedelta(days=days * 2)
    mcp_match = {"log_type": "mcp_server_access"}
    # Agents are non-anonymous callers using a bearer token (non-interactive).
    agent_filter = {
        "identity.username": {"$nin": [ANONYMOUS_USERNAME, None, ""]},
        "identity.credential_type": AGENT_CREDENTIAL_TYPE,
    }

    # Current window matches scoped to MCP-only metrics
    cur_window = _window_match(cur_start, now)
    cur_mcp = _window_match(cur_start, now, mcp_match)
    cur_non_anon = _non_anonymous_match(cur_start, now)
    prior_non_anon = _non_anonymous_match(prior_start, cur_start)

    # Split by credential_type: humans use session cookies (interactive web UI),
    # agents use bearer tokens (programmatic). Both exclude anonymous traffic.
    cur_agents = _window_match(cur_start, now, agent_filter)
    cur_humans = _window_match(
        cur_start,
        now,
        {
            "identity.username": {"$nin": [ANONYMOUS_USERNAME, None, ""]},
            "identity.credential_type": HUMAN_CREDENTIAL_TYPE,
        },
    )
    prior_agents = _window_match(prior_start, cur_start, agent_filter)

    # Rolling DAA/WAA/MAA windows: distinct agent identities over 1/7/30 days.
    daa_match = _window_match(now - timedelta(days=1), now, agent_filter)
    waa_match = _window_match(now - timedelta(days=7), now, agent_filter)
    maa_match = _window_match(now - timedelta(days=30), now, agent_filter)

    # Run all independent queries concurrently
    results = await asyncio.gather(
        _count_distinct_usernames(repository, cur_window),
        repository.distinct("mcp_server.name", cur_mcp),
        repository.distinct("mcp_request.tool_name", cur_mcp),
        _count_distinct_usernames(repository, _window_match(now - timedelta(days=1), now)),
        _count_distinct_usernames(repository, _window_match(now - timedelta(days=7), now)),
        _count_distinct_usernames(repository, _window_match(now - timedelta(days=30), now)),
        repository.count(cur_humans),
        repository.count(cur_agents),
        repository.count(cur_non_anon),
        repository.count(prior_non_anon),
        _count_distinct_usernames(repository, cur_non_anon),
        _count_distinct_usernames(repository, prior_non_anon),
        _count_distinct_usernames(repository, cur_agents),
        _count_distinct_usernames(repository, prior_agents),
        _count_distinct_usernames(repository, daa_match),
        _count_distinct_usernames(repository, waa_match),
        _count_distinct_usernames(repository, maa_match),
        _get_registered_assets(),
    )

    governance = GovernanceScope(
        mcp_servers_governed=len([s for s in results[1] if s]),
        tools_under_policy=len([t for t in results[2] if t]),
        identities_active=results[0],
    )
    registered_assets = results[17]
    active_users = ActiveUsers(
        dau=results[3],
        wau=results[4],
        mau=results[5],
        wau_available=retention_days >= 7,
        mau_available=retention_days >= 30,
    )
    active_agents = ActiveAgents(
        daa=results[14],
        waa=results[15],
        maa=results[16],
        waa_available=retention_days >= 7,
        maa_available=retention_days >= 30,
    )
    traffic_split = _build_traffic_split(human_events=results[6], agent_events=results[7])
    momentum = _build_momentum(
        events_current=results[8],
        events_prior=results[9],
        identities_current=results[10],
        identities_prior=results[11],
        agents_current=results[12],
        agents_prior=results[13],
        prior_window_retained=retention_days >= days * 2,
    )

    elapsed = time.time() - start_time
    logger.info(
        f"Executive summary computed in {elapsed:.2f}s "
        f"(days={days}, retention_days={retention_days})"
    )

    return ExecutiveSummaryResponse(
        window_days=days,
        retention_days=retention_days,
        governance=governance,
        registered_assets=registered_assets,
        active_users=active_users,
        active_agents=active_agents,
        traffic_split=traffic_split,
        momentum=momentum,
    )


async def _get_cached_executive_summary(
    days: int,
) -> ExecutiveSummaryResponse:
    """
    Return the executive summary for a window, using a short-TTL cache.

    The summary runs many aggregations, so results are cached per ``days`` for
    EXEC_SUMMARY_CACHE_TTL_SECONDS to shield the database from repeated page
    loads and manual refreshes.

    Args:
        days: Length of the comparison window in days

    Returns:
        Cached or freshly computed ExecutiveSummaryResponse
    """
    now = datetime.now(UTC)
    cached = _exec_summary_cache.get(days)
    if cached is not None:
        cached_at, cached_value = cached
        if (now - cached_at).total_seconds() < EXEC_SUMMARY_CACHE_TTL_SECONDS:
            logger.debug(f"Executive summary cache hit (days={days})")
            return cached_value

    summary = await _compute_executive_summary(days)
    _exec_summary_cache[days] = (now, summary)
    return summary


@router.get("/executive-summary", response_model=ExecutiveSummaryResponse)
async def get_executive_summary(
    user_context: Annotated[dict[str, Any], Depends(require_admin)],
    days: int = Query(
        7,
        ge=1,
        le=30,
        description="Length of the comparison window in days",
    ),
) -> ExecutiveSummaryResponse:
    """Get a global, cross-stream executive summary. Requires admin access."""
    return await _get_cached_executive_summary(days)


@router.get("/events", response_model=AuditEventsResponse)
async def get_audit_events(
    user_context: Annotated[dict[str, Any], Depends(require_admin)],
    stream: str = Query(
        "registry_api",
        pattern="^(registry_api|mcp_access|token_mint)$",
        description="Log stream type",
    ),
    from_time: datetime | None = Query(
        None,
        alias="from",
        description="Start of time range (ISO 8601)",
    ),
    to_time: datetime | None = Query(
        None,
        alias="to",
        description="End of time range (ISO 8601)",
    ),
    username: str | None = Query(
        None,
        description="Filter by username",
    ),
    operation: str | None = Query(
        None,
        description="Filter by operation type",
    ),
    resource_type: str | None = Query(
        None,
        description="Filter by resource type",
    ),
    resource_id: str | None = Query(
        None,
        description="Filter by resource ID",
    ),
    status_min: int | None = Query(
        None,
        ge=100,
        le=599,
        description="Minimum HTTP status code",
    ),
    status_max: int | None = Query(
        None,
        ge=100,
        le=599,
        description="Maximum HTTP status code",
    ),
    auth_decision: str | None = Query(
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
    user_context: Annotated[dict[str, Any], Depends(require_admin)],
    log_type: str | None = Query(
        default=None,
        description="Filter by log type: registry_api_access or mcp_server_access",
    ),
) -> dict[str, Any]:
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
        query: dict[str, Any] = {"request_id": request_id}
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


def _generate_jsonl(events: list[dict[str, Any]]):
    """Generate JSONL output from events."""
    import json

    for event in events:
        # Convert datetime objects to ISO format strings
        if "timestamp" in event and isinstance(event["timestamp"], datetime):
            event["timestamp"] = event["timestamp"].isoformat()
        yield json.dumps(event) + "\n"


def _generate_csv(events: list[dict[str, Any]]):
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
    user_context: Annotated[dict[str, Any], Depends(require_admin)],
    format: str = Query(
        "jsonl",
        pattern="^(jsonl|csv)$",
        description="Export format: jsonl or csv",
    ),
    stream: str = Query(
        "registry_api",
        pattern="^(registry_api|mcp_access|token_mint)$",
        description="Log stream type",
    ),
    from_time: datetime | None = Query(
        None,
        alias="from",
        description="Start of time range (ISO 8601)",
    ),
    to_time: datetime | None = Query(
        None,
        alias="to",
        description="End of time range (ISO 8601)",
    ),
    username: str | None = Query(
        None,
        description="Filter by username",
    ),
    operation: str | None = Query(
        None,
        description="Filter by operation type",
    ),
    resource_type: str | None = Query(
        None,
        description="Filter by resource type",
    ),
    resource_id: str | None = Query(
        None,
        description="Filter by resource ID",
    ),
    status_min: int | None = Query(
        None,
        ge=100,
        le=599,
        description="Minimum HTTP status code",
    ),
    status_max: int | None = Query(
        None,
        ge=100,
        le=599,
        description="Maximum HTTP status code",
    ),
    auth_decision: str | None = Query(
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
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
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
