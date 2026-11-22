"""GET /api/tool-access: which groups can see and call each tool (issue #2102).

Group names show how an organization is structured, so the listing is limited
by caller:

- Admins and holders of the read-only view_all permission see every row.
- Anyone else sees the rows for servers and virtual servers they own.

A caller who owns nothing gets an empty list, not an error, so the response
says nothing about which servers exist.
"""

import logging
from datetime import UTC, datetime
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)

from ..auth.asset_permissions import can_view_all
from ..auth.dependencies import nginx_proxied_auth
from ..schemas.tool_access_models import (
    AccessSummary,
    GroupRef,
    ServerType,
    ToolAccessListResponse,
    ToolAccessRow,
)
from ..services import tool_access_service

logger = logging.getLogger(__name__)

router = APIRouter()

DEFAULT_LIMIT: int = 20
MAX_LIMIT: int = 500
MAX_QUERY_LENGTH: int = 200


def _sees_every_row(
    user_context: dict,
) -> bool:
    """Return True for admins and view_all holders (a named read site for view_all)."""
    return bool(user_context.get("is_admin")) or can_view_all(user_context)


def _summary_groups(
    summary: AccessSummary,
) -> set[str]:
    """Return every registry group named in a summary."""
    names = set(summary.alone)
    for condition in summary.together:
        names.update(condition.groups)
    return names


def _row_groups(
    row: ToolAccessRow,
) -> set[str]:
    """Return every registry group on a row, for seeing or calling."""
    return _summary_groups(row.can_see) | _summary_groups(row.can_call)


def _row_group_names(
    row: ToolAccessRow,
    groups: dict[str, GroupRef],
) -> set[str]:
    """Return every registry and identity provider group name on a row."""
    names: set[str] = set()
    for name in _row_groups(row):
        names.add(name)
        ref = groups.get(name)
        if ref:
            names.update(ref.idp_groups)
    return names


def _row_matches_query(
    row: ToolAccessRow,
    query: str,
    groups: dict[str, GroupRef],
) -> bool:
    """Case-insensitive substring match on tool, server and group names."""
    needle = query.lower()
    fields = [row.tool_name, row.backend_tool_name or "", row.server_name, row.server_path]
    fields.extend(_row_group_names(row, groups))
    return any(needle in value.lower() for value in fields)


def _apply_filters(
    rows: list[ToolAccessRow],
    groups: dict[str, GroupRef],
    q: str | None,
    server_path: str | None,
    server_type: ServerType | None,
    group: str | None,
    include_disabled: bool,
) -> list[ToolAccessRow]:
    """Keep the rows that pass every filter the caller set."""
    wanted_path = server_path.strip("/") if server_path else None
    kept = []
    for row in rows:
        if not include_disabled and not row.enabled:
            continue
        if wanted_path is not None and row.server_path.strip("/") != wanted_path:
            continue
        if server_type is not None and row.server_type != server_type:
            continue
        if group and group not in _row_group_names(row, groups):
            continue
        if q and not _row_matches_query(row, q, groups):
            continue
        kept.append(row)
    return kept


async def _rows_for_caller(
    user_context: dict,
) -> tuple[list[ToolAccessRow], dict[str, GroupRef], str]:
    """Return the rows this caller may see, the group details, and the tier used."""
    if _sees_every_row(user_context):
        data = await tool_access_service.build_listing()
        tier = "admin" if user_context.get("is_admin") else "view_all"
        return [b.row for b in data.rows], data.groups, tier

    username = user_context.get("username") or ""
    if not username:
        return [], {}, "none"
    data = await tool_access_service.build_listing(owner=username)
    # build_listing already kept only this owner's rows; check again on the
    # stored owner so a row can never reach a caller who does not own it.
    return [b.row for b in data.rows if b.owner == username], data.groups, "owner"


def _groups_on_page(
    rows: list[ToolAccessRow],
    groups: dict[str, GroupRef],
) -> dict[str, GroupRef]:
    """Return details only for the groups the page names, keeping responses small."""
    named: set[str] = set()
    for row in rows:
        named.update(_row_groups(row))
    return {name: groups[name] for name in sorted(named) if name in groups}


@router.get(
    "/tool-access",
    response_model=ToolAccessListResponse,
    summary="List which groups can see and call each tool",
)
async def list_tool_access(
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    q: Annotated[
        str | None,
        Query(max_length=MAX_QUERY_LENGTH, description="Match tool, server or group names."),
    ] = None,
    server_path: Annotated[str | None, Query(description="Exact server path.")] = None,
    server_type: Annotated[ServerType | None, Query(description="mcp, openapi or virtual.")] = None,
    group: Annotated[
        str | None, Query(description="Exact registry or identity provider group name.")
    ] = None,
    include_disabled: Annotated[bool, Query(description="Include disabled servers.")] = True,
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ToolAccessListResponse:
    """List tools with the groups that can see them and the groups that can call them.

    Admins and view_all holders get every tool. Anyone else gets the tools on
    servers they own, or an empty list. The listing is informational and
    grants nothing.
    """
    try:
        rows, groups, tier = await _rows_for_caller(user_context)
    except Exception as exc:
        logger.exception("Tool access listing failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Could not load tool access. Try again, and check the registry "
                "logs if it keeps failing."
            ),
        ) from exc

    filtered = _apply_filters(rows, groups, q, server_path, server_type, group, include_disabled)
    page = filtered[offset : offset + limit]
    logger.info(
        "Tool access listing served: user=%s tier=%s matched=%d returned=%d",
        user_context.get("username"),
        tier,
        len(filtered),
        len(page),
    )
    return ToolAccessListResponse(
        rows=page,
        groups=_groups_on_page(page, groups),
        total_count=len(filtered),
        limit=limit,
        offset=offset,
        has_next=offset + limit < len(filtered),
        generated_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
