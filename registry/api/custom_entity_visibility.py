"""
Visibility + authorization helpers for custom entity records.

These are security-relevant: ``_build_visibility_filter`` (the list/count
Mongo predicate) and ``_user_can_view`` (its single-record Python twin) MUST
stay in lockstep — a record visible in the list but 404 on single-fetch (or
vice-versa) is the bug class they guard against. Custom entities deliberately
filter visibility IN the Mongo query (unlike the three default types which
filter post-fetch), because that is the only approach that delivers an
accurate ``total_count`` AND stays index-covered by ``{entity_type:1, visibility:1}``.
"""

from typing import Any

from fastapi import HTTPException

from ..schemas.custom_entity_models import CustomEntityRecord
from ..services.visibility import can_skip_read_visibility
from ..utils.group_names import group_name_variants, groups_intersect


def _build_visibility_filter(
    user_context: dict,
) -> dict[str, Any] | None:
    """Mongo predicate restricting a list/count to records the caller may see.

    Returns None for admins (no restriction). Composed with
    ``{"entity_type": type}`` by the repo via ``query.update()`` — safe because
    this dict has no top-level ``entity_type`` key, so it never clobbers the
    type discriminator.
    """
    # Admins and view_all holders (issue #2099) see every record. This filter
    # only serves the list/count read path (CustomEntityService.list_records).
    if can_skip_read_visibility(user_context):
        return None
    username = user_context.get("username", "")
    groups = user_context.get("groups", []) or []
    return {
        "$or": [
            {"visibility": "public"},
            {"visibility": "private", "owner": username},
            # Every stored form of the caller's groups, so a record restricted
            # to "/team" matches a session carrying "team" (issue #1689).
            {
                "visibility": "group-restricted",
                "allowed_groups": {"$in": group_name_variants(groups)},
            },
        ]
    }


def _user_can_read_record(
    record: CustomEntityRecord,
    user_context: dict,
) -> bool:
    """Single-record check for READ paths (get, get rating).

    Admins and view_all holders (issue #2099) may read every record; everyone
    else gets exactly ``_user_can_view``. Lockstep with
    ``_build_visibility_filter``, which applies the same bypass.
    """
    if can_skip_read_visibility(user_context):
        return True
    return _user_can_view(record, user_context)


def _user_can_view(
    record: CustomEntityRecord,
    user_context: dict,
) -> bool:
    """Single-record READ analogue of ``_build_visibility_filter``.

    Same public / private-owner / group-restricted logic, evaluated in Python
    against one record. Branch order MUST match the filter's ``$or`` clauses.
    Used by GET single: a record the caller can't see returns 404 (not 403 —
    don't disclose existence). Also the gate for WRITE paths (update, upstream
    headers, delete, rate), which must not honor ``view_all``; read paths go
    through ``_user_can_read_record``.
    """
    if user_context.get("is_admin"):
        return True
    if record.visibility == "public":
        return True
    if record.visibility == "private":
        return record.owner == user_context.get("username")
    if record.visibility == "group-restricted":
        return groups_intersect(user_context.get("groups", []), record.allowed_groups)
    return False  # deny-by-default for any unknown visibility


def _require_owner_or_admin(
    record: CustomEntityRecord,
    user_context: dict,
) -> None:
    """Record-level analogue of ``_require_admin``.

    Raises HTTP 403 unless the caller owns the record or is an admin. Used on
    the MUTATING single-record paths (PUT/DELETE).
    """
    if user_context.get("is_admin"):
        return
    if record.owner and record.owner == user_context.get("username"):
        return
    raise HTTPException(status_code=403, detail="You do not have permission to modify this record")
