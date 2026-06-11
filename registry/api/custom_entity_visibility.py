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


def _build_visibility_filter(
    user_context: dict,
) -> dict[str, Any] | None:
    """Mongo predicate restricting a list/count to records the caller may see.

    Returns None for admins (no restriction). Composed with
    ``{"entity_type": type}`` by the repo via ``query.update()`` — safe because
    this dict has no top-level ``entity_type`` key, so it never clobbers the
    type discriminator.
    """
    if user_context.get("is_admin"):
        return None  # admin sees all
    username = user_context.get("username", "")
    groups = user_context.get("groups", []) or []
    return {
        "$or": [
            {"visibility": "public"},
            {"visibility": "private", "owner": username},
            {"visibility": "group-restricted", "allowed_groups": {"$in": groups}},
        ]
    }


def _user_can_view(
    record: CustomEntityRecord,
    user_context: dict,
) -> bool:
    """Single-record READ analogue of ``_build_visibility_filter``.

    Same public / private-owner / group-restricted logic, evaluated in Python
    against one record. Branch order MUST match the filter's ``$or`` clauses.
    Used by GET single: a record the caller can't see returns 404 (not 403 —
    don't disclose existence).
    """
    if user_context.get("is_admin"):
        return True
    if record.visibility == "public":
        return True
    if record.visibility == "private":
        return record.owner == user_context.get("username")
    if record.visibility == "group-restricted":
        user_groups = set(user_context.get("groups", []) or [])
        return bool(user_groups & set(record.allowed_groups))
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
