"""Admin CRUD + status endpoints for rate-limit definitions (issue #295).

Definitions live in the ``mcp_rate_limits`` collection and are enforced at the
auth-server ``/validate`` hop. All mutating endpoints are admin-only, mirroring
``m2m_management_routes.py``. The ``_id`` is always derived server-side from the
definition body (``<axis>:<entity_type>:<name>:<window_seconds>``); the URL id on
PUT is only a consistency check, never parsed into fields (names may contain
colons).
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ValidationError

from registry.audit.context import set_audit_action
from registry.auth.csrf import verify_csrf_token_flexible
from registry.auth.dependencies import nginx_proxied_auth
from registry.core.config import settings
from registry.rate_limiting.definitions_repository import DefinitionsRepository
from registry.rate_limiting.memberships_repository import MembershipsRepository
from registry.rate_limiting.models import (
    CALLER_SUBJECT_TYPES,
    GROUP_AXES,
    QUARANTINE_CALLER_GROUP,
    QUARANTINE_TARGET_GROUP,
    RESERVED_GROUP_NAMES,
    TARGET_SUBJECT_TYPES,
    UNENFORCED_ENTITY_TYPES,
    RateLimitDefinition,
    RateLimitMembership,
    normalize_target_subject,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_RESOURCE_TYPE: str = "rate_limit"


def _require_admin(
    user_context: dict | None,
) -> None:
    """Enforce admin permission or raise 401/403."""
    if not user_context:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not user_context.get("is_admin"):
        raise HTTPException(status_code=403, detail="Administrator permissions are required")


# Floor is a per-minute rate; enforced only on short windows so daily/hourly
# volume caps (which are legitimately below a per-minute floor) are exempt.
_FLOOR_WINDOW_SECONDS: int = 60


def _validate_enforceable(
    definition: RateLimitDefinition,
) -> None:
    """Reject definitions whose entity_type is not yet enforced (fail-closed footgun guard).

    ``mcp_tool`` / ``a2a_skill`` are modeled but their classifier is a later phase,
    so accepting them would create a silently-inert limit. Reject with a clear message.
    ``server_group`` IS enforced (surfaced per-member at target lookup time), so it is
    intentionally NOT in UNENFORCED_ENTITY_TYPES and passes this guard.
    """
    if definition.entity_type in UNENFORCED_ENTITY_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"entity_type '{definition.entity_type}' is not enforced in this version "
                f"(tool/skill rate limiting is a later phase)"
            ),
        )


def _validate_caller_floor(
    definition: RateLimitDefinition,
) -> None:
    """Reject a group definition whose short-window limit is below its caller-type floor.

    Lockout safeguard: on windows <= 60s, a group's ``user_max_requests`` must be
    >= the user floor and ``agent_max_requests`` >= the agent floor (both config-only,
    no API). Longer windows (hourly/daily volume caps) are exempt because a low
    per-day cap is a legitimate limit that a per-minute floor should not block.
    """
    if definition.axis not in GROUP_AXES or definition.window_seconds > _FLOOR_WINDOW_SECONDS:
        return

    user_floor = settings.rate_limit_user_floor_per_min
    agent_floor = settings.rate_limit_agent_floor_per_min

    if definition.user_max_requests is not None and definition.user_max_requests < user_floor:
        raise HTTPException(
            status_code=400,
            detail=(
                f"user_max_requests {definition.user_max_requests} is below the user floor "
                f"of {user_floor}/min for windows <= {_FLOOR_WINDOW_SECONDS}s. Set it to at "
                f"least {user_floor} (the floor is a fixed config safeguard against lockout)."
            ),
        )
    if definition.agent_max_requests is not None and definition.agent_max_requests < agent_floor:
        raise HTTPException(
            status_code=400,
            detail=(
                f"agent_max_requests {definition.agent_max_requests} is below the agent floor "
                f"of {agent_floor}/min for windows <= {_FLOOR_WINDOW_SECONDS}s. Set it to at "
                f"least {agent_floor} (the floor is a fixed config safeguard against lockout)."
            ),
        )


def _parse_definition(
    body: dict,
) -> RateLimitDefinition:
    """Parse a request body into a RateLimitDefinition or raise 400."""
    try:
        return RateLimitDefinition(**body)
    except ValidationError as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid rate-limit definition: {exc}"
        ) from exc


_repository: DefinitionsRepository | None = None
_memberships_repository: MembershipsRepository | None = None


def _get_repository() -> DefinitionsRepository:
    """Return a shared DefinitionsRepository singleton for the registry process."""
    global _repository
    if _repository is None:
        _repository = DefinitionsRepository()
    return _repository


def _get_memberships_repository() -> MembershipsRepository:
    """Return a shared MembershipsRepository singleton for the registry process."""
    global _memberships_repository
    if _memberships_repository is None:
        _memberships_repository = MembershipsRepository()
    return _memberships_repository


def _parse_membership(
    body: dict,
) -> RateLimitMembership:
    """Parse a request body into a RateLimitMembership or raise 400."""
    try:
        return RateLimitMembership(**body)
    except ValidationError as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid rate-limit membership: {exc}"
        ) from exc


class _DeleteResponse(BaseModel):
    """Response body for a delete operation."""

    deleted: bool


@router.get("/rate-limits")
async def list_rate_limits(
    request: Request,
    user_context: Annotated[dict | None, Depends(nginx_proxied_auth)] = None,
) -> dict:
    """List all rate-limit definitions (admin only)."""
    _require_admin(user_context)
    set_audit_action(request, "list", _RESOURCE_TYPE)
    definitions = await _get_repository().list_all()
    return {"definitions": [d.model_dump() for d in definitions]}


@router.get("/rate-limits/{definition_id:path}")
async def get_rate_limit(
    definition_id: str,
    request: Request,
    user_context: Annotated[dict | None, Depends(nginx_proxied_auth)] = None,
) -> dict:
    """Read a single rate-limit definition by id (admin only). 404 if absent."""
    _require_admin(user_context)
    set_audit_action(request, "read", _RESOURCE_TYPE, resource_id=definition_id)
    definition = await _get_repository().get_by_id(definition_id)
    if definition is None:
        raise HTTPException(status_code=404, detail="Definition not found")
    return definition.model_dump()


@router.post("/rate-limits-enabled/{definition_id:path}")
async def set_rate_limit_enabled(
    definition_id: str,
    request: Request,
    enabled: Annotated[bool, Query()],
    user_context: Annotated[dict | None, Depends(nginx_proxied_auth)] = None,
    _csrf: Annotated[None, Depends(verify_csrf_token_flexible)] = None,
) -> dict:
    """Enable or disable a definition in place without re-specifying it (admin only).

    A distinct ``/rate-limits-enabled/`` prefix avoids the greedy ``:path`` under
    ``/rate-limits/`` swallowing a trailing action segment.
    """
    _require_admin(user_context)
    action = "enable" if enabled else "disable"
    set_audit_action(request, action, _RESOURCE_TYPE, resource_id=definition_id)
    updated = await _get_repository().set_enabled(definition_id, enabled)
    if updated is None:
        raise HTTPException(status_code=404, detail="Definition not found")
    return updated.model_dump()


@router.put("/rate-limits/{definition_id:path}")
async def put_rate_limit(
    definition_id: str,
    body: dict,
    request: Request,
    user_context: Annotated[dict | None, Depends(nginx_proxied_auth)] = None,
    _csrf: Annotated[None, Depends(verify_csrf_token_flexible)] = None,
) -> dict:
    """Create or update a rate-limit definition (admin only).

    The ``_id`` is derived from the body fields; the URL ``{definition_id}`` must
    match it exactly or the request is rejected (400). The URL id is never split.
    """
    _require_admin(user_context)
    set_audit_action(request, "update", _RESOURCE_TYPE, resource_id=definition_id)
    definition = _parse_definition(body)
    _validate_enforceable(definition)
    _validate_caller_floor(definition)

    built_id = definition.build_id()
    if built_id != definition_id:
        raise HTTPException(
            status_code=400,
            detail=(
                f"URL id '{definition_id}' does not match the definition "
                f"'{built_id}' derived from the body"
            ),
        )

    stored = await _get_repository().upsert(definition)
    return stored.model_dump()


@router.delete("/rate-limits/{definition_id:path}")
async def delete_rate_limit(
    definition_id: str,
    request: Request,
    user_context: Annotated[dict | None, Depends(nginx_proxied_auth)] = None,
    _csrf: Annotated[None, Depends(verify_csrf_token_flexible)] = None,
) -> _DeleteResponse:
    """Delete a rate-limit definition by id (admin only)."""
    _require_admin(user_context)
    set_audit_action(request, "delete", _RESOURCE_TYPE, resource_id=definition_id)
    # A reserved quarantine group sentinel cannot be deleted (it must always exist
    # and be visible in the API/UI). It may be DISABLED instead (the global
    # kill-switch off-toggle via /rate-limits-enabled).
    for reserved in RESERVED_GROUP_NAMES:
        if definition_id.endswith(f":{reserved}:1") or f":{reserved}:" in definition_id:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"'{reserved}' is a reserved quarantine group and cannot be deleted; "
                    f"disable it instead to turn the kill switch off globally"
                ),
            )
    deleted = await _get_repository().delete(definition_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Definition not found")
    return _DeleteResponse(deleted=True)


@router.get("/rate-limits-status")
async def rate_limit_status(
    request: Request,
    identity: Annotated[str | None, Query()] = None,
    entity_type: Annotated[str | None, Query()] = None,
    name: Annotated[str | None, Query()] = None,
    user_context: Annotated[dict | None, Depends(nginx_proxied_auth)] = None,
) -> dict:
    """Introspect current definitions for a caller and/or a target entity (admin only).

    Returns the matching definitions (windows and limits) without consuming any
    counter. Counter-value introspection is a later enhancement.
    """
    _require_admin(user_context)
    set_audit_action(request, "read", _RESOURCE_TYPE)
    repository = _get_repository()

    result: dict = {"caller": [], "target": []}
    if identity:
        # The caller's group definitions are what governs them; the caller passes
        # the group name(s) via identity for introspection.
        caller_defs = await repository.list_caller_limits("group", [identity])
        result["caller"] = [d.model_dump() for d in caller_defs]
    if entity_type and name:
        target_defs = await repository.list_target_limits(entity_type, name)
        result["target"] = [d.model_dump() for d in target_defs]
    return result


# ---------------------------------------------------------------------------
# Rate-limit MEMBERSHIPS: which caller (user/client) belongs to which rate-limit
# group. This is the ONLY source of a caller's rate-limit groups (no IdP emits
# them; kept separate from authz groups). Admin-only, mirroring the definitions.
# ---------------------------------------------------------------------------

_MEMBERSHIP_RESOURCE_TYPE: str = "rate_limit_membership"


@router.get("/rate-limit-memberships")
async def list_rate_limit_memberships(
    request: Request,
    user_context: Annotated[dict | None, Depends(nginx_proxied_auth)] = None,
) -> dict:
    """List all rate-limit memberships (admin only)."""
    _require_admin(user_context)
    set_audit_action(request, "list", _MEMBERSHIP_RESOURCE_TYPE)
    memberships = await _get_memberships_repository().list_all()
    return {"memberships": [m.model_dump() for m in memberships]}


@router.get("/rate-limit-memberships/{membership_id:path}")
async def get_rate_limit_membership(
    membership_id: str,
    request: Request,
    user_context: Annotated[dict | None, Depends(nginx_proxied_auth)] = None,
) -> dict:
    """Read a single membership by id ('<subject_type>:<subject>'); 404 if absent (admin only)."""
    _require_admin(user_context)
    set_audit_action(request, "read", _MEMBERSHIP_RESOURCE_TYPE, resource_id=membership_id)
    membership = await _get_memberships_repository().get_by_id(membership_id)
    if membership is None:
        raise HTTPException(status_code=404, detail="Membership not found")
    return membership.model_dump()


@router.put("/rate-limit-memberships/{membership_id:path}")
async def put_rate_limit_membership(
    membership_id: str,
    body: dict,
    request: Request,
    user_context: Annotated[dict | None, Depends(nginx_proxied_auth)] = None,
    _csrf: Annotated[None, Depends(verify_csrf_token_flexible)] = None,
) -> dict:
    """Create or update a membership (admin only).

    The ``_id`` is derived from the body (``<subject_type>:<subject>``); the URL id
    must match it exactly (a colon-bearing subject is never parsed out of the URL).
    """
    _require_admin(user_context)
    set_audit_action(request, "update", _MEMBERSHIP_RESOURCE_TYPE, resource_id=membership_id)
    membership = _parse_membership(body)

    # Normalize a TARGET subject to the classifier's form (server -> bare name,
    # agent -> leading-slash) so a membership stored here can't silently miss at
    # enforcement. The URL id is normalized the same way before the match, so both
    # 'server:/aws-kb' and 'server:aws-kb' are accepted and stored canonically.
    normalized_subject = normalize_target_subject(membership.subject_type, membership.subject)
    if normalized_subject != membership.subject:
        membership = RateLimitMembership(
            subject_type=membership.subject_type,
            subject=normalized_subject,
            groups=membership.groups,
        )
    url_type, _, url_subject = membership_id.partition(":")
    normalized_membership_id = f"{url_type}:{normalize_target_subject(url_type, url_subject)}"

    built_id = membership.build_id()
    if built_id != normalized_membership_id:
        raise HTTPException(
            status_code=400,
            detail=(
                f"URL id '{membership_id}' does not match the membership "
                f"'{built_id}' derived from the body"
            ),
        )

    # Closing the bypass: quarantining is also expressible by adding the reserved
    # quarantine-callers group here. An admin caller must never be quarantined.
    if QUARANTINE_CALLER_GROUP in (membership.groups or []):
        await _refuse_if_admin_caller(membership.subject_type, membership.subject)

    stored = await _get_memberships_repository().upsert(membership)
    return stored.model_dump()


@router.delete("/rate-limit-memberships/{membership_id:path}")
async def delete_rate_limit_membership(
    membership_id: str,
    request: Request,
    user_context: Annotated[dict | None, Depends(nginx_proxied_auth)] = None,
    _csrf: Annotated[None, Depends(verify_csrf_token_flexible)] = None,
) -> _DeleteResponse:
    """Delete a membership by id (admin only)."""
    _require_admin(user_context)
    set_audit_action(request, "delete", _MEMBERSHIP_RESOURCE_TYPE, resource_id=membership_id)
    deleted = await _get_memberships_repository().delete(membership_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Membership not found")
    return _DeleteResponse(deleted=True)


# ---------------------------------------------------------------------------
# QUARANTINE (kill-switch) convenience endpoints. Quarantine = membership in a
# reserved group; these wrap the memberships collection so an operator moves a
# subject in/out with one call and can never mis-scope (the group is chosen from
# the subject type). Admin-only, audited, CSRF-guarded on mutations.
# ---------------------------------------------------------------------------

_QUARANTINE_RESOURCE_TYPE: str = "rate_limit_quarantine"


def _split_subject_id(
    subject_id: str,
) -> tuple[str, str]:
    """Split '<subject_type>:<subject>' into (subject_type, subject); 400 on a bad shape.

    Only the FIRST colon is split so a subject that itself contains a colon (an
    agent path) is preserved intact.
    """
    subject_type, _, subject = subject_id.partition(":")
    if not subject_type or not subject:
        raise HTTPException(
            status_code=400,
            detail="subject_id must be '<subject_type>:<subject>' (e.g. 'user:alice', 'server:mcpgw')",
        )
    if subject_type not in (CALLER_SUBJECT_TYPES | TARGET_SUBJECT_TYPES):
        raise HTTPException(
            status_code=400,
            detail=f"invalid subject_type '{subject_type}'",
        )
    # Normalize a TARGET subject to the form the request classifier keys on
    # (server -> bare name, agent -> leading-slash path). Without this a subject
    # stored as '/aws-kb' never matches the classified 'aws-kb' and the quarantine
    # is silently inert. Caller subjects (user/client) are identities, left as-is.
    subject = normalize_target_subject(subject_type, subject)
    if not subject:
        raise HTTPException(
            status_code=400,
            detail=f"empty subject for subject_type '{subject_type}'",
        )
    return subject_type, subject


def _quarantine_group_for(
    subject_type: str,
) -> str:
    """Pick the reserved group for a subject type (caller vs target). Mis-scope impossible."""
    if subject_type in TARGET_SUBJECT_TYPES:
        return QUARANTINE_TARGET_GROUP
    return QUARANTINE_CALLER_GROUP


async def _subject_is_admin(
    subject_type: str,
    subject: str,
) -> bool:
    """Return True if a caller subject (user/client) resolves to admin privileges.

    Uses the SAME predicate as the request-time ``is_admin`` check: resolve the
    subject's rate-limit-independent authz groups -> scopes -> ui_permissions, and
    apply ``_user_is_admin`` (admin == a MUTATING ui action granted to ``all``).
    This is deliberately NOT the broader privileged-write predicate used by the
    last-admin guard: a read-only ``list_service: ["all"]`` grant (e.g. the
    public-mcp-users group) must NOT count as admin here.

    Raises HTTPException(503) if group resolution fails, so the caller can fail
    closed (refuse the quarantine) rather than proceed on an unverifiable picture.
    """
    from registry.auth.dependencies import (
        _user_is_admin,
        get_ui_permissions_for_user,
        map_cognito_groups_to_scopes,
    )

    try:
        groups = await _resolve_caller_authz_groups(subject_type, subject)
        scopes = await map_cognito_groups_to_scopes(groups)
        ui_permissions = await get_ui_permissions_for_user(scopes)
    except Exception as exc:
        logger.error(
            "quarantine admin-guard: failed to resolve %s:%s groups: %s", subject_type, subject, exc
        )
        raise HTTPException(
            status_code=503,
            detail="Unable to verify whether the subject is an administrator; quarantine refused",
        ) from exc
    return _user_is_admin(ui_permissions)


async def _resolve_caller_authz_groups(
    subject_type: str,
    subject: str,
) -> list[str]:
    """Resolve a caller subject's authz (IdP) groups: username groups or M2M client groups.

    Mirrors the sources the user-management listing (and admin_safety) use so the
    admin picture matches: the IdP user listing for a user, and the
    ``idp_m2m_clients`` collection for an M2M client (its authoritative group store
    on non-Keycloak IdPs). For a user we also union the local fallback-group store
    (used when an IdP JWT carries no groups claim).
    """
    if subject_type == "client":
        # M2M client groups live in idp_m2m_clients (keyed by client_id or name).
        from registry.repositories.documentdb.client import get_documentdb_client

        db = await get_documentdb_client()
        doc = await db["idp_m2m_clients"].find_one(
            {"$or": [{"client_id": subject}, {"name": subject}]}
        )
        return list(doc.get("groups") or []) if doc else []

    # user subject: prefer the IdP listing (authoritative for Keycloak/Cognito),
    # union the local fallback store (non-Keycloak providers without a groups claim).
    groups: set[str] = set()
    from registry.utils.iam_manager import get_iam_manager

    iam = get_iam_manager()
    users = await iam.list_users(search=subject, max_results=50, include_groups=True)
    normalized = subject.strip().casefold()
    for user in users or []:
        if not isinstance(user, dict):
            continue
        uname = user.get("username") or user.get("id") or ""
        if str(uname).strip().casefold() == normalized:
            groups.update(user.get("groups") or [])

    from registry.services.user_group_management_service import (
        get_user_group_management_service,
    )

    svc = await get_user_group_management_service()
    groups.update(await svc.get_user_groups_for_username(subject))
    return list(groups)


async def _refuse_if_admin_caller(
    subject_type: str,
    subject: str,
) -> None:
    """Refuse to quarantine a CALLER subject that resolves to admin privileges.

    A quarantined caller has ALL its data-plane traffic dropped; quarantining an
    administrator would be a self-inflicted lockout of an operator who manages the
    gateway. Only caller subjects (user/client) can be admins -- a target
    (server/agent) never is -- so the check is scoped to callers. Fails closed via
    ``_subject_is_admin`` (503 on an unresolvable picture).
    """
    if subject_type not in CALLER_SUBJECT_TYPES:
        return
    if await _subject_is_admin(subject_type, subject):
        raise HTTPException(
            status_code=403,
            detail=(
                f"Refusing to quarantine '{subject}': it has administrator privileges. "
                "Administrators cannot be quarantined (self-lockout guard)."
            ),
        )


@router.get("/rate-limit-quarantine")
async def list_quarantine(
    request: Request,
    user_context: Annotated[dict | None, Depends(nginx_proxied_auth)] = None,
) -> dict:
    """List everything currently quarantined (members of both reserved groups)."""
    _require_admin(user_context)
    set_audit_action(request, "list", _QUARANTINE_RESOURCE_TYPE)
    repo = _get_memberships_repository()
    callers = await repo.list_group_members(QUARANTINE_CALLER_GROUP)
    targets = await repo.list_group_members(QUARANTINE_TARGET_GROUP)
    return {
        "callers": [m.model_dump() for m in callers],
        "targets": [m.model_dump() for m in targets],
    }


@router.post("/rate-limit-quarantine/{subject_id:path}")
async def add_quarantine(
    subject_id: str,
    request: Request,
    user_context: Annotated[dict | None, Depends(nginx_proxied_auth)] = None,
    _csrf: Annotated[None, Depends(verify_csrf_token_flexible)] = None,
) -> dict:
    """Add a subject to its quarantine group (drops ALL its data-plane traffic).

    Caller subjects (user/client) keep any existing rate-limit groups; target
    subjects (server/agent) get exactly [quarantine-targets].
    """
    _require_admin(user_context)
    subject_type, subject = _split_subject_id(subject_id)
    # An administrator must never be quarantinable (self-lockout guard); fail closed.
    await _refuse_if_admin_caller(subject_type, subject)
    group = _quarantine_group_for(subject_type)
    set_audit_action(request, "quarantine_add", _QUARANTINE_RESOURCE_TYPE, resource_id=subject_id)
    repo = _get_memberships_repository()

    if subject_type in TARGET_SUBJECT_TYPES:
        # A target subject may only ever hold the target quarantine group.
        membership = RateLimitMembership(subject_type=subject_type, subject=subject, groups=[group])
    else:
        # Preserve the caller's existing rate-limit groups; add the reserved one.
        existing = await repo.get_by_id(f"{subject_type}:{subject}")
        groups = list(existing.groups) if existing else []
        if group not in groups:
            groups.append(group)
        membership = RateLimitMembership(subject_type=subject_type, subject=subject, groups=groups)
    stored = await repo.upsert(membership)
    # Refresh the member-count gauge cache (async count; not on the hot path).
    await repo.count_group_members(group)
    return stored.model_dump()


@router.delete("/rate-limit-quarantine/{subject_id:path}")
async def remove_quarantine(
    subject_id: str,
    request: Request,
    user_context: Annotated[dict | None, Depends(nginx_proxied_auth)] = None,
    _csrf: Annotated[None, Depends(verify_csrf_token_flexible)] = None,
) -> dict:
    """Remove a subject from its quarantine group.

    A caller keeps its other rate-limit groups (only the reserved group is
    removed); a target whose only group was quarantine has its membership deleted.
    """
    _require_admin(user_context)
    subject_type, subject = _split_subject_id(subject_id)
    group = _quarantine_group_for(subject_type)
    set_audit_action(
        request, "quarantine_remove", _QUARANTINE_RESOURCE_TYPE, resource_id=subject_id
    )
    repo = _get_memberships_repository()
    doc_id = f"{subject_type}:{subject}"
    existing = await repo.get_by_id(doc_id)
    if existing is None or group not in existing.groups:
        return {"removed": False}

    remaining = [g for g in existing.groups if g != group]
    if remaining:
        await repo.upsert(
            RateLimitMembership(subject_type=subject_type, subject=subject, groups=remaining)
        )
    else:
        await repo.delete(doc_id)
    await repo.count_group_members(group)
    return {"removed": True}
