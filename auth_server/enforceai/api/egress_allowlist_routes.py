from __future__ import annotations

from datetime import (
    datetime,
)
from typing import (
    Optional,
)

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)
from pydantic import (
    BaseModel,
    ConfigDict,
)

from ..auth.dependency import (
    EnforceAIManagementContext,
    get_enforceai_management_context,
    get_enforceai_stores,
)
from ..db.data_layer import (
    EnforceAIStores,
)
from ..egress.allowlist import (
    check_proxy_pass_url,
    normalize_allowlist_entry_value,
)
from ..models.egress_allowlist import (
    EgressAllowlistEntryRecord,
    EgressAllowlistEntryKind,
)
from .management_common import (
    _emit_management_audit_event,
    _get_request_id,
    _require_admin,
)

router = APIRouter()


class EgressAllowlistCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: EgressAllowlistEntryKind
    value: str
    comment: Optional[str] = None
    expires_at: Optional[datetime] = None


class EgressAllowlistUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Optional[EgressAllowlistEntryKind] = None
    value: Optional[str] = None
    comment: Optional[str] = None
    expires_at: Optional[datetime] = None


class EgressAllowlistCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proxy_pass_url: str


class EgressAllowlistCheckResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed: bool
    reason: str
    matched_entry: Optional[EgressAllowlistEntryRecord] = None


@router.get(
    "/admin/egress-allowlist",
    response_model=list[EgressAllowlistEntryRecord],
)
async def admin_list_egress_allowlist(
    include_expired: bool = False,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> list[EgressAllowlistEntryRecord]:
    _require_admin(context)
    return stores.egress_allowlist_store.list_entries(include_expired=include_expired)


@router.post(
    "/admin/egress-allowlist",
    response_model=EgressAllowlistEntryRecord,
)
async def admin_create_egress_allowlist_entry(
    request: Request,
    payload: EgressAllowlistCreateRequest,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> EgressAllowlistEntryRecord:
    _require_admin(context)
    request_id = _get_request_id(request)

    try:
        normalized_value = normalize_allowlist_entry_value(
            kind=payload.kind,
            value=payload.value,
        )
        record = stores.egress_allowlist_store.create_entry(
            kind=payload.kind,
            value=normalized_value,
            comment=payload.comment,
            expires_at=payload.expires_at,
        )
    except ValueError as exc:
        _emit_management_audit_event(
            stores=stores,
            action="admin/egress-allowlist/create",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"error": str(exc)},
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _emit_management_audit_event(
        stores=stores,
        action="admin/egress-allowlist/create",
        outcome="allow",
        user_id=context.user_id,
        agent_id=context.actor_agent_id,
        request_id=request_id,
        details={"entry_id": record.entry_id, "kind": record.kind, "value": record.value},
    )
    return record


@router.put(
    "/admin/egress-allowlist/{entry_id}",
    response_model=EgressAllowlistEntryRecord,
)
async def admin_update_egress_allowlist_entry(
    entry_id: int,
    request: Request,
    payload: EgressAllowlistUpdateRequest,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> EgressAllowlistEntryRecord:
    _require_admin(context)
    request_id = _get_request_id(request)

    existing = stores.egress_allowlist_store.get_entry_by_id(entry_id=entry_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Allowlist entry not found")

    if payload.kind is not None and payload.value is None:
        raise HTTPException(
            status_code=400,
            detail="value is required when updating kind",
        )

    kind = payload.kind
    effective_kind = payload.kind or existing.kind
    value = payload.value
    if value is not None:
        value = normalize_allowlist_entry_value(
            kind=effective_kind,
            value=value,
        )

    updated = stores.egress_allowlist_store.update_entry(
        entry_id=entry_id,
        kind=kind,
        value=value,
        comment=payload.comment,
        expires_at=payload.expires_at,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Allowlist entry not found")

    _emit_management_audit_event(
        stores=stores,
        action="admin/egress-allowlist/update",
        outcome="allow",
        user_id=context.user_id,
        agent_id=context.actor_agent_id,
        request_id=request_id,
        details={"entry_id": entry_id},
    )
    return updated


@router.delete(
    "/admin/egress-allowlist/{entry_id}",
    response_model=dict[str, bool],
)
async def admin_delete_egress_allowlist_entry(
    entry_id: int,
    request: Request,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> dict[str, bool]:
    _require_admin(context)
    request_id = _get_request_id(request)

    deleted = stores.egress_allowlist_store.delete_entry(entry_id=entry_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Allowlist entry not found")

    _emit_management_audit_event(
        stores=stores,
        action="admin/egress-allowlist/delete",
        outcome="allow",
        user_id=context.user_id,
        agent_id=context.actor_agent_id,
        request_id=request_id,
        details={"entry_id": entry_id},
    )

    return {"ok": True}


@router.post(
    "/admin/egress-allowlist/check",
    response_model=EgressAllowlistCheckResponse,
)
async def admin_check_proxy_pass_url_allowlist(
    payload: EgressAllowlistCheckRequest,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> EgressAllowlistCheckResponse:
    _require_admin(context)
    entries = stores.egress_allowlist_store.list_entries(include_expired=False)
    decision = check_proxy_pass_url(
        proxy_pass_url=payload.proxy_pass_url,
        entries=entries,
    )
    return EgressAllowlistCheckResponse(
        allowed=decision.allowed,
        reason=decision.reason,
        matched_entry=decision.matched_entry,
    )

