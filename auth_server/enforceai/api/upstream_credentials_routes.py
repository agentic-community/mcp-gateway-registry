from __future__ import annotations

from typing import (
    Optional,
)

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)

from ..auth.dependency import (
    EnforceAIManagementContext,
    get_enforceai_management_context,
    get_enforceai_stores,
)
from ..db.data_layer import (
    EnforceAIStores,
)
from ..models.upstream_credentials import (
    UpstreamCredentialRecord,
)
from ..models.upstream_management import (
    UpstreamCredentialCreateRequest,
    UpstreamCredentialCreateResponse,
    UpstreamCredentialRevokeRequest,
    UpstreamServerSummary,
)
from .management_common import (
    _emit_management_audit_event,
    _get_request_id,
    _require_upstream_credential_store,
)
from .upstream_common import (
    _credential_key,
    _is_upstream_credential_visible,
    _normalize_server_path,
    _owned_agent_ids_for_user,
)

router = APIRouter()


@router.get(
    "/upstream/servers",
    response_model=list[UpstreamServerSummary],
)
async def list_upstream_servers(
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
    include_service: bool = False,
) -> list[UpstreamServerSummary]:
    upstream_store = _require_upstream_credential_store(stores)

    owned_agent_ids = _owned_agent_ids_for_user(
        stores=stores,
        user_id=context.user_id,
    )

    records = upstream_store.list_credentials(include_revoked=False)
    visible = [
        record
        for record in records
        if _is_upstream_credential_visible(
            record=record,
            context=context,
            owned_agent_ids=owned_agent_ids,
            include_service=include_service,
        )
    ]

    by_server: dict[str, int] = {}
    for record in visible:
        by_server[record.server_path] = by_server.get(record.server_path, 0) + 1

    return [
        UpstreamServerSummary(
            server_path=server_path,
            active_credential_count=count,
        )
        for server_path, count in sorted(by_server.items(), key=lambda item: item[0])
    ]


@router.get(
    "/upstream/servers/{server_path:path}/credentials",
    response_model=list[UpstreamCredentialRecord],
)
async def list_upstream_credentials_for_server(
    server_path: str,
    include_revoked: bool = False,
    include_service: bool = False,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> list[UpstreamCredentialRecord]:
    upstream_store = _require_upstream_credential_store(stores)
    normalized_server_path = _normalize_server_path(server_path)

    owned_agent_ids = _owned_agent_ids_for_user(
        stores=stores,
        user_id=context.user_id,
    )

    records = upstream_store.list_credentials(
        server_path=normalized_server_path,
        include_revoked=include_revoked,
    )
    return [
        record
        for record in records
        if _is_upstream_credential_visible(
            record=record,
            context=context,
            owned_agent_ids=owned_agent_ids,
            include_service=include_service,
        )
    ]


@router.post(
    "/upstream/servers/{server_path:path}/credentials",
    response_model=UpstreamCredentialCreateResponse,
)
async def create_upstream_credential(
    server_path: str,
    payload: UpstreamCredentialCreateRequest,
    request: Request,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> UpstreamCredentialCreateResponse:
    upstream_store = _require_upstream_credential_store(stores)
    request_id = _get_request_id(request)

    normalized_server_path = _normalize_server_path(server_path)

    if payload.credential_binding == "service" and not context.is_admin:
        raise HTTPException(status_code=403, detail="Admin required for service-bound credentials")

    provider = payload.provider.strip() if payload.provider is not None else None
    provider = provider or None

    user_id: Optional[str] = None
    agent_id: Optional[str] = None

    if payload.credential_binding == "service":
        user_id = None
        agent_id = None
    elif payload.credential_binding == "user":
        user_id = context.user_id
        agent_id = None
    elif payload.credential_binding == "agent":
        agent_id = payload.agent_id
        if agent_id is None:
            raise HTTPException(status_code=400, detail="agent_id is required")
        agent = stores.agent_store.get_agent_by_id(agent_id=agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        if not context.is_admin and agent.user_id != context.user_id:
            raise HTTPException(status_code=404, detail="Agent not found")
        user_id = None
    elif payload.credential_binding == "user+agent":
        agent_id = payload.agent_id
        if agent_id is None:
            raise HTTPException(status_code=400, detail="agent_id is required")
        agent = stores.agent_store.get_agent_by_id(agent_id=agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        if agent.user_id != context.user_id:
            raise HTTPException(status_code=404, detail="Agent not found")
        user_id = context.user_id
    else:
        raise HTTPException(status_code=400, detail="Invalid credential_binding")

    try:
        existing = upstream_store.list_credentials(
            server_path=normalized_server_path,
            include_revoked=False,
        )
        desired_key = (
            normalized_server_path,
            payload.credential_type,
            payload.credential_binding,
            user_id,
            agent_id,
            provider,
        )
        for record in existing:
            if _credential_key(record) == desired_key:
                upstream_store.revoke_credential(credential_id=record.credential_id)

        created = upstream_store.create_credential(
            server_path=normalized_server_path,
            credential_type=payload.credential_type,
            credential_binding=payload.credential_binding,
            user_id=user_id,
            agent_id=agent_id,
            provider=provider,
            scopes=payload.scopes,
            token_type=payload.token_type,
            expires_at=payload.expires_at,
            secret_payload=payload.secret_payload,
        )
    except ValueError as exc:
        _emit_management_audit_event(
            stores=stores,
            action="upstream/credentials/create",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"error": str(exc)},
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _emit_management_audit_event(
        stores=stores,
        action="upstream/credentials/create",
        outcome="allow",
        user_id=context.user_id,
        agent_id=context.actor_agent_id,
        request_id=request_id,
        details={
            "credential_id": created.credential_id,
            "server_path": created.server_path,
            "credential_type": created.credential_type,
            "credential_binding": created.credential_binding,
        },
    )

    return UpstreamCredentialCreateResponse(
        credential=created,
        secret_payload=payload.secret_payload,
    )


@router.post(
    "/upstream/credentials/{credential_id}/revoke",
    response_model=UpstreamCredentialRecord,
)
async def revoke_upstream_credential(
    credential_id: str,
    payload: UpstreamCredentialRevokeRequest,
    request: Request,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> UpstreamCredentialRecord:
    upstream_store = _require_upstream_credential_store(stores)
    request_id = _get_request_id(request)

    record = upstream_store.get_credential_by_id(credential_id=credential_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Credential not found")

    owned_agent_ids = _owned_agent_ids_for_user(
        stores=stores,
        user_id=context.user_id,
    )
    if not _is_upstream_credential_visible(
        record=record,
        context=context,
        owned_agent_ids=owned_agent_ids,
        include_service=True,
    ):
        raise HTTPException(status_code=404, detail="Credential not found")

    revoked = upstream_store.revoke_credential(
        credential_id=credential_id,
    )
    if revoked is None:
        raise HTTPException(status_code=404, detail="Credential not found")

    _emit_management_audit_event(
        stores=stores,
        action="upstream/credentials/revoke",
        outcome="allow",
        user_id=context.user_id,
        agent_id=context.actor_agent_id,
        request_id=request_id,
        details={
            "credential_id": credential_id,
            "reason": payload.reason,
        },
    )

    return revoked
