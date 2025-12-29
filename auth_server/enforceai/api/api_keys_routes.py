from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    Request,
)

from ..auth.dependency import (
    EnforceAIManagementContext,
    get_enforceai_management_context,
    get_enforceai_settings,
    get_enforceai_stores,
)
from ..config import (
    EnforceAISettings,
)
from ..db.data_layer import (
    EnforceAIStores,
)
from ..management.models import (
    ApiKeySummary,
)
from .management_api_models import (
    CreateApiKeyRequest,
    CreateApiKeyResponse,
)
from .management_common import (
    _emit_management_audit_event,
    _get_request_id,
    _map_management_error,
)
from .management_service_factory import (
    _build_management_service,
)

router = APIRouter()


@router.post("/agents/{agent_id}/api-keys", response_model=CreateApiKeyResponse)
async def create_api_key(
    agent_id: str,
    body: CreateApiKeyRequest,
    request: Request,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    settings: EnforceAISettings = Depends(get_enforceai_settings),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> CreateApiKeyResponse:
    service = _build_management_service(
        settings=settings,
        stores=stores,
        context=context,
    )
    request_id = _get_request_id(request)

    try:
        key_id, secret, api_key_value = service.create_api_key(
            user_id=context.user_id,
            agent_id=agent_id,
            scopes=body.scopes,
            expires_at=body.expires_at,
        )
        _emit_management_audit_event(
            stores=stores,
            action="management/api-keys/create",
            outcome="allow",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={
                "target_agent_id": agent_id,
                "key_id": key_id,
                "scopes": body.scopes,
                "expires_at": body.expires_at,
            },
        )
        return CreateApiKeyResponse(
            key_id=key_id,
            secret=secret,
            api_key_value=api_key_value,
        )
    except Exception as exc:
        _emit_management_audit_event(
            stores=stores,
            action="management/api-keys/create",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"target_agent_id": agent_id, "error_type": type(exc).__name__},
        )
        raise _map_management_error(exc) from exc


@router.get("/agents/{agent_id}/api-keys", response_model=list[ApiKeySummary])
async def list_api_keys(
    agent_id: str,
    request: Request,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    settings: EnforceAISettings = Depends(get_enforceai_settings),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> list[ApiKeySummary]:
    service = _build_management_service(
        settings=settings,
        stores=stores,
        context=context,
    )
    request_id = _get_request_id(request)

    try:
        keys = service.list_api_keys(
            user_id=context.user_id,
            agent_id=agent_id,
        )
        _emit_management_audit_event(
            stores=stores,
            action="management/api-keys/list",
            outcome="allow",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"target_agent_id": agent_id, "count": len(keys)},
        )
        return keys
    except Exception as exc:
        _emit_management_audit_event(
            stores=stores,
            action="management/api-keys/list",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"target_agent_id": agent_id, "error_type": type(exc).__name__},
        )
        raise _map_management_error(exc) from exc


@router.post("/api-keys/{key_id}/revoke", response_model=ApiKeySummary)
async def revoke_api_key(
    key_id: str,
    request: Request,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    settings: EnforceAISettings = Depends(get_enforceai_settings),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> ApiKeySummary:
    service = _build_management_service(
        settings=settings,
        stores=stores,
        context=context,
    )
    request_id = _get_request_id(request)

    try:
        revoked = service.revoke_api_key(
            user_id=context.user_id,
            key_id=key_id,
        )
        _emit_management_audit_event(
            stores=stores,
            action="management/api-keys/revoke",
            outcome="allow",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"key_id": key_id},
        )
        return revoked
    except Exception as exc:
        _emit_management_audit_event(
            stores=stores,
            action="management/api-keys/revoke",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"key_id": key_id, "error_type": type(exc).__name__},
        )
        raise _map_management_error(exc) from exc

