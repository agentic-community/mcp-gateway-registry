from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

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
from ..models.agent import (
    AgentRecord,
)
from ..models.revocation import (
    TokenRevocationRecord,
)
from ..models.user import (
    UserRecord,
)
from .management_api_models import (
    CreateAgentRequest,
    CreateApiKeyRequest,
    CreateApiKeyResponse,
)
from .management_common import (
    _emit_management_audit_event,
    _get_request_id,
    _map_management_error,
    _require_admin,
)
from .management_service_factory import (
    _build_management_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["enforceai-management"],
)


class AdminUserSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str
    email: str
    username: Optional[str] = None
    auth_method: str
    role: str
    last_login_at: Optional[datetime] = None
    disabled_at: Optional[datetime] = None
    agent_count: int = 0


class AdminCreateAgentRequest(CreateAgentRequest):
    pass


class AdminRevokeGatewayTokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str
    jti: str
    reason: Optional[str] = None


def _user_summary(
    *,
    user: UserRecord,
    agent_count: int,
) -> AdminUserSummary:
    return AdminUserSummary(
        user_id=user.user_id,
        email=user.email,
        username=user.username,
        auth_method=user.auth_method,
        role=user.role,
        last_login_at=user.last_login_at,
        disabled_at=user.disabled_at,
        agent_count=agent_count,
    )


@router.get("/admin/users", response_model=list[AdminUserSummary])
async def admin_search_users(
    query: str,
    request: Request,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
    limit: int = 50,
) -> list[AdminUserSummary]:
    _require_admin(context)
    request_id = _get_request_id(request)

    resolved_limit = max(1, min(limit, 200))
    try:
        users = stores.user_store.search_users(
            query=query,
            limit=resolved_limit,
        )
        summaries: list[AdminUserSummary] = []
        for user in users:
            agent_count = len(stores.agent_store.list_agents_by_user_id(user_id=user.user_id))
            summaries.append(_user_summary(user=user, agent_count=agent_count))

        _emit_management_audit_event(
            stores=stores,
            action="admin/users/search",
            outcome="allow",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"query": query, "count": len(summaries), "limit": resolved_limit},
        )
        return summaries
    except Exception as exc:
        _emit_management_audit_event(
            stores=stores,
            action="admin/users/search",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"error_type": type(exc).__name__},
        )
        raise _map_management_error(exc) from exc


@router.get("/admin/users/{user_id:path}/agents", response_model=list[AgentRecord])
async def admin_list_user_agents(
    user_id: str,
    request: Request,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> list[AgentRecord]:
    _require_admin(context)
    request_id = _get_request_id(request)

    try:
        agents = stores.agent_store.list_agents_by_user_id(user_id=user_id)
        _emit_management_audit_event(
            stores=stores,
            action="admin/users/agents/list",
            outcome="allow",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"target_user_id": user_id, "count": len(agents)},
        )
        return agents
    except Exception as exc:
        _emit_management_audit_event(
            stores=stores,
            action="admin/users/agents/list",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"target_user_id": user_id, "error_type": type(exc).__name__},
        )
        raise _map_management_error(exc) from exc


@router.post("/admin/users/{user_id:path}/agents", response_model=AgentRecord)
async def admin_create_agent_for_user(
    user_id: str,
    body: AdminCreateAgentRequest,
    request: Request,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    settings: EnforceAISettings = Depends(get_enforceai_settings),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> AgentRecord:
    _require_admin(context)
    service = _build_management_service(
        settings=settings,
        stores=stores,
        context=context,
    )
    request_id = _get_request_id(request)

    try:
        created = service.create_agent(
            user_id=user_id,
            scopes=body.scopes,
            allowed_tools=body.allowed_tools,
            alias=body.alias,
            metadata=body.metadata,
        )
        _emit_management_audit_event(
            stores=stores,
            action="admin/users/agents/create",
            outcome="allow",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"target_user_id": user_id, "created_agent_id": created.agent_id},
        )
        return created
    except Exception as exc:
        _emit_management_audit_event(
            stores=stores,
            action="admin/users/agents/create",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"target_user_id": user_id, "error_type": type(exc).__name__},
        )
        raise _map_management_error(exc) from exc


@router.post(
    "/admin/users/{user_id:path}/agents/{agent_id}/revoke",
    response_model=AgentRecord,
)
async def admin_revoke_agent_for_user(
    user_id: str,
    agent_id: str,
    request: Request,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    settings: EnforceAISettings = Depends(get_enforceai_settings),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> AgentRecord:
    _require_admin(context)
    service = _build_management_service(
        settings=settings,
        stores=stores,
        context=context,
    )
    request_id = _get_request_id(request)

    try:
        existing = stores.agent_store.get_agent_by_id(agent_id=agent_id)
        if existing is None or existing.user_id != user_id:
            raise HTTPException(status_code=404, detail="Agent not found")

        revoked = service.revoke_agent(
            user_id=user_id,
            agent_id=agent_id,
        )
        _emit_management_audit_event(
            stores=stores,
            action="admin/users/agents/revoke",
            outcome="allow",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"target_user_id": user_id, "target_agent_id": agent_id},
        )
        return revoked
    except HTTPException as exc:
        _emit_management_audit_event(
            stores=stores,
            action="admin/users/agents/revoke",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={
                "target_user_id": user_id,
                "target_agent_id": agent_id,
                "error_type": "HTTPException",
            },
        )
        raise exc
    except Exception as exc:
        _emit_management_audit_event(
            stores=stores,
            action="admin/users/agents/revoke",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={
                "target_user_id": user_id,
                "target_agent_id": agent_id,
                "error_type": type(exc).__name__,
            },
        )
        raise _map_management_error(exc) from exc


@router.post(
    "/admin/users/{user_id:path}/api-keys/{key_id}/revoke",
    response_model=ApiKeySummary,
)
async def admin_revoke_api_key_for_user(
    user_id: str,
    key_id: str,
    request: Request,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    settings: EnforceAISettings = Depends(get_enforceai_settings),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> ApiKeySummary:
    _require_admin(context)
    service = _build_management_service(
        settings=settings,
        stores=stores,
        context=context,
    )
    request_id = _get_request_id(request)

    try:
        revoked = service.revoke_api_key(
            user_id=user_id,
            key_id=key_id,
        )
        _emit_management_audit_event(
            stores=stores,
            action="admin/users/api-keys/revoke",
            outcome="allow",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"target_user_id": user_id, "key_id": key_id},
        )
        return revoked
    except Exception as exc:
        _emit_management_audit_event(
            stores=stores,
            action="admin/users/api-keys/revoke",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"target_user_id": user_id, "key_id": key_id, "error_type": type(exc).__name__},
        )
        raise _map_management_error(exc) from exc


@router.post(
    "/admin/users/{user_id:path}/agents/{agent_id}/api-keys",
    response_model=CreateApiKeyResponse,
)
async def admin_create_api_key_for_user(
    user_id: str,
    agent_id: str,
    body: CreateApiKeyRequest,
    request: Request,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    settings: EnforceAISettings = Depends(get_enforceai_settings),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> CreateApiKeyResponse:
    _require_admin(context)
    service = _build_management_service(
        settings=settings,
        stores=stores,
        context=context,
    )
    request_id = _get_request_id(request)

    try:
        key_id, secret, api_key_value = service.create_api_key(
            user_id=user_id,
            agent_id=agent_id,
            scopes=body.scopes,
            expires_at=body.expires_at,
        )
        _emit_management_audit_event(
            stores=stores,
            action="admin/users/api-keys/create",
            outcome="allow",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"target_user_id": user_id, "target_agent_id": agent_id, "key_id": key_id},
        )
        return CreateApiKeyResponse(
            key_id=key_id,
            secret=secret,
            api_key_value=api_key_value,
        )
    except Exception as exc:
        _emit_management_audit_event(
            stores=stores,
            action="admin/users/api-keys/create",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"target_user_id": user_id, "target_agent_id": agent_id, "error_type": type(exc).__name__},
        )
        raise _map_management_error(exc) from exc


@router.post(
    "/admin/users/{user_id:path}/tokens/revoke",
    response_model=TokenRevocationRecord,
)
async def admin_revoke_gateway_token_for_user(
    user_id: str,
    body: AdminRevokeGatewayTokenRequest,
    request: Request,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    settings: EnforceAISettings = Depends(get_enforceai_settings),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> TokenRevocationRecord:
    _require_admin(context)
    service = _build_management_service(
        settings=settings,
        stores=stores,
        context=context,
    )
    request_id = _get_request_id(request)

    try:
        record = service.revoke_token_jti(
            user_id=user_id,
            agent_id=body.agent_id,
            jti=body.jti,
            reason=body.reason,
        )
        _emit_management_audit_event(
            stores=stores,
            action="admin/users/tokens/revoke",
            outcome="allow",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={
                "target_user_id": user_id,
                "target_agent_id": body.agent_id,
                "jti": body.jti,
            },
        )
        return record
    except Exception as exc:
        _emit_management_audit_event(
            stores=stores,
            action="admin/users/tokens/revoke",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"target_user_id": user_id, "error_type": type(exc).__name__},
        )
        raise _map_management_error(exc) from exc


@router.get("/admin/users/{user_id:path}", response_model=AdminUserSummary)
async def admin_get_user(
    user_id: str,
    request: Request,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> AdminUserSummary:
    _require_admin(context)
    request_id = _get_request_id(request)

    try:
        user = stores.user_store.get_user_by_id(user_id=user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")

        agent_count = len(stores.agent_store.list_agents_by_user_id(user_id=user.user_id))
        summary = _user_summary(user=user, agent_count=agent_count)
        _emit_management_audit_event(
            stores=stores,
            action="admin/users/get",
            outcome="allow",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"target_user_id": user_id},
        )
        return summary
    except HTTPException as exc:
        _emit_management_audit_event(
            stores=stores,
            action="admin/users/get",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"target_user_id": user_id, "error_type": "HTTPException"},
        )
        raise exc
    except Exception as exc:
        _emit_management_audit_event(
            stores=stores,
            action="admin/users/get",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"target_user_id": user_id, "error_type": type(exc).__name__},
        )
        raise _map_management_error(exc) from exc

