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
from ..models.agent import (
    AgentRecord,
)
from .management_api_models import (
    CreateAgentRequest,
    UpdateAgentRequest,
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


@router.get("/agents", response_model=list[AgentRecord])
async def list_agents(
    request: Request,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    settings: EnforceAISettings = Depends(get_enforceai_settings),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> list[AgentRecord]:
    service = _build_management_service(
        settings=settings,
        stores=stores,
        context=context,
    )

    request_id = _get_request_id(request)
    try:
        agents = service.list_agents(user_id=context.user_id)
        _emit_management_audit_event(
            stores=stores,
            action="management/agents/list",
            outcome="allow",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"count": len(agents)},
        )
        return agents
    except Exception as exc:
        _emit_management_audit_event(
            stores=stores,
            action="management/agents/list",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"error_type": type(exc).__name__},
        )
        raise _map_management_error(exc) from exc


@router.post("/agents", response_model=AgentRecord)
async def create_agent(
    body: CreateAgentRequest,
    request: Request,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    settings: EnforceAISettings = Depends(get_enforceai_settings),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> AgentRecord:
    service = _build_management_service(
        settings=settings,
        stores=stores,
        context=context,
    )
    request_id = _get_request_id(request)

    try:
        agent = service.create_agent(
            user_id=context.user_id,
            scopes=body.scopes,
            allowed_tools=body.allowed_tools,
            alias=body.alias,
            metadata=body.metadata,
        )
        _emit_management_audit_event(
            stores=stores,
            action="management/agents/create",
            outcome="allow",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"created_agent_id": agent.agent_id},
        )
        return agent
    except Exception as exc:
        _emit_management_audit_event(
            stores=stores,
            action="management/agents/create",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"error_type": type(exc).__name__},
        )
        raise _map_management_error(exc) from exc


@router.get("/agents/{agent_id}", response_model=AgentRecord)
async def get_agent(
    agent_id: str,
    request: Request,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    settings: EnforceAISettings = Depends(get_enforceai_settings),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> AgentRecord:
    service = _build_management_service(
        settings=settings,
        stores=stores,
        context=context,
    )
    request_id = _get_request_id(request)

    try:
        agent = service.get_agent(
            user_id=context.user_id,
            agent_id=agent_id,
        )
        _emit_management_audit_event(
            stores=stores,
            action="management/agents/get",
            outcome="allow",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"target_agent_id": agent_id},
        )
        return agent
    except Exception as exc:
        _emit_management_audit_event(
            stores=stores,
            action="management/agents/get",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"target_agent_id": agent_id, "error_type": type(exc).__name__},
        )
        raise _map_management_error(exc) from exc


@router.patch("/agents/{agent_id}", response_model=AgentRecord)
async def update_agent(
    agent_id: str,
    body: UpdateAgentRequest,
    request: Request,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    settings: EnforceAISettings = Depends(get_enforceai_settings),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> AgentRecord:
    service = _build_management_service(
        settings=settings,
        stores=stores,
        context=context,
    )
    request_id = _get_request_id(request)

    try:
        updated = service.update_agent(
            user_id=context.user_id,
            agent_id=agent_id,
            scopes=body.scopes,
            allowed_tools=body.allowed_tools,
            alias=body.alias,
            metadata=body.metadata,
        )
        _emit_management_audit_event(
            stores=stores,
            action="management/agents/update",
            outcome="allow",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"target_agent_id": agent_id},
        )
        return updated
    except Exception as exc:
        _emit_management_audit_event(
            stores=stores,
            action="management/agents/update",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"target_agent_id": agent_id, "error_type": type(exc).__name__},
        )
        raise _map_management_error(exc) from exc


@router.post("/agents/{agent_id}/revoke", response_model=AgentRecord)
async def revoke_agent(
    agent_id: str,
    request: Request,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    settings: EnforceAISettings = Depends(get_enforceai_settings),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> AgentRecord:
    service = _build_management_service(
        settings=settings,
        stores=stores,
        context=context,
    )
    request_id = _get_request_id(request)

    try:
        revoked = service.revoke_agent(
            user_id=context.user_id,
            agent_id=agent_id,
        )
        _emit_management_audit_event(
            stores=stores,
            action="management/agents/revoke",
            outcome="allow",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"target_agent_id": agent_id},
        )
        return revoked
    except Exception as exc:
        _emit_management_audit_event(
            stores=stores,
            action="management/agents/revoke",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"target_agent_id": agent_id, "error_type": type(exc).__name__},
        )
        raise _map_management_error(exc) from exc


@router.post("/agents/{agent_id}/tokens/revoke-all", response_model=AgentRecord)
async def revoke_all_tokens(
    agent_id: str,
    request: Request,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    settings: EnforceAISettings = Depends(get_enforceai_settings),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> AgentRecord:
    service = _build_management_service(
        settings=settings,
        stores=stores,
        context=context,
    )
    request_id = _get_request_id(request)

    try:
        updated = service.revoke_all_tokens(
            user_id=context.user_id,
            agent_id=agent_id,
        )
        _emit_management_audit_event(
            stores=stores,
            action="management/tokens/revoke-all",
            outcome="allow",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"target_agent_id": agent_id},
        )
        return updated
    except Exception as exc:
        _emit_management_audit_event(
            stores=stores,
            action="management/tokens/revoke-all",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"target_agent_id": agent_id, "error_type": type(exc).__name__},
        )
        raise _map_management_error(exc) from exc

