from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Request,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
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
from ..errors import (
    DependencyUnavailableError,
)
from .management_common import (
    _emit_management_audit_event,
    _get_request_id,
    _require_admin,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["enforceai-management"],
)


def _require_if_match(
    if_match: Optional[str],
) -> str:
    if if_match is None or not if_match.strip():
        raise HTTPException(
            status_code=428,
            detail="Missing If-Match header",
        )
    return if_match.strip()


class ScopeMutationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool
    scope_name: str
    etag: str
    last_modified: Optional[str] = None


class MethodPolicyUpsert(BaseModel):
    model_config = ConfigDict(extra="forbid")

    all_methods: bool
    methods: list[str]


class ToolPolicyUpsert(BaseModel):
    model_config = ConfigDict(extra="forbid")

    all_tools: bool
    tools: list[str]


class ServerPermissionUpsert(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server: str
    methods: MethodPolicyUpsert
    tools: Optional[ToolPolicyUpsert] = None


class AgentPermissionUpsert(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str
    resources: list[str]


class CreateScopeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    server_permissions: list[ServerPermissionUpsert] = Field(default_factory=list)
    agent_permissions: list[AgentPermissionUpsert] = Field(default_factory=list)


class ReplaceScopeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    server_permissions: list[ServerPermissionUpsert] = Field(default_factory=list)
    agent_permissions: list[AgentPermissionUpsert] = Field(default_factory=list)


def _scope_entries_from_request(
    *,
    server_permissions: list[ServerPermissionUpsert],
    agent_permissions: list[AgentPermissionUpsert],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    for server_permission in server_permissions:
        methods: list[str]
        if server_permission.methods.all_methods:
            methods = ["*"]
        else:
            methods = list(server_permission.methods.methods)

        entry: dict[str, Any] = {
            "server": server_permission.server,
            "methods": methods,
        }

        if server_permission.tools is not None:
            if server_permission.tools.all_tools:
                entry["tools"] = "*"
            else:
                entry["tools"] = list(server_permission.tools.tools)

        entries.append(entry)

    if agent_permissions:
        actions: list[dict[str, Any]] = []
        for permission in agent_permissions:
            actions.append(
                {
                    "action": permission.action,
                    "resources": list(permission.resources),
                }
            )

        entries.append(
            {
                "agents": {
                    "actions": actions,
                }
            }
        )

    return entries


@router.post(
    "/admin/scopes",
    response_model=ScopeMutationResponse,
)
async def admin_create_scope(
    request: Request,
    payload: CreateScopeRequest,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
    settings: EnforceAISettings = Depends(get_enforceai_settings),
    if_match: Optional[str] = Header(default=None, alias="If-Match"),
) -> ScopeMutationResponse:
    from ..fgac.policy_writer import (
        PolicyConflictError,
        PolicyPreconditionFailedError,
        write_scope_catalog_scope,
    )

    _require_admin(context)
    request_id = _get_request_id(request)

    scope_name = payload.name
    entries = _scope_entries_from_request(
        server_permissions=payload.server_permissions,
        agent_permissions=payload.agent_permissions,
    )

    try:
        result = write_scope_catalog_scope(
            path=settings.scopes_catalog_path,
            scope_name=scope_name,
            entries=entries,
            mode="create",
            if_match=if_match,
        )
        _emit_management_audit_event(
            stores=stores,
            action="admin/scopes/create",
            outcome="allow",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={
                "scope_name": scope_name,
                "server_permissions": len(payload.server_permissions),
                "agent_permissions": len(payload.agent_permissions),
            },
        )
        return ScopeMutationResponse(
            ok=True,
            scope_name=scope_name,
            etag=result.etag,
            last_modified=result.last_modified,
        )
    except PolicyPreconditionFailedError as exc:
        _emit_management_audit_event(
            stores=stores,
            action="admin/scopes/create",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"scope_name": scope_name, "reason": "etag_mismatch"},
        )
        raise HTTPException(status_code=412, detail=str(exc)) from exc
    except PolicyConflictError as exc:
        _emit_management_audit_event(
            stores=stores,
            action="admin/scopes/create",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"scope_name": scope_name, "reason": "conflict"},
        )
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        _emit_management_audit_event(
            stores=stores,
            action="admin/scopes/create",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"scope_name": scope_name, "reason": "invalid"},
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DependencyUnavailableError as exc:
        _emit_management_audit_event(
            stores=stores,
            action="admin/scopes/create",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"scope_name": scope_name, "reason": "unavailable"},
        )
        raise HTTPException(status_code=503, detail=exc.public_message) from exc


@router.put(
    "/admin/scopes/{scope_name}",
    response_model=ScopeMutationResponse,
)
async def admin_replace_scope(
    scope_name: str,
    request: Request,
    payload: ReplaceScopeRequest,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
    settings: EnforceAISettings = Depends(get_enforceai_settings),
    if_match: Optional[str] = Header(default=None, alias="If-Match"),
) -> ScopeMutationResponse:
    from ..fgac.policy_writer import (
        PolicyConflictError,
        PolicyNotFoundError,
        PolicyPreconditionFailedError,
        write_scope_catalog_scope,
    )

    _require_admin(context)
    request_id = _get_request_id(request)

    if payload.name is not None and payload.name != scope_name:
        raise HTTPException(status_code=400, detail="Scope name mismatch")

    required_if_match = _require_if_match(if_match)
    entries = _scope_entries_from_request(
        server_permissions=payload.server_permissions,
        agent_permissions=payload.agent_permissions,
    )

    try:
        result = write_scope_catalog_scope(
            path=settings.scopes_catalog_path,
            scope_name=scope_name,
            entries=entries,
            mode="replace",
            if_match=required_if_match,
        )
        _emit_management_audit_event(
            stores=stores,
            action="admin/scopes/replace",
            outcome="allow",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={
                "scope_name": scope_name,
                "server_permissions": len(payload.server_permissions),
                "agent_permissions": len(payload.agent_permissions),
            },
        )
        return ScopeMutationResponse(
            ok=True,
            scope_name=scope_name,
            etag=result.etag,
            last_modified=result.last_modified,
        )
    except PolicyPreconditionFailedError as exc:
        _emit_management_audit_event(
            stores=stores,
            action="admin/scopes/replace",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"scope_name": scope_name, "reason": "etag_mismatch"},
        )
        raise HTTPException(status_code=412, detail=str(exc)) from exc
    except PolicyNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PolicyConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        _emit_management_audit_event(
            stores=stores,
            action="admin/scopes/replace",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"scope_name": scope_name, "reason": "invalid"},
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DependencyUnavailableError as exc:
        _emit_management_audit_event(
            stores=stores,
            action="admin/scopes/replace",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"scope_name": scope_name, "reason": "unavailable"},
        )
        raise HTTPException(status_code=503, detail=exc.public_message) from exc


@router.delete(
    "/admin/scopes/{scope_name}",
    response_model=ScopeMutationResponse,
)
async def admin_delete_scope(
    scope_name: str,
    request: Request,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
    settings: EnforceAISettings = Depends(get_enforceai_settings),
    if_match: Optional[str] = Header(default=None, alias="If-Match"),
) -> ScopeMutationResponse:
    from ..fgac.policy_writer import (
        PolicyConflictError,
        PolicyNotFoundError,
        PolicyPreconditionFailedError,
        write_scope_catalog_scope,
    )

    _require_admin(context)
    request_id = _get_request_id(request)
    required_if_match = _require_if_match(if_match)

    try:
        result = write_scope_catalog_scope(
            path=settings.scopes_catalog_path,
            scope_name=scope_name,
            entries=[],
            mode="delete",
            if_match=required_if_match,
        )
        _emit_management_audit_event(
            stores=stores,
            action="admin/scopes/delete",
            outcome="allow",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"scope_name": scope_name},
        )
        return ScopeMutationResponse(
            ok=True,
            scope_name=scope_name,
            etag=result.etag,
            last_modified=result.last_modified,
        )
    except PolicyPreconditionFailedError as exc:
        _emit_management_audit_event(
            stores=stores,
            action="admin/scopes/delete",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"scope_name": scope_name, "reason": "etag_mismatch"},
        )
        raise HTTPException(status_code=412, detail=str(exc)) from exc
    except PolicyNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PolicyConflictError as exc:
        _emit_management_audit_event(
            stores=stores,
            action="admin/scopes/delete",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"scope_name": scope_name, "reason": "conflict"},
        )
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        _emit_management_audit_event(
            stores=stores,
            action="admin/scopes/delete",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"scope_name": scope_name, "reason": "invalid"},
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DependencyUnavailableError as exc:
        _emit_management_audit_event(
            stores=stores,
            action="admin/scopes/delete",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"scope_name": scope_name, "reason": "unavailable"},
        )
        raise HTTPException(status_code=503, detail=exc.public_message) from exc

