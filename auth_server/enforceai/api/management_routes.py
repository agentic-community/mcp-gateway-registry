from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from ..auth.dependency import (
    EnforceAIRequestContext,
    get_enforceai_request_context,
    get_enforceai_settings,
    get_enforceai_stores,
)
from ..config import (
    EnforceAISettings,
)
from ..crypto.keyring import (
    GatewayKeyring,
    load_gateway_keyring_cached,
)
from ..db.data_layer import (
    EnforceAIStores,
)
from ..errors import (
    DependencyUnavailableError,
    EnforceAIError,
)
from ..secrets.pepper import (
    load_api_key_pepper,
)
from ..tokens.verify import (
    verify_gateway_token,
)
from ..management.models import (
    ApiKeySummary,
)
from ..management.service import (
    ManagementService,
)
from ..models.agent import (
    AgentRecord,
)
from ..models.revocation import (
    TokenRevocationRecord,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/enforceai",
    tags=["enforceai-management"],
)


class CreateAgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scopes: list[str]
    allowed_tools: Optional[list[str]] = None
    alias: Optional[str] = None
    metadata: Optional[dict[str, object]] = None


class UpdateAgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scopes: Optional[list[str]] = None
    allowed_tools: Optional[list[str]] = None
    alias: Optional[str] = None
    metadata: Optional[dict[str, object]] = None


class CreateApiKeyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scopes: Optional[list[str]] = None
    expires_at: Optional[datetime] = None


class CreateApiKeyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key_id: str
    secret: str
    api_key_value: str


class MintTokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scopes: list[str]
    ttl_seconds: Optional[int] = Field(default=None, ge=1)
    expires_at: Optional[datetime] = None


class MintTokenResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    token: str


class RevokeTokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: Optional[str] = None
    jti: Optional[str] = None
    gateway_token: Optional[str] = None
    reason: Optional[str] = None

    @model_validator(mode="after")
    def _validate(self) -> "RevokeTokenRequest":
        if self.gateway_token is not None:
            if self.agent_id is not None or self.jti is not None:
                raise ValueError("Provide either gateway_token or (agent_id and jti), not both")
            if not self.gateway_token.strip():
                raise ValueError("gateway_token must be a non-empty string")
            return self

        if self.agent_id is None or self.jti is None:
            raise ValueError("Provide either gateway_token or both agent_id and jti")
        if not self.agent_id.strip():
            raise ValueError("agent_id must be a non-empty string")
        if not self.jti.strip():
            raise ValueError("jti must be a non-empty string")
        return self


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _emit_management_audit_event(
    *,
    stores: EnforceAIStores,
    action: str,
    outcome: str,
    user_id: str,
    agent_id: str,
    request_id: Optional[str],
    details: dict[str, Any],
) -> None:
    payload = {
        "event_type": "enforceai_audit",
        "action": action,
        "outcome": outcome,
        "user_id": user_id,
        "agent_id": agent_id,
        "request_id": request_id,
        "details": details,
    }

    try:
        print(
            json.dumps(
                payload,
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            ),
            flush=True,
        )
    except Exception:  # noqa: BLE001 - best-effort
        logger.exception("Failed to emit EnforceAI audit event to stdout")

    try:
        stores.audit_store.append_event(
            occurred_at=_utc_now(),
            user_id=user_id,
            agent_id=agent_id,
            action=action,
            outcome=outcome,
            request_id=request_id,
            details=details,
        )
    except Exception:  # noqa: BLE001 - best-effort
        logger.exception("Failed to persist EnforceAI audit event")


def _map_management_error(
    exc: Exception,
) -> HTTPException:
    if isinstance(exc, EnforceAIError):
        return exc.as_http_exception()
    if isinstance(exc, ValueError):
        return HTTPException(
            status_code=400,
            detail=str(exc),
        )
    return HTTPException(
        status_code=503,
        detail="Enforcement dependency unavailable",
    )


def _load_gateway_keyring(
    *,
    settings: EnforceAISettings,
) -> Optional[GatewayKeyring]:
    if (
        settings.gateway_private_key_path is None
        or settings.gateway_public_keys_dir is None
        or settings.gateway_active_kid is None
    ):
        return None

    try:
        return load_gateway_keyring_cached(
            private_key_path=settings.gateway_private_key_path,
            public_keys_dir=settings.gateway_public_keys_dir,
            active_kid=settings.gateway_active_kid,
        )
    except Exception as exc:  # noqa: BLE001 - map to 503
        raise DependencyUnavailableError(
            "Gateway keyring unavailable",
            public_message="Enforcement misconfigured",
        ) from exc


def _build_management_service(
    *,
    settings: EnforceAISettings,
    stores: EnforceAIStores,
    context: EnforceAIRequestContext,
) -> ManagementService:
    pepper: Optional[bytes] = None
    if settings.api_key_pepper_path is not None:
        try:
            pepper = load_api_key_pepper(settings.api_key_pepper_path)
        except ValueError as exc:
            raise DependencyUnavailableError(
                "API key pepper unavailable",
                public_message="Enforcement misconfigured",
            ) from exc

    keyring = _load_gateway_keyring(settings=settings)

    return ManagementService(
        agent_store=stores.agent_store,
        api_key_store=stores.api_key_store,
        revocation_store=stores.revocation_store,
        scope_catalog=context.catalog,
        api_key_pepper=pepper,
        gateway_keyring=keyring,
        gateway_issuer=settings.gateway_issuer,
    )


def _get_request_id(
    request: Request,
) -> Optional[str]:
    value = request.headers.get("X-Request-Id")
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


@router.get("/agents", response_model=list[AgentRecord])
async def list_agents(
    request: Request,
    context: EnforceAIRequestContext = Depends(get_enforceai_request_context),
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
        agents = service.list_agents(user_id=context.identity.user_id)
        _emit_management_audit_event(
            stores=stores,
            action="management/agents/list",
            outcome="allow",
            user_id=context.identity.user_id,
            agent_id=context.identity.agent_id,
            request_id=request_id,
            details={"count": len(agents)},
        )
        return agents
    except Exception as exc:
        _emit_management_audit_event(
            stores=stores,
            action="management/agents/list",
            outcome="deny",
            user_id=context.identity.user_id,
            agent_id=context.identity.agent_id,
            request_id=request_id,
            details={"error_type": type(exc).__name__},
        )
        raise _map_management_error(exc) from exc


@router.post("/agents", response_model=AgentRecord)
async def create_agent(
    body: CreateAgentRequest,
    request: Request,
    context: EnforceAIRequestContext = Depends(get_enforceai_request_context),
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
            user_id=context.identity.user_id,
            scopes=body.scopes,
            allowed_tools=body.allowed_tools,
            alias=body.alias,
            metadata=body.metadata,
        )
        _emit_management_audit_event(
            stores=stores,
            action="management/agents/create",
            outcome="allow",
            user_id=context.identity.user_id,
            agent_id=context.identity.agent_id,
            request_id=request_id,
            details={"created_agent_id": agent.agent_id},
        )
        return agent
    except Exception as exc:
        _emit_management_audit_event(
            stores=stores,
            action="management/agents/create",
            outcome="deny",
            user_id=context.identity.user_id,
            agent_id=context.identity.agent_id,
            request_id=request_id,
            details={"error_type": type(exc).__name__},
        )
        raise _map_management_error(exc) from exc


@router.get("/agents/{agent_id}", response_model=AgentRecord)
async def get_agent(
    agent_id: str,
    request: Request,
    context: EnforceAIRequestContext = Depends(get_enforceai_request_context),
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
            user_id=context.identity.user_id,
            agent_id=agent_id,
        )
        _emit_management_audit_event(
            stores=stores,
            action="management/agents/get",
            outcome="allow",
            user_id=context.identity.user_id,
            agent_id=context.identity.agent_id,
            request_id=request_id,
            details={"target_agent_id": agent_id},
        )
        return agent
    except Exception as exc:
        _emit_management_audit_event(
            stores=stores,
            action="management/agents/get",
            outcome="deny",
            user_id=context.identity.user_id,
            agent_id=context.identity.agent_id,
            request_id=request_id,
            details={"target_agent_id": agent_id, "error_type": type(exc).__name__},
        )
        raise _map_management_error(exc) from exc


@router.patch("/agents/{agent_id}", response_model=AgentRecord)
async def update_agent(
    agent_id: str,
    body: UpdateAgentRequest,
    request: Request,
    context: EnforceAIRequestContext = Depends(get_enforceai_request_context),
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
            user_id=context.identity.user_id,
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
            user_id=context.identity.user_id,
            agent_id=context.identity.agent_id,
            request_id=request_id,
            details={"target_agent_id": agent_id},
        )
        return updated
    except Exception as exc:
        _emit_management_audit_event(
            stores=stores,
            action="management/agents/update",
            outcome="deny",
            user_id=context.identity.user_id,
            agent_id=context.identity.agent_id,
            request_id=request_id,
            details={"target_agent_id": agent_id, "error_type": type(exc).__name__},
        )
        raise _map_management_error(exc) from exc


@router.post("/agents/{agent_id}/revoke", response_model=AgentRecord)
async def revoke_agent(
    agent_id: str,
    request: Request,
    context: EnforceAIRequestContext = Depends(get_enforceai_request_context),
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
            user_id=context.identity.user_id,
            agent_id=agent_id,
        )
        _emit_management_audit_event(
            stores=stores,
            action="management/agents/revoke",
            outcome="allow",
            user_id=context.identity.user_id,
            agent_id=context.identity.agent_id,
            request_id=request_id,
            details={"target_agent_id": agent_id},
        )
        return revoked
    except Exception as exc:
        _emit_management_audit_event(
            stores=stores,
            action="management/agents/revoke",
            outcome="deny",
            user_id=context.identity.user_id,
            agent_id=context.identity.agent_id,
            request_id=request_id,
            details={"target_agent_id": agent_id, "error_type": type(exc).__name__},
        )
        raise _map_management_error(exc) from exc


@router.post("/agents/{agent_id}/tokens/revoke-all", response_model=AgentRecord)
async def revoke_all_tokens(
    agent_id: str,
    request: Request,
    context: EnforceAIRequestContext = Depends(get_enforceai_request_context),
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
            user_id=context.identity.user_id,
            agent_id=agent_id,
        )
        _emit_management_audit_event(
            stores=stores,
            action="management/tokens/revoke-all",
            outcome="allow",
            user_id=context.identity.user_id,
            agent_id=context.identity.agent_id,
            request_id=request_id,
            details={"target_agent_id": agent_id},
        )
        return updated
    except Exception as exc:
        _emit_management_audit_event(
            stores=stores,
            action="management/tokens/revoke-all",
            outcome="deny",
            user_id=context.identity.user_id,
            agent_id=context.identity.agent_id,
            request_id=request_id,
            details={"target_agent_id": agent_id, "error_type": type(exc).__name__},
        )
        raise _map_management_error(exc) from exc


@router.post("/agents/{agent_id}/api-keys", response_model=CreateApiKeyResponse)
async def create_api_key(
    agent_id: str,
    body: CreateApiKeyRequest,
    request: Request,
    context: EnforceAIRequestContext = Depends(get_enforceai_request_context),
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
            user_id=context.identity.user_id,
            agent_id=agent_id,
            scopes=body.scopes,
            expires_at=body.expires_at,
        )
        _emit_management_audit_event(
            stores=stores,
            action="management/api-keys/create",
            outcome="allow",
            user_id=context.identity.user_id,
            agent_id=context.identity.agent_id,
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
            user_id=context.identity.user_id,
            agent_id=context.identity.agent_id,
            request_id=request_id,
            details={"target_agent_id": agent_id, "error_type": type(exc).__name__},
        )
        raise _map_management_error(exc) from exc


@router.get("/agents/{agent_id}/api-keys", response_model=list[ApiKeySummary])
async def list_api_keys(
    agent_id: str,
    request: Request,
    context: EnforceAIRequestContext = Depends(get_enforceai_request_context),
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
            user_id=context.identity.user_id,
            agent_id=agent_id,
        )
        _emit_management_audit_event(
            stores=stores,
            action="management/api-keys/list",
            outcome="allow",
            user_id=context.identity.user_id,
            agent_id=context.identity.agent_id,
            request_id=request_id,
            details={"target_agent_id": agent_id, "count": len(keys)},
        )
        return keys
    except Exception as exc:
        _emit_management_audit_event(
            stores=stores,
            action="management/api-keys/list",
            outcome="deny",
            user_id=context.identity.user_id,
            agent_id=context.identity.agent_id,
            request_id=request_id,
            details={"target_agent_id": agent_id, "error_type": type(exc).__name__},
        )
        raise _map_management_error(exc) from exc


@router.post("/api-keys/{key_id}/revoke", response_model=ApiKeySummary)
async def revoke_api_key(
    key_id: str,
    request: Request,
    context: EnforceAIRequestContext = Depends(get_enforceai_request_context),
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
            user_id=context.identity.user_id,
            key_id=key_id,
        )
        _emit_management_audit_event(
            stores=stores,
            action="management/api-keys/revoke",
            outcome="allow",
            user_id=context.identity.user_id,
            agent_id=context.identity.agent_id,
            request_id=request_id,
            details={"key_id": key_id},
        )
        return revoked
    except Exception as exc:
        _emit_management_audit_event(
            stores=stores,
            action="management/api-keys/revoke",
            outcome="deny",
            user_id=context.identity.user_id,
            agent_id=context.identity.agent_id,
            request_id=request_id,
            details={"key_id": key_id, "error_type": type(exc).__name__},
        )
        raise _map_management_error(exc) from exc


@router.post("/agents/{agent_id}/tokens/mint", response_model=MintTokenResponse)
async def mint_token(
    agent_id: str,
    body: MintTokenRequest,
    request: Request,
    context: EnforceAIRequestContext = Depends(get_enforceai_request_context),
    settings: EnforceAISettings = Depends(get_enforceai_settings),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> MintTokenResponse:
    service = _build_management_service(
        settings=settings,
        stores=stores,
        context=context,
    )
    request_id = _get_request_id(request)

    try:
        token = service.mint_gateway_token(
            user_id=context.identity.user_id,
            agent_id=agent_id,
            scopes=body.scopes,
            ttl_seconds=body.ttl_seconds,
            expires_at=body.expires_at,
        )
        _emit_management_audit_event(
            stores=stores,
            action="management/tokens/mint",
            outcome="allow",
            user_id=context.identity.user_id,
            agent_id=context.identity.agent_id,
            request_id=request_id,
            details={
                "target_agent_id": agent_id,
                "scopes": body.scopes,
                "ttl_seconds": body.ttl_seconds,
                "expires_at": body.expires_at,
            },
        )
        return MintTokenResponse(token=token)
    except Exception as exc:
        _emit_management_audit_event(
            stores=stores,
            action="management/tokens/mint",
            outcome="deny",
            user_id=context.identity.user_id,
            agent_id=context.identity.agent_id,
            request_id=request_id,
            details={"target_agent_id": agent_id, "error_type": type(exc).__name__},
        )
        raise _map_management_error(exc) from exc


@router.post("/tokens/revoke", response_model=TokenRevocationRecord)
async def revoke_token(
    body: RevokeTokenRequest,
    request: Request,
    context: EnforceAIRequestContext = Depends(get_enforceai_request_context),
    settings: EnforceAISettings = Depends(get_enforceai_settings),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> TokenRevocationRecord:
    service = _build_management_service(
        settings=settings,
        stores=stores,
        context=context,
    )
    request_id = _get_request_id(request)

    try:
        if body.gateway_token is not None:
            keyring = _load_gateway_keyring(settings=settings)
            if keyring is None:
                raise DependencyUnavailableError(
                    "Gateway keyring unavailable",
                    public_message="Enforcement misconfigured",
                )
            if settings.gateway_issuer is None:
                raise DependencyUnavailableError(
                    "Gateway issuer missing",
                    public_message="Enforcement misconfigured",
                )

            claims = verify_gateway_token(
                body.gateway_token,
                keyring=keyring,
                expected_issuer=settings.gateway_issuer,
            )
            if claims.sub != context.identity.user_id:
                raise HTTPException(status_code=403, detail="Forbidden")

            record = service.revoke_token_jti(
                user_id=context.identity.user_id,
                agent_id=claims.agent_id,
                jti=claims.jti,
                expires_at=claims.expires_at,
                reason=body.reason,
            )
            _emit_management_audit_event(
                stores=stores,
                action="management/tokens/revoke",
                outcome="allow",
                user_id=context.identity.user_id,
                agent_id=context.identity.agent_id,
                request_id=request_id,
                details={"target_agent_id": claims.agent_id},
            )
            return record

        record = service.revoke_token_jti(
            user_id=context.identity.user_id,
            agent_id=body.agent_id or "",
            jti=body.jti or "",
            reason=body.reason,
        )
        _emit_management_audit_event(
            stores=stores,
            action="management/tokens/revoke",
            outcome="allow",
            user_id=context.identity.user_id,
            agent_id=context.identity.agent_id,
            request_id=request_id,
            details={"target_agent_id": body.agent_id},
        )
        return record
    except HTTPException as exc:
        _emit_management_audit_event(
            stores=stores,
            action="management/tokens/revoke",
            outcome="deny",
            user_id=context.identity.user_id,
            agent_id=context.identity.agent_id,
            request_id=request_id,
            details={"error_type": "HTTPException"},
        )
        raise exc
    except Exception as exc:
        _emit_management_audit_event(
            stores=stores,
            action="management/tokens/revoke",
            outcome="deny",
            user_id=context.identity.user_id,
            agent_id=context.identity.agent_id,
            request_id=request_id,
            details={"error_type": type(exc).__name__},
        )
        raise _map_management_error(exc) from exc
