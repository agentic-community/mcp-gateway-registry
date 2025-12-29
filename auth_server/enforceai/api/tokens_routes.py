from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
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
from ..errors import (
    DependencyUnavailableError,
)
from ..models.revocation import (
    TokenRevocationRecord,
)
from ..tokens.verify import (
    verify_gateway_token,
)
from .management_api_models import (
    MintTokenRequest,
    MintTokenResponse,
    RevokeTokenRequest,
)
from .management_common import (
    _emit_management_audit_event,
    _get_request_id,
    _map_management_error,
)
from .management_service_factory import (
    _build_management_service,
    _load_gateway_keyring,
)

router = APIRouter()


@router.post("/agents/{agent_id}/tokens/mint", response_model=MintTokenResponse)
async def mint_token(
    agent_id: str,
    body: MintTokenRequest,
    request: Request,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
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
            user_id=context.user_id,
            agent_id=agent_id,
            scopes=body.scopes,
            ttl_seconds=body.ttl_seconds,
            expires_at=body.expires_at,
        )
        _emit_management_audit_event(
            stores=stores,
            action="management/tokens/mint",
            outcome="allow",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
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
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"target_agent_id": agent_id, "error_type": type(exc).__name__},
        )
        raise _map_management_error(exc) from exc


@router.post("/tokens/revoke", response_model=TokenRevocationRecord)
async def revoke_token(
    body: RevokeTokenRequest,
    request: Request,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
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
            if claims.sub != context.user_id:
                raise HTTPException(status_code=403, detail="Forbidden")

            record = service.revoke_token_jti(
                user_id=context.user_id,
                agent_id=claims.agent_id,
                jti=claims.jti,
                expires_at=claims.expires_at,
                reason=body.reason,
            )
            _emit_management_audit_event(
                stores=stores,
                action="management/tokens/revoke",
                outcome="allow",
                user_id=context.user_id,
                agent_id=context.actor_agent_id,
                request_id=request_id,
                details={"target_agent_id": claims.agent_id},
            )
            return record

        record = service.revoke_token_jti(
            user_id=context.user_id,
            agent_id=body.agent_id or "",
            jti=body.jti or "",
            reason=body.reason,
        )
        _emit_management_audit_event(
            stores=stores,
            action="management/tokens/revoke",
            outcome="allow",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"target_agent_id": body.agent_id},
        )
        return record
    except HTTPException as exc:
        _emit_management_audit_event(
            stores=stores,
            action="management/tokens/revoke",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"error_type": "HTTPException"},
        )
        raise exc
    except Exception as exc:
        _emit_management_audit_event(
            stores=stores,
            action="management/tokens/revoke",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"error_type": type(exc).__name__},
        )
        raise _map_management_error(exc) from exc

