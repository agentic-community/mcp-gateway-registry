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
from ..models.upstream_oauth_provider import (
    UpstreamOAuthProviderCreate,
    UpstreamOAuthProviderPublic,
    UpstreamOAuthProviderUpdate,
)
from ..upstream.server_catalog import (
    list_servers_referencing_upstream_oauth_provider,
)
from .management_common import (
    _emit_management_audit_event,
    _get_request_id,
    _require_admin,
    _require_upstream_oauth_provider_store,
)

router = APIRouter()


@router.get(
    "/admin/upstream-oauth-providers",
    response_model=list[UpstreamOAuthProviderPublic],
)
async def admin_list_upstream_oauth_providers(
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> list[UpstreamOAuthProviderPublic]:
    _require_admin(context)
    provider_store = _require_upstream_oauth_provider_store(stores)
    return provider_store.list_providers()


@router.post(
    "/admin/upstream-oauth-providers",
    response_model=UpstreamOAuthProviderPublic,
)
async def admin_create_upstream_oauth_provider(
    request: Request,
    payload: UpstreamOAuthProviderCreate,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> UpstreamOAuthProviderPublic:
    _require_admin(context)
    request_id = _get_request_id(request)
    provider_store = _require_upstream_oauth_provider_store(stores)

    try:
        created = provider_store.create_provider(payload=payload)
    except ValueError as exc:
        _emit_management_audit_event(
            stores=stores,
            action="admin/upstream-oauth-providers/create",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"error": str(exc), "provider_id": payload.provider_id},
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _emit_management_audit_event(
        stores=stores,
        action="admin/upstream-oauth-providers/create",
        outcome="allow",
        user_id=context.user_id,
        agent_id=context.actor_agent_id,
        request_id=request_id,
        details={
            "provider_id": created.provider.provider_id,
            "authorization_endpoint": created.provider.authorization_endpoint,
            "token_endpoint": created.provider.token_endpoint,
        },
    )
    return created


@router.get(
    "/admin/upstream-oauth-providers/{provider_id}",
    response_model=UpstreamOAuthProviderPublic,
)
async def admin_get_upstream_oauth_provider(
    provider_id: str,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> UpstreamOAuthProviderPublic:
    _require_admin(context)
    provider_store = _require_upstream_oauth_provider_store(stores)
    record = provider_store.get_provider(provider_id=provider_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Provider not found")
    return record


@router.put(
    "/admin/upstream-oauth-providers/{provider_id}",
    response_model=UpstreamOAuthProviderPublic,
)
async def admin_update_upstream_oauth_provider(
    provider_id: str,
    request: Request,
    payload: UpstreamOAuthProviderUpdate,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> UpstreamOAuthProviderPublic:
    _require_admin(context)
    request_id = _get_request_id(request)
    provider_store = _require_upstream_oauth_provider_store(stores)

    try:
        updated = provider_store.update_provider(
            provider_id=provider_id,
            payload=payload,
        )
    except ValueError as exc:
        _emit_management_audit_event(
            stores=stores,
            action="admin/upstream-oauth-providers/update",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"error": str(exc), "provider_id": provider_id},
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if updated is None:
        raise HTTPException(status_code=404, detail="Provider not found")

    _emit_management_audit_event(
        stores=stores,
        action="admin/upstream-oauth-providers/update",
        outcome="allow",
        user_id=context.user_id,
        agent_id=context.actor_agent_id,
        request_id=request_id,
        details={
            "provider_id": provider_id,
            "secret_rotated": payload.client_secret is not None,
        },
    )
    return updated


@router.delete(
    "/admin/upstream-oauth-providers/{provider_id}",
    response_model=dict[str, bool],
)
async def admin_delete_upstream_oauth_provider(
    provider_id: str,
    request: Request,
    force: bool = False,
    settings: EnforceAISettings = Depends(get_enforceai_settings),
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> dict[str, bool]:
    _require_admin(context)
    request_id = _get_request_id(request)
    provider_store = _require_upstream_oauth_provider_store(stores)

    referenced_by: list[str] = []
    registry_servers_dir = settings.resolve_registry_servers_dir()
    if registry_servers_dir is None:
        if not force:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Cannot verify provider references (registry servers dir unavailable). "
                    "Use force=true to delete."
                ),
            )
    else:
        try:
            referenced_by = list_servers_referencing_upstream_oauth_provider(
                provider_id=provider_id,
                servers_dir=registry_servers_dir,
            )
        except Exception as exc:  # noqa: BLE001 - fail closed
            _emit_management_audit_event(
                stores=stores,
                action="admin/upstream-oauth-providers/delete",
                outcome="deny",
                user_id=context.user_id,
                agent_id=context.actor_agent_id,
                request_id=request_id,
                details={"error": str(exc), "provider_id": provider_id},
            )
            raise HTTPException(
                status_code=503,
                detail="Enforcement dependency unavailable",
            ) from exc

        if referenced_by and not force:
            raise HTTPException(
                status_code=409,
                detail="Provider is referenced by one or more servers",
            )

    deleted = provider_store.delete_provider(provider_id=provider_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Provider not found")

    _emit_management_audit_event(
        stores=stores,
        action="admin/upstream-oauth-providers/delete",
        outcome="allow",
        user_id=context.user_id,
        agent_id=context.actor_agent_id,
        request_id=request_id,
        details={
            "provider_id": provider_id,
            "force": force,
            "referenced_by": referenced_by,
        },
    )

    return {"ok": True}
