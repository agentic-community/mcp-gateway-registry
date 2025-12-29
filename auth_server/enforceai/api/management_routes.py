from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import (
    parse_qsl,
    urlencode,
    urlsplit,
    urlunsplit,
)

import csv
import io

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Header,
    Query,
    Request,
)
from fastapi.responses import (
    RedirectResponse,
    StreamingResponse,
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
    get_upstream_oauth_token_client,
)
from ..config import (
    EnforceAISettings,
)
from ..db.data_layer import (
    EnforceAIStores,
)
from ..errors import (
    DependencyUnavailableError,
    EnforceAIError,
)
from ..tokens.verify import (
    verify_gateway_token,
)
from ..management.models import (
    ApiKeySummary,
)
from ..models.egress_allowlist import (
    EgressAllowlistEntryRecord,
    EgressAllowlistEntryKind,
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
from ..models.upstream_oauth_provider import (
    UpstreamOAuthProviderCreate,
    UpstreamOAuthProviderPublic,
    UpstreamOAuthProviderUpdate,
)
from ..models.upstream_oauth import (
    UpstreamOAuthCallbackResponse,
    UpstreamOAuthDisconnectRequest,
    UpstreamOAuthDisconnectResponse,
    UpstreamOAuthServerDisconnectRequest,
    UpstreamOAuthServerStartRequest,
    UpstreamOAuthStartRequest,
    UpstreamOAuthStartResponse,
)
from ..models.agent import (
    AgentRecord,
)
from ..models.revocation import (
    TokenRevocationRecord,
)
from ..models.audit import (
    AuditEventsQueryResult,
    DEFAULT_AUDIT_PAGE_SIZE,
    DEFAULT_AUDIT_WINDOW_SECONDS,
    MAX_AUDIT_PAGE_SIZE,
)
from ..egress.allowlist import (
    check_proxy_pass_url,
    normalize_allowlist_entry_value,
)
from ..upstream.oauth_client import (
    OAuthTokenClient,
    OAuthTokenClientError,
)
from ..upstream.oauth_flow import (
    consume_oauth_state,
    start_oauth_flow,
)
from ..upstream.headers import (
    ENFORCEAI_ERROR_CODE_HEADER,
)
from ..upstream.server_catalog import (
    load_upstream_auth_for_server,
    list_servers_referencing_upstream_oauth_provider,
)
from ..upstream.oauth_provider_resolver import (
    UPSTREAM_OAUTH_PROVIDER_NOT_CONFIGURED,
    resolve_upstream_oauth_provider,
)
from .management_common import (
    _emit_management_audit_event,
    _get_request_id,
    _map_management_error,
    _require_admin,
    _utc_now,
)
from .admin_scopes_routes import (
    router as admin_scopes_router,
)
from .admin_users_routes import (
    router as admin_users_router,
)
from .management_api_models import (
    CreateAgentRequest,
    CreateApiKeyRequest,
    CreateApiKeyResponse,
    MintTokenRequest,
    MintTokenResponse,
    RevokeTokenRequest,
    UpdateAgentRequest,
)
from .management_service_factory import (
    _build_management_service,
    _load_gateway_keyring,
)
from .scope_catalog_routes import (
    router as scope_catalog_router,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/enforceai",
    tags=["enforceai-management"],
)
router.include_router(admin_scopes_router)
router.include_router(admin_users_router)
router.include_router(scope_catalog_router)


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


def _require_upstream_credential_store(
    stores: EnforceAIStores,
) -> object:
    store = getattr(stores, "upstream_credential_store", None)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="Upstream credential store unavailable (missing ENFORCEAI_UPSTREAM_KEK_PATH)",
        )
    return store


def _require_upstream_oauth_state_store(
    stores: EnforceAIStores,
) -> object:
    store = getattr(stores, "upstream_oauth_state_store", None)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="Upstream OAuth state store unavailable (missing ENFORCEAI_UPSTREAM_KEK_PATH)",
        )
    return store


def _require_upstream_oauth_provider_store(
    stores: EnforceAIStores,
) -> object:
    store = getattr(stores, "upstream_oauth_provider_store", None)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="Upstream OAuth provider store unavailable (missing ENFORCEAI_UPSTREAM_KEK_PATH)",
        )
    return store


def _normalize_server_path(
    raw: str,
) -> str:
    stripped = raw.strip()
    if not stripped:
        raise HTTPException(status_code=400, detail="server_path is required")
    if not stripped.startswith("/"):
        stripped = "/" + stripped
    return stripped.rstrip("/") or "/"


def _append_query_params(
    *,
    url: str,
    params: dict[str, str],
) -> str:
    split = urlsplit(url)
    query = list(parse_qsl(split.query, keep_blank_values=True))
    query.extend((key, value) for key, value in params.items() if value is not None)
    return urlunsplit(
        (
            split.scheme,
            split.netloc,
            split.path,
            urlencode(query),
            split.fragment,
        )
    )


def _owned_agent_ids_for_user(
    *,
    stores: EnforceAIStores,
    user_id: str,
) -> set[str]:
    return {
        record.agent_id for record in stores.agent_store.list_agents_by_user_id(user_id=user_id)
    }


def _is_upstream_credential_visible(
    *,
    record: UpstreamCredentialRecord,
    context: EnforceAIManagementContext,
    owned_agent_ids: set[str],
    include_service: bool,
) -> bool:
    if record.credential_binding == "service":
        return include_service and context.is_admin
    if record.credential_binding == "user":
        return record.user_id == context.user_id
    if record.credential_binding == "agent":
        return record.agent_id in owned_agent_ids
    if record.credential_binding == "user+agent":
        return record.user_id == context.user_id and record.agent_id in owned_agent_ids
    return False


def _credential_key(
    record: UpstreamCredentialRecord,
) -> tuple[object, ...]:
    return (
        record.server_path,
        record.credential_type,
        record.credential_binding,
        record.user_id,
        record.agent_id,
        record.provider,
    )


@router.get("/admin/ping")
async def admin_ping(
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
) -> dict[str, bool]:
    _require_admin(context)
    return {"ok": True}


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
                detail="Cannot verify provider references (registry servers dir unavailable). Use force=true to delete.",
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


@router.post(
    "/upstream/oauth/start",
    response_model=UpstreamOAuthStartResponse,
)
async def start_upstream_oauth_flow(
    payload: UpstreamOAuthStartRequest,
    request: Request,
    settings: EnforceAISettings = Depends(get_enforceai_settings),
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> UpstreamOAuthStartResponse:
    request_id = _get_request_id(request)
    state_store = _require_upstream_oauth_state_store(stores)

    resolved_provider = resolve_upstream_oauth_provider(
        provider_id=payload.provider,
        stores=stores,
        settings=settings,
        env_providers=settings.upstream_oauth_providers,
        require_client_secret=False,
    )
    if resolved_provider is None:
        raise HTTPException(
            status_code=424,
            detail="Upstream OAuth provider not configured",
            headers={ENFORCEAI_ERROR_CODE_HEADER: UPSTREAM_OAUTH_PROVIDER_NOT_CONFIGURED},
        )

    agent_id: Optional[str] = None
    if payload.credential_binding == "user+agent":
        agent_id = payload.agent_id
        if agent_id is None:
            raise HTTPException(status_code=400, detail="agent_id is required")
        record = stores.agent_store.get_agent_by_id(agent_id=agent_id)
        if record is None or record.user_id != context.user_id:
            raise HTTPException(status_code=404, detail="Agent not found")

    redirect_uri = str(request.url_for("upstream_oauth_callback"))

    try:
        started = start_oauth_flow(
            state_store=state_store,
            authorization_endpoint=resolved_provider.authorization_endpoint,
            client_id=resolved_provider.client_id,
            default_scopes=resolved_provider.default_scopes,
            extra_authorize_params=resolved_provider.extra_authorize_params,
            provider_id=payload.provider,
            server_path=payload.server_path,
            credential_type=payload.credential_type,
            credential_binding=payload.credential_binding,
            user_id=context.user_id,
            agent_id=agent_id,
            redirect_uri=redirect_uri,
            ui_return_url=payload.ui_return_url,
            scopes=payload.scopes,
            ttl_seconds=settings.upstream_oauth_state_ttl_seconds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _emit_management_audit_event(
        stores=stores,
        action="upstream/oauth/start",
        outcome="allow",
        user_id=context.user_id,
        agent_id=context.actor_agent_id,
        request_id=request_id,
        details={
            "server_path": payload.server_path,
            "credential_type": payload.credential_type,
            "credential_binding": payload.credential_binding,
            "provider": payload.provider,
        },
    )

    return UpstreamOAuthStartResponse(
        authorization_url=started.authorization_url,
        state_id=started.state_id,
        expires_at=started.expires_at,
    )


@router.post(
    "/upstream/servers/{server_path:path}/oauth/start",
    response_model=UpstreamOAuthStartResponse,
)
async def start_upstream_server_oauth_flow(
    server_path: str,
    payload: UpstreamOAuthServerStartRequest,
    request: Request,
    settings: EnforceAISettings = Depends(get_enforceai_settings),
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> UpstreamOAuthStartResponse:
    request_id = _get_request_id(request)
    state_store = _require_upstream_oauth_state_store(stores)

    normalized_server_path = _normalize_server_path(server_path)
    server_path_param = normalized_server_path.lstrip("/")
    if not server_path_param:
        raise HTTPException(status_code=400, detail="server_path is required")

    registry_servers_dir = settings.resolve_registry_servers_dir()
    if registry_servers_dir is None:
        raise HTTPException(status_code=503, detail="Registry server catalog unavailable")

    try:
        upstream_auth = load_upstream_auth_for_server(
            server_path=normalized_server_path,
            servers_dir=registry_servers_dir,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Registry server catalog unavailable")
    except ValueError as exc:
        message = str(exc) or "Server not found"
        if message == "Server not found":
            raise HTTPException(status_code=404, detail="Server not found")
        raise HTTPException(status_code=400, detail=message) from exc

    if upstream_auth.type not in {"oauth2", "oidc", "provider-oauth"}:
        raise HTTPException(status_code=400, detail="Server does not require upstream OAuth")

    if upstream_auth.provider is None:
        raise HTTPException(status_code=400, detail="Server upstream_auth.provider is required")

    expected_credential_type: str = upstream_auth.type
    if payload.credential_type != expected_credential_type:
        raise HTTPException(status_code=400, detail="OAuth credential_type mismatch")

    if payload.credential_binding != upstream_auth.credential_binding:
        raise HTTPException(status_code=400, detail="OAuth credential_binding mismatch")

    if payload.provider != upstream_auth.provider:
        raise HTTPException(status_code=400, detail="OAuth provider mismatch")

    resolved_provider = resolve_upstream_oauth_provider(
        provider_id=upstream_auth.provider,
        stores=stores,
        settings=settings,
        env_providers=settings.upstream_oauth_providers,
        require_client_secret=False,
    )
    if resolved_provider is None:
        raise HTTPException(
            status_code=424,
            detail="Upstream OAuth provider not configured",
            headers={ENFORCEAI_ERROR_CODE_HEADER: UPSTREAM_OAUTH_PROVIDER_NOT_CONFIGURED},
        )

    agent_id: Optional[str] = None
    if payload.credential_binding == "user+agent":
        agent_id = payload.agent_id
        if agent_id is None:
            raise HTTPException(status_code=400, detail="agent_id is required")
        record = stores.agent_store.get_agent_by_id(agent_id=agent_id)
        if record is None or record.user_id != context.user_id:
            raise HTTPException(status_code=404, detail="Agent not found")

    redirect_uri = str(
        request.url_for(
            "upstream_server_oauth_callback",
            server_path=server_path_param,
        )
    )

    try:
        started = start_oauth_flow(
            state_store=state_store,
            authorization_endpoint=resolved_provider.authorization_endpoint,
            client_id=resolved_provider.client_id,
            default_scopes=resolved_provider.default_scopes,
            extra_authorize_params=resolved_provider.extra_authorize_params,
            provider_id=upstream_auth.provider,
            server_path=normalized_server_path,
            credential_type=payload.credential_type,
            credential_binding=payload.credential_binding,
            user_id=context.user_id,
            agent_id=agent_id,
            redirect_uri=redirect_uri,
            ui_return_url=payload.ui_return_url,
            scopes=payload.scopes,
            ttl_seconds=settings.upstream_oauth_state_ttl_seconds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _emit_management_audit_event(
        stores=stores,
        action="upstream/servers/oauth/start",
        outcome="allow",
        user_id=context.user_id,
        agent_id=context.actor_agent_id,
        request_id=request_id,
        details={
            "server_path": normalized_server_path,
            "credential_type": payload.credential_type,
            "credential_binding": payload.credential_binding,
            "provider": payload.provider,
        },
    )

    return UpstreamOAuthStartResponse(
        authorization_url=started.authorization_url,
        state_id=started.state_id,
        expires_at=started.expires_at,
    )


@router.get(
    "/upstream/servers/{server_path:path}/oauth/callback",
    name="upstream_server_oauth_callback",
)
async def upstream_server_oauth_callback(
    server_path: str,
    request: Request,
    state: str,
    code: Optional[str] = None,
    error: Optional[str] = None,
    settings: EnforceAISettings = Depends(get_enforceai_settings),
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
    token_client: OAuthTokenClient = Depends(get_upstream_oauth_token_client),
) -> RedirectResponse:
    request_id = _get_request_id(request)
    normalized_server_path = _normalize_server_path(server_path)

    state_store = _require_upstream_oauth_state_store(stores)
    upstream_store = _require_upstream_credential_store(stores)

    try:
        consumed = consume_oauth_state(
            state_store=state_store,
            state_id=state,
            actor_user_id=context.user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if consumed.server_path != normalized_server_path:
        raise HTTPException(status_code=400, detail="OAuth state does not match server_path")

    ui_return_url = consumed.ui_return_url
    if ui_return_url is None or not ui_return_url.strip():
        raise HTTPException(status_code=400, detail="OAuth state missing ui_return_url")

    if error is not None and error.strip():
        target = _append_query_params(
            url=ui_return_url,
            params={
                "upstream_oauth": "error",
                "error_code": "authorization_failed",
                "server_path": consumed.server_path,
                "provider": consumed.provider,
            },
        )
        return RedirectResponse(url=target, status_code=302)

    if code is None or not code.strip():
        target = _append_query_params(
            url=ui_return_url,
            params={
                "upstream_oauth": "error",
                "error_code": "missing_code",
                "server_path": consumed.server_path,
                "provider": consumed.provider,
            },
        )
        return RedirectResponse(url=target, status_code=302)

    resolved_provider = None
    try:
        resolved_provider = resolve_upstream_oauth_provider(
            provider_id=consumed.provider,
            stores=stores,
            settings=settings,
            env_providers=settings.upstream_oauth_providers,
            require_client_secret=True,
        )
    except ValueError:
        resolved_provider = None

    if resolved_provider is None or resolved_provider.client_secret is None:
        target = _append_query_params(
            url=ui_return_url,
            params={
                "upstream_oauth": "error",
                "error_code": "provider_not_configured",
                "server_path": consumed.server_path,
                "provider": consumed.provider,
            },
        )
        return RedirectResponse(url=target, status_code=302)

    try:
        tokens = await token_client.exchange_authorization_code(
            token_endpoint=resolved_provider.token_endpoint,
            client_id=resolved_provider.client_id,
            client_secret=resolved_provider.client_secret,
            code=code.strip(),
            redirect_uri=consumed.redirect_uri,
            code_verifier=consumed.code_verifier,
        )
    except OAuthTokenClientError as exc:
        _emit_management_audit_event(
            stores=stores,
            action="upstream/servers/oauth/callback",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"error": exc.message},
        )
        target = _append_query_params(
            url=ui_return_url,
            params={
                "upstream_oauth": "error",
                "error_code": "token_exchange_failed",
                "server_path": consumed.server_path,
                "provider": consumed.provider,
            },
        )
        return RedirectResponse(url=target, status_code=302)

    existing = upstream_store.list_credentials(
        server_path=consumed.server_path,
        user_id=context.user_id,
        agent_id=consumed.agent_id,
        include_revoked=False,
    )
    for record in existing:
        if (
            record.credential_type == consumed.credential_type
            and record.credential_binding == consumed.credential_binding
            and record.provider == consumed.provider
        ):
            upstream_store.revoke_credential(credential_id=record.credential_id)

    secret_payload: dict[str, object] = {
        "access_token": tokens.access_token,
    }
    if tokens.refresh_token is not None:
        secret_payload["refresh_token"] = tokens.refresh_token
    if tokens.id_token is not None:
        secret_payload["id_token"] = tokens.id_token

    created = upstream_store.create_credential(
        server_path=consumed.server_path,
        credential_type=consumed.credential_type,
        credential_binding=consumed.credential_binding,
        user_id=context.user_id,
        agent_id=consumed.agent_id,
        provider=consumed.provider,
        scopes=tokens.scopes,
        token_type=tokens.token_type,
        expires_at=tokens.expires_at,
        secret_payload=secret_payload,
    )

    _emit_management_audit_event(
        stores=stores,
        action="upstream/servers/oauth/callback",
        outcome="allow",
        user_id=context.user_id,
        agent_id=context.actor_agent_id,
        request_id=request_id,
        details={
            "credential_id": created.credential_id,
            "server_path": created.server_path,
            "credential_type": created.credential_type,
            "credential_binding": created.credential_binding,
            "provider": created.provider,
        },
    )

    target = _append_query_params(
        url=ui_return_url,
        params={
            "upstream_oauth": "success",
            "server_path": created.server_path,
            "provider": created.provider or "",
            "credential_id": created.credential_id,
        },
    )
    return RedirectResponse(url=target, status_code=302)


@router.get(
    "/upstream/oauth/callback",
    response_model=UpstreamOAuthCallbackResponse,
    name="upstream_oauth_callback",
)
async def upstream_oauth_callback(
    request: Request,
    state: str,
    code: Optional[str] = None,
    error: Optional[str] = None,
    settings: EnforceAISettings = Depends(get_enforceai_settings),
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
    token_client: OAuthTokenClient = Depends(get_upstream_oauth_token_client),
) -> UpstreamOAuthCallbackResponse:
    request_id = _get_request_id(request)
    if error is not None and error.strip():
        raise HTTPException(status_code=400, detail="Upstream OAuth authorization failed")
    if code is None or not code.strip():
        raise HTTPException(status_code=400, detail="Missing authorization code")

    state_store = _require_upstream_oauth_state_store(stores)
    upstream_store = _require_upstream_credential_store(stores)

    try:
        consumed = consume_oauth_state(
            state_store=state_store,
            state_id=state,
            actor_user_id=context.user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        resolved_provider = resolve_upstream_oauth_provider(
            provider_id=consumed.provider,
            stores=stores,
            settings=settings,
            env_providers=settings.upstream_oauth_providers,
            require_client_secret=True,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=424,
            detail="Upstream OAuth provider not configured",
            headers={ENFORCEAI_ERROR_CODE_HEADER: UPSTREAM_OAUTH_PROVIDER_NOT_CONFIGURED},
        ) from exc

    if resolved_provider is None or resolved_provider.client_secret is None:
        raise HTTPException(
            status_code=424,
            detail="Upstream OAuth provider not configured",
            headers={ENFORCEAI_ERROR_CODE_HEADER: UPSTREAM_OAUTH_PROVIDER_NOT_CONFIGURED},
        )

    try:
        tokens = await token_client.exchange_authorization_code(
            token_endpoint=resolved_provider.token_endpoint,
            client_id=resolved_provider.client_id,
            client_secret=resolved_provider.client_secret,
            code=code.strip(),
            redirect_uri=consumed.redirect_uri,
            code_verifier=consumed.code_verifier,
        )
    except OAuthTokenClientError as exc:
        _emit_management_audit_event(
            stores=stores,
            action="upstream/oauth/callback",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id,
            details={"error": exc.message},
        )
        raise HTTPException(status_code=502, detail="Upstream OAuth token exchange failed") from exc

    existing = upstream_store.list_credentials(
        server_path=consumed.server_path,
        user_id=context.user_id,
        agent_id=consumed.agent_id,
        include_revoked=False,
    )
    for record in existing:
        if (
            record.credential_type == consumed.credential_type
            and record.credential_binding == consumed.credential_binding
            and record.provider == consumed.provider
        ):
            upstream_store.revoke_credential(credential_id=record.credential_id)

    secret_payload: dict[str, object] = {
        "access_token": tokens.access_token,
    }
    if tokens.refresh_token is not None:
        secret_payload["refresh_token"] = tokens.refresh_token
    if tokens.id_token is not None:
        secret_payload["id_token"] = tokens.id_token

    created = upstream_store.create_credential(
        server_path=consumed.server_path,
        credential_type=consumed.credential_type,
        credential_binding=consumed.credential_binding,
        user_id=context.user_id,
        agent_id=consumed.agent_id,
        provider=consumed.provider,
        scopes=tokens.scopes,
        token_type=tokens.token_type,
        expires_at=tokens.expires_at,
        secret_payload=secret_payload,
    )

    _emit_management_audit_event(
        stores=stores,
        action="upstream/oauth/callback",
        outcome="allow",
        user_id=context.user_id,
        agent_id=context.actor_agent_id,
        request_id=request_id,
        details={
            "credential_id": created.credential_id,
            "server_path": created.server_path,
            "credential_type": created.credential_type,
            "credential_binding": created.credential_binding,
            "provider": created.provider,
        },
    )

    return UpstreamOAuthCallbackResponse(
        credential_id=created.credential_id,
        server_path=created.server_path,
        provider=created.provider or "",
    )


@router.post(
    "/upstream/oauth/disconnect",
    response_model=UpstreamOAuthDisconnectResponse,
)
async def disconnect_upstream_oauth(
    payload: UpstreamOAuthDisconnectRequest,
    request: Request,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> UpstreamOAuthDisconnectResponse:
    request_id = _get_request_id(request)
    upstream_store = _require_upstream_credential_store(stores)

    agent_id: Optional[str] = None
    if payload.credential_binding == "user+agent":
        agent_id = payload.agent_id
        if agent_id is None:
            raise HTTPException(status_code=400, detail="agent_id is required")
        record = stores.agent_store.get_agent_by_id(agent_id=agent_id)
        if record is None or record.user_id != context.user_id:
            raise HTTPException(status_code=404, detail="Agent not found")

    records = upstream_store.list_credentials(
        server_path=payload.server_path,
        user_id=context.user_id,
        agent_id=agent_id,
        include_revoked=False,
    )
    revoked = 0
    for record in records:
        if (
            record.credential_type == payload.credential_type
            and record.credential_binding == payload.credential_binding
            and record.provider == payload.provider
        ):
            upstream_store.revoke_credential(credential_id=record.credential_id)
            revoked += 1

    _emit_management_audit_event(
        stores=stores,
        action="upstream/oauth/disconnect",
        outcome="allow",
        user_id=context.user_id,
        agent_id=context.actor_agent_id,
        request_id=request_id,
        details={
            "server_path": payload.server_path,
            "credential_type": payload.credential_type,
            "credential_binding": payload.credential_binding,
            "provider": payload.provider,
            "revoked_count": revoked,
        },
    )

    return UpstreamOAuthDisconnectResponse(revoked_count=revoked)


@router.post(
    "/upstream/servers/{server_path:path}/oauth/disconnect",
    response_model=UpstreamOAuthDisconnectResponse,
)
async def disconnect_upstream_server_oauth(
    server_path: str,
    payload: UpstreamOAuthServerDisconnectRequest,
    request: Request,
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> UpstreamOAuthDisconnectResponse:
    request_id = _get_request_id(request)
    upstream_store = _require_upstream_credential_store(stores)

    normalized_server_path = _normalize_server_path(server_path)

    agent_id: Optional[str] = None
    if payload.credential_binding == "user+agent":
        agent_id = payload.agent_id
        if agent_id is None:
            raise HTTPException(status_code=400, detail="agent_id is required")
        record = stores.agent_store.get_agent_by_id(agent_id=agent_id)
        if record is None or record.user_id != context.user_id:
            raise HTTPException(status_code=404, detail="Agent not found")

    records = upstream_store.list_credentials(
        server_path=normalized_server_path,
        user_id=context.user_id,
        agent_id=agent_id,
        include_revoked=False,
    )
    revoked = 0
    for record in records:
        if (
            record.credential_type == payload.credential_type
            and record.credential_binding == payload.credential_binding
            and record.provider == payload.provider
        ):
            upstream_store.revoke_credential(credential_id=record.credential_id)
            revoked += 1

    _emit_management_audit_event(
        stores=stores,
        action="upstream/servers/oauth/disconnect",
        outcome="allow",
        user_id=context.user_id,
        agent_id=context.actor_agent_id,
        request_id=request_id,
        details={
            "server_path": normalized_server_path,
            "credential_type": payload.credential_type,
            "credential_binding": payload.credential_binding,
            "provider": payload.provider,
            "revoked_count": revoked,
        },
    )

    return UpstreamOAuthDisconnectResponse(revoked_count=revoked)


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


# ============================================================================
# Audit Events API (Self-Service)
# ============================================================================


@router.get("/audit/events", response_model=AuditEventsQueryResult)
async def list_audit_events(
    request: Request,
    since: Optional[datetime] = Query(
        None,
        description="Filter events after this time (ISO 8601)",
    ),
    until: Optional[datetime] = Query(
        None,
        description="Filter events before this time (ISO 8601)",
    ),
    limit: int = Query(
        DEFAULT_AUDIT_PAGE_SIZE,
        ge=1,
        le=MAX_AUDIT_PAGE_SIZE,
        description="Maximum number of events to return",
    ),
    cursor: Optional[str] = Query(
        None,
        description="Pagination cursor from previous response",
    ),
    agent_id: Optional[str] = Query(
        None,
        description="Filter by agent ID",
    ),
    action: Optional[list[str]] = Query(
        None,
        description="Filter by action names",
    ),
    outcome: Optional[list[str]] = Query(
        None,
        description="Filter by outcome values (allow, deny)",
    ),
    request_id: Optional[str] = Query(
        None,
        description="Filter by exact request ID match",
    ),
    server: Optional[str] = Query(
        None,
        description="Filter by server (from event details)",
    ),
    tool: Optional[str] = Query(
        None,
        description="Filter by tool (from event details)",
    ),
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> AuditEventsQueryResult:
    """
    List audit events for the authenticated user.

    Events are returned in descending order by time (most recent first).
    Use the `cursor` parameter from the response to fetch the next page.

    Time window defaults to the last 60 minutes if not specified.
    """
    now = datetime.now(timezone.utc)

    if until is None:
        until = now
    else:
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)

    if since is None:
        since = until - timedelta(seconds=DEFAULT_AUDIT_WINDOW_SECONDS)
    else:
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)

    if since > until:
        raise HTTPException(
            status_code=400,
            detail="'since' must be before 'until'",
        )

    result = stores.audit_store.query_events(
        user_id=context.user_id,
        agent_id=agent_id,
        actions=action,
        outcomes=outcome,
        request_id=request_id,
        server=server,
        tool=tool,
        since=since,
        until=until,
        limit=limit,
        cursor=cursor,
    )

    return result


# ============================================================================
# Admin Audit Events API
# ============================================================================


@router.get("/admin/audit/events", response_model=AuditEventsQueryResult)
async def list_admin_audit_events(
    request: Request,
    user_id: Optional[str] = Query(
        None,
        description="Filter events by user ID (admin only, omit to query all users)",
    ),
    since: Optional[datetime] = Query(
        None,
        description="Filter events after this time (ISO 8601)",
    ),
    until: Optional[datetime] = Query(
        None,
        description="Filter events before this time (ISO 8601)",
    ),
    limit: int = Query(
        DEFAULT_AUDIT_PAGE_SIZE,
        ge=1,
        le=MAX_AUDIT_PAGE_SIZE,
        description=f"Maximum events to return (max {MAX_AUDIT_PAGE_SIZE})",
    ),
    cursor: Optional[str] = Query(
        None,
        description="Pagination cursor from previous response",
    ),
    agent_id: Optional[str] = Query(
        None,
        description="Filter by agent ID",
    ),
    action: Optional[list[str]] = Query(
        None,
        description="Filter by action(s)",
    ),
    outcome: Optional[list[str]] = Query(
        None,
        description="Filter by outcome(s) (allow, deny)",
    ),
    request_id: Optional[str] = Query(
        None,
        description="Filter by request ID",
    ),
    server: Optional[str] = Query(
        None,
        description="Filter by server (from event details)",
    ),
    tool: Optional[str] = Query(
        None,
        description="Filter by tool (from event details)",
    ),
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> AuditEventsQueryResult:
    """
    List audit events across all users (admin only).

    Events are returned in descending order by time (most recent first).
    Use the `cursor` parameter from the response to fetch the next page.

    Time window defaults to the last 60 minutes if not specified.
    """
    _require_admin(context)
    request_id_header = _get_request_id(request)

    now = datetime.now(timezone.utc)

    if until is None:
        until = now
    else:
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)

    if since is None:
        since = until - timedelta(seconds=DEFAULT_AUDIT_WINDOW_SECONDS)
    else:
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)

    if since > until:
        raise HTTPException(
            status_code=400,
            detail="'since' must be before 'until'",
        )

    result = stores.audit_store.query_events(
        user_id=user_id,  # None means all users
        agent_id=agent_id,
        actions=action,
        outcomes=outcome,
        request_id=request_id,
        server=server,
        tool=tool,
        since=since,
        until=until,
        limit=limit,
        cursor=cursor,
    )

    _emit_management_audit_event(
        stores=stores,
        action="admin/audit/query",
        outcome="allow",
        user_id=context.user_id,
        agent_id=context.actor_agent_id,
        request_id=request_id_header,
        details={
            "target_user_id": user_id,
            "count": len(result.items),
            "has_more": result.next_cursor is not None,
        },
    )

    return result


# Maximum events to export in a single request
MAX_EXPORT_EVENTS = 10000


@router.get("/admin/audit/events/export")
async def export_admin_audit_events(
    request: Request,
    user_id: Optional[str] = Query(
        None,
        description="Filter events by user ID (admin only, omit to query all users)",
    ),
    since: Optional[datetime] = Query(
        None,
        description="Filter events after this time (ISO 8601)",
    ),
    until: Optional[datetime] = Query(
        None,
        description="Filter events before this time (ISO 8601)",
    ),
    agent_id: Optional[str] = Query(
        None,
        description="Filter by agent ID",
    ),
    action: Optional[list[str]] = Query(
        None,
        description="Filter by action(s)",
    ),
    outcome: Optional[list[str]] = Query(
        None,
        description="Filter by outcome(s) (allow, deny)",
    ),
    request_id: Optional[str] = Query(
        None,
        description="Filter by request ID",
    ),
    server: Optional[str] = Query(
        None,
        description="Filter by server (from event details)",
    ),
    tool: Optional[str] = Query(
        None,
        description="Filter by tool (from event details)",
    ),
    context: EnforceAIManagementContext = Depends(get_enforceai_management_context),
    stores: EnforceAIStores = Depends(get_enforceai_stores),
) -> StreamingResponse:
    """
    Export audit events as CSV (admin only).

    Returns up to 10,000 events matching the filters. If more than 10,000 events
    match, returns HTTP 413 and requires narrowing filters.
    Time window defaults to the last 60 minutes if not specified.
    """
    _require_admin(context)
    request_id_header = _get_request_id(request)

    now = datetime.now(timezone.utc)

    if until is None:
        until = now
    else:
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)

    if since is None:
        since = until - timedelta(seconds=DEFAULT_AUDIT_WINDOW_SECONDS)
    else:
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)

    if since > until:
        raise HTTPException(
            status_code=400,
            detail="'since' must be before 'until'",
        )

    result = stores.audit_store.query_events(
        user_id=user_id,
        agent_id=agent_id,
        actions=action,
        outcomes=outcome,
        request_id=request_id,
        server=server,
        tool=tool,
        since=since,
        until=until,
        limit=MAX_EXPORT_EVENTS,
        cursor=None,
        max_limit=MAX_EXPORT_EVENTS,
    )

    if result.next_cursor is not None:
        _emit_management_audit_event(
            stores=stores,
            action="admin/audit/export",
            outcome="deny",
            user_id=context.user_id,
            agent_id=context.actor_agent_id,
            request_id=request_id_header,
            details={
                "target_user_id": user_id,
                "reason": "too_many_events",
                "limit": MAX_EXPORT_EVENTS,
            },
        )
        raise HTTPException(
            status_code=413,
            detail=f"Too many events to export (>{MAX_EXPORT_EVENTS}). Narrow filters and try again.",
        )

    def _csv_stream() -> Iterator[str]:
        output = io.StringIO()
        writer = csv.writer(output)

        header = [
            "event_id",
            "occurred_at",
            "user_id",
            "agent_id",
            "action",
            "outcome",
            "request_id",
            "server",
            "tool",
            "reason",
            "matched_scope",
            "provider",
            "details_json",
        ]
        writer.writerow(header)
        yield output.getvalue()
        output.seek(0)
        output.truncate(0)

        for event in result.items:
            details = event.details or {}
            writer.writerow(
                [
                    event.event_id,
                    event.occurred_at.isoformat(),
                    event.user_id,
                    event.agent_id,
                    event.action,
                    event.outcome,
                    event.request_id or "",
                    str(details.get("server") or ""),
                    str(details.get("tool") or ""),
                    str(details.get("reason") or ""),
                    str(details.get("matched_scope") or ""),
                    str(details.get("provider") or ""),
                    json.dumps(
                        event.details,
                        separators=(",", ":"),
                        sort_keys=True,
                        default=str,
                    )
                    if event.details
                    else "",
                ]
            )
            yield output.getvalue()
            output.seek(0)
            output.truncate(0)

    # Generate filename with timestamp
    export_timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f"audit_events_{export_timestamp}.csv"

    _emit_management_audit_event(
        stores=stores,
        action="admin/audit/export",
        outcome="allow",
        user_id=context.user_id,
        agent_id=context.actor_agent_id,
        request_id=request_id_header,
        details={
            "target_user_id": user_id,
            "count": len(result.items),
            "filename": filename,
        },
    )

    # Return CSV as streaming response
    return StreamingResponse(
        _csv_stream(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )
