from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import (
    parse_qsl,
    urlencode,
    urlsplit,
    urlunsplit,
)

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Header,
    Request,
)
from fastapi.responses import (
    RedirectResponse,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
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
from ..models.user import (
    UserRecord,
)
from ..models.agent import (
    AgentRecord,
)
from ..models.revocation import (
    TokenRevocationRecord,
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

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/enforceai",
    tags=["enforceai-management"],
)


def _compute_etag_for_path(
    path: Path,
) -> str:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise DependencyUnavailableError(
            f"Failed to read scope catalog at {path}",
            public_message="Scope catalog unavailable",
        ) from exc

    return hashlib.sha256(payload).hexdigest()


def _get_last_modified_iso(
    path: Path,
) -> Optional[str]:
    try:
        modified = datetime.fromtimestamp(
            path.stat().st_mtime,
            tz=timezone.utc,
        ).replace(microsecond=0)
    except OSError:
        return None

    return modified.isoformat()


def _require_if_match(
    if_match: Optional[str],
) -> str:
    if if_match is None or not if_match.strip():
        raise HTTPException(
            status_code=428,
            detail="Missing If-Match header",
        )
    return if_match.strip()


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


class AdminCreateAgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scopes: list[str]
    allowed_tools: Optional[list[str]] = None
    alias: Optional[str] = None
    metadata: Optional[dict[str, object]] = None


class AdminRevokeGatewayTokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str
    jti: str
    reason: Optional[str] = None


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


# ============================================================================
# Scopes Catalog Response Models
# ============================================================================


class MethodPolicyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    all_methods: bool
    methods: list[str]


class ToolPolicyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    all_tools: bool
    tools: list[str]


class ServerPermissionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    server: str
    methods: MethodPolicyResponse
    tools: Optional[ToolPolicyResponse] = None


class AgentPermissionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: str
    resources: list[str]


class ScopeDefinitionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    server_permissions: list[ServerPermissionResponse]
    agent_permissions: list[AgentPermissionResponse]


class ScopeCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    generated_at: str
    etag: str
    last_modified: Optional[str] = None
    scopes: dict[str, ScopeDefinitionResponse]
    group_mappings: dict[str, list[str]]


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


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _require_admin(
    context: EnforceAIManagementContext,
) -> None:
    if context.is_admin:
        return
    raise HTTPException(
        status_code=403,
        detail="Admin required",
    )


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


def _scope_entries_from_request(
    *,
    server_permissions: list[ServerPermissionUpsert],
    agent_permissions: list[AgentPermissionUpsert],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    for sp in server_permissions:
        methods: list[str]
        if sp.methods.all_methods:
            methods = ["*"]
        else:
            methods = list(sp.methods.methods)

        entry: dict[str, Any] = {
            "server": sp.server,
            "methods": methods,
        }

        if sp.tools is not None:
            if sp.tools.all_tools:
                entry["tools"] = "*"
            else:
                entry["tools"] = list(sp.tools.tools)

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
    context: EnforceAIManagementContext,
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


@router.get("/scopes/catalog", response_model=ScopeCatalogResponse)
async def get_scopes_catalog(
    settings: EnforceAISettings = Depends(get_enforceai_settings),
) -> ScopeCatalogResponse:
    """
    Get the scopes catalog for UI display.

    Returns the full scope definitions and group mappings.
    This endpoint is publicly accessible (no authentication required)
    since the scope catalog is configuration data meant for display.
    """
    from ..fgac.catalog import load_scope_catalog

    try:
        catalog = load_scope_catalog(path=settings.scopes_catalog_path)
        etag = _compute_etag_for_path(catalog.path)
        last_modified = _get_last_modified_iso(catalog.path)
    except DependencyUnavailableError as exc:
        logger.warning(f"Failed to load scope catalog: {exc}")
        raise HTTPException(
            status_code=503,
            detail=exc.public_message or "Scope catalog unavailable",
        ) from exc
    except Exception as exc:
        logger.warning(f"Failed to load scope catalog: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Scope catalog unavailable",
        )

    # Transform internal models to response models
    scopes_response: dict[str, ScopeDefinitionResponse] = {}
    for scope_name, scope_def in catalog.scopes.items():
        server_perms: list[ServerPermissionResponse] = []
        for sp in scope_def.server_permissions:
            tools_resp = None
            if sp.tools is not None:
                tools_resp = ToolPolicyResponse(
                    all_tools=sp.tools.all_tools,
                    tools=list(sp.tools.tools),
                )
            server_perms.append(
                ServerPermissionResponse(
                    server=sp.server,
                    methods=MethodPolicyResponse(
                        all_methods=sp.methods.all_methods,
                        methods=list(sp.methods.methods),
                    ),
                    tools=tools_resp,
                )
            )

        agent_perms: list[AgentPermissionResponse] = []
        for ap in scope_def.agent_permissions:
            agent_perms.append(
                AgentPermissionResponse(
                    action=ap.action,
                    resources=list(ap.resources),
                )
            )

        scopes_response[scope_name] = ScopeDefinitionResponse(
            name=scope_def.name,
            server_permissions=server_perms,
            agent_permissions=agent_perms,
        )

    # Convert group_mappings tuples to lists
    group_mappings_response: dict[str, list[str]] = {
        group: list(scopes)
        for group, scopes in catalog.group_mappings.items()
    }

    return ScopeCatalogResponse(
        version="1.0",
        generated_at=_utc_now().isoformat(),
        etag=etag,
        last_modified=last_modified,
        scopes=scopes_response,
        group_mappings=group_mappings_response,
    )


@router.post("/admin/scopes", response_model=ScopeMutationResponse)
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


@router.put("/admin/scopes/{scope_name}", response_model=ScopeMutationResponse)
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


@router.delete("/admin/scopes/{scope_name}", response_model=ScopeMutationResponse)
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
            agent_count = len(
                stores.agent_store.list_agents_by_user_id(user_id=user.user_id)
            )
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
