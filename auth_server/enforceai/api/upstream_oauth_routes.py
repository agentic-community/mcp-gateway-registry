from __future__ import annotations

from typing import (
    Optional,
)
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
    Request,
)
from fastapi.responses import (
    RedirectResponse,
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
from ..models.upstream_oauth import (
    UpstreamOAuthCallbackResponse,
    UpstreamOAuthDisconnectRequest,
    UpstreamOAuthDisconnectResponse,
    UpstreamOAuthServerDisconnectRequest,
    UpstreamOAuthServerStartRequest,
    UpstreamOAuthStartRequest,
    UpstreamOAuthStartResponse,
)
from ..upstream.headers import (
    ENFORCEAI_ERROR_CODE_HEADER,
)
from ..upstream.oauth_client import (
    OAuthTokenClient,
    OAuthTokenClientError,
)
from ..upstream.oauth_flow import (
    consume_oauth_state,
    start_oauth_flow,
)
from ..upstream.oauth_provider_resolver import (
    UPSTREAM_OAUTH_PROVIDER_NOT_CONFIGURED,
    resolve_upstream_oauth_provider,
)
from ..upstream.server_catalog import (
    load_upstream_auth_for_server,
)
from .management_common import (
    _emit_management_audit_event,
    _get_request_id,
    _require_upstream_credential_store,
    _require_upstream_oauth_state_store,
)
from .upstream_common import (
    _normalize_server_path,
)

router = APIRouter()


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

