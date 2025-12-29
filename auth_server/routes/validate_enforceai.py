from __future__ import annotations

import json
import logging
from typing import (
    Callable,
    Protocol,
)

from fastapi import (
    HTTPException,
    Request,
)
from fastapi.responses import (
    JSONResponse,
)

try:
    from .validate_common import (
        SESSION_COOKIE_NAME,
        _SESSION_COOKIE_PREFIX,
        _extract_cookie_value,
    )
except ImportError:  # pragma: no cover
    from validate_common import (  # type: ignore[no-redef]
        SESSION_COOKIE_NAME,
        _SESSION_COOKIE_PREFIX,
        _extract_cookie_value,
    )

try:
    from ..enforceai_runtime import (
        _load_enforceai_runtime,
        evaluate_tool_call,
        get_enforceai_settings,
        get_upstream_oauth_token_client,
        resolve_callable_tools_for_server,
    )
    from ..enforceai_support import (
        emit_enforceai_audit_event as _emit_enforceai_audit_event,
        resolve_enforceai_scopes_catalog_path as _resolve_enforceai_scopes_catalog_path,
    )
    from ..validation_utils import (
        validate_session_cookie,
    )
except ImportError:  # pragma: no cover
    from enforceai_runtime import (  # type: ignore[no-redef]
        _load_enforceai_runtime,
        evaluate_tool_call,
        get_enforceai_settings,
        get_upstream_oauth_token_client,
        resolve_callable_tools_for_server,
    )
    from enforceai_support import (  # type: ignore[no-redef]
        emit_enforceai_audit_event as _emit_enforceai_audit_event,
        resolve_enforceai_scopes_catalog_path as _resolve_enforceai_scopes_catalog_path,
    )
    from validation_utils import (  # type: ignore[no-redef]
        validate_session_cookie,
    )

logger = logging.getLogger(__name__)


class _IdentityResolver(Protocol):
    async def resolve_identity(
        self,
        *,
        headers: dict[str, str],
    ) -> object: ...


async def _validate_request_with_enforceai(
    request: Request,
    original_path: str,
    server_name_from_url: str | None,
    server_name: str | None,
    tool_name: str | None,
    request_payload: object,
    *,
    get_identity_resolver: Callable[[], _IdentityResolver],
    load_scope_catalog: Callable[..., object],
    get_enforceai_stores: Callable[[], object],
) -> JSONResponse:
    runtime = _load_enforceai_runtime()
    dependency_unavailable_error = runtime.DependencyUnavailableError
    enforceai_error = runtime.EnforceAIError

    has_non_cookie_credentials = any(
        value and value.strip()
        for value in (
            request.headers.get("authorization"),
            request.headers.get("x-authorization"),
            request.headers.get("x-gateway-token"),
            request.headers.get("x-api-key"),
        )
    )

    allow_cookie_auth = original_path.startswith("/api/")
    cookie_header = request.headers.get("Cookie", "")

    if allow_cookie_auth and not has_non_cookie_credentials:
        if _SESSION_COOKIE_PREFIX not in cookie_header and SESSION_COOKIE_NAME not in request.cookies:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"Connection": "close"},
            )

        cookie_value = _extract_cookie_value(
            request=request,
            cookie_header=cookie_header,
            cookie_name=SESSION_COOKIE_NAME,
        )

        if not cookie_value:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"Connection": "close"},
            )

        try:
            validation_result = validate_session_cookie(cookie_value)
        except ValueError as exc:
            raise HTTPException(
                status_code=401,
                detail=str(exc),
                headers={"Connection": "close"},
            ) from exc

        response_data = {
            "valid": True,
            "username": validation_result.get("username") or "",
            "client_id": validation_result.get("client_id") or "",
            "scopes": validation_result.get("scopes", []),
            "method": validation_result.get("method") or "",
            "groups": validation_result.get("groups", []),
            "server_name": server_name_from_url,
            "tool_name": tool_name,
        }
        response = JSONResponse(
            content=response_data,
            status_code=200,
        )
        response.headers["X-User"] = validation_result.get("username") or ""
        response.headers["X-Username"] = validation_result.get("username") or ""
        response.headers["X-Client-Id"] = validation_result.get("client_id") or ""
        response.headers["X-Scopes"] = " ".join(validation_result.get("scopes", []))
        response.headers["X-Auth-Method"] = validation_result.get("method") or ""
        response.headers["X-Server-Name"] = server_name_from_url or ""
        response.headers["X-Tool-Name"] = tool_name or ""
        return response

    try:
        resolver = get_identity_resolver()
        catalog_path = _resolve_enforceai_scopes_catalog_path()
        if catalog_path is None:
            catalog = load_scope_catalog()
        else:
            catalog = load_scope_catalog(path=catalog_path)
        identity = await resolver.resolve_identity(headers=dict(request.headers))
    except dependency_unavailable_error as exc:
        raise HTTPException(
            status_code=503,
            detail=exc.public_message,
            headers={"Connection": "close"},
        ) from exc
    except enforceai_error as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.public_message,
            headers={"Connection": "close"},
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected EnforceAI failure during identity resolution")
        raise HTTPException(
            status_code=503,
            detail="Enforcement dependency unavailable",
            headers={"Connection": "close"},
        ) from exc

    method = tool_name or "initialize"
    actual_tool_name = None
    if method == "tools/call" and isinstance(request_payload, dict):
        params = request_payload.get("params")
        if isinstance(params, dict):
            actual_tool_name = params.get("name")

    request_id = request.headers.get("X-Request-Id")
    if not request_id and isinstance(request_payload, dict) and "id" in request_payload:
        request_id_value = request_payload.get("id")
        if request_id_value is not None:
            request_id = str(request_id_value)

    allowed_tools = None
    identity_metadata = getattr(identity, "metadata", None)
    if isinstance(identity_metadata, dict):
        allowed_tools = identity_metadata.get("agent_allowed_tools")

    allowed_tools_header_value = ""
    if server_name and method == "tools/list":
        tool_policy = resolve_callable_tools_for_server(
            identity=identity,
            catalog=catalog,
            server=server_name,
            allowed_tools=allowed_tools,
        )
        if tool_policy.all_tools:
            allowed_tools_header_value = "*"
        else:
            allowed_tools_header_value = json.dumps(sorted(tool_policy.tools))

    if server_name and method in {"tools/list", "tools/call"}:
        if method == "tools/list":
            _emit_enforceai_audit_event(
                action="tools/list",
                outcome="allow",
                user_id=identity.user_id,
                agent_id=identity.agent_id,
                request_id=request_id,
                details={
                    "provider": identity.provider,
                    "server": server_name,
                    "allowed_tools": allowed_tools_header_value,
                },
                get_stores=get_enforceai_stores,
            )

        if method == "tools/call":
            if not actual_tool_name:
                _emit_enforceai_audit_event(
                    action="tools/call",
                    outcome="deny",
                    user_id=identity.user_id,
                    agent_id=identity.agent_id,
                    request_id=request_id,
                    details={
                        "provider": identity.provider,
                        "server": server_name,
                        "reason": "missing_tool_name",
                    },
                    get_stores=get_enforceai_stores,
                )
                raise HTTPException(
                    status_code=403,
                    detail="Forbidden",
                    headers={"Connection": "close"},
                )

            decision = evaluate_tool_call(
                identity=identity,
                catalog=catalog,
                server=server_name,
                tool=actual_tool_name,
                allowed_tools=allowed_tools,
            )
            _emit_enforceai_audit_event(
                action="tools/call",
                outcome="allow" if decision.allowed else "deny",
                user_id=identity.user_id,
                agent_id=identity.agent_id,
                request_id=request_id,
                details={
                    "provider": identity.provider,
                    "server": server_name,
                    "tool": actual_tool_name,
                    "reason": decision.reason,
                    "matched_scope": decision.matched_scope,
                },
                get_stores=get_enforceai_stores,
            )
            if not decision.allowed:
                raise HTTPException(
                    status_code=403,
                    detail="Forbidden",
                    headers={"Connection": "close"},
                )

    try:
        from auth_server.enforceai.models.upstream_auth import (
            UpstreamAuthConfig,
            UpstreamAuthInjection,
        )
        from auth_server.enforceai.upstream.headers import (
            ENFORCEAI_ERROR_CODE_HEADER,
            ENFORCEAI_UPSTREAM_API_KEY_HEADER,
            ENFORCEAI_UPSTREAM_API_KEY_HEADER_NAME_HEADER,
            ENFORCEAI_UPSTREAM_AUTHORIZATION_HEADER,
            ENFORCEAI_UPSTREAM_MODE_HEADER,
            MCP_AUTH_TYPE_HEADER,
            MCP_CLAIMS_HEADER,
            MCP_PRINCIPAL_HEADER,
            MCP_PROVIDER_HEADER,
            MCP_SCOPES_HEADER,
        )
        from auth_server.enforceai.upstream.resolver import (
            UpstreamInjectionError,
            resolve_upstream_injection,
        )
    except ImportError:  # pragma: no cover
        from enforceai.models.upstream_auth import (  # type: ignore[no-redef]
            UpstreamAuthConfig,
            UpstreamAuthInjection,
        )
        from enforceai.upstream.headers import (  # type: ignore[no-redef]
            ENFORCEAI_ERROR_CODE_HEADER,
            ENFORCEAI_UPSTREAM_API_KEY_HEADER,
            ENFORCEAI_UPSTREAM_API_KEY_HEADER_NAME_HEADER,
            ENFORCEAI_UPSTREAM_AUTHORIZATION_HEADER,
            ENFORCEAI_UPSTREAM_MODE_HEADER,
            MCP_AUTH_TYPE_HEADER,
            MCP_CLAIMS_HEADER,
            MCP_PRINCIPAL_HEADER,
            MCP_PROVIDER_HEADER,
            MCP_SCOPES_HEADER,
        )
        from enforceai.upstream.resolver import (  # type: ignore[no-redef]
            UpstreamInjectionError,
            resolve_upstream_injection,
        )

    server_path = request.headers.get("X-EnforceAI-Server-Path")
    if not server_path and server_name:
        server_path = f"/{server_name}"

    upstream_type = (request.headers.get("X-EnforceAI-Upstream-Auth-Type") or "none").strip()
    upstream_binding = (
        request.headers.get("X-EnforceAI-Upstream-Credential-Binding") or "service"
    ).strip()
    upstream_provider = request.headers.get("X-EnforceAI-Upstream-Provider")
    default_upstream_mode = "none" if upstream_type == "none" else "gateway-managed"
    upstream_mode = (request.headers.get("X-EnforceAI-Upstream-Mode") or default_upstream_mode).strip()
    upstream_mode = upstream_mode or default_upstream_mode
    upstream_header_name = request.headers.get("X-EnforceAI-Upstream-Header-Name")
    upstream_scheme = request.headers.get("X-EnforceAI-Upstream-Scheme")

    injection = None
    if upstream_type in {"api-key", "jwt", "oauth2", "oidc", "provider-oauth"}:
        header_name = upstream_header_name
        scheme = upstream_scheme
        if not header_name:
            if upstream_type == "api-key":
                header_name = "X-API-Key"
                scheme = None
            else:
                header_name = "Authorization"
                scheme = scheme or "Bearer"
        injection = UpstreamAuthInjection(
            header_name=header_name,
            scheme=scheme,
        )

    upstream_auth = UpstreamAuthConfig(
        mode=upstream_mode,
        type=upstream_type,
        provider=upstream_provider,
        credential_binding=upstream_binding,
        injection=injection,
    )

    try:
        oauth_providers = None
        oauth_token_client = None
        oauth_refresh_skew_seconds = 0
        if upstream_auth.type in {"oauth2", "oidc", "provider-oauth"}:
            settings = get_enforceai_settings()
            oauth_providers = settings.upstream_oauth_providers
            oauth_token_client = get_upstream_oauth_token_client()
            oauth_refresh_skew_seconds = settings.upstream_oauth_refresh_skew_seconds

        allow_missing_upstream = False
        if tool_name == "tools/list" and upstream_auth.type in {
            "oauth2",
            "oidc",
            "provider-oauth",
        }:
            allow_missing_upstream = True

        injection_result = await resolve_upstream_injection(
            server_path=server_path,
            upstream_auth=upstream_auth,
            identity=identity,
            stores=get_enforceai_stores(),
            oauth_providers=oauth_providers,
            oauth_token_client=oauth_token_client,
            oauth_refresh_skew_seconds=oauth_refresh_skew_seconds,
            allow_missing_credential=allow_missing_upstream,
        )
    except UpstreamInjectionError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.public_message,
            headers={
                "Connection": "close",
                ENFORCEAI_ERROR_CODE_HEADER: exc.error_code,
            },
        ) from exc

    response_data = {
        "valid": True,
        "username": identity.user_id,
        "client_id": "",
        "scopes": identity.scopes,
        "method": identity.provider,
        "groups": [],
        "server_name": server_name,
        "tool_name": tool_name,
    }

    response = JSONResponse(
        content=response_data,
        status_code=200,
    )
    response.headers["X-User"] = identity.user_id
    response.headers["X-Username"] = identity.user_id
    response.headers["X-Client-Id"] = ""
    response.headers["X-Scopes"] = " ".join(identity.scopes)
    response.headers["X-Auth-Method"] = identity.provider
    response.headers["X-Server-Name"] = server_name or ""
    response.headers["X-Tool-Name"] = tool_name or ""
    response.headers["X-Agent-Id"] = identity.agent_id
    response.headers["X-Allowed-Tools"] = allowed_tools_header_value
    response.headers[MCP_PRINCIPAL_HEADER] = injection_result.mcp_principal
    response.headers[MCP_AUTH_TYPE_HEADER] = injection_result.mcp_auth_type
    response.headers[MCP_SCOPES_HEADER] = injection_result.mcp_scopes
    response.headers[MCP_PROVIDER_HEADER] = injection_result.mcp_provider
    response.headers[MCP_CLAIMS_HEADER] = injection_result.mcp_claims
    response.headers[ENFORCEAI_UPSTREAM_MODE_HEADER] = injection_result.mode
    response.headers[ENFORCEAI_UPSTREAM_AUTHORIZATION_HEADER] = injection_result.upstream_authorization
    response.headers[ENFORCEAI_UPSTREAM_API_KEY_HEADER] = injection_result.upstream_api_key
    response.headers[ENFORCEAI_UPSTREAM_API_KEY_HEADER_NAME_HEADER] = injection_result.upstream_api_key_header
    return response
