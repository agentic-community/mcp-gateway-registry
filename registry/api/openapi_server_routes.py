"""Routes for OpenAPI MCP servers: /api/openapi-servers (issue #2022).

Registered only when OPENAPI_MCP_ENABLED is on. Every write needs an
administrator and a CSRF token: a target turns into outbound calls that carry a
stored credential, so ``can_modify_servers`` is not enough. Reads return 404,
not 403, to a caller who cannot see the server.
"""

import logging
import time
from collections.abc import Awaitable
from typing import (
    Annotated,
    Any,
)

from fastapi import (
    APIRouter,
    Depends,
    Path,
    Request,
)
from fastapi.responses import (
    JSONResponse,
    Response,
)

from ..audit.context import set_audit_action
from ..auth.csrf import verify_csrf_token_flexible
from ..auth.dependencies import nginx_proxied_auth
from ..core.config import settings
from ..schemas.openapi_server_models import (
    SLUG_PATTERN,
    TARGET_NAME_PATTERN,
    AddTargetResponse,
    MyCredentialRequest,
    MyCredentialStatus,
    OpenApiServer,
    OpenApiServerCreateRequest,
    OpenApiServerUpdateRequest,
    OpenApiTarget,
    OpenApiTargetCreateRequest,
    OpenApiTargetUpdateRequest,
)
from ..services import (
    openapi_mcp_client,
    openapi_server_service,
    openapi_user_credentials,
)
from ..services.openapi_errors import (
    OpenApiServerError,
    not_found,
)
from ..services.openapi_server_views import (
    server_response,
    target_response,
)
from ..services.visibility import user_can_access_server

# The same admin check the virtual server routes use: an explicit admin signal
# only, never a per-server execute scope.
from .virtual_server_routes import _require_admin

logger = logging.getLogger(__name__)

router = APIRouter()

SlugParam = Annotated[str, Path(pattern=SLUG_PATTERN, max_length=64)]
TargetNameParam = Annotated[str, Path(pattern=TARGET_NAME_PATTERN)]
UserContext = Annotated[dict, Depends(nginx_proxied_auth)]
Csrf = Annotated[None, Depends(verify_csrf_token_flexible)]


def _error(
    e: OpenApiServerError,
) -> JSONResponse:
    if e.status_code >= 500:
        logger.error(f"OpenAPI MCP request failed: {e.code}")
    else:
        logger.warning(f"OpenAPI MCP request rejected: {e.code}: {e.detail}")
    return JSONResponse(status_code=e.status_code, content=e.to_body())


async def _measure(
    entity: str,
    operation: str,
    call: Awaitable[Any],
) -> Any:
    """Await a service call and record its outcome and latency."""
    start = time.perf_counter()
    result = "ok"
    try:
        return await call
    except OpenApiServerError as e:
        result = e.code
        raise
    except Exception:
        result = "internal_error"
        raise
    finally:
        _record(entity, operation, result, (time.perf_counter() - start) * 1000)


def _record(
    entity: str,
    operation: str,
    result: str,
    duration_ms: float,
) -> None:
    try:
        from ..observability.meters import (
            openapi_mcp_operation_duration_ms,
            openapi_mcp_operation_total,
        )

        openapi_mcp_operation_total.add(
            1, {"entity": entity, "operation": operation, "result": result}
        )
        openapi_mcp_operation_duration_ms.record(
            duration_ms, {"entity": entity, "operation": operation}
        )
    except Exception:  # noqa: BLE001 - metrics are optional
        logger.debug("openapi_mcp operation metrics unavailable", exc_info=True)


async def _can_read(
    server: OpenApiServer,
    user_context: dict,
) -> bool:
    """Admins see everything; others need the visibility and a scope that reaches the server."""
    if user_context.get("is_admin"):
        return True
    if server.visibility == "private":
        return False
    if server.visibility == "group-restricted":
        groups = set(user_context.get("groups") or [])
        if not groups & set(server.allowed_groups):
            return False
    return await user_can_access_server(server.path, server.name, user_context)


async def _runtime() -> dict:
    """Per-server status from the container, or empty when it does not answer."""
    try:
        return (await openapi_mcp_client.status()).get("servers", {})
    except OpenApiServerError:
        return {}


def _add_target_body(
    server: OpenApiServer,
    target: OpenApiTarget,
    user_context: dict,
    warnings: list[str],
    sync_status: str,
) -> dict:
    base = target_response(target, user_context).model_dump(mode="json")
    body = AddTargetResponse(
        **base,
        warnings=warnings,
        findings=[],
        generation=server.generation,
        sync_status=sync_status,
    )
    return body.model_dump(mode="json")


@router.post("/openapi-servers", status_code=201)
async def create_openapi_server(
    request: Request,
    body: OpenApiServerCreateRequest,
    user_context: UserContext,
    _csrf: Csrf = None,
):
    """Create an empty OpenAPI MCP server and its gateway route. The name becomes the path."""
    _require_admin(user_context)
    set_audit_action(
        request,
        "create",
        "openapi_server",
        resource_id=body.name,
        description="Create OpenAPI MCP server",
    )
    try:
        server, sync_status = await _measure(
            "server",
            "create",
            openapi_server_service.create_server(
                body, user_context, request.scope.get("headers", [])
            ),
        )
    except OpenApiServerError as e:
        return _error(e)
    content = server_response(server, user_context, sync_status=sync_status).model_dump(mode="json")
    return JSONResponse(status_code=201, content=content)


@router.get("/openapi-servers")
async def list_openapi_servers(
    user_context: UserContext,
):
    """List the OpenAPI MCP servers the caller can see."""
    from ..repositories.factory import get_openapi_server_repository

    runtime = await _runtime()
    servers = await get_openapi_server_repository().list_all()
    visible = [s for s in servers if await _can_read(s, user_context)]
    return {
        "servers": [
            server_response(s, user_context, runtime.get(s.slug)).model_dump(mode="json")
            for s in visible
        ]
    }


@router.get("/openapi-servers/{slug}")
async def get_openapi_server(
    slug: SlugParam,
    user_context: UserContext,
):
    """One server with its targets and the container's status for each target."""
    try:
        server = await openapi_server_service.get_server_or_404(slug)
        if not await _can_read(server, user_context):
            raise not_found(f"OpenAPI MCP server '{slug}'")
    except OpenApiServerError as e:
        return _error(e)
    runtime = (await _runtime()).get(slug)
    return server_response(server, user_context, runtime).model_dump(mode="json")


@router.patch("/openapi-servers/{slug}")
async def update_openapi_server(
    request: Request,
    slug: SlugParam,
    body: OpenApiServerUpdateRequest,
    user_context: UserContext,
    _csrf: Csrf = None,
):
    """Update description, tags, visibility or allowed groups."""
    _require_admin(user_context)
    set_audit_action(
        request,
        "update",
        "openapi_server",
        resource_id=slug,
        description="Update OpenAPI MCP server",
    )
    try:
        server, sync_status = await _measure(
            "server", "update", openapi_server_service.update_server(slug, body)
        )
    except OpenApiServerError as e:
        return _error(e)
    return server_response(server, user_context, sync_status=sync_status).model_dump(mode="json")


@router.delete("/openapi-servers/{slug}", status_code=204)
async def delete_openapi_server(
    request: Request,
    slug: SlugParam,
    user_context: UserContext,
    _csrf: Csrf = None,
):
    """Delete a server, its path, its tools and any spec no other target uses."""
    _require_admin(user_context)
    set_audit_action(
        request,
        "delete",
        "openapi_server",
        resource_id=slug,
        description="Delete OpenAPI MCP server",
    )
    try:
        await _measure("server", "delete", openapi_server_service.delete_server(slug, user_context))
    except OpenApiServerError as e:
        return _error(e)
    return Response(status_code=204)


@router.post("/openapi-servers/{slug}/targets", status_code=201)
async def add_openapi_target(
    request: Request,
    slug: SlugParam,
    body: OpenApiTargetCreateRequest,
    user_context: UserContext,
    _csrf: Csrf = None,
):
    """Add a REST target. Fails closed: if any check fails, nothing is stored."""
    _require_admin(user_context)
    set_audit_action(
        request,
        "create",
        "openapi_target",
        resource_id=f"{slug}/{body.name}",
        description="Add OpenAPI target",
    )
    try:
        server, target, warnings, sync_status = await _measure(
            "target", "create", openapi_server_service.add_target(slug, body, user_context)
        )
    except OpenApiServerError as e:
        return _error(e)
    return JSONResponse(
        status_code=201,
        content=_add_target_body(server, target, user_context, warnings, sync_status),
    )


@router.get("/openapi-servers/{slug}/targets/{name}")
async def get_openapi_target(
    slug: SlugParam,
    name: TargetNameParam,
    user_context: UserContext,
):
    """One target. The credential appears only as credential_set."""
    try:
        server = await openapi_server_service.get_server_or_404(slug)
        if not await _can_read(server, user_context):
            raise not_found(f"OpenAPI MCP server '{slug}'")
        target = openapi_server_service.get_target_or_404(server, name)
    except OpenApiServerError as e:
        return _error(e)
    runtime = (await _runtime()).get(slug)
    return target_response(target, user_context, runtime).model_dump(mode="json")


@router.patch("/openapi-servers/{slug}/targets/{name}")
async def update_openapi_target(
    request: Request,
    slug: SlugParam,
    name: TargetNameParam,
    body: OpenApiTargetUpdateRequest,
    user_context: UserContext,
    _csrf: Csrf = None,
):
    """Update a target, rotate its credential, or refresh its spec."""
    _require_admin(user_context)
    set_audit_action(
        request,
        "update",
        "openapi_target",
        resource_id=f"{slug}/{name}",
        description="Update OpenAPI target",
    )
    try:
        server, target, warnings, sync_status = await _measure(
            "target", "update", openapi_server_service.update_target(slug, name, body, user_context)
        )
    except OpenApiServerError as e:
        return _error(e)
    return _add_target_body(server, target, user_context, warnings, sync_status)


@router.delete("/openapi-servers/{slug}/targets/{name}", status_code=204)
async def delete_openapi_target(
    request: Request,
    slug: SlugParam,
    name: TargetNameParam,
    user_context: UserContext,
    _csrf: Csrf = None,
):
    """Remove a target and its tools."""
    _require_admin(user_context)
    set_audit_action(
        request,
        "delete",
        "openapi_target",
        resource_id=f"{slug}/{name}",
        description="Remove OpenAPI target",
    )
    try:
        await _measure(
            "target", "delete", openapi_server_service.delete_target(slug, name, user_context)
        )
    except OpenApiServerError as e:
        return _error(e)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Per-user keys: each user brings their own key for a per-user target.
# ---------------------------------------------------------------------------


def _principal_for(
    user_context: dict,
    sub: str | None = None,
    auth_method: str | None = None,
):
    """The caller's vault principal, or an admin's on-behalf target (same rules as egress-pat)."""
    from .egress_auth_routes import _resolve_target_principal

    if not settings.egress_auth_enabled:
        raise OpenApiServerError(
            409, "egress_auth_disabled", "per-user keys need EGRESS_AUTH_ENABLED=true"
        )
    principal = _resolve_target_principal(sub, auth_method, user_context)
    if principal is None:
        raise OpenApiServerError(
            403, "no_user_identity", "this caller has no per-user identity to store a key under"
        )
    return principal


async def _readable_target(
    slug: str,
    name: str,
    user_context: dict,
) -> tuple[OpenApiServer, OpenApiTarget]:
    server = await openapi_server_service.get_server_or_404(slug)
    if not await _can_read(server, user_context):
        raise not_found(f"OpenAPI MCP server '{slug}'")
    return server, openapi_server_service.get_target_or_404(server, name)


def _status_body(
    target_name: str,
    token,
) -> dict:
    return MyCredentialStatus(
        target=target_name,
        configured=token is not None,
        expires_at=token.expires_at if token else None,
    ).model_dump()


@router.put("/openapi-servers/{slug}/targets/{name}/my-credential")
async def put_my_credential(
    request: Request,
    slug: SlugParam,
    name: TargetNameParam,
    body: MyCredentialRequest,
    user_context: UserContext,
    _csrf: Csrf = None,
):
    """Store your own key for a per-user target (an admin may store one for another user)."""
    from .egress_auth_routes import _resolve_pat_ttl_seconds

    set_audit_action(
        request,
        "update",
        "openapi_user_credential",
        resource_id=f"{slug}/{name}",
        description="Store per-user key",
    )
    try:
        server, target = await _readable_target(slug, name, user_context)
        principal = _principal_for(user_context, body.sub, body.auth_method)
        try:
            ttl_seconds = _resolve_pat_ttl_seconds(body.ttl_value, body.ttl_unit)
        except ValueError as e:
            raise OpenApiServerError(400, "invalid_ttl", str(e)) from e
        token = await openapi_user_credentials.store_key(
            server, target, principal, body.secret.get_secret_value(), ttl_seconds
        )
    except OpenApiServerError as e:
        return _error(e)
    return _status_body(name, token)


@router.get("/openapi-servers/{slug}/targets/{name}/my-credential")
async def get_my_credential(
    slug: SlugParam,
    name: TargetNameParam,
    user_context: UserContext,
):
    """Whether you have a key for this target, and when it expires. Never the key."""
    try:
        server, target = await _readable_target(slug, name, user_context)
        openapi_user_credentials.require_per_user(target)
        token = await openapi_user_credentials.key_status(
            server, target, _principal_for(user_context)
        )
    except OpenApiServerError as e:
        return _error(e)
    return _status_body(name, token)


@router.delete("/openapi-servers/{slug}/targets/{name}/my-credential", status_code=204)
async def delete_my_credential(
    request: Request,
    slug: SlugParam,
    name: TargetNameParam,
    user_context: UserContext,
    _csrf: Csrf = None,
):
    """Delete your key for this target."""
    set_audit_action(
        request,
        "delete",
        "openapi_user_credential",
        resource_id=f"{slug}/{name}",
        description="Delete per-user key",
    )
    try:
        server, target = await _readable_target(slug, name, user_context)
        openapi_user_credentials.require_per_user(target)
        await openapi_user_credentials.delete_key(server, target, _principal_for(user_context))
    except OpenApiServerError as e:
        return _error(e)
    return Response(status_code=204)


@router.get("/openapi-servers/{slug}/my-credentials")
async def list_my_credentials(
    slug: SlugParam,
    user_context: UserContext,
):
    """Your key status for every per-user target on this server."""
    try:
        server = await openapi_server_service.get_server_or_404(slug)
        if not await _can_read(server, user_context):
            raise not_found(f"OpenAPI MCP server '{slug}'")
        principal = _principal_for(user_context)
        statuses = []
        for target in server.targets:
            if target.auth.credential_mode == "per_user":
                token = await openapi_user_credentials.key_status(server, target, principal)
                statuses.append(_status_body(target.name, token))
    except OpenApiServerError as e:
        return _error(e)
    return {"slug": slug, "targets": statuses}
