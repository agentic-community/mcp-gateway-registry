"""Create, update and delete OpenAPI MCP servers and their REST targets (issue #2022).

The registry database is the only stored state. Every change bumps the
server's generation, pushes the server's full state to the openapi-mcp
container, and refreshes the tool list before the call returns, within
REQUEST_WAIT_BUDGET_SECONDS. When the budget runs out the call reports
``pending`` and the reconciler finishes the work.
"""

import asyncio
import logging
import re
from collections.abc import Callable
from typing import Any

from ..core.config import settings
from ..core.nginx_service import PROXY_ROUTE_PREFIXES
from ..repositories.factory import (
    get_openapi_server_repository,
    get_openapi_spec_repository,
)
from ..schemas.openapi_server_models import (
    OpenApiServer,
    OpenApiServerCreateRequest,
    OpenApiServerUpdateRequest,
    OpenApiTarget,
    OpenApiTargetAuthInput,
    OpenApiTargetCreateRequest,
    OpenApiTargetUpdateRequest,
    StoredSpec,
    StoredTargetAuth,
)
from ..utils.credential_encryption import encrypt_credential
from ..utils.url_guard import openapi_mcp_target_url
from . import (
    openapi_mcp_client,
    openapi_spec_checks,
    server_lifecycle,
)
from .openapi_errors import (
    OpenApiServerError,
    conflict,
    not_found,
)
from .openapi_server_views import build_hub_state
from .server_service import server_service

logger = logging.getLogger(__name__)


# Must stay below nginx's 60 s default proxy_read_timeout on /api/.
REQUEST_WAIT_BUDGET_SECONDS: float = 45.0
SAVE_RETRIES: int = 3
SOURCE_API: str = "/api/openapi-servers"
# egress_oauth.provider on the ServerInfo of a server with per-user targets.
OPENAPI_EGRESS_PROVIDER: str = "openapi-mcp"
MAX_SLUG_CHARS: int = 64
# First path segments an OpenAPI MCP server may not take: nginx's proxy
# namespaces, the registry's own routes, and the container's /s/ prefix.
RESERVED_SLUGS: frozenset[str] = frozenset(
    PROXY_ROUTE_PREFIXES
    | {
        "api",
        "static",
        "assets",
        "oauth2",
        "auth",
        "login",
        "logout",
        "health",
        "healthcheck",
        "metrics",
        "well-known",
        "internal",
        "s",
        "mcp",
        "validate",
        "all",
        "docs",
        "openapi",
        "ws",
    }
)


def _slug_from_name(
    name: str,
) -> str:
    """Lowercase, non-alphanumerics to single hyphens, the same rule virtual servers use."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not slug or len(slug) > MAX_SLUG_CHARS:
        raise OpenApiServerError(
            400,
            "invalid_name",
            "the name must contain letters or digits and make a path of at most 64 characters",
        )
    if slug in RESERVED_SLUGS:
        raise OpenApiServerError(
            400, "reserved_name", f"the name makes the path /{slug}, which is reserved"
        )
    return slug


def _actor(
    user_context: dict,
) -> str:
    return str(user_context.get("username") or "unknown")


def _server_info_entry(
    server: OpenApiServer,
) -> dict[str, Any]:
    """The ServerInfo record that routes, scopes and searches this server like any other."""
    return {
        "server_name": server.name,
        "description": server.description,
        "path": server.path,
        "proxy_pass_url": openapi_mcp_target_url(server.slug),
        "transport": "streamable-http",
        "supported_transports": ["streamable-http"],
        "tags": ["openapi", *[t for t in server.tags if t != "openapi"]],
        "visibility": server.visibility,
        "allowed_groups": server.allowed_groups,
        "metadata": {"openapi_server": {"slug": server.slug}},
        "registered_by": server.created_by,
        **_egress_fields(server),
    }


def _egress_fields(
    server: OpenApiServer,
) -> dict[str, Any]:
    """Route a server with per-user targets through the auth server's mcp-proxy hop.

    The existing ``pat`` mode routes it there and asks the registry vend for a
    credential; the vend sees ``metadata.openapi_server`` and returns the signed
    per-user envelope instead of a single PAT.
    """
    if any(t.auth.credential_mode == "per_user" for t in server.targets):
        return {"egress_auth_mode": "pat", "egress_oauth": {"provider": OPENAPI_EGRESS_PROVIDER}}
    return {"egress_auth_mode": "none", "egress_oauth": None}


def _require_egress_for(
    auth: OpenApiTargetAuthInput | None,
) -> None:
    """Per-user targets keep each user's key in the egress vault, so they need egress auth on."""
    if auth is not None and auth.credential_mode == "per_user" and not settings.egress_auth_enabled:
        raise OpenApiServerError(
            409,
            "egress_auth_disabled",
            "credential_mode 'per_user' stores each user's key in the egress vault; set "
            "EGRESS_AUTH_ENABLED=true and configure SECRET_STORE_BACKEND first",
        )


async def _sync_server_info(
    server: OpenApiServer,
) -> None:
    """Keep the ServerInfo's egress routing in step with the targets' credential modes."""
    existing = await server_service.get_server_info(server.path) or {}
    wanted = _egress_fields(server)
    if existing.get("egress_auth_mode", "none") == wanted["egress_auth_mode"]:
        return
    await server_service.update_server(server.path, {**existing, **_server_info_entry(server)})
    await server_lifecycle.flush_nginx()
    logger.info(f"'{server.slug}' egress routing set to {wanted['egress_auth_mode']}")


async def _load_specs(
    servers: list[OpenApiServer],
) -> dict[str, StoredSpec]:
    shas = {t.spec_sha256 for s in servers for t in s.targets}
    return await get_openapi_spec_repository().get_many(shas)


def _count_push(
    result: str,
    trigger: str,
) -> None:
    try:
        from ..observability.meters import openapi_mcp_sync_push_total

        openapi_mcp_sync_push_total.add(1, {"result": result, "trigger": trigger})
    except Exception:  # noqa: BLE001 - metrics are optional
        logger.debug("openapi_mcp push counter unavailable", exc_info=True)


async def push_server(
    server: OpenApiServer,
    trigger: str = "mutation",
) -> dict[str, Any]:
    """Push one server's full state to the container. Returns the container's reply."""
    specs = await _load_specs([server])
    try:
        reply = await openapi_mcp_client.put_server(build_hub_state(server, specs))
    except OpenApiServerError as e:
        _count_push(e.code, trigger)
        raise
    _count_push("ok", trigger)
    return reply


async def _push_and_refresh(
    server: OpenApiServer,
    rescan: bool,
) -> str:
    """Push, then refresh tools, within the request budget. Returns applied or pending."""

    async def _work() -> None:
        await push_server(server)
        await server_lifecycle.refresh_managed_server(server.path, rescan=rescan)

    try:
        await asyncio.wait_for(_work(), timeout=REQUEST_WAIT_BUDGET_SECONDS)
    except TimeoutError:
        logger.warning(
            f"Sync of '{server.slug}' generation {server.generation} passed the budget; reconciler will finish"
        )
        return "pending"
    except OpenApiServerError as e:
        logger.warning(f"Sync of '{server.slug}' failed ({e.code}); reconciler will retry")
        return "pending"
    return "applied"


async def _save_with_retry(
    slug: str,
    change: Callable[[OpenApiServer], None],
) -> OpenApiServer:
    """Read, change, bump generation and write, retrying when another writer won."""
    repo = get_openapi_server_repository()
    for _attempt in range(SAVE_RETRIES):
        server = await repo.get(slug)
        if server is None:
            raise not_found(f"OpenAPI MCP server '{slug}'")
        expected = server.generation
        change(server)
        server.generation = expected + 1
        server.updated_at = openapi_spec_checks.now_utc()
        if await repo.replace(server, expected):
            return server
    raise conflict("concurrent_update", f"'{slug}' changed while saving; try again")


async def get_server_or_404(
    slug: str,
) -> OpenApiServer:
    server = await get_openapi_server_repository().get(slug)
    if server is None:
        raise not_found(f"OpenAPI MCP server '{slug}'")
    return server


async def _rollback_create(
    server: OpenApiServer,
    actor: str,
) -> None:
    """Undo a failed create in reverse order. Every step tolerates a missing target."""
    try:
        info = await server_service.get_server_info(server.path)
        if info:
            await server_lifecycle.remove_server_everywhere(server.path, info, actor)
    except Exception:
        logger.exception(f"Rollback of '{server.slug}': removing the ServerInfo failed")
    try:
        await openapi_mcp_client.delete_server(server.slug)
    except Exception:
        logger.warning(
            f"Rollback of '{server.slug}': deleting the hub failed; the reconciler will remove it"
        )
    try:
        await get_openapi_server_repository().delete(server.slug)
    except Exception:
        logger.exception(f"Rollback of '{server.slug}': deleting the stored record failed")


async def create_server(
    request: OpenApiServerCreateRequest,
    user_context: dict,
    raw_headers: list[tuple[bytes, bytes]],
) -> tuple[OpenApiServer, str]:
    """Create an empty server and its gateway route. Returns the server and its sync status."""
    slug = _slug_from_name(request.name)
    repo = get_openapi_server_repository()
    if await repo.count() >= settings.openapi_mcp_max_servers:
        raise conflict(
            "limit_reached",
            f"OPENAPI_MCP_MAX_SERVERS ({settings.openapi_mcp_max_servers}) reached",
            {"limit": settings.openapi_mcp_max_servers},
        )
    if await server_service.get_server_info(f"/{slug}"):
        raise conflict("duplicate_name", f"a server already uses the path /{slug}")
    now = openapi_spec_checks.now_utc()
    server = OpenApiServer(
        slug=slug,
        name=request.name,
        description=request.description,
        path=f"/{slug}",
        tags=request.tags,
        visibility=request.visibility,
        allowed_groups=request.allowed_groups,
        generation=1,
        created_by=_actor(user_context),
        created_at=now,
        updated_at=now,
    )
    if not await repo.create(server):
        raise conflict("duplicate_name", f"an OpenAPI MCP server already uses the path /{slug}")
    try:
        await push_server(server)
        await server_lifecycle.register_and_enable_managed_server(
            _server_info_entry(server), _actor(user_context), SOURCE_API, raw_headers
        )
    except Exception:
        logger.exception(f"Creating OpenAPI MCP server '{slug}' failed; rolling back")
        await _rollback_create(server, _actor(user_context))
        raise
    logger.info(f"OpenAPI MCP server '{slug}' created by {_actor(user_context)}")
    return server, "applied"


async def update_server(
    slug: str,
    request: OpenApiServerUpdateRequest,
) -> tuple[OpenApiServer, str]:
    """Update description, tags, visibility or groups, on the record and its ServerInfo."""
    updates = request.model_dump(exclude_none=True)

    def _change(server: OpenApiServer) -> None:
        for key, value in updates.items():
            setattr(server, key, value)

    server = await _save_with_retry(slug, _change)
    existing = await server_service.get_server_info(server.path) or {}
    await server_service.update_server(server.path, {**existing, **_server_info_entry(server)})
    return server, await _push_and_refresh(server, rescan=False)


async def _delete_unused_specs(
    shas: set[str],
) -> None:
    for sha in shas:
        if await get_openapi_server_repository().count_spec_references(sha) == 0:
            await get_openapi_spec_repository().delete(sha)


async def delete_server(
    slug: str,
    user_context: dict,
) -> None:
    """Delete the hub, the ServerInfo and everything derived from it, the record, and unused specs."""
    server = await get_server_or_404(slug)
    try:
        await openapi_mcp_client.delete_server(slug)
    except OpenApiServerError:
        logger.warning(f"Could not delete hub '{slug}' now; the reconciler will remove it")
    info = await server_service.get_server_info(server.path)
    if info:
        await server_lifecycle.remove_server_everywhere(server.path, info, _actor(user_context))
    await get_openapi_server_repository().delete(slug)
    await _delete_unused_specs({t.spec_sha256 for t in server.targets})
    logger.info(f"OpenAPI MCP server '{slug}' deleted by {_actor(user_context)}")


def _stored_auth(
    auth: OpenApiTargetAuthInput,
) -> StoredTargetAuth:
    """Encrypt the credential for storage. The plaintext goes no further."""
    encrypted = encrypt_credential(auth.credential.get_secret_value()) if auth.credential else None
    return StoredTargetAuth(
        scheme=auth.scheme,
        credential_mode=auth.credential_mode,
        header_name=auth.header_name,
        query_param=auth.query_param,
        username=auth.username,
        credential_encrypted=encrypted,
        credential_updated_at=openapi_spec_checks.now_utc() if encrypted else None,
    )


async def _admit_spec(
    spec_url: str,
    target_name: str,
    options: dict[str, Any],
) -> tuple[openapi_spec_checks.FetchedSpec, dict[str, Any]]:
    """Fetch, parse and preview a spec. Raises on any failure; stores nothing."""
    raw, content_type = await openapi_spec_checks.fetch_spec_bytes(
        spec_url, settings.openapi_mcp_spec_max_bytes
    )
    fetched = openapi_spec_checks.parse_spec(raw, content_type)
    preview = await openapi_mcp_client.preview(
        {"spec": fetched.spec, "target_name": target_name, **options}
    )
    openapi_spec_checks.check_tool_count(
        preview["tools"], preview.get("tags", {}), settings.openapi_mcp_max_tools_per_target
    )
    openapi_spec_checks.check_tool_names(preview["tools"])
    return fetched, preview


async def _store_spec(
    fetched: openapi_spec_checks.FetchedSpec,
) -> None:
    await get_openapi_spec_repository().upsert(
        StoredSpec(
            sha256=fetched.sha256,
            content=fetched.raw.decode("utf-8"),
            size_bytes=len(fetched.raw),
            content_type=fetched.content_type[:100],
            fetched_at=openapi_spec_checks.now_utc(),
        )
    )


def _tool_options(
    source: OpenApiTargetCreateRequest | OpenApiTarget,
) -> dict[str, Any]:
    return {
        "tool_name_overrides": source.tool_name_overrides,
        "description_overrides": source.description_overrides,
        "include_tags": source.include_tags,
        "include_operations": source.include_operations,
        "keep_output_schemas": source.keep_output_schemas,
    }


def _check_target_slot(
    server: OpenApiServer,
    name: str,
) -> None:
    if any(t.name == name for t in server.targets):
        raise conflict("duplicate_name", f"'{server.slug}' already has a target named '{name}'")
    if len(server.targets) >= settings.openapi_mcp_max_targets_per_server:
        raise conflict(
            "limit_reached",
            f"OPENAPI_MCP_MAX_TARGETS_PER_SERVER ({settings.openapi_mcp_max_targets_per_server}) reached",
            {"limit": settings.openapi_mcp_max_targets_per_server},
        )


async def add_target(
    slug: str,
    request: OpenApiTargetCreateRequest,
    user_context: dict,
) -> tuple[OpenApiServer, OpenApiTarget, list[str], str]:
    """Add a REST target. Fails closed: if any check fails, nothing is stored."""
    server = await get_server_or_404(slug)
    _check_target_slot(server, request.name)
    _require_egress_for(request.auth)
    openapi_spec_checks.validate_target_url(str(request.base_url), "base_url")
    openapi_spec_checks.validate_target_url(str(request.spec_url), "spec_url")
    fetched, preview = await _admit_spec(
        str(request.spec_url), request.name, _tool_options(request)
    )
    await _store_spec(fetched)
    now = openapi_spec_checks.now_utc()
    target = OpenApiTarget(
        name=request.name,
        description=request.description,
        spec_url=str(request.spec_url),
        spec_sha256=fetched.sha256,
        spec_title=fetched.title,
        spec_version=fetched.version,
        openapi_version=fetched.openapi_version,
        base_url=str(request.base_url).rstrip("/"),
        auth=_stored_auth(request.auth),
        tool_names=preview["tools"],
        created_by=_actor(user_context),
        created_at=now,
        updated_at=now,
        **_tool_options(request),
    )

    def _change(stored: OpenApiServer) -> None:
        _check_target_slot(stored, request.name)
        stored.targets.append(target)

    updated = await _save_with_retry(slug, _change)
    await _sync_server_info(updated)
    logger.info(
        f"Target '{request.name}' ({len(target.tool_names)} tools, spec {fetched.sha256[:12]}) added to '{slug}' by {_actor(user_context)}"
    )
    return (
        updated,
        target,
        preview.get("warnings", []),
        await _push_and_refresh(updated, rescan=True),
    )


def get_target_or_404(
    server: OpenApiServer,
    name: str,
) -> OpenApiTarget:
    for target in server.targets:
        if target.name == name:
            return target
    raise not_found(f"target '{name}' on '{server.slug}'")


def _apply_target_update(
    target: OpenApiTarget,
    request: OpenApiTargetUpdateRequest,
) -> None:
    fields = request.model_dump(exclude_none=True, exclude={"auth", "refresh_spec", "base_url"})
    for key, value in fields.items():
        setattr(target, key, value)
    if request.base_url is not None:
        target.base_url = str(request.base_url).rstrip("/")
    if request.auth is not None:
        target.auth = _stored_auth(request.auth)
    target.updated_at = openapi_spec_checks.now_utc()


async def update_target(
    slug: str,
    name: str,
    request: OpenApiTargetUpdateRequest,
    user_context: dict,
) -> tuple[OpenApiServer, OpenApiTarget, list[str], str]:
    """Update a target. Re-runs every admission check before anything is stored."""
    server = await get_server_or_404(slug)
    _require_egress_for(request.auth)
    candidate = get_target_or_404(server, name).model_copy(deep=True)
    old_sha = candidate.spec_sha256
    _apply_target_update(candidate, request)
    openapi_spec_checks.validate_target_url(candidate.base_url, "base_url")
    if request.refresh_spec:
        openapi_spec_checks.validate_target_url(candidate.spec_url, "spec_url")
        fetched, preview = await _admit_spec(candidate.spec_url, name, _tool_options(candidate))
        await _store_spec(fetched)
        candidate.spec_sha256, candidate.spec_title = fetched.sha256, fetched.title
        candidate.spec_version, candidate.openapi_version = fetched.version, fetched.openapi_version
    else:
        spec = (await get_openapi_spec_repository().get_many({old_sha})).get(old_sha)
        if spec is None:
            raise OpenApiServerError(
                409, "spec_missing", "the stored spec is missing; send refresh_spec: true"
            )
        parsed = openapi_spec_checks.parse_spec(spec.content.encode("utf-8"))
        preview = await openapi_mcp_client.preview(
            {"spec": parsed.spec, "target_name": name, **_tool_options(candidate)}
        )
        openapi_spec_checks.check_tool_count(
            preview["tools"], preview.get("tags", {}), settings.openapi_mcp_max_tools_per_target
        )
        openapi_spec_checks.check_tool_names(preview["tools"])
    candidate.tool_names = preview["tools"]

    def _change(stored: OpenApiServer) -> None:
        stored.targets = [candidate if t.name == name else t for t in stored.targets]

    updated = await _save_with_retry(slug, _change)
    await _sync_server_info(updated)
    if candidate.spec_sha256 != old_sha:
        await _delete_unused_specs({old_sha})
    logger.info(f"Target '{name}' on '{slug}' updated by {_actor(user_context)}")
    return (
        updated,
        candidate,
        preview.get("warnings", []),
        await _push_and_refresh(updated, rescan=True),
    )


async def delete_target(
    slug: str,
    name: str,
    user_context: dict,
) -> str:
    """Remove a target and its tools. Returns the sync status."""
    server = await get_server_or_404(slug)
    target = get_target_or_404(server, name)

    def _change(stored: OpenApiServer) -> None:
        stored.targets = [t for t in stored.targets if t.name != name]

    updated = await _save_with_retry(slug, _change)
    await _sync_server_info(updated)
    await _delete_unused_specs({target.spec_sha256})
    logger.info(f"Target '{name}' removed from '{slug}' by {_actor(user_context)}")
    return await _push_and_refresh(updated, rescan=False)
