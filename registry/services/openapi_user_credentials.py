"""Per-user credentials for OpenAPI MCP server targets (issue #2022).

Each user brings their own API key or personal access token for a target in
``per_user`` mode. It is stored in the egress vault exactly like a ``vault-pat``
PAT: write-only, keyed on the user's immutable id, with a bounded lifetime. It is
bound to the target's canonical base URL, so a key never follows a repointed
target to another host.

At call time the registry's egress vend packs the caller's keys for one server
into a signed envelope (``mint_credential_envelope``) for the auth server's
mcp-proxy hop to forward to the container.
"""

import logging
from datetime import (
    UTC,
    datetime,
    timedelta,
)

from ..auth.openapi_mcp_sync import mint_credential_envelope
from ..core.config import settings
from ..egress_auth.schemas import StoredToken
from ..exceptions import UrlValidationError
from ..schemas.openapi_server_models import (
    PER_USER_PROVIDER_PREFIX,
    OpenApiServer,
    OpenApiTarget,
)
from ..utils.url_guard import normalize_url_identity
from .openapi_errors import (
    OpenApiServerError,
    conflict,
)

logger = logging.getLogger(__name__)


def provider_for(
    target_name: str,
) -> str:
    """Vault provider segment of a target's per-user keys."""
    return f"{PER_USER_PROVIDER_PREFIX}{target_name}"


def bound_upstream(
    target: OpenApiTarget,
) -> str:
    """The canonical destination a target's per-user keys are bound to."""
    try:
        return normalize_url_identity(target.base_url).rstrip("/")
    except UrlValidationError as e:
        raise OpenApiServerError(
            500, "invalid_base_url", f"target '{target.name}' has an invalid base_url"
        ) from e


def require_per_user(
    target: OpenApiTarget,
) -> None:
    if target.auth.credential_mode != "per_user":
        raise conflict(
            "not_per_user",
            f"target '{target.name}' uses a shared credential; only an admin can change it",
        )


def _egress_service():
    from ..egress_auth.factory import get_egress_auth_service

    return get_egress_auth_service()


async def store_key(
    server: OpenApiServer,
    target: OpenApiTarget,
    principal,
    secret: str,
    ttl_seconds: int,
) -> StoredToken:
    """Store a user's key for one per-user target."""
    require_per_user(target)
    now = datetime.now(UTC)
    token = StoredToken(
        access_token=secret,
        token_type="Bearer",  # nosec B106 - token type label, not a credential
        created_at=now.isoformat(),
        expires_at=(now + timedelta(seconds=ttl_seconds)).isoformat(),
        bound_upstreams=[bound_upstream(target)],
    )
    await _egress_service().set_pat(
        principal.auth_method, principal.user_id, provider_for(target.name), server.path, token
    )
    logger.info(
        f"Stored a per-user key for {server.slug}/{target.name} (expires {token.expires_at})"
    )
    return token


async def key_status(
    server: OpenApiServer,
    target: OpenApiTarget,
    principal,
) -> StoredToken | None:
    """The stored entry for presence and expiry. Callers must never return its key."""
    return await _egress_service().get_pat_status(
        principal.auth_method, principal.user_id, provider_for(target.name), server.path
    )


async def delete_key(
    server: OpenApiServer,
    target: OpenApiTarget,
    principal,
) -> None:
    await _egress_service().delete_pat(
        principal.auth_method, principal.user_id, provider_for(target.name), server.path
    )


async def _key_for(
    server: OpenApiServer,
    target: OpenApiTarget,
    auth_method: str,
    user_id: str,
) -> str | None:
    """The caller's live key for a target, or None (missing, expired, or bound elsewhere)."""
    return await _egress_service().get_pat(
        auth_method=auth_method,
        user_id=user_id,
        provider=provider_for(target.name),
        server_path=server.path,
        requested_upstream=bound_upstream(target),
    )


async def build_envelope(
    server: OpenApiServer,
    auth_method: str,
    user_id: str,
) -> str:
    """Sign the caller's keys for every per-user target of one server.

    Targets the caller has no live key for are left out; the container then
    answers those tools with "add your key". The envelope is minted even when
    empty, so the server's shared targets keep working for that caller.
    """
    credentials: dict[str, dict[str, str]] = {}
    for target in server.targets:
        if target.auth.credential_mode != "per_user":
            continue
        key = await _key_for(server, target, auth_method, user_id)
        if key is not None:
            credentials[target.name] = {"key": key, "base_url": bound_upstream(target)}
    return mint_credential_envelope(settings.openapi_mcp_sync_secret, server.slug, credentials)
