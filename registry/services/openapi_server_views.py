"""Projections of OpenAPI MCP servers (issue #2022).

Two directions:
- To callers: response models that never carry a credential, and that drop
  spec and base URLs for anyone who is not an admin.
- To the container: the full server state, with each credential decrypted and
  each spec parsed. Only ``build_hub_state`` decrypts, and its result goes only
  to ``openapi_mcp_client.put_server``.
"""

import logging
from typing import Any

from ..core.config import settings
from ..schemas.openapi_server_models import (
    OpenApiServer,
    OpenApiServerResponse,
    OpenApiTarget,
    OpenApiTargetResponse,
    StoredSpec,
    TargetAuthResponse,
)
from ..utils.credential_encryption import decrypt_credential
from .openapi_errors import OpenApiServerError
from .openapi_spec_checks import parse_spec
from .visibility import should_redact_backend_urls

logger = logging.getLogger(__name__)


def mcp_url_for(
    server: OpenApiServer,
) -> str:
    """The gateway URL agents connect to."""
    base = (settings.registry_url or "").rstrip("/")
    return f"{base}{server.path}/mcp"


def _runtime_for(
    target_name: str,
    runtime: dict[str, Any] | None,
) -> tuple[str, str | None]:
    """(status, error) the container reports for a target; pending when it reports nothing."""
    entry = ((runtime or {}).get("targets") or {}).get(target_name)
    if not isinstance(entry, dict) or entry.get("status") not in ("ok", "error"):
        return "pending", None
    return entry["status"], entry.get("error")


def target_response(
    target: OpenApiTarget,
    user_context: dict | None,
    runtime: dict[str, Any] | None = None,
) -> OpenApiTargetResponse:
    """One target as a caller may see it."""
    redact = should_redact_backend_urls(user_context)
    status, error = _runtime_for(target.name, runtime)
    return OpenApiTargetResponse(
        name=target.name,
        description=target.description,
        spec_url=None if redact else target.spec_url,
        spec_source=target.spec_source,
        base_url=None if redact else target.base_url,
        spec_title=target.spec_title,
        spec_version=target.spec_version,
        openapi_version=target.openapi_version,
        auth=TargetAuthResponse(
            scheme=target.auth.scheme,
            credential_mode=target.auth.credential_mode,
            header_name=target.auth.header_name,
            query_param=target.auth.query_param,
            username=None if redact else target.auth.username,
            credential_set=bool(target.auth.credential_encrypted),
        ),
        tool_names=target.tool_names,
        tool_name_overrides=target.tool_name_overrides,
        description_overrides=target.description_overrides,
        include_tags=target.include_tags,
        include_operations=target.include_operations,
        keep_output_schemas=target.keep_output_schemas,
        status=status,
        error=error,
        updated_at=target.updated_at,
    )


def server_response(
    server: OpenApiServer,
    user_context: dict | None,
    runtime: dict[str, Any] | None = None,
    sync_status: str | None = None,
) -> OpenApiServerResponse:
    """One server as a caller may see it."""
    return OpenApiServerResponse(
        slug=server.slug,
        name=server.name,
        description=server.description,
        path=server.path,
        mcp_url=mcp_url_for(server),
        tags=server.tags,
        visibility=server.visibility,
        allowed_groups=server.allowed_groups,
        generation=server.generation,
        sync_status=sync_status,
        targets=[target_response(t, user_context, runtime) for t in server.targets],
        created_at=server.created_at,
        updated_at=server.updated_at,
    )


def _decrypted(
    target: OpenApiTarget,
) -> str | None:
    if not target.auth.credential_encrypted:
        return None
    plain = decrypt_credential(target.auth.credential_encrypted)
    if plain is None:
        # Fail closed: a credential that no longer decrypts (SECRET_KEY changed)
        # must not silently turn into an unauthenticated target.
        raise OpenApiServerError(
            500,
            "credential_unreadable",
            f"the stored credential of target '{target.name}' cannot be decrypted; "
            "SECRET_KEY may have changed. Rotate the credential with PATCH.",
        )
    return plain


def _hub_target(
    target: OpenApiTarget,
    spec: StoredSpec,
) -> dict[str, Any]:
    return {
        "name": target.name,
        "description": target.description,
        "base_url": target.base_url,
        "spec": parse_spec(spec.content.encode("utf-8")).spec,
        "auth_scheme": target.auth.scheme,
        "auth_credential_mode": target.auth.credential_mode,
        "auth_header_name": target.auth.header_name,
        "auth_query_param": target.auth.query_param,
        "auth_username": target.auth.username,
        "auth_credential": _decrypted(target),
        "tool_name_overrides": target.tool_name_overrides,
        "description_overrides": target.description_overrides,
        "include_tags": target.include_tags,
        "include_operations": target.include_operations,
        "keep_output_schemas": target.keep_output_schemas,
    }


def build_hub_state(
    server: OpenApiServer,
    specs: dict[str, StoredSpec],
) -> dict[str, Any]:
    """The full state the container needs to build this server's hub.

    A target whose spec is missing from storage is left out and logged; the
    container then serves the server's other targets.
    """
    targets = []
    for target in server.targets:
        spec = specs.get(target.spec_sha256)
        if spec is None:
            logger.error(
                f"Spec {target.spec_sha256[:12]} for target '{target.name}' on '{server.slug}' is missing"
            )
            continue
        targets.append(_hub_target(target, spec))
    return {
        "slug": server.slug,
        "name": server.name,
        "description": server.description,
        "generation": server.generation,
        "targets": targets,
    }
