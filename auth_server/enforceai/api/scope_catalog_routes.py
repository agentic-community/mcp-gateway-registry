from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)
from pydantic import (
    BaseModel,
    ConfigDict,
)

from ..auth.dependency import (
    get_enforceai_settings,
)
from ..config import (
    EnforceAISettings,
)
from ..errors import (
    DependencyUnavailableError,
)

logger = logging.getLogger(__name__)

router = APIRouter(
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


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


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


@router.get(
    "/scopes/catalog",
    response_model=ScopeCatalogResponse,
)
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
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to load scope catalog: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Scope catalog unavailable",
        ) from exc

    scopes_response: dict[str, ScopeDefinitionResponse] = {}
    for scope_name, scope_def in catalog.scopes.items():
        server_perms: list[ServerPermissionResponse] = []
        for server_permission in scope_def.server_permissions:
            tools_resp = None
            if server_permission.tools is not None:
                tools_resp = ToolPolicyResponse(
                    all_tools=server_permission.tools.all_tools,
                    tools=list(server_permission.tools.tools),
                )
            server_perms.append(
                ServerPermissionResponse(
                    server=server_permission.server,
                    methods=MethodPolicyResponse(
                        all_methods=server_permission.methods.all_methods,
                        methods=list(server_permission.methods.methods),
                    ),
                    tools=tools_resp,
                )
            )

        agent_perms: list[AgentPermissionResponse] = []
        for agent_permission in scope_def.agent_permissions:
            agent_perms.append(
                AgentPermissionResponse(
                    action=agent_permission.action,
                    resources=list(agent_permission.resources),
                )
            )

        scopes_response[scope_name] = ScopeDefinitionResponse(
            name=scope_def.name,
            server_permissions=server_perms,
            agent_permissions=agent_perms,
        )

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

