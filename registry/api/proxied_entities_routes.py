"""Admin read-only listing of proxied non-MCP entities for the scope editor.

The IAM scope editor needs to offer proxied skills/agents/custom entities as
selectable targets so an admin can author a per-verb scope for them. This
endpoint returns exactly those entities, reusing the same ``list_proxied()``
repository queries and ``resolve_proxy_target`` gating the nginx render uses —
so the editor's list matches what will actually route (federated / disabled /
auto-disabled / targetless rows are excluded identically).

MCP servers are intentionally NOT included: they are authored against their
existing ``/path`` scope key, unchanged by this feature. Virtual servers are
alias-only and likewise excluded.

Read-only + admin-gated. Not flag-gated: an admin may pre-author scopes before
GATEWAY_GENERIC_PROXY_ENABLED is flipped (the entities simply won't route until
then), and listing issues the same indexed query regardless.
"""

import logging
from typing import Annotated, Any, Protocol

from fastapi import APIRouter, Depends, HTTPException

from registry.auth.dependencies import nginx_proxied_auth
from registry.repositories.factory import (
    get_agent_repository,
    get_custom_entity_repository,
    get_skill_repository,
)
from registry.schemas.proxy_mixin import resolve_proxy_target

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/iam/proxied-entities", tags=["IAM Proxied Entities"])


class _ProxiedRepository(Protocol):
    """Repository capability required by the scope-editor projection."""

    async def list_proxied(self) -> list[dict[str, Any]]: ...


def _proxyable_repos() -> tuple[tuple[str, _ProxiedRepository], ...]:
    """Resolve (default_type, repo) at CALL time.

    Calling the getters here (rather than capturing them in a module-level tuple
    at import) means the same three proxyable non-MCP repos the render feed uses
    are returned, and tests can patch the getters on this module. MCP servers keep
    their legacy /path scope key; virtual servers are alias-only.
    """
    return (
        ("a2a_agent", get_agent_repository()),
        ("skill", get_skill_repository()),
        ("custom", get_custom_entity_repository()),
    )


def _require_admin(
    user_context: dict | None,
) -> None:
    """Enforce admin permission or raise 401/403 (mirrors iam_user_groups_routes)."""
    if not user_context:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not user_context.get("is_admin"):
        raise HTTPException(status_code=403, detail="Administrator permissions are required")


@router.get("", summary="List proxied non-MCP entities for the scope editor")
async def list_proxied_entities(
    user_context: Annotated[dict | None, Depends(nginx_proxied_auth)] = None,
) -> dict:
    """Return proxied skills/agents/custom entities that resolve to a live target.

    Each item: ``{entity_type, path, name, authz_key}`` where ``authz_key`` is the
    canonical ``{entity_type}/{registered_path}`` string the scope editor must
    write (NOT the bare slash-stripped path — that would 403). ``name`` falls back
    to the path (the render-path projections don't carry a display name).
    """
    _require_admin(user_context)

    items: list[dict] = []
    for default_type, repo in _proxyable_repos():
        try:
            rows = await repo.list_proxied()
        except Exception as e:  # noqa: BLE001 - one repo failing must not 500 the editor
            logger.error("list_proxied failed for %s: %s", default_type, e, exc_info=True)
            continue
        for doc in rows:
            # Custom records carry their own type token; use it so the authz key
            # matches the record's actual type (mirrors the render feed).
            entity_type = doc.get("entity_type") or default_type
            if not resolve_proxy_target(entity_type, doc):
                continue  # federated / disabled / auto-disabled / targetless -> not offered
            path = doc.get("path", "")
            registered = path.strip("/")
            items.append(
                {
                    "entity_type": entity_type,
                    "path": path,
                    "name": doc.get("name") or path,
                    # The canonical authz key the scope must be written against.
                    "authz_key": f"{entity_type}/{registered}",
                }
            )

    items.sort(key=lambda i: (i["entity_type"], i["path"]))
    return {"proxied_entities": items, "count": len(items)}
