"""Public, anonymous per-record endpoints for ARD catalog dereferencing.

The ARD catalog (``/.well-known/ai-catalog.json``) is crawled anonymously, so
each entry's ``url`` must be fetchable without auth. The existing per-record
endpoints (``/api/servers/{path}/server.json`` etc.) all require auth, so this
module adds parallel public endpoints under ``/api/public`` that:

1. Return ONLY public + enabled records.
2. Return 404 for anything not public+enabled (private/group/disabled/missing),
   so existence of non-public records is never disclosed.
3. Strip internal fields (backend URLs, auth schemes, group lists, owner) from
   the response body.

See issue #1294 (Blocker B1, SEC-1) and ``.scratchpad/issue-1294/lld.md``.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Path

from ..repositories.factory import (
    get_agent_repository,
    get_server_repository,
    get_skill_repository,
)
from ..services.ard_mapping import _sanitize_name
from ..services.canonical_export import redact_backend_urls, to_canonical

logger = logging.getLogger(__name__)

router = APIRouter()

# Fields that must never appear in an anonymous agent record body.
_AGENT_SENSITIVE_FIELDS = frozenset(
    {
        "security",
        "security_schemes",
        "allowed_groups",
        "registered_by",
        "auth_credential_encrypted",
        "sync_metadata",
        "ans_metadata",
        "_identity_url_normalized",
    }
)

# Fields that must never appear in an anonymous skill record body.
_SKILL_SENSITIVE_FIELDS = frozenset(
    {
        "auth_credential_encrypted",
        "allowed_groups",
        "owner",
    }
)


def _strip_fields(
    record: dict[str, Any],
    sensitive: frozenset[str],
) -> dict[str, Any]:
    """Return a shallow copy with sensitive keys removed."""
    return {k: v for k, v in record.items() if k not in sensitive}


@router.get("/public/servers/{leaf:path}/server.json")
async def get_public_server(
    leaf: str = Path(..., description="Sanitized server name from the catalog URN"),
) -> dict[str, Any]:
    """Public canonical server.json for a public + enabled MCP server.

    Matches by the same sanitized leaf used in the catalog entry's URN, so the
    URL resolves regardless of how the server path is stored. Only public +
    enabled servers are considered; anything else returns 404.
    """
    target = leaf.strip("/")
    repo = get_server_repository()
    records = await repo.find_with_filter(
        {"is_enabled": True, "visibility": "public"}, limit=None
    )
    for path, record in records.items():
        if _sanitize_name(path) == target:
            canonical, _ = to_canonical({**record, "path": path})
            # Anonymous callers never receive backend URLs.
            return redact_backend_urls(canonical)
    raise HTTPException(status_code=404, detail="Server not found")


@router.get("/public/agents/{leaf:path}")
async def get_public_agent(
    leaf: str = Path(..., description="Sanitized agent name from the catalog URN"),
) -> dict[str, Any]:
    """Public agent card for a public + enabled A2A agent."""
    target = leaf.strip("/")
    repo = get_agent_repository()
    records = await repo.find_with_filter(
        {"is_enabled": True, "visibility": "public"}, limit=None
    )
    for path, record in records.items():
        if _sanitize_name(path) == target:
            return _strip_fields(record, _AGENT_SENSITIVE_FIELDS)
    raise HTTPException(status_code=404, detail="Agent not found")


@router.get("/public/skills/{leaf:path}")
async def get_public_skill(
    leaf: str = Path(..., description="Sanitized skill name from the catalog URN"),
) -> dict[str, Any]:
    """Public skill card for a public + enabled local skill."""
    target = leaf.strip("/")
    repo = get_skill_repository()
    skills = await repo.list_filtered(
        include_disabled=False,
        visibility="public",
        registry_name="local",
    )
    for skill in skills:
        if _sanitize_name(skill.path) == target:
            body = skill.model_dump(mode="json")
            return _strip_fields(body, _SKILL_SENSITIVE_FIELDS)
    raise HTTPException(status_code=404, detail="Skill not found")
