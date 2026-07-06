"""Shared visibility helpers used by the search route and the
duplicate-check service.

These helpers were previously defined as private functions inside
``registry/api/search_routes.py``. Moving them here lets services
import them without crossing back into the API layer (which would
create a circular import — search_routes imports services).

Each helper returns True iff the caller is allowed to *view* the
entity, given the visibility metadata of that entity and the
caller's auth context (``is_admin``, ``accessible_*`` scope lists,
``username``, ``groups``).
"""

import logging
from typing import Any

from ..core.config import DeploymentMode, settings

logger = logging.getLogger(__name__)

# Internal backend-connection fields that must never reach a non-admin caller in
# with-gateway mode: the raw upstream URL and any explicit endpoint override the
# registry proxies on the user's behalf. Non-admins reach servers through the
# gateway, so exposing these maps the internal network. In registry-only mode
# there is no gateway, so users legitimately need the URL to connect directly
# and these are NOT stripped.
_BACKEND_URL_FIELDS: tuple[str, ...] = (
    "proxy_pass_url",
    "mcp_endpoint",
    "sse_endpoint",
)


def should_redact_backend_urls(
    user_context: dict | None,
) -> bool:
    """Return True when internal backend URLs must be stripped for this caller.

    Single source of truth for the redaction decision enforced by the server
    read endpoints (``GET /servers``, ``GET /servers/{path}``,
    ``GET /servers/{path}/versions``, ``server.json``) and semantic search.

    Fails closed: an admin sees the raw URLs, and in registry-only mode every
    caller needs the URL to connect directly (no gateway to proxy through). In
    every other case — including a missing/empty ``user_context`` — the backend
    URLs are redacted.

    Args:
        user_context: Authenticated user context, or None if auth is absent.

    Returns:
        True if backend URL fields should be removed before returning the
        response; False only for admins or when running in registry-only mode.
    """
    if user_context and user_context.get("is_admin"):
        return False
    if settings.deployment_mode == DeploymentMode.REGISTRY_ONLY:
        return False
    return True


def redact_server_backend_fields(
    server_dict: dict,
) -> dict:
    """Strip internal backend URL fields from a flat server/version dict in place.

    Removes ``proxy_pass_url``, ``mcp_endpoint`` and ``sse_endpoint`` from the
    top-level dict and, if present, from each entry of an embedded ``versions``
    list. Mutates and returns the same dict so callers can chain. This mirrors
    the ``proxy_pass_url`` stripping in ``GET /servers/{path}`` but also covers
    the endpoint-override fields and per-version URLs that other read surfaces
    expose.

    Callers MUST gate this behind :func:`should_redact_backend_urls`; this
    function performs no authorization decision of its own.

    Args:
        server_dict: A server or version dict to redact in place.

    Returns:
        The same dict, with backend URL fields removed.
    """
    for field in _BACKEND_URL_FIELDS:
        server_dict.pop(field, None)

    versions = server_dict.get("versions")
    if isinstance(versions, list):
        for version in versions:
            if isinstance(version, dict):
                for field in _BACKEND_URL_FIELDS:
                    version.pop(field, None)

    return server_dict


async def user_can_access_server(
    path: str,
    server_name: str,
    user_context: dict,
) -> bool:
    """Visibility check for an MCP server.

    Mirrors the legacy ``_user_can_access_server`` from
    ``search_routes.py``. Calls into ``server_service`` for the
    accessible-servers comparison; falls back to string checks if
    that lookup raises.
    """
    # Local import to avoid pulling server_service at module load
    # (server_service imports the repository factory which imports
    # config which imports settings which... etc.).
    from .server_service import server_service

    if user_context.get("is_admin"):
        return True

    accessible_servers = user_context.get("accessible_servers") or []
    if "all" in accessible_servers:
        return True
    if not accessible_servers:
        return False

    try:
        if await server_service.user_can_access_server_path(path, accessible_servers):
            return True
    except Exception:
        logger.debug("Unable to validate server path via service for %s", path, exc_info=True)

    technical_name = path.strip("/")
    return technical_name in accessible_servers or (
        bool(server_name) and server_name in accessible_servers
    )


def user_can_access_server_from_doc(
    path: str,
    server_name: str,
    user_context: dict,
) -> bool:
    """Visibility check for an MCP server whose existence is already confirmed.

    Same access rules as :func:`user_can_access_server` but for callers that
    have already fetched the server (e.g. the ANS status endpoints), so it
    performs the ``accessible_servers`` comparison purely in memory instead of
    re-fetching the server through ``server_service`` just to re-confirm it
    exists. The ``"*"`` wildcard grants access here because the caller has
    proven the server exists by holding its document.
    """
    if user_context.get("is_admin"):
        return True

    accessible_servers = user_context.get("accessible_servers") or []
    if "all" in accessible_servers or "*" in accessible_servers:
        return True
    if not accessible_servers:
        return False

    technical_name = path.strip("/")
    return technical_name in accessible_servers or (
        bool(server_name) and server_name in accessible_servers
    )


async def user_can_access_agent(
    agent_path: str,
    user_context: dict,
) -> bool:
    """Visibility check for an A2A agent (path-based).

    Fetches the agent card from ``agent_service`` to read visibility
    metadata. For callers that already have the agent dict in hand
    (e.g. dedup advisory results), prefer
    :func:`user_can_access_agent_from_doc` to avoid the per-call
    fetch.
    """
    from .agent_service import agent_service

    if user_context.get("is_admin"):
        return True

    accessible_agents = user_context.get("accessible_agents") or []
    if "all" not in accessible_agents and agent_path not in accessible_agents:
        return False

    agent_card = await agent_service.get_agent_info(agent_path)
    if not agent_card:
        return False

    if agent_card.visibility == "public":
        return True
    if agent_card.visibility == "private":
        return agent_card.registered_by == user_context.get("username")
    if agent_card.visibility == "group-restricted":
        allowed_groups = set(agent_card.allowed_groups)
        user_groups = set(user_context.get("groups", []))
        return bool(allowed_groups & user_groups)
    return False


def user_can_access_agent_from_doc(
    document: dict,
    user_context: dict,
) -> bool:
    """Visibility check for an agent using fields already in hand.

    Same logic as :func:`user_can_access_agent` but reads visibility
    / registered_by / allowed_groups from the supplied document
    instead of re-fetching the agent card. Handles both shapes:

    - Repo dump: fields live at the top level.
    - Search hit: fields live nested under ``agent_card``.

    Used by ``DuplicateCheckService`` to avoid an N+1 fetch when
    iterating over advisory candidates.
    """
    if user_context.get("is_admin"):
        return True

    path = str(document.get("path") or "")
    accessible_agents = user_context.get("accessible_agents") or []
    if "all" not in accessible_agents and path not in accessible_agents:
        return False

    agent_card = document.get("agent_card")
    if isinstance(agent_card, dict):
        visibility = str(agent_card.get("visibility") or "")
        registered_by = str(agent_card.get("registered_by") or "")
        allowed_groups = list(agent_card.get("allowed_groups") or [])
    else:
        visibility = str(document.get("visibility") or "")
        registered_by = str(document.get("registered_by") or "")
        allowed_groups = list(document.get("allowed_groups") or [])

    if visibility == "public":
        return True
    if visibility == "private":
        return registered_by == user_context.get("username")
    if visibility == "group-restricted":
        user_groups = set(user_context.get("groups", []))
        return bool(set(allowed_groups) & user_groups)
    return False


async def user_can_access_skill(
    skill_path: str,
    visibility: str,
    owner: str,
    allowed_groups: list[Any],
    user_context: dict,
) -> bool:
    """Visibility check for a skill.

    Takes visibility / owner / allowed_groups as arguments rather
    than looking them up — every caller already has these fields
    from the search hit or repo dump.
    """
    if user_context.get("is_admin"):
        return True
    if visibility == "public":
        return True
    if visibility == "private":
        return owner == user_context.get("username")
    if visibility == "group":
        user_groups = set(user_context.get("groups", []))
        skill_groups = set(allowed_groups or [])
        return bool(user_groups & skill_groups)
    return False


async def user_can_access_custom_entity(
    visibility: str,
    owner: str,
    allowed_groups: list[Any],
    user_context: dict,
) -> bool:
    """Visibility check for a custom entity record.

    Mirrors :func:`user_can_access_agent`'s value set (``group-restricted``,
    not skills' legacy ``group``) so it stays in lockstep with the custom
    entity list/single-record paths (``custom_entity_visibility._user_can_view``).
    Branch order MUST match those so a record can't be visible in one surface
    and 404 in another.
    """
    if user_context.get("is_admin"):
        return True
    if visibility == "public":
        return True
    if visibility == "private":
        return owner == user_context.get("username")
    if visibility == "group-restricted":
        user_groups = set(user_context.get("groups", []) or [])
        return bool(user_groups & set(allowed_groups or []))
    return False  # deny-by-default for any unknown visibility
