"""ARD Registry adapter service (issue #1295, Phase 2).

Wraps the existing semantic-search engine and the catalog asset repositories in
the ARD Registry contract, reusing the Phase 1 ``ard_mapping`` helpers and the
``visibility`` access-scoping helpers. Pure logic only: no FastAPI/HTTP and no
metrics here (those live in ``registry/api/ard_routes.py``).

Access-scoping uses the *no-fetch* visibility helpers (the search hit / repo doc
already carries the visibility fields), so neither search nor browse issues a
per-item database read.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
from typing import Any

from ..repositories.factory import (
    get_agent_repository,
    get_search_repository,
    get_server_repository,
    get_skill_repository,
)
from ..schemas.ard_models import ArdCatalogEntry, ArdSearchResult
from . import ard_mapping
from .ard_service import (
    _namespace_for,
    _public_record_url,
    _resolve_publisher_domain,
)
from .visibility import (
    user_can_access_agent_from_doc,
    user_can_access_server,
    user_can_access_skill,
)

logger = logging.getLogger(__name__)

# ARD filter ``type`` values (media string OR short name) -> engine entity_type.
_TYPE_TO_ENTITY: dict[str, str] = {
    "mcp_server": "mcp_server",
    "server": "mcp_server",
    ard_mapping.MEDIA_TYPE_SERVER: "mcp_server",
    "a2a_agent": "a2a_agent",
    "agent": "a2a_agent",
    ard_mapping.MEDIA_TYPE_AGENT: "a2a_agent",
    "skill": "skill",
    ard_mapping.MEDIA_TYPE_SKILL: "skill",
}

_ALLOWED_ORDER_BY = {"identifier", "displayName", "updatedAt"}


class ArdValidationError(ValueError):
    """Raised on a malformed ARD request (bad filter key, bad pageToken, etc.).

    The route layer maps this to a 400 ``INVALID_REQUEST`` ARD error.
    """


# ---------------------------------------------------------------------------
# Pagination cursor (opaque base64 offset)
# ---------------------------------------------------------------------------


def encode_page_token(offset: int) -> str:
    return base64.urlsafe_b64encode(json.dumps({"offset": offset}).encode()).decode()


def decode_page_token(token: str | None) -> int:
    if not token:
        return 0
    try:
        offset = int(json.loads(base64.urlsafe_b64decode(token.encode()))["offset"])
    except (binascii.Error, ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
        raise ArdValidationError("Invalid pageToken") from e
    if offset < 0:
        raise ArdValidationError("Invalid pageToken")
    return offset


# ---------------------------------------------------------------------------
# ARD query.filter -> engine (entity_types, tags)
# ---------------------------------------------------------------------------


def filter_to_engine(
    ard_filter: dict[str, str | list[str]] | None,
) -> tuple[list[str] | None, list[str] | None]:
    """Map an ARD ``query.filter`` to (entity_types, tags).

    Values OR within a key, AND across keys. Supported keys: ``type`` /
    ``entity_type`` and ``tags``. Unknown keys raise ``ArdValidationError``.
    """
    if not ard_filter:
        return None, None
    entity_types: list[str] | None = None
    tags: list[str] | None = None
    for key, raw in ard_filter.items():
        values = raw if isinstance(raw, list) else [raw]
        if key in ("type", "entity_type"):
            mapped: list[str] = []
            for v in values:
                ent = _TYPE_TO_ENTITY.get(v)
                if ent is None:
                    raise ArdValidationError(f"Unsupported filter type value: {v!r}")
                mapped.append(ent)
            entity_types = mapped
        elif key == "tags":
            tags = [str(v) for v in values]
        else:
            raise ArdValidationError(f"Unsupported filter key: {key!r}")
    return entity_types, tags


def _has_all_tags(record_tags: Any, wanted: list[str] | None) -> bool:
    if not wanted:
        return True
    have = {str(t).lower() for t in (record_tags or [])}
    return all(str(w).lower() in have for w in wanted)


def _to_result(entry: ArdCatalogEntry, relevance: float, source: str) -> ArdSearchResult:
    score = max(0, min(100, round((relevance or 0.0) * 100)))
    return ArdSearchResult(
        **entry.model_dump(by_alias=True, exclude_none=True),
        score=score,
        source=source,
    )


# ---------------------------------------------------------------------------
# POST /search
# ---------------------------------------------------------------------------


async def search_and_scope(
    text: str,
    entity_types: list[str] | None,
    tags: list[str] | None,
    window: int,
    user_context: dict,
    source_uri: str,
) -> tuple[list[ArdSearchResult], int]:
    """Run semantic search, access-scope each hit (no per-item DB read), map to
    ARD results, rescale score to 0-100. Returns (ordered results, scoped_out).

    Ordering is deterministic (score desc, then identifier asc) so the opaque
    offset pageToken is stable across pages.
    """
    publisher = _resolve_publisher_domain()
    base_url = source_uri.rsplit("/api/ard", 1)[0]
    search_repo = get_search_repository()
    raw = await search_repo.search(
        query=text,
        entity_types=entity_types,
        max_results=min(max(window, 1), 50),
    )

    out: list[ArdSearchResult] = []
    scoped_out = 0

    for hit in raw.get("servers", []):
        path = hit.get("path", "")
        if not _has_all_tags(hit.get("tags"), tags):
            continue
        if not await user_can_access_server(path, hit.get("server_name", ""), user_context):
            scoped_out += 1
            continue
        entry = ard_mapping.map_server(
            path, hit, publisher, _public_record_url(base_url, "servers", path),
            _namespace_for("server"),
        )
        if entry:
            out.append(_to_result(entry, hit.get("relevance_score", 0.0), source_uri))

    for hit in raw.get("agents", []):
        path = hit.get("path", "")
        card = hit.get("agent_card") or {}
        if not _has_all_tags(card.get("tags"), tags):
            continue
        if not user_can_access_agent_from_doc({"path": path, "agent_card": card}, user_context):
            scoped_out += 1
            continue
        entry = ard_mapping.map_agent(
            path, card, publisher, _public_record_url(base_url, "agents", path),
            _namespace_for("agent"),
        )
        if entry:
            out.append(_to_result(entry, hit.get("relevance_score", 0.0), source_uri))

    for hit in raw.get("skills", []):
        path = hit.get("path", "")
        if not _has_all_tags(hit.get("tags"), tags):
            continue
        allowed = await user_can_access_skill(
            path, hit.get("visibility", ""), hit.get("owner", ""),
            hit.get("allowed_groups", []) or [], user_context,
        )
        if not allowed:
            scoped_out += 1
            continue
        entry = ard_mapping.map_skill(
            path, hit.get("skill_name", ""), hit.get("description"), hit.get("tags") or [],
            [], hit.get("version"), hit.get("last_checked_time"), publisher,
            _public_record_url(base_url, "skills", path), _namespace_for("skill"),
        )
        if entry:
            out.append(_to_result(entry, hit.get("relevance_score", 0.0), source_uri))

    out.sort(key=lambda r: (-r.score, r.identifier))
    return out, scoped_out


# ---------------------------------------------------------------------------
# GET /agents (browse) — all asset types, access-scoped, paginated
# ---------------------------------------------------------------------------


async def browse(
    filter_pairs: list[str],
    order_by: str,
    offset: int,
    limit: int,
    user_context: dict,
    base_url: str,
) -> tuple[list[ArdCatalogEntry], int]:
    """Browse all catalog asset types (servers + agents + skills), access-scoped.

    Bulk-loads each type once (mirrors the Phase 1 publisher), filters in memory
    with the no-fetch visibility helpers, applies the ARD filter, maps to ARD
    entries, sorts deterministically, returns (page, total).
    """
    if order_by not in _ALLOWED_ORDER_BY:
        raise ArdValidationError(f"Unsupported orderBy: {order_by!r}")
    entity_types, tags = filter_to_engine(_parse_filter_pairs(filter_pairs))
    want = set(entity_types) if entity_types else None  # engine entity names
    publisher = _resolve_publisher_domain()

    entries: list[ArdCatalogEntry] = []

    if want is None or "mcp_server" in want:
        servers = await get_server_repository().find_with_filter({"is_enabled": True}, limit=None)
        for path, doc in servers.items():
            if not _has_all_tags(doc.get("tags"), tags):
                continue
            if not await user_can_access_server(path, doc.get("server_name", ""), user_context):
                continue
            e = ard_mapping.map_server(
                path, doc, publisher, _public_record_url(base_url, "servers", path),
                _namespace_for("server"),
            )
            if e:
                entries.append(e)

    if want is None or "a2a_agent" in want:
        agents = await get_agent_repository().find_with_filter({"is_enabled": True}, limit=None)
        for path, doc in agents.items():
            if not _has_all_tags(doc.get("tags"), tags):
                continue
            if not user_can_access_agent_from_doc({**doc, "path": path}, user_context):
                continue
            e = ard_mapping.map_agent(
                path, doc, publisher, _public_record_url(base_url, "agents", path),
                _namespace_for("agent"),
            )
            if e:
                entries.append(e)

    if want is None or "skill" in want:
        skills = await get_skill_repository().list_filtered(include_disabled=False)
        for skill in skills:
            allowed = await user_can_access_skill(
                skill.path, getattr(skill, "visibility", "") or "",
                getattr(skill, "owner", "") or "",
                list(getattr(skill, "allowed_groups", []) or []), user_context,
            )
            if not allowed:
                continue
            if not _has_all_tags(getattr(skill, "tags", []), tags):
                continue
            tool_names = [t.get("name") for t in (getattr(skill, "tools", []) or []) if t.get("name")]
            e = ard_mapping.map_skill(
                skill.path, getattr(skill, "skill_name", "") or skill.path,
                getattr(skill, "description", None), list(getattr(skill, "tags", []) or []),
                tool_names, getattr(skill, "version", None), getattr(skill, "updated_at", None),
                publisher, _public_record_url(base_url, "skills", skill.path),
                _namespace_for("skill"),
            )
            if e:
                entries.append(e)

    entries.sort(key=lambda e: (_order_key(e, order_by), e.identifier))
    total = len(entries)
    return entries[offset : offset + limit], total


def _order_key(entry: ArdCatalogEntry, order_by: str) -> str:
    if order_by == "displayName":
        return (entry.display_name or "").lower()
    if order_by == "updatedAt":
        return entry.updated_at or ""
    return entry.identifier


def _parse_filter_pairs(pairs: list[str]) -> dict[str, str | list[str]]:
    """Parse repeated ``key=value`` query params into an ARD-style filter dict.

    Repeated keys accumulate into a list (OR within the key).
    """
    out: dict[str, str | list[str]] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ArdValidationError(f"Invalid filter (expected key=value): {pair!r}")
        key, _, value = pair.partition("=")
        key, value = key.strip(), value.strip()
        if key in out:
            existing = out[key]
            out[key] = [*existing, value] if isinstance(existing, list) else [existing, value]
        else:
            out[key] = value
    return out
