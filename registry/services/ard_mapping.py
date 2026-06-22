"""Pure mapping helpers that convert registry records into ARD catalog entries.

These functions perform NO I/O. They take an already-loaded record (dict for
servers/agents, SkillCard for skills), the publisher FQDN, and a pre-built record
URL, and return an :class:`ArdCatalogEntry` (or ``None`` when the record cannot
produce a valid URN, in which case the caller skips it).

See issue #1294 and ``.scratchpad/issue-1294/lld.md`` for the mapping rules.
"""

import logging
import re
from datetime import datetime
from typing import Any

from ..schemas.ard_models import ArdCatalogEntry

logger = logging.getLogger(__name__)


# A conformant identifier must satisfy BOTH the JSON Schema regex and the
# conformance tool's stricter regex. We validate against both.
_SCHEMA_URN_RE = re.compile(r"^urn:air:[a-zA-Z0-9.-]+(:[a-zA-Z0-9._-]+)+$")
_TOOL_URN_RE = re.compile(r"^urn:air:([a-zA-Z0-9.-]+)(?::([a-zA-Z0-9._:-]+))?:([a-zA-Z0-9._-]+)$")

# IANA media types per entity type. Skill type is a single constant so the
# value can be flipped if the working group prefers the markdown profile
# (see issue #1294 open question Q3).
MEDIA_TYPE_SERVER = "application/mcp-server-card+json"
MEDIA_TYPE_AGENT = "application/a2a-agent-card+json"
MEDIA_TYPE_SKILL = "application/ai-skill"

_MIN_QUERIES = 2
_MAX_QUERIES = 5
_MAX_CAPABILITIES = 50
_MAX_QUERY_LEN = 120


def _sanitize_name(
    path_or_name: str,
) -> str:
    """Reduce a path or name to the URN-safe ``[A-Za-z0-9._-]`` leaf segment.

    Strips any leading ``/servers/`` style prefix by taking the last path
    segment, then replaces every other character with a hyphen.
    """
    leaf = path_or_name.strip("/").split("/")[-1]
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "-", leaf).strip("-")
    return cleaned


def _build_urn(
    publisher: str,
    namespace: str,
    name: str,
) -> str | None:
    """Build a URN and validate it against both regexes.

    Returns ``None`` (and logs a warning) when the result is invalid so the
    caller can skip the record rather than emit a non-conformant entry.
    """
    if not name:
        logger.warning("Skipping record: empty sanitized name for namespace=%s", namespace)
        return None
    urn = f"urn:air:{publisher}:{namespace}:{name}"
    if _SCHEMA_URN_RE.match(urn) and _TOOL_URN_RE.match(urn):
        return urn
    logger.warning("Skipping record: generated URN failed validation: %s", urn)
    return None


def _normalize_timestamp(
    value: Any,
) -> str | None:
    """Normalize a datetime or ISO string to ISO 8601 UTC with a ``Z`` suffix."""
    if value is None:
        return None
    if isinstance(value, datetime):
        text = value.isoformat()
    else:
        text = str(value)
    if not text:
        return None
    # Collapse an explicit +00:00 offset to Z; append Z when no zone is present.
    if text.endswith("+00:00"):
        return text[:-6] + "Z"
    if text.endswith("Z"):
        return text
    if "+" in text[10:] or text[10:].count("-") > 0:
        return text
    return text + "Z"


def _derive_representative_queries(
    tags: list[str],
    description: str | None,
) -> list[str] | None:
    """Return 2-5 representative queries, or ``None`` when fewer than 2 derivable.

    The field is optional, so omitting it (returning ``None``) is preferable to
    emitting a single query, which would violate the schema's ``minItems: 2``.
    """
    queries: list[str] = []
    for tag in (tags or [])[:3]:
        if tag:
            queries.append(f"{tag} tools")
    if len(queries) < _MIN_QUERIES and description:
        first = description.split(".")[0].strip()
        if first:
            queries.append(first[:_MAX_QUERY_LEN])
    deduped = list(dict.fromkeys(q for q in queries if q))
    if len(deduped) < _MIN_QUERIES:
        return None
    return deduped[:_MAX_QUERIES]


def _cap_capabilities(
    names: list[str],
) -> list[str] | None:
    """Dedupe, drop empties, and cap the capabilities list. ``None`` when empty."""
    cleaned = list(dict.fromkeys(n for n in (names or []) if n))
    if not cleaned:
        return None
    if len(cleaned) > _MAX_CAPABILITIES:
        logger.info("Capping capabilities from %d to %d", len(cleaned), _MAX_CAPABILITIES)
    return cleaned[:_MAX_CAPABILITIES]


def map_server(
    path: str,
    record: dict[str, Any],
    publisher: str,
    record_url: str,
    namespace: str = "server",
) -> ArdCatalogEntry | None:
    """Map an MCP server record to an ARD catalog entry."""
    name = _sanitize_name(path)
    urn = _build_urn(publisher, namespace, name)
    if urn is None:
        return None
    tags = record.get("tags") or []
    description = record.get("description")
    tool_names = [t.get("name") for t in (record.get("tool_list") or []) if t.get("name")]
    return ArdCatalogEntry(
        identifier=urn,
        display_name=record.get("server_name") or name,
        type=MEDIA_TYPE_SERVER,
        url=record_url,
        description=description,
        tags=tags or None,
        capabilities=_cap_capabilities(tool_names),
        representative_queries=_derive_representative_queries(tags, description),
        version=record.get("version"),
        updated_at=_normalize_timestamp(record.get("updated_at")),
    )


def map_agent(
    path: str,
    record: dict[str, Any],
    publisher: str,
    record_url: str,
    namespace: str = "agent",
) -> ArdCatalogEntry | None:
    """Map an A2A agent record to an ARD catalog entry."""
    name = _sanitize_name(path)
    urn = _build_urn(publisher, namespace, name)
    if urn is None:
        return None
    tags = record.get("tags") or []
    description = record.get("description")
    skill_names = [s.get("name") for s in (record.get("skills") or []) if s.get("name")]
    return ArdCatalogEntry(
        identifier=urn,
        display_name=record.get("name") or name,
        type=MEDIA_TYPE_AGENT,
        url=record_url,
        description=description,
        tags=tags or None,
        capabilities=_cap_capabilities(skill_names),
        representative_queries=_derive_representative_queries(tags, description),
        version=record.get("version"),
        updated_at=_normalize_timestamp(record.get("updated_at")),
    )


def map_skill(
    path: str,
    name: str,
    description: str | None,
    tags: list[str],
    tool_names: list[str],
    version: str | None,
    updated_at: Any,
    publisher: str,
    record_url: str,
    namespace: str = "skill",
) -> ArdCatalogEntry | None:
    """Map a skill record to an ARD catalog entry.

    Takes already-extracted primitives rather than the SkillCard model so the
    function stays a pure value mapper that is trivial to unit test.
    """
    sanitized = _sanitize_name(path or name)
    urn = _build_urn(publisher, namespace, sanitized)
    if urn is None:
        return None
    return ArdCatalogEntry(
        identifier=urn,
        display_name=name or sanitized,
        type=MEDIA_TYPE_SKILL,
        url=record_url,
        description=description,
        tags=tags or None,
        capabilities=_cap_capabilities(tool_names),
        representative_queries=_derive_representative_queries(tags, description),
        version=version,
        updated_at=_normalize_timestamp(updated_at),
    )
