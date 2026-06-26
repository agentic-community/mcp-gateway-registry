"""Pure helpers to map ingested ARD catalog entries back to internal records.

This is the reverse of ``ard_mapping`` (which maps internal records -> ARD
catalog entries for the Publisher). Ingestion needs to take an
:class:`ArdCatalogEntry` from an external ``ai-catalog.json`` and produce an
internal record dict that the search engine can index.

The mapping is lossy by design: an ARD entry carries only discovery metadata
(name, description, tags, url, version), not transport/credential details. That
is sufficient for *search*, and ingested records are explicitly marked
read-only and non-connectable (``record_kind="ard_ingested"``) so they are never
proxied as live servers. The original entry JSON is retained under
``ard_source_entry`` for fidelity/debugging.

No I/O. See issue #1296 and ``.scratchpad/issue-1296/lld.md``.
"""

from __future__ import annotations

import logging

from ..schemas.ard_models import ArdCatalogEntry
from . import ard_mapping

logger = logging.getLogger(__name__)

# ARD media type -> internal entity kind. application/ai-registry+json and
# application/ai-catalog+json are handled elsewhere (referral hint / recursion).
_KIND_BY_MEDIA_TYPE: dict[str, str] = {
    ard_mapping.MEDIA_TYPE_SERVER: "server",
    ard_mapping.MEDIA_TYPE_AGENT: "agent",
    ard_mapping.MEDIA_TYPE_SKILL: "skill",
}


def parse_urn_publisher(
    identifier: str,
) -> str | None:
    """Return the ``<publisher>`` FQDN segment from a ``urn:air:<publisher>:...``.

    Uses the same strict URN regex as the Publisher so trust extraction and URN
    emission cannot drift. Returns ``None`` for a malformed/empty identifier.
    """
    match = ard_mapping._TOOL_URN_RE.match(identifier or "")
    return match.group(1) if match else None


def entry_to_record(
    entry: ArdCatalogEntry,
    source_id: str,
) -> tuple[str, str, dict] | None:
    """Map an ARD entry to ``(kind, path, record)`` for indexing.

    Args:
        entry: The catalog entry from an ingested ai-catalog.json.
        source_id: Stable id of the ingestion source; becomes the record's
            ``registry_name`` and the ``/{source_id}/{leaf}`` path prefix.

    Returns:
        ``(kind, path, record)`` where ``kind`` is server/agent/skill, or
        ``None`` for unsupported entry types (registry/catalog) or an
        unmappable identifier.
    """
    kind = _KIND_BY_MEDIA_TYPE.get(entry.type)
    if kind is None:
        return None
    leaf = ard_mapping._sanitize_name(entry.identifier.split(":")[-1])
    if not leaf:
        logger.warning("Skipping ingested entry with unmappable identifier: %s", entry.identifier)
        return None
    # Un-prefixed path; the peer-sync storage layer prepends "/{source_id}" so the
    # stored path becomes "/{source_id}/{leaf}" and sync_metadata.original_path is
    # "/{leaf}" (used for orphan detection). source_id is still recorded here via
    # registry_name for belt-and-suspenders origin attribution.
    path = f"/{leaf}"
    name_key = "server_name" if kind == "server" else "name"
    # Tag with origin markers so the UI can classify these as external/federated
    # ("federated"), as ARD discovery-only imports ("ard"), and group them by
    # source ("{source_id}"). Mirrors entry_to_skill_data.
    tags = list(entry.tags or [])
    for marker in ("federated", "ard", source_id):
        if marker not in tags:
            tags.append(marker)
    record: dict = {
        name_key: entry.display_name,
        "description": entry.description,
        "tags": tags,
        "version": entry.version,
        "updated_at": entry.updated_at,
        "registry_name": source_id,
        "is_enabled": True,
        "visibility": "public",
        "is_read_only": True,
        "record_kind": "ard_ingested",
        # The ORIGINAL entry's url/identifier on the source registry. This is the
        # "resolve" link: a client dereferences ard_source_url to fetch the full
        # artifact descriptor (server.json / agent card) and connect at the source.
        "ard_source_url": entry.url,
        "ard_source_identifier": entry.identifier,
        "ard_source_entry": entry.model_dump(by_alias=True, exclude_none=True),
    }
    if kind == "agent":
        # The A2A agent-card model requires url + a non-null version (plus
        # capabilities/skills/modes which have defaults). A sparse ARD discovery
        # entry has none of these, so registration silently dropped every agent.
        # Fill the minimum so the discovery record stores and is searchable; it
        # stays read-only/non-connectable. The real card lives at ard_source_url.
        record["url"] = entry.url or f"https://{source_id}/agent/{leaf}"
        record["version"] = entry.version or "1.0.0"
        record["capabilities"] = {}
        record["skills"] = []
    return kind, path, record


def entry_to_skill_data(
    entry: ArdCatalogEntry,
    source_id: str,
) -> dict | None:
    """Build a SkillCard-compatible dict for an ingested ARD skill entry.

    Skills are stored directly via the skill repository (peer-sync storage only
    covers servers/agents), so this produces the full record shape:
    ``path`` is ``/{source_id}/{leaf}`` (origin-prefixed for search attribution),
    ``name`` is source-qualified (the skill ``name`` index is global/unique), and
    ``skill_md_url`` reuses the entry's ``url`` (required HttpUrl). Returns ``None``
    when the entry is not a skill or lacks a dereferenceable ``url``.
    """
    if _KIND_BY_MEDIA_TYPE.get(entry.type) != "skill":
        return None
    if not entry.url:
        logger.warning("Skipping ingested skill without url: %s", entry.identifier)
        return None
    leaf = ard_mapping._sanitize_name(entry.identifier.split(":")[-1])
    if not leaf:
        return None
    tags = list(entry.tags or [])
    for marker in ("federated", "ard", source_id):
        if marker not in tags:
            tags.append(marker)
    return {
        # SkillCard requires a /skills/ path prefix; the source_id segment after it
        # carries origin (matched by ard_search_service._origin_id, which scans
        # path segments) and registry_name is the secondary origin signal.
        "path": f"/skills/{source_id}/{leaf}",
        "name": f"{source_id}-{leaf}",
        "description": entry.description or entry.display_name or "Ingested ARD skill",
        "skill_md_url": entry.url,
        "version": entry.version,
        "tags": tags,
        "visibility": "public",
        "is_enabled": True,
        "is_read_only": True,
        "registry_name": source_id,
    }
