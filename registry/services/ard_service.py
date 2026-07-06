"""ARD catalog assembly service (the only I/O layer for the publisher).

``build_catalog`` performs exactly three bulk reads (one per entity type) over
public + enabled records and maps them to an :class:`AICatalogManifest`. The
"public" predicate is strict and uniform: ``visibility == "public"``. A null or
unset visibility is treated as NOT public (fail-closed), which matters for MCP
servers where the field is frequently unset (see issue #1294, BE-1).

See ``.scratchpad/issue-1294/lld.md`` for the full design.
"""

import logging
import time
from urllib.parse import urlparse

from fastapi import Request

from ..core.config import settings
from ..repositories.factory import (
    get_agent_repository,
    get_server_repository,
    get_skill_repository,
)
from ..schemas.ard_models import (
    AICatalogManifest,
    ArdCatalogEntry,
    ArdHost,
    ArdTrustManifest,
)
from . import ard_mapping

logger = logging.getLogger(__name__)

_FALLBACK_DOMAIN = "example.com"  # RFC 2606 placeholder, never localhost
_DEFAULT_HOST_DISPLAY_NAME = "MCP Gateway Registry"


def _is_local_origin(
    record: dict,
) -> bool:
    """Return True iff a record is locally owned (not synced from a peer/ingested).

    The Publisher catalog must advertise only this registry's own assets, never
    re-publish another registry's entries. Synced/ingested items are tagged by
    ``sync_metadata.is_federated`` (peer sync) and/or ``registry_name != "local"``
    (issue #1296). Skills are already filtered to ``registry_name="local"`` at the
    query level; this guards servers and agents.
    """
    sync_metadata = record.get("sync_metadata") or {}
    if sync_metadata.get("is_federated"):
        return False
    return (record.get("registry_name") or "local") == "local"


def _resolve_publisher_domain() -> str:
    """Resolve the URN publisher FQDN.

    Priority: explicit ``ard_publisher_domain`` -> host of ``registry_url`` ->
    RFC 2606 placeholder. ``localhost`` is rejected as a publisher per the ARD
    URN naming guide.
    """
    explicit = (settings.ard_publisher_domain or "").strip()
    if explicit:
        return explicit
    parsed = urlparse(settings.registry_url or "")
    host = (parsed.hostname or "").strip()
    if host and host != "localhost":
        return host
    return _FALLBACK_DOMAIN


def _base_url_from_request(
    request: Request,
) -> str:
    """Build the public base URL (scheme://host) from the incoming request."""
    host = request.headers.get("host", "localhost:7860")
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    return f"{proto}://{host}"


def _public_record_url(
    base_url: str,
    kind: str,
    path: str,
) -> str:
    """Build the anonymous per-record URL an ARD client dereferences.

    ``kind`` is one of servers/agents/skills. The URL leaf is the same sanitized
    token used in the entry's URN, so the public endpoint can resolve it by
    matching the leaf regardless of how the record's path is stored (servers,
    agents, and skills use different path conventions). Servers use the canonical
    ``/server.json`` suffix to mirror the existing authed endpoint shape.
    """
    leaf = ard_mapping._sanitize_name(path)
    if kind == "servers":
        return f"{base_url}/api/public/servers/{leaf}/server.json"
    return f"{base_url}/api/public/{kind}/{leaf}"


def _namespace_for(
    entity_type: str,
) -> str:
    """Return the URN namespace segment for an entity type."""
    override = (settings.ard_catalog_default_namespace or "").strip()
    return override or entity_type


def _build_host(
    publisher: str,
) -> ArdHost:
    """Build the catalog host block from registry config.

    We intentionally omit ``host.identifier`` (a ``did:web``) in Phase 1.
    Advertising ``did:web:<publisher>`` is a promise that
    ``https://<publisher>/.well-known/did.json`` resolves to our public keys,
    but Phase 1 defers signing and publishes no key material, so the DID would
    be unresolvable. Identity is instead asserted via the ``https`` trust
    manifest (proven by the existing TLS certificate). The ``did:web`` is added
    back together with the catalog signing follow-up (issue #1294, Q on trust).
    """
    display_name = settings.registry_name or _DEFAULT_HOST_DISPLAY_NAME
    issuer = settings.registry_url or f"https://{publisher}"
    return ArdHost(
        display_name=display_name,
        trust_manifest=ArdTrustManifest(identity=issuer, identity_type="https"),
    )


def _build_registry_self_entry(
    publisher: str,
    base_url: str,
) -> ArdCatalogEntry | None:
    """Build the self-referential application/ai-registry+json catalog entry that
    points crawlers at this registry's ARD search endpoint (issue #1295)."""
    urn = ard_mapping._build_urn(publisher, "registry", "self")
    if urn is None:
        return None
    return ArdCatalogEntry(
        identifier=urn,
        display_name=settings.registry_name or _DEFAULT_HOST_DISPLAY_NAME,
        type=ard_mapping.MEDIA_TYPE_REGISTRY,
        url=f"{base_url}/api/ard",
        description="ARD Registry search API for this registry (POST /search, GET /agents).",
    )


async def _load_server_entries(
    publisher: str,
    base_url: str,
) -> tuple[list[ArdCatalogEntry], int]:
    """Load public + enabled servers and map them. Returns (entries, skipped)."""
    repo = get_server_repository()
    records = await repo.find_with_filter({"is_enabled": True, "visibility": "public"}, limit=None)
    namespace = _namespace_for("server")
    entries: list[ArdCatalogEntry] = []
    skipped = 0
    for path, record in records.items():
        if not _is_local_origin(record):
            continue  # never re-publish synced/ingested items (issue #1296)
        url = _public_record_url(base_url, "servers", path)
        entry = ard_mapping.map_server(path, record, publisher, url, namespace)
        if entry is None:
            skipped += 1
            continue
        entries.append(entry)
    return entries, skipped


async def _load_agent_entries(
    publisher: str,
    base_url: str,
) -> tuple[list[ArdCatalogEntry], int]:
    """Load public + enabled agents and map them. Returns (entries, skipped)."""
    repo = get_agent_repository()
    records = await repo.find_with_filter({"is_enabled": True, "visibility": "public"}, limit=None)
    namespace = _namespace_for("agent")
    entries: list[ArdCatalogEntry] = []
    skipped = 0
    for path, record in records.items():
        if not _is_local_origin(record):
            continue  # never re-publish synced/ingested items (issue #1296)
        url = _public_record_url(base_url, "agents", path)
        entry = ard_mapping.map_agent(path, record, publisher, url, namespace)
        if entry is None:
            skipped += 1
            continue
        entries.append(entry)
    return entries, skipped


async def _load_skill_entries(
    publisher: str,
    base_url: str,
) -> tuple[list[ArdCatalogEntry], int]:
    """Load public + enabled local skills and map them. Returns (entries, skipped)."""
    repo = get_skill_repository()
    skills = await repo.list_filtered(
        include_disabled=False,
        visibility="public",
        registry_name="local",
    )
    namespace = _namespace_for("skill")
    entries: list[ArdCatalogEntry] = []
    skipped = 0
    for skill in skills:
        tool_names = [
            tool.tool_name
            for tool in (getattr(skill, "allowed_tools", None) or [])
            if getattr(tool, "tool_name", None)
        ]
        url = _public_record_url(base_url, "skills", skill.path)
        entry = ard_mapping.map_skill(
            path=skill.path,
            name=skill.name,
            description=skill.description,
            tags=list(skill.tags or []),
            tool_names=tool_names,
            version=getattr(skill, "version", None),
            updated_at=getattr(skill, "updated_at", None),
            publisher=publisher,
            record_url=url,
            namespace=namespace,
        )
        if entry is None:
            skipped += 1
            continue
        entries.append(entry)
    return entries, skipped


async def build_catalog(
    request: Request,
) -> AICatalogManifest:
    """Build the full ARD catalog manifest from public + enabled records.

    Performs three bulk reads (servers, agents, skills) and maps each record to a
    catalog entry. Records that cannot produce a valid URN are skipped and
    counted, never raised, so one bad record cannot fail the whole catalog.
    """
    start = time.time()
    publisher = _resolve_publisher_domain()
    base_url = _base_url_from_request(request)

    server_entries, server_skipped = await _load_server_entries(publisher, base_url)
    agent_entries, agent_skipped = await _load_agent_entries(publisher, base_url)
    skill_entries, skill_skipped = await _load_skill_entries(publisher, base_url)

    entries = server_entries + agent_entries + skill_entries

    # Self-reference: advertise the ARD Registry adapter (POST /api/ard/search,
    # GET /api/ard/agents) as an application/ai-registry+json entry so crawlers
    # can move from the Publisher half to the Registry half (issue #1295).
    if settings.ard_registry_enabled:
        registry_entry = _build_registry_self_entry(publisher, base_url)
        if registry_entry is not None:
            entries.append(registry_entry)

    elapsed_ms = (time.time() - start) * 1000
    logger.info(
        "Built ARD catalog: publisher=%s servers=%d agents=%d skills=%d skipped=%d elapsed_ms=%.1f",
        publisher,
        len(server_entries),
        len(agent_entries),
        len(skill_entries),
        server_skipped + agent_skipped + skill_skipped,
        elapsed_ms,
    )

    return AICatalogManifest(host=_build_host(publisher), entries=entries)
