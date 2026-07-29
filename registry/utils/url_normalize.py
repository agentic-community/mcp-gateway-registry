"""URL normalization helpers for registration deduplication.

These are intentionally separate from registry/utils/url_utils.py, which
deals with GitHub URL translation. The functions here normalize URLs to a
canonical string form for *exact-match* collision detection in the
duplicate-check service.
"""

import logging
from typing import Literal
from urllib.parse import urlparse

from registry.common.log_redaction import redact_url

logger = logging.getLogger(__name__)

DEFAULT_PORTS: dict[str, int] = {"http": 80, "https": 443}

# Entity type identifiers used by the duplicate-check service. Typed
# as Literal so type-checkers can narrow them where the broader
# ``EntityType`` (Literal of all three) is expected.
ENTITY_TYPE_SERVER: Literal["mcp_server"] = "mcp_server"
ENTITY_TYPE_AGENT: Literal["a2a_agent"] = "a2a_agent"
ENTITY_TYPE_SKILL: Literal["skill"] = "skill"

# Sidecar field name for the normalized identity URL. Stored alongside
# the user-supplied URL field on each document so duplicate-check
# lookups can do an indexed ``$eq`` query instead of a client-side
# scan-and-normalize. The leading underscore signals "internal,
# registry-managed" — not part of the public document shape and not
# user-editable. Same name across all three entity types so the
# DocumentDB index is identical per collection.
NORMALIZED_IDENTITY_URL_FIELD: str = "_identity_url_normalized"

# Per-entity-type mapping from the user-facing URL field (the one the
# document already stores) to the sidecar field. Used by repository
# write paths to derive the sidecar value at insert/update time.
IDENTITY_URL_FIELD_BY_ENTITY: dict[str, str] = {
    ENTITY_TYPE_SERVER: "proxy_pass_url",
    ENTITY_TYPE_AGENT: "url",
    ENTITY_TYPE_SKILL: "skill_md_url",
}


def _parse_url(
    url: str,
) -> tuple[str, str, int | None, str] | None:
    """Parse a URL into (scheme, host, port, path) with default-port stripping.

    Returns None when the input is unparseable or missing scheme/host.
    """
    try:
        parsed = urlparse(url.strip())
    except Exception as exc:
        logger.debug("Failed to parse url for normalization: %s (%s)", redact_url(url), exc)
        return None
    if not parsed.scheme or not parsed.hostname:
        return None
    scheme = parsed.scheme.lower()
    host = parsed.hostname.lower()
    port: int | None = parsed.port
    if port is not None and DEFAULT_PORTS.get(scheme) == port:
        port = None
    path = parsed.path or ""
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    if path == "/":
        path = ""
    return scheme, host, port, path


def _normalize_http_identity(
    url: str,
) -> str | None:
    """Canonical scheme-insensitive identity form for an HTTP-style URL.

    Used by mcp_server (proxy_pass_url) and a2a_agent (agent endpoint url).
    Two URLs that differ only by scheme (http vs https) collapse to the
    same identity — same host + port + path means the same service in
    practice.

    Form: ``{host}[:{port}]{path}`` (no scheme, default ports stripped,
    query/fragment dropped, trailing slash trimmed).
    """
    parsed = _parse_url(url)
    if parsed is None:
        return None
    _scheme, host, port, path = parsed
    if port is not None:
        return f"{host}:{port}{path}"
    return f"{host}{path}"


def _normalize_github_identity(
    url: str,
) -> str | None:
    """Canonical identity form for a GitHub URL (used for skills).

    Skills carry a ``skill_md_url`` that points at a SKILL.md file on
    GitHub. Two skills pointing at the same file should collide even if
    they were registered with slightly different URL spellings:

    - Trailing ``.git`` is stripped (``repo.git`` ≡ ``repo``).
    - Host is lowercased.
    - Path is preserved case-sensitive (GitHub paths are case-sensitive).
    - Default ports stripped, query/fragment dropped.
    - Scheme is collapsed (http and https forms collapse together).

    Returns None when the URL is unparseable.
    """
    parsed = _parse_url(url)
    if parsed is None:
        return None
    _scheme, host, port, path = parsed
    if path.endswith(".git"):
        path = path[: -len(".git")]
    if path.endswith(".git/"):
        path = path[: -len(".git/")]
    if port is not None:
        return f"{host}:{port}{path}"
    return f"{host}{path}"


def normalize_proxy_url(
    url: str | None,
) -> dict[str, str | int | None] | None:
    """Parse a proxy URL into a canonical (scheme, host, port, path) dict.

    Normalization:
    - lowercase scheme + host
    - drop default ports (80 for http, 443 for https) -> port=None
    - strip trailing slash from path (but preserve a single "/" for root)
    - drop query and fragment

    Returns None on parse failure or None/empty input.

    Examples:
        >>> normalize_proxy_url("https://Example.com/")
        {"scheme": "https", "host": "example.com", "port": None, "path": ""}
        >>> normalize_proxy_url("https://example.com:443/foo/?q=1")
        {"scheme": "https", "host": "example.com", "port": None, "path": "/foo"}
    """
    if not url:
        return None
    parsed = _parse_url(url)
    if parsed is None:
        return None
    scheme, host, port, path = parsed
    return {"scheme": scheme, "host": host, "port": port, "path": path}


def derive_normalized_identity_url(
    doc: dict,
    entity_type: str,
) -> str | None:
    """Compute the sidecar identity-URL value for a repository document.

    Looks up the user-facing URL field (``proxy_pass_url`` / ``url`` /
    ``skill_md_url``) on the document and returns its normalized form
    per :func:`normalize_identity_url`. Returns None when the document
    has no URL or the URL is unparseable — callers should ``$unset``
    the sidecar in that case rather than persist a None.

    Repositories call this at write time so the sidecar field stays
    in sync with the user-facing URL field.
    """
    source_field = IDENTITY_URL_FIELD_BY_ENTITY.get(entity_type)
    if source_field is None:
        return None
    raw_value = doc.get(source_field)
    if raw_value is None:
        return None
    return normalize_identity_url(str(raw_value), entity_type)


def normalize_identity_url(
    url: str | None,
    entity_type: str,
) -> str | None:
    """Canonical scheme-insensitive identity form per entity type.

    Used by ``DuplicateCheckService`` to find exact-URL collisions.

    Args:
        url: Raw URL from the registration request. May be None.
        entity_type: One of ``mcp_server``, ``a2a_agent``, ``skill``.

    Returns:
        A canonical string suitable for exact equality comparison, or None
        if the URL is missing/unparseable. Two URLs that should be treated
        as the same service produce the same string; URLs that differ in
        a meaningful way produce different strings.

    Notes:
        Scheme is intentionally collapsed: ``http://x/mcp`` and
        ``https://x/mcp`` both normalize to the same identity. Two
        registrations at the same host/port/path are almost always the
        same service regardless of scheme.
    """
    if not url:
        return None
    if entity_type == ENTITY_TYPE_SKILL:
        return _normalize_github_identity(url)
    if entity_type in (ENTITY_TYPE_SERVER, ENTITY_TYPE_AGENT):
        return _normalize_http_identity(url)
    logger.debug("normalize_identity_url: unknown entity_type %r", entity_type)
    return None
