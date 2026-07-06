"""Domain-anchored trust verification for ARD ingestion (issue #1296, Phase 3).

Trust is *additive* to the OAuth2/scope model: domain anchoring proves a
catalog entry's claimed *publisher identity*; OAuth2/scopes still gate *access*.

The check: extract the publisher FQDN from each entry's URN
(``urn:air:<publisher>:...``) and require it to match the catalog host's
``trustManifest.identity`` domain (proven over TLS via the ``https`` identity
type), plus any per-source pinned ``expected_identity``. A mismatch is handled
per the configured policy:

- ``reject`` (default): skip the entry, do not index it, count it.
- ``flag``: index the entry but annotate it with the mismatch reason.
- ``off``: disable the check entirely.

Pure logic, no I/O.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

from ..schemas.ard_models import ArdCatalogEntry
from ..schemas.federation_schema import AiCatalogSourceConfig
from .ard_ingest_mapping import parse_urn_publisher

logger = logging.getLogger(__name__)


def host_identity_domain(
    identity: str | None,
) -> str | None:
    """Extract the FQDN from a trustManifest.identity value.

    Accepts an ``https://host[...]`` URL or a bare domain. Returns the lowercased
    hostname, or ``None`` when nothing usable is present.
    """
    text = (identity or "").strip()
    if not text:
        return None
    parsed = urlparse(text if "://" in text else f"https://{text}")
    return (parsed.hostname or "").lower() or None


def _source_anchor_domain(
    source: AiCatalogSourceConfig,
) -> str | None:
    """Return the operator-anchored identity domain for a source.

    Priority: pinned ``expected_identity`` -> configured ``domain`` -> host of
    the configured ``uri``. This is the domain the operator chose to trust when
    they added the source, and is what entry URNs are anchored against — NOT the
    catalog's self-declared ``trustManifest.identity`` (which the served document
    controls and could use to impersonate another publisher).
    """
    if source.expected_identity:
        return host_identity_domain(source.expected_identity)
    if source.domain:
        return host_identity_domain(source.domain)
    if source.uri:
        return host_identity_domain(source.uri)
    return None


def verify_entry_trust(
    entry: ArdCatalogEntry,
    host_domain: str | None,
    source: AiCatalogSourceConfig,
    policy: str,
) -> tuple[bool, str | None]:
    """Decide whether an ingested entry is trusted under ``policy``.

    Args:
        entry: The catalog entry being ingested.
        host_domain: FQDN of the catalog host's trustManifest.identity (the
            document actually served — never an entry-supplied identity).
        source: The ingestion source config (may pin ``expected_identity``).
        policy: ``reject`` | ``flag`` | ``off``.

    Returns:
        ``(accept, reason)``. ``reason`` is ``None`` on a clean match, else a
        short human-readable mismatch description (recorded as an annotation
        under ``flag`` or for logging/metrics under ``reject``).
    """
    if policy == "off":
        return True, None

    publisher = parse_urn_publisher(entry.identifier)
    if publisher is None:
        reason = "unparseable-urn"
        return (policy != "reject"), reason

    # Anchor to the OPERATOR-configured source identity (where we chose to fetch
    # from), not the manifest's self-declared identity. Otherwise a configured
    # source served from acme.com could declare identity=victim.com with
    # victim.com URNs and pass, impersonating another publisher in our index.
    # Fall back to the served identity only if the source carries no
    # domain/uri/pin (which cannot happen for a configured source).
    anchor = _source_anchor_domain(source) or host_domain
    if anchor and publisher.lower() == anchor.lower():
        return True, None

    reason = f"urn-publisher={publisher} != source={anchor}"
    return (policy != "reject"), reason
