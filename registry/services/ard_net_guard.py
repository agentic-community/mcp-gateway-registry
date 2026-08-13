"""ARD structural/domain policy layered on the canonical SSRF URL guard.

ARD ingestion adds two policies to the repository-wide URL/IP classifier:
HTTPS is mandatory, and nested catalogs may be restricted to the root publisher
host or its subdomains. DNS resolution, literal normalization, cloud/workload
metadata denial, and embedded-IPv4 handling are delegated to
:mod:`registry.utils.url_guard`; actual fetches use its pinned guarded transport.
"""

from __future__ import annotations

from urllib.parse import urlparse

from ..exceptions import UrlValidationError
from ..utils.url_guard import SKILL_PROFILE, validate_url
from .ard_search_service import ArdValidationError


def assert_fetchable(
    url: str,
    allowed_domain: str | None = None,
) -> str:
    """Apply ARD domain policy, then canonical structural/IP validation."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if allowed_domain:
        allowed = allowed_domain.lower()
        if not (host == allowed or host.endswith("." + allowed)):
            raise ArdValidationError(
                f"Nested catalog host {host!r} is outside the root domain {allowed!r}"
            )
    try:
        validate_url(
            url,
            profile=SKILL_PROFILE,
            require_https=True,
        )
    except UrlValidationError as exc:
        raise ArdValidationError(f"Ingestion URL rejected: {exc.reason}") from exc
    return url
