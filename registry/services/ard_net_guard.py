"""SSRF guard for ARD web ingestion (issue #1296, Phase 3).

Every outbound catalog fetch must use HTTPS, stay within the optional root
publisher domain, and resolve exclusively to public addresses. IP-category
classification is delegated to :mod:`registry.utils.ip_guard` so ARD ingestion
shares the same cloud-credential and embedded-IPv4 protections as other egress.
"""

from __future__ import annotations

import logging
import socket
from urllib.parse import urlparse

from ..utils.ip_guard import coerce_ip_literal, ip_denial_reason
from .ard_search_service import ArdValidationError

logger = logging.getLogger(__name__)


def _is_blocked_ip(ip_text: str) -> bool:
    """Return whether an address is denied by the canonical egress policy."""
    ip = coerce_ip_literal(ip_text)
    return ip is None or ip_denial_reason(ip, allow_private=False) is not None


def assert_fetchable(
    url: str,
    allowed_domain: str | None = None,
) -> str:
    """Validate that ``url`` is safe to fetch, or raise ``ArdValidationError``."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ArdValidationError(f"Refusing non-https ingestion URL: {url!r}")
    host = (parsed.hostname or "").lower()
    if not host:
        raise ArdValidationError(f"Ingestion URL has no host: {url!r}")
    if allowed_domain:
        allowed = allowed_domain.lower()
        if not (host == allowed or host.endswith("." + allowed)):
            raise ArdValidationError(
                f"Nested catalog host {host!r} is outside the root domain {allowed!r}"
            )
    try:
        resolved = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ArdValidationError(f"Cannot resolve ingestion host {host!r}: {exc}") from exc
    for _family, _type, _proto, _canon, sockaddr in resolved:
        ip_text = str(sockaddr[0])
        if _is_blocked_ip(ip_text):
            raise ArdValidationError(
                f"Ingestion host {host!r} resolves to blocked IP {ip_text} (SSRF guard)"
            )
    return url
