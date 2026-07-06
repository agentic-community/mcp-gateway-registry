"""SSRF guard for ARD web ingestion (issue #1296, Phase 3).

Every outbound fetch during catalog crawling (the root source URL and every
nested ``application/ai-catalog+json`` URL found inside a fetched document) must
pass :func:`assert_fetchable` first. The guard is intentionally strict:

- ``https`` only (no ``http``/``file``/``gopher``/...).
- The host must resolve **only** to public IPs. The check runs *after* DNS
  resolution, which defeats DNS-rebinding (a hostname that resolves to a public
  IP once and a private IP on the real fetch).
- Optional same-domain restriction for nested recursion, so a compromised
  catalog cannot pivot the crawler to an unrelated host.

Pure logic with the single unavoidable side effect of a DNS lookup; raises
:class:`ArdValidationError` (the same error the ingestion layer treats as a
skip-and-log) on any unsafe URL.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

from .ard_search_service import ArdValidationError

logger = logging.getLogger(__name__)

# Networks that must never be the target of an ingestion fetch. ``is_private`` /
# ``is_loopback`` / ``is_link_local`` / ``is_reserved`` / ``is_multicast`` cover
# most of these already; the explicit list documents intent and guards the cloud
# metadata endpoint (169.254.169.254) and IPv6 unique-local / link-local ranges.
_BLOCKED_NETS: list[ipaddress._BaseNetwork] = [
    ipaddress.ip_network(n)
    for n in (
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "::1/128",
        "fc00::/7",
        "fe80::/10",
    )
]


def _is_blocked_ip(ip: ipaddress._BaseAddress) -> bool:
    """Return True if an IP is private/loopback/link-local/reserved/blocked.

    IPv4-mapped IPv6 addresses (``::ffff:10.0.0.1``) are unwrapped to their
    embedded IPv4 first so a private target cannot be smuggled past the checks.
    """
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    ):
        return True
    return any(ip in net for net in _BLOCKED_NETS)


def assert_fetchable(
    url: str,
    allowed_domain: str | None = None,
) -> str:
    """Validate that ``url`` is safe to fetch, or raise ``ArdValidationError``.

    Args:
        url: The absolute URL about to be fetched.
        allowed_domain: When set, the URL host must equal it or be a subdomain
            (used to keep nested-catalog recursion on the root source's domain).

    Returns:
        The validated URL (unchanged) on success.

    Raises:
        ArdValidationError: For non-https schemes, missing/blocked hosts, hosts
            that resolve to private/metadata IPs, or out-of-domain nested URLs.
    """
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
    except socket.gaierror as e:
        raise ArdValidationError(f"Cannot resolve ingestion host {host!r}: {e}") from e
    for _family, _type, _proto, _canon, sockaddr in resolved:
        ip = ipaddress.ip_address(sockaddr[0])
        if _is_blocked_ip(ip):
            raise ArdValidationError(
                f"Ingestion host {host!r} resolves to blocked IP {ip} (SSRF guard)"
            )
    return url
