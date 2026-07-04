"""SSRF guard for outbound A2A requests to registry-discovered remote agents.

A remote agent's endpoint URL comes from the registry (registrant-controlled)
and is therefore not fully trusted. Before this agent fetches an untrusted
agent's ``/.well-known/agent-card`` or sends it a message, the destination must
pass :func:`assert_fetchable`. The guard is intentionally strict and fails
closed:

- ``http`` / ``https`` only (no ``file`` / ``gopher`` / ``ftp`` / ...).
- The host must resolve **only** to public IPs. The check runs *after* DNS
  resolution, which defeats DNS-rebinding (a hostname that resolves to a public
  IP once and a private IP on the real fetch). Redirects must additionally be
  disabled on the transport so a public host cannot 302 to an internal one.
- The cloud metadata endpoint (``169.254.169.254``) and all RFC-1918 /
  loopback / link-local / reserved / multicast ranges are blocked and can never
  be allowlisted.

This mirrors the registry-side ARD network guard; the a2a agent is a standalone
package that cannot import from the registry, so a small self-contained copy
lives here. Keep the two in sync when the blocked-network policy changes.
"""

import ipaddress
import logging
import socket
from urllib.parse import urlparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


class UnsafeUrlError(ValueError):
    """Raised when an outbound URL fails the SSRF guard."""


# Networks that must never be the target of an outbound A2A fetch. The
# per-address property checks (``is_private`` etc.) cover most of these; the
# explicit list documents intent and pins the cloud metadata endpoint plus the
# IPv6 unique-local / link-local ranges.
_BLOCKED_NETS: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
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

_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})


def _is_blocked_ip(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
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
) -> str:
    """Validate that ``url`` is safe to fetch, or raise ``UnsafeUrlError``.

    Args:
        url: The absolute URL of a discovered remote agent about to be
            contacted.

    Returns:
        The validated URL (unchanged) on success.

    Raises:
        UnsafeUrlError: For non-http(s) schemes, missing/blocked hosts, or hosts
            that resolve to private/metadata IPs.
    """
    if not url or not isinstance(url, str):
        raise UnsafeUrlError(f"Refusing empty or non-string agent URL: {url!r}")

    parsed = urlparse(url)
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise UnsafeUrlError(f"Refusing non-http(s) agent URL scheme: {url!r}")

    host = (parsed.hostname or "").lower()
    if not host:
        raise UnsafeUrlError(f"Agent URL has no host: {url!r}")

    try:
        resolved = socket.getaddrinfo(host, parsed.port or None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise UnsafeUrlError(f"Cannot resolve agent host {host!r}: {e}") from e

    for _family, _type, _proto, _canon, sockaddr in resolved:
        ip = ipaddress.ip_address(sockaddr[0])
        if _is_blocked_ip(ip):
            raise UnsafeUrlError(f"Agent host {host!r} resolves to blocked IP {ip} (SSRF guard)")

    return url
