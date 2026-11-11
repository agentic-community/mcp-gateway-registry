"""
Shared request utilities for extracting client information.

Provides validated, safe extraction of client IP from proxied requests.
"""

import ipaddress
import logging
import os

from fastapi import Request

logger = logging.getLogger(__name__)


def _trusted_proxy_hops() -> int:
    """Number of trusted reverse-proxy hops in front of the app.

    The client IP is taken from the Nth-from-the-right entry of
    ``X-Forwarded-For`` (the hop our own trusted proxy appended), NOT the
    left-most entry (which the client fully controls and can forge). Defaults to
    1 (the bundled nginx front door). Set ``TRUSTED_PROXY_HOPS`` higher when
    additional trusted proxies (e.g. an ALB + CloudFront) sit in front.

    A value < 1 is treated as 0: no proxy is trusted and only the direct peer is
    used, so a forged header can never win. Fails closed on a malformed value.
    """
    raw = os.environ.get("TRUSTED_PROXY_HOPS", "1").strip()
    try:
        hops = int(raw)
    except ValueError:
        logger.warning("Invalid TRUSTED_PROXY_HOPS=%r; defaulting to 1", raw)
        return 1
    return max(hops, 0)


def _valid_ip(candidate: str) -> str | None:
    """Return the candidate if it is a well-formed IP address, else None."""
    candidate = candidate.strip()
    if not candidate:
        return None
    try:
        ipaddress.ip_address(candidate)
        return candidate
    except ValueError:
        return None


def get_client_ip(request: Request) -> str:
    """
    Extract the client IP from a request behind a trusted reverse proxy.

    Security: the left-most ``X-Forwarded-For`` value is attacker-controlled and
    must never be trusted for audit records. This resolver derives the client IP
    from a NON-SPOOFABLE source and fails toward the direct peer:

    1. ``X-Real-IP`` when present (the bundled nginx sets this to the real
       ``$remote_addr`` and overwrites any client-supplied value).
    2. Otherwise the entry ``TRUSTED_PROXY_HOPS`` positions from the RIGHT of
       ``X-Forwarded-For`` (the hop our own trusted proxy appended). Client-
       supplied left-most entries are ignored.
    3. Otherwise the direct socket peer (``request.client.host``).

    All candidates are validated as well-formed IP addresses to prevent log
    injection / XSS via crafted headers; a malformed candidate is discarded and
    the next non-spoofable source is used.

    Args:
        request: FastAPI Request object

    Returns:
        A validated IP address string, or "unknown" if unavailable.
    """
    # 1. X-Real-IP is set by the trusted proxy to the real peer and overwrites
    #    any client value, so it is the preferred non-spoofable source.
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        validated = _valid_ip(real_ip)
        if validated:
            return validated
        logger.warning("Malformed IP in X-Real-IP header, ignoring")

    # 2. Take the trusted hop from the RIGHT of X-Forwarded-For. With N trusted
    #    proxies, the right-most N entries are appended by trusted hops; the
    #    client-controlled portion is everything to their left. We pick the
    #    left-most of the trusted suffix — the address the outermost trusted
    #    proxy observed as its peer — which a client cannot forge past our proxy.
    forwarded_for = request.headers.get("X-Forwarded-For")
    hops = _trusted_proxy_hops()
    if forwarded_for and hops > 0:
        parts = [p.strip() for p in forwarded_for.split(",") if p.strip()]
        # Only trust the appended hop when the chain is at least as long as the
        # trusted-hop count. If it is SHORTER than expected, the header did not
        # traverse the configured proxy depth: every entry is client-controlled,
        # so we must NOT index into it (parts[0] would be attacker-supplied).
        # Fail closed by ignoring XFF entirely and falling through to the peer.
        if len(parts) >= hops:
            # Index from the right: hop=1 -> parts[-1], hop=2 -> parts[-2], ...
            index = len(parts) - hops
            validated = _valid_ip(parts[index])
            if validated:
                return validated
            logger.warning("Malformed IP in trusted X-Forwarded-For hop, ignoring")
        else:
            logger.warning(
                "X-Forwarded-For has fewer entries (%d) than TRUSTED_PROXY_HOPS "
                "(%d); ignoring header and using direct peer",
                len(parts),
                hops,
            )

    # 3. Direct socket peer — never spoofable, the ultimate fail-closed source.
    if request.client:
        peer = _valid_ip(request.client.host)
        if peer:
            return peer

    return "unknown"


def is_request_https(request: Request) -> bool:
    """Return True when the browser reached the gateway over HTTPS.

    Mirrors ``auth_server.server.is_request_https``, which decides the Secure
    flag of the cookies the auth server sets, so a cookie the registry sets
    follows the same rule. TLS usually ends at a load balancer or CloudFront,
    so the forwarded protocol headers are checked before the request's own
    scheme. A browser drops a Secure cookie set over plain HTTP, so the flag is
    only worth sending when this is True. A spoofed header only changes the
    flag on the response to the request that carried it.

    Args:
        request: The incoming request.

    Returns:
        True if ``X-Cloudfront-Forwarded-Proto`` or ``X-Forwarded-Proto`` is
        ``https``, or the request scheme is ``https``.
    """
    if request.headers.get("x-cloudfront-forwarded-proto", "").lower() == "https":
        return True
    if request.headers.get("x-forwarded-proto", "").lower() == "https":
        return True
    return request.url.scheme == "https"
