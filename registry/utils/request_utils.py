"""
Shared request utilities for extracting client information.

Provides validated, safe extraction of client IP from proxied requests.
"""

import ipaddress
import logging

from fastapi import Request

logger = logging.getLogger(__name__)


def get_client_ip(request: Request) -> str:
    """
    Extract the client IP from a request, preferring X-Forwarded-For when present.

    Validates that the extracted value is a well-formed IP address to prevent
    log injection or XSS via crafted headers.

    Args:
        request: FastAPI Request object

    Returns:
        A validated IP address string, or "unknown" if unavailable.
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        candidate = forwarded_for.split(",")[0].strip()
        try:
            ipaddress.ip_address(candidate)
            return candidate
        except ValueError:
            logger.warning("Malformed IP in X-Forwarded-For header, ignoring")

    if request.client:
        return request.client.host

    return "unknown"
