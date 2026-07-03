"""
Shared request utilities for extracting client information.

Provides validated, safe extraction of client IP from proxied requests.
"""

import ipaddress
import logging
from collections.abc import Mapping

from fastapi import Request

logger = logging.getLogger(__name__)


# Substrings that mark a request-header name (case-insensitive) as sensitive.
# Any header whose name contains one of these is redacted before it is logged.
# This is the fail-closed layer for diagnostic header dumps: a new
# credential-bearing header (e.g. ``X-Auth-Credential``, ``X-Api-Key``) is
# redacted by default rather than requiring an explicit per-name entry.
_SENSITIVE_HEADER_SUBSTRINGS: tuple[str, ...] = (
    "authorization",
    "cookie",
    "token",
    "secret",
    "credential",
    "password",
    "api-key",
    "apikey",
    "auth",
)

_REDACTED_PLACEHOLDER: str = "[REDACTED]"


def is_sensitive_header(name: str) -> bool:
    """Return True if a header name should be redacted before logging.

    Matching is case-insensitive substring matching against a set of
    credential-bearing markers, so variant header names are redacted by default
    (fail closed).

    Args:
        name: The HTTP header name.

    Returns:
        True when the header value must not be logged.
    """
    lowered = name.lower()
    return any(marker in lowered for marker in _SENSITIVE_HEADER_SUBSTRINGS)


def redact_sensitive_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Return a copy of ``headers`` with sensitive values replaced.

    Use this for any diagnostic log line that would otherwise dump raw request
    headers. Credential-bearing headers (Authorization, Cookie, and anything
    carrying a token/secret/credential/api-key) are replaced with a redaction
    placeholder so secrets never reach logs — including at DEBUG level, which
    lands in CI/CloudWatch/shell history.

    Args:
        headers: The request headers to sanitize.

    Returns:
        A new dict with sensitive header values redacted.
    """
    return {
        name: (_REDACTED_PLACEHOLDER if is_sensitive_header(name) else value)
        for name, value in headers.items()
    }


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
