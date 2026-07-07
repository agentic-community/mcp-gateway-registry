"""Shared redaction helpers for safe logging.

Central chokepoint for scrubbing sensitive data out of values before they are
written to application logs. Application logs land in CI, CloudWatch, and shell
history, so tokens, cookies, credentials, and PII must never be emitted at any
level (INFO, WARNING, or DEBUG).

Use these helpers instead of hand-rolling per-call-site redaction:

- ``redact_headers`` for request/response header mappings (masks
  Authorization / Cookie / API-key headers).
- ``redact_mapping`` for arbitrary dicts (request bodies, ``updates`` payloads,
  user-context dicts) that may carry a token/secret/password under a
  sensitive key.

Fail-safe: prefer logging identifiers and counts over values. When in doubt,
redact.
"""

from typing import Any

REDACTED: str = "[REDACTED]"

# Header names whose values carry a bearer token, session cookie, or other
# credential material. Compared case-insensitively.
SENSITIVE_HEADER_NAMES: frozenset[str] = frozenset(
    {
        "authorization",
        "x-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "api-key",
        "apikey",
        "x-auth-token",
        "x-amz-security-token",
        "proxy-authorization",
        "x-internal-token",
        "x-internal-token-registry",
        "x-federation-token",
        "x-secret",
        "x-session-token",
        "x-auth-credential",
    }
)

# Substrings that, when present in a header NAME (case-insensitive), force
# redaction even if the exact name is not enumerated above. Header names use
# hyphens, so this list is hyphen-oriented. Substring matching keeps the helper
# fail-closed: a variant credential header (e.g. ``X-Custom-Auth-Token``) is
# redacted by default rather than leaking because it was not listed.
#
# KEEP IN SYNC with ``auth_server.server._SENSITIVE_HEADER_SUBSTRINGS`` -- the
# auth server is a separate deployable and cannot import this module, so the
# set is duplicated there. ``test_header_substrings_match_shared_redactor``
# (tests/auth_server/unit/test_server.py) fails if the two drift. This differs
# from the audit query-param list (registry.audit.models) on purpose: header
# names carry ``cookie``/``jwt``/``bearer``/``session``, query keys do not.
SENSITIVE_HEADER_SUBSTRINGS: tuple[str, ...] = (
    "authorization",
    "cookie",
    "token",
    "secret",
    "credential",
    "password",
    "api-key",
    "apikey",
    "auth",
    "jwt",
    "bearer",
    "session",
    "key",
)

# Substrings that, when present in a mapping key (case-insensitive), mark the
# value as sensitive and force redaction. Substring matching is intentional so
# that keys like ``federation_token`` or ``client_secret`` are caught without
# enumerating every variant.
SENSITIVE_KEY_SUBSTRINGS: tuple[str, ...] = (
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
    "authorization",
    "api_key",
    "apikey",
    "session",
    "cookie",
    "private_key",
)


def _is_sensitive_key(key: str) -> bool:
    """Return True when a mapping key names a sensitive value.

    Args:
        key: The mapping key to test.

    Returns:
        True if the key contains any known sensitive substring.
    """
    key_lower = key.lower()
    return any(marker in key_lower for marker in SENSITIVE_KEY_SUBSTRINGS)


def _is_sensitive_header_name(name: str) -> bool:
    """Return True when a header name carries credential material.

    Matches the exact-name allowlist first, then falls back to case-insensitive
    substring matching so variant credential headers redact by default (fail
    closed).

    Args:
        name: The header name to test.

    Returns:
        True if the header value must be redacted before logging.
    """
    lowered = name.lower()
    if lowered in SENSITIVE_HEADER_NAMES:
        return True
    return any(marker in lowered for marker in SENSITIVE_HEADER_SUBSTRINGS)


def redact_headers(headers: Any) -> dict[str, str]:
    """Return a copy of an HTTP header mapping with credential values masked.

    Authorization, Cookie, API-key, and any other credential-bearing header
    (matched by exact name or a credential substring in the name) are replaced
    with a fixed ``[REDACTED]`` marker so the token/cookie value never reaches
    the log, while non-sensitive headers are preserved for diagnostics. The
    full value is dropped (not truncated) because even a prefix of a bearer
    token or session cookie is sensitive.

    Args:
        headers: Any object that exposes ``.items()`` yielding
            ``(name, value)`` pairs (e.g. Starlette ``Headers``, ``dict``).

    Returns:
        A plain dict with sensitive header values replaced by ``[REDACTED]``.
    """
    redacted: dict[str, str] = {}
    for name, value in headers.items():
        if _is_sensitive_header_name(name):
            redacted[name] = REDACTED
        else:
            redacted[name] = value
    return redacted


def redact_mapping(
    data: Any,
    _depth: int = 0,
) -> Any:
    """Return a copy of a mapping with sensitive values recursively masked.

    Any key whose name contains a sensitive substring (``token``, ``secret``,
    ``password``, ``cookie``, ...) has its value replaced by ``[REDACTED]``,
    regardless of the value's type. Nested dicts and lists are traversed so a
    token buried inside a body or user-context dict cannot leak.

    Non-mapping inputs are returned unchanged, so callers can pass an arbitrary
    payload defensively.

    Args:
        data: The value to redact. Dicts and lists are traversed; other types
            are returned as-is.
        _depth: Internal recursion guard; not part of the public contract.

    Returns:
        A redacted copy of the input (same shape), or the input unchanged when
        it is not a dict/list.
    """
    if _depth > 10:
        # Defensive bound against pathological/cyclic structures.
        return REDACTED

    if isinstance(data, dict):
        result: dict[Any, Any] = {}
        for key, value in data.items():
            if isinstance(key, str) and _is_sensitive_key(key):
                result[key] = REDACTED
            else:
                result[key] = redact_mapping(value, _depth + 1)
        return result

    if isinstance(data, list):
        return [redact_mapping(item, _depth + 1) for item in data]

    return data
