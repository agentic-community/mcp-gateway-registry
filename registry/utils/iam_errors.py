"""
Typed IdP error hierarchy and raw-message detectors.

Providers raise free-form exceptions (`EntraAdminError`, Keycloak/Okta/Auth0
errors) whose message text is the only signal we have for status classification.
The IAM manager subclasses inspect those exceptions and re-raise them as one
of the typed errors below so route handlers can branch on exception type
rather than substring match.
"""

from __future__ import annotations


class IdPError(RuntimeError):
    """Base class for IdP admin API errors."""


class IdPForbiddenError(IdPError):
    """Raised when the IdP returns HTTP 403 / forbidden on an admin call.

    Means the registry's IdP app lacks the required write scope for this
    operation. PATCH/DELETE callers treat this as non-fatal and fall through
    to the MongoDB mutation.
    """


class IdPNotFoundError(IdPError):
    """Raised when the IdP returns HTTP 404 on an admin call.

    Means the group does not exist (or is not visible) in the upstream IdP.
    PATCH/DELETE callers treat this as non-fatal.
    """


_FORBIDDEN_MARKERS: tuple[str, ...] = (
    "403",
    "forbidden",
    "insufficient privileges",
    "authorization_requestdenied",
)

_NOT_FOUND_MARKERS: tuple[str, ...] = (
    "not found",
    "404",
)


def looks_forbidden(
    exc: Exception,
) -> bool:
    """Raw-message fallback: is this exception an IdP 403?"""
    detail = str(exc).lower()
    return any(marker in detail for marker in _FORBIDDEN_MARKERS)


def looks_not_found(
    exc: Exception,
) -> bool:
    """Raw-message fallback: is this exception an IdP 404?"""
    detail = str(exc).lower()
    return any(marker in detail for marker in _NOT_FOUND_MARKERS)


def wrap_idp_admin_error(
    exc: Exception,
) -> Exception:
    """Translate raw provider errors into typed IdP errors when detectable.

    Returns the typed error if the message matches; otherwise returns the
    original exception unchanged so the existing 502 translation still works
    for unknown error types.
    """
    if looks_forbidden(exc):
        return IdPForbiddenError(str(exc))
    if looks_not_found(exc):
        return IdPNotFoundError(str(exc))
    return exc
