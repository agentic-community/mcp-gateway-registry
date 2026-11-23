"""CSRF token generation and validation utilities.

Provides signed CSRF tokens bound to the user's opaque session_id (resolved
from the signed session cookie via the server-side session store). Tokens
expire based on session max age.
"""

import logging

from fastapi import Form, HTTPException, Request, status
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from ..core.config import settings
from .dependencies import resolve_session_from_cookie
from .session_cookie import (
    request_carries_session_cookie,
    request_session_cookie,
)

logger = logging.getLogger(__name__)

CSRF_SALT: str = "csrf-salt"

_csrf_signer = URLSafeTimedSerializer(settings.secret_key)


def generate_csrf_token(
    session_id: str,
) -> str:
    """Generate a signed CSRF token bound to the given session_id.

    Args:
        session_id: The opaque server-side session identifier (NOT the raw
            cookie value). Callers obtain this by resolving the session
            cookie first.

    Returns:
        A signed CSRF token string.
    """
    token = _csrf_signer.dumps(session_id, salt=CSRF_SALT)
    logger.debug("Generated CSRF token for session")
    return token


def validate_csrf_token(
    token: str,
    session_id: str,
) -> bool:
    """Validate a CSRF token against the session_id.

    Args:
        token: The CSRF token to validate.
        session_id: The opaque session identifier the token should be bound to.

    Returns:
        True if the token is valid, False otherwise.
    """
    try:
        data = _csrf_signer.loads(
            token,
            salt=CSRF_SALT,
            max_age=settings.session_max_age_seconds,
        )
        if data != session_id:
            logger.warning("CSRF token session mismatch")
            return False
        logger.debug("CSRF token validated successfully")
        return True
    except SignatureExpired:
        logger.warning("CSRF token has expired")
        return False
    except BadSignature:
        logger.warning("CSRF token has invalid signature")
        return False
    except Exception as e:
        logger.error(f"Unexpected error validating CSRF token: {e}")
        return False


async def _resolve_session_id(request: Request) -> str | None:
    """Resolve the request's session cookie to the underlying session_id."""
    cookie = request_session_cookie(request)
    if not cookie:
        return None
    data = await resolve_session_from_cookie(cookie)
    if not data:
        return None
    return data.get("session_id")


async def _cookie_session_id(
    request: Request,
) -> str | None:
    """Return the session_id a CSRF token must match, or None for a request with no session cookie.

    A request that carries the session cookie, even more than once or with an
    empty value, is a browser request with ambient credentials. It must
    resolve to a session; skipping CSRF for it would rest on the registry and
    ``/validate`` agreeing that the cookie is no session, which a transient
    store error or a duplicate breaks.

    Args:
        request: The incoming request.

    Returns:
        The resolved session_id, or None when the request has no session cookie.

    Raises:
        HTTPException: 403 when the request carries the session cookie and it
            does not resolve to a session.
    """
    if not request_carries_session_cookie(request):
        logger.debug("No session cookie present, skipping CSRF check (non-browser client)")
        return None
    session_id = await _resolve_session_id(request)
    if not session_id:
        logger.warning("CSRF validation failed: the session cookie does not resolve to a session")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF validation failed: no valid session; sign in again",
        )
    return session_id


async def verify_csrf_token(
    request: Request,
    csrf_token: str = Form(...),
) -> None:
    """FastAPI dependency that validates the CSRF token from form data.

    Resolves the request's session cookie to its session_id and validates the
    submitted CSRF token against it.
    """
    session_id = await _resolve_session_id(request)
    if not session_id:
        logger.warning("CSRF validation failed: no session")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF validation failed: no session",
        )

    if not validate_csrf_token(csrf_token, session_id):
        logger.warning("CSRF validation failed: invalid token")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF validation failed: invalid token",
        )

    logger.debug("CSRF token verified via dependency")


async def verify_csrf_token_flexible(
    request: Request,
) -> None:
    """FastAPI dependency that validates CSRF token from header or form.

    Skips CSRF validation only when the request carries no session cookie, as
    it is from a non-browser client (e.g. Bearer token auth) and CSRF attacks
    require a browser session with cookies. A request that carries the session
    cookie must resolve to a session and present its token, or gets a 403.
    """
    session_id = await _cookie_session_id(request)
    if session_id is None:
        return

    csrf_token = request.headers.get("X-CSRF-Token")
    if not csrf_token:
        try:
            form_data = await request.form()
            form_value = form_data.get("csrf_token")
            # A form field may be an UploadFile; only a string is a valid token.
            csrf_token = form_value if isinstance(form_value, str) else None
        except Exception as e:
            logger.debug(f"Form body parsing failed (likely non-form request): {e}")

    if not csrf_token:
        logger.warning("CSRF validation failed: no token provided (session-based request)")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF validation failed: no token provided",
        )

    if not validate_csrf_token(csrf_token, session_id):
        logger.warning("CSRF validation failed: invalid token")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF validation failed: invalid token",
        )

    logger.debug("CSRF token verified via flexible dependency")


async def verify_csrf_token_header_only(
    request: Request,
) -> None:
    """FastAPI dependency for GET endpoints that return sensitive data.

    Behavior:
    - If the request has no session cookie (e.g. Bearer-token client
      like the CLI or an external integration), CSRF is skipped. Bearer
      auth is already not accessible to browser XHR from a third-party
      origin without the explicit token, so CSRF is not applicable.
    - If the request carries a session cookie, it MUST resolve to a session
      and the X-CSRF-Token header MUST be present AND valid for it. Anything
      else returns 403.

    Rationale: browsers attach cookies to cross-origin GET requests made
    with credentials:"include" even without the CSRF header. Requiring
    the header closes that read vector.
    """
    # The CSRF token is signed with the resolved session_id (see
    # generate_csrf_token in auth/routes.py), NOT the raw cookie blob.
    session_id = await _cookie_session_id(request)
    if session_id is None:
        return

    token = request.headers.get("X-CSRF-Token")
    if not token:
        logger.warning("CSRF header missing on cookie-authenticated GET")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF validation failed: X-CSRF-Token header required",
        )
    if not validate_csrf_token(token, session_id):
        logger.warning("CSRF header invalid on cookie-authenticated GET")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF validation failed: invalid token",
        )
