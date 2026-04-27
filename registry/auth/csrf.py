"""CSRF token generation and validation utilities.

Provides signed CSRF tokens bound to user sessions using itsdangerous.
Tokens are validated against the session ID and expire based on session max age.
"""

import logging

from fastapi import Form, HTTPException, Request, status
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from ..core.config import settings

logger = logging.getLogger(__name__)

CSRF_SALT: str = "csrf-salt"

_csrf_signer = URLSafeTimedSerializer(settings.secret_key)


def generate_csrf_token(
    session_id: str,
) -> str:
    """Generate a signed CSRF token bound to the given session ID.

    Args:
        session_id: The session cookie value to bind the token to.

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
    """Validate a CSRF token against the session ID.

    Args:
        token: The CSRF token to validate.
        session_id: The session cookie value the token should be bound to.

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


async def verify_csrf_token(
    request: Request,
    csrf_token: str = Form(...),
) -> None:
    """FastAPI dependency that validates the CSRF token from form data.

    Reads the session cookie from the request and validates the submitted
    CSRF token against it.

    Args:
        request: The incoming FastAPI request.
        csrf_token: The CSRF token submitted via form data.

    Raises:
        HTTPException: If the CSRF token is missing, invalid, or the session
            cookie is not present.
    """
    session_id = request.cookies.get(settings.session_cookie_name)
    if not session_id:
        logger.warning("CSRF validation failed: no session cookie present")
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
    """FastAPI dependency that validates CSRF token from multiple sources.

    Accepts CSRF token from:
    - Form data (for traditional HTML forms)
    - X-CSRF-Token header (for React/SPA applications)

    Skips CSRF validation when no session cookie is present, as the request
    is from a non-browser client (e.g. Bearer token auth) and CSRF attacks
    require a browser session with cookies.

    Args:
        request: The incoming FastAPI request.

    Raises:
        HTTPException: If CSRF token is missing or invalid for session-based requests.
    """
    session_id = request.cookies.get(settings.session_cookie_name)
    if not session_id:
        logger.debug("No session cookie present, skipping CSRF check (non-browser client)")
        return

    # Try to get token from header first (for JSON requests)
    csrf_token = request.headers.get("X-CSRF-Token")

    # If not in header, try form data (for HTML form requests)
    if not csrf_token:
        try:
            form_data = await request.form()
            csrf_token = form_data.get("csrf_token")
        except Exception:
            pass

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
