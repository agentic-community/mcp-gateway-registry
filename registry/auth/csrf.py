from __future__ import annotations

import logging
from typing import Optional

from fastapi import (
    HTTPException,
    Request,
)
from itsdangerous import (
    BadSignature,
    SignatureExpired,
)

from gateway_csrf import (
    validate_csrf_token,
)
from gateway_session import (
    normalize_session_data,
)
from registry.auth.dependencies import (
    signer,
)
from registry.core.config import (
    settings,
)

logger = logging.getLogger(__name__)

SAFE_METHODS: set[str] = {"GET", "HEAD", "OPTIONS", "TRACE"}


def _has_non_cookie_credentials(
    request: Request,
) -> bool:
    authorization = request.headers.get("authorization") or ""
    if authorization.strip():
        return True

    for header_name in ("x-api-key", "x-gateway-token", "x-authorization"):
        value = request.headers.get(header_name)
        if value and value.strip():
            return True

    return False


def _is_csrf_exempt_path(
    path: str,
) -> bool:
    if not path.startswith("/api/auth/"):
        return False

    if path == "/api/auth/logout":
        return False

    return True


def enforce_csrf_for_request(
    request: Request,
) -> None:
    """Enforce CSRF for cookie-authenticated state-changing requests.

    Enforcement triggers only when:
    - method is state-changing (non-safe), and
    - the session cookie is present, and
    - the request is not authenticated via a non-cookie credential.
    """
    if request.method in SAFE_METHODS:
        return

    if not request.url.path.startswith("/api/"):
        return

    if _is_csrf_exempt_path(request.url.path):
        return

    if _has_non_cookie_credentials(request):
        return

    cookie_value = request.cookies.get(settings.session_cookie_name)
    if cookie_value is None or not cookie_value.strip():
        return

    try:
        session_payload = signer.loads(
            cookie_value,
            max_age=settings.session_max_age_seconds,
        )
    except (SignatureExpired, BadSignature):
        return
    except Exception:
        return

    normalized = normalize_session_data(
        session_payload,
        default_provider="local",
        max_age_seconds=settings.session_max_age_seconds,
    )

    csrf_header = request.headers.get("x-csrf-token") or ""
    error = validate_csrf_token(
        secret_key=settings.secret_key,
        token=csrf_header,
        session_id=normalized.session_id,
        max_age_seconds=settings.csrf_token_max_age_seconds,
    )
    if error is not None:
        logger.warning(f"CSRF validation failed for {request.url.path}: {error}")
        raise HTTPException(
            status_code=403,
            detail=error,
        )

