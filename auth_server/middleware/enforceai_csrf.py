from __future__ import annotations

import logging
from typing import Any, Callable, Awaitable

from fastapi import (
    FastAPI,
    Request,
)
from fastapi.responses import (
    JSONResponse,
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

logger = logging.getLogger(__name__)

SAFE_CSRF_METHODS: set[str] = {"GET", "HEAD", "OPTIONS", "TRACE"}


def _has_non_cookie_credentials_for_csrf(
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


def add_enforceai_csrf_middleware(
    app: FastAPI,
    *,
    secret_key: str,
    session_cookie_name: str,
    signer: Any,
    csrf_token_max_age_seconds: int,
) -> None:
    @app.middleware("http")
    async def enforce_csrf_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Any]],
    ):
        if request.method in SAFE_CSRF_METHODS:
            return await call_next(request)

        if not request.url.path.startswith("/enforceai"):
            return await call_next(request)

        if _has_non_cookie_credentials_for_csrf(request):
            return await call_next(request)

        cookie_value = request.cookies.get(session_cookie_name)
        if cookie_value is None or not cookie_value.strip():
            return await call_next(request)

        try:
            session_payload = signer.loads(cookie_value, max_age=28800)
        except (SignatureExpired, BadSignature):
            return await call_next(request)
        except Exception:
            return await call_next(request)

        normalized = normalize_session_data(
            session_payload,
            default_provider="local",
            max_age_seconds=28800,
        )

        csrf_header = request.headers.get("x-csrf-token") or ""
        error = validate_csrf_token(
            secret_key=secret_key,
            token=csrf_header,
            session_id=normalized.session_id,
            max_age_seconds=csrf_token_max_age_seconds,
        )
        if error is not None:
            return JSONResponse(
                status_code=403,
                content={"detail": error},
            )

        return await call_next(request)

    logger.info("Mounted EnforceAI CSRF middleware")

