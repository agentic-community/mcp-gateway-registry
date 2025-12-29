from __future__ import annotations

import json
import logging

from fastapi import (
    Request,
)

try:
    from ..session_config import (
        SESSION_COOKIE_NAME,
    )
except ImportError:  # pragma: no cover
    from session_config import (
        SESSION_COOKIE_NAME,
    )

logger = logging.getLogger(__name__)

_SESSION_COOKIE_PREFIX: str = f"{SESSION_COOKIE_NAME}="


def _extract_cookie_value(
    request: Request,
    cookie_header: str,
    cookie_name: str,
) -> str | None:
    value = request.cookies.get(cookie_name)
    if value:
        return value

    for cookie in cookie_header.split(";"):
        stripped = cookie.strip()
        if stripped.startswith(f"{cookie_name}="):
            return stripped.split("=", 1)[1]
    return None


def _parse_request_payload(
    body: str | None,
) -> object | None:
    if not body:
        logger.debug("No request body provided, skipping payload parsing")
        return None

    payload_text = body
    logger.debug(
        "Raw Request Payload (%s chars): %s...",
        len(payload_text),
        payload_text[:1000],
    )
    try:
        request_payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse JSON RPC payload: %s", exc)
        return None

    logger.debug("JSON RPC Request Payload: %s", json.dumps(request_payload, indent=2))
    return request_payload

