from __future__ import annotations

from typing import Any, Optional

from itsdangerous import (
    BadSignature,
    SignatureExpired,
    URLSafeTimedSerializer,
)

CSRF_TOKEN_SALT: str = "mcp-gateway-csrf-v1"


def mint_csrf_token(
    *,
    secret_key: str,
    session_id: str,
) -> str:
    serializer = URLSafeTimedSerializer(secret_key=secret_key, salt=CSRF_TOKEN_SALT)
    return serializer.dumps({"v": 1, "sid": session_id})


def validate_csrf_token(
    *,
    secret_key: str,
    token: str,
    session_id: str,
    max_age_seconds: int,
) -> Optional[str]:
    """Return error message if invalid; None if valid."""
    if not token.strip():
        return "Missing X-CSRF-Token"

    serializer = URLSafeTimedSerializer(secret_key=secret_key, salt=CSRF_TOKEN_SALT)
    try:
        payload = serializer.loads(
            token,
            max_age=max_age_seconds,
        )
    except SignatureExpired:
        return "CSRF token expired"
    except BadSignature:
        return "Invalid CSRF token"
    except Exception:
        return "Invalid CSRF token"

    if not isinstance(payload, dict):
        return "Invalid CSRF token"

    if payload.get("v") != 1:
        return "Invalid CSRF token"

    sid = payload.get("sid")
    if not isinstance(sid, str) or not sid.strip():
        return "Invalid CSRF token"

    if sid != session_id:
        return "Invalid CSRF token"

    return None

