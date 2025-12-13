from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

MASKED_VALUE: str = "***MASKED***"

SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "authorization",
        "cookie",
        "x-authorization",
        "x-gateway-token",
        "token",
        "access_token",
        "refresh_token",
        "id_token",
        "api_key",
        "api-key",
        "secret",
        "client_secret",
        "private_key",
        "password",
    }
)


def _is_sensitive_key(
    key: str,
) -> bool:
    return key.strip().lower() in SENSITIVE_KEYS


def mask_secret(
    value: str,
    *,
    show_start: int = 0,
    show_end: int = 4,
) -> str:
    if not value:
        return MASKED_VALUE

    if show_start < 0 or show_end < 0:
        raise ValueError("show_start and show_end must be >= 0")

    if len(value) <= show_start + show_end:
        return MASKED_VALUE

    start = value[:show_start] if show_start else ""
    end = value[-show_end:] if show_end else ""
    return f"{start}...{end}"


def redact_headers(
    headers: Mapping[str, str],
) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for key, value in headers.items():
        if not _is_sensitive_key(key):
            redacted[key] = value
            continue

        lowered = key.lower()
        if lowered == "authorization" and value.lower().startswith("bearer "):
            token = value.split(" ", maxsplit=1)[1]
            redacted[key] = f"Bearer {mask_secret(token)}"
            continue

        redacted[key] = MASKED_VALUE

    return redacted


def redact_mapping(
    data: Mapping[str, Any],
) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in data.items():
        if _is_sensitive_key(key):
            redacted[key] = MASKED_VALUE
            continue

        if isinstance(value, Mapping):
            redacted[key] = redact_mapping(value)
            continue

        redacted[key] = value

    return redacted


def safe_json_dumps(
    data: Any,
    *,
    indent: int = 2,
) -> str:
    if isinstance(data, Mapping):
        data = redact_mapping(data)
    return json.dumps(
        data,
        indent=indent,
        default=str,
    )

