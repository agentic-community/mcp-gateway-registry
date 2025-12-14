from __future__ import annotations

import uuid
from typing import Optional

USER_ID_SEPARATOR: str = "|"
USER_ID_FORMAT: str = "'<iss>|<sub>' format"


def _validate_user_id(
    user_id: str,
    *,
    label: str = "user_id",
    prefix: Optional[str] = None,
) -> str:
    parts = user_id.split(USER_ID_SEPARATOR)
    if len(parts) != 2:
        if prefix is not None:
            raise ValueError(f"{prefix} in {USER_ID_FORMAT}")
        raise ValueError(f"{label} must be in {USER_ID_FORMAT}")

    issuer, subject = parts
    if not issuer or not subject:
        if prefix is not None:
            raise ValueError(f"{prefix} in {USER_ID_FORMAT}")
        raise ValueError(f"{label} must be in {USER_ID_FORMAT}")

    return user_id


def _validate_uuid4(
    value: str,
    *,
    label: str = "agent_id",
) -> str:
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a UUIDv4 string") from exc

    if parsed.version != 4:
        raise ValueError(f"{label} must be a UUIDv4 string")

    return value


def _normalize_non_empty_str_list(
    values: list[str],
    *,
    label: str,
) -> list[str]:
    normalized: list[str] = []
    for item in values:
        stripped = item.strip()
        if not stripped:
            raise ValueError(f"{label} must not contain empty strings")
        normalized.append(stripped)
    return normalized


def _normalize_optional_non_empty_str_list(
    values: Optional[list[str]],
    *,
    label: str,
) -> Optional[list[str]]:
    if values is None:
        return None
    return _normalize_non_empty_str_list(values, label=label)


def _intersect_preserving_order(
    *,
    primary: list[str],
    allowed: list[str],
) -> list[str]:
    allowed_set = set(allowed)
    return [item for item in primary if item in allowed_set]

