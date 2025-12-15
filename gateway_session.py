from __future__ import annotations

import time
import uuid
from typing import Any, Mapping, Optional, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class NormalizedSessionData(TypedDict, total=False):
    v: int
    session_id: str
    auth_method: str
    legacy_auth_method: str
    provider: str
    user_id: str
    username: str
    email: str
    name: str
    groups: list[str]
    issued_at: int
    expires_at: int


class SessionCookieV1(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    v: int = Field(default=1, ge=1)
    session_id: str
    auth_method: str
    legacy_auth_method: Optional[str] = None
    provider: str
    user_id: str
    username: Optional[str] = None
    email: Optional[str] = None
    name: Optional[str] = None
    groups: list[str] = Field(default_factory=list)
    issued_at: int
    expires_at: int


def _coerce_int(
    value: object,
    *,
    default: int,
) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return default


def _derive_user_id(
    *,
    raw: Mapping[str, Any],
    provider: str,
    username: Optional[str],
) -> str:
    existing = raw.get("user_id")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()

    issuer = raw.get("iss") or raw.get("issuer")
    subject = raw.get("sub") or raw.get("subject")
    if isinstance(issuer, str) and issuer.strip() and isinstance(subject, str) and subject.strip():
        return f"{issuer.strip()}|{subject.strip()}"

    if username is None or not username.strip():
        return "local|unknown"

    if provider == "local":
        return f"local|{username.strip()}"

    return f"{provider}|{username.strip()}"


def normalize_session_data(
    raw: Mapping[str, Any],
    *,
    default_provider: str = "local",
    max_age_seconds: int = 60 * 60 * 8,
    now_epoch_seconds: Optional[int] = None,
) -> SessionCookieV1:
    """Normalize legacy/new session cookie payloads into a versioned schema.

    This function is intentionally tolerant: it accepts legacy cookies and fills in
    derived fields so callers can rely on a stable shape.

    Args:
        raw: Decoded cookie payload (pre-signature validation).
        default_provider: Provider to use when missing in legacy cookies.
        max_age_seconds: Session TTL (used to derive expires_at when missing).
        now_epoch_seconds: Override "now" for deterministic tests.

    Returns:
        Normalized session payload in `SessionCookieV1`.
    """
    now = int(time.time()) if now_epoch_seconds is None else now_epoch_seconds

    provider = raw.get("provider")
    if not isinstance(provider, str) or not provider.strip():
        provider = default_provider

    legacy_auth_method = raw.get("legacy_auth_method")
    if not isinstance(legacy_auth_method, str) or not legacy_auth_method.strip():
        auth_method_value = raw.get("auth_method")
        if isinstance(auth_method_value, str) and auth_method_value.strip():
            legacy_auth_method = auth_method_value.strip()
        else:
            legacy_auth_method = None

    normalized_auth_method = raw.get("auth_method")
    if not isinstance(normalized_auth_method, str) or not normalized_auth_method.strip():
        normalized_auth_method = "password"
    else:
        normalized_auth_method = normalized_auth_method.strip()

    if normalized_auth_method in {"traditional", "oauth2"}:
        normalized_auth_method = "password" if normalized_auth_method == "traditional" else "oidc"

    if legacy_auth_method == "oauth2":
        normalized_auth_method = "oidc"

    username = raw.get("username")
    if not isinstance(username, str) or not username.strip():
        username = None

    session_id = raw.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        session_id = str(uuid.uuid4())

    issued_at = _coerce_int(raw.get("issued_at"), default=now)
    expires_at = _coerce_int(raw.get("expires_at"), default=issued_at + max_age_seconds)

    return SessionCookieV1(
        v=_coerce_int(raw.get("v"), default=1),
        session_id=session_id,
        auth_method=normalized_auth_method,
        legacy_auth_method=legacy_auth_method,
        provider=provider,
        user_id=_derive_user_id(raw=raw, provider=provider, username=username),
        username=username,
        email=raw.get("email") if isinstance(raw.get("email"), str) else None,
        name=raw.get("name") if isinstance(raw.get("name"), str) else None,
        groups=list(raw.get("groups") or []) if isinstance(raw.get("groups"), list) else [],
        issued_at=issued_at,
        expires_at=expires_at,
    )


def build_session_cookie_payload(
    *,
    username: str,
    email: Optional[str],
    name: Optional[str],
    groups: Optional[list[str]],
    provider: str,
    legacy_auth_method: str,
    max_age_seconds: int,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    now_epoch_seconds: Optional[int] = None,
) -> NormalizedSessionData:
    """Build a v1 session cookie payload while preserving legacy fields.

    This is the preferred way to mint session payloads going forward because it
    produces both:
    - stable v1 fields (v, session_id, user_id, expires_at)
    - legacy compatibility fields (auth_method legacy value via legacy_auth_method)
    """
    now = int(time.time()) if now_epoch_seconds is None else now_epoch_seconds
    payload: NormalizedSessionData = {
        "v": 1,
        "session_id": session_id or str(uuid.uuid4()),
        "auth_method": "oidc" if legacy_auth_method == "oauth2" else "password",
        "legacy_auth_method": legacy_auth_method,
        "provider": provider,
        "username": username,
        "groups": groups or [],
        "issued_at": now,
        "expires_at": now + max_age_seconds,
    }
    if user_id is not None and user_id.strip():
        payload["user_id"] = user_id.strip()
    else:
        payload["user_id"] = _derive_user_id(
            raw=payload,
            provider=provider,
            username=username,
        )
    if email is not None:
        payload["email"] = email
    if name is not None:
        payload["name"] = name
    return payload
