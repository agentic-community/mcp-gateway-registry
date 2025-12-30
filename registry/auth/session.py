from __future__ import annotations

import logging
import secrets
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from typing import (
    Any,
    Dict,
    Optional,
)

from fastapi import (
    HTTPException,
    status,
)
from itsdangerous import (
    BadSignature,
    SignatureExpired,
    URLSafeTimedSerializer,
)

from auth_server.enforceai.db.data_layer import (
    EnforceAIDataLayer,
)
from auth_server.enforceai.stores.sqlite.session_store import (
    SqliteSessionStore,
)
from gateway_session import (
    build_session_cookie_payload,
    normalize_session_data,
)

logger = logging.getLogger(__name__)


def _load_session_cookie(
    signer: URLSafeTimedSerializer,
    session_cookie: str,
    max_age_seconds: int,
) -> Dict[str, Any]:
    try:
        return signer.loads(session_cookie, max_age=max_age_seconds)
    except SignatureExpired:
        logger.warning("Session cookie has expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session has expired",
        )
    except BadSignature:
        logger.warning("Invalid session cookie signature")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session",
        )


def _require_session_cookie(
    session_cookie: Optional[str],
) -> str:
    if not session_cookie:
        logger.warning("No session cookie provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    return session_cookie


def _touch_enforceai_session_record(
    enforceai_db_path: str,
    session_id: Optional[str],
) -> None:
    if session_id is None:
        return

    try:
        EnforceAIDataLayer(db_path=enforceai_db_path).initialize()
        store = SqliteSessionStore(db_path=enforceai_db_path)
        record = store.get_session_by_id(session_id=session_id)
    except OSError:
        logger.warning(
            "Skipping server-side session validation; EnforceAI DB unavailable",
            exc_info=True,
        )
        return
    except Exception:
        logger.exception("Skipping server-side session validation due to unexpected error")
        return

    if record is None or record.revoked_at is not None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session invalidated",
        )

    store.touch_session(
        session_id=session_id,
        now=datetime.now(timezone.utc).replace(microsecond=0),
    )


def get_current_user_from_cookie(
    signer: URLSafeTimedSerializer,
    session_cookie: Optional[str],
    max_age_seconds: int,
) -> str:
    """Return the authenticated username from the session cookie.

    Args:
        signer: Session signer used to validate cookies.
        session_cookie: Signed session cookie value.
        max_age_seconds: Maximum cookie age for validation.

    Returns:
        The authenticated username.

    Raises:
        HTTPException: When the session is missing/invalid/expired.
    """
    try:
        session_cookie_value = _require_session_cookie(session_cookie)
        data = _load_session_cookie(
            signer=signer,
            session_cookie=session_cookie_value,
            max_age_seconds=max_age_seconds,
        )
        username = data.get("username")

        if not username:
            logger.warning("No username found in session data")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid session data",
            )

        logger.debug("Authentication successful for user: %s", username)
        return username
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Session validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
        )


def get_user_session_data_from_cookie(
    signer: URLSafeTimedSerializer,
    session_cookie: Optional[str],
    max_age_seconds: int,
    enforceai_db_path: Optional[str] = None,
    default_provider: str = "local",
) -> Dict[str, Any]:
    """Return the normalized session data for the current user.

    Args:
        signer: Session signer used to validate cookies.
        session_cookie: Signed session cookie value.
        max_age_seconds: Maximum cookie age for validation.
        enforceai_db_path: Optional EnforceAI DB path to validate the session server-side.
        default_provider: Provider name to default to when missing from the cookie payload.

    Returns:
        Session payload dict containing username, groups, scopes, auth method, provider, etc.

    Raises:
        HTTPException: When the session is missing/invalid/expired.
    """
    try:
        session_cookie_value = _require_session_cookie(session_cookie)
        data = _load_session_cookie(
            signer=signer,
            session_cookie=session_cookie_value,
            max_age_seconds=max_age_seconds,
        )

        normalized = normalize_session_data(
            data,
            default_provider=default_provider,
            max_age_seconds=max_age_seconds,
        )

        if normalized.username is None or not normalized.username.strip():
            logger.warning("No username found in session data")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid session data",
            )

        legacy_auth_method = normalized.legacy_auth_method or ""

        if legacy_auth_method != "oauth2":
            data.setdefault("groups", ["mcp-registry-admin"])
            data.setdefault(
                "scopes",
                [
                    "mcp-servers-unrestricted/read",
                    "mcp-servers-unrestricted/execute",
                ],
            )

        data["v"] = normalized.v
        data["session_id"] = normalized.session_id
        data["user_id"] = normalized.user_id
        data["auth_method"] = normalized.auth_method
        data["legacy_auth_method"] = normalized.legacy_auth_method
        data["provider"] = normalized.provider
        data["username"] = normalized.username
        data["email"] = normalized.email
        data["name"] = normalized.name
        data["groups"] = normalized.groups or data.get("groups", [])
        data["issued_at"] = normalized.issued_at
        data["expires_at"] = normalized.expires_at

        if enforceai_db_path is not None:
            _touch_enforceai_session_record(
                enforceai_db_path=enforceai_db_path,
                session_id=normalized.session_id,
            )

        logger.debug("Session data extracted for user: %s", data.get("username"))
        return data
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Session data extraction error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
        )


def create_session_cookie_value(
    signer: URLSafeTimedSerializer,
    username: str,
    session_max_age_seconds: int,
    auth_method: str = "traditional",
    provider: str = "local",
    enforceai_db_path: Optional[str] = None,
) -> str:
    """Create a signed session cookie for the given user."""
    is_oidc_session = auth_method == "oauth2"
    user_id = f"{provider}|{username}" if is_oidc_session else f"local|{username}"
    groups = [] if is_oidc_session else ["enforceai-admin", "mcp-registry-admin"]
    session_id: Optional[str] = None

    if enforceai_db_path is not None:
        try:
            EnforceAIDataLayer(db_path=enforceai_db_path).initialize()
            store = SqliteSessionStore(db_path=enforceai_db_path)
            session_id = secrets.token_urlsafe(24)
            store.create_session(
                session_id=session_id,
                user_id=user_id,
                auth_method="oidc" if is_oidc_session else "password",
                expires_at=datetime.now(timezone.utc).replace(microsecond=0)
                + timedelta(seconds=session_max_age_seconds),
            )
        except OSError:
            logger.warning(
                "Failed to persist EnforceAI session record; continuing without server-side session",
                exc_info=True,
            )
            session_id = None
        except Exception:
            logger.exception(
                "Unexpected error persisting EnforceAI session record; continuing without server-side session"
            )
            session_id = None

    session_data = build_session_cookie_payload(
        username=username,
        email=None,
        name=None,
        groups=groups,
        provider=provider,
        legacy_auth_method=auth_method,
        max_age_seconds=session_max_age_seconds,
        session_id=session_id,
        user_id=user_id,
    )
    return signer.dumps(session_data)
