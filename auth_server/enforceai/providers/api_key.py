from __future__ import annotations

import hashlib
import hmac
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from auth_server.enforceai.errors import (
    DependencyUnavailableError,
    ForbiddenError,
    UnauthorizedError,
)
from auth_server.enforceai.identity import (
    IdentityContext,
)
from auth_server.enforceai.secrets.pepper import (
    load_api_key_pepper,
)
from auth_server.enforceai.stores.interfaces import (
    AgentStore,
    ApiKeyStore,
)

logger = logging.getLogger(__name__)

API_KEY_PREFIX: str = "eak_"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _parse_api_key_value(
    raw_value: str,
) -> tuple[str, str]:
    stripped = raw_value.strip()
    if not stripped:
        raise UnauthorizedError("Malformed API key")

    if not stripped.startswith(API_KEY_PREFIX):
        raise UnauthorizedError("Malformed API key")

    without_prefix = stripped[len(API_KEY_PREFIX) :]
    key_id, sep, secret = without_prefix.partition(".")
    if sep != "." or not key_id or not secret:
        raise UnauthorizedError("Malformed API key")

    return key_id, secret


def _compute_secret_hash(
    *,
    pepper: bytes,
    secret: str,
) -> str:
    digest = hmac.new(
        pepper,
        secret.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return digest


def _is_expired(
    *,
    expires_at: Optional[datetime],
    now: datetime,
) -> bool:
    if expires_at is None:
        return False

    normalized = expires_at
    if normalized.tzinfo is None:
        normalized = normalized.replace(tzinfo=timezone.utc)

    return now > normalized


def _effective_scopes(
    *,
    agent_scopes: list[str],
    api_key_scopes: Optional[list[str]],
) -> list[str]:
    if api_key_scopes is None:
        return list(agent_scopes)

    allowed = set(api_key_scopes)
    return [scope for scope in agent_scopes if scope in allowed]


class ApiKeyProvider:
    def __init__(
        self,
        *,
        api_key_store: ApiKeyStore,
        agent_store: AgentStore,
        pepper_path: Path,
    ) -> None:
        self._api_key_store = api_key_store
        self._agent_store = agent_store
        self._pepper = load_api_key_pepper(pepper_path)

    def resolve_identity(
        self,
        *,
        api_key_value: str,
        now: Optional[datetime] = None,
    ) -> IdentityContext:
        resolved_now = now or _utc_now()

        key_id, secret = _parse_api_key_value(api_key_value)

        try:
            record = self._api_key_store.get_key_by_id(key_id=key_id)
        except Exception as exc:
            logger.exception("API key store lookup failed")
            raise DependencyUnavailableError("API key store lookup failed") from exc

        if record is None:
            raise UnauthorizedError("Invalid API key")

        if record.revoked_at is not None:
            raise ForbiddenError("API key revoked")

        if _is_expired(expires_at=record.expires_at, now=resolved_now):
            raise ForbiddenError("API key expired")

        computed_hash = _compute_secret_hash(
            pepper=self._pepper,
            secret=secret,
        )
        if not hmac.compare_digest(record.secret_hash, computed_hash):
            raise UnauthorizedError("Invalid API key")

        try:
            agent = self._agent_store.get_agent_by_id(agent_id=record.agent_id)
        except Exception as exc:
            logger.exception("Agent store lookup failed")
            raise DependencyUnavailableError("Agent store lookup failed") from exc

        if agent is None:
            raise ForbiddenError("Agent not found for API key")

        if agent.user_id != record.user_id:
            raise ForbiddenError("API key agent binding mismatch")

        if agent.revoked_at is not None:
            raise ForbiddenError("Agent revoked")

        scopes = _effective_scopes(
            agent_scopes=agent.scopes,
            api_key_scopes=record.scopes,
        )

        return IdentityContext(
            user_id=record.user_id,
            agent_id=record.agent_id,
            provider="api-key",
            scopes=scopes,
            user_roles=None,
            metadata={
                "api_key_id": record.key_id,
            },
        )
