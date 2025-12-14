from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..crypto.keyring import (
    load_gateway_keyring_cached,
)
from ..errors import (
    DependencyUnavailableError,
    ForbiddenError,
)
from ..identity import (
    IdentityContext,
)
from .._validation import (
    _intersect_preserving_order,
)
from ..stores.interfaces import (
    AgentStore,
    RevocationStore,
)
from ..tokens.verify import (
    verify_gateway_token,
)

logger = logging.getLogger(__name__)


def _ensure_aware_utc(
    value: datetime,
) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


class GatewayTokenProvider:
    def __init__(
        self,
        *,
        agent_store: AgentStore,
        revocation_store: RevocationStore,
        private_key_path: Path,
        public_keys_dir: Path,
        active_kid: str,
        expected_issuer: str,
    ) -> None:
        self._agent_store = agent_store
        self._revocation_store = revocation_store
        self._private_key_path = private_key_path
        self._public_keys_dir = public_keys_dir
        self._active_kid = active_kid
        self._expected_issuer = expected_issuer

    def resolve_identity(
        self,
        *,
        token: str,
        now: Optional[datetime] = None,
    ) -> IdentityContext:
        effective_now = _ensure_aware_utc(now or _utc_now()).replace(microsecond=0)

        try:
            keyring = load_gateway_keyring_cached(
                private_key_path=self._private_key_path,
                public_keys_dir=self._public_keys_dir,
                active_kid=self._active_kid,
            )
        except Exception as exc:  # noqa: BLE001 - map to 503
            logger.exception("Gateway keyring load failed")
            raise DependencyUnavailableError("Gateway keyring unavailable") from exc

        claims = verify_gateway_token(
            token,
            keyring=keyring,
            now=effective_now,
            expected_issuer=self._expected_issuer,
        )

        try:
            agent = self._agent_store.get_agent_by_id(agent_id=claims.agent_id)
        except Exception as exc:  # noqa: BLE001 - map to 503
            logger.exception("Agent store lookup failed")
            raise DependencyUnavailableError("Agent store lookup failed") from exc

        if agent is None:
            raise ForbiddenError("Agent not found")

        if agent.user_id != claims.sub:
            raise ForbiddenError("Agent ownership mismatch")

        if agent.revoked_at is not None:
            raise ForbiddenError("Agent revoked")

        try:
            is_revoked = self._revocation_store.is_jti_revoked(
                jti=claims.jti,
                now=effective_now,
            )
        except Exception as exc:  # noqa: BLE001 - map to 503
            logger.exception("Revocation store lookup failed")
            raise DependencyUnavailableError("Revocation store lookup failed") from exc

        if is_revoked:
            raise ForbiddenError("Token revoked")

        if agent.tokens_valid_after is not None:
            tokens_valid_after = _ensure_aware_utc(agent.tokens_valid_after).replace(
                microsecond=0
            )
            if claims.issued_at < tokens_valid_after:
                raise ForbiddenError("Token revoked")

        scopes = _intersect_preserving_order(
            primary=claims.scopes,
            allowed=agent.scopes,
        )

        metadata = {
            "jti": claims.jti,
            "issuer": claims.iss,
        }
        if agent.allowed_tools is not None:
            metadata["agent_allowed_tools"] = agent.allowed_tools

        return IdentityContext(
            user_id=claims.sub,
            agent_id=claims.agent_id,
            provider="gateway-token",
            scopes=scopes,
            user_roles=None,
            metadata=metadata,
        )
