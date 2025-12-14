from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional

from ..errors import (
    DependencyUnavailableError,
    ForbiddenError,
)
from ..identity import (
    IdentityContext,
)
from ..oidc.verify import (
    OIDCVerifier,
)
from ..stores.interfaces import (
    AgentStore,
)

logger = logging.getLogger(__name__)


def _validate_uuid4(
    raw_value: str,
) -> str:
    try:
        parsed = uuid.UUID(raw_value)
    except ValueError as exc:
        raise ForbiddenError("Invalid X-Agent-Id") from exc

    if parsed.version != 4:
        raise ForbiddenError("Invalid X-Agent-Id")

    return raw_value


class OidcProvider:
    def __init__(
        self,
        *,
        verifier: OIDCVerifier,
        agent_store: AgentStore,
    ) -> None:
        self._verifier = verifier
        self._agent_store = agent_store

    async def resolve_identity(
        self,
        *,
        bearer_token: str,
        agent_id_header: Optional[str],
        now: Optional[datetime] = None,
    ) -> IdentityContext:
        if agent_id_header is None or not agent_id_header.strip():
            raise ForbiddenError("Missing X-Agent-Id")

        agent_id = _validate_uuid4(agent_id_header.strip())

        validated = await self._verifier.verify_bearer_token(bearer_token)

        try:
            agent = self._agent_store.get_agent_by_id(agent_id=agent_id)
        except Exception as exc:  # noqa: BLE001 - map to dependency failure
            logger.exception("Agent store lookup failed")
            raise DependencyUnavailableError("Agent store lookup failed") from exc

        if agent is None:
            raise ForbiddenError("Agent not found")

        if agent.user_id != validated.user_id:
            raise ForbiddenError("Agent ownership mismatch")

        if agent.revoked_at is not None:
            raise ForbiddenError("Agent revoked")

        metadata = {
            "issuer": validated.issuer,
            "audiences": validated.audiences,
            "oidc_scopes": validated.scopes,
            "oidc_roles": validated.roles,
            "claims": validated.claims,
        }
        if agent.allowed_tools is not None:
            metadata["agent_allowed_tools"] = agent.allowed_tools

        return IdentityContext(
            user_id=validated.user_id,
            agent_id=agent.agent_id,
            provider="oidc",
            scopes=list(agent.scopes),
            user_roles=validated.roles,
            metadata=metadata,
        )
