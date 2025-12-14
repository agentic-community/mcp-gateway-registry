from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import uuid
from datetime import datetime, timezone
from typing import Optional

from ..crypto.keyring import (
    GatewayKeyring,
)
from ..errors import (
    DependencyUnavailableError,
    ForbiddenError,
)
from ..fgac.models import (
    ScopeCatalog,
)
from ..models.agent import (
    AgentRecord,
)
from ..models.api_key import (
    ApiKeyRecord,
)
from ..models.revocation import (
    TokenRevocationRecord,
)
from ..stores.interfaces import (
    AgentStore,
    ApiKeyStore,
    RevocationStore,
)
from ..tokens.mint import (
    mint_gateway_token,
)
from .models import (
    ApiKeySummary,
)

logger = logging.getLogger(__name__)

API_KEY_PREFIX: str = "eak_"
DEFAULT_API_KEY_SECRET_BYTES: int = 32


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _ensure_aware_utc(
    value: datetime,
) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _normalize_str_list(
    values: Optional[list[str]],
    *,
    label: str,
) -> Optional[list[str]]:
    if values is None:
        return None

    normalized: list[str] = []
    for idx, raw in enumerate(values):
        stripped = raw.strip()
        if not stripped:
            raise ValueError(f"{label}[{idx}] must be a non-empty string")
        normalized.append(stripped)

    return normalized


def _dedupe_preserving_order(
    values: list[str],
) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _validate_scopes_exist(
    *,
    catalog: ScopeCatalog,
    scopes: list[str],
) -> list[str]:
    normalized = _dedupe_preserving_order(
        _normalize_str_list(scopes, label="scopes") or []
    )
    unknown = [scope for scope in normalized if catalog.get_scope(scope) is None]
    if unknown:
        display = ", ".join(unknown[:10])
        suffix = "..." if len(unknown) > 10 else ""
        raise ValueError(f"Unknown scopes: {display}{suffix}")
    return normalized


def _ensure_subset_or_forbidden(
    *,
    requested: list[str],
    allowed: list[str],
    label: str,
) -> None:
    allowed_set = set(allowed)
    violations = [scope for scope in requested if scope not in allowed_set]
    if violations:
        display = ", ".join(violations[:10])
        suffix = "..." if len(violations) > 10 else ""
        raise ForbiddenError(f"{label} exceed agent scopes: {display}{suffix}")


def _hash_api_key_secret(
    *,
    pepper: bytes,
    secret: str,
) -> str:
    return hmac.new(
        pepper,
        secret.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _api_key_value(
    *,
    key_id: str,
    secret: str,
) -> str:
    return f"{API_KEY_PREFIX}{key_id}.{secret}"


def _summarize_api_key(
    record: ApiKeyRecord,
) -> ApiKeySummary:
    return ApiKeySummary(
        key_id=record.key_id,
        user_id=record.user_id,
        agent_id=record.agent_id,
        scopes=record.scopes,
        expires_at=record.expires_at,
        revoked_at=record.revoked_at,
        created_at=record.created_at,
        last_used_at=record.last_used_at,
    )


def _map_store_exception(
    *,
    exc: Exception,
    message: str,
) -> Exception:
    if isinstance(exc, ValueError):
        return exc
    return DependencyUnavailableError(message)  # public message handled upstream


class ManagementService:
    def __init__(
        self,
        *,
        agent_store: AgentStore,
        api_key_store: ApiKeyStore,
        revocation_store: RevocationStore,
        scope_catalog: ScopeCatalog,
        api_key_pepper: Optional[bytes] = None,
        gateway_keyring: Optional[GatewayKeyring] = None,
        gateway_issuer: Optional[str] = None,
    ) -> None:
        self._agent_store = agent_store
        self._api_key_store = api_key_store
        self._revocation_store = revocation_store
        self._catalog = scope_catalog
        self._api_key_pepper = api_key_pepper
        self._gateway_keyring = gateway_keyring
        self._gateway_issuer = gateway_issuer

    def create_agent(
        self,
        *,
        user_id: str,
        scopes: list[str],
        allowed_tools: Optional[list[str]] = None,
        alias: Optional[str] = None,
        metadata: Optional[dict[str, object]] = None,
        agent_id: Optional[str] = None,
    ) -> AgentRecord:
        validated_scopes = _validate_scopes_exist(
            catalog=self._catalog,
            scopes=scopes,
        )
        normalized_allowed_tools = _normalize_str_list(
            allowed_tools,
            label="allowed_tools",
        )
        resolved_agent_id = agent_id or str(uuid.uuid4())

        try:
            return self._agent_store.create_agent(
                user_id=user_id,
                agent_id=resolved_agent_id,
                scopes=validated_scopes,
                allowed_tools=normalized_allowed_tools,
                alias=alias,
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001 - mapped below
            raise _map_store_exception(
                exc=exc,
                message="Agent store create failed",
            ) from exc

    def list_agents(
        self,
        *,
        user_id: str,
    ) -> list[AgentRecord]:
        try:
            return self._agent_store.list_agents_by_user_id(user_id=user_id)
        except Exception as exc:  # noqa: BLE001 - mapped below
            raise _map_store_exception(
                exc=exc,
                message="Agent store list failed",
            ) from exc

    def get_agent(
        self,
        *,
        user_id: str,
        agent_id: str,
    ) -> AgentRecord:
        agent = self._get_owned_agent(
            user_id=user_id,
            agent_id=agent_id,
        )
        return agent

    def update_agent(
        self,
        *,
        user_id: str,
        agent_id: str,
        scopes: Optional[list[str]] = None,
        allowed_tools: Optional[list[str]] = None,
        alias: Optional[str] = None,
        metadata: Optional[dict[str, object]] = None,
    ) -> AgentRecord:
        existing = self._get_owned_agent(
            user_id=user_id,
            agent_id=agent_id,
        )

        if existing.revoked_at is not None:
            raise ForbiddenError("Agent revoked")

        validated_scopes: Optional[list[str]] = None
        if scopes is not None:
            validated_scopes = _validate_scopes_exist(
                catalog=self._catalog,
                scopes=scopes,
            )

        normalized_allowed_tools = _normalize_str_list(
            allowed_tools,
            label="allowed_tools",
        )

        try:
            updated = self._agent_store.update_agent(
                agent_id=existing.agent_id,
                scopes=validated_scopes,
                allowed_tools=normalized_allowed_tools,
                alias=alias,
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001 - mapped below
            raise _map_store_exception(
                exc=exc,
                message="Agent store update failed",
            ) from exc

        if updated is None:
            raise ForbiddenError("Agent not found")

        if updated.user_id != user_id:
            raise ForbiddenError("Agent ownership mismatch")

        return updated

    def revoke_agent(
        self,
        *,
        user_id: str,
        agent_id: str,
        revoked_at: Optional[datetime] = None,
    ) -> AgentRecord:
        existing = self._get_owned_agent(
            user_id=user_id,
            agent_id=agent_id,
        )

        try:
            revoked = self._agent_store.revoke_agent(
                agent_id=existing.agent_id,
                revoked_at=revoked_at,
            )
        except Exception as exc:  # noqa: BLE001 - mapped below
            raise _map_store_exception(
                exc=exc,
                message="Agent store revoke failed",
            ) from exc

        if revoked is None:
            raise ForbiddenError("Agent not found")

        if revoked.user_id != user_id:
            raise ForbiddenError("Agent ownership mismatch")

        return revoked

    def create_api_key(
        self,
        *,
        user_id: str,
        agent_id: str,
        scopes: Optional[list[str]] = None,
        expires_at: Optional[datetime] = None,
        key_id: Optional[str] = None,
        secret: Optional[str] = None,
    ) -> tuple[str, str, str]:
        pepper = self._api_key_pepper
        if pepper is None:
            raise DependencyUnavailableError(
                "API key pepper missing",
                public_message="Enforcement misconfigured",
            )

        agent = self._get_owned_agent(
            user_id=user_id,
            agent_id=agent_id,
        )
        if agent.revoked_at is not None:
            raise ForbiddenError("Agent revoked")

        validated_scopes: Optional[list[str]] = None
        if scopes is not None:
            validated_scopes = _validate_scopes_exist(
                catalog=self._catalog,
                scopes=scopes,
            )
            _ensure_subset_or_forbidden(
                requested=validated_scopes,
                allowed=agent.scopes,
                label="API key scopes",
            )

        resolved_key_id = key_id or str(uuid.uuid4())
        resolved_secret = secret or secrets.token_urlsafe(DEFAULT_API_KEY_SECRET_BYTES)
        secret_hash = _hash_api_key_secret(
            pepper=pepper,
            secret=resolved_secret,
        )

        try:
            self._api_key_store.create_key(
                key_id=resolved_key_id,
                secret_hash=secret_hash,
                user_id=user_id,
                agent_id=agent.agent_id,
                scopes=validated_scopes,
                expires_at=expires_at,
            )
        except Exception as exc:  # noqa: BLE001 - mapped below
            raise _map_store_exception(
                exc=exc,
                message="API key store create failed",
            ) from exc

        return (
            resolved_key_id,
            resolved_secret,
            _api_key_value(
                key_id=resolved_key_id,
                secret=resolved_secret,
            ),
        )

    def list_api_keys(
        self,
        *,
        user_id: str,
        agent_id: str,
    ) -> list[ApiKeySummary]:
        agent = self._get_owned_agent(
            user_id=user_id,
            agent_id=agent_id,
        )

        try:
            records = self._api_key_store.list_keys(
                user_id=user_id,
                agent_id=agent.agent_id,
            )
        except Exception as exc:  # noqa: BLE001 - mapped below
            raise _map_store_exception(
                exc=exc,
                message="API key store list failed",
            ) from exc

        summaries: list[ApiKeySummary] = []
        for record in records:
            if record.user_id != user_id:
                raise DependencyUnavailableError("API key store returned foreign user keys")
            if record.agent_id != agent.agent_id:
                raise DependencyUnavailableError("API key store returned foreign agent keys")
            summaries.append(_summarize_api_key(record))
        return summaries

    def revoke_api_key(
        self,
        *,
        user_id: str,
        key_id: str,
        revoked_at: Optional[datetime] = None,
    ) -> ApiKeySummary:
        try:
            existing = self._api_key_store.get_key_by_id(key_id=key_id)
        except Exception as exc:  # noqa: BLE001 - mapped below
            raise _map_store_exception(
                exc=exc,
                message="API key store lookup failed",
            ) from exc

        if existing is None:
            raise ForbiddenError("API key not found")

        if existing.user_id != user_id:
            raise ForbiddenError("API key ownership mismatch")

        try:
            revoked = self._api_key_store.revoke_key(
                key_id=key_id,
                revoked_at=revoked_at,
            )
        except Exception as exc:  # noqa: BLE001 - mapped below
            raise _map_store_exception(
                exc=exc,
                message="API key store revoke failed",
            ) from exc

        if revoked is None:
            raise ForbiddenError("API key not found")

        if revoked.user_id != user_id:
            raise ForbiddenError("API key ownership mismatch")

        return _summarize_api_key(revoked)

    def mint_gateway_token(
        self,
        *,
        user_id: str,
        agent_id: str,
        scopes: list[str],
        ttl_seconds: Optional[int] = None,
        expires_at: Optional[datetime] = None,
        issued_at: Optional[datetime] = None,
        jti: Optional[str] = None,
    ) -> str:
        keyring = self._gateway_keyring
        issuer = self._gateway_issuer
        if keyring is None or issuer is None or not issuer.strip():
            raise DependencyUnavailableError(
                "Gateway token configuration missing",
                public_message="Enforcement misconfigured",
            )

        if not scopes:
            raise ValueError("scopes must be a non-empty list")

        agent = self._get_owned_agent(
            user_id=user_id,
            agent_id=agent_id,
        )
        if agent.revoked_at is not None:
            raise ForbiddenError("Agent revoked")

        validated_scopes = _validate_scopes_exist(
            catalog=self._catalog,
            scopes=scopes,
        )
        _ensure_subset_or_forbidden(
            requested=validated_scopes,
            allowed=agent.scopes,
            label="Token scopes",
        )

        return mint_gateway_token(
            keyring=keyring,
            issuer=issuer,
            user_id=user_id,
            agent_id=agent.agent_id,
            scopes=validated_scopes,
            issued_at=issued_at,
            expires_at=expires_at,
            ttl_seconds=ttl_seconds,
            jti=jti,
        )

    def revoke_token_jti(
        self,
        *,
        user_id: str,
        agent_id: str,
        jti: str,
        expires_at: Optional[datetime] = None,
        reason: Optional[str] = None,
        revoked_at: Optional[datetime] = None,
    ) -> TokenRevocationRecord:
        agent = self._get_owned_agent(
            user_id=user_id,
            agent_id=agent_id,
        )

        if not jti.strip():
            raise ValueError("jti must be a non-empty string")

        try:
            return self._revocation_store.revoke_jti(
                jti=jti.strip(),
                user_id=user_id,
                agent_id=agent.agent_id,
                revoked_at=revoked_at,
                expires_at=expires_at,
                reason=reason,
            )
        except Exception as exc:  # noqa: BLE001 - mapped below
            raise _map_store_exception(
                exc=exc,
                message="Revocation store revoke failed",
            ) from exc

    def revoke_all_tokens(
        self,
        *,
        user_id: str,
        agent_id: str,
        now: Optional[datetime] = None,
    ) -> AgentRecord:
        agent = self._get_owned_agent(
            user_id=user_id,
            agent_id=agent_id,
        )

        tokens_valid_after = _ensure_aware_utc(now or _utc_now()).replace(microsecond=0)

        try:
            updated = self._agent_store.bump_tokens_valid_after(
                agent_id=agent.agent_id,
                tokens_valid_after=tokens_valid_after,
            )
        except Exception as exc:  # noqa: BLE001 - mapped below
            raise _map_store_exception(
                exc=exc,
                message="Agent store bump tokens_valid_after failed",
            ) from exc

        if updated is None:
            raise ForbiddenError("Agent not found")

        if updated.user_id != user_id:
            raise ForbiddenError("Agent ownership mismatch")

        return updated

    def _get_owned_agent(
        self,
        *,
        user_id: str,
        agent_id: str,
    ) -> AgentRecord:
        try:
            agent = self._agent_store.get_agent_by_id(agent_id=agent_id)
        except Exception as exc:  # noqa: BLE001 - mapped below
            logger.exception("Agent store lookup failed")
            raise _map_store_exception(
                exc=exc,
                message="Agent store lookup failed",
            ) from exc

        if agent is None:
            raise ForbiddenError("Agent not found")

        if agent.user_id != user_id:
            raise ForbiddenError("Agent ownership mismatch")

        return agent
