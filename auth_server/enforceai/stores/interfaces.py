from __future__ import annotations

from datetime import datetime
from typing import Optional, Protocol

from ..models.agent import (
    AgentRecord,
)
from ..models.api_key import (
    ApiKeyRecord,
)
from ..models.revocation import (
    TokenRevocationRecord,
)
from ..models.audit import (
    AuditEventRecord,
)


class AgentStore(Protocol):
    def create_agent(
        self,
        *,
        user_id: str,
        agent_id: str,
        scopes: list[str],
        allowed_tools: Optional[list[str]] = None,
        alias: Optional[str] = None,
        metadata: Optional[dict[str, object]] = None,
    ) -> AgentRecord:
        ...

    def get_agent_by_id(
        self,
        *,
        agent_id: str,
    ) -> Optional[AgentRecord]:
        ...

    def list_agents_by_user_id(
        self,
        *,
        user_id: str,
    ) -> list[AgentRecord]:
        ...

    def update_agent(
        self,
        *,
        agent_id: str,
        scopes: Optional[list[str]] = None,
        allowed_tools: Optional[list[str]] = None,
        alias: Optional[str] = None,
        metadata: Optional[dict[str, object]] = None,
    ) -> Optional[AgentRecord]:
        ...

    def revoke_agent(
        self,
        *,
        agent_id: str,
        revoked_at: Optional[datetime] = None,
    ) -> Optional[AgentRecord]:
        ...

    def bump_tokens_valid_after(
        self,
        *,
        agent_id: str,
        tokens_valid_after: datetime,
    ) -> Optional[AgentRecord]:
        ...


class ApiKeyStore(Protocol):
    def create_key(
        self,
        *,
        key_id: str,
        secret_hash: str,
        user_id: str,
        agent_id: str,
        scopes: Optional[list[str]] = None,
        expires_at: Optional[datetime] = None,
    ) -> ApiKeyRecord:
        ...

    def get_key_by_id(
        self,
        *,
        key_id: str,
    ) -> Optional[ApiKeyRecord]:
        ...

    def list_keys(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> list[ApiKeyRecord]:
        ...

    def revoke_key(
        self,
        *,
        key_id: str,
        revoked_at: Optional[datetime] = None,
    ) -> Optional[ApiKeyRecord]:
        ...

    def update_last_used_at(
        self,
        *,
        key_id: str,
        last_used_at: datetime,
    ) -> Optional[ApiKeyRecord]:
        ...


class RevocationStore(Protocol):
    def revoke_jti(
        self,
        *,
        jti: str,
        user_id: str,
        agent_id: str,
        revoked_at: Optional[datetime] = None,
        expires_at: Optional[datetime] = None,
        reason: Optional[str] = None,
    ) -> TokenRevocationRecord:
        ...

    def is_jti_revoked(
        self,
        *,
        jti: str,
        now: Optional[datetime] = None,
    ) -> bool:
        ...

    def list_revocations_by_agent_id(
        self,
        *,
        agent_id: str,
    ) -> list[TokenRevocationRecord]:
        ...

    def delete_expired_revocations(
        self,
        *,
        now: datetime,
    ) -> int:
        ...


class AuditStore(Protocol):
    def append_event(
        self,
        *,
        occurred_at: datetime,
        user_id: str,
        agent_id: str,
        action: str,
        outcome: str,
        request_id: Optional[str] = None,
        details: Optional[dict[str, object]] = None,
    ) -> AuditEventRecord:
        ...

    def list_recent_events(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[AuditEventRecord]:
        ...

    def delete_events_older_than(
        self,
        *,
        cutoff: datetime,
    ) -> int:
        ...

    def delete_oldest_events(
        self,
        *,
        limit: int,
    ) -> int:
        ...
