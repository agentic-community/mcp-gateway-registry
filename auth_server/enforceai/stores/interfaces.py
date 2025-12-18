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
from ..models.user import (
    UserRecord,
)
from ..models.session import (
    SessionRecord,
)
from ..models.upstream_credentials import (
    UpstreamCredentialRecord,
    UpstreamCredentialSecret,
)
from ..models.egress_allowlist import (
    EgressAllowlistEntryRecord,
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


class UpstreamCredentialStore(Protocol):
    def create_credential(
        self,
        *,
        server_path: str,
        credential_type: str,
        credential_binding: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        provider: Optional[str] = None,
        scopes: Optional[list[str]] = None,
        token_type: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        secret_payload: Optional[dict[str, object]] = None,
    ) -> UpstreamCredentialRecord:
        ...

    def get_credential_by_id(
        self,
        *,
        credential_id: str,
    ) -> Optional[UpstreamCredentialRecord]:
        ...

    def get_credential_secret(
        self,
        *,
        credential_id: str,
    ) -> Optional[UpstreamCredentialSecret]:
        ...

    def list_credentials(
        self,
        *,
        server_path: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        include_revoked: bool = False,
    ) -> list[UpstreamCredentialRecord]:
        ...

    def revoke_credential(
        self,
        *,
        credential_id: str,
        revoked_at: Optional[datetime] = None,
    ) -> Optional[UpstreamCredentialRecord]:
        ...

    def update_last_used_at(
        self,
        *,
        credential_id: str,
        last_used_at: datetime,
    ) -> Optional[UpstreamCredentialRecord]:
        ...


class EgressAllowlistStore(Protocol):
    def create_entry(
        self,
        *,
        kind: str,
        value: str,
        comment: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ) -> EgressAllowlistEntryRecord:
        ...

    def get_entry_by_id(
        self,
        *,
        entry_id: int,
    ) -> Optional[EgressAllowlistEntryRecord]:
        ...

    def list_entries(
        self,
        *,
        include_expired: bool = False,
        now: Optional[datetime] = None,
    ) -> list[EgressAllowlistEntryRecord]:
        ...

    def update_entry(
        self,
        *,
        entry_id: int,
        kind: Optional[str] = None,
        value: Optional[str] = None,
        comment: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ) -> Optional[EgressAllowlistEntryRecord]:
        ...

    def delete_entry(
        self,
        *,
        entry_id: int,
    ) -> bool:
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


class UserStore(Protocol):
    def upsert_oidc_user(
        self,
        *,
        user_id: str,
        email: str,
        role: str = "user",
        now: Optional[datetime] = None,
    ) -> UserRecord:
        ...

    def create_local_user(
        self,
        *,
        username: str,
        email: str,
        password_hash: str,
        role: str = "user",
        now: Optional[datetime] = None,
    ) -> UserRecord:
        ...

    def get_user_by_id(
        self,
        *,
        user_id: str,
    ) -> Optional[UserRecord]:
        ...

    def get_user_by_username(
        self,
        *,
        username: str,
    ) -> Optional[UserRecord]:
        ...

    def search_users(
        self,
        *,
        query: str,
        limit: int = 50,
    ) -> list[UserRecord]:
        ...

    def disable_user(
        self,
        *,
        user_id: str,
        disabled_at: Optional[datetime] = None,
    ) -> Optional[UserRecord]:
        ...

    def update_password_hash(
        self,
        *,
        user_id: str,
        password_hash: str,
        updated_at: Optional[datetime] = None,
    ) -> Optional[UserRecord]:
        ...


class SessionStore(Protocol):
    def create_session(
        self,
        *,
        session_id: str,
        user_id: str,
        auth_method: str,
        expires_at: datetime,
        now: Optional[datetime] = None,
    ) -> SessionRecord:
        ...

    def get_session_by_id(
        self,
        *,
        session_id: str,
    ) -> Optional[SessionRecord]:
        ...

    def touch_session(
        self,
        *,
        session_id: str,
        now: datetime,
    ) -> Optional[SessionRecord]:
        ...

    def revoke_session(
        self,
        *,
        session_id: str,
        revoked_at: Optional[datetime] = None,
        revoked_reason: Optional[str] = None,
    ) -> Optional[SessionRecord]:
        ...
