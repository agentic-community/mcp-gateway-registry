from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .connection import (
    sqlite_connection,
)
from .migrations import (
    DEFAULT_MIGRATIONS_SQL_DIR,
    upgrade_to_latest,
)
from ..stores.sqlite.agent_store import (
    SqliteAgentStore,
)
from ..stores.sqlite.api_key_store import (
    SqliteApiKeyStore,
)
from ..stores.sqlite.audit_store import (
    SqliteAuditStore,
)
from ..stores.sqlite.revocation_store import (
    SqliteRevocationStore,
)
from ..stores.sqlite.session_store import (
    SqliteSessionStore,
)
from ..stores.sqlite.user_store import (
    SqliteUserStore,
)
from ..stores.sqlite.egress_allowlist_store import (
    SqliteEgressAllowlistStore,
)
from ..stores.sqlite.upstream_credential_store import (
    SqliteUpstreamCredentialStore,
)
from ..stores.sqlite.upstream_oauth_state_store import (
    SqliteUpstreamOAuthStateStore,
)


@dataclass(frozen=True)
class EnforceAIStores:
    agent_store: SqliteAgentStore
    api_key_store: SqliteApiKeyStore
    revocation_store: SqliteRevocationStore
    audit_store: SqliteAuditStore
    user_store: SqliteUserStore
    session_store: SqliteSessionStore
    egress_allowlist_store: SqliteEgressAllowlistStore
    upstream_credential_store: Optional[SqliteUpstreamCredentialStore]
    upstream_oauth_state_store: Optional[SqliteUpstreamOAuthStateStore]


class EnforceAIDataLayer:
    def __init__(
        self,
        *,
        db_path: Path,
        migrations_sql_dir: Path = DEFAULT_MIGRATIONS_SQL_DIR,
    ) -> None:
        self._db_path = db_path
        self._migrations_sql_dir = migrations_sql_dir

    @property
    def db_path(self) -> Path:
        return self._db_path

    def initialize(self) -> None:
        self._db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        with sqlite_connection(self._db_path) as connection:
            upgrade_to_latest(
                connection,
                migrations_sql_dir=self._migrations_sql_dir,
            )

    def build_stores(
        self,
        *,
        upstream_kek: Optional[bytes] = None,
    ) -> EnforceAIStores:
        upstream_credential_store: Optional[SqliteUpstreamCredentialStore] = None
        upstream_oauth_state_store: Optional[SqliteUpstreamOAuthStateStore] = None
        if upstream_kek is not None:
            upstream_credential_store = SqliteUpstreamCredentialStore(
                db_path=self._db_path,
                kek=upstream_kek,
            )
            upstream_oauth_state_store = SqliteUpstreamOAuthStateStore(
                db_path=self._db_path,
                kek=upstream_kek,
            )

        return EnforceAIStores(
            agent_store=SqliteAgentStore(db_path=self._db_path),
            api_key_store=SqliteApiKeyStore(db_path=self._db_path),
            revocation_store=SqliteRevocationStore(db_path=self._db_path),
            audit_store=SqliteAuditStore(db_path=self._db_path),
            user_store=SqliteUserStore(db_path=self._db_path),
            session_store=SqliteSessionStore(db_path=self._db_path),
            egress_allowlist_store=SqliteEgressAllowlistStore(db_path=self._db_path),
            upstream_credential_store=upstream_credential_store,
            upstream_oauth_state_store=upstream_oauth_state_store,
        )
