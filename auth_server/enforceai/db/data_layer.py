from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class EnforceAIStores:
    agent_store: SqliteAgentStore
    api_key_store: SqliteApiKeyStore
    revocation_store: SqliteRevocationStore
    audit_store: SqliteAuditStore


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

    def build_stores(self) -> EnforceAIStores:
        return EnforceAIStores(
            agent_store=SqliteAgentStore(db_path=self._db_path),
            api_key_store=SqliteApiKeyStore(db_path=self._db_path),
            revocation_store=SqliteRevocationStore(db_path=self._db_path),
            audit_store=SqliteAuditStore(db_path=self._db_path),
        )
