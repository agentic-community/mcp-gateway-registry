"""
Unit tests for EnforceAI API key provider (Stage 4.2).
"""

from __future__ import annotations

import hashlib
import hmac
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from auth_server.enforceai.db.migrations import (
    upgrade_to_latest,
)
from auth_server.enforceai.errors import (
    ForbiddenError,
    UnauthorizedError,
)
from auth_server.enforceai.providers.api_key import (
    ApiKeyProvider,
)
from auth_server.enforceai.stores.sqlite.agent_store import (
    SqliteAgentStore,
)
from auth_server.enforceai.stores.sqlite.api_key_store import (
    SqliteApiKeyStore,
)


def _migrate_db(
    db_path: Path,
) -> None:
    connection = sqlite3.connect(db_path)
    try:
        upgrade_to_latest(connection)
    finally:
        connection.close()


def _compute_hash(
    *,
    pepper: bytes,
    secret: str,
) -> str:
    return hmac.new(
        pepper,
        secret.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


@pytest.mark.unit
class TestApiKeyProvider:
    def test_happy_path_scopes_unset_uses_agent_scopes(
        self,
        tmp_path: Path,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        agent_store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)
        api_key_store = SqliteApiKeyStore(db_path=enforceai_sqlite_db_path)

        pepper = b"pepper-1"
        pepper_path = tmp_path / "pepper"
        pepper_path.write_bytes(pepper)

        user_id = "https://issuer.example|sub-1"
        agent_id = str(uuid.uuid4())
        agent_store.create_agent(
            user_id=user_id,
            agent_id=agent_id,
            scopes=[
                "mcp-servers-restricted/read",
                "mcp-tools/call",
            ],
        )

        secret = "supersecret"
        api_key_store.create_key(
            key_id="key-1",
            secret_hash=_compute_hash(
                pepper=pepper,
                secret=secret,
            ),
            user_id=user_id,
            agent_id=agent_id,
            scopes=None,
        )

        provider = ApiKeyProvider(
            api_key_store=api_key_store,
            agent_store=agent_store,
            pepper_path=pepper_path,
        )

        identity = provider.resolve_identity(api_key_value=f"eak_key-1.{secret}")
        assert identity.provider == "api-key"
        assert identity.user_id == user_id
        assert identity.agent_id == agent_id
        assert identity.scopes == [
            "mcp-servers-restricted/read",
            "mcp-tools/call",
        ]
        assert identity.metadata == {"api_key_id": "key-1"}

    def test_happy_path_scopes_intersection(
        self,
        tmp_path: Path,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        agent_store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)
        api_key_store = SqliteApiKeyStore(db_path=enforceai_sqlite_db_path)

        pepper = b"pepper-1"
        pepper_path = tmp_path / "pepper"
        pepper_path.write_bytes(pepper)

        user_id = "https://issuer.example|sub-1"
        agent_id = str(uuid.uuid4())
        agent_store.create_agent(
            user_id=user_id,
            agent_id=agent_id,
            scopes=[
                "a",
                "b",
                "c",
            ],
        )

        secret = "supersecret"
        api_key_store.create_key(
            key_id="key-1",
            secret_hash=_compute_hash(
                pepper=pepper,
                secret=secret,
            ),
            user_id=user_id,
            agent_id=agent_id,
            scopes=[
                "c",
                "a",
            ],
        )

        provider = ApiKeyProvider(
            api_key_store=api_key_store,
            agent_store=agent_store,
            pepper_path=pepper_path,
        )

        identity = provider.resolve_identity(api_key_value=f"eak_key-1.{secret}")
        assert identity.scopes == [
            "a",
            "c",
        ]

    def test_malformed_key_rejected_unauthorized(
        self,
        tmp_path: Path,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        pepper_path = tmp_path / "pepper"
        pepper_path.write_bytes(b"pepper")

        provider = ApiKeyProvider(
            api_key_store=SqliteApiKeyStore(db_path=enforceai_sqlite_db_path),
            agent_store=SqliteAgentStore(db_path=enforceai_sqlite_db_path),
            pepper_path=pepper_path,
        )

        with pytest.raises(UnauthorizedError):
            provider.resolve_identity(api_key_value="not-an-api-key")

    def test_unknown_key_id_rejected_unauthorized(
        self,
        tmp_path: Path,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)

        pepper_path = tmp_path / "pepper"
        pepper_path.write_bytes(b"pepper")

        provider = ApiKeyProvider(
            api_key_store=SqliteApiKeyStore(db_path=enforceai_sqlite_db_path),
            agent_store=SqliteAgentStore(db_path=enforceai_sqlite_db_path),
            pepper_path=pepper_path,
        )

        with pytest.raises(UnauthorizedError):
            provider.resolve_identity(api_key_value="eak_unknown.secret")

    def test_bad_secret_rejected_unauthorized(
        self,
        tmp_path: Path,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        agent_store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)
        api_key_store = SqliteApiKeyStore(db_path=enforceai_sqlite_db_path)

        pepper = b"pepper-1"
        pepper_path = tmp_path / "pepper"
        pepper_path.write_bytes(pepper)

        user_id = "https://issuer.example|sub-1"
        agent_id = str(uuid.uuid4())
        agent_store.create_agent(
            user_id=user_id,
            agent_id=agent_id,
            scopes=["a"],
        )

        api_key_store.create_key(
            key_id="key-1",
            secret_hash=_compute_hash(
                pepper=pepper,
                secret="correct",
            ),
            user_id=user_id,
            agent_id=agent_id,
        )

        provider = ApiKeyProvider(
            api_key_store=api_key_store,
            agent_store=agent_store,
            pepper_path=pepper_path,
        )

        with pytest.raises(UnauthorizedError):
            provider.resolve_identity(api_key_value="eak_key-1.wrong")

    def test_revoked_key_rejected_forbidden(
        self,
        tmp_path: Path,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        agent_store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)
        api_key_store = SqliteApiKeyStore(db_path=enforceai_sqlite_db_path)

        pepper = b"pepper-1"
        pepper_path = tmp_path / "pepper"
        pepper_path.write_bytes(pepper)

        user_id = "https://issuer.example|sub-1"
        agent_id = str(uuid.uuid4())
        agent_store.create_agent(
            user_id=user_id,
            agent_id=agent_id,
            scopes=["a"],
        )

        secret = "supersecret"
        api_key_store.create_key(
            key_id="key-1",
            secret_hash=_compute_hash(
                pepper=pepper,
                secret=secret,
            ),
            user_id=user_id,
            agent_id=agent_id,
        )
        api_key_store.revoke_key(key_id="key-1")

        provider = ApiKeyProvider(
            api_key_store=api_key_store,
            agent_store=agent_store,
            pepper_path=pepper_path,
        )

        with pytest.raises(ForbiddenError):
            provider.resolve_identity(api_key_value=f"eak_key-1.{secret}")

    def test_expired_key_rejected_forbidden(
        self,
        tmp_path: Path,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        agent_store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)
        api_key_store = SqliteApiKeyStore(db_path=enforceai_sqlite_db_path)

        pepper = b"pepper-1"
        pepper_path = tmp_path / "pepper"
        pepper_path.write_bytes(pepper)

        user_id = "https://issuer.example|sub-1"
        agent_id = str(uuid.uuid4())
        agent_store.create_agent(
            user_id=user_id,
            agent_id=agent_id,
            scopes=["a"],
        )

        secret = "supersecret"
        api_key_store.create_key(
            key_id="key-1",
            secret_hash=_compute_hash(
                pepper=pepper,
                secret=secret,
            ),
            user_id=user_id,
            agent_id=agent_id,
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )

        provider = ApiKeyProvider(
            api_key_store=api_key_store,
            agent_store=agent_store,
            pepper_path=pepper_path,
        )

        with pytest.raises(ForbiddenError):
            provider.resolve_identity(api_key_value=f"eak_key-1.{secret}")

    def test_agent_revoked_rejected_forbidden(
        self,
        tmp_path: Path,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        agent_store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)
        api_key_store = SqliteApiKeyStore(db_path=enforceai_sqlite_db_path)

        pepper = b"pepper-1"
        pepper_path = tmp_path / "pepper"
        pepper_path.write_bytes(pepper)

        user_id = "https://issuer.example|sub-1"
        agent_id = str(uuid.uuid4())
        agent_store.create_agent(
            user_id=user_id,
            agent_id=agent_id,
            scopes=["a"],
        )
        agent_store.revoke_agent(agent_id=agent_id)

        secret = "supersecret"
        api_key_store.create_key(
            key_id="key-1",
            secret_hash=_compute_hash(
                pepper=pepper,
                secret=secret,
            ),
            user_id=user_id,
            agent_id=agent_id,
        )

        provider = ApiKeyProvider(
            api_key_store=api_key_store,
            agent_store=agent_store,
            pepper_path=pepper_path,
        )

        with pytest.raises(ForbiddenError):
            provider.resolve_identity(api_key_value=f"eak_key-1.{secret}")

    def test_error_messages_do_not_leak_api_key_material(
        self,
        tmp_path: Path,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        agent_store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)
        api_key_store = SqliteApiKeyStore(db_path=enforceai_sqlite_db_path)

        pepper_path = tmp_path / "pepper"
        pepper_path.write_bytes(b"pepper-1")

        provider = ApiKeyProvider(
            api_key_store=api_key_store,
            agent_store=agent_store,
            pepper_path=pepper_path,
        )

        raw_key = "eak_key-1.supersecret"
        with pytest.raises(UnauthorizedError) as exc:
            provider.resolve_identity(api_key_value=raw_key)

        message = str(exc.value)
        assert "supersecret" not in message
        assert raw_key not in message
