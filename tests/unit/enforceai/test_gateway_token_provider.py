"""
Unit tests for EnforceAI gateway token provider (Stage 4.3).
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from auth_server.enforceai.crypto.keyring import (
    GatewayKeyring,
)
from auth_server.enforceai.db.migrations import (
    upgrade_to_latest,
)
from auth_server.enforceai.errors import (
    DependencyUnavailableError,
    ForbiddenError,
)
from auth_server.enforceai.providers.gateway_token import (
    GatewayTokenProvider,
)
from auth_server.enforceai.stores.sqlite.agent_store import (
    SqliteAgentStore,
)
from auth_server.enforceai.stores.sqlite.revocation_store import (
    SqliteRevocationStore,
)
from auth_server.enforceai.tokens.mint import (
    mint_gateway_token,
)


def _migrate_db(
    db_path: Path,
) -> None:
    connection = sqlite3.connect(db_path)
    try:
        upgrade_to_latest(connection)
    finally:
        connection.close()


@pytest.mark.unit
class TestGatewayTokenProvider:
    def test_happy_path(
        self,
        enforceai_sqlite_db_path: Path,
        enforceai_gateway_key_files,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        agent_store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)
        revocation_store = SqliteRevocationStore(db_path=enforceai_sqlite_db_path)

        key_files = enforceai_gateway_key_files
        keyring = GatewayKeyring.load(
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
        )

        now = datetime.now(timezone.utc).replace(microsecond=0)
        user_id = "https://issuer.example|sub-1"
        agent_id = str(uuid.uuid4())
        agent_store.create_agent(
            user_id=user_id,
            agent_id=agent_id,
            scopes=[
                "a",
                "b",
            ],
        )

        token = mint_gateway_token(
            keyring=keyring,
            issuer="enforceai-gateway",
            user_id=user_id,
            agent_id=agent_id,
            scopes=[
                "b",
                "c",
            ],
            issued_at=now,
            ttl_seconds=3600,
            jti="jti-1",
        )

        provider = GatewayTokenProvider(
            agent_store=agent_store,
            revocation_store=revocation_store,
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
            expected_issuer="enforceai-gateway",
        )

        identity = provider.resolve_identity(
            token=token,
            now=now,
        )
        assert identity.provider == "gateway-token"
        assert identity.user_id == user_id
        assert identity.agent_id == agent_id
        assert identity.scopes == ["b"]
        assert identity.metadata == {
            "issuer": "enforceai-gateway",
            "jti": "jti-1",
        }

    def test_agent_revoked_denied(
        self,
        enforceai_sqlite_db_path: Path,
        enforceai_gateway_key_files,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        agent_store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)
        revocation_store = SqliteRevocationStore(db_path=enforceai_sqlite_db_path)

        key_files = enforceai_gateway_key_files
        keyring = GatewayKeyring.load(
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
        )

        now = datetime.now(timezone.utc).replace(microsecond=0)
        user_id = "https://issuer.example|sub-1"
        agent_id = str(uuid.uuid4())
        agent_store.create_agent(
            user_id=user_id,
            agent_id=agent_id,
            scopes=["a"],
        )
        agent_store.revoke_agent(agent_id=agent_id)

        token = mint_gateway_token(
            keyring=keyring,
            issuer="enforceai-gateway",
            user_id=user_id,
            agent_id=agent_id,
            scopes=["a"],
            issued_at=now,
            ttl_seconds=3600,
            jti="jti-1",
        )

        provider = GatewayTokenProvider(
            agent_store=agent_store,
            revocation_store=revocation_store,
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
            expected_issuer="enforceai-gateway",
        )

        with pytest.raises(ForbiddenError, match="Agent revoked"):
            provider.resolve_identity(token=token, now=now)

    def test_jti_revoked_denied(
        self,
        enforceai_sqlite_db_path: Path,
        enforceai_gateway_key_files,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        agent_store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)
        revocation_store = SqliteRevocationStore(db_path=enforceai_sqlite_db_path)

        key_files = enforceai_gateway_key_files
        keyring = GatewayKeyring.load(
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
        )

        now = datetime.now(timezone.utc).replace(microsecond=0)
        user_id = "https://issuer.example|sub-1"
        agent_id = str(uuid.uuid4())
        agent_store.create_agent(
            user_id=user_id,
            agent_id=agent_id,
            scopes=["a"],
        )

        token = mint_gateway_token(
            keyring=keyring,
            issuer="enforceai-gateway",
            user_id=user_id,
            agent_id=agent_id,
            scopes=["a"],
            issued_at=now,
            ttl_seconds=3600,
            jti="jti-1",
        )
        revocation_store.revoke_jti(
            jti="jti-1",
            user_id=user_id,
            agent_id=agent_id,
            revoked_at=now,
        )

        provider = GatewayTokenProvider(
            agent_store=agent_store,
            revocation_store=revocation_store,
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
            expected_issuer="enforceai-gateway",
        )

        with pytest.raises(ForbiddenError, match="Token revoked"):
            provider.resolve_identity(token=token, now=now)

    def test_tokens_valid_after_denied(
        self,
        enforceai_sqlite_db_path: Path,
        enforceai_gateway_key_files,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        agent_store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)
        revocation_store = SqliteRevocationStore(db_path=enforceai_sqlite_db_path)

        key_files = enforceai_gateway_key_files
        keyring = GatewayKeyring.load(
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
        )

        now = datetime.now(timezone.utc).replace(microsecond=0)
        issued_at = now - timedelta(seconds=30)
        user_id = "https://issuer.example|sub-1"
        agent_id = str(uuid.uuid4())
        agent_store.create_agent(
            user_id=user_id,
            agent_id=agent_id,
            scopes=["a"],
        )
        agent_store.bump_tokens_valid_after(
            agent_id=agent_id,
            tokens_valid_after=now,
        )

        token = mint_gateway_token(
            keyring=keyring,
            issuer="enforceai-gateway",
            user_id=user_id,
            agent_id=agent_id,
            scopes=["a"],
            issued_at=issued_at,
            ttl_seconds=3600,
            jti="jti-1",
        )

        provider = GatewayTokenProvider(
            agent_store=agent_store,
            revocation_store=revocation_store,
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
            expected_issuer="enforceai-gateway",
        )

        with pytest.raises(ForbiddenError, match="Token revoked"):
            provider.resolve_identity(token=token, now=now)

    def test_agent_ownership_mismatch_denied(
        self,
        enforceai_sqlite_db_path: Path,
        enforceai_gateway_key_files,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        agent_store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)
        revocation_store = SqliteRevocationStore(db_path=enforceai_sqlite_db_path)

        key_files = enforceai_gateway_key_files
        keyring = GatewayKeyring.load(
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
        )

        now = datetime.now(timezone.utc).replace(microsecond=0)
        token_user_id = "https://issuer.example|sub-1"
        agent_id = str(uuid.uuid4())
        agent_store.create_agent(
            user_id="https://issuer.example|other",
            agent_id=agent_id,
            scopes=["a"],
        )

        token = mint_gateway_token(
            keyring=keyring,
            issuer="enforceai-gateway",
            user_id=token_user_id,
            agent_id=agent_id,
            scopes=["a"],
            issued_at=now,
            ttl_seconds=3600,
            jti="jti-1",
        )

        provider = GatewayTokenProvider(
            agent_store=agent_store,
            revocation_store=revocation_store,
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
            expected_issuer="enforceai-gateway",
        )

        with pytest.raises(ForbiddenError, match="ownership mismatch"):
            provider.resolve_identity(token=token, now=now)

    def test_keyring_load_failure_maps_to_dependency_unavailable(
        self,
        enforceai_sqlite_db_path: Path,
        enforceai_gateway_key_files,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        agent_store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)
        revocation_store = SqliteRevocationStore(db_path=enforceai_sqlite_db_path)

        key_files = enforceai_gateway_key_files
        provider = GatewayTokenProvider(
            agent_store=agent_store,
            revocation_store=revocation_store,
            private_key_path=key_files.private_key_path,
            public_keys_dir=(key_files.public_keys_dir / "missing"),
            active_kid=key_files.active_kid,
            expected_issuer="enforceai-gateway",
        )

        with pytest.raises(DependencyUnavailableError, match="keyring unavailable"):
            provider.resolve_identity(token="dummy")

