"""
Unit tests for EnforceAI management service layer (Stage 6.1).
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
from auth_server.enforceai.fgac.models import (
    ScopeCatalog,
    ScopeDefinition,
)
from auth_server.enforceai.management.service import (
    ManagementService,
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
from auth_server.enforceai.stores.sqlite.revocation_store import (
    SqliteRevocationStore,
)
from auth_server.enforceai.tokens.verify import (
    verify_gateway_token,
)


def _migrate_db(
    db_path: Path,
) -> None:
    connection = sqlite3.connect(db_path)
    try:
        upgrade_to_latest(connection)
    finally:
        connection.close()


def _synthetic_catalog(
    *,
    tmp_path: Path,
    scope_names: list[str],
) -> ScopeCatalog:
    scopes: dict[str, ScopeDefinition] = {}
    for name in scope_names:
        scopes[name] = ScopeDefinition(
            name=name,
            server_permissions=tuple(),
            agent_permissions=tuple(),
        )

    return ScopeCatalog(
        path=tmp_path / "scopes.yml",
        ui_scopes={},
        group_mappings={},
        scopes=scopes,
    )


@pytest.mark.unit
class TestManagementService:
    def test_agent_crud_and_ownership_enforced(
        self,
        tmp_path: Path,
        enforceai_sqlite_db_path: Path,
        enforceai_gateway_key_files,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        agent_store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)
        api_key_store = SqliteApiKeyStore(db_path=enforceai_sqlite_db_path)
        revocation_store = SqliteRevocationStore(db_path=enforceai_sqlite_db_path)

        keyring = GatewayKeyring.load(
            private_key_path=enforceai_gateway_key_files.private_key_path,
            public_keys_dir=enforceai_gateway_key_files.public_keys_dir,
            active_kid=enforceai_gateway_key_files.active_kid,
        )

        catalog = _synthetic_catalog(
            tmp_path=tmp_path,
            scope_names=[
                "a",
                "b",
                "c",
            ],
        )

        pepper = b"pepper-1"
        service = ManagementService(
            agent_store=agent_store,
            api_key_store=api_key_store,
            revocation_store=revocation_store,
            scope_catalog=catalog,
            api_key_pepper=pepper,
            gateway_keyring=keyring,
            gateway_issuer="enforceai-gateway",
        )

        user_1 = "https://issuer.example|sub-1"
        user_2 = "https://issuer.example|sub-2"

        agent = service.create_agent(
            user_id=user_1,
            scopes=["a", "b"],
            alias="agent-1",
        )

        agents = service.list_agents(user_id=user_1)
        assert [item.agent_id for item in agents] == [agent.agent_id]

        fetched = service.get_agent(
            user_id=user_1,
            agent_id=agent.agent_id,
        )
        assert fetched.agent_id == agent.agent_id

        with pytest.raises(ForbiddenError):
            service.get_agent(
                user_id=user_2,
                agent_id=agent.agent_id,
            )

        updated = service.update_agent(
            user_id=user_1,
            agent_id=agent.agent_id,
            alias="agent-1b",
            scopes=["a"],
        )
        assert updated.alias == "agent-1b"
        assert updated.scopes == ["a"]

        with pytest.raises(ValueError, match="Unknown scopes"):
            service.create_agent(
                user_id=user_1,
                scopes=["unknown-scope"],
            )

        revoked = service.revoke_agent(
            user_id=user_1,
            agent_id=agent.agent_id,
        )
        assert revoked.revoked_at is not None

        with pytest.raises(ForbiddenError, match="Agent revoked"):
            service.update_agent(
                user_id=user_1,
                agent_id=agent.agent_id,
                alias="nope",
            )

    def test_api_key_create_list_revoke_and_secret_returned_once(
        self,
        tmp_path: Path,
        enforceai_sqlite_db_path: Path,
        enforceai_gateway_key_files,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        agent_store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)
        api_key_store = SqliteApiKeyStore(db_path=enforceai_sqlite_db_path)
        revocation_store = SqliteRevocationStore(db_path=enforceai_sqlite_db_path)

        catalog = _synthetic_catalog(
            tmp_path=tmp_path,
            scope_names=["a", "b", "c"],
        )

        pepper = b"pepper-1"
        pepper_path = tmp_path / "pepper"
        pepper_path.write_bytes(pepper)

        keyring = GatewayKeyring.load(
            private_key_path=enforceai_gateway_key_files.private_key_path,
            public_keys_dir=enforceai_gateway_key_files.public_keys_dir,
            active_kid=enforceai_gateway_key_files.active_kid,
        )

        service = ManagementService(
            agent_store=agent_store,
            api_key_store=api_key_store,
            revocation_store=revocation_store,
            scope_catalog=catalog,
            api_key_pepper=pepper,
            gateway_keyring=keyring,
            gateway_issuer="enforceai-gateway",
        )

        user_id = "https://issuer.example|sub-1"
        agent_id = service.create_agent(
            user_id=user_id,
            scopes=["a", "b"],
        ).agent_id

        key_id, secret, api_key_value = service.create_api_key(
            user_id=user_id,
            agent_id=agent_id,
            scopes=["a"],
        )
        assert key_id
        assert secret
        assert api_key_value.startswith("eak_")
        assert f"{key_id}." in api_key_value

        provider = ApiKeyProvider(
            api_key_store=api_key_store,
            agent_store=agent_store,
            pepper_path=pepper_path,
        )
        identity = provider.resolve_identity(api_key_value=api_key_value)
        assert identity.user_id == user_id
        assert identity.agent_id == agent_id
        assert identity.scopes == ["a"]

        summaries = service.list_api_keys(
            user_id=user_id,
            agent_id=agent_id,
        )
        assert [item.key_id for item in summaries] == [key_id]
        assert all(not hasattr(item, "secret_hash") for item in summaries)
        assert all(not hasattr(item, "secret") for item in summaries)

        revoked = service.revoke_api_key(
            user_id=user_id,
            key_id=key_id,
        )
        assert revoked.key_id == key_id
        assert revoked.revoked_at is not None

        with pytest.raises(ForbiddenError, match="API key scopes exceed agent scopes"):
            service.create_api_key(
                user_id=user_id,
                agent_id=agent_id,
                scopes=["c"],
            )

        with pytest.raises(ForbiddenError):
            service.revoke_api_key(
                user_id="https://issuer.example|sub-2",
                key_id=key_id,
            )

    def test_gateway_token_mint_scope_subset_enforced(
        self,
        tmp_path: Path,
        enforceai_sqlite_db_path: Path,
        enforceai_gateway_key_files,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        agent_store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)
        api_key_store = SqliteApiKeyStore(db_path=enforceai_sqlite_db_path)
        revocation_store = SqliteRevocationStore(db_path=enforceai_sqlite_db_path)

        keyring = GatewayKeyring.load(
            private_key_path=enforceai_gateway_key_files.private_key_path,
            public_keys_dir=enforceai_gateway_key_files.public_keys_dir,
            active_kid=enforceai_gateway_key_files.active_kid,
        )

        catalog = _synthetic_catalog(
            tmp_path=tmp_path,
            scope_names=["a", "b", "c"],
        )

        service = ManagementService(
            agent_store=agent_store,
            api_key_store=api_key_store,
            revocation_store=revocation_store,
            scope_catalog=catalog,
            api_key_pepper=b"pepper-1",
            gateway_keyring=keyring,
            gateway_issuer="enforceai-gateway",
        )

        user_id = "https://issuer.example|sub-1"
        agent_id = service.create_agent(
            user_id=user_id,
            scopes=["a", "b"],
        ).agent_id

        token = service.mint_gateway_token(
            user_id=user_id,
            agent_id=agent_id,
            scopes=["a"],
            ttl_seconds=60,
            jti=str(uuid.uuid4()),
        )
        assert isinstance(token, str)
        assert token

        claims = verify_gateway_token(
            token,
            keyring=keyring,
            expected_issuer="enforceai-gateway",
        )
        assert claims.sub == user_id
        assert claims.agent_id == agent_id
        assert claims.scopes == ["a"]

        with pytest.raises(ForbiddenError, match="Token scopes exceed agent scopes"):
            service.mint_gateway_token(
                user_id=user_id,
                agent_id=agent_id,
                scopes=["c"],
                ttl_seconds=60,
            )

    def test_revoke_all_bumps_tokens_valid_after(
        self,
        tmp_path: Path,
        enforceai_sqlite_db_path: Path,
        enforceai_gateway_key_files,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        agent_store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)
        api_key_store = SqliteApiKeyStore(db_path=enforceai_sqlite_db_path)
        revocation_store = SqliteRevocationStore(db_path=enforceai_sqlite_db_path)

        keyring = GatewayKeyring.load(
            private_key_path=enforceai_gateway_key_files.private_key_path,
            public_keys_dir=enforceai_gateway_key_files.public_keys_dir,
            active_kid=enforceai_gateway_key_files.active_kid,
        )

        catalog = _synthetic_catalog(
            tmp_path=tmp_path,
            scope_names=["a"],
        )

        service = ManagementService(
            agent_store=agent_store,
            api_key_store=api_key_store,
            revocation_store=revocation_store,
            scope_catalog=catalog,
            api_key_pepper=b"pepper-1",
            gateway_keyring=keyring,
            gateway_issuer="enforceai-gateway",
        )

        user_id = "https://issuer.example|sub-1"
        agent_id = service.create_agent(
            user_id=user_id,
            scopes=["a"],
        ).agent_id

        before = agent_store.get_agent_by_id(agent_id=agent_id)
        assert before is not None
        assert before.tokens_valid_after is None

        bumped = service.revoke_all_tokens(
            user_id=user_id,
            agent_id=agent_id,
        )
        assert bumped.tokens_valid_after is not None

        now = datetime.now(timezone.utc).replace(microsecond=0)
        bumped_again = service.revoke_all_tokens(
            user_id=user_id,
            agent_id=agent_id,
            now=now + timedelta(seconds=10),
        )
        assert bumped_again.tokens_valid_after is not None
        assert bumped_again.tokens_valid_after >= bumped.tokens_valid_after

    def test_revoke_token_jti_creates_revocation_record(
        self,
        tmp_path: Path,
        enforceai_sqlite_db_path: Path,
        enforceai_gateway_key_files,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        agent_store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)
        api_key_store = SqliteApiKeyStore(db_path=enforceai_sqlite_db_path)
        revocation_store = SqliteRevocationStore(db_path=enforceai_sqlite_db_path)

        keyring = GatewayKeyring.load(
            private_key_path=enforceai_gateway_key_files.private_key_path,
            public_keys_dir=enforceai_gateway_key_files.public_keys_dir,
            active_kid=enforceai_gateway_key_files.active_kid,
        )

        catalog = _synthetic_catalog(
            tmp_path=tmp_path,
            scope_names=["a"],
        )

        service = ManagementService(
            agent_store=agent_store,
            api_key_store=api_key_store,
            revocation_store=revocation_store,
            scope_catalog=catalog,
            api_key_pepper=b"pepper-1",
            gateway_keyring=keyring,
            gateway_issuer="enforceai-gateway",
        )

        user_id = "https://issuer.example|sub-1"
        agent_id = service.create_agent(
            user_id=user_id,
            scopes=["a"],
        ).agent_id

        jti = "jti-1"
        record = service.revoke_token_jti(
            user_id=user_id,
            agent_id=agent_id,
            jti=jti,
        )
        assert record.jti == jti
        assert record.user_id == user_id
        assert record.agent_id == agent_id
        assert revocation_store.is_jti_revoked(jti=jti) is True

    def test_dependency_failures_map_to_dependency_unavailable(
        self,
        tmp_path: Path,
    ) -> None:
        class ExplodingAgentStore:
            def list_agents_by_user_id(self, *, user_id: str):  # type: ignore[no-untyped-def]
                raise RuntimeError("boom")

            def get_agent_by_id(self, *, agent_id: str):  # type: ignore[no-untyped-def]
                raise RuntimeError("boom")

        catalog = _synthetic_catalog(
            tmp_path=tmp_path,
            scope_names=["a"],
        )

        service = ManagementService(
            agent_store=ExplodingAgentStore(),
            api_key_store=SqliteApiKeyStore(db_path=tmp_path / "db.sqlite"),
            revocation_store=SqliteRevocationStore(db_path=tmp_path / "db.sqlite"),
            scope_catalog=catalog,
            api_key_pepper=b"pepper-1",
        )

        with pytest.raises(DependencyUnavailableError):
            service.list_agents(user_id="https://issuer.example|sub-1")
