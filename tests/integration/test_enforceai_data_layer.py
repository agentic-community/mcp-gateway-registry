"""
Integration test for EnforceAI data layer initialization.

This is intentionally "integration-lite": it validates the settings -> data layer
-> stores wiring without putting EnforceAI on the request path yet.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from auth_server.enforceai.config import (
    EnforceAISettings,
)
from auth_server.enforceai.db.data_layer import (
    EnforceAIDataLayer,
)


@pytest.mark.integration
class TestEnforceAIDataLayerIntegration:
    def test_settings_to_data_layer_roundtrip(
        self,
        enforceai_env,
        enforceai_oidc_issuers_env_json: str,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        enforceai_env(
            {
                "OIDC_ISSUERS": enforceai_oidc_issuers_env_json,
                "ENFORCEAI_DB_PATH": str(enforceai_sqlite_db_path),
            }
        )
        settings = EnforceAISettings(_env_file=None)

        layer = EnforceAIDataLayer(db_path=settings.db_path)
        layer.initialize()
        stores = layer.build_stores()

        user_id = "https://issuer.example|sub-1"
        agent_id = str(uuid.uuid4())

        stores.agent_store.create_agent(
            user_id=user_id,
            agent_id=agent_id,
            scopes=["mcp-servers-restricted/read"],
        )
        stores.api_key_store.create_key(
            key_id="key-1",
            secret_hash="hash-1",
            user_id=user_id,
            agent_id=agent_id,
        )
        stores.revocation_store.revoke_jti(
            jti="jti-1",
            user_id=user_id,
            agent_id=agent_id,
        )
        stores.audit_store.append_event(
            occurred_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            user_id=user_id,
            agent_id=agent_id,
            action="auth/validate",
            outcome="allow",
        )

        assert stores.agent_store.get_agent_by_id(agent_id=agent_id) is not None
        assert stores.api_key_store.get_key_by_id(key_id="key-1") is not None
        assert stores.revocation_store.is_jti_revoked(jti="jti-1") is True
        assert stores.audit_store.list_recent_events(user_id=user_id, limit=10)

