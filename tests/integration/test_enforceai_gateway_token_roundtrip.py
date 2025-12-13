"""
Integration-lite test for EnforceAI gateway token mint + verify roundtrip.

No FastAPI wiring and no network access; validates crypto/keyring/token primitives
work together with realistic inputs.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from auth_server.enforceai.crypto.keyring import (
    GatewayKeyring,
)
from auth_server.enforceai.errors import (
    UnauthorizedError,
)
from auth_server.enforceai.tokens.mint import (
    mint_gateway_token,
)
from auth_server.enforceai.tokens.verify import (
    verify_gateway_token,
)


@pytest.mark.integration
class TestEnforceAIGatewayTokenRoundtrip:
    def test_mint_and_verify_roundtrip(
        self,
        enforceai_gateway_key_files,
    ) -> None:
        key_files = enforceai_gateway_key_files
        keyring = GatewayKeyring.load(
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
        )

        now = datetime.now(timezone.utc).replace(microsecond=0)
        agent_id = str(uuid.uuid4())

        token = mint_gateway_token(
            keyring=keyring,
            issuer="enforceai-gateway",
            user_id="https://issuer.example|sub-1",
            agent_id=agent_id,
            scopes=["mcp-servers-restricted/read"],
            issued_at=now,
            ttl_seconds=3600,
            jti="jti-1",
        )

        claims = verify_gateway_token(
            token,
            keyring=keyring,
            now=now,
            expected_issuer="enforceai-gateway",
        )
        assert claims.agent_id == agent_id
        assert claims.jti == "jti-1"

    def test_issuer_mismatch_rejected(
        self,
        enforceai_gateway_key_files,
    ) -> None:
        key_files = enforceai_gateway_key_files
        keyring = GatewayKeyring.load(
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
        )
        now = datetime.now(timezone.utc).replace(microsecond=0)

        token = mint_gateway_token(
            keyring=keyring,
            issuer="enforceai-gateway",
            user_id="https://issuer.example|sub-1",
            agent_id=str(uuid.uuid4()),
            scopes=["s1"],
            issued_at=now,
            ttl_seconds=3600,
            jti="jti-1",
        )

        with pytest.raises(UnauthorizedError, match="issuer mismatch"):
            verify_gateway_token(
                token,
                keyring=keyring,
                now=now,
                expected_issuer="different-gateway",
            )

    def test_expired_within_leeway_is_accepted(
        self,
        enforceai_gateway_key_files,
    ) -> None:
        key_files = enforceai_gateway_key_files
        keyring = GatewayKeyring.load(
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
        )

        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        token = mint_gateway_token(
            keyring=keyring,
            issuer="enforceai-gateway",
            user_id="https://issuer.example|sub-1",
            agent_id=str(uuid.uuid4()),
            scopes=["s1"],
            issued_at=now - timedelta(hours=1),
            expires_at=now - timedelta(seconds=30),
            jti="jti-1",
        )

        verify_gateway_token(
            token,
            keyring=keyring,
            now=now,
            clock_skew_seconds=60,
        )

