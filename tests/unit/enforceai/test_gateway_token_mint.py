"""
Unit tests for EnforceAI gateway token minting (Stage 2.3).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import jwt
import pytest

from auth_server.enforceai.crypto.keyring import (
    GatewayKeyring,
)
from auth_server.enforceai.tokens.mint import (
    mint_gateway_token,
)


@pytest.mark.unit
class TestGatewayTokenMint:
    def test_mints_rs256_token_with_kid_header(
        self,
        enforceai_gateway_key_files,
    ) -> None:
        key_files = enforceai_gateway_key_files
        keyring = GatewayKeyring.load(
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
        )

        issued_at = datetime.now(timezone.utc).replace(microsecond=0)
        token = mint_gateway_token(
            keyring=keyring,
            issuer="enforceai-gateway",
            user_id="https://issuer.example|sub-1",
            agent_id=str(uuid.uuid4()),
            scopes=["mcp-servers-restricted/read"],
            issued_at=issued_at,
            ttl_seconds=3600,
            jti="jti-1",
        )

        header = jwt.get_unverified_header(token)
        assert header["kid"] == key_files.active_kid
        assert header["alg"] == "RS256"

        public_key = keyring.get_public_key(kid=key_files.active_kid)
        assert public_key is not None

        decoded = jwt.decode(
            token,
            key=public_key,
            algorithms=["RS256"],
            options={"verify_aud": False},
        )
        assert decoded["iss"] == "enforceai-gateway"
        assert decoded["sub"] == "https://issuer.example|sub-1"
        assert decoded["jti"] == "jti-1"
        assert decoded["scopes"] == ["mcp-servers-restricted/read"]

    def test_rejects_empty_scopes(
        self,
        enforceai_gateway_key_files,
    ) -> None:
        key_files = enforceai_gateway_key_files
        keyring = GatewayKeyring.load(
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
        )

        with pytest.raises(ValueError, match="scopes must be a non-empty list"):
            mint_gateway_token(
                keyring=keyring,
                issuer="enforceai-gateway",
                user_id="https://issuer.example|sub-1",
                agent_id=str(uuid.uuid4()),
                scopes=[],
                issued_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
                ttl_seconds=3600,
            )

    def test_rejects_both_expires_at_and_ttl(self, enforceai_gateway_key_files) -> None:
        key_files = enforceai_gateway_key_files
        keyring = GatewayKeyring.load(
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
        )
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)

        with pytest.raises(ValueError, match="only one of expires_at or ttl_seconds"):
            mint_gateway_token(
                keyring=keyring,
                issuer="enforceai-gateway",
                user_id="https://issuer.example|sub-1",
                agent_id=str(uuid.uuid4()),
                scopes=["s1"],
                issued_at=now,
                expires_at=now + timedelta(hours=1),
                ttl_seconds=3600,
            )
