"""
Hardening tests for EnforceAI gateway token primitives (Stage 2.5).
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


@pytest.mark.unit
class TestGatewayTokenHardening:
    def test_no_private_key_or_token_leaks_to_output_or_logs(
        self,
        capsys: pytest.CaptureFixture[str],
        caplog: pytest.LogCaptureFixture,
        enforceai_gateway_key_files,
    ) -> None:
        key_files = enforceai_gateway_key_files
        keyring = GatewayKeyring.load(
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
        )
        now = datetime.now(timezone.utc).replace(microsecond=0)

        caplog.clear()
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
        verify_gateway_token(
            token,
            keyring=keyring,
            now=now,
            expected_issuer="enforceai-gateway",
        )

        captured = capsys.readouterr()
        assert token not in captured.out
        assert token not in captured.err

        combined_logs = "\n".join(record.message for record in caplog.records)
        assert token not in combined_logs
        assert "BEGIN PRIVATE KEY" not in combined_logs
        assert "BEGIN RSA PRIVATE KEY" not in combined_logs
        assert "BEGIN PUBLIC KEY" not in combined_logs

    def test_verify_allows_expired_within_leeway(self, enforceai_gateway_key_files) -> None:
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

    def test_verify_rejects_expired_outside_leeway(self, enforceai_gateway_key_files) -> None:
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
            expires_at=now - timedelta(seconds=61),
            jti="jti-1",
        )
        with pytest.raises(UnauthorizedError, match="Invalid gateway token claims"):
            verify_gateway_token(
                token,
                keyring=keyring,
                now=now,
                clock_skew_seconds=60,
            )

    def test_verify_rejects_iat_too_far_in_future(self, enforceai_gateway_key_files) -> None:
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
            issued_at=now + timedelta(minutes=5),
            ttl_seconds=3600,
            jti="jti-1",
        )
        with pytest.raises(UnauthorizedError, match="Invalid gateway token claims"):
            verify_gateway_token(
                token,
                keyring=keyring,
                now=now,
                clock_skew_seconds=60,
            )

