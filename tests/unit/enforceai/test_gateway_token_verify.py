"""
Unit tests for EnforceAI gateway token verification (Stage 2.4).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import jwt
import pytest

from auth_server.enforceai.crypto.keyring import (
    GatewayKeyring,
)
from auth_server.enforceai.errors import (
    DependencyUnavailableError,
    UnauthorizedError,
)
from auth_server.enforceai.tokens.mint import (
    mint_gateway_token,
)
from auth_server.enforceai.tokens.verify import (
    verify_gateway_token,
)


@pytest.mark.unit
class TestGatewayTokenVerify:
    def test_valid_token_verifies(self, enforceai_gateway_key_files) -> None:
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

        claims = verify_gateway_token(
            token,
            keyring=keyring,
            now=now,
            expected_issuer="enforceai-gateway",
        )
        assert claims.jti == "jti-1"

    def test_unknown_kid_fails(self, enforceai_gateway_key_files) -> None:
        key_files = enforceai_gateway_key_files
        keyring = GatewayKeyring.load(
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
        )
        now = datetime.now(timezone.utc).replace(microsecond=0)

        payload = {
            "iss": "enforceai-gateway",
            "sub": "https://issuer.example|sub-1",
            "agent_id": str(uuid.uuid4()),
            "scopes": ["s1"],
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(hours=1)).timestamp()),
            "jti": "jti-1",
        }
        token = jwt.encode(
            payload=payload,
            key=keyring.signing_private_key,
            algorithm="RS256",
            headers={"kid": "kid-unknown"},
        )

        with pytest.raises(UnauthorizedError, match="Unknown gateway token kid"):
            verify_gateway_token(token, keyring=keyring, now=now)

    def test_tampered_token_fails(self, enforceai_gateway_key_files) -> None:
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
        header_b64, payload_b64, signature_b64 = token.split(".")
        tampered_payload_b64 = payload_b64[:-1] + ("a" if payload_b64[-1] != "a" else "b")
        tampered = ".".join([header_b64, tampered_payload_b64, signature_b64])

        with pytest.raises(UnauthorizedError, match="Invalid gateway token signature"):
            verify_gateway_token(tampered, keyring=keyring, now=now)

    def test_expired_token_fails(self, enforceai_gateway_key_files) -> None:
        key_files = enforceai_gateway_key_files
        keyring = GatewayKeyring.load(
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
        )
        issued_at = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(days=2)
        token = mint_gateway_token(
            keyring=keyring,
            issuer="enforceai-gateway",
            user_id="https://issuer.example|sub-1",
            agent_id=str(uuid.uuid4()),
            scopes=["s1"],
            issued_at=issued_at,
            ttl_seconds=60,
            jti="jti-1",
        )

        with pytest.raises(UnauthorizedError, match="Invalid gateway token claims"):
            verify_gateway_token(token, keyring=keyring)

    def test_algorithm_mismatch_fails(self, enforceai_gateway_key_files) -> None:
        key_files = enforceai_gateway_key_files
        keyring = GatewayKeyring.load(
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
        )
        now = datetime.now(timezone.utc).replace(microsecond=0)

        token = jwt.encode(
            payload={
                "iss": "enforceai-gateway",
                "sub": "https://issuer.example|sub-1",
                "agent_id": str(uuid.uuid4()),
                "scopes": ["s1"],
                "iat": int(now.timestamp()),
                "exp": int((now + timedelta(hours=1)).timestamp()),
                "jti": "jti-1",
            },
            key="secret",
            algorithm="HS256",
            headers={"kid": key_files.active_kid},
        )
        with pytest.raises(UnauthorizedError, match="Unsupported gateway token algorithm"):
            verify_gateway_token(token, keyring=keyring, now=now)

    def test_missing_required_claim_fails(self, enforceai_gateway_key_files) -> None:
        key_files = enforceai_gateway_key_files
        keyring = GatewayKeyring.load(
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
        )
        now = datetime.now(timezone.utc).replace(microsecond=0)

        token = jwt.encode(
            payload={
                "iss": "enforceai-gateway",
                "sub": "https://issuer.example|sub-1",
                "agent_id": str(uuid.uuid4()),
                "scopes": ["s1"],
                "iat": int(now.timestamp()),
                "jti": "jti-1",
            },
            key=keyring.signing_private_key,
            algorithm="RS256",
            headers={"kid": key_files.active_kid},
        )

        with pytest.raises(UnauthorizedError, match="Invalid gateway token claims"):
            verify_gateway_token(token, keyring=keyring, now=now)

    def test_keyring_failure_maps_to_dependency_unavailable(
        self,
        enforceai_gateway_key_files,
        monkeypatch: pytest.MonkeyPatch,
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

        def _boom(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(GatewayKeyring, "get_public_key", _boom)

        with pytest.raises(DependencyUnavailableError, match="Keyring unavailable"):
            verify_gateway_token(token, keyring=keyring, now=now)
