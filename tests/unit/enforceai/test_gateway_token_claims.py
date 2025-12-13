"""
Unit tests for EnforceAI gateway token claims model and validation helpers (Stage 2.1).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from auth_server.enforceai.tokens.claims import (
    GatewayTokenClaims,
    datetime_to_jwt_timestamp,
    jwt_timestamp_to_datetime,
    validate_gateway_token_claims,
)


@pytest.mark.unit
class TestGatewayTokenClaims:
    def test_datetime_timestamp_roundtrip(self) -> None:
        value = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        ts = datetime_to_jwt_timestamp(value)
        assert ts > 0
        assert jwt_timestamp_to_datetime(ts) == value

    def test_model_parse_and_serialize(self) -> None:
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        claims = GatewayTokenClaims(
            iss="enforceai-gateway",
            sub="https://issuer.example|sub-1",
            agent_id=str(uuid.uuid4()),
            scopes=["mcp-servers-restricted/read"],
            iat=datetime_to_jwt_timestamp(now),
            exp=datetime_to_jwt_timestamp(now + timedelta(hours=1)),
            jti="jti-1",
        )
        dumped = claims.model_dump()
        assert dumped["iss"] == "enforceai-gateway"
        assert dumped["sub"] == "https://issuer.example|sub-1"
        assert dumped["jti"] == "jti-1"

    def test_missing_required_claims_rejected(self) -> None:
        with pytest.raises(ValueError, match="Field required"):
            GatewayTokenClaims.model_validate(
                {
                    "iss": "enforceai-gateway",
                    "sub": "https://issuer.example|sub-1",
                }
            )

    def test_invalid_agent_id_rejected(self) -> None:
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="agent_id must be a UUIDv4"):
            GatewayTokenClaims(
                iss="enforceai-gateway",
                sub="https://issuer.example|sub-1",
                agent_id="not-a-uuid",
                scopes=["s1"],
                iat=datetime_to_jwt_timestamp(now),
                exp=datetime_to_jwt_timestamp(now + timedelta(hours=1)),
                jti="jti-1",
            )

    def test_invalid_user_id_rejected(self) -> None:
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="sub must be a canonical user_id"):
            GatewayTokenClaims(
                iss="enforceai-gateway",
                sub="not-a-user-id",
                agent_id=str(uuid.uuid4()),
                scopes=["s1"],
                iat=datetime_to_jwt_timestamp(now),
                exp=datetime_to_jwt_timestamp(now + timedelta(hours=1)),
                jti="jti-1",
            )

    def test_empty_scopes_item_rejected(self) -> None:
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="scopes must not contain empty"):
            GatewayTokenClaims(
                iss="enforceai-gateway",
                sub="https://issuer.example|sub-1",
                agent_id=str(uuid.uuid4()),
                scopes=[" "],
                iat=datetime_to_jwt_timestamp(now),
                exp=datetime_to_jwt_timestamp(now + timedelta(hours=1)),
                jti="jti-1",
            )

    def test_validate_rejects_expired_outside_leeway(self) -> None:
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        claims = GatewayTokenClaims(
            iss="enforceai-gateway",
            sub="https://issuer.example|sub-1",
            agent_id=str(uuid.uuid4()),
            scopes=["s1"],
            iat=datetime_to_jwt_timestamp(now - timedelta(hours=2)),
            exp=datetime_to_jwt_timestamp(now - timedelta(minutes=10)),
            jti="jti-1",
        )
        with pytest.raises(ValueError, match="Token is expired"):
            validate_gateway_token_claims(
                claims,
                now=now,
                clock_skew_seconds=60,
            )

    def test_validate_allows_expired_within_leeway(self) -> None:
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        claims = GatewayTokenClaims(
            iss="enforceai-gateway",
            sub="https://issuer.example|sub-1",
            agent_id=str(uuid.uuid4()),
            scopes=["s1"],
            iat=datetime_to_jwt_timestamp(now - timedelta(hours=1)),
            exp=datetime_to_jwt_timestamp(now - timedelta(seconds=30)),
            jti="jti-1",
        )
        validate_gateway_token_claims(
            claims,
            now=now,
            clock_skew_seconds=60,
        )

    def test_validate_rejects_iat_too_far_in_future(self) -> None:
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        claims = GatewayTokenClaims(
            iss="enforceai-gateway",
            sub="https://issuer.example|sub-1",
            agent_id=str(uuid.uuid4()),
            scopes=["s1"],
            iat=datetime_to_jwt_timestamp(now + timedelta(minutes=5)),
            exp=datetime_to_jwt_timestamp(now + timedelta(hours=1)),
            jti="jti-1",
        )
        with pytest.raises(ValueError, match="iat is too far"):
            validate_gateway_token_claims(
                claims,
                now=now,
                clock_skew_seconds=60,
            )

    def test_validate_rejects_exp_not_greater_than_iat(self) -> None:
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        claims = GatewayTokenClaims(
            iss="enforceai-gateway",
            sub="https://issuer.example|sub-1",
            agent_id=str(uuid.uuid4()),
            scopes=["s1"],
            iat=datetime_to_jwt_timestamp(now),
            exp=datetime_to_jwt_timestamp(now),
            jti="jti-1",
        )
        with pytest.raises(ValueError, match="exp must be greater than iat"):
            validate_gateway_token_claims(claims, now=now)

    def test_validate_enforces_max_lifetime(self) -> None:
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        claims = GatewayTokenClaims(
            iss="enforceai-gateway",
            sub="https://issuer.example|sub-1",
            agent_id=str(uuid.uuid4()),
            scopes=["s1"],
            iat=datetime_to_jwt_timestamp(now),
            exp=datetime_to_jwt_timestamp(now + timedelta(days=10)),
            jti="jti-1",
        )
        with pytest.raises(ValueError, match="Token lifetime exceeds"):
            validate_gateway_token_claims(
                claims,
                now=now,
                max_lifetime_seconds=60 * 60,
            )

