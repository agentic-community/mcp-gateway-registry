from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    timezone,
)

import pytest

from auth_server.enforceai.errors import (
    UnauthorizedError,
)
from auth_server.enforceai.tokens.ui_session import (
    ENFORCEAI_UI_SESSION_TOKEN_AUDIENCE,
    ENFORCEAI_UI_SESSION_TOKEN_ISSUER,
    mint_enforceai_ui_session_token,
    verify_enforceai_ui_session_token,
)


@pytest.mark.unit
class TestEnforceAIUISessionToken:
    def test_mint_and_verify_roundtrip(self) -> None:
        issued_at = datetime(2025, 12, 20, 12, 0, 0, tzinfo=timezone.utc)

        token, expires_at = mint_enforceai_ui_session_token(
            secret_key="secret-1",
            user_id="local|user-1",
            session_id="session-1",
            groups=["enforceai-admin"],
            issued_at=issued_at,
            ttl_seconds=60,
            jti="jti-1",
        )

        assert expires_at == issued_at + timedelta(seconds=60)

        claims = verify_enforceai_ui_session_token(
            token,
            secret_key="secret-1",
            now=issued_at + timedelta(seconds=1),
        )
        assert claims.iss == ENFORCEAI_UI_SESSION_TOKEN_ISSUER
        assert ENFORCEAI_UI_SESSION_TOKEN_AUDIENCE in claims.aud
        assert claims.sub == "local|user-1"
        assert claims.sid == "session-1"
        assert claims.groups == ["enforceai-admin"]
        assert claims.jti == "jti-1"

    def test_verify_rejects_wrong_issuer(self) -> None:
        issued_at = datetime(2025, 12, 20, 12, 0, 0, tzinfo=timezone.utc)
        token, _expires_at = mint_enforceai_ui_session_token(
            secret_key="secret-1",
            user_id="local|user-1",
            session_id="session-1",
            groups=[],
            issuer="not-the-issuer",
            issued_at=issued_at,
            ttl_seconds=60,
        )

        with pytest.raises(UnauthorizedError, match="issuer"):
            verify_enforceai_ui_session_token(
                token,
                secret_key="secret-1",
                now=issued_at + timedelta(seconds=1),
            )

    def test_verify_rejects_wrong_audience(self) -> None:
        issued_at = datetime(2025, 12, 20, 12, 0, 0, tzinfo=timezone.utc)
        token, _expires_at = mint_enforceai_ui_session_token(
            secret_key="secret-1",
            user_id="local|user-1",
            session_id="session-1",
            groups=[],
            audience="other-audience",
            issued_at=issued_at,
            ttl_seconds=60,
        )

        with pytest.raises(UnauthorizedError, match="audience"):
            verify_enforceai_ui_session_token(
                token,
                secret_key="secret-1",
                now=issued_at + timedelta(seconds=1),
            )

    def test_verify_rejects_expired_token(self) -> None:
        issued_at = datetime(2025, 12, 20, 12, 0, 0, tzinfo=timezone.utc)
        token, _expires_at = mint_enforceai_ui_session_token(
            secret_key="secret-1",
            user_id="local|user-1",
            session_id="session-1",
            groups=[],
            issued_at=issued_at,
            ttl_seconds=1,
        )

        with pytest.raises(UnauthorizedError, match="expired"):
            verify_enforceai_ui_session_token(
                token,
                secret_key="secret-1",
                now=issued_at + timedelta(minutes=5),
                clock_skew_seconds=0,
            )

