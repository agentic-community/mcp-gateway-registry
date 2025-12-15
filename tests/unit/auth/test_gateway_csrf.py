"""
Unit tests for CSRF token minting and validation.
"""

from __future__ import annotations

import pytest

from gateway_csrf import (
    mint_csrf_token,
    validate_csrf_token,
)


@pytest.mark.unit
@pytest.mark.auth
class TestGatewayCsrf:
    def test_mint_and_validate_success(self) -> None:
        token = mint_csrf_token(secret_key="secret-1", session_id="sid-1")
        assert (
            validate_csrf_token(
                secret_key="secret-1",
                token=token,
                session_id="sid-1",
                max_age_seconds=60,
            )
            is None
        )

    def test_missing_token_rejected(self) -> None:
        assert (
            validate_csrf_token(
                secret_key="secret-1",
                token="",
                session_id="sid-1",
                max_age_seconds=60,
            )
            == "Missing X-CSRF-Token"
        )

    def test_wrong_session_id_rejected(self) -> None:
        token = mint_csrf_token(secret_key="secret-1", session_id="sid-1")
        assert (
            validate_csrf_token(
                secret_key="secret-1",
                token=token,
                session_id="sid-2",
                max_age_seconds=60,
            )
            == "Invalid CSRF token"
        )

