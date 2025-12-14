"""
Unit tests for EnforceAI credential extraction (Stage 4.1).
"""

import pytest

from auth_server.enforceai.auth.credentials import (
    extract_credential_input,
)
from auth_server.enforceai.errors import UnauthorizedError


@pytest.mark.unit
class TestIdentityCredentials:
    def test_authorization_bearer_parses(self):
        credential = extract_credential_input(
            {
                "Authorization": "Bearer test-token",
            }
        )

        assert credential.kind == "bearer"
        assert credential.value == "test-token"
        assert credential.agent_id_header is None

    def test_x_authorization_bearer_parses(self):
        credential = extract_credential_input(
            {
                "X-Authorization": "Bearer test-token",
            }
        )

        assert credential.kind == "bearer"
        assert credential.value == "test-token"

    def test_authorization_rejects_non_bearer_scheme(self):
        with pytest.raises(UnauthorizedError):
            extract_credential_input(
                {
                    "Authorization": "Basic abc123",
                }
            )

    def test_authorization_rejects_missing_token(self):
        with pytest.raises(UnauthorizedError):
            extract_credential_input(
                {
                    "Authorization": "Bearer",
                }
            )

    def test_rejects_multi_credential_authorization_and_gateway_token(self):
        with pytest.raises(UnauthorizedError):
            extract_credential_input(
                {
                    "Authorization": "Bearer test-token",
                    "X-Gateway-Token": "gateway-token",
                }
            )

    def test_rejects_multi_credential_api_key_and_token(self):
        with pytest.raises(UnauthorizedError):
            extract_credential_input(
                {
                    "X-API-Key": "eak_abc.def",
                    "Authorization": "Bearer test-token",
                }
            )

    def test_rejects_multi_credential_authorization_and_x_authorization(self):
        with pytest.raises(UnauthorizedError):
            extract_credential_input(
                {
                    "Authorization": "Bearer test-token",
                    "X-Authorization": "Bearer test-token-2",
                }
            )

    def test_missing_credentials_unauthorized(self):
        with pytest.raises(UnauthorizedError):
            extract_credential_input({})

    def test_x_agent_id_passthrough_captured(self):
        credential = extract_credential_input(
            {
                "Authorization": "Bearer test-token",
                "X-Agent-Id": "not-a-uuid",
            }
        )

        assert credential.agent_id_header == "not-a-uuid"
