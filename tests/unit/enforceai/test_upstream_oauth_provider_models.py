"""
Unit tests for upstream OAuth provider registry models (Phase 1).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from auth_server.enforceai.models.upstream_oauth_provider import (
    UpstreamOAuthProviderCreate,
    UpstreamOAuthProviderRecord,
    UpstreamOAuthProviderUpdate,
)


@pytest.mark.unit
class TestUpstreamOAuthProviderModels:
    def test_provider_id_normalization_rejects_whitespace_and_invalid_chars(
        self,
    ) -> None:
        now = datetime.now(timezone.utc)

        with pytest.raises(ValueError, match="must not include leading/trailing whitespace"):
            UpstreamOAuthProviderRecord(
                provider_id=" github ",
                authorization_endpoint="https://example.com/auth",
                token_endpoint="https://example.com/token",
                client_id="client",
                created_at=now,
                updated_at=now,
            )

        with pytest.raises(ValueError, match="provider_id must match"):
            UpstreamOAuthProviderRecord(
                provider_id="github.com",
                authorization_endpoint="https://example.com/auth",
                token_endpoint="https://example.com/token",
                client_id="client",
                created_at=now,
                updated_at=now,
            )

    def test_endpoint_validation_requires_https_or_localhost_http(
        self,
    ) -> None:
        now = datetime.now(timezone.utc)

        with pytest.raises(ValueError, match="only allowed for localhost development"):
            UpstreamOAuthProviderRecord(
                provider_id="github",
                authorization_endpoint="http://example.com/auth",
                token_endpoint="https://example.com/token",
                client_id="client",
                created_at=now,
                updated_at=now,
            )

        record = UpstreamOAuthProviderRecord(
            provider_id="github",
            authorization_endpoint="http://localhost:9000/auth",
            token_endpoint="http://127.0.0.1:9000/token",
            client_id=" client ",
            default_scopes=["repo", " repo ", "", "user:email"],
            extra_authorize_params={"prompt": " consent ", "": "x", "y": " "},
            created_at=now,
            updated_at=now,
        )
        assert record.authorization_endpoint == "http://localhost:9000/auth"
        assert record.token_endpoint == "http://127.0.0.1:9000/token"
        assert record.client_id == "client"
        assert record.default_scopes == ["repo", "user:email"]
        assert record.extra_authorize_params == {"prompt": "consent"}

    def test_create_requires_client_secret_and_normalizes_scopes(
        self,
    ) -> None:
        payload = UpstreamOAuthProviderCreate(
            provider_id="github",
            authorization_endpoint="https://example.com/auth",
            token_endpoint="https://example.com/token",
            client_id="client",
            client_secret=" secret ",
            default_scopes=["repo", "repo", " user:email "],
        )
        assert payload.client_secret == "secret"
        assert payload.default_scopes == ["repo", "user:email"]

    def test_update_requires_at_least_one_field(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="At least one field must be provided"):
            UpstreamOAuthProviderUpdate()

