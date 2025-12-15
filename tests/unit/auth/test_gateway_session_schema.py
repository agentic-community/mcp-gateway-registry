"""
Unit tests for the shared gateway session cookie schema.
"""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from gateway_session import (
    build_session_cookie_payload,
    normalize_session_data,
)


@pytest.mark.unit
@pytest.mark.auth
class TestGatewaySessionSchema:
    def test_normalize_legacy_traditional_cookie(self) -> None:
        raw: Mapping[str, Any] = {
            "username": "alice",
            "auth_method": "traditional",
            "provider": "local",
        }

        normalized = normalize_session_data(
            raw,
            max_age_seconds=3600,
            now_epoch_seconds=1000,
        )

        assert normalized.v == 1
        assert normalized.auth_method == "password"
        assert normalized.legacy_auth_method == "traditional"
        assert normalized.provider == "local"
        assert normalized.username == "alice"
        assert normalized.user_id == "local|alice"
        assert normalized.issued_at == 1000
        assert normalized.expires_at == 4600
        assert normalized.session_id

    def test_normalize_legacy_oauth2_cookie(self) -> None:
        raw: Mapping[str, Any] = {
            "username": "bob",
            "auth_method": "oauth2",
            "provider": "keycloak",
            "groups": ["group-1"],
            "email": "bob@example.com",
        }

        normalized = normalize_session_data(
            raw,
            max_age_seconds=60,
            now_epoch_seconds=10,
        )

        assert normalized.auth_method == "oidc"
        assert normalized.legacy_auth_method == "oauth2"
        assert normalized.provider == "keycloak"
        assert normalized.username == "bob"
        assert normalized.email == "bob@example.com"
        assert normalized.groups == ["group-1"]
        assert normalized.user_id == "keycloak|bob"

    def test_build_payload_preserves_legacy_auth_method(self) -> None:
        payload = build_session_cookie_payload(
            username="carol",
            email=None,
            name=None,
            groups=None,
            provider="local",
            legacy_auth_method="traditional",
            max_age_seconds=30,
            now_epoch_seconds=200,
        )

        assert payload["v"] == 1
        assert payload["auth_method"] == "password"
        assert payload["legacy_auth_method"] == "traditional"
        assert payload["provider"] == "local"
        assert payload["username"] == "carol"
        assert payload["user_id"] == "local|carol"
        assert payload["issued_at"] == 200
        assert payload["expires_at"] == 230
        assert payload["session_id"]

