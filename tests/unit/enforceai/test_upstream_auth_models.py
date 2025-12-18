"""
Unit tests for upstream auth normalization models (Phase 1).
"""

from __future__ import annotations

import json

import pytest

from auth_server.enforceai.models.upstream_auth import (
    normalize_upstream_auth,
)


@pytest.mark.unit
class TestUpstreamAuthModels:
    def test_normalize_legacy_none(
        self,
    ) -> None:
        normalized = normalize_upstream_auth(
            auth_type=None,
            auth_provider=None,
            headers=None,
        )
        assert normalized.type == "none"
        assert normalized.injection is None

    def test_normalize_legacy_api_key_defaults_to_x_api_key(
        self,
    ) -> None:
        normalized = normalize_upstream_auth(
            auth_type="api_key",
            auth_provider=None,
            headers=None,
        )
        assert normalized.type == "api-key"
        assert normalized.injection is not None
        assert normalized.injection.header_name == "X-API-Key"
        assert normalized.injection.scheme is None

    def test_normalize_legacy_oauth_infers_authorization_bearer(
        self,
    ) -> None:
        normalized = normalize_upstream_auth(
            auth_type="oauth",
            auth_provider="github",
            headers=[
                {"Authorization": "Bearer $TOKEN"},
            ],
        )
        assert normalized.type == "oauth2"
        assert normalized.provider == "github"
        assert normalized.credential_binding == "user"
        assert normalized.injection is not None
        assert normalized.injection.header_name == "Authorization"
        assert normalized.injection.scheme == "Bearer"

    def test_upstream_auth_object_is_validated(
        self,
    ) -> None:
        raw = {
            "type": "jwt",
            "provider": None,
            "credential_binding": "service",
            "injection": {"kind": "header", "header_name": "Authorization", "scheme": "Bearer"},
        }
        normalized = normalize_upstream_auth(
            upstream_auth=json.dumps(raw),
            auth_type="none",
        )
        assert normalized.type == "jwt"
        assert normalized.injection is not None
        assert normalized.injection.header_name == "Authorization"

    def test_mtls_rejected_in_phase_1(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="mtls is not supported yet"):
            normalize_upstream_auth(
                upstream_auth={"type": "mtls", "credential_binding": "service"},
            )

