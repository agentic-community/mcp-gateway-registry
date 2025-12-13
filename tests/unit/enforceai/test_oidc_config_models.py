"""
Unit tests for OIDC issuer configuration models (Stage 3.1).
"""

import json

import pytest
from pydantic import ValidationError

from auth_server.enforceai.config import (
    DEFAULT_OIDC_ROLE_CLAIMS,
    DEFAULT_OIDC_SCOPE_CLAIMS,
    EnforceAISettings,
    OIDCIssuerConfig,
)


@pytest.mark.unit
class TestOIDCIssuerConfig:
    def test_legacy_fields_are_accepted(self):
        issuer = OIDCIssuerConfig.model_validate(
            {
                "jwks_url": "https://issuer.example/jwks.json",
                "audience": "mcp-registry",
            }
        )

        assert issuer.jwks_uri == "https://issuer.example/jwks.json"
        assert issuer.audiences == ["mcp-registry"]

    def test_invalid_jwks_uri_scheme_is_rejected(self):
        with pytest.raises(ValidationError, match="jwks_uri"):
            OIDCIssuerConfig.model_validate(
                {
                    "jwks_uri": "ftp://issuer.example/jwks.json",
                    "audiences": ["mcp-registry"],
                }
            )

    def test_empty_audiences_are_rejected(self):
        with pytest.raises(ValidationError, match="audiences"):
            OIDCIssuerConfig.model_validate(
                {
                    "jwks_uri": "https://issuer.example/jwks.json",
                    "audiences": [],
                }
            )

    def test_default_claim_precedence_is_stable(self):
        issuer = OIDCIssuerConfig.model_validate(
            {
                "jwks_uri": "https://issuer.example/jwks.json",
                "audiences": ["mcp-registry"],
            }
        )

        assert issuer.scope_claims == DEFAULT_OIDC_SCOPE_CLAIMS
        assert issuer.role_claims == DEFAULT_OIDC_ROLE_CLAIMS


@pytest.mark.unit
class TestEnforceAISettingsOIDCIssuers:
    def test_single_issuer_map_parses(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(
            "OIDC_ISSUERS",
            json.dumps(
                {
                    "https://issuer.example": {
                        "jwks_uri": "https://issuer.example/jwks.json",
                        "audiences": ["mcp-registry"],
                    }
                }
            ),
        )
        monkeypatch.setenv(
            "ENFORCEAI_DB_PATH",
            "/tmp/enforceai.db",
        )

        settings = EnforceAISettings(_env_file=None)

        assert settings.oidc_issuers["https://issuer.example"].jwks_uri.endswith("/jwks.json")

    def test_multi_issuer_map_parses(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(
            "OIDC_ISSUERS",
            json.dumps(
                {
                    "https://issuer-one.example": {
                        "jwks_uri": "https://issuer-one.example/jwks.json",
                        "audiences": ["mcp-registry"],
                    },
                    "https://issuer-two.example": {
                        "jwks_uri": "https://issuer-two.example/jwks.json",
                        "audiences": ["mcp-registry", "mcp-gateway"],
                    },
                }
            ),
        )
        monkeypatch.setenv(
            "ENFORCEAI_DB_PATH",
            "/tmp/enforceai.db",
        )

        settings = EnforceAISettings(_env_file=None)

        assert set(settings.oidc_issuers.keys()) == {
            "https://issuer-one.example",
            "https://issuer-two.example",
        }

