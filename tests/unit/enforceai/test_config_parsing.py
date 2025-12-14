"""
Unit tests for EnforceAI configuration parsing (env-driven).
"""

import json

import pytest
from pydantic import ValidationError

from auth_server.enforceai.config import EnforceAISettings


def _set_minimal_valid_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "OIDC_ISSUERS",
        json.dumps(
            {
                "https://issuer.example": {
                    "jwks_uri": "https://issuer.example/.well-known/jwks.json",
                    "audiences": ["mcp-registry"],
                },
            }
        ),
    )
    monkeypatch.setenv(
        "ENFORCEAI_DB_PATH",
        "/tmp/enforceai.db",
    )


@pytest.mark.unit
class TestEnforceAIConfigParsing:
    """Test suite for EnforceAI env var parsing."""

    def test_oidc_issuers_map_of_one_parses(self, monkeypatch: pytest.MonkeyPatch):
        _set_minimal_valid_env(monkeypatch)

        settings = EnforceAISettings(_env_file=None)

        assert "https://issuer.example" in settings.oidc_issuers
        assert settings.oidc_issuers["https://issuer.example"].jwks_uri.endswith(
            "/.well-known/jwks.json"
        )

    def test_oidc_issuers_map_of_two_parses(self, monkeypatch: pytest.MonkeyPatch):
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
                        "audience": "mcp-registry",
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
        assert settings.oidc_issuers["https://issuer-two.example"].audiences == ["mcp-registry"]

    def test_invalid_oidc_issuers_json_fails_with_clear_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv(
            "OIDC_ISSUERS",
            "{not valid json",
        )
        monkeypatch.setenv(
            "ENFORCEAI_DB_PATH",
            "/tmp/enforceai.db",
        )

        with pytest.raises(ValidationError, match="Invalid JSON in OIDC_ISSUERS"):
            EnforceAISettings(_env_file=None)

    def test_missing_required_vars_fails_with_clear_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.delenv("OIDC_ISSUERS", raising=False)
        monkeypatch.delenv("ENFORCEAI_DB_PATH", raising=False)

        with pytest.raises(ValidationError) as exc:
            EnforceAISettings(_env_file=None)

        message = str(exc.value)
        assert "ENFORCEAI_DB_PATH" in message

    def test_missing_oidc_issuers_when_oidc_mode_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("OIDC_ISSUERS", raising=False)
        monkeypatch.setenv(
            "ENFORCEAI_DB_PATH",
            "/tmp/enforceai.db",
        )

        with pytest.raises(ValidationError, match="OIDC_ISSUERS"):
            EnforceAISettings(_env_file=None)
