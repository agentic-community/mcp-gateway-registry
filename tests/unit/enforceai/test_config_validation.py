"""
Unit tests for EnforceAI configuration validation rules.
"""

import json

import pytest
from pydantic import ValidationError

from auth_server.enforceai.config import EnforceAISettings


def _set_base_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "OIDC_ISSUERS",
        json.dumps(
            {
                "https://issuer.example": {
                    "jwks_url": "https://issuer.example/.well-known/jwks.json",
                },
            }
        ),
    )
    monkeypatch.setenv(
        "ENFORCEAI_DB_PATH",
        "/tmp/enforceai.db",
    )


@pytest.mark.unit
class TestEnforceAIConfigValidation:
    """Test suite for EnforceAI settings validation."""

    def test_missing_public_keys_dir_is_rejected(self, monkeypatch: pytest.MonkeyPatch):
        _set_base_env(monkeypatch)

        monkeypatch.setenv(
            "ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH",
            "/tmp/private.pem",
        )
        monkeypatch.setenv(
            "GATEWAY_ACTIVE_KID",
            "kid-1",
        )

        with pytest.raises(ValidationError, match="Gateway token key configuration incomplete"):
            EnforceAISettings(_env_file=None)

    def test_missing_active_kid_is_rejected(self, monkeypatch: pytest.MonkeyPatch):
        _set_base_env(monkeypatch)

        monkeypatch.setenv(
            "ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH",
            "/tmp/private.pem",
        )
        monkeypatch.setenv(
            "ENFORCEAI_GATEWAY_PUBLIC_KEYS_DIR",
            "/tmp/public",
        )

        with pytest.raises(ValidationError, match="GATEWAY_ACTIVE_KID"):
            EnforceAISettings(_env_file=None)

    def test_negative_retention_values_are_rejected(self, monkeypatch: pytest.MonkeyPatch):
        _set_base_env(monkeypatch)

        monkeypatch.setenv(
            "ENFORCEAI_AUDIT_RETENTION_DAYS",
            "-1",
        )
        monkeypatch.setenv(
            "ENFORCEAI_AUDIT_MAX_DB_BYTES",
            "-10",
        )

        with pytest.raises(ValidationError):
            EnforceAISettings(_env_file=None)

