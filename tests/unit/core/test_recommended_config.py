"""Tests for registry.core.recommended_config.

The egress credential encryption key is the first recommended-but-optional
setting: the vault runs without it but then stores credentials in plaintext, so
we nudge (startup warning + metric + UI badge) only when the vault is enabled and
the key is unset. These tests pin the evaluate/gate/fail-closed behavior that all
three surfaces depend on.
"""

import logging
from unittest.mock import MagicMock

from registry.core.recommended_config import (
    evaluate_recommendations,
    log_recommended_config_warnings,
)


def _settings(egress_auth_enabled=True, egress_credential_encryption_key=""):
    s = MagicMock()
    s.egress_auth_enabled = egress_auth_enabled
    s.egress_credential_encryption_key = egress_credential_encryption_key
    return s


class TestEvaluateRecommendations:
    """The applies() gate and configured status."""

    def test_not_applicable_when_egress_disabled(self):
        """A deployment that doesn't use the vault is never nagged about the key."""
        result = evaluate_recommendations(_settings(egress_auth_enabled=False))
        assert all(r["id"] != "egress_credential_encryption" for r in result)

    def test_applicable_and_key_unset_is_a_nudge(self):
        result = evaluate_recommendations(
            _settings(egress_auth_enabled=True, egress_credential_encryption_key="")
        )
        rec = next(r for r in result if r["id"] == "egress_credential_encryption")
        assert rec["configured"] is False
        assert rec["component"] == "egress"
        assert rec["severity"] == "security"

    def test_applicable_and_key_set_is_configured(self):
        result = evaluate_recommendations(
            _settings(egress_auth_enabled=True, egress_credential_encryption_key="a" * 43)
        )
        rec = next(r for r in result if r["id"] == "egress_credential_encryption")
        assert rec["configured"] is True

    def test_whitespace_only_key_is_not_configured(self):
        result = evaluate_recommendations(
            _settings(egress_auth_enabled=True, egress_credential_encryption_key="   ")
        )
        rec = next(r for r in result if r["id"] == "egress_credential_encryption")
        assert rec["configured"] is False

    def test_message_contains_no_secret_value(self):
        key = "super-secret-key-value-do-not-leak-1234567890"
        result = evaluate_recommendations(
            _settings(egress_auth_enabled=True, egress_credential_encryption_key=key)
        )
        rec = next(r for r in result if r["id"] == "egress_credential_encryption")
        assert key not in rec["message"]

    def test_fail_closed_on_broken_settings(self):
        """A settings object that raises must not crash evaluation; treat as a nudge."""

        class _Boom:
            egress_auth_enabled = True

            @property
            def egress_credential_encryption_key(self):
                raise RuntimeError("boom")

        result = evaluate_recommendations(_Boom())
        rec = next(r for r in result if r["id"] == "egress_credential_encryption")
        assert rec["configured"] is False  # fail closed


class TestLogRecommendedConfigWarnings:
    def test_warns_when_unset(self, caplog):
        with caplog.at_level(logging.WARNING, logger="registry.core.recommended_config"):
            log_recommended_config_warnings(_settings(egress_auth_enabled=True))
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "egress_credential_encryption" in m and "not set" in m.lower() for m in warnings
        ), warnings

    def test_no_warning_when_configured(self, caplog):
        with caplog.at_level(logging.WARNING, logger="registry.core.recommended_config"):
            log_recommended_config_warnings(
                _settings(egress_auth_enabled=True, egress_credential_encryption_key="a" * 43)
            )
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_no_warning_when_not_applicable(self, caplog):
        with caplog.at_level(logging.WARNING, logger="registry.core.recommended_config"):
            log_recommended_config_warnings(_settings(egress_auth_enabled=False))
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]
