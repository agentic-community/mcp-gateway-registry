"""Tests for the startup AWS credential posture check.

The check warns when the registry can read instance-metadata credentials that no
enabled feature needs, and stays quiet otherwise. It must never raise, because a
diagnostic that can stop startup is worse than the finding it reports.

Covers issue #1861.
"""

import asyncio
import logging
import time

import pytest

from registry.core import aws_credential_posture
from registry.core.aws_credential_posture import (
    _aws_backed_features,
    _caller_identity_arn,
    _resolved_credential_provider,
    log_aws_credential_posture,
)

EXAMPLE_ARN = "arn:aws:sts::123456789012:assumed-role/example-role/i-0123456789abcdef0"


@pytest.fixture
def imds_credentials(monkeypatch):
    """Resolve credentials through the instance metadata service."""
    monkeypatch.setattr(
        aws_credential_posture,
        "_resolved_credential_provider",
        lambda: "iam-role",
    )
    monkeypatch.setattr(
        aws_credential_posture,
        "_caller_identity_arn",
        lambda: EXAMPLE_ARN,
    )


@pytest.fixture
def no_aws_features(monkeypatch):
    """Turn off every feature that would make an AWS API call."""
    monkeypatch.setattr(aws_credential_posture.settings, "auth_provider", "keycloak")
    monkeypatch.setattr(
        aws_credential_posture.settings,
        "secret_store_backend",
        "openbao",
    )


class TestAwsBackedFeatures:
    """Only the features that really call AWS count."""

    def test_keycloak_and_openbao_with_federation_off_needs_nothing(
        self,
        no_aws_features,
    ):
        assert _aws_backed_features(aws_federation_enabled=False) == []

    def test_cognito_counts(self, monkeypatch, no_aws_features):
        monkeypatch.setattr(aws_credential_posture.settings, "auth_provider", "cognito")
        features = _aws_backed_features(aws_federation_enabled=False)
        assert len(features) == 1
        assert "Cognito" in features[0]

    def test_secrets_manager_counts(self, monkeypatch, no_aws_features):
        monkeypatch.setattr(
            aws_credential_posture.settings,
            "secret_store_backend",
            "secrets-manager",
        )
        features = _aws_backed_features(aws_federation_enabled=False)
        assert len(features) == 1
        assert "Secrets Manager" in features[0]

    def test_federation_counts(self, no_aws_features):
        features = _aws_backed_features(aws_federation_enabled=True)
        assert features == ["AgentCore registry federation"]


class TestWarnPath:
    """The warning fires only when credentials come from IMDS and go unused."""

    @pytest.mark.asyncio
    async def test_warns_when_imds_credentials_are_unused(
        self,
        caplog,
        imds_credentials,
        no_aws_features,
    ):
        with caplog.at_level(logging.WARNING):
            await log_aws_credential_posture(aws_federation_enabled=False)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert EXAMPLE_ARN in warnings[0].message
        assert "docs/installation.md" in warnings[0].message

    @pytest.mark.asyncio
    async def test_an_enabled_feature_suppresses_the_warning(
        self,
        caplog,
        imds_credentials,
        no_aws_features,
    ):
        with caplog.at_level(logging.INFO):
            await log_aws_credential_posture(aws_federation_enabled=True)

        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []
        assert any("AgentCore registry federation" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_credentials_from_another_provider_do_not_warn(
        self,
        caplog,
        monkeypatch,
        no_aws_features,
    ):
        monkeypatch.setattr(
            aws_credential_posture,
            "_resolved_credential_provider",
            lambda: "container-role",
        )
        monkeypatch.setattr(
            aws_credential_posture,
            "_caller_identity_arn",
            lambda: EXAMPLE_ARN,
        )

        with caplog.at_level(logging.INFO):
            await log_aws_credential_posture(aws_federation_enabled=False)

        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []
        assert any("container-role" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_no_credentials_reports_nothing_to_do(
        self,
        caplog,
        monkeypatch,
        no_aws_features,
    ):
        monkeypatch.setattr(
            aws_credential_posture,
            "_resolved_credential_provider",
            lambda: None,
        )

        with caplog.at_level(logging.INFO):
            await log_aws_credential_posture(aws_federation_enabled=False)

        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []
        assert any("No AWS credentials resolve" in r.message for r in caplog.records)


class TestFailuresNeverBlockStartup:
    """Every failure mode degrades to a log line."""

    @pytest.mark.asyncio
    async def test_warns_without_an_arn_when_get_caller_identity_fails(
        self,
        caplog,
        monkeypatch,
        no_aws_features,
    ):
        monkeypatch.setattr(
            aws_credential_posture,
            "_resolved_credential_provider",
            lambda: "iam-role",
        )
        monkeypatch.setattr(
            aws_credential_posture,
            "_caller_identity_arn",
            lambda: None,
        )

        with caplog.at_level(logging.WARNING):
            await log_aws_credential_posture(aws_federation_enabled=False)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "sts:GetCallerIdentity did not return" in warnings[0].message

    @pytest.mark.asyncio
    async def test_an_unexpected_error_does_not_propagate(
        self,
        monkeypatch,
        no_aws_features,
    ):
        def explode():
            raise RuntimeError("boom")

        monkeypatch.setattr(
            aws_credential_posture,
            "_resolved_credential_provider",
            explode,
        )

        # No assertion beyond the absence of an exception: startup must continue.
        await log_aws_credential_posture(aws_federation_enabled=False)

    def test_provider_resolution_survives_a_botocore_error(self, monkeypatch):
        import botocore.session

        def explode(*args, **kwargs):
            raise RuntimeError("no config")

        monkeypatch.setattr(botocore.session, "get_session", explode)
        assert _resolved_credential_provider() is None

    def test_caller_identity_survives_a_client_error(self, monkeypatch):
        import boto3

        def explode(*args, **kwargs):
            raise RuntimeError("no endpoint")

        monkeypatch.setattr(boto3, "client", explode)
        assert _caller_identity_arn() is None


class TestStartupIsNeverDelayed:
    """A diagnostic on the startup path must be bounded."""

    @pytest.mark.asyncio
    async def test_a_hanging_check_is_abandoned_at_the_deadline(
        self,
        monkeypatch,
        no_aws_features,
    ):
        monkeypatch.setattr(
            aws_credential_posture,
            "POSTURE_CHECK_DEADLINE_SECONDS",
            0.05,
        )

        async def never_finishes(aws_federation_enabled):
            await asyncio.sleep(30)

        monkeypatch.setattr(
            aws_credential_posture,
            "_report_credential_posture",
            never_finishes,
        )

        started = time.monotonic()
        await log_aws_credential_posture(aws_federation_enabled=False)
        assert time.monotonic() - started < 5

    def test_the_sts_client_uses_short_timeouts_and_one_attempt(self, monkeypatch):
        captured = {}

        class FakeClient:
            def get_caller_identity(self):
                return {"Arn": EXAMPLE_ARN}

        def fake_client(service, config=None):
            captured["service"] = service
            captured["config"] = config
            return FakeClient()

        import boto3

        monkeypatch.setattr(boto3, "client", fake_client)

        assert _caller_identity_arn() == EXAMPLE_ARN
        assert captured["service"] == "sts"

        config = captured["config"]
        assert config is not None, "the STS client must not use boto3's 60s defaults"
        assert config.connect_timeout == aws_credential_posture.STS_TIMEOUT_SECONDS
        assert config.read_timeout == aws_credential_posture.STS_TIMEOUT_SECONDS
        assert config.retries["max_attempts"] == aws_credential_posture.STS_MAX_ATTEMPTS


class TestNoPolicyReads:
    """The check must not need IAM read permissions on the role it reports."""

    def test_module_makes_no_iam_or_simulate_call(self):
        from pathlib import Path

        source = Path(aws_credential_posture.__file__).read_text()
        body = "\n".join(
            line for line in source.splitlines() if not line.strip().startswith("#")
        )
        for forbidden in [
            "list_attached_role_policies",
            "get_policy_version",
            "list_role_policies",
            "get_role_policy",
            "simulate_principal_policy",
        ]:
            assert forbidden not in body
