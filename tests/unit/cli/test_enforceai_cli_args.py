"""
Unit tests for EnforceAI CLI argument parsing and header construction (Stage 6.3).
"""

from __future__ import annotations

import pytest

from cli import enforceai_cli


def _parse_args(
    argv: list[str],
):
    return enforceai_cli._build_parser().parse_args(argv)


@pytest.mark.unit
class TestEnforceAICliArgs:
    def test_authorization_coerces_to_bearer(
        self,
    ) -> None:
        args = _parse_args(
            [
                "--authorization",
                "raw-token",
                "agents",
                "list",
            ]
        )
        headers = enforceai_cli._build_headers(
            args=args,
            env={},
        )
        assert headers["Authorization"] == "Bearer raw-token"

    def test_env_fallback_for_api_key(
        self,
    ) -> None:
        args = _parse_args(["agents", "list"])
        headers = enforceai_cli._build_headers(
            args=args,
            env={
                enforceai_cli.ENV_X_API_KEY: "eak_key-1.secret-1",
            },
        )
        assert headers["X-API-Key"] == "eak_key-1.secret-1"

    def test_multiple_credentials_rejected_without_leaking_values(
        self,
    ) -> None:
        secret_token = "Bearer super-secret-token"
        api_key = "eak_key-1.super-secret"

        args = _parse_args(
            [
                "--authorization",
                secret_token,
                "--x-api-key",
                api_key,
                "agents",
                "list",
            ]
        )
        with pytest.raises(enforceai_cli.CLIError) as excinfo:
            enforceai_cli._build_headers(
                args=args,
                env={},
            )

        message = str(excinfo.value)
        assert "Multiple credentials provided" in message
        assert "super-secret-token" not in message
        assert "super-secret" not in message

    def test_missing_credentials_rejected(
        self,
    ) -> None:
        args = _parse_args(["agents", "list"])
        with pytest.raises(enforceai_cli.CLIError, match="Missing credentials"):
            enforceai_cli._build_headers(
                args=args,
                env={},
            )

    def test_tokens_revoke_requires_token_or_agent_id_and_jti(
        self,
    ) -> None:
        args = _parse_args(
            [
                "--authorization",
                "token",
                "tokens",
                "revoke",
            ]
        )
        with pytest.raises(enforceai_cli.CLIError, match="tokens revoke:"):
            enforceai_cli._validate_tokens_revoke_args(args=args)

