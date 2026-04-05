"""Unit tests for cross-account support in AgentCore auto-registration.

Tests the assume-role logic, multi-account iteration, account ID parsing,
and that scanner/builder correctly use cross-account sessions.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _parse_account_ids
# ---------------------------------------------------------------------------


class TestParseAccountIds:
    """Tests for _parse_account_ids helper."""

    def test_empty_string_returns_empty_list(self):
        from cli.agentcore.sync import _parse_account_ids

        assert _parse_account_ids("") == []

    def test_whitespace_only_returns_empty_list(self):
        from cli.agentcore.sync import _parse_account_ids

        assert _parse_account_ids("   ") == []

    def test_single_account(self):
        from cli.agentcore.sync import _parse_account_ids

        assert _parse_account_ids("111122223333") == ["111122223333"]

    def test_multiple_accounts(self):
        from cli.agentcore.sync import _parse_account_ids

        result = _parse_account_ids("111122223333,444455556666,777788889999")
        assert result == ["111122223333", "444455556666", "777788889999"]

    def test_strips_whitespace(self):
        from cli.agentcore.sync import _parse_account_ids

        result = _parse_account_ids(" 111122223333 , 444455556666 ")
        assert result == ["111122223333", "444455556666"]

    def test_ignores_empty_entries(self):
        from cli.agentcore.sync import _parse_account_ids

        result = _parse_account_ids("111122223333,,444455556666,")
        assert result == ["111122223333", "444455556666"]


# ---------------------------------------------------------------------------
# _assume_role_session
# ---------------------------------------------------------------------------


class TestAssumeRoleSession:
    """Tests for _assume_role_session helper."""

    @patch("boto3.client")
    @patch("boto3.Session")
    def test_assume_role_creates_session(self, mock_session_cls, mock_client_fn):
        from cli.agentcore.sync import _assume_role_session

        mock_sts = MagicMock()
        mock_client_fn.return_value = mock_sts
        mock_sts.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "AKID",
                "SecretAccessKey": "SECRET",
                "SessionToken": "TOKEN",
            }
        }
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        result = _assume_role_session("111122223333", "MyRole", "us-east-2")

        mock_sts.assume_role.assert_called_once_with(
            RoleArn="arn:aws:iam::111122223333:role/MyRole",
            RoleSessionName="agentcore-sync-111122223333",
            DurationSeconds=3600,
        )
        mock_session_cls.assert_called_once_with(
            aws_access_key_id="AKID",
            aws_secret_access_key="SECRET",
            aws_session_token="TOKEN",
            region_name="us-east-2",
        )
        assert result == mock_session

    @patch("boto3.client")
    def test_assume_role_propagates_error(self, mock_client_fn):
        from botocore.exceptions import ClientError

        from cli.agentcore.sync import _assume_role_session

        mock_sts = MagicMock()
        mock_client_fn.return_value = mock_sts
        mock_sts.assume_role.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Not authorized"}},
            "AssumeRole",
        )

        with pytest.raises(ClientError):
            _assume_role_session("111122223333", "MyRole", "us-east-2")


# ---------------------------------------------------------------------------
# Scanner with cross-account session
# ---------------------------------------------------------------------------


class TestScannerCrossAccount:
    """Tests that AgentCoreScanner uses the provided session."""

    @patch("cli.agentcore.discovery.boto3")
    def test_scanner_uses_session_client(self, mock_boto3):
        from cli.agentcore.discovery import AgentCoreScanner

        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client

        scanner = AgentCoreScanner(
            region="us-east-2", timeout=5, session=mock_session
        )

        # Should use session.client, not boto3.client
        mock_session.client.assert_called_once()
        mock_boto3.client.assert_not_called()
        assert scanner.client == mock_client

    @patch("cli.agentcore.discovery.boto3")
    def test_scanner_without_session_uses_default(self, mock_boto3):
        from cli.agentcore.discovery import AgentCoreScanner

        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        scanner = AgentCoreScanner(region="us-east-2", timeout=5)

        mock_boto3.client.assert_called_once()
        assert scanner.client == mock_client


# ---------------------------------------------------------------------------
# RegistrationBuilder with cross-account session
# ---------------------------------------------------------------------------


class TestRegistrationBuilderCrossAccount:
    """Tests that RegistrationBuilder uses the provided session for STS."""

    @patch("cli.agentcore.registration.boto3")
    def test_builder_uses_session_for_account_id(self, mock_boto3):
        from cli.agentcore.registration import RegistrationBuilder

        mock_session = MagicMock()
        mock_sts = MagicMock()
        mock_session.client.return_value = mock_sts
        mock_sts.get_caller_identity.return_value = {
            "Account": "999988887777"
        }

        builder = RegistrationBuilder(
            region="us-east-2", session=mock_session
        )

        mock_session.client.assert_called_once_with("sts")
        mock_boto3.client.assert_not_called()
        assert builder.account_id == "999988887777"

    @patch("cli.agentcore.registration.boto3")
    def test_builder_without_session_uses_default(self, mock_boto3):
        mock_sts = MagicMock()
        mock_boto3.client.return_value = mock_sts
        mock_sts.get_caller_identity.return_value = {
            "Account": "111122223333"
        }

        from cli.agentcore.registration import RegistrationBuilder

        builder = RegistrationBuilder(region="us-east-2")

        mock_boto3.client.assert_called_once_with("sts")
        assert builder.account_id == "111122223333"


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLIAccountArgs:
    """Tests that --accounts and --assume-role-name are parsed correctly."""

    def test_accounts_flag_parsed(self):
        from cli.agentcore.sync import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "sync",
            "--accounts", "111122223333,444455556666",
            "--assume-role-name", "CrossAccountRole",
        ])
        assert args.accounts == "111122223333,444455556666"
        assert args.assume_role_name == "CrossAccountRole"

    def test_accounts_defaults_to_empty(self):
        from cli.agentcore.sync import build_parser

        parser = build_parser()
        args = parser.parse_args(["sync"])
        # Default is empty string (or env var)
        assert hasattr(args, "accounts")

    def test_list_subcommand_has_accounts_flag(self):
        from cli.agentcore.sync import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "list",
            "--accounts", "111122223333",
        ])
        assert args.accounts == "111122223333"

    def test_default_role_name(self):
        from cli.agentcore.sync import build_parser

        parser = build_parser()
        args = parser.parse_args(["sync"])
        assert args.assume_role_name == "AgentCoreSyncRole"
