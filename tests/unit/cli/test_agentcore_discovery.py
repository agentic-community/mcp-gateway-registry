"""Unit tests for cli.agentcore.discovery — AgentCoreScanner.

Tests pagination (multi-page nextToken handling), READY filtering,
and error handling (AccessDeniedException, ThrottlingException).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_scanner(region: str = "us-east-1", timeout: int = 5):
    """Create an AgentCoreScanner with a mocked boto3 client."""
    with patch("cli.agentcore.discovery.boto3") as mock_boto3:
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        from cli.agentcore.discovery import AgentCoreScanner

        scanner = AgentCoreScanner(region=region, timeout=timeout)
        scanner.client = mock_client
        return scanner, mock_client


# ---------------------------------------------------------------------------
# Gateway pagination tests
# ---------------------------------------------------------------------------


class TestGatewayPagination:
    """Tests for scan_gateways() pagination via nextToken."""

    def test_single_page_no_next_token(self):
        scanner, client = _make_scanner()
        client.list_gateways.return_value = {
            "items": [
                {"gatewayId": "gw-1", "status": "READY"},
            ],
        }
        client.get_gateway.return_value = {
            "gatewayId": "gw-1",
            "name": "Gateway One",
            "status": "READY",
        }
        client.list_gateway_targets.return_value = {"items": []}

        result = scanner.scan_gateways()
        assert len(result) == 1
        assert result[0]["name"] == "Gateway One"
        client.list_gateways.assert_called_once()

    def test_multi_page_pagination(self):
        scanner, client = _make_scanner()
        client.list_gateways.side_effect = [
            {
                "items": [{"gatewayId": "gw-1", "status": "READY"}],
                "nextToken": "page2",
            },
            {
                "items": [{"gatewayId": "gw-2", "status": "READY"}],
                "nextToken": "page3",
            },
            {
                "items": [{"gatewayId": "gw-3", "status": "READY"}],
            },
        ]
        client.get_gateway.side_effect = [
            {"gatewayId": "gw-1", "name": "GW1", "status": "READY"},
            {"gatewayId": "gw-2", "name": "GW2", "status": "READY"},
            {"gatewayId": "gw-3", "name": "GW3", "status": "READY"},
        ]
        client.list_gateway_targets.return_value = {"items": []}

        result = scanner.scan_gateways()
        assert len(result) == 3
        assert client.list_gateways.call_count == 3
        # Verify nextToken was passed on subsequent calls
        calls = client.list_gateways.call_args_list
        assert calls[0] == ((), {})
        assert calls[1] == ((), {"nextToken": "page2"})
        assert calls[2] == ((), {"nextToken": "page3"})

    def test_empty_response(self):
        scanner, client = _make_scanner()
        client.list_gateways.return_value = {"items": []}

        result = scanner.scan_gateways()
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Gateway READY filtering tests
# ---------------------------------------------------------------------------


class TestGatewayReadyFiltering:
    """Tests for READY status filtering in scan_gateways()."""

    def test_only_ready_gateways_returned(self):
        scanner, client = _make_scanner()
        client.list_gateways.return_value = {
            "items": [
                {"gatewayId": "gw-ready", "status": "READY"},
                {"gatewayId": "gw-creating", "status": "CREATING"},
                {"gatewayId": "gw-failed", "status": "FAILED"},
                {"gatewayId": "gw-deleting", "status": "DELETING"},
            ],
        }
        client.get_gateway.return_value = {
            "gatewayId": "gw-ready",
            "name": "Ready GW",
            "status": "READY",
        }
        client.list_gateway_targets.return_value = {"items": []}

        result = scanner.scan_gateways()
        assert len(result) == 1
        assert result[0]["gatewayId"] == "gw-ready"
        client.get_gateway.assert_called_once_with(gatewayIdentifier="gw-ready")

    def test_no_ready_gateways(self):
        scanner, client = _make_scanner()
        client.list_gateways.return_value = {
            "items": [
                {"gatewayId": "gw-1", "status": "CREATING"},
                {"gatewayId": "gw-2", "status": "FAILED"},
            ],
        }

        result = scanner.scan_gateways()
        assert len(result) == 0
        client.get_gateway.assert_not_called()


# ---------------------------------------------------------------------------
# Runtime pagination tests
# ---------------------------------------------------------------------------


class TestRuntimePagination:
    """Tests for scan_runtimes() pagination via nextToken."""

    def test_single_page(self):
        scanner, client = _make_scanner()
        client.list_agent_runtimes.return_value = {
            "agentRuntimes": [
                {"agentRuntimeId": "rt-1", "status": "READY"},
            ],
        }
        client.get_agent_runtime.return_value = {
            "agentRuntimeId": "rt-1",
            "agentRuntimeName": "Runtime One",
            "status": "READY",
        }
        client.list_agent_runtime_endpoints.return_value = {
            "runtimeEndpoints": [],
        }

        result = scanner.scan_runtimes()
        assert len(result) == 1
        assert result[0]["agentRuntimeName"] == "Runtime One"

    def test_multi_page_pagination(self):
        scanner, client = _make_scanner()
        client.list_agent_runtimes.side_effect = [
            {
                "agentRuntimes": [{"agentRuntimeId": "rt-1", "status": "READY"}],
                "nextToken": "page2",
            },
            {
                "agentRuntimes": [{"agentRuntimeId": "rt-2", "status": "READY"}],
            },
        ]
        client.get_agent_runtime.side_effect = [
            {"agentRuntimeId": "rt-1", "agentRuntimeName": "RT1", "status": "READY"},
            {"agentRuntimeId": "rt-2", "agentRuntimeName": "RT2", "status": "READY"},
        ]
        client.list_agent_runtime_endpoints.return_value = {
            "runtimeEndpoints": [],
        }

        result = scanner.scan_runtimes()
        assert len(result) == 2
        assert client.list_agent_runtimes.call_count == 2


# ---------------------------------------------------------------------------
# Runtime READY filtering tests
# ---------------------------------------------------------------------------


class TestRuntimeReadyFiltering:
    """Tests for READY status filtering in scan_runtimes()."""

    def test_only_ready_runtimes_returned(self):
        scanner, client = _make_scanner()
        client.list_agent_runtimes.return_value = {
            "agentRuntimes": [
                {"agentRuntimeId": "rt-ready", "status": "READY"},
                {"agentRuntimeId": "rt-creating", "status": "CREATING"},
                {"agentRuntimeId": "rt-failed", "status": "FAILED"},
            ],
        }
        client.get_agent_runtime.return_value = {
            "agentRuntimeId": "rt-ready",
            "agentRuntimeName": "Ready RT",
            "status": "READY",
        }
        client.list_agent_runtime_endpoints.return_value = {
            "runtimeEndpoints": [],
        }

        result = scanner.scan_runtimes()
        assert len(result) == 1
        assert result[0]["agentRuntimeId"] == "rt-ready"


# ---------------------------------------------------------------------------
# Gateway target pagination tests
# ---------------------------------------------------------------------------


class TestGatewayTargetPagination:
    """Tests for _get_gateway_targets() pagination."""

    def test_target_pagination(self):
        scanner, client = _make_scanner()
        client.list_gateway_targets.side_effect = [
            {
                "items": [{"targetId": "t-1", "status": "READY"}],
                "nextToken": "tpage2",
            },
            {
                "items": [{"targetId": "t-2", "status": "READY"}],
            },
        ]
        client.get_gateway_target.side_effect = [
            {"targetId": "t-1", "name": "Target1"},
            {"targetId": "t-2", "name": "Target2"},
        ]

        targets = scanner._get_gateway_targets("gw-1")
        assert len(targets) == 2
        assert client.list_gateway_targets.call_count == 2

    def test_target_ready_filtering(self):
        scanner, client = _make_scanner()
        client.list_gateway_targets.return_value = {
            "items": [
                {"targetId": "t-ready", "status": "READY"},
                {"targetId": "t-creating", "status": "CREATING"},
            ],
        }
        client.get_gateway_target.return_value = {
            "targetId": "t-ready",
            "name": "Ready Target",
        }

        targets = scanner._get_gateway_targets("gw-1")
        assert len(targets) == 1
        client.get_gateway_target.assert_called_once()


# ---------------------------------------------------------------------------
# Error handling tests (Task 4.4 — discovery portion)
# ---------------------------------------------------------------------------


class TestDiscoveryErrorHandling:
    """Tests for AWS API error handling in AgentCoreScanner."""

    def test_access_denied_exception_propagates(self):
        scanner, client = _make_scanner()
        client.list_gateways.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Not authorized"}},
            "ListGateways",
        )

        with pytest.raises(ClientError) as exc_info:
            scanner.scan_gateways()
        assert "AccessDeniedException" in str(exc_info.value)

    def test_throttling_exception_propagates(self):
        scanner, client = _make_scanner()
        client.list_agent_runtimes.side_effect = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
            "ListAgentRuntimes",
        )

        with pytest.raises(ClientError) as exc_info:
            scanner.scan_runtimes()
        assert "ThrottlingException" in str(exc_info.value)
