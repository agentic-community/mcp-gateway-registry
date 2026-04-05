"""Integration tests for AgentCore auto-registration sync flow.

Tests the SyncOrchestrator end-to-end with mocked external dependencies
(boto3 AWS calls and registry HTTP calls). Validates discovery -> registration
-> manifest generation pipeline.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

ACCOUNT_ID = "111122223333"
REGION = "us-east-1"

GATEWAY_CUSTOM_JWT = {
    "gatewayId": "gw-jwt-1",
    "gatewayArn": "arn:aws:bedrock:us-east-1:111122223333:gateway/gw-jwt-1",
    "gatewayUrl": "https://gateway-jwt.example.com",
    "name": "jwt-gateway",
    "description": "OAuth2 gateway",
    "status": "READY",
    "authorizerType": "CUSTOM_JWT",
    "authorizerConfiguration": {
        "customJWTAuthorizer": {
            "discoveryUrl": "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_pnikLWYzO/.well-known/openid-configuration",
            "allowedClients": ["7kqi2l0n47mnfmhfapsf29ch4h"],
        }
    },
    "targets": [],
}

GATEWAY_IAM = {
    "gatewayId": "gw-iam-1",
    "gatewayArn": "arn:aws:bedrock:us-east-1:111122223333:gateway/gw-iam-1",
    "gatewayUrl": "https://gateway-iam.example.com",
    "name": "iam-gateway",
    "description": "IAM gateway",
    "status": "READY",
    "authorizerType": "AWS_IAM",
    "targets": [],
}

GATEWAY_NONE = {
    "gatewayId": "gw-none-1",
    "gatewayArn": "arn:aws:bedrock:us-east-1:111122223333:gateway/gw-none-1",
    "gatewayUrl": "https://gateway-none.example.com",
    "name": "none-gateway",
    "description": "No-auth gateway",
    "status": "READY",
    "authorizerType": "NONE",
    "targets": [],
}

MCP_RUNTIME = {
    "agentRuntimeId": "rt-mcp-1",
    "agentRuntimeArn": "arn:aws:bedrock:us-east-1:111122223333:runtime/rt-mcp-1",
    "agentRuntimeName": "test-mcp-runtime",
    "description": "Test MCP runtime",
    "status": "READY",
    "protocolConfiguration": {"serverProtocol": "MCP"},
    "endpoints": [],
}

HTTP_RUNTIME = {
    "agentRuntimeId": "rt-http-1",
    "agentRuntimeArn": "arn:aws:bedrock:us-east-1:111122223333:runtime/rt-http-1",
    "agentRuntimeName": "test-http-runtime",
    "description": "Test HTTP runtime",
    "status": "READY",
    "protocolConfiguration": {"serverProtocol": "HTTP"},
    "endpoints": [],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mock_sts():
    """Create a mock STS client that returns a fixed account ID."""
    mock = MagicMock()
    mock.get_caller_identity.return_value = {"Account": ACCOUNT_ID}
    return mock


def _mock_agentcore_client(gateways=None, runtimes=None):
    """Create a mock bedrock-agentcore-control client."""
    client = MagicMock()

    # list_gateways
    gw_items = []
    for gw in (gateways or []):
        gw_items.append({"gatewayId": gw["gatewayId"], "status": gw["status"]})
    client.list_gateways.return_value = {"items": gw_items}

    # get_gateway -- return the full gateway dict for each ID
    def _get_gateway(gatewayIdentifier):
        for gw in (gateways or []):
            if gw["gatewayId"] == gatewayIdentifier:
                return dict(gw)
        return {}

    client.get_gateway.side_effect = _get_gateway

    # list_gateway_targets -- return empty by default
    client.list_gateway_targets.return_value = {"items": []}
    client.get_gateway_target.return_value = {}

    # list_agent_runtimes
    rt_items = []
    for rt in (runtimes or []):
        rt_items.append({"agentRuntimeId": rt["agentRuntimeId"], "status": rt["status"]})
    client.list_agent_runtimes.return_value = {"agentRuntimes": rt_items}

    # get_agent_runtime
    def _get_runtime(agentRuntimeId):
        for rt in (runtimes or []):
            if rt["agentRuntimeId"] == agentRuntimeId:
                return dict(rt)
        return {}

    client.get_agent_runtime.side_effect = _get_runtime

    # list_agent_runtime_endpoints
    client.list_agent_runtime_endpoints.return_value = {"runtimeEndpoints": []}

    return client


def _build_orchestrator(
    gateways=None,
    runtimes=None,
    dry_run=False,
    overwrite=False,
    include_mcp_targets=False,
    registry_client=None,
    manifest_path="/tmp/test_manifest.json",
):
    """Build a SyncOrchestrator with mocked AWS and registry dependencies."""
    mock_ac_client = _mock_agentcore_client(gateways=gateways, runtimes=runtimes)
    mock_sts = _mock_sts()

    def _boto3_client(service, **kwargs):
        if service == "sts":
            return mock_sts
        if service == "bedrock-agentcore-control":
            return mock_ac_client
        return MagicMock()

    with patch("cli.agentcore.registration.boto3") as reg_boto3, \
         patch("cli.agentcore.discovery.boto3") as disc_boto3:
        reg_boto3.client.side_effect = _boto3_client
        disc_boto3.client.side_effect = _boto3_client

        from cli.agentcore.discovery import AgentCoreScanner
        from cli.agentcore.registration import RegistrationBuilder, SyncOrchestrator

        scanner = AgentCoreScanner(region=REGION)
        scanner.client = mock_ac_client

        builder = RegistrationBuilder(region=REGION)

    if registry_client is None:
        registry_client = MagicMock()

    orch = SyncOrchestrator(
        scanner=scanner,
        builder=builder,
        registry_client=registry_client,
        dry_run=dry_run,
        overwrite=overwrite,
        include_mcp_targets=include_mcp_targets,
        manifest_path=manifest_path,
    )
    return orch, registry_client


# ---------------------------------------------------------------------------
# End-to-end flow: discovery -> registration -> manifest
# ---------------------------------------------------------------------------


class TestEndToEndFlow:
    """Full sync pipeline with gateways and runtimes."""

    def test_gateway_discovery_registration_manifest(self):
        """CUSTOM_JWT gateway: register and collect manifest entry with OIDC metadata."""
        orch, registry = _build_orchestrator(gateways=[GATEWAY_CUSTOM_JWT])

        orch.sync_gateways()

        # Gateway registered
        assert len(orch.results) == 1
        assert orch.results[0]["status"] == "registered"
        assert orch.results[0]["resource_type"] == "gateway"
        registry.register_service.assert_called_once()

        # Manifest entry collected
        assert len(orch._manifest_entries) == 1
        entry = orch._manifest_entries[0]
        assert entry["server_path"] == "/jwt-gateway"
        assert "cognito-idp" in entry["discovery_url"]
        assert entry["allowed_clients"] == ["7kqi2l0n47mnfmhfapsf29ch4h"]
        assert entry["idp_vendor"] == "cognito"

    def test_mcp_runtime_registered_as_server(self):
        """MCP runtime -> registered as MCP Server via register_service."""
        orch, registry = _build_orchestrator(runtimes=[MCP_RUNTIME])

        orch.sync_runtimes()

        assert len(orch.results) == 1
        assert orch.results[0]["status"] == "registered"
        assert orch.results[0]["registration_type"] == "mcp_server"
        assert orch.results[0]["resource_type"] == "runtime"
        registry.register_service.assert_called_once()
        registry.register_agent.assert_not_called()

    def test_http_runtime_registered_as_agent(self):
        """HTTP runtime -> registered as A2A Agent via register_agent."""
        orch, registry = _build_orchestrator(runtimes=[HTTP_RUNTIME])

        orch.sync_runtimes()

        assert len(orch.results) == 1
        assert orch.results[0]["status"] == "registered"
        assert orch.results[0]["registration_type"] == "agent"
        registry.register_agent.assert_called_once()
        registry.register_service.assert_not_called()

    def test_full_sync_gateways_and_runtimes(self):
        """Sync both gateways and runtimes in a single run."""
        orch, registry = _build_orchestrator(
            gateways=[GATEWAY_NONE],
            runtimes=[MCP_RUNTIME, HTTP_RUNTIME],
        )

        orch.sync_gateways()
        orch.sync_runtimes()

        assert len(orch.results) == 3
        statuses = [r["status"] for r in orch.results]
        assert all(s == "registered" for s in statuses)
        # 1 gateway + 1 MCP runtime = 2 register_service calls
        assert registry.register_service.call_count == 2
        # 1 HTTP runtime = 1 register_agent call
        assert registry.register_agent.call_count == 1


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------


class TestDryRunMode:
    """Dry-run: no registry calls, manifest entries collected but not written."""

    def test_dry_run_skips_registry_calls(self):
        orch, registry = _build_orchestrator(
            gateways=[GATEWAY_CUSTOM_JWT, GATEWAY_NONE],
            runtimes=[MCP_RUNTIME, HTTP_RUNTIME],
            dry_run=True,
        )

        orch.sync_gateways()
        orch.sync_runtimes()

        # No registry calls
        registry.register_service.assert_not_called()
        registry.register_agent.assert_not_called()

        # All results are dry_run
        assert len(orch.results) == 4
        assert all(r["status"] == "dry_run" for r in orch.results)

    def test_dry_run_collects_manifest_entries(self):
        """Dry-run still collects manifest entries for CUSTOM_JWT gateways."""
        orch, _ = _build_orchestrator(
            gateways=[GATEWAY_CUSTOM_JWT],
            dry_run=True,
        )

        orch.sync_gateways()

        assert len(orch._manifest_entries) == 1
        assert orch._manifest_entries[0]["idp_vendor"] == "cognito"

    def test_dry_run_does_not_write_manifest(self, tmp_path):
        """Dry-run mode does not create the manifest file."""
        manifest_file = tmp_path / "manifest.json"
        orch, _ = _build_orchestrator(
            gateways=[GATEWAY_CUSTOM_JWT],
            dry_run=True,
            manifest_path=str(manifest_file),
        )

        orch.sync_gateways()
        orch.write_manifest()

        assert not manifest_file.exists()


# ---------------------------------------------------------------------------
# Mixed deployment: CUSTOM_JWT, IAM, NONE gateways
# ---------------------------------------------------------------------------


class TestMixedDeployment:
    """Mixed authorizer types in a single sync run."""

    def test_mixed_gateways_all_registered(self):
        """All three authorizer types register successfully."""
        orch, registry = _build_orchestrator(
            gateways=[GATEWAY_CUSTOM_JWT, GATEWAY_IAM, GATEWAY_NONE],
        )

        orch.sync_gateways()

        # All 3 gateways registered
        assert len(orch.results) == 3
        assert all(r["status"] == "registered" for r in orch.results)
        assert registry.register_service.call_count == 3

    def test_only_custom_jwt_collects_manifest(self):
        """Only CUSTOM_JWT gateways produce manifest entries."""
        orch, _ = _build_orchestrator(
            gateways=[GATEWAY_CUSTOM_JWT, GATEWAY_IAM, GATEWAY_NONE],
        )

        orch.sync_gateways()

        assert len(orch._manifest_entries) == 1
        assert orch._manifest_entries[0]["gateway_arn"] == GATEWAY_CUSTOM_JWT["gatewayArn"]

    def test_mixed_with_runtimes(self):
        """Mixed gateways + mixed runtimes in a single sync."""
        orch, registry = _build_orchestrator(
            gateways=[GATEWAY_IAM, GATEWAY_NONE],
            runtimes=[MCP_RUNTIME, HTTP_RUNTIME],
        )

        orch.sync_gateways()
        orch.sync_runtimes()

        assert len(orch.results) == 4
        types = {r["resource_type"] for r in orch.results}
        assert types == {"gateway", "runtime"}

        # 2 gateways + 1 MCP runtime = 3 register_service
        assert registry.register_service.call_count == 3
        # 1 HTTP runtime = 1 register_agent
        assert registry.register_agent.call_count == 1

    def test_iam_gateway_auth_scheme_is_bearer(self):
        """IAM gateways get auth_scheme=bearer in registration."""
        orch, registry = _build_orchestrator(gateways=[GATEWAY_IAM])

        orch.sync_gateways()

        assert len(orch.results) == 1
        assert orch.results[0]["status"] == "registered"
        call_args = registry.register_service.call_args
        reg = call_args[0][0]
        assert reg.auth_scheme == "bearer"

    def test_none_gateway_auth_scheme_is_none(self):
        """NONE gateways get auth_scheme=none in registration."""
        orch, registry = _build_orchestrator(gateways=[GATEWAY_NONE])

        orch.sync_gateways()

        call_args = registry.register_service.call_args
        reg = call_args[0][0]
        assert reg.auth_scheme == "none"


# ---------------------------------------------------------------------------
# Manifest file writing
# ---------------------------------------------------------------------------


class TestManifestWriting:
    """Tests for token refresh manifest file output."""

    def test_manifest_written_with_correct_structure(self, tmp_path):
        """Manifest file contains correct OIDC metadata for CUSTOM_JWT gateways."""
        manifest_file = tmp_path / "manifest.json"
        orch, _ = _build_orchestrator(
            gateways=[GATEWAY_CUSTOM_JWT],
            manifest_path=str(manifest_file),
        )

        orch.sync_gateways()
        orch.write_manifest()

        data = json.loads(manifest_file.read_text())
        assert len(data) == 1
        entry = data[0]
        assert entry["server_path"] == "/jwt-gateway"
        assert entry["gateway_arn"] == GATEWAY_CUSTOM_JWT["gatewayArn"]
        assert "cognito-idp" in entry["discovery_url"]
        assert entry["allowed_clients"] == ["7kqi2l0n47mnfmhfapsf29ch4h"]
        assert entry["idp_vendor"] == "cognito"

    def test_no_manifest_for_non_jwt_gateways(self, tmp_path):
        """IAM and NONE gateways produce no manifest entries."""
        manifest_file = tmp_path / "manifest.json"
        orch, _ = _build_orchestrator(
            gateways=[GATEWAY_IAM, GATEWAY_NONE],
            manifest_path=str(manifest_file),
        )

        orch.sync_gateways()
        orch.write_manifest()

        # No manifest file created (no CUSTOM_JWT gateways)
        assert not manifest_file.exists()

    def test_runtimes_produce_no_manifest_entries(self, tmp_path):
        """Runtimes do not contribute to the manifest."""
        manifest_file = tmp_path / "manifest.json"
        orch, _ = _build_orchestrator(
            runtimes=[MCP_RUNTIME, HTTP_RUNTIME],
            manifest_path=str(manifest_file),
        )

        orch.sync_runtimes()
        orch.write_manifest()

        assert not manifest_file.exists()
