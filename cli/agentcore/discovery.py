"""AWS AgentCore resource discovery via boto3.

Scans AgentCore Gateways and Agent Runtimes using the
``bedrock-agentcore-control`` boto3 client, filtering to READY
resources and paginating through all pages.
"""

from __future__ import annotations

import logging
from typing import Any

import boto3
from botocore.config import Config as BotoConfig

from .models import DEFAULT_TIMEOUT, READY_STATUS

logger = logging.getLogger(__name__)


class AgentCoreScanner:
    """Scans AWS AgentCore resources using boto3.

    Configures the boto3 client with connect/read timeouts and
    standard retry mode (3 attempts). All list operations paginate
    via ``nextToken`` and only READY resources are returned.
    """

    def __init__(
        self,
        region: str,
        timeout: int = DEFAULT_TIMEOUT,
        session: boto3.Session | None = None,
    ) -> None:
        """Initialize scanner with AWS region, timeout, and optional boto3 session.

        Args:
            region: AWS region to scan.
            timeout: AWS API call timeout in seconds.
            session: Optional boto3 session (e.g. from STS AssumeRole for
                     cross-account scanning). Uses default credentials if None.
        """
        self.region = region
        self.timeout = timeout

        boto_config = BotoConfig(
            connect_timeout=timeout,
            read_timeout=timeout,
            retries={"max_attempts": 3, "mode": "standard"},
        )
        if session:
            self.client = session.client(
                "bedrock-agentcore-control",
                region_name=region,
                config=boto_config,
            )
        else:
            self.client = boto3.client(
                "bedrock-agentcore-control",
                region_name=region,
                config=boto_config,
            )
        logger.info(
            f"Initialized AgentCore scanner for region: {region} "
            f"(timeout: {timeout}s, cross_account: {session is not None})"
        )

    # ------------------------------------------------------------------
    # Gateway scanning
    # ------------------------------------------------------------------

    def scan_gateways(self) -> list[dict[str, Any]]:
        """Scan all AgentCore Gateways in the region.

        Paginates through ``list_gateways()``, filters to READY status,
        fetches full details via ``get_gateway()``, and collects targets.
        """
        gateways: list[dict[str, Any]] = []
        paginator_params: dict[str, Any] = {}

        while True:
            response = self.client.list_gateways(**paginator_params)

            for item in response.get("items", []):
                if item.get("status") == READY_STATUS:
                    gateway = self.client.get_gateway(
                        gatewayIdentifier=item["gatewayId"]
                    )
                    gateway["targets"] = self._get_gateway_targets(
                        item["gatewayId"]
                    )
                    gateways.append(gateway)
                else:
                    logger.debug(
                        f"Skipping gateway {item['gatewayId']} "
                        f"with status {item['status']}"
                    )

            if "nextToken" in response:
                paginator_params["nextToken"] = response["nextToken"]
            else:
                break

        logger.info(f"Found {len(gateways)} READY gateways")
        return gateways

    def _get_gateway_targets(
        self,
        gateway_id: str,
    ) -> list[dict[str, Any]]:
        """Get all targets for a gateway.

        Paginates through ``list_gateway_targets()`` and fetches full
        details for READY targets.
        """
        targets: list[dict[str, Any]] = []
        paginator_params: dict[str, Any] = {"gatewayIdentifier": gateway_id}

        while True:
            response = self.client.list_gateway_targets(**paginator_params)

            for item in response.get("items", []):
                if item.get("status") == READY_STATUS:
                    target = self.client.get_gateway_target(
                        gatewayIdentifier=gateway_id,
                        targetId=item["targetId"],
                    )
                    targets.append(target)

            if "nextToken" in response:
                paginator_params["nextToken"] = response["nextToken"]
            else:
                break

        return targets

    # ------------------------------------------------------------------
    # Runtime scanning
    # ------------------------------------------------------------------

    def scan_runtimes(self) -> list[dict[str, Any]]:
        """Scan all AgentCore Runtimes in the region.

        Paginates through ``list_agent_runtimes()``, filters to READY
        status, fetches full details via ``get_agent_runtime()``, and
        collects endpoints.
        """
        runtimes: list[dict[str, Any]] = []
        paginator_params: dict[str, Any] = {}

        while True:
            response = self.client.list_agent_runtimes(**paginator_params)

            for item in response.get("agentRuntimes", []):
                if item.get("status") == READY_STATUS:
                    runtime = self.client.get_agent_runtime(
                        agentRuntimeId=item["agentRuntimeId"]
                    )
                    runtime["endpoints"] = self._get_runtime_endpoints(
                        item["agentRuntimeId"]
                    )
                    runtimes.append(runtime)
                else:
                    logger.debug(
                        f"Skipping runtime {item['agentRuntimeId']} "
                        f"with status {item['status']}"
                    )

            if "nextToken" in response:
                paginator_params["nextToken"] = response["nextToken"]
            else:
                break

        logger.info(f"Found {len(runtimes)} READY runtimes")
        return runtimes

    def _get_runtime_endpoints(
        self,
        runtime_id: str,
    ) -> list[dict[str, Any]]:
        """Get all endpoints for a runtime.

        Paginates through ``list_agent_runtime_endpoints()`` and
        returns READY endpoints.
        """
        endpoints: list[dict[str, Any]] = []
        paginator_params: dict[str, Any] = {"agentRuntimeId": runtime_id}

        while True:
            response = self.client.list_agent_runtime_endpoints(
                **paginator_params
            )

            for item in response.get("runtimeEndpoints", []):
                if item.get("status") == READY_STATUS:
                    endpoints.append(item)

            if "nextToken" in response:
                paginator_params["nextToken"] = response["nextToken"]
            else:
                break

        return endpoints
