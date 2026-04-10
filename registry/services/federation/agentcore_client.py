"""
AWS Agent Registry federation client.

Fetches registry records from AWS Agent Registry via boto3 control plane API
(bedrock-agentcore-control) and transforms them to the gateway's internal format.

Descriptor type mapping:
    MCP -> MCP Servers (server model)
    A2A -> Agents (agent card model)
    CUSTOM -> Agents (agent card model, self-referencing URL)
    AGENT_SKILLS -> Skills (skill card model, inline content stored in DB)
"""

import json
import logging
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    as_completed,
)
from datetime import UTC, datetime
from typing import Any

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
)

from ...schemas.federation_schema import AgentCoreRegistryConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


# Constants
AGENTCORE_SOURCE: str = "agentcore"
AGENTCORE_ATTRIBUTION: str = "AWS Agent Registry"
DEFAULT_AWS_REGION: str = "us-east-1"
DEFAULT_SYNC_STATUS: str = "APPROVED"
DEFAULT_SYNC_TIMEOUT_SECONDS: int = 300
DEFAULT_MAX_CONCURRENT_FETCHES: int = 5
MAX_RESULTS_PER_PAGE: int = 100


def _safe_parse_json(
    content: str,
    context: str = "",
) -> dict[str, Any]:
    """Parse JSON string safely, returning empty dict on failure.

    Args:
        content: JSON string to parse
        context: Description for error logging

    Returns:
        Parsed dict or empty dict on failure
    """
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse JSON for {context}: {e}")
        return {}


def _sanitize_path_segment(
    name: str,
) -> str:
    """Sanitize a name for use in a URL path segment.

    Replaces slashes and spaces with hyphens, lowercases.

    Args:
        name: Raw name string

    Returns:
        Sanitized path-safe string
    """
    return name.replace("/", "-").replace(" ", "-").replace("_", "-").lower().strip("-")


def _extract_transport_info(
    server_content: dict[str, Any],
) -> tuple[str, str | None]:
    """Extract transport type and proxy URL from MCP server definition.

    Args:
        server_content: Parsed MCP server inlineContent

    Returns:
        Tuple of (transport_type, proxy_url)
    """
    transport_type = "streamable-http"
    proxy_url = None

    remotes = server_content.get("remotes", [])
    if remotes:
        remote = remotes[0]
        transport_type = remote.get("type", "streamable-http")
        proxy_url = remote.get("url")
    else:
        packages = server_content.get("packages", [])
        if packages:
            transport = packages[0].get("transport", {})
            transport_type = transport.get("type", "streamable-http")
            if transport_type in ("streamable-http", "http"):
                proxy_url = transport.get("url")

    return transport_type, proxy_url


class AgentCoreFederationClient:
    """Client for fetching records from AWS Agent Registry.

    Uses boto3 bedrock-agentcore-control client (control plane) to:
    - List registries in the AWS account
    - List and fetch registry records
    - Transform records to gateway internal format by descriptor type
    """

    def __init__(
        self,
        aws_region: str = DEFAULT_AWS_REGION,
        timeout_seconds: int = 30,
        retry_attempts: int = 3,
    ) -> None:
        """Initialize AWS Agent Registry federation client.

        Args:
            aws_region: AWS region for API calls
            timeout_seconds: boto3 read timeout
            retry_attempts: Number of retry attempts for API calls
        """
        self.aws_region = aws_region
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts

        boto_config = BotoConfig(
            region_name=aws_region,
            read_timeout=timeout_seconds,
            retries={"max_attempts": retry_attempts, "mode": "adaptive"},
        )
        self._client = boto3.client(
            "bedrock-agentcore-control",
            config=boto_config,
        )

        # Health indicator state
        self._last_sync_success: bool = False
        self._last_sync_time: str | None = None
        self._last_sync_record_count: int = 0
        self._last_sync_error: str | None = None

        # Cache for per-registry clients (keyed by cache key)
        self._registry_clients: dict[str, Any] = {}

        logger.info(
            f"AgentCoreFederationClient initialized "
            f"(region={aws_region}, timeout={timeout_seconds}s, retries={retry_attempts})"
        )

    def _get_client_for_registry(
        self,
        reg_config: AgentCoreRegistryConfig,
    ) -> Any:
        """Get a boto3 client for the given registry config.

        Returns the default client when the registry uses the same region
        and no cross-account role. Creates a region-specific or cross-account
        client via STS AssumeRole when needed.

        Args:
            reg_config: Registry configuration (may include aws_region, assume_role_arn)

        Returns:
            boto3 bedrock-agentcore-control client
        """
        registry_region = reg_config.aws_region or self.aws_region
        has_custom_region = registry_region != self.aws_region
        has_role = bool(reg_config.assume_role_arn)

        # Same region, no role assumption -> use default client
        if not has_custom_region and not has_role:
            return self._client

        # Build cache key from region + role
        cache_key = f"{registry_region}:{reg_config.assume_role_arn or 'default'}"
        if cache_key in self._registry_clients:
            return self._registry_clients[cache_key]

        return self._create_registry_client(
            reg_config=reg_config,
            registry_region=registry_region,
            cache_key=cache_key,
        )

    def _create_registry_client(
        self,
        reg_config: AgentCoreRegistryConfig,
        registry_region: str,
        cache_key: str,
    ) -> Any:
        """Create a boto3 client for cross-account or cross-region access.

        Args:
            reg_config: Registry configuration
            registry_region: Resolved AWS region for this registry
            cache_key: Cache key for storing the created client

        Returns:
            boto3 bedrock-agentcore-control client
        """
        boto_config = BotoConfig(
            region_name=registry_region,
            read_timeout=self.timeout_seconds,
            retries={"max_attempts": self.retry_attempts, "mode": "adaptive"},
        )

        if reg_config.assume_role_arn:
            logger.info(
                f"Assuming role {reg_config.assume_role_arn} for registry "
                f"{reg_config.registry_id} (region={registry_region})"
            )
            client = self._create_cross_account_client(
                role_arn=reg_config.assume_role_arn,
                registry_id=reg_config.registry_id,
                registry_region=registry_region,
                boto_config=boto_config,
            )
        else:
            # Different region, same account
            logger.info(
                f"Creating region-specific client for registry "
                f"{reg_config.registry_id} (region={registry_region})"
            )
            client = boto3.client(
                "bedrock-agentcore-control",
                config=boto_config,
            )

        self._registry_clients[cache_key] = client
        return client

    def _create_cross_account_client(
        self,
        role_arn: str,
        registry_id: str,
        registry_region: str,
        boto_config: BotoConfig,
    ) -> Any:
        """Create a boto3 client using STS AssumeRole for cross-account access.

        Args:
            role_arn: IAM role ARN to assume
            registry_id: Registry ID (for session name)
            registry_region: AWS region for the target registry
            boto_config: Boto3 client config

        Returns:
            boto3 bedrock-agentcore-control client with assumed role credentials
        """
        try:
            sts_client = boto3.client("sts", region_name=registry_region)
            assumed = sts_client.assume_role(
                RoleArn=role_arn,
                RoleSessionName=f"agentcore-federation-{registry_id[:20]}",
                DurationSeconds=3600,
            )

            credentials = assumed["Credentials"]
            cross_account_client = boto3.client(
                "bedrock-agentcore-control",
                config=boto_config,
                aws_access_key_id=credentials["AccessKeyId"],
                aws_secret_access_key=credentials["SecretAccessKey"],
                aws_session_token=credentials["SessionToken"],
            )

            logger.info(f"Cross-account client created for role {role_arn}")
            return cross_account_client

        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to assume role {role_arn}: {e}")
            raise

    def get_health_status(self) -> dict[str, Any]:
        """Return health indicator for AWS Agent Registry federation.

        Returns:
            Dict with last sync status, time, record count, and error (if any)
        """
        return {
            "source": AGENTCORE_SOURCE,
            "healthy": self._last_sync_success,
            "last_sync_time": self._last_sync_time,
            "last_sync_record_count": self._last_sync_record_count,
            "last_sync_error": self._last_sync_error,
            "aws_region": self.aws_region,
        }

    def list_registries(self) -> list[dict[str, Any]]:
        """List all AgentCore registries in the AWS account.

        Returns:
            List of registry summary dicts
        """
        registries: list[dict[str, Any]] = []
        next_token = None

        try:
            while True:
                params: dict[str, Any] = {
                    "maxResults": MAX_RESULTS_PER_PAGE,
                    "status": "READY",
                }
                if next_token:
                    params["nextToken"] = next_token

                response = self._client.list_registries(**params)
                registries.extend(response.get("registries", []))

                next_token = response.get("nextToken")
                if not next_token:
                    break

            logger.info(f"Found {len(registries)} AgentCore registries")
            return registries

        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to list AgentCore registries: {e}")
            return []

    def list_registry_records(
        self,
        registry_id: str,
        descriptor_type: str | None = None,
        status: str = DEFAULT_SYNC_STATUS,
    ) -> list[dict[str, Any]]:
        """List all registry records from an AgentCore registry.

        Handles pagination automatically.

        Args:
            registry_id: Registry ID or ARN
            descriptor_type: Filter by descriptor type (MCP, A2A, CUSTOM, AGENT_SKILLS)
            status: Filter by record status (default: APPROVED)

        Returns:
            List of registry record summary dicts
        """
        records: list[dict[str, Any]] = []
        next_token = None

        try:
            while True:
                params: dict[str, Any] = {
                    "registryId": registry_id,
                    "maxResults": MAX_RESULTS_PER_PAGE,
                }
                if descriptor_type:
                    params["descriptorType"] = descriptor_type
                if status:
                    params["status"] = status
                if next_token:
                    params["nextToken"] = next_token

                response = self._client.list_registry_records(**params)
                records.extend(response.get("registryRecords", []))

                next_token = response.get("nextToken")
                if not next_token:
                    break

            logger.info(
                f"Found {len(records)} records in registry {registry_id} "
                f"(descriptor_type={descriptor_type}, status={status})"
            )
            return records

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            logger.error(
                f"Failed to list records from registry {registry_id}: "
                f"{error_code} - {e}"
            )
            return []
        except BotoCoreError as e:
            logger.error(f"Failed to list records from registry {registry_id}: {e}")
            return []

    def get_registry_record(
        self,
        registry_id: str,
        record_id: str,
    ) -> dict[str, Any] | None:
        """Get full details of a single registry record.

        Args:
            registry_id: Registry ID or ARN
            record_id: Record ID or ARN

        Returns:
            Full registry record dict or None if fetch fails
        """
        try:
            response = self._client.get_registry_record(
                registryId=registry_id,
                recordId=record_id,
            )

            # Remove ResponseMetadata added by boto3
            response.pop("ResponseMetadata", None)
            return response

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "ResourceNotFoundException":
                logger.warning(
                    f"Record {record_id} not found in registry {registry_id}"
                )
            else:
                logger.error(
                    f"Failed to get record {record_id} from {registry_id}: "
                    f"{error_code} - {e}"
                )
            return None
        except BotoCoreError as e:
            logger.error(
                f"Failed to get record {record_id} from {registry_id}: {e}"
            )
            return None

    def fetch_all_records(
        self,
        registry_configs: list[AgentCoreRegistryConfig],
        sync_timeout_seconds: int = DEFAULT_SYNC_TIMEOUT_SECONDS,
        max_concurrent_fetches: int = DEFAULT_MAX_CONCURRENT_FETCHES,
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch and transform all records from configured AWS Agent Registry registries.

        Uses ThreadPoolExecutor for parallel get_registry_record calls
        and enforces an overall sync timeout.

        Args:
            registry_configs: List of registry configurations
            sync_timeout_seconds: Max time for entire sync (default 300s / 5 min)
            max_concurrent_fetches: Thread pool size for parallel fetches (default 5)

        Returns:
            Dict with keys "servers", "agents", "skills" containing
            transformed record dicts ready for registration
        """
        start_time = time.monotonic()

        result: dict[str, list[dict[str, Any]]] = {
            "servers": [],
            "agents": [],
            "skills": [],
        }

        try:
            for reg_config in registry_configs:
                # Check overall timeout before starting next registry
                elapsed = time.monotonic() - start_time
                if elapsed >= sync_timeout_seconds:
                    logger.warning(
                        f"Sync timeout reached ({sync_timeout_seconds}s) "
                        f"after processing some registries. Stopping gracefully."
                    )
                    break

                self._fetch_from_registry(
                    reg_config=reg_config,
                    result=result,
                    start_time=start_time,
                    sync_timeout_seconds=sync_timeout_seconds,
                    max_concurrent_fetches=max_concurrent_fetches,
                )

        except Exception as e:
            logger.error(f"Unexpected error during AgentCore sync: {e}")
            self._last_sync_error = str(e)

        # Log timing
        elapsed_total = time.monotonic() - start_time
        minutes = int(elapsed_total // 60)
        seconds = elapsed_total % 60
        if minutes > 0:
            logger.info(f"AgentCore sync completed in {minutes} minutes and {seconds:.1f} seconds")
        else:
            logger.info(f"AgentCore sync completed in {seconds:.1f} seconds")

        # Update health indicator
        total_count = len(result["servers"]) + len(result["agents"]) + len(result["skills"])
        self._last_sync_success = True
        self._last_sync_time = datetime.now(UTC).isoformat()
        self._last_sync_record_count = total_count
        self._last_sync_error = None

        return result

    def _fetch_from_registry(
        self,
        reg_config: AgentCoreRegistryConfig,
        result: dict[str, list[dict[str, Any]]],
        start_time: float,
        sync_timeout_seconds: int,
        max_concurrent_fetches: int,
    ) -> None:
        """Fetch and transform records from a single registry.

        Args:
            reg_config: Registry configuration
            result: Result dict to append transformed records to
            start_time: Monotonic start time for timeout calculation
            sync_timeout_seconds: Overall sync timeout
            max_concurrent_fetches: Thread pool size
        """
        registry_id = reg_config.registry_id
        status_filter = reg_config.sync_status_filter
        account_info = f" (account={reg_config.aws_account_id})" if reg_config.aws_account_id else ""
        logger.info(f"Fetching records from AWS Agent Registry: {registry_id}{account_info}")

        # Swap to cross-account client if needed
        original_client = self._client
        try:
            self._client = self._get_client_for_registry(reg_config)
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Skipping registry {registry_id}: failed to get client: {e}")
            return

        try:
            record_summaries = self.list_registry_records(
                registry_id=registry_id,
                status=status_filter,
            )

            # Filter to configured descriptor types
            filtered_summaries = [
                s for s in record_summaries
                if s.get("descriptorType", "") in reg_config.descriptor_types
            ]

            skipped = len(record_summaries) - len(filtered_summaries)
            if skipped > 0:
                logger.debug(f"Skipped {skipped} records with non-configured descriptor types")

            # Fetch full details in parallel
            timeout_remaining = sync_timeout_seconds - (time.monotonic() - start_time)
            fetched_records = self._fetch_records_parallel(
                registry_id=registry_id,
                summaries=filtered_summaries,
                max_workers=max_concurrent_fetches,
                timeout_remaining=timeout_remaining,
            )
        finally:
            # Restore original client
            self._client = original_client

        # Transform and route to correct bucket
        for full_record in fetched_records:
            transformed = self._transform_record(full_record, registry_id)
            if not transformed:
                continue

            descriptor_type = full_record.get("descriptorType", "")
            if descriptor_type == "MCP":
                result["servers"].append(transformed)
            elif descriptor_type in ("A2A", "CUSTOM"):
                result["agents"].append(transformed)
            elif descriptor_type == "AGENT_SKILLS":
                result["skills"].append(transformed)

        logger.info(
            f"Registry {registry_id}: "
            f"{len(result['servers'])} servers, "
            f"{len(result['agents'])} agents, "
            f"{len(result['skills'])} skills"
        )

    def _fetch_records_parallel(
        self,
        registry_id: str,
        summaries: list[dict[str, Any]],
        max_workers: int = DEFAULT_MAX_CONCURRENT_FETCHES,
        timeout_remaining: float = DEFAULT_SYNC_TIMEOUT_SECONDS,
    ) -> list[dict[str, Any]]:
        """Fetch full record details in parallel using ThreadPoolExecutor.

        boto3 clients are thread-safe for read operations.

        Args:
            registry_id: Registry ID to fetch from
            summaries: List of record summaries to fetch details for
            max_workers: Maximum concurrent threads (default 5)
            timeout_remaining: Seconds remaining before sync timeout

        Returns:
            List of full record dicts (failed fetches are excluded)
        """
        if not summaries:
            return []

        if timeout_remaining <= 0:
            logger.warning("No time remaining for parallel fetch, skipping")
            return []

        records: list[dict[str, Any]] = []
        failed_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_record_id = {
                executor.submit(
                    self.get_registry_record,
                    registry_id,
                    summary.get("recordId", ""),
                ): summary.get("recordId", "")
                for summary in summaries
            }

            try:
                for future in as_completed(future_to_record_id, timeout=timeout_remaining):
                    record_id = future_to_record_id[future]
                    try:
                        full_record = future.result()
                        if full_record:
                            records.append(full_record)
                        else:
                            failed_count += 1
                            logger.warning(f"Skipping record {record_id}: fetch returned None")
                    except Exception as e:
                        failed_count += 1
                        logger.error(f"Error fetching record {record_id}: {e}")
            except FuturesTimeoutError:
                failed_count += len(future_to_record_id) - len(records)
                logger.warning(
                    f"Parallel fetch timed out for registry {registry_id}, "
                    f"got {len(records)} of {len(summaries)} records"
                )

        logger.info(
            f"Parallel fetch from {registry_id}: "
            f"{len(records)} succeeded, {failed_count} failed "
            f"(of {len(summaries)} total, {max_workers} workers)"
        )

        return records

    def _transform_record(
        self,
        record: dict[str, Any],
        registry_id: str,
    ) -> dict[str, Any] | None:
        """Transform a record to internal format based on descriptor type.

        Args:
            record: Full record from get_registry_record
            registry_id: Source registry ID

        Returns:
            Transformed dict ready for registration, or None on failure
        """
        descriptor_type = record.get("descriptorType", "")
        descriptors = record.get("descriptors", {})

        if descriptor_type == "MCP":
            return self._transform_mcp_record(record, descriptors, registry_id)
        elif descriptor_type == "A2A":
            return self._transform_a2a_record(record, descriptors, registry_id)
        elif descriptor_type == "CUSTOM":
            return self._transform_custom_record(record, descriptors, registry_id)
        elif descriptor_type == "AGENT_SKILLS":
            return self._transform_skills_record(record, descriptors, registry_id)
        else:
            logger.warning(f"Unknown descriptor type: {descriptor_type}")
            return None

    def _transform_mcp_record(
        self,
        record: dict[str, Any],
        descriptors: dict[str, Any],
        registry_id: str,
    ) -> dict[str, Any]:
        """Transform MCP descriptor record to server registration data.

        Args:
            record: Full AgentCore record
            descriptors: Descriptors section of the record
            registry_id: Source registry ID

        Returns:
            Server data dict for ServerService.register_server()
        """
        record_name = record.get("name", "")
        record_id = record.get("recordId", "")
        description = record.get("description", "")
        version = record.get("recordVersion", "1.0.0")

        # Parse MCP server definition
        mcp_desc = descriptors.get("mcp", {})
        server_content = _safe_parse_json(
            mcp_desc.get("server", {}).get("inlineContent", "{}"),
            context=f"MCP server for {record_name}",
        )
        tools_content = _safe_parse_json(
            mcp_desc.get("tools", {}).get("inlineContent", "{}"),
            context=f"MCP tools for {record_name}",
        )

        # Extract transport info
        transport_type, proxy_url = _extract_transport_info(server_content)

        # Check synchronizationConfiguration for URL as fallback
        sync_config = record.get("synchronizationConfiguration", {})
        from_url = sync_config.get("fromUrl", {})
        sync_url = from_url.get("url")
        if sync_url and not proxy_url:
            proxy_url = sync_url

        # Count tools
        tools_list = tools_content.get("tools", [])
        num_tools = len(tools_list) if isinstance(tools_list, list) else 0

        # Use description from server content if record description is empty
        if not description:
            description = server_content.get("description", "")

        path_segment = _sanitize_path_segment(record_name)

        tags = [
            "agentcore",
            "bedrock",
            "federated",
            "mcp",
            f"registry-{registry_id[:12]}",
        ]

        # Extract AWS timestamps (datetime objects from boto3)
        created_at = record.get("createdAt")
        updated_at = record.get("lastUpdatedAt")

        return {
            "source": AGENTCORE_SOURCE,
            "server_name": record_name,
            "description": description,
            "version": version,
            "title": server_content.get("title", record_name),
            "proxy_pass_url": proxy_url,
            "transport_type": transport_type,
            "requires_auth": False,
            "auth_headers": [],
            "tags": tags,
            "metadata": {
                "agentcore_registry_id": registry_id,
                "agentcore_record_id": record_id,
                "descriptor_type": "MCP",
                "created_at": created_at.isoformat() if created_at else None,
                "updated_at": updated_at.isoformat() if updated_at else None,
            },
            "cached_at": datetime.now(UTC).isoformat(),
            "is_read_only": True,
            "attribution_label": AGENTCORE_ATTRIBUTION,
            "path": f"/agentcore-{path_segment}",
            "is_enabled": True,
            "health_status": "unknown",
            "num_tools": num_tools,
        }

    def _transform_a2a_record(
        self,
        record: dict[str, Any],
        descriptors: dict[str, Any],
        registry_id: str,
    ) -> dict[str, Any]:
        """Transform A2A descriptor record to agent card registration data.

        Args:
            record: Full AgentCore record
            descriptors: Descriptors section
            registry_id: Source registry ID

        Returns:
            Agent data dict for AgentService.register_agent()
        """
        record_name = record.get("name", "")
        record_id = record.get("recordId", "")
        description = record.get("description", "")
        version = record.get("recordVersion", "1.0.0")

        # Parse A2A agent card
        a2a_desc = descriptors.get("a2a", {})
        agent_card_content = _safe_parse_json(
            a2a_desc.get("agentCard", {}).get("inlineContent", "{}"),
            context=f"A2A agent card for {record_name}",
        )

        path_segment = _sanitize_path_segment(record_name)

        # Extract A2A protocol fields
        agent_url = agent_card_content.get("url", "")
        agent_name = agent_card_content.get("name", record_name)
        agent_description = agent_card_content.get("description", description)
        agent_version = agent_card_content.get("version", version)

        tags = [
            "agentcore",
            "bedrock",
            "federated",
            "a2a",
            f"registry-{registry_id[:12]}",
        ]

        # Extract AWS timestamps (datetime objects from boto3)
        created_at = record.get("createdAt")
        updated_at = record.get("lastUpdatedAt")

        return {
            "source": AGENTCORE_SOURCE,
            "name": agent_name,
            "description": agent_description,
            "url": agent_url,
            "path": f"/agents/agentcore-{path_segment}",
            "version": agent_version,
            "protocol_version": agent_card_content.get("protocolVersion", "1.0"),
            "capabilities": agent_card_content.get("capabilities", {}),
            "skills": agent_card_content.get("skills", []),
            "provider": agent_card_content.get("provider"),
            "security_schemes": agent_card_content.get("securitySchemes", {}),
            "default_input_modes": agent_card_content.get("defaultInputModes", ["text/plain"]),
            "default_output_modes": agent_card_content.get("defaultOutputModes", ["text/plain"]),
            "tags": tags,
            "is_enabled": True,
            "is_read_only": True,
            "attribution_label": AGENTCORE_ATTRIBUTION,
            "supported_protocol": "a2a",
            "metadata": {
                "agentcore_registry_id": registry_id,
                "agentcore_record_id": record_id,
                "descriptor_type": "A2A",
                "created_at": created_at.isoformat() if created_at else None,
                "updated_at": updated_at.isoformat() if updated_at else None,
            },
            "cached_at": datetime.now(UTC).isoformat(),
        }

    def _transform_custom_record(
        self,
        record: dict[str, Any],
        descriptors: dict[str, Any],
        registry_id: str,
    ) -> dict[str, Any]:
        """Transform CUSTOM descriptor record to agent card registration data.

        Custom descriptors are treated as agents. Uses a self-referencing URL
        pointing to our own agent detail endpoint.

        Args:
            record: Full AgentCore record
            descriptors: Descriptors section
            registry_id: Source registry ID

        Returns:
            Agent data dict for AgentService.register_agent()
        """
        record_name = record.get("name", "")
        record_id = record.get("recordId", "")
        description = record.get("description", "")
        version = record.get("recordVersion", "1.0.0")

        # Parse custom content
        custom_desc = descriptors.get("custom", {})
        custom_content = _safe_parse_json(
            custom_desc.get("inlineContent", "{}"),
            context=f"CUSTOM descriptor for {record_name}",
        )

        path_segment = _sanitize_path_segment(record_name)
        agent_path = f"/agents/agentcore-custom-{path_segment}"

        # Use our own agent detail endpoint as the URL
        from registry.core.config import settings

        agent_url = f"{settings.registry_url}/api{agent_path}"
        original_url = (
            custom_content.get("url")
            or custom_content.get("endpoint")
            or custom_content.get("baseUrl")
        )

        tags = [
            "agentcore",
            "bedrock",
            "federated",
            "custom",
            f"registry-{registry_id[:12]}",
        ]

        # Map custom provider to AgentProvider format (needs organization + url)
        raw_provider = custom_content.get("provider")
        provider_data = None
        if isinstance(raw_provider, dict):
            org = raw_provider.get("organization") or raw_provider.get("name", "")
            provider_url = raw_provider.get("url", "")
            if org and provider_url:
                provider_data = {"organization": org, "url": provider_url}

        # Extract AWS timestamps (datetime objects from boto3)
        created_at = record.get("createdAt")
        updated_at = record.get("lastUpdatedAt")

        # Extract record ARN and status for CUSTOM card display
        record_arn = record.get("recordArn", "")
        record_status = record.get("status", "")

        return {
            "source": AGENTCORE_SOURCE,
            "name": record_name,
            "description": description or "Custom protocol record",
            "url": agent_url,
            "path": agent_path,
            "version": version,
            "protocol_version": "1.0",
            "capabilities": custom_content.get("capabilities", {}),
            "skills": [],
            "provider": provider_data,
            "security_schemes": {},
            "default_input_modes": ["text/plain"],
            "default_output_modes": ["text/plain"],
            "tags": tags,
            "is_enabled": True,
            "is_read_only": True,
            "attribution_label": AGENTCORE_ATTRIBUTION,
            "supported_protocol": "other",
            "metadata": {
                "agentcore_registry_id": registry_id,
                "agentcore_record_id": record_id,
                "descriptor_type": "CUSTOM",
                "custom_content": custom_content,
                "original_url": original_url,
                "record_arn": record_arn,
                "record_status": record_status,
                "created_at": created_at.isoformat() if created_at else None,
                "updated_at": updated_at.isoformat() if updated_at else None,
            },
            "cached_at": datetime.now(UTC).isoformat(),
        }

    def _transform_skills_record(
        self,
        record: dict[str, Any],
        descriptors: dict[str, Any],
        registry_id: str,
    ) -> dict[str, Any]:
        """Transform AGENT_SKILLS descriptor record to skill registration data.

        Uses a self-referencing URL for skill_md_url and stores inline
        markdown content in skill_md_content for DB storage.

        Args:
            record: Full AgentCore record
            descriptors: Descriptors section
            registry_id: Source registry ID

        Returns:
            Skill data dict for SkillService registration
        """
        record_name = record.get("name", "")
        record_id = record.get("recordId", "")
        description = record.get("description", "")
        version = record.get("recordVersion", "1.0.0")

        # Sanitize name for SkillCard: lowercase alphanumeric and hyphens only
        sanitized_name = record_name.replace("_", "-").replace(" ", "-").lower().strip("-")

        # Parse skill descriptors
        skills_desc = descriptors.get("agentSkills", {})
        skill_md_content = skills_desc.get("skillMd", {}).get("inlineContent", "")
        skill_def_content = _safe_parse_json(
            skills_desc.get("skillDefinition", {}).get("inlineContent", "{}"),
            context=f"AGENT_SKILLS definition for {record_name}",
        )

        path_segment = _sanitize_path_segment(record_name)
        skill_path = f"/skills/agentcore-{path_segment}"

        # Build self-referencing URL for skill_md_url
        from registry.core.config import settings

        skill_md_url = f"{settings.registry_url}/api/skills/agentcore-{path_segment}/content"

        # Extract fields from skill definition
        if not description:
            description = skill_def_content.get("description", "")

        target_agents = skill_def_content.get("targetAgents", [])
        allowed_tools = skill_def_content.get("allowedTools", [])

        tags = [
            "agentcore",
            "bedrock",
            "federated",
            "skill",
            f"registry-{registry_id[:12]}",
        ]

        # Extract AWS timestamps (datetime objects from boto3)
        created_at = record.get("createdAt")
        updated_at = record.get("lastUpdatedAt")

        return {
            "source": AGENTCORE_SOURCE,
            "name": sanitized_name,
            "description": description,
            "skill_md_url": skill_md_url,
            "skill_md_content": skill_md_content,
            "path": skill_path,
            "version": version,
            "tags": tags,
            "target_agents": target_agents,
            "allowed_tools": allowed_tools,
            "is_enabled": True,
            "is_read_only": True,
            "attribution_label": AGENTCORE_ATTRIBUTION,
            "registry_name": AGENTCORE_SOURCE,
            "metadata": {
                "agentcore_registry_id": registry_id,
                "agentcore_record_id": record_id,
                "descriptor_type": "AGENT_SKILLS",
                "skill_definition": skill_def_content,
                "created_at": created_at.isoformat() if created_at else None,
                "updated_at": updated_at.isoformat() if updated_at else None,
            },
            "cached_at": datetime.now(UTC).isoformat(),
        }

    # BaseFederationClient interface methods (for compatibility)

    def fetch_server(
        self,
        server_name: str,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Fetch a single server record by name.

        Not the primary usage pattern -- prefer fetch_all_records().

        Args:
            server_name: Record name to search for

        Returns:
            Server data dict or None
        """
        registry_id = kwargs.get("registry_id", "")
        if not registry_id:
            logger.error("registry_id required for AgentCore fetch_server")
            return None

        records = self.list_registry_records(
            registry_id=registry_id,
            descriptor_type="MCP",
        )

        for rec in records:
            if rec.get("name") == server_name:
                full_record = self.get_registry_record(registry_id, rec["recordId"])
                if full_record:
                    return self._transform_record(full_record, registry_id)

        return None

    def fetch_all_servers(
        self,
        server_names: list[str],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Fetch multiple servers by name.

        Args:
            server_names: List of record names to fetch

        Returns:
            List of server data dicts
        """
        registry_id = kwargs.get("registry_id", "")
        if not registry_id:
            logger.error("registry_id required for AgentCore fetch_all_servers")
            return []

        records = self.list_registry_records(
            registry_id=registry_id,
            descriptor_type="MCP",
        )

        servers: list[dict[str, Any]] = []
        name_set = set(server_names)
        for rec in records:
            if rec.get("name") in name_set:
                full_record = self.get_registry_record(registry_id, rec["recordId"])
                if full_record:
                    transformed = self._transform_record(full_record, registry_id)
                    if transformed:
                        servers.append(transformed)

        return servers
