"""
Unit tests for AgentCoreFederationClient.

Tests boto3 API interactions (mocked), descriptor type transformations,
parallel fetching, sync timeout, and health indicator.
"""

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from registry.services.federation.agentcore_client import (
    AGENTCORE_ATTRIBUTION,
    AGENTCORE_SOURCE,
    AgentCoreFederationClient,
    _safe_parse_json,
    _sanitize_path_segment,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_boto3():
    """Patch boto3.client so no real AWS calls are made."""
    with patch("registry.services.federation.agentcore_client.boto3") as mock:
        mock_client = MagicMock()
        mock.client.return_value = mock_client
        yield mock_client


@pytest.fixture
def client(mock_boto3):
    """Return an AgentCoreFederationClient with a mocked boto3 backend."""
    return AgentCoreFederationClient(aws_region="us-east-1")


# ---------------------------------------------------------------------------
# Sample AgentCore API responses
# ---------------------------------------------------------------------------


def _mcp_record(
    name: str = "my-mcp-server",
    record_id: str = "rec-001",
    registry_id: str = "reg-abc123",
) -> dict:
    """Build a sample MCP descriptor record."""
    server_content = json.dumps({
        "title": "My MCP Server",
        "description": "A test MCP server",
        "remotes": [{"type": "streamable-http", "url": "https://example.com/mcp"}],
    })
    tools_content = json.dumps({
        "tools": [{"name": "tool1"}, {"name": "tool2"}],
    })
    return {
        "recordId": record_id,
        "name": name,
        "description": "Test MCP server record",
        "descriptorType": "MCP",
        "recordVersion": "1.0.0",
        "descriptors": {
            "mcp": {
                "server": {"inlineContent": server_content},
                "tools": {"inlineContent": tools_content},
            }
        },
    }


def _a2a_record(
    name: str = "my-a2a-agent",
    record_id: str = "rec-002",
) -> dict:
    """Build a sample A2A descriptor record."""
    agent_card = json.dumps({
        "name": "My A2A Agent",
        "description": "An A2A agent",
        "url": "https://agent.example.com",
        "version": "2.0.0",
        "protocolVersion": "1.0",
        "capabilities": {"streaming": True},
        "skills": [{"name": "chat"}],
    })
    return {
        "recordId": record_id,
        "name": name,
        "description": "Test A2A agent",
        "descriptorType": "A2A",
        "recordVersion": "2.0.0",
        "descriptors": {
            "a2a": {
                "agentCard": {"inlineContent": agent_card},
            }
        },
    }


def _custom_record(
    name: str = "my-custom-thing",
    record_id: str = "rec-003",
) -> dict:
    """Build a sample CUSTOM descriptor record."""
    custom_content = json.dumps({
        "url": "https://original.example.com/api",
        "capabilities": {"invoke": True},
        "provider": {"organization": "TestCorp"},
    })
    return {
        "recordId": record_id,
        "name": name,
        "description": "A custom descriptor",
        "descriptorType": "CUSTOM",
        "recordVersion": "1.0.0",
        "descriptors": {
            "custom": {
                "inlineContent": custom_content,
            }
        },
    }


def _skills_record(
    name: str = "my-skill",
    record_id: str = "rec-004",
) -> dict:
    """Build a sample AGENT_SKILLS descriptor record."""
    skill_md = "# My Skill\n\nDo something useful."
    skill_def = json.dumps({
        "description": "A useful skill",
        "targetAgents": ["claude-code"],
        "allowedTools": ["Read", "Write"],
    })
    return {
        "recordId": record_id,
        "name": name,
        "description": "Test skill",
        "descriptorType": "AGENT_SKILLS",
        "recordVersion": "1.0.0",
        "descriptors": {
            "agentSkills": {
                "skillMd": {"inlineContent": skill_md},
                "skillDefinition": {"inlineContent": skill_def},
            }
        },
    }


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestSafeParseJson:
    """Tests for _safe_parse_json utility."""

    def test_valid_json(self):
        result = _safe_parse_json('{"key": "value"}', "test")
        assert result == {"key": "value"}

    def test_invalid_json_returns_empty_dict(self):
        result = _safe_parse_json("not json at all", "test")
        assert result == {}

    def test_none_input_returns_empty_dict(self):
        result = _safe_parse_json(None, "test")
        assert result == {}

    def test_empty_string_returns_empty_dict(self):
        result = _safe_parse_json("", "test")
        assert result == {}


class TestSanitizePathSegment:
    """Tests for _sanitize_path_segment utility."""

    def test_simple_name(self):
        assert _sanitize_path_segment("my-server") == "my-server"

    def test_slashes_replaced(self):
        assert _sanitize_path_segment("org/server") == "org-server"

    def test_spaces_replaced(self):
        assert _sanitize_path_segment("my cool server") == "my-cool-server"

    def test_uppercase_lowered(self):
        assert _sanitize_path_segment("MyServer") == "myserver"

    def test_leading_trailing_hyphens_stripped(self):
        assert _sanitize_path_segment("-server-") == "server"


# ---------------------------------------------------------------------------
# Client API tests (boto3 mocked)
# ---------------------------------------------------------------------------


class TestListRegistries:
    """Tests for list_registries."""

    def test_success(self, client, mock_boto3):
        mock_boto3.list_registries.return_value = {
            "registries": [{"name": "reg-1", "registryId": "id-1", "status": "READY"}],
        }

        result = client.list_registries()
        assert len(result) == 1
        assert result[0]["name"] == "reg-1"

    def test_pagination(self, client, mock_boto3):
        mock_boto3.list_registries.side_effect = [
            {
                "registries": [{"name": "reg-1"}],
                "nextToken": "page2",
            },
            {
                "registries": [{"name": "reg-2"}],
            },
        ]

        result = client.list_registries()
        assert len(result) == 2

    def test_error_returns_empty(self, client, mock_boto3):
        mock_boto3.list_registries.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "forbidden"}},
            "ListRegistries",
        )

        result = client.list_registries()
        assert result == []


class TestListRegistryRecords:
    """Tests for list_registry_records."""

    def test_success_with_filters(self, client, mock_boto3):
        mock_boto3.list_registry_records.return_value = {
            "registryRecords": [
                {"recordId": "rec-1", "descriptorType": "MCP", "name": "s1"},
            ],
        }

        result = client.list_registry_records(
            registry_id="reg-123",
            descriptor_type="MCP",
            status="APPROVED",
        )
        assert len(result) == 1

        call_kwargs = mock_boto3.list_registry_records.call_args[1]
        assert call_kwargs["registryId"] == "reg-123"
        assert call_kwargs["descriptorType"] == "MCP"
        assert call_kwargs["status"] == "APPROVED"

    def test_pagination(self, client, mock_boto3):
        mock_boto3.list_registry_records.side_effect = [
            {"registryRecords": [{"recordId": "r1"}], "nextToken": "tok"},
            {"registryRecords": [{"recordId": "r2"}]},
        ]

        result = client.list_registry_records(registry_id="reg-123")
        assert len(result) == 2

    def test_client_error_returns_empty(self, client, mock_boto3):
        mock_boto3.list_registry_records.side_effect = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "bad"}},
            "ListRegistryRecords",
        )

        result = client.list_registry_records(registry_id="reg-123")
        assert result == []


class TestGetRegistryRecord:
    """Tests for get_registry_record."""

    def test_success(self, client, mock_boto3):
        mock_boto3.get_registry_record.return_value = {
            "recordId": "rec-1",
            "name": "test",
            "ResponseMetadata": {"RequestId": "xxx"},
        }

        result = client.get_registry_record("reg-123", "rec-1")
        assert result is not None
        assert result["recordId"] == "rec-1"
        assert "ResponseMetadata" not in result

    def test_not_found_returns_none(self, client, mock_boto3):
        mock_boto3.get_registry_record.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "nope"}},
            "GetRegistryRecord",
        )

        result = client.get_registry_record("reg-123", "rec-1")
        assert result is None

    def test_other_error_returns_none(self, client, mock_boto3):
        mock_boto3.get_registry_record.side_effect = ClientError(
            {"Error": {"Code": "InternalServerException", "Message": "oops"}},
            "GetRegistryRecord",
        )

        result = client.get_registry_record("reg-123", "rec-1")
        assert result is None


# ---------------------------------------------------------------------------
# Transformation tests
# ---------------------------------------------------------------------------


class TestTransformMcpRecord:
    """Tests for MCP descriptor -> server dict transformation."""

    def test_basic_transform(self, client):
        record = _mcp_record()
        result = client._transform_record(record, "reg-abc123")

        assert result is not None
        assert result["source"] == AGENTCORE_SOURCE
        assert result["server_name"] == "my-mcp-server"
        assert result["description"] == "Test MCP server record"
        assert result["proxy_pass_url"] == "https://example.com/mcp"
        assert result["transport_type"] == "streamable-http"
        assert result["is_read_only"] is True
        assert result["attribution_label"] == AGENTCORE_ATTRIBUTION
        assert result["num_tools"] == 2
        assert result["path"] == "/agentcore-my-mcp-server"
        assert "agentcore" in result["tags"]
        assert "mcp" in result["tags"]

    def test_fallback_to_sync_url(self, client):
        """When no remotes/packages, fall back to synchronizationConfiguration URL."""
        record = _mcp_record()
        record["descriptors"]["mcp"]["server"]["inlineContent"] = json.dumps({})
        record["synchronizationConfiguration"] = {
            "fromUrl": {"url": "https://sync.example.com/mcp"}
        }

        result = client._transform_record(record, "reg-abc")
        assert result["proxy_pass_url"] == "https://sync.example.com/mcp"


class TestTransformA2aRecord:
    """Tests for A2A descriptor -> agent dict transformation."""

    def test_basic_transform(self, client):
        record = _a2a_record()
        result = client._transform_record(record, "reg-abc123")

        assert result is not None
        assert result["source"] == AGENTCORE_SOURCE
        assert result["name"] == "My A2A Agent"
        assert result["url"] == "https://agent.example.com"
        assert result["version"] == "2.0.0"
        assert result["supported_protocol"] == "a2a"
        assert result["path"] == "/agents/agentcore-my-a2a-agent"
        assert "a2a" in result["tags"]
        assert result["is_read_only"] is True

    def test_capabilities_preserved(self, client):
        record = _a2a_record()
        result = client._transform_record(record, "reg-abc123")
        assert result["capabilities"] == {"streaming": True}


class TestTransformCustomRecord:
    """Tests for CUSTOM descriptor -> agent dict transformation."""

    @patch("registry.core.config.settings")
    def test_basic_transform(self, mock_settings, client):
        mock_settings.registry_url = "https://my-registry.com"
        record = _custom_record()
        result = client._transform_record(record, "reg-abc123")

        assert result is not None
        assert result["source"] == AGENTCORE_SOURCE
        assert result["name"] == "my-custom-thing"
        assert result["supported_protocol"] == "other"
        assert result["path"] == "/agents/agentcore-custom-my-custom-thing"
        # Self-referencing URL
        assert result["url"] == "https://my-registry.com/api/agents/agentcore-custom-my-custom-thing"
        # Original URL preserved in metadata
        assert result["metadata"]["original_url"] == "https://original.example.com/api"

    @patch("registry.core.config.settings")
    def test_no_original_url(self, mock_settings, client):
        mock_settings.registry_url = "http://localhost:8000"
        record = _custom_record()
        record["descriptors"]["custom"]["inlineContent"] = json.dumps({"foo": "bar"})
        result = client._transform_record(record, "reg-abc123")
        assert result["metadata"]["original_url"] is None


class TestTransformSkillsRecord:
    """Tests for AGENT_SKILLS descriptor -> skill dict transformation."""

    @patch("registry.core.config.settings")
    def test_basic_transform(self, mock_settings, client):
        mock_settings.registry_url = "https://my-registry.com"
        record = _skills_record()
        result = client._transform_record(record, "reg-abc123")

        assert result is not None
        assert result["source"] == AGENTCORE_SOURCE
        assert result["name"] == "my-skill"
        assert result["path"] == "/skills/agentcore-my-skill"
        assert result["skill_md_content"] == "# My Skill\n\nDo something useful."
        assert result["skill_md_url"] == "https://my-registry.com/api/skills/agentcore-my-skill/content"
        assert result["target_agents"] == ["claude-code"]
        assert result["registry_name"] == AGENTCORE_SOURCE
        assert result["is_read_only"] is True

    @patch("registry.core.config.settings")
    def test_empty_skill_md_content(self, mock_settings, client):
        mock_settings.registry_url = "http://localhost:8000"
        record = _skills_record()
        record["descriptors"]["agentSkills"]["skillMd"]["inlineContent"] = ""
        result = client._transform_record(record, "reg-abc123")
        assert result["skill_md_content"] == ""


class TestTransformUnknownDescriptor:
    """Tests for unknown descriptor type handling."""

    def test_unknown_returns_none(self, client):
        record = {"descriptorType": "FUTURE_TYPE", "descriptors": {}}
        result = client._transform_record(record, "reg-abc")
        assert result is None


# ---------------------------------------------------------------------------
# fetch_all_records tests
# ---------------------------------------------------------------------------


class TestFetchAllRecords:
    """Tests for fetch_all_records (parallel fetch, grouping, timeout)."""

    def test_grouped_by_type(self, client, mock_boto3):
        """Records should be routed to servers/agents/skills buckets."""
        from registry.schemas.federation_schema import AgentCoreRegistryConfig

        # Mock list_registry_records to return 3 records
        mock_boto3.list_registry_records.return_value = {
            "registryRecords": [
                {"recordId": "r1", "descriptorType": "MCP", "name": "s1"},
                {"recordId": "r2", "descriptorType": "A2A", "name": "a1"},
                {"recordId": "r3", "descriptorType": "AGENT_SKILLS", "name": "sk1"},
            ],
        }

        # Mock get_registry_record to return full records
        mock_boto3.get_registry_record.side_effect = [
            {**_mcp_record(name="s1", record_id="r1"), "ResponseMetadata": {}},
            {**_a2a_record(name="a1", record_id="r2"), "ResponseMetadata": {}},
            {**_skills_record(name="sk1", record_id="r3"), "ResponseMetadata": {}},
        ]

        config = AgentCoreRegistryConfig(registry_id="reg-123")
        with patch("registry.core.config.settings") as mock_s:
            mock_s.registry_url = "http://localhost:8000"
            result = client.fetch_all_records([config])

        assert len(result["servers"]) == 1
        assert len(result["agents"]) == 1
        assert len(result["skills"]) == 1

    def test_health_updated_after_sync(self, client, mock_boto3):
        """Health indicator should be updated after successful sync."""
        from registry.schemas.federation_schema import AgentCoreRegistryConfig

        mock_boto3.list_registry_records.return_value = {"registryRecords": []}
        config = AgentCoreRegistryConfig(registry_id="reg-123")

        assert client._last_sync_success is False

        client.fetch_all_records([config])

        assert client._last_sync_success is True
        assert client._last_sync_time is not None
        assert client._last_sync_record_count == 0

    def test_descriptor_type_filter(self, client, mock_boto3):
        """Records with descriptor types not in config should be skipped."""
        from registry.schemas.federation_schema import AgentCoreRegistryConfig

        mock_boto3.list_registry_records.return_value = {
            "registryRecords": [
                {"recordId": "r1", "descriptorType": "MCP", "name": "s1"},
                {"recordId": "r2", "descriptorType": "CUSTOM", "name": "c1"},
            ],
        }

        mock_boto3.get_registry_record.return_value = {
            **_mcp_record(name="s1", record_id="r1"),
            "ResponseMetadata": {},
        }

        # Only sync MCP, not CUSTOM
        config = AgentCoreRegistryConfig(
            registry_id="reg-123",
            descriptor_types=["MCP"],
        )
        result = client.fetch_all_records([config])

        assert len(result["servers"]) == 1
        assert len(result["agents"]) == 0
        # get_registry_record should only be called once (for MCP)
        assert mock_boto3.get_registry_record.call_count == 1

    def test_empty_registries(self, client, mock_boto3):
        """No configs means no API calls."""
        result = client.fetch_all_records([])
        assert result == {"servers": [], "agents": [], "skills": []}
        mock_boto3.list_registry_records.assert_not_called()


# ---------------------------------------------------------------------------
# Health indicator tests
# ---------------------------------------------------------------------------


class TestHealthStatus:
    """Tests for get_health_status."""

    def test_initial_state(self, client):
        health = client.get_health_status()
        assert health["source"] == AGENTCORE_SOURCE
        assert health["healthy"] is False
        assert health["last_sync_time"] is None
        assert health["last_sync_record_count"] == 0

    def test_after_sync(self, client, mock_boto3):
        from registry.schemas.federation_schema import AgentCoreRegistryConfig

        mock_boto3.list_registry_records.return_value = {"registryRecords": []}
        client.fetch_all_records([AgentCoreRegistryConfig(registry_id="reg-1")])

        health = client.get_health_status()
        assert health["healthy"] is True
        assert health["last_sync_time"] is not None
        assert health["aws_region"] == "us-east-1"


# ---------------------------------------------------------------------------
# Cross-account client tests
# ---------------------------------------------------------------------------


class TestGetClientForRegistry:
    """Tests for _get_client_for_registry (cross-account/cross-region)."""

    def test_same_account_same_region_returns_default(self, client):
        """When no role or custom region, return the default client."""
        from registry.schemas.federation_schema import AgentCoreRegistryConfig

        config = AgentCoreRegistryConfig(registry_id="reg-123")
        result = client._get_client_for_registry(config)
        assert result is client._client

    def test_same_region_explicit_returns_default(self, client):
        """Explicitly setting aws_region to same as client still returns default."""
        from registry.schemas.federation_schema import AgentCoreRegistryConfig

        config = AgentCoreRegistryConfig(
            registry_id="reg-123",
            aws_region="us-east-1",
        )
        result = client._get_client_for_registry(config)
        assert result is client._client

    def test_different_region_creates_new_client(self, client):
        """Different aws_region should create a region-specific client."""
        from registry.schemas.federation_schema import AgentCoreRegistryConfig

        mock_regional_client = MagicMock()
        with patch("registry.services.federation.agentcore_client.boto3") as mock_b3:
            mock_b3.client.return_value = mock_regional_client

            config = AgentCoreRegistryConfig(
                registry_id="reg-eu",
                aws_region="eu-west-1",
            )
            result = client._get_client_for_registry(config)

        assert result is mock_regional_client

    def test_cross_account_calls_sts_assume_role(self, client):
        """When assume_role_arn is set, STS AssumeRole should be called."""
        from registry.schemas.federation_schema import AgentCoreRegistryConfig

        mock_sts = MagicMock()
        mock_sts.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "AKIA_TEMP",
                "SecretAccessKey": "secret_temp",
                "SessionToken": "token_temp",
            }
        }

        mock_cross_client = MagicMock()

        with patch("registry.services.federation.agentcore_client.boto3") as mock_b3:
            mock_b3.client.side_effect = lambda service, **kwargs: (
                mock_sts if service == "sts" else mock_cross_client
            )

            config = AgentCoreRegistryConfig(
                registry_id="reg-cross",
                aws_account_id="123456789012",
                assume_role_arn="arn:aws:iam::123456789012:role/ReadRole",
            )
            result = client._get_client_for_registry(config)

        assert result is mock_cross_client
        mock_sts.assume_role.assert_called_once()
        call_kwargs = mock_sts.assume_role.call_args[1]
        assert call_kwargs["RoleArn"] == "arn:aws:iam::123456789012:role/ReadRole"

    def test_cross_account_with_custom_region(self, client):
        """Role assumption should use the per-registry region."""
        from registry.schemas.federation_schema import AgentCoreRegistryConfig

        mock_sts = MagicMock()
        mock_sts.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "AK",
                "SecretAccessKey": "SK",
                "SessionToken": "ST",
            }
        }
        mock_cross_client = MagicMock()

        with patch("registry.services.federation.agentcore_client.boto3") as mock_b3:
            mock_b3.client.side_effect = lambda service, **kwargs: (
                mock_sts if service == "sts" else mock_cross_client
            )

            config = AgentCoreRegistryConfig(
                registry_id="reg-eu-cross",
                aws_account_id="999888777666",
                aws_region="eu-west-1",
                assume_role_arn="arn:aws:iam::999888777666:role/EuRole",
            )
            result = client._get_client_for_registry(config)

        assert result is mock_cross_client
        # STS client should be created in the registry's region
        sts_call = mock_b3.client.call_args_list[0]
        assert sts_call[0][0] == "sts"
        assert sts_call[1]["region_name"] == "eu-west-1"

    def test_client_is_cached_by_region_and_role(self, client):
        """Second call with same region+role should return cached client."""
        from registry.schemas.federation_schema import AgentCoreRegistryConfig

        mock_cached = MagicMock()
        cache_key = "eu-west-1:arn:aws:iam::111111111111:role/CachedRole"
        client._registry_clients[cache_key] = mock_cached

        config = AgentCoreRegistryConfig(
            registry_id="reg-cached",
            aws_region="eu-west-1",
            assume_role_arn="arn:aws:iam::111111111111:role/CachedRole",
        )
        result = client._get_client_for_registry(config)
        assert result is mock_cached


# ---------------------------------------------------------------------------
# Compatibility interface tests
# ---------------------------------------------------------------------------


class TestFetchServerInterface:
    """Tests for BaseFederationClient interface methods."""

    def test_fetch_server_no_registry_id(self, client):
        result = client.fetch_server("test-server")
        assert result is None

    def test_fetch_all_servers_no_registry_id(self, client):
        result = client.fetch_all_servers(["s1", "s2"])
        assert result == []
