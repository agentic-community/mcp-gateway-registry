"""
Tests for MCP Logger functionality.

Validates: Requirements 9.3, 9.5
"""

import json
import os
import tempfile

import pytest
from hypothesis import given, settings, strategies as st

from registry.audit.mcp_logger import MCPLogger
from registry.audit.models import Identity, MCPServer
from registry.audit.service import AuditLogger


class TestJSONRPCParsing:
    """Property 14: JSON-RPC parsing extracts method and tool name."""

    @given(
        tool_name=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        jsonrpc_id=st.one_of(st.integers(), st.text(min_size=1, max_size=20)),
    )
    @settings(max_examples=50)
    def test_tools_call_extracts_tool_name(self, tool_name: str, jsonrpc_id):
        """For tools/call requests, parse_jsonrpc_body extracts the tool_name."""
        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": tool_name},
                "id": jsonrpc_id,
            }
        )
        result = MCPLogger(None).parse_jsonrpc_body(body)
        assert result["method"] == "tools/call"
        assert result["tool_name"] == tool_name

    @given(
        resource_uri=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
    )
    @settings(max_examples=50)
    def test_resources_read_extracts_uri(self, resource_uri: str):
        """For resources/read requests, parse_jsonrpc_body extracts the resource_uri."""
        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "resources/read",
                "params": {"uri": resource_uri},
                "id": 1,
            }
        )
        result = MCPLogger(None).parse_jsonrpc_body(body)
        assert result["method"] == "resources/read"
        assert result["resource_uri"] == resource_uri

    def test_invalid_json_returns_unknown(self):
        """Invalid JSON returns method='unknown'."""
        result = MCPLogger(None).parse_jsonrpc_body(b"not valid json")
        assert result["method"] == "unknown"
        assert result["jsonrpc_id"] == ""

    def test_empty_body_returns_unknown(self):
        """Empty body returns method='unknown'."""
        result = MCPLogger(None).parse_jsonrpc_body(b"")
        assert result["method"] == "unknown"


class TestLogMCPAccess:
    """Tests for log_mcp_access method."""

    @pytest.mark.asyncio
    async def test_creates_audit_record(self):
        """log_mcp_access creates a complete audit record via MongoDB."""
        from unittest.mock import AsyncMock

        # Create mock repository to capture the audit record
        mock_repository = AsyncMock()
        captured_records = []

        async def capture_insert(record):
            captured_records.append(record)

        mock_repository.insert.side_effect = capture_insert

        # Create AuditLogger with MongoDB enabled
        audit_logger = AuditLogger(
            stream_name="mcp-server-access",
            mongodb_enabled=True,
            audit_repository=mock_repository,
        )
        mcp_logger = MCPLogger(audit_logger)

        identity = Identity(
            username="test-user",
            auth_method="oauth2",
            credential_type="bearer_token",
            credential_hint="abc123xyz789",
        )
        mcp_server = MCPServer(
            name="weather-server",
            path="/mcp/weather",
            proxy_target="http://localhost:8080",
        )

        await mcp_logger.log_mcp_access(
            request_id="req-123",
            identity=identity,
            mcp_server=mcp_server,
            request_body=b'{"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "get_weather"}, "id": 1}',
            response_status="success",
            duration_ms=150.5,
            mcp_session_id="session-456",
        )
        await audit_logger.close()

        # Verify audit record was captured
        assert len(captured_records) == 1

        record = captured_records[0]
        assert record.log_type == "mcp_server_access"
        assert record.mcp_request.method == "tools/call"
        assert record.mcp_request.tool_name == "get_weather"
        assert record.identity.credential_hint == "***xyz789"
