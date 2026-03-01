"""
MCP Logger for protocol-level audit logging.

This module provides the MCPLogger class that handles logging of
MCP (Model Context Protocol) server access events, including
JSON-RPC request parsing and tool/resource invocation tracking.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional, Union

from .models import (
    Identity,
    MCPRequest,
    MCPResponse,
    MCPServer,
    MCPServerAccessRecord,
    Request,
)
from .service import AuditLogger

logger = logging.getLogger(__name__)


class MCPLogger:
    """
    MCP protocol-level audit logger.

    Handles logging of MCP server access events by parsing JSON-RPC
    request bodies and creating structured audit records. Delegates
    actual file/MongoDB writing to the AuditLogger service.

    Attributes:
        audit_logger: The underlying AuditLogger service for writing events
    """

    def __init__(self, audit_logger: AuditLogger):
        """
        Initialize the MCPLogger.

        Args:
            audit_logger: The AuditLogger service to use for writing events
        """
        self.audit_logger = audit_logger

    def parse_jsonrpc_body(self, body: Union[bytes, str]) -> dict:
        """
        Parse JSON-RPC request body to extract method and params.

        Extracts the JSON-RPC method name and relevant parameters
        based on the method type:
        - For 'tools/call': extracts tool_name from params.name
        - For 'resources/read': extracts resource_uri from params.uri

        Args:
            body: The JSON-RPC request body as bytes or string

        Returns:
            Dictionary containing:
            - method: The JSON-RPC method name (or 'unknown' if parsing fails)
            - jsonrpc_id: The JSON-RPC request ID as string
            - tool_name: (optional) The tool name for tools/call requests
            - resource_uri: (optional) The resource URI for resources/read requests
        """
        try:
            # Handle both bytes and string input
            if isinstance(body, bytes):
                body_str = body.decode("utf-8")
            else:
                body_str = body

            # Handle empty body
            if not body_str or not body_str.strip():
                return {"method": "unknown", "jsonrpc_id": ""}

            data = json.loads(body_str)

            # Handle non-dict responses (e.g., arrays for batch requests)
            if not isinstance(data, dict):
                return {"method": "unknown", "jsonrpc_id": ""}

            method = data.get("method", "")
            params = data.get("params", {})

            # Ensure params is a dict
            if not isinstance(params, dict):
                params = {}

            result = {
                "method": method if method else "unknown",
                "jsonrpc_id": str(data.get("id", "")),
            }

            # Extract tool_name for tools/call
            if method == "tools/call":
                tool_name = params.get("name")
                if tool_name:
                    result["tool_name"] = str(tool_name)

            # Extract resource_uri for resources/read
            if method == "resources/read":
                resource_uri = params.get("uri")
                if resource_uri:
                    result["resource_uri"] = str(resource_uri)

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON-RPC body: {e}")
            return {"method": "unknown", "jsonrpc_id": ""}
        except Exception as e:
            logger.warning(f"Unexpected error parsing JSON-RPC body: {e}")
            return {"method": "unknown", "jsonrpc_id": ""}

    async def log_mcp_access(
        self,
        request_id: str,
        identity: Identity,
        mcp_server: MCPServer,
        request_body: Union[bytes, str],
        response_status: str,
        duration_ms: float,
        mcp_session_id: Optional[str] = None,
        transport: str = "streamable-http",
        error_code: Optional[int] = None,
        error_message: Optional[str] = None,
        client_ip: str = "unknown",
        forwarded_for: Optional[str] = None,
        user_agent: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Log an MCP server access event.

        Creates an MCPServerAccessRecord from the provided parameters
        and writes it to the audit log via the AuditLogger service.

        Args:
            request_id: Unique identifier for this request
            identity: Identity of the user making the request
            mcp_server: Information about the target MCP server
            request_body: The JSON-RPC request body (for method/tool extraction)
            response_status: Response status: 'success', 'error', or 'timeout'
            duration_ms: Request duration in milliseconds
            mcp_session_id: Optional MCP session identifier
            transport: Transport protocol (default: 'streamable-http')
            error_code: JSON-RPC error code (if status is 'error')
            error_message: Error message (if status is 'error')
            client_ip: Client IP address
            forwarded_for: X-Forwarded-For header value
            user_agent: User-Agent header value
            correlation_id: Optional correlation ID for tracing
        """
        # Parse the JSON-RPC body to extract method and tool/resource info
        parsed = self.parse_jsonrpc_body(request_body)

        # Build the MCP request model
        mcp_request = MCPRequest(
            method=parsed.get("method", "unknown"),
            tool_name=parsed.get("tool_name"),
            resource_uri=parsed.get("resource_uri"),
            mcp_session_id=mcp_session_id,
            transport=transport,
            jsonrpc_id=parsed.get("jsonrpc_id"),
        )

        # Build the MCP response model
        mcp_response = MCPResponse(
            status=response_status,
            duration_ms=duration_ms,
            error_code=error_code,
            error_message=error_message,
        )

        # Build optional HTTP request info
        request_info = None
        if client_ip != "unknown" or forwarded_for or user_agent:
            request_info = Request(
                method="POST",  # MCP requests are typically POST
                path=mcp_server.path,
                client_ip=client_ip,
                forwarded_for=forwarded_for,
                user_agent=user_agent,
            )

        # Create the complete audit record
        record = MCPServerAccessRecord(
            timestamp=datetime.now(timezone.utc),
            request_id=request_id,
            correlation_id=correlation_id,
            identity=identity,
            mcp_server=mcp_server,
            mcp_request=mcp_request,
            mcp_response=mcp_response,
            request=request_info,
        )

        # Write to audit log
        await self.audit_logger.log_event(record)
