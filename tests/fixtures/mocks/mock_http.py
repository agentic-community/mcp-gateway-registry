"""
Mock HTTP client implementations for testing.

This module provides mock implementations of HTTP clients to avoid
making real network requests during tests.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MockResponse:
    """
    Mock HTTP response object.

    Mimics the interface of httpx.Response.
    """

    def __init__(
        self,
        status_code: int = 200,
        json_data: dict[str, Any] | None = None,
        text: str = "",
        headers: dict[str, str] | None = None,
    ):
        """
        Initialize mock response.

        Args:
            status_code: HTTP status code
            json_data: JSON response data
            text: Response text
            headers: Response headers
        """
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text or ""
        self.headers = headers or {}
        self.content = text.encode() if text else b""

    def json(self) -> dict[str, Any]:
        """Get JSON response data."""
        return self._json_data

    def raise_for_status(self) -> None:
        """Raise exception for error status codes."""
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def __repr__(self) -> str:
        return f"MockResponse(status={self.status_code})"


class MockAsyncClient:
    """
    Mock async HTTP client.

    Mimics the interface of httpx.AsyncClient.
    """

    def __init__(self, responses: dict[str, MockResponse] | None = None):
        """
        Initialize mock async client.

        Args:
            responses: Dictionary mapping URLs to mock responses
        """
        self.responses = responses or {}
        self.request_history: list[dict[str, Any]] = []

    async def get(self, url: str, **kwargs: Any) -> MockResponse:
        """
        Mock GET request.

        Args:
            url: Request URL
            **kwargs: Additional request arguments

        Returns:
            Mock response
        """
        self.request_history.append({"method": "GET", "url": url, "kwargs": kwargs})

        if url in self.responses:
            return self.responses[url]

        return MockResponse(status_code=404, json_data={"error": "Not found"})

    async def post(self, url: str, **kwargs: Any) -> MockResponse:
        """
        Mock POST request.

        Args:
            url: Request URL
            **kwargs: Additional request arguments

        Returns:
            Mock response
        """
        self.request_history.append({"method": "POST", "url": url, "kwargs": kwargs})

        if url in self.responses:
            return self.responses[url]

        return MockResponse(status_code=200, json_data={"success": True})

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass


def create_mock_httpx_client(responses: dict[str, MockResponse] | None = None) -> MockAsyncClient:
    """
    Create a mock httpx async client.

    Args:
        responses: Dictionary mapping URLs to mock responses

    Returns:
        Mock async client
    """
    return MockAsyncClient(responses)


def create_mock_mcp_server_response(
    tools: list[dict[str, Any]] | None = None,
    prompts: list[dict[str, Any]] | None = None,
    resources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Create a mock MCP server response.

    Args:
        tools: List of tool definitions
        prompts: List of prompt definitions
        resources: List of resource definitions

    Returns:
        Mock server response dictionary
    """
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"tools": tools or [], "prompts": prompts or [], "resources": resources or []},
    }


def create_mock_tool_definition(
    name: str, description: str = "Test tool", input_schema: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Create a mock MCP tool definition.

    Args:
        name: Tool name
        description: Tool description
        input_schema: Tool input schema

    Returns:
        Mock tool definition
    """
    return {
        "name": name,
        "description": description,
        "inputSchema": input_schema or {"type": "object", "properties": {}},
    }
