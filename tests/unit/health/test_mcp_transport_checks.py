"""
Unit tests for MCP transport-aware health checks.
"""

from __future__ import annotations

from unittest.mock import (
    AsyncMock,
)

import httpx
import pytest

from registry.constants import (
    HealthStatus,
)
from registry.health.mcp_transport_checks import (
    McpTransportChecker,
    _redact_headers_for_logging,
)


def _make_response(
    method: str,
    url: str,
    status_code: int,
    *,
    headers: dict[str, str] | None = None,
    content: bytes | None = None,
) -> httpx.Response:
    request = httpx.Request(method, url)
    return httpx.Response(
        status_code=status_code,
        headers=headers or {},
        content=content or b"",
        request=request,
    )


@pytest.mark.unit
@pytest.mark.health
class TestMcpTransportChecker:
    def test_redact_headers_for_logging_redacts_sensitive_values(
        self,
    ) -> None:
        redacted = _redact_headers_for_logging(
            {
                "Authorization": "Bearer secret-token",
                "cookie": "session=supersecret",
                "Content-Type": "application/json",
            }
        )

        assert redacted["Authorization"] == "***REDACTED***"
        assert redacted["cookie"] == "***REDACTED***"
        assert redacted["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_check_missing_proxy_url_is_unhealthy(
        self,
    ) -> None:
        checker = McpTransportChecker()
        client = AsyncMock(spec=httpx.AsyncClient)

        is_healthy, status = await checker._check_server_endpoint_transport_aware(
            client,
            "",
            {},
        )

        assert is_healthy is False
        assert status == HealthStatus.UNHEALTHY_MISSING_PROXY_URL

    @pytest.mark.asyncio
    async def test_check_stdio_transport_is_skipped(
        self,
    ) -> None:
        checker = McpTransportChecker()
        client = AsyncMock(spec=httpx.AsyncClient)

        is_healthy, status = await checker._check_server_endpoint_transport_aware(
            client,
            "http://example.test",
            {"supported_transports": ["stdio"]},
        )

        assert is_healthy is True
        assert status == HealthStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_check_sse_url_path_considered_healthy_on_200(
        self,
    ) -> None:
        checker = McpTransportChecker()
        client = AsyncMock(spec=httpx.AsyncClient)
        proxy_pass_url = "http://example.test/sse"

        client.get = AsyncMock(
            return_value=_make_response(
                "GET",
                proxy_pass_url,
                200,
            )
        )

        is_healthy, status = await checker._check_server_endpoint_transport_aware(
            client,
            proxy_pass_url,
            {"supported_transports": ["sse"]},
        )

        assert is_healthy is True
        assert status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_check_streamable_http_initialize_then_ping_is_healthy(
        self,
    ) -> None:
        checker = McpTransportChecker()
        client = AsyncMock(spec=httpx.AsyncClient)
        proxy_pass_url = "http://example.test"
        endpoint = "http://example.test/mcp"

        init_response = _make_response(
            "POST",
            endpoint,
            200,
            headers={"Mcp-Session-Id": "session-123"},
        )
        ping_response = _make_response(
            "POST",
            endpoint,
            200,
        )

        client.post = AsyncMock(side_effect=[init_response, ping_response])

        is_healthy, status = await checker._check_server_endpoint_transport_aware(
            client,
            proxy_pass_url,
            {"supported_transports": ["streamable-http"]},
        )

        assert is_healthy is True
        assert status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_check_streamable_http_auth_failure_falls_back_to_ping_without_auth(
        self,
    ) -> None:
        checker = McpTransportChecker()
        client = AsyncMock(spec=httpx.AsyncClient)
        proxy_pass_url = "http://example.test"
        endpoint = "http://example.test/mcp"

        init_response = _make_response(
            "POST",
            endpoint,
            200,
            headers={"Mcp-Session-Id": "session-123"},
        )
        ping_response = _make_response(
            "POST",
            endpoint,
            401,
        )

        client.post = AsyncMock(side_effect=[init_response, ping_response])
        checker._try_ping_without_auth = AsyncMock(return_value=True)

        is_healthy, status = await checker._check_server_endpoint_transport_aware(
            client,
            proxy_pass_url,
            {"supported_transports": ["streamable-http"]},
        )

        assert is_healthy is True
        assert status == HealthStatus.HEALTHY
