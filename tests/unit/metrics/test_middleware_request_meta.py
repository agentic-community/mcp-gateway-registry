"""Tests that tool metrics read the client of an MCP 2026-07-28 request from _meta.

A 2026-07-28 client sends no initialize, so the protocol version and client
name exist only in params._meta, on every request.
"""

import json
from unittest.mock import MagicMock

import pytest

from auth_server.metrics_middleware import AuthMetricsMiddleware

CLIENT: dict = {"name": "claude-code", "version": "3.1.0"}


def _request(
    body: dict,
) -> MagicMock:
    request = MagicMock()
    request.headers = {"X-Body": json.dumps(body)}
    return request


async def _extract(
    body: dict,
) -> dict:
    middleware = AuthMetricsMiddleware(app=lambda *a, **k: None)
    return await middleware.extract_tool_and_method_info(_request(body))


@pytest.mark.asyncio
async def test_modern_tools_call_reads_client_and_version_from_meta():
    info = await _extract(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "search",
                "_meta": {
                    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                    "io.modelcontextprotocol/clientInfo": CLIENT,
                },
            },
        }
    )

    assert info["tool_name"] == "search"
    assert info["protocol_version"] == "2026-07-28"
    assert info["client_info"] == CLIENT


@pytest.mark.asyncio
async def test_legacy_initialize_is_read_as_before():
    info = await _extract(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-11-25", "clientInfo": CLIENT},
        }
    )

    assert info["protocol_version"] == "2025-11-25"
    assert info["client_info"] == CLIENT


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "params",
    [
        {"name": "search"},
        {"name": "search", "_meta": "not a dict"},
        {"name": "search", "_meta": {"io.modelcontextprotocol/clientInfo": "not a dict"}},
    ],
)
async def test_legacy_or_malformed_meta_leaves_client_unknown(params):
    """The session fallback to the stored initialize client still applies."""
    info = await _extract({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params})

    assert info["client_info"] == {}
    assert info["protocol_version"] is None
