"""tools/list filtering must apply to SSE responses, not just JSON.

`should_filter` required `application/json`, but streamable-http upstreams answer
tools/list with `text/event-stream`. For those servers the tool-level filter never
ran and the unfiltered tool list was forwarded, which has been true since #1026
(2026-05-13). Two things leaked: a tool the caller has no scope for, and (after
#1719) a tool blocked by the per-tool security block.

The block half matters more than it looks. A tool is auto-blocked for
`HIGH:PROMPT INJECTION` when the *description* carries the injection, and
tools/list is what hands descriptions to the model. Listing a tool blocked for
prompt injection therefore delivers the payload even though the call is refused.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from auth_server.server import _filter_sse_tools_list_body, _should_filter_tools_list

pytestmark = [pytest.mark.unit]

SERVER = "airegistry-tools"
SCOPES = ["mcp-servers-unrestricted/execute"]


def _sse(payload: dict) -> str:
    """One SSE message frame, matching what a streamable-http upstream emits."""
    return f"event: message\ndata: {json.dumps(payload)}\n\n"


def _tools_payload(*names: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"tools": [{"name": n, "description": f"desc of {n}"} for n in names]},
    }


class TestFilterSseToolsListBody:
    """The SSE body is rewritten with the filtered tool set."""

    async def test_drops_the_tool_the_filter_rejects(self):
        body = _sse(_tools_payload("list_services", "healthcheck"))

        with patch(
            "auth_server.server.filter_tools_list_response",
            new=AsyncMock(return_value=[{"name": "list_services", "description": "desc"}]),
        ):
            out = await _filter_sse_tools_list_body(body, SERVER, SCOPES)

        assert out is not None
        assert "healthcheck" not in out
        assert "list_services" in out

    async def test_preserves_sse_framing(self):
        """A rewritten body must still parse as SSE, or every client breaks."""
        body = _sse(_tools_payload("a"))

        with patch(
            "auth_server.server.filter_tools_list_response",
            new=AsyncMock(return_value=[{"name": "a"}]),
        ):
            out = await _filter_sse_tools_list_body(body, SERVER, SCOPES)

        assert out.startswith("event: message\n")
        assert "\ndata: " in out
        # The payload still round-trips as JSON-RPC.
        data = [ln for ln in out.split("\n") if ln.startswith("data: ")][0]
        assert json.loads(data[len("data: ") :])["result"]["tools"] == [{"name": "a"}]

    async def test_unparseable_payload_fails_closed(self):
        """None tells the caller to refuse, never to forward what it cannot inspect."""
        body = "event: message\ndata: {not json at all\n\n"

        out = await _filter_sse_tools_list_body(body, SERVER, SCOPES)

        assert out is None

    async def test_frame_without_tools_is_untouched(self):
        """An error result has nothing to withhold, so pass it through as-is."""
        body = _sse({"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "nope"}})

        out = await _filter_sse_tools_list_body(body, SERVER, SCOPES)

        assert out == body.rstrip("\n") or out == body
        assert "nope" in out

    async def test_empty_filter_result_yields_empty_tools(self):
        """Filtering everything out is a valid, fail-closed outcome."""
        body = _sse(_tools_payload("blocked_one"))

        with patch(
            "auth_server.server.filter_tools_list_response",
            new=AsyncMock(return_value=[]),
        ):
            out = await _filter_sse_tools_list_body(body, SERVER, SCOPES)

        data = [ln for ln in out.split("\n") if ln.startswith("data: ")][0]
        assert json.loads(data[len("data: ") :])["result"]["tools"] == []
        assert "blocked_one" not in out


class TestShouldFilterToolsList:
    """The gate itself must accept SSE.

    These are the tests that protect the wiring. The body-filter tests above
    exercise the helper directly, so they keep passing even if this gate is
    narrowed back to application/json, which is how the hole stayed open.
    """

    def test_sse_is_filtered(self):
        assert _should_filter_tools_list(
            filter_enabled=True,
            incoming_method="tools/list",
            status_code=200,
            content_type="text/event-stream; charset=utf-8",
        )

    def test_json_is_filtered(self):
        assert _should_filter_tools_list(
            filter_enabled=True,
            incoming_method="tools/list",
            status_code=200,
            content_type="application/json",
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"filter_enabled": False},
            {"incoming_method": "tools/call"},
            {"status_code": 500},
            {"content_type": "text/plain"},
        ],
    )
    def test_not_filtered(self, kwargs):
        base = {
            "filter_enabled": True,
            "incoming_method": "tools/list",
            "status_code": 200,
            "content_type": "text/event-stream",
        }
        assert not _should_filter_tools_list(**{**base, **kwargs})
