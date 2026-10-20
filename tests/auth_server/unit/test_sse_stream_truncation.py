"""A chunked SSE stream that ends at connection close must not lose its events.

`_read_bounded` buffers the whole upstream body, so a stream cut without its
terminating chunk used to fail as one opaque 502 (issue #1908) even when every
event had already arrived. These tests pin the replacement at both levels: the
read reports truncation as a typed error carrying what it received, and the
proxy route delivers the events that completed.

The default path matters as much as the new one. `_read_bounded` has a second
caller that cannot use a partial body, so the opt-out has to keep raising the
original httpx error rather than the domain one.

The route-level cases drive the real FastAPI handler with a mocked httpx
upstream, the way test_server.py's header-passthrough class does, so the
wiring is under test and not a copy of it.
"""

import os
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-definitely-long-enough-32b")

import httpx  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import auth_server.server as server  # noqa: E402

pytestmark = pytest.mark.unit

TRUNCATED = "peer closed connection without sending complete message body (incomplete chunked read)"

EVENT_ONE = b'id: 1\ndata: {"jsonrpc":"2.0"}\n\n'
EVENT_TWO = b'id: 2\ndata: {"result":{"content":[]}}\n\n'
PARTIAL_TAIL = b'id: 3\ndata: {"res'

SERVER_NAME = "office-docs"
UPSTREAM_URL = "https://upstream.example/mcp"


class _StreamingResponse:
    """Minimal stand-in for a streaming httpx.Response.

    Args:
        chunks: Body chunks the stream yields before anything else.
        then: Exception to raise once the chunks are exhausted.
    """

    def __init__(
        self,
        chunks: list[bytes],
        then: Exception | None = None,
    ) -> None:
        self._chunks = chunks
        self._then = then

    async def aiter_bytes(
        self,
        *,
        chunk_size: int = 0,
    ) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
        if self._then is not None:
            raise self._then


class TestReadBoundedTruncation:
    async def test_reports_truncation_with_the_bytes_received(self):
        response = _StreamingResponse(
            [EVENT_ONE, EVENT_TWO], then=httpx.RemoteProtocolError(TRUNCATED)
        )

        with pytest.raises(server._UpstreamStreamTruncated) as excinfo:
            await server._read_bounded(response, 4096, partial_on_truncation=True)

        assert excinfo.value.partial == EVENT_ONE + EVENT_TWO

    async def test_default_caller_still_sees_the_httpx_error(self):
        response = _StreamingResponse([EVENT_ONE], then=httpx.RemoteProtocolError(TRUNCATED))

        with pytest.raises(httpx.RemoteProtocolError):
            await server._read_bounded(response, 4096)

    async def test_still_rejects_an_oversized_body_while_truncation_is_opted_in(self):
        response = _StreamingResponse([b"x" * 64])

        with pytest.raises(HTTPException) as excinfo:
            await server._read_bounded(response, 10, partial_on_truncation=True)

        assert excinfo.value.status_code == 413

    async def test_returns_the_whole_body_when_the_stream_completes(self):
        response = _StreamingResponse([EVENT_ONE, EVENT_TWO])

        body = await server._read_bounded(response, 4096, partial_on_truncation=True)

        assert body == EVENT_ONE + EVENT_TWO


class TestCompleteEventsOnly:
    def test_keeps_completed_events_and_drops_a_partial_tail(self):
        assert (
            server._complete_events_only(EVENT_ONE + EVENT_TWO + PARTIAL_TAIL)
            == EVENT_ONE + EVENT_TWO
        )

    def test_keeps_events_terminated_with_crlf(self):
        assert server._complete_events_only(b"data: a\r\n\r\ndata: b") == b"data: a\r\n\r\n"

    def test_returns_empty_when_no_event_ever_completed(self):
        assert server._complete_events_only(b"data: {") == b""

    def test_returns_empty_for_no_bytes_at_all(self):
        assert server._complete_events_only(b"") == b""


def _build_upstream_response(
    headers: dict[str, str],
    chunks: list[bytes],
    then: Exception | None,
) -> MagicMock:
    """Build the httpx.Response surface mcp_proxy uses: status, headers, stream.

    Args:
        headers: Response headers, including the content type.
        chunks: Body chunks the stream yields.
        then: Exception to raise after the chunks, or None for a clean end.

    Returns:
        A MagicMock mcp_proxy can stream from.
    """
    upstream = MagicMock()
    upstream.status_code = 200
    upstream.headers = headers

    async def _aiter(
        chunk_size: int = 64 * 1024,
    ) -> AsyncIterator[bytes]:
        for chunk in chunks:
            yield chunk
        if then is not None:
            raise then

    upstream.aiter_bytes = _aiter
    return upstream


def _patch_upstream(upstream_response: MagicMock):
    """Patch httpx.AsyncClient so mcp_proxy's stream chain yields this response."""
    stream_cm = AsyncMock()
    stream_cm.__aenter__ = AsyncMock(return_value=upstream_response)
    stream_cm.__aexit__ = AsyncMock(return_value=False)

    client = AsyncMock()
    client.stream = MagicMock(return_value=stream_cm)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    return patch("auth_server.server.httpx.AsyncClient", return_value=client)


def _patch_scope_repo_allow_all():
    """Grant the wildcard scope so the handler's re-authorization passes."""
    repo = AsyncMock()

    async def _get_server_scopes(scope_name: str):
        if scope_name == "admin:all":
            return [{"server": "*", "methods": ["*"], "tools": ["*"]}]
        return []

    repo.get_server_scopes.side_effect = _get_server_scopes
    return patch("auth_server.server.get_scope_repository", return_value=repo)


def _proxy_token_headers() -> dict:
    """Mint the X-Internal-Token nginx's /validate hop would forward."""
    from auth_server.internal_request_token import mint_mcp_proxy_token

    token = mint_mcp_proxy_token(
        subject="test-user",
        scopes=["admin:all"],
        server_name=SERVER_NAME,
        upstream_url=UPSTREAM_URL,
    )
    return {
        "X-Internal-Token": token,
        "X-Registered-Server-Path": "/" + SERVER_NAME,
        "X-Registered-Route-Mode": "direct",
    }


class TestMcpProxyHandlesTruncatedStream:
    """The route-level outcome for a stream that ends at connection close."""

    def _post(self, upstream_response: MagicMock):
        with (
            _patch_upstream(upstream_response),
            _patch_scope_repo_allow_all(),
            patch.object(server, "_read_mcp_filter_enabled", return_value=False),
        ):
            client = TestClient(server.app)
            return client.post(
                f"/mcp-proxy/{SERVER_NAME}",
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
                headers=_proxy_token_headers(),
            )

    def test_delivers_completed_events_and_drops_the_cut_tail(self):
        upstream = _build_upstream_response(
            headers={"content-type": "text/event-stream", "mcp-session-id": "sess-cut-001"},
            chunks=[EVENT_ONE, EVENT_TWO, PARTIAL_TAIL],
            then=httpx.RemoteProtocolError(TRUNCATED),
        )

        response = self._post(upstream)

        assert response.status_code == 200
        assert response.content == EVENT_ONE + EVENT_TWO

    def test_still_fails_when_no_event_ever_completed(self):
        upstream = _build_upstream_response(
            headers={"content-type": "text/event-stream"},
            chunks=[b'{"jsonrpc":"2.0"'],
            then=httpx.RemoteProtocolError(TRUNCATED),
        )

        response = self._post(upstream)

        assert response.status_code == 502
        assert response.json()["detail"] == "Upstream MCP server error"

    def test_a_stream_that_ends_cleanly_is_untouched(self):
        upstream = _build_upstream_response(
            headers={"content-type": "text/event-stream"},
            chunks=[EVENT_ONE, EVENT_TWO],
            then=None,
        )

        response = self._post(upstream)

        assert response.status_code == 200
        assert response.content == EVENT_ONE + EVENT_TWO
