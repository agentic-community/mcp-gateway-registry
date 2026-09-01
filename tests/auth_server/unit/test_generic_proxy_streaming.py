"""Runtime tests for the generic-proxy STREAMING hop (_generic_proxy_streaming).

The streaming handler is the highest-risk new code on the generic proxy: it holds
a concurrency slot for the whole (long-lived) stream lifetime, disables the httpx
read timeout, and must release the slot + close the transport on every exit path
(normal completion, byte-cap, duration/idle timeout, upstream error, capacity
rejection). These tests drive the handler directly with a fake guarded client and
assert both the byte-level behavior AND that no slot is leaked, plus that the new
observability counters fire with the right outcome labels.
"""

import asyncio
import os
from unittest.mock import patch

import httpx
import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-definitely-long-enough-32b")

from fastapi import HTTPException  # noqa: E402

import auth_server.server as server_module  # noqa: E402

pytestmark = pytest.mark.unit

class _FakeRawResponse:
    def __init__(self, status_code, headers, chunks):
        self.status_code = status_code
        self.headers = headers
        self._chunks = chunks

    async def aiter_raw(self):
        for chunk in self._chunks:
            yield chunk


class _FakeStreamCtx:
    def __init__(self, response, raise_on_enter=None):
        self._response = response
        self._raise = raise_on_enter

    async def __aenter__(self):
        if self._raise is not None:
            raise self._raise
        return self._response

    async def __aexit__(self, *a):
        return False


class _FakeStreamClient:
    """Stand-in for guarded_async_client(): opened manually on the streaming path,
    aclose()'d by the cleanup task."""

    def __init__(self, response=None, raise_on_enter=None):
        self._response = response
        self._raise = raise_on_enter
        self.closed = False

    def stream(self, method, url, **kw):
        return _FakeStreamCtx(self._response, self._raise)

    async def aclose(self):
        self.closed = True


async def _drain(resp):
    return b"".join([chunk async for chunk in resp.body_iterator])


async def _call(sem, client):
    with patch.object(server_module, "guarded_async_client", lambda *a, **kw: client):
        return await server_module._generic_proxy_streaming(
            semaphore=sem,
            method="GET",
            outbound_url="https://backend.example/stream",
            request_body=b"",
            forward_headers={},
            verify=True,
        )


class TestStreamingHappyPath:
    async def test_streams_chunks_incrementally_and_releases_slot(self):
        sem = asyncio.Semaphore(2)
        client = _FakeStreamClient(
            _FakeRawResponse(200, {"content-type": "text/event-stream"}, [b"da", b"ta"])
        )
        with patch.object(server_module, "record_generic_proxy_stream_outcome") as rec:
            resp = await _call(sem, client)
            assert resp.status_code == 200
            # Gateway security headers are stamped on the streamed response.
            assert resp.headers.get("X-Content-Type-Options") == "nosniff"
            body = await _drain(resp)
        assert body == b"data"
        # Slot released and transport closed after the stream completes.
        assert sem._value == 2
        assert client.closed is True
        outcomes = [c.args[0] for c in rec.call_args_list]
        assert "started" in outcomes and "completed" in outcomes


class TestStreamingByteCap:
    async def test_byte_cap_aborts_stream_records_and_releases_slot(self):
        sem = asyncio.Semaphore(1)
        client = _FakeStreamClient(
            _FakeRawResponse(200, {"content-type": "application/octet-stream"}, [b"AAAA"])
        )
        with (
            patch.object(server_module, "_read_generic_stream_max_bytes", return_value=3),
            patch.object(server_module, "record_generic_proxy_stream_outcome") as rec,
        ):
            resp = await _call(sem, client)
            with pytest.raises(HTTPException) as ei:
                await _drain(resp)
        assert ei.value.status_code == 413
        assert sem._value == 1  # no leaked slot even on mid-stream abort
        assert client.closed is True
        assert "byte_cap" in [c.args[0] for c in rec.call_args_list]


class TestStreamingSetupError:
    async def test_upstream_error_releases_slot_and_records(self):
        sem = asyncio.Semaphore(1)
        client = _FakeStreamClient(raise_on_enter=httpx.ConnectError("refused"))
        with patch.object(server_module, "record_generic_proxy_stream_outcome") as rec:
            with pytest.raises(HTTPException) as ei:
                await _call(sem, client)
        assert ei.value.status_code == 502
        assert sem._value == 1  # slot released on setup failure
        assert client.closed is True
        assert "upstream_error" in [c.args[0] for c in rec.call_args_list]


class TestStreamingCapacity:
    async def test_capacity_rejection_503_records_pool_stream(self):
        sem = asyncio.Semaphore(1)
        await sem.acquire()  # pool is now full
        client = _FakeStreamClient(_FakeRawResponse(200, {}, [b"x"]))
        with (
            patch.object(server_module, "_read_generic_acquire_timeout_seconds", return_value=0.05),
            patch.object(server_module, "record_generic_proxy_slot_rejected") as rec,
        ):
            with pytest.raises(HTTPException) as ei:
                await _call(sem, client)
        assert ei.value.status_code == 503
        rec.assert_called_once_with("stream")
        # The rejected request must not have consumed the (already-held) slot.
        assert sem._value == 0
