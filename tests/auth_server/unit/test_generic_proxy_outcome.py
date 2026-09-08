"""One test per hop outcome value (issue #1735, item 4).

``mcpgw_registry_generic_proxy_request_total`` is the counter that answers what
the CALLER saw. ``auth_request_total`` cannot: the metrics middleware returns
early on anything but ``/validate``, so its ``success`` label records the
authorization decision and nothing after it — a 502 credential-vend failure
reaches the client while that metric says ``success=true``.

Every value gets a case because a missed exit path produces **silence, not a
failure**: nothing errors, the series simply never moves. These drive the real
decorated handler (``server.generic_proxy`` is the wrapper) and the real
streaming function, so the wiring is under test, not a copy of it.

Two collisions are the reason the outcome set is not just "status code":

- two different 503s — ``disabled`` (feature latched off) vs ``capacity``
  (pool saturated). Conflating them makes a switched-off deployment page
  somebody for saturation.
- three different 502s — ``auth_unavailable`` (vend failed), ``egress_blocked``
  (SSRF refusal at connect time) and ``upstream_error`` (dead backend). An SSRF
  refusal must not read as a credential outage.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-definitely-long-enough-32b")

from fastapi import HTTPException  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import StreamingResponse  # noqa: E402

import auth_server.server as server  # noqa: E402
from auth_server.observability.meters import HOP_ENTITY_TYPES, HOP_OUTCOMES  # noqa: E402
from registry.exceptions import UrlValidationError  # noqa: E402

pytestmark = pytest.mark.unit

UPSTREAM = "https://backend.example/api"


def _request(*, streaming: bool = False, has_upstream_auth: bool = False) -> Request:
    """A hop request whose verified claims are already stashed by the dependency."""

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/proxy/skill/skills/demo",
        "query_string": b"",
        "headers": [(b"accept", b"application/json")],
        "client": ("10.0.0.1", 1234),
        "server": ("auth-server", 8888),
        "scheme": "http",
    }
    request = Request(scope, receive=receive)
    request.state.generic_proxy_claims = {
        "upstream_url": UPSTREAM,
        "server": "skills/demo",
        "streaming": streaming,
        "has_upstream_auth": has_upstream_auth,
    }
    return request


class _FakeUpstreamResponse:
    """Minimal stand-in for the httpx streaming response the hop consumes."""

    def __init__(self, status_code: int = 200, body: bytes = b"{}", chunks=None):
        self.status_code = status_code
        self.headers = {"content-type": "application/json"}
        self._body = body
        self._chunks = chunks

    async def aiter_bytes(self, chunk_size: int = 65536):
        yield self._body

    def aiter_raw(self):
        chunks = self._chunks if self._chunks is not None else [self._body]

        async def _gen():
            for chunk in chunks:
                if isinstance(chunk, BaseException):
                    raise chunk
                yield chunk

        return _gen()


def _client_returning(response, *, stream_raises: BaseException | None = None):
    """Patch target for guarded_async_client: async CM -> client with .stream()."""
    stream_cm = MagicMock()
    if stream_raises is not None:
        stream_cm.__aenter__ = AsyncMock(side_effect=stream_raises)
    else:
        stream_cm.__aenter__ = AsyncMock(return_value=response)
    stream_cm.__aexit__ = AsyncMock(return_value=False)

    client = MagicMock()
    client.stream = MagicMock(return_value=stream_cm)
    client.aclose = AsyncMock()

    client_cm = MagicMock()
    client_cm.__aenter__ = AsyncMock(return_value=client)
    client_cm.__aexit__ = AsyncMock(return_value=False)
    # The streaming path uses the client WITHOUT `async with`, so the factory's
    # return value must double as the client itself.
    client_cm.stream = client.stream
    client_cm.aclose = client.aclose
    return client_cm


async def _call_hop(request, **patches):
    """Drive the decorated handler, returning the recorded (entity_type, outcome)."""
    with patch("auth_server.server.record_generic_proxy_request") as recorder:
        stack = []
        try:
            for target, kwargs in patches.items():
                ctx = patch(f"auth_server.server.{target}", **kwargs)
                stack.append(ctx)
                ctx.start()
            with patch.object(server, "_generic_proxy_feature_active", True):
                try:
                    response = await server.generic_proxy("skill", "skills/demo", request)
                except BaseException as exc:  # noqa: BLE001 - the outcome is the assertion
                    response = exc
        finally:
            for ctx in reversed(stack):
                ctx.stop()
    calls = [c.args for c in recorder.call_args_list]
    return response, calls


class TestBufferedOutcomes:
    """The buffered path: one record per request, whoever knows best records it."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (200, "ok"),
            (204, "ok"),
            (404, "upstream_4xx"),
            (429, "upstream_4xx"),
            (500, "upstream_5xx"),
            (503, "upstream_5xx"),
        ],
    )
    async def test_returned_status_maps_to_outcome(self, status, expected):
        response, calls = await _call_hop(
            _request(),
            guarded_async_client={
                "return_value": _client_returning(_FakeUpstreamResponse(status_code=status))
            },
        )
        assert calls == [("skill", expected)]

    @pytest.mark.asyncio
    async def test_disabled_is_not_capacity(self):
        """Both return 503. Deriving from the status alone would page for saturation."""
        request = _request()
        with (
            patch("auth_server.server.record_generic_proxy_request") as recorder,
            patch.object(server, "_generic_proxy_feature_active", False),
        ):
            with pytest.raises(HTTPException) as exc:
                await server.generic_proxy("skill", "skills/demo", request)
        assert exc.value.status_code == 503
        assert [c.args for c in recorder.call_args_list] == [("skill", "disabled")]

    @pytest.mark.asyncio
    async def test_capacity_from_the_acquire_helper(self):
        """The 503 is raised inside a helper that never touches the handler's returns."""
        _, calls = await _call_hop(
            _request(),
            _acquire_generic_proxy_slot={
                "side_effect": HTTPException(status_code=503, detail="Generic proxy is at capacity")
            },
        )
        assert calls == [("skill", "capacity")]

    @pytest.mark.asyncio
    async def test_helper_raised_400_is_rejected_and_not_a_500(self):
        """The SSRF sub-path confinement refusals raise before the handler's own try.

        Revision 1 of the design read the outcome in a `finally` without assigning
        it first, so these turned a 400 into an UnboundLocalError-driven 500. This
        asserts both halves: the status survives and the outcome is `rejected`.
        """
        response, calls = await _call_hop(
            _request(),
            _build_generic_outbound_url={
                "side_effect": HTTPException(status_code=400, detail="Illegal sub-path")
            },
        )
        assert isinstance(response, HTTPException)
        assert response.status_code == 400
        assert calls == [("skill", "rejected")]

    @pytest.mark.asyncio
    async def test_host_pin_failure_is_rejected(self):
        _, calls = await _call_hop(
            _request(),
            _assert_outbound_host_pinned={
                "side_effect": HTTPException(status_code=400, detail="Proxy target host mismatch")
            },
        )
        assert calls == [("skill", "rejected")]

    @pytest.mark.asyncio
    async def test_byte_cap_from_read_bounded(self):
        _, calls = await _call_hop(
            _request(),
            guarded_async_client={"return_value": _client_returning(_FakeUpstreamResponse())},
            _read_bounded={
                "side_effect": HTTPException(status_code=413, detail="Upstream response too large")
            },
        )
        assert calls == [("skill", "byte_cap")]

    @pytest.mark.asyncio
    async def test_vend_failure_is_auth_unavailable_not_upstream_error(self):
        """Shares 502 with a dead backend; a credential outage needs its own signal."""
        _, calls = await _call_hop(
            _request(has_upstream_auth=True),
            _vend_generic_upstream_headers={"return_value": None},
        )
        assert calls == [("skill", "auth_unavailable")]

    @pytest.mark.asyncio
    async def test_egress_block_is_not_auth_unavailable(self):
        """UrlValidationError = the guarded transport refused a rebound/private IP."""
        _, calls = await _call_hop(
            _request(),
            guarded_async_client={
                "return_value": _client_returning(
                    None, stream_raises=UrlValidationError("blocked", reason="private_ip")
                )
            },
        )
        assert calls == [("skill", "egress_blocked")]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "raised",
        [httpx.ConnectError("refused"), httpx.TimeoutException("slow"), httpx.ReadError("reset")],
    )
    async def test_transport_failures_are_upstream_error(self, raised):
        _, calls = await _call_hop(
            _request(),
            guarded_async_client={"return_value": _client_returning(None, stream_raises=raised)},
        )
        assert calls == [("skill", "upstream_error")]

    @pytest.mark.asyncio
    async def test_client_hangup_is_client_closed_not_internal_error(self):
        """CancelledError derives from BaseException.

        Without its own arm it lands in the bug bucket, so a caller pressing
        Ctrl-C would page whoever alerts on internal_error.
        """
        _, calls = await _call_hop(
            _request(),
            guarded_async_client={
                "return_value": _client_returning(None, stream_raises=asyncio.CancelledError())
            },
        )
        assert calls == [("skill", "client_closed")]

    @pytest.mark.asyncio
    async def test_unexpected_exception_is_internal_error(self):
        _, calls = await _call_hop(
            _request(),
            guarded_async_client={
                "return_value": _client_returning(None, stream_raises=RuntimeError("bug"))
            },
        )
        assert calls == [("skill", "internal_error")]

    @pytest.mark.asyncio
    async def test_exactly_one_record_per_request(self):
        """The wrapper sees every failure the sites already recorded; first wins."""
        _, calls = await _call_hop(
            _request(has_upstream_auth=True),
            _vend_generic_upstream_headers={"return_value": None},
        )
        assert len(calls) == 1, f"double counted: {calls}"


class TestEntityTypeLabel:
    """entity_type collapses to three values, whatever operators name their types."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("entity_type", "expected"),
        [
            ("skill", "skill"),
            ("a2a_agent", "a2a_agent"),
            ("rest-endpoint", "custom"),
            ("some-operator-type", "custom"),
        ],
    )
    async def test_label_is_bounded_to_three_values(self, entity_type, expected):
        request = _request()
        with (
            patch("auth_server.server.record_generic_proxy_request") as recorder,
            patch.object(server, "_generic_proxy_feature_active", False),
        ):
            with pytest.raises(HTTPException):
                await server.generic_proxy(entity_type, "x/y", request)
        assert recorder.call_args_list[0].args[0] == expected


class TestStreamingOutcomes:
    """The streaming path owns its terminal; the wrapper must stay silent for it."""

    @staticmethod
    async def _drive_stream(response=None, *, stream_raises=None, consume=True, chunks=None):
        recorded = []
        upstream = response or _FakeUpstreamResponse(chunks=chunks)
        with (
            patch(
                "auth_server.server.record_generic_proxy_request",
                side_effect=lambda e, o: recorded.append((e, o)),
            ),
            patch("auth_server.server.record_generic_proxy_stream_outcome"),
            patch(
                "auth_server.server.guarded_async_client",
                return_value=_client_returning(upstream, stream_raises=stream_raises),
            ),
            patch("auth_server.server._acquire_generic_proxy_slot", AsyncMock()),
        ):
            semaphore = asyncio.Semaphore(1)
            try:
                result = await server._generic_proxy_streaming(
                    semaphore=semaphore,
                    method="GET",
                    outbound_url=UPSTREAM,
                    request_body=b"",
                    forward_headers={},
                    verify=True,
                    metrics_entity="custom",
                )
            except BaseException as exc:  # noqa: BLE001
                return exc, recorded
            if consume and isinstance(result, StreamingResponse):
                # A terminal failure mid-body propagates to the ASGI server in
                # production; here it just must not hide the recorded outcome.
                try:
                    async for _ in result.body_iterator:
                        pass
                except BaseException as exc:  # noqa: BLE001 - the record is the assertion
                    return exc, recorded
            return result, recorded

    @staticmethod
    async def _drive_stream_through_handler(*, stream_raises):
        """Pre-header failures must be driven through the wrapper.

        The site records via the idempotent per-request recorder, which only exists
        inside the wrapper — that is the whole point: whichever layer knows the
        precise meaning records first, and the wrapper's status-derived attempt
        then no-ops. Calling the streaming function bare would record nothing and
        prove nothing.
        """
        recorded = []
        with (
            patch(
                "auth_server.server.record_generic_proxy_request",
                side_effect=lambda e, o: recorded.append((e, o)),
            ),
            patch("auth_server.server.record_generic_proxy_stream_outcome"),
            patch(
                "auth_server.server.guarded_async_client",
                return_value=_client_returning(None, stream_raises=stream_raises),
            ),
            patch("auth_server.server._acquire_generic_proxy_slot", AsyncMock()),
            patch.object(server, "_generic_proxy_feature_active", True),
        ):
            try:
                result = await server.generic_proxy(
                    "skill", "skills/demo", _request(streaming=True)
                )
            except BaseException as exc:  # noqa: BLE001
                result = exc
        return result, recorded

    @pytest.mark.asyncio
    async def test_completed_stream_records_ok_once(self):
        result, recorded = await self._drive_stream(chunks=[b"data: 1\n", b"data: 2\n"])
        assert isinstance(result, StreamingResponse)
        assert recorded == [("custom", "ok")]

    @pytest.mark.asyncio
    async def test_pre_header_egress_block(self):
        result, recorded = await self._drive_stream_through_handler(
            stream_raises=UrlValidationError("blocked", reason="metadata_ip")
        )
        assert isinstance(result, HTTPException) and result.status_code == 502
        assert recorded == [("skill", "egress_blocked")]

    @pytest.mark.asyncio
    async def test_pre_header_duration_timeout_is_not_upstream_error(self):
        """Raised as 504, which maps to upstream_error by status — so it records itself."""
        result, recorded = await self._drive_stream_through_handler(
            stream_raises=TimeoutError("deadline")
        )
        assert isinstance(result, HTTPException) and result.status_code == 504
        assert recorded == [("skill", "duration_timeout")]

    @pytest.mark.asyncio
    async def test_mid_stream_upstream_error(self):
        result, recorded = await self._drive_stream(chunks=[b"partial", httpx.ReadError("dropped")])
        assert recorded == [("custom", "upstream_error")]

    @pytest.mark.asyncio
    async def test_byte_cap_mid_stream(self):
        with patch("auth_server.server._read_generic_stream_max_bytes", return_value=4):
            result, recorded = await self._drive_stream(chunks=[b"12345678"])
        assert recorded == [("custom", "byte_cap")]

    @pytest.mark.asyncio
    async def test_client_disconnect_mid_stream(self):
        recorded = []
        with (
            patch(
                "auth_server.server.record_generic_proxy_request",
                side_effect=lambda e, o: recorded.append((e, o)),
            ),
            patch("auth_server.server.record_generic_proxy_stream_outcome"),
            patch(
                "auth_server.server.guarded_async_client",
                return_value=_client_returning(_FakeUpstreamResponse(chunks=[b"a", b"b", b"c"])),
            ),
            patch("auth_server.server._acquire_generic_proxy_slot", AsyncMock()),
        ):
            result = await server._generic_proxy_streaming(
                semaphore=asyncio.Semaphore(1),
                method="GET",
                outbound_url=UPSTREAM,
                request_body=b"",
                forward_headers={},
                verify=True,
                metrics_entity="custom",
            )
            iterator = result.body_iterator
            await iterator.__anext__()
            await iterator.aclose()  # client hung up mid-stream
        assert recorded == [("custom", "client_closed")]

    @pytest.mark.asyncio
    async def test_wrapper_does_not_double_count_a_stream(self):
        """The handler returns a StreamingResponse; only the generator records."""
        request = _request(streaming=True)
        upstream = _FakeUpstreamResponse(chunks=[b"x"])
        with (
            patch("auth_server.server.record_generic_proxy_request") as recorder,
            patch("auth_server.server.record_generic_proxy_stream_outcome"),
            patch(
                "auth_server.server.guarded_async_client",
                return_value=_client_returning(upstream),
            ),
            patch("auth_server.server._acquire_generic_proxy_slot", AsyncMock()),
            patch.object(server, "_generic_proxy_feature_active", True),
        ):
            response = await server.generic_proxy("skill", "skills/demo", request)
            assert isinstance(response, StreamingResponse)
            # Nothing recorded yet: the terminal belongs to the generator.
            assert recorder.call_args_list == []
            async for _ in response.body_iterator:
                pass
        assert [c.args for c in recorder.call_args_list] == [("skill", "ok")]


class TestLabelSets:
    """The declared label sets are the contract zero-init and dashboards rely on."""

    def test_thirteen_outcomes_three_entity_types(self):
        assert len(HOP_OUTCOMES) == 13
        assert len(HOP_ENTITY_TYPES) == 3
        assert len(HOP_ENTITY_TYPES) * len(HOP_OUTCOMES) == 39

    def test_every_outcome_this_suite_records_is_declared(self):
        """A typo in a record call would otherwise ship an undeclared, unseeded series."""
        recorded_here = {
            "ok",
            "upstream_4xx",
            "upstream_5xx",
            "upstream_error",
            "egress_blocked",
            "auth_unavailable",
            "capacity",
            "disabled",
            "rejected",
            "byte_cap",
            "client_closed",
            "duration_timeout",
            "internal_error",
        }
        assert recorded_here == set(HOP_OUTCOMES)

    def test_status_map_covers_the_hop_statuses(self):
        assert server._HOP_STATUS_OUTCOMES == {
            400: "rejected",
            413: "byte_cap",
            502: "upstream_error",
            503: "capacity",
            504: "upstream_error",
        }
