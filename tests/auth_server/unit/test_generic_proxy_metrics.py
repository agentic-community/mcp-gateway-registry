"""Metrics classification and labeling for gateway-proxied requests (issue #1735).

Two properties are under test.

``classify_target_kind`` must attribute a gateway-proxied request to its entity
type instead of ``unknown``. It discriminates on the ``X-Generic-Proxy-Kind``
marker nginx sets per generated location, NOT on a path prefix:
``GATEWAY_PROXY_PREFIX`` reaches the registry container only, so a prefix-keyed
rule would silently keep filing gateway traffic as ``unknown`` whenever an
operator changed it.

``_emit_auth_metric`` must put the entity's authz key in the ``server`` label on
the counter, and must keep ``server`` OFF the duration histogram. A per-target
label costs 18 series on a 16-bucket histogram against the counter's 1, and no
query asks for /validate latency per target.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from starlette.requests import Request
from starlette.responses import Response

from auth_server.metrics_middleware import AuthMetricsMiddleware
from registry.observability.label_bounding import LabelCardinalityLimiter

pytestmark = pytest.mark.unit

UUID = "1a546ca6-4336-4164-8fd3-b5d7e6bccb56"


@pytest.fixture
def middleware() -> AuthMetricsMiddleware:
    return AuthMetricsMiddleware(app=MagicMock())


class TestClassifyGenericProxy:
    """A gateway request lands on its entity type, never on unknown."""

    @pytest.mark.parametrize(
        ("marker", "expected"),
        [
            ("skill", "generic_proxy_skill"),
            ("a2a_agent", "generic_proxy_agent"),
            ("rest-endpoint", "generic_proxy_custom"),
            # Operators define custom types at will, so anything that is not a
            # skill or an agent collapses to one label and the set stays at three.
            ("some-operator-type", "generic_proxy_custom"),
            ("llm-endpoint", "generic_proxy_custom"),
        ],
    )
    def test_marker_selects_the_label(self, middleware, marker, expected):
        url = f"http://localhost/gateway/{marker}/{UUID}/v1/models"
        assert middleware.classify_target_kind(url, marker) == expected

    def test_no_marker_leaves_a_gateway_shaped_path_unknown(self, middleware):
        """Without the marker a gateway-shaped path must NOT get a data-plane label.

        The marker is the only trustworthy signal. A caller can shape a URL to look
        like a gateway path, so path shape alone must not grant one.
        """
        url = f"http://localhost/gateway/skill/{UUID}/v1/models"
        assert middleware.classify_target_kind(url, "") == "unknown"

    @pytest.mark.parametrize("url", ["", None])
    def test_marker_classifies_without_x_original_url(self, middleware, url):
        """A missing X-Original-URL must not file gateway traffic under unknown.

        `server_name` is built from the markers alone, so classification has to be
        too: otherwise one request records the per-entity `server` label AND
        `target_kind="unknown"`, contradicting the documented invariant ("no gateway
        request lands in unknown") that OBSERVABILITY.md ships an alert query for.
        """
        assert middleware.classify_target_kind(url, "skill") == "generic_proxy_skill"
        assert middleware.classify_target_kind(url, "rest-endpoint") == "generic_proxy_custom"

    def test_no_marker_and_no_url_is_unknown(self, middleware):
        assert middleware.classify_target_kind("", "") == "unknown"

    def test_control_plane_still_wins_when_the_url_is_present(self, middleware):
        """Resolving the marker earlier must not reorder it ahead of control plane."""
        assert (
            middleware.classify_target_kind("http://localhost/api/skills/pdf", "skill")
            == "control_plane"
        )

    def test_marker_works_for_any_prefix(self, middleware):
        """A non-default GATEWAY_PROXY_PREFIX still classifies.

        This is the regression guard for the original design, which read
        GATEWAY_PROXY_PREFIX -- a variable docker-compose passes to the registry
        container and never to auth-server.
        """
        assert (
            middleware.classify_target_kind("http://localhost/edge/skill/pdf/x", "skill")
            == "generic_proxy_skill"
        )

    def test_control_plane_wins_over_the_marker(self, middleware):
        """An /api/ path stays control_plane even if a marker is somehow present.

        Control-plane-first ordering is what stops a control-plane call being
        counted as routed data-plane traffic.
        """
        assert (
            middleware.classify_target_kind("http://localhost/api/skills/pdf", "skill")
            == "control_plane"
        )

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("http://localhost/agent/travel-assistant-agent/", "a2a_agent"),
            ("http://localhost/virtual/vs-1/mcp", "virtual_mcp_server"),
            ("http://localhost/airegistry-tools/mcp", "mcp_server"),
            ("http://localhost/api/skills/pdf", "control_plane"),
            ("http://localhost/", "unknown"),
        ],
    )
    def test_existing_labels_are_unchanged(self, middleware, url, expected):
        assert middleware.classify_target_kind(url, "") == expected


class TestServerLabel:
    """The counter carries the authz key; the histogram carries no server label."""

    @staticmethod
    async def _emit(middleware, **kwargs):
        with (
            patch("auth_server.metrics_middleware.auth_request_total") as counter,
            patch("auth_server.metrics_middleware.auth_request_duration_ms") as hist,
        ):
            await middleware._emit_auth_metric(
                success=kwargs.get("success", True),
                duration_ms=kwargs.get("duration_ms", 12.0),
                method=kwargs.get("method", "self_signed"),
                server_name=kwargs["server_name"],
                target_kind=kwargs["target_kind"],
                user_hash="",
            )
        return counter, hist

    @pytest.mark.asyncio
    async def test_counter_carries_the_server_label(self, middleware):
        authz_key = f"rest-endpoint/rest-endpoint/{UUID}"
        counter, _ = await self._emit(
            middleware, server_name=authz_key, target_kind="generic_proxy_custom"
        )
        attrs = counter.add.call_args[0][1]
        assert attrs["server"] == authz_key
        assert attrs["target_kind"] == "generic_proxy_custom"

    @pytest.mark.asyncio
    async def test_histogram_omits_the_server_label(self, middleware):
        """18 of every 19 series came from `server` on a 16-bucket histogram."""
        counter, hist = await self._emit(
            middleware,
            server_name=f"rest-endpoint/rest-endpoint/{UUID}",
            target_kind="generic_proxy_custom",
        )
        hist_attrs = hist.record.call_args[0][1]
        assert "server" not in hist_attrs
        # Everything else survives, so the documented (le, target_kind) query
        # keeps working.
        assert hist_attrs["target_kind"] == "generic_proxy_custom"
        assert hist_attrs["success"] == "true"
        assert hist_attrs["method"] == "self_signed"

    @pytest.mark.asyncio
    async def test_a_64_character_authz_key_is_not_truncated(self, middleware):
        """openai-proxy's key is exactly 64 characters, the old label-length cap.

        A truncated key matches no scope rule, so the label would point an operator
        at a rule that cannot exist -- defeating its only purpose.
        """
        authz_key = f"rest-endpoint/rest-endpoint/{UUID}"
        assert len(authz_key) == 64
        counter, _ = await self._emit(
            middleware, server_name=authz_key, target_kind="generic_proxy_custom"
        )
        assert counter.add.call_args[0][1]["server"] == authz_key

    @pytest.mark.asyncio
    async def test_a_long_custom_type_key_survives(self, middleware):
        """A type name longer than 'rest-endpoint' pushes the key past 64 chars."""
        authz_key = f"model-inference-endpoint/model-inference-endpoint/{UUID}"
        assert len(authz_key) > 64
        counter, _ = await self._emit(
            middleware, server_name=authz_key, target_kind="generic_proxy_custom"
        )
        assert counter.add.call_args[0][1]["server"] == authz_key


class TestServerLabelFromMarkers:
    """server_name comes from the nginx markers, not from the client path.

    Parsing the client path yields "skill/pdf" for a skill registered at
    /skills/pdf, because build_proxy_client_path strips the namespace segment. The
    key /validate authorizes against is "skill/skills/pdf", so a parsed key names
    no real scope rule. The markers also cannot be spoofed, unlike X-Original-URL,
    which is $request_uri on a trailing-slash prefix match.
    """

    @pytest.mark.parametrize(
        ("kind", "entity_path", "expected"),
        [
            # A skill's registered path keeps its namespace segment, which is what
            # the scope rule names. The client path would have said "skill/pdf".
            ("skill", "skills/pdf", "skill/skills/pdf"),
            ("rest-endpoint", f"rest-endpoint/{UUID}", f"rest-endpoint/rest-endpoint/{UUID}"),
            ("a2a_agent", "agents/travel-assistant", "a2a_agent/agents/travel-assistant"),
        ],
    )
    def test_key_matches_the_validate_expression(self, kind, entity_path, expected):
        """The middleware must build the SAME string /validate authorizes against.

        nginx pre-strips the value it puts in X-Entity-Path
        (registry/core/nginx_service.py:1884 does path.strip("/")), so the join
        needs no interior cleanup. The expression is deliberately identical to
        server.py:3955 -- a metric label that disagrees with the authz key would
        name a rule the authorization layer never consulted.
        """
        assert f"{kind}/{entity_path}".strip("/") == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize("with_original_url", [True, False])
    async def test_dispatch_labels_a_gateway_request_from_the_markers(
        self, middleware, with_original_url
    ):
        """Drive the real dispatch path, not a re-typed copy of the expression.

        Covers both labels at once and both header shapes: with X-Original-URL and
        without it. The second case is the one that used to record the per-entity
        `server` alongside `target_kind="unknown"`.
        """
        headers = [
            (b"x-generic-proxy-kind", b"rest-endpoint"),
            (b"x-entity-path", f"rest-endpoint/{UUID}".encode()),
        ]
        if with_original_url:
            headers.append(
                (b"x-original-url", f"http://localhost/gateway/rest-endpoint/{UUID}/v1".encode())
            )
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/validate",
                "query_string": b"",
                "headers": headers,
                "client": ("10.0.0.1", 1234),
                "server": ("localhost", 80),
                "scheme": "http",
            }
        )

        async def call_next(_request):
            return Response(status_code=200, headers={"X-Auth-Method": "self_signed"})

        with (
            patch("auth_server.metrics_middleware.auth_request_total") as counter,
            patch("auth_server.metrics_middleware.auth_request_duration_ms"),
            patch("auth_server.metrics_middleware.record_emission_path"),
        ):
            await middleware.dispatch(request, call_next)
            # Emission is fire-and-forget via create_task; let those tasks run.
            for _ in range(5):
                await asyncio.sleep(0)

        attrs = counter.add.call_args[0][1]
        assert attrs["server"] == f"rest-endpoint/rest-endpoint/{UUID}"
        assert attrs["target_kind"] == "generic_proxy_custom"


class TestZeroInit:
    """Counters must expose a zero series before any traffic."""

    def test_seeds_every_known_label_value(self):
        from auth_server.observability import meters

        with (
            patch.object(meters.generic_proxy_slot_rejected_total, "add") as slot_add,
            patch.object(meters.generic_proxy_stream_outcome_total, "add") as stream_add,
            patch.object(meters.metrics, "get_meter_provider", return_value=MagicMock()),
        ):
            meters.zero_init_metrics()

        assert {c[0][1]["pool"] for c in slot_add.call_args_list} == {"buffered", "stream"}
        assert {c[0][1]["outcome"] for c in stream_add.call_args_list} == {
            "started",
            "completed",
            "duration_timeout",
            "byte_cap",
            "upstream_error",
            "client_closed",
        }
        # Zero, not one: seeding must not change any total.
        assert all(c[0][0] == 0 for c in slot_add.call_args_list)
        assert all(c[0][0] == 0 for c in stream_add.call_args_list)

    def test_seeds_all_39_hop_outcome_combinations(self):
        """The hop counter is the one an alert binds to, so it must exist at zero.

        39 series = 3 entity types x 13 outcomes, flat whatever the endpoint count.
        A missed combination is invisible until the day it increments, which is the
        day you needed the alert to have been armed already.
        """
        from auth_server.observability import meters

        with (
            patch.object(meters.generic_proxy_request_total, "add") as hop_add,
            patch.object(meters.generic_proxy_slot_rejected_total, "add"),
            patch.object(meters.generic_proxy_stream_outcome_total, "add"),
            patch.object(meters.metrics, "get_meter_provider", return_value=MagicMock()),
        ):
            meters.zero_init_metrics()

        seeded = {(c[0][1]["entity_type"], c[0][1]["outcome"]) for c in hop_add.call_args_list}
        expected = {(e, o) for e in meters.HOP_ENTITY_TYPES for o in meters.HOP_OUTCOMES}
        assert seeded == expected
        assert len(seeded) == 39
        assert all(c[0][0] == 0 for c in hop_add.call_args_list)

    def test_skips_and_reports_a_no_op_provider(self, caplog):
        """With no SDK provider every add(0) is discarded, so say so once.

        _init_meter_provider_if_needed returns early unless
        OTEL_EXPORTER_PROMETHEUS_HOST is set, and a failed start_http_server leaves
        a proxy provider. Silence would leave an operator believing the series exist.
        """
        from auth_server.observability import meters

        provider = MagicMock()
        type(provider).__name__ = "NoOpMeterProvider"
        with (
            patch.object(meters.generic_proxy_slot_rejected_total, "add") as slot_add,
            patch.object(meters.metrics, "get_meter_provider", return_value=provider),
            caplog.at_level("INFO"),
        ):
            meters.zero_init_metrics()

        slot_add.assert_not_called()
        assert "zero-init skipped" in caplog.text

    def test_one_failure_does_not_skip_the_rest(self):
        """Each add() gets its own try, so a single raise cannot void the seeding."""
        from auth_server.observability import meters

        with (
            patch.object(
                meters.generic_proxy_slot_rejected_total,
                "add",
                side_effect=[RuntimeError("boom"), None],
            ) as slot_add,
            patch.object(meters.generic_proxy_stream_outcome_total, "add") as stream_add,
            patch.object(meters.metrics, "get_meter_provider", return_value=MagicMock()),
        ):
            meters.zero_init_metrics()

        assert slot_add.call_count == 2
        assert stream_add.call_count == 6
        # The hop counter is seeded after both of those, so a raise in the first
        # instrument must not cost the 39 series an alert actually binds to.
        with (
            patch.object(
                meters.generic_proxy_slot_rejected_total, "add", side_effect=RuntimeError("boom")
            ),
            patch.object(meters.generic_proxy_stream_outcome_total, "add"),
            patch.object(meters.generic_proxy_request_total, "add") as hop_add,
            patch.object(meters.metrics, "get_meter_provider", return_value=MagicMock()),
        ):
            meters.zero_init_metrics()
        assert hop_add.call_count == 39

    def test_zero_init_seeds_both_flush_outcomes(self):
        """The flush-health counter must exist at zero before the first flush.

        An OTel counter is absent until its first increment, and the alert on
        `outcome="error"` is the only thing that distinguishes a broken auth-path
        flush from a deployment with no traffic. Unseeded, it cannot bind at deploy
        -- and it only ever increments on the failure it was meant to have caught.
        """
        from auth_server.observability import meters

        with (
            patch.object(meters.auth_path_flush_total, "add") as flush_add,
            patch.object(meters.generic_proxy_slot_rejected_total, "add") as slot_add,
            patch.object(meters.generic_proxy_stream_outcome_total, "add") as stream_add,
            patch.object(meters.generic_proxy_request_total, "add") as hop_add,
            patch.object(meters.metrics, "get_meter_provider", return_value=MagicMock()),
        ):
            meters.zero_init_metrics()

        seeded = {c[0][1]["outcome"] for c in flush_add.call_args_list}
        assert seeded == set(meters.AUTH_PATH_FLUSH_OUTCOMES) == {"ok", "error"}
        assert all(c[0][0] == 0 for c in flush_add.call_args_list)
        # One seeding mechanism, not two: the flush outcomes join the existing
        # totals, which is the count the startup log line and docs report.
        assert (
            slot_add.call_count + stream_add.call_count + hop_add.call_count + flush_add.call_count
            == 49
        )


class TestServerNameIsBounded:
    """`server_name` is cardinality-bounded on every instrument that carries it.

    The gateway change makes `server_name` per-entity, and `tool_execution_total`
    shares it with a 16-bucket histogram (18 series per value) while
    `protocol_latency_ms` is a histogram too. Left unbounded, UUID-keyed custom
    records mint series per create-and-delete cycle with no `_other` collapse.

    Gateway routes should not reach these instruments at all -- the emission needs a
    JSON-RPC `method`, which comes from X-Body, and nginx clears the client-authored
    copy on generic locations (tests/unit/core/test_nginx_generic_proxy.py). This is
    the second line of defense, and it is what keeps the label honest if any future
    route forwards a body.

    Each test installs its own limiter with a small cap: the module-level limiter is
    process-global, so flooding the real one would spend the shared budget and leak
    into other tests.
    """

    @staticmethod
    def _keys(count: int) -> list[str]:
        return [
            f"rest-endpoint/rest-endpoint/{i:08d}-4336-4164-8fd3-b5d7e6bccb56" for i in range(count)
        ]

    @pytest.mark.asyncio
    async def test_tool_execution_server_name_collapses_past_the_cap(self, middleware):
        with (
            patch(
                "auth_server.metrics_middleware._label_limiter",
                LabelCardinalityLimiter(max_cardinality=3),
            ),
            patch("auth_server.metrics_middleware.tool_execution_total") as counter,
            patch("auth_server.metrics_middleware.tool_execution_duration_ms") as hist,
        ):
            for key in self._keys(20):
                await middleware._emit_tool_execution_metric(
                    tool_info={"method": "tools/call", "tool_name": "x"},
                    server_name=key,
                    success=True,
                    duration_ms=1.0,
                    user_hash="",
                )

        emitted = {c[0][1]["server_name"] for c in counter.add.call_args_list}
        assert "_other" in emitted
        assert len(emitted - {"_other"}) == 3
        # The histogram carries the same bounded value, not the raw key: it is the
        # instrument where an unbounded value costs 18 series instead of 1.
        assert {c[0][1]["server_name"] for c in hist.record.call_args_list} == emitted

    @pytest.mark.asyncio
    async def test_protocol_latency_server_name_collapses_past_the_cap(self, middleware):
        middleware.session_timings = {}
        with (
            patch(
                "auth_server.metrics_middleware._label_limiter",
                LabelCardinalityLimiter(max_cardinality=3),
            ),
            patch("auth_server.metrics_middleware.protocol_latency_ms") as hist,
        ):
            for key in self._keys(20):
                session_key = f"{key}:anonymous"
                middleware.session_timings[session_key] = {
                    "initialize": 1000.0,
                    "tools/list": 1001.0,
                }
                await middleware._emit_protocol_latency_metric(
                    session_key=session_key,
                    current_method="tools/list",
                    server_name=key,
                    user_hash="",
                    request_id="req_test",
                )

        emitted = {c[0][1]["server_name"] for c in hist.record.call_args_list}
        assert emitted, "no protocol-latency series recorded; the flow-step setup broke"
        assert "_other" in emitted
        assert len(emitted - {"_other"}) == 3
        # flow_step survives untouched: it is a server-set enum, not a bounded label.
        assert {c[0][1]["flow_step"] for c in hist.record.call_args_list} == {
            "initialize_to_tools_list"
        }

    @pytest.mark.asyncio
    async def test_a_single_real_server_name_keeps_full_fidelity(self, middleware):
        """Bounding must not degrade the common case: one value, emitted verbatim."""
        with (
            patch(
                "auth_server.metrics_middleware._label_limiter",
                LabelCardinalityLimiter(max_cardinality=150),
            ),
            patch("auth_server.metrics_middleware.tool_execution_total") as counter,
            patch("auth_server.metrics_middleware.tool_execution_duration_ms"),
        ):
            await middleware._emit_tool_execution_metric(
                tool_info={"method": "tools/call", "tool_name": "get_weather"},
                server_name="airegistry-tools",
                success=True,
                duration_ms=1.0,
                user_hash="",
            )

        assert counter.add.call_args[0][1]["server_name"] == "airegistry-tools"


def _validate_request(path: str = "/validate") -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": path,
            "query_string": b"",
            "headers": [],
            "client": ("10.0.0.1", 1234),
            "server": ("localhost", 80),
            "scheme": "http",
        }
    )


class TestBlankAuthMethod:
    """A blank `method` reads as `unknown` on the metrics, and nowhere else.

    server.py writes `X-Auth-Method: validation_result.get("method") or ""`, so on
    a validation result with no method the header is PRESENT and empty and
    dispatch's `.get(..., "unknown")` default never fires. Prometheus treats a
    blank label value as absent, so those requests drop out of `sum by (method)`
    entirely. The coercion therefore belongs at emission, not at the header.
    """

    @staticmethod
    async def _dispatch_blank(middleware):
        async def call_next(_request):
            return Response(status_code=200, headers={"X-Auth-Method": ""})

        with (
            patch("auth_server.metrics_middleware.auth_request_total") as counter,
            patch("auth_server.metrics_middleware.auth_request_duration_ms") as hist,
            patch("auth_server.metrics_middleware.record_emission_path"),
            patch("auth_server.metrics_middleware.record_auth_path"),
        ):
            response = await middleware.dispatch(_validate_request(), call_next)
            # Emission is fire-and-forget via create_task; let those tasks run.
            for _ in range(5):
                await asyncio.sleep(0)
        return response, counter, hist

    @pytest.mark.asyncio
    async def test_blank_auth_method_becomes_unknown_on_both_metrics(self, middleware):
        """One `otel_attrs` dict feeds both instruments, so both must read `unknown`."""
        _, counter, hist = await self._dispatch_blank(middleware)

        assert counter.add.call_args[0][1]["method"] == "unknown"
        assert hist.record.call_args[0][1]["method"] == "unknown"

    @pytest.mark.asyncio
    async def test_auth_method_response_header_unchanged(self, middleware):
        """The regression guard for the whole approach: the header is byte-identical.

        nginx forwards X-Auth-Method from the auth_request subrequest and downstream
        consumers read it. Repairing the header instead of the label would change a
        request contract to fix a metric.
        """
        response, _, _ = await self._dispatch_blank(middleware)

        assert response.headers["X-Auth-Method"] == ""

    @pytest.mark.asyncio
    async def test_histogram_still_carries_no_server_label(self, middleware):
        """The 1.30.0 contract survives the shared-dict coercion.

        `hist_attrs` is derived from the same `otel_attrs` the counter gets, so a
        change there is exactly where `server` would leak back onto 16 buckets.
        """
        _, counter, hist = await self._dispatch_blank(middleware)

        assert "server" not in hist.record.call_args[0][1]
        assert "server" in counter.add.call_args[0][1]


class TestAuthPathRecording:
    """Every /validate request is counted once; nothing else is counted at all."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("outcome", [200, 401, 403, 502, "raise"])
    async def test_record_called_once_per_request_for_every_outcome(self, middleware, outcome):
        """The count feeds the fleet's auth-path mix, so a missing request is a
        missing request in that mix. dispatch's `finally` is what makes it
        exhaustive across success, every failure status, and a raising handler --
        and the raise must still propagate untouched.
        """

        async def call_next(_request):
            if outcome == "raise":
                raise RuntimeError("validator exploded")
            headers = {"X-Auth-Method": "session_cookie"} if outcome == 200 else {}
            return Response(status_code=outcome, headers=headers)

        with (
            patch("auth_server.metrics_middleware.auth_request_total"),
            patch("auth_server.metrics_middleware.auth_request_duration_ms"),
            patch("auth_server.metrics_middleware.record_emission_path"),
            patch("auth_server.metrics_middleware.record_auth_path") as record,
        ):
            if outcome == "raise":
                with pytest.raises(RuntimeError, match="validator exploded"):
                    await middleware.dispatch(_validate_request(), call_next)
            else:
                await middleware.dispatch(_validate_request(), call_next)
            for _ in range(5):
                await asyncio.sleep(0)

        record.assert_called_once()
        # Only a 200 reports a path: the auth-server sets X-Auth-Method on success
        # only, so every failure files as `unknown`.
        assert record.call_args[0][0] == ("session_cookie" if outcome == 200 else "unknown")

    @pytest.mark.asyncio
    async def test_non_validate_path_records_nothing(self, middleware):
        """dispatch returns before any instrumentation for a non-/validate path."""

        async def call_next(_request):
            return Response(status_code=200, headers={"X-Auth-Method": "session_cookie"})

        with (
            patch("auth_server.metrics_middleware.auth_request_total") as counter,
            patch("auth_server.metrics_middleware.auth_request_duration_ms"),
            patch("auth_server.metrics_middleware.record_emission_path"),
            patch("auth_server.metrics_middleware.record_auth_path") as record,
        ):
            await middleware.dispatch(_validate_request(path="/health"), call_next)
            for _ in range(5):
                await asyncio.sleep(0)

        record.assert_not_called()
        counter.add.assert_not_called()
