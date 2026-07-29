"""Tests for attacker-influenced label cardinality bounding.

Attacker-controlled request data (tool names, client names, search queries)
flows into Prometheus labels. Without a bound, a client sending randomized
values explodes the time-series count and exhausts Prometheus -- a DoS. These
tests prove the processor charset-normalizes, length-caps, and cardinality-caps
the risky dimensions while leaving legitimate internal telemetry untouched.
"""

import pytest
from unittest.mock import MagicMock, patch

from app.core.processor import (
    _CardinalityLimiter,
    _MAX_LABEL_CARDINALITY,
    _MAX_LABEL_LENGTH,
    _OVERFLOW_LABEL_VALUE,
    _EMPTY_LABEL_VALUE,
    _BOUNDED_LABEL_DIMENSIONS,
    _SAFE_LABEL_CHARS,
    _normalize_label_value,
    MetricsProcessor,
)
from app.core.models import MetricType, Metric


class TestNormalizeLabelValue:
    """Test the boolean/Prometheus normalization helper."""

    def test_bool_true_lowercased(self):
        assert _normalize_label_value(True) == "true"

    def test_bool_false_lowercased(self):
        assert _normalize_label_value(False) == "false"

    def test_string_passthrough(self):
        assert _normalize_label_value("calculator") == "calculator"

    def test_int_stringified(self):
        assert _normalize_label_value(200) == "200"


class TestCardinalityLimiterCharset:
    """Test charset + length normalization."""

    def test_legitimate_value_unchanged(self):
        limiter = _CardinalityLimiter()
        assert limiter.bound("tool_name", "get_weather-v2.1") == "get_weather-v2.1"

    def test_illegal_chars_replaced(self):
        limiter = _CardinalityLimiter()
        # spaces, semicolons, newlines and other unsafe chars collapse to
        # underscore. Slash is deliberately preserved (legitimate method/tool
        # separator) so it survives here.
        result = limiter.bound("tool_name", "rm -rf /; drop\ntable")
        assert result == "rm_-rf_/__drop_table"
        assert " " not in result
        assert ";" not in result
        assert "\n" not in result

    def test_oversized_value_truncated(self):
        limiter = _CardinalityLimiter()
        oversized = "a" * (_MAX_LABEL_LENGTH + 100)
        result = limiter.bound("client_name", oversized)
        assert len(result) == _MAX_LABEL_LENGTH
        assert result == "a" * _MAX_LABEL_LENGTH

    def test_all_illegal_maps_to_empty_sentinel(self):
        limiter = _CardinalityLimiter()
        # A value that is empty after char normalization uses the empty sentinel
        assert limiter.bound("tool_name", "") == _EMPTY_LABEL_VALUE

    def test_configurable_length_cap(self):
        limiter = _CardinalityLimiter(max_length=4)
        assert limiter.bound("tool_name", "abcdefgh") == "abcd"


class TestCardinalityLimiterCap:
    """Test the distinct-value (cardinality) cap and overflow bucket."""

    def test_flood_collapses_to_overflow(self):
        limiter = _CardinalityLimiter(max_cardinality=5)
        emitted = {limiter.bound("tool_name", f"tool{i}") for i in range(1000)}
        # Only up to the cap distinct real values, plus the overflow sentinel
        assert _OVERFLOW_LABEL_VALUE in emitted
        real_values = emitted - {_OVERFLOW_LABEL_VALUE}
        assert len(real_values) == 5

    def test_distinct_count_bounds_emitted_values(self):
        limiter = _CardinalityLimiter(max_cardinality=10)
        for i in range(500):
            limiter.bound("client_name", f"client{i}")
        # The tracked set never exceeds the cap
        assert len(limiter._seen["client_name"]) == 10

    def test_known_value_after_cap_still_passes(self):
        limiter = _CardinalityLimiter(max_cardinality=2)
        assert limiter.bound("tool_name", "known_a") == "known_a"
        assert limiter.bound("tool_name", "known_b") == "known_b"
        # cap reached; a new value overflows
        assert limiter.bound("tool_name", "known_c") == _OVERFLOW_LABEL_VALUE
        # but an already-admitted value keeps passing unchanged
        assert limiter.bound("tool_name", "known_a") == "known_a"

    def test_cap_is_per_label_name(self):
        limiter = _CardinalityLimiter(max_cardinality=1)
        assert limiter.bound("tool_name", "t1") == "t1"
        # different label name has its own budget
        assert limiter.bound("client_name", "c1") == "c1"

    def test_overflow_sentinel_not_consuming_budget(self):
        limiter = _CardinalityLimiter(max_cardinality=1)
        assert limiter.bound("tool_name", "real") == "real"
        # overflow many times; the sentinel itself must not evict "real"
        for _ in range(50):
            assert limiter.bound("tool_name", "x" + str(_)) == _OVERFLOW_LABEL_VALUE
        assert limiter.bound("tool_name", "real") == "real"

    def test_default_cardinality_default(self):
        # sanity: shipped defaults are bounded, not disabled
        assert _MAX_LABEL_CARDINALITY > 0
        assert _MAX_LABEL_LENGTH > 0

    def test_semantics_match_registry_sibling_limiter(self):
        # This limiter is an intentional sibling of
        # registry/observability/label_bounding.py::LabelCardinalityLimiter,
        # duplicated across the metrics-service deployable boundary (that service
        # cannot import `registry`). The charset, sentinels, and default length
        # MUST stay in sync between the two. These literal pins fail loudly if
        # one side drifts, forcing a matching update to the other (whose test
        # test_sibling_limiter_semantics_are_pinned holds the mirror-image pins).
        assert _OVERFLOW_LABEL_VALUE == "_other"
        assert _EMPTY_LABEL_VALUE == "_unset"
        assert _SAFE_LABEL_CHARS.pattern == r"[^A-Za-z0-9\-_.:/]"
        # Default length pin (cardinality default is env-derived; length is the
        # charset-normalization constant that must match the sibling).
        limiter = _CardinalityLimiter()
        assert limiter._max_length == 64


class TestBoundDimensions:
    """Test which dimensions get bounded and which are left alone."""

    def test_bounded_dimensions_are_the_attacker_influenced_ones(self):
        assert _BOUNDED_LABEL_DIMENSIONS == frozenset(
            {"tool_name", "client_name", "client_version", "query", "method"}
        )

    def test_internal_dimensions_untouched(self):
        limiter = _CardinalityLimiter(max_cardinality=1)
        # server_name/success are internal (routing/derived) and not bounded
        # even under a tiny cap. method is attacker-controlled and IS bounded
        # (covered separately below).
        labels = limiter.bound_dimensions(
            "tool_execution",
            {
                "server_name": "mcpgw",
                "success": True,
                "tool_name": "calc",
            },
        )
        assert labels["server_name"] == "mcpgw"
        assert labels["success"] == "true"
        assert labels["tool_name"] == "calc"

    def test_legitimate_method_passes_through_with_slash(self):
        # A slash is preserved so real JSON-RPC methods survive normalization.
        limiter = _CardinalityLimiter(max_cardinality=10)
        labels = limiter.bound_dimensions("tool_execution", {"method": "tools/call"})
        assert labels["method"] == "tools/call"

    def test_method_cardinality_is_bounded(self):
        # method comes verbatim from the JSON-RPC body; a client flooding
        # randomized method names must collapse to the overflow sentinel.
        limiter = _CardinalityLimiter(max_cardinality=2)
        seen = set()
        for i in range(100):
            labels = limiter.bound_dimensions("tool_execution", {"method": f"m{i}"})
            seen.add(labels["method"])
        assert "_other" in seen
        assert len(seen) <= 3  # 2 admitted + the overflow bucket

    def test_internal_dimension_not_cardinality_capped(self):
        # server_name is derived from the URI path (registry-controlled), so it
        # must NOT be capped -- flooding it should never produce the sentinel.
        limiter = _CardinalityLimiter(max_cardinality=2)
        for i in range(100):
            labels = limiter.bound_dimensions("tool_execution", {"server_name": f"srv{i}"})
            assert labels["server_name"] == f"srv{i}"

    def test_bounded_dimension_flood_collapses(self):
        limiter = _CardinalityLimiter(max_cardinality=3)
        seen = set()
        for i in range(200):
            labels = limiter.bound_dimensions("tool_execution", {"tool_name": f"tool{i}"})
            seen.add(labels["tool_name"])
        assert _OVERFLOW_LABEL_VALUE in seen
        assert len(seen - {_OVERFLOW_LABEL_VALUE}) == 3

    def test_unknown_dimension_key_is_dropped(self):
        # A direct authenticated POST cannot invent arbitrary dimension keys:
        # keys outside the metric type's allowlist are dropped fail-closed so the
        # Prometheus label-NAME axis cannot be flooded.
        limiter = _CardinalityLimiter(max_cardinality=100)
        labels = limiter.bound_dimensions(
            "tool_execution",
            {"tool_name": "calc", "rogue_key_0": "x", "another_bogus": "y"},
        )
        assert labels == {"tool_name": "calc"}

    def test_unknown_key_flood_does_not_create_labels(self):
        # Rotating key NAMES (the label-name cardinality bomb) never survive.
        limiter = _CardinalityLimiter(max_cardinality=100)
        label_names = set()
        for i in range(500):
            labels = limiter.bound_dimensions("tool_execution", {f"rot_key_{i}": "v"})
            label_names.update(labels.keys())
        assert label_names == set()

    def test_unlabelled_metric_type_emits_no_dimension_labels(self):
        # registry_operation / custom have no labelled _emit_to_otel branch, so
        # their (possibly arbitrary) dimension keys must never become labels.
        limiter = _CardinalityLimiter(max_cardinality=100)
        assert (
            limiter.bound_dimensions("registry_operation", {"resource_id": "r", "user_id": "u"})
            == {}
        )
        assert limiter.bound_dimensions("custom", {"metric_name": "x", "whatever": "y"}) == {}
        assert limiter.bound_dimensions("totally_unknown_type", {"a": "b"}) == {}

    def test_legitimate_emitter_dimensions_are_preserved(self):
        # Regression guard: every dimension a real emitter sends for a labelled
        # metric type must survive bound_dimensions (i.e. be on the allowlist).
        # A dropped key here is a silent telemetry loss. The expected sets are
        # the union of what the nginx-lua flush and the registry/auth metrics
        # clients actually send per type.
        limiter = _CardinalityLimiter(max_cardinality=1000)
        emitter_dimensions = {
            "tool_execution": {
                "method": "tools/call",
                "server_name": "srv",
                "server_path": "/srv/",
                "tool_name": "calc",
                "client_name": "cursor",
                "client_version": "1.2.3",
                "success": True,
                "user_hash": "abc123",
            },
            "auth_request": {
                "success": True,
                "method": "bearer",
                "server": "srv",
                "target_kind": "mcp_server",
                "user_hash": "abc123",
            },
            "tool_discovery": {
                "query": "weather",
                "results_count": 3,
                "top_k_services": 5,
                "top_n_tools": 10,
            },
            "protocol_latency": {
                "flow_step": "initialize_to_tools_list",
                "server_name": "srv",
                "user_hash": "abc123",
                "session_key": "sess",
            },
            "health_check": {"endpoint": "/health", "status_code": 200, "healthy": True},
        }
        for metric_type, dims in emitter_dimensions.items():
            labels = limiter.bound_dimensions(metric_type, dims)
            assert set(labels.keys()) == set(dims.keys()), (
                f"{metric_type}: dropped legitimate dimension(s) "
                f"{set(dims.keys()) - set(labels.keys())}"
            )


class TestEmitToOtelBounding:
    """Test bounding is applied end-to-end through _emit_to_otel."""

    @patch("app.core.processor.MetricsStorage")
    async def test_tool_name_flood_bounded_in_labels(self, mock_storage_class):
        processor = MetricsProcessor()
        processor._cardinality_limiter = _CardinalityLimiter(max_cardinality=4)
        processor.otel = MagicMock()
        processor.otel.tool_counter = MagicMock()
        processor.otel.tool_histogram = MagicMock()

        emitted_tool_names = set()
        for i in range(500):
            metric = Metric(
                type=MetricType.TOOL_EXECUTION,
                value=1.0,
                duration_ms=10.0,
                dimensions={
                    "tool_name": f"tool{i}",
                    "client_name": f"client{i}",
                    "server_name": "mcpgw",
                    "success": True,
                },
            )
            await processor._emit_to_otel(metric, "mcpgw-service")

        for call in processor.otel.tool_counter.add.call_args_list:
            labels = call.args[1]
            emitted_tool_names.add(labels["tool_name"])
            # server_name is internal and never bucketed
            assert labels["server_name"] == "mcpgw"

        assert _OVERFLOW_LABEL_VALUE in emitted_tool_names
        assert len(emitted_tool_names - {_OVERFLOW_LABEL_VALUE}) == 4

    @patch("app.core.processor.MetricsStorage")
    async def test_illegal_client_name_normalized_in_labels(self, mock_storage_class):
        processor = MetricsProcessor()
        processor.otel = MagicMock()
        processor.otel.tool_counter = MagicMock()
        processor.otel.tool_histogram = MagicMock()

        metric = Metric(
            type=MetricType.TOOL_EXECUTION,
            value=1.0,
            duration_ms=10.0,
            dimensions={"client_name": "evil client\n\r injection", "success": True},
        )
        await processor._emit_to_otel(metric, "mcpgw-service")

        labels = processor.otel.tool_counter.add.call_args.args[1]
        assert "\n" not in labels["client_name"]
        assert "\r" not in labels["client_name"]
        assert " " not in labels["client_name"]

    @patch("app.core.processor.MetricsStorage")
    async def test_legitimate_values_pass_through(self, mock_storage_class):
        processor = MetricsProcessor()
        processor.otel = MagicMock()
        processor.otel.tool_counter = MagicMock()
        processor.otel.tool_histogram = MagicMock()

        metric = Metric(
            type=MetricType.TOOL_EXECUTION,
            value=1.0,
            duration_ms=10.0,
            dimensions={
                "tool_name": "get_weather",
                "client_name": "cursor-ide",
                "server_name": "mcpgw",
                "success": True,
            },
        )
        await processor._emit_to_otel(metric, "mcpgw-service")

        labels = processor.otel.tool_counter.add.call_args.args[1]
        assert labels["tool_name"] == "get_weather"
        assert labels["client_name"] == "cursor-ide"
        assert labels["server_name"] == "mcpgw"
        assert labels["success"] == "true"
        assert labels["service"] == "mcpgw-service"
        assert labels["metric_type"] == "tool_execution"
