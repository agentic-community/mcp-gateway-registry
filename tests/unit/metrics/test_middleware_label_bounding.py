"""Unit tests for cardinality bounding on the auth-server OTel emission path.

The auth server emits tool-execution metrics directly to an in-process
Prometheus instrument (the default path when ``METRICS_LEGACY_HTTP_POST`` is
off), bypassing the metrics-service processor. The request-derived attributes
(``tool_name``/``method``/``client_name``/``client_version``) come verbatim from
the JSON-RPC body / clientInfo, so a client sending randomized values would
otherwise explode the Prometheus time-series count -- a DoS. These tests prove
those attributes are charset-normalized and distinct-value-capped before they
reach the instrument, while server-set attributes pass through untouched.
"""

from __future__ import annotations

from unittest.mock import patch

from auth_server.metrics_middleware import (
    _TOOL_EXECUTION_BOUNDED_ATTRS,
    AuthMetricsMiddleware,
)


def _middleware() -> AuthMetricsMiddleware:
    return AuthMetricsMiddleware(app=lambda *a, **k: None)


def _tool_info(method: str, tool_name: str, client_name: str, client_version: str) -> dict:
    return {
        "method": method,
        "tool_name": tool_name,
        "client_info": {"name": client_name, "version": client_version},
    }


class TestToolExecutionLabelBounding:
    async def test_flood_of_tool_names_collapses_to_overflow(self) -> None:
        mw = _middleware()
        mw.legacy_http_post_enabled = False
        emitted_tool_names = set()

        with (
            patch("auth_server.metrics_middleware.tool_execution_total") as counter,
            patch("auth_server.metrics_middleware.tool_execution_duration_ms"),
            patch("auth_server.metrics_middleware.record_emission_path"),
        ):
            # Use the module-level limiter with a small cap for a fast assertion.
            from auth_server import metrics_middleware as mm

            mm._label_limiter = mm.LabelCardinalityLimiter(max_cardinality=3)
            for i in range(200):
                await mw._emit_tool_execution_metric(
                    tool_info=_tool_info("tools/call", f"tool{i}", f"client{i}", f"{i}.0"),
                    server_name="mcpgw",
                    success=True,
                    duration_ms=1.0,
                    user_hash="u",
                )
            for call in counter.add.call_args_list:
                attrs = call.args[1]
                emitted_tool_names.add(attrs["tool_name"])
                # server_name is registry-derived and never bucketed.
                assert attrs["server_name"] == "mcpgw"

        assert "_other" in emitted_tool_names
        # 3 admitted distinct values + the overflow bucket.
        assert len(emitted_tool_names) <= 4

    async def test_illegal_chars_normalized_in_attrs(self) -> None:
        mw = _middleware()
        mw.legacy_http_post_enabled = False

        with (
            patch("auth_server.metrics_middleware.tool_execution_total") as counter,
            patch("auth_server.metrics_middleware.tool_execution_duration_ms"),
            patch("auth_server.metrics_middleware.record_emission_path"),
        ):
            await mw._emit_tool_execution_metric(
                tool_info=_tool_info("tools/call", "evil name\n\r x", "c", "1.0"),
                server_name="mcpgw",
                success=True,
                duration_ms=1.0,
                user_hash="u",
            )
            attrs = counter.add.call_args.args[1]
            assert "\n" not in attrs["tool_name"]
            assert "\r" not in attrs["tool_name"]
            assert " " not in attrs["tool_name"]

    async def test_legitimate_method_with_slash_preserved(self) -> None:
        mw = _middleware()
        mw.legacy_http_post_enabled = False

        with (
            patch("auth_server.metrics_middleware.tool_execution_total") as counter,
            patch("auth_server.metrics_middleware.tool_execution_duration_ms"),
            patch("auth_server.metrics_middleware.record_emission_path"),
        ):
            from auth_server import metrics_middleware as mm

            mm._label_limiter = mm.LabelCardinalityLimiter(max_cardinality=100)
            await mw._emit_tool_execution_metric(
                tool_info=_tool_info("tools/call", "get_weather", "cursor-ide", "1.2.3"),
                server_name="mcpgw",
                success=True,
                duration_ms=1.0,
                user_hash="u",
            )
            attrs = counter.add.call_args.args[1]
            assert attrs["method"] == "tools/call"
            assert attrs["tool_name"] == "get_weather"
            assert attrs["client_name"] == "cursor-ide"
            assert attrs["client_version"] == "1.2.3"


def test_bounded_attrs_set_covers_request_derived_keys() -> None:
    # Guard against a future edit re-adding a raw request-derived attr without
    # bounding it. Every key here is copied verbatim from the JSON-RPC body.
    assert _TOOL_EXECUTION_BOUNDED_ATTRS == frozenset(
        {"tool_name", "method", "client_name", "client_version"}
    )
