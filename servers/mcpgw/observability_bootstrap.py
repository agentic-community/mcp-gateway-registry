"""OTel bootstrap and tool-level instrumentation for the mcpgw service (Issue #1122).

This module:

1. Starts a Prometheus exposition listener on :9464 so the in-cluster
   Prometheus can scrape mcpgw alongside registry and auth-server.
2. Provides a ``track_tool`` decorator that records invocations,
   latency, and success/failure per FastMCP tool. Apply it underneath
   ``@mcp.tool()`` on each tool function.

Mcpgw is a separate Docker image that does NOT include the registry
codebase, so this module is self-contained.
"""

from __future__ import annotations

import functools
import logging
import os
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


# Tool-level OTel instruments. Lazily initialized on first use so the
# helpful no-op behavior is preserved when the SDK isn't running.
_tool_invocations_counter: Any = None
_tool_duration_histogram: Any = None


def _get_tool_instruments() -> tuple[Any, Any] | tuple[None, None]:
    """Return (invocations Counter, duration Histogram), creating them lazily."""
    global _tool_invocations_counter, _tool_duration_histogram

    if _tool_invocations_counter is not None:
        return _tool_invocations_counter, _tool_duration_histogram

    try:
        from opentelemetry import metrics
    except ImportError:
        return None, None

    meter = metrics.get_meter("mcp-gateway-mcpgw")
    _tool_invocations_counter = meter.create_counter(
        name="mcpgw_registry_tool_invocations_total",
        description="FastMCP tool invocation count, labeled by tool and outcome",
        unit="1",
    )
    _tool_duration_histogram = meter.create_histogram(
        name="mcpgw_registry_tool_duration",
        description="FastMCP tool invocation duration",
        unit="ms",
    )
    return _tool_invocations_counter, _tool_duration_histogram


def track_tool(
    tool_name: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that records OTel metrics for an async FastMCP tool function.

    Apply directly underneath ``@mcp.tool()``. Records:

    - ``mcpgw_registry_tool_invocations_total{tool, success}`` Counter
      (``success="True"`` on a normal return, ``"False"`` on any raised
      exception).
    - ``mcpgw_registry_tool_duration_ms{tool, success}`` Histogram (per-call
      duration in milliseconds).

    Args:
        tool_name: Override the metric label. Defaults to the wrapped
            function's ``__name__``.

    Falls back to a transparent no-op when ``opentelemetry`` is not
    importable, so test environments that exclude OTel still run.
    """

    def _wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        label_name = tool_name or func.__name__

        @functools.wraps(func)
        async def _async_inner(*args: Any, **kwargs: Any) -> Any:
            counter, histogram = _get_tool_instruments()
            start = time.perf_counter()
            success = "True"
            try:
                return await func(*args, **kwargs)
            except Exception:
                success = "False"
                raise
            finally:
                duration_ms = (time.perf_counter() - start) * 1000.0
                if counter is not None:
                    counter.add(1, {"tool": label_name, "success": success})
                if histogram is not None:
                    histogram.record(duration_ms, {"tool": label_name, "success": success})

        return _async_inner

    return _wrap


def init_meter_provider_if_needed() -> None:
    """Bootstrap the OTel SDK + Prometheus exporter when applicable.

    No-op when:
    - ``OTEL_EXPORTER_PROMETHEUS_HOST`` is unset (operator hasn't opted in).
    - A real ``MeterProvider`` is already installed (e.g., by
      ``opentelemetry-instrument`` doing its job).
    """
    prom_host = os.getenv("OTEL_EXPORTER_PROMETHEUS_HOST", "").strip()
    if not prom_host:
        return

    try:
        from opentelemetry import metrics
    except ImportError:
        logger.debug("opentelemetry SDK not installed; skipping bootstrap")
        return

    current = metrics.get_meter_provider()
    current_name = type(current).__name__
    if "ProxyMeterProvider" not in current_name and "NoOp" not in current_name:
        return

    try:
        from opentelemetry.exporter.prometheus import PrometheusMetricReader
        from opentelemetry.sdk.metrics import MeterProvider
        from prometheus_client import start_http_server
    except ImportError as exc:
        logger.warning(
            "Cannot start Prometheus exporter: %s. Install "
            "opentelemetry-exporter-prometheus and prometheus-client.",
            exc,
        )
        return

    _port_str = os.getenv("OTEL_EXPORTER_PROMETHEUS_PORT", "9464")
    try:
        prom_port = int(_port_str)
    except ValueError:
        logger.warning(
            "Invalid OTEL_EXPORTER_PROMETHEUS_PORT=%r, falling back to 9464",
            _port_str,
        )
        prom_port = 9464
    try:
        start_http_server(port=prom_port, addr=prom_host)
        reader = PrometheusMetricReader()
        provider = MeterProvider(metric_readers=[reader])
        metrics.set_meter_provider(provider)
        logger.info(
            "Started OTel Prometheus exporter on %s:%d (provider=MeterProvider)",
            prom_host,
            prom_port,
        )
    except OSError as exc:
        logger.warning(
            "Could not start Prometheus exporter on %s:%d: %s",
            prom_host,
            prom_port,
            exc,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("OTel Prometheus exporter init failed: %s", exc)
