"""OpenTelemetry meter and instrument declarations for the auth server.

Mirrors the pattern in ``registry/observability/meters.py``: declare a single
global meter and the instruments it owns at module scope, import where
needed, increment via ``counter.add(value, attributes)``.

All Path-2 events that auth_server previously POSTed to metrics-service
(``auth_request``, ``tool_execution``, ``protocol_latency``) are migrated
here. Cardinality-risky dimensions (``user_hash``, ``request_id``,
``server_path``, ``session_key``) have been removed from the canonical
attribute set; see issue #1122 for the rationale.
"""

from __future__ import annotations

import logging
import os

from opentelemetry import metrics

logger = logging.getLogger(__name__)


def _init_meter_provider_if_needed() -> None:
    """Bootstrap the OTel SDK when ``opentelemetry-instrument`` did not.

    See registry/observability/meters.py for the full rationale. Mirrors
    the same logic for the auth-server process.
    """
    prom_host = os.getenv("OTEL_EXPORTER_PROMETHEUS_HOST", "").strip()
    if not prom_host:
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


_init_meter_provider_if_needed()
_meter = metrics.get_meter("mcp-auth-server")


# =============================================================================
# Authentication-request metrics
# =============================================================================

auth_request_total = _meter.create_counter(
    name="mcpgw_registry_auth_request_total",
    description=(
        "Authentication request count, labeled by success, method, server, and "
        "target_kind (a2a_agent | virtual_mcp_server | mcp_server | "
        "generic_proxy_skill | generic_proxy_agent | generic_proxy_custom | "
        "control_plane | unknown) for routing breakdown. For a gateway-proxied "
        "request `server` holds the entity's authz key, which is the string a "
        "server_access rule names, so a success=false series points at the rule "
        "to write"
    ),
    unit="1",
)

auth_request_duration_ms = _meter.create_histogram(
    name="mcpgw_registry_auth_request_duration",
    description=(
        "Authentication request duration. Carries no `server` label: /validate "
        "does the same work for every target, and a per-target label costs 18 "
        "series on this histogram against 1 on the counter. Group by target_kind"
    ),
    unit="ms",
)


# =============================================================================
# Tool-execution metrics (auth-side, with full client info from headers)
# =============================================================================

tool_execution_total = _meter.create_counter(
    name="mcpgw_registry_tool_execution_total",
    description="Tool execution count detected at the auth layer",
    unit="1",
)

tool_execution_duration_ms = _meter.create_histogram(
    name="tool_execution_duration",
    description="Tool execution duration",
    unit="ms",
)


# =============================================================================
# Protocol-latency metrics (time between MCP protocol stages)
# =============================================================================

protocol_latency_ms = _meter.create_histogram(
    name="mcpgw_registry_protocol_latency",
    description=(
        "Time between MCP protocol stages: initialize -> tools/list, "
        "tools/list -> tools/call, initialize -> tools/call"
    ),
    unit="ms",
)


# =============================================================================
# Token-mint metrics
# =============================================================================

token_mint_total = _meter.create_counter(
    name="mcpgw_registry_token_mint_total",
    description="Token mint count, labeled by kind, resource type, path, and outcome",
    unit="1",
)


# =============================================================================
# Redirect-validation metrics (open-redirect hardening, PR #1475 follow-up)
# =============================================================================

redirect_rejected_total = _meter.create_counter(
    name="mcpgw_registry_redirect_rejected_total",
    description=(
        "Login/logout redirect URIs rejected by the allowlist, labeled by flow "
        "(login | logout) and reason (backslash | protocol_relative | scheme | "
        "not_in_allowlist | cookie_domain | empty)"
    ),
    unit="1",
)


# =============================================================================
# Self-observability of the migration itself
# =============================================================================

_metrics_emission_path_counter = _meter.create_counter(
    name="mcpgw_registry_metrics_emission_path_total",
    description=(
        "Counts which emission path produced a metric. Helps operators verify "
        "the OTel migration: when METRICS_LEGACY_HTTP_POST=false the legacy "
        "count should be zero. Issue #1122."
    ),
    unit="1",
)


def record_emission_path(path: str) -> None:
    """Record that a metric was emitted via the given path.

    Args:
        path: Either ``"otel"`` or ``"legacy"``.
    """
    _metrics_emission_path_counter.add(1, {"path": path})


# =============================================================================
# Generic reverse-proxy (gateway-proxy-any-resource) hop metrics
# =============================================================================

generic_proxy_slot_rejected_total = _meter.create_counter(
    name="mcpgw_registry_generic_proxy_slot_rejected_total",
    description=(
        "Generic-proxy requests rejected with 503 because a concurrency slot "
        "could not be acquired within the acquire timeout, labeled by pool "
        "(buffered | stream). A non-zero rate means the pool is saturated."
    ),
    unit="1",
)

generic_proxy_stream_outcome_total = _meter.create_counter(
    name="mcpgw_registry_generic_proxy_stream_outcome_total",
    description=(
        "Generic-proxy streaming request outcomes, labeled by outcome "
        "(started | completed | duration_timeout | byte_cap | upstream_error | "
        "client_closed). "
        "duration_timeout/byte_cap track the new stream ceilings."
    ),
    unit="1",
)


def record_generic_proxy_slot_rejected(pool: str) -> None:
    """Record a generic-proxy capacity rejection (503) for the given pool."""
    try:
        generic_proxy_slot_rejected_total.add(1, {"pool": pool})
    except Exception:  # pragma: no cover - metrics must never break the hop
        pass


def record_generic_proxy_stream_outcome(outcome: str) -> None:
    """Record a start/terminal outcome for a streaming generic-proxy request."""
    try:
        generic_proxy_stream_outcome_total.add(1, {"outcome": outcome})
    except Exception:  # pragma: no cover - metrics must never break the hop
        pass


# Known label values for the generic-proxy counters. An OTel counter reads as
# ABSENT until its first increment, so a fresh deployment shows an empty panel and
# a rate() alert that can never fire -- including when it should. Seeding each
# combination with zero materializes the series without changing any total.
# Verified against this repo's pinned SDK: an add(0)-only counter does export
# through PrometheusMetricReader.
_SLOT_POOLS: tuple[str, ...] = ("buffered", "stream")
_STREAM_OUTCOMES: tuple[str, ...] = (
    "started",
    "completed",
    "duration_timeout",
    "byte_cap",
    "upstream_error",
    "client_closed",
)


def zero_init_generic_proxy_metrics() -> None:
    """Seed every known generic-proxy label combination with zero.

    Each add() gets its own try, so one failure cannot skip the rest.

    Logs when the SDK provider was never installed. _init_meter_provider_if_needed
    returns early unless OTEL_EXPORTER_PROMETHEUS_HOST is set, and a failed
    start_http_server leaves a proxy provider, so in both cases every add(0) is
    discarded and an operator would otherwise see no signal that seeding did
    nothing.
    """
    provider = metrics.get_meter_provider()
    if type(provider).__name__ in ("_ProxyMeterProvider", "NoOpMeterProvider"):
        logger.info(
            "zero-init skipped: meter provider is %s, so no series were seeded "
            "(OTEL_EXPORTER_PROMETHEUS_HOST unset or exporter failed to start)",
            type(provider).__name__,
        )
        return

    seeds = [(generic_proxy_slot_rejected_total, {"pool": pool}) for pool in _SLOT_POOLS]
    seeds += [
        (generic_proxy_stream_outcome_total, {"outcome": outcome}) for outcome in _STREAM_OUTCOMES
    ]
    seeded = 0
    for instrument, attrs in seeds:
        try:
            instrument.add(0, attrs)
            seeded += 1
        except Exception:  # pragma: no cover - metrics must never break startup
            pass
    logger.info("zero-init seeded %d/%d generic-proxy series", seeded, len(seeds))


# =============================================================================
# Public helpers
# =============================================================================


def is_otel_enabled() -> bool:
    """Return True when OTel SDK is configured to export metrics."""
    return bool(os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip())
