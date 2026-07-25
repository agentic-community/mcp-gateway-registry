"""OpenTelemetry meter and instrument declarations for the registry.

This module owns the single global meter used by the registry process. Every
in-process metric (counters, up/down counters, histograms, observable gauges)
is declared here at module scope and imported wherever it's incremented.

Why module scope: the OTel SDK lazily initializes the global MeterProvider on
first call to ``opentelemetry.metrics.get_meter_provider()``. When the
``opentelemetry-instrument`` wrapper has bootstrapped uvicorn (i.e.,
``OTEL_EXPORTER_OTLP_ENDPOINT`` is set), the provider is real and instruments
emit. When unset, ``get_meter_provider()`` returns a NoOp provider and
instruments become no-ops; this is safe and zero-cost.

Migration shape (issue #1122):
- Path-2 events (replaces ``MetricsClient.emit_*``): ``registry_operation_*``,
  ``tool_discovery_*``, ``tool_execution_*``, ``health_check_*``.
- Path-3 in-process counters (migrated from ``prometheus_client.Counter`` /
  ``Gauge`` declarations across registry/core/metrics.py, registry/auth/*.py,
  registry/api/m2m_management_routes.py): preserved metric names and label
  semantics, exposed via the OTel exporter pipeline.

The legacy compatibility shim ``_CounterAdapter`` lives in
``registry.observability._compat``. Existing call sites that use
``METRIC.labels(...).inc()`` continue to work unchanged via the adapter.
New code should call the OTel instruments directly via
``counter.add(value, attributes)``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from opentelemetry import metrics

from registry.observability._compat import (
    _CounterAdapter,
    _HistogramAdapter,
)

logger = logging.getLogger(__name__)


def _init_meter_provider_if_needed() -> None:
    """Bootstrap the OTel SDK when ``opentelemetry-instrument`` did not.

    Why this exists: when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is unset, the
    docker entrypoint does NOT prefix uvicorn with ``opentelemetry-instrument``,
    so the SDK is never auto-bootstrapped. Without a bootstrap, ``metrics.
    get_meter_provider()`` returns a ``_ProxyMeterProvider`` and no exporter
    runs. The Prometheus exporter env vars (``OTEL_EXPORTER_PROMETHEUS_HOST``
    / ``_PORT``) are read by the SDK only when it initializes.

    This helper covers the OTLP-disabled-but-Prometheus-pull-enabled case:
    if the user has set ``OTEL_EXPORTER_PROMETHEUS_HOST``, install a
    ``MeterProvider`` with a ``PrometheusMetricReader`` so the
    :9464/metrics listener actually starts and the in-process counters
    become scrape-able.

    No-op when:
    - ``OTEL_EXPORTER_PROMETHEUS_HOST`` is unset (operator hasn't opted in).
    - A real ``MeterProvider`` is already installed (auto-instrumentation
      did its job).
    """
    prom_host = os.getenv("OTEL_EXPORTER_PROMETHEUS_HOST", "").strip()
    if not prom_host:
        return

    current = metrics.get_meter_provider()
    current_name = type(current).__name__
    # If a real SDK MeterProvider is already installed (e.g., by
    # opentelemetry-instrument), don't fight it.
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
        # The exporter relies on prometheus_client's default REGISTRY.
        # Start the HTTP server first so that the listener is bound by the
        # time the reader registers.
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
        # Most likely cause: port already in use (e.g., a re-import in tests
        # or a second process binding). Don't crash the app.
        logger.warning(
            "Could not start Prometheus exporter on %s:%d: %s",
            prom_host,
            prom_port,
            exc,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("OTel Prometheus exporter init failed: %s", exc)


_init_meter_provider_if_needed()
_meter = metrics.get_meter("mcp-gateway-registry")


# =============================================================================
# Path-2 events (replaces MetricsClient.emit_*)
#
# These instruments capture custom application events that were previously
# POSTed as JSON to metrics-service:8890. The dimensions are deliberately
# pruned vs the legacy payload to control time-series cardinality:
#
# - ``user_hash``: removed (per-user labels create unbounded series on large
#   deployments). The user identity is still available in span attributes
#   via auto-instrumentation for per-request debugging.
# - ``request_id``: removed (already a span attribute).
# - ``query`` text (tool_discovery): removed (unbounded). Log it instead.
# - ``resource_id`` / ``user_id`` (registry_operation): removed (unbounded).
# =============================================================================

registry_operation_total = _meter.create_counter(
    name="mcpgw_registry_operation_total",
    description="Registry API operations (read/create/update/delete/list/search)",
    unit="1",
)

registry_operation_duration_ms = _meter.create_histogram(
    name="mcpgw_registry_operation_duration",
    description="Registry API operation duration",
    unit="ms",
)

tool_discovery_total = _meter.create_counter(
    name="mcpgw_registry_tool_discovery_total",
    description="Semantic search calls",
    unit="1",
)

tool_discovery_duration_ms = _meter.create_histogram(
    name="mcpgw_registry_tool_discovery_duration",
    description="Semantic search duration",
    unit="ms",
)

tool_execution_total = _meter.create_counter(
    name="mcpgw_registry_tool_execution_total",
    description="Tool execution count (registry side)",
    unit="1",
)

health_check_total = _meter.create_counter(
    name="health_check_total",
    description="Health check probe count",
    unit="1",
)


# =============================================================================
# Rate limiting (issue #295)
#
# Bounded labels only: axis (caller/target), entity_type (small allowlist),
# window_seconds (distinct configured windows), backend/op. NEVER the subject
# name (user/client/server/tool) -- that is unbounded. Per-entity-name
# attribution lives in the WARNING log and the /api/rate-limits/status endpoint.
# =============================================================================

rate_limit_throttled_total = _meter.create_counter(
    name="mcpgw_rate_limit_throttled_total",
    description="Times an axis entity was throttled (a gate denied a request)",
    unit="1",
)

rate_limit_checks_total = _meter.create_counter(
    name="mcpgw_rate_limit_checks_total",
    description="Total rate-limit gate checks (denominator for a throttle rate)",
    unit="1",
)

rate_limit_errors_total = _meter.create_counter(
    name="mcpgw_rate_limit_errors_total",
    description="Rate-limit backend errors (fail-open events)",
    unit="1",
)

rate_limit_backend_duration_ms = _meter.create_histogram(
    name="mcpgw_rate_limit_backend_duration",
    description="Per-op latency of the rate-limit counter-store round trip",
    unit="ms",
)

rate_limit_quarantine_denied_total = _meter.create_counter(
    name="mcpgw_rate_limit_quarantine_denied_total",
    description="Requests dropped because a caller or target is quarantined (kill-switch)",
    unit="1",
)


def _quarantine_members_callback(options: Any) -> Any:
    """ObservableGauge: current member count of each reserved quarantine group.

    Runs on the metrics-export cycle (NOT the request path). Each replica reports
    the same DB-backed ``count_documents`` per reserved group, so the gauge is
    correct across replicas. Bounded labels (only the fixed group name).
    """
    try:
        from registry.rate_limiting.memberships_repository import (
            get_shared_memberships_repo,
        )
        from registry.rate_limiting.models import (
            QUARANTINE_CALLER_GROUP,
            QUARANTINE_TARGET_GROUP,
        )

        repo = get_shared_memberships_repo()
        for group in (QUARANTINE_CALLER_GROUP, QUARANTINE_TARGET_GROUP):
            count = repo.count_group_members_cached(group)
            yield metrics.Observation(count, {"group": group})
    except Exception as exc:  # pragma: no cover - defensive; never break export
        logger.debug(f"quarantine_members callback failed: {exc}")
        return


_meter.create_observable_gauge(
    name="mcpgw_rate_limit_quarantine_members",
    callbacks=[_quarantine_members_callback],
    description="Current members in each rate-limit quarantine (kill-switch) group",
    unit="1",
)


# =============================================================================
# Path-3 in-process metrics
#
# Migrated from ``prometheus_client.Counter`` / ``Gauge`` declarations across
# registry/core/metrics.py, registry/auth/routes.py, registry/auth/session_store.py,
# registry/api/m2m_management_routes.py.
#
# Names are preserved verbatim so existing Grafana dashboards and PromQL alerts
# continue to function. Each instrument is wrapped by an adapter so call sites
# that use ``METRIC.labels(...).inc()`` keep working without modification.
# =============================================================================

# Configuration viewer metrics (registry/core/metrics.py:6,12)
_config_view_requests_counter = _meter.create_counter(
    name="mcpgw_registry_config_view_requests_total",
    description="Configuration view requests",
    unit="1",
)
config_view_requests_total = _CounterAdapter(_config_view_requests_counter)

_config_export_requests_counter = _meter.create_counter(
    name="mcpgw_registry_config_export_requests_total",
    description="Configuration export requests",
    unit="1",
)
config_export_requests_total = _CounterAdapter(_config_export_requests_counter)

# Outbound registration-webhook delivery health (Issue #1330).
# Labels: event_type (registration|update|deletion|scan_complete),
#         outcome (success|timeout|error|skipped_no_url).
_webhook_send_counter = _meter.create_counter(
    name="mcpgw_registry_webhook_send_total",
    description="Outbound registration webhook deliveries by event type and outcome",
    unit="1",
)
webhook_send_total = _CounterAdapter(_webhook_send_counter)

# Registrations rejected because REGISTRATION_ENFORCED_STATUS was set and the
# request's status did not match (Issue #1330). Labels: registration_type
# (server|agent|skill).
_registration_status_rejected_counter = _meter.create_counter(
    name="mcpgw_registry_registration_status_rejected_total",
    description="Registrations rejected due to enforced-status mismatch (4xx)",
    unit="1",
)
registration_status_rejected_total = _CounterAdapter(_registration_status_rejected_counter)


# Caller-supplied asset id metrics (#1276). Labels: asset_type (server|agent|skill).
_asset_id_supplied_counter = _meter.create_counter(
    name="registry_asset_id_supplied_total",
    description="Registrations where a caller-supplied asset id was honored",
    unit="1",
)
registry_asset_id_supplied_total = _CounterAdapter(_asset_id_supplied_counter)

_asset_id_conflict_counter = _meter.create_counter(
    name="registry_asset_id_conflict_total",
    description="Asset registrations rejected due to an id collision (409)",
    unit="1",
)
registry_asset_id_conflict_total = _CounterAdapter(_asset_id_conflict_counter)

_asset_id_federation_conflict_counter = _meter.create_counter(
    name="registry_asset_id_federation_conflict_total",
    description="Federated assets skipped due to a local id collision",
    unit="1",
)
registry_asset_id_federation_conflict_total = _CounterAdapter(_asset_id_federation_conflict_counter)

_asset_id_index_build_failed_counter = _meter.create_counter(
    name="registry_asset_id_index_build_failed_total",
    description=(
        "Unique id index build failures. When this is non-zero the registry is "
        "running WITHOUT the DB-level uniqueness guarantee and relies on the racy "
        "service-layer pre-check; alert on it."
    ),
    unit="1",
)
registry_asset_id_index_build_failed_total = _CounterAdapter(_asset_id_index_build_failed_counter)


# Deployment mode info (registry/core/metrics.py:19)
#
# Originally a Prometheus Gauge with .labels(...).set(1) called once at startup.
# Migrated to an ObservableGauge whose callback returns the current settings
# values. Callbacks fire on every collection cycle; the lookup is cheap (one
# attribute access on the imported ``settings`` singleton). Importing inside
# the callback avoids a startup-time circular import (settings -> meters ->
# settings) when the module is loaded eagerly.


def _deployment_mode_callback(options: Any) -> Any:
    """ObservableGauge callback: emit a value of 1 with current deployment-mode attributes."""
    try:
        from registry.core.config import settings

        yield metrics.Observation(
            1,
            {
                "deployment_mode": str(getattr(settings, "deployment_mode", "unknown")),
                "registry_mode": str(getattr(settings, "registry_mode", "unknown")),
            },
        )
    except Exception as exc:  # pragma: no cover - defensive only
        logger.debug("deployment_mode observable callback failed: %s", exc)
        return


_meter.create_observable_gauge(
    name="mcpgw_registry_deployment_mode_info",
    callbacks=[_deployment_mode_callback],
    description="Current deployment mode configuration (observed each export cycle)",
    unit="1",
)


# Nginx-related counters (registry/core/metrics.py:26,36)
_nginx_updates_skipped_counter = _meter.create_counter(
    name="mcpgw_registry_nginx_updates_skipped_total",
    description="Number of nginx updates skipped due to registry-only mode",
    unit="1",
)
nginx_updates_skipped_total = _CounterAdapter(_nginx_updates_skipped_counter)

_nginx_config_writes_counter = _meter.create_counter(
    name="mcpgw_registry_nginx_config_writes_total",
    description="Total nginx config file writes performed by the registry, by outcome",
    unit="1",
)
nginx_config_writes_total = _CounterAdapter(_nginx_config_writes_counter)


# Mode-blocked requests (registry/core/metrics.py:43)
_mode_blocked_requests_counter = _meter.create_counter(
    name="mcpgw_registry_mode_blocked_requests_total",
    description="Requests blocked due to registry mode restrictions",
    unit="1",
)
mode_blocked_requests_total = _CounterAdapter(_mode_blocked_requests_counter)


# Peer federation metrics (registry/core/metrics.py:50,56,61)
#
# NOTE: peer_token_missing_total was previously declared as a Gauge but had
# ZERO incrementing call sites in the codebase (confirmed at migration time
# via grep). Deleted rather than migrated. If a real call site is added
# later, name the new metric correctly per Prometheus convention (Gauges
# should not end in _total). Tracked in follow-up issue #1124.
_peer_sync_failures_counter = _meter.create_counter(
    name="peer_sync_failures_total",
    description="Total peer sync failures by failure type",
    unit="1",
)
peer_sync_failures_total = _CounterAdapter(_peer_sync_failures_counter)

_peer_sync_duration_histogram = _meter.create_histogram(
    name="peer_sync_duration",
    description="Duration of peer sync operations",
    unit="s",
)
# Legacy call sites use .labels(...).set(duration_seconds) per sync,
# which is one-shot histogram-record semantics. The adapter translates.
peer_sync_duration_seconds = _HistogramAdapter(_peer_sync_duration_histogram)


# Application log handler (registry/core/metrics.py:66)
_app_log_flush_failures_counter = _meter.create_counter(
    name="app_log_mongodb_flush_failures_total",
    description="Total MongoDB log handler flush failures",
    unit="1",
)
app_log_flush_failures_total = _CounterAdapter(_app_log_flush_failures_counter)


# Telemetry (registry/core/metrics.py:73)
_telemetry_sends_counter = _meter.create_counter(
    name="telemetry_sends_total",
    description="Total telemetry events sent",
    unit="1",
)
telemetry_sends_total = _CounterAdapter(_telemetry_sends_counter)


# M2M orphan cleanup (registry/core/metrics.py:81)
_m2m_orphan_cleanups_counter = _meter.create_counter(
    name="m2m_orphan_cleanups_total",
    description="Total M2M orphan cleanup deletions",
    unit="1",
)
m2m_orphan_cleanups_total = _CounterAdapter(_m2m_orphan_cleanups_counter)


# Search index cleanup on entity delete (issue #1145)
_embedding_removal_failures_counter = _meter.create_counter(
    name="mcpgw_registry_embedding_removal_failures_total",
    description="Failed search-index embedding removals during entity delete",
    unit="1",
)
embedding_removal_failures_total = _CounterAdapter(_embedding_removal_failures_counter)


# Cloud detection (registry/core/metrics.py:87)
_cloud_detection_counter = _meter.create_counter(
    name="mcpgw_registry_cloud_detection_total",
    description="Cloud-detection outcomes labeled by cloud and detection method",
    unit="1",
)
cloud_detection_total = _CounterAdapter(_cloud_detection_counter)


# Logout-related counters (registry/auth/routes.py:59-77)
_logout_id_token_hint_present_counter = _meter.create_counter(
    name="mcpgw_registry_logout_id_token_hint_present_total",
    description="Logouts where id_token hint was present",
    unit="1",
)
logout_id_token_hint_present_total = _CounterAdapter(_logout_id_token_hint_present_counter)

_logout_id_token_hint_missing_counter = _meter.create_counter(
    name="mcpgw_registry_logout_id_token_hint_missing_total",
    description="Logouts where id_token hint was missing",
    unit="1",
)
logout_id_token_hint_missing_total = _CounterAdapter(_logout_id_token_hint_missing_counter)

_logout_jwt_validation_failed_counter = _meter.create_counter(
    name="mcpgw_registry_logout_jwt_validation_failed_total",
    description="Logout JWT validation failures",
    unit="1",
)
logout_jwt_validation_failed_total = _CounterAdapter(_logout_jwt_validation_failed_counter)

_logout_url_length_warning_counter = _meter.create_counter(
    name="mcpgw_registry_logout_url_length_warning_total",
    description="Logout URLs exceeding recommended length",
    unit="1",
)
logout_url_length_warning_total = _CounterAdapter(_logout_url_length_warning_counter)


# Session store (registry/auth/session_store.py:32)
_session_store_resolve_counter = _meter.create_counter(
    name="mcpgw_registry_session_store_resolve_total",
    description="Session store resolve outcomes (hit/miss/expired/store_error)",
    unit="1",
)
session_store_resolve_total = _CounterAdapter(_session_store_resolve_counter)


# M2M management API (registry/api/m2m_management_routes.py:42)
_m2m_management_requests_counter = _meter.create_counter(
    name="mcpgw_registry_m2m_management_requests_total",
    description="Direct M2M client management API calls",
    unit="1",
)
m2m_management_requests_total = _CounterAdapter(_m2m_management_requests_counter)


# Internal service-token single-use enforcement (registry/auth/internal_replay_store.py).
# Emitted from both the registry and auth-server processes (validate_internal_auth
# runs in both), same as the session store counter above.
# result: accepted | replay | missing_jti | store_error. store_error is the
# fail-closed signal: DocumentDB unreachable => every internal call 401s, so
# rate(...{result="store_error"}) distinguishes a store outage from
# rate(...{result="replay"}) which flags an actual replay attack.
_internal_token_replay_check_counter = _meter.create_counter(
    name="mcpgw_registry_internal_token_replay_check_total",
    description="Internal service-token single-use checks by outcome",
    unit="1",
)
internal_token_replay_check_total = _CounterAdapter(_internal_token_replay_check_counter)


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
# ARD Registry adapter metrics (issue #1295)
# Every ARD operation (search, browse) emits through this family via the
# `operation` label, so each operation is independently observable.
# =============================================================================

ard_requests_total = _meter.create_counter(
    name="mcpgw_ard_requests_total",
    description="ARD Registry adapter requests by operation, status, and federation mode",
    unit="1",
)

ard_request_duration_ms = _meter.create_histogram(
    name="mcpgw_ard_request_duration",
    description="ARD Registry adapter request duration by operation",
    unit="ms",
)

ard_results_returned = _meter.create_histogram(
    name="mcpgw_ard_results_returned",
    description="Number of results/items returned per ARD request by operation",
    unit="1",
)

ard_access_filtered_total = _meter.create_counter(
    name="mcpgw_ard_access_filtered_total",
    description="ARD candidate entries removed by access-scoping, by operation",
    unit="1",
)

ard_errors_total = _meter.create_counter(
    name="mcpgw_ard_errors_total",
    description="ARD Registry adapter errors by operation and error_code",
    unit="1",
)

# --- ARD Phase 3: ingestion + federation trust (issue #1296) ---

ard_ingestion_runs_total = _meter.create_counter(
    name="mcpgw_ard_ingestion_runs_total",
    description="ARD ai-catalog ingestion runs by source_id and status",
    unit="1",
)

ard_ingestion_entries_total = _meter.create_counter(
    name="mcpgw_ard_ingestion_entries_total",
    description="ARD ingestion entries by source_id and outcome (indexed/rejected/orphaned)",
    unit="1",
)

ard_ingestion_duration_ms = _meter.create_histogram(
    name="mcpgw_ard_ingestion_duration",
    description="ARD ingestion run wall time in milliseconds, by source_id",
    unit="ms",
)

ard_trust_mismatch_total = _meter.create_counter(
    name="mcpgw_ard_trust_mismatch_total",
    description="ARD domain-anchored trust mismatches by source_id and policy",
    unit="1",
)


# =============================================================================
# Public helpers
# =============================================================================


def is_otel_enabled() -> bool:
    """Return True when OTel SDK is configured to export metrics.

    Used by middleware to decide whether to call the OTel emission path or
    fall back to the legacy HTTP POST. Auto-instrumentation initializes the
    SDK only when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set.
    """
    return bool(os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip())
