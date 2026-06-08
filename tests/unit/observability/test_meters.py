"""Unit tests for the OpenTelemetry meter modules and compatibility shim.

Covers the migration from prometheus_client.Counter/Gauge declarations to
native OTel meter instruments. The tests are deliberately implementation-
oriented: they validate that the adapter shape is correct so existing
.labels(...).inc() / .set() call sites keep working without modification.

When OTel is not configured (no OTEL_EXPORTER_OTLP_ENDPOINT), the global
meter provider returns a NoOp meter and instrument calls become no-ops.
The tests verify that this no-op state is safe (no exceptions, no leaks).
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.unit]


class TestCounterAdapter:
    """Tests for registry.observability._compat._CounterAdapter."""

    def test_labels_then_inc_calls_underlying_add(self):
        from registry.observability._compat import _CounterAdapter

        otel_counter = MagicMock()
        adapter = _CounterAdapter(otel_counter)
        adapter.labels(status="success", outcome="ok").inc()

        otel_counter.add.assert_called_once_with(1.0, {"status": "success", "outcome": "ok"})

    def test_labels_returns_new_instance_not_self(self):
        from registry.observability._compat import _CounterAdapter

        otel_counter = MagicMock()
        adapter = _CounterAdapter(otel_counter)
        adapter_with_labels = adapter.labels(status="success")

        # Verify original adapter has empty pending_attrs, new instance has the kwargs
        assert adapter is not adapter_with_labels
        assert adapter._pending_attrs == {}
        assert adapter_with_labels._pending_attrs == {"status": "success"}

    def test_inc_with_custom_value(self):
        from registry.observability._compat import _CounterAdapter

        otel_counter = MagicMock()
        adapter = _CounterAdapter(otel_counter)
        adapter.labels(category="bulk").inc(42.0)

        otel_counter.add.assert_called_once_with(42.0, {"category": "bulk"})

    def test_int_label_values_coerced_to_str(self):
        from registry.observability._compat import _CounterAdapter

        otel_counter = MagicMock()
        adapter = _CounterAdapter(otel_counter)
        adapter.labels(top_k=5).inc()

        otel_counter.add.assert_called_once_with(1.0, {"top_k": "5"})

    def test_inc_without_labels_uses_empty_attrs(self):
        from registry.observability._compat import _CounterAdapter

        otel_counter = MagicMock()
        adapter = _CounterAdapter(otel_counter)
        adapter.inc()

        otel_counter.add.assert_called_once_with(1.0, {})


class TestHistogramAdapter:
    """Tests for registry.observability._compat._HistogramAdapter."""

    def test_labels_then_set_calls_underlying_record(self):
        from registry.observability._compat import _HistogramAdapter

        otel_histogram = MagicMock()
        adapter = _HistogramAdapter(otel_histogram)
        adapter.labels(peer_id="peer-a", success="true").set(2.5)

        otel_histogram.record.assert_called_once_with(2.5, {"peer_id": "peer-a", "success": "true"})

    def test_observe_is_alias_for_set(self):
        from registry.observability._compat import _HistogramAdapter

        otel_histogram = MagicMock()
        adapter = _HistogramAdapter(otel_histogram)
        adapter.labels(x="y").observe(1.23)

        otel_histogram.record.assert_called_once_with(1.23, {"x": "y"})


class TestUpDownCounterAdapter:
    """Tests for registry.observability._compat._UpDownCounterAdapter."""

    def test_inc_adds_positive(self):
        from registry.observability._compat import _UpDownCounterAdapter

        otel_updown = MagicMock()
        adapter = _UpDownCounterAdapter(otel_updown)
        adapter.labels(state="active").inc()

        otel_updown.add.assert_called_once_with(1.0, {"state": "active"})

    def test_dec_adds_negative(self):
        from registry.observability._compat import _UpDownCounterAdapter

        otel_updown = MagicMock()
        adapter = _UpDownCounterAdapter(otel_updown)
        adapter.labels(state="active").dec()

        otel_updown.add.assert_called_once_with(-1.0, {"state": "active"})

    def test_set_raises_clear_error(self):
        from registry.observability._compat import _UpDownCounterAdapter

        adapter = _UpDownCounterAdapter(MagicMock())
        with pytest.raises(NotImplementedError, match="ObservableGauge"):
            adapter.set(5.0)


class TestRegistryMeterDeclarations:
    """Verify all expected instruments are exposed by registry.observability.meters."""

    def test_path2_event_instruments_exposed(self):
        from registry.observability import meters

        # Counter instruments for path-2 events (formerly MetricsClient.emit_*)
        assert hasattr(meters, "registry_operation_total")
        assert hasattr(meters, "tool_discovery_total")
        assert hasattr(meters, "tool_execution_total")
        assert hasattr(meters, "health_check_total")

        # Histograms for duration measurements
        assert hasattr(meters, "registry_operation_duration_ms")
        assert hasattr(meters, "tool_discovery_duration_ms")

    def test_path3_in_process_instruments_exposed(self):
        from registry.observability import meters

        # Adapter-wrapped Counter instruments for path-3 in-process metrics
        for name in [
            "config_view_requests_total",
            "config_export_requests_total",
            "nginx_updates_skipped_total",
            "nginx_config_writes_total",
            "mode_blocked_requests_total",
            "peer_sync_failures_total",
            "app_log_flush_failures_total",
            "telemetry_sends_total",
            "m2m_orphan_cleanups_total",
            "cloud_detection_total",
            "logout_id_token_hint_present_total",
            "logout_id_token_hint_missing_total",
            "logout_jwt_validation_failed_total",
            "logout_url_length_warning_total",
            "session_store_resolve_total",
            "m2m_management_requests_total",
        ]:
            assert hasattr(meters, name), f"meters module missing instrument: {name}"

    def test_dead_peer_token_missing_not_exposed(self):
        """peer_token_missing_total was confirmed dead at migration time and removed."""
        from registry.observability import meters

        assert not hasattr(meters, "peer_token_missing_total")

    def test_peer_sync_duration_seconds_supports_legacy_set_api(self):
        """peer_sync_duration_seconds was a Gauge using .labels().set(); adapter must support it."""
        from registry.observability.meters import peer_sync_duration_seconds

        # Should not raise
        peer_sync_duration_seconds.labels(peer_id="p1", success="true").set(1.5)

    def test_record_emission_path_helper_exposed(self):
        from registry.observability.meters import record_emission_path

        # Should not raise even when OTel is not configured (no-op meter)
        record_emission_path("otel")
        record_emission_path("legacy")

    def test_is_otel_enabled_reads_env(self, monkeypatch):
        from registry.observability.meters import is_otel_enabled

        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        assert is_otel_enabled() is False

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        assert is_otel_enabled() is True


class TestRegistryCoreMetricsShim:
    """Verify the legacy registry/core/metrics.py re-exports work."""

    def test_legacy_uppercase_names_re_exported(self):
        from registry.core import metrics as legacy

        # Path-3 counters under their historical UPPER_SNAKE names
        for name in [
            "CONFIG_VIEW_REQUESTS",
            "CONFIG_EXPORT_REQUESTS",
            "NGINX_UPDATES_SKIPPED",
            "NGINX_CONFIG_WRITES",
            "MODE_BLOCKED_REQUESTS",
            "PEER_SYNC_FAILURES",
            "PEER_SYNC_DURATION_SECONDS",
            "APP_LOG_FLUSH_FAILURES",
            "M2M_ORPHAN_CLEANUPS_TOTAL",
            "CLOUD_DETECTION_TOTAL",
        ]:
            assert hasattr(legacy, name), f"legacy metrics shim missing: {name}"

    def test_telemetry_sends_total_lowercase_preserved(self):
        """Pre-existing inconsistency: telemetry_sends_total was already lowercase."""
        from registry.core import metrics as legacy

        assert hasattr(legacy, "telemetry_sends_total")

    def test_dead_peer_token_missing_not_in_shim(self):
        """The Gauge with zero call sites is gone; importers will fail loudly."""
        from registry.core import metrics as legacy

        assert not hasattr(legacy, "PEER_TOKEN_MISSING")

    def test_deployment_mode_info_shim_is_noop(self):
        """DEPLOYMENT_MODE_INFO is preserved as a no-op stand-in.

        The OTel ObservableGauge in the meters module produces the actual
        time series via its callback; the legacy call site (.labels().set(1))
        is a no-op for backward compatibility.
        """
        from registry.core.metrics import DEPLOYMENT_MODE_INFO

        # Should not raise
        DEPLOYMENT_MODE_INFO.labels(deployment_mode="local", registry_mode="full").set(1)


class TestAuthServerMeterDeclarations:
    """Verify all expected instruments are exposed by auth_server.observability.meters."""

    def test_auth_request_instruments_exposed(self):
        from auth_server.observability import meters

        assert hasattr(meters, "auth_request_total")
        assert hasattr(meters, "auth_request_duration_ms")

    def test_tool_execution_instruments_exposed(self):
        from auth_server.observability import meters

        assert hasattr(meters, "tool_execution_total")
        assert hasattr(meters, "tool_execution_duration_ms")

    def test_protocol_latency_instrument_exposed(self):
        from auth_server.observability import meters

        assert hasattr(meters, "protocol_latency_ms")

    def test_record_emission_path_helper_exposed(self):
        from auth_server.observability.meters import record_emission_path

        # No-op when OTel is not configured; verifying it doesn't raise
        record_emission_path("otel")
        record_emission_path("legacy")


class TestNoOpBehaviorWhenOTelDisabled:
    """When OTel SDK is uninitialized (no OTEL_EXPORTER_OTLP_ENDPOINT), instrument
    calls must be safe no-ops. The OTel SDK provides this guarantee via the
    NoOpMeterProvider; these tests just verify our shims don't add any
    surprising failure modes on top of that.
    """

    def test_counter_adapter_inc_does_not_raise_on_noop_meter(self, caplog):
        from registry.observability.meters import nginx_config_writes_total

        with caplog.at_level(logging.WARNING):
            # Should not raise even if no OTel SDK is initialized
            nginx_config_writes_total.labels(status="success").inc()

    def test_histogram_adapter_set_does_not_raise_on_noop_meter(self):
        from registry.observability.meters import peer_sync_duration_seconds

        # Should not raise
        peer_sync_duration_seconds.labels(peer_id="x", success="true").set(0.5)
