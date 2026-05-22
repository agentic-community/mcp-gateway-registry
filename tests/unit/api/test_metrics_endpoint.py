"""Unit tests for the /api/metrics endpoint (issue #867)."""

from fastapi.testclient import TestClient

from registry.main import app

client = TestClient(app)


class TestMetricsEndpoint:
    """Tests for the Prometheus metrics endpoint."""

    def test_metrics_endpoint_returns_200(self):
        """GET /api/metrics returns HTTP 200."""
        response = client.get("/api/metrics")
        assert response.status_code == 200

    def test_metrics_endpoint_content_type_is_prometheus_text(self):
        """Response Content-Type is Prometheus text exposition format."""
        response = client.get("/api/metrics")
        assert "text/plain" in response.headers["content-type"]

    def test_metrics_endpoint_contains_help_and_type_lines(self):
        """Response body contains # HELP and # TYPE lines."""
        response = client.get("/api/metrics")
        assert "# HELP" in response.text
        assert "# TYPE" in response.text

    def test_known_counters_present_in_output(self):
        """All known in-process counters appear in scrape output (value 0 is fine).

        These counters are declared at module level and register with the default
        REGISTRY collector at import time, so they appear even before any events fire.
        """
        response = client.get("/api/metrics")
        assert "registry_logout_id_token_hint_present_total" in response.text
        assert "mcp_config_view_requests_total" in response.text
        assert "m2m_management_requests_total" in response.text
        assert "registry_deployment_mode_info" in response.text
        assert "peer_sync_failures_total" in response.text
        assert "nginx_config_writes_total" in response.text
        assert "telemetry_sends_total" in response.text

    def test_python_runtime_metrics_present(self):
        """Default collector includes Python runtime metrics (GC stats, python_info)."""
        response = client.get("/api/metrics")
        # python_gc_* metrics are available on all platforms (Linux + macOS)
        assert "python_gc_objects_collected_total" in response.text
