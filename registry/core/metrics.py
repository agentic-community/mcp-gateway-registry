"""Compatibility shim for in-process metrics.

Re-exports OpenTelemetry-backed instruments from
``registry.observability.meters`` under the historical names that existing
call sites use.

This shim is temporary and will be deleted in 1.26.0 alongside the
``METRICS_LEGACY_HTTP_POST`` flag. New code should import directly from
``registry.observability.meters``.

Migration notes (issue #1122):
- All ``Counter``-style metrics are now ``_CounterAdapter`` instances wrapping
  OTel ``Counter`` instruments. The legacy ``.labels(...).inc()`` API works
  unchanged.
- ``PEER_SYNC_DURATION_SECONDS`` was a Prometheus ``Gauge`` whose only
  call sites used ``.labels(...).set(duration)``. It is now backed by an
  OTel ``Histogram`` (semantically a one-shot duration measurement). The
  ``_HistogramAdapter`` translates ``.set(value)`` to ``histogram.record``.
- ``DEPLOYMENT_MODE_INFO`` was a Prometheus ``Gauge`` set to 1 once at
  startup. It is now an OTel ``ObservableGauge`` whose callback (see
  ``registry/observability/meters.py``) reads the current deployment mode
  on every export cycle. The historical ``DEPLOYMENT_MODE_INFO`` symbol is
  preserved as a no-op shim so existing call sites still compile; it has
  no effect because the observable gauge handles emission.
- ``PEER_TOKEN_MISSING`` was confirmed dead (zero call sites at migration
  time) and is not re-exported. See follow-up issue #1124.
"""

from __future__ import annotations

import logging
from typing import Any

from registry.observability.meters import (
    app_log_flush_failures_total as APP_LOG_FLUSH_FAILURES,
)
from registry.observability.meters import (
    cloud_detection_total as CLOUD_DETECTION_TOTAL,
)
from registry.observability.meters import (
    config_export_requests_total as CONFIG_EXPORT_REQUESTS,
)
from registry.observability.meters import (
    config_view_requests_total as CONFIG_VIEW_REQUESTS,
)
from registry.observability.meters import (
    m2m_orphan_cleanups_total as M2M_ORPHAN_CLEANUPS_TOTAL,
)
from registry.observability.meters import (
    mode_blocked_requests_total as MODE_BLOCKED_REQUESTS,
)
from registry.observability.meters import (
    nginx_config_writes_total as NGINX_CONFIG_WRITES,
)
from registry.observability.meters import (
    nginx_updates_skipped_total as NGINX_UPDATES_SKIPPED,
)
from registry.observability.meters import (
    peer_sync_duration_seconds as PEER_SYNC_DURATION_SECONDS,
)
from registry.observability.meters import (
    peer_sync_failures_total as PEER_SYNC_FAILURES,
)
from registry.observability.meters import (
    telemetry_sends_total,
)

logger = logging.getLogger(__name__)


class _NoOpDeploymentModeShim:
    """No-op stand-in for the legacy ``DEPLOYMENT_MODE_INFO`` Gauge.

    The current deployment-mode emission is handled by the OTel
    ``ObservableGauge`` callback in ``registry.observability.meters``. The
    historical ``DEPLOYMENT_MODE_INFO.labels(...).set(1)`` call site (in
    ``registry/main.py:_initialize_deployment_metrics``) is preserved
    intentionally so the import still works during the migration window.
    The shim returns itself from ``.labels(...)`` and does nothing on
    ``.set(...)``; the observable gauge produces the same data on every
    export cycle.
    """

    def labels(self, **_kwargs: Any) -> _NoOpDeploymentModeShim:
        return self

    def set(self, _value: float = 1.0) -> None:  # noqa: A003 - mirroring prometheus_client API
        # Intentionally a no-op; ObservableGauge callback handles emission.
        pass


DEPLOYMENT_MODE_INFO = _NoOpDeploymentModeShim()


__all__ = [
    "APP_LOG_FLUSH_FAILURES",
    "CLOUD_DETECTION_TOTAL",
    "CONFIG_EXPORT_REQUESTS",
    "CONFIG_VIEW_REQUESTS",
    "DEPLOYMENT_MODE_INFO",
    "M2M_ORPHAN_CLEANUPS_TOTAL",
    "MODE_BLOCKED_REQUESTS",
    "NGINX_CONFIG_WRITES",
    "NGINX_UPDATES_SKIPPED",
    "PEER_SYNC_DURATION_SECONDS",
    "PEER_SYNC_FAILURES",
    "telemetry_sends_total",
]
