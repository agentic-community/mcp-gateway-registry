import asyncio
import logging
import os
import re
import threading
from datetime import datetime
from typing import Any
from ..core.models import MetricRequest, Metric, MetricType
from ..storage.database import MetricsStorage
from ..core.validator import validator

logger = logging.getLogger(__name__)


def _int_from_env(
    name: str,
    default: int,
) -> int:
    """Read a positive integer from the environment, falling back to default.

    A missing, empty, non-integer, or non-positive value falls back to the
    default so a misconfiguration never disables the cardinality bound (fail
    bounded, not open).

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset or malformed.

    Returns:
        The parsed positive integer, or the default.
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        parsed = int(raw.strip())
    except ValueError:
        logger.warning("Ignoring non-integer %s=%r; using default %d", name, raw, default)
        return default
    if parsed <= 0:
        logger.warning("Ignoring non-positive %s=%d; using default %d", name, parsed, default)
        return default
    return parsed


# Maximum number of distinct values a single attacker-influenced label may take
# per process. Once exceeded, further distinct values collapse to the overflow
# sentinel so a client sending randomized values cannot explode the Prometheus
# time-series count (DoS). 150 comfortably covers a realistic set of tools and
# clients while capping the blast radius.
_MAX_LABEL_CARDINALITY: int = _int_from_env("METRICS_MAX_LABEL_CARDINALITY", 150)

# Maximum length (characters) of a bounded label value before truncation. Keeps
# a single label value from bloating the series name and provides a second cap
# even before the distinct-count limit is reached.
_MAX_LABEL_LENGTH: int = _int_from_env("METRICS_MAX_LABEL_LENGTH", 64)

# Sentinel emitted once a bounded label exceeds the distinct-value cap. Groups
# all overflow values into a single Prometheus time series.
_OVERFLOW_LABEL_VALUE: str = "_other"

# Sentinel emitted when a value normalizes to the empty string (e.g. all
# characters were illegal). Distinct from the overflow bucket so operators can
# tell "malformed" from "too many".
_EMPTY_LABEL_VALUE: str = "_unset"

# Dimension names whose values are attacker-influenced and therefore bounded
# before becoming Prometheus labels. Purely internal dimensions (server_name,
# success, status_code, ...) are intentionally left untouched so legitimate
# telemetry keeps full fidelity.
#   - tool_name:   JSON-RPC params.name / caller-supplied tool identifier
#   - client_name: X-Client-Name request header (fully attacker-controlled)
#   - query:       free-text search query from tool-discovery metrics
#   - method:      JSON-RPC method copied verbatim from the request body; a
#                  client can send an arbitrary method string (it is recorded
#                  even when the upstream rejects the call), so it is as
#                  attacker-controllable as tool_name and must be bounded too.
#   - client_version: from the client's `clientInfo.version` in the JSON-RPC
#                  body; attacker-controllable like client_name.
_BOUNDED_LABEL_DIMENSIONS: frozenset[str] = frozenset(
    {"tool_name", "client_name", "client_version", "query", "method"}
)

# Per-metric-type allowlist of dimension KEYS permitted to become Prometheus
# label NAMES. The value-axis cap above bounds how many distinct VALUES a
# risky label may take; this bounds the label-NAME axis so a direct authenticated
# POST cannot invent arbitrary dimension keys (each distinct key name is a
# distinct Prometheus series signature -- an unbounded-key flood is a cardinality
# bomb the value cap cannot catch). Keys are the union of what every emitter
# actually sends for that type (nginx lua flush, registry/auth metrics clients);
# anything outside the set is dropped fail-closed before it reaches an
# instrument. Only metric types that actually attach labels in _emit_to_otel
# appear here. registry_operation and custom have no _emit_to_otel branch, so
# their dimensions never become labels and are intentionally omitted (custom
# carries operator-defined keys by design).
#
# IMPORTANT: if a new metric type is given a labelled _emit_to_otel branch, or a
# new dimension is added to an existing emitter, add it here or it will be
# silently dropped from Prometheus.
_ALLOWED_DIMENSION_KEYS: dict[str, frozenset[str]] = {
    "tool_execution": frozenset(
        {
            "method",
            "server_name",
            "server_path",
            "tool_name",
            "client_name",
            "client_version",
            "success",
            "user_hash",
        }
    ),
    "auth_request": frozenset({"success", "method", "server", "target_kind", "user_hash"}),
    "tool_discovery": frozenset({"query", "results_count", "top_k_services", "top_n_tools"}),
    "protocol_latency": frozenset({"flow_step", "server_name", "user_hash", "session_key"}),
    "health_check": frozenset({"endpoint", "status_code", "healthy"}),
}

# Characters permitted in a bounded label value. Anything outside this set is
# replaced with an underscore so control characters, whitespace, and other
# high-cardinality noise cannot reach Prometheus. Alphanumerics plus a small
# set of separators commonly seen in legitimate tool/client names and JSON-RPC
# methods (the slash keeps values like "tools/call" and "namespace/tool"
# intact; it is safe inside a Prometheus label value). Must stay in sync with
# the lua ingest charset in docker/lua/emit_metrics.lua.
_SAFE_LABEL_CHARS: re.Pattern[str] = re.compile(r"[^A-Za-z0-9\-_.:/]")


def _normalize_label_value(value: object) -> str:
    """Normalize a label value for Prometheus compatibility.

    Python's str(True) produces "True" (capital T), but Prometheus convention
    is lowercase "true"/"false". Lua's tostring() already produces lowercase,
    so without normalization the same metric gets split into two timeseries.

    Args:
        value: The raw dimension value of any type.

    Returns:
        A Prometheus-compatible string form of the value.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


class _CardinalityLimiter:
    """Per-process bound on the distinct values a set of labels may take.

    Attacker-controlled request data (tool names, client names, search queries)
    flows verbatim into Prometheus labels. Without a bound, a client sending
    randomized values explodes the time-series count and exhausts Prometheus
    memory/storage -- a DoS affecting every scraped service. This limiter is the
    reliable, last-line-of-defense enforcement point: each bounded label is
    charset-normalized, length-capped, and its distinct-value count is tracked
    per label name. Once the cap is exceeded, further new values collapse to an
    overflow sentinel. Failing bounded (never passing a value through raw) is
    the intended behavior.
    """

    def __init__(
        self,
        max_cardinality: int = _MAX_LABEL_CARDINALITY,
        max_length: int = _MAX_LABEL_LENGTH,
    ) -> None:
        self._max_cardinality = max_cardinality
        self._max_length = max_length
        # label name -> set of already-admitted normalized values
        self._seen: dict[str, set[str]] = {}
        self._lock = threading.Lock()

    def _normalize_charset(
        self,
        value: str,
    ) -> str:
        """Apply charset and length bounds to a single value.

        Illegal characters are replaced with underscore and the result is
        truncated to the configured maximum length. An empty result maps to the
        empty sentinel so it never becomes a blank label.

        Args:
            value: The already string-coerced label value.

        Returns:
            The charset- and length-bounded value.
        """
        cleaned = _SAFE_LABEL_CHARS.sub("_", value)
        if len(cleaned) > self._max_length:
            cleaned = cleaned[: self._max_length]
        if not cleaned:
            return _EMPTY_LABEL_VALUE
        return cleaned

    def bound(
        self,
        name: str,
        value: str,
    ) -> str:
        """Return a cardinality-bounded, normalized value for a label.

        Args:
            name: The label (dimension) name, used to track distinct values
                independently per label.
            value: The string-coerced label value to bound.

        Returns:
            The normalized value if it fits within the per-label distinct-value
            cap; otherwise the overflow sentinel.
        """
        cleaned = self._normalize_charset(value)
        # Sentinels are always admitted; they are the bounded outcomes and must
        # not themselves consume a cardinality slot in a way that could evict
        # legitimate values.
        if cleaned in (_OVERFLOW_LABEL_VALUE, _EMPTY_LABEL_VALUE):
            return cleaned

        with self._lock:
            seen = self._seen.setdefault(name, set())
            if cleaned in seen:
                return cleaned
            if len(seen) >= self._max_cardinality:
                return _OVERFLOW_LABEL_VALUE
            seen.add(cleaned)
            return cleaned

    def bound_dimensions(
        self,
        metric_type: str,
        dimensions: dict[str, Any],
    ) -> dict[str, str]:
        """Normalize dimensions into Prometheus labels, bounding risky ones.

        Two independent cardinality axes are bounded:

        - Label NAME axis: only keys on this metric type's allowlist (see
          ``_ALLOWED_DIMENSION_KEYS``) may become labels. Unknown keys are
          dropped fail-closed so a direct authenticated POST cannot invent
          arbitrary dimension keys and explode the label-name axis. A metric
          type with no allowlist entry (no labelled instrument) contributes no
          dimension labels at all.
        - Label VALUE axis: attacker-influenced dimensions (see
          ``_BOUNDED_LABEL_DIMENSIONS``) are charset-normalized, length-capped,
          and distinct-value-capped. Remaining allowed dimensions are only
          value-normalized for Prometheus compatibility, so legitimate internal
          telemetry keeps full fidelity.

        Args:
            metric_type: The metric's type value (e.g. ``"tool_execution"``),
                used to select the permitted dimension keys.
            dimensions: The raw metric dimensions.

        Returns:
            A mapping of label name to bounded string value.
        """
        allowed = _ALLOWED_DIMENSION_KEYS.get(metric_type)
        if not allowed:
            # No labelled instrument for this type (or an unrecognized type):
            # emit no dimension labels rather than trust arbitrary keys.
            return {}
        labels: dict[str, str] = {}
        for key, value in dimensions.items():
            if key not in allowed:
                logger.debug(
                    "Dropping unexpected dimension %r for metric type %s",
                    key,
                    metric_type,
                )
                continue
            normalized = _normalize_label_value(value)
            if key in _BOUNDED_LABEL_DIMENSIONS:
                labels[key] = self.bound(key, normalized)
            else:
                labels[key] = normalized
        return labels


class ProcessingResult:
    def __init__(self):
        self.accepted = 0
        self.rejected = 0
        self.errors = []


class MetricsProcessor:
    """Core metrics processing engine."""

    def __init__(self):
        self.storage = MetricsStorage()
        self._buffer = []
        self._buffer_lock = asyncio.Lock()

        # Bounds attacker-influenced label cardinality before values reach
        # Prometheus (last line of defense against label-cardinality DoS).
        self._cardinality_limiter = _CardinalityLimiter()

        # Try to initialize OTel instruments, but don't fail if it doesn't work
        self.otel = None
        try:
            from ..otel.instruments import MetricsInstruments

            self.otel = MetricsInstruments()
            logger.info("OpenTelemetry instruments initialized")
        except Exception as e:
            logger.warning(f"OpenTelemetry instruments not available: {e}")

    async def process_metrics(
        self, request: MetricRequest, request_id: str, api_key: str
    ) -> ProcessingResult:
        """Process incoming metrics request."""
        result = ProcessingResult()

        # Validate the entire request first
        validation_result = validator.validate_metric_request(request)
        if not validation_result.is_valid:
            result.rejected = len(request.metrics)
            result.errors.extend(validation_result.get_error_messages())
            return result

        # Log any validation warnings
        for warning in validation_result.warnings:
            logger.warning(f"Metrics validation warning: {warning}")

        for metric in request.metrics:
            try:
                # Additional runtime validation
                if not self._validate_metric(metric):
                    result.rejected += 1
                    result.errors.append(f"Invalid metric: {metric.type}")
                    continue

                # Emit to OpenTelemetry if available
                if self.otel:
                    try:
                        await self._emit_to_otel(metric, request.service)
                    except Exception as e:
                        logger.warning(f"Failed to emit to OTel: {e}")

                # Store in SQLite (buffered)
                await self._buffer_for_storage(metric, request, request_id)

                result.accepted += 1

            except Exception as e:
                result.rejected += 1
                result.errors.append(f"Error processing metric: {str(e)}")
                logger.error(f"Error processing metric: {e}")

        return result

    def _validate_metric(self, metric: Metric) -> bool:
        """Validate metric data."""
        if metric.value is None:
            return False
        if metric.type not in MetricType:
            return False
        return True

    async def _emit_to_otel(self, metric: Metric, service: str):
        """Emit metric to OpenTelemetry instruments."""
        if not self.otel:
            return

        labels = {
            "service": service,
            "metric_type": metric.type.value,
            **self._cardinality_limiter.bound_dimensions(metric.type.value, metric.dimensions),
        }

        # Route to appropriate OTel instrument
        if metric.type == MetricType.AUTH_REQUEST:
            self.otel.auth_counter.add(metric.value, labels)
            if metric.duration_ms:
                self.otel.auth_histogram.record(metric.duration_ms / 1000, labels)

        elif metric.type == MetricType.TOOL_DISCOVERY:
            self.otel.discovery_counter.add(metric.value, labels)
            if metric.duration_ms:
                self.otel.discovery_histogram.record(metric.duration_ms / 1000, labels)

        elif metric.type == MetricType.TOOL_EXECUTION:
            self.otel.tool_counter.add(metric.value, labels)
            if metric.duration_ms:
                self.otel.tool_histogram.record(metric.duration_ms / 1000, labels)

        elif metric.type == MetricType.PROTOCOL_LATENCY:
            # For protocol latency, record the value as latency seconds
            self.otel.latency_histogram.record(metric.value, labels)

        elif metric.type == MetricType.HEALTH_CHECK:
            self.otel.health_counter.add(metric.value, labels)
            if metric.duration_ms:
                self.otel.health_histogram.record(metric.duration_ms / 1000, labels)

    async def _buffer_for_storage(self, metric: Metric, request: MetricRequest, request_id: str):
        """Buffer metric for batch SQLite storage."""
        async with self._buffer_lock:
            self._buffer.append({"metric": metric, "request": request, "request_id": request_id})

            # Flush buffer if it's full
            if len(self._buffer) >= 100:
                await self._flush_buffer()

    async def _flush_buffer(self):
        """Flush buffered metrics to SQLite."""
        if not self._buffer:
            return

        buffer_copy = self._buffer.copy()
        self._buffer.clear()

        try:
            await self.storage.store_metrics_batch(buffer_copy)
            logger.debug(f"Flushed {len(buffer_copy)} metrics to storage")
        except Exception as e:
            logger.error(f"Failed to flush metrics buffer: {e}")
            # Re-add to buffer for retry
            self._buffer.extend(buffer_copy)

    async def force_flush(self):
        """Force flush all buffered metrics."""
        async with self._buffer_lock:
            await self._flush_buffer()
