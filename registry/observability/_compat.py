"""Compatibility adapters that mimic the prometheus_client API on top of OTel meters.

These adapters let existing call sites that look like
``METRIC.labels(key=value).inc()`` keep working without modification, while
the underlying instrument is an OpenTelemetry Counter / UpDownCounter.

The compatibility layer is temporary; new code should call the OTel
instruments directly via ``counter.add(value, attributes)``. The shim is
expected to be removed in 1.26.0 along with the legacy HTTP POST path.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class _CounterAdapter:
    """Wraps an OTel Counter so legacy ``.labels(k=v).inc()`` calls work.

    IMPORTANT semantic difference from ``prometheus_client.Counter``: this
    ``.labels(...)`` returns a NEW adapter instance and does NOT mutate
    ``self``. Calling ``counter.labels(x="a")`` and storing the result in a
    variable gives you a fresh, single-shot adapter. Each call to
    ``.labels()`` produces a new instance carrying ONLY the kwargs from
    that call.

    ``prometheus_client.Counter`` is stateful (the labels stack on ``self``),
    which is why this difference matters. If you write
    ``c = counter.labels(x="a")`` then ``c.inc()`` here, you get exactly
    what you'd expect: increment with ``{"x": "a"}``. If you write
    ``counter.labels(x="a").labels(y="b").inc()``, the first ``.labels()``
    is discarded, NOT merged. That matches the pattern observed in the
    repo (always one ``.labels()`` call per ``.inc()``), so this is safe
    in practice; just don't chain.
    """

    def __init__(
        self,
        otel_counter: Any,
        pending_attrs: dict[str, str] | None = None,
    ) -> None:
        self._counter = otel_counter
        self._pending_attrs: dict[str, str] = pending_attrs or {}

    def labels(self, **kwargs: Any) -> _CounterAdapter:
        """Return a new adapter carrying only the kwargs from this call."""
        return _CounterAdapter(
            self._counter,
            pending_attrs={k: str(v) for k, v in kwargs.items()},
        )

    def inc(self, value: float = 1.0) -> None:
        """Increment the underlying OTel Counter."""
        self._counter.add(value, self._pending_attrs)


class _UpDownCounterAdapter:
    """Wraps an OTel UpDownCounter so legacy Gauge ``.set()`` / ``.inc()`` calls work.

    Prometheus Gauges support ``.set()``, ``.inc()``, ``.dec()``. OTel does
    not have a direct Gauge equivalent for synchronous mutation; the closest
    is ``UpDownCounter``, which only supports ``.add(delta)``. We approximate:

    - ``.inc()`` -> ``add(+value)``
    - ``.dec()`` -> ``add(-value)``
    - ``.set(value)`` -> NOT supported (raises) because there's no consistent
      way to translate "set absolute value" into a delta without tracking
      the previous value, which would create a memory leak across labels.

    For Prometheus Gauges that historically used ``.set()``, the migration
    plan is to convert them to ``ObservableGauge`` callbacks instead. See
    ``registry.observability.meters.create_observable_gauge`` for the
    pattern.
    """

    def __init__(
        self,
        otel_updown_counter: Any,
        pending_attrs: dict[str, str] | None = None,
    ) -> None:
        self._counter = otel_updown_counter
        self._pending_attrs: dict[str, str] = pending_attrs or {}

    def labels(self, **kwargs: Any) -> _UpDownCounterAdapter:
        return _UpDownCounterAdapter(
            self._counter,
            pending_attrs={k: str(v) for k, v in kwargs.items()},
        )

    def inc(self, value: float = 1.0) -> None:
        self._counter.add(value, self._pending_attrs)

    def dec(self, value: float = 1.0) -> None:
        self._counter.add(-value, self._pending_attrs)

    def set(self, value: float) -> None:  # noqa: A003 - mirroring prometheus_client API
        """Not supported on UpDownCounter adapters.

        Use an ObservableGauge with a callback instead. Raises a clear error
        so a migration regression surfaces immediately rather than silently
        recording wrong values.
        """
        raise NotImplementedError(
            "UpDownCounter adapter does not support .set(). Use "
            "ObservableGauge with a callback in meters.py instead."
        )


class _HistogramAdapter:
    """Wraps an OTel Histogram so legacy ``.labels(k=v).set(value)`` calls work.

    Several Prometheus Gauges in the codebase are used as one-shot duration
    recorders: ``.labels(...).set(duration_seconds)`` is called once per
    operation. That semantics is actually a Histogram, not a Gauge — and
    OTel Histograms are the natural fit. This adapter translates the
    legacy ``.set()`` API into ``histogram.record(value, attrs)``.

    ``.observe(value)`` is also supported (the prometheus_client Histogram
    spelling) for symmetry, in case any call site uses it.

    Like ``_CounterAdapter``, ``.labels()`` returns a NEW adapter instance
    rather than mutating ``self``. Don't chain ``.labels()`` calls.
    """

    def __init__(
        self,
        otel_histogram: Any,
        pending_attrs: dict[str, str] | None = None,
    ) -> None:
        self._histogram = otel_histogram
        self._pending_attrs: dict[str, str] = pending_attrs or {}

    def labels(self, **kwargs: Any) -> _HistogramAdapter:
        return _HistogramAdapter(
            self._histogram,
            pending_attrs={k: str(v) for k, v in kwargs.items()},
        )

    def set(self, value: float) -> None:  # noqa: A003 - mirroring prometheus_client API
        """Record a single observation on the underlying OTel Histogram."""
        self._histogram.record(value, self._pending_attrs)

    def observe(self, value: float) -> None:
        """Alias for ``set`` matching prometheus_client.Histogram API."""
        self._histogram.record(value, self._pending_attrs)
