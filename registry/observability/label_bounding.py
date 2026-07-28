"""Bound attacker-influenced OTel/Prometheus label values against a cardinality DoS.

Request-derived values (tool names, JSON-RPC methods, client name/version) flow
into in-process Prometheus instrument attributes. Without a bound, a client
sending randomized values explodes the emitted time-series count and exhausts
Prometheus memory/storage -- a denial of service affecting every scraped
service. This module provides the canonical limiter for that in-process
(OTel-native) emission path, used by both the registry and the auth server.

The metrics-service processor carries its own equivalent limiter
(``metrics-service/app/core/processor.py::_CardinalityLimiter``) because it is a
separate deployable that cannot import ``registry`` in its container. The two
implementations are intentional siblings across that hard boundary; keep the
charset, length, and sentinel semantics in sync when either changes.
"""

from __future__ import annotations

import logging
import os
import re
import threading

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


# Maximum number of distinct values a single bounded label may take per process
# before further distinct values collapse to the overflow sentinel. Matches the
# metrics-service default so both emission paths behave identically.
_MAX_LABEL_CARDINALITY: int = _int_from_env("METRICS_MAX_LABEL_CARDINALITY", 150)

# Maximum length (characters) of a bounded label value before truncation.
_MAX_LABEL_LENGTH: int = _int_from_env("METRICS_MAX_LABEL_LENGTH", 64)

# Emitted once a bounded label exceeds the distinct-value cap; groups all
# overflow values into a single time series.
_OVERFLOW_LABEL_VALUE: str = "_other"

# Emitted when a value normalizes to the empty string (all illegal characters);
# distinct from the overflow bucket so operators can tell "malformed" from "too
# many".
_EMPTY_LABEL_VALUE: str = "_unset"

# Characters permitted in a bounded label value. Anything outside is replaced
# with an underscore so control chars, whitespace, and other high-cardinality
# noise cannot reach Prometheus. Alphanumerics plus a small separator set,
# including the slash so values like "tools/call" and "namespace/tool" survive
# (safe inside a Prometheus label value). Must stay in sync with the lua ingest
# charset (docker/lua/emit_metrics.lua) and the metrics-service processor.
_SAFE_LABEL_CHARS: re.Pattern[str] = re.compile(r"[^A-Za-z0-9\-_.:/]")


class LabelCardinalityLimiter:
    """Per-process bound on the distinct values a set of labels may take.

    Each bounded label is charset-normalized, length-capped, and its
    distinct-value count is tracked independently per label name. Once the cap
    is exceeded, further new values collapse to an overflow sentinel. Failing
    bounded (never passing a value through raw) is the intended behavior.
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
            name: The label name, used to track distinct values independently
                per label.
            value: The string-coerced label value to bound.

        Returns:
            The normalized value if it fits within the per-label distinct-value
            cap; otherwise the overflow sentinel.
        """
        cleaned = self._normalize_charset(value)
        # Sentinels are always admitted; they are the bounded outcomes and must
        # not themselves consume a cardinality slot that could evict a
        # legitimate value.
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

    def bound_attrs(
        self,
        attrs: dict[str, str],
        bounded_keys: frozenset[str],
    ) -> dict[str, str]:
        """Return a copy of ``attrs`` with the named keys cardinality-bounded.

        Keys in ``bounded_keys`` are charset-normalized, length-capped, and
        distinct-value-capped. All other keys are passed through unchanged (they
        are expected to be low-cardinality server-set enums).

        Args:
            attrs: The instrument attribute mapping (all values already strings).
            bounded_keys: The subset of keys whose values are attacker-influenced
                and must be bounded.

        Returns:
            A new attribute mapping safe to attach to an instrument.
        """
        result: dict[str, str] = {}
        for key, value in attrs.items():
            if key in bounded_keys:
                result[key] = self.bound(key, value)
            else:
                result[key] = value
        return result
