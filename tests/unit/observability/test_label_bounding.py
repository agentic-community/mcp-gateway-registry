"""Unit tests for the shared in-process label cardinality limiter.

``registry/observability/label_bounding.py::LabelCardinalityLimiter`` bounds
attacker-influenced OTel/Prometheus label values on the in-process emission path
used by both the registry and the auth server. It is an intentional sibling of
the metrics-service processor's ``_CardinalityLimiter`` (that service is a
separate deployable that cannot import ``registry``); the drift-guard test at
the bottom pins the shared semantics.
"""

from __future__ import annotations

from registry.observability.label_bounding import (
    _EMPTY_LABEL_VALUE,
    _OVERFLOW_LABEL_VALUE,
    _SAFE_LABEL_CHARS,
    LabelCardinalityLimiter,
)


class TestBound:
    def test_flood_collapses_to_overflow(self) -> None:
        limiter = LabelCardinalityLimiter(max_cardinality=3)
        seen = set()
        for i in range(200):
            seen.add(limiter.bound("tool_name", f"tool{i}"))
        assert _OVERFLOW_LABEL_VALUE in seen
        assert len(seen - {_OVERFLOW_LABEL_VALUE}) == 3

    def test_legitimate_value_passes_through(self) -> None:
        limiter = LabelCardinalityLimiter(max_cardinality=10)
        assert limiter.bound("method", "tools/call") == "tools/call"

    def test_illegal_chars_replaced_slash_preserved(self) -> None:
        limiter = LabelCardinalityLimiter()
        result = limiter.bound("tool_name", "rm -rf /; drop\ntable")
        assert result == "rm_-rf_/__drop_table"
        assert " " not in result
        assert ";" not in result
        assert "\n" not in result

    def test_oversized_truncated(self) -> None:
        limiter = LabelCardinalityLimiter(max_length=10)
        assert len(limiter.bound("client_name", "a" * 100)) == 10

    def test_empty_maps_to_sentinel(self) -> None:
        # An empty string (or any value that has no representable chars) maps to
        # the empty sentinel rather than becoming a blank label. Note "!!!"
        # would normalize to "___" since underscore is in the safe charset; only
        # a genuinely empty result hits the sentinel.
        limiter = LabelCardinalityLimiter()
        assert limiter.bound("method", "") == _EMPTY_LABEL_VALUE

    def test_sentinel_does_not_consume_budget(self) -> None:
        limiter = LabelCardinalityLimiter(max_cardinality=1)
        assert limiter.bound("tool_name", "real") == "real"
        for i in range(50):
            assert limiter.bound("tool_name", f"x{i}") == _OVERFLOW_LABEL_VALUE
        assert limiter.bound("tool_name", "real") == "real"

    def test_per_label_independent_budgets(self) -> None:
        limiter = LabelCardinalityLimiter(max_cardinality=1)
        assert limiter.bound("tool_name", "t1") == "t1"
        # a different label name has its own budget
        assert limiter.bound("method", "m1") == "m1"


class TestBoundAttrs:
    def test_only_named_keys_bounded(self) -> None:
        limiter = LabelCardinalityLimiter(max_cardinality=1)
        # flood two calls so the 2nd distinct value of a bounded key overflows,
        # while an unbounded key passes raw both times
        limiter.bound_attrs({"tool_name": "a", "server_name": "srv1"}, frozenset({"tool_name"}))
        out = limiter.bound_attrs(
            {"tool_name": "b", "server_name": "srv2"}, frozenset({"tool_name"})
        )
        assert out["tool_name"] == _OVERFLOW_LABEL_VALUE
        assert out["server_name"] == "srv2"  # unbounded, raw

    def test_does_not_mutate_input(self) -> None:
        limiter = LabelCardinalityLimiter()
        original = {"tool_name": "evil name", "success": "true"}
        limiter.bound_attrs(dict(original), frozenset({"tool_name"}))
        assert original == {"tool_name": "evil name", "success": "true"}


def test_sibling_limiter_semantics_are_pinned() -> None:
    # This limiter is duplicated in metrics-service/app/core/processor.py across
    # the deployable boundary (that service cannot import `registry`). The
    # charset, sentinels, and default length MUST match the sibling. These
    # literal pins fail loudly if this side drifts; the metrics-service suite
    # (test_semantics_match_registry_sibling_limiter) holds the mirror-image
    # pins. If you change one, change both.
    assert _OVERFLOW_LABEL_VALUE == "_other"
    assert _EMPTY_LABEL_VALUE == "_unset"
    assert _SAFE_LABEL_CHARS.pattern == r"[^A-Za-z0-9\-_.:/]"
    assert LabelCardinalityLimiter()._max_length == 64
