"""Unit tests for the caller_target (per-caller-per-target) rate-limit axis.

Verifies the composite counter subject isolates targets from each other and
callers from each other, that the gate is skipped without a classified target,
and that admins bypass it (fake backend, no DB).
"""

from registry.rate_limiting.backend import IncrResult, RateLimiterBackend
from registry.rate_limiting.limiter import RateLimiter
from registry.rate_limiting.models import RateLimitDefinition


class _FakeBackend(RateLimiterBackend):
    """In-memory conditional counter honoring deny-does-not-consume."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    async def incr_if_allowed(self, key, window_seconds, max_requests) -> IncrResult:
        current = self.counts.get(key, 0)
        if current >= max_requests:
            return IncrResult(allowed=False, current=max_requests)
        current += 1
        self.counts[key] = current
        return IncrResult(allowed=True, current=current)

    async def get(self, key, window_seconds) -> int:
        return self.counts.get(key, 0)


class _FakeDefs:
    """DefinitionsRepository stand-in for caller_target + quarantine-enabled reads."""

    def __init__(self, caller_target_defs: list[RateLimitDefinition]) -> None:
        self._ct_defs = caller_target_defs

    async def list_caller_limits(self, entity_type, names):
        return []

    async def list_caller_target_limits(self, entity_type, names):
        return [d for d in self._ct_defs if d.entity_type == entity_type and d.name in names]

    async def list_target_limits(self, entity_type, name):
        return []

    async def is_quarantine_group_enabled(self, group):
        return True


class _FakeMemberships:
    """MembershipsRepository stand-in: caller groups + no target quarantine."""

    def __init__(self, by_user: dict[str, list[str]] | None = None) -> None:
        self._by_user = by_user or {}

    async def get_groups_for(self, username, client_id):
        return list(self._by_user.get(username, []))

    async def is_target_quarantined(self, target_entity_type, target_name):
        return False


def _ct_def(name, max_requests, window_seconds=60):
    """A caller_target group definition (per-caller-per-target quota)."""
    return RateLimitDefinition(
        axis="caller_target",
        entity_type="group",
        name=name,
        user_max_requests=max_requests,
        agent_max_requests=max_requests,
        window_seconds=window_seconds,
    )


def _limiter(caller_target_defs, memberships):
    return RateLimiter(
        _FakeBackend(),
        _FakeDefs(caller_target_defs),
        memberships,
        fail_open=True,
        backend_timeout_seconds=0.25,
    )


async def _count_allowed(limiter, times, **kwargs):
    allowed = 0
    for _ in range(times):
        decision = await limiter.check(**kwargs)
        if decision.allowed:
            allowed += 1
    return allowed


class TestCallerTargetGates:
    async def test_composite_subject_isolates_targets(self):
        """Bursting caller A against server X does not consume A's quota against server Y."""
        limiter = _limiter(
            [_ct_def("cap", 3)],
            _FakeMemberships(by_user={"alice": ["cap"]}),
        )
        # Exhaust against server X.
        x_allowed = await _count_allowed(
            limiter,
            5,
            username="alice",
            client_id=None,
            target_entity_type="mcp_server",
            target_name="X",
        )
        assert x_allowed == 3
        # Quota against server Y is untouched.
        y_allowed = await _count_allowed(
            limiter,
            5,
            username="alice",
            client_id=None,
            target_entity_type="mcp_server",
            target_name="Y",
        )
        assert y_allowed == 3

    async def test_composite_subject_isolates_callers(self):
        """Caller A tripping the limit against X does not affect caller B against X."""
        limiter = _limiter(
            [_ct_def("cap", 2)],
            _FakeMemberships(by_user={"alice": ["cap"], "bob": ["cap"]}),
        )
        a_allowed = await _count_allowed(
            limiter,
            4,
            username="alice",
            client_id=None,
            target_entity_type="mcp_server",
            target_name="X",
        )
        b_allowed = await _count_allowed(
            limiter,
            4,
            username="bob",
            client_id=None,
            target_entity_type="mcp_server",
            target_name="X",
        )
        assert a_allowed == 2
        assert b_allowed == 2

    async def test_caller_target_skipped_without_target(self):
        """No caller_target gate is built when no target is classified."""
        limiter = _limiter(
            [_ct_def("cap", 1)],
            _FakeMemberships(by_user={"alice": ["cap"]}),
        )
        allowed = await _count_allowed(
            limiter,
            5,
            username="alice",
            client_id=None,
            target_entity_type=None,
            target_name=None,
        )
        assert allowed == 5  # no target => no gate => all allowed

    async def test_caller_target_admin_bypass(self):
        """An admin caller bypasses the caller_target gate."""
        limiter = _limiter(
            [_ct_def("cap", 1)],
            _FakeMemberships(by_user={"alice": ["cap"]}),
        )
        allowed = await _count_allowed(
            limiter,
            5,
            username="alice",
            client_id=None,
            is_admin=True,
            target_entity_type="mcp_server",
            target_name="X",
        )
        assert allowed == 5
