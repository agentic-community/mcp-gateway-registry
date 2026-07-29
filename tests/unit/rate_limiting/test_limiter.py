"""Unit tests for RateLimiter orchestration (fake backend, no DB).

Covers the behaviors the design calls out as critical: the Blocker-1 regression
(a denied burst must not consume a wider window's quota), most-restrictive
resolution within a window, gate stacking, and fail-open / fail-closed.
"""

from registry.rate_limiting.backend import IncrResult, RateLimiterBackend
from registry.rate_limiting.limiter import RateLimiter
from registry.rate_limiting.models import RateLimitDefinition


class _FakeBackend(RateLimiterBackend):
    """In-memory conditional counter honoring the deny-does-not-consume contract."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    async def incr_if_allowed(
        self,
        key: str,
        window_seconds: int,
        max_requests: int,
    ) -> IncrResult:
        # The key here omits the window index (the fake keeps one window per key).
        current = self.counts.get(key, 0)
        if current >= max_requests:
            return IncrResult(allowed=False, current=max_requests)
        current += 1
        self.counts[key] = current
        return IncrResult(allowed=True, current=current)

    async def get(
        self,
        key: str,
        window_seconds: int,
    ) -> int:
        return self.counts.get(key, 0)


class _BoomBackend(RateLimiterBackend):
    """A backend whose ops always raise, to exercise fail-open / fail-closed."""

    async def incr_if_allowed(self, key, window_seconds, max_requests):
        raise RuntimeError("db down")

    async def get(self, key, window_seconds):
        raise RuntimeError("db down")


class _SlowBackend(RateLimiterBackend):
    """A backend that hangs longer than the limiter's timeout, to exercise fail-fast."""

    def __init__(self, delay_seconds: float) -> None:
        self._delay_seconds = delay_seconds

    async def incr_if_allowed(self, key, window_seconds, max_requests):
        import asyncio

        await asyncio.sleep(self._delay_seconds)
        return IncrResult(allowed=True, current=1)

    async def get(self, key, window_seconds):
        return 0


class _FakeDefs:
    """Minimal DefinitionsRepository stand-in returning fixed lists.

    Filters caller defs by entity_type and name (like the real repo), so a
    group/user/client query only sees its own definitions.
    """

    def __init__(
        self,
        caller_defs: list[RateLimitDefinition],
        target_defs: list[RateLimitDefinition],
    ) -> None:
        self._caller_defs = caller_defs
        self._target_defs = target_defs

    async def list_caller_limits(self, entity_type, names):
        return [d for d in self._caller_defs if d.entity_type == entity_type and d.name in names]

    async def list_caller_target_limits(self, entity_type, names):
        # Existing limiter tests do not exercise the caller_target axis.
        return []

    async def list_target_limits(self, entity_type, name):
        return [d for d in self._target_defs if d.entity_type == entity_type and d.name == name]

    async def is_quarantine_group_enabled(self, group):
        return True


class _FakeMemberships:
    """MembershipsRepository stand-in mapping username/client_id -> rate-limit groups."""

    def __init__(
        self,
        by_user: dict[str, list[str]] | None = None,
        by_client: dict[str, list[str]] | None = None,
    ) -> None:
        self._by_user = by_user or {}
        self._by_client = by_client or {}

    async def get_groups_for(self, username, client_id):
        groups: list[str] = []
        if username and username in self._by_user:
            groups.extend(self._by_user[username])
        if client_id and client_id in self._by_client:
            groups.extend(self._by_client[client_id])
        return groups

    async def is_target_quarantined(self, target_entity_type, target_name):
        # Existing limiter tests do not exercise target quarantine.
        return False


def _make_limiter(
    backend,
    caller_defs=None,
    target_defs=None,
    memberships=None,
    fail_open=True,
    backend_timeout_seconds=0.25,
):
    """Build a RateLimiter with fake defs + memberships (keeps tests concise)."""
    return RateLimiter(
        backend,
        _FakeDefs(caller_defs or [], target_defs or []),
        memberships or _FakeMemberships(),
        fail_open=fail_open,
        backend_timeout_seconds=backend_timeout_seconds,
    )


async def _count_allowed(
    limiter: RateLimiter,
    times: int,
    **check_kwargs,
) -> int:
    """Drive ``times`` checks and count how many were allowed."""
    allowed = 0
    for _ in range(times):
        decision = await limiter.check(**check_kwargs)
        if decision.allowed:
            allowed += 1
    return allowed


def _caller(name, max_requests, window_seconds):
    """A group caller definition setting both user + agent limits to max_requests.

    (Model-level construction; the config-time floor is a route-layer check and
    does not constrain direct model instances, so small test values are fine.)
    """
    return RateLimitDefinition(
        axis="caller",
        entity_type="group",
        name=name,
        user_max_requests=max_requests,
        agent_max_requests=max_requests,
        window_seconds=window_seconds,
    )


class TestRateLimiter:
    """Behavioral tests for gate orchestration.

    A caller's rate-limit groups come from the memberships stub (keyed by
    username), never from the token, mirroring production.
    """

    async def test_no_definitions_allows(self):
        """With no matching definitions, every call is allowed."""
        limiter = _make_limiter(_FakeBackend())
        decision = await limiter.check(username="u", client_id=None)
        assert decision.allowed is True

    async def test_membership_group_gate_enforced(self):
        """A caller in a rate-limited group (via memberships) is limited; the token is not consulted."""
        backend = _FakeBackend()
        limiter = _make_limiter(
            backend,
            caller_defs=[_caller("dev", 5, 60)],
            memberships=_FakeMemberships(by_user={"alice": ["dev"]}),
        )
        allowed = await _count_allowed(limiter, 8, username="alice", client_id=None)
        assert allowed == 5
        # A user with no membership is unlimited (no group resolved).
        other = await _count_allowed(limiter, 4, username="bob", client_id=None)
        assert other == 4

    async def test_throttle_log_carries_caller_identity(self, caplog):
        """A throttle logs caller_type + username/client_id so it is attributable.

        The identity is NOT a metric label (unbounded cardinality); the app log is
        where an operator answers "which user/client got throttled".
        """
        import logging

        backend = _FakeBackend()
        limiter = _make_limiter(
            backend,
            caller_defs=[_caller("dev", 1, 60)],
            memberships=_FakeMemberships(by_user={"alice": ["dev"]}),
        )
        with caplog.at_level(logging.WARNING, logger="registry.rate_limiting.limiter"):
            # First call allowed, second throttled (limit 1/60s).
            await _count_allowed(limiter, 2, username="alice", client_id=None)

        throttle_lines = [r.getMessage() for r in caplog.records if "throttled" in r.getMessage()]
        assert throttle_lines, "expected a throttle log line"
        msg = throttle_lines[0]
        assert "caller_type=user" in msg
        assert "caller_username=alice" in msg
        assert "caller_client_id=" in msg

    async def test_client_membership_resolves_groups(self):
        """An agent's rate-limit group resolves from its client_id membership."""
        backend = _FakeBackend()
        limiter = _make_limiter(
            backend,
            caller_defs=[_caller("agents", 3, 60)],
            memberships=_FakeMemberships(by_client={"agent-1": ["agents"]}),
        )
        allowed = await _count_allowed(limiter, 6, username=None, client_id="agent-1")
        assert allowed == 3

    async def test_user_and_agent_limits_differ_within_a_group(self):
        """A group's user vs agent number applies by caller type."""
        # Group sets user=4, agent=2 on the same window.
        group = RateLimitDefinition(
            axis="caller",
            entity_type="group",
            name="mixed",
            user_max_requests=4,
            agent_max_requests=2,
            window_seconds=60,
        )
        # A human (username, no client_id) gets the user number.
        b1 = _FakeBackend()
        lim1 = _make_limiter(
            b1, caller_defs=[group], memberships=_FakeMemberships(by_user={"alice": ["mixed"]})
        )
        assert await _count_allowed(lim1, 8, username="alice", client_id=None) == 4
        # An agent (client_id) gets the agent number.
        b2 = _FakeBackend()
        lim2 = _make_limiter(
            b2, caller_defs=[group], memberships=_FakeMemberships(by_client={"cli": ["mixed"]})
        )
        assert await _count_allowed(lim2, 8, username=None, client_id="cli") == 2

    async def test_group_with_only_user_limit_does_not_gate_agents(self):
        """A group that sets only user_max_requests does not limit agent callers."""
        group = RateLimitDefinition(
            axis="caller",
            entity_type="group",
            name="u-only",
            user_max_requests=3,
            window_seconds=60,
        )
        backend = _FakeBackend()
        limiter = _make_limiter(
            backend,
            caller_defs=[group],
            memberships=_FakeMemberships(by_client={"cli": ["u-only"]}),
        )
        # Agent has no agent-limit in this group -> unlimited.
        assert await _count_allowed(limiter, 6, username=None, client_id="cli") == 6

    async def test_admin_bypasses_caller_gates(self):
        """An admin caller is not subject to caller (group) limits."""
        backend = _FakeBackend()
        limiter = _make_limiter(
            backend,
            caller_defs=[_caller("dev", 2, 60)],
            memberships=_FakeMemberships(by_user={"admin": ["dev"]}),
        )
        # is_admin=True -> caller gates skipped, all allowed.
        allowed = 0
        for _ in range(6):
            d = await limiter.check(username="admin", client_id=None, is_admin=True)
            allowed += 1 if d.allowed else 0
        assert allowed == 6

    async def test_admin_still_subject_to_target_gates(self):
        """Admin bypass covers caller gates only; target gates still protect a backend."""
        backend = _FakeBackend()
        target_defs = [
            RateLimitDefinition(
                axis="target",
                entity_type="mcp_server",
                name="mcpgw",
                max_requests=2,
                window_seconds=60,
            )
        ]
        limiter = _make_limiter(backend, target_defs=target_defs)
        allowed = 0
        for _ in range(5):
            d = await limiter.check(
                username="admin",
                client_id=None,
                is_admin=True,
                target_entity_type="mcp_server",
                target_name="mcpgw",
            )
            allowed += 1 if d.allowed else 0
        assert allowed == 2

    async def test_burst_denial_does_not_consume_daily_cap(self):
        """BLOCKER-1 regression: a burst-denied request must NOT advance the daily gate."""
        backend = _FakeBackend()
        limiter = _make_limiter(
            backend,
            caller_defs=[_caller("dev", 5, 60), _caller("dev", 20, 86400)],
            memberships=_FakeMemberships(by_user={"alice": ["dev"]}),
        )
        allowed = await _count_allowed(limiter, 30, username="alice", client_id=None)
        assert allowed == 5
        assert backend.counts.get("clr:group:alice:86400") == 5

    async def test_most_restrictive_wins_within_same_window(self):
        """Among a caller's groups sharing a window, the smallest max_requests governs."""
        backend = _FakeBackend()
        limiter = _make_limiter(
            backend,
            caller_defs=[_caller("g1", 100, 60), _caller("g2", 3, 60)],
            memberships=_FakeMemberships(by_user={"bob": ["g1", "g2"]}),
        )
        allowed = await _count_allowed(limiter, 10, username="bob", client_id=None)
        assert allowed == 3

    async def test_target_gate_counts_across_callers(self):
        """A target (A2A agent) gate is enforced independent of caller identity."""
        backend = _FakeBackend()
        target_defs = [
            RateLimitDefinition(
                axis="target",
                entity_type="a2a_agent",
                name="booking",
                max_requests=2,
                window_seconds=60,
            )
        ]
        limiter = _make_limiter(backend, target_defs=target_defs)
        allowed = await _count_allowed(
            limiter,
            5,
            username="anyone",
            client_id=None,
            target_entity_type="a2a_agent",
            target_name="booking",
        )
        assert allowed == 2
        assert backend.counts.get("tgt:a2a_agent:booking:60") == 2

    async def test_server_group_keys_bucket_per_member_server(self):
        """A server_group limit buckets per the LIVE server (tgt:server_group:<server>:w).

        The counter key uses the request's server name, so each member server gets
        its own independent bucket even though they share one definition.
        """
        backend = _FakeBackend()
        group_def = RateLimitDefinition(
            axis="target",
            entity_type="server_group",
            name="fragile",
            max_requests=2,
            window_seconds=60,
            members=["mcpgw", "atlas"],
        )
        # Stub: the group def is returned for any of its member servers, mirroring
        # the real repo's $or expansion.
        limiter = _make_limiter(backend)

        async def _list_target_limits(entity_type, name):
            if entity_type == "mcp_server" and name in group_def.members:
                return [group_def]
            return []

        limiter._definitions.list_target_limits = _list_target_limits  # type: ignore[attr-defined]

        # Exhaust mcpgw's own bucket (2/60s).
        mcpgw_allowed = await _count_allowed(
            limiter,
            5,
            username="u",
            client_id=None,
            target_entity_type="mcp_server",
            target_name="mcpgw",
        )
        assert mcpgw_allowed == 2
        assert backend.counts.get("tgt:server_group:mcpgw:60") == 2
        # atlas has its OWN independent bucket (per-member uniform, not pooled).
        atlas_allowed = await _count_allowed(
            limiter,
            5,
            username="u",
            client_id=None,
            target_entity_type="mcp_server",
            target_name="atlas",
        )
        assert atlas_allowed == 2
        assert backend.counts.get("tgt:server_group:atlas:60") == 2

    async def test_server_group_and_individual_target_are_independent_buckets(self):
        """A server in a group AND with its own mcp_server limit gets two buckets, both enforced."""
        backend = _FakeBackend()
        group_def = RateLimitDefinition(
            axis="target",
            entity_type="server_group",
            name="grp",
            max_requests=5,
            window_seconds=60,
            members=["mcpgw"],
        )
        indiv_def = RateLimitDefinition(
            axis="target",
            entity_type="mcp_server",
            name="mcpgw",
            max_requests=2,
            window_seconds=60,
        )
        limiter = _make_limiter(backend)

        async def _list_target_limits(entity_type, name):
            if entity_type == "mcp_server" and name == "mcpgw":
                return [group_def, indiv_def]
            return []

        limiter._definitions.list_target_limits = _list_target_limits  # type: ignore[attr-defined]
        # The tighter individual limit (2) governs; both buckets exist.
        allowed = await _count_allowed(
            limiter,
            6,
            username="u",
            client_id=None,
            target_entity_type="mcp_server",
            target_name="mcpgw",
        )
        assert allowed == 2
        assert backend.counts.get("tgt:mcp_server:mcpgw:60") == 2
        assert "tgt:server_group:mcpgw:60" in backend.counts

    async def test_caller_and_target_gates_stack(self):
        """When both a caller and a target gate apply, the tighter one governs."""
        backend = _FakeBackend()
        target_defs = [
            RateLimitDefinition(
                axis="target",
                entity_type="mcp_server",
                name="mcpgw",
                max_requests=3,
                window_seconds=60,
            )
        ]
        limiter = _make_limiter(
            backend,
            caller_defs=[_caller("dev", 10, 60)],
            target_defs=target_defs,
            memberships=_FakeMemberships(by_user={"alice": ["dev"]}),
        )
        allowed = await _count_allowed(
            limiter,
            8,
            username="alice",
            client_id=None,
            target_entity_type="mcp_server",
            target_name="mcpgw",
        )
        assert allowed == 3

    async def test_deny_decision_has_retry_after(self):
        """A denied call returns 429-shaped info with a positive Retry-After."""
        limiter = _make_limiter(
            _FakeBackend(),
            caller_defs=[_caller("dev", 1, 60)],
            memberships=_FakeMemberships(by_user={"alice": ["dev"]}),
        )
        await limiter.check(username="alice", client_id=None)  # consume the 1 allowed
        decision = await limiter.check(username="alice", client_id=None)
        assert decision.allowed is False
        assert decision.limit == 1
        assert int(decision.headers()["Retry-After"]) >= 1

    async def test_fail_open_allows_on_backend_error(self):
        """With fail_open, a backend error results in allow (availability guardrail)."""
        limiter = _make_limiter(
            _BoomBackend(),
            caller_defs=[_caller("dev", 5, 60)],
            memberships=_FakeMemberships(by_user={"x": ["dev"]}),
            fail_open=True,
        )
        decision = await limiter.check(username="x", client_id=None)
        assert decision.allowed is True

    async def test_fail_closed_definition_denies_on_backend_error(self):
        """A per-limit fail_closed=True denies when the backend errors."""
        fc = RateLimitDefinition(
            axis="caller",
            entity_type="group",
            name="dev",
            user_max_requests=5,
            agent_max_requests=5,
            window_seconds=60,
            fail_closed=True,
        )
        limiter = _make_limiter(
            _BoomBackend(),
            caller_defs=[fc],
            memberships=_FakeMemberships(by_user={"x": ["dev"]}),
            fail_open=True,
        )
        decision = await limiter.check(username="x", client_id=None)
        assert decision.allowed is False

    async def test_global_fail_open_false_denies_on_error(self):
        """With fail_open=False globally, a backend error denies even without fail_closed."""
        limiter = _make_limiter(
            _BoomBackend(),
            caller_defs=[_caller("dev", 5, 60)],
            memberships=_FakeMemberships(by_user={"x": ["dev"]}),
            fail_open=False,
        )
        decision = await limiter.check(username="x", client_id=None)
        assert decision.allowed is False

    async def test_slow_backend_times_out_and_fails_open(self):
        """A backend slower than the timeout is treated as an error and fails open fast."""
        limiter = _make_limiter(
            _SlowBackend(delay_seconds=1.0),
            caller_defs=[_caller("dev", 5, 60)],
            memberships=_FakeMemberships(by_user={"x": ["dev"]}),
            fail_open=True,
            backend_timeout_seconds=0.05,
        )
        decision = await limiter.check(username="x", client_id=None)
        assert decision.allowed is True

    async def test_slow_backend_times_out_and_fails_closed_when_configured(self):
        """A slow backend on a fail_closed limit denies (does not hang, does not allow)."""
        fc = RateLimitDefinition(
            axis="caller",
            entity_type="group",
            name="dev",
            user_max_requests=5,
            agent_max_requests=5,
            window_seconds=60,
            fail_closed=True,
        )
        limiter = _make_limiter(
            _SlowBackend(delay_seconds=1.0),
            caller_defs=[fc],
            memberships=_FakeMemberships(by_user={"x": ["dev"]}),
            fail_open=True,
            backend_timeout_seconds=0.05,
        )
        decision = await limiter.check(username="x", client_id=None)
        assert decision.allowed is False
