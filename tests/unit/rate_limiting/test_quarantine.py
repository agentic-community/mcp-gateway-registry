"""Unit tests for the quarantine (kill-switch) short-circuit in the limiter.

Covers caller + target quarantine denials, admin bypass (caller yes, target no),
deny-does-not-consume, the global off-switch (disabled sentinel), and the
fail-open / opt-in fail-closed policy on a backend error.
"""

from registry.rate_limiting.backend import IncrResult, RateLimiterBackend
from registry.rate_limiting.limiter import RateLimiter
from registry.rate_limiting.models import (
    QUARANTINE_CALLER_GROUP,
)


class _FakeBackend(RateLimiterBackend):
    """Counter backend that records every increment so we can assert none happened."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def incr_if_allowed(self, key, window_seconds, max_requests) -> IncrResult:
        self.calls.append(key)
        return IncrResult(allowed=True, current=1)

    async def get(self, key, window_seconds) -> int:
        return 0


class _FakeDefs:
    """DefinitionsRepository stand-in; quarantine sentinel enabled unless overridden."""

    def __init__(self, enabled: dict[str, bool] | None = None, raise_enabled: bool = False) -> None:
        self._enabled = enabled or {}
        self._raise_enabled = raise_enabled

    async def list_caller_limits(self, entity_type, names):
        return []

    async def list_caller_target_limits(self, entity_type, names):
        return []

    async def list_target_limits(self, entity_type, name):
        return []

    async def is_quarantine_group_enabled(self, group):
        if self._raise_enabled:
            raise RuntimeError("defs down")
        return self._enabled.get(group, True)


class _FakeMemberships:
    """Memberships stand-in for caller-group + target-quarantine lookups."""

    def __init__(
        self,
        by_user: dict[str, list[str]] | None = None,
        quarantined_targets: set[str] | None = None,
        raise_target: bool = False,
    ) -> None:
        self._by_user = by_user or {}
        self._quarantined_targets = quarantined_targets or set()
        self._raise_target = raise_target

    async def get_groups_for(self, username, client_id):
        return list(self._by_user.get(username, []))

    async def is_target_quarantined(self, target_entity_type, target_name):
        if self._raise_target:
            raise RuntimeError("memberships down")
        return target_name in self._quarantined_targets


def _limiter(defs=None, memberships=None, fail_open=True, quarantine_fail_closed=False):
    return RateLimiter(
        _FakeBackend(),
        defs or _FakeDefs(),
        memberships or _FakeMemberships(),
        fail_open=fail_open,
        backend_timeout_seconds=0.25,
        quarantine_fail_closed=quarantine_fail_closed,
    )


class TestQuarantine:
    async def test_caller_quarantine_denies(self):
        """A caller in quarantine-callers is denied with a quarantine (plain-403) decision."""
        backend = _FakeBackend()
        limiter = RateLimiter(
            backend,
            _FakeDefs(),
            _FakeMemberships(by_user={"alice": [QUARANTINE_CALLER_GROUP]}),
            fail_open=True,
        )
        decision = await limiter.check(
            username="alice",
            client_id=None,
            target_entity_type="mcp_server",
            target_name="X",
        )
        assert decision.allowed is False
        assert decision.quarantined is True
        # No counter was consumed on a quarantine deny.
        assert backend.calls == []

    async def test_target_quarantine_denies_all(self):
        """A quarantined target denies even an admin caller."""
        limiter = _limiter(
            memberships=_FakeMemberships(quarantined_targets={"X"}),
        )
        decision = await limiter.check(
            username="admin",
            client_id=None,
            is_admin=True,
            target_entity_type="mcp_server",
            target_name="X",
        )
        assert decision.allowed is False
        assert decision.quarantined is True
        assert decision.axis == "tgt"

    async def test_caller_quarantine_admin_bypass(self):
        """An admin caller is NOT blocked by caller quarantine (no self-lockout)."""
        limiter = _limiter(
            memberships=_FakeMemberships(by_user={"admin": [QUARANTINE_CALLER_GROUP]}),
        )
        decision = await limiter.check(
            username="admin",
            client_id=None,
            is_admin=True,
            target_entity_type="mcp_server",
            target_name="X",
        )
        assert decision.allowed is True

    async def test_disabled_reserved_group_is_global_off(self):
        """Disabling the reserved caller sentinel skips the short-circuit even with members."""
        limiter = _limiter(
            defs=_FakeDefs(enabled={QUARANTINE_CALLER_GROUP: False}),
            memberships=_FakeMemberships(by_user={"alice": [QUARANTINE_CALLER_GROUP]}),
        )
        decision = await limiter.check(
            username="alice",
            client_id=None,
            target_entity_type="mcp_server",
            target_name="X",
        )
        assert decision.allowed is True

    async def test_target_quarantine_fail_open_on_error(self):
        """A memberships read error allows (fail-open default)."""
        limiter = _limiter(
            memberships=_FakeMemberships(raise_target=True),
            fail_open=True,
            quarantine_fail_closed=False,
        )
        decision = await limiter.check(
            username="alice",
            client_id=None,
            target_entity_type="mcp_server",
            target_name="X",
        )
        assert decision.allowed is True

    async def test_target_quarantine_fail_closed_when_configured(self):
        """A memberships read error denies (quarantine) when fail-closed is set."""
        limiter = _limiter(
            memberships=_FakeMemberships(raise_target=True),
            quarantine_fail_closed=True,
        )
        decision = await limiter.check(
            username="alice",
            client_id=None,
            target_entity_type="mcp_server",
            target_name="X",
        )
        assert decision.allowed is False
        assert decision.quarantined is True
