"""Staggered tool refresh: which healthy servers are fetched on which cycle (issue #2050)."""

import random

import pytest
from pydantic import ValidationError

from registry.constants import HealthStatus
from registry.core.config import (
    Settings,
    settings,
)
from registry.health.service import HealthMonitoringService

SERVER_COUNT: int = 300
SEED: int = 2050
STORED_TOOLS: list[dict] = [{"name": "echo", "description": "Echo text."}]


@pytest.fixture
def service() -> HealthMonitoringService:
    """A health service with no background tasks running."""
    return HealthMonitoringService()


@pytest.fixture
def refresh_on(monkeypatch):
    """Turn tool refresh on with a period of 3 cycles."""
    monkeypatch.setattr(settings, "tool_refresh_enabled", True)
    monkeypatch.setattr(settings, "tool_refresh_every_cycles", 3)


def _due_cycles(
    service: HealthMonitoringService,
    path: str,
    cycles: int,
) -> list[int]:
    """Return the cycle numbers on which a server is due, over 1..cycles."""
    due = []
    for cycle in range(1, cycles + 1):
        service._health_cycle = cycle
        if service._is_tool_refresh_due(path):
            due.append(cycle)
    return due


@pytest.mark.unit
class TestRefreshDue:
    """_is_tool_refresh_due spreads servers and fetches each once per period."""

    def test_refresh_disabled_never_due(self, service, monkeypatch):
        monkeypatch.setattr(settings, "tool_refresh_enabled", False)
        assert _due_cycles(service, "/a", 9) == []
        assert service._tool_refresh_offsets == {}

    def test_each_server_due_once_per_period(self, service, refresh_on):
        random.seed(SEED)
        for index in range(20):
            due = _due_cycles(service, f"/server-{index}", 9)
            assert len(due) == 3
            assert due[1] - due[0] == 3
            assert due[2] - due[1] == 3

    def test_offset_drawn_once(self, service, refresh_on):
        service._is_tool_refresh_due("/a")
        first = service._tool_refresh_offsets["/a"]
        for cycle in range(1, 10):
            service._health_cycle = cycle
            service._is_tool_refresh_due("/a")
        assert service._tool_refresh_offsets["/a"] == first
        assert 1 <= first <= 3

    def test_offsets_spread_evenly(self, service, refresh_on):
        random.seed(SEED)
        per_cycle = {1: 0, 2: 0, 3: 0}
        for index in range(SERVER_COUNT):
            for cycle in _due_cycles(service, f"/server-{index}", 3):
                per_cycle[cycle] += 1
        assert sum(per_cycle.values()) == SERVER_COUNT
        for count in per_cycle.values():
            assert 80 <= count <= 120

    def test_period_of_one_is_every_cycle(self, service, monkeypatch):
        monkeypatch.setattr(settings, "tool_refresh_enabled", True)
        monkeypatch.setattr(settings, "tool_refresh_every_cycles", 1)
        assert _due_cycles(service, "/a", 4) == [1, 2, 3, 4]

    def test_prune_drops_removed_servers(self, service, refresh_on):
        service._is_tool_refresh_due("/kept")
        service._is_tool_refresh_due("/gone")
        service._prune_tool_refresh_offsets({"/kept"})
        assert set(service._tool_refresh_offsets) == {"/kept"}


@pytest.mark.unit
class TestFetchTrigger:
    """_tool_fetch_trigger keeps today's triggers and adds the refresh."""

    def test_first_check(self, service):
        trigger = service._tool_fetch_trigger(
            "/a", HealthStatus.UNKNOWN, {"tool_list": STORED_TOOLS}
        )
        assert trigger == "first_check"

    def test_recovered(self, service):
        trigger = service._tool_fetch_trigger(
            "/a", HealthStatus.UNHEALTHY_TIMEOUT, {"tool_list": STORED_TOOLS}
        )
        assert trigger == "recovered"

    def test_empty_tool_list(self, service):
        assert service._tool_fetch_trigger("/a", HealthStatus.HEALTHY, {"tool_list": []}) == (
            "empty"
        )

    def test_healthy_with_tools_and_refresh_off(self, service, monkeypatch):
        monkeypatch.setattr(settings, "tool_refresh_enabled", False)
        trigger = service._tool_fetch_trigger(
            "/a", HealthStatus.HEALTHY, {"tool_list": STORED_TOOLS}
        )
        assert trigger is None

    def test_healthy_with_tools_and_refresh_due(self, service, monkeypatch):
        monkeypatch.setattr(settings, "tool_refresh_enabled", True)
        monkeypatch.setattr(settings, "tool_refresh_every_cycles", 1)
        trigger = service._tool_fetch_trigger(
            "/a", HealthStatus.HEALTHY, {"tool_list": STORED_TOOLS}
        )
        assert trigger == "refresh"


@pytest.mark.unit
class TestRefreshSettings:
    """Out-of-range values stop the registry from starting."""

    @pytest.mark.parametrize(
        "field",
        ["tool_refresh_every_cycles", "tool_fetch_max_concurrency"],
    )
    def test_zero_is_rejected(self, field):
        with pytest.raises(ValidationError, match=field):
            Settings(**{field: 0})

    def test_defaults(self, monkeypatch):
        # _env_file=None and the delenv calls isolate the assertion from a
        # developer's local .env, which may turn refresh on for testing.
        for name in (
            "TOOL_REFRESH_ENABLED",
            "TOOL_REFRESH_EVERY_CYCLES",
            "TOOL_FETCH_MAX_CONCURRENCY",
        ):
            monkeypatch.delenv(name, raising=False)
        fresh = Settings(_env_file=None)
        assert fresh.tool_refresh_enabled is False
        assert fresh.tool_refresh_every_cycles == 3
        assert fresh.tool_fetch_max_concurrency == 5


@pytest.mark.unit
class TestStartupLog:
    """The startup log states the refresh period in cycles and seconds."""

    def test_refresh_on_logs_period(self, monkeypatch, caplog):
        from registry.health.service import _log_tool_refresh_settings

        monkeypatch.setattr(settings, "tool_refresh_enabled", True)
        monkeypatch.setattr(settings, "tool_refresh_every_cycles", 3)
        monkeypatch.setattr(settings, "health_check_interval_seconds", 300)
        with caplog.at_level("INFO", logger="registry.health.service"):
            _log_tool_refresh_settings()
        assert "every 3 health cycles (at least 900s)" in caplog.text

    def test_refresh_off_logs_off(self, monkeypatch, caplog):
        from registry.health.service import _log_tool_refresh_settings

        monkeypatch.setattr(settings, "tool_refresh_enabled", False)
        with caplog.at_level("INFO", logger="registry.health.service"):
            _log_tool_refresh_settings()
        assert "Tool refresh off" in caplog.text
