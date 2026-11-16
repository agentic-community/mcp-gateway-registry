"""Background tool fetch scheduling: concurrency cap, one fetch per server (issue #2050)."""

import asyncio
from unittest.mock import patch

import pytest

from registry.health.service import HealthMonitoringService

CONCURRENCY_LIMIT: int = 3
FETCH_COUNT: int = 20
FETCH_SECONDS: float = 0.02


@pytest.fixture
def service() -> HealthMonitoringService:
    """A health service with a small fetch semaphore."""
    instance = HealthMonitoringService()
    instance._tool_fetch_semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    return instance


class _SlowFetch:
    """Stand-in for _update_tools_background that records peak concurrency."""

    def __init__(self, seconds: float = FETCH_SECONDS):
        self.seconds = seconds
        self.running = 0
        self.peak = 0
        self.calls = 0

    async def __call__(self, service_path: str, proxy_pass_url: str) -> str:
        self.running += 1
        self.calls += 1
        self.peak = max(self.peak, self.running)
        try:
            await asyncio.sleep(self.seconds)
        finally:
            self.running -= 1
        return "unchanged"


async def _drain(service: HealthMonitoringService) -> None:
    """Wait for every scheduled fetch to finish."""
    while service._tool_fetch_tasks:
        await asyncio.gather(*list(service._tool_fetch_tasks), return_exceptions=True)


@pytest.mark.unit
@pytest.mark.asyncio
class TestToolFetchScheduler:
    """_schedule_tool_fetch bounds concurrency and deduplicates per server."""

    async def test_concurrency_never_exceeds_limit(self, service):
        fake = _SlowFetch()
        with patch.object(service, "_update_tools_background", fake):
            for index in range(FETCH_COUNT):
                service._schedule_tool_fetch(f"/s-{index}", "http://h/mcp", "refresh")
            await _drain(service)
        assert fake.calls == FETCH_COUNT
        assert fake.peak == CONCURRENCY_LIMIT

    async def test_second_schedule_skipped_while_in_flight(self, service):
        fake = _SlowFetch()
        with patch.object(service, "_update_tools_background", fake):
            assert service._schedule_tool_fetch("/a", "http://h/mcp", "refresh") is True
            assert service._schedule_tool_fetch("/a", "http://h/mcp", "manual") is False
            await _drain(service)
        assert fake.calls == 1

    async def test_server_can_be_fetched_again_after_finishing(self, service):
        fake = _SlowFetch()
        with patch.object(service, "_update_tools_background", fake):
            service._schedule_tool_fetch("/a", "http://h/mcp", "refresh")
            await _drain(service)
            assert service._schedule_tool_fetch("/a", "http://h/mcp", "refresh") is True
            await _drain(service)
        assert fake.calls == 2

    async def test_in_flight_cleared_after_failure(self, service):
        async def boom(service_path: str, proxy_pass_url: str) -> str:
            raise RuntimeError("upstream exploded")

        with patch.object(service, "_update_tools_background", boom):
            service._schedule_tool_fetch("/a", "http://h/mcp", "refresh")
            await _drain(service)
        assert "/a" not in service._tool_fetch_in_flight
        assert not service._tool_fetch_tasks

    async def test_task_reference_kept_until_done(self, service):
        fake = _SlowFetch()
        with patch.object(service, "_update_tools_background", fake):
            service._schedule_tool_fetch("/a", "http://h/mcp", "refresh")
            assert len(service._tool_fetch_tasks) == 1
            await _drain(service)
        assert not service._tool_fetch_tasks

    async def test_shutdown_cancels_pending_fetches(self, service):
        fake = _SlowFetch(seconds=10)
        with patch.object(service, "_update_tools_background", fake):
            for index in range(5):
                service._schedule_tool_fetch(f"/s-{index}", "http://h/mcp", "refresh")
            await asyncio.sleep(0)
            await service.shutdown()
        assert not service._tool_fetch_tasks
        assert not service._tool_fetch_in_flight


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fetch_timeout_returns_timeout(service, monkeypatch):
    """A hung upstream fetch stops at TOOL_FETCH_TIMEOUT_SECONDS and frees its slot."""

    async def hang(*_args, **_kwargs):
        await asyncio.sleep(10)

    monkeypatch.setattr("registry.health.service.TOOL_FETCH_TIMEOUT_SECONDS", 0.05)
    with (
        patch("registry.core.mcp_client.mcp_client_service") as mock_mcp,
        patch("registry.services.server_service.server_service") as mock_servers,
    ):
        mock_mcp.get_mcp_connection_result = hang
        mock_servers.get_server_info = hang
        result = await service._update_tools_background("/a", "http://h/mcp")
    assert result == "timeout"
