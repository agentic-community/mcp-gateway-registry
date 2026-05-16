"""Tests for the shared reload lock on the NginxConfigService singleton.

Covers issue #1044: every call site that invokes ``generate_config_async``
or ``reload_nginx`` must acquire ``nginx_service.reload_lock`` so concurrent
CRUD operations cannot race on the config file or on ``nginx -s reload``.
"""

import asyncio

import pytest

from registry.core.nginx_service import nginx_service


class TestReloadLock:
    @pytest.mark.asyncio
    async def test_lock_serializes_concurrent_acquires(self) -> None:
        """Three concurrent acquires complete one at a time."""
        in_critical_section: list[int] = []
        max_concurrent = [0]

        async def acquirer(idx: int) -> None:
            async with nginx_service.reload_lock:
                in_critical_section.append(idx)
                max_concurrent[0] = max(max_concurrent[0], len(in_critical_section))
                await asyncio.sleep(0.05)
                in_critical_section.remove(idx)

        await asyncio.gather(acquirer(1), acquirer(2), acquirer(3))
        # If the lock works, only one task ever held the critical section
        assert max_concurrent[0] == 1

    @pytest.mark.asyncio
    async def test_lock_released_on_exception(self) -> None:
        """An exception inside the lock block must still release the lock."""
        with pytest.raises(RuntimeError):
            async with nginx_service.reload_lock:
                raise RuntimeError("boom")

        assert not nginx_service.reload_lock.locked()

    def test_virtual_server_does_not_define_its_own_lock(self) -> None:
        """The previous module-level _nginx_reload_lock has been removed.

        virtual_server_service used to declare its own lock; this PR
        consolidated all call sites onto the shared nginx_service.reload_lock.
        """
        from registry.services import virtual_server_service

        assert not hasattr(virtual_server_service, "_nginx_reload_lock"), (
            "virtual_server_service should no longer define its own lock; "
            "it must use nginx_service.reload_lock"
        )

    def test_lock_is_asyncio_lock(self) -> None:
        """Sanity check on the lock type."""
        assert isinstance(nginx_service.reload_lock, asyncio.Lock)
