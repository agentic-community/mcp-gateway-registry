# registry/services/ans_sync_scheduler.py

import asyncio
import logging

from registry.core.config import settings
from registry.services.ans_service import sync_all_ans_status

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

SCHEDULER_CHECK_INTERVAL_SECONDS: int = 300


class ANSSyncScheduler:
    """Background scheduler for ANS verification status sync."""

    def __init__(self):
        self._task: asyncio.Task | None = None
        self._running: bool = False

    async def start(self) -> None:
        """Start the ANS sync scheduler."""
        if not settings.ans_integration_enabled:
            logger.info("ANS integration disabled, skipping scheduler start")
            return

        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info(
            f"ANS sync scheduler started (interval: {settings.ans_sync_interval_hours} hours)"
        )

    async def stop(self) -> None:
        """Stop the ANS sync scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("ANS sync scheduler stopped")

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        interval_seconds = settings.ans_sync_interval_hours * 3600

        while self._running:
            try:
                await asyncio.sleep(interval_seconds)
                if not self._running:
                    break
                logger.info("ANS sync scheduler: starting sync cycle")
                stats = await sync_all_ans_status()
                logger.info(f"ANS sync scheduler: cycle complete - {stats.model_dump()}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"ANS sync scheduler error: {e}")


_scheduler: ANSSyncScheduler | None = None


def get_ans_sync_scheduler() -> ANSSyncScheduler:
    """Get the global ANS sync scheduler singleton."""
    global _scheduler
    if _scheduler is None:
        _scheduler = ANSSyncScheduler()
    return _scheduler
