"""Background scheduler for periodic ARD ai-catalog.json ingestion (issue #1296).

Mirrors :class:`PeerSyncScheduler`: every check interval it reads the enabled
``ai_catalog`` sources from the DB federation config and triggers ingestion for
each source whose interval has elapsed. Per-source overlap is prevented by the
ingestion service's in-process lock; run this scheduler on a single replica
(parity with peer sync).
"""

import asyncio
import logging
from datetime import UTC, datetime

from .ard_ingestion_service import get_ard_ingestion_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

SCHEDULER_CHECK_INTERVAL_SECONDS: int = 60


class ArdIngestionScheduler:
    """Background scheduler for ARD ai-catalog ingestion."""

    def __init__(self):
        self._task: asyncio.Task | None = None
        self._running: bool = False
        self._last_run: dict[str, datetime] = {}

    async def start(self) -> None:
        """Start the background scheduler."""
        if self._running:
            logger.warning("ARD ingestion scheduler already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info("ARD ingestion scheduler started")

    async def stop(self) -> None:
        """Stop the background scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("ARD ingestion scheduler stopped")

    async def _scheduler_loop(self) -> None:
        logger.info(
            "ARD ingestion scheduler loop started, checking every %ds",
            SCHEDULER_CHECK_INTERVAL_SECONDS,
        )
        while self._running:
            try:
                await self._check_and_ingest()
            except Exception as e:  # noqa: BLE001
                logger.error("Error in ARD ingestion scheduler: %s", e, exc_info=True)
            await asyncio.sleep(SCHEDULER_CHECK_INTERVAL_SECONDS)

    async def _check_and_ingest(self) -> None:
        """Trigger ingestion for each enabled source whose interval has elapsed."""
        service = get_ard_ingestion_service()
        cfg = await service.get_config()
        if not cfg.enabled or not cfg.sources:
            return
        now = datetime.now(UTC)
        for source in cfg.sources:
            last = self._last_run.get(source.source_id)
            if last is not None:
                elapsed_minutes = (now - last).total_seconds() / 60
                if elapsed_minutes < cfg.sync_interval_minutes:
                    continue
            logger.info(
                "Scheduled ARD ingestion triggered for source '%s' (interval: %dm)",
                source.source_id, cfg.sync_interval_minutes,
            )
            self._last_run[source.source_id] = now
            try:
                result = await service.ingest_source(source, cfg)
                if result.success:
                    logger.info(
                        "Scheduled ARD ingestion completed for '%s': %d servers, %d agents",
                        source.source_id, result.servers_synced, result.agents_synced,
                    )
                else:
                    logger.warning(
                        "Scheduled ARD ingestion failed for '%s': %s",
                        source.source_id, result.error_message,
                    )
            except Exception as e:  # noqa: BLE001
                logger.error("Error during scheduled ARD ingestion for '%s': %s", source.source_id, e)


_scheduler: ArdIngestionScheduler | None = None


def get_ard_ingestion_scheduler() -> ArdIngestionScheduler:
    """Get or create the global ARD ingestion scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = ArdIngestionScheduler()
    return _scheduler
