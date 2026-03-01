"""
Background scheduler for periodic peer federation sync.

Uses asyncio to periodically check enabled peers and trigger sync
when their configured interval has elapsed.
"""

import asyncio
import logging
from datetime import datetime
from datetime import timezone
from typing import Optional

from registry.repositories.factory import get_peer_federation_repository
from registry.services.peer_federation_service import PeerFederationService


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


# Check interval in seconds (how often to check if any peer needs sync)
SCHEDULER_CHECK_INTERVAL_SECONDS: int = 60


class PeerSyncScheduler:
    """
    Background scheduler for peer federation sync.

    Periodically checks all enabled peers and triggers sync when
    the configured interval has elapsed since last successful sync.
    """

    def __init__(self):
        self._task: Optional[asyncio.Task] = None
        self._running: bool = False

    async def start(self) -> None:
        """Start the background scheduler."""
        if self._running:
            logger.warning("Peer sync scheduler already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info("Peer sync scheduler started")

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
        logger.info("Peer sync scheduler stopped")

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop that checks peers and triggers sync."""
        logger.info(
            f"Peer sync scheduler loop started, checking every {SCHEDULER_CHECK_INTERVAL_SECONDS}s"
        )

        while self._running:
            try:
                await self._check_and_sync_peers()
            except Exception as e:
                logger.error(f"Error in peer sync scheduler: {e}", exc_info=True)

            # Wait before next check
            await asyncio.sleep(SCHEDULER_CHECK_INTERVAL_SECONDS)

    async def _check_and_sync_peers(self) -> None:
        """Check all peers and sync those that need it."""
        try:
            peer_repo = get_peer_federation_repository()
            peers = await peer_repo.list_peers()

            if not peers:
                return

            federation_service = PeerFederationService()
            now = datetime.now(timezone.utc)

            for peer in peers:
                # Skip disabled peers
                if not peer.enabled:
                    continue

                # Skip peers with no scheduled sync (interval = 0)
                if peer.sync_interval_minutes <= 0:
                    continue

                # Check if sync is needed
                should_sync = await self._should_sync_peer(
                    peer.peer_id, peer.sync_interval_minutes, now
                )

                if should_sync:
                    logger.info(
                        f"Scheduled sync triggered for peer '{peer.peer_id}' "
                        f"(interval: {peer.sync_interval_minutes}m)"
                    )
                    try:
                        result = await federation_service.sync_peer(peer.peer_id)
                        if result.success:
                            logger.info(
                                f"Scheduled sync completed for peer '{peer.peer_id}': "
                                f"{result.servers_synced} servers, {result.agents_synced} agents"
                            )
                        else:
                            logger.warning(
                                f"Scheduled sync failed for peer '{peer.peer_id}': "
                                f"{result.error_message}"
                            )
                    except Exception as e:
                        logger.error(f"Error during scheduled sync for peer '{peer.peer_id}': {e}")

        except Exception as e:
            logger.error(f"Error checking peers for scheduled sync: {e}", exc_info=True)

    async def _should_sync_peer(self, peer_id: str, interval_minutes: int, now: datetime) -> bool:
        """
        Determine if a peer should be synced based on last sync time.

        Args:
            peer_id: The peer identifier
            interval_minutes: Configured sync interval in minutes
            now: Current UTC time

        Returns:
            True if sync should be triggered
        """
        try:
            peer_repo = get_peer_federation_repository()
            status = await peer_repo.get_sync_status(peer_id)

            if not status:
                # No status record means never synced - should sync
                return True

            if status.sync_in_progress:
                # Sync already in progress - skip
                return False

            last_sync = status.last_successful_sync
            if not last_sync:
                # Never successfully synced - should sync
                return True

            # Ensure last_sync is timezone-aware
            if last_sync.tzinfo is None:
                last_sync = last_sync.replace(tzinfo=timezone.utc)

            # Calculate time since last sync
            elapsed_minutes = (now - last_sync).total_seconds() / 60

            return elapsed_minutes >= interval_minutes

        except Exception as e:
            logger.error(f"Error checking sync status for peer '{peer_id}': {e}")
            return False


# Global scheduler instance
_scheduler: Optional[PeerSyncScheduler] = None


def get_peer_sync_scheduler() -> PeerSyncScheduler:
    """Get or create the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = PeerSyncScheduler()
    return _scheduler
