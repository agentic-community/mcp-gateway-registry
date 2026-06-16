"""
Background task that checks GitHub for newer registry releases.

Fetches the latest release tag from the GitHub Releases API on startup and
periodically thereafter. Compares against the running ``__version__`` and
caches the result so the admin-only ``GET /api/system/update-check`` endpoint
can read it without I/O.

Design notes:
- Fail-silent: any network/parse error is logged at INFO and never affects
  registry operation (air-gapped safe).
- Dev builds are skipped: when ``BUILD_VERSION`` is unset the running version
  came from ``git describe`` or the ``DEFAULT_VERSION`` fallback, neither of
  which should be nudged.
- The GitHub Releases API URL is hardcoded — it tracks this repo and is not
  operator-configurable.
"""

import asyncio
import json
import logging
import os
from datetime import UTC, datetime

import httpx
from packaging.version import InvalidVersion, Version

from registry.core.config import settings
from registry.version import __version__

logger = logging.getLogger(__name__)

GITHUB_RELEASES_URL = (
    "https://api.github.com/repos/agentic-community/mcp-gateway-registry/releases/latest"
)
HTTP_TIMEOUT_SECONDS = 5
MAX_RESPONSE_BYTES = 256 * 1024


class UpdateCheckState:
    """Cached result of the most recent update check."""

    def __init__(self) -> None:
        self.latest: str | None = None
        self.update_available: bool = False
        self.release_notes_url: str | None = None
        self.checked_at: datetime | None = None

    def to_dict(self) -> dict:
        return {
            "current": __version__,
            "latest": self.latest,
            "update_available": self.update_available,
            "release_notes_url": self.release_notes_url,
            "checked_at": self.checked_at.isoformat() if self.checked_at else None,
            "check_enabled": settings.update_check_enabled,
        }


_state = UpdateCheckState()
_scheduler: "UpdateCheckScheduler | None" = None


def get_state() -> UpdateCheckState:
    """Return the current cached state. Always safe to call."""
    return _state


def _is_dev_build() -> bool:
    """Return True when the running version is a local/dev build.

    Production images set ``BUILD_VERSION`` at Docker build time. When that
    env var is absent the version came from git or the default fallback, and
    we should not nudge.
    """
    return not os.getenv("BUILD_VERSION")


def _parse_release_tag(tag: str) -> Version | None:
    """Parse a GitHub release tag (with or without ``v`` prefix)."""
    if tag.startswith("v"):
        tag = tag[1:]
    try:
        return Version(tag)
    except InvalidVersion:
        return None


async def _fetch_latest_release() -> tuple[str, str] | None:
    """Fetch the latest release from GitHub.

    Returns:
        Tuple of (tag_name, html_url) on success, None on any failure.
    """
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            response = await client.get(
                GITHUB_RELEASES_URL,
                headers={"Accept": "application/vnd.github+json"},
            )
            if response.status_code != 200:
                logger.info(f"[update-check] GitHub releases API returned {response.status_code}")
                return None

            content = response.content[:MAX_RESPONSE_BYTES]
            data = json.loads(content)
            tag = data.get("tag_name")
            html_url = data.get("html_url")
            if not isinstance(tag, str) or not isinstance(html_url, str):
                logger.info("[update-check] Release payload missing tag_name or html_url")
                return None
            if not html_url.startswith(("http://", "https://")):
                logger.info("[update-check] Release html_url has unexpected scheme")
                return None
            return tag, html_url
    except Exception as e:
        logger.info(f"[update-check] Failed to fetch latest release: {type(e).__name__}: {e}")
        return None


async def _run_check_once() -> None:
    """Perform a single update check and update the cached state.

    Never raises. Skips outright on dev builds and when the feature is
    disabled.
    """
    if not settings.update_check_enabled:
        return
    if _is_dev_build():
        logger.debug("[update-check] Skipping: BUILD_VERSION not set (dev build)")
        return

    current = _parse_release_tag(__version__)
    if current is None:
        logger.info(f"[update-check] Skipping: cannot parse current version '{__version__}'")
        return

    fetched = await _fetch_latest_release()
    if fetched is None:
        return
    tag, html_url = fetched

    latest = _parse_release_tag(tag)
    if latest is None:
        logger.info(f"[update-check] Skipping: cannot parse release tag '{tag}'")
        return

    _state.latest = str(latest)
    _state.release_notes_url = html_url
    _state.update_available = latest > current
    _state.checked_at = datetime.now(UTC)

    if _state.update_available:
        logger.info(f"[update-check] Update available: {current} -> {latest} ({html_url})")
    else:
        logger.debug(f"[update-check] Up to date at {current}")


class UpdateCheckScheduler:
    """Background scheduler that runs the update check on a fixed interval."""

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._running: bool = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            f"[update-check] Scheduler started (interval={settings.update_check_interval_hours}h)"
        )

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        interval_seconds = max(1, settings.update_check_interval_hours) * 3600
        # Run once on startup
        try:
            await _run_check_once()
        except Exception as e:  # noqa: BLE001 - fail-silent
            logger.info(f"[update-check] Initial check failed: {type(e).__name__}: {e}")

        while self._running:
            try:
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break
            if not self._running:
                break
            try:
                await _run_check_once()
            except Exception as e:  # noqa: BLE001 - fail-silent
                logger.info(f"[update-check] Periodic check failed: {type(e).__name__}: {e}")


async def start_update_checker() -> None:
    """Start the background update checker. No-op if disabled or dev build."""
    global _scheduler

    if not settings.update_check_enabled:
        logger.info("[update-check] Disabled via UPDATE_CHECK_ENABLED=false")
        return
    if _is_dev_build():
        logger.info("[update-check] Disabled: BUILD_VERSION not set (dev build)")
        return
    if _scheduler is not None:
        return

    _scheduler = UpdateCheckScheduler()
    await _scheduler.start()


async def stop_update_checker() -> None:
    """Stop the background update checker."""
    global _scheduler
    if _scheduler is not None:
        await _scheduler.stop()
        _scheduler = None
