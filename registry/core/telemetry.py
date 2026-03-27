"""
Anonymous telemetry module for tracking registry adoption.

Privacy-first design:
- Opt-out by default (telemetry ON but easy to disable)
- No PII: no IP addresses, hostnames, file paths, or user data
- Conspicuous disclosure at every startup
- Fail-silent: never impact registry operation
- Cloud-agnostic: no dependency on any specific provider
"""

import asyncio
import json
import logging
import os
import platform
import sys
import uuid
from datetime import UTC, datetime, timedelta

import httpx

from registry.core.config import settings
from registry.version import __version__

logger = logging.getLogger(__name__)

# Telemetry constants
STARTUP_LOCK_INTERVAL_SECONDS = 60  # Don't send startup ping more than once per minute
HEARTBEAT_INTERVAL_HOURS = 24  # Send heartbeat once per day
HEARTBEAT_LOCK_INTERVAL_SECONDS = HEARTBEAT_INTERVAL_HOURS * 3600
TELEMETRY_TIMEOUT_SECONDS = 5  # HTTP request timeout


def _is_telemetry_enabled() -> bool:
    """Check if telemetry is enabled (respects MCP_TELEMETRY_DISABLED env var)."""
    # Environment variable override takes precedence
    disabled_env = os.getenv("MCP_TELEMETRY_DISABLED", "").lower()
    if disabled_env in ("1", "true", "yes"):
        return False

    return settings.telemetry_enabled


def _is_opt_in_enabled() -> bool:
    """Check if opt-in telemetry (heartbeat) is enabled."""
    # Must have base telemetry enabled
    if not _is_telemetry_enabled():
        return False

    # Check environment variable override
    opt_in_env = os.getenv("MCP_TELEMETRY_OPT_IN", "").lower()
    if opt_in_env in ("1", "true", "yes"):
        return True

    return settings.telemetry_opt_in


async def _get_or_create_instance_id() -> str:
    """
    Get or create anonymous instance ID.

    - For MongoDB/DocumentDB: Store in _telemetry_state collection
    - For file-based storage: Store in {data_dir}/.telemetry_id

    Returns:
        UUID v4 string (e.g., "a1b2c3d4-e5f6-7890-abcd-ef1234567890")
    """
    if settings.storage_backend in ("mongodb-ce", "documentdb"):
        # MongoDB-based storage
        from registry.repositories.documentdb.client import get_documentdb_client

        try:
            db = await get_documentdb_client()
            collection = db["_telemetry_state"]

            # Try to get existing document
            doc = await collection.find_one({"_id": "telemetry_config"})

            if doc and "instance_id" in doc:
                return doc["instance_id"]

            # Create new instance ID
            instance_id = str(uuid.uuid4())
            now = datetime.now(UTC).isoformat()

            # Insert or update
            await collection.update_one(
                {"_id": "telemetry_config"},
                {"$setOnInsert": {"instance_id": instance_id, "created_at": now}},
                upsert=True,
            )

            return instance_id

        except Exception as e:
            logger.warning(f"Failed to get instance ID from MongoDB: {e}")
            # Fall through to file-based fallback

    # File-based fallback
    telemetry_file = settings.data_dir / ".telemetry_id"

    try:
        # Ensure data directory exists
        settings.data_dir.mkdir(parents=True, exist_ok=True)

        if telemetry_file.exists():
            instance_id = telemetry_file.read_text().strip()
            if instance_id:
                return instance_id

        # Create new instance ID
        instance_id = str(uuid.uuid4())
        telemetry_file.write_text(instance_id)
        return instance_id

    except Exception as e:
        logger.warning(f"Failed to read/write telemetry ID file: {e}")
        # Last resort: generate ephemeral ID (will be different each startup)
        return str(uuid.uuid4())


async def _acquire_telemetry_lock(event_type: str, interval_seconds: int) -> bool:
    """
    Acquire a distributed lock for sending telemetry.

    Uses MongoDB findOneAndUpdate with a staleness check to ensure
    only one replica sends telemetry within the interval window.

    Args:
        event_type: "startup" or "heartbeat"
        interval_seconds: Lock interval (e.g., 60 for startup, 86400 for heartbeat)

    Returns:
        True if lock acquired (caller should send), False if already sent recently
    """
    if settings.storage_backend not in ("mongodb-ce", "documentdb"):
        # File-based storage: no multi-replica concerns, always allow
        return True

    try:
        from registry.repositories.documentdb.client import get_documentdb_client

        db = await get_documentdb_client()
        collection = db["_telemetry_state"]

        now = datetime.now(UTC)
        cutoff = now - timedelta(seconds=interval_seconds)

        field_name = f"last_{event_type}_sent_at"

        # Atomic update: only update if last sent is None or older than cutoff
        # NOTE: Use BSON datetime objects for proper comparison (not ISO-8601 strings)
        result = await collection.find_one_and_update(
            {
                "_id": "telemetry_config",
                "$or": [
                    {field_name: {"$exists": False}},
                    {field_name: None},
                    {field_name: {"$lt": cutoff}},  # Motor converts datetime to BSON date
                ],
            },
            {"$set": {field_name: now}},  # Store as BSON datetime
            upsert=False,
        )

        # Lock acquired if document was found and updated
        return result is not None

    except Exception as e:
        logger.warning(f"Failed to acquire telemetry lock: {e}")
        # If lock mechanism fails, don't block telemetry
        return True


async def _build_startup_payload() -> dict:
    """Build the anonymous startup event payload."""
    from registry.repositories.stats_repository import get_search_count

    search_queries_total = await get_search_count()

    return {
        "event": "startup",
        "schema_version": "1",
        "v": __version__,
        "py": f"{sys.version_info.major}.{sys.version_info.minor}",
        "os": platform.system().lower(),  # linux, darwin, windows
        "arch": platform.machine(),  # x86_64, arm64, aarch64
        "mode": settings.deployment_mode.value,  # with-gateway, registry-only
        "registry_mode": settings.registry_mode.value,  # full, skills-only, etc.
        "storage": settings.storage_backend,  # file, documentdb, mongodb-ce
        "auth": settings.auth_provider,  # cognito, keycloak, entra, github, google
        "federation": settings.federation_static_token_auth_enabled,
        "search_queries_total": search_queries_total,
        "ts": datetime.now(UTC).isoformat(),
    }


async def _build_heartbeat_payload() -> dict:
    """Build the richer opt-in heartbeat payload with aggregate counts."""
    from registry.api.system_routes import get_server_start_time
    from registry.repositories.factory import (
        get_agent_repository,
        get_peer_federation_repository,
        get_server_repository,
        get_skill_repository,
    )
    from registry.repositories.stats_repository import get_search_count

    # Calculate uptime
    uptime_hours = 0
    server_start_time = get_server_start_time()
    if server_start_time:
        elapsed = datetime.now(UTC) - server_start_time
        uptime_hours = int(elapsed.total_seconds() / 3600)

    # Get aggregate counts (with detailed error logging)
    try:
        server_repo = get_server_repository()
        servers = await server_repo.list_all()
        servers_count = len(servers)
    except Exception as e:
        logger.warning(f"[telemetry] Failed to get server count: {e}")
        servers_count = 0

    try:
        agent_repo = get_agent_repository()
        agents = await agent_repo.list_all()
        agents_count = len(agents)
    except Exception as e:
        logger.warning(f"[telemetry] Failed to get agent count: {e}")
        agents_count = 0

    try:
        skill_repo = get_skill_repository()
        skills = await skill_repo.list_all()
        skills_count = len(skills)
    except Exception as e:
        logger.warning(f"[telemetry] Failed to get skill count: {e}")
        skills_count = 0

    try:
        peer_repo = get_peer_federation_repository()
        peers = await peer_repo.list_peers()
        peers_count = len(peers)
    except Exception as e:
        logger.warning(f"[telemetry] Failed to get peer count: {e}")
        peers_count = 0

    # Determine search backend from storage backend
    # documentdb/mongodb-ce uses DocumentDB search, file uses FAISS
    search_backend = (
        "documentdb" if settings.storage_backend in ("documentdb", "mongodb-ce") else "faiss"
    )

    search_queries_total = await get_search_count()

    return {
        "event": "heartbeat",
        "schema_version": "1",
        "v": __version__,
        "servers_count": servers_count,
        "agents_count": agents_count,
        "skills_count": skills_count,
        "peers_count": peers_count,
        "search_backend": search_backend,
        "embeddings_provider": settings.embeddings_provider,
        "uptime_hours": uptime_hours,
        "search_queries_total": search_queries_total,
        "search_queries_daily_7d_moving_avg": None,
        "search_queries_hourly_moving_avg": None,
        "ts": datetime.now(UTC).isoformat(),
    }


async def _send_telemetry(payload: dict) -> None:
    """
    Send telemetry payload to the collector endpoint.

    - 5-second timeout
    - Fail-silent: log errors but never raise
    - Debug mode: log payload instead of sending

    Args:
        payload: Telemetry event payload (startup or heartbeat)
    """
    # Add instance ID to payload
    try:
        instance_id = await _get_or_create_instance_id()
        payload["instance_id"] = instance_id
    except Exception as e:
        logger.warning(f"Failed to get instance ID: {e}")
        # Continue without instance ID (collector will still accept it)

    # Debug mode: log payload instead of sending
    if settings.telemetry_debug:
        logger.info(f"[telemetry] Debug mode - payload:\n{json.dumps(payload, indent=2)}")
        return

    # Send telemetry with retry logic
    max_retries = 1  # Single retry
    retry_delay = 1.0  # 1 second delay

    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=TELEMETRY_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    settings.telemetry_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code in (200, 204):
                    logger.debug(f"[telemetry] {payload['event']} event sent successfully")

                    # Track success in Datadog
                    from registry.core.metrics import telemetry_sends_total

                    telemetry_sends_total.labels(event=payload["event"], status="success").inc()

                    return  # Success, exit

                else:
                    logger.warning(
                        f"[telemetry] Unexpected response {response.status_code} from collector"
                    )

                    # Track failure in Datadog
                    from registry.core.metrics import telemetry_sends_total

                    status_category = f"{response.status_code // 100}xx"
                    telemetry_sends_total.labels(
                        event=payload["event"], status=status_category
                    ).inc()

        except httpx.TimeoutException:
            logger.debug(f"[telemetry] Request timed out (attempt {attempt + 1}/{max_retries + 1})")

            # Track timeout in Datadog
            from registry.core.metrics import telemetry_sends_total

            telemetry_sends_total.labels(
                event=payload.get("event", "unknown"), status="timeout"
            ).inc()

        except Exception as e:
            logger.debug(
                f"[telemetry] Failed to send (attempt {attempt + 1}/{max_retries + 1}): {e}"
            )

            # Track error in Datadog
            from registry.core.metrics import telemetry_sends_total

            telemetry_sends_total.labels(
                event=payload.get("event", "unknown"), status="error"
            ).inc()

        # Retry after delay (but not on last attempt)
        if attempt < max_retries:
            await asyncio.sleep(retry_delay)


async def _initialize_telemetry_collection() -> None:
    """
    Proactively create the _telemetry_state collection with proper schema.

    Called during application startup to ensure MongoDB permissions are correct
    and avoid silent failures on first telemetry send.
    """
    if settings.storage_backend not in ("mongodb-ce", "documentdb"):
        return  # File-based storage, no collection needed

    try:
        from registry.repositories.documentdb.client import get_documentdb_client

        db = await get_documentdb_client()

        # Check if collection exists
        existing_collections = await db.list_collection_names()

        if "_telemetry_state" not in existing_collections:
            # Create collection
            await db.create_collection("_telemetry_state")
            logger.info("[telemetry] Created _telemetry_state collection")

        # Ensure the singleton document exists
        collection = db["_telemetry_state"]
        doc = await collection.find_one({"_id": "telemetry_config"})

        if not doc:
            # Create initial document with instance_id
            instance_id = str(uuid.uuid4())
            now = datetime.now(UTC)

            await collection.insert_one(
                {"_id": "telemetry_config", "instance_id": instance_id, "created_at": now}
            )
            logger.info(f"[telemetry] Initialized instance_id: {instance_id}")

    except Exception as e:
        logger.warning(f"[telemetry] Failed to initialize collection: {e}")
        # Non-fatal: will fall back to lazy creation or file-based storage


# Global scheduler instance
_telemetry_scheduler: "TelemetryScheduler | None" = None


async def initialize_telemetry() -> None:
    """
    Initialize telemetry system (create MongoDB collection, etc.).

    Called during lifespan startup, before send_startup_ping().
    """
    await _initialize_telemetry_collection()


async def send_startup_ping() -> None:
    """
    Send anonymous startup ping (Tier 1 - Opt-Out).

    Called once during lifespan startup. Checks lock to prevent
    duplicate sends in multi-replica deployments.
    """
    if not _is_telemetry_enabled():
        logger.info("[telemetry] Telemetry is disabled")
        return

    # Log conspicuous disclosure
    logger.info(
        "[telemetry] Anonymous usage telemetry is ON. To disable: set MCP_TELEMETRY_DISABLED=1"
    )
    logger.info("[telemetry] Details: https://mcpgateway.io/telemetry")

    try:
        # Acquire lock (60-second interval)
        lock_acquired = await _acquire_telemetry_lock("startup", STARTUP_LOCK_INTERVAL_SECONDS)

        if not lock_acquired:
            logger.debug("[telemetry] Startup ping already sent recently by another replica")
            return

        # Build and send payload
        payload = await _build_startup_payload()
        await _send_telemetry(payload)

    except Exception as e:
        logger.warning(f"[telemetry] Startup ping failed: {e}")


async def start_heartbeat_scheduler() -> None:
    """
    Start the daily heartbeat scheduler (Tier 2 - Opt-In).

    No-op if opt-in not enabled. Called during lifespan startup.
    """
    global _telemetry_scheduler

    if not _is_opt_in_enabled():
        logger.debug("[telemetry] Heartbeat scheduler not enabled (opt-in required)")
        return

    if _telemetry_scheduler is not None:
        logger.warning("[telemetry] Heartbeat scheduler already running")
        return

    _telemetry_scheduler = TelemetryScheduler()
    await _telemetry_scheduler.start()
    logger.info("[telemetry] Enhanced telemetry is ON (opted in)")


async def stop_heartbeat_scheduler() -> None:
    """Stop the heartbeat scheduler. Called during lifespan shutdown."""
    global _telemetry_scheduler

    if _telemetry_scheduler is not None:
        await _telemetry_scheduler.stop()
        _telemetry_scheduler = None


class TelemetryScheduler:
    """
    Background scheduler for daily heartbeat telemetry.

    Follows the same pattern as PeerSyncScheduler.
    """

    def __init__(self):
        self._task: asyncio.Task | None = None
        self._running: bool = False

    async def start(self) -> None:
        """Start the background scheduler."""
        if self._running:
            logger.warning("[telemetry] Heartbeat scheduler already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info("[telemetry] Heartbeat scheduler started")

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
        logger.info("[telemetry] Heartbeat scheduler stopped")

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop that sends heartbeat every 24 hours."""
        logger.info("[telemetry] Heartbeat loop started (24-hour interval)")

        while self._running:
            try:
                await self._send_heartbeat()
            except Exception as e:
                logger.error(f"[telemetry] Error in heartbeat scheduler: {e}", exc_info=True)

            # Wait 24 hours before next heartbeat
            await asyncio.sleep(HEARTBEAT_INTERVAL_HOURS * 3600)

    async def _send_heartbeat(self) -> None:
        """Send heartbeat event if lock acquired."""
        # Acquire lock (24-hour interval)
        lock_acquired = await _acquire_telemetry_lock("heartbeat", HEARTBEAT_LOCK_INTERVAL_SECONDS)

        if not lock_acquired:
            logger.debug("[telemetry] Heartbeat already sent recently by another replica")
            return

        # Build and send payload
        payload = await _build_heartbeat_payload()
        await _send_telemetry(payload)
