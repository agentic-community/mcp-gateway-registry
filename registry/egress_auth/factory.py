"""EgressAuthService factory.

Builds the service singleton from settings + the SecretStore (mirrors the
repositories/secrets factory pattern). The cross-replica replay guard and
refresh lease are backed by the Mongo operational-state repository (no
credential material) -- the config check guarantees a Mongo-family
storage_backend whenever the feature is enabled, so the repo is always present.
"""

import logging

from registry.core.config import settings
from registry.egress_auth.service import EgressAuthService
from registry.secrets.factory import get_secret_store

logger = logging.getLogger(__name__)

_egress_service: EgressAuthService | None = None


class _MongoReplayGuard:
    """ReplayGuard adapter over the Mongo operational-state repository."""

    def __init__(self, repo) -> None:
        self._repo = repo

    async def check_and_consume(self, nonce: str, ttl_seconds: int) -> bool:
        return await self._repo.consume_nonce(nonce, ttl_seconds)


class _MongoLeaseManager:
    """LeaseManager adapter over the Mongo operational-state repository."""

    def __init__(self, repo) -> None:
        self._repo = repo

    async def acquire(self, key: str, holder: str, ttl_seconds: int) -> bool:
        return await self._repo.acquire_lease(key, holder, ttl_seconds)

    async def release(self, key: str, holder: str) -> None:
        await self._repo.release_lease(key, holder)


def get_egress_auth_service() -> EgressAuthService:
    """Get the EgressAuthService singleton."""
    global _egress_service
    if _egress_service is not None:
        return _egress_service

    logger.info("Creating EgressAuthService (secret_store=%s)", settings.secret_store_backend)

    # Cross-replica replay/lease via the Mongo operational-state repo (no
    # credential material). Feature requires a Mongo-family backend, so the
    # import/repo is always usable when we reach here; fall back to the service's
    # in-process defaults only if construction fails (single-replica dev safety).
    replay_guard = None
    lease_manager = None
    try:
        from registry.repositories.documentdb.egress_operational_repository import (
            EgressOperationalRepository,
        )

        repo = EgressOperationalRepository()
        replay_guard = _MongoReplayGuard(repo)
        lease_manager = _MongoLeaseManager(repo)
    except Exception as exc:  # pragma: no cover - dev/file-backend fallback
        logger.warning(
            "Egress operational repo unavailable (%s); using in-process replay/lease "
            "(single-replica only)",
            exc,
        )

    _egress_service = EgressAuthService(
        secret_store=get_secret_store(),
        callback_base_url=settings.egress_oauth_callback_base,
        refresh_skew_seconds=settings.egress_token_refresh_skew_seconds,
        state_ttl_seconds=settings.egress_state_ttl_seconds,
        replay_guard=replay_guard,
        lease_manager=lease_manager,
    )
    return _egress_service


def reset_egress_auth_service() -> None:
    """Reset the singleton. USE ONLY IN TESTS."""
    global _egress_service
    _egress_service = None
