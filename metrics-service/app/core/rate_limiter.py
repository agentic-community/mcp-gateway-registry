"""Rate limiting implementation for API keys."""

import asyncio
import time
import logging
from typing import Dict, Tuple
from datetime import datetime, timedelta
from ..storage.database import MetricsStorage

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter for API keys."""

    def __init__(self):
        # In-memory token buckets: {key_hash: (tokens, last_refill, rate_limit)}
        self._buckets: Dict[str, Tuple[int, float, int]] = {}
        self._lock = asyncio.Lock()

    async def check_rate_limit(self, key_hash: str, rate_limit: int) -> Tuple[bool, int]:
        """
        Check if request is allowed under rate limit.

        Args:
            key_hash: The hashed API key
            rate_limit: Requests per minute limit

        Returns:
            Tuple of (is_allowed, remaining_tokens)
        """
        async with self._lock:
            now = time.time()

            # Get or create bucket
            if key_hash not in self._buckets:
                # New bucket starts full, consume one token for this request
                tokens = rate_limit - 1
                self._buckets[key_hash] = (tokens, now, rate_limit)
                return True, tokens
            else:
                tokens, last_refill, limit = self._buckets[key_hash]

                # Update rate limit if it changed
                if limit != rate_limit:
                    # Scale existing tokens proportionally
                    tokens = int(tokens * (rate_limit / limit))
                    limit = rate_limit

            # Refill tokens based on elapsed time
            time_elapsed = now - last_refill
            minutes_elapsed = time_elapsed / 60.0

            # Add tokens for elapsed time (rate_limit tokens per minute)
            tokens_to_add = int(minutes_elapsed * rate_limit)
            tokens = min(tokens + tokens_to_add, rate_limit)

            # Update last refill time if we added tokens
            if tokens_to_add > 0:
                last_refill = now

            # Check if request is allowed
            if tokens > 0:
                tokens -= 1
                self._buckets[key_hash] = (tokens, last_refill, rate_limit)
                logger.debug(f"Rate limit check passed. Remaining: {tokens}")
                return True, tokens
            else:
                self._buckets[key_hash] = (tokens, last_refill, rate_limit)
                logger.warning(f"Rate limit exceeded for key: {key_hash[:8]}...")
                return False, 0

    async def get_bucket_status(self, key_hash: str, rate_limit: int) -> Dict[str, int]:
        """Get current bucket status without consuming a token."""
        async with self._lock:
            now = time.time()

            if key_hash not in self._buckets:
                return {
                    "available_tokens": rate_limit,
                    "rate_limit": rate_limit,
                    "reset_time_seconds": 0,
                }

            tokens, last_refill, limit = self._buckets[key_hash]

            # Calculate tokens after refill
            time_elapsed = now - last_refill
            minutes_elapsed = time_elapsed / 60.0
            tokens_to_add = int(minutes_elapsed * rate_limit)
            current_tokens = min(tokens + tokens_to_add, rate_limit)

            # Calculate time until bucket is full
            if current_tokens < rate_limit:
                tokens_needed = rate_limit - current_tokens
                reset_time_seconds = int((tokens_needed / rate_limit) * 60)
            else:
                reset_time_seconds = 0

            return {
                "available_tokens": current_tokens,
                "rate_limit": rate_limit,
                "reset_time_seconds": reset_time_seconds,
            }

    async def cleanup_old_buckets(self, max_age_hours: int = 24):
        """Remove buckets that haven't been used recently."""
        async with self._lock:
            now = time.time()
            cutoff = now - (max_age_hours * 3600)

            old_keys = []
            for key_hash, (_, last_refill, _) in self._buckets.items():
                if last_refill < cutoff:
                    old_keys.append(key_hash)

            for key in old_keys:
                del self._buckets[key]

            if old_keys:
                logger.info(f"Cleaned up {len(old_keys)} old rate limit buckets")


# Global rate limiter instance
rate_limiter = RateLimiter()


class IPThrottle:
    """Fixed-window per-client-IP throttle for unauthenticated endpoints.

    Used to blunt abuse of endpoints that must accept a request before it can
    be authenticated (e.g. the ``/rate-limit`` lookup, which would otherwise be
    a high-throughput key-validity oracle). Fails closed: any bookkeeping error
    results in the request being denied.

    This is process-local. In multi-replica deployments it bounds per-replica
    abuse rather than global; a shared store would be required for global
    enforcement (documented as future hardening).
    """

    def __init__(self) -> None:
        # {client_ip: (window_start_epoch, request_count)}
        self._windows: Dict[str, Tuple[float, int]] = {}
        self._lock = asyncio.Lock()

    async def allow(
        self,
        client_ip: str,
        max_requests: int,
        window_seconds: int,
    ) -> bool:
        """Return True iff the client is within its request budget.

        Args:
            client_ip: The caller's IP address (or a stable stand-in).
            max_requests: Maximum requests allowed within the window.
            window_seconds: Length of the fixed window in seconds.

        Returns:
            True if the request is permitted; False if the budget is exhausted.
        """
        async with self._lock:
            now = time.time()
            window_start, count = self._windows.get(client_ip, (now, 0))

            # Reset the window if it has elapsed.
            if now - window_start >= window_seconds:
                window_start, count = now, 0

            if count >= max_requests:
                self._windows[client_ip] = (window_start, count)
                logger.warning("IP throttle exceeded for client")
                return False

            self._windows[client_ip] = (window_start, count + 1)
            return True

    async def cleanup_old_windows(self, max_age_seconds: int = 3600) -> None:
        """Remove windows whose start time is older than the cutoff."""
        async with self._lock:
            cutoff = time.time() - max_age_seconds
            stale = [ip for ip, (start, _) in self._windows.items() if start < cutoff]
            for ip in stale:
                del self._windows[ip]
            if stale:
                logger.info(f"Cleaned up {len(stale)} old IP throttle windows")


# Global IP throttle instance for unauthenticated endpoints
ip_throttle = IPThrottle()
