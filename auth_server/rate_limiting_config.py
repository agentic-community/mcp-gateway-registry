"""Auth-server-side rate-limiting configuration and limiter singleton.

The enforcement runs inside the auth-server ``/validate`` path, which reads
module-level ``os.environ`` constants (the convention in ``server.py``). This
module owns those constants and lazily builds a single shared ``RateLimiter``
(and its DocumentDB backend + definitions repository) on first use.

Everything is gated by ``RATE_LIMITING_ENABLED`` (default false), so importing
this module has no effect until an operator opts in.
"""

import logging
import os

from registry.rate_limiting.definitions_repository import DefinitionsRepository
from registry.rate_limiting.documentdb_backend import DocumentDBRateLimiterBackend
from registry.rate_limiting.limiter import RateLimiter
from registry.rate_limiting.memberships_repository import MembershipsRepository

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


def _env_bool(
    name: str,
    default: str,
) -> bool:
    """Parse a boolean environment variable ('true'/'false', case-insensitive)."""
    return os.environ.get(name, default).strip().lower() == "true"


def _env_int(
    name: str,
    default: int,
) -> int:
    """Parse an integer environment variable, falling back to ``default`` on error."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"invalid int for {name}={raw!r}; using default {default}")
        return default


# Master switch. Off = no checks, current behavior (backwards compatible).
RATE_LIMITING_ENABLED: bool = _env_bool("RATE_LIMITING_ENABLED", "false")

# Counter backend selector. Only "documentdb" is implemented in v1.
RATE_LIMIT_BACKEND: str = os.environ.get("RATE_LIMIT_BACKEND", "documentdb").strip().lower()

# Global fail-open on backend error (per-limit fail_closed overrides to closed).
RATE_LIMIT_FAIL_OPEN: bool = _env_bool("RATE_LIMIT_FAIL_OPEN", "true")

# When true, a backend error reading quarantine membership DENIES (fails closed)
# instead of the default fail-open. Lets an operator make quarantine a hard block
# at the cost of denying data-plane traffic during a memberships-store outage.
# Default false: quarantine is best-effort, NOT breach containment -- pair it with
# IdP credential revocation.
RATE_LIMIT_QUARANTINE_FAIL_CLOSED: bool = _env_bool("RATE_LIMIT_QUARANTINE_FAIL_CLOSED", "false")

# In-process definitions cache TTL (seconds).
RATE_LIMIT_DEFINITIONS_CACHE_TTL_SECONDS: int = _env_int(
    "RATE_LIMIT_DEFINITIONS_CACHE_TTL_SECONDS", 30
)

# Hard timeout per counter op (milliseconds); a slow store fails fast fail-open.
RATE_LIMIT_BACKEND_TIMEOUT_MS: int = _env_int("RATE_LIMIT_BACKEND_TIMEOUT_MS", 250)


# Lazily-built singleton so import is side-effect free and the DB client is only
# touched on the first enforced call.
_rate_limiter: RateLimiter | None = None


def _build_backend() -> DocumentDBRateLimiterBackend:
    """Build the configured counter backend. Only DocumentDB in v1."""
    if RATE_LIMIT_BACKEND != "documentdb":
        logger.warning(
            f"RATE_LIMIT_BACKEND={RATE_LIMIT_BACKEND!r} not implemented; using 'documentdb'"
        )
    return DocumentDBRateLimiterBackend()


def get_rate_limiter() -> RateLimiter:
    """Return the shared RateLimiter singleton, building it on first use."""
    global _rate_limiter
    if _rate_limiter is None:
        backend = _build_backend()
        cache_ttl = float(RATE_LIMIT_DEFINITIONS_CACHE_TTL_SECONDS)
        definitions = DefinitionsRepository(cache_ttl_seconds=cache_ttl)
        memberships = MembershipsRepository(cache_ttl_seconds=cache_ttl)
        _rate_limiter = RateLimiter(
            backend=backend,
            definitions=definitions,
            memberships=memberships,
            fail_open=RATE_LIMIT_FAIL_OPEN,
            backend_timeout_seconds=RATE_LIMIT_BACKEND_TIMEOUT_MS / 1000.0,
            quarantine_fail_closed=RATE_LIMIT_QUARANTINE_FAIL_CLOSED,
        )
        logger.info(
            "Rate limiter initialized: backend=%s, fail_open=%s, quarantine_fail_closed=%s, "
            "cache_ttl=%ss, timeout=%sms",
            RATE_LIMIT_BACKEND,
            RATE_LIMIT_FAIL_OPEN,
            RATE_LIMIT_QUARANTINE_FAIL_CLOSED,
            RATE_LIMIT_DEFINITIONS_CACHE_TTL_SECONDS,
            RATE_LIMIT_BACKEND_TIMEOUT_MS,
        )
    return _rate_limiter
