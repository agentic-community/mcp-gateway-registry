import hmac

from fastapi import HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
from ..config import settings
from ..storage.database import MetricsStorage
from ..utils.helpers import hash_api_key
from ..core.rate_limiter import rate_limiter

logger = logging.getLogger(__name__)
security = HTTPBearer()


async def verify_api_key(request: Request) -> str:
    """Verify API key from X-API-Key header and check rate limits.

    Fails closed: a missing/invalid/inactive key, a rate-limit breach, or a
    server-side hashing error (e.g. the deployment pepper is not configured)
    all result in denial rather than acceptance.
    """
    api_key = request.headers.get("X-API-Key")

    if not api_key:
        raise HTTPException(status_code=401, detail="API key required in X-API-Key header")

    # Hash the provided API key. If the deployment pepper is missing/weak this
    # raises; treat any hashing failure as a server misconfiguration and deny
    # (never fall back to an unpeppered/predictable hash).
    try:
        key_hash = hash_api_key(api_key)
    except ValueError:
        logger.error("API key hashing is misconfigured (METRICS_KEY_PEPPER)")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")

    # Verify against database
    storage = MetricsStorage()
    key_info = await storage.get_api_key(key_hash)

    if not key_info:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not key_info["is_active"]:
        raise HTTPException(status_code=401, detail="API key is inactive")

    # Check rate limit
    rate_limit = key_info.get("rate_limit", 1000)
    allowed, remaining = await rate_limiter.check_rate_limit(key_hash, rate_limit)

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Limit: {rate_limit} requests/minute",
            headers={
                "X-RateLimit-Limit": str(rate_limit),
                "X-RateLimit-Remaining": "0",
                "Retry-After": "60",
            },
        )

    # Update last used timestamp
    await storage.update_api_key_usage(key_hash)

    # Add rate limit headers
    request.state.rate_limit_remaining = remaining
    request.state.rate_limit_limit = rate_limit

    logger.debug(
        f"API key verified for service: {key_info['service_name']}, remaining: {remaining}"
    )
    return key_info["service_name"]


async def verify_admin_api_key(request: Request) -> str:
    """Authorize an ``/admin/*`` request against the dedicated admin API key.

    This is the privilege-separation boundary for the metrics service. Ordinary
    ingest keys authenticate with :func:`verify_api_key` and MUST NOT be able to
    perform admin operations (change retention policies, run cleanup, read
    database stats). ``/admin/*`` routes therefore require a SEPARATE credential,
    the ``METRICS_ADMIN_API_KEY``, presented in the ``X-API-Key`` header.

    Fails closed on every ambiguous or misconfigured state:

    - No/empty ``X-API-Key`` header -> 401.
    - ``METRICS_ADMIN_API_KEY`` unset, empty, whitespace, or too short -> 503
      (admin surface is unusable until an operator configures a strong key; we
      never fall back to accepting an ingest key).
    - Presented key does not match the configured admin key -> 401.

    The comparison is constant-time (:func:`hmac.compare_digest`, on the UTF-8
    bytes of each side so a non-ASCII header cannot raise) so it does not leak
    how many leading characters matched.

    The admin key MUST be configured DISTINCT from every ingest key. Nothing in
    this function can enforce that -- ingest keys live in the database as salted
    hashes, not in a comparable form here -- so if an operator sets the admin key
    equal to an ingest key value, that ingest key would gain admin access. This
    is the operator's responsibility; the shipped IaC (Terraform/CDK) generates
    the admin key independently of any ingest key, and compose operators must do
    the same.

    Args:
        request: The incoming request.

    Returns:
        A static principal label (``"admin"``) on success. Callers use this only
        as an authorization marker, not as an identity.

    Raises:
        HTTPException: 401 if the header is missing or the key does not match;
            503 if no valid admin key is configured.
    """
    presented_key = request.headers.get("X-API-Key")

    if not presented_key:
        raise HTTPException(status_code=401, detail="API key required in X-API-Key header")

    # Load the configured admin key. If it is missing/weak, this raises and we
    # DENY the admin operation rather than falling back to ingest-key auth.
    try:
        admin_key = settings.get_admin_api_key()
    except ValueError:
        logger.error(
            "Admin API key is not configured or is too weak (METRICS_ADMIN_API_KEY); "
            "denying admin request. Configure a strong, distinct admin key to enable /admin/*."
        )
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")

    # Constant-time comparison so match time does not leak matched-prefix length.
    # Compare the UTF-8 bytes of each side: Starlette decodes header values as
    # latin-1, so a header byte > 0x7F yields a non-ASCII str and
    # hmac.compare_digest on two str would raise TypeError (surfacing as an
    # uncaught 500). Encoding both sides keeps the comparison constant-time and
    # yields a clean 401 for any non-matching key, ASCII or not.
    if not hmac.compare_digest(presented_key.encode("utf-8"), admin_key.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid API key")

    logger.debug("Admin API key verified for an /admin/* request")
    return "admin"


async def get_rate_limit_status(api_key: str) -> dict:
    """Get current rate limit status for an API key.

    Fails closed on a hashing misconfiguration (missing/weak pepper) and on an
    unknown key, exactly like :func:`verify_api_key`, so callers cannot use this
    path to distinguish those states from a normal auth failure.
    """
    try:
        key_hash = hash_api_key(api_key)
    except ValueError:
        logger.error("API key hashing is misconfigured (METRICS_KEY_PEPPER)")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")

    # Get key info from database
    storage = MetricsStorage()
    key_info = await storage.get_api_key(key_hash)

    # Return the SAME 401 for both an unknown key and a known-but-inactive key,
    # mirroring verify_api_key. Distinguishing "no such key" from "exists but
    # inactive" would leak that a given key hash is present in the database.
    if not key_info or not key_info["is_active"]:
        raise HTTPException(status_code=401, detail="Invalid API key")

    rate_limit = key_info.get("rate_limit", 1000)
    status = await rate_limiter.get_bucket_status(key_hash, rate_limit)

    return {
        "service": key_info["service_name"],
        "rate_limit": status["rate_limit"],
        "available_tokens": status["available_tokens"],
        "reset_time_seconds": status["reset_time_seconds"],
    }
