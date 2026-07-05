from fastapi import APIRouter, HTTPException, Depends, Request, Response
from typing import List, Dict, Any, Optional
import uuid
import logging
from ..config import settings
from ..core.models import MetricRequest, MetricResponse, ErrorResponse
from ..core.processor import MetricsProcessor
from ..core.retention import retention_manager
from ..core.rate_limiter import ip_throttle
from ..api.auth import verify_api_key, get_rate_limit_status
from ..utils.helpers import generate_request_id, generate_api_key, hash_api_key
from ..storage.database import MetricsStorage

router = APIRouter()
logger = logging.getLogger(__name__)
processor = MetricsProcessor()


def _client_ip(request: Request) -> str:
    """Return a best-effort client IP for per-caller throttling.

    Uses the direct socket peer only. We deliberately do NOT trust
    ``X-Forwarded-For`` here: an attacker could rotate spoofed values to evade
    the throttle. In deployments behind a trusted proxy the socket peer is the
    proxy, which still bounds aggregate abuse.

    Args:
        request: The incoming request.

    Returns:
        The client host string, or "unknown" if it cannot be determined.
    """
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


@router.post("/metrics", response_model=MetricResponse)
async def collect_metrics(
    metric_request: MetricRequest,
    request: Request,
    response: Response,
    api_key: str = Depends(verify_api_key),
):
    """Collect metrics from MCP components."""
    request_id = generate_request_id()

    try:
        # Add rate limit headers
        if hasattr(request.state, "rate_limit_remaining") and hasattr(
            request.state, "rate_limit_limit"
        ):
            response.headers["X-RateLimit-Limit"] = str(request.state.rate_limit_limit)
            response.headers["X-RateLimit-Remaining"] = str(request.state.rate_limit_remaining)

        # Process metrics
        result = await processor.process_metrics(metric_request, request_id, api_key)

        logger.info(
            f"Processed {result.accepted} metrics from {metric_request.service} (request: {request_id})"
        )

        return MetricResponse(
            status="success",
            accepted=result.accepted,
            rejected=result.rejected,
            errors=result.errors,
            request_id=request_id,
        )

    except Exception as e:
        logger.exception("Error processing metrics")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/flush")
async def flush_metrics(
    request: Request, response: Response, api_key: str = Depends(verify_api_key)
):
    """Force flush buffered metrics to storage."""
    try:
        # Add rate limit headers
        if hasattr(request.state, "rate_limit_remaining") and hasattr(
            request.state, "rate_limit_limit"
        ):
            response.headers["X-RateLimit-Limit"] = str(request.state.rate_limit_limit)
            response.headers["X-RateLimit-Remaining"] = str(request.state.rate_limit_remaining)

        await processor.force_flush()
        return {"status": "success", "message": "Metrics flushed to storage"}
    except Exception as e:
        logger.exception("Error flushing metrics")
        raise HTTPException(status_code=500, detail="Failed to flush metrics")


@router.get("/rate-limit")
async def get_rate_limit(request: Request):
    """Get current rate limit status for the caller's own API key.

    This endpoint accepts an unauthenticated request before it can verify the
    key, so it is throttled per client IP to prevent it being used as a
    high-throughput key-validity oracle. All failure modes (missing key,
    throttled, invalid key, misconfiguration) return without revealing whether
    a given key exists beyond the standard 401 that every authenticated
    endpoint already returns for an unknown key.
    """
    # Throttle by client IP before doing any key work. Fail closed: a throttle
    # breach returns 429 regardless of whether the key is valid.
    if not await ip_throttle.allow(
        _client_ip(request),
        settings.RATE_LIMIT_ENDPOINT_MAX_REQUESTS,
        settings.RATE_LIMIT_ENDPOINT_WINDOW_SECONDS,
    ):
        raise HTTPException(
            status_code=429,
            detail="Too many requests",
            headers={"Retry-After": str(settings.RATE_LIMIT_ENDPOINT_WINDOW_SECONDS)},
        )

    api_key = request.headers.get("X-API-Key")

    if not api_key:
        raise HTTPException(status_code=401, detail="API key required in X-API-Key header")

    try:
        return await get_rate_limit_status(api_key)
    except HTTPException:
        # Preserve intentional auth/misconfig status codes (401/503); do not
        # rewrite them to 500.
        raise
    except Exception:
        logger.exception("Error getting rate limit status")
        raise HTTPException(status_code=500, detail="Failed to get rate limit status")


@router.get("/admin/retention/preview")
async def get_cleanup_preview(
    table_name: str | None = None, api_key: str = Depends(verify_api_key)
):
    """Preview what would be cleaned up by retention policies.

    Args:
        table_name: Optional table name to preview. Must be a valid table
            with a configured retention policy.
        api_key: API key for authentication.

    Raises:
        HTTPException: 400 if table_name is not in the allowlist.
        HTTPException: 404 if table_name has no retention policy.
        HTTPException: 500 for other errors.
    """
    try:
        preview = await retention_manager.get_cleanup_preview(table_name)
        return preview
    except ValueError as e:
        # Security: Invalid table name - not in allowlist. Log the detail
        # server-side but return static text so the response does not disclose
        # the set of valid table names.
        logger.warning(f"Invalid table name in cleanup preview request: {e}")
        raise HTTPException(status_code=400, detail="Invalid table name")
    except KeyError as e:
        logger.warning(f"Table not found in cleanup preview request: {e}")
        raise HTTPException(status_code=404, detail="No retention policy for table")
    except Exception:
        logger.exception("Error getting cleanup preview")
        raise HTTPException(status_code=500, detail="Failed to get cleanup preview")


@router.post("/admin/retention/cleanup")
async def run_cleanup(
    table_name: str | None = None, dry_run: bool = True, api_key: str = Depends(verify_api_key)
):
    """Run data cleanup according to retention policies.

    Args:
        table_name: Optional table name to clean. Must be in the allowlist.
        dry_run: If True, only preview what would be deleted.
        api_key: API key for authentication.

    Raises:
        HTTPException: 400 if table_name is not in the allowlist.
        HTTPException: 500 for other errors.
    """
    try:
        if table_name:
            result = await retention_manager.cleanup_table(table_name, dry_run)
        else:
            result = await retention_manager.cleanup_all_tables(dry_run)
        return result
    except ValueError as e:
        # Security: Invalid table name - not in allowlist. Static response text.
        logger.warning(f"Invalid table name in cleanup request: {e}")
        raise HTTPException(status_code=400, detail="Invalid table name")
    except Exception:
        logger.exception("Error running cleanup")
        raise HTTPException(status_code=500, detail="Failed to run cleanup")


@router.get("/admin/retention/policies")
async def get_retention_policies(api_key: str = Depends(verify_api_key)):
    """Get current retention policies."""
    try:
        policies = {}
        for name, policy in retention_manager.policies.items():
            policies[name] = {
                "table_name": policy.table_name,
                "retention_days": policy.retention_days,
                "is_active": policy.is_active,
                "timestamp_column": policy.timestamp_column,
            }
        return policies
    except Exception as e:
        logger.exception("Error getting retention policies")
        raise HTTPException(status_code=500, detail="Failed to get retention policies")


@router.put("/admin/retention/policies/{table_name}")
async def update_retention_policy(
    table_name: str,
    retention_days: int,
    is_active: bool = True,
    api_key: str = Depends(verify_api_key),
):
    """Update retention policy for a table.

    Args:
        table_name: The table name. Must be in the allowlist of valid tables.
        retention_days: Number of days to retain data.
        is_active: Whether the policy is active.
        api_key: API key for authentication.

    Raises:
        HTTPException: 400 if table_name is not in the allowlist.
        HTTPException: 500 for other errors.
    """
    try:
        await retention_manager.update_policy(table_name, retention_days, is_active)
        return {
            "status": "success",
            "message": f"Updated retention policy for {table_name}",
            "table_name": table_name,
            "retention_days": retention_days,
            "is_active": is_active,
        }
    except ValueError as e:
        # Security: Invalid table name - not in allowlist. Static response text.
        logger.warning(f"Invalid table name in update policy request: {e}")
        raise HTTPException(status_code=400, detail="Invalid table name")
    except Exception:
        logger.exception("Error updating retention policy")
        raise HTTPException(status_code=500, detail="Failed to update retention policy")


@router.get("/admin/database/stats")
async def get_database_stats(api_key: str = Depends(verify_api_key)):
    """Get database table statistics."""
    try:
        stats = await retention_manager.get_table_stats()
        return stats
    except Exception as e:
        logger.exception("Error getting database stats")
        raise HTTPException(status_code=500, detail="Failed to get database stats")


@router.get("/admin/database/size")
async def get_database_size(api_key: str = Depends(verify_api_key)):
    """Get database size information."""
    try:
        size_info = await retention_manager.get_database_size()
        return size_info
    except Exception as e:
        logger.exception("Error getting database size")
        raise HTTPException(status_code=500, detail="Failed to get database size")
