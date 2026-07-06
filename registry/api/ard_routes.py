"""ARD Registry adapter routes (issue #1295, Phase 2).

Exposes the ARD Registry HTTP contract over the existing semantic-search engine:

    POST /api/ard/search   (mandatory)   - ARD SearchRequest -> SearchResponse
    GET  /api/ard/agents   (browse)       - ARD ListResponse over all asset types

Both are JWT-required (``nginx_proxied_auth``) and access-scoped. All errors use
the ARD ``{errorCode, message}`` envelope with static, client-safe messages; the
detail goes to logs only. Per-operation metrics are emitted via ``ard_metrics``.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.exception_handlers import (
    http_exception_handler as _default_http_handler,
)
from fastapi.exception_handlers import (
    request_validation_exception_handler as _default_validation_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..auth.dependencies import nginx_proxied_auth
from ..core.config import settings
from ..observability.meters import (
    ard_access_filtered_total,
    ard_errors_total,
    ard_request_duration_ms,
    ard_requests_total,
    ard_results_returned,
)
from ..schemas.ard_models import ArdListResponse, ArdSearchRequest, ArdSearchResponse
from ..services import ard_search_service
from ..services.ard_search_service import ArdValidationError
from ..services.ard_service import _base_url_from_request

logger = logging.getLogger(__name__)

router = APIRouter()

# Static, client-safe messages keyed by ARD error code. Raw exception text and
# stack traces are NEVER returned to the client; they go to logs only.
_ARD_MESSAGES = {
    "UNAUTHENTICATED": "Authentication required.",
    "INVALID_REQUEST": "The request was invalid.",
    "RATE_LIMITED": "Rate limit exceeded.",
    "NOT_FOUND": "Resource not found.",
    "INTERNAL": "Internal error.",
}

# Map an arbitrary HTTP status (e.g. from the auth dependency) to an ARD code.
_STATUS_TO_CODE = {
    400: "INVALID_REQUEST",
    401: "UNAUTHENTICATED",
    403: "UNAUTHENTICATED",
    404: "NOT_FOUND",
    422: "INVALID_REQUEST",
    429: "RATE_LIMITED",
}


class ARDHTTPException(HTTPException):
    """HTTPException carrying an ARD error code. ``detail`` is the client-safe
    message; ``log_detail`` is server-only."""

    def __init__(self, status_code: int, error_code: str, log_detail: str | None = None):
        super().__init__(status_code=status_code, detail=_ARD_MESSAGES.get(error_code, "Error."))
        self.error_code = error_code
        self.log_detail = log_detail


# ---------------------------------------------------------------------------
# Per-operation metrics
# ---------------------------------------------------------------------------


@dataclass
class _ArdMetricCtx:
    returned: int | None = None
    filtered: int = 0


@contextmanager
def ard_metrics(operation: str, federation: str = "n_a"):
    """Record duration + success/error + error_code for an ARD operation, so
    every ARD operation has its own metric series."""
    start = time.monotonic()
    ctx = _ArdMetricCtx()
    try:
        yield ctx
        ard_requests_total.add(
            1, {"operation": operation, "status": "success", "federation": federation}
        )
    except ARDHTTPException as e:
        ard_requests_total.add(
            1, {"operation": operation, "status": "error", "federation": federation}
        )
        ard_errors_total.add(1, {"operation": operation, "error_code": e.error_code})
        raise
    except Exception:
        ard_requests_total.add(
            1, {"operation": operation, "status": "error", "federation": federation}
        )
        ard_errors_total.add(1, {"operation": operation, "error_code": "INTERNAL"})
        raise
    finally:
        ard_request_duration_ms.record(
            (time.monotonic() - start) * 1000.0, {"operation": operation}
        )
        if ctx.returned is not None:
            ard_results_returned.record(ctx.returned, {"operation": operation})
        if ctx.filtered:
            ard_access_filtered_total.add(ctx.filtered, {"operation": operation})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/search", response_model=ArdSearchResponse)
async def ard_search(
    http_request: Request,
    body: ArdSearchRequest,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
) -> ArdSearchResponse:
    """ARD POST /search: own-index, access-scoped results (Phase 2)."""
    if not settings.ard_registry_enabled:
        raise ARDHTTPException(404, "NOT_FOUND")
    with ard_metrics("search", federation=body.federation) as m:
        try:
            offset = ard_search_service.decode_page_token(body.page_token)
            entity_types, tags = ard_search_service.filter_to_engine(body.query.filter)
        except ArdValidationError as e:
            raise ARDHTTPException(400, "INVALID_REQUEST", str(e)) from e
        window = offset + body.page_size
        source = f"{_base_url_from_request(http_request)}/api/ard/search"
        try:
            results, scoped_out, referrals = await ard_search_service.search_and_scope(
                body.query.text, entity_types, tags, window, user_context, source,
                federation=body.federation,
            )
        except Exception as e:  # noqa: BLE001 - reshape to ARD envelope, detail to logs
            raise ARDHTTPException(500, "INTERNAL", repr(e)) from e
        page = results[offset : offset + body.page_size]
        next_token = (
            ard_search_service.encode_page_token(offset + body.page_size)
            if offset + body.page_size < len(results)
            else None
        )
        m.returned, m.filtered = len(page), scoped_out
        # Federation (#1296): auto -> unified index source-tagged; none -> local
        # only; referrals -> local results + application/ai-registry+json peers.
        return ArdSearchResponse(results=page, referrals=referrals, page_token=next_token)


@router.get("/agents", response_model=ArdListResponse)
async def ard_browse(
    http_request: Request,
    user_context: Annotated[dict, Depends(nginx_proxied_auth)],
    filter: Annotated[list[str], Query()] = [],  # noqa: B006 - FastAPI Query default
    orderBy: str = Query(default="identifier"),
    pageSize: int = Query(default=20, ge=1, le=100),
    pageToken: str | None = Query(default=None),
) -> ArdListResponse:
    """ARD GET /agents: browse ALL asset types (servers + agents + skills),
    access-scoped, deterministic order, paginated."""
    if not settings.ard_registry_enabled:
        raise ARDHTTPException(404, "NOT_FOUND")
    with ard_metrics("browse") as m:
        try:
            offset = ard_search_service.decode_page_token(pageToken)
        except ArdValidationError as e:
            raise ARDHTTPException(400, "INVALID_REQUEST", str(e)) from e
        try:
            items, total = await ard_search_service.browse(
                filter_pairs=filter,
                order_by=orderBy,
                offset=offset,
                limit=pageSize,
                user_context=user_context,
                base_url=_base_url_from_request(http_request),
            )
        except ArdValidationError as e:
            raise ARDHTTPException(400, "INVALID_REQUEST", str(e)) from e
        except Exception as e:  # noqa: BLE001
            raise ARDHTTPException(500, "INTERNAL", repr(e)) from e
        next_token = (
            ard_search_service.encode_page_token(offset + pageSize)
            if offset + pageSize < total
            else None
        )
        m.returned = len(items)
        return ArdListResponse(items=items, total=total, page_token=next_token)


# ---------------------------------------------------------------------------
# Exception handlers (registered on the app in main.py) — ARD envelope for
# /api/ard/*; default behavior everywhere else.
# ---------------------------------------------------------------------------


async def ard_http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Reshape HTTPExceptions on /api/ard/* into the ARD envelope (incl. the auth
    401, which is therefore a clean ARD error, never a login redirect)."""
    if request.url.path.startswith("/api/ard"):
        if isinstance(exc, ARDHTTPException):
            error_code = exc.error_code
            if exc.log_detail:
                logger.warning("ard error %s: %s", error_code, exc.log_detail)
        else:
            error_code = _STATUS_TO_CODE.get(exc.status_code, "INTERNAL")
        return JSONResponse(
            status_code=exc.status_code,
            content={"errorCode": error_code, "message": _ARD_MESSAGES.get(error_code, "Error.")},
        )
    return await _default_http_handler(request, exc)


async def ard_validation_exception_handler(request: Request, exc: RequestValidationError):
    """Reshape request-validation errors on /api/ard/* into a generic ARD 400
    (no raw exception text to the client; detail to logs)."""
    if request.url.path.startswith("/api/ard"):
        logger.warning("ard validation error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=400,
            content={"errorCode": "INVALID_REQUEST", "message": _ARD_MESSAGES["INVALID_REQUEST"]},
        )
    return await _default_validation_handler(request, exc)
