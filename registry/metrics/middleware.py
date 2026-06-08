"""FastAPI middleware for registry metrics collection.

Tracks registry operations, request headers, and API usage patterns.

Issue #1122 migration: metrics now flow primarily through native OpenTelemetry
``Counter.add()`` / ``Histogram.record()`` calls in-process. The legacy HTTP
POST path to metrics-service is preserved for one release behind the
``METRICS_LEGACY_HTTP_POST=true`` env var. Both paths are removed in 1.26.0.
"""

import asyncio
import logging
import os
import time
from collections.abc import Callable
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from registry.observability.meters import (
    record_emission_path,
    registry_operation_duration_ms,
    registry_operation_total,
    tool_discovery_duration_ms,
    tool_discovery_total,
)

from .client import create_metrics_client
from .utils import extract_headers_for_analysis, hash_user_id

logger = logging.getLogger(__name__)


def _bucket_results_count(count: int) -> str:
    """Map a search results count to a low-cardinality bucket label.

    Avoids creating one time series per unique count value (which can be
    arbitrary). Buckets keep the cardinality bounded.
    """
    if count < 0:
        return "unknown"
    if count == 0:
        return "zero"
    if count <= 10:
        return "low"
    if count <= 50:
        return "med"
    return "high"


class RegistryMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect registry operation and request metrics.

    Tracks:
    - Registry operations (server CRUD, search, health)
    - Request headers for nginx config analysis (legacy path only)
    - API usage patterns
    """

    def __init__(self, app, service_name: str = "registry"):
        super().__init__(app)
        self.metrics_client = create_metrics_client(service_name=service_name)

        # OTel-native emission gate (issue #1122). When the legacy flag is
        # off (the default in 1.25.0+), metrics flow only via the in-process
        # OTel meters in registry/observability/meters.py. The flag is
        # removed in 1.26.0 along with the metrics-service container.
        self.legacy_http_post_enabled = (
            os.getenv("METRICS_LEGACY_HTTP_POST", "false").lower() == "true"
        )

    def extract_operation_info(self, request: Request) -> dict[str, Any]:
        """Extract operation type and resource information from the request."""
        path = request.url.path
        method = request.method

        # Skip non-API endpoints
        if not path.startswith("/api/"):
            return None

        # Determine operation and resource type
        operation = "unknown"
        resource_type = "unknown"
        resource_id = ""

        # Map HTTP methods to operations
        method_mapping = {
            "GET": "read",
            "POST": "create",
            "PUT": "update",
            "PATCH": "update",
            "DELETE": "delete",
        }

        operation = method_mapping.get(method, "unknown")

        # Parse path to determine resource type and ID
        path_parts = [p for p in path.split("/") if p]  # Remove empty parts

        if len(path_parts) >= 2 and path_parts[0] == "api":
            if path_parts[1] == "servers":
                resource_type = "server"
                if len(path_parts) >= 3:
                    resource_id = path_parts[2]
                # Special case for GET /api/servers - this is a list operation
                if method == "GET" and len(path_parts) == 2:
                    operation = "list"
            elif path_parts[1] == "search":
                resource_type = "search"
                operation = "search"
            elif path_parts[1] == "health":
                resource_type = "health"
                operation = "check"
            elif path_parts[1] == "auth":
                resource_type = "auth"
                if len(path_parts) >= 3:
                    if path_parts[2] == "login":
                        operation = "login"
                    elif path_parts[2] == "logout":
                        operation = "logout"
                    elif path_parts[2] == "me":
                        operation = "profile"

        return {
            "operation": operation,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "path": path,
        }

    def extract_user_info(self, request: Request) -> str:
        """Extract user information from request headers or auth context."""
        # Try to get user from various headers
        user_id = request.headers.get("X-User", "")
        if not user_id:
            user_id = request.headers.get("X-Username", "")

        return hash_user_id(user_id)

    def should_track_request(self, request: Request) -> bool:
        """Determine if the request should be tracked for metrics."""
        path = request.url.path

        # Skip static files and non-API endpoints
        if (
            path.startswith("/static/")
            or path.startswith("/favicon.ico")
            or path == "/"
            or path == "/docs"
            or path == "/openapi.json"
        ):
            return False

        return True

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics."""
        # Skip tracking for certain endpoints
        if not self.should_track_request(request):
            return await call_next(request)

        # Start timing
        start_time = time.perf_counter()

        # Extract operation information
        operation_info = self.extract_operation_info(request)
        if not operation_info:
            return await call_next(request)

        # Extract user and header information
        user_hash = self.extract_user_info(request)
        headers_info = extract_headers_for_analysis(dict(request.headers))

        # Process the request
        response = None
        success = False
        error_code = None

        try:
            response = await call_next(request)

            # Work around Starlette BaseHTTPMiddleware bug with 204 responses
            if response.status_code == 204:
                response = Response(status_code=204, headers=dict(response.headers))

            # Determine success based on response status
            success = 200 <= response.status_code < 400

            if not success:
                error_code = str(response.status_code)

        except Exception as e:
            # Handle exceptions during request processing
            success = False
            error_code = type(e).__name__
            logger.error(f"Error in registry request: {e}")
            # Re-raise the exception to maintain normal error handling
            raise

        finally:
            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Native OTel emission (always-on, in-process, non-blocking).
            # Cardinality-controlled attributes: drop resource_id, user_id,
            # error_code (logged separately) from the OTel attribute set.
            otel_attrs = {
                "operation": str(operation_info["operation"]),
                "resource_type": str(operation_info["resource_type"]),
                "success": str(success),
            }
            registry_operation_total.add(1, otel_attrs)
            registry_operation_duration_ms.record(duration_ms, otel_attrs)
            record_emission_path("otel")

            # Legacy HTTP POST path (gated, one-release dual-write)
            if self.legacy_http_post_enabled:
                asyncio.create_task(
                    self._emit_registry_metric_legacy(
                        operation=operation_info["operation"],
                        resource_type=operation_info["resource_type"],
                        success=success,
                        duration_ms=duration_ms,
                        resource_id=operation_info["resource_id"],
                        user_id=user_hash,
                        error_code=error_code,
                    )
                )

                # Headers metric was a debug-only custom metric. Preserved
                # only in the legacy path; not migrated to OTel because it
                # is not used in any dashboard.
                if success and operation_info["resource_type"] != "health":
                    asyncio.create_task(
                        self._emit_headers_metric_legacy(
                            path=operation_info["path"],
                            method=request.method,
                            headers_info=headers_info,
                            status_code=response.status_code if response else 500,
                        )
                    )

            # Emit search-specific OTel metric and (optionally) legacy
            if operation_info["resource_type"] == "search" and success:
                discovery_attrs = {
                    "results_count_bucket": "unknown",  # not available without response body
                }
                tool_discovery_total.add(1, discovery_attrs)
                tool_discovery_duration_ms.record(duration_ms, discovery_attrs)
                record_emission_path("otel")

                if self.legacy_http_post_enabled:
                    asyncio.create_task(
                        self._emit_discovery_metric_from_request_legacy(
                            request=request, duration_ms=duration_ms
                        )
                    )

        return response

    async def _emit_registry_metric_legacy(
        self,
        operation: str,
        resource_type: str,
        success: bool,
        duration_ms: float,
        resource_id: str = "",
        user_id: str = "",
        error_code: str = None,
    ):
        """Legacy HTTP POST path; only invoked when METRICS_LEGACY_HTTP_POST is true."""
        try:
            await self.metrics_client.emit_registry_metric(
                operation=operation,
                resource_type=resource_type,
                success=success,
                duration_ms=duration_ms,
                resource_id=resource_id,
                user_id=user_id,
                error_code=error_code,
            )
            record_emission_path("legacy")
        except Exception as e:
            logger.debug(f"Legacy registry metric emit failed: {e}")

    async def _emit_headers_metric_legacy(
        self,
        path: str,
        method: str,
        headers_info: dict[str, Any],
        status_code: int,
    ):
        """Legacy custom-metric path used only for nginx-config debug analysis."""
        try:
            await self.metrics_client.emit_custom_metric(
                metric_name="request_headers_analysis",
                value=1.0,
                dimensions={
                    "path": path,
                    "method": method,
                    "status_code": status_code,
                    "has_auth": headers_info.get("authorization_present", False),
                    "user_agent_type": headers_info.get("user_agent_type", "unknown"),
                    "content_type": headers_info.get("content_type", "unknown")[:50],
                    "has_origin": headers_info.get("origin", "unknown") != "unknown",
                },
                metadata={"headers_sample": str(headers_info)[:500]},
            )
            record_emission_path("legacy")
        except Exception as e:
            logger.debug(f"Legacy headers metric emit failed: {e}")

    async def _emit_discovery_metric_from_request_legacy(
        self,
        request: Request,
        duration_ms: float,
    ):
        """Legacy HTTP POST path for search/discovery events."""
        try:
            query_params = request.query_params
            query = query_params.get("q", query_params.get("query", "unknown"))
            results_count = -1  # not available from middleware
            await self.metrics_client.emit_discovery_metric(
                query=query, results_count=results_count, duration_ms=duration_ms
            )
            record_emission_path("legacy")
        except Exception as e:
            logger.debug(f"Legacy discovery metric emit failed: {e}")


def add_registry_metrics_middleware(app, service_name: str = "registry"):
    """Convenience function to add registry metrics middleware to a FastAPI app.

    Args:
        app: FastAPI application instance
        service_name: Name of the service for metrics identification
    """
    app.add_middleware(RegistryMetricsMiddleware, service_name=service_name)
    logger.info(f"Registry metrics middleware added for service: {service_name}")
    logger.info(
        "Metrics emission: native OTel (always-on); "
        f"legacy HTTP POST: {os.getenv('METRICS_LEGACY_HTTP_POST', 'false').lower() == 'true'}"
    )
