"""
FastAPI middleware for comprehensive metrics collection in the auth server.

This middleware automatically tracks detailed authentication metrics including:
- Validation steps and scope checking
- Tool access control decisions
- Method/tool usage patterns
- Error analysis with specific reasons
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

# Import metrics client - use HTTP API instead of local import
import httpx
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Dual-path import: in-container the auth_server package is flattened into
# /app (so siblings are imported without the ``auth_server.`` prefix); under
# pytest the package is rooted at the repo so the prefix is required.
try:
    from observability.meters import (
        auth_request_duration_ms,
        auth_request_total,
        protocol_latency_ms,
        record_emission_path,
        tool_execution_duration_ms,
        tool_execution_total,
    )
except ImportError:
    from auth_server.observability.meters import (
        auth_request_duration_ms,
        auth_request_total,
        protocol_latency_ms,
        record_emission_path,
        tool_execution_duration_ms,
        tool_execution_total,
    )

from registry.observability.label_bounding import LabelCardinalityLimiter

logger = logging.getLogger(__name__)

# Tool-execution OTel attributes derived from request data (JSON-RPC body /
# clientInfo) and therefore attacker-influenced. These must be cardinality-
# bounded before reaching the in-process Prometheus instrument, or a client
# sending randomized values explodes the time-series count (DoS). server_name is
# derived from the registry-controlled URI path and success is a boolean, so
# both are left unbounded.
_TOOL_EXECUTION_BOUNDED_ATTRS: frozenset[str] = frozenset(
    {"tool_name", "method", "client_name", "client_version"}
)

# Shared per-process limiter for this module's OTel-native emission path (mirrors
# the metrics-service processor's limiter across the deployable boundary).
_label_limiter = LabelCardinalityLimiter()

# MCP transport endpoints. A data-plane MCP request ends in one of these (e.g.
# "{server}/mcp", "{server}/sse"); its presence is what distinguishes a genuine
# MCP server call from a control-plane path that merely has segments.
_MCP_TRANSPORT_ENDPOINTS: frozenset[str] = frozenset({"mcp", "sse", "messages"})

# Path prefixes that are CONTROL PLANE, not a routed target. These reach
# /validate (the /api/* and static locations set auth_request in nginx) but must
# NEVER be counted as MCP-server or agent routing. This mirrors the auth server's
# own rule that /api/ paths do not yield a server_name (see server.py: the
# `path_parts[0] != "api"` guard). Matched on the FIRST path segment after the
# REGISTRY_ROOT_PATH prefix is stripped.
_CONTROL_PLANE_FIRST_SEGMENTS: frozenset[str] = frozenset({"api", "static", "oauth2"})


# Target-kind classification for the routing metric. Each rule recognizes a
# routed DATA-PLANE target by its URL shape, checked in order (most specific
# first). Anything that matches no rule and is not control plane is "unknown"
# (never silently attributed to MCP servers) -- the classifier is an ALLOWLIST,
# fail-safe by design, so a new/unrecognized route cannot inflate mcp_server.
#
# To track a NEW routed target type in the future: add one entry to
# _TARGET_KIND_RULES with a predicate over the (already root-stripped) path
# segments, and add the matching nginx route. No change to classify/emit logic.
#
# Each rule is (kind_label, predicate) where predicate(path_parts) -> bool.
def _is_a2a_agent(path_parts: list[str]) -> bool:
    # "{root}/agent/{agent_path}/..." -> at least "agent" + one path segment.
    return len(path_parts) >= 2 and path_parts[0] == "agent"


def _is_virtual_server(path_parts: list[str]) -> bool:
    # "{root}/virtual/{id}/{transport}" -> a virtual MCP server data-plane call.
    return len(path_parts) >= 2 and path_parts[0] == "virtual"


def _is_mcp_server(path_parts: list[str]) -> bool:
    # A real MCP server call ends in an MCP transport endpoint
    # ("{server}/mcp", "{server}/sse", ...). Requiring the transport suffix is
    # what keeps control-plane paths (which never carry it) out of mcp_server.
    return len(path_parts) >= 2 and path_parts[-1] in _MCP_TRANSPORT_ENDPOINTS


_TARGET_KIND_RULES: tuple[tuple[str, Any], ...] = (
    ("a2a_agent", _is_a2a_agent),
    ("virtual_mcp_server", _is_virtual_server),
    ("mcp_server", _is_mcp_server),
)


class AuthMetricsMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive middleware to collect detailed authentication and tool execution metrics.

    Tracks:
    - Authentication flow with detailed validation steps
    - Scope checking and access control decisions
    - Tool and method execution patterns
    - Error analysis with specific failure reasons
    - User activity patterns (hashed for privacy)
    """

    def __init__(self, app, service_name: str = "auth-server"):
        super().__init__(app)
        self.service_name = service_name
        self.metrics_url = os.getenv("METRICS_SERVICE_URL", "http://localhost:8890")
        self.api_key = os.getenv("METRICS_API_KEY", "")
        self.client = httpx.AsyncClient(timeout=5.0)

        # OTel-native emission gate (issue #1122). When the legacy flag is
        # off (the default in 1.25.0+), HTTP POSTs to metrics-service are
        # skipped entirely and metrics flow only via the in-process OTel
        # meters declared in auth_server/observability/meters.py. The flag
        # is removed in 1.26.0 along with the metrics-service container.
        self.legacy_http_post_enabled = (
            os.getenv("METRICS_LEGACY_HTTP_POST", "false").lower() == "true"
        )

        # Track request contexts for detailed metrics
        self.request_contexts: dict[str, dict[str, Any]] = {}

        # Track session timings for protocol flow analysis
        self.session_timings: dict[str, dict[str, float]] = {}

        # Track session client info for consistent metrics across requests
        self.session_client_info: dict[str, dict[str, str]] = {}

        # Scalability configuration
        self.max_sessions = 1000  # Limit concurrent sessions
        self.session_ttl = 3600  # 1 hour TTL
        self.cleanup_interval = 300  # Cleanup every 5 minutes
        self.last_cleanup = time.time()

    def hash_username(self, username: str) -> str:
        """Hash username for privacy in metrics."""
        if not username:
            return ""
        return hashlib.sha256(username.encode()).hexdigest()[:12]

    async def _cleanup_sessions_if_needed(self):
        """Perform periodic cleanup of old sessions to prevent memory leaks."""
        current_time = time.time()

        # Only cleanup every cleanup_interval seconds
        if current_time - self.last_cleanup < self.cleanup_interval:
            return

        self.last_cleanup = current_time

        # Clean up old session timings
        sessions_to_remove = []
        for session_key, methods in self.session_timings.items():
            # Remove if all methods are old
            if all(current_time - timestamp > self.session_ttl for timestamp in methods.values()):
                sessions_to_remove.append(session_key)

        # Also remove oldest sessions if we exceed max_sessions
        if len(self.session_timings) > self.max_sessions:
            # Sort by oldest timestamp and remove excess
            session_ages = [
                (session_key, min(methods.values()) if methods else 0)
                for session_key, methods in self.session_timings.items()
            ]
            session_ages.sort(key=lambda x: x[1])
            excess_count = len(self.session_timings) - self.max_sessions
            sessions_to_remove.extend([s[0] for s in session_ages[:excess_count]])

        # Remove sessions
        for session_key in sessions_to_remove:
            self.session_timings.pop(session_key, None)
            self.session_client_info.pop(session_key, None)

        if sessions_to_remove:
            logger.debug(f"Cleaned up {len(sessions_to_remove)} old sessions")

    def extract_server_name_from_url(self, original_url: str) -> str:
        """Extract server name from the original URL."""
        if not original_url:
            return "unknown"

        try:
            from urllib.parse import urlparse

            parsed_url = urlparse(original_url)
            path = parsed_url.path.strip("/")
            path_parts = path.split("/") if path else []
            return path_parts[0] if path_parts else "unknown"
        except Exception:
            return "unknown"

    def classify_target_kind(self, original_url: str) -> str:
        """Classify a validated request by the kind of target it routes to.

        Splits the auth metric by routed data-plane target type so routing
        volume can be tracked per kind. Returns one of:

        - ``a2a_agent``          - an A2A agent reverse-proxy call
        - ``virtual_mcp_server`` - a virtual MCP server call
        - ``mcp_server``         - a (real) MCP server transport call
        - ``control_plane``      - an /api/, static, or oauth2 request (NOT a
                                   routed target: the dashboard, login, config,
                                   skill/agent CRUD, ARD, public endpoints)
        - ``unknown``            - no path, or a shape we do not recognize

        This is an ALLOWLIST classifier: a path is attributed to a data-plane
        target ONLY when it matches an explicit rule. Everything else is
        control_plane/unknown, never silently counted as an MCP server. That
        mirrors the auth server's own rule that ``/api/`` paths yield no
        server_name and do not engage the rate-limit target axis, so this metric
        cannot inadvertently count control-plane API calls as MCP-server traffic.

        Note on skills: skills have no data-plane proxy route -- they are managed
        only via ``/api/skills/...`` CRUD -- so they correctly classify as
        control_plane, not a routed target.

        Path shapes honor an optional ``REGISTRY_ROOT_PATH`` prefix. Kept
        self-contained (not imported from ``server``) to avoid a circular
        import: ``server`` imports this middleware.

        Args:
            original_url: The X-Original-URL header value from nginx.

        Returns:
            The target-kind label.
        """
        if not original_url:
            return "unknown"

        try:
            from urllib.parse import urlparse

            parsed_url = urlparse(original_url)
            path = parsed_url.path.strip("/")

            registry_prefix = os.environ.get("REGISTRY_ROOT_PATH", "").strip("/")
            if registry_prefix and path.startswith(registry_prefix):
                path = path[len(registry_prefix) :].lstrip("/")

            path_parts = path.split("/") if path else []
            if not path_parts:
                return "unknown"

            # Control plane is checked FIRST so an /api/* path can never fall
            # through to a data-plane target label.
            if path_parts[0] in _CONTROL_PLANE_FIRST_SEGMENTS:
                return "control_plane"

            # Allowlist: attribute to a data-plane target only on an explicit match.
            for kind, predicate in _TARGET_KIND_RULES:
                if predicate(path_parts):
                    return kind

            # Recognized as neither control plane nor a known routed target.
            return "unknown"
        except Exception:
            return "unknown"

    async def extract_tool_and_method_info(self, request: Request) -> dict[str, Any]:
        """Extract detailed tool and method information from headers (X-Body) instead of consuming body."""
        tool_info = {
            "method": "unknown",
            "tool_name": None,
            "request_id": None,
            "protocol_version": None,
            "client_info": {},
            "params": {},
        }

        try:
            # Get the request body from X-Body header set by Lua script instead of consuming it
            x_body = request.headers.get("X-Body")
            if x_body:
                request_payload = json.loads(x_body)

                if isinstance(request_payload, dict):
                    tool_info["method"] = request_payload.get("method", "unknown")
                    tool_info["request_id"] = request_payload.get("id")
                    tool_info["jsonrpc"] = request_payload.get("jsonrpc")

                    # Extract parameters
                    params = request_payload.get("params", {})
                    tool_info["params"] = params

                    # For tools/call, extract the actual tool name from params
                    if tool_info["method"] == "tools/call" and isinstance(params, dict):
                        tool_info["tool_name"] = params.get("name", "")

                    # For initialize, extract client info and capabilities
                    elif tool_info["method"] == "initialize" and isinstance(params, dict):
                        tool_info["protocol_version"] = params.get("protocolVersion")
                        tool_info["client_info"] = params.get("clientInfo", {})

        except Exception as e:
            logger.debug(f"Could not extract tool information from X-Body header: {e}")

        return tool_info

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and collect comprehensive metrics.
        """
        # Skip metrics collection for non-validation endpoints
        if not request.url.path.startswith("/validate"):
            return await call_next(request)

        # Start timing and generate request ID
        start_time = time.perf_counter()
        current_timestamp = time.time()
        request_id = f"req_{uuid.uuid4().hex[:16]}"

        # Extract comprehensive request data
        server_name = "unknown"
        user_hash = ""
        auth_method = "unknown"
        tool_info = {}

        # Extract server name from original URL header
        original_url = request.headers.get("X-Original-URL")
        target_kind = "unknown"
        if original_url:
            server_name = self.extract_server_name_from_url(original_url)
            target_kind = self.classify_target_kind(original_url)

        # Extract detailed tool/method information
        tool_info = await self.extract_tool_and_method_info(request)

        # Process the request
        response = None
        success = False
        error_code = None

        try:
            response = await call_next(request)

            # Determine success based on response status
            success = response.status_code == 200

            if success:
                # Extract user info from response headers if available
                username = response.headers.get("X-Username", "")
                user_hash = self.hash_username(username)
                auth_method = response.headers.get("X-Auth-Method", "unknown")

                # Track session timing for protocol flow analysis
                session_key = (
                    f"{server_name}:{user_hash}" if user_hash else f"{server_name}:anonymous"
                )
                method = tool_info.get("method", "unknown")

                # Perform periodic cleanup to prevent memory leaks
                await self._cleanup_sessions_if_needed()

                if session_key not in self.session_timings:
                    self.session_timings[session_key] = {}

                # Store timestamp for this method
                self.session_timings[session_key][method] = current_timestamp

                # Store client info for initialize requests
                if method == "initialize" and tool_info.get("client_info"):
                    self.session_client_info[session_key] = tool_info["client_info"]
            else:
                error_code = str(response.status_code)
                session_key = f"{server_name}:anonymous"

        except Exception as e:
            # Handle exceptions during request processing
            success = False
            error_code = type(e).__name__
            logger.error(f"Error in auth request: {e}")
            # Re-raise the exception to maintain normal error handling
            raise

        finally:
            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Emit comprehensive metrics asynchronously (fire and forget)
            # 1. Main auth metric
            asyncio.create_task(
                self._emit_auth_metric(
                    success=success,
                    method=auth_method,
                    duration_ms=duration_ms,
                    server_name=server_name,
                    target_kind=target_kind,
                    user_hash=user_hash,
                    error_code=error_code,
                    request_id=request_id,
                )
            )

            # 2. Tool execution metric (if applicable)
            if tool_info.get("method") and tool_info["method"] != "unknown":
                asyncio.create_task(
                    self._emit_tool_execution_metric(
                        tool_info=tool_info,
                        server_name=server_name,
                        success=success,
                        duration_ms=duration_ms,
                        user_hash=user_hash,
                        error_code=error_code,
                        request_id=request_id,
                        auth_method=auth_method,
                    )
                )

            # 3. Protocol flow latency metric (if we can calculate it)
            if success and session_key in self.session_timings:
                asyncio.create_task(
                    self._emit_protocol_latency_metric(
                        session_key=session_key,
                        current_method=method,
                        server_name=server_name,
                        user_hash=user_hash,
                        request_id=request_id,
                    )
                )

        return response

    async def _emit_auth_metric(
        self,
        success: bool,
        method: str,
        duration_ms: float,
        server_name: str,
        target_kind: str,
        user_hash: str,
        error_code: str = None,
        request_id: str = None,
    ):
        """Emit authentication metric via OTel and (optionally) legacy HTTP POST.

        Cardinality-controlled OTel attributes: ``success``, ``method``,
        ``server``, ``target_kind``. ``target_kind`` (``a2a_agent`` /
        ``mcp_server`` / ``unknown``) is a bounded label that lets the same
        counter answer "how much traffic routed to agents vs MCP servers?"
        without the unbounded per-target ``server`` name. The legacy
        ``user_hash`` and ``request_id`` dimensions are intentionally not OTel
        attributes; per-user identification stays available in
        auto-instrumentation span attributes for per-request debugging.
        """
        # 1) OTel emission (always-on, in-process, non-blocking)
        otel_attrs = {
            "success": str(success),
            "method": method,
            "server": server_name,
            "target_kind": target_kind,
        }
        auth_request_total.add(1, otel_attrs)
        auth_request_duration_ms.record(duration_ms, otel_attrs)
        record_emission_path("otel")

        # 2) Legacy HTTP POST (one-release dual-write window, issue #1122)
        if not self.legacy_http_post_enabled:
            return
        try:
            if not self.api_key:
                return

            payload = {
                "service": self.service_name,
                "version": "1.0.0",
                "metrics": [
                    {
                        "type": "auth_request",
                        "timestamp": datetime.utcnow().isoformat(),
                        "value": 1.0,
                        "duration_ms": duration_ms,
                        "dimensions": {
                            "success": success,
                            "method": method,
                            "server": server_name,
                            "target_kind": target_kind,
                            "user_hash": user_hash,
                        },
                        "metadata": {
                            "error_code": error_code,
                            "request_id": request_id or f"req_{uuid.uuid4().hex[:16]}",
                        },
                    }
                ],
            }

            await self.client.post(
                f"{self.metrics_url}/metrics", json=payload, headers={"X-API-Key": self.api_key}
            )
            record_emission_path("legacy")
        except httpx.RequestError as e:
            logger.debug(f"Legacy metrics-service POST failed (non-fatal): {e}")
        except Exception as e:
            logger.debug(f"Failed to emit legacy auth metric: {e}")

    async def _emit_tool_execution_metric(
        self,
        tool_info: dict[str, Any],
        server_name: str,
        success: bool,
        duration_ms: float,
        user_hash: str,
        error_code: str = None,
        request_id: str = None,
        auth_method: str = "unknown",
    ):
        """Emit tool execution metric via OTel and (optionally) legacy HTTP POST."""
        # Extract tool/method details (used by both paths)
        method_name = tool_info.get("method", "unknown")
        actual_tool_name = tool_info.get("tool_name")
        client_info = tool_info.get("client_info", {})

        # If no client_info in current request, try to get it from session
        if not client_info or client_info.get("name") == "unknown":
            session_key = f"{server_name}:{user_hash}" if user_hash else f"{server_name}:anonymous"
            stored_client_info = self.session_client_info.get(session_key, {})
            if stored_client_info:
                client_info = stored_client_info

        # 1) OTel emission. Cardinality-controlled: drops user_hash, request_id,
        # server_path (redundant with server_name), and the metadata block. The
        # request-derived attributes (tool_name/method/client_name/client_version)
        # are charset-normalized and distinct-value-capped so a client sending
        # randomized values cannot explode the Prometheus time-series count.
        otel_attrs = _label_limiter.bound_attrs(
            {
                "tool_name": str(actual_tool_name or method_name),
                "server_name": str(server_name),
                "success": str(success),
                "method": str(method_name),
                "client_name": str(client_info.get("name", "unknown")),
                "client_version": str(client_info.get("version", "unknown")),
            },
            _TOOL_EXECUTION_BOUNDED_ATTRS,
        )
        tool_execution_total.add(1, otel_attrs)
        tool_execution_duration_ms.record(duration_ms, otel_attrs)
        record_emission_path("otel")

        # 2) Legacy HTTP POST (one-release dual-write window, issue #1122)
        if not self.legacy_http_post_enabled:
            return
        try:
            if not self.api_key:
                return

            metric_data = {
                "type": "tool_execution",
                "timestamp": datetime.utcnow().isoformat(),
                "value": 1.0,
                "duration_ms": duration_ms,
                "dimensions": {
                    "tool_name": actual_tool_name or method_name,
                    "server_name": server_name,
                    "success": success,
                    "method": method_name,
                    "user_hash": user_hash,
                    "server_path": f"/{server_name}/",
                    "client_name": client_info.get("name", "unknown"),
                    "client_version": client_info.get("version", "unknown"),
                },
                "metadata": {
                    "error_code": error_code,
                    "auth_method": auth_method,
                    "request_id": request_id or f"req_{uuid.uuid4().hex[:16]}",
                    "protocol_version": tool_info.get("protocol_version"),
                    "jsonrpc_id": tool_info.get("request_id"),
                    "actual_tool_name": actual_tool_name,
                    "method_type": method_name,
                    "input_size_bytes": len(json.dumps(tool_info.get("params", {})).encode()),
                    "output_size_bytes": 0,
                },
            }

            payload = {"service": self.service_name, "version": "1.0.0", "metrics": [metric_data]}

            await self.client.post(
                f"{self.metrics_url}/metrics", json=payload, headers={"X-API-Key": self.api_key}
            )
            record_emission_path("legacy")
        except httpx.RequestError as e:
            logger.debug(f"Legacy metrics-service POST failed (non-fatal): {e}")
        except Exception as e:
            logger.debug(f"Failed to emit legacy tool execution metric: {e}")

    async def _emit_protocol_latency_metric(
        self,
        session_key: str,
        current_method: str,
        server_name: str,
        user_hash: str,
        request_id: str,
    ):
        """Emit protocol flow latency metric via OTel and (optionally) legacy HTTP POST.

        OTel attributes drop ``user_hash`` and ``session_key`` (high cardinality,
        ephemeral). The legacy POST preserves the full dimension set for the
        one-release dual-write window.
        """
        try:
            session_data = self.session_timings.get(session_key, {})

            # 1) OTel emission for each completed flow step
            for flow_step, latency_seconds in self._compute_completed_latencies(session_data):
                protocol_latency_ms.record(
                    latency_seconds * 1000.0,
                    {
                        "flow_step": flow_step,
                        "server_name": str(server_name),
                    },
                )
                record_emission_path("otel")

            # 2) Legacy HTTP POST (one-release dual-write window, issue #1122)
            if not self.legacy_http_post_enabled:
                return
            if not self.api_key:
                return

            current_time = time.time()

            # Calculate latencies between protocol steps
            latency_metrics = []

            # Initialize -> Tools List latency
            if "initialize" in session_data and "tools/list" in session_data:
                init_to_list_latency = session_data["tools/list"] - session_data["initialize"]
                if (
                    init_to_list_latency > 0 and init_to_list_latency < 300
                ):  # Max 5 minutes reasonable
                    latency_metrics.append(
                        {
                            "type": "protocol_latency",
                            "timestamp": datetime.utcnow().isoformat(),
                            "value": init_to_list_latency,
                            "dimensions": {
                                "flow_step": "initialize_to_tools_list",
                                "server_name": server_name,
                                "user_hash": user_hash,
                                "session_key": session_key,
                            },
                            "metadata": {
                                "request_id": request_id,
                                "latency_seconds": init_to_list_latency,
                                "from_method": "initialize",
                                "to_method": "tools/list",
                            },
                        }
                    )

            # Tools List -> Tools Call latency
            if "tools/list" in session_data and "tools/call" in session_data:
                list_to_call_latency = session_data["tools/call"] - session_data["tools/list"]
                if (
                    list_to_call_latency > 0 and list_to_call_latency < 300
                ):  # Max 5 minutes reasonable
                    latency_metrics.append(
                        {
                            "type": "protocol_latency",
                            "timestamp": datetime.utcnow().isoformat(),
                            "value": list_to_call_latency,
                            "dimensions": {
                                "flow_step": "tools_list_to_tools_call",
                                "server_name": server_name,
                                "user_hash": user_hash,
                                "session_key": session_key,
                            },
                            "metadata": {
                                "request_id": request_id,
                                "latency_seconds": list_to_call_latency,
                                "from_method": "tools/list",
                                "to_method": "tools/call",
                            },
                        }
                    )

            # Initialize -> Tools Call (total flow latency)
            if "initialize" in session_data and "tools/call" in session_data:
                total_flow_latency = session_data["tools/call"] - session_data["initialize"]
                if total_flow_latency > 0 and total_flow_latency < 600:  # Max 10 minutes reasonable
                    latency_metrics.append(
                        {
                            "type": "protocol_latency",
                            "timestamp": datetime.utcnow().isoformat(),
                            "value": total_flow_latency,
                            "dimensions": {
                                "flow_step": "full_protocol_flow",
                                "server_name": server_name,
                                "user_hash": user_hash,
                                "session_key": session_key,
                            },
                            "metadata": {
                                "request_id": request_id,
                                "latency_seconds": total_flow_latency,
                                "from_method": "initialize",
                                "to_method": "tools/call",
                            },
                        }
                    )

            # Emit metrics if we have any
            if latency_metrics:
                payload = {
                    "service": self.service_name,
                    "version": "1.0.0",
                    "metrics": latency_metrics,
                }

                await self.client.post(
                    f"{self.metrics_url}/metrics", json=payload, headers={"X-API-Key": self.api_key}
                )
                record_emission_path("legacy")

            # Cleanup is now handled by _cleanup_sessions_if_needed method

        except httpx.RequestError as e:
            logger.debug(f"Legacy metrics-service POST failed (non-fatal): {e}")
        except Exception as e:
            logger.debug(f"Failed to emit protocol latency metric: {e}")

    def _compute_completed_latencies(
        self,
        session_data: dict[str, float],
    ) -> list[tuple[str, float]]:
        """Return (flow_step, latency_seconds) tuples for completed protocol flows.

        Identical bounds to the legacy logic: each step is only emitted if both
        endpoints exist in ``session_data`` and the computed latency falls in
        a reasonable range.
        """
        out: list[tuple[str, float]] = []

        if "initialize" in session_data and "tools/list" in session_data:
            init_to_list = session_data["tools/list"] - session_data["initialize"]
            if 0 < init_to_list < 300:
                out.append(("initialize_to_tools_list", init_to_list))

        if "tools/list" in session_data and "tools/call" in session_data:
            list_to_call = session_data["tools/call"] - session_data["tools/list"]
            if 0 < list_to_call < 300:
                out.append(("tools_list_to_tools_call", list_to_call))

        if "initialize" in session_data and "tools/call" in session_data:
            total_flow = session_data["tools/call"] - session_data["initialize"]
            if 0 < total_flow < 600:
                out.append(("full_protocol_flow", total_flow))

        return out


def add_auth_metrics_middleware(app, service_name: str = "auth-server"):
    """
    Convenience function to add auth metrics middleware to a FastAPI app.

    Args:
        app: FastAPI application instance
        service_name: Name of the service for metrics identification
    """
    app.add_middleware(AuthMetricsMiddleware, service_name=service_name)
    logger.info(f"Auth metrics middleware added for service: {service_name}")
