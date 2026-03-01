"""Prometheus metrics for deployment mode monitoring."""

from prometheus_client import Counter, Gauge

# Configuration viewer metrics
CONFIG_VIEW_REQUESTS = Counter(
    "mcp_config_view_requests_total",
    "Total configuration view requests",
    ["user_type"],
)

CONFIG_EXPORT_REQUESTS = Counter(
    "mcp_config_export_requests_total",
    "Total configuration export requests",
    ["format", "includes_sensitive"],
)

# Deployment mode info gauge
DEPLOYMENT_MODE_INFO = Gauge(
    "registry_deployment_mode_info",
    "Current deployment mode configuration",
    ["deployment_mode", "registry_mode"],
)

# Counter for skipped nginx updates
NGINX_UPDATES_SKIPPED = Counter(
    "registry_nginx_updates_skipped_total",
    "Number of nginx updates skipped due to registry-only mode",
    ["operation"],  # generate_config, reload
)

# Counter for blocked requests due to registry mode
MODE_BLOCKED_REQUESTS = Counter(
    "registry_mode_blocked_requests_total",
    "Requests blocked due to registry mode restrictions",
    ["path_category", "mode"],  # servers, agents, skills, federation
)

# Peer federation metrics (issue #561)
PEER_SYNC_FAILURES = Counter(
    "peer_sync_failures_total",
    "Total peer sync failures by failure type",
    ["peer_id", "failure_type"],  # auth_error, network_error, etc.
)

PEER_TOKEN_MISSING = Gauge(
    "peer_token_missing_total",
    "Number of peers missing federation tokens",
)

PEER_SYNC_DURATION_SECONDS = Gauge(
    "peer_sync_duration_seconds", "Duration of peer sync operations", ["peer_id", "success"]
)
