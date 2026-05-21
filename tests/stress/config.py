"""Path helpers and environment-variable plumbing for the stress harness."""

import logging
import os
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def project_root() -> Path:
    """Return the mcp-gateway-registry project root."""
    return Path(__file__).resolve().parents[2]


def stress_root() -> Path:
    return project_root() / "tests" / "stress"


def default_data_dir() -> Path:
    return stress_root() / "data"


def default_results_dir() -> Path:
    override = os.getenv("STRESS_RESULTS_DIR")
    return Path(override) if override else stress_root() / "results"


def default_cache_dir() -> Path:
    return default_data_dir() / ".cache"


def data_dir_for(
    entity_type: str,
    count: int,
    data_dir: Path | None = None,
) -> Path:
    base = data_dir or default_data_dir()
    return base / entity_type / str(count)


def results_dir_for(
    backend: str,
    size: int,
    results_dir: Path | None = None,
) -> Path:
    base = results_dir or default_results_dir()
    return base / backend / f"size-{size}"


def default_base_url() -> str:
    return os.getenv("STRESS_BASE_URL", "http://localhost")


def default_token_file() -> Path:
    override = os.getenv("STRESS_TOKEN_FILE")
    if override:
        return Path(override)
    return project_root() / ".oauth-tokens" / "ingress.json"


def default_queries_path() -> Path:
    return stress_root() / "queries.json"


def _detect_instance_count(
    base_url: str,
    headers: dict[str, str],
    sample_requests: int = 20,
) -> int:
    """Detect the number of backend instances behind the load balancer.

    Hits /api/stats N times and counts distinct `started_at` values.
    Each ECS task/container has a unique boot timestamp, and the ALB
    round-robins across them.
    """
    started_at_values: set[str] = set()
    stats_url = f"{base_url.rstrip('/')}/api/stats"
    for _ in range(sample_requests):
        try:
            resp = httpx.get(stats_url, headers=headers, timeout=5)
            if resp.status_code == 200:
                started_at = resp.json().get("started_at", "")
                if started_at:
                    started_at_values.add(started_at)
        except Exception:
            pass
    count = len(started_at_values)
    if count > 0:
        logger.info("Detected %d backend instance(s) behind load balancer", count)
    return count


def fetch_registry_info(
    base_url: str,
    token: str,
) -> dict[str, Any]:
    """Fetch deployment info from GET /api/registry-management/telemetry/info.

    Falls back to GET /api/stats if the telemetry/info endpoint is not
    available (older deployments). Also probes the load balancer to detect
    the number of backend instances. Returns the registry's hardware/software
    configuration snapshot for embedding in benchmark result files.

    Returns an empty dict if both endpoints are unreachable.
    """
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{base_url.rstrip('/')}/api/registry-management/telemetry/info"
    info: dict[str, Any] = {}

    try:
        resp = httpx.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            info = resp.json()
            logger.info(
                "Fetched registry info: v=%s, compute=%s, storage=%s",
                info.get("v"),
                info.get("compute"),
                info.get("storage"),
            )
    except Exception:
        pass

    # Fallback: /api/stats provides version, backend, and auth info
    if not info:
        stats_url = f"{base_url.rstrip('/')}/api/stats"
        try:
            resp = httpx.get(stats_url, headers=headers, timeout=10)
            if resp.status_code == 200:
                stats = resp.json()
                info = {
                    "v": stats.get("version", ""),
                    "storage": stats.get("database_status", {}).get("backend", ""),
                    "auth": stats.get("auth_status", {}).get("provider", ""),
                    "uptime_hours": int(stats.get("uptime_seconds", 0) / 3600),
                    "servers_count": stats.get("registry_stats", {}).get("servers", 0),
                    "agents_count": stats.get("registry_stats", {}).get("agents", 0),
                    "skills_count": stats.get("registry_stats", {}).get("skills", 0),
                    "mode": stats.get("deployment_mode", ""),
                    "source": "api_stats_fallback",
                }
                logger.info(
                    "Fetched registry info (fallback): v=%s, storage=%s",
                    info["v"],
                    info["storage"],
                )
        except Exception as exc:
            logger.warning("Failed to fetch registry info: %s", exc)

    # Detect instance count behind the load balancer
    if info:
        instance_count = _detect_instance_count(base_url, headers)
        if instance_count > 0:
            info["instance_count"] = instance_count

    return info
