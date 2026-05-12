"""Path helpers and environment-variable plumbing for the stress harness."""

import os
from pathlib import Path


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
