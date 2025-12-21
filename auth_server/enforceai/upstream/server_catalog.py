from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..models.upstream_auth import (
    UpstreamAuthConfig,
    normalize_upstream_auth,
)


def _normalize_server_path(
    raw: str,
) -> str:
    stripped = raw.strip()
    if not stripped:
        raise ValueError("server_path is required")
    if not stripped.startswith("/"):
        stripped = "/" + stripped
    return stripped.rstrip("/") or "/"


def load_upstream_auth_for_server(
    *,
    server_path: str,
    servers_dir: Path,
) -> UpstreamAuthConfig:
    """Load and normalize upstream_auth for a server from registry JSON files.

    Args:
        server_path: Canonical server path (leading '/', no trailing slash).
        servers_dir: Directory containing registry server JSON definitions.

    Returns:
        Normalized upstream_auth config for the matching server.

    Raises:
        FileNotFoundError: If `servers_dir` is missing.
        ValueError: If no matching server is found or JSON is invalid.
    """
    normalized_server_path = _normalize_server_path(server_path)

    if not servers_dir.exists() or not servers_dir.is_dir():
        raise FileNotFoundError(f"servers_dir not found: {servers_dir}")

    candidates = sorted(servers_dir.glob("*.json"))
    for candidate in candidates:
        if candidate.name == "server_state.json":
            continue

        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {candidate}: {exc.msg}") from exc
        except OSError as exc:
            raise ValueError(f"Failed to read {candidate}") from exc

        if not isinstance(payload, dict):
            continue

        raw_path = payload.get("path")
        if not isinstance(raw_path, str):
            continue

        try:
            candidate_path = _normalize_server_path(raw_path)
        except ValueError:
            continue

        if candidate_path != normalized_server_path:
            continue

        return normalize_upstream_auth(
            upstream_auth=payload.get("upstream_auth"),
            auth_type=payload.get("auth_type") if isinstance(payload.get("auth_type"), str) else None,
            auth_provider=payload.get("auth_provider")
            if isinstance(payload.get("auth_provider"), str)
            else None,
            headers=payload.get("headers"),
        )

    raise ValueError("Server not found")


def list_servers_referencing_upstream_oauth_provider(
    *,
    provider_id: str,
    servers_dir: Path,
) -> list[str]:
    """List server paths whose normalized upstream_auth.provider matches provider_id.

    Args:
        provider_id: Provider id to search for (exact match).
        servers_dir: Directory containing registry server JSON definitions.

    Returns:
        Sorted list of canonical server paths referencing the provider id.

    Raises:
        FileNotFoundError: If `servers_dir` is missing.
        ValueError: If JSON cannot be read/parsed.
    """
    normalized_provider_id = provider_id.strip()
    if not normalized_provider_id:
        raise ValueError("provider_id must be a non-empty string")

    if not servers_dir.exists() or not servers_dir.is_dir():
        raise FileNotFoundError(f"servers_dir not found: {servers_dir}")

    matches: list[str] = []
    candidates = sorted(servers_dir.glob("*.json"))
    for candidate in candidates:
        if candidate.name == "server_state.json":
            continue

        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {candidate}: {exc.msg}") from exc
        except OSError as exc:
            raise ValueError(f"Failed to read {candidate}") from exc

        if not isinstance(payload, dict):
            continue

        raw_path = payload.get("path")
        if not isinstance(raw_path, str):
            continue

        try:
            candidate_path = _normalize_server_path(raw_path)
        except ValueError:
            continue

        upstream_auth = normalize_upstream_auth(
            upstream_auth=payload.get("upstream_auth"),
            auth_type=payload.get("auth_type") if isinstance(payload.get("auth_type"), str) else None,
            auth_provider=payload.get("auth_provider")
            if isinstance(payload.get("auth_provider"), str)
            else None,
            headers=payload.get("headers"),
        )
        if upstream_auth.provider == normalized_provider_id:
            matches.append(candidate_path)

    return sorted(set(matches))
