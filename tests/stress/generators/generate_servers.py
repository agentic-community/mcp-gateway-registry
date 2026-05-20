#!/usr/bin/env python3
"""Generate MCP server registration payloads from the Anthropic MCP Registry.

Pages https://registry.modelcontextprotocol.io/v0/servers, runs each entry
through cli.anthropic_transformer.transform_anthropic_to_gateway, then
overrides fields for stress testing (synthetic proxy URLs, draft status,
unique paths via -stress-{i} suffix when augmenting).

Validates each payload as api.registry_client.InternalServiceRegistration
before writing.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import httpx

from tests.stress.constants import (
    ANTHROPIC_PAGE_LIMIT,
    ANTHROPIC_REGISTRY_BASE,
    STRESS_TAG,
)
from tests.stress.generators._base import (
    cache_read_json,
    cache_write_json,
    collect_unique,
    ensure_project_on_path,
    run_generator,
    unique_suffix,
)

ensure_project_on_path()

from api.registry_client import InternalServiceRegistration  # noqa: E402
from cli.anthropic_transformer import transform_anthropic_to_gateway  # noqa: E402

logger = logging.getLogger(__name__)

CACHE_BASENAME = "anthropic-registry-pages.json"


def _fetch_anthropic_catalog(cache_dir: Path) -> list[dict[str, Any]]:
    """Page the Anthropic registry, caching the merged response on disk."""
    cache_path = cache_dir / CACHE_BASENAME
    cached = cache_read_json(cache_path)
    if cached:
        logger.info("Using cached Anthropic catalog from %s", cache_path)
        return cached

    all_records: list[dict[str, Any]] = []
    cursor: str | None = None

    with httpx.Client(timeout=60.0) as client:
        while True:
            params: dict[str, Any] = {"limit": ANTHROPIC_PAGE_LIMIT}
            if cursor:
                params["cursor"] = cursor

            logger.info(
                "Fetching Anthropic registry page (records so far=%d, cursor=%s)",
                len(all_records),
                cursor,
            )
            resp = client.get(ANTHROPIC_REGISTRY_BASE, params=params)
            resp.raise_for_status()
            payload = resp.json()

            page = payload.get("servers", [])
            if not page:
                break
            all_records.extend(page)

            cursor = payload.get("metadata", {}).get("nextCursor")
            if not cursor:
                break

    deduped = collect_unique(all_records, key_fn=_record_key)
    cache_write_json(cache_path, deduped)
    logger.info("Cached %d unique Anthropic records to %s", len(deduped), cache_path)
    return deduped


def _record_key(record: dict[str, Any]) -> str:
    server = record.get("server", record)
    return server.get("name", json.dumps(record, sort_keys=True))


def _build_payload(
    record: dict[str, Any],
    suffix_index: int | None,
) -> dict[str, Any]:
    """Transform an Anthropic record and override fields for stress testing."""
    transformed = transform_anthropic_to_gateway(record)

    base_name = transformed.get("server_name") or "unnamed"
    base_path = transformed.get("path") or f"/{base_name.replace('/', '-')}"

    if suffix_index is not None:
        suffix = unique_suffix(suffix_index)
        name = f"{base_name}{suffix}"
        path = f"{base_path}{suffix}"
    else:
        name = base_name
        path = base_path

    # Filter out tags that the frontend treats as "external registry" markers
    EXTERNAL_REGISTRY_TAGS = {"anthropic-registry", "workday-asor", "asor", "federated"}
    tags = [t for t in (transformed.get("tags") or []) if t not in EXTERNAL_REGISTRY_TAGS]
    if STRESS_TAG not in tags:
        tags.append(STRESS_TAG)

    # Preserve the original proxy_pass_url from the Anthropic registry when available
    original_url = transformed.get("proxy_pass_url") or transformed.get("remote_url")
    proxy_url = original_url or f"http://stress-test-{abs(hash(path)) % 100000:05d}.invalid:8100"

    payload: dict[str, Any] = {
        "server_name": name,
        "description": transformed.get("description") or f"Stress-test MCP server: {name}",
        "path": path,
        "proxy_pass_url": proxy_url,
        "supported_transports": transformed.get("supported_transports") or ["streamable-http"],
        "tags": tags,
        "status": "active",
        "visibility": "public",
    }
    return payload


def _validate_payload(payload: dict[str, Any]) -> None:
    """Construct InternalServiceRegistration to surface validation errors."""
    InternalServiceRegistration(
        path=payload["path"],
        name=payload["server_name"],
        description=payload["description"],
        proxy_pass_url=payload["proxy_pass_url"],
        supported_transports=payload.get("supported_transports"),
        tags=payload.get("tags"),
        status=payload.get("status"),
    )


def _payload_seed(record: dict[str, Any]) -> str:
    return _record_key(record)


def main() -> int:
    return run_generator(
        entity_type="servers",
        description="Generate MCP server registration payloads from the Anthropic Registry.",
        fetch_records=_fetch_anthropic_catalog,
        build_payload=_build_payload,
        validate_payload=_validate_payload,
        payload_seed=_payload_seed,
    )


if __name__ == "__main__":
    sys.exit(main())
