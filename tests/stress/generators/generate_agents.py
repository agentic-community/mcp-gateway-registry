#!/usr/bin/env python3
"""Generate A2A agent registration payloads from the GoDaddy ANS catalog.

Pages https://api.godaddy.com/v1/agents using `Authorization: sso-key {key}:{secret}`,
maps each ANS record into an A2A Agent Card shape compatible with
api.registry_client.AgentRegistration, augments with -stress-{i} when the
upstream returns fewer records than the target count.

Required environment variables: ANS_API_KEY, ANS_API_SECRET.
Optional: ANS_API_ENDPOINT (default https://api.godaddy.com).
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from tests.stress.constants import (
    ANS_AGENTS_PATH,
    ANS_DEFAULT_ENDPOINT,
    ANS_PAGE_LIMIT,
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

from api.registry_client import AgentRegistration  # noqa: E402

logger = logging.getLogger(__name__)

CACHE_BASENAME = "ans-agents.json"

ANS_NAME_PREFIX = "ans://"
SLUG_RE = re.compile(r"[^a-z0-9-]+")


def _slugify(name: str) -> str:
    lowered = name.lower().strip()
    lowered = re.sub(r"\s+", "-", lowered)
    cleaned = SLUG_RE.sub("-", lowered)
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    return cleaned or "agent"


def _ans_credentials() -> tuple[str, str, str]:
    """Resolve ANS credentials and endpoint.

    Uses the documented names from `docs/design/ans-integration.md`:
    `ANS_API_KEY`, `ANS_API_SECRET`, and an optional `ANS_API_ENDPOINT`
    override. The default endpoint is the production host
    (`api.godaddy.com`); customer-tier credentials issued for the test
    environment require `ANS_API_ENDPOINT=https://api.ote-godaddy.com`,
    since /v1/agents on prod is gated behind GoDaddy's internal SSO.
    """
    api_key = os.getenv("ANS_API_KEY")
    api_secret = os.getenv("ANS_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError(
            "ANS_API_KEY and ANS_API_SECRET must be set to generate agents. "
            "See docs/design/ans-integration.md. For customer-tier credentials "
            "also set ANS_API_ENDPOINT=https://api.ote-godaddy.com."
        )
    endpoint = os.getenv("ANS_API_ENDPOINT", ANS_DEFAULT_ENDPOINT)
    return api_key, api_secret, endpoint


def _fetch_ans_agents(cache_dir: Path) -> list[dict[str, Any]]:
    """Page the ANS catalog with exponential backoff on 429."""
    cache_path = cache_dir / CACHE_BASENAME
    cached = cache_read_json(cache_path)
    if cached:
        logger.info("Using cached ANS catalog from %s", cache_path)
        return cached

    api_key, api_secret, endpoint = _ans_credentials()
    headers = {
        "Authorization": f"sso-key {api_key}:{api_secret}",
        "Accept": "application/json",
    }
    base_url = f"{endpoint.rstrip('/')}{ANS_AGENTS_PATH}"

    all_records: list[dict[str, Any]] = []
    offset = 0
    total_count: int | None = None
    max_records = 10_000

    with httpx.Client(timeout=60.0) as client:
        while True:
            page = _fetch_page_with_retry(client, base_url, headers, offset)
            agents = page.get("agents", [])
            if not agents:
                break
            all_records.extend(agents)

            if total_count is None:
                total_count = page.get("totalCount", 0) or 0
                logger.info("ANS reports totalCount=%d", total_count)

            offset += ANS_PAGE_LIMIT
            if len(all_records) >= max_records:
                logger.info("Reached %d records cap, stopping pagination", max_records)
                break
            if total_count and offset >= total_count:
                break
            if not total_count and len(agents) < ANS_PAGE_LIMIT:
                break

    deduped = collect_unique(all_records, key_fn=_record_key)
    cache_write_json(cache_path, deduped)
    logger.info("Cached %d unique ANS records to %s", len(deduped), cache_path)
    return deduped


def _fetch_page_with_retry(
    client: httpx.Client,
    base_url: str,
    headers: dict[str, str],
    offset: int,
) -> dict[str, Any]:
    """GET a single page, retrying on 429 with exponential backoff (max 5 retries)."""
    max_retries = 5
    base_delay = 2.0

    for attempt in range(max_retries):
        resp = client.get(
            base_url,
            headers=headers,
            params={"limit": ANS_PAGE_LIMIT, "offset": offset},
        )
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", base_delay * (2**attempt)))
            logger.warning(
                "ANS rate-limited at offset=%d, sleeping %.1fs (attempt %d/%d)",
                offset,
                retry_after,
                attempt + 1,
                max_retries,
            )
            time.sleep(retry_after)
            continue
        resp.raise_for_status()
        return resp.json()

    raise RuntimeError(f"ANS API exceeded retry budget at offset={offset}")


def _record_key(record: dict[str, Any]) -> str:
    return record.get("agentId") or record.get("ansName") or str(id(record))


def _base_name(record: dict[str, Any]) -> str:
    """Derive a base name from an ANS record.

    Prefers `agentDisplayName` (human-readable) when present. Falls back to
    the last DNS-style component of `ansName` (e.g. for
    `ans://v1.0.0.canary-ote.itest.example.com` returns `example`), then
    `agentHost`, then `agentId`.
    """
    display = record.get("agentDisplayName")
    if display:
        return display
    ans_name = record.get("ansName") or ""
    if ans_name.startswith(ANS_NAME_PREFIX):
        ans_name = ans_name[len(ANS_NAME_PREFIX) :]
    if ans_name:
        parts = [p for p in ans_name.split(".") if p]
        # Pick a meaningful component: skip semver prefix like "v1.0.0"
        for part in reversed(parts):
            if not re.fullmatch(r"v?\d+", part):
                return part
        return parts[-1] if parts else ans_name
    return record.get("agentHost") or record.get("agentId") or "ans-agent"


def _build_payload(
    record: dict[str, Any],
    suffix_index: int | None,
) -> dict[str, Any]:
    base = _base_name(record)
    slug = _slugify(base)
    if suffix_index is not None:
        suffix = unique_suffix(suffix_index)
        name = f"{slug}{suffix}"
        path = f"/{name}"
    else:
        name = slug
        path = f"/{slug}"

    endpoints = record.get("endpoints") or []
    url = None
    for ep in endpoints:
        if isinstance(ep, dict):
            url = ep.get("agentUrl") or ep.get("url") or ep.get("endpoint")
            if url:
                break
    if not url:
        url = f"https://stress-test-{abs(hash(path)) % 100000:05d}.invalid/agent"

    provider_org = record.get("organization") or record.get("provider") or "Stress Test Org"
    tags_in = record.get("tags") or []
    tags = [STRESS_TAG, *tags_in] if STRESS_TAG not in tags_in else list(tags_in)

    description = (
        record.get("agentDescription")
        or record.get("description")
        or f"ANS-sourced stress test agent: {name}"
    )

    payload: dict[str, Any] = {
        "protocolVersion": "1.0",
        "name": name,
        "description": description,
        "url": url,
        "version": record.get("version") or "1.0.0",
        "capabilities": {"streaming": True},
        "defaultInputModes": ["text/plain", "application/json"],
        "defaultOutputModes": ["text/plain", "application/json"],
        "skills": [],
        "preferredTransport": "JSONRPC",
        "provider": {
            "organization": provider_org,
            "url": record.get("organizationUrl") or "https://stress-test.invalid",
        },
        "path": path,
        "tags": tags,
        "isEnabled": True,
        "visibility": "public",
        "supportedProtocol": "a2a",
    }
    return payload


def _validate_payload(payload: dict[str, Any]) -> None:
    AgentRegistration.model_validate(payload)


def _payload_seed(record: dict[str, Any]) -> str:
    return _record_key(record)


def main() -> int:
    return run_generator(
        entity_type="agents",
        description="Generate A2A agent payloads from the GoDaddy ANS catalog.",
        fetch_records=_fetch_ans_agents,
        build_payload=_build_payload,
        validate_payload=_validate_payload,
        payload_seed=_payload_seed,
    )


if __name__ == "__main__":
    sys.exit(main())
