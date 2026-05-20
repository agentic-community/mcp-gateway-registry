#!/usr/bin/env python3
"""Generate Agent Skill registration payloads from the anthropics/skills repo.

Uses the GitHub trees API to enumerate every `*/SKILL.md` blob, builds a
skill registration payload pointing at the raw GitHub URL, and validates
each as api.registry_client.SkillRegistrationRequest with an extra regex
check to mirror the server-side name validator.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx

from tests.stress.constants import (
    GITHUB_RAW_BASE,
    GITHUB_TREE_API,
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

from api.registry_client import SkillRegistrationRequest  # noqa: E402

logger = logging.getLogger(__name__)

CACHE_BASENAME = "anthropics-skills-tree.json"

SKILL_MD_SUFFIX = "/SKILL.md"
SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
NAME_SANITIZE_RE = re.compile(r"[^a-z0-9-]+")


def _sanitize_skill_name(raw: str) -> str:
    lowered = raw.lower().strip()
    lowered = re.sub(r"[_\s/]+", "-", lowered)
    lowered = NAME_SANITIZE_RE.sub("-", lowered)
    lowered = re.sub(r"-+", "-", lowered).strip("-")
    return lowered or "skill"


def _fetch_skill_records(cache_dir: Path) -> list[dict[str, Any]]:
    """Enumerate every SKILL.md path in the anthropics/skills repo."""
    cache_path = cache_dir / CACHE_BASENAME
    cached = cache_read_json(cache_path)
    if cached:
        logger.info("Using cached GitHub tree from %s", cache_path)
        return cached

    headers = {"Accept": "application/vnd.github+json"}
    # Accept either GITHUB_TOKEN (the documented stress-test name) or
    # GITHUB_PAT (the project's existing convention used elsewhere in .env)
    # so we don't force users to duplicate the same secret under two names.
    github_token = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PAT")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    else:
        logger.warning(
            "Neither GITHUB_TOKEN nor GITHUB_PAT is set. "
            "GitHub anonymous rate limit (60 req/hr) may apply."
        )

    with httpx.Client(timeout=60.0) as client:
        resp = client.get(GITHUB_TREE_API, headers=headers)
        resp.raise_for_status()
        tree = resp.json()

    if tree.get("truncated"):
        logger.warning(
            "GitHub tree response was truncated -- some skills may be missing. "
            "If this becomes a problem, switch to git-clone-based enumeration."
        )

    records: list[dict[str, Any]] = []
    for entry in tree.get("tree", []):
        path = entry.get("path", "")
        if entry.get("type") != "blob" or not path.endswith(SKILL_MD_SUFFIX):
            continue

        parent = path[: -len(SKILL_MD_SUFFIX)]
        raw_name = parent.rsplit("/", 1)[-1]
        records.append(
            {
                "raw_name": raw_name,
                "repo_path": path,
                "skill_md_url": f"{GITHUB_RAW_BASE}/{path}",
            }
        )

    deduped = collect_unique(records, key_fn=lambda r: r["repo_path"])
    cache_write_json(cache_path, deduped)
    logger.info("Cached %d unique skills to %s", len(deduped), cache_path)
    return deduped


def _build_payload(
    record: dict[str, Any],
    suffix_index: int | None,
) -> dict[str, Any]:
    base_name = _sanitize_skill_name(record["raw_name"])
    if suffix_index is not None:
        suffix = unique_suffix(suffix_index)
        name = f"{base_name}{suffix}"
    else:
        name = base_name

    payload: dict[str, Any] = {
        "name": name,
        "skill_md_url": record["skill_md_url"],
        "description": f"Skill sourced from anthropics/skills: {record['raw_name']}",
        "version": "1.0.0",
        "tags": [STRESS_TAG, "anthropic-skills"],
        "target_agents": ["claude-code"],
        "visibility": "public",
    }
    return payload


def _validate_payload(payload: dict[str, Any]) -> None:
    """Validate via SkillRegistrationRequest + the server-side name regex."""
    if not SKILL_NAME_RE.match(payload["name"]):
        raise ValueError(
            f"Skill name {payload['name']!r} does not match server regex ^[a-z0-9]+(-[a-z0-9]+)*$"
        )
    SkillRegistrationRequest.model_validate(payload)


def _payload_seed(record: dict[str, Any]) -> str:
    return record["repo_path"]


def main() -> int:
    return run_generator(
        entity_type="skills",
        description="Generate Agent Skill payloads from anthropics/skills.",
        fetch_records=_fetch_skill_records,
        build_payload=_build_payload,
        validate_payload=_validate_payload,
        payload_seed=_payload_seed,
    )


if __name__ == "__main__":
    sys.exit(main())
