#!/usr/bin/env python3
"""Delete all stress-test entities (servers, agents, skills) from the registry.

Identifies stress-test entities by the 'stress-test' tag that all generators
apply. Supports deleting a single entity type or all three.

Usage:
    uv run python -m tests.stress.cleanup \
        --base-url http://localhost \
        --token-file .token \
        --entity-type all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from tests.stress.config import (
    default_base_url,
    default_token_file,
)
from tests.stress.constants import (
    ENTITY_TYPES,
    HTTP_TIMEOUT_SECONDS,
    STRESS_TAG,
    EntityType,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


def _load_token(token_file: Path) -> str:
    """Load JWT token from file (JSON or plain text)."""
    raw = token_file.read_text()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        token = raw.strip()
        if not token:
            raise RuntimeError(f"Empty token file: {token_file}") from None
        return token

    token = (
        data.get("access_token")
        or data.get("tokens", {}).get("access_token")
        or data.get("token_data", {}).get("access_token")
    )
    if not token:
        raise RuntimeError(f"No 'access_token' field found in token file: {token_file}")
    return token


async def _list_servers(
    client: httpx.AsyncClient,
    base_url: str,
    headers: dict[str, str],
) -> list[dict[str, Any]]:
    """List all servers and filter to those with the stress-test tag."""
    all_servers: list[dict[str, Any]] = []
    offset = 0
    limit = 500

    while True:
        resp = await client.get(
            f"{base_url}/api/servers",
            headers=headers,
            params={"limit": limit, "offset": offset},
        )
        resp.raise_for_status()
        data = resp.json()
        servers = data.get("servers", [])
        if not servers:
            break
        all_servers.extend(servers)
        if not data.get("has_next", False):
            break
        offset += limit

    stress_servers = [
        s for s in all_servers
        if STRESS_TAG in (s.get("tags") or [])
    ]
    return stress_servers


async def _list_agents(
    client: httpx.AsyncClient,
    base_url: str,
    headers: dict[str, str],
) -> list[dict[str, Any]]:
    """List all agents and filter to those with the stress-test tag."""
    all_agents: list[dict[str, Any]] = []
    offset = 0
    limit = 100

    while True:
        resp = await client.get(
            f"{base_url}/api/agents",
            headers=headers,
            params={"limit": limit, "offset": offset},
        )
        resp.raise_for_status()
        data = resp.json()
        agents = data.get("agents", [])
        if not agents:
            break
        all_agents.extend(agents)
        if not data.get("has_next", False):
            break
        offset += limit

    stress_agents = [
        a for a in all_agents
        if STRESS_TAG in (a.get("tags") or [])
    ]
    return stress_agents


async def _list_skills(
    client: httpx.AsyncClient,
    base_url: str,
    headers: dict[str, str],
) -> list[dict[str, Any]]:
    """List all skills and filter to those with the stress-test tag."""
    all_skills: list[dict[str, Any]] = []
    offset = 0
    limit = 100

    while True:
        resp = await client.get(
            f"{base_url}/api/skills",
            headers=headers,
            params={"limit": limit, "offset": offset},
        )
        resp.raise_for_status()
        data = resp.json()
        skills = data.get("skills", [])
        if not skills:
            break
        all_skills.extend(skills)
        if not data.get("has_next", False):
            break
        offset += limit

    stress_skills = [
        s for s in all_skills
        if STRESS_TAG in (s.get("tags") or [])
    ]
    return stress_skills


async def _delete_servers(
    client: httpx.AsyncClient,
    base_url: str,
    headers: dict[str, str],
    servers: list[dict[str, Any]],
) -> tuple[int, int]:
    """Delete servers by path. Returns (deleted, failed) counts."""
    deleted = 0
    failed = 0
    for server in servers:
        path = server.get("path", "")
        if not path:
            failed += 1
            continue
        resp = await client.post(
            f"{base_url}/api/servers/remove",
            headers=headers,
            data={"path": path},
        )
        if resp.status_code in (200, 204):
            deleted += 1
            logger.debug("Deleted server: %s", path)
        else:
            failed += 1
            logger.warning("Failed to delete server %s: %d %s", path, resp.status_code, resp.text[:100])
    return deleted, failed


async def _delete_agents(
    client: httpx.AsyncClient,
    base_url: str,
    headers: dict[str, str],
    agents: list[dict[str, Any]],
) -> tuple[int, int]:
    """Delete agents by path. Returns (deleted, failed) counts."""
    deleted = 0
    failed = 0
    for agent in agents:
        path = agent.get("path", "")
        if not path:
            failed += 1
            continue
        # Strip leading slash for the URL path param
        url_path = path.lstrip("/")
        resp = await client.delete(
            f"{base_url}/api/agents/{url_path}",
            headers=headers,
        )
        if resp.status_code in (200, 204):
            deleted += 1
            logger.debug("Deleted agent: %s", path)
        else:
            failed += 1
            logger.warning("Failed to delete agent %s: %d %s", path, resp.status_code, resp.text[:100])
    return deleted, failed


async def _delete_skills(
    client: httpx.AsyncClient,
    base_url: str,
    headers: dict[str, str],
    skills: list[dict[str, Any]],
) -> tuple[int, int]:
    """Delete skills by path/name. Returns (deleted, failed) counts."""
    deleted = 0
    failed = 0
    for skill in skills:
        skill_path = skill.get("path") or skill.get("name", "")
        if not skill_path:
            failed += 1
            continue
        # Strip the leading "/skills/" prefix since the API endpoint is /api/skills/{name}
        url_path = skill_path.lstrip("/")
        if url_path.startswith("skills/"):
            url_path = url_path[len("skills/"):]
        resp = await client.delete(
            f"{base_url}/api/skills/{url_path}",
            headers=headers,
        )
        if resp.status_code in (200, 204):
            deleted += 1
            logger.debug("Deleted skill: %s", skill_path)
        else:
            failed += 1
            logger.warning("Failed to delete skill %s: %d %s", skill_path, resp.status_code, resp.text[:100])
    return deleted, failed


async def _main_async(args: argparse.Namespace) -> int:
    token = _load_token(args.token_file)
    headers = {"Authorization": f"Bearer {token}"}
    base_url = args.base_url.rstrip("/")

    if args.entity_type == "all":
        entity_types: list[EntityType] = list(ENTITY_TYPES)
    else:
        entity_types = [args.entity_type]

    start = time.time()

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        for entity_type in entity_types:
            logger.info("Listing %s with tag '%s'...", entity_type, STRESS_TAG)

            if entity_type == "servers":
                entities = await _list_servers(client, base_url, headers)
                logger.info("Found %d stress-test servers", len(entities))
                if entities and not args.dry_run:
                    deleted, failed = await _delete_servers(client, base_url, headers, entities)
                    logger.info("Servers: deleted=%d failed=%d", deleted, failed)

            elif entity_type == "agents":
                entities = await _list_agents(client, base_url, headers)
                logger.info("Found %d stress-test agents", len(entities))
                if entities and not args.dry_run:
                    deleted, failed = await _delete_agents(client, base_url, headers, entities)
                    logger.info("Agents: deleted=%d failed=%d", deleted, failed)

            elif entity_type == "skills":
                entities = await _list_skills(client, base_url, headers)
                logger.info("Found %d stress-test skills", len(entities))
                if entities and not args.dry_run:
                    deleted, failed = await _delete_skills(client, base_url, headers, entities)
                    logger.info("Skills: deleted=%d failed=%d", deleted, failed)

    elapsed = time.time() - start
    if args.dry_run:
        logger.info("Dry run complete (%.1fs). No entities were deleted.", elapsed)
    else:
        logger.info("Cleanup complete in %.1fs.", elapsed)
    return 0


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Delete all stress-test entities from the registry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Delete all stress-test entities
    uv run python -m tests.stress.cleanup --base-url http://localhost --token-file .token

    # Dry run (list only, don't delete)
    uv run python -m tests.stress.cleanup --dry-run --base-url http://localhost --token-file .token

    # Delete only servers
    uv run python -m tests.stress.cleanup --entity-type servers --base-url http://localhost --token-file .token
""",
    )
    parser.add_argument(
        "--entity-type",
        choices=[*ENTITY_TYPES, "all"],
        default="all",
        help="Which entity type to clean up (default: all).",
    )
    parser.add_argument(
        "--base-url",
        default=default_base_url(),
        help="Registry base URL (default: $STRESS_BASE_URL or http://localhost).",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        default=default_token_file(),
        help="JWT token file (default: .token).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List entities that would be deleted without actually deleting them.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main() -> int:
    args = _build_argparser().parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
