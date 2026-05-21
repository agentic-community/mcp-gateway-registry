#!/usr/bin/env python3
"""Bulk-register generated stress-test payloads against a running registry.

Reads JSON payloads from tests/stress/data/<entity>/<count>/ and POSTs each
one to the appropriate registry endpoint using httpx.AsyncClient bounded by
an asyncio.Semaphore. Aggregates p50/p95/p99 latency and reports to
tests/stress/results/<backend>/size-<count>/registration.json.

Idempotent: 409 Conflict (already exists) is recorded as 'skipped', not
'failed'. 401 triggers one token-file re-read + retry before giving up.

Endpoints (from api/registry_client.py):
  servers:  POST /api/servers/register   (form-encoded)
  agents:   POST /api/agents/register    (JSON)
  skills:   POST /api/skills             (JSON)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from tests.stress.config import (
    data_dir_for,
    default_base_url,
    default_data_dir,
    default_results_dir,
    default_token_file,
    fetch_registry_info,
    results_dir_for,
)
from tests.stress.constants import (
    BACKENDS,
    DEFAULT_CONCURRENCY,
    ENTITY_TYPES,
    HTTP_TIMEOUT_SECONDS,
    TARGET_SIZES,
    EntityType,
)
from tests.stress.generators._base import ensure_project_on_path

ensure_project_on_path()

from api.registry_client import (  # noqa: E402
    AgentRegistration,
    InternalServiceRegistration,
    SkillRegistrationRequest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class RegistrationRecord(BaseModel):
    payload_file: str
    outcome: str  # success | skipped | failed
    status_code: int | None = None
    latency_ms: float
    error: str | None = None


class RegistrationAggregate(BaseModel):
    entity_type: EntityType
    target_count: int
    registered: int
    skipped: int
    failed: int
    failure_rate: float
    wall_clock_seconds: float
    latency_ms: dict[str, float] = Field(default_factory=dict)
    failures: list[dict[str, str]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Token loading (mirrors api/registry_management.py:491)
# ---------------------------------------------------------------------------


def _load_token(token_file: Path) -> str:
    if not token_file.exists():
        raise FileNotFoundError(
            f"Token file not found: {token_file}. "
            "Generate one via the registry UI's 'Get JWT Token' button "
            "and save it as .oauth-tokens/ingress.json."
        )

    raw = token_file.read_text()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        token = raw.strip()
        if not token:
            raise RuntimeError(f"Empty token file: {token_file}") from None
        return token

    token = data.get("access_token")
    if not token and "tokens" in data:
        token = data["tokens"].get("access_token")
    if not token and "token_data" in data:
        token = data["token_data"].get("access_token")
    if not token:
        raise RuntimeError(f"No 'access_token' field found in token file: {token_file}")
    return token


# ---------------------------------------------------------------------------
# Payload normalization
# ---------------------------------------------------------------------------


def _server_form_data(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert a server payload JSON into form-encoded fields for /api/servers/register."""
    reg = InternalServiceRegistration(
        service_path=raw.get("path") or raw.get("service_path"),
        name=raw.get("server_name") or raw.get("name"),
        description=raw.get("description"),
        proxy_pass_url=raw.get("proxy_pass_url"),
        version=raw.get("version"),
        status=raw.get("status"),
        auth_scheme=raw.get("auth_scheme"),
        auth_provider=raw.get("auth_provider"),
        supported_transports=raw.get("supported_transports"),
        headers=raw.get("headers"),
        tags=raw.get("tags"),
        metadata=raw.get("metadata"),
        external_tags=raw.get("external_tags"),
        provider_organization=raw.get("provider_organization"),
        provider_url=raw.get("provider_url"),
    )
    data = reg.model_dump(exclude_none=True, by_alias=True)

    if isinstance(data.get("tags"), list):
        data["tags"] = ",".join(data["tags"])
    if isinstance(data.get("external_tags"), list):
        data["external_tags"] = ",".join(data["external_tags"])
    if isinstance(data.get("metadata"), dict):
        data["metadata"] = json.dumps(data["metadata"])
    return data


def _agent_json_data(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate and serialize an agent payload for /api/agents/register."""
    reg = AgentRegistration.model_validate(raw)
    data = reg.model_dump(exclude_none=True, by_alias=True)
    # The server-side schema requires supportedProtocol but the client model
    # treats it as optional. Re-inject from the raw payload if needed.
    if "supportedProtocol" not in data and raw.get("supportedProtocol"):
        data["supportedProtocol"] = raw["supportedProtocol"]
    return data


def _skill_json_data(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate and serialize a skill payload for /api/skills."""
    reg = SkillRegistrationRequest.model_validate(raw)
    return reg.model_dump(exclude_none=True)


# ---------------------------------------------------------------------------
# Per-entity HTTP plumbing
# ---------------------------------------------------------------------------


class EntityOps:
    """Encapsulates the differences between entity types for the loader."""

    def __init__(
        self,
        entity_type: EntityType,
        endpoint: str,
        send_kind: str,  # "form" | "json"
        payload_transform: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        self.entity_type = entity_type
        self.endpoint = endpoint
        self.send_kind = send_kind
        self.payload_transform = payload_transform


ENTITY_OPS: dict[EntityType, EntityOps] = {
    "servers": EntityOps("servers", "/api/servers/register", "form", _server_form_data),
    "agents": EntityOps("agents", "/api/agents/register", "json", _agent_json_data),
    "skills": EntityOps("skills", "/api/skills", "json", _skill_json_data),
}


# ---------------------------------------------------------------------------
# Core registration logic
# ---------------------------------------------------------------------------


async def _register_one(
    client: httpx.AsyncClient,
    base_url: str,
    ops: EntityOps,
    payload_file: Path,
    sem: asyncio.Semaphore,
    token_getter: Callable[[], str],
    token_refresher: Callable[[], Awaitable[None]],
) -> RegistrationRecord:
    async with sem:
        try:
            raw = json.loads(payload_file.read_text())
            raw.setdefault("status", "active")
            body = ops.payload_transform(raw)
        except Exception as exc:
            return RegistrationRecord(
                payload_file=payload_file.name,
                outcome="failed",
                latency_ms=0.0,
                error=f"payload_error: {exc}",
            )

        url = f"{base_url.rstrip('/')}{ops.endpoint}"
        result = await _post_with_retry(
            client=client,
            url=url,
            body=body,
            send_kind=ops.send_kind,
            payload_file=payload_file,
            token_getter=token_getter,
            token_refresher=token_refresher,
        )

        # Enable the entity after successful registration (all types default to disabled)
        if result.outcome == "success":
            path = raw.get("path") or raw.get("service_path", "")
            if not path.startswith("/"):
                path = "/" + path
            if path:
                auth_header = {"Authorization": f"Bearer {token_getter()}"}
                if ops.entity_type == "servers":
                    await client.post(
                        f"{base_url.rstrip('/')}/api/servers/toggle",
                        headers={**auth_header, "Content-Type": "application/x-www-form-urlencoded"},
                        content=f"path={path}&new_state=true",
                    )
                elif ops.entity_type == "agents":
                    agent_path = path.lstrip("/")
                    await client.post(
                        f"{base_url.rstrip('/')}/api/agents/{agent_path}/toggle",
                        headers=auth_header,
                        params={"enabled": "true"},
                    )
                elif ops.entity_type == "skills":
                    skill_path = path.lstrip("/")
                    if skill_path.startswith("skills/"):
                        skill_path = skill_path[len("skills/"):]
                    await client.post(
                        f"{base_url.rstrip('/')}/api/skills/{skill_path}/toggle",
                        headers={**auth_header, "Content-Type": "application/json"},
                        json={"enabled": True},
                    )

        return result


async def _post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    body: dict[str, Any],
    send_kind: str,
    payload_file: Path,
    token_getter: Callable[[], str],
    token_refresher: Callable[[], Awaitable[None]],
) -> RegistrationRecord:
    attempted_refresh = False
    while True:
        headers = {"Authorization": f"Bearer {token_getter()}"}
        start = time.perf_counter()
        try:
            if send_kind == "form":
                resp = await client.post(url, headers=headers, data=body)
            else:
                resp = await client.post(url, headers=headers, json=body)
        except httpx.HTTPError as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return RegistrationRecord(
                payload_file=payload_file.name,
                outcome="failed",
                latency_ms=elapsed_ms,
                error=f"http_error: {exc}",
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        if resp.status_code in (200, 201):
            return RegistrationRecord(
                payload_file=payload_file.name,
                outcome="success",
                status_code=resp.status_code,
                latency_ms=elapsed_ms,
            )

        if resp.status_code == 409 or _looks_like_conflict(resp):
            return RegistrationRecord(
                payload_file=payload_file.name,
                outcome="skipped",
                status_code=resp.status_code,
                latency_ms=elapsed_ms,
            )

        if resp.status_code == 401 and not attempted_refresh:
            attempted_refresh = True
            logger.warning("Got 401; refreshing token from file and retrying once")
            await token_refresher()
            continue

        return RegistrationRecord(
            payload_file=payload_file.name,
            outcome="failed",
            status_code=resp.status_code,
            latency_ms=elapsed_ms,
            error=_truncate(resp.text, 500),
        )


def _looks_like_conflict(resp: httpx.Response) -> bool:
    """Some endpoints return 400/422 with 'already exists' instead of 409."""
    if resp.status_code not in (400, 422):
        return False
    body_text = resp.text.lower()
    return "already exists" in body_text or "duplicate" in body_text


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(
    entity_type: EntityType,
    target_count: int,
    records: list[RegistrationRecord],
    wall_clock_seconds: float,
) -> RegistrationAggregate:
    registered = sum(1 for r in records if r.outcome == "success")
    skipped = sum(1 for r in records if r.outcome == "skipped")
    failed = sum(1 for r in records if r.outcome == "failed")

    failure_rate = failed / target_count if target_count else 0.0

    successful_latencies = [r.latency_ms for r in records if r.outcome == "success"]
    latency_ms = _percentiles(successful_latencies)

    failures = [
        {"payload": r.payload_file, "error": r.error or "unknown"}
        for r in records
        if r.outcome == "failed"
    ]

    return RegistrationAggregate(
        entity_type=entity_type,
        target_count=target_count,
        registered=registered,
        skipped=skipped,
        failed=failed,
        failure_rate=failure_rate,
        wall_clock_seconds=wall_clock_seconds,
        latency_ms=latency_ms,
        failures=failures,
    )


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    sorted_vals = sorted(values)
    return {
        "p50": _pct(sorted_vals, 50),
        "p95": _pct(sorted_vals, 95),
        "p99": _pct(sorted_vals, 99),
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "mean": statistics.mean(sorted_vals),
    }


def _pct(sorted_vals: list[float], pct: int) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * pct / 100
    lower = int(k)
    upper = min(lower + 1, len(sorted_vals) - 1)
    frac = k - lower
    return sorted_vals[lower] + (sorted_vals[upper] - sorted_vals[lower]) * frac


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def _register_entity_type(
    client: httpx.AsyncClient,
    base_url: str,
    entity_type: EntityType,
    payload_files: list[Path],
    concurrency: int,
    target_count: int,
    token_state: dict[str, Any],
    token_file: Path,
) -> RegistrationAggregate:
    if not payload_files:
        logger.warning("No payloads for %s; skipping", entity_type)
        return RegistrationAggregate(
            entity_type=entity_type,
            target_count=target_count,
            registered=0,
            skipped=0,
            failed=0,
            failure_rate=0.0,
            wall_clock_seconds=0.0,
        )

    ops = ENTITY_OPS[entity_type]
    sem = asyncio.Semaphore(concurrency)

    def token_getter() -> str:
        return token_state["token"]

    async def token_refresher() -> None:
        token_state["token"] = _load_token(token_file)

    start = time.time()
    tasks = [
        _register_one(client, base_url, ops, pf, sem, token_getter, token_refresher)
        for pf in payload_files
    ]
    records: list[RegistrationRecord] = await asyncio.gather(*tasks)
    elapsed = time.time() - start

    aggregate = _aggregate(entity_type, target_count, records, elapsed)
    logger.info(
        "%s: registered=%d skipped=%d failed=%d in %.1fs (p50=%.1fms, p95=%.1fms)",
        entity_type,
        aggregate.registered,
        aggregate.skipped,
        aggregate.failed,
        elapsed,
        aggregate.latency_ms.get("p50", 0.0),
        aggregate.latency_ms.get("p95", 0.0),
    )
    return aggregate


async def _main_async(args: argparse.Namespace) -> int:
    base_url = args.base_url
    token_file = args.token_file
    try:
        token = _load_token(token_file)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1
    token_state: dict[str, Any] = {"token": token}

    data_dir: Path = args.data_dir
    results_dir: Path = args.results_dir

    if args.entity_type == "all":
        entity_types: list[EntityType] = list(ENTITY_TYPES)
    else:
        entity_types = [args.entity_type]

    output_dir = results_dir_for(args.backend, args.count, results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_start = time.time()
    per_entity: dict[str, dict[str, Any]] = {}

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        for entity_type in entity_types:
            entity_dir = data_dir_for(entity_type, args.count, data_dir)
            payload_files = sorted(entity_dir.glob("*.json"))
            logger.info(
                "Registering %d %s payloads from %s",
                len(payload_files),
                entity_type,
                entity_dir,
            )
            aggregate = await _register_entity_type(
                client=client,
                base_url=base_url,
                entity_type=entity_type,
                payload_files=payload_files,
                concurrency=args.concurrency,
                target_count=args.count,
                token_state=token_state,
                token_file=token_file,
            )
            per_entity[entity_type] = aggregate.model_dump()

    registry_info = fetch_registry_info(base_url, token)

    overall = {
        "backend": args.backend,
        "size": args.count,
        "base_url": base_url,
        "concurrency": args.concurrency,
        "wall_clock_seconds": time.time() - overall_start,
        "registry_info": registry_info,
        "entity_types": per_entity,
    }

    out_file = output_dir / "registration.json"
    out_file.write_text(json.dumps(overall, indent=2, default=str))
    logger.info("Wrote registration report: %s", out_file)

    any_failed = any(per_entity[et]["failure_rate"] >= 0.01 for et in per_entity)
    return 1 if any_failed else 0


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bulk-register stress-test payloads against a running registry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--entity-type",
        choices=[*ENTITY_TYPES, "all"],
        default="all",
        help="Which entity type to register (default: all).",
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        choices=TARGET_SIZES,
        help=f"Source size to register from (one of {TARGET_SIZES}).",
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        required=True,
        help="Backend label for the results directory.",
    )
    parser.add_argument(
        "--base-url",
        default=default_base_url(),
        help="Registry base URL (default: $STRESS_BASE_URL or http://localhost).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Concurrent registrations (default: {DEFAULT_CONCURRENCY}).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="Root data directory (default: tests/stress/data/).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results_dir(),
        help="Root results directory (default: tests/stress/results/).",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        default=default_token_file(),
        help="JWT token file (default: .oauth-tokens/ingress.json).",
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
