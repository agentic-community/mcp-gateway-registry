#!/usr/bin/env python3
"""Live smoke test for the agent PUT-update path and the async batch endpoint.

Runs against a deployed registry (through nginx, port 80 by default) and
exercises two flows end to end:

1. PUT update path. Registers a fresh agent, fetches it (capturing the ETag),
   PUT-updates a field, and verifies the change took effect on a re-GET.

2. Async batch path. Submits a single batch job mixing register/patch/delete
   items to POST /api/agents/batch, then polls GET /api/agents/batch/{job_id}
   until the job reaches a terminal state, reporting per-item results.

Everything created is cleaned up at the end (best effort), unless --keep is set.

Auth mirrors cli/agent_mgmt.py: a JWT Bearer token is loaded from
.oauth-tokens/ingress.json by default, or supplied via --token / $AGENT_TOKEN.

Usage:
    uv run python scripts/test_agent_batch_live.py
    uv run python scripts/test_agent_batch_live.py --base-url https://registry.example.com
    uv run python scripts/test_agent_batch_live.py --token "$AGENT_TOKEN" --keep
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger("agent-batch-live")

DEFAULT_BASE_URL: str = "http://localhost"
DEFAULT_TOKEN_FILE: str = ".token"
API_BASE: str = "/api/agents"
REQUEST_TIMEOUT: int = 30
TERMINAL_STATES: frozenset[str] = frozenset({"succeeded", "partial", "failed"})
MODE_REQUESTS: str = "requests"
MODE_CLIENT: str = "client"


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class JobTiming:
    """Timing record for a completed batch job."""

    flow: str
    job_id: str
    items: int
    duration_s: float
    state: str


@dataclass
class Summary:
    base_url: str
    checks: list[CheckResult] = field(default_factory=list)
    created_paths: list[str] = field(default_factory=list)
    job_timings: list[JobTiming] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> bool:
        self.checks.append(CheckResult(name=name, passed=passed, detail=detail))
        marker = "PASS" if passed else "FAIL"
        logger.info(f"[{marker}] {name}{f' — {detail}' if detail else ''}")
        return passed

    def record_job(
        self,
        flow: str,
        job_id: str,
        items: int,
        duration_s: float,
        state: str,
    ) -> None:
        self.job_timings.append(
            JobTiming(flow=flow, job_id=job_id, items=items, duration_s=duration_s, state=state)
        )

    def print_timing_report(self) -> None:
        if not self.job_timings:
            return
        logger.info("")
        logger.info("=" * 72)
        logger.info("BATCH JOB TIMING REPORT")
        logger.info("=" * 72)
        logger.info(f"{'Flow':<30} {'Job ID':<12} {'Items':>5} {'Time (s)':>9} {'State':<10}")
        logger.info("-" * 72)
        total_items = 0
        total_time = 0.0
        for t in self.job_timings:
            logger.info(
                f"{t.flow:<30} {t.job_id[:10]:<12} {t.items:>5} {t.duration_s:>9.2f} {t.state:<10}"
            )
            total_items += t.items
            total_time += t.duration_s
        logger.info("-" * 72)
        logger.info(
            f"{'TOTAL':<30} {'':<12} {total_items:>5} {total_time:>9.2f}"
        )
        if total_items > 0 and total_time > 0:
            logger.info(f"Throughput: {total_items / total_time:.1f} items/sec (wall clock)")
        logger.info("=" * 72)


def _load_token(token_file: str) -> str:
    """Load a JWT Bearer token from a file.

    Supports three formats:
    - Raw JWT string (no JSON, just the token text)
    - Flat JSON: {"access_token": "..."} or {"token": "..."}
    - Nested JSON: {"tokens": {"access_token": "..."}}
    """
    abs_path = os.path.abspath(token_file)
    with open(abs_path) as f:
        content = f.read().strip()

    # Try raw JWT (starts with eyJ)
    if content.startswith("eyJ"):
        logger.info(f"Token loaded from {abs_path} (raw JWT, length {len(content)})")
        return content

    # Try JSON formats
    data = json.loads(content)

    # Nested: {"tokens": {"access_token": "..."}}
    if isinstance(data.get("tokens"), dict):
        token = data["tokens"].get("access_token") or data["tokens"].get("token")
        if token:
            logger.info(f"Token loaded from {abs_path} (nested JSON, length {len(token)})")
            return token

    # Flat: {"access_token": "..."} or {"token": "..."}
    token = data.get("access_token") or data.get("token")
    if token:
        logger.info(f"Token loaded from {abs_path} (flat JSON, length {len(token)})")
        return token

    raise ValueError(f"No access_token/token found in {abs_path}")


def _resolve_token(
    cli_token: str | None,
    token_file: str,
) -> str:
    """Resolve the Bearer token from --token, $AGENT_TOKEN, or the token file."""
    if cli_token:
        logger.info("Using token from --token argument")
        return cli_token

    env_token = os.getenv("AGENT_TOKEN")
    if env_token:
        logger.info("Using token from $AGENT_TOKEN")
        return env_token

    return _load_token(token_file)


def _request(
    session: requests.Session,
    method: str,
    url: str,
    json_body: dict | None = None,
    extra_headers: dict[str, str] | None = None,
) -> requests.Response:
    """Issue an authenticated request and log a concise summary."""
    headers = dict(extra_headers or {})
    logger.info(f"-> {method} {url}")
    response = session.request(
        method=method,
        url=url,
        json=json_body,
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )
    logger.info(f"<- {response.status_code} ({len(response.content)} bytes)")
    if response.status_code >= 400:
        try:
            logger.warning(f"   body: {json.dumps(response.json())[:500]}")
        except ValueError:
            logger.warning(f"   body: {response.text[:500]}")
    return response


class _ClientTransport:
    """Transport adapter that uses api/registry_client.py instead of raw requests.

    Provides the same submit/poll/delete interface as the requests-based path
    but exercises the RegistryClient SDK layer that CLI users call.
    """

    def __init__(self, base_url: str, token: str):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        from api.registry_client import RegistryClient

        self._client = RegistryClient(registry_url=base_url, token=token)
        self._base_url = base_url
        self._token = token

    def submit_batch(
        self,
        items: list[dict[str, Any]],
        idempotency_key: str | None = None,
    ) -> tuple[int, dict]:
        """Submit a batch job via the client. Returns (status_code, response_dict)."""
        try:
            resp = self._client.submit_agent_batch(items, idempotency_key=idempotency_key)
            return 202, {
                "job_id": resp.job_id,
                "status_url": resp.status_url,
            }
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", 500)
            return status, {"error": str(e)}

    def get_batch_status(self, job_id: str) -> tuple[int, dict]:
        """Poll a batch job via the client. Returns (status_code, job_dict)."""
        try:
            job = self._client.get_agent_batch(job_id)
            return 200, job.model_dump()
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", 500)
            return status, {"error": str(e)}

    def register_agent(self, payload: dict) -> tuple[int, dict]:
        """Register an agent via raw POST (client doesn't have a typed method for this)."""
        resp = requests.post(
            f"{self._base_url}{API_BASE}/register",
            json=payload,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
            timeout=REQUEST_TIMEOUT,
        )
        body = resp.json() if resp.content else {}
        return resp.status_code, body

    def delete_agent(self, path: str) -> int:
        """Delete an agent. Returns status code."""
        resp = requests.delete(
            f"{self._base_url}{API_BASE}{path}",
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=REQUEST_TIMEOUT,
        )
        return resp.status_code


def _register_payload(
    name: str,
    path: str,
    description: str,
) -> dict:
    """Build a minimal valid AgentRegistrationRequest body."""
    return {
        "name": name,
        "path": path,
        "description": description,
        "url": f"http://{path.strip('/')}.invalid:9000/",
        "supportedProtocol": "other",
        "version": "0.0.1",
        "tags": ["batch-live-test"],
        "skills": [
            {
                "id": "echo",
                "name": "Echo",
                "description": "Echoes input back.",
                "tags": [],
            }
        ],
    }


def _run_put_flow(
    session: requests.Session,
    base_url: str,
    summary: Summary,
) -> None:
    """Register an agent, PUT-update it, and verify the change."""
    run_id = uuid.uuid4().hex[:8]
    path = f"/batch-live-put-{run_id}"
    name = f"Batch Live PUT {run_id}"

    reg = _request(
        session,
        "POST",
        f"{base_url}{API_BASE}/register",
        json_body=_register_payload(name, path, "Original description"),
    )
    if not summary.add(
        "PUT flow: register agent", reg.status_code == 201, f"HTTP {reg.status_code}"
    ):
        return
    summary.created_paths.append(path)

    get_before = _request(session, "GET", f"{base_url}{API_BASE}{path}")
    etag = get_before.headers.get("ETag")
    summary.add(
        "PUT flow: GET returns ETag",
        bool(etag),
        f"ETag={etag}",
    )

    updated_desc = "Updated description via PUT"
    put = _request(
        session,
        "PUT",
        f"{base_url}{API_BASE}{path}",
        json_body=_register_payload(name, path, updated_desc),
    )
    if not summary.add("PUT flow: update agent", put.status_code == 200, f"HTTP {put.status_code}"):
        return

    get_after = _request(session, "GET", f"{base_url}{API_BASE}{path}")
    actual_desc = get_after.json().get("description") if get_after.ok else None
    summary.add(
        "PUT flow: description updated",
        actual_desc == updated_desc,
        f"description={actual_desc!r}",
    )
    new_etag = get_after.headers.get("ETag")
    summary.add(
        "PUT flow: ETag changed after update",
        bool(new_etag) and new_etag != etag,
        f"{etag} -> {new_etag}",
    )


def _poll_batch_job(
    session: requests.Session,
    status_url: str,
    timeout_s: float,
    interval_s: float,
) -> dict | None:
    """Poll a batch job until it reaches a terminal state or times out."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = _request(session, "GET", status_url)
        if not resp.ok:
            logger.warning(f"   poll returned HTTP {resp.status_code}")
            return None
        job = resp.json()
        state = job.get("state")
        logger.info(
            f"   job {job.get('job_id')} state={state} "
            f"succeeded={job.get('succeeded')} failed={job.get('failed')}/{job.get('total')}"
        )
        if state in TERMINAL_STATES:
            return job
        time.sleep(interval_s)
    logger.error("   batch job did not reach a terminal state before timeout")
    return None


def _poll_batch_job_client(
    transport: _ClientTransport,
    job_id: str,
    timeout_s: float,
    interval_s: float,
) -> dict | None:
    """Poll a batch job via the registry_client transport."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status_code, job = transport.get_batch_status(job_id)
        if status_code != 200:
            logger.warning(f"   client poll returned status {status_code}")
            return None
        state = job.get("state")
        logger.info(
            f"   job {job.get('job_id')} state={state} "
            f"succeeded={job.get('succeeded')} failed={job.get('failed')}/{job.get('total')}"
        )
        if state in TERMINAL_STATES:
            return job
        time.sleep(interval_s)
    logger.error("   batch job did not reach a terminal state before timeout")
    return None


def _run_batch_flow(
    session: requests.Session,
    base_url: str,
    summary: Summary,
    poll_timeout_s: float,
    poll_interval_s: float,
) -> None:
    """Submit a mixed batch job and verify per-item outcomes."""
    run_id = uuid.uuid4().hex[:8]
    seed_path = f"/batch-live-seed-{run_id}"
    new_path = f"/batch-live-new-{run_id}"
    seed_name = f"Batch Live Seed {run_id}"

    # Seed an agent that the batch will patch and delete.
    seed = _request(
        session,
        "POST",
        f"{base_url}{API_BASE}/register",
        json_body=_register_payload(seed_name, seed_path, "Seed agent for batch"),
    )
    if not summary.add(
        "Batch flow: seed agent registered",
        seed.status_code == 201,
        f"HTTP {seed.status_code}",
    ):
        return
    summary.created_paths.append(seed_path)

    batch_body = {
        "idempotency_key": f"batch-live-{run_id}",
        "items": [
            {
                "op": "register",
                "card": _register_payload(f"Batch Live New {run_id}", new_path, "Created by batch"),
            },
            {
                "op": "patch",
                "path": seed_path,
                "card": {"description": "Patched by batch"},
            },
            {
                "op": "delete",
                "path": seed_path,
            },
        ],
    }

    submit = _request(
        session,
        "POST",
        f"{base_url}{API_BASE}/batch",
        json_body=batch_body,
    )
    if not summary.add(
        "Batch flow: submit accepted (202)",
        submit.status_code == 202,
        f"HTTP {submit.status_code}",
    ):
        return

    payload = submit.json()
    status_url = f"{base_url}{payload['status_url']}"
    # Track the register item's path so cleanup removes it even if delete ran.
    summary.created_paths.append(new_path)

    job = _poll_batch_job(session, status_url, poll_timeout_s, poll_interval_s)
    if not summary.add("Batch flow: job reached terminal state", job is not None):
        return

    summary.add(
        "Batch flow: all 3 items succeeded",
        job.get("state") == "succeeded" and job.get("succeeded") == 3,
        f"state={job.get('state')} succeeded={job.get('succeeded')}/{job.get('total')}",
    )

    results = {r["op"]: r for r in job.get("results", [])}
    for op, r in results.items():
        if r.get("status", 0) >= 400:
            logger.warning(f"   item op={op} status={r.get('status')} error={r.get('error')}")
    summary.add(
        "Batch flow: register item ok",
        results.get("register", {}).get("status") in (200, 201),
        f"status={results.get('register', {}).get('status')}",
    )
    summary.add(
        "Batch flow: patch item ok",
        results.get("patch", {}).get("status") == 200,
        f"status={results.get('patch', {}).get('status')}",
    )
    summary.add(
        "Batch flow: delete item ok",
        results.get("delete", {}).get("status") in (200, 204),
        f"status={results.get('delete', {}).get('status')}",
    )


def _run_bulk_batch_flow(
    session: requests.Session,
    base_url: str,
    summary: Summary,
    poll_timeout_s: float,
    poll_interval_s: float,
    count: int = 100,
) -> None:
    """Register `count` agents in one batch, then patch half and delete half."""
    run_id = uuid.uuid4().hex[:8]
    logger.info(f"Bulk batch flow: registering {count} agents (run_id={run_id})")

    paths = [f"/bulk-{run_id}-{i:04d}" for i in range(count)]
    register_items = [
        {
            "op": "register",
            "card": _register_payload(
                f"Bulk Agent {run_id}/{i:04d}",
                paths[i],
                f"Bulk-registered agent {i} of {count}",
            ),
        }
        for i in range(count)
    ]

    batch_body = {
        "idempotency_key": f"bulk-register-{run_id}",
        "items": register_items,
    }

    start = time.time()
    submit = _request(session, "POST", f"{base_url}{API_BASE}/batch", json_body=batch_body)
    if not summary.add(
        f"Bulk register: submit {count} items accepted (202)",
        submit.status_code == 202,
        f"HTTP {submit.status_code}",
    ):
        return

    payload = submit.json()
    status_url = f"{base_url}{payload['status_url']}"
    job_id = payload["job_id"]
    summary.created_paths.extend(paths)

    job = _poll_batch_job(session, status_url, poll_timeout_s, poll_interval_s)
    duration = time.time() - start
    if not summary.add("Bulk register: job reached terminal state", job is not None):
        return

    summary.add(
        f"Bulk register: all {count} succeeded",
        job.get("state") == "succeeded" and job.get("succeeded") == count,
        f"state={job.get('state')} succeeded={job.get('succeeded')}/{job.get('total')}",
    )
    summary.record_job("bulk-register", job_id, count, duration, job.get("state", "?"))

    half = count // 2
    logger.info(f"Bulk batch flow: patching {half} agents, deleting {half} agents")

    mutate_items = []
    for i in range(half):
        mutate_items.append({
            "op": "patch",
            "path": paths[i],
            "card": {"description": f"Bulk-patched at index {i}"},
        })
    for i in range(half, count):
        mutate_items.append({
            "op": "delete",
            "path": paths[i],
        })

    mutate_body = {
        "idempotency_key": f"bulk-mutate-{run_id}",
        "items": mutate_items,
    }

    start2 = time.time()
    submit2 = _request(session, "POST", f"{base_url}{API_BASE}/batch", json_body=mutate_body)
    if not summary.add(
        f"Bulk mutate: submit {count} items accepted (202)",
        submit2.status_code == 202,
        f"HTTP {submit2.status_code}",
    ):
        return

    payload2 = submit2.json()
    status_url2 = f"{base_url}{payload2['status_url']}"
    job_id2 = payload2["job_id"]

    job2 = _poll_batch_job(session, status_url2, poll_timeout_s, poll_interval_s)
    duration2 = time.time() - start2
    if not summary.add("Bulk mutate: job reached terminal state", job2 is not None):
        return

    summary.add(
        f"Bulk mutate: all {count} items succeeded",
        job2.get("state") == "succeeded" and job2.get("succeeded") == count,
        f"state={job2.get('state')} succeeded={job2.get('succeeded')}/{job2.get('total')}",
    )
    summary.record_job("bulk-mutate", job_id2, count, duration2, job2.get("state", "?"))

    failed_items = [r for r in job2.get("results", []) if r.get("status", 0) >= 400]
    if failed_items:
        for r in failed_items[:5]:
            logger.warning(
                f"   failed item: op={r.get('op')} path={r.get('path')} "
                f"status={r.get('status')} error={r.get('error')}"
            )

    # Remove deleted paths from cleanup list (they're already gone)
    deleted_paths = set(paths[half:])
    summary.created_paths = [p for p in summary.created_paths if p not in deleted_paths]


def _run_parallel_batch_flow(
    session: requests.Session,
    base_url: str,
    summary: Summary,
    poll_timeout_s: float,
    poll_interval_s: float,
    num_jobs: int = 3,
    agents_per_job: int = 20,
) -> None:
    """Submit multiple batch jobs concurrently and track each to completion."""
    run_id = uuid.uuid4().hex[:8]
    logger.info(
        f"Parallel batch flow: submitting {num_jobs} jobs x {agents_per_job} agents "
        f"(run_id={run_id})"
    )

    job_ids = []
    all_paths = []
    submit_start = time.time()

    for job_num in range(num_jobs):
        paths = [f"/parallel-{run_id}-j{job_num}-{i:03d}" for i in range(agents_per_job)]
        all_paths.extend(paths)
        items = [
            {
                "op": "register",
                "card": _register_payload(
                    f"Parallel {run_id}/j{job_num}/{i:03d}",
                    paths[i],
                    f"Parallel job {job_num} agent {i}",
                ),
            }
            for i in range(agents_per_job)
        ]
        batch_body = {
            "idempotency_key": f"parallel-{run_id}-j{job_num}",
            "items": items,
        }

        submit = _request(session, "POST", f"{base_url}{API_BASE}/batch", json_body=batch_body)
        if not summary.add(
            f"Parallel: job {job_num} submit accepted (202)",
            submit.status_code == 202,
            f"HTTP {submit.status_code}",
        ):
            continue
        payload = submit.json()
        job_ids.append((job_num, payload["job_id"], f"{base_url}{payload['status_url']}"))

    summary.created_paths.extend(all_paths)

    if not job_ids:
        return

    logger.info(f"Parallel batch flow: tracking {len(job_ids)} jobs to completion")

    for job_num, job_id, status_url in job_ids:
        job = _poll_batch_job(session, status_url, poll_timeout_s, poll_interval_s)
        duration = time.time() - submit_start
        if not summary.add(
            f"Parallel: job {job_num} ({job_id[:8]}) reached terminal state",
            job is not None,
        ):
            continue
        summary.add(
            f"Parallel: job {job_num} all {agents_per_job} succeeded",
            job.get("state") == "succeeded" and job.get("succeeded") == agents_per_job,
            f"state={job.get('state')} succeeded={job.get('succeeded')}/{job.get('total')}",
        )
        summary.record_job(
            f"parallel-j{job_num}", job_id, agents_per_job, duration, job.get("state", "?")
        )


def _run_client_mode_flows(
    transport: _ClientTransport,
    summary: Summary,
    poll_timeout_s: float,
    poll_interval_s: float,
) -> None:
    """Run the batch flows using the registry_client transport layer."""
    run_id = uuid.uuid4().hex[:8]
    logger.info(f"CLIENT MODE: Running batch flows via registry_client (run_id={run_id})")

    count = 100
    paths = [f"/client-bulk-{run_id}-{i:04d}" for i in range(count)]
    register_items = [
        {
            "op": "register",
            "card": _register_payload(
                f"Client Bulk {run_id}/{i:04d}",
                paths[i],
                f"Client-mode bulk agent {i}",
            ),
        }
        for i in range(count)
    ]

    start = time.time()
    status_code, resp = transport.submit_batch(
        register_items, idempotency_key=f"client-bulk-{run_id}"
    )
    if not summary.add(
        f"Client: bulk register {count} submit accepted",
        status_code == 202,
        f"HTTP {status_code}",
    ):
        return
    summary.created_paths.extend(paths)

    job_id = resp["job_id"]
    job = _poll_batch_job_client(transport, job_id, poll_timeout_s, poll_interval_s)
    duration = time.time() - start
    if not summary.add("Client: bulk register job completed", job is not None):
        return
    summary.add(
        f"Client: bulk register all {count} succeeded",
        job.get("state") == "succeeded" and job.get("succeeded") == count,
        f"state={job.get('state')} succeeded={job.get('succeeded')}/{job.get('total')}",
    )
    summary.record_job("client-bulk-register", job_id, count, duration, job.get("state", "?"))

    # Submit 3 parallel jobs via client
    num_jobs = 3
    per_job = 20
    logger.info(f"CLIENT MODE: submitting {num_jobs} parallel jobs x {per_job} agents")
    job_starts: list[tuple[int, str, float]] = []

    for j in range(num_jobs):
        j_paths = [f"/client-par-{run_id}-j{j}-{i:03d}" for i in range(per_job)]
        summary.created_paths.extend(j_paths)
        items = [
            {
                "op": "register",
                "card": _register_payload(
                    f"Client Par {run_id}/j{j}/{i:03d}",
                    j_paths[i],
                    f"Client parallel job {j} agent {i}",
                ),
            }
            for i in range(per_job)
        ]
        j_start = time.time()
        sc, r = transport.submit_batch(items, idempotency_key=f"client-par-{run_id}-j{j}")
        if not summary.add(
            f"Client parallel: job {j} submit accepted",
            sc == 202,
            f"HTTP {sc}",
        ):
            continue
        job_starts.append((j, r["job_id"], j_start))

    for j, jid, j_start in job_starts:
        job = _poll_batch_job_client(transport, jid, poll_timeout_s, poll_interval_s)
        j_duration = time.time() - j_start
        if not summary.add(f"Client parallel: job {j} completed", job is not None):
            continue
        summary.add(
            f"Client parallel: job {j} all {per_job} succeeded",
            job.get("state") == "succeeded" and job.get("succeeded") == per_job,
            f"state={job.get('state')} succeeded={job.get('succeeded')}/{job.get('total')}",
        )
        summary.record_job(
            f"client-parallel-j{j}", jid, per_job, j_duration, job.get("state", "?")
        )


def _cleanup(
    session: requests.Session,
    base_url: str,
    paths: list[str],
) -> None:
    """Best-effort delete of every agent created during the run."""
    for path in paths:
        resp = _request(session, "DELETE", f"{base_url}{API_BASE}{path}")
        if resp.status_code in (200, 204, 404):
            logger.info(f"   cleaned up {path} (HTTP {resp.status_code})")
        else:
            logger.warning(f"   cleanup of {path} returned HTTP {resp.status_code}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live smoke test for agent PUT-update and async batch endpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Registry base URL through nginx (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="JWT Bearer token. Falls back to $AGENT_TOKEN, then the token file.",
    )
    parser.add_argument(
        "--token-file",
        default=DEFAULT_TOKEN_FILE,
        help=f"Path to credentials JSON (default: {DEFAULT_TOKEN_FILE})",
    )
    parser.add_argument(
        "--poll-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for the batch job to finish (default: 60)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between batch status polls (default: 2)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Do not delete agents created during the run",
    )
    parser.add_argument(
        "--mode",
        choices=[MODE_REQUESTS, MODE_CLIENT, "both"],
        default=MODE_REQUESTS,
        help=(
            "Transport mode: 'requests' uses raw HTTP (default), "
            "'client' uses api/registry_client.py, 'both' runs both in sequence"
        ),
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    base_url = args.base_url.rstrip("/")

    token = _resolve_token(args.token, args.token_file)
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
    )

    summary = Summary(base_url=base_url)
    try:
        run_requests = args.mode in (MODE_REQUESTS, "both")
        run_client = args.mode in (MODE_CLIENT, "both")

        if run_requests:
            logger.info("=" * 60)
            logger.info("REQUESTS MODE (raw HTTP)")
            logger.info("=" * 60)
            _run_put_flow(session, base_url, summary)
            _run_batch_flow(session, base_url, summary, args.poll_timeout, args.poll_interval)
            _run_bulk_batch_flow(
                session, base_url, summary, args.poll_timeout, args.poll_interval, count=100
            )
            _run_parallel_batch_flow(
                session, base_url, summary, args.poll_timeout, args.poll_interval,
                num_jobs=3, agents_per_job=20,
            )

        if run_client:
            logger.info("=" * 60)
            logger.info("CLIENT MODE (api/registry_client.py)")
            logger.info("=" * 60)
            transport = _ClientTransport(base_url, token)
            _run_client_mode_flows(
                transport, summary, args.poll_timeout, args.poll_interval
            )
    finally:
        if args.keep:
            logger.info(f"--keep set; leaving {len(summary.created_paths)} agent(s) in place")
        elif summary.created_paths:
            logger.info(f"Cleaning up {len(summary.created_paths)} agent(s)...")
            _cleanup(session, base_url, summary.created_paths)

    passed = sum(1 for c in summary.checks if c.passed)
    total = len(summary.checks)
    logger.info("")
    logger.info(f"Summary: {passed}/{total} checks passed against {base_url}")
    for check in summary.checks:
        marker = "PASS" if check.passed else "FAIL"
        logger.info(f"  [{marker}] {check.name}")

    summary.print_timing_report()

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
