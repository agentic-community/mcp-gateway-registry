"""Compute infra + embeddings spend for the AI Registry customer fleet.

Cost model (auditable, documented below):

1. Infra compute -- per-compute-platform daily rate.
   Rates are grounded in the committed deployment artefacts: the ECS
   rate is derived from the live terraform.tfstate in
   terraform/aws-ecs/, and the EKS rate is derived from the Helm chart
   defaults in charts/ plus the standard aws-load-balancer-controller
   ingress pattern. All prices are us-east-1 on-demand.

     docker     -> $3.99/day : one t3.xlarge running docker-compose.
                   24h * $0.1664/hr = $3.99.
                   Customer VM; no managed-AWS services implied.

     ecs        -> $26.04/day : grounded in a measured AWS Cost Explorer
                   day (May-30) for the terraform/aws-ecs deployment.
                   Excludes shared-account overhead (CloudTrail, Others) and
                   the standalone EC2-Instances row (a separate box, not ECS).
                   Itemized (per-day, from the bill):
                     Elastic Container Service (Fargate vCPU + memory)
                       = $7.66/day
                     EC2-Other (NAT-per-AZ + EBS + inter-AZ data transfer)
                       = $7.39/day
                     RDS (Keycloak)
                       = $4.47/day
                     DocumentDB (with MongoDB compatibility)
                       = $2.03/day
                     VPC (endpoints, etc.)
                       = $1.80/day
                     CloudWatch (logs + metric alarms)
                       = $1.61/day
                     Elastic Load Balancing (ALBs)
                       = $1.08/day
                   Total = $26.04/day.
                   ECS tasks run on FARGATE (no EC2-backed cluster); the NAT
                   cost lands under EC2-Other because Fargate tasks in private
                   subnets egress through it. See terraform/aws-ecs/*.tf.

     kubernetes -> $18.58/day : grounded in the measured EKS reference
                   deployment (3-node managed node group + single ALB).
                   Itemized:
                     EKS control plane ($0.10/hr)
                       = $2.40/day
                     3 x m6i.xlarge worker nodes
                       = $13.82/day
                     EBS gp3 (~41 Gi)
                       = $0.11/day
                     1 Application Load Balancer
                       = $0.92/day
                     1 NAT Gateway + 5 GB egress
                       = $1.31/day
                     ACM public cert
                       = $0.00/day
                     Route 53 hosted zone
                       = $0.02/day
                   Total = $18.58/day.

     ec2        -> $3.99/day : single VM, same as docker fallback.

     unknown / anything else -> $3.99/day : conservative docker fallback.

   A per-instance-day charge is assessed for every distinct (AWS customer
   instance, day) pair we saw any event for. Platform is determined from
   the instance's most-recent non-empty `compute` field -- if an instance
   migrates from docker to kubernetes mid-window, it's billed at the
   kubernetes rate for the entire window (the alternative is per-event
   attribution, which double-counts).

2. Bedrock Titan embeddings (only for instances whose most recent
   embeddings_backend_kind == "bedrock"):
     Titan Text Embeddings v2 = $0.00002 per 1K tokens.
     We assume 100 tokens per search query.
     Per-instance daily Bedrock cost =
         delta(search_queries_total) on that day * 100 / 1000 * $0.00002

     delta is computed from the instance's search_queries_total timeseries.
     If the first event we see for an instance on a day already has a
     non-zero counter, that counter value is NOT retroactively charged
     to earlier days (we only charge the delta since the previous event
     we saw). This is conservative; it matches "how many queries hit
     Bedrock DURING the reporting window" rather than "how many queries
     did this instance ever run".

Filters:
- Customer-only: known-internal instance UUIDs are excluded.
- AWS-only: only instances whose last-seen cloud == "aws" are charged.
  GCP, Azure, and unknown clouds are excluded from BOTH compute and
  embeddings totals. (Those customers may still be running Bedrock via
  cross-account roles, but we don't have visibility to attribute it.)

Outputs:
- PNG chart with three panels: daily compute $, daily Bedrock $, and
  cumulative LTV $.
- CSV sidecar with per-day rows (date, aws_instances, queries_today,
  compute_usd, bedrock_usd, total_usd, cum_total_usd) for diffing in
  future reports.
- JSON summary with headline numbers (yesterday_usd, ltv_usd, etc.) that
  the report narrative can quote. Includes a per-platform breakdown so
  the report can show which compute platform drives most of the spend.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
import sys as _sys

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tufte_style import apply_tufte_style, tufte_axes  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

EC2_INSTANCE_TYPE: str = "t3.xlarge"
EC2_HOURLY_RATE: float = 0.1664
EC2_DAILY_RATE: float = 24 * EC2_HOURLY_RATE

# Per-compute-platform daily infra rate in USD.
# See the module docstring for the tfstate + Helm chart grounding.
# docker / ec2 / unknown / vm all map to a single customer VM running
# docker-compose, priced off EC2_DAILY_RATE (t3.xlarge) so the hourly-rate
# constant above stays the single source of truth (no hardcoded duplicate).
COMPUTE_PLATFORM_DAILY_RATE_USD: dict[str, float] = {
    "docker": EC2_DAILY_RATE,
    "ecs": 26.04,
    "kubernetes": 18.58,
    "ec2": EC2_DAILY_RATE,
    "unknown": EC2_DAILY_RATE,
    "vm": EC2_DAILY_RATE,
    "": EC2_DAILY_RATE,
}

BEDROCK_MODEL: str = "amazon.titan-embed-text-v2"
BEDROCK_PRICE_PER_1K_TOKENS: float = 0.00002
TOKENS_PER_QUERY: int = 100
BEDROCK_COST_PER_QUERY: float = (TOKENS_PER_QUERY / 1000.0) * BEDROCK_PRICE_PER_1K_TOKENS


def _daily_rate_for_platform(
    platform: str,
) -> float:
    """Return the daily USD rate for a given compute platform string."""
    key = (platform or "").strip().lower()
    return COMPUTE_PLATFORM_DAILY_RATE_USD.get(key, COMPUTE_PLATFORM_DAILY_RATE_USD[""])

FIGURE_WIDTH: int = 14
FIGURE_HEIGHT: int = 9
CHART_TITLE: str = (
    "AI Registry -- Customer AWS infra spend "
    "(per-platform: docker $3.99 / ecs $26.04 / k8s $18.58 per day + Bedrock Titan)"
)


def _find_csv_files(
    directory: str,
) -> list[str]:
    """Find all registry_metrics.csv files in the directory and dated subdirectories."""
    csv_files: list[str] = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith(".csv"):
            csv_files.append(filepath)
        elif os.path.isdir(filepath):
            for subfile in os.listdir(filepath):
                if subfile.endswith(".csv"):
                    csv_files.append(os.path.join(filepath, subfile))
    csv_files.sort()
    logger.info(f"Found {len(csv_files)} CSV files in {directory}")
    return csv_files


def _load_internal_instance_ids(
    path: str | None,
) -> set[str]:
    """Parse the known-internal-instances.md file into a set of full UUIDs."""
    if not path or not Path(path).exists():
        return set()
    pattern = re.compile(r"`([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})`")
    ids: set[str] = set()
    with open(path) as f:
        for line in f:
            for match in pattern.findall(line):
                ids.add(match)
    logger.info(f"Loaded {len(ids)} known internal instance IDs from {path}")
    return ids


def _read_all_csvs(
    csv_files: list[str],
) -> list[dict[str, str]]:
    """Read and concatenate all CSV files."""
    all_rows: list[dict[str, str]] = []
    for csv_path in csv_files:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        logger.info(f"Read {len(rows)} rows from {csv_path}")
        all_rows.extend(rows)
    logger.info(f"Total rows across all CSVs: {len(all_rows)}")
    return all_rows


def _dedupe_by_id_ts(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Drop duplicate rows by (registry_id, ts)."""
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for r in rows:
        rid = (r.get("registry_id") or "").strip()
        ts = (r.get("ts") or "").strip()
        key = (rid, ts)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    logger.info(f"Deduplicated: {len(rows)} -> {len(out)} unique events")
    return out


def _is_aws_customer(
    r: dict[str, str],
    internal_ids: set[str],
) -> bool:
    """Return True if this row should contribute to spend."""
    rid = (r.get("registry_id") or "").strip()
    if not rid or rid in internal_ids:
        return False
    cloud = (r.get("cloud") or "").strip()
    return cloud == "aws"


def _parse_int(
    v: str | None,
) -> int:
    """Parse an int safely, returning 0 on failure or empty."""
    if not v:
        return 0
    try:
        return int(v.strip())
    except (ValueError, AttributeError):
        return 0


def _compute_per_instance_latest_backend(
    rows: list[dict[str, str]],
    internal_ids: set[str],
) -> dict[str, str]:
    """For each AWS customer instance, return the most-recent non-empty embeddings_backend_kind.

    Instances without a populated value get "unknown" (pre-v1.0.22 registries).
    """
    latest: dict[str, tuple[str, str]] = {}
    for r in rows:
        if not _is_aws_customer(r, internal_ids):
            continue
        rid = r["registry_id"].strip()
        ebk = (r.get("embeddings_backend_kind") or "").strip()
        ts = r.get("ts", "")
        if not ebk:
            continue
        prior = latest.get(rid)
        if prior is None or ts > prior[0]:
            latest[rid] = (ts, ebk)
    return {rid: v[1] for rid, v in latest.items()}


def _compute_per_instance_latest_platform(
    rows: list[dict[str, str]],
    internal_ids: set[str],
) -> dict[str, str]:
    """For each AWS customer instance, return the most-recent non-empty compute platform.

    Falls back to "unknown" for instances that never reported a platform.
    """
    latest: dict[str, tuple[str, str]] = {}
    for r in rows:
        if not _is_aws_customer(r, internal_ids):
            continue
        rid = r["registry_id"].strip()
        platform = (r.get("compute") or "").strip().lower()
        ts = r.get("ts", "")
        if not platform:
            continue
        prior = latest.get(rid)
        if prior is None or ts > prior[0]:
            latest[rid] = (ts, platform)
    return {rid: v[1] for rid, v in latest.items()}


def _compute_daily_spend(
    rows: list[dict[str, str]],
    internal_ids: set[str],
    exclude_day: str | None = None,
) -> tuple[list[dict[str, float | int | str]], dict[str, int]]:
    """Compute per-day spend rows and per-platform instance counts.

    For each day D (YYYY-MM-DD):
      aws_instances    = count of distinct AWS customer registry_ids that sent
                         any event on D
      bedrock_queries  = sum over those instances of delta(search_queries_total)
                         on D (only for instances whose latest
                         embeddings_backend_kind == "bedrock")
      compute_usd      = sum over active instances on D of
                         _daily_rate_for_platform(latest_platform[instance])
      bedrock_usd      = bedrock_queries * BEDROCK_COST_PER_QUERY
      total_usd        = compute_usd + bedrock_usd

    Platform is resolved via the instance's most-recent non-empty `compute`
    field (see _compute_per_instance_latest_platform). If an instance never
    reported a platform, it's billed at the "unknown"/docker fallback rate.

    Returns a (per_day_rows, per_platform_unique_instance_counts) tuple.
    """
    rows_sorted = sorted(rows, key=lambda r: (r.get("registry_id", ""), r.get("ts", "")))

    # Build per-day sets and per-instance-per-day max-seen search_queries_total
    by_day_instances: dict[str, set[str]] = defaultdict(set)
    by_instance_daily_max: dict[str, dict[str, int]] = defaultdict(dict)
    skipped_incomplete = 0
    for r in rows_sorted:
        if not _is_aws_customer(r, internal_ids):
            continue
        rid = r["registry_id"].strip()
        d = r.get("ts", "")[:10]
        if not d or len(d) < 10:
            continue
        if exclude_day and d == exclude_day:
            skipped_incomplete += 1
            continue
        by_day_instances[d].add(rid)
        sqt = _parse_int(r.get("search_queries_total"))
        prev_max = by_instance_daily_max[rid].get(d, 0)
        if sqt > prev_max:
            by_instance_daily_max[rid][d] = sqt
    if exclude_day:
        logger.info(
            f"Excluded incomplete day {exclude_day}: dropped {skipped_incomplete} AWS customer events"
        )

    # Per-instance latest compute platform -> daily rate
    latest_platform = _compute_per_instance_latest_platform(rows, internal_ids)

    # Count unique customer AWS instances per platform (for the per-platform summary)
    all_customer_ids: set[str] = set()
    for s in by_day_instances.values():
        all_customer_ids.update(s)
    platform_instance_counts: dict[str, int] = defaultdict(int)
    for rid in all_customer_ids:
        p = latest_platform.get(rid) or "unknown"
        platform_instance_counts[p] += 1

    # Per-instance first-observed day (needed for the "proven-persistence" model:
    # an instance is only charged on day D if it had events on D AND any prior day).
    # Equivalently: the instance's first-ever active day is free.
    instance_first_day: dict[str, str] = {}
    for d, ids in by_day_instances.items():
        for rid in ids:
            cur = instance_first_day.get(rid)
            if cur is None or d < cur:
                instance_first_day[rid] = d

    # Per-instance distinct-reporting-day count (needed for the "persisted" model:
    # an instance is only charged at all if it reported on >= 2 distinct days, i.e.
    # it came back on a separate day rather than installing and vanishing the same
    # day. Once it qualifies, EVERY day it reported is charged -- including its
    # first day. This is the "real running deployment" definition: excludes
    # install-and-delete instances, double-counts nothing.
    instance_distinct_days: dict[str, int] = defaultdict(int)
    for ids in by_day_instances.values():
        for rid in ids:
            instance_distinct_days[rid] += 1
    persisted_ids = {rid for rid, n in instance_distinct_days.items() if n >= 2}

    # Bedrock instances (latest-backend = bedrock)
    latest_backend = _compute_per_instance_latest_backend(rows, internal_ids)
    bedrock_instance_ids = {rid for rid, ebk in latest_backend.items() if ebk == "bedrock"}

    # Compute per-instance daily deltas (non-negative; counter resets => zero out)
    by_instance_daily_delta: dict[str, dict[str, int]] = {}
    for rid in bedrock_instance_ids:
        day_max = by_instance_daily_max.get(rid, {})
        sorted_days = sorted(day_max.keys())
        deltas: dict[str, int] = {}
        prev = 0
        for d in sorted_days:
            cur = day_max[d]
            delta = max(cur - prev, 0)
            deltas[d] = delta
            prev = cur
        by_instance_daily_delta[rid] = deltas

    # Roll up per day
    all_days = sorted(by_day_instances.keys())
    if not all_days:
        return [], dict(platform_instance_counts)

    out: list[dict[str, float | int | str]] = []
    cum = 0.0
    cum_persistent = 0.0
    cum_persisted = 0.0
    for d in _date_range(all_days[0], all_days[-1]):
        active = by_day_instances.get(d, set())
        n_inst = len(active)

        # "Proven-persistence" subset: instance was active on D AND had events
        # on any prior day. The instance's first-ever active day is excluded.
        active_persistent = {rid for rid in active if instance_first_day.get(rid) != d}
        n_inst_persistent = len(active_persistent)

        # "Persisted" subset: instance reported on >= 2 distinct days total, AND
        # was active on D. Unlike "proven", the instance's first day IS charged
        # once it qualifies. This is the "real running deployment, count every
        # day it phoned home" definition (excludes install-and-delete instances).
        active_persisted = active & persisted_ids
        n_inst_persisted = len(active_persisted)

        # Per-platform breakdown for this day (permissive / all-days model)
        platform_counts: dict[str, int] = defaultdict(int)
        platform_usd: dict[str, float] = defaultdict(float)
        for rid in active:
            p = latest_platform.get(rid) or "unknown"
            rate = _daily_rate_for_platform(p)
            platform_counts[p] += 1
            platform_usd[p] += rate

        # Per-platform breakdown for the proven subset
        platform_counts_p: dict[str, int] = defaultdict(int)
        platform_usd_p: dict[str, float] = defaultdict(float)
        for rid in active_persistent:
            p = latest_platform.get(rid) or "unknown"
            rate = _daily_rate_for_platform(p)
            platform_counts_p[p] += 1
            platform_usd_p[p] += rate

        # Per-platform breakdown for the persisted subset
        platform_counts_d: dict[str, int] = defaultdict(int)
        platform_usd_d: dict[str, float] = defaultdict(float)
        for rid in active_persisted:
            p = latest_platform.get(rid) or "unknown"
            rate = _daily_rate_for_platform(p)
            platform_counts_d[p] += 1
            platform_usd_d[p] += rate

        compute_usd = sum(platform_usd.values())
        compute_usd_persistent = sum(platform_usd_p.values())
        compute_usd_persisted = sum(platform_usd_d.values())

        queries = 0
        queries_persistent = 0
        queries_persisted = 0
        for rid in bedrock_instance_ids:
            if rid in active:
                delta = by_instance_daily_delta.get(rid, {}).get(d, 0)
                queries += delta
                if rid in active_persistent:
                    queries_persistent += delta
                if rid in active_persisted:
                    queries_persisted += delta
        bedrock_usd = queries * BEDROCK_COST_PER_QUERY
        bedrock_usd_persistent = queries_persistent * BEDROCK_COST_PER_QUERY
        bedrock_usd_persisted = queries_persisted * BEDROCK_COST_PER_QUERY

        total_usd = compute_usd + bedrock_usd
        total_usd_persistent = compute_usd_persistent + bedrock_usd_persistent
        total_usd_persisted = compute_usd_persisted + bedrock_usd_persisted
        cum += total_usd
        cum_persistent += total_usd_persistent
        cum_persisted += total_usd_persisted

        out.append(
            {
                "date": d,
                "aws_instances": n_inst,
                "aws_instances_persistent": n_inst_persistent,
                "aws_instances_persisted": n_inst_persisted,
                "docker_instances": platform_counts.get("docker", 0),
                "ecs_instances": platform_counts.get("ecs", 0),
                "kubernetes_instances": platform_counts.get("kubernetes", 0),
                "other_platform_instances": sum(
                    v for k, v in platform_counts.items()
                    if k not in ("docker", "ecs", "kubernetes")
                ),
                "docker_instances_persistent": platform_counts_p.get("docker", 0),
                "ecs_instances_persistent": platform_counts_p.get("ecs", 0),
                "kubernetes_instances_persistent": platform_counts_p.get("kubernetes", 0),
                "other_platform_instances_persistent": sum(
                    v for k, v in platform_counts_p.items()
                    if k not in ("docker", "ecs", "kubernetes")
                ),
                "docker_instances_persisted": platform_counts_d.get("docker", 0),
                "ecs_instances_persisted": platform_counts_d.get("ecs", 0),
                "kubernetes_instances_persisted": platform_counts_d.get("kubernetes", 0),
                "other_platform_instances_persisted": sum(
                    v for k, v in platform_counts_d.items()
                    if k not in ("docker", "ecs", "kubernetes")
                ),
                "bedrock_queries": queries,
                "bedrock_queries_persistent": queries_persistent,
                "bedrock_queries_persisted": queries_persisted,
                "compute_usd": round(compute_usd, 4),
                "compute_usd_persistent": round(compute_usd_persistent, 4),
                "compute_usd_persisted": round(compute_usd_persisted, 4),
                "bedrock_usd": round(bedrock_usd, 6),
                "bedrock_usd_persistent": round(bedrock_usd_persistent, 6),
                "bedrock_usd_persisted": round(bedrock_usd_persisted, 6),
                "total_usd": round(total_usd, 4),
                "total_usd_persistent": round(total_usd_persistent, 4),
                "total_usd_persisted": round(total_usd_persisted, 4),
                "cum_total_usd": round(cum, 4),
                "cum_total_usd_persistent": round(cum_persistent, 4),
                "cum_total_usd_persisted": round(cum_persisted, 4),
            }
        )
    return out, dict(platform_instance_counts)


def _date_range(
    start: str,
    end: str,
) -> list[str]:
    """Return contiguous YYYY-MM-DD strings from start..end inclusive."""
    s = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()
    out: list[str] = []
    d = s
    while d <= e:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _write_csv_sidecar(
    daily: list[dict[str, float | int | str]],
    path: str,
) -> None:
    """Write per-day spend rows to a CSV sidecar."""
    if not daily:
        return
    fieldnames = list(daily[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in daily:
            w.writerow(row)
    logger.info(f"CSV sidecar written to {path}")


def _write_summary_json(
    daily: list[dict[str, float | int | str]],
    platform_instance_counts: dict[str, int],
    path: str,
) -> None:
    """Write a JSON summary with headline numbers for the report narrative."""
    if not daily:
        return
    yesterday = daily[-1]
    ltv = yesterday["cum_total_usd"]
    ltv_persistent = yesterday["cum_total_usd_persistent"]
    ltv_persisted = yesterday["cum_total_usd_persisted"]
    seven = daily[-7:]
    ltv_7d = round(sum(r["total_usd"] for r in seven), 2)
    ltv_7d_persistent = round(sum(r["total_usd_persistent"] for r in seven), 2)
    ltv_7d_persisted = round(sum(r["total_usd_persisted"] for r in seven), 2)
    ltv_compute = round(sum(r["compute_usd"] for r in daily), 2)
    ltv_bedrock = round(sum(r["bedrock_usd"] for r in daily), 2)
    ltv_compute_persistent = round(sum(r["compute_usd_persistent"] for r in daily), 2)
    ltv_bedrock_persistent = round(sum(r["bedrock_usd_persistent"] for r in daily), 2)
    ltv_compute_persisted = round(sum(r["compute_usd_persisted"] for r in daily), 2)
    ltv_bedrock_persisted = round(sum(r["bedrock_usd_persisted"] for r in daily), 2)

    # Platform-level instance-day totals and compute-USD totals (LTV) -- permissive
    platform_instance_days: dict[str, int] = defaultdict(int)
    platform_compute_usd: dict[str, float] = defaultdict(float)
    for row in daily:
        for p in ("docker", "ecs", "kubernetes"):
            n = int(row.get(f"{p}_instances", 0))
            platform_instance_days[p] += n
            platform_compute_usd[p] += n * _daily_rate_for_platform(p)
        other_n = int(row.get("other_platform_instances", 0))
        platform_instance_days["other"] += other_n
        platform_compute_usd["other"] += other_n * _daily_rate_for_platform("unknown")

    # Same breakdown for the persistent model
    platform_instance_days_p: dict[str, int] = defaultdict(int)
    platform_compute_usd_p: dict[str, float] = defaultdict(float)
    for row in daily:
        for p in ("docker", "ecs", "kubernetes"):
            n = int(row.get(f"{p}_instances_persistent", 0))
            platform_instance_days_p[p] += n
            platform_compute_usd_p[p] += n * _daily_rate_for_platform(p)
        other_n = int(row.get("other_platform_instances_persistent", 0))
        platform_instance_days_p["other"] += other_n
        platform_compute_usd_p["other"] += other_n * _daily_rate_for_platform("unknown")

    # Same breakdown for the persisted model (>= 2 distinct days, all reported days charged)
    platform_instance_days_d: dict[str, int] = defaultdict(int)
    platform_compute_usd_d: dict[str, float] = defaultdict(float)
    for row in daily:
        for p in ("docker", "ecs", "kubernetes"):
            n = int(row.get(f"{p}_instances_persisted", 0))
            platform_instance_days_d[p] += n
            platform_compute_usd_d[p] += n * _daily_rate_for_platform(p)
        other_n = int(row.get("other_platform_instances_persisted", 0))
        platform_instance_days_d["other"] += other_n
        platform_compute_usd_d["other"] += other_n * _daily_rate_for_platform("unknown")

    summary = {
        "cost_model": {
            "per_platform_daily_rate_usd": dict(COMPUTE_PLATFORM_DAILY_RATE_USD),
            "docker_breakdown": "1 x t3.xlarge on-demand ($0.1664/hr)",
            "ecs_breakdown": (
                "Grounded in a measured Cost Explorer day (May-30) for the "
                "terraform/aws-ecs deployment, excl shared-account overhead "
                "(CloudTrail, Others) and the standalone EC2-Instances box: "
                "Fargate/ECS ($7.66) + EC2-Other NAT/EBS/data ($7.39) "
                "+ RDS Keycloak ($4.47) + DocumentDB ($2.03) + VPC ($1.80) "
                "+ CloudWatch ($1.61) + ELB/ALBs ($1.08) "
                "= $26.04/day"
            ),
            "kubernetes_breakdown": (
                "Grounded in the measured EKS reference deployment: "
                "EKS control plane ($2.40) + 3 x m6i.xlarge nodes ($13.82) "
                "+ EBS gp3 ~41Gi ($0.11) + 1 ALB ($0.92) "
                "+ 1 NAT Gateway + 5 GB egress ($1.31) + ACM public cert ($0.00) "
                "+ Route 53 hosted zone ($0.02) = $18.58/day"
            ),
            "bedrock_model": BEDROCK_MODEL,
            "bedrock_price_per_1k_tokens_usd": BEDROCK_PRICE_PER_1K_TOKENS,
            "tokens_per_query": TOKENS_PER_QUERY,
            "bedrock_cost_per_query_usd": BEDROCK_COST_PER_QUERY,
            "filters": "customer-only (internal UUIDs excluded), AWS-only (cloud=aws)",
        },
        "counting_rule": {
            "permissive": (
                "Charge every distinct (AWS customer instance, day) pair -- "
                "including 1-day trial installs. Headline numbers labeled "
                "'all-days'."
            ),
            "proven_persistence": (
                "Charge an instance on day D only if it had events on D AND "
                "any prior day. Conservative filter that excludes every "
                "instance's first-ever active day. Headline numbers labeled "
                "'proven'. ~59% of the current fleet never sends a second "
                "day of events (one-day wonders) -- they contribute $0 under "
                "this model."
            ),
            "persisted": (
                "Charge an instance on EVERY day it reported, but only if it "
                "reported on >= 2 distinct days total (i.e. it came back on a "
                "separate day rather than installing and vanishing the same "
                "day). Unlike 'proven', the instance's first day IS charged "
                "once it qualifies. This is the 'real running deployment' "
                "definition: excludes install-and-delete instances, "
                "double-counts nothing. Headline numbers labeled 'persisted'."
            ),
        },
        "yesterday": {
            "date": yesterday["date"],
            "all_days": {
                "aws_instances": yesterday["aws_instances"],
                "docker_instances": yesterday.get("docker_instances", 0),
                "ecs_instances": yesterday.get("ecs_instances", 0),
                "kubernetes_instances": yesterday.get("kubernetes_instances", 0),
                "other_platform_instances": yesterday.get("other_platform_instances", 0),
                "bedrock_queries": yesterday["bedrock_queries"],
                "compute_usd": yesterday["compute_usd"],
                "bedrock_usd": yesterday["bedrock_usd"],
                "total_usd": yesterday["total_usd"],
            },
            "proven": {
                "aws_instances": yesterday["aws_instances_persistent"],
                "docker_instances": yesterday.get("docker_instances_persistent", 0),
                "ecs_instances": yesterday.get("ecs_instances_persistent", 0),
                "kubernetes_instances": yesterday.get("kubernetes_instances_persistent", 0),
                "other_platform_instances": yesterday.get("other_platform_instances_persistent", 0),
                "bedrock_queries": yesterday["bedrock_queries_persistent"],
                "compute_usd": yesterday["compute_usd_persistent"],
                "bedrock_usd": yesterday["bedrock_usd_persistent"],
                "total_usd": yesterday["total_usd_persistent"],
            },
            "persisted": {
                "aws_instances": yesterday["aws_instances_persisted"],
                "docker_instances": yesterday.get("docker_instances_persisted", 0),
                "ecs_instances": yesterday.get("ecs_instances_persisted", 0),
                "kubernetes_instances": yesterday.get("kubernetes_instances_persisted", 0),
                "other_platform_instances": yesterday.get("other_platform_instances_persisted", 0),
                "bedrock_queries": yesterday["bedrock_queries_persisted"],
                "compute_usd": yesterday["compute_usd_persisted"],
                "bedrock_usd": yesterday["bedrock_usd_persisted"],
                "total_usd": yesterday["total_usd_persisted"],
            },
        },
        "per_platform_unique_instance_totals": platform_instance_counts,
        "per_platform_ltv_breakdown_all_days": {
            p: {
                "instance_days": platform_instance_days[p],
                "compute_usd": round(platform_compute_usd[p], 2),
            }
            for p in ("docker", "ecs", "kubernetes", "other")
        },
        "per_platform_ltv_breakdown_proven": {
            p: {
                "instance_days": platform_instance_days_p[p],
                "compute_usd": round(platform_compute_usd_p[p], 2),
            }
            for p in ("docker", "ecs", "kubernetes", "other")
        },
        "per_platform_ltv_breakdown_persisted": {
            p: {
                "instance_days": platform_instance_days_d[p],
                "compute_usd": round(platform_compute_usd_d[p], 2),
            }
            for p in ("docker", "ecs", "kubernetes", "other")
        },
        "last_7_days": {
            "all_days_total_usd": ltv_7d,
            "proven_total_usd": ltv_7d_persistent,
            "persisted_total_usd": ltv_7d_persisted,
        },
        "ltv": {
            "all_days": {
                "compute_usd": ltv_compute,
                "bedrock_usd": ltv_bedrock,
                "total_usd": ltv,
                "total_instance_days": sum(r["aws_instances"] for r in daily),
            },
            "proven": {
                "compute_usd": ltv_compute_persistent,
                "bedrock_usd": ltv_bedrock_persistent,
                "total_usd": ltv_persistent,
                "total_instance_days": sum(r["aws_instances_persistent"] for r in daily),
            },
            "persisted": {
                "compute_usd": ltv_compute_persisted,
                "bedrock_usd": ltv_bedrock_persisted,
                "total_usd": ltv_persisted,
                "total_instance_days": sum(r["aws_instances_persisted"] for r in daily),
            },
            "first_day": daily[0]["date"],
            "last_day": yesterday["date"],
        },
    }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary JSON written to {path}")


def _generate_chart(
    daily: list[dict[str, float | int | str]],
    output_path: str,
) -> None:
    """Render a three-panel chart: daily compute $, daily Bedrock $, cumulative $."""
    apply_tufte_style()

    fig, (ax_compute, ax_bedrock, ax_cum) = plt.subplots(
        3,
        1,
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
        sharex=True,
    )
    fig.suptitle(CHART_TITLE, fontsize=13, fontweight="bold", y=0.995)

    dates = [datetime.strptime(r["date"], "%Y-%m-%d") for r in daily]
    compute = [r["compute_usd"] for r in daily]
    compute_p = [r["compute_usd_persistent"] for r in daily]
    compute_d = [r["compute_usd_persisted"] for r in daily]
    bedrock = [r["bedrock_usd"] for r in daily]
    cum = [r["cum_total_usd"] for r in daily]
    cum_p = [r["cum_total_usd_persistent"] for r in daily]
    cum_d = [r["cum_total_usd_persisted"] for r in daily]
    colors = sns.color_palette("Set2", 5)

    # Daily compute: show all-days as the faint bar height, overlay the persisted
    # subset (the headline rule: >= 2 distinct reporting days) as the solid bar.
    ax_compute.bar(dates, compute, color=colors[0], alpha=0.4, label="all-days (incl. install-and-vanish)")
    ax_compute.bar(dates, compute_d, color=colors[0], alpha=0.95, label="persisted (>= 2 distinct reporting days)")
    ax_compute.set_title(
        "Daily EC2 compute cost -- per-platform rate * active AWS customer instances",
        fontsize=10,
    )
    ax_compute.set_ylabel("USD / day")
    ax_compute.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax_compute.legend(loc="upper left", fontsize=8)

    ax_bedrock.bar(dates, bedrock, color=colors[1], alpha=0.8)
    ax_bedrock.set_title(
        "Daily Bedrock Titan embeddings cost -- search queries * 100 tok * $0.00002/1K",
        fontsize=10,
    )
    ax_bedrock.set_ylabel("USD / day")
    ax_bedrock.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))

    ax_cum.plot(
        dates, cum, linewidth=2.5, color=colors[2], marker="o", markersize=3,
        label="all-days (upper bound)",
    )
    ax_cum.plot(
        dates, cum_d, linewidth=2.5, color=colors[4], marker="D", markersize=3,
        label="persisted (headline: >= 2 distinct days)",
    )
    ax_cum.plot(
        dates, cum_p, linewidth=1.8, color=colors[3], marker="s", markersize=3,
        linestyle="--", label="proven (first day free, lower bound)",
    )
    ax_cum.fill_between(dates, cum_p, cum, color=colors[2], alpha=0.12)
    ax_cum.set_title("Cumulative LTV spend (compute + Bedrock) -- range across counting rules", fontsize=10)
    ax_cum.set_ylabel("USD total")
    ax_cum.legend(loc="upper left", fontsize=8)
    ax_cum.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))

    ax_cum.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax_cum.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates) // 14)))
    plt.setp(ax_cum.xaxis.get_majorticklabels(), rotation=45, ha="right")

    for _ax in fig.axes:
        tufte_axes(_ax)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Chart saved to {output_path}")


def main() -> None:
    """Parse arguments and compute LTV spend artifacts."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute AWS-only customer infra + Bedrock-embeddings spend for the "
            "AI Registry. Produces a chart, a CSV sidecar of per-day values, "
            "and a JSON summary with headline numbers."
        ),
    )
    parser.add_argument(
        "--csv-dir",
        required=True,
        help="Directory containing CSV files (scans subdirectories too)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the output PNG chart",
    )
    parser.add_argument(
        "--internal-instances",
        default=None,
        help="Path to known-internal-instances.md. Internal IDs are excluded from spend.",
    )
    parser.add_argument(
        "--csv-out",
        default=None,
        help="Optional path to write per-day spend CSV",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path to write JSON summary with headline numbers",
    )
    parser.add_argument(
        "--exclude-incomplete-day",
        default=None,
        help=(
            "Optional YYYY-MM-DD. Events on this date are dropped from the chart "
            "and headline numbers so a still-in-progress day doesn't show as a dip."
        ),
    )
    args = parser.parse_args()

    if not os.path.isdir(args.csv_dir):
        logger.error(f"Directory not found: {args.csv_dir}")
        raise SystemExit(1)

    csv_files = _find_csv_files(args.csv_dir)
    if not csv_files:
        logger.error(f"No CSV files found in {args.csv_dir}")
        raise SystemExit(1)

    internal_ids = _load_internal_instance_ids(args.internal_instances)

    all_rows = _read_all_csvs(csv_files)
    unique_rows = _dedupe_by_id_ts(all_rows)
    daily, platform_instance_counts = _compute_daily_spend(
        unique_rows,
        internal_ids,
        args.exclude_incomplete_day,
    )
    if not daily:
        logger.error("No AWS customer events found after filtering")
        raise SystemExit(1)

    _generate_chart(daily, args.output)

    if args.csv_out:
        _write_csv_sidecar(daily, args.csv_out)
    if args.summary_json:
        _write_summary_json(daily, platform_instance_counts, args.summary_json)


if __name__ == "__main__":
    main()
