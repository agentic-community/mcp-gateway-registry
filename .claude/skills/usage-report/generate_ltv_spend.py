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

     ecs        -> $19.03/day : grounded in terraform/aws-ecs/terraform.tfstate.
                   Itemized:
                     10 Fargate tasks (5 @ 1vCPU/2GB + 5 @ 0.5vCPU/1GB)
                       = $7.67/day
                       (vCPU $0.04048/hr + GB $0.004445/hr, 24h, us-east-1)
                     DocumentDB db.t3.medium  ($0.078/hr, 1 instance)
                       = $1.87/day
                     RDS Aurora Serverless v2 (Keycloak, avg 1 ACU @ $0.12/hr)
                       = $2.88/day
                     2 Application Load Balancers ($0.0225/hr + LCUs)
                       = $1.35/day
                     3 NAT Gateways ($0.045/hr each, excludes data processing)
                       = $3.24/day
                     2 CloudFront distributions (PriceClass_100, low traffic)
                       = $0.50/day
                     S3 (ALB + CloudFront logs)
                       = $0.05/day
                     CloudWatch log groups + metric alarms (14 + 12)
                       = $1.00/day
                     EFS + Secrets Manager + data transfer overhead
                       = $0.50/day
                   See terraform/aws-ecs/modules/mcp-gateway/*.tf for
                   resource definitions, cpu/memory defaults, and the
                   ALB / NAT / CloudFront wiring.

     kubernetes -> $11.17/day : grounded in charts/registry/values.yaml
                   and the stack chart's 4 ALB-backed ingresses.
                   Itemized:
                     EKS control plane ($0.10/hr)
                       = $2.40/day
                     2 x t3.large worker nodes ($0.0832/hr each)
                       = $3.99/day
                     4 Application Load Balancers (keycloak, registry,
                       mcpgw, stack-level) @ $0.0225/hr each + LCUs
                       = $2.70/day
                     1 NAT Gateway for private subnets ($0.045/hr)
                       = $1.08/day
                     EBS volumes (node storage + mongodb PV)
                       = $0.50/day
                     CloudWatch Container Insights
                       = $0.30/day
                     Data transfer overhead
                       = $0.20/day
                   Pod resource requests from chart defaults:
                     registry      1 vCPU / 1 GiB
                     auth-server   1 vCPU / 1 GiB
                     mcpgw       0.5 vCPU / 1 GiB
                     keycloak    ~1 vCPU / 1 GiB (in-cluster)
                     mongodb    ~0.5 vCPU / 2 GiB (in-cluster)
                   Total ~4 vCPU / 6 GiB fits comfortably on 2 x t3.large.

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
COMPUTE_PLATFORM_DAILY_RATE_USD: dict[str, float] = {
    "docker": 3.99,
    "ecs": 19.03,
    "kubernetes": 11.17,
    "ec2": 3.99,
    "unknown": 3.99,
    "vm": 3.99,
    "": 3.99,
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
    "(per-platform: docker $3.99 / ecs $19.03 / k8s $11.17 per day + Bedrock Titan)"
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
    for r in rows_sorted:
        if not _is_aws_customer(r, internal_ids):
            continue
        rid = r["registry_id"].strip()
        d = r.get("ts", "")[:10]
        if not d or len(d) < 10:
            continue
        by_day_instances[d].add(rid)
        sqt = _parse_int(r.get("search_queries_total"))
        prev_max = by_instance_daily_max[rid].get(d, 0)
        if sqt > prev_max:
            by_instance_daily_max[rid][d] = sqt

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
    for d in _date_range(all_days[0], all_days[-1]):
        active = by_day_instances.get(d, set())
        n_inst = len(active)

        # "Proven-persistence" subset: instance was active on D AND had events
        # on any prior day. The instance's first-ever active day is excluded.
        active_persistent = {rid for rid in active if instance_first_day.get(rid) != d}
        n_inst_persistent = len(active_persistent)

        # Per-platform breakdown for this day (permissive / all-days model)
        platform_counts: dict[str, int] = defaultdict(int)
        platform_usd: dict[str, float] = defaultdict(float)
        for rid in active:
            p = latest_platform.get(rid) or "unknown"
            rate = _daily_rate_for_platform(p)
            platform_counts[p] += 1
            platform_usd[p] += rate

        # Per-platform breakdown for the persistent subset
        platform_counts_p: dict[str, int] = defaultdict(int)
        platform_usd_p: dict[str, float] = defaultdict(float)
        for rid in active_persistent:
            p = latest_platform.get(rid) or "unknown"
            rate = _daily_rate_for_platform(p)
            platform_counts_p[p] += 1
            platform_usd_p[p] += rate

        compute_usd = sum(platform_usd.values())
        compute_usd_persistent = sum(platform_usd_p.values())

        queries = 0
        queries_persistent = 0
        for rid in bedrock_instance_ids:
            if rid in active:
                delta = by_instance_daily_delta.get(rid, {}).get(d, 0)
                queries += delta
                if rid in active_persistent:
                    queries_persistent += delta
        bedrock_usd = queries * BEDROCK_COST_PER_QUERY
        bedrock_usd_persistent = queries_persistent * BEDROCK_COST_PER_QUERY

        total_usd = compute_usd + bedrock_usd
        total_usd_persistent = compute_usd_persistent + bedrock_usd_persistent
        cum += total_usd
        cum_persistent += total_usd_persistent

        out.append(
            {
                "date": d,
                "aws_instances": n_inst,
                "aws_instances_persistent": n_inst_persistent,
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
                "bedrock_queries": queries,
                "bedrock_queries_persistent": queries_persistent,
                "compute_usd": round(compute_usd, 4),
                "compute_usd_persistent": round(compute_usd_persistent, 4),
                "bedrock_usd": round(bedrock_usd, 6),
                "bedrock_usd_persistent": round(bedrock_usd_persistent, 6),
                "total_usd": round(total_usd, 4),
                "total_usd_persistent": round(total_usd_persistent, 4),
                "cum_total_usd": round(cum, 4),
                "cum_total_usd_persistent": round(cum_persistent, 4),
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
    seven = daily[-7:]
    ltv_7d = round(sum(r["total_usd"] for r in seven), 2)
    ltv_7d_persistent = round(sum(r["total_usd_persistent"] for r in seven), 2)
    ltv_compute = round(sum(r["compute_usd"] for r in daily), 2)
    ltv_bedrock = round(sum(r["bedrock_usd"] for r in daily), 2)
    ltv_compute_persistent = round(sum(r["compute_usd_persistent"] for r in daily), 2)
    ltv_bedrock_persistent = round(sum(r["bedrock_usd_persistent"] for r in daily), 2)

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

    summary = {
        "cost_model": {
            "per_platform_daily_rate_usd": dict(COMPUTE_PLATFORM_DAILY_RATE_USD),
            "docker_breakdown": "1 x t3.xlarge on-demand ($0.1664/hr)",
            "ecs_breakdown": (
                "Grounded in terraform/aws-ecs/terraform.tfstate: "
                "10 Fargate tasks ($7.67) + DocumentDB db.t3.medium ($1.87) "
                "+ RDS Aurora Serverless v2 avg 1 ACU ($2.88) + 2 ALBs ($1.35) "
                "+ 3 NAT Gateways ($3.24) + 2 CloudFront distributions ($0.50) "
                "+ S3 logs ($0.05) + CloudWatch ($1.00) + EFS/SM/DT ($0.50) "
                "= $19.03/day"
            ),
            "kubernetes_breakdown": (
                "Grounded in charts/ Helm defaults + aws-load-balancer-controller: "
                "EKS control plane ($2.40) + 2 x t3.large nodes ($3.99) "
                "+ 4 ALB ingresses ($2.70) + 1 NAT Gateway ($1.08) "
                "+ EBS ($0.50) + CloudWatch Container Insights ($0.30) "
                "+ Data transfer ($0.20) = $11.17/day"
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
        "last_7_days": {
            "all_days_total_usd": ltv_7d,
            "proven_total_usd": ltv_7d_persistent,
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
    sns.set_theme(style="whitegrid")

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
    bedrock = [r["bedrock_usd"] for r in daily]
    cum = [r["cum_total_usd"] for r in daily]
    cum_p = [r["cum_total_usd_persistent"] for r in daily]
    colors = sns.color_palette("Set2", 4)

    # Daily compute: show permissive as the bar height and overlay the proven
    # subset so the "first-day wedge" is visible at a glance.
    ax_compute.bar(dates, compute, color=colors[0], alpha=0.5, label="all-days (incl. first-day installs)")
    ax_compute.bar(dates, compute_p, color=colors[0], alpha=0.95, label="proven (seen on any prior day)")
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
        dates, cum_p, linewidth=2.5, color=colors[3], marker="s", markersize=3,
        label="proven (lower bound)",
    )
    ax_cum.fill_between(dates, cum_p, cum, color=colors[2], alpha=0.15)
    ax_cum.set_title("Cumulative LTV spend (compute + Bedrock) -- range between two counting rules", fontsize=10)
    ax_cum.set_ylabel("USD total")
    ax_cum.legend(loc="upper left", fontsize=8)
    ax_cum.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))

    ax_cum.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax_cum.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates) // 14)))
    plt.setp(ax_cum.xaxis.get_majorticklabels(), rotation=45, ha="right")

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
    daily, platform_instance_counts = _compute_daily_spend(unique_rows, internal_ids)
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
