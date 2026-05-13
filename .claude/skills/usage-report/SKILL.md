---
name: usage-report
description: Generate a usage report for MCP Gateway Registry by SSHing into the telemetry bastion host, exporting telemetry data from DocumentDB, and producing a formatted markdown report with deployment insights.
license: Apache-2.0
metadata:
  author: mcp-gateway-registry
  version: "1.1"
---

# Usage Report Skill

Export telemetry data from the MCP Gateway Registry's DocumentDB telemetry collector and generate a usage report showing deployment patterns, version adoption, and feature usage in the wild.

## Prerequisites

1. **SSH key** at `~/.ssh/id_ed25519` with access to the bastion host
2. **Terraform state** available in `terraform/telemetry-collector/` (to read bastion IP)
3. **Bastion host enabled** (`bastion_enabled = true` in `terraform/telemetry-collector/terraform.tfvars`)
4. **AWS credentials** configured on the bastion host (for Secrets Manager access)
5. **GitHub CLI (`gh`)** authenticated with read access to the upstream repo (`agentic-community/mcp-gateway-registry`) for collecting stars, forks, and contributor counts

## Input

The skill accepts optional parameters:

```
/usage-report [OUTPUT_DIR]
```

- **OUTPUT_DIR** - Base directory for reports (default: `.scratchpad/usage-reports/`)

If OUTPUT_DIR is not provided, save to `.scratchpad/usage-reports/`.

All artifacts for a given run are placed in a **dated subfolder**: `OUTPUT_DIR/YYYY-MM-DD/`. This keeps each report self-contained and avoids a flat directory of hundreds of files. Previous metrics and CSV files are discovered by scanning both the base directory and all dated subdirectories.

## Workflow

### Step 1: Get Bastion IP

```bash
cd terraform/telemetry-collector && terraform output -raw bastion_public_ip
```

If the output is "Bastion not enabled", tell the user to set `bastion_enabled = true` in `terraform/telemetry-collector/terraform.tfvars` and run `terraform apply`.

### Step 2: Copy Export Script to Bastion

```bash
scp -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519 \
  terraform/telemetry-collector/bastion-scripts/telemetry_db.py \
  ec2-user@$BASTION_IP:~/telemetry_db.py
```

### Step 3: Run Export on Bastion

```bash
ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519 \
  ec2-user@$BASTION_IP \
  'python3 telemetry_db.py export --output /tmp/registry_metrics.csv 2>&1'
```

Capture the full output -- it contains the summary statistics printed by `telemetry_db.py`.

### Step 4: Create Dated Subfolder and Download the CSV

Create a dated subfolder for this run's artifacts, then download the CSV into it:

```bash
DATE_DIR=OUTPUT_DIR/YYYY-MM-DD
mkdir -p $DATE_DIR

scp -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519 \
  ec2-user@$BASTION_IP:/tmp/registry_metrics.csv \
  $DATE_DIR/registry_metrics.csv
```

### Step 5: Install Python Dependencies and Generate Charts

First, ensure matplotlib and seaborn are available on the system Python:

```bash
/usr/bin/python3 -c "import matplotlib, seaborn" 2>/dev/null || pip install --break-system-packages matplotlib seaborn
```

Then generate the **instance-based** deployment distribution chart (counts unique registry instances, not events):

```bash
/usr/bin/python3 .claude/skills/usage-report/generate_instance_distribution_chart.py \
  --csv $DATE_DIR/registry_metrics.csv \
  --output $DATE_DIR/instance-distribution-YYYY-MM-DD.png
```

This produces a single faceted PNG with 6 subplots: Cloud Provider, Compute Platform, Storage Backend, Auth Provider, Architecture, and Deployment Mode. Each subplot shows unique instance counts and percentages.

### Step 5b: Generate Timeseries Chart

Generate a timeseries chart showing unique registry installs per cloud provider over time. This reads ALL CSV files in the base output directory and dated subdirectories to build a complete historical view:

```bash
/usr/bin/python3 .claude/skills/usage-report/generate_timeseries_chart.py \
  --csv-dir OUTPUT_DIR \
  --output $DATE_DIR/registry-installs-timeseries-YYYY-MM-DD.png
```

This produces a PNG with two subplots:
- **Cumulative Unique Registry Installs** -- running total of unique registry_ids per cloud provider
- **Daily Active Registry Installs** -- unique registry_ids seen each day per cloud provider

### Step 5b2: Generate Compute Platform Timeseries Chart

Generate a second timeseries chart, parallel to the cloud-provider one, showing unique registry installs per **compute platform** (docker, kubernetes, ecs, ec2, etc.) over time. Same data-sourcing behavior (scans all CSV files across dated subdirectories). Pass `--snapshots-table` to also emit a markdown per-snapshot table ready to embed in the report:

```bash
/usr/bin/python3 .claude/skills/usage-report/generate_compute_timeseries_chart.py \
  --csv-dir OUTPUT_DIR \
  --output $DATE_DIR/compute-installs-timeseries-YYYY-MM-DD.png \
  --snapshots-table $DATE_DIR/compute-platform-snapshots-YYYY-MM-DD.md
```

This produces:
- A PNG with two subplots:
  - **Cumulative Unique Registry Installs per Compute Platform** -- running total of unique registry_ids per platform
  - **Daily Active Registry Installs per Compute Platform** -- unique registry_ids seen each day per platform
- A markdown file with the **Per-Platform Growth (Unique Installs)** table, one row per dated CSV snapshot, sorted **descending by date** (newest first, bolded). The column order is `docker | kubernetes | ecs | ec2 | unknown` when present, plus any other platforms alphabetically. Unique-instance counts per snapshot are computed directly from each dated CSV using the `compute` column (not `compute_platform` -- that's the schema key but not the CSV column name).

Embed the chart in the report's "Compute Platform Growth" section and drop the contents of the snapshots-table markdown file in under the "Per-Platform Growth (Unique Installs)" subheading. Add a short narrative on which platforms are growing fastest in absolute and percentage terms; the newest (bolded) row is the current total for the report.

### Step 5c: Generate Instance Lifetime Chart

Generate a density plot showing the distribution of instance lifetimes (age in days). This reads the metrics JSON produced by the analysis step, so it must run after Step 6. However, the SKILL.md lists it here for logical grouping with other charts:

```bash
/usr/bin/python3 .claude/skills/usage-report/generate_lifetime_chart.py \
  --metrics $DATE_DIR/metrics-YYYY-MM-DD.json \
  --output $DATE_DIR/instance-lifetime-YYYY-MM-DD.png
```

This produces a PNG with two panels:
- **Age Distribution** -- histogram with KDE density overlay showing instance ages in days, with stats annotation (mean, max, multi-day vs single-day counts)
- **Age Buckets** -- horizontal bar chart grouping instances into age ranges (0 days, 1-2 days, 3-5 days, etc.) with counts and percentages

**Note**: Run this after Step 6 (telemetry analysis) since it reads the metrics JSON.

### Step 5c2: Generate Customer-Active-Instances Chart

Generate a chart of customer engagement over time, with three overlaid series:

- **Daily Active Instances (DAI)** -- unique `registry_id`s that sent at least one event (startup OR heartbeat) on that day.
- **7-day moving average (MA7)** -- trailing 7-day average of DAI.
- **7-day consistency streak (S7)** -- unique `registry_id`s that sent at least one event on EACH of the 7 days in the window `[D-6..D]`.

Customer-only: internal instances loaded from `known-internal-instances.md` are excluded so the numbers align with the Liveness section (which is also customer-only). A CSV sidecar of the per-day values is written alongside the PNG so the report narrative can quote exact numbers and future reports can diff against it.

```bash
/usr/bin/python3 .claude/skills/usage-report/generate_active_instances_chart.py \
  --csv-dir OUTPUT_DIR \
  --output $DATE_DIR/active-instances-YYYY-MM-DD.png \
  --internal-instances .claude/skills/usage-report/known-internal-instances.md \
  --csv-out $DATE_DIR/active-instances-YYYY-MM-DD.csv
```

Same data-sourcing behavior as the other historical charts (scans all CSVs across dated subdirectories). Embed in the Liveness section under an "Engagement over Time" subheading.

### Step 5c3: Generate LTV Infra-Spend Chart and Summary

Compute daily and cumulative AWS customer infra spend (EC2 compute + Bedrock Titan embeddings). The script emits **two cost numbers framed as a range** so the report can be conservative about not over-estimating spend:

- **All-days (upper bound):** charges every distinct (AWS customer instance, day) pair, including 1-day trial installs. Matches "every real AWS billing day this solution caused".
- **Proven-persistence (lower bound):** charges an instance on day D only if it had events on D AND on any prior day. This excludes every instance's first-ever active day, so instances that phone home once and never again contribute $0. Gap-tolerant: if an instance is silent on day D-1 and comes back on D, it's still charged on D (because it had prior events on earlier days).

Customer-only (internal UUIDs excluded), AWS-only (GCP/Azure/unknown excluded because we can't attribute their AWS-side usage). On the current fleet ~59% of AWS customer instances are "one-day wonders" — they show up once and never return — so the proven-persistence number is typically ~30% lower than all-days.

**Cost model (per-compute-platform, grounded in deployment artefacts):**

| Platform | Daily rate | Grounding |
|---|---:|---|
| `docker` | $3.99 | 1 × t3.xlarge on-demand ($0.1664/hr), customer VM |
| `ecs` | $19.03 | From `terraform/aws-ecs/terraform.tfstate`: 10 Fargate tasks ($7.67) + DocumentDB db.t3.medium ($1.87) + RDS Aurora Serverless v2 avg 1 ACU ($2.88) + 2 ALBs ($1.35) + 3 NAT Gateways ($3.24) + 2 CloudFront ($0.50) + S3 logs ($0.05) + CloudWatch ($1.00) + EFS/SM/DT ($0.50) |
| `kubernetes` | $11.17 | From `charts/` Helm defaults + aws-load-balancer-controller: EKS control plane ($2.40) + 2 × t3.large nodes ($3.99) + 4 ALB ingresses ($2.70) + 1 NAT Gateway ($1.08) + EBS ($0.50) + CloudWatch Container Insights ($0.30) + data transfer ($0.20) |
| `ec2` / `unknown` / other | $3.99 | Docker-compose fallback (single VM) |

Platform for a given instance is resolved via its **most-recent non-empty `compute` field**. If an instance migrates across platforms mid-window, it's billed at the latest platform's rate for the whole window.

**Bedrock Titan embeddings:** only for instances whose latest `embeddings_backend_kind == "bedrock"`. Cost = `delta(search_queries_total)` on that day × 100 tokens/query × $0.00002 / 1K tokens. The delta is computed from the instance's own `search_queries_total` timeseries (monotonic counter), so we never double-count queries that were already charged on a previous day.

```bash
/usr/bin/python3 .claude/skills/usage-report/generate_ltv_spend.py \
  --csv-dir OUTPUT_DIR \
  --output $DATE_DIR/ltv-spend-YYYY-MM-DD.png \
  --internal-instances .claude/skills/usage-report/known-internal-instances.md \
  --csv-out $DATE_DIR/ltv-spend-YYYY-MM-DD.csv \
  --summary-json $DATE_DIR/ltv-spend-YYYY-MM-DD.json
```

Outputs:
- PNG chart with three panels: daily EC2 compute USD (all-days + proven overlay), daily Bedrock USD, cumulative LTV USD (both lines with shaded range between them).
- CSV sidecar per day: `date, aws_instances, aws_instances_persistent, <platform>_instances[_persistent], bedrock_queries[_persistent], compute_usd[_persistent], bedrock_usd[_persistent], total_usd[_persistent], cum_total_usd[_persistent]`.
- JSON summary with headline numbers under `yesterday.all_days` vs `yesterday.proven`, `last_7_days.{all_days_total_usd, proven_total_usd}`, `ltv.{all_days, proven}`, and per-platform LTV breakdown for both models.

Embed the chart in the report's **Customer Infra Spend (AWS)** section. Include a single summary table that shows both numbers as a range (e.g. "yesterday: $292.67 – $346.01"), one short paragraph explaining the two counting rules, and the per-platform LTV breakdown (both models side by side). Flag clearly that the cost model is hypothetical (we don't actually bill these customers; these are "what it would cost them at list price").

### Step 5d: Fetch GitHub Repository Stats

Collect community-growth signals for the upstream repo (`agentic-community/mcp-gateway-registry`) using the authenticated `gh` CLI. These numbers complement telemetry by showing project interest outside of deployed instances.

```bash
# Star, fork, watcher, open-issue counts (single API call)
gh api repos/agentic-community/mcp-gateway-registry \
  --jq '{stars: .stargazers_count, forks: .forks_count, watchers: .subscribers_count, open_issues: .open_issues_count}' \
  > $DATE_DIR/github_stats.json

# Unique contributors (paginate through all pages, count unique logins)
gh api --paginate repos/agentic-community/mcp-gateway-registry/contributors \
  --jq '.[].login' | sort -u | wc -l > $DATE_DIR/github_contributors_count.txt
```

Record these numbers in the report and compare them against the previous report's `github_stats.json` (if present in the previous dated subfolder). Compute deltas for stars, forks, and contributors the same way telemetry metrics are compared.

**Note**: If `gh` is not authenticated or the API call fails, skip the GitHub section in the report and log a short note instead of failing the entire run.

### Step 6: Run Telemetry Analysis

Run the analysis script to compute all distributions, instance timelines, and metrics. This produces two files:
- `tables-YYYY-MM-DD.md` -- pre-formatted markdown tables ready to embed in the report (with executive summary comparison at the top)
- `metrics-YYYY-MM-DD.json` -- raw computed metrics as JSON (includes `per_cloud_unique_installs`)

The script automatically finds the most recent previous `metrics-*.json` file. Since output files are written to the dated subfolder (`$DATE_DIR`) but previous metrics live in *sibling* dated subfolders, you **must** pass `--search-dir OUTPUT_DIR` so the script searches the parent directory containing all dated subfolders:

```bash
INTERNAL_INSTANCES_FILE=".claude/skills/usage-report/known-internal-instances.md"
INTERNAL_FLAG=""
if [ -f "$INTERNAL_INSTANCES_FILE" ]; then
  INTERNAL_FLAG="--internal-instances $INTERNAL_INSTANCES_FILE"
fi

/usr/bin/python3 .claude/skills/usage-report/analyze_telemetry.py \
  --csv $DATE_DIR/registry_metrics.csv \
  --output-dir $DATE_DIR \
  --search-dir OUTPUT_DIR \
  --date YYYY-MM-DD \
  $INTERNAL_FLAG
```

- `--output-dir $DATE_DIR` -- where to write `tables-*.md` and `metrics-*.json`
- `--search-dir OUTPUT_DIR` -- where to search for previous `metrics-*.json` files (scans this directory and all subdirectories). **If omitted, defaults to the parent of `--output-dir`.**
- `--internal-instances` -- path to `known-internal-instances.md` listing known internal registry instance IDs. When provided, internal instances are labeled "(internal)" in the Instance Lifetime and Identified Instances tables, a Most Active Instances table is generated with an Internal column, and stickiness metrics (3+ day non-internal count, longest-running non-internal instance) are computed and included in the JSON output.

Or with an explicit previous metrics file (skips auto-detection):

```bash
/usr/bin/python3 .claude/skills/usage-report/analyze_telemetry.py \
  --csv $DATE_DIR/registry_metrics.csv \
  --output-dir $DATE_DIR \
  --date YYYY-MM-DD \
  --previous-metrics OUTPUT_DIR/PREVIOUS-DATE/metrics-PREVIOUS-DATE.json \
  $INTERNAL_FLAG
```

### Step 6b: Identify Internal vs Customer Instances

The `--internal-instances` flag passed in Step 6 handles internal instance identification automatically. The analysis script reads `.claude/skills/usage-report/known-internal-instances.md` (if it exists, since it is gitignored and may not be present on all machines) and:

1. Labels internal instances with "(internal)" in the Instance Lifetime and Identified Instances tables
2. Generates a "Most Active Instances" table ranked by activity score (max servers + agents + skills + search), with an Internal column and a Version column
3. Computes stickiness metrics (3+ day non-internal count, longest-running non-internal) and writes them to the JSON output under the `stickiness` key
4. Writes the list of internal instance IDs to the JSON output under `internal_instance_ids`

If the file does not exist, the script treats all instances as external (no internal labeling, stickiness counts all instances).

When writing the report:

1. **Clearly label known internal instances** in the Instance Lifetime table and Registry Instances table (e.g., add "(internal)" suffix or a dedicated column)
2. **Separate metrics**: Report total fleet numbers AND customer-only numbers (excluding internal instances). For example: "97 total instances (3 known internal + possibly more, ~94 potential customer instances)"
3. **Flag unusual activity from internal instances**: If internal instances show disproportionate activity (e.g., many registered servers/agents/skills, heavy search usage, frequent restarts/heartbeats), explicitly note this is internal testing activity and NOT indicative of customer usage patterns
4. **Note that additional internal instances may exist** beyond the known list -- short-lived CI/CD runs, developer local setups, etc. may not be in the known list

The known internal instances are typically the longest-running, highest-activity instances since they are always-on development environments.

### Step 6c: Run Liveness Analysis

Classify customer (non-internal) instances into liveness tiers based on recent heartbeat activity. Registry heartbeats are emitted once per 24 hours by default (`MCP_TELEMETRY_HEARTBEAT_INTERVAL_MINUTES=1440`, see [registry/core/telemetry.py](../../../registry/core/telemetry.py) and [registry/core/config.py](../../../registry/core/config.py)), which makes heartbeat counts a direct proxy for "is this deployment still running".

The script produces two files:
- `liveness-YYYY-MM-DD.md` -- a pre-formatted markdown section (tier summary table, confirmed-alive instance list, cloud/compute/auth breakdowns) ready to embed in the report
- `liveness-YYYY-MM-DD.json` -- raw counts and instance ID lists, used for delta tracking in future reports

```bash
/usr/bin/python3 .claude/skills/usage-report/analyze_liveness.py \
  --csv $DATE_DIR/registry_metrics.csv \
  --metrics-json $DATE_DIR/metrics-YYYY-MM-DD.json \
  --output-dir $DATE_DIR \
  --search-dir OUTPUT_DIR \
  --date YYYY-MM-DD \
  $INTERNAL_FLAG
```

**Tiers defined:**
- **Confirmed Alive** (leading, revenue-countable): ≥ 5 heartbeats in the last 7 days -- a registry that has phoned home almost every day for a week
- **Stronger Alive** (trailing): ≥ 10 heartbeats in the last 14 days -- durable two-week signal
- **Likely Alive**: any event (startup or heartbeat) in the last 7 days
- **Silent-but-recent**: event in last 7 days but < 5 heartbeats (new installs or heartbeat-disabled)
- **Dormant**: no event in the last 14 days (probably deprovisioned)

If a previous `liveness-*.json` file is found in `--search-dir`, the "vs Previous" column in the tier summary table is populated with deltas. On first run, it shows "baseline".

**Note:** Run this after Step 6 since it reads `metrics-YYYY-MM-DD.json` for per-instance cloud/compute/auth metadata.

### Step 7: Generate the Usage Report

Read the generated `tables-YYYY-MM-DD.md` and include its tables directly in the report. Add narrative sections (Executive Summary, Architecture Patterns, Recommendations) around the data tables. The tables file contains:

- Key Metrics table
- Registry Instance Lifetime table (age in days, sorted descending, internal instances labeled)
- Identified and Unidentified instance tables (internal instances labeled)
- Cloud, Compute, Architecture, Storage, Auth distribution tables
- Version Adoption table (with Events, % Events, unique Instances, and % Instances columns)
- Feature Adoption table
- Search Usage table
- Sticky Instance Breakdown table (one row per cloud/compute/storage/auth profile, with count, percentage, and change vs previous)
- Most Active Instances table (top 10 non-internal instances by activity score, with Version and Embeddings columns)
- Per-instance daily timelines (with servers, agents, skills, search queries)

Also read the generated `liveness-YYYY-MM-DD.md` (from Step 6c) and include its tier summary, confirmed-alive instance list, and cloud/compute/auth breakdowns as a dedicated **Liveness** section in the report (placed after "Registry Instance Lifetime" and before "Version Adoption"). The Executive Summary should mention the Confirmed-Alive and Stronger-Alive counts as the revenue-countable leading and trailing indicators.

#### Report Structure

The main body focuses on insights and charts. Detailed event-count distribution tables are moved to an appendix.

```markdown
# AI Registry -- Usage Report

*Report Date: YYYY-MM-DD*
*Data Source: Telemetry Collector (DocumentDB)*
*Collection Period: [earliest ts] to [latest ts]*

---

## Executive Summary
Lead with new installs since last report, total unique installs, dominant cloud/compute/IdP, growth trends. Also include the current GitHub star count (with delta vs previous report) as a top-line community signal. Include timeseries chart.

Include an **instance stickiness** line: "N instances have been running for 3+ days (up/down from M in the previous report). The longest-running non-internal instance is `REGISTRY_ID` at D days (previously P days)." This signals real adoption beyond one-time trials.

These numbers are pre-computed by `analyze_telemetry.py` (when `--internal-instances` is provided) and available in `metrics-YYYY-MM-DD.json` under the `stickiness` key:
- `stickiness.sticky_3plus_days`: count of non-internal instances where age_days >= 3
- `stickiness.longest_non_internal_id`: registry_id with max age_days (filtered)
- `stickiness.longest_non_internal_days`: max age_days value

Compare both numbers against the same `stickiness` values from the previous report's `metrics-*.json`.

![Registry Installs Timeseries](registry-installs-timeseries-YYYY-MM-DD.png)

### Comparison with Previous Report
- Deltas for total events, unique instances, heartbeat events, null registry_id count
- Per-cloud-provider unique registry installs comparison table
- GitHub stars delta (and forks/contributors if notable)
- Customer instances running 3+ days: current vs previous count
- Longest-running non-internal instance: current age vs previous age
- Confirmed-Alive and Stronger-Alive counts (from `liveness-*.json`): current vs previous

## Deployment Distribution (by Unique Instances)
![Instance Distribution](instance-distribution-YYYY-MM-DD.png)

## Key Metrics
| Metric | Value |
|--------|-------|
| Total Events | N |
| Unique Registry Instances | N |
| Known Internal Instances | 3 (+ possibly more) |
| Potential Customer Instances | N - internal |
| ... | ... |

## Internal Instances (Development/Testing)
List the known internal instances from known-internal-instances.md.
Note their disproportionate activity (high search, many servers/agents/skills, long uptime).
Clearly state: "Activity from these instances reflects internal testing and should not be interpreted as customer usage patterns."
Flag any unusual spikes (e.g., restart storms, heavy search bursts) with context.

## Registry Instance Lifetime
Commentary on average/max lifetime, multi-day vs single-day.
Density chart and top-10 table by age. Mark internal instances.

## Liveness (Currently Active Instances)
This section is pre-generated by `analyze_liveness.py` in `liveness-YYYY-MM-DD.md`. Include it verbatim, plus a short narrative below the tables explaining:
- How the confirmed-alive count correlates with cloud/compute/auth distributions
- Any notable shifts vs the previous report (the tier summary table has a "vs Previous" column pre-filled)
- Which confirmed-alive instances overlap with "Most Active Instances" (signaling strong customer intent)

The `liveness-YYYY-MM-DD.json` file persists counts and instance ID lists so future reports can compute deltas.

## Version Adoption
Table of version strings with event counts AND unique-instance counts. Columns: Version, Type, Events, % Events, Instances, % Instances. The Instances column is the count of distinct `registry_id` values reporting that version; % Instances is computed against the total identified-instance count. Notes on release vs dev versions, and commentary on versions with high events-per-instance (few long-running deployments) vs low events-per-instance (spreading across more distinct deployments).

## Feature Adoption
Federation, gateway mode, heartbeat rates.

## Search Usage
Total queries (deduplicated), average per instance, max from single instance.

## Heartbeat Metrics
Server/agent/skill counts, uptime, search backend, embeddings provider.

## Sticky Instance Breakdown (3+ Days)
This section is pre-generated by `analyze_telemetry.py` in the `tables-YYYY-MM-DD.md` file. It shows a single table of unique non-internal registry instances with age >= 3 days, grouped by deployment profile (cloud, compute, storage, auth combination). Each row shows the instance count, percentage, and change vs the previous report. The profile counts are also saved in the metrics JSON as `sticky_profiles` so future reports can compute deltas.

Add a short narrative highlighting the top deployment profiles and any notable shifts in the mix.

## Most Active Instances (by Feature Usage)
This table is pre-generated by `analyze_telemetry.py` in the `tables-YYYY-MM-DD.md` file. It shows the top 10 non-internal instances ranked by activity score (max servers + agents + skills + lifetime search queries), with columns: Rank, Registry ID, Cloud/Compute/Auth, Version, Embeddings, Servers, Agents, Skills, Search, Total. The Embeddings column (placed right after Version) shows the most recent non-empty `embeddings_provider` value from heartbeat events (e.g., `sentence-transformers`, `litellm`), or `unknown` if no heartbeat data is available.

Add a short narrative below the table highlighting distinct usage patterns among the top customer instances (e.g., full-featured vs search-only vs skills-catalog).

## GitHub Repository
Community-growth signals for `agentic-community/mcp-gateway-registry` pulled via the `gh` CLI in Step 5d. Include a table with current values and deltas vs the previous report:

| Metric | Previous | Current | Change |
|--------|----------|---------|--------|
| Stars | N | N | +N |
| Forks | N | N | +N |
| Contributors | N | N | +N |

Add a short narrative: direction of growth (stars/week trend), any notable jumps (e.g., post-launch spike, blog-post-driven traffic), and whether contributor count is broadening (new external contributors) or concentrating.

## Architecture Patterns Observed
3-5 distinct deployment patterns from the data.

## Recommendations
3-5 actionable insights based on the data.

## Appendix: Raw Distribution Tables
Event-count-based distribution tables for cloud, compute, architecture, storage, and auth.
These provide the raw numbers behind the instance-based chart above.
```

Save the report to `$DATE_DIR/ai-registry-usage-report-YYYY-MM-DD.md`.

### Step 8: Generate Self-Contained HTML

Convert the markdown report to a single self-contained HTML file using pandoc. The chart PNG is base64-embedded so the HTML works standalone. Run from the DATE_DIR so relative image paths resolve:

```bash
cd $DATE_DIR && pandoc ai-registry-usage-report-YYYY-MM-DD.md \
  -o ai-registry-usage-report-YYYY-MM-DD.html \
  --embed-resources --standalone \
  --css=.claude/skills/usage-report/report-style.css \
  --metadata title="AI Registry - Usage Report YYYY-MM-DD"
```

The `report-style.css` file in the skill directory provides a clean, professional layout. Pandoc must be installed:
```bash
which pandoc >/dev/null || sudo apt-get install -y pandoc
```

### Step 9: Present Results

After generating the report:
1. Display the Executive Summary (with comparison deltas, including GitHub stars delta) and Key Metrics directly in the conversation
2. Tell the user the full report path, HTML path, CSV path, and chart paths
3. Highlight the most interesting findings and notable changes from the previous report (telemetry + GitHub)

## Error Handling

- **SSH connection fails**: Check that the bastion IP is correct and security group allows your IP. The allowed CIDRs are in `terraform/telemetry-collector/terraform.tfvars` under `bastion_allowed_cidrs`.
- **Export returns 0 documents**: The telemetry collector may not have received any events yet. Check that `telemetry_enabled` is true in registry settings and the collector endpoint is reachable.
- **Terraform output fails**: Make sure you're in the right directory and have run `terraform init`.

## Example Usage

```
User: /usage-report
```

Output:
```
Executive Summary: 1074 events from 97 unique registry instances over 21 days...
Compared to previous report (2026-04-16): +327 events (+44%), +5 new instances (+5%)

Full report: .scratchpad/usage-reports/2026-04-18/ai-registry-usage-report-2026-04-18.md
HTML report: .scratchpad/usage-reports/2026-04-18/ai-registry-usage-report-2026-04-18.html
Charts:
  - .scratchpad/usage-reports/2026-04-18/instance-distribution-2026-04-18.png
  - .scratchpad/usage-reports/2026-04-18/registry-installs-timeseries-2026-04-18.png
  - .scratchpad/usage-reports/2026-04-18/compute-installs-timeseries-2026-04-18.png
  - .scratchpad/usage-reports/2026-04-18/instance-lifetime-2026-04-18.png
CSV data: .scratchpad/usage-reports/2026-04-18/registry_metrics.csv
```

```
User: /usage-report /tmp/reports
```

Output saved to `/tmp/reports/2026-04-18/`.

## Output Directory Structure

```
.scratchpad/usage-reports/
  2026-04-16/
    ai-registry-usage-report-2026-04-16.md
    ai-registry-usage-report-2026-04-16.html
    instance-distribution-2026-04-16.png
    registry-installs-timeseries-2026-04-16.png
    compute-installs-timeseries-2026-04-16.png
    compute-platform-snapshots-2026-04-16.md
    active-instances-2026-04-16.png
    active-instances-2026-04-16.csv
    ltv-spend-2026-04-16.png
    ltv-spend-2026-04-16.csv
    ltv-spend-2026-04-16.json
    instance-lifetime-2026-04-16.png
    tables-2026-04-16.md
    metrics-2026-04-16.json
    liveness-2026-04-16.md
    liveness-2026-04-16.json
    registry_metrics.csv
  2026-04-18/
    ai-registry-usage-report-2026-04-18.md
    ai-registry-usage-report-2026-04-18.html
    github_stats.json
    github_contributors_count.txt
    ...
```
