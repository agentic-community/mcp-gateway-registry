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

### Step 7: Generate the Usage Report

Read the generated `tables-YYYY-MM-DD.md` and include its tables directly in the report. Add narrative sections (Executive Summary, Architecture Patterns, Recommendations) around the data tables. The tables file contains:

- Key Metrics table
- Registry Instance Lifetime table (age in days, sorted descending, internal instances labeled)
- Identified and Unidentified instance tables (internal instances labeled)
- Cloud, Compute, Architecture, Storage, Auth distribution tables
- Version Adoption table
- Feature Adoption table
- Search Usage table
- Sticky Instance Breakdown table (one row per cloud/compute/storage/auth profile, with count, percentage, and change vs previous)
- Most Active Instances table (top 10 non-internal instances by activity score, with Version and Embeddings columns)
- Per-instance daily timelines (with servers, agents, skills, search queries)

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

## Version Adoption
Table of version strings with event counts. Notes on release vs dev versions.

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
    instance-lifetime-2026-04-16.png
    tables-2026-04-16.md
    metrics-2026-04-16.json
    registry_metrics.csv
  2026-04-18/
    ai-registry-usage-report-2026-04-18.md
    ai-registry-usage-report-2026-04-18.html
    github_stats.json
    github_contributors_count.txt
    ...
```
