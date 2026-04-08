---
name: usage-report
description: Generate a usage report for MCP Gateway Registry by SSHing into the telemetry bastion host, exporting telemetry data from DocumentDB, and producing a formatted markdown report with deployment insights.
license: Apache-2.0
metadata:
  author: mcp-gateway-registry
  version: "1.0"
---

# Usage Report Skill

Export telemetry data from the MCP Gateway Registry's DocumentDB telemetry collector and generate a usage report showing deployment patterns, version adoption, and feature usage in the wild.

## Prerequisites

1. **SSH key** at `~/.ssh/id_ed25519` with access to the bastion host
2. **Terraform state** available in `terraform/telemetry-collector/` (to read bastion IP)
3. **Bastion host enabled** (`bastion_enabled = true` in `terraform/telemetry-collector/terraform.tfvars`)
4. **AWS credentials** configured on the bastion host (for Secrets Manager access)

## Input

The skill accepts optional parameters:

```
/usage-report [OUTPUT_DIR]
```

- **OUTPUT_DIR** - Directory to save the report (default: `.scratchpad/usage-reports/`)

If OUTPUT_DIR is not provided, save to `.scratchpad/usage-reports/`.

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

### Step 4: Download the CSV

```bash
scp -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519 \
  ec2-user@$BASTION_IP:/tmp/registry_metrics.csv \
  OUTPUT_DIR/registry_metrics.csv
```

### Step 5: Install Python Dependencies and Generate Charts

First, ensure matplotlib and seaborn are available on the system Python:

```bash
/usr/bin/python3 -c "import matplotlib, seaborn" 2>/dev/null || pip install --break-system-packages matplotlib seaborn
```

Then generate the deployment distribution chart:

```bash
/usr/bin/python3 .claude/skills/usage-report/generate_charts.py \
  --csv OUTPUT_DIR/registry_metrics.csv \
  --output OUTPUT_DIR/deployment-distribution-YYYY-MM-DD.png
```

This produces a single faceted PNG with 6 subplots: Cloud Provider, Compute Platform, Storage Backend, Auth Provider, Version Type, and Deployment Mode. Each subplot shows counts and percentages.

### Step 5b: Generate Timeseries Chart

Generate a timeseries chart showing unique registry installs per cloud provider over time. This reads ALL CSV files in the output directory to build a complete historical view:

```bash
/usr/bin/python3 .claude/skills/usage-report/generate_timeseries_chart.py \
  --csv-dir OUTPUT_DIR \
  --output OUTPUT_DIR/registry-installs-timeseries-YYYY-MM-DD.png
```

This produces a PNG with two subplots:
- **Cumulative Unique Registry Installs** -- running total of unique registry_ids per cloud provider
- **Daily Active Registry Installs** -- unique registry_ids seen each day per cloud provider

### Step 6: Run Telemetry Analysis

Run the analysis script to compute all distributions, instance timelines, and metrics. This produces two files:
- `tables-YYYY-MM-DD.md` -- pre-formatted markdown tables ready to embed in the report (with executive summary comparison at the top)
- `metrics-YYYY-MM-DD.json` -- raw computed metrics as JSON (includes `per_cloud_unique_installs`)

The script automatically finds the most recent previous `metrics-*.json` file in the output directory and generates an executive summary with deltas at the top of the tables file. You can also specify a previous metrics file explicitly:

```bash
/usr/bin/python3 .claude/skills/usage-report/analyze_telemetry.py \
  --csv OUTPUT_DIR/registry_metrics.csv \
  --output-dir OUTPUT_DIR \
  --date YYYY-MM-DD
```

Or with an explicit previous metrics file:

```bash
/usr/bin/python3 .claude/skills/usage-report/analyze_telemetry.py \
  --csv OUTPUT_DIR/registry_metrics.csv \
  --output-dir OUTPUT_DIR \
  --date YYYY-MM-DD \
  --previous-metrics OUTPUT_DIR/metrics-PREVIOUS-DATE.json
```

### Step 7: Generate the Usage Report

Read the generated `tables-YYYY-MM-DD.md` and include its tables directly in the report. Add narrative sections (Executive Summary, Architecture Patterns, Recommendations) around the data tables. The tables file contains:

- Key Metrics table
- Identified and Unidentified instance tables
- Cloud, Compute, Architecture, Storage, Auth distribution tables
- Version Adoption table
- Feature Adoption table
- Search Usage table
- Per-instance daily timelines (with servers, agents, skills, search queries)

#### Report Structure

```markdown
# AI Registry -- Usage Report

*Report Date: YYYY-MM-DD*
*Data Source: Telemetry Collector (DocumentDB)*
*Collection Period: [earliest ts] to [latest ts]*

---

## Executive Summary
- Comparison with previous report (auto-detected from most recent metrics JSON)
- Deltas for total events, unique instances, heartbeat events, null registry_id count
- Per-cloud-provider unique registry installs comparison table

## Deployment Distribution

![Deployment Distribution](deployment-distribution-YYYY-MM-DD.png)

## Registry Installs Timeseries

![Registry Installs Timeseries](registry-installs-timeseries-YYYY-MM-DD.png)

## Key Metrics
| Metric | Value |
|--------|-------|
| Total Events | N |
| Unique Registry Instances | N |
| ... | ... |

## Deployment Landscape

### Registry Instances
Table of unique registry_id values with their cloud, compute, storage, auth, federation status.

### Cloud Provider Distribution
Count and percentage of each cloud value (aws, azure, gcp, unknown).

### Compute Platform Distribution
Count and percentage of each compute value (docker, ecs, kubernetes, etc).

### Storage Backend Distribution
Count and percentage of each storage value (mongodb-ce, documentdb, etc).

### Auth Provider Distribution
Count and percentage of each auth value (auth0, keycloak, entra, cognito, none).

## Version Adoption
Table of version strings with counts and percentages. Note which are release vs dev/branch versions.

## Feature Adoption
- Federation enabled rate
- with-gateway vs registry-only mode
- Heartbeat opt-in rate

## Search Usage
- Total queries, average per instance, max from single instance

## Architecture Patterns Observed
Identify 3-5 distinct deployment patterns from the data (e.g., "Dev Setup", "AWS Production", "Azure Enterprise").

## Recommendations
3-5 actionable insights based on the data.
```

Save the report to `OUTPUT_DIR/ai-registry-usage-report-YYYY-MM-DD.md`.

### Step 8: Generate Self-Contained HTML

Convert the markdown report to a single self-contained HTML file using pandoc. The chart PNG is base64-embedded so the HTML works standalone. Run from the OUTPUT_DIR so relative image paths resolve:

```bash
cd OUTPUT_DIR && pandoc ai-registry-usage-report-YYYY-MM-DD.md \
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
1. Display the Executive Summary (with comparison deltas) and Key Metrics directly in the conversation
2. Tell the user the full report path, HTML path, CSV path, and chart paths
3. Highlight the most interesting findings and notable changes from the previous report

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
Executive Summary: 68 startup events from ~7 unique registry instances over 4 days...
Compared to previous report (2026-03-30): +12 events (+21%), +2 new instances (+40%)

Full report: .scratchpad/usage-reports/ai-registry-usage-report-2026-03-31.md
HTML report: .scratchpad/usage-reports/ai-registry-usage-report-2026-03-31.html
Charts:
  - .scratchpad/usage-reports/deployment-distribution-2026-03-31.png
  - .scratchpad/usage-reports/registry-installs-timeseries-2026-03-31.png
CSV data: .scratchpad/usage-reports/registry_metrics.csv
```

```
User: /usage-report /tmp/reports
```

Output saved to `/tmp/reports/`.
