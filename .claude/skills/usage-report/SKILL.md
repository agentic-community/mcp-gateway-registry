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

### Step 6: Run Telemetry Analysis

Run the analysis script to compute all distributions, instance timelines, and metrics. This produces two files:
- `tables-YYYY-MM-DD.md` -- pre-formatted markdown tables ready to embed in the report (with executive summary comparison at the top)
- `metrics-YYYY-MM-DD.json` -- raw computed metrics as JSON (includes `per_cloud_unique_installs`)

The script automatically finds the most recent previous `metrics-*.json` file by scanning the base output directory and all dated subdirectories. You can also specify a previous metrics file explicitly:

```bash
/usr/bin/python3 .claude/skills/usage-report/analyze_telemetry.py \
  --csv $DATE_DIR/registry_metrics.csv \
  --output-dir $DATE_DIR \
  --date YYYY-MM-DD
```

Or with an explicit previous metrics file:

```bash
/usr/bin/python3 .claude/skills/usage-report/analyze_telemetry.py \
  --csv $DATE_DIR/registry_metrics.csv \
  --output-dir $DATE_DIR \
  --date YYYY-MM-DD \
  --previous-metrics OUTPUT_DIR/PREVIOUS-DATE/metrics-PREVIOUS-DATE.json
```

**Important**: Pass the **base** OUTPUT_DIR (not DATE_DIR) as the directory to search for previous metrics, so the script can find metrics from earlier dated subfolders.

### Step 7: Generate the Usage Report

Read the generated `tables-YYYY-MM-DD.md` and include its tables directly in the report. Add narrative sections (Executive Summary, Architecture Patterns, Recommendations) around the data tables. The tables file contains:

- Key Metrics table
- Registry Instance Lifetime table (age in days, sorted descending)
- Identified and Unidentified instance tables
- Cloud, Compute, Architecture, Storage, Auth distribution tables
- Version Adoption table
- Feature Adoption table
- Search Usage table
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
Lead with new installs since last report, total unique installs, dominant cloud/compute/IdP, growth trends. Include timeseries chart.

![Registry Installs Timeseries](registry-installs-timeseries-YYYY-MM-DD.png)

### Comparison with Previous Report
- Deltas for total events, unique instances, heartbeat events, null registry_id count
- Per-cloud-provider unique registry installs comparison table

## Deployment Distribution (by Unique Instances)
![Instance Distribution](instance-distribution-YYYY-MM-DD.png)

## Key Metrics
| Metric | Value |
|--------|-------|
| Total Events | N |
| Unique Registry Instances | N |
| ... | ... |

## Registry Instance Lifetime
Commentary on average/max lifetime, multi-day vs single-day.
Density chart and top-10 table by age.

## Version Adoption
Table of version strings with event counts. Notes on release vs dev versions.

## Feature Adoption
Federation, gateway mode, heartbeat rates.

## Search Usage
Total queries (deduplicated), average per instance, max from single instance.

## Heartbeat Metrics
Server/agent/skill counts, uptime, search backend, embeddings provider.

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
    ...
```
