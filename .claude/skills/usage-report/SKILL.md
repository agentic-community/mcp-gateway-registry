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

### Step 5: Generate the Usage Report

Read the downloaded CSV and the captured export output. Generate a markdown report with the following sections:

#### Report Structure

```markdown
# MCP Gateway Registry -- Usage Report

*Report Date: YYYY-MM-DD*
*Data Source: Telemetry Collector (DocumentDB)*
*Collection Period: [earliest ts] to [latest ts]*

---

## Executive Summary
- Total events, unique instances, collection period, key highlights

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

Save the report to `OUTPUT_DIR/usage-report-YYYY-MM-DD.md`.

### Step 6: Present Results

After generating the report:
1. Display the Executive Summary and Key Metrics directly in the conversation
2. Tell the user the full report path and CSV path
3. Highlight the most interesting findings

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
Executive Summary: 52 startup events from ~6 unique registry instances over 3 days...

Full report: .scratchpad/usage-reports/usage-report-2026-03-30.md
CSV data: .scratchpad/usage-reports/registry_metrics.csv
```

```
User: /usage-report /tmp/reports
```

Output saved to `/tmp/reports/usage-report-2026-03-30.md` and `/tmp/reports/registry_metrics.csv`.
