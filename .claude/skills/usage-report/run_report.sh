#!/bin/bash
#
# run_report.sh - Half A of the usage-report pipeline (everything up to commentary).
#
# Runs: telemetry export (bastion) -> all 14 charts -> telemetry + liveness
# analysis -> deterministic report render -> commentary manifest extract.
#
# Stops at the LLM hinge: it prints the path to commentary-manifest.json. The
# agent running the skill then writes commentary.json and calls finish_report.sh.
#
# Usage:
#   run_report.sh [YYYY-MM-DD] [OUTPUT_DIR]
#
#   YYYY-MM-DD   Report date (default: today, UTC).
#   OUTPUT_DIR   Base reports dir (default: .scratchpad/usage-reports).
#
# Environment overrides:
#   FORCE_EXPORT=1   Re-export from the bastion even if the CSV already exists.
#   BASTION_IP=...   Skip the terraform lookup and use this IP.
#   SSH_KEY=...      SSH identity file (default: ~/.ssh/id_ed25519).
#
# All steps are fail-loud: any error aborts the whole run (set -euo pipefail).

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration and derived values
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repo root is four levels up: .claude/skills/usage-report -> repo root.
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

REPORT_DATE="${1:-$(date -u +%F)}"
OUTPUT_DIR="${2:-.scratchpad/usage-reports}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
PY="/usr/bin/python3"

# Previous complete day (report date - 1) for active-on-date / yesterday args.
PREV_DATE="$(date -u -d "$REPORT_DATE - 1 day" +%F)"

# The dated subfolder that holds all artifacts for this run.
DATE_DIR="$OUTPUT_DIR/$REPORT_DATE"
CSV="$DATE_DIR/registry_metrics.csv"
INTERNAL_FILE="$SCRIPT_DIR/known-internal-instances.md"

# Always operate from the repo root so relative paths resolve consistently.
cd "$REPO_ROOT"

echo "=============================================================="
echo "Usage report - Half A (run_report.sh)"
echo "  Report date:   $REPORT_DATE"
echo "  Previous day:  $PREV_DATE"
echo "  Output dir:    $OUTPUT_DIR"
echo "  Dated subdir:  $DATE_DIR"
echo "=============================================================="

mkdir -p "$DATE_DIR"

# Internal-instances flag is optional (the file is gitignored).
INTERNAL_FLAG=()
if [ -f "$INTERNAL_FILE" ]; then
    INTERNAL_FLAG=(--internal-instances "$INTERNAL_FILE")
    echo "Using internal-instances allowlist: $INTERNAL_FILE"
else
    echo "No internal-instances allowlist found; treating all instances as external."
fi


# ---------------------------------------------------------------------------
# Step 1-4: Export telemetry from the bastion and download the CSV
# ---------------------------------------------------------------------------

_export_telemetry() {
    if [ -f "$CSV" ] && [ "${FORCE_EXPORT:-0}" != "1" ]; then
        echo ">>> CSV already exists ($CSV); skipping export. Set FORCE_EXPORT=1 to re-export."
        return 0
    fi

    local bastion_ip="${BASTION_IP:-}"
    if [ -z "$bastion_ip" ]; then
        echo ">>> Looking up bastion IP from terraform..."
        bastion_ip="$(cd "$REPO_ROOT/terraform/telemetry-collector" && terraform output -raw bastion_public_ip)"
    fi
    if [ -z "$bastion_ip" ] || [ "$bastion_ip" = "Bastion not enabled" ]; then
        echo "ERROR: bastion IP unavailable. Set bastion_enabled=true and terraform apply, or pass BASTION_IP=..." >&2
        exit 1
    fi
    echo ">>> Bastion IP: $bastion_ip"

    echo ">>> Copying export script to bastion..."
    scp -o StrictHostKeyChecking=no -i "$SSH_KEY" \
        "$REPO_ROOT/terraform/telemetry-collector/bastion-scripts/telemetry_db.py" \
        "ec2-user@$bastion_ip:~/telemetry_db.py"

    echo ">>> Running export on bastion (this can take ~30s)..."
    ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" \
        "ec2-user@$bastion_ip" \
        'python3 telemetry_db.py export --output /tmp/registry_metrics.csv 2>&1'

    echo ">>> Downloading CSV..."
    scp -o StrictHostKeyChecking=no -i "$SSH_KEY" \
        "ec2-user@$bastion_ip:/tmp/registry_metrics.csv" \
        "$CSV"

    echo ">>> Exported $(wc -l < "$CSV") rows to $CSV"

    # Safety net: a combined export must contain BOTH event types. A CSV with
    # startups but zero heartbeats is the classic silent-drop failure (mongosh
    # fetch timeout on the t2.micro) and would zero out the entire Liveness /
    # Engagement / LTV half of the report. Fail loud here so the workaround is
    # never needed manually. (The bastion telemetry_db.py also guards this now;
    # this covers the case where an older script is present on the bastion.)
    _verify_both_event_types "$CSV"
}


# Verify a combined export CSV has both startup and heartbeat rows.
_verify_both_event_types() {
    local csv="$1"
    local has_startup has_heartbeat
    has_startup="$(awk -F, 'NR>1 && $1=="startup" {print "yes"; exit}' "$csv")"
    has_heartbeat="$(awk -F, 'NR>1 && $1=="heartbeat" {print "yes"; exit}' "$csv")"
    if [ "$has_startup" != "yes" ] || [ "$has_heartbeat" != "yes" ]; then
        echo "ERROR: export CSV is missing an event type (startup=${has_startup:-no}, heartbeat=${has_heartbeat:-no})." >&2
        echo "       This is the silent heartbeat-drop failure. Not proceeding with a partial export." >&2
        echo "       Re-run with FORCE_EXPORT=1; if it recurs, export each collection separately on the bastion." >&2
        rm -f "$csv"
        exit 1
    fi
    echo ">>> Verified export contains both startup and heartbeat events."
}


# ---------------------------------------------------------------------------
# Step 5: Ensure chart dependencies, then generate charts
# ---------------------------------------------------------------------------

_ensure_deps() {
    echo ">>> Checking chart dependencies (matplotlib, seaborn)..."
    "$PY" -c "import matplotlib, seaborn" 2>/dev/null \
        || pip install --break-system-packages matplotlib seaborn
}


_generate_charts_no_metrics() {
    # Charts that read only the CSV (or scan all CSVs); independent of metrics JSON.
    echo ">>> Generating CSV-only charts..."

    "$PY" "$SCRIPT_DIR/generate_instance_distribution_chart.py" \
        --csv "$CSV" \
        --output "$DATE_DIR/instance-distribution-$REPORT_DATE.png"

    "$PY" "$SCRIPT_DIR/generate_instance_distribution_chart.py" \
        --csv "$CSV" \
        --output "$DATE_DIR/instance-distribution-active-$PREV_DATE.png" \
        --active-on-date "$PREV_DATE"

    "$PY" "$SCRIPT_DIR/generate_timeseries_chart.py" \
        --csv-dir "$OUTPUT_DIR" \
        --output "$DATE_DIR/registry-installs-timeseries-$REPORT_DATE.png" \
        --exclude-incomplete-day "$REPORT_DATE"

    "$PY" "$SCRIPT_DIR/generate_compute_timeseries_chart.py" \
        --csv-dir "$OUTPUT_DIR" \
        --output "$DATE_DIR/compute-installs-timeseries-$REPORT_DATE.png" \
        --snapshots-table "$DATE_DIR/compute-platform-snapshots-$REPORT_DATE.md" \
        --exclude-incomplete-day "$REPORT_DATE"

    "$PY" "$SCRIPT_DIR/generate_active_instances_chart.py" \
        --csv-dir "$OUTPUT_DIR" \
        --output "$DATE_DIR/active-instances-$REPORT_DATE.png" \
        "${INTERNAL_FLAG[@]}" \
        --csv-out "$DATE_DIR/active-instances-$REPORT_DATE.csv" \
        --exclude-incomplete-day "$REPORT_DATE"

    "$PY" "$SCRIPT_DIR/generate_ltv_spend.py" \
        --csv-dir "$OUTPUT_DIR" \
        --output "$DATE_DIR/ltv-spend-$REPORT_DATE.png" \
        "${INTERNAL_FLAG[@]}" \
        --csv-out "$DATE_DIR/ltv-spend-$REPORT_DATE.csv" \
        --summary-json "$DATE_DIR/ltv-spend-$REPORT_DATE.json" \
        --exclude-incomplete-day "$REPORT_DATE"

    "$PY" "$SCRIPT_DIR/generate_daily_reporters_chart.py" \
        --csv-dir "$OUTPUT_DIR" \
        --output "$DATE_DIR/daily-reporters-$REPORT_DATE.png" \
        "${INTERNAL_FLAG[@]}" \
        --csv-out "$DATE_DIR/daily-reporters-$REPORT_DATE.csv" \
        --exclude-incomplete-day "$REPORT_DATE"

    "$PY" "$SCRIPT_DIR/generate_install_forecast.py" \
        --csv-dir "$OUTPUT_DIR" \
        --output "$DATE_DIR/install-forecast-$REPORT_DATE.png" \
        --summary-json "$DATE_DIR/install-forecast-$REPORT_DATE.json"

    "$PY" "$SCRIPT_DIR/generate_detection_by_version_chart.py" \
        --csv "$CSV" \
        --output "$DATE_DIR/detection-by-version-$REPORT_DATE.png" \
        --csv-out "$DATE_DIR/detection-by-version-$REPORT_DATE.csv" \
        --snapshot-date "$REPORT_DATE"

    "$PY" "$SCRIPT_DIR/generate_prod_internal_chart.py" \
        --csv-dir "$OUTPUT_DIR" \
        --output "$DATE_DIR/prod-internal-timeseries-$REPORT_DATE.png" \
        --summary-json "$DATE_DIR/prod-internal-$REPORT_DATE.json" \
        "${INTERNAL_FLAG[@]}" \
        --yesterday "$PREV_DATE" \
        --exclude-incomplete-day "$REPORT_DATE"
}


_generate_charts_with_metrics() {
    # Charts that read metrics-*.json / liveness-*.json (run after analysis).
    echo ">>> Generating metrics-dependent charts..."
    local metrics="$DATE_DIR/metrics-$REPORT_DATE.json"
    local liveness="$DATE_DIR/liveness-$REPORT_DATE.json"

    "$PY" "$SCRIPT_DIR/generate_lifetime_chart.py" \
        --metrics "$metrics" \
        --output "$DATE_DIR/instance-lifetime-$REPORT_DATE.png"

    "$PY" "$SCRIPT_DIR/generate_lifetime_by_compute_chart.py" \
        --metrics "$metrics" \
        --output "$DATE_DIR/instance-lifetime-by-compute-$REPORT_DATE.png" \
        --box-output "$DATE_DIR/instance-lifetime-box-by-compute-$REPORT_DATE.png"

    "$PY" "$SCRIPT_DIR/generate_lifetime_buckets_chart.py" \
        --csv-dir "$OUTPUT_DIR" \
        --output "$DATE_DIR/lifetime-buckets-$REPORT_DATE.png" \
        --csv-out "$DATE_DIR/lifetime-buckets-$REPORT_DATE.csv"

    "$PY" "$SCRIPT_DIR/generate_adoption_funnel_chart.py" \
        --metrics "$metrics" \
        --liveness "$liveness" \
        --output "$DATE_DIR/adoption-funnel-$REPORT_DATE.png"
}


# ---------------------------------------------------------------------------
# Step 5g: GitHub stats (non-fatal; the section is skipped if this fails)
# ---------------------------------------------------------------------------

_fetch_github_stats() {
    echo ">>> Fetching GitHub stats..."
    "$SCRIPT_DIR/fetch_github_stats.sh" "$DATE_DIR" || \
        echo "WARNING: GitHub stats fetch failed; the report will omit the GitHub section."
}


# ---------------------------------------------------------------------------
# Step 6 / 6c: Telemetry and liveness analysis
# ---------------------------------------------------------------------------

_run_analysis() {
    echo ">>> Running telemetry analysis..."
    "$PY" "$SCRIPT_DIR/analyze_telemetry.py" \
        --csv "$CSV" \
        --output-dir "$DATE_DIR" \
        --search-dir "$OUTPUT_DIR" \
        --date "$REPORT_DATE" \
        "${INTERNAL_FLAG[@]}"

    echo ">>> Running liveness analysis..."
    "$PY" "$SCRIPT_DIR/analyze_liveness.py" \
        --csv "$CSV" \
        --metrics-json "$DATE_DIR/metrics-$REPORT_DATE.json" \
        --output-dir "$DATE_DIR" \
        --search-dir "$OUTPUT_DIR" \
        --date "$REPORT_DATE" \
        "${INTERNAL_FLAG[@]}"
}


# ---------------------------------------------------------------------------
# Chart completeness gate: fail before rendering if any of the 14 are missing
# ---------------------------------------------------------------------------

_verify_charts() {
    echo ">>> Verifying all 14 mandatory charts are present..."
    local charts=(
        "registry-installs-timeseries-$REPORT_DATE.png"
        "instance-distribution-$REPORT_DATE.png"
        "instance-distribution-active-$PREV_DATE.png"
        "instance-lifetime-$REPORT_DATE.png"
        "instance-lifetime-box-by-compute-$REPORT_DATE.png"
        "lifetime-buckets-$REPORT_DATE.png"
        "active-instances-$REPORT_DATE.png"
        "compute-installs-timeseries-$REPORT_DATE.png"
        "install-forecast-$REPORT_DATE.png"
        "daily-reporters-$REPORT_DATE.png"
        "ltv-spend-$REPORT_DATE.png"
        "adoption-funnel-$REPORT_DATE.png"
        "detection-by-version-$REPORT_DATE.png"
        "prod-internal-timeseries-$REPORT_DATE.png"
    )
    local missing=0
    for c in "${charts[@]}"; do
        if [ ! -f "$DATE_DIR/$c" ]; then
            echo "  MISSING: $c"
            missing=$((missing + 1))
        fi
    done
    if [ "$missing" -gt 0 ]; then
        echo "ERROR: $missing mandatory chart(s) missing; aborting before render." >&2
        exit 1
    fi
    echo "  All 14 charts present."
}


# ---------------------------------------------------------------------------
# Step 7: Render report; extract commentary manifest
# ---------------------------------------------------------------------------

_render_and_extract() {
    echo ">>> Rendering deterministic report..."
    "$PY" "$SCRIPT_DIR/render_report.py" \
        --date "$REPORT_DATE" \
        --output-dir "$DATE_DIR" \
        --search-dir "$OUTPUT_DIR"

    echo ">>> Extracting commentary manifest..."
    "$PY" "$SCRIPT_DIR/augment_with_commentary.py" extract \
        --md "$DATE_DIR/ai-registry-usage-report-$REPORT_DATE.md" \
        --date "$REPORT_DATE" \
        --output "$DATE_DIR/commentary-manifest.json"
}


# ---------------------------------------------------------------------------
# Control flow
# ---------------------------------------------------------------------------

main() {
    _export_telemetry
    _ensure_deps
    _generate_charts_no_metrics
    _fetch_github_stats
    _run_analysis
    _generate_charts_with_metrics
    _verify_charts
    _render_and_extract

    echo "=============================================================="
    echo "Half A complete."
    echo ""
    echo "NEXT: the agent writes analyst commentary to:"
    echo "  $DATE_DIR/commentary.json"
    echo "using the manifest at:"
    echo "  $DATE_DIR/commentary-manifest.json"
    echo ""
    echo "Then run:"
    echo "  $SCRIPT_DIR/finish_report.sh $REPORT_DATE $OUTPUT_DIR"
    echo "=============================================================="
}

main "$@"
