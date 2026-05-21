---
name: benchmark-report
description: Generate a benchmark report from stress test results (registration, API performance, search concurrency). Reads JSON result files and produces a markdown report suitable for docs/benchmarks/.
license: Apache-2.0
metadata:
  author: mcp-gateway-registry
  version: "1.0"
---

# Benchmark Report Skill

Generate a markdown benchmark report from stress test result files. The report documents the registry's performance characteristics under load, including registration throughput, API latency, and semantic search concurrency scaling.

## Prerequisites

1. Stress test results must exist in `tests/stress/results/<backend>/size-<N>/`
2. Required files: `registration.json`, `api_perf.json`, `search_concurrency.json`
3. All JSON files should include `registry_info` (deployment configuration snapshot)

## Input

```
/benchmark-report [RESULTS_DIR] [OUTPUT_PATH]
```

- **RESULTS_DIR** - Path to the results directory (default: `tests/stress/results/documentdb/size-100`)
- **OUTPUT_PATH** - Where to write the report (default: `docs/benchmarks/benchmark-report.md`)

## Workflow

### Step 1: Run the Report Generator

```bash
/usr/bin/python3 .claude/skills/benchmark-report/generate_benchmark_report.py \
  --results-dir tests/stress/results/documentdb/size-100 \
  --output docs/benchmarks/benchmark-report.md
```

The script reads all three JSON files, extracts the key metrics, and produces a structured markdown report.

### Step 2: Review and Present

After generation:
1. Display the executive summary in the conversation
2. Tell the user the output path
3. Note any missing data (e.g., if one of the JSON files is absent)

## Report Structure

The generated report includes:

1. **Deployment Configuration** - from `registry_info` (version, cloud, compute, storage, auth, embeddings, corpus size)
2. **Registration Throughput** - from `registration.json` (success rates, latency percentiles per entity type)
3. **API Latency (Serial)** - from `api_perf.json` (list endpoints and semantic search at k=5/10/50)
4. **Search Concurrency Scaling** - from `search_concurrency.json` (latency and throughput at concurrency 1/10/100)
5. **Scaling Observations** - interpretation of how latency degrades under load

## Output

```
docs/benchmarks/
  benchmark-report.md    # The generated report (committed to repo)
```
