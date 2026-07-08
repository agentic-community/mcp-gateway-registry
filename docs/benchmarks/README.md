# Benchmark Reports and Scaling Guide

## Default Deployment

The MCP Gateway Registry deploys with **2 replicas** of the registry service by default on both ECS and EKS:

| Platform | Default Replicas | Autoscaling | Config Location |
|----------|-----------------|-------------|-----------------|
| AWS ECS | 2 (min), 4 (max) | Enabled (CPU 70%, Memory 80%) | `terraform/aws-ecs/terraform.tfvars` |
| Kubernetes (Helm) | 2 | Not enabled by default | `charts/registry/values.yaml` |

This configuration handles **up to 10 concurrent semantic search requests** with sub-2-second p99 latency against a corpus of 1000 servers, 900 agents, and 1000 skills.

## When to Scale

| Concurrent Search Users | Recommended Replicas | Expected p99 |
|------------------------|---------------------|--------------|
| 1-10 | 2 (default) | < 4s |
| 10-50 | 4 | < 10s |
| 50-100 | 4-8 | < 15s |

Scaling is linear: doubling replicas roughly halves latency and doubles throughput at a given concurrency level.

## How to Scale

### ECS (Terraform)

Edit `terraform/aws-ecs/terraform.tfvars`:

```hcl
autoscaling_min_capacity  = 4   # Always run at least 4 tasks
autoscaling_max_capacity  = 8   # Allow up to 8 under load
autoscaling_target_cpu    = 70  # Scale up at 70% CPU
autoscaling_target_memory = 80  # Scale up at 80% memory
```

Then apply:

```bash
cd terraform/aws-ecs && terraform apply
```

ECS autoscaling reacts to sustained CPU/memory pressure over a few minutes. For burst traffic (100 concurrent requests in < 1 second), set the min capacity to the level you need since autoscaling cannot respond fast enough to sub-minute bursts.

### EKS (Helm)

Edit `charts/registry/values.yaml`:

```yaml
app:
  replicas: 4
```

Or override at install time:

```bash
helm upgrade mcp-gateway-registry ./charts/mcp-gateway-registry-stack \
  --set registry.app.replicas=4
```

Kubernetes HPA is not configured by default. To add autoscaling on EKS, create a HorizontalPodAutoscaler targeting the registry deployment with CPU/memory thresholds matching the ECS configuration (70% CPU, 80% memory).

## Benchmark Reports

Reports in this directory document measured performance at specific configurations. Each filename encodes the test parameters:

```
benchmark-{date}-{compute}-{backend}-{replicas}x.md
```

For example, `benchmark-2026-05-21-ecs-documentdb-4x.md` was measured on ECS with DocumentDB backend and 4 registry task replicas.

### Available Reports

- [benchmark-2026-05-21-ecs-documentdb-4x.md](benchmark-2026-05-21-ecs-documentdb-4x.md): 4 ECS tasks, DocumentDB, 1000 entities per type, 50 iterations

More reports with other configurations (2x replicas, mongodb-ce backend, Kubernetes) will be added as testing continues.

## Running Your Own Benchmarks

See [`tests/stress/README.md`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/tests/stress/README.md) for the full benchmark procedure. The sequence is:

```bash
# 1. Register entities
bash tests/stress/run_stress_test.sh 1000 \
    --base-url <REGISTRY_URL> --token-file .token

# 2. Measure API latency
uv run python -m tests.stress.measure_api_performance \
    --size 1000 --base-url <REGISTRY_URL> --iterations 50 --token-file .token

# 3. Measure search concurrency
uv run python -m tests.stress.measure_search_concurrency \
    --size 1000 --base-url <REGISTRY_URL> --token-file .token --iterations 50

# 4. Generate report
/usr/bin/python3 .claude/skills/benchmark-report/generate_benchmark_report.py \
    --results-dir tests/stress/results/documentdb/size-1000
```

The report generator auto-detects the deployment configuration (version, compute, backend, replica count) from the registry's telemetry info endpoint and includes it in the report.
