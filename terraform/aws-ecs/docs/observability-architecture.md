# MCP Gateway Observability Architecture for AWS ECS (Terraform)

This document describes the observability architecture for the MCP Gateway Registry when deployed on AWS ECS using Terraform.

## Executive Summary

The Terraform ECS deployment uses the existing **metrics-service** to aggregate application metrics from the registry and auth-server, with AWS-native services (Amazon Managed Prometheus, Grafana OSS on ECS) providing durable storage and visualization. The ADOT collector runs as a sidecar in the metrics-service task, scraping its Prometheus endpoint and remote-writing to AMP. All observability resources are gated by `var.enable_observability` (default: `true`) and can be fully disabled with a single variable.

## Architecture Overview

The pipeline reuses the same metrics-service that runs in local docker-compose development. The registry and auth-server already emit metrics to `METRICS_SERVICE_URL` via HTTP POST -- no application code changes are required. In the AWS deployment, AMP replaces the local Prometheus container as the durable time-series store, and Grafana OSS on ECS replaces the local Grafana container. The same Grafana dashboards work in both environments because they query identical Prometheus metric names.

### ADOT Sidecar Pattern

The ADOT collector runs as a sidecar container within the metrics-service ECS task definition, scraping `localhost:9465`. This is necessary because the Terraform deployment uses the `terraform-aws-modules/ecs` module, which creates HTTP-type Cloud Map services. HTTP-type services register in Cloud Map but do **not** create Route53 A records -- DNS resolution is handled by the ECS Service Connect Envoy sidecar proxy. A standalone ADOT service cannot resolve Service Connect hostnames via system DNS, but co-locating as a sidecar eliminates the DNS dependency entirely.

See [issue #496](https://github.com/agentic-community/mcp-gateway-registry/issues/496) for the broader discussion on Service Connect DNS behavior.

### Ephemeral SQLite

The metrics-service uses SQLite for local buffering but ECS Fargate task storage is ephemeral. AMP serves as the durable store (150-day default retention), replacing SQLite's historical analysis role. SQLite data loss on task restart has no impact on the metrics pipeline.

## Architecture

```
+---------------------------------------------------------------------+
|                         ECS Services                                 |
|                                                                      |
|  +-----------------+  +-----------------+  +---------------------+   |
|  |    Registry     |  |   Auth Server   |  |   Other Services    |   |
|  |                 |  |                 |  |   (MCP Servers)     |   |
|  | METRICS_SERVICE |  | METRICS_SERVICE |  |  No custom metrics  |   |
|  | _URL=http://    |  | _URL=http://    |  |  (CloudWatch only)  |   |
|  | metrics:8890    |  | metrics:8890    |  |                     |   |
|  | METRICS_API_KEY |  | METRICS_API_KEY |  |                     |   |
|  +--------+--------+  +--------+--------+  +---------------------+   |
|           |                    |                                     |
|           +--------------------+                                     |
|                                |                                     |
+--------------------------------+-------------------------------------+
                                 |
                                 | HTTP POST /metrics
                                 | Header: X-API-Key: <METRICS_API_KEY>
                                 v
              +---------------------------------------------------+
              |      metrics-service ECS Task (512 CPU, 1024 MB)  |
              |                                                   |
              |  +---------------------------------------------+  |
              |  | metrics-service container                   |  |
              |  |                                             |  |
              |  |  +---------------------------------------+  |  |
              |  |  | FastAPI Application                   |  |  |
              |  |  | - Receives metrics via HTTP API       |  |  |
              |  |  | - API key auth (METRICS_API_KEY_*)    |  |  |
              |  |  | - Rate limiting (1000 req/min)        |  |  |
              |  |  | - Request validation                  |  |  |
              |  |  | - In-memory buffering (5s flush)      |  |  |
              |  |  +---------------------------------------+  |  |
              |  |                                             |  |
              |  |  +---------------------------------------+  |  |
              |  |  | OpenTelemetry Instrumentation         |  |  |
              |  |  | - Counters: auth, tool, discovery     |  |  |
              |  |  | - Histograms: latency, duration       |  |  |
              |  |  | - Custom bucket boundaries (5ms-300s) |  |  |
              |  |  | - Prometheus exporter :9465           |  |  |
              |  |  +---------------------------------------+  |  |
              |  |                                             |  |
              |  |  Ports: 8890 (API), 9465 (Prometheus)       |  |
              |  +---------------------------------------------+  |
              |                                                   |
              |  +---------------------------------------------+  |
              |  | adot-collector sidecar container            |  |
              |  |                                             |  |
              |  |  +---------------------------------------+  |  |
              |  |  | Prometheus Receiver                   |  |  |
              |  |  | - Scrapes localhost:9465              |  |  |
              |  |  | - 15s scrape interval                 |  |  |
              |  |  +---------------------------------------+  |  |
              |  |                                             |  |
              |  |  +---------------------------------------+  |  |
              |  |  | Prometheus Remote Write Exporter      |  |  |
              |  |  | - SigV4 authentication                |  |  |
              |  |  | - Writes to AMP workspace             |  |  |
              |  |  +---------------------------------------+  |  |
              |  |                                             |  |
              |  |  Health check: :13133                       |  |
              |  |  essential: false (metrics-service can run  |  |
              |  |  without ADOT; metrics just won't reach AMP)|  |
              |  +---------------------------------------------+  |
              |                                                   |
              +------------------------+--------------------------+
                                       |
                                       | Remote Write (SigV4)
                                       | https://aps-workspaces.region.amazonaws.com
                                       v
              +---------------------------------------------------+
              |   Amazon Managed Prometheus (AMP)                 |
              |                                                   |
              |  - Fully managed Prometheus-compatible            |
              |  - Automatic scaling                              |
              |  - 150-day default retention                      |
              |  - PromQL query support                           |
              |  - SigV4 authentication                           |
              |  - No infrastructure to manage                    |
              |                                                   |
              |  Alert Rules:                                     |
              |  - MCPHighErrorRate (>10% for 5 min)              |
              |  - MCPRegistryDown (no requests for 5 min)        |
              |  - MCPHighLatency (P95 > 5s for 5 min)            |
              +------------------------+--------------------------+
                                       |
                                       | PromQL Queries (SigV4)
                                       v
              +---------------------------------------------------+
              |      Grafana OSS ECS Task (512 CPU, 1024 MB)      |
              |                                                   |
              |  +---------------------------------------------+  |
              |  | Pre-configured Datasource                   |  |
              |  | - Amazon Managed Prometheus (AMP)           |  |
              |  |   - SigV4 auth via IAM task role            |  |
              |  +---------------------------------------------+  |
              |                                                   |
              |  +---------------------------------------------+  |
              |  | Pre-loaded Dashboard: MCP Analytics         |  |
              |  | - Real-time Protocol Activity               |  |
              |  | - Authentication Flow Analysis              |  |
              |  | - Active MCP Servers                        |  |
              |  | - Tool Executions per Hour                  |  |
              |  | - MCP Latency P95 (by Server & Method)      |  |
              |  | - Server Performance Dashboard              |  |
              |  | - Tool Usage Rankings                       |  |
              |  | - Error Rate Analysis                       |  |
              |  | - Client Applications Distribution          |  |
              |  | - 19 panels total                           |  |
              |  +---------------------------------------------+  |
              |                                                   |
              |  Access: https://<cloudfront>/grafana/            |
              |  Auth: admin / grafana_admin_password (from tfvars)|
              +---------------------------------------------------+
```

## Component Details

### Services Emitting Metrics

The following services emit custom metrics to the metrics-service:

| Service | Metrics Emitted | Configuration |
|---------|-----------------|---------------|
| **Registry** | Tool discovery, registry operations, health checks | `METRICS_SERVICE_URL` + `METRICS_API_KEY` env vars |
| **Auth-server** | Authentication requests (via `/validate` subrequest), session operations | `METRICS_SERVICE_URL` + `METRICS_API_KEY` env vars |
| **Nginx (Lua)** | MCP tool execution counters and duration histograms | `METRICS_API_KEY_NGINX` env var. See PR #488. |

**Note**: MCP servers (CurrentTime, MCPGW, RealServerFakeTools, etc.) do not emit custom metrics directly. However, nginx emits tool execution metrics on their behalf via `log_by_lua` -- capturing method, tool name, duration, and success/failure for all MCP protocol traffic flowing through nginx location blocks.

**MCP data-plane metrics**: MCP protocol traffic (initialize, tools/list, tools/call) is handled by nginx location blocks and proxied directly to backend servers, bypassing FastAPI entirely. The middleware in `registry/metrics/middleware.py` never observes these requests. The auth-server sees every request via `auth_request /validate`, but the auth check fires *before* `proxy_pass` -- so it captures auth latency but cannot observe tool execution duration, success/failure, or which tool was called. The nginx Lua metrics pipeline (`emit_metrics.lua` + `flush_metrics.lua`, PR #488) fills this gap.

The metrics emission flow:
1. **Registry/Auth-server** (control plane): Instantiate `MetricsClient` from `registry/metrics/client.py`. The client reads `METRICS_SERVICE_URL` and `METRICS_API_KEY` from environment variables. Metrics are sent via HTTP POST to `{METRICS_SERVICE_URL}/metrics` with `X-API-Key` header.
2. **Nginx** (data plane, PR #488): `emit_metrics.lua` runs in `log_by_lua` phase after each MCP request, writing metrics to `lua_shared_dict metrics_buffer 10m` (no network I/O). A background timer in `flush_metrics.lua` (`init_worker_by_lua`) batch-POSTs buffered metrics to metrics-service every 5-10 seconds, authenticating with `METRICS_API_KEY_NGINX`.

### API Key Authentication Configuration

The metrics-service uses a dual naming convention for API keys:

**Client Side** (registry, auth-server):
- Environment variable: `METRICS_API_KEY`
- Used to authenticate when sending metrics to metrics-service
- In Terraform: sourced from `aws_secretsmanager_secret.metrics_api_key`, auto-generated via `random_password`

**Server Side** (metrics-service):
- Environment variable pattern: `METRICS_API_KEY_<SERVICE>`
- The `setup_preshared_api_keys()` function in `metrics-service/app/main.py` discovers all environment variables matching `METRICS_API_KEY_*` on startup
- Each key is automatically registered with the service name derived from the suffix (e.g., `METRICS_API_KEY_REGISTRY` registers key for service `registry`)

**Terraform Implementation**:
- `random_password.metrics_api_key` generates a 32-character key
- `aws_secretsmanager_secret.metrics_api_key` stores it in Secrets Manager
- Registry and auth-server task definitions reference the secret as `METRICS_API_KEY`
- metrics-service receives the same secret as both `METRICS_API_KEY_REGISTRY` and `METRICS_API_KEY_AUTH`
- All secret resources are gated by `var.enable_observability`

To rotate the API key: update the secret in Secrets Manager and force redeploy the affected ECS services.

### metrics-service

The metrics-service is deployed as an ECS Fargate task:

| Configuration | Value | Notes |
|--------------|-------|-------|
| Image | `var.metrics_service_image_uri` | Built via CodeBuild or provided |
| CPU | 512 | 0.5 vCPU (shared with ADOT sidecar) |
| Memory | 1024 | 1 GB (shared with ADOT sidecar) |
| Port 8890 | HTTP API | Receives metrics from services |
| Port 9465 | Prometheus | Scraped by ADOT sidecar on localhost |
| Health Check | `GET /health` | 30s interval, 30s start period |
| Service Connect | `metrics-service:8890` | Discoverable by registry and auth-server |

Environment variables:
```
METRICS_SERVICE_HOST=0.0.0.0
PORT=8890
OTEL_SERVICE_NAME=mcp-metrics-service
OTEL_PROMETHEUS_ENABLED=true
OTEL_PROMETHEUS_PORT=9465
METRICS_RATE_LIMIT=1000
HISTOGRAM_BUCKET_BOUNDARIES=0.005,0.01,0.025,0.05,0.075,0.1,0.25,0.5,0.75,1.0,2.5,5.0,7.5,10.0,30.0,60.0,120.0,300.0
SQLITE_DB_PATH=/tmp/metrics.db
METRICS_API_KEY_REGISTRY=<from Secrets Manager>
METRICS_API_KEY_AUTH=<from Secrets Manager>
```

### ADOT Collector (Sidecar)

AWS Distro for OpenTelemetry collector runs as a sidecar in the metrics-service task:

| Configuration | Value | Notes |
|--------------|-------|-------|
| Image | `public.ecr.aws/aws-observability/aws-otel-collector:latest` | AWS-managed |
| CPU | 256 | Allocated within the 512 task CPU |
| Memory | 512 | Allocated within the 1024 task memory |
| essential | false | metrics-service continues if ADOT fails |
| Health Check | `:13133` | ADOT health extension |
| Dependency | metrics-service HEALTHY | Waits for metrics-service to start |

Configuration (embedded YAML via `AOT_CONFIG_CONTENT` env var):
```yaml
receivers:
  prometheus:
    config:
      scrape_configs:
        - job_name: 'mcp-metrics-service'
          scrape_interval: 15s
          static_configs:
            - targets: ['localhost:9465']

exporters:
  prometheusremotewrite:
    endpoint: https://aps-workspaces.<region>.amazonaws.com/workspaces/<id>/api/v1/remote_write
    auth:
      authenticator: sigv4auth

extensions:
  sigv4auth:
    region: <region>
  health_check:
    endpoint: 0.0.0.0:13133

service:
  extensions: [sigv4auth, health_check]
  pipelines:
    metrics:
      receivers: [prometheus]
      exporters: [prometheusremotewrite]
```

### Grafana OSS

Pre-configured Grafana container:

| Configuration | Value |
|--------------|-------|
| Image | `var.grafana_image_uri` |
| CPU | 512 |
| Memory | 1024 |
| Port | 3000 |
| Root URL | `/grafana/` |
| Auth | Login required (admin / `grafana_admin_password`) |
| ALB Path | `/grafana/*` |

**Note**: Anonymous access is disabled by default. The admin password is configured via `grafana_admin_password` in `terraform.tfvars` (marked as `sensitive` to prevent exposure in plan output). Generate a strong random password with: `python3 -c "import secrets; print(secrets.token_urlsafe(24))"`

**Critical Environment Variables for SigV4 Authentication:**

| Variable | Value | Purpose |
|----------|-------|---------|
| `AWS_REGION` | `<deployment region>` | AWS region for SDK |
| `GF_AUTH_SIGV4_AUTH_ENABLED` | `true` | Enables SigV4 signing for AWS datasources |
| `GF_AWS_ALLOWED_AUTH_PROVIDERS` | `default,ec2_iam_role` | Allows ECS task role credential chain |

Without `GF_AUTH_SIGV4_AUTH_ENABLED=true`, Grafana will not sign requests to AMP even if `sigV4Auth: true` is set in the datasource configuration. Without `GF_AWS_ALLOWED_AUTH_PROVIDERS`, Grafana on ECS Fargate will reject the task role credentials. Both are required.

Datasource (provisioned):
- **Amazon Managed Prometheus** -- Default datasource, SigV4 auth via IAM task role

Dashboard (provisioned):
- **MCP Analytics Comprehensive** -- 19 panels covering MCP protocol metrics (see "Grafana Dashboard Panels" below)

### Prometheus Alert Rules

Three alert rules are configured in the AMP workspace:

| Alert | Condition | Duration |
|-------|-----------|----------|
| MCPHighErrorRate | Error rate > 10% | 5 minutes |
| MCPRegistryDown | No requests received | 5 minutes |
| MCPHighLatency | P95 latency > 5 seconds | 5 minutes |

## Terraform Configuration

### Enabling Observability (default)

```hcl
enable_observability       = true
metrics_service_image_uri  = "<account>.dkr.ecr.<region>.amazonaws.com/mcp-gateway-metrics-service:latest"
grafana_image_uri          = "<account>.dkr.ecr.<region>.amazonaws.com/mcp-gateway-grafana:latest"
```

### Disabling Observability

```hcl
enable_observability = false
# No image URIs needed -- all observability resources are skipped
```

When `enable_observability = false`:
- Zero observability resources are created
- No AMP workspace, no metrics-service, no ADOT, no Grafana
- Registry and auth-server deploy without `METRICS_SERVICE_URL` or `METRICS_API_KEY`
- No cost impact from observability
- Existing functionality is completely unaffected

### Resource Gating

All observability resources use `count = var.enable_observability ? 1 : 0`:

| Resource | File |
|----------|------|
| `aws_prometheus_workspace.mcp` | `observability.tf` |
| `module.ecs_service_metrics` | `observability.tf` |
| `aws_iam_policy.adot_amp_write` | `observability.tf` |
| `aws_iam_policy.grafana_amp_query` | `observability.tf` |
| `aws_lb_target_group.grafana` | `observability.tf` |
| `aws_lb_listener_rule.grafana` | `observability.tf` |
| `aws_lb_listener_rule.grafana_https` | `observability.tf` (also gated by `enable_https`) |
| `module.ecs_service_grafana` | `observability.tf` |
| `random_password.metrics_api_key` | `secrets.tf` |
| `aws_secretsmanager_secret.metrics_api_key` | `secrets.tf` |

Conditional references in `ecs-services.tf` for the registry and auth-server environment variables use the same gate to avoid referencing resources that do not exist when observability is disabled.

## Grafana Dashboard Panels

The pre-provisioned "MCP Gateway - Analytics Dashboard" contains 19 panels:

| Panel | Type | Description |
|-------|------|-------------|
| Real-time Protocol Activity | timeseries | Live MCP request/response volume |
| Authentication Flow Analysis | timeseries | Auth method breakdown over time |
| Authentication Success Rate | stat | Current auth success percentage |
| Active MCP Servers | stat | Count of registered, enabled servers |
| Tool Executions per Hour | stat | Aggregate tool call volume |
| Most Popular Tool | stat | Highest-traffic tool name |
| MCP Latency P95 (by Server & Method) | timeseries | Tail latency per server and method |
| Request Volume Over Time | timeseries | Total request throughput |
| Error Rate Analysis | timeseries | Error percentage with threshold |
| Average Response Times | timeseries | Mean latency trends |
| Server Performance Dashboard | table | Per-server request counts, error rates, avg latency |
| Tool Usage Rankings | table | Most-called tools across all servers |
| MCP Protocol Methods Distribution | bargauge | Breakdown by MCP method type |
| Tool Usage by Call Count | barchart | Tool call volume comparison |
| Client Applications Distribution | bargauge | Traffic by MCP client |
| MCP Protocol Flow Analysis | table | Protocol step timing |
| Authentication Methods Distribution | bargauge | Auth method usage |
| Tool Execution Success Rate | timeseries | Success/failure ratio over time |
| Session Activity by Client | bargauge | Session counts per client |

Metrics begin appearing within 1-2 minutes of the first MCP request passing through the gateway.

## Metric Types Collected

### Authentication Metrics
- `mcp_auth_requests_total` -- Counter by success, method, server
- `mcp_auth_request_duration_seconds` -- Histogram of auth latency

### Tool Execution Metrics
- `mcp_tool_executions_total` -- Counter by tool, server, success
- `mcp_tool_execution_duration_seconds` -- Histogram of execution time

### Discovery Metrics
- `mcp_tool_discovery_total` -- Counter of semantic search requests
- `mcp_discovery_duration_seconds` -- Histogram of search latency

### Protocol Flow Metrics
- `mcp_protocol_latency_seconds` -- Time between protocol steps
  - initialize -> tools/list
  - tools/list -> tools/call
  - initialize -> tools/call (full flow)

### Histogram Bucket Boundaries

The default OTel SDK bucket boundaries have a smallest non-zero boundary of 5 seconds. Since most MCP responses are sub-second, `histogram_quantile(0.95, ...)` interpolates within the 0-5s bucket and reports misleading values (e.g., ~4.75s P95 for a 50ms response).

This deployment configures `ExplicitBucketHistogramAggregation` with boundaries from 5ms to 300s:
```
0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 30.0, 60.0, 120.0, 300.0
```

Configurable via the `HISTOGRAM_BUCKET_BOUNDARIES` environment variable on the metrics-service.

## Security Considerations

### Network Security
- metrics-service is deployed in private subnets
- Only accessible via Service Connect (internal) or from registry/auth-server security groups
- ADOT sidecar communicates with metrics-service on localhost (no network hop)
- Grafana is exposed via ALB path `/grafana/*` behind CloudFront (when enabled)
- No direct public internet exposure for metrics-service or ADOT

### Authentication
- Service-to-metrics-service: API key authentication (auto-generated, stored in Secrets Manager)
- ADOT-to-AMP: IAM task role with SigV4
- Grafana-to-AMP: IAM task role with SigV4
- User-to-Grafana: Login required (admin / `grafana_admin_password` from `terraform.tfvars`)

### IAM Roles

**metrics-service Task Role**:
- `SecretsManagerAccess` -- read metrics API key
- `EcsExecTask` -- ECS Exec for debugging
- `AMPRemoteWrite` -- ADOT sidecar writes to AMP

**Grafana Task Role**:
```json
{
  "Effect": "Allow",
  "Action": [
    "aps:QueryMetrics",
    "aps:GetMetricMetadata",
    "aps:GetSeries",
    "aps:GetLabels"
  ],
  "Resource": "arn:aws:aps:<region>:<account>:workspace/<workspace-id>"
}
```

## Cost Considerations

| Component | Estimated Monthly Cost | Notes |
|-----------|----------------------|-------|
| AMP | $0.90/10M samples ingested | ~$5-10/month typical |
| metrics-service + ADOT (Fargate) | ~$15/month | 512 CPU, 1024 MB (shared task) |
| Grafana OSS (Fargate) | ~$15/month | 512 CPU, 1024 MB |
| Secrets Manager | ~$0.40/month | 1 secret |
| **Total** | **~$35-40/month** | For full observability stack |

Setting `enable_observability = false` reduces this to $0.

## Differences from CloudFormation Deployment

| Aspect | CloudFormation | Terraform |
|--------|---------------|-----------|
| ADOT deployment | Standalone ECS service | Sidecar in metrics-service task |
| ADOT scrape target | `metrics-service.internal:9465` | `localhost:9465` |
| Service discovery | DNS-type Cloud Map (Route53 A records) | HTTP-type Cloud Map (no Route53) |
| API key management | CloudFormation parameter (static default) | Secrets Manager (auto-generated) |
| Resource gating | Separate nested stack | `count` on each resource |
| Grafana datasources | AMP + CloudWatch | AMP only |
| Grafana dashboards | MCP Analytics + AWS Infrastructure | MCP Analytics |

The sidecar pattern used in Terraform is a direct consequence of the HTTP-type Cloud Map limitation. See [issue #496](https://github.com/agentic-community/mcp-gateway-registry/issues/496) for details.

## References

- [MCP Gateway Metrics Architecture](../../../docs/metrics-architecture.md)
- [metrics-service Deployment Guide](../../../metrics-service/docs/deployment.md)
- [metrics-service API Reference](../../../metrics-service/docs/api-reference.md)
- [AWS ADOT Documentation](https://aws-otel.github.io/docs/introduction)
- [Amazon Managed Prometheus User Guide](https://docs.aws.amazon.com/prometheus/latest/userguide/)
- [Issue #496: Health gate blocks nginx routing to reachable servers](https://github.com/agentic-community/mcp-gateway-registry/issues/496)
