# MCP Gateway Observability Guide

> Looking for the pre-1.25.0 architecture (HTTP POSTs to metrics-service +
> SQLite + port 9465)? See [OBSERVABILITY-LEGACY.md](OBSERVABILITY-LEGACY.md).
> That document is retained for the 1.25.0 transition window and will be
> removed in 1.26.0.

This guide describes the **current** observability architecture (1.25.0+),
the metrics each service emits, and a cookbook of PromQL queries for the
investigations operators most often need to run.

## Table of Contents

- [Architecture in one diagram](#architecture-in-one-diagram)
- [Configuration](#configuration)
- [Metric inventory](#metric-inventory)
- [Query cookbook](#query-cookbook)
- [Verifying the migration is working](#verifying-the-migration-is-working)
- [Troubleshooting](#troubleshooting)

## Architecture in one diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                Registry / Auth-Server / Mcpgw                      │
│                                                                    │
│  In-process OTel SDK with:                                         │
│   • Path-2 events (mcpgw_registry_operation_total, mcpgw_registry_auth_request_total,   │
│     mcpgw_registry_tool_execution_total, tool_discovery_total, protocol_latency)  │
│   • Path-3 in-process counters (mcpgw_registry_nginx_config_writes_total,         │
│     peer_sync_failures_total, m2m_orphan_cleanups_total, ...)      │
│   • HTTP auto-instrumentation (http_server_duration_milliseconds_*)│
│   • Mcpgw per-tool metrics (mcpgw_registry_tool_invocations_total,          │
│     mcpgw_registry_tool_duration)                                           │
└────────────────────────┬───────────────────────────────────────────┘
                         │
   ┌─────────────────────┴─────────────────────┐
   │                                           │
   │ HTTP GET /metrics on :9464                │ OTLP push (when configured)
   │ (always-on Prometheus exporter)           │ to OTEL_EXPORTER_OTLP_ENDPOINT
   │                                           │
   ▼                                           ▼
Prometheus (Compose) /                    Per-task ADOT sidecar (ECS)
in-cluster Prometheus (EKS)                  → Amazon Managed Prometheus
   │
   ▼
Grafana (or any Prometheus-compatible UI)
```

Three differences from the legacy architecture:

1. **No metrics-service container.** Each service emits metrics in-process via
   the OpenTelemetry SDK, not via HTTP POSTs to a separate Python service.
2. **No SQLite store.** Long-term retention is the operator's observability
   backend's job (Prometheus TSDB, AMP, Datadog, Grafana Cloud, etc.).
3. **No API keys.** The legacy `METRICS_API_KEY_*` family is unused; operators
   can remove them from `.env`. They are removed entirely in 1.26.0.

## Configuration

Two new application-level settings introduced in 1.25.0, plus a handful of
standard OTel SDK env vars. The full cross-surface mapping (Docker Compose
env vars, Terraform tfvars, Helm values paths) lives in
[docs/unified-parameter-reference.md, Group 25](unified-parameter-reference.md#group-25--otlp--opentelemetry-export).
The summary below highlights each setting, its default, and where to find it
on each deployment surface.

### Setting 1: `METRICS_LEGACY_HTTP_POST` — transition flag

| Surface | Where to set | Default |
|---|---|---|
| Docker Compose | `METRICS_LEGACY_HTTP_POST` in `.env` | `false` |
| Terraform / ECS | Hardcoded to `"false"` in the container env block in `terraform/aws-ecs/modules/mcp-gateway/ecs-services.tf` | `"false"` |
| Helm / EKS | `app.metricsLegacyHttpPost` in `charts/registry/values.yaml`, `metrics.legacyHttpPost` in `charts/auth-server/values.yaml` and `charts/mcpgw/values.yaml` | `false` |

When `true`, services ALSO POST JSON events to the legacy `metrics-service:8890`
in addition to the native OTel emission. Used during the 1.25.0 → 1.26.0
transition window to verify dashboards before the cutover. **Removed in 1.26.0
along with the metrics-service container itself.**

### Setting 2: `OTEL_METRIC_EXPORT_INTERVAL_MS` — SDK flush interval

| Surface | Where to set | Default |
|---|---|---|
| Docker Compose | `OTEL_METRIC_EXPORT_INTERVAL_MS` in `.env` | `15000` |
| Terraform / ECS | Hardcoded to `"15000"` in the container env block in `ecs-services.tf` | `"15000"` |
| Helm / EKS | `app.otelMetricExportIntervalMs` (registry), `metrics.otelExportIntervalMs` (auth-server, mcpgw) | `"15000"` |

Lower (e.g., `5000`) for near-real-time dashboards during incident response;
raise (e.g., `30000`) for high-traffic production where reduced OTLP push
frequency matters.

### Setting 3: `OTEL_EXPORTER_PROMETHEUS_HOST` / `OTEL_EXPORTER_PROMETHEUS_PORT`

| Surface | Where to set | Default |
|---|---|---|
| Docker Compose | `OTEL_EXPORTER_PROMETHEUS_HOST` / `_PORT` in `.env` | `0.0.0.0` / `9464` |
| Terraform / ECS | Not set explicitly; SDK default `0.0.0.0:9464` is used | `0.0.0.0:9464` |
| Helm / EKS | `app.otelExporterPrometheusHost` / `app.otelExporterPrometheusPort` (registry), `metrics.exporterPrometheusHost` / `metrics.exporterPrometheusPort` (auth-server, mcpgw) | `0.0.0.0:9464` |

Bind address and port for the in-process Prometheus exporter listener. EKS
needs `0.0.0.0` because Prometheus runs in a different pod (the chart's
`NetworkPolicy` template gates access). Compose can use `127.0.0.1` to keep
the port unreachable from outside the Docker network.

### Setting 4: `OTEL_EXPORTER_OTLP_ENDPOINT` and friends

| Surface | Where to set | Default |
|---|---|---|
| Docker Compose | `OTEL_EXPORTER_OTLP_ENDPOINT` (and `_PROTOCOL`, `_HEADERS`) in `.env`. Default compose ships an `otel-collector` service; uncomment the line `OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317` to enable push. | unset |
| Terraform / ECS | The container env block in `ecs-services.tf` sets `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317` when `var.enable_observability=true` (so the per-task ADOT sidecar receives it). | `http://localhost:4317` when observability is on, else unset |
| Helm / EKS | Operators inject this via the chart's `extraEnv` block (or via `OTEL_EXPORTER_OTLP_ENDPOINT` in the chart's configmap). Point at an in-cluster collector or the OTLP receiver of your chosen vendor. | unset |

When set, the OTel SDK ALSO pushes metrics + traces via OTLP in addition to
serving the Prometheus exporter on `:9464`. The docker entrypoint additionally
wraps uvicorn with `opentelemetry-instrument`, which auto-activates the
`opentelemetry-instrumentation-fastapi`, `-httpx`, `-asyncio`, `-pymongo`,
and `-logging` packages. This produces the standard HTTP semantic-convention
metrics (`http_server_duration_*`, `http_server_active_requests`) and per-route
spans without any application code change.

`OTEL_EXPORTER_OTLP_HEADERS` is **secret-bearing** when used with backends
like Datadog (`dd-api-key=...`) or Grafana Cloud. On ECS, source it from
AWS Secrets Manager. On EKS, use a `secretKeyRef`. On Compose, put it in
`.env` (which is gitignored).

### Setting 5: `OTEL_SERVICE_NAME` — trace attribution

| Surface | Where to set | Default |
|---|---|---|
| Docker Compose | Hardcoded per service in `docker-compose.yml`: `mcp-gateway-registry`, `mcp-auth-server`, `mcp-mcpgw` | per-service |
| Terraform / ECS | Set via the auto-instrumentation distro (defaults to the container name) | container-name-based |
| Helm / EKS | Set via `extraEnv` per pod, or rely on auto-instrumentation defaults | unset (becomes `unknown_service`) |

Without this set, OTel traces are tagged `unknown_service` in your tracing
backend, making it impossible to tell which container produced which span.
Set it explicitly when adding a new service.

## Where to view metrics

Each deployment surface has a different "I want to look at the metrics" UX,
because each has a different observability stack. The metrics themselves
are the same — only the viewer changes.

### Docker Compose

Everything is wired in the `docker-compose.yml` file: Prometheus container,
Grafana container, and (optional) the OTel collector for OTLP push
verification. Direct access from your laptop:

| URL | What you see |
|---|---|
| `http://localhost:9090/targets` | Prometheus targets page. All four `mcp-*` jobs (registry, auth-server, mcpgw, metrics-service for the legacy path) should show **UP**. |
| `http://localhost:9090/graph` | Prometheus query workbench. Type any of the queries from [Query cookbook](#query-cookbook) here. |
| `http://localhost:3000/` | Grafana UI. Login admin / `${GRAFANA_ADMIN_PASSWORD}` from `.env`. Add a Prometheus data source pointing at `http://prometheus:9090`. |
| `http://localhost:8889/metrics` | The OTel collector's re-exposed Prometheus surface (only meaningful when `OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317` is set in `.env`). |

For a graphical trace browser, follow the instructions in
[Adding a graphical trace UI](#adding-a-graphical-trace-ui).

To inspect raw metrics from any container without going through Prometheus:

```bash
docker compose exec prometheus wget -qO- http://registry:9464/metrics | head -30
docker compose exec prometheus wget -qO- http://auth-server:9464/metrics | head -30
docker compose exec prometheus wget -qO- http://mcpgw-server:9464/metrics | head -30
```

### AWS ECS (Terraform)

The Terraform module at `terraform/aws-ecs/` provisions a complete
observability stack when `var.enable_observability=true`:

- **Amazon Managed Prometheus (AMP) workspace** receives metrics via
  per-task ADOT sidecars (see Phase E of #1122).
- **Grafana ECS service** runs as a containerized Grafana, connected to AMP
  via the `grafana_amp_query` IAM policy. Routed by the ALB on path
  `/grafana/*`.
- **AMP query endpoint** exposed as the Terraform output `amp_endpoint`.

Direct access:

| Where | What you see |
|---|---|
| `https://<your-domain>/grafana/` | Grafana UI. Login is admin / `${GRAFANA_ADMIN_PASSWORD}` set in `terraform.tfvars`. The Prometheus data source is pre-wired to AMP via SigV4 auth. |
| AWS Console → Amazon Managed Service for Prometheus | AMP workspace with all metrics. Use the AMP-native query UI or any vendor that speaks the AMP query API. |
| `terraform output amp_endpoint` | The query endpoint URL, useful for `aws-prom-query` CLI calls or custom dashboards. |

Quick AMP query from the CLI:

```bash
AMP_ENDPOINT=$(terraform -chdir=terraform/aws-ecs output -raw amp_endpoint)
awscurl --service aps --region <region> "${AMP_ENDPOINT}api/v1/query?query=mcpgw_registry_auth_request_total"
```

To verify metrics are flowing into AMP after a deploy:

```bash
# In CloudWatch Logs, look at the ADOT sidecar log group for any service
aws logs tail /ecs/mcp-gateway-registry-adot --follow --since 5m
# Should see "metrics" entries. No errors mentioning "remote_write" or "sigv4".
```

### Kubernetes / EKS (Helm)

The Helm chart **does NOT ship Prometheus or Grafana**. EKS operators bring
their own observability stack — typically `kube-prometheus-stack` (which
gives you Prometheus + Grafana + Alertmanager) or a vendor
(Datadog, New Relic, Honeycomb, Grafana Cloud).

What the chart DOES provide:

- The OTel SDK Prometheus exporter listens on `:9464` inside each pod
  (registry, auth-server, mcpgw).
- A `NetworkPolicy` template that gates ingress to `:9464` to allow only
  pods matching `metricsScrape.networkPolicy.fromPodSelector` (default:
  `app.kubernetes.io/name: prometheus` in the `monitoring` namespace).

Operators wire one of two paths:

**Path 1: in-cluster Prometheus pulls from `:9464`.** If you're running
`kube-prometheus-stack`, add a `ServiceMonitor` resource:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: mcp-gateway-services
  namespace: monitoring
spec:
  namespaceSelector:
    matchNames: [<your-mcp-gateway-namespace>]
  selector:
    matchLabels:
      app.kubernetes.io/name: registry  # repeat for auth-server, mcpgw
  endpoints:
    - port: metrics
      path: /metrics
      interval: 10s
```

Make sure `metricsScrape.networkPolicy.fromPodSelector` in the Helm values
matches your in-cluster Prometheus pod labels.

**Path 2: OTLP push to your collector or vendor.** Set `OTEL_EXPORTER_OTLP_ENDPOINT`
via the chart's `extraEnv` block to point at your in-cluster collector
(typically an OTel collector deployed via the `opentelemetry-operator` or
the `kube-prometheus-stack`'s OTLP receiver) or directly at a vendor's
OTLP endpoint:

```yaml
# In your values.yaml override
extraEnv:
  - name: OTEL_EXPORTER_OTLP_ENDPOINT
    value: "http://otel-collector.monitoring:4317"
  - name: OTEL_EXPORTER_OTLP_PROTOCOL
    value: "grpc"
  # For vendor backends with auth headers:
  - name: OTEL_EXPORTER_OTLP_HEADERS
    valueFrom:
      secretKeyRef:
        name: my-otlp-secret
        key: headers
```

After either path, query metrics through whatever UI your stack exposes
(Grafana via kube-prometheus-stack's ingress, the vendor's web UI, etc.).

### Quick triage cheat sheet

| Question | Compose | ECS | EKS |
|---|---|---|---|
| Is the metric being emitted at all? | `docker compose exec prometheus wget -qO- http://<service>:9464/metrics` | CloudWatch Logs on the ADOT sidecar | `kubectl exec` into a pod and `curl localhost:9464/metrics` |
| Is the scrape working? | `http://localhost:9090/targets` | AMP query for the metric returns rows | `kubectl get servicemonitor -A` and Prometheus `/targets` |
| Where do I look at the data? | `http://localhost:9090/graph` or `:3000` Grafana | `https://<domain>/grafana/` | Your in-cluster Grafana / vendor UI |

## Metric inventory

The full list of metrics emitted as of 1.25.0. **Names below are the Prometheus
exposition form** (after the OTel exporter appends the unit suffix).

### Counters and gauges (Path-2 + Path-3)

| Metric | Source | Labels | What it counts |
|---|---|---|---|
| `mcpgw_registry_auth_request_total` | auth-server | `success`, `method`, `server` | Authenticated /validate calls |
| `mcpgw_registry_tool_execution_total` | auth-server | `tool_name`, `server_name`, `success`, `method`, `client_name`, `client_version` | MCP tool calls detected at the auth layer |
| `mcpgw_registry_operation_total` | registry middleware | `operation`, `resource_type`, `success` | Registry API operations (list/create/update/delete/search) |
| `tool_discovery_total` | registry middleware | `results_count_bucket` | Semantic search calls |
| `health_check_total` | registry | `endpoint`, `status_code`, `healthy` | Health check probe count |
| `mcpgw_registry_tool_invocations_total` | mcpgw | `tool`, `success` | FastMCP tool invocations |
| `mcpgw_registry_nginx_config_writes_total` | registry | `status` | Nginx config file writes by outcome |
| `mcpgw_registry_nginx_updates_skipped_total` | registry | `operation` | Nginx updates skipped due to mode |
| `mcpgw_registry_mode_blocked_requests_total` | registry | `path_category`, `mode` | Requests blocked by registry mode |
| `peer_sync_failures_total` | registry | `peer_id`, `failure_type` | Federation peer sync failures |
| `app_log_mongodb_flush_failures_total` | registry | `service` | MongoDB log handler failures |
| `telemetry_sends_total` | registry | `event`, `status` | Telemetry events sent |
| `m2m_orphan_cleanups_total` | registry | `idp_had_record` | M2M orphan cleanup deletions |
| `mcpgw_registry_cloud_detection_total` | registry | `cloud`, `method` | Cloud-detection outcomes |
| `mcpgw_registry_config_view_requests_total` | registry | `user_type` | Configuration view requests |
| `mcpgw_registry_config_export_requests_total` | registry | `format`, `includes_sensitive` | Configuration export requests |
| `mcpgw_registry_logout_id_token_hint_present_total` | registry | — | Logouts with id_token hint present |
| `mcpgw_registry_logout_id_token_hint_missing_total` | registry | — | Logouts without id_token hint |
| `mcpgw_registry_logout_jwt_validation_failed_total` | registry | — | Logout JWT validation failures |
| `mcpgw_registry_logout_url_length_warning_total` | registry | — | Logout URLs over recommended length |
| `mcpgw_registry_session_store_resolve_total` | registry | `result` | Session store lookups |
| `m2m_management_requests_total` | registry | `operation`, `outcome` | Direct M2M client API calls |
| `mcpgw_registry_metrics_emission_path_total` | registry, auth-server, mcpgw | `path` (`otel`/`legacy`) | Migration self-observability |
| `mcpgw_registry_deployment_mode_info` (Gauge) | registry | `deployment_mode`, `registry_mode` | Current deployment mode (always 1, observed each cycle) |

### Histograms

| Metric | Source | Labels | What it measures |
|---|---|---|---|
| `mcpgw_registry_auth_request_duration_milliseconds` (`_count`, `_sum`, `_bucket`) | auth-server | `success`, `method`, `server` | Auth /validate latency |
| `tool_execution_duration_milliseconds` | auth-server | same as `mcpgw_registry_tool_execution_total` | Tool call latency at auth layer |
| `mcpgw_registry_protocol_latency_milliseconds` | auth-server | `flow_step`, `server_name` | Time between MCP protocol stages (init → tools/list, etc.) |
| `mcpgw_registry_operation_duration_milliseconds` | registry middleware | same as `mcpgw_registry_operation_total` | Registry API operation latency |
| `mcpgw_registry_tool_discovery_duration_milliseconds` | registry middleware | same as `tool_discovery_total` | Semantic search latency |
| `peer_sync_duration_seconds` | registry | `peer_id`, `success` | Peer sync operation duration |
| `mcpgw_registry_tool_duration_milliseconds` | mcpgw | `tool`, `success` | Per-tool invocation latency |

### HTTP auto-instrumentation (when OTel auto-instrument is active)

| Metric | Source | Labels | What it measures |
|---|---|---|---|
| `http_server_duration_milliseconds` | every service | `http_method`, `http_target`, `http_status_code`, `http_scheme`, `http_host`, `http_server_name`, `net_host_port`, `http_flavor` | Per-route HTTP request latency |
| `http_server_active_requests` | every service | same | In-flight requests right now |

> Note on `http_target`: this is the **raw URL path** (e.g.
> `/api/servers/airegistry-tools/rating`), not the FastAPI route template
> (`/api/servers/{path:path}/rating`). For paths with high-cardinality IDs
> this can produce many time series; in production with large catalogs you
> may want to add a label-relabel rule in Prometheus to collapse them.

## Query cookbook

Open `http://localhost:9090/graph` (Compose) or your AMP/Grafana UI and try
these. Most are useful in **Graph view** with the time window dropped to 5
minutes.

### Semantic search endpoint

| Goal | Query |
|---|---|
| Calls per second | `sum by (http_status_code)(rate(http_server_duration_milliseconds_count{http_target="/api/search/semantic"}[5m]))` |
| p95 latency | `histogram_quantile(0.95, sum by (le)(rate(http_server_duration_milliseconds_bucket{http_target="/api/search/semantic"}[5m])))` |
| Average latency | `rate(http_server_duration_milliseconds_sum{http_target="/api/search/semantic"}[5m]) / rate(http_server_duration_milliseconds_count{http_target="/api/search/semantic"}[5m])` |
| Application-level view (results-bucket dimension) | `sum by (results_count_bucket)(rate(tool_discovery_total[5m]))` |
| Search latency from middleware (alternative source) | `histogram_quantile(0.95, sum by (le)(rate(mcpgw_registry_tool_discovery_duration_milliseconds_bucket[5m])))` |

### Mcpgw — per-tool stats

| Goal | Query |
|---|---|
| Total invocations per tool | `sum by (tool)(mcpgw_registry_tool_invocations_total)` |
| Tool QPS | `sum by (tool)(rate(mcpgw_registry_tool_invocations_total[5m]))` |
| Per-tool error rate | `sum by (tool)(rate(mcpgw_registry_tool_invocations_total{success="False"}[5m])) / sum by (tool)(rate(mcpgw_registry_tool_invocations_total[5m]))` |
| Most-called tool right now | `topk(3, sum by (tool)(rate(mcpgw_registry_tool_invocations_total[5m])))` |
| p95 latency per tool | `histogram_quantile(0.95, sum by (le, tool)(rate(mcpgw_registry_tool_duration_milliseconds_bucket[5m])))` |
| Average duration per tool | `sum by (tool)(rate(mcpgw_registry_tool_duration_milliseconds_sum[5m])) / sum by (tool)(rate(mcpgw_registry_tool_duration_milliseconds_count[5m]))` |
| Slowest tool right now | `topk(1, histogram_quantile(0.95, sum by (le, tool)(rate(mcpgw_registry_tool_duration_milliseconds_bucket[5m]))))` |

### Any API endpoint — invocations + success/failure

Replace `<TARGET>` with the path you care about (e.g. `/api/servers`,
`/api/agents`, `/api/skills`, `/validate`, `/api/auth/login`).

| Goal | Query |
|---|---|
| List all routes that have ever been hit | `group by (http_target, http_method, job)(http_server_duration_milliseconds_count)` |
| Calls per second on `<TARGET>` | `sum by (http_status_code)(rate(http_server_duration_milliseconds_count{http_target="<TARGET>"}[5m]))` |
| Success/failure split | `sum by (http_status_code)(rate(http_server_duration_milliseconds_count{http_target="<TARGET>"}[5m]))` |
| Error rate (4xx/5xx as fraction of total) | `sum(rate(http_server_duration_milliseconds_count{http_target="<TARGET>",http_status_code=~"4..|5.."}[5m])) / sum(rate(http_server_duration_milliseconds_count{http_target="<TARGET>"}[5m]))` |
| p50 / p95 / p99 latency on `<TARGET>` | `histogram_quantile(0.95, sum by (le)(rate(http_server_duration_milliseconds_bucket{http_target="<TARGET>"}[5m])))` |
| Top 5 most-called routes | `topk(5, sum by (http_target, job)(rate(http_server_duration_milliseconds_count[5m])))` |
| Top 5 slowest routes (p95) | `topk(5, histogram_quantile(0.95, sum by (le, http_target)(rate(http_server_duration_milliseconds_bucket[5m]))))` |
| In-flight requests right now | `http_server_active_requests` |
| Request rate per service | `sum by (job)(rate(http_server_duration_milliseconds_count[5m]))` |

### Auth, sessions, federation

| Goal | Query |
|---|---|
| Auth requests per second by outcome | `sum by (success)(rate(mcpgw_registry_auth_request_total[5m]))` |
| Auth p95 latency | `histogram_quantile(0.95, sum by (le)(rate(mcpgw_registry_auth_request_duration_milliseconds_bucket[5m])))` |
| Session-store hit rate | `sum(rate(mcpgw_registry_session_store_resolve_total{result="hit"}[5m])) / sum(rate(mcpgw_registry_session_store_resolve_total[5m]))` |
| Federation peer sync failures by type | `sum by (peer_id, failure_type)(rate(peer_sync_failures_total[5m]))` |
| Logout JWT validation failure rate | `rate(mcpgw_registry_logout_jwt_validation_failed_total[5m])` |

### Registry health and operations

| Goal | Query |
|---|---|
| Registry API operations per second by type | `sum by (operation, resource_type)(rate(mcpgw_registry_operation_total[5m]))` |
| Operations p95 latency | `histogram_quantile(0.95, sum by (le, operation)(rate(mcpgw_registry_operation_duration_milliseconds_bucket[5m])))` |
| Nginx config write outcomes | `sum by (status)(rate(mcpgw_registry_nginx_config_writes_total[5m]))` |
| M2M orphan cleanups | `sum by (idp_had_record)(rate(m2m_orphan_cleanups_total[5m]))` |
| Cloud detection method distribution | `sum by (cloud, method)(mcpgw_registry_cloud_detection_total)` |
| Telemetry pings success rate | `sum(rate(telemetry_sends_total{status="success"}[5m])) / sum(rate(telemetry_sends_total[5m]))` |

## Verifying the migration is working

Three checks operators can run after upgrading from 1.24.x to 1.25.0:

**1. All Prometheus targets UP**

```
http://localhost:9090/targets
```

You should see `mcp-registry`, `mcp-auth-server`, `mcp-mcpgw`, and (until
1.26.0) `mcp-metrics-service` in the targets list, all with state `UP`.

**2. Migration self-observability**

```
mcpgw_registry_metrics_emission_path_total
```

Should show `path="otel"` rows incrementing on every request.
`path="legacy"` should be empty (or zero) when `METRICS_LEGACY_HTTP_POST=false`.
If both are incrementing, you have the dual-write transition flag enabled.

**3. Previously-invisible counters are now visible**

```
mcpgw_registry_nginx_config_writes_total
peer_sync_failures_total
m2m_orphan_cleanups_total
```

These were `prometheus_client.Counter` instances declared in the registry
process for releases but never exposed anywhere. If they return rows now,
the migration to OTel-native emission worked.

## Troubleshooting

### A Prometheus target shows DOWN with "connection refused"

The OTel SDK Prometheus exporter starts during application startup, after
the SDK initializes. Containers go `Up` before this completes. Wait one or
two scrape intervals (10-20 seconds) and re-check. If still DOWN:

```bash
docker compose exec <service> ss -tlnp | grep 9464
```

If that shows nothing, the Prometheus exporter never bound. Most common
cause: `OTEL_EXPORTER_PROMETHEUS_HOST` is unset, so the bootstrap helper
in the meter module took the no-op path. Set it in `.env` and recreate the
container.

### A query returns "Empty query result"

In order:

1. Wait one minute. Counters appear after first emission. Histograms appear
   after first observation. Rate functions need at least 2 data points in
   the rate window.
2. Try the simpler form: drop labels and aggregations, just type the metric
   name. If that returns rows, your filter is wrong (most often: a label
   typo).
3. Check the raw exposition: `docker compose exec prometheus wget -qO- http://<service>:9464/metrics 2>/dev/null | grep <metric>`.
   If the metric is there, Prometheus's scrape didn't pick it up yet
   (10-second scrape interval). If it's not there, the application code
   didn't emit it.

### `*_milliseconds_*` metric names look weird

That's standard. The OTel-to-Prometheus exporter appends the OTel `unit=`
annotation to histogram metric names. So a Histogram declared with
`unit="ms"` exports as `<name>_milliseconds`, regardless of what we named
the OTel instrument. The naming follows the OTel spec.

### I want to inspect what's actually being scraped without going through Prometheus

```bash
docker compose exec <service> curl -s http://localhost:9464/metrics
```

Or from any other container on the Docker network:

```bash
docker compose exec prometheus wget -qO- http://<service>:9464/metrics
```

### I want metrics flowing to AMP / Datadog / Honeycomb instead of (or in addition to) Prometheus

Set `OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_EXPORTER_OTLP_HEADERS` in `.env`.
The OTel SDK will then push every metric over OTLP in addition to serving
the Prometheus exporter on `:9464`. On ECS, the AMP push is wired
automatically via the per-task ADOT sidecar (Phase E of #1122).

### How do I disable OTel emission entirely

Don't set `OTEL_EXPORTER_PROMETHEUS_HOST`, don't set `OTEL_EXPORTER_OTLP_ENDPOINT`.
The bootstrap helpers will detect the unset state and leave the SDK in
NoOp mode. Every `Counter.add()` and `Histogram.record()` call across the
codebase becomes a zero-cost no-op.

## Adding a graphical trace UI

The default `docker-compose.yml` ships an `otel-collector` container that
receives traces from registry/auth-server/mcpgw and logs them to stdout
via the debug exporter (good for verification, awkward for browsing). If
you want a visual trace browser locally, drop in a Jaeger or Tempo
container and route the collector's traces pipeline to it. The eight
lines below give you Jaeger at `http://localhost:16686/`.

**Step 1**: add a `jaeger` service to your `docker-compose.yml`
(e.g., before the `prometheus` service):

```yaml
  jaeger:
    image: jaegertracing/all-in-one:latest
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    ports:
      - "16686:16686" # Jaeger UI
    restart: unless-stopped
```

**Step 2**: in `config/otel/collector.yaml`, add a Jaeger exporter and
include it in the traces pipeline:

```yaml
exporters:
  # ... existing exporters ...
  otlp/jaeger:
    endpoint: jaeger:4317
    tls:
      insecure: true

service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [debug, otlp/jaeger]   # add otlp/jaeger here
```

**Step 3**: bring it up:

```
docker compose up -d jaeger
docker compose restart otel-collector
```

Open `http://localhost:16686/`, pick a service from the dropdown
(`mcp-gateway-registry`, `mcp-auth-server`, `mcp-mcpgw`), click
**Find Traces**. Each trace shows the full waterfall: HTTP span at the
top, child spans for downstream calls (MongoDB queries, httpx requests
to the registry, FastMCP tool dispatch, etc.).

The same pattern works with Tempo, Zipkin, or any OTLP-receiving
backend — swap the exporter type. We don't ship Jaeger by default to
keep the Compose footprint minimal and to avoid pulling an extra image
in restricted environments.

## Related docs

- [docs/metrics-architecture.md](metrics-architecture.md) — design-level
  diagrams (component view, sequence diagrams)
- [docs/unified-parameter-reference.md](unified-parameter-reference.md) —
  cross-surface env var mapping
- [docs/OBSERVABILITY-LEGACY.md](OBSERVABILITY-LEGACY.md) — pre-1.25.0
  architecture, retained for the transition window
