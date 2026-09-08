# MCP Gateway Observability Guide

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
| `mcpgw_registry_auth_request_total` | auth-server | `success`, `method`, `server`, `target_kind` | Authenticated /validate calls. `target_kind` = `a2a_agent` \| `virtual_mcp_server` \| `mcp_server` \| `generic_proxy_skill` \| `generic_proxy_agent` \| `generic_proxy_custom` \| `control_plane` \| `unknown` (routing breakdown; `control_plane` = `/api/*`, static, oauth2 — never counted as a data-plane target). For a gateway-proxied request `server` holds the entity's **authz key** (`skill/skills/pdf`, `rest-endpoint/rest-endpoint/<uuid>`), which is the exact string a `server_access` rule names, so a `success="false"` series points at the rule to write |
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
| `mcpgw_rate_limit_checks_total` | auth-server (`/validate`) | `axis` (`clr`/`tgt`/`ctg`), `entity_type`, `window_seconds`, `outcome` (`allow`/`deny`) | Every rate-limit gate evaluation (denominator for a throttle rate) |
| `mcpgw_rate_limit_throttled_total` | auth-server (`/validate`) | `axis` (`clr`/`tgt`/`ctg`), `entity_type`, `window_seconds` | Times a gate denied a request. `ctg` = the per-caller-per-target axis; `clr`/`tgt` are the caller/target axes |
| `mcpgw_rate_limit_quarantine_denied_total` | auth-server (`/validate`) | `scope` (`caller`/`target`), `entity_type` | Requests dropped because a caller or target is **quarantined** (kill switch) |
| `mcpgw_rate_limit_quarantine_members` (Gauge) | registry | `group` (`quarantine-callers`/`quarantine-targets`) | Current member count of each kill-switch group, observed each export cycle (correct across replicas) |
| `mcpgw_rate_limit_errors_total` | auth-server (`/validate`) | `axis` (incl. `qtn` for a quarantine-membership read error) | Rate-limit backend errors (counter-store unreachable/timeout → fail-open/closed) |
| `mcpgw_registry_generic_proxy_request_total` | auth-server (gateway hop) | `entity_type` (`skill`/`a2a_agent`/`custom`), `outcome` (13 values, below) | **What the caller actually got.** One record per request that enters the hop handler, on both the buffered and streaming paths. `auth_request_total` cannot answer this — its `success` label is the `/validate` decision, so a 502 credential-vend failure reads there as a success. 3 × 13 = 39 series, flat whatever the endpoint count (no label carries entity identity); every combination exists at zero from startup. Does **not** cover the 401 token gate, which rejects in a route dependency before the handler runs |
| `mcpgw_registry_generic_proxy_slot_rejected_total` | auth-server (gateway hop) | `pool` (`buffered`/`stream`) | Gateway-proxy requests rejected with 503 because a concurrency slot could not be acquired inside the acquire timeout. A non-zero rate means the pool is saturated: raise `GATEWAY_GENERIC_STREAM_MAX_CONCURRENCY` or shed load. Both `pool` values exist at zero from startup |
| `mcpgw_registry_generic_proxy_stream_outcome_total` | auth-server (gateway hop) | `outcome` (`started`/`completed`/`duration_timeout`/`byte_cap`/`upstream_error`/`client_closed`) | Streaming gateway-proxy lifecycle. In-flight streams are `started` minus the sum of the five terminals. All six values exist at zero from startup |
| `mcpgw_registry_gateway_generic_blocks_dropped_total` | registry | `reason` (`invalid`/`collision`) | Gateway routes the nginx render path refused. Non-zero means an entity is registered and unreachable |
| `mcpgw_registry_gateway_egress_policy_unverified` (Gauge) | registry | none | `1` means the startup self-check reached cloud metadata, so the generic proxy latched **off** for the process. A standing `1` on an enabled deployment is an alert |

#### The 13 hop outcomes, and why the set is not just "status code"

`mcpgw_registry_generic_proxy_request_total{outcome}` exists because HTTP statuses collapse failures an operator must tell apart. Two 503s and three 502s leave the hop for entirely different reasons:

| Outcome | Status the caller saw | What happened |
|---|---|---|
| `ok` | 2xx/3xx | Forwarded and returned, or a stream that completed |
| `upstream_4xx` | 4xx from the backend | The hop worked; the backend refused. A 404 for a path that does not exist upstream lands here |
| `upstream_5xx` | 5xx from the backend | The hop worked; the backend broke |
| `upstream_error` | 502 / 504 | No usable response: connection refused, reset, or timed out |
| `egress_blocked` | 502 | **Security event.** The guarded transport refused the outbound because the pinned host resolved to a private, metadata, or rebound IP |
| `auth_unavailable` | 502 | The upstream-credential vend failed, so the hop refused to forward unauthenticated |
| `capacity` | 503 | A concurrency slot could not be acquired inside the acquire timeout (either pool) |
| `disabled` | 503 | The feature latch is off — flag disabled, or the egress self-check failed at startup |
| `rejected` | 400 | Sub-path confinement or host-pin refusal: the SSRF guards saying no |
| `byte_cap` | 413 | The response exceeded the buffered or stream byte ceiling |
| `client_closed` | — | The caller hung up (buffered `CancelledError`, or a stream abandoned mid-body) |
| `duration_timeout` | 504 / aborted stream | The absolute stream-duration ceiling fired |
| `internal_error` | 500 | A bug. If this moves, read the logs — nothing else records here |

Collapse those and you get paged for the wrong thing: `disabled` would fire a saturation alert, and `egress_blocked` — an SSRF refusal — would fire a credential-outage alert.

| Goal | Query |
|---|---|
| Hop failure rate per entity type (the headline SLI) | `sum by (entity_type)(rate(mcpgw_registry_generic_proxy_request_total{outcome!~"ok\|upstream_4xx"}[5m])) / sum by (entity_type)(rate(mcpgw_registry_generic_proxy_request_total[5m]))` |
| What is failing, ranked | `topk(10, sum by (outcome)(rate(mcpgw_registry_generic_proxy_request_total{outcome!="ok"}[15m])))` |
| **Alert:** SSRF egress refusals (would have read as a credential outage before) | `sum by (entity_type)(rate(mcpgw_registry_generic_proxy_request_total{outcome="egress_blocked"}[15m])) > 0` |
| **Alert:** credential vend failing | `sum(rate(mcpgw_registry_generic_proxy_request_total{outcome="auth_unavailable"}[15m])) > 0` |
| **Alert:** a bug, not a backend fault | `sum(rate(mcpgw_registry_generic_proxy_request_total{outcome="internal_error"}[15m])) > 0` |
| Feature switched off while callers still arrive | `sum(rate(mcpgw_registry_generic_proxy_request_total{outcome="disabled"}[5m])) > 0` |
| The disagreement this counter exists for: `/validate` says success, the hop failed | `sum(rate(mcpgw_registry_generic_proxy_request_total{outcome=~"auth_unavailable\|upstream_error\|egress_blocked\|capacity"}[15m])) > 0 and sum(rate(mcpgw_registry_auth_request_total{target_kind=~"generic_proxy_.*", success="true"}[15m])) > 0` |

Both counters are correct at once: `/validate` did succeed, and the request still failed at the hop. That is the point — `auth_request_total` reports the authorization decision, this counter reports the caller's outcome.

### Histograms

| Metric | Source | Labels | What it measures |
|---|---|---|---|
| `mcpgw_registry_auth_request_duration_milliseconds` (`_count`, `_sum`, `_bucket`) | auth-server | `success`, `method`, `target_kind` | Auth /validate latency. Carries **no `server` label**: the histogram holds 16 `le` buckets plus `_count` and `_sum`, so a per-target label would cost 18 series per combination where the counter costs 1, and `/validate` does the same work for every target — a token check and a scope lookup. Group by `target_kind`. Per-endpoint hop latency is a separate metric |
| `tool_execution_duration_milliseconds` | auth-server | same as `mcpgw_registry_tool_execution_total` | Tool call latency at auth layer |
| `mcpgw_registry_protocol_latency_milliseconds` | auth-server | `flow_step`, `server_name` | Time between MCP protocol stages (init → tools/list, etc.) |
| `mcpgw_registry_operation_duration_milliseconds` | registry middleware | same as `mcpgw_registry_operation_total` | Registry API operation latency |
| `mcpgw_registry_tool_discovery_duration_milliseconds` | registry middleware | same as `tool_discovery_total` | Semantic search latency |
| `peer_sync_duration_seconds` | registry | `peer_id`, `success` | Peer sync operation duration |
| `mcpgw_registry_tool_duration_milliseconds` | mcpgw | `tool`, `success` | Per-tool invocation latency |
| `mcpgw_rate_limit_backend_duration_milliseconds` (`_count`, `_sum`, `_bucket`) | auth-server (`/validate`) | `backend` (`documentdb`), `op` | Counter-store round-trip latency per rate-limit op (the hop bounded by `RATE_LIMIT_BACKEND_TIMEOUT_MS`, default 250 ms) |

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
| Per-tool error rate | `sum by (tool)(rate(mcpgw_registry_tool_invocations_total{success="false"}[5m])) / sum by (tool)(rate(mcpgw_registry_tool_invocations_total[5m]))` |
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

#### Target-type routing (`target_kind`)

`mcpgw_registry_auth_request_total` carries a `target_kind` label so you can see how `/validate` traffic splits across routed targets: `a2a_agent`, `virtual_mcp_server`, `mcp_server`, the three gateway-proxy kinds (`generic_proxy_skill`, `generic_proxy_agent`, `generic_proxy_custom`), `control_plane` (the `/api/*` / static / oauth2 control plane, never a data-plane target), and `unknown`. Chart these as a Grafana **Time series** panel with legend `{{target_kind}}`.

The three `generic_proxy_*` kinds come from the `X-Generic-Proxy-Kind` marker nginx sets on each generated gateway location, so they hold for any `GATEWAY_PROXY_PREFIX`. Every operator-defined custom type collapses to `generic_proxy_custom`, which keeps the label set fixed at three values however many custom types exist. A gateway request landing in `unknown` means the marker did not arrive, so check the rendered nginx location.

**The `success` label is lowercase `true` / `false` on every exporter.** It used to be `True` / `False` on the native OTel path (auth-server and registry emitted `str(bool)`, a Python repr) while the standalone metrics-service normalized the same booleans to lowercase. A query written for one exporter matched nothing on the other, and Prometheus reports a label-value miss as an **empty result rather than an error** — so `success="false"` read as "no failures" when it really meant "no such label value". Both paths now emit lowercase, which is the Prometheus/OpenTelemetry convention.

| Exporter | `success` values |
|---|---|
| auth-server `:9464` (native OTel) | `true` / `false` |
| registry `:9464` (native OTel) | `true` / `false` |
| metrics-service (legacy POST path) | `true` / `false` |

**Upgrading:** any saved query, alert rule, or panel filtering `success="True"` / `success="False"` must be lowercased. Series carrying the old capitalized values stay in the TSDB until retention expires, so a `sum(...)` over a window that straddles the upgrade needs `success=~"[Tt]rue"` to cover both. The shipped Grafana dashboards are already updated.

| Goal | Query |
|---|---|
| Routing split — one line per target kind (main timeseries) | `sum by (target_kind)(rate(mcpgw_registry_auth_request_total[5m]))` |
| Data-plane only (drop control-plane / dashboard noise) | `sum by (target_kind)(rate(mcpgw_registry_auth_request_total{target_kind!="control_plane"}[5m]))` |
| A2A agent routing only | `sum(rate(mcpgw_registry_auth_request_total{target_kind="a2a_agent"}[5m]))` |
| Routing split by success/failure | `sum by (target_kind, success)(rate(mcpgw_registry_auth_request_total[5m]))` |
| Share of total per target kind (stacked %) | `sum by (target_kind)(rate(mcpgw_registry_auth_request_total[5m])) / ignoring(target_kind) group_left sum(rate(mcpgw_registry_auth_request_total[5m]))` |
| p95 auth latency per target kind | `histogram_quantile(0.95, sum by (le, target_kind)(rate(mcpgw_registry_auth_request_duration_milliseconds_bucket[5m])))` |
| Cumulative counts (raw growth, not rate) | `sum by (target_kind)(mcpgw_registry_auth_request_total)` |
| Gateway-proxy volume, all three kinds | `sum by (target_kind)(rate(mcpgw_registry_auth_request_total{target_kind=~"generic_proxy_.*"}[5m]))` |
| Skill routing only | `sum(rate(mcpgw_registry_auth_request_total{target_kind="generic_proxy_skill"}[5m]))` |
| Busiest proxied endpoints by authz key | `topk(10, sum by (server)(rate(mcpgw_registry_auth_request_total{target_kind=~"generic_proxy_.*"}[1h])))` |
| Which proxied endpoint is being denied (the `server` value is the scope rule to write) | `sum by (server)(rate(mcpgw_registry_auth_request_total{target_kind=~"generic_proxy_.*", success="false"}[15m])) > 0` |
| Gateway requests that failed to classify (should stay flat) | `sum(rate(mcpgw_registry_auth_request_total{target_kind="unknown"}[5m]))` |
| Per-server rate across **all** routed targets (MCP, virtual, agent, gateway in one panel) | `topk(20, sum by (server, target_kind)(rate(mcpgw_registry_auth_request_total{target_kind!="control_plane"}[6h])))` |
| Per-server denial rate, any target type | `sum by (server)(rate(mcpgw_registry_auth_request_total{target_kind!="control_plane", success="false"}[6h])) > 0` |
| Which tool ran on which MCP server | `sum by (server_name, tool_name)(increase(mcpgw_registry_tool_execution_total{method="tools/call"}[6h]))` |
| Average MCP flow latency per server | `sum by (server_name)(rate(mcpgw_registry_protocol_latency_milliseconds_sum[6h])) / sum by (server_name)(rate(mcpgw_registry_protocol_latency_milliseconds_count[6h]))` |
| Label-cardinality headroom (compare against `METRICS_MAX_LABEL_CARDINALITY`, default 150) | `count(count by (server)(mcpgw_registry_auth_request_total))` |
| **Alert:** a bounded label has hit the cap and is collapsing values | `sum(mcpgw_registry_auth_request_total{server="_other"}) or sum(mcpgw_registry_tool_execution_total{server_name="_other"}) or sum(mcpgw_registry_tool_execution_total{tool_name="_other"})` |
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

### Rate limiting

Application-level rate limiting runs in the auth-server `/validate` hop. The metrics carry only low-cardinality labels (`axis`, `entity_type`, `window_seconds`, `scope`, `group`) — deliberately **not** the caller's username/client_id, which would be unbounded cardinality. To attribute a throttle or a quarantine deny to a specific user or client, use the app logs instead (see below).

| Goal | Query |
|---|---|
| Throttles per second by axis / entity type | `sum by (axis, entity_type)(rate(mcpgw_rate_limit_throttled_total[5m]))` |
| Per-caller-per-target throttles only (the `ctg` axis) | `sum by (entity_type)(rate(mcpgw_rate_limit_throttled_total{axis="ctg"}[5m]))` |
| Throttle rate (fraction of checks denied) | `sum(rate(mcpgw_rate_limit_throttled_total[5m])) / sum(rate(mcpgw_rate_limit_checks_total[5m]))` |
| Quarantine denies per second (caller vs target) | `sum by (scope)(rate(mcpgw_rate_limit_quarantine_denied_total[5m]))` |
| Currently quarantined callers / targets | `mcpgw_rate_limit_quarantine_members` |
| Backend-error rate (fail-open/closed events; `qtn` = quarantine-read error) | `sum by (axis)(rate(mcpgw_rate_limit_errors_total[5m]))` |
| Counter-store p95 latency | `histogram_quantile(0.95, sum by (le)(rate(mcpgw_rate_limit_backend_duration_milliseconds_bucket[5m])))` |
| Counter-store calls exceeding the 250 ms budget | `sum(rate(mcpgw_rate_limit_backend_duration_milliseconds_bucket{le="250"}[5m])) / sum(rate(mcpgw_rate_limit_backend_duration_milliseconds_count[5m]))` (fraction **within** budget; alert when it drops) |

**Quarantine attribution (app logs).** A quarantine deny logs a structured `WARNING` line: `rate-limit quarantine deny: scope=caller entity_type=group caller_username=alice caller_client_id=`. Search the auth-server app logs for `rate-limit quarantine deny` and filter on `caller_username=` / `caller_client_id=`. The metric labels stay bounded (scope only), so a nonzero `mcpgw_rate_limit_quarantine_denied_total` is your signal that quarantine is actively dropping traffic.

**Attributing a throttle to a user / client_id (app logs, not metrics).** On every denial the limiter logs a structured `WARNING` line carrying the validated-token identity, e.g.:

```
rate-limit throttled: axis=clr entity_type=group name=alice limit=5/60s caller_type=user caller_username=alice caller_client_id=
```

`caller_type` is `user` or `agent` (agent = a client_id was present); exactly one of `caller_username` / `caller_client_id` is populated. Search the auth-server application logs (MongoDB `application_logs` collection, or your log backend) for `rate-limit throttled` and filter on `caller_username=` / `caller_client_id=`.

#### Recommended alerts

These are the two signals worth alarming on. Both indicate the limiter is degrading, not that a caller is merely hitting a limit (throttles themselves are expected and are not an alert condition).

| Alert | Condition (PromQL) | Why it matters |
|---|---|---|
| **Rate-limit backend errors** | `sum(rate(mcpgw_rate_limit_errors_total[5m])) > 0` for 5m | The DocumentDB counter store is unreachable or timing out. With the default `RATE_LIMIT_FAIL_OPEN=true` this means limits are **not being enforced** (requests fail open); with a fail-closed definition it means those callers are being **denied**. Either way the limiter is not doing its job. |
| **Counter-store latency near budget** | `histogram_quantile(0.95, sum by (le)(rate(mcpgw_rate_limit_backend_duration_milliseconds_bucket[5m]))) > 200` for 10m | Each `/validate` waits on this hop up to `RATE_LIMIT_BACKEND_TIMEOUT_MS` (default **250 ms**). A p95 creeping toward 250 ms means throttle checks are about to start timing out (→ `mcpgw_rate_limit_errors_total` fail-open) and are adding latency to every gated request. Investigate DocumentDB health / connection pool before it crosses the timeout. |

Tune the `200` threshold relative to your configured `RATE_LIMIT_BACKEND_TIMEOUT_MS`: alert at roughly 80% of the timeout so you have headroom before ops start failing open.

### Verifying gateway-proxy metrics end to end

#### Three PromQL assertions to run after a deploy

These answer "did the routing labels ship correctly?" from Grafana Explore, without an exec into the container. Run them as **Instant** queries in **Table** format.

```promql
# 1. Every gateway request carries an entity kind and an authz key. Expect one row per
#    proxied entity that has taken traffic; `server` must read like "skill/skills/pdf" or
#    "rest-endpoint/rest-endpoint/<uuid>", never the bare literal "gateway".
sum by (target_kind, server, success) (mcpgw_registry_auth_request_total{target_kind=~"generic_proxy_.*"})

# 2. Both hop counters expose every label value, including the ones nothing has triggered.
#    Expect 8 rows on a fresh deployment, the untriggered ones at 0 — that is what lets a
#    rate() alert bind at deploy instead of at first failure.
sum by (outcome)(mcpgw_registry_generic_proxy_stream_outcome_total) or sum by (pool)(mcpgw_registry_generic_proxy_slot_rejected_total)

# 3. The latency histogram must carry NO `server` label. This MUST return "No data".
#    The contrast query returns a count > 0, proving the label lives on the counter only.
count(mcpgw_registry_auth_request_duration_milliseconds_count{server!=""})
count(mcpgw_registry_auth_request_total{server!=""})
```

A ratio query that divides by a per-server rate returns `NaN` for any server with no traffic in the window (`0/0`). Prefer the `> 0` forms above, or append `and on(server) sum by (server)(rate(mcpgw_registry_auth_request_total[$__range])) > 0`.

**Read a quantile together with its sample count.** `histogram_quantile` over a handful of observations reports a bucket boundary, not a measurement: a series with 5 samples once read `429 ms` at p95 while every sample actually sat in the 25–50 ms bucket. Pair any `histogram_quantile(...)` panel with `sum by (target_kind)(increase(mcpgw_registry_auth_request_duration_milliseconds_count[$__range]))`, and prefer `rate(_sum)/rate(_count)` when traffic is thin — the average is immune to bucket granularity. Measured with adequate load on ECS, `/validate` runs ≈3 ms for control-plane requests and ≈21 ms for gateway-proxied ones (p95 9 ms vs 54 ms); the gateway hop additionally mints the generic-audience token and runs the CSRF gate, so a several-fold ratio there is expected, not a fault.

**On ECS these PromQL assertions are the only way in — there is no `:9464` to curl.** The task runs under `opentelemetry-instrument`, which installs the SDK meter provider before application code runs, so `_init_meter_provider_if_needed` returns early and never starts the Prometheus HTTP server. `OTEL_EXPORTER_PROMETHEUS_HOST/PORT` are set on the task but nothing listens on that port; verified by exec'ing into a running `mcp-gateway-v2-auth` task, where `127.0.0.1:9464` gives `Connection refused`. Metrics leave the task through the **`adot-collector` sidecar** instead (`OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317`, gRPC) and land in Amazon Managed Prometheus, where the same queries work. Query AMP directly with a SigV4-signed request when you have no Grafana:

```bash
# AMP needs SigV4, so sign the request; POST avoids query-string canonicalization pitfalls
python3 - <<'PY'
import boto3, json, urllib.parse, urllib.request
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
AMP = "https://aps-workspaces.<region>.amazonaws.com/workspaces/<ws-id>"
creds = boto3.Session().get_credentials().get_frozen_credentials()
body = urllib.parse.urlencode({"query": 'count by (success)(mcpgw_registry_auth_request_total)'}).encode()
req = AWSRequest(method="POST", url=f"{AMP}/api/v1/query", data=body,
                 headers={"Content-Type": "application/x-www-form-urlencoded"})
SigV4Auth(creds, "aps", "<region>").add_auth(req)
r = urllib.request.Request(req.url, data=body, headers=dict(req.headers), method="POST")
print(json.load(urllib.request.urlopen(r))["data"]["result"])
PY
```

On **docker compose**, where the SDK is bootstrapped by the application itself, the exporter does listen and reading it straight from the container avoids the 10s scrape delay:

```bash
export AUTH=mcp-gateway-registry-auth-server-1     # compose only; on ECS use the AMP query above

counters() {
  docker exec $AUTH sh -c 'curl -s localhost:9464/metrics' \
    | grep -E '^mcpgw_registry_(auth_request_total|generic_proxy_)' \
    | grep -v duration \
    | sed 's/otel_scope_name="mcp-auth-server",//; s/otel_scope_schema_url="",//; s/otel_scope_version="",//' \
    | sort
}
counters
```

On a stack that has just started, before any gateway traffic, 47 series already exist at zero — 39 hop outcomes plus the 8 lifecycle values below. That is the point of zero-initialization: a `rate()` alert can bind at deploy instead of waiting for the first failure.

```
mcpgw_registry_generic_proxy_slot_rejected_total{pool="buffered"} 0.0
mcpgw_registry_generic_proxy_slot_rejected_total{pool="stream"} 0.0
mcpgw_registry_generic_proxy_stream_outcome_total{outcome="started"} 0.0
mcpgw_registry_generic_proxy_stream_outcome_total{outcome="completed"} 0.0
mcpgw_registry_generic_proxy_stream_outcome_total{outcome="duration_timeout"} 0.0
mcpgw_registry_generic_proxy_stream_outcome_total{outcome="byte_cap"} 0.0
mcpgw_registry_generic_proxy_stream_outcome_total{outcome="upstream_error"} 0.0
mcpgw_registry_generic_proxy_stream_outcome_total{outcome="client_closed"} 0.0
```

Confirm the startup log said so:

```bash
docker compose logs auth-server | grep zero-init
# zero-init seeded 47/47 generic-proxy series
```

A line reading `zero-init skipped: meter provider is ...` means the SDK meter provider was never installed, so no series were seeded and none of the checks below will show anything.

On ECS, seeding still happens — `opentelemetry-instrument` installs a real SDK provider, which is exactly the case the guard lets through. The log line is there too, in the **per-container v2 log group** named by the task definition, not the older `/ecs/mcp-gateway-auth-server`:

```bash
aws logs filter-log-events --log-group-name /ecs/mcp-gateway-v2-auth-server \
  --filter-pattern 'zero-init' --start-time $(( ($(date +%s) - 3600) * 1000 )) \
  --query 'events[-3:].message' --output text
# zero-init seeded 47/47 generic-proxy series

# the group name comes from the task definition, so read it rather than guessing:
aws ecs describe-task-definition --task-definition mcp-gateway-v2-auth \
  --query 'taskDefinition.containerDefinitions[].logConfiguration.options."awslogs-group"' --output text
```

A rolling deploy leaves the replaced task's stream in the same group, so filter by a stream whose id matches a **currently running** task (`aws ecs describe-tasks ... --query 'tasks[].taskArn'`) or you may read a startup line from the previous revision.

Then confirm the effect in AMP: `sum by (outcome)(mcpgw_registry_generic_proxy_stream_outcome_total)` returns all six values with the untriggered ones at `0`. Verified on the ECS deployment: log line at 15:50:09 UTC and 6 stream outcomes plus both `slot_rejected` pools seeded, with no `:9464` involved.

#### A worked example: OpenAI registered as a generic REST endpoint

Say an operator has put the OpenAI API behind the gateway as a proxied custom record of type `rest-endpoint`, so callers reach `api.openai.com` through a registry-issued URL and never hold the backend key themselves. Registering it looks like this:

```bash
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" --token-file .token \
  custom-proxy-create --type rest-endpoint --name openai-proxy \
  --target-url https://api.openai.com \
  --streaming true --auth-passthrough
```

`--auth-passthrough` registers `Authorization` as a caller-overridable upstream header, so each caller supplies their own OpenAI key and the registry stores none. `--streaming true` sets `proxy_streaming` on the record, which matters for the metrics below.

**`rest-endpoint` is not a reserved word.** It is the name of a custom entity type somebody created on this deployment with `custom-type-create`, and the examples below use it because that is what the type happens to be called here. It is a literal string in every URL, authz key, and metric label **for this deployment only**. If your type is called `llm-endpoint` or `http-backend`, substitute that everywhere `rest-endpoint` appears below, including inside the `server` label value. List what exists before copying anything:

```bash
uv run python api/registry_management.py --registry-url "$REGISTRY_URL" --token-file .token \
  custom-type-list --json | jq -r '.custom_types[].name'
```

The metric label `target_kind="generic_proxy_custom"` is the part that does not vary: every custom type collapses to that one value, however many types exist and whatever they are named. Only `server` carries the type name.

Once callers start using it, four questions come up, and each maps to one of the labels this feature adds:

| Question | Where the answer is |
|---|---|
| How much traffic is this endpoint taking, next to my skills and MCP servers? | `target_kind="generic_proxy_custom"` on `mcpgw_registry_auth_request_total` |
| Which proxied endpoint, out of the several registered? | the `server` label, which holds the entity's authz key |
| Is anyone being denied, and what scope rule would fix it? | `success="false"` on that same series; the `server` value is the string to put in `server_access` |
| Are its streams completing, or hitting a ceiling? | `mcpgw_registry_generic_proxy_stream_outcome_total` |

Every gateway request carries a `generic_proxy_*` kind, so `target_kind="unknown"` holds no gateway traffic. A gateway call appearing there means the `X-Generic-Proxy-Kind` marker did not reach the auth-server.

The walkthrough below drives that endpoint two ways, then contrasts it with a buffered entity, a skill, and an A2A agent, checking the counters after each.

#### Shell variables

```bash
export GW=$(jq -r .tokens.access_token .token)      # gateway token, valid for THIS deployment
export OAI=$(tr -d '\n' < .scratchpad/.oai)         # your OpenAI key, sent as Authorization
export REGISTRY_URL=http://localhost

# read the client URL off the record rather than assembling it
export OPENAI_BASE=$REGISTRY_URL$(curl -sS --compressed -H "Authorization: Bearer $GW" \
  "$REGISTRY_URL/api/custom/rest-endpoint" \
  | jq -r '.records[] | select(.name=="openai-proxy") | .proxy_client_url')
```

The gateway token goes in `X-Authorization` and the OpenAI key in `Authorization`. Sending the gateway token in both trips the equal-token guard and returns 401.

Every `curl` needs `--compressed`: the gateway gzips JSON, and without it a working call prints unreadable bytes while `-w '%{http_code}'` still reports 200.

#### Case 1 — the OpenAI endpoint, non-streaming request body

```bash
curl -sS --compressed -X POST \
  -H "X-Authorization: Bearer $GW" -H "Authorization: Bearer $OAI" \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Reply with the single word: alpha"}],"stream":false}' \
  "$OPENAI_BASE/v1/chat/completions" | jq -r '.choices[0].message.content'
# alpha
```

#### Case 2 — the same endpoint, streaming request body

```bash
curl -sS --compressed -N -X POST \
  -H "X-Authorization: Bearer $GW" -H "Authorization: Bearer $OAI" \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Count 1 to 5 slowly"}],"stream":true}' \
  "$OPENAI_BASE/v1/chat/completions" | grep -c '^data:'
# 33
```

#### What the counters show after cases 1 and 2, and the trap

```
mcpgw_registry_auth_request_total{...,server="rest-endpoint/rest-endpoint/6160de6a-...",target_kind="generic_proxy_custom"} 4.0
mcpgw_registry_generic_proxy_stream_outcome_total{outcome="started"}   4.0
mcpgw_registry_generic_proxy_stream_outcome_total{outcome="completed"} 4.0
```

**Both cases incremented the stream counter**, including the one that asked for `"stream": false`. `proxy_streaming` is a property of the **registered entity**, not of the request: it decides whether the hop streams the upstream response or buffers it. OpenAI's `"stream"` body field only changes what the upstream sends back. An entity with `proxy_streaming=true` therefore takes the streaming hop path on every request.

So `generic_proxy_stream_outcome_total` answers "how are this deployment's streaming-enabled entities behaving", not "how many callers asked for SSE".

#### Case 3 — a second custom record, this one buffered

`openmeteo-forecast` is registered with `proxy_streaming=false`:

```bash
export OM_BASE=$REGISTRY_URL$(curl -sS --compressed -H "Authorization: Bearer $GW" \
  "$REGISTRY_URL/api/custom/rest-endpoint" \
  | jq -r '.records[] | select(.name=="openmeteo-forecast") | .proxy_client_url')

curl -sS --compressed -H "X-Authorization: Bearer $GW" \
  "$OM_BASE/v1/forecast?latitude=38.9&longitude=-77.03&current=temperature_2m" \
  | jq '.current.temperature_2m'
# 21.4
```

```
mcpgw_registry_auth_request_total{...,server="rest-endpoint/rest-endpoint/1f32aefe-...",target_kind="generic_proxy_custom"} 1.0
mcpgw_registry_generic_proxy_stream_outcome_total{outcome="started"} 4.0     <- unchanged
```

Classification incremented; the stream counter did not. Two entities of the same `entity_type` land in the same `generic_proxy_custom` series while keeping separate `server` values, which is the intended split: `entity_type` is bounded and stays at three values, `server` identifies the endpoint.

#### Case 4 — a proxied skill, showing a different entity_type

A skill is the other kind of entity that reaches the data plane through the gateway, and it classifies separately, so the same walkthrough is worth running against one.

The example here is a skill named `pdf` that somebody flipped to proxied with a backend of `https://raw.githubusercontent.com`. As with `rest-endpoint` above, `pdf` is nothing special — it is whatever the skill was named at registration. Substitute yours, and list what is proxied first:

```bash
curl -sS --compressed -H "Authorization: Bearer $GW" "$REGISTRY_URL/api/skills" \
  | jq -r '.skills[] | select(.is_proxied) | "\(.path)  ->  \(.proxy_target_url)  (client: \(.proxy_client_url))"'
# /skills/pdf  ->  https://raw.githubusercontent.com  (client: /gateway/skill/pdf)
```

A skill has no backend of its own, so an operator has to name one. This backend is a bare origin, which means whatever the caller appends travels to it — so the path below is a real file on `raw.githubusercontent.com`, not a made-up one:

```bash
curl -sS --compressed -H "X-Authorization: Bearer $GW" \
  "$REGISTRY_URL/gateway/skill/pdf/anthropics/courses/refs/heads/master/prompt_engineering_interactive_tutorial/README.md" \
  | head -1
# # Welcome to Anthropic's Prompt Engineering Interactive Tutorial
```

```
mcpgw_registry_auth_request_total{...,server="skill/skills/pdf",target_kind="generic_proxy_skill"} 2.0
```

Two things to read off that line.

`target_kind` is `generic_proxy_skill`, so skill traffic separates from the custom-record traffic in cases 1 to 3 without either one needing a per-entity label.

`server` reads `skill/skills/pdf`, not `skill/pdf`. The label is built from the nginx marker headers, which carry the full registered path, so it equals the authz key `/validate` authorizes against. The client URL is the shorter `/gateway/skill/pdf`, because the generated path strips the registered path's leading namespace segment. Read the label, not the URL, when writing a scope rule.

A non-2xx from the upstream still increments this counter — authorization happens at `/validate`, before the upstream is reached — so the series counts requests routed, not requests the backend served. A request for a path that does not exist on the origin returns 404 and still counts here, which is why a drop in this series means callers stopped arriving rather than the backend breaking.

#### Case 5 — an A2A agent card, which is a different route entirely

```bash
curl -sS --compressed -H "X-Authorization: Bearer $GW" \
  "$REGISTRY_URL/agent/<agent-name>/.well-known/agent-card.json" | jq '{name, url}'
```

```
mcpgw_registry_auth_request_total{...,server="agent",target_kind="a2a_agent"} 1.0
```

This is the **pre-existing** A2A reverse proxy at `/agent/...`, so it keeps `target_kind="a2a_agent"` and the coarse `server="agent"`. `generic_proxy_agent` appears only for an agent opted into the generic proxy, served at `/gateway/a2a_agent/<name>/`.

#### Confirm the histogram carries no `server` label

```bash
docker exec $AUTH sh -c 'curl -s localhost:9464/metrics' \
  | grep '^mcpgw_registry_auth_request_duration' | grep -c 'server='
# 0
```

```bash
docker exec $AUTH sh -c 'curl -s localhost:9464/metrics' \
  | grep -c '^mcpgw_registry_auth_request_duration'    # 108
docker exec $AUTH sh -c 'curl -s localhost:9464/metrics' \
  | grep -c '^mcpgw_registry_auth_request_total'       # 6
```

108 histogram series across 6 label groups is 18 per group: 16 `le` buckets plus `_count` and `_sum`. Each new proxied endpoint adds 1 series to the counter and 0 to the histogram.

#### Nothing should land in `unknown`

```bash
docker exec $AUTH sh -c 'curl -s localhost:9464/metrics' \
  | grep 'auth_request_total.*target_kind="unknown"'
```

Empty, or a count that does not grow when you repeat the cases above. A gateway request landing in `unknown` means the `X-Generic-Proxy-Kind` marker did not reach the auth-server, so check the rendered location:

```bash
docker exec mcp-gateway-registry-registry-1 \
  grep -A3 'X-Generic-Proxy-Kind' /etc/nginx/conf.d/nginx_rev_proxy.conf | head -20
```

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
