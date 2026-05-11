# Telemetry Documentation

## Overview

The MCP Gateway Registry collects anonymous usage telemetry to understand adoption patterns and improve the product. This document describes what data is collected, how to opt-out, and our privacy commitments.

## What Data is Collected

### Tier 1: Startup Ping (Opt-Out, Default ON)

Sent once at startup:

| Field | Example | Description |
|-------|---------|-------------|
| `registry_id` | `c546a650-...` | Registry Card UUID (public, not PII) |
| `v` | `1.0.16` | Registry version |
| `py` | `3.12` | Python version (major.minor) |
| `os` | `linux` | Operating system (linux, darwin, windows) |
| `arch` | `x86_64` | CPU architecture |
| `cloud` | `aws` | Cloud provider (aws, gcp, azure, unknown) |
| `compute` | `ecs` | Compute platform (ecs, eks, kubernetes, docker, ec2, unknown) |
| `mode` | `with-gateway` | Deployment mode |
| `registry_mode` | `full` | Registry operating mode |
| `storage` | `documentdb` | Storage backend (file, documentdb, mongodb-ce) |
| `auth` | `keycloak` | Auth provider |
| `federation` | `true` | Whether federation is enabled |
| `search_queries_total` | `150` | Lifetime semantic search query count |
| `search_queries_24h` | `12` | Search queries in the last 24 hours |
| `search_queries_1h` | `3` | Search queries in the last hour |
| `embeddings_provider` | `litellm` | Embeddings code path: `sentence-transformers` or `litellm` (added in schema v2) |
| `embeddings_backend_kind` | `bedrock` | Derived coarse backend category: `sentence-transformers`, `bedrock`, `openai`, `azure-openai`, `voyage`, `cohere`, `other`, or `unknown` (added in schema v2) |
| `ts` | `2026-03-18T00:00:00Z` | ISO 8601 timestamp |

### Tier 2: Daily Heartbeat (Opt-Out, Default ON)

> **Behavior change (post v1.0.18):** The daily heartbeat was previously opt-in (`MCP_TELEMETRY_OPT_IN=1`). It is now opt-out and sent by default every 24 hours. Since the heartbeat contains only aggregate counts (no PII), this aligns it with the startup ping behavior.

Sent at a configurable interval (default: every 24 hours). Includes all Tier 1 fields plus:

| Field | Example | Description |
|-------|---------|-------------|
| `servers_count` | `15` | Number of registered MCP servers |
| `agents_count` | `8` | Number of registered A2A agents |
| `skills_count` | `23` | Number of registered skills |
| `peers_count` | `2` | Number of federation peers |
| `search_backend` | `documentdb` | Search backend (faiss or documentdb) |
| `embeddings_provider` | `sentence-transformers` | Embeddings code path: `sentence-transformers` or `litellm` |
| `embeddings_backend_kind` | `bedrock` | Derived coarse backend category: `sentence-transformers`, `bedrock`, `openai`, `azure-openai`, `voyage`, `cohere`, `other`, or `unknown` (added in schema v2) |
| `uptime_hours` | `48` | Hours since server started |

## Request Signing (HMAC)

All telemetry requests are signed with HMAC-SHA256 to prevent unauthorized use of the collector endpoint. The registry computes a signature over the JSON request body and sends it in the `X-Telemetry-Signature` HTTP header. The server-side Lambda collector verifies this signature before processing any event.

This is not a secret-based authentication mechanism -- the signing key is embedded in the open-source code. Its purpose is to raise the bar against casual abuse (e.g., random `curl` requests to the endpoint). Combined with IP-based rate limiting and strict Pydantic schema validation, this makes endpoint abuse impractical.

### Example HTTP Request

A startup event request (schema v2) looks like this:

```http
POST /v1/collect HTTP/1.1
Host: m3ijrhd020.execute-api.us-east-1.amazonaws.com
Content-Type: application/json
X-Telemetry-Signature: 8a3f2b...c9d1e0

{"arch":"x86_64","auth":"keycloak","cloud":"aws","compute":"ecs","embeddings_backend_kind":"bedrock","embeddings_provider":"litellm","event":"startup","federation":true,"mode":"with-gateway","os":"linux","py":"3.12","registry_id":"c546a650-8af9-4721-9efb-7df221b2a0d9","registry_mode":"full","schema_version":"2","search_queries_1h":3,"search_queries_24h":12,"search_queries_total":150,"storage":"documentdb","ts":"2026-03-18T00:00:00+00:00","v":"1.0.22"}
```

A heartbeat event request (schema v2):

```http
POST /v1/collect HTTP/1.1
Host: m3ijrhd020.execute-api.us-east-1.amazonaws.com
Content-Type: application/json
X-Telemetry-Signature: 5b7e1a...d4f2c3

{"agents_count":8,"cloud":"aws","compute":"ecs","embeddings_backend_kind":"sentence-transformers","embeddings_provider":"sentence-transformers","event":"heartbeat","peers_count":2,"registry_id":"c546a650-8af9-4721-9efb-7df221b2a0d9","schema_version":"2","search_backend":"documentdb","search_queries_1h":3,"search_queries_24h":12,"search_queries_total":150,"servers_count":15,"skills_count":23,"ts":"2026-03-18T12:00:00+00:00","uptime_hours":48,"v":"1.0.22"}
```

Registries running versions earlier than v1.0.22 emit `schema_version":"1"` events without `embeddings_backend_kind`. The collector accepts both versions.

Notes:
- JSON body keys are sorted alphabetically (`sort_keys=True`) and compact (`separators=(",",":")`) for deterministic HMAC computation
- The `X-Telemetry-Signature` header is the HMAC-SHA256 hex digest of the raw JSON body

## Force Telemetry (Admin API)

Admins can trigger telemetry events on demand (bypasses the distributed lock):

```bash
# Force heartbeat
uv run python api/registry_management.py --registry-url http://localhost --token-file .token-local telemetry-heartbeat

# Force startup ping
uv run python api/registry_management.py --registry-url http://localhost --token-file .token-local telemetry-startup
```

API endpoints (require admin auth):
- `POST /api/registry-management/telemetry/heartbeat`
- `POST /api/registry-management/telemetry/startup`

## What is NOT Collected

We never collect any personally identifiable information (PII):

- IP addresses, MAC addresses, hostnames
- Server names, URLs, file paths
- User data, credentials, tokens
- Query content, agent card content, skill code
- Any data that could identify a person or organization

### Embeddings configuration -- what is NOT sent

The registry reads the embeddings settings locally to derive the coarse `embeddings_backend_kind` category, but these operator-configured values are **never** included in any telemetry payload:

- `EMBEDDINGS_MODEL_NAME` (e.g. `amazon.titan-embed-text-v2:0`, `text-embedding-3-small`, or any custom/deployment-specific name)
- `EMBEDDINGS_MODEL_DIMENSIONS` (the configured vector size)
- `EMBEDDINGS_API_KEY`, `EMBEDDINGS_SECRET_KEY` (credentials)
- `EMBEDDINGS_API_BASE` (endpoint URL)
- `EMBEDDINGS_AWS_REGION` (infrastructure region)

Only the derived `embeddings_backend_kind` (one of a fixed 8-value allowlist) and the existing `embeddings_provider` code-path string leave the registry process.

## Schema Versioning

Telemetry events include a `schema_version` field. We bump it whenever fields are added, removed, or have their semantics changed. Additive optional fields still bump from N to N+1 so downstream consumers can tell at a glance which events carry the newer data.

| Version | Introduced in | Changes |
|---------|---------------|---------|
| `"1"` | Initial release | Original startup + heartbeat schema |
| `"2"` | v1.0.22 | Added `embeddings_provider` to startup; added `embeddings_backend_kind` to both startup and heartbeat |
| `"3"` | v1.23.0 | Added `cloud_detection_method` to both startup and heartbeat (issue #986) |

Older registry versions continue to send schema v1/v2 events; the collector accepts all versions side-by-side.

## Cloud Provider Detection

Cloud detection cascades through env vars, DMI files, ECS task metadata, Kubernetes node-name heuristics, and (last resort) cloud IMDS endpoints. The first tier that produces a signal wins; the rest are skipped.

| Method (`cloud_detection_method`) | Signal |
|-----------------------------------|--------|
| `env` | One of `AWS_REGION`, `AWS_DEFAULT_REGION`, `GOOGLE_CLOUD_PROJECT`, `GCLOUD_PROJECT`, `WEBSITE_INSTANCE_ID`, `AZURE_CLIENT_ID` is set |
| `dmi` | `/sys/devices/virtual/dmi/id/{board_asset_tag,product_name,sys_vendor}` readable and matches known vendor strings |
| `ecs_meta` | `ECS_CONTAINER_METADATA_URI_V4` or `ECS_CONTAINER_METADATA_URI` is set (AWS only) |
| `k8s_heuristic` | `KUBERNETES_SERVICE_HOST` set and `NODE_NAME` matches a cloud-specific pattern (`*.compute.internal` for AWS, `gke-*` for GCP, `aks-*` for Azure) |
| `imds` | Live HTTP probe to `http://169.254.169.254/` (AWS IMDSv2, Azure) or `http://metadata.google.internal/` (GCP). 300ms timeout per provider, tried sequentially AWS -> GCP -> Azure. Worst case: ~900ms once per process at startup. |
| `unknown` | No signal fired. |

### Zero-cost opt-in (no outbound network)

Operators who want accurate telemetry without enabling IMDS probes can set any of the following env vars. Detection short-circuits at tier 1 (`env`) and never calls out:

- `AWS_REGION=us-east-1`
- `GOOGLE_CLOUD_PROJECT=my-project`
- `AZURE_CLIENT_ID=00000000-0000-0000-0000-000000000000`

### EKS IMDS hop-limit (the #1 operator trap)

On EKS, IMDS probes from pods require the node's `httpPutResponseHopLimit` to be set to `2` (the AWS default is `1`). Without this, IMDSv2 requests originating from pods are dropped at the hypervisor and detection falls back to `unknown`.

Fix options:

- Set `--metadata-options '{"httpPutResponseHopLimit": 2}'` on your managed node group. See [AWS docs on IMDS hop-limit](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/configuring-instance-metadata-options.html#configuring-instance-metadata-options-put-response-hop-limit).
- OR inject `AWS_REGION` directly into pod env via the downward API or a ConfigMap. Detection resolves at tier 1 without hitting IMDS.

### What the IMDS probe does NOT do

The probe checks whether the cloud metadata service responds at all. It does **not** read or emit the instance ID, region, availability zone, role ARN, tags, or any other metadata content.

- AWS IMDSv2 token TTL is set to 1 second; the token value is discarded when the response goes out of scope.
- The response body is never logged, persisted, or forwarded.
- Only the detected cloud label (`aws`/`gcp`/`azure`) and the detection method (`env`/`dmi`/`ecs_meta`/`k8s_heuristic`/`imds`/`unknown`) are ever emitted.
- No URLs, headers, or response content are logged on success OR failure; debug logs include only the exception type.

### Opt-out of IMDS probing

Set `MCP_TELEMETRY_IMDS_PROBE_DISABLED=1` to skip the IMDS tier. The env, DMI, ECS metadata, and k8s heuristic tiers still run. Setting `MCP_TELEMETRY_DISABLED=1` (the master switch) also disables IMDS probing since detection only runs as part of telemetry payload building.

## Startup Banner

When telemetry is enabled (the default), you will see this banner at startup:

```
==============================================================================
[telemetry] Anonymous usage telemetry is ON (startup ping + daily heartbeat)
[telemetry] No PII is collected (no IPs, hostnames, or user data)
[telemetry] Endpoint: https://m3ijrhd020.execute-api.us-east-1.amazonaws.com/v1/collect
[telemetry] To disable all: set MCP_TELEMETRY_DISABLED=1
[telemetry] Details: https://github.com/agentic-community/mcp-gateway-registry/blob/main/docs/TELEMETRY.md
==============================================================================
```

## Telemetry Configuration Parameters

| Environment Variable | Purpose | Default |
|---------------------|---------|---------|
| `MCP_TELEMETRY_DISABLED` | Set to `1` to disable all telemetry (startup ping + heartbeat) | _(not set, telemetry ON)_ |
| `MCP_TELEMETRY_OPT_OUT` | Set to `1` to disable daily heartbeat only (startup ping still sent) | _(not set, heartbeat ON)_ |
| `MCP_TELEMETRY_HEARTBEAT_INTERVAL_MINUTES` | Heartbeat send frequency in minutes | `1440` (24 hours) |
| `MCP_TELEMETRY_ENDPOINT` | HTTPS URL for a self-hosted telemetry collector | _(built-in endpoint)_ |
| `MCP_TELEMETRY_DEBUG` | Set to `true` to log payloads instead of sending | `false` |
| `MCP_TELEMETRY_IMDS_PROBE_DISABLED` | Set to `1` to skip IMDS probing in cloud detection (env, DMI, ECS metadata, k8s heuristic tiers still run) | _(not set, probes enabled)_ |

### Docker Compose

Add these to your `.env` file in the project root:

```bash
# .env
MCP_TELEMETRY_DISABLED=1          # Disable all telemetry (startup ping + heartbeat)
MCP_TELEMETRY_OPT_OUT=1           # Disable heartbeat only (startup ping still sent)
MCP_TELEMETRY_HEARTBEAT_INTERVAL_MINUTES=1440  # Heartbeat interval in minutes (default: 1440 = 24h)
MCP_TELEMETRY_ENDPOINT=https://your-collector.example.com/v1/collect  # Self-hosted (optional)
MCP_TELEMETRY_DEBUG=true           # Debug mode (optional)
```

These are automatically picked up by the `docker-compose.yml`, `docker-compose.prebuilt.yml`, and `docker-compose.podman.yml` files.

### ECS (Terraform)

Add these to your `terraform.tfvars`:

```hcl
# terraform.tfvars
mcp_telemetry_disabled                   = "1"     # Disable all telemetry
mcp_telemetry_opt_out                    = "1"     # Disable heartbeat only (startup ping still sent)
mcp_telemetry_heartbeat_interval_minutes = "1440"  # Heartbeat interval in minutes (default: 1440 = 24h)
telemetry_debug                          = "true"  # Debug mode (optional)
```

The corresponding Terraform variables are defined in `terraform/aws-ecs/variables.tf`.

### Kubernetes (Helm)

Set these in your `values.yaml` or pass with `--set`:

```yaml
# values.yaml (standalone chart)
app:
  mcpTelemetryDisabled: true       # Disable all telemetry
  mcpTelemetryOptOut: true         # Disable heartbeat only (startup ping still sent)
  telemetryHeartbeatIntervalMinutes: "1440"  # Heartbeat interval in minutes (default: 1440 = 24h)
  telemetryDebug: true             # Debug mode (optional)

# -- OR for the stack chart --
# values.yaml (mcp-gateway-registry-stack)
registry:
  app:
    mcpTelemetryDisabled: true
    mcpTelemetryOptOut: true
    telemetryHeartbeatIntervalMinutes: "1440"
    telemetryDebug: true
```

Or with `helm install`/`helm upgrade`:

```bash
helm upgrade my-release charts/registry \
  --set app.mcpTelemetryDisabled=true \
  --set app.mcpTelemetryOptOut=true
```

These values are injected as environment variables via the `registry-otel-config` ConfigMap.

## How to Opt-Out

Set `MCP_TELEMETRY_DISABLED=1` using the method for your deployment (see above).

When telemetry is disabled, you'll see this message at startup:

```
[telemetry] Telemetry is disabled.
```

## How to Opt-Out of Heartbeat Only

Both startup ping and daily heartbeat are enabled by default. To disable the heartbeat while keeping the startup ping:

Set `MCP_TELEMETRY_OPT_OUT=1` using the method for your deployment (see above).

When heartbeat is opted out, you'll see:

```
[telemetry] Heartbeat scheduler not started (opted out or telemetry disabled)
```

## Debug Mode

Set `MCP_TELEMETRY_DEBUG=true` using the method for your deployment (see above).

This logs the full JSON payload to stderr instead of sending it to the collector.

## Privacy Commitments

1. **Privacy First**: No PII is ever collected or stored
2. **Conspicuous Disclosure**: Every startup logs a clear message about telemetry
3. **Easy Opt-Out**: Multiple methods to disable telemetry
4. **Fail-Silent**: Telemetry failures never impact registry operation
5. **No Tracking**: No user identification or cross-session tracking
6. **Open Source**: The telemetry code is open source and auditable

## Multi-Replica Deployments

In multi-replica deployments (ECS, Kubernetes), telemetry uses MongoDB-based distributed locks to prevent duplicate sends. Only one replica will send telemetry within the configured interval:

- **Startup ping**: At most once per 60 seconds
- **Heartbeat**: At most once per configured interval (default: 1440 minutes = 24 hours)

## Self-Hosted Telemetry Collector

If you want to run your own telemetry collector instead of using the default endpoint, you can deploy the server-side infrastructure from issue #559.

### Why Self-Host?

- **Data Sovereignty**: Keep telemetry data in your own AWS account
- **Compliance**: Meet specific regulatory requirements
- **Custom Analytics**: Run your own queries and dashboards
- **Air-Gapped Deployments**: Collect telemetry without external network access

### Quick Start

The telemetry collector infrastructure is available in `terraform/telemetry-collector/`:

```bash
cd terraform/telemetry-collector

# Configure deployment
cp terraform.tfvars.example terraform.tfvars
vi terraform.tfvars  # Set aws_region, deployment_stage, etc.

# Deploy infrastructure (~15-20 minutes)
terraform init
terraform apply

# Get your collector URL
terraform output collector_url
```

### Point Registry to Your Collector

```bash
# Set custom endpoint
export MCP_TELEMETRY_ENDPOINT=https://your-collector-url.execute-api.us-east-1.amazonaws.com/v1/collect

# Start registry
uv run python -m registry
```

### Infrastructure Components

The self-hosted collector includes:

- **API Gateway HTTP API**: HTTPS endpoint (`/v1/collect`)
- **Lambda Function**: VPC-enabled, validates events with Pydantic schemas
- **DynamoDB**: Privacy-preserving rate limiting (hashed IPs)
- **DocumentDB**: MongoDB-compatible storage with 365-day TTL
- **Secrets Manager**: Secure credential management
- **CloudWatch**: Logs and alarms (production)

### Cost Estimate

- **Testing**: ~$85-90/month (db.t3.medium DocumentDB)
- **Production**: ~$195-200/month (db.r5.large DocumentDB)

See `terraform/telemetry-collector/README.md` for detailed cost breakdown.

### Security Features

- **No IP Logging**: Source IPs are hashed (SHA-256) for rate limiting only
- **HMAC Signed**: Requests signed with HMAC-SHA256 to reject unauthorized callers
- **Rate Limited**: DynamoDB-based per-IP rate limiting (10 requests/minute)
- **Schema Validated**: Strict Pydantic validation rejects malformed payloads
- **VPC Isolated**: DocumentDB not accessible from internet
- **TLS Everywhere**: All connections encrypted
- **Always Returns 204**: No information leakage (same response for valid, invalid, or rejected)
- **IAM Least Privilege**: Minimal Lambda permissions

### Bastion Host Scripts

The bastion host provides scripts for querying and managing telemetry data in DocumentDB. Scripts are located in `terraform/telemetry-collector/bastion-scripts/` and should be copied to the bastion home directory.

#### Interactive Shell (connect.sh)

Open an interactive mongosh session against DocumentDB:

```bash
~/connect.sh
```

#### Quick Summary (query.sh)

Print a summary of telemetry collections (counts, last 5 events, storage backend breakdown):

```bash
~/query.sh
```

#### Export to CSV (telemetry_db.py export)

Dump telemetry data to a CSV file:

```bash
# Export all collections to registry_metrics.csv
python3 ~/telemetry_db.py export

# Export to a custom path
python3 ~/telemetry_db.py export --output /tmp/metrics.csv

# Export only startup events
python3 ~/telemetry_db.py export --collection startup_events

# Export only heartbeat events
python3 ~/telemetry_db.py export --collection heartbeat_events
```

#### Purge Data (telemetry_db.py purge)

Delete all telemetry data from DocumentDB (with interactive confirmation):

```bash
# Purge all collections (prompts for confirmation)
python3 ~/telemetry_db.py purge

# Purge only startup events
python3 ~/telemetry_db.py purge --collection startup_events

# Purge only heartbeat events
python3 ~/telemetry_db.py purge --collection heartbeat_events

# Skip confirmation prompt
python3 ~/telemetry_db.py purge --confirm
```

#### Deploying Scripts to Bastion

Copy scripts to the bastion host after initial setup:

```bash
BASTION_IP=$(terraform output -raw bastion_public_ip)

scp -i ~/.ssh/id_ed25519 \
    bastion-scripts/connect.sh \
    bastion-scripts/query.sh \
    bastion-scripts/telemetry_db.py \
    ec2-user@$BASTION_IP:~/

ssh -i ~/.ssh/id_ed25519 ec2-user@$BASTION_IP 'chmod +x ~/connect.sh ~/query.sh'
```

### Full Documentation

See `terraform/telemetry-collector/README.md` for:
- Prerequisites and deployment steps
- DocumentDB index setup
- Testing procedures
- Troubleshooting guide
- Production deployment (custom domain, alarms)

## Questions?

For more information or questions about telemetry:

- **GitHub Issue**: https://github.com/agentic-community/mcp-gateway-registry/issues/558
- **Telemetry Source Code**: https://github.com/agentic-community/mcp-gateway-registry/blob/main/registry/core/telemetry.py
