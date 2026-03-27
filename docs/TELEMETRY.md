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
| `ts` | `2026-03-18T00:00:00Z` | ISO 8601 timestamp |

### Tier 2: Daily Heartbeat (Opt-In, Default OFF)

Sent daily when explicitly enabled. Includes all Tier 1 fields plus:

| Field | Example | Description |
|-------|---------|-------------|
| `servers_count` | `15` | Number of registered MCP servers |
| `agents_count` | `8` | Number of registered A2A agents |
| `skills_count` | `23` | Number of registered skills |
| `peers_count` | `2` | Number of federation peers |
| `search_backend` | `documentdb` | Search backend (faiss or documentdb) |
| `embeddings_provider` | `sentence-transformers` | Embeddings provider |
| `uptime_hours` | `48` | Hours since server started |

## Request Signing (HMAC)

All telemetry requests are signed with HMAC-SHA256 to prevent unauthorized use of the collector endpoint. The registry computes a signature over the JSON request body and sends it in the `X-Telemetry-Signature` HTTP header. The server-side Lambda collector verifies this signature before processing any event.

This is not a secret-based authentication mechanism -- the signing key is embedded in the open-source code. Its purpose is to raise the bar against casual abuse (e.g., random `curl` requests to the endpoint). Combined with IP-based rate limiting and strict Pydantic schema validation, this makes endpoint abuse impractical.

## What is NOT Collected

We never collect any personally identifiable information (PII):

- ❌ IP addresses, MAC addresses, hostnames
- ❌ Server names, URLs, file paths
- ❌ User data, credentials, tokens
- ❌ Query content, agent card content, skill code
- ❌ Any data that could identify a person or organization

## Startup Banner

When telemetry is enabled (the default), you will see this banner at startup:

```
==============================================================================
[telemetry] Anonymous usage telemetry is ON
[telemetry] No PII is collected (no IPs, hostnames, or user data)
[telemetry] Endpoint: https://m3ijrhd020.execute-api.us-east-1.amazonaws.com/v1/collect
[telemetry] To disable: set MCP_TELEMETRY_DISABLED=1
[telemetry] Details: https://github.com/agentic-community/mcp-gateway-registry/blob/main/docs/TELEMETRY.md
==============================================================================
```

## How to Opt-Out

### Method 1: Environment Variable (Recommended)

```bash
export MCP_TELEMETRY_DISABLED=1
```

### Method 2: Configuration File

In your `.env` file or environment:

```bash
TELEMETRY_ENABLED=false
```

### Method 3: CLI Flag

If using the registry management CLI:

```bash
registry-management --no-telemetry
```

### Verify Opt-Out

When telemetry is disabled, you'll see this message at startup:

```
[telemetry] Telemetry is disabled.
```

## How to Opt-In to Richer Data

To enable the daily heartbeat with aggregate counts:

```bash
export MCP_TELEMETRY_OPT_IN=1
```

When opted in, you'll see:

```
[telemetry] Enhanced telemetry is ON (opted in)
```

## Debug Mode

To see what data would be sent without actually sending it:

```bash
export MCP_TELEMETRY_DEBUG=1
```

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
- **Heartbeat**: At most once per 24 hours

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

### Querying Your Data

Connect to DocumentDB to analyze telemetry:

```bash
# Get DocumentDB endpoint
DOCDB_ENDPOINT=$(terraform output -raw documentdb_endpoint)

# Get credentials
aws secretsmanager get-secret-value --secret-id telemetry-collector-docdb

# Connect with mongosh
mongosh --host $DOCDB_ENDPOINT --username telemetry_admin --tls --tlsCAFile global-bundle.pem

# Query telemetry
use telemetry;
db.startup_events.find({"v": "1.0.16"}).count();
db.heartbeat_events.find({"search_backend": "documentdb"});
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
