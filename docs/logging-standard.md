# Logging Standard

This document describes the log file paths, format, and rotation behavior for
the AI Registry deployment. It is the reference document customers and their
log-aggregation teams (Splunk, ELK, CloudWatch, etc.) can use to configure
ingestion.

> **Related:** For the Admin API / Log Viewer UI, MongoDB-based centralized log
> retrieval across pods, and the full `APP_LOG_*` configuration table, see
> [logging.md](logging.md). This document focuses specifically on the on-disk
> file format and host paths consumed by external log shippers.

Scope: Docker Compose deployments. Helm/EKS and Terraform/ECS deployments use
their platform-native log collection (pod stdout via `/var/log/containers/` on
the node for EKS; `awslogs` driver to CloudWatch for ECS) and are not affected
by the host filesystem paths documented here.

## Log File Paths

All service log files land on the Docker host under
`/var/log/containers/ai-registry/`:

| Service | Host file path |
| --- | --- |
| registry | `/var/log/containers/ai-registry/registry.log` |
| auth-server | `/var/log/containers/ai-registry/auth-server.log` |
| mcpgw (a.k.a. ai-registry-tools) | `/var/log/containers/ai-registry/ai-registry-tools.log` |
| nginx access | `/var/log/containers/ai-registry/nginx-access.log` |
| nginx error | `/var/log/containers/ai-registry/nginx-error.log` |

Rotated files live alongside:
`registry.log.1`, `registry.log.2.gz`, `nginx-access.log.1.gz`, etc.

### Why the `ai-registry/` subdirectory

Kubernetes kubelet writes pod logs flat under `/var/log/containers/`.
Splunk Connect for Kubernetes (and most stock inputs.conf shipped with Splunk
Universal Forwarder) watches `/var/log/containers/*.log` non-recursively and
expects the kubelet filename pattern
`<pod>_<namespace>_<container>-<hash>.log`.

Placing our files one directory deeper means:

1. On hosts running both Docker and Kubernetes, our files do not show up under
   the stock K8s monitor stanza and therefore do not get misclassified.
2. Customer Splunk config for AI Registry stays isolated in its own input
   stanza, easy to enable/disable independently of K8s scraping.

## Log Format Standard

Python services (registry, auth-server, mcpgw) emit JSON Lines (JSONL) to
their `.log` files by default. One JSON object per line, UTF-8 encoded,
newline terminated.

Console / stdout output remains the existing human-readable comma-separated
format so `docker logs <container>` stays skimmable without a JSON parser.

### Mandatory fields (every record)

| Field | Type | Source | Example |
| --- | --- | --- | --- |
| `timestamp` | string (ISO 8601 UTC, microsecond precision, with offset) | Python `datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()` | `"2026-05-09T14:23:45.123456+00:00"` |
| `service` | string | Fixed per container: `registry`, `auth-server`, `ai-registry-tools` | `"registry"` |
| `level` | string | Python `record.levelname` | `"INFO"` |
| `logger` | string | Python `record.name` | `"registry.core.config"` |
| `filename` | string | Python `record.filename` | `"config.py"` |
| `lineno` | integer | Python `record.lineno` | `42` |
| `process_id` | integer | Python `record.process` | `12345` |
| `message` | string | `record.getMessage()` (format args already applied) | `"Starting registry on port 8000"` |

### Additional fields (exception records)

| Field | Type | Source |
| --- | --- | --- |
| `exc_type` | string | `record.exc_info[0].__name__` |
| `exc_message` | string | `str(record.exc_info[1])` |
| `stack_trace` | string | `"".join(traceback.format_exception(*record.exc_info))` |

### Example records

Info record:

```json
{"timestamp":"2026-05-09T14:23:45.123456+00:00","service":"registry","level":"INFO","logger":"registry.core.config","filename":"config.py","lineno":42,"process_id":12345,"message":"Starting registry on port 8000"}
```

Exception record:

```json
{"timestamp":"2026-05-09T14:23:45.987654+00:00","service":"auth-server","level":"ERROR","logger":"auth_server.oauth","filename":"oauth.py","lineno":156,"process_id":67890,"message":"Token exchange failed","exc_type":"httpx.ConnectError","exc_message":"Cannot connect to keycloak:8080","stack_trace":"Traceback (most recent call last):\n  File \"/app/auth_server/oauth.py\", line 156, in exchange\n    ...\nhttpx.ConnectError: Cannot connect to keycloak:8080\n"}
```

### Nginx log formats

Nginx logs use the standard well-known formats; no custom parsing required:

- `nginx-access.log`: `combined` (Apache-style). Splunk sourcetype
  `access_combined`.
- `nginx-error.log`: nginx default (`YYYY/MM/DD HH:MM:SS [level] pid#tid:
  message`). Splunk sourcetype `nginx:error`.

## Rotation

- Python service logs rotate via `RotatingFileHandler` at
  `APP_LOG_MAX_BYTES` bytes (default 50 MB) with `APP_LOG_BACKUP_COUNT`
  backups (default 5).
- Nginx logs rotate daily via `/etc/logrotate.d/nginx-mcp` inside the registry
  container, compressed with 14-day retention. A background loop in the
  registry entrypoint invokes `logrotate` once every 24 hours.

Expected total on-disk footprint per deployment is roughly 1 GB worst case.

## Configuration Knobs

| Environment variable | Default | Description |
| --- | --- | --- |
| `APP_LOG_DIR` | `/var/log/containers/ai-registry` | Directory where service `.log` files are written. Must be absolute; `..` segments are rejected. |
| `APP_LOG_FILE_FORMAT` | `json` | `json` for JSONL (default) or `text` for the legacy comma-separated format. Console format is not affected. |
| `APP_LOG_LEVEL` | `INFO` | Root log level. `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL`. |
| `APP_LOG_MAX_BYTES` | `52428800` (50 MB) | Python file rotation size threshold. |
| `APP_LOG_BACKUP_COUNT` | `5` | Number of rotated backups to keep. |
| `APP_LOG_CENTRALIZED_ENABLED` | `true` | Also write log records to MongoDB collection `application_logs_<namespace>`. Unrelated to file logging; controlled independently. |

## Splunk Ingestion Recipe

The following stanzas assume a Splunk Universal Forwarder or Splunk Heavy
Forwarder running on the Docker host. Index name and host assignment should
match your conventions.

### `inputs.conf`

```ini
[monitor:///var/log/containers/ai-registry/*.log]
disabled = false
index = ai_registry
sourcetype = ai-registry:json

[monitor:///var/log/containers/ai-registry/nginx-access.log]
disabled = false
index = ai_registry
sourcetype = access_combined

[monitor:///var/log/containers/ai-registry/nginx-error.log]
disabled = false
index = ai_registry
sourcetype = nginx:error
```

### `props.conf`

```ini
[ai-registry:json]
INDEXED_EXTRACTIONS = json
KV_MODE = none
TIMESTAMP_FIELDS = timestamp
TIME_FORMAT = %Y-%m-%dT%H:%M:%S.%6N%:z
TRUNCATE = 0
SHOULD_LINEMERGE = false
```

With this configuration analysts can query structured fields directly:

```
index=ai_registry sourcetype=ai-registry:json service=registry level=ERROR
```

## Host Preparation

Before starting containers, the host log directory must exist with
`uid 1000` ownership so the non-root container user can write to it.
`build_and_run.sh` does this automatically via
`scripts/prepare-log-dirs.sh`. To prepare the directory manually:

```bash
sudo mkdir -p /var/log/containers/ai-registry
sudo chown -R 1000:1000 /var/log/containers/ai-registry
sudo chmod 0750 /var/log/containers/ai-registry
```

## Troubleshooting

### Log files are missing on the host

Check the container log:

```bash
docker logs registry 2>&1 | grep -i "cannot write\|log file"
```

An error like `Cannot write to log file /var/log/containers/ai-registry/
registry.log` indicates the host directory is either missing or not owned
by uid 1000. Run the host preparation commands above and restart the
container.

### I want the old /app/logs behavior back

Set `APP_LOG_DIR=/app/logs` in your `.env` and ensure a matching volume
mount is available (historical docker-compose.yml named volume
`registry-logs` is still present for audit logs and can be reused).
This keeps file logging in the container-local overlay filesystem as it
was before v1.23.0.

### I want the legacy plain-text format in my files

Set `APP_LOG_FILE_FORMAT=text` to revert to the comma-separated format.
Useful during a phased rollout when downstream parsers still expect the
old format.

## Scope Boundaries

The paths and rotation mechanisms above apply to the Docker Compose
deployment model. Other deployment modes are unaffected:

- **Helm / EKS**: logs flow via pod stdout; Splunk Connect for
  Kubernetes picks them up from node `/var/log/containers/<pod>_<ns>_
  <container>-<hash>.log` automatically. No changes to Helm charts.
- **Terraform / ECS**: logs flow via the `awslogs` log driver to
  CloudWatch. The `APP_LOG_DIR` env var is set in the task definition
  but does not influence log shipping (files land on task-local
  ephemeral storage).

## Related

- Issue: https://github.com/agentic-community/mcp-gateway-registry/issues/987
- Audit logs (separate subsystem): `docs/OBSERVABILITY.md`
- MongoDB centralized logging (also a separate path):
  `registry/utils/mongodb_log_handler.py`
