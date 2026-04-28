# Centralized Application Logging

The MCP Gateway Registry provides centralized application log collection, storage, and retrieval across all service instances. Logs from both the `registry` and `auth-server` services are written to a shared MongoDB/DocumentDB collection, enabling cross-pod log queries through the admin API and the Settings UI.

## Architecture

```
registry / auth-server
        |
        v
  RotatingFileHandler  (always active, local file rotation)
        |
        v
  MongoDBLogHandler    (optional, buffered writes via background thread)
        |
        v
  MongoDB / DocumentDB  (application_logs collection with TTL index)
        |
        v
  Admin REST API  (/api/admin/logs)
        |
        v
  Log Viewer UI   (Settings > Application Logs)
```

### Components

- **RotatingFileHandler**: Always active. Writes logs to local files with size-based rotation (default 50 MB, 5 backups). No external dependencies.
- **MongoDBLogHandler**: Optional. Buffers log records in memory and flushes them to MongoDB periodically (default every 5 seconds or every 50 records). Uses a background daemon thread to avoid blocking the async event loop.
- **TTL Index**: MongoDB automatically deletes log documents older than the configured retention period.
- **Admin API**: Three endpoints for querying, exporting, and discovering log metadata. All require admin authentication.
- **Log Viewer UI**: Filter by service, level, hostname, time range, and message content. Supports pagination and JSONL export.

## Configuration Parameters

All parameters use the `APP_LOG_` prefix. The centralized (MongoDB) storage parameters use `APP_LOG_CENTRALIZED_`.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `APP_LOG_CENTRALIZED_ENABLED` | Write application logs to MongoDB/DocumentDB for centralized retrieval | `true` |
| `APP_LOG_CENTRALIZED_TTL_DAYS` | Days to retain log entries before automatic deletion | `1` |
| `APP_LOG_MAX_BYTES` | Maximum size per log file in bytes before rotation | `52428800` (50 MB) |
| `APP_LOG_BACKUP_COUNT` | Number of rotated backup files to keep | `5` |
| `APP_LOG_MONGODB_BUFFER_SIZE` | Number of log records to buffer before flushing to MongoDB | `50` |
| `APP_LOG_MONGODB_FLUSH_INTERVAL_SECONDS` | Seconds between periodic flushes to MongoDB | `5.0` |
| `APP_LOG_LEVEL` | Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL | `INFO` |
| `APP_LOG_EXCLUDED_LOGGERS` | Comma-separated logger names to exclude from MongoDB writes | `uvicorn.access,httpx,pymongo,motor` |

## Deployment Configuration

### Docker Compose

Set parameters in your `.env` file:

```bash
# Enable centralized logging (default: true)
APP_LOG_CENTRALIZED_ENABLED=true

# Retain logs for 1 day (default: 1)
APP_LOG_CENTRALIZED_TTL_DAYS=1

# Optional overrides
APP_LOG_MAX_BYTES=52428800
APP_LOG_BACKUP_COUNT=5
APP_LOG_MONGODB_BUFFER_SIZE=50
APP_LOG_MONGODB_FLUSH_INTERVAL_SECONDS=5.0
APP_LOG_LEVEL=INFO
APP_LOG_EXCLUDED_LOGGERS=uvicorn.access,httpx,pymongo,motor
```

All `APP_LOG_*` variables are passed to both the `registry` and `auth-server` services in `docker-compose.yml`, `docker-compose.podman.yml`, and `docker-compose.prebuilt.yml`.

### Terraform / ECS

Set parameters in `terraform.tfvars`:

```hcl
# Enable centralized logging (default: true)
app_log_centralized_enabled = true

# Retain logs for 1 day (default: 1)
app_log_centralized_ttl_days = 1

# Optional overrides
app_log_max_bytes         = 52428800
app_log_backup_count      = 5
app_log_mongodb_buffer_size = 50
app_log_mongodb_flush_interval_seconds = 5.0
app_log_level             = "INFO"
app_log_excluded_loggers  = "uvicorn.access,httpx,pymongo,motor"
```

Variables are defined in `terraform/aws-ecs/variables.tf` and passed through to the ECS task definitions in `terraform/aws-ecs/modules/mcp-gateway/ecs-services.tf`. Both the registry and auth-server containers receive these environment variables.

### Helm / EKS

Set parameters in your values override file:

```yaml
registry:
  app:
    appLogCentralizedEnabled: "true"
    appLogCentralizedTtlDays: "1"
    appLogMaxBytes: "52428800"
    appLogBackupCount: "5"
    appLogMongodbBufferSize: "50"
    appLogMongodbFlushIntervalSeconds: "5.0"
    appLogLevel: "INFO"
    appLogExcludedLoggers: "uvicorn.access,httpx,pymongo,motor"
```

Configuration is managed via dedicated ConfigMaps (`registry-app-log-config` and `auth-server-app-log-config`), mounted using `envFrom` in the deployment templates.

In the umbrella chart (`mcp-gateway-registry-stack`), a YAML anchor (`&appLogConfig`) defines values once under the `registry.app` section and merges them into `auth-server.app` via `<<: *appLogConfig`, so you only need to set values in one place.

## Admin API Endpoints

All endpoints require admin authentication and are rate-limited to 10 requests per 60 seconds per user.

### Query Logs

```
GET /api/admin/logs
```

Query parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `service` | string | Filter by service name (e.g., `registry`, `auth-server`) |
| `level` | string | Minimum log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |
| `hostname` | string | Filter by pod/hostname |
| `search` | string | Substring search in message (max 200 chars, regex-escaped) |
| `start` | datetime | Start of time range (ISO 8601) |
| `end` | datetime | End of time range (ISO 8601) |
| `limit` | int | Max entries to return (1-10000, default 100) |
| `offset` | int | Number of entries to skip (default 0) |

### Export Logs

```
GET /api/admin/logs/export
```

Streams logs as newline-delimited JSON (JSONL) for download. Accepts the same filter parameters as the query endpoint, with a higher limit (up to 50,000 entries).

### Log Metadata

```
GET /api/admin/logs/metadata
```

Returns available filter values: service names, hostnames, and log levels.

## Log Viewer UI

Navigate to **Settings > Application Logs > Log Viewer** in the web UI.

Features:
- Filter by service, level, hostname, time range, and message content
- Click any row to expand and view the full log message
- Pagination (50 entries per page)
- Download filtered results as JSONL

## Observability

The `app_log_mongodb_flush_failures_total` Prometheus counter (labeled by `service`) tracks failed flush attempts. Use this metric to alert on write failures.

## Disabling Centralized Logging

To disable MongoDB log storage while keeping file-based rotation active:

```bash
APP_LOG_CENTRALIZED_ENABLED=false
```

When disabled, the admin API returns `503 Service Unavailable` and the Log Viewer UI shows an informational message. Local file rotation continues regardless of this setting.

## Prerequisites

Centralized logging requires:
- `STORAGE_BACKEND` set to `documentdb` or `mongodb-ce`
- A running MongoDB/DocumentDB instance accessible by both services
