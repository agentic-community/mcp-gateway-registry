# Metadata Field Projection

The `metadata_fields` parameter lets API consumers request only specific fields from the `metadata` subdocument on servers, agents, and skills. This reduces payload size when metadata contains large nested objects that the caller does not need.

## How it works

Pass a comma-separated list of dot-notation paths as the `metadata_fields` query parameter (or body field on the search endpoint). Only the listed paths are returned; everything else in metadata is dropped. All fields outside metadata (name, description, tags, etc.) are always returned in full.

When `metadata_fields` is omitted, responses are unchanged from their existing behavior.

## Supported endpoints

| Endpoint | Parameter location |
|----------|-------------------|
| `GET /api/servers` | Query parameter |
| `GET /api/servers/{path}` | Query parameter |
| `GET /api/agents` | Query parameter |
| `GET /api/agents/{path}` | Query parameter |
| `GET /api/skills` | Query parameter |
| `GET /api/skills/{path}` | Query parameter |
| `POST /api/search/semantic` | Request body field |

## Examples

### Basic projection

```bash
# Return only the "owner" metadata field
curl "http://localhost/api/servers?metadata_fields=owner" \
  -H "Authorization: Bearer $TOKEN"
```

Response metadata:
```json
{"owner": "team-platform"}
```

### Nested paths (dot-notation)

```bash
# Return owner and the region nested inside config
curl "http://localhost/api/servers?metadata_fields=owner,config.region" \
  -H "Authorization: Bearer $TOKEN"
```

Response metadata:
```json
{"owner": "team-platform", "config": {"region": "us-east-1"}}
```

### Repeated query parameters

Both formats are supported and can be mixed:

```bash
# Comma-separated
?metadata_fields=owner,config.region

# Repeated params
?metadata_fields=owner&metadata_fields=config.region

# Mixed
?metadata_fields=owner,config.region&metadata_fields=limits.rps
```

### Semantic search

```bash
curl -X POST "http://localhost/api/search/semantic" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "deployment tools",
    "metadata_fields": "owner,config.region"
  }'
```

### CLI

```bash
registry_management.py list --metadata-fields "owner,config.region" --json
registry_management.py agent-list --metadata-fields "team" --json
registry_management.py server-search --query "tools" --metadata-fields "owner"
```

## Behavior

| Scenario | Result |
|----------|--------|
| `metadata_fields` omitted | Full metadata returned (no change) |
| `metadata_fields=owner` | Only `owner` in metadata |
| `metadata_fields=config.region` | Only `config: {region: ...}` in metadata |
| `metadata_fields=nonexistent` | Empty metadata `{}` (no error) |
| `metadata_fields=config,config.region` | Ancestor wins: full `config` subtree returned |
| Path into a scalar (`owner.sub` where owner is a string) | Treated as missing, no value |

## Validation

Invalid input returns HTTP 422 with a descriptive error message.

| Constraint | Limit | Example rejection |
|------------|-------|-------------------|
| Maximum paths | 20 | `?metadata_fields=a,b,c,...` (21 fields) |
| Maximum depth | 5 levels | `?metadata_fields=a.b.c.d.e.f` |
| Segment length | 64 characters | A single segment exceeding 64 chars |
| No `$` prefix | Any segment | `?metadata_fields=$set.injection` |
| No empty segments | Between dots | `?metadata_fields=config..region` |

## Performance

For large metadata blobs (>100KB), the server list endpoint projects metadata at the database level using a MongoDB aggregation pipeline. This avoids transferring the full blob from the database to the application. If the pipeline fails for any reason (e.g., older database engine), it falls back to Python-level projection transparently.

All other endpoints apply projection in Python after fetching the data. This is fast (microseconds per document) and correct for typical metadata sizes.
