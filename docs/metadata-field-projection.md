# Metadata Field Projection

The `metadata_fields` parameter lets API consumers request only specific fields from the `metadata` subdocument on servers, agents, and skills. This reduces payload size when metadata contains large nested objects that the caller does not need.

## How it works

Pass a comma-separated list of dot-notation paths as the `metadata_fields` query parameter (or body field on the search endpoint). Only the listed paths are returned; everything else in metadata is dropped. All fields outside metadata (name, description, tags, etc.) are always returned in full.

When `metadata_fields` is omitted, the REST endpoints return metadata in full. The mcpgw MCP tools differ: they return no metadata unless asked (see below).

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

## From the mcpgw MCP tools

The gateway's own MCP tools sit on top of these endpoints, so the same rules apply from an agent's point of view:

| Tool | Metadata behavior |
|------|-------------------|
| `list_services` / `list_agents` / `list_skills` | Return no metadata unless the caller passes `metadata_fields`; with it, return only the named keys |
| `search_registry` | Accepts a `metadata_fields` argument and forwards it; without it, results carry no metadata. This includes each agent hit's `agent_card.metadata`, which the semantic search endpoint itself still returns in full |

Metadata is opt-in on every model-facing tool. It is free-form text the registrant controls, and an MCP tool result lands directly in a model's context window, so a tool returns none of it until the caller names the keys it needs: `list_services(metadata_fields="owner_team")` or `search_registry(query="billing", metadata_fields="owner_team")`. A skill's metadata holds `author`, `version` and `extra`, with free-form keys under `extra`, so skill keys are named like `author,extra.owner_team`.

The REST endpoints keep their default and return full metadata when `metadata_fields` is omitted, because the registry's edit forms read it from there. Only the mcpgw tools change the default.

Requested keys are also subject to the metadata YARA scan: a key the scan flagged is removed even when `metadata_fields` names it, and a record without a current verdict returns no metadata to these tools. See [Registrant Metadata Scanning](metadata-security.md).

Key syntax is the same everywhere, and the tools inherit the endpoints' validation. Each key is a single word of letters, digits, `_` or `-`, with `.` separating levels of nesting, up to 20 keys and 5 levels deep. A key no asset happens to carry is simply absent from the response, so guessing a name is harmless. A malformed key is not: it is rejected with a 422, which fails the whole call.

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
| Allowed characters only | Letters (incl. accented/non-Latin), digits, `_`, `-` | `?metadata_fields=owner name` or `re@gion` |

## Performance

For large metadata blobs (>100KB), the server list endpoint projects metadata at the database level using a MongoDB aggregation pipeline. This avoids transferring the full blob from the database to the application. If the pipeline fails for any reason (e.g., older database engine), it falls back to Python-level projection transparently.

All other endpoints apply projection in Python after fetching the data. This is fast (microseconds per document) and correct for typical metadata sizes.

## Size and nesting limits

Every path that writes registrant-supplied metadata rejects a document that breaks either limit, with a 422 whose message names the limit, the limit's value and the measured value.

| Limit | Value | How it is measured |
|-------|-------|--------------------|
| Size | 10,240 bytes | The metadata serialized as compact JSON (no whitespace), encoded as UTF-8. A non-ASCII character counts its real UTF-8 width, for example 2 bytes for `é` and 4 for most emoji |
| Nesting | 8 levels | Each object or list is one level. The top-level metadata object is level 1, so `{"a": {"b": 1}}` is 2 levels deep |

There is no separate cap on the number of keys; the size limit bounds it.

Two malformed inputs are also rejected with a 422 on the form-based server routes: text that is not valid Unicode (an unpaired UTF-16 surrogate such as `"\ud800"`, which JSON allows but UTF-8 and MongoDB cannot store), and a metadata string nested so deeply that the JSON decoder cannot read it.

The limits apply on server registration (`POST /api/servers/register`, `POST /api/register`, `POST /api/internal/register`), server edits (`PUT` and `PATCH /api/servers/{path}`, `POST /api/edit/{path}`), agent registration and replacement (`POST /api/agents/register`, `PUT /api/agents/{path}`, and register and replace items in `POST /api/agents/batch`), and skill registration and replacement (`POST /api/skills`, `PUT /api/skills/{path}`). `PATCH /api/agents/{path}` does not accept metadata at all.

A skill's metadata is checked as it is stored: `author` and `version` lift out of the flat request and every other key moves under `extra`, which adds one level of nesting and a few bytes, and an explicit `version` field counts in place of `metadata.version`.

Records stored before these limits existed still load and list normally. The limits are checked when metadata is written, not when it is read, so an oversized record is only rejected the next time someone saves it.
