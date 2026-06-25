# ARD Registry Adapter

This registry is a conformant **ARD (Agent Registry Discovery) Registry**: it exposes
its catalog through ARD's standard HTTP search and browse interface, so any ARD-aware
client, agent, or peer registry can query it the same way it queries any other registry.

This is the **Registry** half of ARD. The **Publisher** half (the static
`/.well-known/ai-catalog.json` document) is described in
[`ard-catalog-publisher.md`](./ard-catalog-publisher.md). The published catalog links to
this adapter via a self entry of type `application/ai-registry+json`.

## Endpoints

Both endpoints are mounted under `/api/ard` and are **JWT-required**; results are
**access-scoped** to what the authenticated caller can see (the same visibility rules as
the rest of the registry). The static `ai-catalog.json` remains the anonymous,
public-only discovery surface.

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/ard/search` | ARD `SearchRequest` → `SearchResponse` (semantic search) |
| `GET`  | `/api/ard/agents` | ARD `ListResponse` browse over **all** asset types |

The ARD "Registry base URL" advertised in `ai-catalog.json` is `<registry_url>/api/ard`,
so a conformance tool hits `<base>/search` and `<base>/agents`.

### POST /api/ard/search

Request body (`additionalProperties: false`):

```json
{
  "query": { "text": "financial data tools", "filter": { "type": ["mcp_server"], "tags": ["finance"] } },
  "federation": "auto",
  "pageSize": 10,
  "pageToken": null
}
```

- `query.text` — required natural-language query.
- `query.filter` — dot-path field → string|array. Supported keys: `type` / `entity_type`
  (`mcp_server` | `a2a_agent` | `skill`, or their media-type strings) and `tags`. Values
  OR within a key, AND across keys.
- `federation` — `auto` | `referrals` | `none`. **Phase 2** returns own-index results for
  all modes; `referrals` is empty because no ARD peers are configured. Cross-registry
  querying/merging is Phase 3 (issue #1296).
- `pageSize` — 1–100 (default 10). `pageToken` — opaque cursor from a prior response.

Each result is a full `catalogEntry` plus `score` (integer **0–100**, rescaled from the
internal 0–1 relevance) and `source` (the search endpoint URI).

### GET /api/ard/agents

Browses **all** catalog asset types — MCP servers, A2A agents, and skills — not agents
only. Query params: `filter` (repeatable `key=value`, same keys as search), `orderBy`
(`identifier` | `displayName` | `updatedAt`, default `identifier`), `pageSize` (1–100,
default 20), `pageToken`. Deterministic order, no relevance ranking. Returns
`{ "items": [...], "total": N, "pageToken": ... }`.

## Errors

All `/api/ard/*` errors use the ARD envelope:

```json
{ "errorCode": "INVALID_REQUEST", "message": "The request was invalid." }
```

`errorCode` is one of `UNAUTHENTICATED` (401), `INVALID_REQUEST` (400/422),
`RATE_LIMITED` (429), `NOT_FOUND` (404), `INTERNAL` (500). Messages are static and
client-safe; exception detail goes to the server logs only. An unauthenticated request
returns a clean `401` ARD envelope, never a login redirect.

> Operator note: an unauthenticated request is rejected by nginx's `auth_request`
> gate before it reaches the app, so the `401` ARD envelope is produced by a dedicated
> `^~ /api/ard/` location (`error_page 401 = @ard_auth_error`) in
> `docker/nginx_rev_proxy_*.conf`. If you change nginx routing for `/api/ard/*`, keep
> that location (and its error handler) so the 401 stays ARD-shaped.

## CLI

The management CLI exposes both endpoints:

```bash
# Search
uv run python api/registry_management.py ard-search \
  --token-file .oauth-tokens/ingress.json --registry-url http://localhost \
  --query "financial data" --filter type=mcp_server --filter tags=finance --page-size 5

# Browse (all asset types; narrow with --filter type=...)
uv run python api/registry_management.py ard-agents \
  --token-file .oauth-tokens/ingress.json --registry-url http://localhost \
  --order-by identifier --page-size 20
```

Add `--json` for the raw ARD response. The same calls are available programmatically via
`RegistryClient.ard_search(...)` and `RegistryClient.ard_browse(...)` in
`api/registry_client.py`.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ARD_REGISTRY_ENABLED` | `true` | Enable the adapter and the self `ai-registry` catalog entry. When `false`, `/api/ard/*` returns `404` and the self entry is omitted. Independent of `ARD_CATALOG_ENABLED` (the publisher). |

## Observability

Every ARD operation emits dedicated, bounded-cardinality metrics (in addition to the
generic request middleware):

| Metric | Labels |
|--------|--------|
| `mcpgw_ard_requests_total` | `operation` (`search`/`browse`), `status`, `federation` |
| `mcpgw_ard_request_duration` | `operation` |
| `mcpgw_ard_results_returned` | `operation` |
| `mcpgw_ard_access_filtered_total` | `operation` (entries removed by access-scoping) |
| `mcpgw_ard_errors_total` | `operation`, `error_code` |

The two histograms are exposed in Prometheus as `_bucket`/`_count`/`_sum` series, and the
duration one carries a `_milliseconds` unit suffix (`mcpgw_ard_request_duration_milliseconds_*`).
`mcpgw_ard_access_filtered_total` stays at zero for admin callers (nothing is filtered) and
increments for access-scoped users.

## Notes

- The existing `POST /api/search/semantic` contract is unchanged; the ARD adapter wraps
  the same engine and does not replace it. The internal relevance score there stays 0–1,
  while the ARD adapter reports 0–100.
- Pagination is an opaque base64 offset cursor over a materialized, deterministically
  ordered result set; deep pagination is bounded by the search engine's top-N.
