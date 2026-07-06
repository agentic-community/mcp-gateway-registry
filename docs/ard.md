# Agentic Resource Discovery (ARD)

The MCP Gateway & Registry implements the [Agentic Resource Discovery (ARD) v1.0
spec](https://github.com/ards-project/ard-spec) end to end — as a **Publisher**, a
**Registry**, and a **federating peer** — so any ARD-aware client, agent, or registry can
discover, search, and cross-reference its assets through vendor-neutral, standards-track
interfaces.

ARD defines two roles, both implemented here, plus federation between registries:

| Role | What it does | Surface |
|------|--------------|---------|
| **Publisher** | Hosts a static, crawlable `/.well-known/ai-catalog.json` of its public assets | anonymous, public |
| **Registry** | Exposes a search/browse API over its catalog | `POST /api/ard/search`, `GET /api/ard/agents` (JWT, access-scoped) |
| **Federation** | Ingests other registries' catalogs and references peers | `federation=none\|auto\|referrals` + ai-catalog ingestion |

Everything is a **format adapter over existing registry data** — no new storage for the
Publisher/Registry roles; federation ingests into the existing collections, clearly marked.

---

## 1. Publisher — `/.well-known/ai-catalog.json`

A conformant, anonymous ARD catalog rendered live from the registry's records.

```
GET /.well-known/ai-catalog.json        # public, no auth, Cache-Control: public
```

- Lists only **public + enabled** MCP servers, A2A agents, and skills. Private,
  group-restricted, and disabled assets never appear.
- Each entry is an ARD `catalogEntry` with a domain-anchored URN
  (`urn:air:<publisher>:<namespace>:<name>`), the correct IANA media type
  (`application/mcp-server-card+json`, `application/a2a-agent-card+json`,
  `application/ai-skill`), derived `representativeQueries`/`capabilities`, and an `https`
  `trustManifest`.
- Entry `url`s point at anonymous, read-only per-record endpoints that strip internal
  fields (backend URLs, auth schemes, group lists, owner) and 404 for anything non-public:

  ```
  GET /api/public/servers/{name}/server.json
  GET /api/public/agents/{name}
  GET /api/public/skills/{name}
  ```

Example (trimmed to one entry):

```json
{
  "specVersion": "1.0",
  "host": {
    "displayName": "AI Registry",
    "trustManifest": { "identity": "https://registry.example.com", "identityType": "https" }
  },
  "entries": [
    {
      "identifier": "urn:air:registry.example.com:server:realserverfaketools",
      "displayName": "Real Server Fake Tools",
      "type": "application/mcp-server-card+json",
      "url": "https://registry.example.com/api/public/servers/realserverfaketools/server.json",
      "description": "A collection of fake tools …",
      "tags": ["demo", "fake", "tools"],
      "capabilities": ["quantum_flux_analyzer", "neural_pattern_synthesizer"],
      "representativeQueries": ["demo tools", "fake tools"],
      "version": "v1.0.0",
      "updatedAt": "2026-06-19T01:07:00.841437Z"
    }
  ],
  "host_self_entry_note": "the catalog also carries a self application/ai-registry+json entry pointing at /api/ard"
}
```

**Identity & trust (discovery, not authorization).** `host.trustManifest.identity` lets a
client verify *who published* the catalog (via the publisher's TLS certificate). The catalog
never carries credentials — to *use* a discovered resource a client authenticates via that
resource's own protocol (for MCP servers, the RFC 9728 flow at
`/.well-known/oauth-protected-resource`). A cryptographically signed `trustManifest` (detached
JWS) / `did:web` is an optional, additive follow-up — the spec marks `signature` optional, so
the `https` trust manifest alone is conformant.

The rendered manifest validates against `ai-catalog.schema.json` and passes the spec's
`conformance-test manifest` CLI with zero critical errors.

---

## 2. Registry — search & browse (`/api/ard/*`)

The catalog is also queryable through ARD's HTTP search/browse contract. Both endpoints are
mounted under `/api/ard`, are **JWT-required**, and are **access-scoped** — a non-admin caller
sees a strict subset, never assets outside its grants. (The static `ai-catalog.json` remains
the anonymous public-only surface.)

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/ard/search` | ARD `SearchRequest` → `SearchResponse` (hybrid semantic search) |
| `GET`  | `/api/ard/agents` | ARD `ListResponse` browse over **all** asset types |

```json
// POST /api/ard/search  (additionalProperties: false)
{
  "query": { "text": "financial data tools", "filter": { "type": ["mcp_server"], "tags": ["finance"] } },
  "federation": "auto",
  "pageSize": 10,
  "pageToken": null
}
```

- `query.filter` supports `type`/`entity_type` (`mcp_server` | `a2a_agent` | `skill`, or their
  media-type strings) and `tags` — values OR within a key, AND across keys.
- Each result is a full `catalogEntry` plus `score` (integer **0–100**, rescaled from the
  internal 0–1 relevance) and `source` (the origin registry).
- `pageToken` is an opaque base64 cursor; `GET /api/ard/agents` browses all asset types in
  deterministic order with `filter` / `orderBy` / `pageSize` / `pageToken`.

**Errors** use the ARD envelope `{ "errorCode": ..., "message": ... }` with codes
`UNAUTHENTICATED` (401) / `INVALID_REQUEST` (400/422) / `RATE_LIMITED` (429) / `NOT_FOUND`
(404) / `INTERNAL` (500). Messages are static and client-safe; an unauthenticated request
returns a clean ARD `401` (produced by a dedicated `^~ /api/ard/` nginx location), never a
login redirect.

```bash
# CLI
uv run python api/registry_management.py ard-search \
  --token-file .token --registry-url http://localhost \
  --query "financial data" --filter type=mcp_server --filter tags=finance --federation auto
uv run python api/registry_management.py ard-agents \
  --token-file .token --registry-url http://localhost --order-by identifier
```

The existing `POST /api/search/semantic` contract is unchanged — the ARD adapter wraps the
same engine.

---

## 3. Federation — ingest & cross-reference other registries

The registry can **federate** with other registries. There are two complementary ways to do
it; pick based on what you want.

### Adding another registry as a peer — which mechanism?

| | **A. ARD ingestion source** | **B. Federation peer (mesh)** |
|---|---|---|
| For | Any ARD-conformant registry/publisher (any vendor) | Another **mcp-gateway-registry** instance you trust |
| Pulls | Its public `ai-catalog.json` (discovery metadata) | Full records via `/api/federation/*` |
| Auth | None (public catalog) | A federation token from the peer |
| Result | Read-only **discovery** entries in your index, with "view at source" links | Fully connectable synced servers/agents + `referrals[]` pointers |
| Configure in | **Settings → Federation → External Registries → ARD Catalog** | **Settings → Federation → Peers** / `peer-add` |

> **Rule of thumb:** to *discover* what another ARD registry offers, add it as an **ARD
> ingestion source (A)**. To *connect to and use* another mcp-gateway-registry's assets through
> your gateway, add it as a **federation peer (B)**.

#### A. Add an external ARD registry as an ingestion source

This crawls the registry's `ai-catalog.json`, validates it, trust-gates each entry, and indexes
servers, agents, and skills into your unified local index as **read-only discovery** records.
Sources live in the DB-backed federation config (`FederationConfig.ai_catalog`) — exactly like
the Anthropic/ASOR/AWS upstreams, with **no per-knob env vars**.

**UI:** Settings → Federation → External Registries → **ARD Catalog (ai-catalog.json)** → Add.
Provide a `source_id` and either a `uri` or a `domain` (and optionally an `expected_identity`
trust pin).

**CLI:**

```bash
RM="uv run python api/registry_management.py --registry-url http://localhost --token-file .token"

# Add the registry as an ARD ingestion source (and enable ingestion)
$RM ard-ingestion-add-source --source-id acme \
  --uri "https://acme.com/.well-known/ai-catalog.json" \
  --expected-identity "https://acme.com" --enable
# (or: --domain acme.com  →  resolves https://acme.com/.well-known/ai-catalog.json)

$RM ard-ingestion-sync   --source-id acme     # crawl + index now
$RM ard-ingestion-status                      # per-source generation, counts, failures
$RM ard-ingestion-remove-source --source-id acme   # stop ingesting + remove the source
```

**API:**

```bash
curl -X POST "$REGISTRY_URL/api/federation/config/default/ai_catalog/sources" \
  -H "Authorization: Bearer $ADMIN_TOKEN" -H "Content-Type: application/json" \
  -d '{ "source_id": "acme", "uri": "https://acme.com/.well-known/ai-catalog.json",
        "expected_identity": "https://acme.com" }'
curl -X POST   "$REGISTRY_URL/api/federation/ai_catalog/sync?source_id=acme"  -H "Authorization: Bearer $ADMIN_TOKEN"
curl           "$REGISTRY_URL/api/federation/ai_catalog/status"               -H "Authorization: Bearer $ADMIN_TOKEN"
curl -X DELETE "$REGISTRY_URL/api/federation/config/default/ai_catalog/sources/acme" -H "Authorization: Bearer $ADMIN_TOKEN"
```

Block-level behavior knobs (shared by all sources, defaults shown) plus identity-only
`sources[]` live in the `ai_catalog` config block:

```json
{
  "ai_catalog": {
    "enabled": true, "sync_on_startup": false, "sync_interval_minutes": 60,
    "max_depth": 3, "fetch_timeout_seconds": 15, "polite_interval_ms": 200,
    "same_domain_only": true, "trust_enforcement": "reject",
    "sources": [
      { "source_id": "acme", "uri": "https://acme.com/.well-known/ai-catalog.json", "expected_identity": "https://acme.com" },
      { "source_id": "foo", "domain": "foo.com" }
    ]
  }
}
```

#### B. Add another mcp-gateway-registry as a federation peer

The peer mesh pulls **full** records (connectable servers/agents) from another
mcp-gateway-registry over `/api/federation/*` using a federation token, and those peers are
what `federation=referrals` points to. Add one with the Peers UI or the CLI:

```bash
uv run python api/registry_management.py peer-add \
  --config peer.json --federation-token "<token-from-the-remote-peer>"
# peer.json: { "peer_id": "lob-1", "name": "LOB 1", "endpoint": "https://lob1.example.com", "enabled": true, ... }
```

See **[federation.md](federation.md)** and **[federation-operational-guide.md](federation-operational-guide.md)**
for the full peer-mesh setup (token issuance, sync modes, orphan handling).

### Federation modes on `POST /api/ard/search`

The mesh and ingestion both feed the **unified local index**, so federated search is one local
query (no per-request network call to a peer). The `federation` parameter selects what's
eligible:

| `federation` | Behavior |
|--------------|----------|
| `none` | Local-origin items only — excludes synced-peer and ingested items. The opt-out. |
| `auto` (default) | The whole unified index. Each result's `source` is its origin registry; federated results resolve to the **source's** original `url`/`identifier` so a client can dereference the peer's descriptor and connect at the source. |
| `referrals` | Local-origin results plus a `referrals[]` array of `application/ai-registry+json` pointers to peer registries. |

**Scores are per-source** (each registry computes relevance independently), so a `score` is
comparable only within its own `source`. Under `none`/`referrals` a tail page may be shorter
than `pageSize` (the engine is over-fetched to mitigate, not guarantee) — page by `pageToken`
until `null`. **Access-scoping applies to every result regardless of mode** — federation
controls *which* items are eligible, never *who* may see them.

### Domain-anchored trust

Each ingested entry's publisher FQDN (from its `urn:air:<publisher>:...`) is anchored to the
**operator-configured source identity** — the `expected_identity` pin, else the configured
`domain`, else the host of the configured `uri`. This prevents a configured source from
publishing entries that impersonate another publisher (e.g. a catalog served from `acme.com`
declaring `victim.com` URNs is rejected). `trust_enforcement` decides the action on a mismatch:

- `reject` (default) — the entry is not indexed (counted in `mcpgw_ard_trust_mismatch_total`).
- `flag` — indexed but annotated with the mismatch reason.
- `off` — check disabled.

Trust is **additive** to the OAuth2/scope model — it proves *publisher identity*; access is
still gated by access-scoping.

### Security: SSRF protection on the crawler

Every fetch (the root source and every nested `application/ai-catalog+json` URL) passes an SSRF
guard before any request:

- **https only** (no `http`/`file`/...).
- **Post-DNS IP block** — the host must resolve only to public IPs; private / loopback /
  link-local / reserved / unspecified (`0.0.0.0`) / cloud-metadata (`169.254.169.254`) and
  IPv4-mapped-IPv6 targets are refused.
- **Same-domain recursion** (`same_domain_only`, default on) — nested catalogs stay on the root
  source's registrable domain.
- **Streamed size + timeout caps** — the body is streamed and aborted once it exceeds 5 MB
  (with an early `Content-Length` check); fetches past the timeout are skipped.
- **No auth header** on outbound catalog fetches — a peer/federation token can never leak to a
  third-party host; redirects are not followed.

### How clients use discovered assets (discover → resolve → connect)

ARD separates *finding* a resource from *connecting* to it; the registry is a directory, not a
proxy. A client: (1) **discovers** via search/catalog → gets `identifier`, `type`, `url`,
`source`; (2) **resolves** the entry's `url` → the full artifact descriptor (the MCP server
card / A2A agent card); (3) **connects** directly to the resource's own endpoint from that
descriptor, with whatever auth it requires. For ARD-discovered (federated) assets this registry
returns the **source** `url`/`identifier`, and the UI surfaces a **"View at source"** link to
the peer's descriptor.

> ARD-discovered assets are **read-only and non-connectable through this gateway**
> (`record_kind = "ard_ingested"`), are origin-tagged, and are **never re-published** in this
> registry's own `/.well-known/ai-catalog.json`. To make another registry's servers connectable
> *through your gateway*, use the federation peer mesh (option B above).

### Operational notes

- **Disabled by default** (`ai_catalog.enabled = false`) — zero behavior change until you add a
  source and enable it.
- **Single-scheduler guidance** — run the ingestion scheduler on one replica (parity with peer
  sync); a per-source in-process lock prevents overlapping runs within a replica.
- **Outbound egress** — the registry must be allowed outbound HTTPS (443) to the catalog hosts
  you configure.

---

## Configuration

ARD Publisher/Registry toggles are environment variables (wired across Docker, Terraform/ECS,
Helm, and the admin **System Config** page). ARD **federation** ingestion has **no env vars** —
its sources + behavior live in the DB federation config (managed via the UI/API above).

| Variable | Default | Description |
|----------|---------|-------------|
| `ARD_CATALOG_ENABLED` | `true` | Enable `/.well-known/ai-catalog.json` (Publisher). |
| `ARD_REGISTRY_ENABLED` | `true` | Enable `/api/ard/*` + the self `ai-registry` catalog entry. |
| `ARD_PUBLISHER_DOMAIN` | *(derived)* | FQDN for URN publisher (`urn:air:<domain>:...`). Empty derives from `REGISTRY_URL`'s host; falls back to `example.com` (never `localhost`). |
| `ARD_CATALOG_DEFAULT_NAMESPACE` | *(entity type)* | Optional URN namespace override (default `server`/`agent`/`skill`). |

See [`docs/unified-parameter-reference.md`](unified-parameter-reference.md) for the cross-surface
mapping.

---

## Observability

| Metric | Type | Labels |
|--------|------|--------|
| `mcpgw_ard_requests_total` | counter | `operation` (search/browse), `status`, `federation` |
| `mcpgw_ard_request_duration` | histogram | `operation` |
| `mcpgw_ard_results_returned` | histogram | `operation` |
| `mcpgw_ard_access_filtered_total` | counter | `operation` (entries removed by access-scoping) |
| `mcpgw_ard_errors_total` | counter | `operation`, `error_code` |
| `mcpgw_ard_ingestion_runs_total` | counter | `source_id`, `status` |
| `mcpgw_ard_ingestion_entries_total` | counter | `source_id`, `outcome` (indexed/rejected/orphaned) |
| `mcpgw_ard_ingestion_duration` | histogram | `source_id` |
| `mcpgw_ard_trust_mismatch_total` | counter | `source_id`, `policy` |

---

## References

- [ARD specification](https://github.com/ards-project/ard-spec)
- [Federation (peer mesh) guide](federation.md) · [Federation operational guide](federation-operational-guide.md)
- [RFC 8141 — URN syntax](https://www.rfc-editor.org/rfc/rfc8141) · [RFC 9728 — OAuth protected resource](https://www.rfc-editor.org/rfc/rfc9728)
