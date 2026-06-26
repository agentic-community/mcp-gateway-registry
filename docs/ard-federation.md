# ARD Federation: ai-catalog.json Ingestion + federation modes + domain-anchored trust

This is **Phase 3** of the registry's Agentic Resource Discovery (ARD) support. It builds on the
ARD Catalog Publisher (Phase 1, `/.well-known/ai-catalog.json`) and the ARD Registry adapter
(Phase 2, `POST /api/ard/search` + `GET /api/ard/agents`, see [ard-registry.md](ard-registry.md)).

Phase 3 makes the registry a **federating** ARD Registry:

1. **Web ingestion** — crawl external `ai-catalog.json` documents (and a domain's `.well-known`),
   validate them, and index their entries into the local search index.
2. **Federation modes** on `POST /api/ard/search` — `none` / `auto` / `referrals`.
3. **Domain-anchored trust** — verify each ingested entry's publisher identity.

## How it works

The registry already syncs peer-registry items **into its own index** (background pull-sync,
origin-tagged, read-only, orphan-tracked). Phase 3 adds the ARD `ai-catalog.json` crawler as one
more feeder into that same index, then exposes read-side federation filters over it. **Search never
makes a per-request network call to a peer** — it always queries the local unified index.

```
External ai-catalog.json ──crawl+validate+trust-gate──┐
Peer ARD registries ──────background pull-sync────────►│  unified local index
                                                       │  (local + synced + ingested)
                                                       ▼
                          POST /api/ard/search?federation=none|auto|referrals
```

## Configuring ingestion sources

Ingestion sources live in the **DB-backed federation config** (`FederationConfig.ai_catalog`),
exactly like the Anthropic / ASOR / AWS Agent Registry upstreams. Manage them three ways:

### 1. UI

**Settings → Federation → External Registries → ARD Catalog (ai-catalog.json) → Add.**
Provide a `source_id` and either a `uri` or a `domain` (and optionally an `expected_identity`
trust pin).

### 2. Federation API (one source at a time)

```bash
# Add a source
curl -X POST "$REGISTRY_URL/api/federation/config/default/ai_catalog/sources" \
  -H "Authorization: Bearer $ADMIN_TOKEN" -H "Content-Type: application/json" \
  -d '{ "source_id": "acme", "uri": "https://acme.com/.well-known/ai-catalog.json",
        "expected_identity": "https://acme.com" }'

# Remove a source
curl -X DELETE "$REGISTRY_URL/api/federation/config/default/ai_catalog/sources/acme" \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Trigger ingestion now (all sources, or ?source_id=acme)
curl -X POST "$REGISTRY_URL/api/federation/ai_catalog/sync" -H "Authorization: Bearer $ADMIN_TOKEN"

# Per-source ingestion status
curl "$REGISTRY_URL/api/federation/ai_catalog/status" -H "Authorization: Bearer $ADMIN_TOKEN"
```

### 3. Full federation config (the `ai_catalog` block)

`POST /api/federation/config` accepts the whole config; the new block:

```json
{
  "ai_catalog": {
    "enabled": true,
    "sync_on_startup": false,
    "sync_interval_minutes": 60,
    "max_depth": 3,
    "fetch_timeout_seconds": 15,
    "polite_interval_ms": 200,
    "same_domain_only": true,
    "trust_enforcement": "reject",
    "sources": [
      { "source_id": "acme", "uri": "https://acme.com/.well-known/ai-catalog.json",
        "expected_identity": "https://acme.com" },
      { "source_id": "foo", "domain": "foo.com" }
    ]
  }
}
```

The behavior knobs are **block-level** (shared by all sources) with the defaults shown above; each
entry in `sources[]` carries only identity. There are **no per-knob environment variables** — this
mirrors `aws_registry` exactly.

### CLI

```bash
uv run python api/registry_management.py ard-ingestion-sync   --registry-url "$REGISTRY_URL" --token-file ../.token
uv run python api/registry_management.py ard-ingestion-status --registry-url "$REGISTRY_URL" --token-file ../.token
```

## Federation modes on `POST /api/ard/search`

| `federation` | Behavior |
|--------------|----------|
| `none` | Local-origin items only — excludes synced-peer and ingested items. The opt-out. |
| `auto` (default) | The whole unified index. Each result's `source` is its origin registry (the peer's ARD endpoint or the catalog URI); local items keep this registry's search URI. |
| `referrals` | Local-origin items only, plus a `referrals[]` array of `application/ai-registry+json` pointers to peer registries. |

**Scores are per-source.** Each registry computes relevance independently, so a result's 0-100
`score` is comparable only within its own `source`. Results are ordered by `(score desc,
identifier asc)`; under `none`/`referrals` a page near the tail of the local ranking may be shorter
than `pageSize` (the engine is over-fetched to mitigate this, but it is not a guarantee) — page by
`pageToken` until it is `null`.

Access-scoping (OAuth2/scopes) is applied to **every** result regardless of mode; federation only
controls *which* items are eligible, never *who* may see them.

## Domain-anchored trust

For each ingested entry, the publisher FQDN is extracted from its URN
(`urn:air:<publisher>:...`) and cross-checked against the catalog host's `trustManifest.identity`
(proven over TLS). `trust_enforcement` decides what happens on a mismatch:

- `reject` (default) — the entry is not indexed (counted in `mcpgw_ard_trust_mismatch_total`).
- `flag` — the entry is indexed but annotated with the mismatch reason.
- `off` — the check is disabled.

This proves *publisher identity*; it is additive to the OAuth2/scope model, which still gates
*access*. Signed `trustManifest` (detached JWS) is a separate follow-up — the spec marks
`signature` optional, so domain anchoring alone is conformant.

## Security: SSRF protection

Every fetch (the root source and every nested `application/ai-catalog+json` URL) passes an SSRF
guard before any request is made:

- **https only** (no `http`/`file`/...).
- **Post-DNS IP check** — the host must resolve only to public IPs; private / loopback /
  link-local / reserved / cloud-metadata (`169.254.169.254`) addresses are refused (defeats
  DNS-rebinding).
- **Same-domain recursion** (`same_domain_only`, default on) — nested catalogs must stay on the
  root source's registrable domain.
- **Size + timeout caps** — documents over 5 MB or past the fetch timeout are skipped.

Outbound catalog fetches carry **no** `Authorization` header, so a peer/federation token can never
leak to a third-party catalog host.

## Operational notes

- **Disabled by default** (`ai_catalog.enabled = false`). No behavior change for existing
  deployments until you add a source and enable it.
- **Single-scheduler guidance** — run the ingestion scheduler on one replica (parity with peer
  sync); a per-source in-process lock prevents overlapping runs within a replica.
- **Ingested items are read-only and non-connectable** (`record_kind = "ard_ingested"`), tagged
  `is_federated` so they show the "federated" badge in the dashboard and are **never re-published**
  in this registry's own `/.well-known/ai-catalog.json`.
- **Outbound egress** — the registry must be allowed to make outbound HTTPS (443) to the catalog
  hosts you configure.
- **All three asset types are ingested** — MCP servers, A2A agents (via the peer-sync storage
  path), and skills (via the skill repository, stored at `/skills/{source_id}/{leaf}`). Each type
  is origin-tagged, read-only, and orphan-reconciled per source.

## Metrics

| Metric | Type | Labels |
|--------|------|--------|
| `mcpgw_ard_ingestion_runs_total` | counter | `source_id`, `status` |
| `mcpgw_ard_ingestion_entries_total` | counter | `source_id`, `outcome` (indexed/rejected/orphaned) |
| `mcpgw_ard_ingestion_duration` | histogram | `source_id` |
| `mcpgw_ard_trust_mismatch_total` | counter | `source_id`, `policy` |
| `mcpgw_ard_requests_total` | counter | `operation`, `status`, `federation` |
