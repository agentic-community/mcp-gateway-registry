# ARD Catalog Publisher

The MCP Gateway & Registry publishes its public assets as an [Agentic Resource Discovery (ARD)](https://github.com/ards-project/ard-spec) catalog at a standard well-known location. This makes the registry a conformant **ARD Catalog Publisher**: any ARD-aware client or peer registry can discover the assets you expose, using a vendor-neutral, standards-track wire format.

## What is ARD?

Agentic Resource Discovery (ARD) is a specification (v1.0, `ards-project/ard-spec`) for advertising AI assets, MCP servers, A2A agents, and skills, through a static, crawlable manifest. It defines two roles:

- **Catalog Publisher** - hosts a static `/.well-known/ai-catalog.json` listing its resources.
- **Registry** - crawls and indexes catalogs and exposes a search API.

This feature implements the **Publisher** role. It is a format adapter over the registry's existing data: nothing new is stored, the catalog is rendered on demand from already-registered records.

## The endpoint

```
GET /.well-known/ai-catalog.json
```

- **Public** - no authentication. A `.well-known` manifest is crawled anonymously.
- **Read-only** - rendered live from the registry's records, with a `Cache-Control: public` header.
- **Scope** - lists only **public + enabled** MCP servers, A2A agents, and skills. Private, group-restricted, and disabled assets never appear.

Try it:

```bash
curl -sS https://registry.example.com/.well-known/ai-catalog.json | jq
```

## Example response

This is a trimmed example showing one entry of each type (MCP server, A2A agent, skill):

```json
{
  "specVersion": "1.0",
  "host": {
    "displayName": "AI Registry",
    "trustManifest": {
      "identity": "https://registry.example.com",
      "identityType": "https"
    }
  },
  "entries": [
    {
      "identifier": "urn:air:registry.example.com:server:realserverfaketools",
      "displayName": "Real Server Fake Tools",
      "type": "application/mcp-server-card+json",
      "url": "https://registry.example.com/api/public/servers/realserverfaketools/server.json",
      "description": "A collection of fake tools with interesting names that take different parameter types",
      "tags": ["demo", "fake", "tools", "testing"],
      "capabilities": [
        "quantum_flux_analyzer",
        "neural_pattern_synthesizer",
        "hyper_dimensional_mapper"
      ],
      "representativeQueries": ["demo tools", "fake tools", "testing tools"],
      "version": "v1.0.0",
      "updatedAt": "2026-06-19T01:07:00.841437Z"
    },
    {
      "identifier": "urn:air:registry.example.com:agent:travel-assistant-agent",
      "displayName": "Travel Assistant Agent",
      "type": "application/a2a-agent-card+json",
      "url": "https://registry.example.com/api/public/agents/travel-assistant-agent",
      "description": "Flight search and trip planning agent",
      "tags": ["travel", "flight-search", "trip-planning", "booking"],
      "capabilities": [
        "search_flights",
        "check_prices",
        "get_recommendations",
        "create_trip_plan"
      ],
      "representativeQueries": ["travel tools", "flight-search tools", "trip-planning tools"],
      "version": "0.0.1",
      "updatedAt": "2026-04-30T04:10:41.574701Z"
    },
    {
      "identifier": "urn:air:registry.example.com:skill:pdf",
      "displayName": "pdf",
      "type": "application/ai-skill",
      "url": "https://registry.example.com/api/public/skills/pdf",
      "description": "Create and manipulate PDF documents",
      "tags": ["pdf", "documents", "conversion"],
      "representativeQueries": ["pdf tools", "documents tools", "conversion tools"],
      "updatedAt": "2026-04-27T06:54:12.760604Z"
    }
  ]
}
```

## How records map to catalog entries

| ARD field | Source | Notes |
|-----------|--------|-------|
| `identifier` | `urn:air:<publisher>:<namespace>:<name>` | Namespace is `server` / `agent` / `skill`; name is the sanitized record path leaf. Validated against the ARD URN regex. |
| `displayName` | Server/agent name (or skill name) | |
| `type` | IANA media type per entity | `application/mcp-server-card+json`, `application/a2a-agent-card+json`, `application/ai-skill` |
| `url` | Public per-record endpoint | `/api/public/...` (see below). Phase 1 always uses `url`, never inline `data`. |
| `tags` | Record `tags` | |
| `capabilities` | Tool names (servers), skill names (agents), allowed-tool names (skills) | Capped at 50. |
| `representativeQueries` | Derived from tags + description | 2-5 items; omitted if fewer than 2 derivable (per schema bounds). |
| `updatedAt` | Record `updated_at` | Normalized to ISO 8601 UTC (`...Z`). |

### Public per-record endpoints

Because ARD clients dereference each entry's `url` **anonymously**, the catalog points at dedicated public, read-only endpoints rather than the registry's authenticated record endpoints:

```
GET /api/public/servers/{name}/server.json
GET /api/public/agents/{name}
GET /api/public/skills/{name}
```

These return only public + enabled records (404 for anything else, so the existence of non-public assets is never disclosed) and strip internal fields (backend URLs, auth schemes, group lists, owner). The original authenticated endpoints (`/api/servers/{path}/server.json`, etc.) are unchanged and still require auth.

## Identity and trust

The catalog is a **discovery + identity** surface, not an authorization surface. The `host.trustManifest` lets a client verify *who published* the catalog:

- `identity`: the registry's public issuer URL (e.g. `https://registry.example.com`).
- `identityType: https`: trust is established by the publisher's TLS certificate, the same way any HTTPS site is trusted.

**Authentication is delegated** (per the ARD spec): the catalog never carries credentials. To actually *use* a discovered resource, a client obtains a token via that resource's own protocol, for MCP servers, that is the existing RFC 9728 flow at `/.well-known/oauth-protected-resource`.

> **Note on `did:web` and signatures.** A stronger, cryptographically verifiable publisher identity (`host.identifier: did:web:<publisher>` plus a signed `trustManifest`) is intentionally **not** emitted today, because it requires publishing a signing key. That is planned as a follow-up; until then the registry asserts identity via the `https` trust manifest only, which is honest for an unsigned catalog.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ARD_CATALOG_ENABLED` | `true` | Enable/disable the `/.well-known/ai-catalog.json` endpoint. Returns `404` when disabled. |
| `ARD_PUBLISHER_DOMAIN` | *(derived)* | FQDN used as the URN publisher (`urn:air:<domain>:...`). Empty derives it from the host of `REGISTRY_URL`; falls back to `example.com` if no resolvable host is available (never `localhost`). |
| `ARD_CATALOG_DEFAULT_NAMESPACE` | *(entity type)* | Optional override for the URN namespace segment. Empty uses the entity type (`server` / `agent` / `skill`). |

These are wired across all three deployment surfaces (Docker Compose, Terraform/ECS, Helm/EKS) and appear on the admin **System Config** page under *Well-Known Discovery*. See [`docs/unified-parameter-reference.md`](unified-parameter-reference.md) for the cross-surface mapping.

## Conformance

The rendered manifest validates against the ARD `ai-catalog.schema.json` and passes the spec's `conformance-test manifest` CLI with **zero critical errors**.

Skill entries use `application/ai-skill` (matching the issue and the spec examples); the conformance tool emits a non-fatal *warning* for this type because its current allowlist does not include it. This does not affect the PASS status, and the media type is a single constant that can be updated if the working group finalizes a different value.

## What's not included (Phase 1)

This is Phase 1 of a three-phase ARD effort. Out of scope here:

- **`POST /search` adapter** (ARD Registry search API) - Phase 2. This is where an *authenticated* caller would receive results scoped to assets they can access; the static catalog itself is public-only by design.
- **`ai-catalog.json` crawling / federation** - Phase 3.
- **Signed (detached JWS) trust manifests and `did:web`** - signing follow-up.

## References

- [ARD specification](https://github.com/ards-project/ard-spec)
- [RFC 8141 - URN syntax](https://www.rfc-editor.org/rfc/rfc8141)
- [Well-known discovery endpoints](api-reference.md)
