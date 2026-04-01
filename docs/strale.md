# Strale Integration

Strale is trust and quality infrastructure for AI agents. It provides 250+ quality-scored data capabilities — IBAN validation, company registry lookups across 27 countries, sanctions screening, web scraping, lead enrichment, Web3 risk analysis — each continuously tested and assigned a dual-profile quality score (0-100). Every execution returns structured JSON with data provenance and a compliance-ready audit trail.

## Architecture

```
┌──────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│              │     │                      │     │                  │
│   AI Agent   │────▶│  MCP Gateway Registry│────▶│  Strale MCP      │
│              │     │                      │     │  api.strale.io   │
└──────────────┘     └──────────────────────┘     └──────────────────┘
                            │                            │
                            │ A2A                        │ 250+ capabilities
                            ▼                            ▼
                     ┌──────────────┐            ┌──────────────────┐
                     │ Strale A2A   │            │ Company regs,    │
                     │ Agent        │            │ sanctions lists, │
                     │ /a2a         │            │ VIES, DNS, etc.  │
                     └──────────────┘            └──────────────────┘
```

Strale is available through two protocols:
- **MCP Server** at `https://api.strale.io/mcp` (Streamable HTTP transport)
- **A2A Agent** at `https://api.strale.io/a2a` (JSON-RPC, `message/send` and `tasks/get`)

## Prerequisites

1. **Strale API key** — Sign up at [strale.dev/signup](https://strale.dev/signup). New accounts get EUR 2.00 in free trial credits, no card required.
2. A running MCP Gateway Registry instance.

## Register as MCP Server

```bash
# Using the CLI
./cli/service_mgmt.sh add examples/strale-gateway-config.json

# Or via the Python API
uv run python api/registry_management.py add-server --config examples/strale-gateway-config.json
```

Set the `STRALE_API_KEY` environment variable so the gateway can pass the Bearer token:

```bash
export STRALE_API_KEY=sk_live_your_key_here
```

## Register as A2A Agent

```bash
uv run python cli/agent_mgmt.py register examples/strale-a2a-agent-config.json
```

## Verify the MCP Server

Test a free capability (no API key needed):

```bash
# Search for capabilities
curl -X POST http://localhost:3000/strale/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "strale_search", "arguments": {"query": "IBAN"}}, "id": 1}'

# Execute a free capability
curl -X POST http://localhost:3000/strale/mcp \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $STRALE_API_KEY" \
  -d '{"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "strale_execute", "arguments": {"slug": "iban-validate", "inputs": {"iban": "DE89370400440532013000"}}}, "id": 2}'
```

## Verify the A2A Agent

```bash
curl -X POST http://localhost:3000/strale/a2a \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $STRALE_API_KEY" \
  -d '{"jsonrpc": "2.0", "method": "message/send", "params": {"message": {"role": "user", "parts": [{"type": "text", "text": "Validate IBAN DE89370400440532013000"}]}}, "id": 1}'
```

## Available Tools (MCP)

| Tool | Auth Required | Description |
|------|:---:|---|
| `strale_ping` | No | Health check |
| `strale_getting_started` | No | Free capabilities with example inputs |
| `strale_search` | No | Search 250+ capabilities by keyword |
| `strale_execute` | Partial | Run any capability. 5 free-tier slugs work without auth. |
| `strale_trust_profile` | No | Quality score and execution guidance |
| `strale_methodology` | No | Quality scoring methodology |
| `strale_balance` | Yes | Check wallet balance |
| `strale_transaction` | Partial | Retrieve past execution records |

## Free Tier

These capabilities work without an API key (10 calls/day per IP):

- `email-validate` — verify email deliverability
- `iban-validate` — validate international bank account numbers
- `dns-lookup` — DNS records for any domain
- `url-to-markdown` — convert any URL to clean markdown
- `json-repair` — fix malformed JSON

## Semantic Search Tags

The Strale server is tagged for discovery via the gateway's semantic search:

`validation`, `compliance`, `enrichment`, `kyb`, `sanctions`, `web3`, `trust`, `audit-trail`, `iban`, `vat`, `domain`, `email`, `company-data`, `quality-scored`

## Links

- [Documentation](https://strale.dev/docs)
- [Capability catalog](https://api.strale.io/v1/capabilities)
- [Quality methodology](https://strale.dev/trust/methodology)
- [MCP server on npm](https://www.npmjs.com/package/strale-mcp)
- [Agent Card](https://api.strale.io/.well-known/agent-card.json)
