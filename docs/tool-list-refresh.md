# Tool list refresh and drift detection

The registry keeps a copy of each MCP server's tool list. Search, the server API, the UI and virtual servers read that copy, while MCP clients on a direct route get the live list from the server through the gateway. This page explains when the registry fetches the list again, how it decides the copy is out of date, and how to turn on a scheduled refresh.

## How the registry detects a change

Each time the registry fetches a server's tools, it hashes the whole list with SHA-256 and compares that with a hash of the stored copy. The hash covers each tool's:

- `name`
- `description`
- `inputSchema`
- `title`, `annotations` and `outputSchema` (see [Tool annotations](tool-annotations.md))

Tool order does not affect the hash. The registry derives `parsed_description` from `description`, so the hash leaves it out.

If the hashes match, the registry writes nothing. If they differ, it saves the new list and sets two fields on the server:

| Field | Meaning |
|-------|---------|
| `tool_list_hash` | The SHA-256 of the saved list. A client can compare it to tell whether anything changed. |
| `tool_list_changed_at` | When the registry last saved a list that differed from the stored one, in ISO 8601 UTC. The first discovery after registration does not set it. |

Both fields appear in `GET /api/servers` and `GET /api/servers/{path}`, and in `registry_management.py server-get`. A server stored before this release gets `tool_list_hash` the first time its tools change, not on upgrade: every save re-indexes the server for search, and a backfill would re-embed every server at once.

A saved change also logs one line naming the tools that were added, removed and changed:

```
Tool list changed for /github (42 tools): added=[] removed=[] changed=['create_issue']
```

When a removed tool had a block override, the registry also logs a warning. Blocks are keyed by tool name, so if the server renamed the tool, the new name is not blocked and you need to block it again.

The registry updates scopes, and reloads the auth server, only when tool names change. A description or schema change does neither.

## When the registry fetches tools

With refresh off (the default), the registry fetches a server's tools:

- on the server's first health check after the registry starts,
- when the server turns healthy after being unhealthy,
- on every health check while the server has no stored tools,
- when someone opens the server's tool list in the UI, or calls `GET /api/servers/tools/{path}`.

A server that stays healthy is not fetched again until one of those happens. With refresh on, the registry also fetches every healthy server on a schedule.

## Turning on scheduled refresh

Set `TOOL_REFRESH_ENABLED=true`. The health loop keeps its own interval, `HEALTH_CHECK_INTERVAL_SECONDS`. Each server is fetched once every `TOOL_REFRESH_EVERY_CYCLES` health cycles, so the refresh period is:

```
period in seconds >= TOOL_REFRESH_EVERY_CYCLES x HEALTH_CHECK_INTERVAL_SECONDS
```

| Surface | Default interval | Period with the default of 3 cycles |
|---------|------------------|-------------------------------------|
| Terraform (ECS), Helm (EKS), CDK | 300 s | 15 minutes |
| Docker Compose | 30 s | 90 seconds |

Each server starts on a random cycle, drawn once, between 1 and the period. With a period of 3, about a third of the servers are fetched on each cycle, and every server is fetched exactly once every 3 cycles. A change is picked up within one period. Fetches count health cycles, not clock time, so a slow cycle does not cause a missed fetch.

The registry logs its refresh settings at startup:

```
Tool refresh on: every 3 health cycles (at least 900s), at most 5 concurrent fetches per registry process
```

The period is a lower bound. The registry sleeps `HEALTH_CHECK_INTERVAL_SECONDS` after each round of checks finishes, so a cycle lasts the interval plus however long the checks took. On a Compose stack with about 60 servers and a 30 s interval, one cycle took about 44 s, so 3 cycles came to about 2 minutes 15 seconds.

The registry does not listen for the MCP `notifications/tools/list_changed` message. It keeps no open connection to a server, and the 2026-07-28 revision of MCP moves change notifications to `subscriptions/listen`, which the gateway does not support yet (see [MCP 2026-07-28 migration](mcp-2026-07-28-migration.md)).

## Replicas

Each registry replica runs its own health loop and draws its own starting cycles. With two replicas, every upstream server sees two refreshes per period. Helm runs two registry replicas by default. If that load is too much for your upstream servers, raise `TOOL_REFRESH_EVERY_CYCLES`.

## Limiting concurrent fetches

`TOOL_FETCH_MAX_CONCURRENCY` (default 5) caps how many tool fetches one registry process runs at once. It applies to every fetch: first checks, recoveries, scheduled refreshes and the UI refresh. A server never has two fetches running at once. If a fetch for a server is still running or waiting when the next one is due, the registry skips the new one.

To size it, a cycle starts about N / P fetches for N servers and a period of P cycles, and they take about (N / P) x fetch time / cap seconds to finish. A healthy server answers in 1 to 3 seconds, and discovery gives up after 25 seconds. Keep that time under half the health interval:

```
cap >= N x 3 s / (P x interval x 0.5)
```

For 300 servers, a period of 3 and a 300 s interval, that gives 2, so 5 leaves room for slow servers. Raise it to 10 for about a thousand servers, or when the health interval is short, as on Docker Compose.

The UI refresh button goes through the same cap. If a scheduled fetch for that server is already waiting, the click does not start a second one, and the new tools appear when the waiting fetch finishes.

## Monitoring

Two metrics cover tool fetches:

| Metric | Type | Attributes |
|--------|------|------------|
| `mcpgw_registry_tool_fetch_total` | Counter | `trigger`: `first_check`, `recovered`, `empty`, `refresh`, `manual`. `result`: `changed`, `unchanged`, `failed`, `timeout`, `skipped_in_flight` |
| `mcpgw_registry_tool_fetch_duration` | Histogram (ms) | `trigger` |

A steady rise in `result="skipped_in_flight"` means fetches from one cycle are still running when the next cycle starts: raise `TOOL_FETCH_MAX_CONCURRENCY` or `TOOL_REFRESH_EVERY_CYCLES`. A rise in `result="timeout"` points at slow upstream servers.

## Parameters

| Variable | Default | Range | Meaning |
|----------|---------|-------|---------|
| `HEALTH_CHECK_INTERVAL_SECONDS` | `300` (`30` on Compose) | 1 and up | Seconds between health checks; one refresh cycle |
| `TOOL_REFRESH_ENABLED` | `false` | | Fetch tools from healthy servers on a schedule |
| `TOOL_REFRESH_EVERY_CYCLES` | `3` | 1 to 1000 | Fetch each healthy server once every this many cycles |
| `TOOL_FETCH_MAX_CONCURRENCY` | `5` | 1 to 100 | Most tool fetches one registry process runs at once |

The registry refuses to start when a value is out of range. The Terraform, Helm and CDK names are in [the parameter reference](unified-parameter-reference.md#group-13a1-health-checks-and-tool-list-refresh-issue-2050).
