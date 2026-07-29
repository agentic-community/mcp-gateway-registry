# How do I quarantine a user, an agent, or an MCP server?

Quarantine is a **kill switch**: it drops *all* data-plane traffic from a caller (a user or an agent/M2M client) or to a target (an MCP server or an A2A agent), immediately and absolutely. It is not a rate limit, it is a hard block. Use it when a caller is compromised or misbehaving, or when a target must be taken out of rotation while you investigate, without deleting the account or unregistering the server.

Quarantine is part of the [application-level rate limiting](../design/rate-limiting.md) feature. The two quarantine groups are **auto-seeded empty at startup**, so there is nothing to create first: you just move a subject into the right group.

## Prerequisites

- Rate limiting is enabled on the deployment (`RATE_LIMITING_ENABLED=true`). Quarantine only acts while rate limiting is on (it rides the same `/validate` enforcement hop).
- You have an **admin** token. The commands below load it from a `.token` file via `--token-file`; substitute your own path.
- The examples use `REG` for the registry URL: `export REG=http://localhost` (or your deployment URL).

## The one thing you do

Move the subject into its quarantine group. There are two reserved groups and the server picks the right one from the subject type, so you cannot mis-scope:

- `user` / `client` subjects → `quarantine-callers`
- `server` / `agent` subjects → `quarantine-targets`

### Quarantine an agent (M2M client)

```bash
cd api
uv run python registry_management.py rate-limit-quarantine-add \
  --subject-type client --subject <client_id> \
  --token-file ../.token --registry-url "$REG"
```

Every call made with that client's token is now denied at the gateway with a plain `403 Access forbidden`.

### Quarantine a user

```bash
uv run python registry_management.py rate-limit-quarantine-add \
  --subject-type user --subject alice \
  --token-file ../.token --registry-url "$REG"
```

### Quarantine an MCP server (or an A2A agent)

```bash
# MCP server (subject = the server path, e.g. mcpgw)
uv run python registry_management.py rate-limit-quarantine-add \
  --subject-type server --subject mcpgw \
  --token-file ../.token --registry-url "$REG"

# A2A agent (subject = the agent path, e.g. /booking-agent)
uv run python registry_management.py rate-limit-quarantine-add \
  --subject-type agent --subject /booking-agent \
  --token-file ../.token --registry-url "$REG"
```

No caller can reach a quarantined target while it is in the group.

## From the UI

**Settings → IAM → Rate Limits → Quarantine** shows the two groups (`quarantine-callers`, `quarantine-targets`) with a live member count, a per-member **Remove** action, and a destructive global **enable/disable** master switch per group. (Adding a subject from the Users / M2M / server-detail pages is a follow-on; today you add from the CLI or the API, and manage/remove from this panel.)

## From the API

The CLI wraps three admin-only endpoints:

```bash
# Add
curl -sS -X POST "$REG/api/rate-limit-quarantine/client:<client_id>" \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# List everything currently quarantined
curl -sS "$REG/api/rate-limit-quarantine" -H "Authorization: Bearer $ADMIN_TOKEN"

# Remove
curl -sS -X DELETE "$REG/api/rate-limit-quarantine/client:<client_id>" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

## Listing and un-quarantining

```bash
cd api
# See who/what is quarantined.
uv run python registry_management.py rate-limit-quarantine-list \
  --token-file ../.token --registry-url "$REG"

# Release a subject (a caller keeps any other rate-limit groups; a target's
# membership is deleted if quarantine was its only group).
uv run python registry_management.py rate-limit-quarantine-remove \
  --subject-type client --subject <client_id> \
  --token-file ../.token --registry-url "$REG"
```

Changes take effect within one cache TTL (~30 seconds).

## Important behaviors

- **Plain 403, not 429.** A quarantined subject gets `403 Access forbidden` with no `X-RateLimit-*` headers. Quarantine is an access decision, not a throttle, so it is deliberately distinguishable from a rate-limit 429.
- **Admin safety.** A quarantined **caller** who is an admin is *not* blocked (you cannot lock yourself out of the tool that manages quarantine). A quarantined **target** is blocked for *everyone*, admins included (it is out of rotation).
- **Data-plane only.** Quarantine blocks MCP/A2A calls, not the control-plane `/api/*` (a quarantined user can still reach the dashboard/login). This is deliberate — quarantine is not a way to lock a user out of the UI.
- **Global off-switch.** To turn a whole kill switch off without emptying the group, disable its reserved definition:
  ```bash
  uv run python registry_management.py rate-limit-disable \
    --id quarantine:group:quarantine-callers:1 --token-file ../.token --registry-url "$REG"
  ```
  The reserved groups cannot be deleted.
- **Failure mode.** Quarantine is **fail-open by default**: if the membership store is briefly unreachable, traffic is allowed rather than denied (consistent with the rate limiter's availability guardrail). Set `RATE_LIMIT_QUARANTINE_FAIL_CLOSED=true` to fail closed instead (a stricter block at the cost of denying data-plane traffic during a store outage).
- **Not breach containment.** Quarantine is best-effort and rides `RATE_LIMITING_ENABLED`. For a compromised credential, also revoke it at your IdP — do not rely on quarantine alone.

## How do I verify it worked?

Drive a call as the quarantined caller (or to the quarantined target) and confirm the plain 403:

```bash
uv run python tests/scripts/call_mcp_tool.py \
  --server-url "$REG/<server>/mcp" --tool <tool> --tool-args '{}' \
  --token-file <caller-token> --registry-url "$REG" --count 1
# → HTTP 403 (Access forbidden), NOT 429
```

Operationally, the `mcpgw_rate_limit_quarantine_denied_total{scope="caller|target"}` metric increments on each blocked call, and the auth-server logs a `rate-limit quarantine deny: scope=... caller_username=... caller_client_id=...` line you can attribute to the exact subject. See the [Observability guide](../OBSERVABILITY.md#rate-limiting).

## Related

- [Rate Limiting: Design and Implementation Guide](../design/rate-limiting.md) — the full feature, including the per-caller-per-target axis.
- [Observability Guide → Rate limiting](../OBSERVABILITY.md#rate-limiting) — the quarantine metrics and queries.
