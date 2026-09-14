# What changes when I upgrade to a release with the gateway generic proxy?

Nothing, until you opt in.

The generic proxy lets the gateway front non-MCP entities — skills, A2A agents, and custom records such as a REST API — at `/{prefix}/{entity_type}/{name}/`. It ships disabled. On a default upgrade your deployment behaves as it did before: no new route renders, `/validate` mints no generic-proxy token, and the nginx render path issues no extra per-tick database queries.

This page answers what an upgrade does and does not change. For how to use the feature, see the [operational guide](../gateway-proxy-operational-guide.md) and [registering OpenAI or Bedrock endpoints](registering-model-inference-endpoints.md).

## What a default upgrade does not change

**Existing routes stay where they are.** MCP servers keep `/mcp-proxy/...`. A2A agents keep `/agent/...`. The generic proxy adds paths alongside them and relocates nothing.

**No schema migration, no backfill.** Eight fields join the entity models — `is_proxied`, `proxy_target_url`, `proxy_streaming`, `proxy_connect_notes`, `custom_headers_encrypted`, `custom_header_names`, `custom_header_overridable_names`, `custom_headers_updated_at`. Existing records do not have them and read as their defaults. Nothing rewrites your data.

**Enabling the feature grants nobody anything.** A legacy `methods: ["all"]` rule authorizes no HTTP verb. That split is deliberate: an existing MCP grant must not turn into arbitrary HTTP access against a proxied backend. Every proxied entity needs an explicit verb rule, or `http:*`, on its authz key.

**Per-entity opt-in.** With the feature on, still no route renders until an entity sets `is_proxied`.

## What an upgrade does change

Two visible differences, both read-only:

- The new fields appear in API read models, with secret values stripped. `custom_header_names` lists which upstream headers an entity has; the values never leave the registry.
- Entity forms in the UI show a proxy toggle.

## Turning it on: the flag and three prerequisites

`GATEWAY_GENERIC_PROXY_ENABLED=true` is necessary and not sufficient. Both the registry and the auth-server must receive the same value — the registry renders the routes, the auth-server serves the hop.

**1. `DEPLOYMENT_MODE=with-gateway`.** In `registry-only` mode there is no nginx to render into, and the feature stays off.

**2. A stable `SECRET_KEY`.** It derives the encryption key for stored upstream credentials. Rotating it makes every stored credential undecryptable, and each affected entity then returns 502 on every request. Rotate the provider keys instead, or re-register the headers after a `SECRET_KEY` change.

**3. An enforced network egress policy.** At startup the auth-server probes a cloud metadata IP (`169.254.169.254`, `fd00:ec2::254`) from inside its own container. If the address answers, the feature disables itself for that process and logs at critical:

```text
Generic proxy egress self-check FAILED: a cloud metadata IP
(169.254.169.254 / fd00:ec2::254) is REACHABLE from the auth-server.
The required network egress policy is NOT enforced — DISABLING the
generic-proxy feature for this process (fail-closed). Deploy the
egress NetworkPolicy/security-group before enabling this feature.
```

The gauge `mcpgw_registry_gateway_egress_policy_unverified` reads `1` in that state. A standing `1` on a deployment you believe is enabled means the routes render and the hop refuses every request. On a passing deployment the log reads `Generic proxy egress self-check PASSED; feature active`.

You can opt out of the check with `GATEWAY_EGRESS_SELFCHECK_ENABLED=false`, which logs a warning and leaves DNS rebinding to the metadata IP unmitigated. Do that only when the egress policy is enforced by something the container cannot see.

## Custom records need a second flag

`CUSTOM_ENTITY_TYPES_ENABLED` is separately false by default. Without it the registry never registers `/api/custom-types` or `/api/custom/{type}`, and both return **404**.

Worth knowing because the symptom misleads: a 404 reads as a missing route or a version mismatch, when the cause is a disabled feature. Skills and A2A agents need only `GATEWAY_GENERIC_PROXY_ENABLED`; custom records need both flags.

## Checking what your deployment is doing

```bash
# the flags each service actually received (both must agree on the first one)
docker compose exec -T registry env | grep -E '^(GATEWAY_GENERIC_PROXY_ENABLED|CUSTOM_ENTITY_TYPES_ENABLED|DEPLOYMENT_MODE)='
docker compose exec -T auth-server env | grep -E '^GATEWAY_GENERIC_PROXY_ENABLED='

# did the hop come up, or did the self-check latch it off?
docker compose logs auth-server | grep -iE 'egress self-check|Generic proxy'

# which entities are proxied right now
curl -sS --compressed -H "Authorization: Bearer $GW" "$REGISTRY_URL/api/skills" \
  | jq -r '.skills[] | select(.is_proxied) | "\(.path) -> \(.proxy_target_url)"'
```

On a working deployment the first command prints all three variables as `true` / `true` / `with-gateway`, and the last prints one line per proxied entity:

```text
/skills/pdf -> https://raw.githubusercontent.com
```

The log line tells you which of three states the hop is in:

| Log line | State |
|---|---|
| `Generic proxy egress self-check PASSED; feature active` | enabled and verified |
| `Generic proxy egress self-check FAILED: ... DISABLING the generic-proxy feature` | routes render, hop refuses every request |
| `Generic proxy ENABLED with egress self-check OPTED OUT` | `GATEWAY_EGRESS_SELFCHECK_ENABLED=false`; you own the egress policy |
| `Generic proxy feature disabled (gateway_generic_proxy_enabled=false)` | off, the default |

The System Config page in the UI shows the same flags. `GET /api/config/full` backs that page and authenticates with a session cookie, so a Bearer token returns `{"detail": "Authentication required"}` — read the values from the container env or the UI instead.

If a route 404s while the feature is on, check that nginx has regenerated. The registry rewrites the config when a proxied entity changes, and a record created seconds ago may not have a location yet.

## If you upgrade past 1.30.0, three metric labels change

The gateway-proxy metrics work in 1.30.0 alters labels on an existing counter. Saved queries and alert rules need a look:

- `mcpgw_registry_auth_request_total` now classifies gateway traffic as `target_kind="generic_proxy_skill"`, `generic_proxy_agent`, or `generic_proxy_custom`. Anything filtering `target_kind="unknown"` stops matching it.
- On those series `server` holds the entity's authz key — `skill/skills/pdf` — instead of the literal `gateway`.
- `success` is lowercase `true` / `false` on every exporter. A query written `success="True"` returns an empty result rather than an error, so it looks like zero traffic.
- `mcpgw_registry_auth_request_duration_milliseconds` no longer carries `server`. Group it by `target_kind`.

The shipped Grafana dashboards are already updated. See [Observability](../OBSERVABILITY.md#target-type-routing-target_kind) for the full label reference.

## Related documentation

- [Gateway generic proxy operational guide](../gateway-proxy-operational-guide.md) — routes, credentials, streaming limits, troubleshooting
- [How do I let my team call OpenAI or Bedrock through the gateway?](registering-model-inference-endpoints.md)
- [Unified parameter reference](../unified-parameter-reference.md) — every flag above across Docker, Terraform, and Helm
- [Observability](../OBSERVABILITY.md) — the metrics and labels named here
