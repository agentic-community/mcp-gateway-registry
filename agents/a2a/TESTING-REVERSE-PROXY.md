# Testing A2A Reverse-Proxy Routing

This is a hands-on runbook for verifying **A2A reverse-proxy mode** end to end using the
sample Travel Assistant and Flight Booking agents in this folder: a calling agent discovers a
target agent through the registry and then invokes it, with **every hop routed through the
gateway**. It documents two setups we actually exercised:

1. [Local Docker Compose](#part-1-local-docker-compose) — registry + agents on one host.
2. [ECS + CloudFront](#part-2-ecs--cloudfront-hybrid) — registry on ECS behind CloudFront,
   agents running on a host and reached back through the gateway.

For what reverse-proxy mode is and how it works, see the main [A2A guide](../../docs/a2a.md#reverse-proxy-mode-routing-a2a-traffic-through-the-gateway)
and the [design doc](../../docs/design/a2a-protocol-integration.md). For the egress trust
model (the `X-Authorization` / `Authorization` header split) see
[a2a-protocol-integration.md#egress-trust-model](../../docs/design/a2a-protocol-integration.md#egress-trust-model-two-header-split).

## What "working" looks like

Reverse-proxy routing is confirmed when all of these hold for a registered, enabled agent:

- Its stored `url` is the **gateway** address (`{REGISTRY_URL}/agent/{path}/`) and its
  `proxy_pass_url` is the **private backend** (visible to admins only).
- Fetching `/agent/{path}/.well-known/agent-card.json` through the gateway returns a card
  whose `url` points back at the gateway (not the backend).
- A JSON-RPC call to `/agent/{path}/` is gated by `invoke_agent` (403 without the grant, 200
  with it) and reaches the backend.
- The gateway logs show the auth subrequest passing and the proxy hop to the backend.

## Auth mode used for these tests

The sample agents support a **presence-only** auth mode (`AGENT_AUTH_PRESENCE_ONLY=true`):
they accept any non-empty `Authorization: Bearer <value>` without verifying it, so you can
exercise the gateway's routing and `invoke_agent` gating before wiring a real per-agent
credential. Because presence-only is effectively unauthenticated, an agent refuses to start
in this mode while bound to all interfaces (which it must be, so the gateway can reach it)
unless you also set `AGENT_AUTH_ALLOW_EXPOSED_PRESENCE_ONLY=true`. Both are TEST/DEMO switches
only — never enable them in a real deployment. See the
`.env.example` files in this folder for the flag.

For a verified run, set both presence-only flags to `false`. The local Compose
fallback and ECS test environment enable them for routing tests. Set `KEYCLOAK_URL`
to the reachable token/JWKS transport and `KEYCLOAK_EXTERNAL_URL` to the canonical
public issuer base URL. The signed issuer must match
`<KEYCLOAK_EXTERNAL_URL>/realms/<KEYCLOAK_REALM>`. Obtain a token with the backend's
configured audience and pass it through the normal credential path. The routing
examples below use presence-only mode and do not verify backend authentication.

---

## Part 1: Local Docker Compose

Registry, gateway, and both agents run on one host on the same Docker network. The gateway
reaches the agents by their Docker service names.

### 1. Enable reverse-proxy mode on the registry

In the repo-root `.env` (the registry's env, not this folder's):

```bash
A2A_REVERSE_PROXY_ENABLED=true
# Let the gateway reach agent backends on the internal Docker network.
SSRF_ALLOWED_HOSTS=flight-booking-agent,travel-assistant-agent
SSRF_ALLOWED_CIDRS=172.18.0.0/16
```

Requires `DEPLOYMENT_MODE=with-gateway` (the default). Start/restart the stack from the repo
root so the registry regenerates nginx with the `/agent/*` blocks:

```bash
./build_and_run.sh          # or: docker compose up -d
```

### 2. Start the sample agents

The agents are a separate uv project with their own venv (do not use the repo-root venv).
They listen on container port 9000, published on host ports 9001 (travel) and 9002 (flight).

```bash
cd agents/a2a
cp .env.example .env        # set MCP_REGISTRY_URL, a registry JWT, AGENT_AUTH_PRESENCE_ONLY=true,
                            # AGENT_AUTH_ALLOW_EXPOSED_PRESENCE_ONLY=true (agents bind 0.0.0.0)
./deploy_local.sh
```

### 3. Register + enable both agents (url rewritten to the gateway)

From the repo root, with a token in `.token`. The registry UI's "Get JWT Token" button
saves the **nested** format (`{"tokens": {"access_token": "<jwt>"}}`), but `cli/agent_mgmt.py`
expects a **flat** file (`{"access_token": "<jwt>"}`). Flatten it into `.token.flat` first
(this one-liner handles both the nested and already-flat cases):

```bash
python3 -c "import json; d=json.load(open('.token')); t=d.get('access_token') or d['tokens']['access_token']; json.dump({'access_token': t}, open('.token.flat','w'))"
CLI="uv run python cli/agent_mgmt.py --token-file .token.flat"

$CLI register cli/examples/flight_booking_agent_card.json
$CLI toggle /flight-booking true
$CLI register cli/examples/travel_assistant_agent_card.json
$CLI toggle /travel-assistant-agent true
```

Confirm the split as an admin (advertised `url` = gateway, `proxy_pass_url` = backend):

```bash
$CLI get /flight-booking | jq '{url, proxy_pass_url}'
```

### 4. Grant invoke access

The token you test with needs an `invoke_agent` grant for the target path. Admins already
have it via `scripts/registry-admins.json` (`{"agent": "*", "actions": [..., "invoke_agent"]}`).
For a least-privilege caller, add a per-agent rule to that group's scope (see
[cli/examples/public-mcp-users.json](../../cli/examples/public-mcp-users.json) and
[docs/scopes-mgmt.md](../../docs/scopes-mgmt.md#agent-rule)).

### 5. Verify each hop through the gateway

```bash
TOKEN=$(jq -r .access_token .token.flat)   # .token.flat created in step 3
REGISTRY_URL=http://localhost

# Agent card THROUGH the gateway -> card.url must be the gateway, not the backend.
curl -s -H "X-Authorization: Bearer $TOKEN" \
  "$REGISTRY_URL/agent/flight-booking/.well-known/agent-card.json" | jq .url

# JSON-RPC invoke THROUGH the gateway. X-Authorization = gateway token (gated + stripped);
# Authorization = the target agent's own credential (forwarded). With presence-only auth the
# target accepts any bearer.
curl -s "$REGISTRY_URL/agent/flight-booking/" \
  -H "X-Authorization: Bearer $TOKEN" \
  -H "Authorization: Bearer any-nonempty-value" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"1","method":"message/send","params":{"message":{"role":"user","messageId":"msg-1","parts":[{"kind":"text","text":"any flights NYC to SFO?"}]}}}'
```

Without an `invoke_agent` grant the JSON-RPC call returns **403** at the gateway (the backend
is never reached). Watch the hops:

```bash
docker compose logs -f nginx auth-server registry
```

### 6. Agent-discovers-agent (the full loop)

Drive the Travel Assistant to discover and call the Flight Booking agent, so the whole path
is agent → gateway → auth → gateway → agent:

```bash
# The test calls the agents directly, so it needs a bearer. Under presence-only
# auth any non-empty value works (set AGENT_BEARER_TOKEN or REGISTRY_JWT_TOKEN).
# --registry-url is the harness's reachability probe target (defaults to
# http://localhost); point it at the gateway you enabled reverse-proxy on.
AGENT_BEARER_TOKEN=any-nonempty-value \
  uv run python agents/a2a/test/simple_agents_test.py \
    --endpoint local --debug --registry-url http://localhost
```

`--debug` prints the JSON-RPC payloads, response bodies, and timing for each hop.

The startup banner prints the topology, including the Travel Assistant's own
**in-container `MCP_REGISTRY_URL`** (read via `docker exec`) — this is the registry
it actually discovers against, which is independent of `--registry-url`. The agent
then connects to the discovered card's URL (e.g. `<registry>/agent/flight-booking/`).
Confirm the exact registry + `/agent/` URL it invokes in the container log:

```bash
docker logs travel-assistant-agent \
  | grep -iE "RegistryDiscoveryClient initialized|Initializing A2A client|HTTP Request"
```

---

## Part 2: ECS + CloudFront (hybrid)

The registry/gateway run on ECS Fargate behind CloudFront + ALB. The agents run on a host
(EC2 or a workstation); the ECS gateway reaches them **back over the internet** on their
public ports. Use `agents/a2a/.env.ecs.example` as the template for this topology.

```
Client ─► CloudFront ─► ALB ─► nginx (ECS) ─► auth-server (ECS)
                                   │  (gateway proxies A2A)
                                   └────────────────► agent backend on the host (NAT EIP inbound)
```

### 1. Enable reverse-proxy mode on ECS

In `terraform/aws-ecs/terraform.tfvars`:

```hcl
a2a_reverse_proxy_enabled = true
# Name the agent backends the ECS gateway must reach (public host or DNS).
ssrf_allowed_hosts = "mcpgateway.ddns.net"
ssrf_allowed_cidrs = ""
```

`terraform apply`, then let the service redeploy. (The nginx `/agent/*` blocks are generated
at registration time once the flag is on.)

### 2. Run the agents on the host, reachable by ECS

Deploy the agents locally on the host and expose their ports so the ECS egress can reach them:

```bash
cd agents/a2a
cp .env.ecs.example .env     # MCP_REGISTRY_URL = the CloudFront URL; presence-only auth ok for the test
./deploy_local.sh
```

Open the agent ports **inbound from the ECS NAT EIPs only** (not `0.0.0.0/0`) in the host's
security group, so only the gateway can reach the agents. In our run the flight agent's
backend was `http://mcpgateway.ddns.net:9000` (travel on `:9001`), and the ECS egress arrived
from the NAT EIP.

### 3. Register with the ECS card (gateway url = CloudFront, backend = host)

Use the ECS variant cards, whose `url` points at the host backend; the registry rewrites the
advertised `url` to the CloudFront `/agent/{path}/` and stores the host backend in
`proxy_pass_url`:

```bash
export REGISTRY_URL=https://<your-distribution>.cloudfront.net
# Flatten the nested UI token as in Part 1 step 3 (creates .token.flat).
python3 -c "import json; d=json.load(open('.token')); t=d.get('access_token') or d['tokens']['access_token']; json.dump({'access_token': t}, open('.token.flat','w'))"
CLI="uv run python cli/agent_mgmt.py --token-file .token.flat --base-url $REGISTRY_URL"

$CLI register cli/examples/flight_booking_agent_ecs.json
$CLI toggle /flight-booking true
$CLI get /flight-booking | jq '{url, proxy_pass_url}'   # url=CloudFront, proxy_pass_url=host
```

### 4. Verify through CloudFront

```bash
TOKEN=$(jq -r .access_token .token.flat)   # .token.flat created in step 3

# Card through CloudFront -> url advertises https + the CloudFront host (see scheme note below).
curl -s -H "X-Authorization: Bearer $TOKEN" \
  "$REGISTRY_URL/agent/flight-booking/.well-known/agent-card.json" | jq .url

# Invoke through CloudFront.
curl -s "$REGISTRY_URL/agent/flight-booking/" \
  -H "X-Authorization: Bearer $TOKEN" \
  -H "Authorization: Bearer any-nonempty-value" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"1","method":"message/send","params":{"message":{"role":"user","messageId":"msg-1","parts":[{"kind":"text","text":"any flights NYC to SFO?"}]}}}'
```

Trace hops with ECS Exec / CloudWatch:

- **auth-server** logs `A2A per-agent scope validation passed` on the `/validate` subrequest.
- **nginx** logs the proxy hop to the host backend.
- the **flight backend** on the host logs the inbound `POST` from the ECS NAT EIP.

### Scheme note (CloudFront + ALB)

CloudFront terminates TLS and forwards to the ALB over plain HTTP, and the ALB **overwrites**
`X-Forwarded-Proto` with the CloudFront→ALB hop scheme (`http`). The card rewriter therefore
prefers `X-Cloudfront-Forwarded-Proto` (CloudFront's original viewer scheme) so the advertised
`url` is `https`, not `http`. If your card comes back advertising `http`, that header path is
the thing to check.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Card `url` still shows the backend, not the gateway | Reverse-proxy mode not effective: flag off, or `registry-only` mode (force-disabled), or agent registered before the flag was on. Re-register. |
| Agent marked unhealthy / 502 from the gateway | Gateway can't reach the backend — add the host/CIDR to `SSRF_ALLOWED_HOSTS`/`SSRF_ALLOWED_CIDRS`; on ECS also open the host SG to the NAT EIPs. |
| JSON-RPC returns 403 | Caller's token lacks an `invoke_agent` grant for that agent path. |
| JSON-RPC returns 401 on an agent path | No `X-Authorization` (agent paths authenticate on `X-Authorization` only, no `Authorization` fallback), or the same token was sent in both headers (rejected by design). |
| No `/agent/*` blocks generated | `DEPLOYMENT_MODE` is `registry-only`, or the agent is disabled — reverse-proxy blocks are emitted only for enabled agents in `with-gateway` mode. Also: the agent must be `healthy` and its backend host must resolve on the registry's network (see next row). |
| A specific enabled agent has no block (others do) | Its backend host does not resolve from the registry container, so the block is skipped on purpose (an unresolvable literal `proxy_pass` would fail the whole nginx reload). Make the host resolvable from the registry (correct Docker service name / ECS Service Connect name / DNS), then re-enable or re-register. |
| Card advertises `http` behind CloudFront | ALB clobbered `X-Forwarded-Proto`; confirm `X-Cloudfront-Forwarded-Proto` reaches nginx. |
