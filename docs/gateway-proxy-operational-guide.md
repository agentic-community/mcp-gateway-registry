# Gateway Proxy Operational Guide

How to put an HTTP backend behind the gateway, which credential model to pick, and how to run the result.

For internals — the request sequence, the nginx render path, storage fields, failure shapes — read [design/gateway-generic-proxy.md](design/gateway-generic-proxy.md). This guide stays on the operator's side of that line.

Four walkthroughs, each end to end:

| Scenario | Backend | Credential | Section |
|---|---|---|---|
| Model inference | OpenAI, Amazon Bedrock | each caller's own key | [Pattern 1](#pattern-1--model-inference-openai-and-bedrock) |
| Generic REST API | any HTTP API | your service key, or none | [Pattern 2](#pattern-2--generic-rest-api) |
| Skills and documents | a file origin | usually none | [Pattern 3](#pattern-3--skills-and-documents) |
| A2A agents | the agent's own URL | none today | [Pattern 4](#pattern-4--a2a-agents) |

Patterns 1 and 2 need a custom entity type first. Patterns 3 and 4 use entities you already have.

## Prerequisites

Work through all four steps before running anything else in this guide.

### 1. Enable the feature

The proxy ships off. Three settings must hold before any route renders.

| Setting | Required value | Why |
|---|---|---|
| `GATEWAY_GENERIC_PROXY_ENABLED` | `true` | with it off, no location block renders and `/validate` mints no generic token |
| `DEPLOYMENT_MODE` | `with-gateway` | the proxy renders only in gateway mode |
| `SECRET_KEY` | set and stable | derives the encryption key for stored upstream credentials; rotating it invalidates every one |

Set them in `.env` and rebuild, then check the startup logs:

```bash
docker compose logs auth-server | grep -iE "generic.proxy|egress" | tail -20
```

Deploy the network egress policy too. At startup the auth-server probes the cloud metadata address from inside its container. If that address answers, the policy is missing, so the auth-server disables the feature for the process and logs `Generic proxy egress self-check FAILED`. Check for that line after every deploy; the healthy counterpart is `Generic proxy egress self-check PASSED; feature active`. The gauge `mcpgw_registry_gateway_egress_policy_unverified` reads 1 in that state.

[unified-parameter-reference.md](unified-parameter-reference.md) lists every `GATEWAY_GENERIC_*` setting with its compose, terraform, and helm equivalent.

### 2. Get a token

In the registry UI, click **Get JWT Token** in the sidebar. Copy the JSON it produces and save it to `.token` at the repository root. That file holds the access token under `tokens.access_token`, which is what every command below reads.

For a machine-to-machine token instead, use the script:

```bash
./api/get-m2m-token.sh --keycloak-url http://localhost:8080 --aws-region us-east-1
```

Tokens expire, and an expired one produces a plain 401 that reads like a permissions problem. Check the lifetime before a long session:

```bash
python3 - <<'PY'
import base64, json, time
t = json.load(open('.token'))['tokens']['access_token']
p = t.split('.')[1]; p += '=' * (-len(p) % 4)
left = json.loads(base64.urlsafe_b64decode(p))['exp'] - int(time.time())
print(f"{left // 60} minutes left" if left > 0 else f"EXPIRED {abs(left) // 60} minutes ago")
PY
```

### 3. Set the shell variables

Every command in this guide uses these three. Set them once per shell:

```bash
cd /path/to/mcp-gateway-registry      # repository root; paths below are relative to it

export REGISTRY_URL=http://localhost  # or your registry endpoint, e.g. https://registry.example.com
export TOKEN_FILE=.token              # the file from step 2
export GW=$(jq -r .tokens.access_token "$TOKEN_FILE")

[ -n "$GW" ] && [ "$GW" != null ] || echo "FATAL: no token in $TOKEN_FILE" >&2
```

`registry_management.py` defaults to `http://localhost` and `.token`, but this guide passes `--registry-url` and `--token-file` on every call so a non-default endpoint needs no edits.

**Behind CloudFront or any gzipping edge, add `--compressed` to every `curl`.** JSON responses come back gzipped, and `curl` does not decompress unless asked, so a working call prints unreadable bytes — or nothing at all, if the shell swallows them — while `-w '%{http_code}'` still reports `200`. That combination reads like a gateway bug and is not one. `requests`, the Python and JS SDKs, and the scripted clients under `tests/scripts/` all decompress transparently, so this only bites hand-run `curl`. A plain `http://localhost` registry is unaffected, which is why it surfaces only once you test a real deployment.

### 4. Check your scopes

Two different permissions govern this feature: one set to **manage** entities, another to **call** them through the gateway. Holding one does not imply the other.

**Control plane — managing types and records**

| Operation | Requires |
|---|---|
| Create, update, or delete a custom **type** | registry admin (`is_admin`, or the `mcp-registry-admin` group or scope) |
| Create a record of type `<type>` | `create_<type>_entity` |
| Update a record, including rotating upstream headers | `modify_<type>_entity` |
| Delete a record | `delete_<type>_entity` |
| List, get, search, or rate records | `list_<type>_entity` |
| Flip a skill or agent to proxied | owner of that entity, or admin |

Registry admin bypasses all of them. For the `rest-endpoint` type used throughout this guide, the mutation scopes read `create_rest-endpoint_entity`, `modify_rest-endpoint_entity`, and `delete_rest-endpoint_entity`.

`list_<type>_entity` is per-record aware. A grant of `"all"` or the type name opens the whole type; a grant naming a record path opens only that record. Missing it returns **404**, not 403 — the registry hides existence rather than confirming it, including for public records. A 404 on a record you know exists means a read-scope gap.

**Data plane — calling a route through the gateway**

Calling `/gateway/...` needs a `server_access` rule on one of the caller's groups. That is separate from every scope above; an admin who can create a record still gets 403 on the route without it. See [Authorizing callers](#authorizing-callers).

Inspect what a group holds:

```bash
curl -sS -H "Authorization: Bearer $GW" \
  "$REGISTRY_URL/api/management/iam/groups/registry-admins" | jq '{scope_config, server_access}'
```

## Quick start

Three steps: define a type, register a public API against it, call it.

```bash
# 1. define the type (once per registry)
cat > /tmp/rest-endpoint-type.json <<'JSON'
{
  "name": "rest-endpoint",
  "display_name": "REST Endpoint",
  "description": "Proxied generic REST/HTTP endpoints",
  "fields": [{"name": "notes", "label": "Notes", "datatype": "string"}]
}
JSON
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" --token-file "$TOKEN_FILE" \
  custom-type-create --config /tmp/rest-endpoint-type.json

# 2. register a record that fronts a public API
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" --token-file "$TOKEN_FILE" \
  custom-proxy-create \
  --type rest-endpoint --name openmeteo-forecast \
  --target-url https://api.open-meteo.com

# 3. call it through the gateway
BASE=$(curl -sS --compressed -H "Authorization: Bearer $GW" "$REGISTRY_URL/api/custom/rest-endpoint" \
  | jq -r --arg n openmeteo-forecast --arg u "$REGISTRY_URL" \
      '.records[] | select(.name==$n) | $u + .proxy_client_url + "/"')

curl -sS --compressed -w '\n%{http_code} %{content_type}\n' -H "Authorization: Bearer $GW" \
  "${BASE}v1/forecast?latitude=38.9&longitude=-77.03&current=temperature_2m"
```

Expect JSON and `200 application/json`. `200 text/html` means nginx has no location for the route yet — most often because the record was registered seconds ago — so wait and retry. The content type is what distinguishes the two: the fall-through also returns `200`, so a status-only check reports success on a route that is not live. That is why the example prints the content type.

## Two ways to turn on proxying

**From the UI.** Click the **edit icon** on a resource tile — a server, agent, skill, or custom record — and the edit form opens. Under **Serve through the gateway proxy**, tick **Enable proxying**. A **Backend URL** field appears; enter the origin the gateway should forward to. Proxied skills and custom records also show an **Upstream headers** editor for the credential. Save, and the tile gains a **Proxied** badge and an active **Connect** button that shows the client URL to hand to callers.

**From the API or CLI.** Everything below. Use this for scripted or repeatable registration.

Both write the same fields. The UI is faster for one entity; the CLI is what you put in a runbook.

## Why

The gateway reverse-proxies MCP servers and A2A agents. Everything else in the registry — skills, custom types, agents wanting a uniform hop — had no gateway route, so clients reached those backends directly. Direct access costs the three things a gateway exists to give: one authenticated ingress, one audit point, one egress policy.

It also pushes credentials outward. A team calling five backends holds five keys, each copied into every client that needs one. Rotating one key means finding every copy.

A proxied entity moves both problems inward. Callers hold a gateway token and nothing else. The backend URL and its credential stay in the registry, and every call lands in the audit trail you already read.

## How a request flows

A client calls the entity's gateway URL with a gateway token:

```
GET /gateway/rest-endpoint/<uuid>/v1/weather?lat=39.28&lon=-77.41
Authorization: Bearer <gateway token>
```

1. **nginx matches the entity's location block.** The registry generated it at registration. The block holds the backend URL and a few markers as nginx variables. No secret goes in that file.
2. **nginx calls `/validate` on the auth-server.** That checks the caller's token, checks the caller's groups against the entity's authz key for this HTTP verb, and mints a short-lived internal token. Streaming and upstream-auth markers become signed claims, so an inbound header cannot switch either on.
3. **nginx forwards to the internal hop** with the internal token attached. The hop verifies it and takes the entity identity and backend URL from the claims, never from a caller header.
4. **The hop builds the outbound request.** It keeps a positive allowlist of protocol headers, strips the whole gateway-internal set, fetches the entity's stored credential when it has one, merges operator and caller headers, and refuses to forward a bearer equal to the gateway's own.
5. **The hop fetches the backend** over a client pinned to the addresses resolved at registration, then returns the response — buffered, or chunk by chunk when the entity is configured to stream.

Two consequences to hold on to. The caller's identity stops at step 2, so the backend sees the gateway rather than the user unless you use the per-user vault. And the backend address is fixed at registration, so a DNS answer that changes later cannot move the request.

## Custom entity types

A custom type is the container for records that front a backend. Define it once, then register as many records against it as you need. Type creation is admin-only.

`custom-type-create` takes a JSON descriptor, not flags:

```bash
cat > /tmp/rest-endpoint-type.json <<'JSON'
{
  "name": "rest-endpoint",
  "display_name": "REST Endpoint",
  "description": "Proxied generic REST/HTTP endpoints",
  "fields": [{"name": "notes", "label": "Notes", "datatype": "string"}]
}
JSON

uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" --token-file "$TOKEN_FILE" \
  custom-type-create --config /tmp/rest-endpoint-type.json

uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" --token-file "$TOKEN_FILE" \
  custom-type-list --json | jq -r '.custom_types[].name'
```

Field datatypes are `string`, `text`, `number`, `bool`, `enum` (with `enum_values`), and `array<string>`. Mark a field `semantic: true` to feed it to search, `show_in_list: true` to show it on cards.

**Keep every field optional.** `custom-proxy-create` sends no attributes, so a type with a required field rejects it and you fall back to `custom-record-create` with a hand-written JSON body. Creating a type twice returns 409, so check `custom-type-list` first.

One type per backend shape works well: `rest-endpoint` for plain APIs, a separate `model-endpoint` if you want inference records to carry their own fields.

## Registering a route

Two fields turn any eligible entity into a route:

```json
{"is_proxied": true, "proxy_target_url": "https://api.example.com"}
```

An A2A agent already has a `url`, so the target falls back to it and you can omit `proxy_target_url`. A skill has no backend of its own and must name one.

The registry derives the client URL and never accepts one you supply:

| Entity | Registry path | Client URL |
|---|---|---|
| Skill `pdf` | `/skills/pdf` | `/gateway/skill/pdf/` |
| Custom record | `/rest-endpoint/<uuid>` | `/gateway/rest-endpoint/<uuid>/` |
| A2A agent | `/agents/<name>` | `/gateway/a2a_agent/<name>/` |

The type segment is singular even where the registry path is plural, and the prefix comes from `GATEWAY_PROXY_PREFIX`. Read `proxy_client_url` off the record rather than assembling it:

```bash
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" --token-file "$TOKEN_FILE" \
  custom-record-list --type rest-endpoint --json 2>/dev/null \
  | jq -r '.records[] | select(.is_proxied) | "\(.name)\t\(.proxy_client_url)/\t-> \(.proxy_target_url)"' \
  | column -t -s$'\t'
```

The trailing slash belongs to the URL. The generated nginx location ends in one, so a request to the bare path answers 301, and a client following that redirect turns a POST into a GET.

Whatever a caller appends travels to the backend. `GET /gateway/skill/pdf/README.md` fetches `README.md` from the target origin. A caller cannot escape the route prefix with `..`, a scheme, or userinfo.

### Telling callers how to use it

A client URL alone does not say what to append or what the request looks like. `proxy_connect_notes` is free text on the record for exactly that, capped at 2000 characters and never interpreted by the gateway:

```bash
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" --token-file "$TOKEN_FILE" \
  custom-proxy-create --type rest-endpoint --name openmeteo-forecast \
  --target-url https://api.open-meteo.com \
  --connect-notes 'Append /v1/forecast?latitude=..&longitude=..&current=temperature_2m — keyless, send only your gateway token.'
```

The notes appear in the UI's **Connect** popover, alongside the full client URL and a copy button, on the entity's card and in its detail view. The popover is the shortest path for a caller: it hands over the URL rather than asking them to derive it from `proxy_client_url`.

On a patch, omitting `proxy_connect_notes` leaves the stored value alone, so editing an unrelated field cannot blank it. Pass an empty string to clear it.

### Authorizing callers

The hop checks `{entity_type}/{registered_path}` against the caller's groups — `rest-endpoint/rest-endpoint/<uuid>` for a custom record, `skill/skills/pdf` for a skill. A legacy `methods: ["all"]` rule grants **no** HTTP verb. That split stops an old MCP grant from becoming arbitrary HTTP access, so every proxied entity needs a rule naming the verb or `http:*`:

```json
{"server": "rest-endpoint/rest-endpoint/<uuid>", "methods": ["GET", "POST"], "tools": []}
```

`{"server": "*", "methods": ["http:*"]}` grants every verb on every proxied entity. Convenient for an admin group, too broad for anything else. A missing rule shows up as 403 from the gateway.

`registry-admins` ships with that wildcard rule in `scripts/registry-admins.json`, so a **fresh** install has it. An **existing** deployment does not: the seed is read when DocumentDB is initialised, not on every boot, so the rule arrives only when the init task is re-run. Until then an admin gets 403 on every proxied route. Note also that a wildcard rule cannot be added through the management API — it refuses wildcard `server_access` entries — so the seed file is the only path for one. Per-entity rules like the example above are writable through the API and the UI scope editor.

## The four credential models

Pick by asking who owns the credential.

| Model | Owner | Per caller | Configure with |
|---|---|---|---|
| 1. Keyless | nobody | — | nothing |
| 2. Fixed operator header | operator | no, one key serves everyone | `custom_headers: [{name, value}]` |
| 3. Caller passthrough | the caller | yes, sent per request | `custom_headers: [{name, overridable: true}]` |
| 4. Per-user vault | the end user | yes, one bucket per person | [egress-credential-vault.md](egress-credential-vault.md) |

### 1. Keyless

The backend needs no credential. Before the request leaves, the gateway strips its whole internal header set, including the caller's `Authorization`, so the gateway token never reaches the backend.

Use it for public APIs and public files. Try it first when a proxied route misbehaves: with no credential in the way, it tests the routing and the token path by themselves.

### 2. Fixed operator header

You store a credential once. The registry encrypts it with a key derived from `SECRET_KEY`, hands it to the auth-server over an internal endpoint gated by a service token, and the hop sets it on the outbound request. The value stays out of nginx config, out of read models, and out of reach of callers.

A caller who sends the same header name gets ignored. The operator's value wins.

When the fetch of that credential fails — wrong `SECRET_KEY`, corrupt ciphertext, unreachable registry — the hop answers 502 and drops the request. It never forwards one stripped of its credential.

This is service-to-service auth, and the shape has consequences. The backend sees one client for all callers, so quotas pool and upstream logs show a single identity. Rotation cuts off everyone at once. Attribution lives on the gateway side. Suits a service API key, a partner key, an internal service token — a credential with no person behind it.

Because one key serves every authorized caller, the entity's scope rule is the only thing deciding who spends it.

### 3. Caller passthrough

You register a header name and mark it overridable. The caller supplies the value per request. Two shapes:

- **Name only.** A pure passthrough slot. The caller must send the header or the backend rejects the call.
- **Name plus an operator value.** A default the caller may override. Send nothing and the default applies.

Only registered overridable names survive the egress merge. A caller who invents a header name gets it dropped, so nobody steers your backend with headers you never approved.

`Authorization` is the one reserved name you may register, and only as overridable. A fixed `Authorization` is refused: a bearer usually stands for a person, and folding one person's token into a shared service credential is the mistake that rule prevents. Callers put the gateway token in `X-Authorization` and the backend token in `Authorization`. The same value in both trips a guard and returns 401, so the gateway's credential cannot leak downstream.

### 4. Per-user vault

Each user connects their own account and the gateway sends that user's token upstream, so the backend attributes and revokes per person. Use it whenever the credential represents a human. Separate mechanism, separate document.

---

## Pattern 1 — model inference (OpenAI and Bedrock)

An LLM API behind a gateway route, streaming token by token, with each caller bringing their own provider key. Credential model 3.

**Step 1. Create the type** if you have not already — see [Custom entity types](#custom-entity-types).

**Step 2. Register the record.** `--streaming true` sets `proxy_streaming`; `--auth-passthrough` registers `Authorization` as a caller-overridable slot with no stored value.

```bash
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" --token-file "$TOKEN_FILE" \
  custom-proxy-create \
  --type rest-endpoint --name openai-proxy \
  --target-url https://api.openai.com \
  --streaming true --auth-passthrough \
  --connect-notes 'Append /v1/chat/completions. Gateway JWT in X-Authorization, OpenAI key in Authorization.'
```

**Step 3. Grant the verbs.** Inference is POST and listing models is GET, so the caller's group needs both. See [Authorizing callers](#authorizing-callers).

**Step 4. Call it with the credential split.** The gateway token goes in `X-Authorization`, the provider key in `Authorization`:

```bash
BASE=$(curl -sS -H "Authorization: Bearer $GW" "$REGISTRY_URL/api/custom/rest-endpoint" \
  | jq -r --arg n openai-proxy --arg u "$REGISTRY_URL" \
      '.records[] | select(.name==$n) | $u + .proxy_client_url + "/"')

curl -N -sS -X POST "${BASE}v1/chat/completions" \
  -H "X-Authorization: Bearer $GW" \
  -H "Authorization: Bearer $(tr -d '\n' < /path/to/openai-key)" \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Count to ten."}],"stream":true}'
```

`-N` stops curl buffering so you can watch chunks arrive. A response that lands all at once means the entity is not streaming.

**Step 5. Verify streaming rather than assuming it.** A 200 proves nothing about incremental delivery. `tests/scripts/openai_gateway_client.py --mode stream` reports time to first chunk, chunk count, and the largest inter-chunk gap, and fails a buffered response.

### Amazon Bedrock

Same pattern, different upstream. Bedrock accepts a long-term API key as a plain bearer, so there is no SigV4 and no boto3:

```bash
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" --token-file "$TOKEN_FILE" \
  custom-proxy-create \
  --type rest-endpoint --name bedrock-proxy-use2 \
  --target-url https://bedrock-runtime.us-east-2.amazonaws.com \
  --streaming true --auth-passthrough
```

```bash
curl -sS -X POST "${BASE}model/us.anthropic.claude-opus-4-8/converse" \
  -H "X-Authorization: Bearer $GW" \
  -H "Authorization: Bearer $(tr -d '\n' < /path/to/bedrock-key)" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":[{"text":"Say hello."}]}],"inferenceConfig":{"maxTokens":128}}'
```

Two Bedrock specifics. The key must be valid for the region in the target URL; a key from another region returns a Bedrock-side 403 and the proxy path itself worked. And `converse-stream` returns an AWS event-stream — length-prefixed binary frames rather than server-sent events — so `curl -N` shows bytes with no `data:` lines and a client needs a frame decoder. `tests/scripts/bedrock_gateway_client.py` uses `botocore.eventstream`.

### Streaming limits

Four ceilings guard a streaming route: a slot pool with a bounded wait (503 when full), an absolute lifetime, an idle read timeout covering both first byte and inter-chunk gaps, and a raw byte cap (413). Every stream records one terminal outcome, so in-flight streams equal `started` minus the sum of the terminals.

Switch to model 2 when one team key should serve everyone, and accept that the provider then sees a single client for all of them.

## Pattern 2 — generic REST API

Any HTTP API. Model 2 when the service key belongs to you, model 1 when the API is public.

### With your service key

**Step 1. Create the type** — see [Custom entity types](#custom-entity-types).

**Step 2. Register the record with the credential.** `custom_headers` is accepted at create, and it needs the API rather than the CLI, which only covers `Authorization` via `--auth-passthrough`. Omitting `overridable` makes the header fixed:

```bash
KEY=$(tr -d '\n' < /path/to/keyfile)

curl -sS -X POST "$REGISTRY_URL/api/custom/rest-endpoint" \
  -H "Authorization: Bearer $GW" -H 'Content-Type: application/json' \
  -d "$(jq -n --arg k "$KEY" '{
    name: "api-ninjas-weather",
    description: "API Ninjas weather",
    visibility: "private",
    is_proxied: true,
    proxy_target_url: "https://api.api-ninjas.com",
    custom_headers: [{name: "X-Api-Key", value: $k}]
  }')" -w '\nHTTP %{http_code}\n'
```

**Step 3. Confirm the secret is hidden.** The 201 body should carry `custom_header_names: ["X-Api-Key"]` with no value and no ciphertext, and the plaintext key should appear nowhere in it.

**Step 4. Call it.** Callers send a gateway token alone; the gateway adds the key:

```bash
curl -sS -H "Authorization: Bearer $GW" "${BASE}v1/weather?lat=39.2806&lon=-77.4136"
```

**Step 5. Check the failure direction.** Break the stored credential and the route must answer 502, never forward the request bare. Section 5 of the regression suite does this with a reversible edit.

### With no credential

```bash
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" --token-file "$TOKEN_FILE" \
  custom-proxy-create \
  --type rest-endpoint --name openmeteo-forecast \
  --target-url https://api.open-meteo.com \
  --connect-notes 'Append /v1/forecast?latitude=&longitude=&current=temperature_2m'
```

`--connect-notes` stores free text the UI shows in the Connect panel. Put the sub-path and a working example there. The gateway never reads it.

### Credentials in the query string

Some APIs take their key as a query parameter. The registered target keeps its query string on every outbound request and registered keys beat caller-supplied ones, so `https://api.example.com/v1?api_key=abc123` does work. Avoid it for a secret: a query string in the target is stored in plaintext, written into nginx config on disk, passed as a header value between nginx and the hop, and returned to admins in the read model. None of the protections around `custom_headers` apply. Where a query-parameter key is the only option, treat that credential as public.

## Pattern 3 — skills and documents

A skill has no backend of its own, so proxying one needs an explicit target. Point it at the origin serving its files and the gateway becomes an authenticated route to them. Usually model 1.

**Step 1. Pick an existing skill.** No type creation and no re-registration. In the UI, click the edit icon on the skill tile and tick **Enable proxying**; the steps below are the API equivalent.

**Step 2. Flip it, naming the origin.** `PUT` patches, so omitted fields keep their values, though `name`, `description`, and `skill_md_url` must appear for the body to validate. You must own the skill, or be admin. `skill-register` carries no proxy flags, so this is API or UI only:

```bash
curl -sS -X PUT "$REGISTRY_URL/api/skills/pdf" \
  -H "Authorization: Bearer $GW" -H 'Content-Type: application/json' \
  -d '{"name":"pdf","description":"PDF processing",
       "skill_md_url":"https://raw.githubusercontent.com/anthropics/skills/main/skills/pdf/SKILL.md",
       "is_proxied":true,
       "proxy_target_url":"https://raw.githubusercontent.com"}' \
  -w '\nHTTP %{http_code}\n'
```

The response carries `proxy_client_url: "/gateway/skill/pdf"` — name-based for skills, not UUID-based.

**Step 3. Fetch a file through the route.** Everything after the base URL goes to the origin:

```bash
curl -sS -H "Authorization: Bearer $GW" \
  "$REGISTRY_URL/gateway/skill/pdf/anthropics/skills/main/skills/pdf/SKILL.md" > /tmp/gw.md
```

**Step 4. Confirm the bytes match the origin.** A 200 alone does not prove the hop forwarded faithfully:

```bash
curl -sS https://raw.githubusercontent.com/anthropics/skills/main/skills/pdf/SKILL.md > /tmp/direct.md
cmp /tmp/gw.md /tmp/direct.md && echo IDENTICAL
```

**Step 5. Narrow the target if you can.** `https://raw.githubusercontent.com` reaches every file on that host through your route. `https://raw.githubusercontent.com/org/repo/main/skills/pdf/SKILL.md` reaches one file, and its route root returns that file with no sub-path.

A proxied route differs from `GET /api/skills/{path}/content`, which returns the copy the registry fetched and cached. The route fetches live and passes the upstream `etag` and `cache-control` through.

Revert with a second PUT carrying `"is_proxied": false`.

## Pattern 4 — A2A agents

An agent carries a `url`, so the effective target falls back to it and `is_proxied: true` is enough.

**Step 1. Flip the agent** — the edit icon on its tile, or:

```bash
curl -sS -X PATCH "$REGISTRY_URL/api/agents/<name>" \
  -H "Authorization: Bearer $GW" -H 'Content-Type: application/json' \
  -d '{"isProxied": true}' -w '\nHTTP %{http_code}\n'
```

**Step 2. Fetch the agent card through the gateway:**

```bash
curl -sS -H "Authorization: Bearer $GW" \
  "$REGISTRY_URL/gateway/a2a_agent/<name>/.well-known/agent.json"
```

Two limits today. An agent takes no `proxy_streaming` and no `custom_headers` through the API, so it gets a route and nothing more. And because the target falls back to `url`, changing `url` repoints the backend: the registry re-validates the new target, re-pins its addresses, and clears stored upstream headers so a credential registered for the old host cannot reach the new one.

A2A itself is untouched. Proxying an agent adds a route; how the registry invokes agents and serves their cards elsewhere stays the same.

## Choosing

| Backend | Credential model | Settings |
|---|---|---|
| Public API or public files | 1 | target only |
| Service API key in a header | 2 | fixed `custom_headers` entry |
| Every caller has their own key | 3 | overridable name, no value |
| Shared default, callers may override | 3 | overridable name with a value |
| Credential belongs to a person | 4 | per-user vault |
| Token-by-token responses | any | add `proxy_streaming: true` |

## Managing a route

### Rotate a credential

Use the dedicated endpoint. The general update refuses `custom_headers` on purpose: accepting it would skip create-path validation and write plaintext values to storage. Needs `modify_<type>_entity`, or admin.

```bash
curl -sS -X PATCH "$REGISTRY_URL/api/custom/rest-endpoint/<uuid>/upstream-headers" \
  -H "Authorization: Bearer $GW" -H 'Content-Type: application/json' \
  -d "$(jq -n --arg k "$NEW_KEY" '{custom_headers: [{name: "X-Api-Key", value: $k}]}')"
```

Skills use `PATCH /api/skills/{path}/upstream-headers`. In the UI, the same edit form carries an **Upstream headers** editor; leave a value box blank to keep the stored secret.

Four rules:

- The call **replaces the whole set**. Omit a header and it is gone.
- Leaving a value out **keeps** the stored one. That is what lets the UI editor show registered names with empty value boxes without wiping credentials on an unrelated edit. It also means you cannot turn an operator default into a caller-only slot by omitting the value — clear the set first, then re-add the name.
- An empty list clears everything.
- Rotation rewrites the nginx route and reloads. For a few seconds after the call, requests can see the old behaviour: adding the first credential to an entity reads as a missing credential at the backend until the reload lands. Wait before asserting.

### Repoint a backend

Change `proxy_target_url` and the registry re-validates the new target and re-pins its addresses. If the host changed, it also **clears stored upstream headers**, so a credential registered for the old host cannot be sent to the new one. Re-register the credential after a repoint.

For an A2A agent, changing `url` counts as a repoint, since the target falls back to it.

### Disable or remove a route

Set `is_proxied: false` on the entity, or untick **Enable proxying** in the UI. The location block disappears on the next render and `proxy_client_url` clears. Deleting the entity does the same.

To turn the whole feature off, set `GATEWAY_GENERIC_PROXY_ENABLED=false` and restart. Every generic block stops rendering, `/validate` stops minting generic tokens, and the render path issues no extra database queries.

## Monitoring

| Metric | Read it as |
|---|---|
| `mcpgw_registry_generic_proxy_slot_rejected_total{pool}` | 503s from a saturated concurrency pool, labeled `buffered` or `stream`. A non-zero rate means raise `GATEWAY_GENERIC_STREAM_MAX_CONCURRENCY` or shed load. |
| `mcpgw_registry_generic_proxy_stream_outcome_total{outcome}` | `started`, `completed`, `duration_timeout`, `byte_cap`, `upstream_error`, `client_closed`. In-flight streams equal `started` minus the sum of the terminals. |
| `mcpgw_registry_gateway_generic_blocks_dropped_total{reason}` | routes the render path refused, `invalid` (bad target) or `collision` (the location path is already claimed). Non-zero means an entity is registered and unreachable. |
| `mcpgw_registry_gateway_egress_policy_unverified` | 1 means the startup self-check reached cloud metadata and the feature is latched off for the process. A standing 1 on an enabled deployment is an alert. |

Log lines worth alerting on. Each is the literal text the code emits, so it is safe to match on:

```
Generic proxy egress self-check FAILED               the self-check reached cloud metadata and latched the feature off (auth_server)
generic upstream-headers REFUSED: ...                a vend refusal; the suffix names the branch (registry)
generic upstream-headers vend: registry unreachable  the auth-server could not reach the registry's vend listener (auth_server)
Dropping generic block for ...                       a route did not render, so the entity is registered and unreachable (registry)
```

`Generic proxy egress self-check PASSED; feature active` is the healthy counterpart, logged once at startup.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `301` | The trailing slash is missing. The location ends in one. `curl -L` follows it, and a POST becomes a GET when it does. |
| `200` with `text/html`, ~889 bytes | No proxy location matched, so nginx served the frontend shell. Wrong entity-type spelling (`skills` for `skill`), the entity is not proxied, or the record was registered seconds ago and nginx has not regenerated its config yet — retry. Check the content type, not the status: this failure returns `200`, so a status-only check reports success. |
| `401` | Expired or missing gateway token, or the equal-token guard fired because `Authorization` matched `X-Authorization`. Check the token lifetime first. |
| `403` from the gateway | The caller's group has no `server_access` rule granting this verb on this authz key. `methods: ["all"]` does not count. |
| `403` on a create or update | Missing `create_<type>_entity` or `modify_<type>_entity`, or a non-admin trying to manage a custom type. |
| `404` on a record you know exists | Missing `list_<type>_entity` for that record. The registry hides existence rather than confirming it. |
| `404` on a gateway route | The feature is off or self-disabled, the client URL was assembled by hand, or nginx has not reloaded. |
| `409` on `custom-type-create` | The type exists. Check `custom-type-list`. |
| `502 {"detail":"Upstream auth unavailable"}` | The credential fetch failed and the hop refused. Causes: `SECRET_KEY` changed since the credential was stored, corrupt ciphertext, the registry vend listener unreachable, or a target-identity mismatch. Grep the registry log for `upstream-headers REFUSED`. |
| `503` | The concurrency pool is full. |
| Backend rejects the credential right after a rotation | The nginx reload has not landed. Wait and retry. |
| Backend 401 or 403 | The key is stale, or wrong for the region. The proxy path itself worked. |

## Security practices

- **Scope every credentialed entity deliberately.** One fixed credential serves every authorized caller, so the scope rule is the access control. Avoid `{"server": "*", "methods": ["http:*"]}` outside an admin group.
- **Aim targets narrowly.** A bare origin exposes every path on that host through your route. Name the sub-tree or the file when you can.
- **Keep credentials in headers.** A key in the target's query string is stored and logged in plaintext.
- **Use the per-user vault for anything representing a person.** A shared credential gives you no upstream attribution and no per-user revocation.
- **Leave the egress self-check on.** It is the DNS-rebind defense for the whole feature. Disable it only for a local test run.
- **Keep `SECRET_KEY` stable and backed up.** Losing it makes every stored upstream credential undecryptable, and every affected route answers 502 until you re-register each one.
- **Rotate through the dedicated endpoint** so values stay write-only and never pass through a general update.

## Testing

`tests/scripts/gateway-proxy-regression.md` covers everything above: routing, ingress auth, per-verb authorization, all four credential models, rotation, streaming, and the fail-closed paths. Run its gate first:

```bash
./tests/scripts/gateway-proxy-preflight.sh
```

The gate checks credential files, token expiry, configuration, and stack health, and refuses to let a half-configured run start.

## See also

- [design/gateway-generic-proxy.md](design/gateway-generic-proxy.md) — request sequence, storage fields, render path, failure shapes
- [egress-credential-vault.md](egress-credential-vault.md) — per-user credentials
- [unified-parameter-reference.md](unified-parameter-reference.md) — every `GATEWAY_GENERIC_*` setting across compose, terraform, and helm
