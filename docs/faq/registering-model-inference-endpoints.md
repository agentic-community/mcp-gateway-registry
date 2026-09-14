# How do I let my team call OpenAI or Bedrock through the gateway?

Put a model-inference endpoint behind the gateway as a **proxied custom record**. Callers then hold a gateway token and reach the provider through a registry URL, and every call carries the registry's authorization, audit, and rate-limit checks.

Two credential models exist, and you pick one per record:

- **Caller passthrough (the default, and what you want for user-facing traffic)** — each caller brings their own provider key. The registry stores nothing. The provider sees one client per user, so quota, rate limits, and billing stay attributable to whoever made the call. Use this whenever a human is behind the request.
- **Shared operator key (optional, for service-to-service traffic)** — the record carries one key, encrypted, and callers never see it. That suits a batch job, a scheduled pipeline, or an agent running under a machine identity. The provider then sees a single client, so its quota is pooled and its rate limits are shared, and per-user attribution at the provider is gone; the registry's audit log still records which caller made each request. Do not reach for this to spare users the trouble of holding a key — a shared credential turns one user's runaway loop into everybody's throttling. Note the header rule: for `Authorization`, which is what OpenAI and Bedrock use, the slot must still be registered as **caller-overridable**, and it may carry an operator default that any caller who sends their own key overrides. A fixed, non-overridable `Authorization` is refused at registration, because operator-owned bearers belong in the [egress credential vault](../egress-credential-vault.md). Providers that read a different header, such as Azure OpenAI's `api-key`, can hold a fixed operator value.

The steps below use caller passthrough, which is what `--auth-passthrough` sets up. For a stored operator credential, register the header without `--auth-passthrough` and see the [operational guide](../gateway-proxy-operational-guide.md); for per-user third-party credentials the gateway can broker OAuth instead, covered in the [per-user egress credential vault](../egress-credential-vault.md).

## What you need before you start

- `GATEWAY_GENERIC_PROXY_ENABLED=true` and `CUSTOM_ENTITY_TYPES_ENABLED=true` in `.env`, with the stack restarted. Both default to false. See [what changes when you upgrade](gateway-proxy-backwards-compatibility.md).
- `DEPLOYMENT_MODE=with-gateway`. In `registry-only` mode no route renders.
- Registry admin access — creating a custom entity type is admin-only.
- A provider key: an OpenAI API key, or a Bedrock long-term API key scoped to the region you target.

Set the two variables the commands below use:

```bash
export REGISTRY_URL=https://mcpgateway.example.com
export TOKEN_FILE=.token
```

## Step 1 — Create the custom type, once

A custom type is a schema for records of that kind. Create it from a JSON descriptor:

```bash
cat > /tmp/rest-endpoint.json <<'EOF'
{
  "name": "rest-endpoint",
  "description": "Proxied generic REST/HTTP endpoints",
  "fields": [
    {"name": "notes", "datatype": "string", "required": false}
  ]
}
EOF

uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" --token-file "$TOKEN_FILE" \
  custom-type-create --config /tmp/rest-endpoint.json
```

**Every field must be optional.** `custom-proxy-create` sends no attributes, so a required field rejects the record you are about to create. One type serves every provider — OpenAI, Bedrock, and any other REST backend become records of `rest-endpoint`.

Check what already exists before creating a duplicate:

```bash
uv run python api/registry_management.py --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" custom-type-list --json | jq -r '.custom_types[].name'
```

## Step 2 — Register the endpoint

```bash
# OpenAI
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" --token-file "$TOKEN_FILE" \
  custom-proxy-create --type rest-endpoint --name openai-proxy \
  --target-url https://api.openai.com \
  --streaming true --auth-passthrough

# Bedrock, us-east-2
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" --token-file "$TOKEN_FILE" \
  custom-proxy-create --type rest-endpoint --name bedrock-proxy-use2 \
  --target-url https://bedrock-runtime.us-east-2.amazonaws.com \
  --streaming true --auth-passthrough
```

`--auth-passthrough` registers `Authorization` as a **caller-overridable** upstream header with no stored value: the caller's key travels to the provider, and the registry keeps no secret. A fixed operator `Authorization` value is refused at registration — operator-owned bearers belong in the [per-user egress credential vault](../egress-credential-vault.md).

`--streaming true` sets `proxy_streaming` on the record. That is a property of the **record**, not of a request: an entity with it on takes the streaming path on every call, whatever the request body asks for.

Read back the client URL rather than assembling it:

```bash
uv run python api/registry_management.py --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" custom-record-list --type rest-endpoint --json \
  | jq -r '.records[] | "\(.name)  \(.proxy_client_url)  -> \(.proxy_target_url)"'
```

```text
openai-proxy         /gateway/rest-endpoint/6160de6a-...  -> https://api.openai.com
bedrock-proxy-use2   /gateway/rest-endpoint/d7ff8465-...  -> https://bedrock-runtime.us-east-2.amazonaws.com
```

## Step 3 — Grant the HTTP verbs

This is where a first attempt usually fails. **A legacy `methods: ["all"]` rule grants no HTTP verb.** The split is deliberate: an existing MCP grant must not become arbitrary HTTP access. The caller's group needs an explicit verb — or `http:*` — on the record's authz key:

```text
{entity_type}/{registered_path}
```

For the record above that key is `rest-endpoint/rest-endpoint/6160de6a-...`. Read it from the record rather than building it by hand, and note it is not the client URL: the generated path drops the namespace segment, so a skill registered at `/skills/pdf` has the key `skill/skills/pdf` and the client URL `/gateway/skill/pdf`.

Without the rule, the gateway returns **403** and the `server` label on `mcpgw_registry_auth_request_total{success="false"}` holds the exact key to name in the rule. The helper scripts write it for you:

```bash
uv run python tests/scripts/openai_gateway_client.py --registry-url "$REGISTRY_URL" \
  --entity openai-proxy --ensure-scope
```

## Step 4 — Call it, with the credentials split

Two headers, two different secrets:

| Header | Value |
|---|---|
| `X-Authorization` | the gateway JWT — authenticates the caller to the registry |
| `Authorization` | the provider key — forwarded to OpenAI or Bedrock |

Sending the gateway token in both trips the equal-token guard and returns **401**. That guard exists so a gateway credential cannot leak to a third-party backend.

```bash
export GW=$(jq -r '.tokens.access_token // .access_token' .token)
export OAI=$(cat .scratchpad/.oai)
export BASE="$REGISTRY_URL/gateway/rest-endpoint/6160de6a-..."

curl -sS --compressed -X POST "$BASE/v1/chat/completions" \
  -H "X-Authorization: Bearer $GW" -H "Authorization: Bearer $OAI" \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Say hello in five words."}]}' \
  | jq -r '.choices[0].message.content'
```

Always pass `--compressed`. The gateway gzips JSON, and without it a working call prints unreadable bytes while `-w '%{http_code}'` still reports 200.

The official SDKs work unchanged — point the base URL at the client path and give the SDK the provider key:

```python
from openai import OpenAI

client = OpenAI(
    base_url=f"{REGISTRY_URL}/gateway/rest-endpoint/6160de6a-.../v1",
    api_key=openai_key,                              # travels as Authorization
    default_headers={"X-Authorization": f"Bearer {gateway_jwt}"},
)
print(client.chat.completions.create(model="gpt-4o-mini", messages=[...]))
```

## Step 5 — Verify streaming instead of assuming it

A 200 says nothing about incremental delivery. A buffered response and a streamed one both return 200, and the difference only shows in the timing of the chunks. Two scripts measure it and fail when the response arrives in one piece:

```bash
uv run python tests/scripts/openai_gateway_client.py --registry-url "$REGISTRY_URL" \
  --entity openai-proxy --mode all
```

Output from a live run against a deployment with the record above:

```text
Entity      : rest-endpoint/openai-proxy
Streaming   : True
Header names: ['Authorization'] (overridable: ['Authorization'])
Authz key   : rest-endpoint/rest-endpoint/6160de6a-...
Scope grant : present in group 'registry-admins' for GET, POST
models      : OK 125 models in 1.30s
chat        : OK in 0.94s -> 'Hello! How are you today?'
stream      : 10 chunks, first at 0.90s, span 0.11s, max gap 0.071s
stream      : OK (incremental delivery confirmed)
Summary     : 3/3 checks passed (models, chat, stream)
```

Ten chunks spread over 0.11s is real streaming. One chunk, or ten chunks with a max gap of 0.000s, means the response was buffered somewhere and the script fails.

## Bedrock specifics

The request the gateway forwards is a plain Converse call with a bearer key — no SigV4, no boto3:

```text
POST https://bedrock-runtime.<region>.amazonaws.com/model/<model-id>/converse
Authorization: Bearer <bedrock api key>
{"messages": [{"role": "user", "content": [{"text": "Hello"}]}]}
```

Three things differ from OpenAI:

**The key is region-scoped.** It must match the region in `proxy_target_url`. A key from another region returns a Bedrock-side 403 while the proxy path itself worked.

**`converse-stream` is not SSE.** It returns `application/vnd.amazon.eventstream` — length-prefixed binary frames. `curl -N` shows bytes and no `data:` lines, and a client needs a frame decoder. `tests/scripts/bedrock_gateway_client.py` decodes them with `botocore.eventstream`:

```bash
uv run python tests/scripts/bedrock_gateway_client.py --registry-url "$REGISTRY_URL" \
  --entity bedrock-proxy-use2 --mode all --model us.anthropic.claude-opus-4-8
```

**Read whose 403 it is.** A gateway scope denial and a provider refusal share the status code, so check the body. This is a real failure from a run where the stored key had lapsed:

```text
converse    : FAILED - HTTP 403
Upstream/gateway reported HTTP 403: {"Message":"Bearer Token has expired"}
```

`{"Message":"Bearer Token has expired"}` is Bedrock's own wording, so the request reached Bedrock and the hop is fine — the provider key needs replacing. A gateway denial returns `{"error": "Access forbidden"}` instead, and that one means the scope rule from step 3 is missing.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `200` with `text/html`, roughly 889 bytes | No gateway location matched, so nginx served the frontend shell. Wrong type spelling, the record is not proxied, or nginx has not regenerated yet. Check the content type, not the status |
| `401` | Expired gateway token, or the same value in `X-Authorization` and `Authorization` |
| `403` with `{"error": "Access forbidden"}` | The caller's group has no verb rule on the authz key. `methods: ["all"]` does not count |
| `403` with a provider message | The provider refused: wrong region, expired key, or no access to that model |
| `404` on the gateway route | The feature is off, self-disabled by the egress self-check, or nginx has not reloaded |
| `301` | The trailing slash is missing. `curl -L` follows it, and a POST becomes a GET when it does |
| `502 Upstream auth unavailable` | The credential vend failed. For a passthrough record, check that the caller sent `Authorization` |
| `503` | The concurrency pool is full |

Watch `mcpgw_registry_generic_proxy_request_total{entity_type,outcome}` for what callers actually get — `auth_request_total{success}` reports the authorization decision, so a request that authorizes and then fails at the hop reads there as a success. See [Observability](../OBSERVABILITY.md#the-13-hop-outcomes-and-why-the-set-is-not-just-status-code).

## Related documentation

- [Gateway generic proxy operational guide](../gateway-proxy-operational-guide.md)
- [What changes when I upgrade to a release with the gateway generic proxy?](gateway-proxy-backwards-compatibility.md)
- [Per-User Egress Credential Vault](../egress-credential-vault.md) — for operator-owned credentials and per-user OAuth
- [Observability](../OBSERVABILITY.md) — the metrics named above
