# Gateway generic proxy — regression suite

Covers the generic reverse proxy that serves non-MCP registry entities (skills, agents, custom types) through the gateway: routing, ingress auth, per-verb authorization, response streaming, and the three upstream-credential models.

Run this before tagging a release that touches `auth_server/server.py`, `registry/schemas/proxy_mixin.py`, `registry/api/egress_auth_routes.py`, `registry/core/nginx_service.py`, `registry/utils/credential_encryption.py`, or any `charts/` and `terraform/` file carrying a `GATEWAY_GENERIC_*` setting.

Every test names its own command and its own pass line. Nothing depends on a previous test's output except where a fixture is named. Section 8 tears down everything section 2 creates.

`pytest` never runs this file. It needs a deployed stack and real backend credentials, so an operator drives it. Every command assumes the repository root as the working directory, not this directory. The three Python clients it calls in section 7 sit beside it.

---

## STOP — required credential files

This suite reads real secrets from four files. Nothing here works without them, and a missing key does not look like a missing key: it looks like a proxy bug. Section 5 in particular turns a missing credential into an upstream `400`, which reads exactly like the fail-open bug that section exists to catch.

| File | Needed by | Contents | How to get it |
|---|---|---|---|
| `.token` | every section | gateway JWT for an admin whose group authorizes HTTP verbs | registry UI sidebar, or `./api/get-m2m-token.sh` |
| `.scratchpad/pr-1714/api-ninja` | sections 5, 6 | api-ninjas.com API key, one line, no newline needed | sign up at api-ninjas.com |
| `.scratchpad/.oai` | section 7 | OpenAI API key, one line | platform.openai.com, API keys |
| `.scratchpad/.bedrock` | section 7 | Amazon Bedrock long-term API key valid for **us-east-2** | Bedrock console, long-term API key |

All four live outside version control. `.scratchpad/` is git-ignored; `.token` is in `.gitignore`. Read each key from its file inside the command that needs it. Never paste one into a shell argument you keep, a test file, a commit, or a log.

### Run the gate first

```bash
./tests/scripts/gateway-proxy-preflight.sh            # gate every section
./tests/scripts/gateway-proxy-preflight.sh 3 4 5 6    # gate only these sections
```

It checks each credential file for existence and non-emptiness, decodes `.token` and fails on an expired JWT, checks the four `.env` settings the suite depends on, checks `/health`, checks that the auth-server has not latched the feature off, and checks that `curl`, `jq`, `python3`, and `docker` are on `PATH`. It prints one line per check and exits non-zero on the first missing prerequisite.

**Do not start testing until it prints `PREFLIGHT PASSED`.** A failing run ends with:

```
=============================================================
 PREFLIGHT FAILED — DO NOT RUN THE SUITE
 Fix every FATAL above. A partial run produces results
 nobody can trust: a missing key looks like a proxy bug.
=============================================================
```

Section 1 below repeats the configuration detail the gate checks, for when you need to fix something it flagged.

---

## 1. Prerequisites

### 1.1 Configuration

Set these in `.env`, then rebuild. The feature ships off, so a release that leaves `GATEWAY_GENERIC_PROXY_ENABLED=false` skips this suite.

```bash
GATEWAY_GENERIC_PROXY_ENABLED=true
DEPLOYMENT_MODE=with-gateway          # the proxy only renders in gateway mode
GATEWAY_PROXY_PREFIX=gateway
GATEWAY_PROXY_ALLOW_PRIVATE_TARGETS=false
GATEWAY_GENERIC_TLS_VERIFY=true
GATEWAY_GENERIC_REQUIRE_BEARER_FOR_WRITES=true
GATEWAY_GENERIC_STREAM_READ_TIMEOUT_SECONDS=3600
GATEWAY_GENERIC_STREAM_MAX_CONCURRENCY=8
GATEWAY_GENERIC_ACQUIRE_TIMEOUT_SECONDS=5
GATEWAY_GENERIC_STREAM_MAX_DURATION_SECONDS=3600
GATEWAY_GENERIC_STREAM_MAX_BYTES=104857600
```

`SECRET_KEY` must stay stable across the run. It derives the encryption key for stored upstream credentials, so rotating it mid-suite turns every section 5 test into a 502.

On an EC2 host whose containers can reach `169.254.169.254`, the startup egress self-check disables the feature for the process. Set `GATEWAY_EGRESS_SELFCHECK_ENABLED=false` for the run and leave it on everywhere else.

### 1.2 Stack

```bash
docker compose up -d
curl -s -o /dev/null -w '%{http_code}\n' http://localhost/health        # expect 200
docker compose logs auth-server | grep -iE "generic.proxy|egress" | tail -20
```

A line reading `Generic proxy egress self-check FAILED` means the self-check latched the feature off for the process; fix 1.1 and restart before going further. `Generic proxy egress self-check PASSED; feature active` is the healthy line. Match these literally — they are the exact text `auth_server/server.py` emits.

### 1.3 Shell variables

```bash
cd /path/to/mcp-gateway-registry          # repository root; every path below is relative to it
export ORIGIN=http://localhost
export DOCUMENTDB_PASSWORD=$(grep -m1 '^DOCUMENTDB_PASSWORD=' .env | cut -d= -f2-)

# Read a secret from a file, or stop. Use this everywhere a key is needed so a
# missing file fails on the spot instead of sending an empty header and turning
# into an upstream 400 that reads like a proxy bug.
secret() {
  [ -s "$1" ] || { printf 'FATAL: missing or empty credential file %s\n' "$1" >&2; return 1; }
  tr -d '\n' < "$1"
}

export GW=$(secret .token | jq -r .tokens.access_token)
[ -n "$GW" ] && [ "$GW" != null ] || echo 'FATAL: no usable token in .token — run the preflight gate' >&2
```

**Against a CloudFront-fronted deployment, add `--compressed` to every `curl`.** CloudFront gzips JSON responses, and `curl` does not decompress unless asked, so a working call prints unreadable bytes — or nothing at all, if the shell swallows them. A `curl -w '%{http_code}'` reports 200 while the body looks empty, which reads as a gateway bug and is not one. `requests` and the scripted clients decompress transparently, so this only bites hand-run `curl`. Local `http://localhost` runs are unaffected, which is why it does not show up until you test a real deployment.

```bash
curl -sS --compressed -H "X-Authorization: Bearer $GW" "${BASE}v1/models" | jq '.data | length'
```

`$GW` must belong to a group that authorizes HTTP verbs on the entities under test. See 1.5.

Tokens expire. When a test returns 401 for no obvious reason, check the lifetime first:

```bash
python3 - <<'PY'
import base64, json, time
t = json.load(open('.token'))['tokens']['access_token']
p = t.split('.')[1]; p += '=' * (-len(p) % 4)
left = json.loads(base64.urlsafe_b64decode(p))['exp'] - int(time.time())
print(f"{left // 60} minutes left" if left > 0 else f"EXPIRED {abs(left) // 60} minutes ago")
PY
```

### 1.4 Backend keys

| File | Used by |
|---|---|
| `.scratchpad/pr-1714/api-ninja` | fixture F3, sections 5 and 6 |
| `.scratchpad/.oai` | fixture F4, section 4 |
| `.scratchpad/.bedrock` | fixture F5, section 4 |

Read each key from its file inside the command. Never paste one into a shell argument you keep, a test file, or a log.

### 1.5 Scope grant

The generic hop authorizes `{entity_type}/{registered_path}` — for a custom record that reads `rest-endpoint/rest-endpoint/<uuid>`, for a skill `skill/skills/<name>`. A legacy `methods: ["all"]` rule does **not** grant an HTTP verb. That split is deliberate, so the group needs an explicit rule:

```bash
curl -sS -H "Authorization: Bearer $GW" \
  $ORIGIN/api/management/iam/groups/registry-admins | jq '.server_access'
```

Pass line: one rule matches the entity (or `"server": "*"`) and its `methods` contain either the verb or `http:*`.

To add it, use the IAM scope editor or let a scripted client write it:

```bash
uv run python tests/scripts/openai_gateway_client.py --ensure-scope --mode models
```

## 2. Fixtures

Six entities cover every path. Create the ones the sections you plan to run need.

### F1 — the `rest-endpoint` custom type

`custom-type-create` takes a JSON descriptor, not flags. Keep every field optional: a type with a required field cannot be registered through `custom-proxy-create` and needs a full `custom-record-create` body instead.

```bash
cat > /tmp/rest-endpoint-type.json <<'JSON'
{
  "name": "rest-endpoint",
  "display_name": "REST Endpoint",
  "description": "Proxied generic REST/HTTP endpoints",
  "fields": [
    {"name": "notes", "label": "Notes", "datatype": "string"}
  ]
}
JSON

uv run python api/registry_management.py custom-type-create --config /tmp/rest-endpoint-type.json
```

Skip this when `custom-type-list` already shows the type; creating it twice fails.

### F2 — keyless upstream (Open-Meteo)

```bash
uv run python api/registry_management.py custom-proxy-create \
  --type rest-endpoint --name openmeteo-forecast \
  --target-url https://api.open-meteo.com \
  --connect-notes 'Append /v1/forecast?latitude=&longitude=&current=temperature_2m'
```

### F3 — fixed operator credential (api-ninjas)

`custom_headers` is accepted at create only. Omitting `overridable` makes the header fixed.

```bash
KEY=$(secret .scratchpad/pr-1714/api-ninja) || echo 'cannot register F3 without the key' >&2

curl -sS -X POST $ORIGIN/api/custom/rest-endpoint \
  -H "Authorization: Bearer $GW" -H 'Content-Type: application/json' \
  -d "$(jq -n --arg k "$KEY" '{
    name: "api-ninjas-weather",
    description: "API Ninjas weather, fixed operator X-Api-Key",
    visibility: "private",
    is_proxied: true,
    proxy_target_url: "https://api.api-ninjas.com",
    custom_headers: [{name: "X-Api-Key", value: $k}]
  }')" -w '\nHTTP %{http_code}\n'
```

### F4 — caller passthrough plus streaming (OpenAI)

```bash
uv run python api/registry_management.py custom-proxy-create \
  --type rest-endpoint --name openai-proxy \
  --target-url https://api.openai.com --streaming true --auth-passthrough
```

### F5 — caller passthrough plus AWS event-stream (Bedrock)

```bash
uv run python api/registry_management.py custom-proxy-create \
  --type rest-endpoint --name bedrock-proxy-use2 \
  --target-url https://bedrock-runtime.us-east-2.amazonaws.com \
  --streaming true --auth-passthrough
```

The key in `.scratchpad/.bedrock` must be valid for `us-east-2`, matching the target. A key from another region returns a Bedrock-side 403 and the passthrough still worked.

### F6 — a proxied skill

A skill has no native backend, so proxying one needs an explicit target. Flip an existing skill; no re-registration. `PUT` patches (it applies `exclude_unset`), but the three required fields must appear or the body fails validation.

```bash
curl -sS -X PUT $ORIGIN/api/skills/pdf \
  -H "Authorization: Bearer $GW" -H 'Content-Type: application/json' \
  -d '{"name": "pdf",
       "description": "Backwards compat update test",
       "skill_md_url": "https://raw.githubusercontent.com/anthropics/courses/refs/heads/master/prompt_engineering_interactive_tutorial/README.md",
       "is_proxied": true,
       "proxy_target_url": "https://raw.githubusercontent.com"}' \
  -w '\nHTTP %{http_code}\n'
```

`skill-register` carries no proxy flags, so this is API or UI only.

### 2.1 Resolve the client URLs

Never assemble a client URL by hand. The server derives it and the tests read it:

```bash
BASE_METEO=$(curl -sS -H "Authorization: Bearer $GW" $ORIGIN/api/custom/rest-endpoint \
  | jq -r --arg n openmeteo-forecast --arg o "$ORIGIN" \
      '.records[] | select(.name==$n) | $o + .proxy_client_url + "/"')

BASE_NINJA=$(curl -sS -H "Authorization: Bearer $GW" $ORIGIN/api/custom/rest-endpoint \
  | jq -r --arg n api-ninjas-weather --arg o "$ORIGIN" \
      '.records[] | select(.name==$n) | $o + .proxy_client_url + "/"')

BASE_OPENAI=$(curl -sS -H "Authorization: Bearer $GW" $ORIGIN/api/custom/rest-endpoint \
  | jq -r --arg n openai-proxy --arg o "$ORIGIN" \
      '.records[] | select(.name==$n) | $o + .proxy_client_url + "/"')
```

Two details the `jq` encodes on purpose. The trailing slash matters: the stored value has none, the nginx location ends in one, so the bare path answers 301 and most clients downgrade a POST to GET when they follow it. `select(.name==$n)` on a record that is not proxied yields an empty string rather than `null` pasted into a URL.

List every proxied record with its route:

```bash
uv run python api/registry_management.py custom-record-list --type rest-endpoint --json 2>/dev/null \
  | jq -r '.records[] | select(.is_proxied) | "\(.name)\t\(.proxy_client_url)/\t-> \(.proxy_target_url)"' \
  | column -t -s$'\t'
```

Diagnostics go to stderr, JSON to stdout, so `2>/dev/null` leaves clean JSON. Listings paginate: compare `total_count` against the array length and pass `?limit=` when they differ, or a proxied record hides on page two and looks unregistered.

## 3. Routing and ingress auth

Fixture: F2.

| ID | Command | Pass |
|---|---|---|
| **T-3.1** | `curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $GW" "${BASE_METEO}v1/forecast?latitude=38.9&longitude=-77.03&current=temperature_2m"` | `200` |
| **T-3.2** | same with no `Authorization` header | `401` |
| **T-3.3** | same with `-H "Authorization: Bearer not-a-jwt"` | `401` |
| **T-3.4** | same with the token in `X-Authorization` instead | `200` — either header carries the gateway JWT |
| **T-3.5** | `curl -s -o /dev/null -w '%{http_code} %{redirect_url}\n' -H "Authorization: Bearer $GW" "${BASE_METEO%/}"` | `301`, `Location` adds the trailing slash |

**T-3.6 — an unmatched gateway path must not look like success.**

```bash
curl -s -o /dev/null -w '%{http_code} %{content_type}\n' -H "Authorization: Bearer $GW" \
  "$ORIGIN/gateway/skill/some-skill-that-is-not-proxied/anything"
```

Today this returns `200 text/html` — the 889-byte frontend shell, because no proxy location matches and nginx falls through to the SPA. Record the result. A release that changes it to `404` is an improvement; a test that reads only the status code passes either way and tells you nothing, so assert on `content_type`.

## 4. Keyless upstream

Fixture: F2. Run this first when anything else fails. With no credential anywhere, it isolates the transport — nginx route, `/validate` markers, signed token, egress hop — from every credential concern.

**T-4.1**

```bash
curl -sS -H "Authorization: Bearer $GW" \
  "${BASE_METEO}v1/forecast?latitude=38.9072&longitude=-77.0369&current=temperature_2m,wind_speed_10m"
```

Pass: `200` and a body carrying `"current"` with a `temperature_2m` number.

**T-4.2 — the gateway strips its own credential before egress.** Send the JWT in `Authorization`, which the keyless upstream ignores. `_strip_generic_internal_headers` drops it and the whole gateway-internal header set on the way out. Pass: `200`, same as T-4.1.

## 5. Static operator credential

Fixture: F3. The operator stores a credential once; the registry encrypts it, vends it to the auth-server over an internal service-token-gated endpoint, and the hop injects it on egress. Callers never send it and never see it.

api-ninjas returns three distinct answers, so each failure mode has a signature. All three are HTTP 400, so **assert on the body, not the status code**.

| Upstream body | Meaning |
|---|---|
| `200` + weather JSON | the right key was injected |
| `{"error": "Missing API Key."}` | the hop forwarded with no credential — fail-open, a bug |
| `{"error": "Invalid API Key."}` | a key arrived and it was the wrong one |

The free tier needs `lat` and `lon`. `city` costs a premium plan and returns 400 with a valid key.

**T-5.1 — the create response hides the secret.**

```bash
curl -sS -H "Authorization: Bearer $GW" \
  "$ORIGIN/api/custom/rest-endpoint/$(basename ${BASE_NINJA%/})" | jq '{custom_header_names, custom_header_overridable_names, custom_headers_updated_at}'
```

Pass: `custom_header_names` is `["X-Api-Key"]`, `custom_header_overridable_names` is `[]`, no value or ciphertext field appears anywhere, and the plaintext key appears nowhere in the body.

**T-5.2 — injection works and the caller sends nothing.**

```bash
curl -sS -H "Authorization: Bearer $GW" "${BASE_NINJA}v1/weather?lat=39.2806&lon=-77.4136"
```

Pass: `200` with `"temp"` in the body. Fails on `Missing API Key.`

**T-5.3 — the credential is encrypted at rest.**

```bash
docker exec mcp-mongodb mongosh --quiet -u admin -p "$DOCUMENTDB_PASSWORD" \
  --authenticationDatabase admin --eval '
  var c = db.getSiblingDB("mcp_registry");
  var e = c.mcp_custom_entities_default.findOne({name: "api-ninjas-weather"}).custom_headers_encrypted[0];
  print(Object.keys(e).join(",") + " len=" + e.value_encrypted.length + " " + e.value_encrypted.substring(0,8));'
```

Pass: entry keys are `name,value_encrypted`; the value starts `gAAAAA` (Fernet); the plaintext key appears nowhere in the document; no plaintext `custom_headers` field exists.

**T-5.4 — a fixed header beats a caller header.**

```bash
curl -sS -H "Authorization: Bearer $GW" -H "X-Api-Key: caller-garbage-value" \
  "${BASE_NINJA}v1/weather?lat=39.2806&lon=-77.4136"
```

Pass: `200`. Fails on `Invalid API Key.`, which means the caller's value reached the backend.

**T-5.5 — a failed vend refuses the request.** The highest-value test here. Save the restore command before breaking anything: an interrupted run leaves the record holding an undecryptable credential.

```bash
REC=$(curl -sS -H "Authorization: Bearer $GW" $ORIGIN/api/custom/rest-endpoint \
  | jq -r '.records[] | select(.name=="api-ninjas-weather") | .path')
M="docker exec mcp-mongodb mongosh --quiet -u admin -p $DOCUMENTDB_PASSWORD --authenticationDatabase admin --eval"

ORIG=$($M "print(db.getSiblingDB('mcp_registry').mcp_custom_entities_default.findOne({_id:'$REC'}).custom_headers_encrypted[0].value_encrypted)")
echo "$ORIG" > /tmp/orig-ciphertext.txt

BAD="${ORIG:0:60}X${ORIG:61}"
$M "db.getSiblingDB('mcp_registry').mcp_custom_entities_default.updateOne({_id:'$REC'},{\$set:{'custom_headers_encrypted.0.value_encrypted':'$BAD'}})"

curl -sS -w '\nHTTP %{http_code}\n' -H "Authorization: Bearer $GW" \
  "${BASE_NINJA}v1/weather?lat=39.2806&lon=-77.4136"

$M "db.getSiblingDB('mcp_registry').mcp_custom_entities_default.updateOne({_id:'$REC'},{\$set:{'custom_headers_encrypted.0.value_encrypted':'$(cat /tmp/orig-ciphertext.txt)'}})"
```

Pass: `502 {"detail":"Upstream auth unavailable"}`. Fails on `Missing API Key.`, which means the hop forwarded the request stripped of its credential.

Both layers must log a refusal:

```
registry     egress_auth_routes.py  generic upstream-headers REFUSED: stored credential decryption failed
registry     POST /api/internal/generic-upstream-headers -> 500
auth-server  generic upstream-headers vend failed ...; refusing to forward unauthenticated
auth-server  GET /proxy/rest-endpoint/... -> 502
```

Re-run T-5.2 after the restore. Pass: `200`, and the registry logs `generic upstream-headers vended for ...: defaults=['X-Api-Key'] overridable=[]`, which also shows the vend runs per request with no cache.

## 6. Credential rotation

Fixture: F3. `PATCH /api/custom/{type}/{uuid}/upstream-headers` replaces the whole header set. An empty list clears it. Values are write-only.

```bash
UUID=$(basename ${BASE_NINJA%/})
ROT=$ORIGIN/api/custom/rest-endpoint/$UUID/upstream-headers
KEY=$(secret .scratchpad/pr-1714/api-ninja) || echo 'section 6 needs the api-ninjas key' >&2
rot() { curl -sS -X PATCH "$ROT" -H "Authorization: Bearer $GW" -H 'Content-Type: application/json' -d "$1" \
        | jq -c '{custom_header_names, custom_header_overridable_names, custom_headers_updated_at}'; }
```

**Wait for the nginx reload after every rotation.** Adding or removing the last header rewrites the route and flips the `$generic_has_upstream_auth` marker. Probing inside that window returns stale behaviour: a rotation that adds the first credential reads as `Missing API Key.` until the reload lands. Sleep 4 seconds, or poll until the answer stops changing.

| ID | Rotate to | Caller sends | Pass |
|---|---|---|---|
| **T-6.1** | `{"custom_headers":[{"name":"X-Api-Key","value":"'$KEY'","overridable":true}]}` | nothing | `200` — operator default applies |
| **T-6.2** | (same) | `X-Api-Key: garbage` | `Invalid API Key.` — the caller wins on an overridable header |
| **T-6.3** | (same) | `X-Api-Key: $KEY` | `200` |
| **T-6.4** | `{"custom_headers":[{"name":"X-Api-Key","overridable":false}]}` | nothing | `200` — a blank value keeps the stored ciphertext |
| **T-6.5** | `{"custom_headers":[]}` | nothing | `Missing API Key.` — cleared |
| **T-6.6** | (cleared) | `X-Api-Key: $KEY` | `Missing API Key.` — an unregistered name is dropped, so a caller cannot inject a header at the backend |
| **T-6.7** | `{"custom_headers":[{"name":"X-Api-Key","overridable":true}]}` | nothing | `Missing API Key.` — the slot is registered and empty |
| **T-6.8** | (same) | `X-Api-Key: $KEY` | `200` — the caller supplies the credential |
| **T-6.9** | `{"custom_headers":[{"name":"X-Api-Key","value":"'$KEY'"}]}` | nothing | `200` — back to fixed |

`custom_headers_updated_at` must advance on every rotation and the two name arrays must track the new shape.

T-6.4 is the one that protects the UI. `UpstreamHeadersField` renders registered names with empty value boxes, so a user who edits an unrelated field and saves must not wipe the credential. A blank value means keep.

The same rule has a consequence worth remembering: rotating with the value omitted cannot turn an operator default into a caller-only slot, because omission keeps the old value. Clear first, then re-add — T-6.5 followed by T-6.7.

**T-6.10 — the general update refuses headers.** `PUT` on the record with a `custom_headers` field must ignore it. Accepting it would skip create-path validation and write plaintext values straight to storage. Pass: the stored `custom_headers_encrypted` and `custom_headers_updated_at` are unchanged after the PUT.

## 7. Caller passthrough, streaming, and skills

### 7.1 Scripted clients

Fixtures F4 and F5. Four files under `tests/scripts/`, none of them pytest tests. They need a live stack and a real backend key.

| File | Role |
|---|---|
| `gateway_test_support.py` | credential split, entity discovery, per-verb scope check, status interpretation |
| `openai_gateway_client.py` | modes `models` (GET authz), `chat` (POST authz plus CSRF bearer exemption), `stream` (SSE), `all` |
| `bedrock_gateway_client.py` | modes `converse`, `stream` (AWS event-stream frames via `botocore.eventstream`), `all` |
| `verify_caller_creds.py` | proves a **caller-supplied** key is the credential reaching the backend, and that the gateway stores none |

```bash
uv run python tests/scripts/openai_gateway_client.py --mode all --ensure-scope
uv run python tests/scripts/bedrock_gateway_client.py --entity bedrock-proxy-use2 --mode all \
  --model us.anthropic.claude-opus-4-8 --prompt "Count slowly from 1 to 10."
```

Shared flags: `--registry-url`, `--token-file`, `--api-key-file`, `--entity`, `--client-path`, `--ensure-scope`, `--scope-group`, `--timeout`, `--debug`.

Pass: every mode reports OK. Both stream modes print time-to-first-chunk, chunk count, and the largest inter-chunk gap, and they fail when a response arrives buffered — one chunk, or every chunk at the same instant. A 200 alone does not pass a streaming test.

Neither client covers section 5 or 6. Both send the credential themselves, which is the opposite of what an operator-injected header does.

**T-7.1a — whose credential is actually in use.** A 200 on a caller-passthrough entity does not tell you: a stored operator default produces the same 200. Two calls are needed, and `verify_caller_creds.py` makes both:

```bash
uv run python tests/scripts/verify_caller_creds.py --entity openai-proxy
```

| Caller sends | Pass |
|---|---|
| the key | `200` with a model list |
| nothing | `401` from the backend — nothing stored, so nothing to inject |

A `200` on the second call is the failure: an operator default is being injected. Demote it to a caller-only slot with two rotations, because omitting a value **keeps** the stored one:

```bash
curl -sS -X PATCH "$ROT" -H "Authorization: Bearer $GW" -H 'Content-Type: application/json' -d '{"custom_headers": []}'
curl -sS -X PATCH "$ROT" -H "Authorization: Bearer $GW" -H 'Content-Type: application/json' \
  -d '{"custom_headers": [{"name": "Authorization", "overridable": true}]}'
```

The script also refuses an expired token up front and names the cause of a 401, 403, 404, or unknown entity, rather than raising. A gateway JWT is only valid for the deployment whose auth-server signed it: `iss` and `aud` are identical across deployments while each signs with its own `SECRET_KEY`, so pointing a valid-looking token at the wrong registry yields a 401 that looks like a scope problem.

### 7.2 The credential split

The gateway JWT goes in `X-Authorization`. The backend key goes in `Authorization`, and the hop forwards it only because the entity registers `Authorization` as caller-overridable. A fixed `Authorization` is refused at registration: an operator bearer belongs in the per-user egress vault.

**T-7.1 — the equal-token guard.** Send the same gateway JWT in both `Authorization` and `X-Authorization`. Pass: `401` from the gateway. This stops the gateway's own credential from reaching a backend.

### 7.3 Skills

Fixture: F6. A skill differs from a custom record in three ways.

The client URL is name-based. `build_proxy_client_path` strips the leading namespace segment and prefixes the entity-type token, which is singular:

| Identifier | Value |
|---|---|
| Registry path (Mongo `_id`) | `/skills/pdf` |
| Client URL | `/gateway/skill/pdf` |
| Authz key | `skill/skills/pdf` |

**T-7.2 — the hop forwards bytes unchanged.**

```bash
curl -sS -H "Authorization: Bearer $GW" \
  "$ORIGIN/gateway/skill/pdf/anthropics/skills/refs/heads/main/skills/pdf/SKILL.md" > /tmp/gw.md
curl -sS https://raw.githubusercontent.com/anthropics/skills/refs/heads/main/skills/pdf/SKILL.md > /tmp/direct.md
cmp /tmp/gw.md /tmp/direct.md && echo IDENTICAL
```

Pass: `cmp` reports identical, and the response carries the upstream `etag` and `cache-control`.

**T-7.3 — the plural path is not a 404.** Request `/gateway/skills/pdf/...`. Today it returns `200 text/html`, the SPA shell. Same class of trap as T-3.6.

**T-7.4 — a bare origin exposes the whole host.** Fetch an unrelated path through the same route:

```bash
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $GW" \
  "$ORIGIN/gateway/skill/pdf/anthropics/courses/refs/heads/master/prompt_engineering_interactive_tutorial/README.md"
```

Returns `200`. The sub-path stays inside the route prefix and the SSRF pin locks the host, so this is the documented behaviour. A target that names one file confines the route to that file, which is tighter. Record which shape each fixture uses.

**T-7.5 — a PUT-flipped skill stores no `proxy_client_url`.** The field is server-derived, so it never appears in a request body and `exclude_unset` never writes it. Reads recompute it through `populate_proxy_client_url`, and nginx generation recomputes it too, so the API, the UI, and routing all agree. Pass: the API returns the client URL even though Mongo has no such field. A raw projection that skips the model sees `null`.

### 7.4 Routing metrics and the body-capture header (issue #1735)

Every command below was run against a live stack on 2026-09-08 and the stated values were observed, not predicted. `AUTH` is the auth-server container; `:9464` is not published, so metrics are read with an exec.

**On ECS, substitute an AMP query for every `counters` call.** The task runs under `opentelemetry-instrument`, so the SDK provider is already installed, `start_http_server` never runs, and `curl localhost:9464/metrics` inside the task returns `Connection refused` — metrics leave through the `adot-collector` sidecar over OTLP instead. Use Grafana's AMP datasource, or a SigV4-signed `POST /api/v1/query` (see [OBSERVABILITY.md](../../docs/OBSERVABILITY.md#verifying-gateway-proxy-metrics-end-to-end)). The whole of 7.4 was re-run that way against the ECS deployment on 2026-09-08 and passed: 8 `clear_header` directives in the rendered config (4 entities × 2 headers, one server block), forged `X-Body` minting zero series, and the 64-character authz key `rest-endpoint/rest-endpoint/1a546ca6-…` intact.

```bash
export AUTH=mcp-gateway-registry-auth-server-1
counters() { docker exec $AUTH sh -c 'curl -s localhost:9464/metrics' \
  | grep -E '^(mcpgw_registry_)?(auth_request_total|tool_execution_total|mcpgw_registry_generic_proxy_)' \
  | sed 's/otel_scope[^,]*,//g' | sort; }
```

**T-7.6 — zero series exist before any traffic.** On a freshly started stack:

```bash
docker compose logs auth-server | grep zero-init     # zero-init seeded 47/47 generic-proxy series
docker exec $AUTH sh -c 'curl -s localhost:9464/metrics' \
  | grep -E '^mcpgw_registry_generic_proxy_(slot_rejected|stream_outcome)_total' | grep -c ' 0.0$'   # 8
docker exec $AUTH sh -c 'curl -s localhost:9464/metrics' \
  | grep -c '^mcpgw_registry_generic_proxy_request_total.* 0.0$'                                     # 39
```

Pass: the log line reports `47/47`, the two lifecycle counters expose 8 series at `0.0`, and the hop outcome counter exposes all **39** (3 entity types × 13 outcomes). A `zero-init skipped: meter provider is ...` line means the SDK meter provider was never installed, so nothing below will show anything.

**T-7.7 — a proxied request is labeled by entity type and authz key.** Drive one buffered custom record and one skill, then:

```bash
counters | grep generic_proxy
```

Pass: `auth_request_total{server="rest-endpoint/rest-endpoint/<uuid>",target_kind="generic_proxy_custom",success="true"}` and `{server="skill/skills/pdf",target_kind="generic_proxy_skill",success="true"}`. The `server` value is the **authz key**, i.e. the exact string a `server_access` rule names — `skill/skills/pdf`, not the client path `skill/pdf`.

**T-7.8 — no gateway request lands in `unknown`.**

```bash
counters | grep 'target_kind="unknown"'
```

Pass: the only values present are non-gateway (MCP servers whose paths match no transport rule). The count must not grow when the gateway calls above are repeated.

**T-7.9 — the latency histogram carries no `server` label.**

```bash
docker exec $AUTH sh -c 'curl -s localhost:9464/metrics' \
  | grep '^mcpgw_registry_auth_request_duration' | grep -c 'server='      # 0
```

Pass: `0`. A per-target label on a 16-bucket histogram costs 18 series where the counter costs 1, and nothing queries latency per endpoint. Group by `target_kind` instead.

**T-7.10 — a client cannot author the body-capture headers on a gateway route.** This is the regression guard for the defect fixed in `fb579306`: `capture_body.lua` does not run on generic locations, and the shared `location = /validate` forwards client headers verbatim, so before the fix a caller could hand the metrics middleware a JSON-RPC body of their choosing — **including with an invalid token**, because emission happens in a `finally`.

```bash
for i in 1 2 3; do
  curl -sS -o /dev/null -w '%{http_code} ' -H "X-Authorization: Bearer $GW" \
    -H "X-Body: {\"method\":\"tools/call\",\"params\":{\"name\":\"forged_$i\"}}" \
    -H 'X-Body-Uninspectable: 1' "${BASE}v1/forecast?latitude=38.9&longitude=-77.03&current=temperature_2m"
done; echo
docker exec $AUTH sh -c 'curl -s localhost:9464/metrics' | grep -c 'forged_'                          # 0
docker exec $AUTH sh -c 'curl -s localhost:9464/metrics' | grep -E 'tool_execution|protocol_latency' \
  | grep -c 'rest-endpoint/'                                                                          # 0
```

Pass: the requests still return their normal status (`200` with a valid token, `401` without — the fix does not change the response), **and** no `forged_*` label and no `tool_execution` / `protocol_latency` series naming a gateway entity appear. Run it with a valid token too: a `403`/`401` alone does not exercise the hop.

Also confirm the rendered config still carries the clears (2 per generated gateway location):

```bash
docker exec mcp-gateway-registry-registry-1 sh -c \
  "grep -c 'clear_header' /etc/nginx/conf.d/nginx_rev_proxy.conf"
```

**T-7.11 — streaming still streams, and the terminals balance.** Drive one SSE request against a `proxy_streaming=true` entity, then:

```bash
counters | grep stream_outcome | grep -v ' 0.0$'
```

Pass: SSE events arrive incrementally (observed: 29 events spread over 257 ms, not one blob), and `started` equals the sum of the terminals (`completed` + `client_closed` + `duration_timeout` + `byte_cap` + `upstream_error`) once nothing is in flight. A truncated client pipe correctly counts as `client_closed`.

**T-7.12 — MCP traffic keeps per-server labels.** `server_name` on `tool_execution` and `protocol_latency` is now cardinality-bounded (150 distinct values, 96 characters), so verify real MCP names still pass through untouched:

```bash
uv run python tests/scripts/call_mcp_tool.py --server-url $ORIGIN/<server>/mcp \
  --tool <tool> --tool-args '{}' --token-file .token --registry-url $ORIGIN
counters | grep tool_execution_total
docker exec $AUTH sh -c 'curl -s localhost:9464/metrics' | grep -cE '"_other"|"_unset"'    # 0
```

Pass: the real server name appears verbatim (`server_name="com-github-github-mcp-server"`), the invoked tool appears as `tool_name`, and no `_other`/`_unset` sentinel exists. **Note the cap:** a deployment whose traffic spreads across more than 150 distinct servers will collapse the long tail into `_other` — this stack has 144 registered MCP servers plus 8 proxied gateway entities, so it is close enough to the cap that `METRICS_MAX_LABEL_CARDINALITY` may need raising (see [unified-parameter-reference.md](../../docs/unified-parameter-reference.md), Group 25).

**T-7.13 — the hop's own outcome is recorded, once per request (issue #1735 item 4).** `auth_request_total{success}` is the `/validate` decision, so a request that authorizes and then fails at the hop reads there as a success. `generic_proxy_request_total{entity_type,outcome}` is what the caller got.

Drive one request per reachable outcome, then read the counter:

```bash
hop() { docker exec $AUTH sh -c 'curl -s localhost:9464/metrics' \
  | grep '^mcpgw_registry_generic_proxy_request_total' | grep -v ' 0.0$' \
  | sed 's/otel_scope[^,]*,//g;s/mcpgw_registry_generic_proxy_request_total//' | sort; }

curl -sS --compressed -o /dev/null -H "X-Authorization: Bearer $GW" "${BASE}v1/forecast?latitude=38.9&longitude=-77.03&current=temperature_2m"   # ok
curl -sS --compressed -o /dev/null -H "X-Authorization: Bearer $GW" "${BASE}v1/forecast?latitude=not-a-number"                                    # upstream_4xx
curl -sS --compressed -o /dev/null -H "X-Authorization: Bearer $GW" "$ORIGIN/gateway/skill/pdf/definitely/not/a/real/file.md"                     # upstream_4xx (skill)
timeout 60 curl -sS --compressed -N -X POST -H "X-Authorization: Bearer $GW" -H "Authorization: Bearer $OAI" \
  -H 'Content-Type: application/json' -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}],"stream":true,"max_tokens":10}' \
  -o /dev/null "$OPENAI_BASE/v1/chat/completions"                                                                                                 # ok (stream)
timeout 60 curl -sS --compressed -N -X POST -H "X-Authorization: Bearer $GW" -H "Authorization: Bearer $OAI" \
  -H 'Content-Type: application/json' -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"count to 40"}],"stream":true,"max_tokens":300}' \
  "$OPENAI_BASE/v1/chat/completions" | head -c 200 > /dev/null                                                                                    # client_closed
hop
```

Observed on 2026-09-08 for exactly those six requests:

```
{entity_type="custom",outcome="client_closed"} 1.0
{entity_type="custom",outcome="ok"} 2.0
{entity_type="custom",outcome="upstream_4xx"} 2.0
{entity_type="skill",outcome="ok"} 1.0
{entity_type="skill",outcome="upstream_4xx"} 1.0
```

Pass criteria, each of which has caught a real bug in review:

- **The totals equal the number of requests that entered the hop.** Six requests, six records. More means double counting (the streaming path recording alongside the wrapper); fewer means an exit path records nothing.
- **A streaming request contributes exactly one record, not one per chunk**, and it reconciles with the lifecycle counter: `stream_outcome{started}` equals the sum of its terminals, and each stream shows up once here.
- **`entity_type` stays within `skill` / `a2a_agent` / `custom`** whatever the operator named the type.
- **A 401 from the token gate records nothing.** Send a bogus token and confirm the counter does not move: the gate is a route dependency that returns before the handler, which is a documented non-goal rather than an oversight.
- **Failures that share a status stay distinct.** `disabled` vs `capacity` (both 503) and `auth_unavailable` vs `egress_blocked` vs `upstream_error` (all 502) are unit-tested per value in `tests/auth_server/unit/test_generic_proxy_outcome.py`; force them here only if you can (a corrupted stored credential gives `auth_unavailable`, `GATEWAY_GENERIC_PROXY_ENABLED=false` gives `disabled`).

## 8. Teardown

```bash
for n in api-ninjas-weather openmeteo-forecast openai-proxy bedrock-proxy-use2; do
  U=$(curl -sS -H "Authorization: Bearer $GW" $ORIGIN/api/custom/rest-endpoint \
      | jq -r --arg n "$n" '.records[] | select(.name==$n) | .path | split("/")[2]')
  [ -n "$U" ] && curl -sS -X DELETE -H "Authorization: Bearer $GW" \
    -o /dev/null -w "$n -> %{http_code}\n" "$ORIGIN/api/custom/rest-endpoint/$U"
done

curl -sS -X PUT $ORIGIN/api/skills/pdf \
  -H "Authorization: Bearer $GW" -H 'Content-Type: application/json' \
  -d '{"name":"pdf","description":"Backwards compat update test",
       "skill_md_url":"https://raw.githubusercontent.com/anthropics/courses/refs/heads/master/prompt_engineering_interactive_tutorial/README.md",
       "is_proxied":false}' -o /dev/null -w 'pdf unproxied -> %{http_code}\n'
```

Then confirm no route survives:

```bash
docker exec mcp-gateway-registry-registry-1 grep -c 'gateway/rest-endpoint\|gateway/skill' \
  /etc/nginx/conf.d/nginx_rev_proxy.conf
```

## 9. Failure reference

| Symptom | Cause |
|---|---|
| `301` | The trailing slash is missing. The location ends in one. `curl -L` follows it, and a POST becomes a GET when it does. |
| `200 text/html`, 889 bytes | No proxy location matched, so nginx served the SPA. Wrong entity-type spelling (`skills` for `skill`), or the entity is not proxied. Check `content_type`, not the status. |
| `401` from the gateway | Expired or missing token, or the equal-token guard fired because `Authorization` matched `X-Authorization`. |
| `403` from the gateway | The group holds no rule granting this verb on this authz key. `methods: ["all"]` does not count. |
| `502 {"detail":"Upstream auth unavailable"}` | The vend failed and the hop refused. Causes: `SECRET_KEY` changed since the credential was stored, corrupt ciphertext, the registry `:8091` vend listener unreachable, or a target-identity mismatch. Grep the registry log for `upstream-headers REFUSED`; it names the branch. |
| `503` | The streaming slot pool is full (`GATEWAY_GENERIC_STREAM_MAX_CONCURRENCY`). |
| `Missing API Key.` right after a rotation | The nginx reload has not landed. Wait and retry. |
| Backend 401 or 403 | The key is stale, or wrong for the region. The passthrough itself worked. |

## 10. Coverage gaps

No test in this suite covers these yet.

- Five of the six vend refusal branches: body and token mismatch, invalid target identity, token target against canonical URL, credential headers on a non-HTTPS target, duplicate stored names. Section 5 covers decrypt failure only.
- Streaming limits: byte cap to 413, absolute duration ceiling, slot exhaustion to 503, idle read timeout, client disconnect. The scripted clients prove incremental delivery, nothing more.
- The `Authorization` carve-out on a stored operator default. Registration accepts the name only as caller-overridable, and a mismatch between what registration accepts and what the strict decrypt allows produces an entity that 502s on every request.
- Repoint clearing. Patch `proxy_target_url` to another host and confirm the stored headers are wiped so the old host's secret cannot reach the new one.
- The scope boundary on a credentialed entity. Call F3 with a non-admin token. One operator credential serves every authorized caller, so the scope check is the only thing standing between a low-privilege caller and spending it.
- Federation stripping of `proxy_connect_notes` and the header fields, both directions.
- Non-admin redaction of `proxy_target_url` while `is_proxied` stays visible.
- Agents. `a2a_agent` is a proxyable type and its target falls back to the agent's own `url`, so flipping `is_proxied` needs no explicit target. Patching `url` repoints the backend and must re-validate, re-pin, and clear stored headers.
- `/api/config` exposure of the five streaming settings.
- All six `stream_outcome_total{outcome}` values, so in-flight equals started minus the terminals. Section 7.4 (T-7.11) covers `started`, `completed`, and `client_closed`; `duration_timeout`, `byte_cap`, and `upstream_error` still need forcing.
- Metrics under the legacy dual-write (`METRICS_LEGACY_HTTP_POST=true`): the middleware POSTs the raw authz key to metrics-service, whose limiter does **not** bound `server_name`. Section 7.4 covers the native OTel path only.

## 11. Result log

| Date | Release | Sections run | Result | Notes |
|---|---|---|---|---|
| | | | | |
