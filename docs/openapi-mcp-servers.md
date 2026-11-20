# OpenAPI MCP servers

An OpenAPI MCP server turns REST APIs into MCP tools without anyone writing an MCP server. An admin creates the server in the registry and adds REST APIs to it, each one given as an OpenAPI 3.x spec URL, a base URL and a credential: either one key the admin enters for every caller, or a key each user brings for themselves. Every operation in the spec becomes a tool, and agents connect to the server like any other MCP server, through the gateway, with the same login and access rules.

The design, with diagrams, is in [design/openapi-mcp-servers-design.md](design/openapi-mcp-servers-design.md). Every setting is listed in [unified-parameter-reference.md](unified-parameter-reference.md) under Group 13d.

## Turn it on

The feature needs the gateway (`DEPLOYMENT_MODE=with-gateway`) and a `REGISTRY_MODE` that includes MCP servers. In any other mode the registry turns it off at startup and logs why.

Docker Compose: set these in `.env`, then run `./build_and_run.sh`, which also starts the `openapi-mcp` container (compose profile `openapi-mcp`).

```bash
OPENAPI_MCP_ENABLED=true
OPENAPI_MCP_SYNC_SECRET=$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')
```

The sync secret signs the registry's calls to the container. Both read the same variable, it must be at least 32 characters, and it must differ from `SECRET_KEY`. The registry and the container both refuse to start without a strong one.

Terraform: set `openapi_mcp_enabled = true` and `openapi_mcp_image_uri` to a pinned image. Terraform generates the sync secret in AWS Secrets Manager.

Helm: set `openapiMcp.enabled: true` (deploys the container), `registry.app.openapiMcpEnabled: true` (turns the registry side on) and `auth-server.app.openapiMcpEnabled: true` (lets the auth server forward calls that use per-user credentials). The stack chart refuses to render when the three differ.

## Create a server and add an API

With the CLI (`api/registry_management.py`):

```bash
uv run python api/registry_management.py openapi-server-create --name "Weather APIs"

export PETSTORE_KEY=YOUR_PETSTORE_KEY
uv run python api/registry_management.py openapi-target-add --slug weather-apis --name petstore \
    --spec-url https://petstore3.swagger.io/api/v3/openapi.json \
    --base-url https://petstore3.swagger.io/api/v3 \
    --auth-scheme api_key_header --header-name api_key --credential-env PETSTORE_KEY \
    --include-tag pet
```

To send the spec itself instead of a URL, use `--spec-file` in place of `--spec-url`:

```bash
uv run python api/registry_management.py openapi-target-add --slug space-apis --name nasa \
    --spec-file docs/examples/openapi/nasa.json --base-url https://api.nasa.gov \
    --auth-scheme api_key_header --header-name X-Api-Key --credential-mode per_user
```

The CLI reads a credential from an environment variable (`--credential-env`) or from stdin (`--credential-stdin`), never from the command line. The same operations are under `/api/openapi-servers` in the REST API, and in the registry UI under the OpenAPI view, which only admins see.

The server's name becomes its path, so "Weather APIs" is served at `/weather-apis/mcp` on the gateway. Tool names are the target name, an underscore, and the operation's `operationId`, for example `petstore_getPetById`.

## What gets checked before a target is stored

Adding a target stores nothing until every check passes:

- The SSRF guard checks the base URL, and the spec URL when there is one, the same way it checks any registered backend. To reach an internal API, an operator adds its host to `SSRF_ALLOWED_HOSTS` or `SSRF_ALLOWED_CIDRS`.
- A target with a credential needs an `https://` base URL. Plain HTTP is accepted only for a host listed in `SSRF_ALLOWED_HOSTS`, or an IP address inside `SSRF_ALLOWED_CIDRS`, such as an API in your own network. The container applies the same rule again before each call.
- The registry fetches the spec without following redirects, so give the final URL. It asks for an uncompressed response and refuses a compressed one; upload the file instead if the host always compresses. It refuses a spec larger than `OPENAPI_MCP_SPEC_MAX_BYTES` (5 MiB by default), fetched or uploaded.
- The spec must be OpenAPI 3.x in JSON or YAML, valid UTF-8, with no YAML anchors or aliases.
- No spec path may contain a `.` or `..` segment or a backslash, raw or percent-encoded, because such a path could send the credential outside the base URL.
- Every `$ref` must be a local `#/...` pointer. A spec split across files, or one that refers to a URL or a local file, is refused; bundle it into one document first (for example with `redocly bundle`). Convert a Swagger 2.0 spec first, for example with [converter.swagger.io](https://converter.swagger.io).
- One target may produce at most `OPENAPI_MCP_MAX_TOOLS_PER_TARGET` tools (100 by default). A larger spec gets a `too_many_tools` error that lists the spec's tags with their operation counts. Choose operations with `include_tags` and `include_operations`; an `include_operations` entry is an `operationId` or `"METHOD /path"`.
- Tool names may be at most 64 characters. Rename long or clashing ones with `tool_name_overrides`, keyed by `operationId`, or by `"METHOD /path"` for an operation that has none.

## Upload a spec instead of a URL

A target gets its spec in one of two ways. With `spec_url`, the registry fetches the spec. With `spec_content`, the admin sends the spec's JSON or YAML text in the request body, which suits a spec that is not published anywhere the registry can reach. The UI offers both on the target form ("Fetch from a URL" or "Upload a file"), and the CLI takes `--spec-url` or `--spec-file`.

An uploaded spec goes through every check in the list above except the fetch. The registry stores it the same way as a fetched one, and the response shows `"spec_source": "upload"` with no `spec_url`.

To change a target's spec later, send one of these in a `PATCH` on the target:

- `spec_content` replaces the spec with new text.
- `spec_url` switches the target to fetching from that URL.
- `refresh_spec: true` fetches the spec again from the stored URL. A target with an uploaded spec has no URL, so this returns `409 spec_url_missing`; send `spec_content` instead.

The gateway accepts request bodies up to 8 MiB on `/api/openapi-servers/`; the rest of `/api/` keeps nginx's 1 MiB default. A spec sent in `spec_content` is larger than the file, because JSON escaping turns every quote and line break into two characters, so a deployment that raises `OPENAPI_MCP_SPEC_MAX_BYTES` above about 6 MiB should give the spec by URL instead.

## Shared or per-user credentials

Each target chooses one `credential_mode`:

- `shared` (the default): the admin enters one credential, which every caller uses. It can be an API key in a header or a query parameter, a bearer token, or HTTP basic auth. The registry stores it encrypted with `SECRET_KEY` and never returns it; responses show only `credential_set`. To rotate it, send a new `auth` in a `PATCH` on the target.
- `per_user`: each user brings their own API key or personal access token, which they get from the API provider themselves. The registry never obtains keys for anyone. The admin sets only the shape, `bearer` or `api_key_header` with its header name. Per-user targets need `EGRESS_AUTH_ENABLED=true` and a vault (`SECRET_STORE_BACKEND`), because the keys are stored there, like egress PATs.

One server can mix both kinds of target. A per-user target, as the admin adds it:

```json
{"name": "nasa", "spec_url": "https://raw.githubusercontent.com/agentic-community/mcp-gateway-registry/main/docs/examples/openapi/nasa.json", "base_url": "https://api.nasa.gov",
 "auth": {"scheme": "api_key_header", "header_name": "X-Api-Key", "credential_mode": "per_user"}}
```

Each user then adds their key, in the UI ("Your API keys" on the server card, or Connected Accounts) or through the API:

```bash
curl -X PUT "$REGISTRY_URL/api/openapi-servers/nasa/targets/nasa/my-credential" \
  -H "Authorization: Bearer $ACCESS_TOKEN" -H "Content-Type: application/json" \
  -d '{"secret": "YOUR_NASA_KEY", "ttl_value": 30, "ttl_unit": "days"}'
```

With the CLI, the key comes from an environment variable or stdin, never the command line:

```bash
export NASA_KEY=YOUR_NASA_KEY
uv run python api/registry_management.py openapi-key-set --slug nasa --name nasa --key-env NASA_KEY
uv run python api/registry_management.py openapi-key-status --slug nasa
uv run python api/registry_management.py openapi-key-delete --slug nasa --name nasa
```

- A key is write-only. `GET .../my-credential` and `GET /api/openapi-servers/{slug}/my-credentials` return only whether a key is set and when it expires. `DELETE .../my-credential` removes it.
- A key lasts at most 30 days, and the user adds a new one after that.
- A key is bound to the target's base URL and to this server. If an admin points the target somewhere else, or deletes the server and creates it again under the same name, stored keys stop working, and users add them again.
- Deleting a target or a server does not delete users' stored keys, because the vault lists keys one user at a time. Those keys can no longer be used, they expire within their 30-day lifetime, and each user can remove theirs from Connected Accounts.
- An admin can store a key on a user's behalf by adding `sub` and that user's `auth_method` to the request, as with egress PATs.
- A call to a per-user tool without a key returns a tool error asking the user to add one. It never falls back to a shared key.

At call time the gateway sends a per-user server's traffic through the auth server, which forwards the caller's keys to the container in a signed, single-use envelope. The auth server never holds the key that signs it.

The registry keeps the exact spec it checked. A change to the spec at its URL reaches agents only after an admin refreshes the target (`refresh_spec: true`) or uploads the new version, and either one runs every check again.

## Try it with the NASA example

[`docs/examples/openapi/nasa.json`](examples/openapi/nasa.json) is a small OpenAPI 3 spec for three NASA APIs: the Astronomy Picture of the Day, Earth images from the DSCOVR EPIC camera, and near-Earth asteroids. It works on any deployment (Compose, ECS or EKS), because the registry can fetch it from GitHub or you can upload the file. Each user needs a free key of their own from [api.nasa.gov](https://api.nasa.gov).

1. Create a server named `NASA`, in the UI or with the CLI. It is served at `/nasa/mcp` on the gateway.

   ```bash
   uv run python api/registry_management.py openapi-server-create --name "NASA" \
       --description "NASA open data; each user adds their own api.nasa.gov key"
   ```

2. Add the target. In the UI, choose "Fetch from a URL" and paste the spec URL below, or choose "Upload a file" and pick `docs/examples/openapi/nasa.json`. Set the base URL to `https://api.nasa.gov`, the credential mode to per user, the auth scheme to an API key in a header, and the header name to `X-Api-Key`.

   ```bash
   uv run python api/registry_management.py openapi-target-add --slug nasa --name nasa \
       --spec-url https://raw.githubusercontent.com/agentic-community/mcp-gateway-registry/main/docs/examples/openapi/nasa.json \
       --base-url https://api.nasa.gov \
       --auth-scheme api_key_header --header-name X-Api-Key --credential-mode per_user
   ```

   Use `--spec-file docs/examples/openapi/nasa.json` instead of `--spec-url` to upload the file.

3. Decide who can use it. The server starts `private`; set it to `group-restricted` with the groups that should see it.

4. Each user adds their key, in the UI ("Your API keys" on the server card) or with `openapi-key-set` (see above).

5. Connect an MCP client to `<gateway URL>/nasa/mcp` and call `nasa_getApod`. A user who has not added a key gets a tool error that tells them to add one.

The registry must reach `raw.githubusercontent.com` to fetch the spec (upload the file if it cannot), and the openapi-mcp container must reach `api.nasa.gov` to call the API.

## Who can use a server

A new server is `private`: the registry lists it only to admins. Visibility controls who sees the server in the registry; the gateway decides who may call it from scopes alone, as for every other server, so a user whose scopes grant the server (for example through a wildcard) can call it even while it is private. Check your scope grants before you add a shared key. This differs from `POST /api/servers/register`, where the default is `public`. Every caller spends the targets' shared credentials, so open a server with shared targets to a group (`group-restricted`) or to everyone (`public`) only when it is ready. A server whose targets are all per-user spends no one's key but the caller's. The server is an ordinary registry server, so scopes, the gateway's access check, search and the per-tool block all apply to it unchanged.

The OpenAPI MCP server owns that registry record. The generic `/api/servers/*` routes that would change or remove it (edit, replace through register, remove, the credential, OAuth and version routes) return `409` and point to `/api/openapi-servers/{slug}`. Toggle, refresh, rescan, rating and the per-tool block still work through the generic routes, so an admin can turn a server back on after an interrupted scan. No other registration may point at the openapi-mcp container.

When a caller who is not an admin reads a server through `/api/openapi-servers`, the response lists only the tools their scopes grant. It leaves out each target's overrides and include filters, because those name operations too, and it replaces a target's load error with a generic message.

Each tool is marked from its HTTP method: `GET`, `HEAD` and `OPTIONS` are read-only, and `DELETE` is destructive. Clients that confirm destructive calls, such as Claude Code, ask before a `DELETE`. To remove a tool entirely, block it with the existing per-tool block (`tool_overrides`) on the server, or leave its operation out with the include filters.

## What callers should expect

- A JSON array response arrives wrapped as `{"result": [...]}`, because MCP structured output has to be an object.
- Tools refuse arguments the spec does not define, so a misspelled argument returns an error.
- If a backend echoes the target's credential in its response, as api.nasa.gov does in its `links` URLs, the container replaces it with `[REDACTED]` before the response reaches the agent. This covers the raw key, its URL-encoded and JSON-escaped forms and a basic-auth header value, for shared and per-user targets alike. A link copied from such a response therefore no longer works on its own; the agent calls the tool again instead.
- The container sends no cookies and drops every `Set-Cookie` a backend returns, so one caller's backend session never reaches another caller. An API that works only with cookie sessions does not work through an OpenAPI MCP server.
- The container asks for uncompressed responses and turns a compressed one into a tool error, because decompressing it could use far more memory than the response cap allows.
- Responses larger than `OPENAPI_MCP_MAX_RESPONSE_BYTES` (2 MiB by default) become a tool error, as do backend calls slower than `OPENAPI_MCP_BACKEND_TIMEOUT_SECONDS` (30 seconds).
- By default tools carry no output schema, because many real APIs return responses that break their own spec. Set `keep_output_schemas` on a target to keep the spec's schemas.

## Operations

The registry database is the only stored state. The container keeps each server in memory and the registry pushes it back whenever needed:

- After a container restart, calls to OpenAPI MCP servers return 404 until the registry notices the restart and pushes every server again. With 10 servers of 10 targets each, that took 23 seconds.
- After a registry restart, the registry compares what it stores with what the container runs and pushes anything stale.
- A change that cannot reach the container in time returns `sync_status: "pending"`. The registry retries every `OPENAPI_MCP_RECONCILE_INTERVAL_SECONDS` (60 by default), and it also finishes refreshing the server's tool list, which search and the UI read, even when the push itself landed late.
- If the container cannot load a target, the add or update still returns, and its `warnings` name the target and the reason. That target serves no tools until it is fixed.
- A server deleted while the container was unreachable and then created again under the same name replaces the old hub on its first push. Each push carries the server's creation time, so the container does not mistake the new server for the old one.
- Two requests that change the same target at once do not overwrite each other: the later one returns `409 target_changed`, and the client reloads the target and tries again.
- Every change starts a background security scan of the server, which takes a few seconds. If the container restarts or the server is deleted during that scan, the scan fails, and the registry's scan policy tags the server `security-pending` and disables it. The hub comes back, but the server stays disabled until an admin turns it on again with `POST /api/servers/toggle`.
- A change replaces a server's hub without cutting off calls already running on the old one: the old hub keeps serving for 30 seconds before it closes.

The registry's calls to the container carry each target's credential in plain HTTP on the internal network, signed with the sync token. Only the registry (with its nginx) and the auth server can reach the container: in Compose it sits on its own `openapi-mcp` network with no published port, on ECS its security group admits only those two services, and on Helm a NetworkPolicy does the same. A Helm NetworkPolicy has no effect on a cluster that does not enforce network policies, so check that yours does. A cluster that needs TLS on internal hops can set `OPENAPI_MCP_INTERNAL_URL` to an `https` URL.

To look at the container directly in Compose, run a command inside it: `docker compose exec openapi-mcp curl -s http://127.0.0.1:9876/health`. A REST API that you run as another Compose service has to join the `openapi-mcp` network, or the container cannot call it.

The container runs one replica, because the registry pushes state to one address. With 10 servers of 10 targets each, including one 4.4 MB spec, it peaked at 293 MiB. Turning the feature off leaves stored servers in place without a backend; the registry logs their names at startup. While it is off, those servers drop out of the server list, search, health checks and the gateway's routes, so nobody sees or connects to a server that cannot answer. Nothing stored changes: turn the feature back on and every server returns, with its users' keys. To remove them for good, delete them with the feature on.

The container logs in the same JSON lines format as the registry, the auth server and mcpgw, with `"service": "openapi-mcp"`, and writes `openapi-mcp.log` under `APP_LOG_DIR` (`/var/log/containers/ai-registry` by default). It reads the same `APP_LOG_*` settings; see [logging-standard.md](logging-standard.md).

Metrics are under `mcpgw_openapi_mcp_*` on the registry's Prometheus endpoint. Alert when `mcpgw_openapi_mcp_state{kind="container_reachable"}` stays at 0 for five minutes. The gauge reports only while the feature is on, so the alert cannot fire on a deployment that never enabled it.

## End-to-end test

`tests/integration/test_openapi_mcp_e2e.py` runs against a local Compose stack with the feature on, and is skipped otherwise. It uses a mock REST API (`tests/integration/openapi_mock_api.py`) that serves its own spec, checks each auth scheme, and echoes the headers it receives. The mock also serves the example NASA spec, [`docs/examples/openapi/nasa.json`](examples/openapi/nasa.json), at `/nasa-spec.json`; the same file is the example for registering a per-user target against `https://api.nasa.gov`.

```bash
docker run -d --name openapi-mock-api --network mcp-gateway-registry_openapi-mcp --no-healthcheck \
    -v "$PWD/tests/integration/openapi_mock_api.py:/mock/tests/integration/openapi_mock_api.py:ro" \
    -v "$PWD/docs/examples/openapi:/mock/docs/examples/openapi:ro" \
    --entrypoint /app/.venv/bin/python mcp-gateway-registry-registry \
    -m uvicorn --app-dir /mock/tests/integration openapi_mock_api:app --host 0.0.0.0 --port 8000
# add openapi-mock-api to SSRF_ALLOWED_HOSTS in .env, then recreate registry and openapi-mcp
OPENAPI_MCP_E2E=1 uv run pytest tests/integration/test_openapi_mcp_e2e.py -v --no-cov
```
