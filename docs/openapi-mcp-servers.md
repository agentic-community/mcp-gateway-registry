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

The CLI reads a credential from an environment variable (`--credential-env`) or from stdin (`--credential-stdin`), never from the command line. The same operations are under `/api/openapi-servers` in the REST API, and in the registry UI under the OpenAPI view, which only admins see.

The server's name becomes its path, so "Weather APIs" is served at `/weather-apis/mcp` on the gateway. Tool names are the target name, an underscore, and the operation's `operationId`, for example `petstore_getPetById`.

## What gets checked before a target is stored

Adding a target stores nothing until every check passes:

- The SSRF guard checks the spec URL and the base URL, the same way it checks any registered backend. To reach an internal API, an operator adds its host to `SSRF_ALLOWED_HOSTS` or `SSRF_ALLOWED_CIDRS`.
- The registry fetches the spec without following redirects, so give the final URL. Specs larger than `OPENAPI_MCP_SPEC_MAX_BYTES` (5 MiB by default) are refused.
- The spec must be OpenAPI 3.x in JSON or YAML, valid UTF-8, with no YAML anchors or aliases.
- Every `$ref` must be a local `#/...` pointer. A spec split across files, or one that refers to a URL or a local file, is refused; bundle it into one document first (for example with `redocly bundle`). Convert a Swagger 2.0 spec first, for example with [converter.swagger.io](https://converter.swagger.io).
- One target may produce at most `OPENAPI_MCP_MAX_TOOLS_PER_TARGET` tools (100 by default). A larger spec gets a `too_many_tools` error that lists the spec's tags with their operation counts. Choose operations with `include_tags` and `include_operations`; an `include_operations` entry is an `operationId` or `"METHOD /path"`.
- Tool names may be at most 64 characters. Rename long or clashing ones with `tool_name_overrides`, keyed by `operationId`, or by `"METHOD /path"` for an operation that has none.

## Shared or per-user credentials

Each target chooses one `credential_mode`:

- `shared` (the default): the admin enters one credential, which every caller uses. It can be an API key in a header or a query parameter, a bearer token, or HTTP basic auth. The registry stores it encrypted with `SECRET_KEY` and never returns it; responses show only `credential_set`. To rotate it, send a new `auth` in a `PATCH` on the target.
- `per_user`: each user brings their own API key or personal access token, which they get from the API provider themselves. The registry never obtains keys for anyone. The admin sets only the shape, `bearer` or `api_key_header` with its header name. Per-user targets need `EGRESS_AUTH_ENABLED=true` and a vault (`SECRET_STORE_BACKEND`), because the keys are stored there, like egress PATs.

One server can mix both kinds of target. A per-user target, as the admin adds it:

```json
{"name": "nasa", "spec_url": "https://example.com/nasa-openapi.json", "base_url": "https://api.nasa.gov",
 "auth": {"scheme": "api_key_header", "header_name": "X-Api-Key", "credential_mode": "per_user"}}
```

Each user then adds their key, in the UI ("Your API keys" on the server card, or Connected Accounts) or through the API:

```bash
curl -X PUT "$REGISTRY_URL/api/openapi-servers/nasa/targets/nasa/my-credential" \
  -H "Authorization: Bearer $ACCESS_TOKEN" -H "Content-Type: application/json" \
  -d '{"secret": "YOUR_NASA_KEY", "ttl_value": 30, "ttl_unit": "days"}'
```

- A key is write-only. `GET .../my-credential` and `GET /api/openapi-servers/{slug}/my-credentials` return only whether a key is set and when it expires. `DELETE .../my-credential` removes it.
- A key lasts at most 30 days, and the user adds a new one after that.
- A key is bound to the target's base URL. If an admin points the target somewhere else, stored keys stop working, and users add them again.
- An admin can store a key on a user's behalf by adding `sub` and that user's `auth_method` to the request, as with egress PATs.
- A call to a per-user tool without a key returns a tool error asking the user to add one. It never falls back to a shared key.

At call time the gateway sends a per-user server's traffic through the auth server, which forwards the caller's keys to the container in a signed, single-use envelope. The auth server never holds the key that signs it.

The registry keeps the exact spec it checked. A change to the spec at its URL reaches agents only after an admin refreshes the target (`refresh_spec: true`), which runs every check again.

## Who can use a server

A new server is `private`: only admins can see and call it. This differs from `POST /api/servers/register`, where the default is `public`. Every caller spends the targets' shared credentials, so open a server with shared targets to a group (`group-restricted`) or to everyone (`public`) only when it is ready. A server whose targets are all per-user spends no one's key but the caller's. The server is an ordinary registry server, so scopes, the gateway's access check, search and the per-tool block all apply to it unchanged.

Each tool is marked from its HTTP method: `GET`, `HEAD` and `OPTIONS` are read-only, and `DELETE` is destructive. Clients that confirm destructive calls, such as Claude Code, ask before a `DELETE`. To remove a tool entirely, block it with the existing per-tool block (`tool_overrides`) on the server, or leave its operation out with the include filters.

## What callers should expect

- A JSON array response arrives wrapped as `{"result": [...]}`, because MCP structured output has to be an object.
- Tools refuse arguments the spec does not define, so a misspelled argument returns an error.
- Responses larger than `OPENAPI_MCP_MAX_RESPONSE_BYTES` (2 MiB by default) become a tool error, as do backend calls slower than `OPENAPI_MCP_BACKEND_TIMEOUT_SECONDS` (30 seconds).
- By default tools carry no output schema, because many real APIs return responses that break their own spec. Set `keep_output_schemas` on a target to keep the spec's schemas.

## Operations

The registry database is the only stored state. The container keeps each server in memory and the registry pushes it back whenever needed:

- After a container restart, calls to OpenAPI MCP servers return 404 until the registry notices the restart and pushes every server again. With 10 servers of 10 targets each, that took 23 seconds.
- After a registry restart, the registry compares what it stores with what the container runs and pushes anything stale.
- A change that cannot reach the container in time returns `sync_status: "pending"`. The registry retries every `OPENAPI_MCP_RECONCILE_INTERVAL_SECONDS` (60 by default).
- Every change starts a background security scan of the server, which takes a few seconds. If the container restarts or the server is deleted during that scan, the scan fails, and the registry's scan policy tags the server `security-pending` and disables it. The hub comes back, but the server stays disabled until an admin turns it on again with `POST /api/servers/toggle`.
- A change replaces a server's hub without cutting off calls already running on the old one: the old hub keeps serving for 30 seconds before it closes.

The registry's calls to the container carry each target's credential in plain HTTP on the internal network, signed with the sync token. Only the registry and nginx can reach the container. A cluster that needs TLS on internal hops can set `OPENAPI_MCP_INTERNAL_URL` to an `https` URL.

The container runs one replica, because the registry pushes state to one address. With 10 servers of 10 targets each, including one 4.4 MB spec, it peaked at 293 MiB. Turning the feature off leaves stored servers in place without a backend; the registry logs their names at startup. Turn the feature back on, or delete them with it on.

Metrics are under `mcpgw_openapi_mcp_*` on the registry's Prometheus endpoint. Alert when `mcpgw_openapi_mcp_state{kind="container_reachable"}` stays at 0 for five minutes.

## End-to-end test

`tests/integration/test_openapi_mcp_e2e.py` runs against a local Compose stack with the feature on, and is skipped otherwise. It uses a mock REST API (`tests/integration/openapi_mock_api.py`) that serves its own spec, checks each auth scheme, and echoes the headers it receives. The mock also serves the example NASA spec, [`docs/examples/openapi/nasa.json`](examples/openapi/nasa.json), at `/nasa-spec.json`; the same file is the example for registering a per-user target against `https://api.nasa.gov`.

```bash
docker run -d --name openapi-mock-api --network mcp-gateway-registry_default --no-healthcheck \
    -v "$PWD/tests/integration/openapi_mock_api.py:/mock/tests/integration/openapi_mock_api.py:ro" \
    -v "$PWD/docs/examples/openapi:/mock/docs/examples/openapi:ro" \
    --entrypoint /app/.venv/bin/python mcp-gateway-registry-registry \
    -m uvicorn --app-dir /mock/tests/integration openapi_mock_api:app --host 0.0.0.0 --port 8000
# add openapi-mock-api to SSRF_ALLOWED_HOSTS in .env, then recreate registry and openapi-mcp
OPENAPI_MCP_E2E=1 uv run pytest tests/integration/test_openapi_mcp_e2e.py -v --no-cov
```

