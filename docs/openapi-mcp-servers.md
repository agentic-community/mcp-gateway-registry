# OpenAPI MCP servers

An OpenAPI MCP server turns REST APIs into MCP tools without anyone writing an MCP server. An admin creates the server in the registry and adds REST APIs to it, each one given as an OpenAPI 3.x spec URL, a base URL and one credential. Every operation in the spec becomes a tool, and agents connect to the server like any other MCP server, through the gateway, with the same login and access rules.

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

Helm: set both `openapiMcp.enabled: true` (deploys the container) and `registry.app.openapiMcpEnabled: true` (turns the registry side on). The stack chart refuses to render when the two differ.

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
- The spec must be OpenAPI 3.x in JSON or YAML, valid UTF-8, with no YAML anchors or aliases. Convert a Swagger 2.0 spec first, for example with [converter.swagger.io](https://converter.swagger.io).
- One target may produce at most `OPENAPI_MCP_MAX_TOOLS_PER_TARGET` tools (100 by default). A larger spec gets a `too_many_tools` error that lists the spec's tags with their operation counts. Choose operations with `include_tags` and `include_operations`; an `include_operations` entry is an `operationId` or `"METHOD /path"`.
- Tool names may be at most 64 characters. Rename long or clashing ones with `tool_name_overrides`, keyed by `operationId`, or by `"METHOD /path"` for an operation that has none.

Each target uses one credential for every caller: an API key in a header or a query parameter, a bearer token, or HTTP basic auth. The registry stores it encrypted and never returns it. Responses show only `credential_set`. To rotate it, send a new `auth` in a `PATCH` on the target.

The registry keeps the exact spec it checked. A change to the spec at its URL reaches agents only after an admin refreshes the target (`refresh_spec: true`), which runs every check again.

## Who can use a server

A new server is `private`: only admins can see and call it. This differs from `POST /api/servers/register`, where the default is `public`. Every caller spends the targets' shared credentials, so open a server to a group (`group-restricted`) or to everyone (`public`) only when it is ready. The server is an ordinary registry server, so scopes, the gateway's access check, search and the per-tool block all apply to it unchanged.

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

`tests/integration/test_openapi_mcp_e2e.py` runs against a local Compose stack with the feature on, and is skipped otherwise. It uses a mock REST API (`tests/integration/openapi_mock_api.py`) that serves its own spec, checks each auth scheme, and echoes the headers it receives.

```bash
docker run -d --name openapi-mock-api --network mcp-gateway-registry_default --no-healthcheck \
    -v "$PWD/tests/integration/openapi_mock_api.py:/mock/openapi_mock_api.py:ro" \
    --entrypoint /app/.venv/bin/python mcp-gateway-registry-registry \
    -m uvicorn --app-dir /mock openapi_mock_api:app --host 0.0.0.0 --port 8000
# add openapi-mock-api to SSRF_ALLOWED_HOSTS in .env, then recreate registry and openapi-mcp
OPENAPI_MCP_E2E=1 uv run pytest tests/integration/test_openapi_mcp_e2e.py -v --no-cov
```

