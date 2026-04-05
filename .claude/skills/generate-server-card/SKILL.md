---
name: generate-server-card
description: Generate an MCP server card JSON (mcp-gateway-registry format) by analyzing server source code in a folder or GitHub URL. Studies code to detect server name, tools, transport, auth, and generates a registry-compatible config.
argument-hint: "[folder-path-or-github-url]"
disable-model-invocation: true
---

# Generate MCP Server Card

Generate an MCP server card JSON file in the mcp-gateway-registry format by analyzing the source code at `$ARGUMENTS`.

This format is used by https://github.com/agentic-community/mcp-gateway-registry for registering MCP servers in the gateway registry.

## Steps

### 1. Carefully study the source code

Do NOT just skim the top-level files. MCP servers can be complex, multi-file projects. You MUST thoroughly explore the entire folder structure before generating the card.

If `$ARGUMENTS` is a local folder path, use Glob to find ALL Python files, YAML/JSON configs, and README files in the folder and its subfolders. Read every relevant file.

If `$ARGUMENTS` is a GitHub URL, use `WebFetch` to read the raw file contents. Follow imports to discover additional files.

**Where to look** (MCP servers are not always a single file):

- **Entrypoints**: `main.py`, `app.py`, `server.py`, `__main__.py`, `mcp_server.py`
- **Tool definitions**: look for `@mcp.tool()`, `@server.tool()`, `@tool`, tool handler functions, tool registrations
- **OpenAPI specs**: `.json` or `.yaml` files in `openapi-specs/`, `specs/`, `api/` folders - these define the tools when an API is exposed as MCP
- **Config files**: `pyproject.toml`, `requirements.txt`, `.bedrock_agentcore.yaml`, `deploy_*.py`, `docker-compose.yml`
- **Deployment scripts**: these reveal the gateway URL, auth scheme, transport type
- **Gateway configs**: `.deployment_info.json`, `.roo.json` - these have the actual MCP endpoint URL
- **Sub-folders**: check ALL subdirectories for additional tools, resources, or shared utilities
- **Resource definitions**: MCP servers can expose resources and prompts in addition to tools

From the code, detect:

- **Server name**: from constants, config files, class names, gateway name
- **Description**: from docstrings, README, module-level comments, OpenAPI spec info
- **Tools**: from `@tool` decorators, tool registrations, OpenAPI operationIds, function definitions
- **Tool schemas**: parameter types, required fields, descriptions from function signatures or OpenAPI specs
- **Supported transports**: `streamable-http`, `sse`, `stdio` - from server framework config
- **Auth scheme**: `none`, bearer JWT, API key, OAuth2, Cognito - from deployment scripts or middleware
- **MCP endpoint URL**: from deployment info, gateway URL, or config files
- **Proxy pass URL**: the backend URL the gateway proxies to (may differ from the MCP endpoint)
- **Version**: from constants, pyproject.toml, or default to `1.0.0`
- **Provider info**: organization name and URL if detectable
- **License**: from pyproject.toml or LICENSE file
- **Status**: `active`, `beta`, etc.

### 2. Generate the server card JSON

Create a JSON file following the mcp-gateway-registry server config format.

#### Required fields

```json
{
  "server_name": "string - Human-readable server name",
  "description": "string - What the server does",
  "path": "string - URL path prefix on the gateway, must start with /",
  "proxy_pass_url": "string - Backend URL to proxy to",
  "tags": ["string - Relevant tags for discovery"],
  "supported_transports": ["streamable-http"]
}
```

#### Recommended optional fields (include when detected)

```json
{
  "auth_scheme": "string - none, bearer, api_key, oauth2",
  "mcp_endpoint": "string - Custom MCP endpoint URL if different from proxy_pass_url",
  "version": "string - Server version e.g. 1.0.0",
  "status": "string - active, beta, deprecated",
  "visibility": "string - public or private",
  "provider_organization": "string - Organization name",
  "provider_url": "string - Organization URL",
  "num_tools": 0,
  "license": "string - License identifier e.g. MIT, Apache-2.0",
  "metadata": {
    "category": "string - e.g. documentation, geolocation, utility",
    "official": false,
    "mcp_compatible": "1.0"
  },
  "tool_list": []
}
```

### 3. Do NOT populate tool_list

- Always set `"tool_list": []` (empty array)
- Do NOT add individual tool entries to tool_list - leave it empty
- Still count the tools and set `num_tools` to the correct count (for informational purposes)
- The tool details are intentionally omitted from the server card

### 4. Set the path

- The `path` must start with `/`
- Use the gateway name or a short descriptive slug, e.g. `/geo-mcp`, `/cloudflare-docs`
- Check `.deployment_info.json` for the gateway name if available

### 5. Set the proxy_pass_url and mcp_endpoint

- Check for `.deployment_info.json`, `.roo.json`, or `.bedrock_agentcore.yaml` for the actual deployed URL
- `proxy_pass_url`: the backend URL the gateway proxies to
- `mcp_endpoint`: the full MCP endpoint URL that clients connect to (if different from proxy_pass_url)
- If no deployment info found, use placeholder: `https://<GATEWAY_URL>/mcp`

### 6. Detect transport type

- `streamable-http`: most common for remote MCP servers, HTTP-based
- `sse`: Server-Sent Events based transport
- `stdio`: local process communication (not for remote servers)
- Default to `["streamable-http"]` for gateway-deployed servers

### 7. Detect auth scheme

- Cognito JWT / bearer token: `"bearer"` or describe as `"custom_jwt"`
- API key: `"api_key"`
- No auth: `"none"`
- Check deployment scripts for `authorizerType`, `customJWTAuthorizer`, bearer token usage

### 8. Save the output

- Name the file `{server_name}_server_card.json` using the detected server name in snake_case
- Save it in the server's folder (same folder as the source code)
- Pretty-print with 2-space indentation

### 9. Validate the generated JSON

After writing the file, validate it by running this Python script via Bash:

```bash
python3 -c "
import json
import sys

file_path = '<OUTPUT_FILE_PATH>'

# Step 1: JSON format check
try:
    with open(file_path) as f:
        card = json.load(f)
    print('PASS: Valid JSON format')
except json.JSONDecodeError as e:
    print(f'FAIL: Invalid JSON - {e}')
    sys.exit(1)

# Step 2: Required fields
errors = []
REQUIRED = ['server_name', 'description', 'path', 'proxy_pass_url', 'tags', 'supported_transports']
for field in REQUIRED:
    if field not in card:
        errors.append(f'Missing required field: {field}')
    elif field in ('server_name', 'description', 'path', 'proxy_pass_url') and not isinstance(card[field], str):
        errors.append(f'{field} must be a string')
    elif field in ('tags', 'supported_transports') and not isinstance(card[field], list):
        errors.append(f'{field} must be an array')

# Step 3: Path must start with /
if 'path' in card and isinstance(card['path'], str) and not card['path'].startswith('/'):
    errors.append('path must start with /')

# Step 4: proxy_pass_url must be a valid URL
if 'proxy_pass_url' in card and isinstance(card['proxy_pass_url'], str):
    if not card['proxy_pass_url'].startswith(('http://', 'https://')):
        errors.append('proxy_pass_url must start with http:// or https://')

# Step 5: tags must be non-empty
if not card.get('tags'):
    errors.append('tags must have at least one entry')

# Step 6: supported_transports must be non-empty
if not card.get('supported_transports'):
    errors.append('supported_transports must have at least one entry')
valid_transports = ['streamable-http', 'sse', 'stdio']
for t in card.get('supported_transports', []):
    if t not in valid_transports:
        errors.append(f'Unknown transport: {t}, must be one of {valid_transports}')

# Step 7: Validate tool_list is an empty array
if 'tool_list' in card:
    if not isinstance(card['tool_list'], list):
        errors.append('tool_list must be an array')
    elif len(card['tool_list']) > 0:
        errors.append('tool_list must be empty ([]) - do not populate individual tool entries')

# Step 8: num_tools must be present and a non-negative integer
if 'num_tools' in card:
    if not isinstance(card['num_tools'], int) or card['num_tools'] < 0:
        errors.append('num_tools must be a non-negative integer')

# Step 9: Report results
if errors:
    print(f'FAIL: {len(errors)} validation error(s):')
    for e in errors:
        print(f'  - {e}')
    sys.exit(1)
else:
    tool_count = len(card.get('tool_list', []))
    transport_count = len(card.get('supported_transports', []))
    print(f'PASS: All required fields present ({tool_count} tool(s), {transport_count} transport(s))')
"
```

Replace `<OUTPUT_FILE_PATH>` with the actual path of the generated JSON file.

If validation fails, fix the errors in the JSON and re-run validation until all checks pass. Do NOT report success to the user until validation passes.

### 10. Report results

After validation passes, output results in EXACTLY this format:

```
Validation passed. Here's a summary of the generated server card:

Output file: <filename>.json

Detected from code:

- Server name: <ServerName> (from <how it was detected>)
- Tools: <count> - <tool_name> (<brief description>), ...
- Transport: <streamable-http|sse|stdio> (<why>)
- Auth: <none|bearer|api_key|oauth2> (<details>)
- MCP endpoint: <URL or placeholder>
- Proxy pass URL: <URL or placeholder>
- Status: <active|beta>
- Provider: <org name> or <not detected>
```

If validation fails, show the errors first, fix them, re-validate, and only show the summary above after all checks pass.

## Reference: Minimal server card

```json
{
  "server_name": "Minimal MCP Server",
  "description": "A minimal server configuration with only required fields",
  "path": "/minimal-server",
  "proxy_pass_url": "http://minimal-server:9001/",
  "supported_transports": ["streamable-http"],
  "tags": ["mcp", "minimal", "example"]
}
```

## Reference: Full server card with tools

```json
{
  "server_name": "Geolocation MCP Server",
  "description": "IP geolocation lookup via ipwho.is, deployed on Amazon Bedrock AgentCore Gateway with Cognito JWT auth",
  "path": "/geo-mcp",
  "proxy_pass_url": "https://geo-mcp-abc123.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp",
  "supported_transports": ["streamable-http"],
  "auth_scheme": "bearer",
  "tags": ["geolocation", "ip-lookup", "aws", "bedrock-agentcore"],
  "num_tools": 1,
  "version": "1.0.0",
  "status": "active",
  "visibility": "public",
  "provider_organization": "AWS",
  "provider_url": "https://aws.amazon.com",
  "license": "MIT",
  "metadata": {
    "category": "geolocation",
    "official": false,
    "mcp_compatible": "1.0"
  },
  "tool_list": []
}
```
