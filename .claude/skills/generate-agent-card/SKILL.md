---
name: generate-agent-card
description: Generate an A2A agent card JSON by analyzing agent source code in a folder or GitHub URL. Studies the code to detect agent name, skills, tools, auth, protocol, and generates a spec-compliant agent card.
argument-hint: "[folder-path-or-github-url]"
disable-model-invocation: true
---

# Generate A2A Agent Card

Generate an A2A (Agent-to-Agent) protocol agent card JSON file by analyzing the source code at `$ARGUMENTS`.

## Steps

### 1. Carefully study the source code

Do NOT just skim the top-level files. Agents can be complex, multi-file projects. You MUST thoroughly explore the entire folder structure before generating the card.

If `$ARGUMENTS` is a local folder path, use Glob to find ALL Python files, YAML/JSON configs, and README files in the folder and its subfolders. Read every relevant file.

If `$ARGUMENTS` is a GitHub URL, use `WebFetch` to read the raw file contents. Follow imports to discover additional files.

**Where to look** (agents are not always a single file):

- **Entrypoints**: `main.py`, `app.py`, `agent_entrypoint.py`, `server.py`, `__main__.py`
- **Tool definitions**: tools may be in separate files like `tools/`, `skills/`, `functions/`, or registered via decorators (`@tool`, `@function_tool`, `@mcp_tool`)
- **Multi-agent setups**: look for orchestrator/supervisor patterns, multiple agent classes, agent registries, sub-agent folders
- **Prompts and system messages**: may be in separate files like `prompts/`, `templates/`, `system_prompt.txt`, or as string constants
- **MCP server connections**: look for MCP client configs, `mcp_servers`, `MCPClient`, tool imports from MCP servers - these are skills the agent can use
- **Config files**: `pyproject.toml`, `requirements.txt`, `.bedrock_agentcore.yaml`, `agent_config.yaml`, `docker-compose.yml`
- **Sub-folders**: check ALL subdirectories for additional agents, tools, or shared utilities

From the code, detect:

- **Agent name**: from constants, CLI args, config files, class names
- **Description**: from docstrings, README, module-level comments
- **Skills/tools**: from `@tool` decorators, tool lists, function definitions passed to agent frameworks (Strands, LangChain, CrewAI, AutoGen, etc.), MCP tool connections, imported tool modules
- **Multi-agent skills**: if there are multiple agents (orchestrator, sub-agents), each agent's capabilities should be represented as skills
- **MCP-sourced tools**: if the agent connects to MCP servers, list those tools as skills too (read the MCP server configs to find tool names and descriptions)
- **Input/output modes**: text, images, files, structured data - check what the agent accepts and returns
- **Protocol**: HTTP (`HTTP+JSON`), A2A (`JSONRPC`), MCP
- **Auth mechanism**: IAM/SigV4, Cognito/JWT, API key, OAuth2
- **Streaming support**: from framework config, capabilities flags
- **Version**: from constants, pyproject.toml, or default to `1.0.0`
- **Endpoint URL**: from `.bedrock_agentcore.yaml` or deployment config if available

### 2. Generate the agent card JSON

Create a JSON file following the official A2A Agent Card specification (https://a2a-protocol.org/latest/specification/).

All mandatory fields MUST be present. Use camelCase for JSON field names.

#### Required fields

```json
{
  "name": "string - Human-readable agent name",
  "description": "string - What the agent does",
  "version": "string - e.g. 1.0.0",
  "supportedInterfaces": [
    {
      "url": "string - Agent endpoint URL",
      "protocolBinding": "string - JSONRPC or HTTP+JSON or GRPC",
      "protocolVersion": "string - e.g. 1.0"
    }
  ],
  "capabilities": {
    "streaming": false,
    "pushNotifications": false
  },
  "defaultInputModes": ["text/plain"],
  "defaultOutputModes": ["text/plain"],
  "skills": [
    {
      "id": "string - unique skill id",
      "name": "string - human-readable name",
      "description": "string - what the skill does",
      "tags": ["string"],
      "examples": ["string - example prompts"]
    }
  ]
}
```

#### Optional fields to include when detected

- `provider`: Include if organization info is available. Requires `organization` (string) and `url` (string).
- `documentationUrl`: Link to docs if found in README or code.
- `iconUrl`: Agent icon if found.
- `securitySchemes`: Include when auth is detected:
  - Cognito/JWT: `{"bearerAuth": {"httpAuthSecurityScheme": {"scheme": "Bearer", "description": "Cognito JWT bearer token"}}}`
  - API Key: `{"apiKey": {"apiKeySecurityScheme": {"name": "x-api-key", "location": "header"}}}`
  - IAM/SigV4: `{"sigv4": {"httpAuthSecurityScheme": {"scheme": "AWS4-HMAC-SHA256", "description": "AWS SigV4 request signing"}}}`
- `securityRequirements`: Reference the schemes defined above, e.g. `[{"schemes": {"bearerAuth": []}}]`

### 3. Populate skills correctly

- Find ALL tools/functions the agent exposes
- For each tool, create a skill entry with:
  - `id`: snake_case identifier
  - `name`: Human-readable name
  - `description`: From the function docstring or tool description
  - `tags`: Relevant categories (e.g. `["math", "calculator"]`, `["search", "web"]`)
  - `examples`: 2-3 example prompts showing usage

### 4. Set the endpoint URL

- Check for `.bedrock_agentcore.yaml` in the agent folder and read the ARN/endpoint if available
- If no config found, use placeholder: `https://<AGENT_ENDPOINT_URL>/`

### 5. Detect protocol binding

- A2A agents (a2a_server, A2AServer, port 9000, protocol="A2A"): use `JSONRPC`
- HTTP agents (BedrockAgentCoreApp, REST endpoints): use `HTTP+JSON`
- Default to `JSONRPC`

### 6. Save the output

- Name the file `{agent_name}_agent_card.json` using the detected agent name in snake_case
- Save it in the agent's folder (same folder as the source code)
- Pretty-print with 2-space indentation

### 7. Validate the generated JSON

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

# Step 2: Required top-level fields
errors = []
TOP_LEVEL_REQUIRED = ['name', 'description', 'version', 'supportedInterfaces', 'capabilities', 'defaultInputModes', 'defaultOutputModes', 'skills']
for field in TOP_LEVEL_REQUIRED:
    if field not in card:
        errors.append(f'Missing required top-level field: {field}')
    elif field in ('name', 'description', 'version') and not isinstance(card[field], str):
        errors.append(f'{field} must be a string')
    elif field in ('defaultInputModes', 'defaultOutputModes', 'skills', 'supportedInterfaces') and not isinstance(card[field], list):
        errors.append(f'{field} must be an array')
    elif field == 'capabilities' and not isinstance(card[field], dict):
        errors.append(f'{field} must be an object')

# Step 3: Validate supportedInterfaces entries
INTERFACE_REQUIRED = ['url', 'protocolBinding', 'protocolVersion']
for i, iface in enumerate(card.get('supportedInterfaces', [])):
    for field in INTERFACE_REQUIRED:
        if field not in iface:
            errors.append(f'supportedInterfaces[{i}] missing required field: {field}')
    binding = iface.get('protocolBinding', '')
    if binding and binding not in ('JSONRPC', 'GRPC', 'HTTP+JSON'):
        errors.append(f'supportedInterfaces[{i}].protocolBinding must be JSONRPC, GRPC, or HTTP+JSON, got: {binding}')
if not card.get('supportedInterfaces'):
    errors.append('supportedInterfaces must have at least one entry')

# Step 4: Validate skills entries
SKILL_REQUIRED = ['id', 'name', 'description', 'tags']
for i, skill in enumerate(card.get('skills', [])):
    for field in SKILL_REQUIRED:
        if field not in skill:
            errors.append(f'skills[{i}] missing required field: {field}')
    if 'tags' in skill and not isinstance(skill['tags'], list):
        errors.append(f'skills[{i}].tags must be an array')

# Step 5: Validate defaultInputModes/defaultOutputModes are non-empty
if not card.get('defaultInputModes'):
    errors.append('defaultInputModes must have at least one entry')
if not card.get('defaultOutputModes'):
    errors.append('defaultOutputModes must have at least one entry')

# Step 6: Validate provider if present
if 'provider' in card and card['provider'] is not None:
    for field in ('organization', 'url'):
        if field not in card['provider']:
            errors.append(f'provider missing required field: {field}')

# Step 7: Report results
if errors:
    print(f'FAIL: {len(errors)} validation error(s):')
    for e in errors:
        print(f'  - {e}')
    sys.exit(1)
else:
    skill_count = len(card.get('skills', []))
    iface_count = len(card.get('supportedInterfaces', []))
    print(f'PASS: All mandatory fields present ({iface_count} interface(s), {skill_count} skill(s))')
"
```

Replace `<OUTPUT_FILE_PATH>` with the actual path of the generated JSON file.

If validation fails, fix the errors in the JSON and re-run validation until all checks pass. Do NOT report success to the user until validation passes.

### 8. Report results

After validation passes, output results in EXACTLY this format:

```
Validation passed. Here's a summary of the generated agent card:

Output file: <filename>.json

Detected from code:

- Agent name: <AgentName> (from <how it was detected, e.g. Strands agent with BedrockAgentCoreApp>)
- Skills: <count> - <skill_id> (<brief description>), <skill_id> (<brief description>), ...
- Protocol: <HTTP+JSON or JSONRPC> (<why, e.g. uses BedrockAgentCoreApp on port 8080>)
- Auth: <auth mechanisms detected, e.g. IAM/SigV4 (default) + Cognito JWT (optional)>
- Streaming: <true or false>
- Endpoint URL: <source, e.g. From .bedrock_agentcore.yaml - uses the my_agent deployment ARN (account ..., region ...)> or <Placeholder used - no deployment config found>
```

If validation fails, show the errors first, fix them, re-validate, and only show the summary above after all checks pass.

## Reference template

Use this as a structural reference (from simple-a2a-agent):

```json
{
  "name": "SimpleCalculatorAgent",
  "description": "A simple calculator agent that can evaluate mathematical expressions.",
  "version": "1.0.0",
  "supportedInterfaces": [
    {
      "url": "https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/<encoded-arn>/invocations/",
      "protocolBinding": "JSONRPC",
      "protocolVersion": "1.0"
    }
  ],
  "capabilities": {
    "streaming": true,
    "pushNotifications": false
  },
  "defaultInputModes": ["text/plain"],
  "defaultOutputModes": ["text/plain"],
  "skills": [
    {
      "id": "calculator",
      "name": "Calculator",
      "description": "Evaluate a mathematical expression and return the result.",
      "tags": ["math", "calculator", "arithmetic"],
      "examples": [
        "What is 42 * 17?",
        "Calculate the square root of 144",
        "What is 15% of 200?"
      ]
    }
  ]
}
```
