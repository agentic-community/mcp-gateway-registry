# Agent Management Examples

This directory contains example JSON files for registering A2A agents using the agent management CLI.

## Quick Start

### Service Account: `mcp-gateway-m2m`

The agent management CLI uses the **`mcp-gateway-m2m`** service account for all operations.

**Token Details:**
- **Service Account ID:** `mcp-gateway-m2m`
- **Token Location:** `.oauth-tokens/ingress.json`
- **Token Generation:** `./credentials-provider/generate_creds.sh`
- **Required Keycloak Groups:**
  - `mcp-servers-unrestricted` (for MCP server access)
  - `a2a-agent-admin` (for agent management permissions)

### Prerequisites

Start the registry service in one terminal:

```bash
python -m uvicorn registry.main:app --reload
```

Wait for: `Uvicorn running on http://127.0.0.1:8000`

### Register an Agent

In another terminal, the agent management CLI will automatically use the `mcp-gateway-m2m` token from `.oauth-tokens/ingress.json`:

```bash
# Register the test code reviewer agent
# Token is automatically loaded from .oauth-tokens/ingress.json (mcp-gateway-m2m service account)
uv run python cli/agent_mgmt.py register cli/examples/test_code_reviewer_agent.json
```

### Verify Registration

```bash
# List all agents
uv run python cli/agent_mgmt.py list

# Get specific agent details
uv run python cli/agent_mgmt.py get /test-reviewer

# Test agent accessibility
uv run python cli/agent_mgmt.py test /test-reviewer
```

## Available Examples

All example files use the complete A2A agent schema with all fields documented:

### code_reviewer_agent.json

Comprehensive code review agent analyzing code quality, bugs, and improvements.

**Skills:**
- Analyze Code Quality
- Detect Bugs
- Suggest Improvements

**Security:** JWT Bearer token authentication
**Features:** Streaming enabled, verified trust level

**Usage:**
```bash
uv run python cli/agent_mgmt.py register cli/examples/code_reviewer_agent.json
```

### test_automation_agent.json

Intelligent test automation agent for generating and executing test cases.

**Skills:**
- Generate Unit Tests
- Execute Tests
- Analyze Test Coverage
- Generate Test Report

**Security:** API Key + OAuth2 authentication
**Features:** Streaming enabled, community trust level

**Usage:**
```bash
uv run python cli/agent_mgmt.py register cli/examples/test_automation_agent.json
```

### data_analysis_agent.json

Advanced data analysis agent for statistical analysis and visualization.

**Skills:**
- Statistical Analysis
- Data Visualization
- Predictive Modeling
- Anomaly Detection
- Data Transformation

**Security:** JWT Bearer + OpenID Connect
**Features:** GPU-enabled, verified trust level, supports large datasets

**Usage:**
```bash
uv run python cli/agent_mgmt.py register cli/examples/data_analysis_agent.json
```

### security_analyzer_agent.json

Comprehensive security analysis agent for vulnerability detection and compliance.

**Skills:**
- Scan for Vulnerabilities
- Check Compliance
- Analyze Authentication
- Penetration Testing
- Generate Security Report

**Security:** Mutual TLS + API Key authentication
**Features:** Trusted level, comprehensive CVE database

**Usage:**
```bash
uv run python cli/agent_mgmt.py register cli/examples/security_analyzer_agent.json
```

### documentation_agent.json

Documentation agent for generating and maintaining API docs and guides.

**Skills:**
- Generate API Documentation
- Extract and Format Docstrings
- Generate README
- Maintain Documentation
- Generate Changelog

**Security:** Basic Auth + API Token
**Features:** Supports multiple documentation formats, community trust level

**Usage:**
```bash
uv run python cli/agent_mgmt.py register cli/examples/documentation_agent.json
```

### devops_deployment_agent.json

DevOps automation agent for infrastructure and deployment management.

**Skills:**
- Deploy Application
- Manage Infrastructure
- Configure CI/CD Pipeline
- Monitor Health and Performance
- Manage Secrets and Credentials
- Auto-Scale Application

**Security:** AWS SigV4 + Client Certificate
**Features:** Multi-cloud support, verified trust level

**Usage:**
```bash
uv run python cli/agent_mgmt.py register cli/examples/devops_deployment_agent.json
```

### a2a-1.0-agent-card.json

This registration example shows A2A 1.0 `supportedInterfaces` and `securityRequirements`. Both declared bindings use the registered `/a2a` backend URL, which lets a gateway deployment route either one. The bearer scheme describes the agent's own authentication; the gateway does not issue that credential.

Replace `agent.example.com` with a running agent that serves both JSON-RPC and HTTP+JSON at `/a2a`. If it serves only one, remove the other interface. The registry checks the endpoint on registration and does not turn one binding into another.

```bash
uv run python cli/agent_mgmt.py register cli/examples/a2a-1.0-agent-card.json
```

## Complete A2A Schema Fields

Examples use the registry's agent registration format. `a2a-1.0-agent-card.json` shows the A2A 1.0 transport and security fields; older examples use earlier A2A versions or registry extensions.

**Required registration fields:**
- `name`: Agent display name
- `url`: Agent endpoint URL
- `supportedProtocol`: `a2a` for an A2A agent

The CLI examples also supply a description, protocol version, and path. If you omit `path`, the registry derives it from `name`.

**Optional A2A fields:**
- `protocolVersion`: A2A protocol version (for example, `1.0`)
- `supportedInterfaces`: Ordered transport bindings, each with a URL, `protocolBinding`, and `protocolVersion`
- `version`: Agent version
- `provider`: Agent provider
- `securitySchemes`: Named authentication schemes
- `securityRequirements`: A2A 1.0 scheme requirements with wrapped scope lists
- `skills`: Agent skills
- `capabilities`: Supported features, such as streaming
- `metadata`: Extra registry metadata

**Registry Extensions:**
- `tags`: Array of categorization tags
- `is_enabled`: Whether agent is enabled
- `num_stars`: Community rating
- `license`: License information
- `visibility`: "public", "private", or "group-restricted"
- `allowed_groups`: Groups with access (for group-restricted)
- `trust_level`: "unverified", "community", "verified", or "trusted"
- `registered_at`: Registration timestamp (auto-set)
- `updated_at`: Last update timestamp (auto-set)
- `registered_by`: Username who registered (auto-set)
- `signature`: JWS signature for integrity

**Federation & Lifecycle Metadata (New):**
- `status`: Lifecycle status - "active", "beta", "draft", or "deprecated"
- `provider`: Provider information object with:
  - `organization`: Provider organization name
  - `url`: Provider website or documentation URL
- `source_created_at`: Original creation timestamp from source system (ISO 8601 format)
- `source_updated_at`: Last update timestamp from source system (ISO 8601 format)

### Complete Examples with All Fields

For reference implementations showing all available fields including the new federation and lifecycle metadata:

**complete-server-example.json**
- Shows all server configuration fields
- Includes lifecycle status, provider info, federation timestamps
- Demonstrates custom metadata usage

**complete-agent-example.json**
- Shows all agent configuration fields
- Includes lifecycle status, provider info, federation timestamps
- Full agent card schema example

**Usage:**
```bash
# Register server with all fields
uv run python cli/registry_mgmt.py register cli/examples/complete-server-example.json

# Register agent with all fields
uv run python cli/agent_mgmt.py register cli/examples/complete-agent-example.json
```

## Creating Your Own Agent File

Copy an example and modify the fields:

```bash
cp cli/examples/test_code_reviewer_agent.json cli/examples/my_custom_agent.json
```

Then edit the JSON with your agent details:

```json
{
  "name": "My Custom Agent",
  "path": "/my-agent",
  "description": "What my agent does",
  "url": "http://my-domain.com/agents/my-agent",
  "version": "1.0.0",
  "visibility": "public",
  "trust_level": "community",
  "tags": ["custom", "my-agent"],
  "security_schemes": {
    "bearer": {
      "type": "bearer"
    }
  },
  "protocol_version": "1.0"
}
```

Register your agent:

```bash
export TOKEN="test-token"
uv run python cli/agent_mgmt.py register cli/examples/my_custom_agent.json
```

## Required Fields

All agent JSON files must include:

- `name` - Agent display name (string)
- `path` - Internal path identifier (string, must start with `/`)
- `description` - Brief description (string)
- `url` - Agent endpoint URL (string)
- `version` - Version number (string, e.g., "1.0.0")
- `visibility` - Public visibility (string: "public", "private", "community")
- `trust_level` - Trust classification (string)
- `tags` - Discovery tags (array of strings)
- `security_schemes` - Authentication config (object)
- `protocol_version` - A2A protocol version (string)

## Error Handling

### Agent Already Exists (HTTP 409)

If you get: `Error: Agent with path '/test-reviewer' already exists`

Solution: Change the path in your JSON file or delete the existing agent.

### Validation Failed (HTTP 422)

If you get: `Error: Validation failed - check agent JSON format`

Solution: Verify all required fields are present and properly formatted. Validate with:

```bash
jq . cli/examples/test_code_reviewer_agent.json
```

### Connection Refused

If you get connection errors:

1. Ensure the registry service is running
2. Check it's on the correct port (default: `localhost:8000`)
3. Verify with: `curl http://localhost:8000/api/health`

## Storage

After registration, agent files are stored in:

```bash
ls registry/agents/
cat registry/agents/test-reviewer.json
cat registry/agents/agent_state.json
```

## Next Steps

1. Register a test agent
2. View agents in the frontend dashboard
3. Test agent accessibility
4. Explore the admin panel for agent management

For complete documentation, see: `.scratchpad/A2A_AGENT_CLI_REGISTRATION_GUIDE.md`
