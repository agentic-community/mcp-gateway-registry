# Integrating Amazon Bedrock AgentCore Gateways

This guide demonstrates how to register and use an Amazon Bedrock AgentCore Gateway as an MCP server through the MCP Gateway Registry.

## Overview

[Amazon Bedrock AgentCore Gateway](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway.html) provides an easy and secure way for developers to build, deploy, discover, and connect to tools at scale. AI agents need tools to perform real-world tasks—from querying databases to sending messages to analyzing documents.

This guide uses the [Customer Support Assistant](https://github.com/awslabs/amazon-bedrock-agentcore-samples/tree/main/02-use-cases/customer-support-assistant) example from the `amazon-bedrock-agentcore-samples` repository to demonstrate how to register and use an AgentCore Gateway as an MCP server through the MCP Gateway Registry.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Integration Flow                                   │
└─────────────────────────────────────────────────────────────────────────────┘

                                                   ┌──────────────────────────┐
                                                   │   AWS Cloud              │
                                                   │                          │
  ┌──────────────┐        ┌──────────────────┐     │  ┌────────────────────┐  │
  │              │        │                  │     │  │  Amazon Bedrock    │  │
  │  AI Agent    │───────▶│  MCP Gateway &   │──────▶│  AgentCore Gateway │  │
  │  (Claude)    │        │  Registry        │     │  │                    │  │
  │              │        │  localhost:7860  │     │  │  Customer Support  │  │
  └──────────────┘        └──────────────────┘     │  │  MCP Server        │  │
                                   │               │  │                    │  │
                                   │               │  │  - Warranty Tool   │  │
                                   │               │  │  - Customer Tool   │  │
                                   │               │  │  - Knowledge Base  │  │
                                   │               │  └────────────────────┘  │
                          ┌────────▼────────┐      │            │             │
                          │                 │      │            │             │
                          │  Upstream Auth  │      │    ┌───────▼──────────┐  │
                          │  - Cognito OAuth│◀─────────│    Cognito User  │  │
                          │  - Managed by   │      │    │  Pool (OAuth)    │  │
                          │    Gateway      │      │    └──────────────────┘  │
                          └─────────────────┘      │                          │
                                                   └──────────────────────────┘

Flow:
1. AI Agent sends request to MCP Gateway Registry
2. Gateway routes to registered AgentCore Gateway MCP server
3. Gateway authenticates to AgentCore using gateway-managed Cognito OAuth
4. Tools execute (warranty lookup, customer profile, knowledge base query)
5. Response flows back through Gateway to AI Agent
```

## Prerequisites

- AWS Account with Amazon Bedrock AgentCore access
- EC2 instance (recommended) - See [Complete Setup Guide](complete-setup-guide.md) for EC2 configuration details
  - Alternatively, you can run on macOS - See [macOS Setup Guide](macos-setup-guide.md)
  - Optional: For direct desktop access to EC2 instead of port forwarding, see [Remote Desktop Setup](remote-desktop-setup.md)
- Docker and Docker Compose installed
- Python 3.11+ with `uv` package manager
- Git

## Step 1: Set Up the MCP Gateway Registry

### 1.1 Deploy the Registry

Deploy the registry using pre-built containers. See [README - Option A: Pre-built Images](../README.md#option-a-pre-built-images-instant-setup) for detailed instructions.

```bash
# Create workspace directory
mkdir -p ${HOME}/workspace
cd ${HOME}/workspace

# Clone the repository
git clone https://github.com/agentic-community/mcp-gateway-registry.git
cd mcp-gateway-registry

# Copy environment file
cp .env.example .env

# Configure environment (see Complete Setup Guide for details)
export DOCKERHUB_ORG=mcpgateway

# Deploy with pre-built images
./build_and_run.sh --prebuilt
```

Follow the [Complete Setup Guide - Initial Environment Configuration](complete-setup-guide.md#initial-environment-configuration) for detailed configuration steps including Keycloak initialization and agent account creation.

### 1.2 Verify Registry is Running

Open your browser and navigate to:
```
http://localhost:7860
```

You should see the MCP Gateway Registry UI with the list of registered services.

## Step 2: Set Up Your AgentCore Gateway

### 2.1 Deploy the Customer Support Assistant

Clone the Amazon Bedrock AgentCore samples repository and follow the setup instructions:

```bash
# Navigate to workspace directory
cd ${HOME}/workspace

# Clone the AgentCore samples repository
git clone https://github.com/awslabs/amazon-bedrock-agentcore-samples.git
cd amazon-bedrock-agentcore-samples/02-use-cases/customer-support-assistant
```

Follow the instructions in the [Customer Support Assistant README](https://github.com/awslabs/amazon-bedrock-agentcore-samples/tree/main/02-use-cases/customer-support-assistant) to deploy the AgentCore gateway in your AWS account. You can do this in a separate terminal on the same EC2 machine.

This will create:
- Amazon Bedrock AgentCore Gateway
- Cognito User Pool for authentication
- Lambda functions for warranty status and customer profile lookup
- Knowledge base integration

**Important:** During deployment, record the AgentCore gateway’s OAuth details so you can configure gateway-managed upstream auth:

- Cognito token URL (OAuth `/token` endpoint)
- OAuth client ID
- OAuth client secret
- Required scope (if any)

**To find your Client Secret (for gateway upstream auth configuration):**
1. Go to AWS Cognito Console
2. Find the User Pool with prefix `customersupport-`
3. Navigate to App Integration → App clients
4. Click on your app client
5. View and copy the Client Secret

### 2.2 Verify AgentCore Gateway is Working

After deployment, test the gateway directly using the test script:

```bash
cd ${HOME}/workspace/amazon-bedrock-agentcore-samples/02-use-cases/customer-support-assistant
python test/test_gateway.py --prompt "Check warranty with serial number MNO33333333"
```

If successful, you should see output showing the gateway endpoint and warranty lookup results with tool execution details.

## Step 3: Register the AgentCore Gateway with MCP Registry

### 3.1 Create Gateway Configuration File

Create a file named `gateway-config.json` in your AgentCore project directory with the following content:

```json
{
  "server_name": "customer-support-assistant",
  "description": "Amazon Bedrock AgentCore Gateway for customer support operations with warranty lookup and knowledge base",
  "path": "/customer-support-assistant",
  "proxy_pass_url": "https://<YOUR-GATEWAY-ID>.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp/",
  "auth_type": "oauth",
  "upstream_auth": {
    "mode": "gateway-managed",
    "type": "oauth2",
    "provider": "cognito",
    "credential_binding": "service",
    "injection": {
      "kind": "header",
      "name": "Authorization",
      "value_format": "Bearer {access_token}"
    }
  },
  "supported_transports": [
    "streamable-http"
  ],
  "tags": [
    "bedrock",
    "agentcore",
    "customer-support",
    "warranty",
    "knowledge-base"
  ],
  "num_tools": 2,
  "num_stars": 0,
  "is_python": false,
  "license": "Apache-2.0",
  "tool_list": [
    {
      "name": "LambdaUsingSDK___check_warranty_status",
      "parsed_description": {
        "main": "Check the warranty status of a product using its serial number and optionally verify via email",
        "args": null,
        "returns": null,
        "raises": null
      },
      "schema": {
        "type": "object",
        "properties": {
          "serial_number": {
            "type": "string",
            "description": "Product serial number to check warranty status"
          },
          "customer_email": {
            "type": "string",
            "description": "Optional customer email for verification"
          }
        },
        "required": [
          "serial_number"
        ]
      }
    },
    {
      "name": "LambdaUsingSDK___get_customer_profile",
      "parsed_description": {
        "main": "Retrieve customer profile using customer ID, email, or phone number",
        "args": null,
        "returns": null,
        "raises": null
      },
      "schema": {
        "type": "object",
        "properties": {
          "customer_id": {
            "type": "string",
            "description": "Unique customer identifier"
          },
          "email": {
            "type": "string",
            "description": "Customer email address"
          },
          "phone": {
            "type": "string",
            "description": "Customer phone number"
          }
        },
        "required": [
          "customer_id"
        ]
      }
    }
  ]
}
```

**Key Configuration Parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `path` | `/customer-support-assistant` | The URL path where this service will be accessible through the registry. The registry will automatically format this to `/customer-support-assistant/` (with trailing slash) for bedrock-agentcore services. |
| `proxy_pass_url` | `https://<YOUR-GATEWAY-ID>.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp/` | The backend AgentCore Gateway URL. The registry will automatically remove the `/mcp/` suffix and ensure it ends with just `/` for bedrock-agentcore services. Replace `<YOUR-GATEWAY-ID>` with your actual Gateway ID from the deployment output. |
| `auth_type` | `oauth` | Specifies OAuth authentication flow |
| `upstream_auth` | `{"type":"oauth2","provider":"cognito",...}` | Declares the upstream authentication requirement; credentials are stored and injected by the gateway. |
| `server_name` | `customer-support-assistant` | Display name for the service in the registry UI |
| `tags` | `["bedrock", "agentcore", "customer-support", ...]` | Searchable tags used by the `intelligent_tool_finder` for hybrid search (combines semantic search with tag-based filtering). AI agents can discover tools by category using these tags. |
| `tool_list` | Array of tool definitions | Defines the available tools/functions with their schemas, descriptions, and parameters. Each tool includes a name, parsed description, and JSON schema for arguments. This metadata enables the registry to catalog and expose tools for dynamic discovery by AI agents. |

**Important Notes:**
- Agents authenticate only to the gateway; the gateway handles all upstream authentication to AgentCore.
- Do not distribute Cognito tokens to agents or include upstream `Authorization` headers in client configuration.
- Replace `<YOUR-GATEWAY-ID>` with your actual AgentCore Gateway ID (shown in deployment output)

### 3.2 Register the Gateway

Navigate to your `mcp-gateway-registry` directory and run:

```bash
cd ${HOME}/workspace/mcp-gateway-registry
source .venv/bin/activate

# Register the AgentCore gateway
./cli/service_mgmt.sh add ${HOME}/workspace/amazon-bedrock-agentcore-samples/02-use-cases/customer-support-assistant/gateway-config.json
```

### 3.3 Verify Registration

#### Via UI

1. Open http://localhost:7860 in your browser
2. You should see "customer-support-assistant" in the services list
3. The health status should show as "healthy"

**Screenshot:**
![AgentCore Gateway Registration](img/mcpgw-ac-1.png)

#### Via Command Line

**Note:** Refresh credentials first since Keycloak access tokens have a 5-minute TTL by default:

```bash
# Refresh authentication credentials
./credentials-provider/generate_creds.sh

# List services and filter for customer-support-assistant
uv run cli/mcp_client.py \
  --url http://localhost/mcpgw/mcp \
  call --tool list_services \
  --args '{}' \
  2>/dev/null | \
tail -n +2 | \
jq '.content[0].text | fromjson | .services[] | select(.server_name == "customer-support-assistant")'
```

Look for the customer-support-assistant entry in the output.

## Step 4: Test the Registered Gateway

### 4.1 Refresh Authentication Credentials

Generate fresh ingress credentials for the registry. **Note:** Keycloak access tokens have a 5-minute TTL by default, so you'll need to refresh credentials before testing:

```bash
./credentials-provider/generate_creds.sh
```

### 4.2 Call the AgentCore Gateway Through the Registry

Now you can call the AgentCore gateway tools through the MCP Gateway Registry:

```bash
uv run cli/mcp_client.py \
  --url http://localhost/customer-support-assistant/mcp \
  call --tool LambdaUsingSDK___check_warranty_status \
  --args '{"serial_number":"MNO33333333"}'
```

### 4.3 Test Customer Profile Lookup

```bash
uv run cli/mcp_client.py \
  --url http://localhost/customer-support-assistant/mcp \
  call --tool LambdaUsingSDK___get_customer_profile \
  --args '{"customer_id":"CUST001"}'
```

Expected: a successful MCP tool response from the upstream AgentCore gateway.

## How It Works

### Gateway-Terminated Upstream Authentication

This integration follows the gateway-terminated model:

- Agents authenticate only to the gateway.
- The gateway resolves and injects upstream `Authorization` (Cognito OAuth) when proxying to AgentCore.
- If upstream credentials are required but not configured, expect `424 Failed Dependency` with `error_code=UPSTREAM_CREDENTIALS_REQUIRED`.

```mermaid
sequenceDiagram
    participant Agent as AI Agent
    participant Gateway as MCP Gateway (Nginx)
    participant Auth as Auth Server (/validate)
    participant AgentCore as AgentCore Gateway
    participant Cognito as Cognito (Upstream IdP)

    Agent->>Gateway: MCP request + gateway auth
    Gateway->>Auth: auth_request (/validate)
    Auth-->>Gateway: allow/deny + upstream injection metadata
    Gateway->>AgentCore: MCP request + injected Authorization
    AgentCore->>Cognito: validate upstream token (as required)
    AgentCore-->>Gateway: MCP response
    Gateway-->>Agent: MCP response
```


## Next Steps

### 1. Configure Fine-Grained Access Control (FGAC)

Set up access controls so the customer-support-assistant service is only accessible to users in specific groups:

```bash
# Create a new group/scope for customer support users
cd ${HOME}/workspace/mcp-gateway-registry

# Add the group and assign users
# See the complete end-to-end example in the Service Management Guide
```

**Learn More:** Follow the [Service Management Guide - Complete Example: LOB1 Services Group](service-management.md#complete-example-lob1-line-of-business-1-services-group) for detailed instructions on:
- Creating groups and scopes
- Assigning users to groups
- Configuring service-level access control
- Testing access permissions

### 2. Additional Integration Options

- **Add more AgentCore gateways** to your registry following this same process
- **Integrate with Claude Desktop** or other MCP clients for AI-powered interactions
- **Browse registered services** through the registry UI at http://localhost:7860
- **Monitor usage and metrics** through Grafana dashboards at http://localhost:3000 - See [Observability Guide](OBSERVABILITY.md)

## Troubleshooting

### 404 Not Found Error

If you get a 404 error, verify:
1. Service is registered: `uv run cli/mcp_client.py --url http://localhost/mcpgw/mcp call --tool list_services --args '{}'`
2. Path matches: Use `/customer-support-assistant/` (with trailing slash)
3. Health status is healthy in the UI

### 401 Authentication Error

If you get a 401 error:
1. Refresh your gateway access token: `./credentials-provider/generate_creds.sh`
2. Verify your client is sending the gateway auth headers expected by your deployment
3. If the gateway reports missing upstream credentials, configure upstream auth for this server in the Registry UI

### Service Not Showing as Healthy

1. Verify AgentCore gateway is accessible from registry container
2. Check network connectivity
3. Review registry logs: `docker logs mcp-gateway-registry-registry-1`

## Additional Resources

- [Amazon Bedrock AgentCore Samples](https://github.com/awslabs/amazon-bedrock-agentcore-samples)
- [MCP Gateway Registry Documentation](../README.md)
- [Service Management Guide](service-management.md)
