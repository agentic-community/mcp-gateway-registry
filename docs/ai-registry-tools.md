# AI Registry Tools

An MCP server that provides AI agents with tools to discover and query MCP servers, tools, agents, and skills registered in the MCP Gateway Registry.

See the [What's New section](../README.md#whats-new) in the main README for the latest updates and features.

## What This Server Does

**AI Registry Tools** gives your AI coding assistants the ability to:
- **Search for MCP tools** using natural language queries
- **List all available MCP servers** in the registry
- **Discover AI agents** that can help with specific tasks
- **Find skills** (like Claude Code skills) available in the ecosystem
- **Check registry health** and statistics

This server is **automatically registered** in the MCP Gateway Registry and ready to use.

## Adding to Your AI Agent

To use AI Registry Tools with your coding assistant (Claude Code, Roo Code, Cline, etc.):

1. **Navigate to the registry web UI** at your registry URL
2. **Find the AI Registry Tools server card** in the servers list
3. **Click the gear icon** on the server card
4. **Follow the configuration instructions** to add it to your AI agent

The gear icon will provide ready-to-use configuration snippets for popular AI coding assistants.

## Available Tools

### intelligent_tool_finder

Search for MCP tools using natural language semantic search.

**Parameters:**
- `query` (string, required) - Natural language description of what you want to do
- `top_n` (integer, optional) - Number of results to return (default: 5, max: 100)

**Returns:**
```typescript
{
  results: Array<{
    tool_name: string;          // Name of the tool
    server_name: string;         // Server providing the tool
    description: string | null;  // Tool description
    score: number | null;        // Relevance score (0-1)
    path: string | null;         // Server path
  }>;
  query: string;                 // Your search query
  total_results: number;         // Number of results found
  status: "success" | "failed";
}
```

**Example Query:** "find tools to help me work with databases"

---

### list_services

List all MCP servers registered in the gateway.

**Parameters:** None

**Returns:**
```typescript
{
  services: Array<{
    server_name: string | null;  // Display name of the server
    path: string;                // URL path (e.g., '/weather-api')
    description: string | null;   // Server description
    enabled: boolean;            // Whether server is active
    tags: string[];              // Server tags
    tool_count: number | null;   // Number of tools provided
  }>;
  total_count: number;           // Total servers
  enabled_count: number;         // Number of enabled servers
  status: "success" | "failed";
}
```

---

### list_agents

List all AI agents registered in the gateway.

**Parameters:** None

**Returns:**
```typescript
{
  agents: Array<{
    name: string | null;         // Agent name
    description: string | null;  // Agent description
    tags: string[];              // Agent tags
    created_at: string | null;   // ISO timestamp
  }>;
  total_count: number;           // Total agents
  status: "success" | "failed";
}
```

---

### list_skills

List all skills (Claude Code skills, etc.) registered in the gateway.

**Parameters:** None

**Returns:**
```typescript
{
  skills: Array<{
    path: string;                // Skill path
    name: string | null;         // Skill name
    description: string | null;  // Skill description
    tags: string[];              // Skill tags
    created_at: string | null;   // ISO timestamp
  }>;
  total_count: number;           // Total skills
  status: "success" | "failed";
}
```

---

### healthcheck

Get registry health status and statistics.

**Parameters:** None

**Returns:**
```typescript
{
  // Dynamic fields from registry health endpoint
  // May include: total_servers, enabled_servers,
  // total_tools, uptime, version, etc.
  [key: string]: any;
  status: "success" | "failed";
}
```

## Use Cases

### For AI Coding Assistants

**Discover new capabilities:**
```
You: "What tools are available for working with AWS?"
AI: *calls intelligent_tool_finder(query="AWS tools")*
AI: "I found 12 AWS-related tools including aws-kb for documentation,
     aws-bedrock for AI models, and cloudformation for infrastructure..."
```

**Check what's available:**
```
You: "Show me all MCP servers in the registry"
AI: *calls list_services()*
AI: "There are 47 MCP servers registered, including weather-api,
     github-mcp, slack-tools, and more..."
```

**Find specialized agents:**
```
You: "Are there any agents that can help with travel planning?"
AI: *calls list_agents()*
AI: "Yes, there's a travel-assistant-agent that can help with
     flight bookings, hotel searches, and itinerary planning."
```

### For Development Workflows

- **Tool discovery during development** - Find the right MCP tool before building custom solutions
- **Registry exploration** - Understand what's available in your organization's MCP ecosystem
- **Integration planning** - Identify which servers and tools to integrate into your projects
- **Capability mapping** - Map business requirements to available MCP tools

## Authentication

All tools require bearer token authentication. The authentication is handled automatically when you configure the server through your AI agent's settings.

**How authentication works:**
1. Your AI agent includes an `Authorization: Bearer <token>` header with each request
2. AI Registry Tools forwards this token to the registry API
3. The registry validates your token and returns the requested data

**If you see authentication errors:**
- Verify your token is valid and not expired
- Check that your token has appropriate permissions in the registry
- Contact your registry administrator if issues persist

## Technical Architecture

```
AI Agent → AI Registry Tools → MCP Gateway Registry
         (MCP Protocol)      (HTTP/JSON API)
         Bearer Token        Token Forwarding
```

**Design principles:**
- **Lightweight** - Minimal dependencies, fast startup
- **Stateless** - No session management, horizontally scalable
- **Pass-through authentication** - Tokens forwarded to registry
- **Protocol adapter** - Translates MCP tool calls to HTTP API requests

## Configuration

AI Registry Tools is configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `REGISTRY_BASE_URL` | `http://localhost` | Registry API endpoint |
| `HOST` | `127.0.0.1` | Bind host (use `0.0.0.0` for Docker/K8s) |
| `PORT` | `8003` | Server port |

For Docker/Kubernetes deployments, the registry automatically configures these variables.

## Support

- **Documentation**: See the MCP Gateway Registry docs
- **Issues**: Report issues in the main registry repository
- **Configuration help**: Use the gear icon on the server card for setup guidance

---

**Server Status**: Auto-registered and ready to use
**Protocol**: Model Context Protocol (MCP)
**Transport**: Streamable HTTP
**Authentication**: Bearer token (forwarded to registry)
