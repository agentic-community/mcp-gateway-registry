---
name: search-registry
description: Search the MCP Gateway Registry using semantic search. Takes a natural language query, calls the /api/search/semantic endpoint, and returns formatted results directly in the conversation.
license: Apache-2.0
metadata:
  author: mcp-gateway-registry
  version: "2.0"
---

# Search Registry Skill

Search the MCP Gateway Registry for MCP servers, tools, A2A agents, skills, and virtual servers using the ai-registry-tools MCP server.

## Prerequisites

The user must have the AI Registry MCP server connected to their AI Assistant session. It may be registered under various names including:

- `ai-registry-tools`
- `airegistry-tools`
- `ai_registry_tools`
- `registry-tools`
- `mcp-registry`

The key indicator is that one of these tools should be available:
- `mcp__ai-registry-tools__search_registry`
- `mcp__airegistry-tools__search_registry`
- or similar variations

## Input

The skill accepts a natural language search query:

```
/search-registry <query>
```

- **query** - What capability, tool, server, or agent the user is looking for (natural language)

If no query is provided, ask the user: "What would you like to search for in the registry?"

### Examples

```
/search-registry tools for weather data
/search-registry exchange rates currency conversion
/search-registry document processing PDF
/search-registry travel booking flights
```

## Workflow

### Step 1: Find the Registry Search Tool

Look for the `search_registry` tool from the ai-registry-tools MCP server. It may be available under different prefixes depending on how the user registered the server. Try these in order:

1. `mcp__ai-registry-tools__search_registry`
2. `mcp__airegistry-tools__search_registry`
3. `mcp__ai_registry_tools__search_registry`
4. `mcp__registry-tools__search_registry`
5. `mcp__mcp-registry__search_registry`

Use ToolSearch to find the correct tool name if none of the above work. Search for "search_registry" or "registry search".

If the tool cannot be found, tell the user:
> "The AI Registry MCP server is not connected to this AI Assistant session. You need to add it first. See the registry documentation for connection instructions specific to your AI Assistant (Claude Code, Codex CLI, etc.)."

### Step 2: Execute the Search

Call the `search_registry` tool with:
- `query`: the user's natural language query
- `max_results`: 10

### Step 3: Present Results

Display the results in this format:

```
### Registry Search Results

**Query:** {query}
**Found:** {count} servers, {count} tools, {count} agents, {count} skills

---
```

For each result category that has results, show a table:

#### MCP Servers

| Name | Path | Score | Tools | Transport | Endpoint |
|------|------|-------|-------|-----------|----------|
| {server_name} | {path} | {relevance_score} | {num_tools} | {supported_transports} | {endpoint_url} |

For servers with matching_tools, list them below:

**{server_name}** matching tools:
| Tool | Description |
|------|-------------|
| {tool_name} | {description truncated to 80 chars} |

#### Tools

| Tool | Server | Score | Endpoint | Description |
|------|--------|-------|----------|-------------|
| {tool_name} | {server_name} | {relevance_score} | {endpoint_url} | {description truncated to 80 chars} |

#### A2A Agents

| Name | Path | Score | URL | Description |
|------|------|-------|-----|-------------|
| {name} | {path} | {relevance_score} | {url} | {description truncated to 80 chars} |

#### Skills

| Name | Path | Score | Description |
|------|------|-------|-------------|
| {skill_name} | {path} | {relevance_score} | {description truncated to 80 chars} |

After the tables, provide a brief plain-text summary of the top 3-5 most relevant results.

### Step 4: Offer to Add MCP Servers

**IMPORTANT**: After presenting results, if any MCP servers were found that have an `endpoint_url`, ask the user:

> "Would you like to add any of these MCP servers to your AI Assistant configuration? If so, which one(s)?"

Present the available servers as options (only those with a valid `endpoint_url`).

### Step 5: Add Server to AI Assistant

If the user says yes and picks a server, determine which AI Assistant is being used and add the server using the appropriate method for that assistant.

#### Detecting the AI Assistant

Check for configuration files to determine the assistant in use:

1. **Claude Code**: Look for `.mcp.json` in the project root or `~/.claude/settings.json`
2. **Codex CLI**: Look for `codex.json` or `.codex/` configuration directory
3. **Other assistants**: Ask the user which assistant they are using if it cannot be determined automatically

#### Adding for Claude Code

**Method 1 (preferred): Use the claude mcp add command**

Run this command (either in a terminal or prefixed with `!` inside a Claude Code session):

```bash
claude mcp add --transport http --scope user <server-name> <endpoint_url>
```

For example:

```bash
claude mcp add --transport http --scope user real-server-fake-tools https://mcpgateway.ddns.net/realserverfaketools/mcp
```

This saves the server to `~/.claude.json` at user scope, making it available across all projects.

**Method 2: Update `~/.claude.json` directly**

Add the new server to the `mcpServers` object in `~/.claude.json` (the user-level config file where `ai-registry-tools` and other MCP Gateway servers are configured):

```json
{
  "mcpServers": {
    "existing-server": { ... },
    "new-server-name": {
      "type": "http",
      "url": "<endpoint_url from the search result>"
    }
  }
}
```

Note: Do NOT add servers to the project-level `.mcp.json` file. That file is committed to git and should not contain user-specific MCP server configurations. Use `~/.claude.json` (user scope) for servers that persist across projects.

#### Adding for Codex CLI

Update the Codex MCP configuration file (typically `~/.codex/mcp.json` or the project-level equivalent):

```json
{
  "servers": {
    "new-server-name": {
      "url": "<endpoint_url from the search result>"
    }
  }
}
```

Or use the Codex CLI command if available:

```bash
codex mcp add <server-name> --url <endpoint_url>
```

#### Adding for Other Assistants

If the AI Assistant cannot be determined or is not one of the above, provide the user with the raw connection details and let them configure it manually:

> "Here are the connection details for the server:
> - **Name**: {server_name}
> - **Endpoint URL**: {endpoint_url}
> - **Transport**: streamable-http
>
> Add this to your AI Assistant's MCP server configuration using the method appropriate for your tool."

#### Server Name Convention

Claude Code only allows server names containing letters, numbers, hyphens, and underscores. Dots and slashes are not allowed.

To derive a valid server name from the server's `path` field:
1. Remove the leading slash
2. Replace slashes with hyphens
3. Remove dots or replace them with hyphens
4. If the result is still unclear, use the `server_name` field as inspiration for a short, descriptive name

Examples:
- Path `/ai.agenticshelf-graffeo` with server_name "ai.agenticshelf/graffeo" (a coffee catalog) becomes `graffeo-coffee`
- Path `/realserverfaketools/` becomes `real-server-fake-tools`
- Path `/ai.com.mcp-openai-tools` becomes `openai-tools`

### Step 6: Inform User About New Session

After successfully adding the server, tell the user:

> "Server '{server_name}' has been added to your AI Assistant's MCP configuration.
>
> **You will need to start a new session to use this server's tools.** I cannot start the new session for you, but you can continue this conversation in that session. The new tools from '{server_name}' will be available once you restart."

## Fallback: Browse the Full Catalog

If the semantic search returns no relevant results (or the user is unsure what to search for), fall back to listing the full catalog using these tools:

- `list_services` - Lists all registered MCP servers (unfiltered). Good for browsing what is available.
- `list_agents` - Lists all registered A2A agents.
- `list_skills` - Lists all registered skills.

Tell the user:
> "The semantic search did not find a close match. Let me list the full catalog so you can browse what is available."

Then call the appropriate listing tool(s) and present the results in a similar table format. If the user finds something of interest from the listing, continue with Step 4 (offer to add it).

Additionally, if a skill is found that looks useful, offer to fetch its full instructions using `get_skill_content` with the skill name.

## Available Registry Tools Reference

The ai-registry-tools MCP server provides these tools:

| Tool | Purpose |
|------|---------|
| `search_registry` | Semantic search across all entity types (primary search tool) |
| `list_services` | List all MCP servers in the registry (unfiltered) |
| `list_agents` | List all A2A agents in the registry (unfiltered) |
| `list_skills` | List all skills in the registry (unfiltered) |
| `get_skill_content` | Fetch full SKILL.md instructions for a skill by name |
| `healthcheck` | Check registry health status and statistics |

## Error Handling

- **Tool not found**: Tell the user the ai-registry-tools MCP server is not connected and how to add it.
- **No results from search**: Tell the user no results were found, then offer to list the full catalog using `list_services`, `list_agents`, or `list_skills` so they can browse what is available.
- **Search error**: Display the error message from the tool response and suggest the user check their registry connection.
- **Server already exists in config**: If the user picks a server that's already in their MCP configuration, tell them it's already configured.

## Notes

- Always search ALL entity types. Do not ask the user to filter by category.
- The `endpoint_url` field on servers is the streamable-http MCP endpoint that can be added directly to any AI Assistant that supports MCP.
- Some servers may have authentication requirements. If the endpoint requires auth headers, note this to the user when offering to add the server. They may need to add headers manually to the config after.
- Do NOT add servers without user confirmation. Always ask first.
