# Virtual MCP Server - How It Works

This document explains how Virtual MCP Servers work using diagrams and examples. For detailed implementation specifics, see [virtual-mcp-server.md](virtual-mcp-server.md).

---

## What Problem Are We Solving?

Consider a typical development setup: you have separate MCP servers for GitHub (code search, PRs), Slack (messaging), and Jira (issue tracking). Your AI agent needs tools from all three, which means:

- Managing three separate connections
- Handling three different sessions
- Dealing with tool name conflicts (both GitHub and Jira have a `search` tool)

A Virtual MCP Server solves this by providing a **single endpoint** that aggregates tools from multiple backends. Your agent connects once and gets access to all the tools it needs.

```
WITHOUT Virtual Server:              WITH Virtual Server:

  You                                  You
   |                                    |
   +---> GitHub Server                  |
   |        |-> search                  v
   |        |-> create_pr         +------------+
   |                              |  Virtual   |
   +---> Slack Server             |  Server    |
   |        |-> send_message      +-----+------+
   |        |-> list_channels           |
   |                              +-----+-----+-----+
   +---> Jira Server              |           |     |
            |-> create_issue      v           v     v
            |-> search_issues   GitHub     Slack   Jira
                                Server     Server  Server
```

**Benefits:**
- Your app only connects to ONE server instead of many
- You can pick exactly which tools you want from each backend
- You can rename tools to avoid confusion (like "github_search" vs "jira_search")
- You can control who has access to which tools

---

## The Big Picture

Request flow when a client connects to a Virtual MCP Server:

```
+----------------+                    +------------------+
|   Your App     |                    |   MCP Gateway    |
|                |                    |                  |
|  "I want to    | ---(1) Request --> |  Nginx receives  |
|   search on    |                    |  your request    |
|   GitHub"      |                    +--------+---------+
|                |                             |
|                |                             v
|                |                    +------------------+
|                |                    |  Lua Router      |
|                |                    |  (the brain)     |
|                |                    |                  |
|                |                    |  "Ah, this tool  |
|                |                    |   belongs to     |
|                |                    |   GitHub backend"|
|                |                    +--------+---------+
|                |                             |
|                |                             v
|                |                    +------------------+
|                |                    |  GitHub Backend  |
|                |                    |                  |
|                | <--(4) Response -- |  (does the       |
|                |                    |   actual work)   |
+----------------+                    +------------------+
```

Each component is described below.

---

## The Three Key Players

### 1. Nginx (Reverse Proxy)

Nginx receives incoming requests and handles:

- JWT authentication via `auth_request` subrequest
- Path-based routing to determine which virtual server
- Invoking the Lua content handler for MCP protocol processing

```
Request arrives at /virtual/dev-tools
                |
                v
        +---------------+
        |    Nginx      |
        |               |
        |  1. Check JWT |  <-- "Is this token valid?"
        |  2. Read path |  <-- "Which virtual server?"
        |  3. Call Lua  |  <-- "Hand off to the router"
        +---------------+
```

### 2. Lua Router (Content Handler)

The Lua router (`virtual_router.lua`) runs as an nginx content handler. It:

- Reads tool-to-backend mappings from JSON config files
- Translates tool aliases back to original names
- Manages session multiplexing across backends
- Authorizes each backing request before dispatch, then aggregates results

### 3. Backend Servers

The actual MCP servers (GitHub, Slack, Jira, etc.) that execute tool calls. The virtual server coordinates requests but delegates all execution to backends.

---

## How Tool Mapping Works

This is the core mechanism. The process works as follows:

### Step 1: Configuration is Created

When someone creates a virtual server, they specify which tools to include:

```
Virtual Server: "dev-tools"
Path: /virtual/dev-tools

Tool Mappings:
  +------------------+------------------+------------------+
  | Tool Name        | Backend Server   | Alias            |
  +------------------+------------------+------------------+
  | search           | /github          | github_search    |
  | search           | /jira            | jira_search      |
  | send_message     | /slack           | (none - use as-is)|
  +------------------+------------------+------------------+
```

Both GitHub and Jira have a tool called "search". Aliases resolve this naming conflict.

### Step 2: Mapping File is Generated

The system writes a JSON file that the Lua router will read:

```
File: /etc/nginx/lua/virtual_mappings/dev-tools.json

{
  "tool_backend_map": {
    "github_search": {
      "original_name": "search",
      "backend_location": "/_vs_backend_github"
    },
    "jira_search": {
      "original_name": "search",
      "backend_location": "/_vs_backend_jira"
    },
    "send_message": {
      "original_name": "send_message",
      "backend_location": "/_vs_backend_slack"
    }
  }
}
```

This file is a lookup table mapping tool names to their backend locations.

### Step 3: Request Comes In

When your app calls a tool:

```
Your app sends:
{
  "method": "tools/call",
  "params": {
    "name": "github_search",      <-- The alias you see
    "arguments": { "query": "bug fixes" }
  }
}
```

### Step 4: Lua Router Translates

The Lua router:
1. Reads the mapping file
2. Looks up "github_search"
3. Finds: backend is `/_vs_backend_github`, original name is `search`
4. Rewrites the request:

```
Forwarded to /_vs_backend_github:
{
  "method": "tools/call",
  "params": {
    "name": "search",             <-- Original name the backend knows
    "arguments": { "query": "bug fixes" }
  }
}
```

### Step 5: Response Goes Back

The GitHub backend responds. The Lua router passes it back to your app unchanged.

---

## The Complete Request Flow

For `tools/call: github_search` on `/virtual/dev-tools/mcp`:

1. Nginx validates the gateway credential and the caller's virtual-server grant.
2. Lua resolves `github_search` to `/github` and rewrites the tool name to `search`. It enforces the virtual server's required scopes and alias override.
3. Lua calls the internal `/_vs_auth_github` location with the **rewritten** JSON-RPC body. Auth-server checks the backing server's scope and original tool name, returning a short-lived token signed for `/github` and its registered upstream.
4. Lua reuses or initializes that backend's stateful or stateless session and calls `/_vs_backend_github` with the signed token. For a PAT or OAuth-backed registration, this location sends the request through `/mcp-proxy/github/`, which vends the calling user's `/github` credential and injects it. For a plain registration, it proxies directly after stripping gateway credentials.
5. The backend result returns through Lua to the client. If either the virtual or backing grant fails, there is no backend call or egress vend.

---

## Session Management (The Tricky Part)

Your app gets one virtual `vs-...` client session. Each backend is initialized lazily when it is first used. A stateful backend returns its own `Mcp-Session-Id`; a stateless Streamable-HTTP backend returns a successful `initialize` without one. Lua remembers either result separately for each owner, backend and pinned version. It does not send a fake session ID to stateless backends.

The nginx shared-dictionary cache keeps this initialization state for 30 seconds. On a miss, Lua reads an owner-bound MongoDB record that survives nginx restarts and expires after one hour of inactivity. Only if both miss does Lua authorize and initialize the backend again. A failed initialize is not cached as a stateless success. Both tool calls and cached tool-list reads still require a fresh backing-server grant.

If the gateway answers a credentialed backend's initialize locally before the user connects, Lua does not cache it as a backend session. An authorized tool call still reaches the credential broker to return the PAT submission instruction or OAuth connect URL. Resource/prompt discovery skips an unavailable, unsupported or access-denied sibling while retaining items from connected backends; the list is an error only when no backend answered.

---

## Listing Tools (Aggregation)

When your app asks "what tools do you have?", the virtual server needs to ask ALL backends:

```
Your app asks: tools/list

Lua Router:
  +-- Ask GitHub: "What tools do you have?"
  |     Response: [search, create_pr, list_repos]
  |
  +-- Ask Slack: "What tools do you have?"
  |     Response: [send_message, list_channels]
  |
  +-- Ask Jira: "What tools do you have?"
        Response: [create_issue, search_issues]

Lua Router combines them:
  [github_search, create_pr, list_repos,     <-- Applied aliases
   send_message, list_channels,
   jira_search, create_issue]                 <-- Renamed "search" to "jira_search"
```

Each backend's `tools/list` request first passes a backing-server scope check. Calls are sequential so a denied backing grant never becomes an upstream request. Plain-backend results can be cached per user with grant rechecks; credentialed-backend discovery is fetched anew because consent and available tools vary by user.

---

## What the Nginx Config Looks Like

When a virtual server is enabled, the registry generates its virtual ingress, a mapping file, and internal authorization/dispatch locations per backing server:

### 1. A Location Block (for routing)

```nginx
location /virtual/dev-tools/ {
    # Tell Lua which virtual server this is
    set $virtual_server_id "dev-tools";

    # Check authentication first
    auth_request /validate;

    # Run the Lua router
    content_by_lua_file /etc/nginx/lua/virtual_router.lua;
}
```

### 2. Internal Backend Locations

```nginx
location = /_vs_auth_github {
    internal;
    set $backend_url "https://github-mcp.example.com/mcp";
    proxy_set_header X-Original-URL $scheme://$host/github/mcp;
    proxy_set_header X-Body $http_x_body;
    proxy_set_header X-Resolved-Upstream $backend_url;
    proxy_pass http://auth-server:8888/validate;
}
location = /_vs_backend_github {
    internal;
    proxy_set_header X-Internal-Token $http_x_internal_token;
    proxy_pass http://auth-server:8888/mcp-proxy/github/;
}
```

The example uses a credentialed GitHub backend; a plain backend dispatches directly after clearing all gateway credentials. Only Lua can call these `internal` locations, and the authorization location must return a signed backend token before Lua dispatches.

---

## Error Handling

What happens when things go wrong?

### Backend is Down

After both grants pass, the backend request can still fail (for example, a network error). Lua returns a backend failure to the client; it does not invent a successful tool result or cache a failed `initialize` as a stateless server.

### Session Expired

```
Lua Router uses cached session sess-gh-001
  |
  v
GitHub returns: "400 Bad Request - Invalid session"
  |
  v
Lua Router:
  1. Delete sess-gh-001 from both caches
  2. Send new "initialize" to GitHub
  3. Get new session: sess-gh-002
  4. Cache it in both levels
  5. Retry the original request with new session
```

### User Lacks Permission

If the virtual alias is not granted, the virtual request is denied. If the alias is granted but the original tool is not granted on the backing server, the explicit backing check denies it before a credential vend or backend call. A backend `initialize` permission does not imply permission to call its tools.

---

## Access Control in Simple Terms

Access requires **both** a virtual-server grant and a backing-server grant. The virtual grant is checked when entering `/virtual/dev-tools`; Lua checks its server-level scopes and any override on the requested alias. The backing grant is checked before every actual backend request, including `initialize`, `tools/list`, and `tools/call`. The backend sees the original tool name, so the backing grant must allow `/github` to call `search`, not just the virtual alias `github_search`.

For example, `mcp-access` plus a `github-write` virtual override is not enough to invoke `create_pr` unless the caller also has an appropriate `/github` backing-server grant. Conversely, a direct `/github` grant does not grant entry to `/virtual/dev-tools`. Discovery is authorized for each backing server; egress discovery is never shared across users or cached while consent changes.

---

## How Changes Are Applied

When you create or update a virtual server:

```
1. You call the API: POST /api/virtual-servers
          |
          v
2. Service validates the configuration
   - Does each backend server exist?
   - Does each tool exist on its backend?
   - Are all alias names unique?
          |
          v
3. Configuration saved to MongoDB
          |
          v
4. Nginx config regenerated
   - New location block written
   - New mapping JSON file written
          |
          v
5. Nginx reloaded
   - nginx -s reload
   - New config takes effect immediately
          |
          v
6. Virtual server is live!
```

---

## Quick Reference

### Files You Should Know

| File | What It Does |
|------|-------------|
| `virtual_router.lua` | The Lua brain that routes requests |
| `nginx_service.py` | Generates nginx config + mapping files |
| `virtual_server_service.py` | Business logic and validation |
| `virtual_server_routes.py` | REST API endpoints |
| `/etc/nginx/lua/virtual_mappings/*.json` | Tool mapping files read by Lua |

### Key Concepts

| Term | Plain English |
|------|--------------|
| Virtual Server | A fake server that coordinates real servers |
| Tool Mapping | "This tool comes from that backend" |
| Alias | A renamed tool to avoid confusion |
| Backend Location | Where to forward requests (internal nginx path) |
| Session Multiplexing | One client session, many backend sessions |
| Scope | A permission string that controls access |

### Common Operations

| What You Want | What Happens |
|---------------|--------------|
| List tools | Checks each backing grant, asks backends sequentially, combines allowed results |
| Call a tool | Looks up backend, translates name, forwards request |
| Initialize | Creates client session, backend sessions are lazy |
| Ping | Responds immediately, no backend calls |

---

## Summary

1. **Virtual servers aggregate tools** from multiple backends into one endpoint
2. **Nginx routes requests** to the Lua router based on path
3. **Lua router reads mapping files** to know which tool goes where
4. **Aliases solve naming conflicts** when two backends have same tool names
5. **Sessions are cached in two levels** for speed and reliability
6. **Both virtual and backing server grants are required** for backend calls
7. **Backing requests are authorized before dispatch**, and egress credentials are per backend and user

The virtual server acts as a coordinator - all tool execution happens on the backend servers. The virtual server's role is to present a unified endpoint to clients.
