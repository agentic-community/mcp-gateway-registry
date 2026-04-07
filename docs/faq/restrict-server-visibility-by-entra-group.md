# How do I restrict which MCP servers a user can see based on their Entra ID group?

The registry has a built-in IAM system that lets you control which servers/tools each group can access. You create the group in Entra ID first, then map it in the registry with the servers that group should have access to.

## Option A: Via the Web UI

1. **Create the group in Entra ID first** (Azure Portal > Groups)
2. In the registry UI, go to **Settings > IAM > Groups**
3. Click **Create Group**
4. **Uncheck "Create group in IdP"** -- since the group already exists in Entra ID
5. Enter the group name (must match the Entra ID group name or Object ID depending on your claims configuration)
6. Under **Server Access**, select which MCP servers, methods, and tools this group should have access to
7. Under **UI Permissions**, configure what actions group members can perform in the dashboard
8. Save the group

## Option B: Via CLI (registry_management.py)

You can import a group definition from a JSON file with the `import-group` command:

```bash
python api/registry_management.py \
  --registry-url https://your-registry-url \
  --token-file .token \
  import-group \
  --file my-group.json
```

Example JSON file (`my-group.json`) -- adapted from [`cli/examples/public-mcp-users.json`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/cli/examples/public-mcp-users.json):
```json
{
  "scope_name": "restricted-mcp-users",
  "description": "Users with access to specific MCP servers only",
  "create_in_idp": false,
  "group_mappings": ["restricted-mcp-users", "your-entra-group-object-id-guid"],
  "server_access": [
    {
      "server": "your-server-1",
      "methods": ["initialize", "notifications/initialized", "ping", "tools/list", "tools/call"],
      "tools": ["*"]
    },
    {
      "server": "/your-server-1",
      "methods": ["initialize", "notifications/initialized", "ping", "tools/list", "tools/call"],
      "tools": ["*"]
    },
    {
      "server": "api",
      "methods": ["initialize", "GET", "POST", "servers", "agents", "search"],
      "tools": []
    }
  ],
  "ui_permissions": {
    "list_service": ["all"],
    "list_agents": [],
    "get_agent": []
  }
}
```

Set `"create_in_idp": false` since the group already exists in Entra ID. The `group_mappings` array should include both the group name and the Entra ID Group Object ID (GUID).

Example scope JSON files are also available in [`scripts/registry-admins.json`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/scripts/registry-admins.json) and [`cli/examples/public-mcp-users.json`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/cli/examples/public-mcp-users.json).

## Related Documentation

- [IAM Settings UI Guide](https://github.com/agentic-community/mcp-gateway-registry/blob/main/docs/iam-settings-ui.md) -- full walkthrough of the Groups UI with server access, tools, and permissions configuration
- [Entra ID Setup Guide](https://github.com/agentic-community/mcp-gateway-registry/blob/main/docs/entra-id-setup.md) -- Steps 5-10 cover configuring group claims in Azure and mapping Entra ID Group Object IDs to registry scopes
- [Scopes Management](https://github.com/agentic-community/mcp-gateway-registry/blob/main/docs/scopes-mgmt.md) -- detailed field reference for scope/group JSON configuration
- [Entra ID Setup - IAM API for Groups](https://github.com/agentic-community/mcp-gateway-registry/blob/main/docs/entra-id-setup.md#using-the-iam-api-to-manage-groups-users-and-m2m-accounts) -- covers `import-group`, `group-create`, `group-delete` commands with full JSON examples
