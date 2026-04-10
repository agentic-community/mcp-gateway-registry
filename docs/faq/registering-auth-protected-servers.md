# How do I register and manage MCP servers that require authentication?

The MCP Gateway Registry fully supports registering MCP servers that are behind access control (Bearer token or API key). When a server requires authentication, the registry stores the credential securely (encrypted at rest) and automatically injects it when performing health checks, tool discovery, and proxying requests.

## Registering a Server with Authentication

Use the `POST /api/servers/register` endpoint with JWT Bearer authentication. Include the `auth_scheme` and `auth_credential` fields to specify how the registry should authenticate with your backend MCP server.

### Bearer Token Authentication

```bash
curl -X POST https://registry.example.com/api/servers/register \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -F "name=My Protected Server" \
  -F "description=An MCP server behind Bearer auth" \
  -F "path=/my-protected-server" \
  -F "proxy_pass_url=http://my-server:8000" \
  -F "auth_scheme=bearer" \
  -F "auth_credential=my-backend-server-token"
```

### API Key Authentication

```bash
curl -X POST https://registry.example.com/api/servers/register \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -F "name=My API Key Server" \
  -F "description=An MCP server behind API key auth" \
  -F "path=/my-apikey-server" \
  -F "proxy_pass_url=http://my-server:8000" \
  -F "auth_scheme=api_key" \
  -F "auth_credential=my-api-key-value" \
  -F "auth_header_name=X-API-Key"
```

### Supported `auth_scheme` Values

| Value | Behavior |
|-------|----------|
| `none` | No authentication (default) |
| `bearer` | Sends `Authorization: Bearer <credential>` header |
| `api_key` | Sends credential in a custom header (default: `X-API-Key`) |

### Custom Header Name

When using `api_key`, you can specify a custom header name via `auth_header_name`. For example, if your server expects `X-My-Custom-Key`, pass `auth_header_name=X-My-Custom-Key`.

## How Tool Discovery Works with Auth

Once registered with credentials, the registry automatically:

1. **Health checks** -- Injects the decrypted credential when checking if the server is reachable
2. **Tool discovery** -- Uses the credential to call the MCP `tools/list` method on the backend server
3. **Request proxying** -- When clients connect through the gateway, the credential is injected into proxied requests

This means tool discovery works the same way for protected servers as it does for public ones -- no additional configuration is needed beyond providing the credential at registration time.

## Manually Providing Tools

If your server is behind a firewall or tool auto-discovery is not possible, you can provide tools manually at registration time using the `tool_list_json` parameter:

```bash
curl -X POST https://registry.example.com/api/servers/register \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -F "name=My Server" \
  -F "description=Server with manually defined tools" \
  -F "path=/my-server" \
  -F "proxy_pass_url=http://my-server:8000" \
  -F 'tool_list_json=[{"name": "get_weather", "description": "Get weather for a city", "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]'
```

The `tool_list_json` field accepts a JSON array of MCP tool definitions. These will be stored in the registry and returned to clients during tool discovery, even if the backend server is unreachable for live tool listing.

## Updating or Rotating Credentials

Use the `PATCH /api/servers/{path}/auth-credential` endpoint to update credentials without re-registering the server:

```bash
# Rotate a Bearer token
curl -X PATCH https://registry.example.com/api/servers/my-protected-server/auth-credential \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "auth_scheme": "bearer",
    "auth_credential": "new-backend-server-token"
  }'
```

```bash
# Switch from Bearer to API key
curl -X PATCH https://registry.example.com/api/servers/my-protected-server/auth-credential \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "auth_scheme": "api_key",
    "auth_credential": "new-api-key",
    "auth_header_name": "X-API-Key"
  }'
```

```bash
# Remove authentication (make server public)
curl -X PATCH https://registry.example.com/api/servers/my-protected-server/auth-credential \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "auth_scheme": "none"
  }'
```

## Credential Security

- Credentials are **encrypted at rest** using the `SECRET_KEY` configured in your deployment
- Credentials are **never returned** in API responses or displayed in the UI
- Credentials are **decrypted only in memory** when needed for health checks, tool discovery, or request proxying

## Getting a JWT Token

To authenticate with the registry API, you need a JWT token. Click the **Get JWT Token** button in the top-left corner of the registry UI, or use the token generation API:

```bash
curl -X POST https://registry.example.com/api/tokens/generate \
  -H "Cookie: session=<your-session-cookie>" \
  -H "Content-Type: application/json" \
  -d '{
    "expires_in_hours": 8,
    "description": "Server registration token"
  }'
```

## Troubleshooting

### Tools not discovered for a protected server

1. Verify the credential is correct by testing it directly against your MCP server
2. Check the server health status in the registry UI -- if unhealthy, the credential may be invalid or expired
3. Use the credential update endpoint to provide a fresh credential
4. As a fallback, provide tools manually via `tool_list_json` at registration time

### "Failed to fetch tools" error

This typically means:
- The backend server is unreachable from the registry
- The credential is invalid or expired
- The server does not implement the standard MCP `tools/list` method

Check the registry logs for detailed error messages about the connection failure.
