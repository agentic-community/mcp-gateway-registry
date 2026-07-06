# MCP Gateway Registry API Documentation

This document provides a comprehensive overview of all API endpoints available in the MCP Gateway Registry service.

## Table of Contents

- [Authentication](#authentication)
  - [OAuth2 Login](#oauth2-login)
  - [Logout](#logout)
- [Server Management](#server-management)
  - [Register a New Service](#register-a-new-service)
  - [Toggle Service Status](#toggle-service-status)
  - [Edit Service Details](#edit-service-details)
- [API Endpoints](#api-endpoints)
  - [Get Server Details](#get-server-details)
  - [Get Service Tools](#get-service-tools)
  - [Refresh Service](#refresh-service)
- [WebSocket Endpoints](#websocket-endpoints)
  - [Health Status Updates](#health-status-updates)

## Authentication

> **IMPORTANT**: Most endpoints in this API require authentication via OAuth2 (Keycloak). Users authenticate through the browser-based OAuth2 flow, which sets a session cookie. The examples below use `-b cookies.txt` to include the session cookie in requests. For programmatic API access, use a JWT Bearer token obtained from your OAuth2 provider (`-H "Authorization: Bearer <token>"`). A static API token is also accepted via the same `Authorization: Bearer <token>` header when `REGISTRY_STATIC_TOKEN_AUTH_ENABLED=true`.

### OAuth2 Login

Authentication is handled via OAuth2 providers (Keycloak). Navigate to `/login` in your
browser to open the login page; selecting a provider kicks off the OAuth2 flow via
`/oauth2/login/{provider}`. After successful authentication a session cookie is set.

**URL:** `/login`
**Method:** `GET`
**Response:** Login page (served by the web UI) with OAuth2 provider buttons

Supporting API routes the login page uses:

**URL:** `/api/auth/providers`
**Method:** `GET`
**Description:** Returns the enabled OAuth2 providers shown on the login page.

**URL:** `/api/auth/me`
**Method:** `GET`
**Description:** Returns the authenticated user's identity, scopes, groups, and permissions.

### Logout

Logs out the current user by invalidating their session.

**URL:** `/api/auth/logout`
**Method:** `POST` (also `GET`)
**Authentication:** Required (session cookie)
**Response:** `303` redirect; clears the session cookie. (The web UI exposes this as `/logout`.)

**Example:**

```bash
curl -X POST http://localhost/api/auth/logout \
  -b cookies.txt
```

## Server Management

> **Note**: All endpoints in this section require authentication via a session cookie obtained from the OAuth2 login flow.

### Register a New Service

Registers a new MCP service with the gateway.

**URL:** `/api/servers/register`  
**Method:** `POST`  
**Content-Type:** `application/x-www-form-urlencoded`  
**Authentication:** Required (session cookie)  
**Parameters:**
- `name` (required): Display name of the service
- `description` (required): Description of the service
- `path` (required): URL path for the service
- `proxy_pass_url` (required): URL to proxy requests to
- `tags` (optional): Comma-separated list of tags
- `num_tools` (optional): Number of tools provided by the service
- `license` (optional): License information
- `metadata` (optional): JSON object with custom metadata for organization, compliance, and integration tracking. Fully searchable via semantic search.

**Metadata Examples:**
```json
{
  "team": "data-platform",
  "owner": "alice@example.com",
  "compliance_level": "PCI-DSS",
  "cost_center": "engineering",
  "deployment_region": "us-east-1"
}
```

**Response:**
- Success: JSON response with status code 201
- Failure: JSON response with error details

**Example:**

```bash
# Uses the session cookie from the login request
curl -X POST http://localhost/api/servers/register \
  -b cookies.txt \
  -d "name=Weather Service&description=Provides weather forecasts&path=/weather&proxy_pass_url=http://localhost:8000&tags=weather,forecast&num_tools=3&license=MIT"
```

### Toggle Service Status

Enables or disables a registered service.

**URL:** `/api/servers/toggle`
**Method:** `POST`
**Content-Type:** `application/x-www-form-urlencoded` (form fields)
**Authentication:** Required (session cookie or Bearer token)
**Form Parameters:**
- `path` (required): Path of the service to toggle (in the body, not the URL)
- `new_state` (required): `true` to enable, `false` to disable

**Response:** JSON with the updated service status

**Example:**

```bash
# Enable a service
curl -X POST http://localhost/api/servers/toggle \
  -b cookies.txt \
  -F "path=/weather" \
  -F "new_state=true"

# Disable a service
curl -X POST http://localhost/api/servers/toggle \
  -b cookies.txt \
  -F "path=/weather" \
  -F "new_state=false"
```

### Edit Service Details

Updates the details of an existing service.

**URL:** `/api/edit/{service_path}`  
**Method:** `POST`  
**Content-Type:** `application/x-www-form-urlencoded`  
**Authentication:** Required (session cookie)  
**URL Parameters:**
- `service_path`: Path of the service to edit
**Form Parameters:**
- `name` (required): Display name of the service
- `proxy_pass_url` (required): URL to proxy requests to
- `description` (optional): Description of the service
- `tags` (optional): Comma-separated list of tags
- `num_tools` (optional): Number of tools provided by the service
- `license` (optional): License information

**Response:** Redirects to the main page on success

**Example:**

```bash
# Requires session cookie from login
curl -X POST http://localhost/api/edit/weather \
  -b cookies.txt \
  -d "name=Weather API&description=Updated weather service&proxy_pass_url=http://localhost:8001&tags=weather,api&num_tools=5&license=MIT"
```

## API Endpoints

> **Note**: All endpoints in this section require authentication via a session cookie obtained from the OAuth2 login flow.

### Get Server Details

Retrieves detailed information about registered services. Use `GET /api/servers` to list all servers, or `GET /api/servers/{path}` to fetch a single server.

**URL:** `/api/servers`  
**Method:** `GET`  
**Authentication:** Required (session cookie)  
**Description:** Lists all registered services.

**URL:** `/api/servers/{path}`  
**Method:** `GET`  
**Authentication:** Required (session cookie)  
**URL Parameters:**
- `path`: Path of the service to get details for

**Response:** JSON with server details

**Example:**

```bash
# Get details for a specific service (requires session cookie)
curl -X GET http://localhost/api/servers/weather \
  -b cookies.txt

# List all services (requires session cookie)
curl -X GET http://localhost/api/servers \
  -b cookies.txt
```

### Get Service Tools

Retrieves the list of tools provided by a service.

**URL:** `/api/tools/{service_path}`  
**Method:** `GET`  
**Authentication:** Required (session cookie)  
**URL Parameters:**
- `service_path`: Path of the service to get tools for, or "all" to get tools from all services

**Response:** JSON with tool details

**Example:**

```bash
# Get tools for a specific service (requires session cookie)
curl -X GET http://localhost/api/tools/weather \
  -b cookies.txt

# Get tools from all services (requires session cookie)
curl -X GET http://localhost/api/tools/all \
  -b cookies.txt
```

### Refresh Service

Manually triggers a health check and tool discovery (rescan) for a service.

**URL:** `/api/servers/{path}/rescan`  
**Method:** `POST`  
**Authentication:** Required (session cookie)  
**URL Parameters:**
- `path`: Path of the service to rescan

**Response:** JSON with updated service status

**Example:**

```bash
# Requires session cookie from login
curl -X POST http://localhost/api/servers/weather/rescan \
  -b cookies.txt
```

## WebSocket Endpoints

### Health Status Updates

Provides real-time updates on the health status of all registered services.

**URL:** `/api/health/ws/health_status`  
**Protocol:** WebSocket  
**Authentication:** Not required (public endpoint)  
**Response:** JSON messages with health status updates

**Example using websocat:**

First, install websocat:

```bash
sudo wget -qO /usr/local/bin/websocat https://github.com/vi/websocat/releases/latest/download/websocat.x86_64-unknown-linux-musl
sudo chmod +x /usr/local/bin/websocat
```

Then connect to the WebSocket endpoint:

```bash
websocat ws://localhost/api/health/ws/health_status
```

This will display the JSON messages with health status updates in real-time in your terminal.

**Example using Python:**

```python
# Python example using websockets library
import asyncio
import json
import websockets

async def health_status_monitor():
    uri = "ws://localhost/api/health/ws/health_status"
    async with websockets.connect(uri) as websocket:
        print("WebSocket connection established")
        
        while True:
            try:
                # Receive health status updates
                message = await websocket.recv()
                data = json.loads(message)
                
                print("Health status update received:")
                for path, info in data.items():
                    print(f"Service {path}: {info['status']}")
                    print(f"Last checked: {info['last_checked_iso']}")
                    print(f"Number of tools: {info['num_tools']}")
                    print("---")
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed")
                break

# Run the async function
asyncio.run(health_status_monitor())
```

## Authentication Flow

1. **Login**: Navigate to `/login` in your browser and authenticate via your OAuth2 provider (Keycloak). The session cookie is set automatically after successful authentication.

2. **Programmatic Access**: For API access, obtain a JWT Bearer token from your OAuth2 provider (or use a static API token when `REGISTRY_STATIC_TOKEN_AUTH_ENABLED=true`) and include it in the `Authorization` header:
   ```bash
   curl -X GET http://localhost/api/servers \
     -H "Authorization: Bearer <token>"
   ```

3. **Session Expiration**: The session cookie is valid for 8 hours. After expiration, you'll need to login again.

## API Summary

* `GET /login`: Login page (web UI) with OAuth2 provider options.
* `GET /api/auth/providers`: List enabled OAuth2 providers.
* `GET /api/auth/me`: Authenticated user's identity, scopes, groups, and permissions.
* `POST /api/auth/logout`: Log out user and invalidate session cookie (also accepts `GET`; web UI exposes `/logout`).
* `GET /`: Main dashboard (web UI, requires authentication).
* `POST /api/servers/register`: Register a new service (requires authentication).
* `POST /api/servers/toggle`: Enable/disable a service; form fields `path` + `new_state` (requires authentication).
* `POST /api/edit/{service_path}`: Update service details (requires authentication).
* `GET /api/servers`: List all registered services (requires authentication).
* `GET /api/servers/{path}`: Get full details for a single service (requires authentication).
* `GET /api/tools/{service_path}`: Get the discovered tool list for a service (requires authentication).
* `POST /api/servers/{path}/rescan`: Manually trigger a health check/tool update (requires authentication).
* `WebSocket /api/health/ws/health_status`: Real-time connection for receiving server health status updates.