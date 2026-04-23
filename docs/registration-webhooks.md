# Registration Webhooks

MCP Gateway Registry can send HTTP webhook notifications when servers, agents, or skills are registered (added) or deleted (removed). This enables external systems to react to registry changes in real time, for example updating a CMDB, triggering a CI/CD pipeline, sending a Slack notification, or syncing with a third-party inventory.

## Overview

Registration webhooks are **fire-and-forget**: the registry sends an async POST to a configurable URL after a successful registration or deletion, logs the result, and moves on. A webhook failure never blocks or rolls back the operation that triggered it.

### Supported Events

| Event Type | Trigger | Asset Types |
|------------|---------|-------------|
| `registration` | A new asset is added to the registry | server, agent, skill |
| `deletion` | An existing asset is removed from the registry | server, agent, skill |

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Delivery model | Fire-and-forget | Registry availability is never affected by webhook failures |
| Failure handling | Log at WARNING level | Operators can monitor via CloudWatch or log aggregation |
| Auth header handling | Auto-prefix Bearer for Authorization header | Follows RFC 6750 convention without extra config |
| HTTPS enforcement | Warn but allow HTTP | Avoids breaking dev/test setups while flagging insecure production use |

## Configuration

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `REGISTRATION_WEBHOOK_URL` | string | `""` (disabled) | Full URL to POST to. Only `http://` and `https://` schemes are accepted. Leave empty to disable. |
| `REGISTRATION_WEBHOOK_AUTH_HEADER` | string | `Authorization` | Name of the HTTP header used for authentication. If set to `Authorization`, the token is auto-prefixed with `Bearer `. For any other header (e.g. `X-API-Key`), the token is sent as-is. |
| `REGISTRATION_WEBHOOK_AUTH_TOKEN` | string | `""` | Auth token value. Leave empty for unauthenticated webhooks. |
| `REGISTRATION_WEBHOOK_TIMEOUT_SECONDS` | int | `10` | HTTP timeout per request in seconds. |

### Example Configurations

**Unauthenticated webhook (dev/test):**

```bash
REGISTRATION_WEBHOOK_URL=https://hooks.example.com/registry
REGISTRATION_WEBHOOK_AUTH_HEADER=Authorization
REGISTRATION_WEBHOOK_AUTH_TOKEN=
REGISTRATION_WEBHOOK_TIMEOUT_SECONDS=10
```

**Bearer token authentication:**

```bash
REGISTRATION_WEBHOOK_URL=https://hooks.example.com/registry
REGISTRATION_WEBHOOK_AUTH_HEADER=Authorization
REGISTRATION_WEBHOOK_AUTH_TOKEN=my-secret-bearer-token
REGISTRATION_WEBHOOK_TIMEOUT_SECONDS=10
```

The request will include `Authorization: Bearer my-secret-bearer-token`.

**Custom API key header:**

```bash
REGISTRATION_WEBHOOK_URL=https://hooks.example.com/registry
REGISTRATION_WEBHOOK_AUTH_HEADER=X-API-Key
REGISTRATION_WEBHOOK_AUTH_TOKEN=my-api-key-value
REGISTRATION_WEBHOOK_TIMEOUT_SECONDS=5
```

The request will include `X-API-Key: my-api-key-value`.

## Webhook Payload

Every webhook POST sends a JSON body with the following structure:

```json
{
    "event_type": "registration",
    "registration_type": "agent",
    "timestamp": "2026-04-23T14:30:00.000000+00:00",
    "performed_by": "admin@example.com",
    "card": {
        "name": "My Agent",
        "path": "/agents/my-agent",
        "description": "An example A2A agent",
        "...": "full card data as stored in the registry"
    }
}
```

### Payload Fields

| Field | Type | Description |
|-------|------|-------------|
| `event_type` | string | `"registration"` (asset added) or `"deletion"` (asset removed) |
| `registration_type` | string | `"server"`, `"agent"`, or `"skill"` |
| `timestamp` | string | ISO 8601 timestamp in UTC |
| `performed_by` | string or null | Username of the operator who performed the action (null if unknown) |
| `card` | object | The full card JSON as stored in the registry |

### HTTP Request Details

| Aspect | Value |
|--------|-------|
| Method | `POST` |
| Content-Type | `application/json` |
| Timeout | Configurable via `REGISTRATION_WEBHOOK_TIMEOUT_SECONDS` |
| Retries | None (fire-and-forget) |
| TLS verification | Enabled by default (httpx default behavior) |

## Deployment Configuration

The webhook environment variables must be set on the **registry** service (not the auth server).

### Docker Compose

All three Compose files (`docker-compose.yml`, `docker-compose.podman.yml`, `docker-compose.prebuilt.yml`) pass the variables to the `mcp-gateway-registry` service:

```yaml
services:
  mcp-gateway-registry:
    environment:
      - REGISTRATION_WEBHOOK_URL=${REGISTRATION_WEBHOOK_URL:-}
      - REGISTRATION_WEBHOOK_AUTH_HEADER=${REGISTRATION_WEBHOOK_AUTH_HEADER:-Authorization}
      - REGISTRATION_WEBHOOK_AUTH_TOKEN=${REGISTRATION_WEBHOOK_AUTH_TOKEN:-}
      - REGISTRATION_WEBHOOK_TIMEOUT_SECONDS=${REGISTRATION_WEBHOOK_TIMEOUT_SECONDS:-10}
```

### Terraform / ECS

The variables are defined in `terraform/aws-ecs/variables.tf` and wired into the registry ECS task definition via `terraform/aws-ecs/modules/mcp-gateway/ecs-services.tf` (inside `module "ecs_service_registry"`).

Set values in `terraform.tfvars`:

```hcl
registration_webhook_url             = "https://hooks.example.com/registry"
registration_webhook_auth_header     = "X-API-Key"
registration_webhook_auth_token      = "my-api-key"
registration_webhook_timeout_seconds = 10
```

For sensitive values (tokens), use AWS Secrets Manager references instead of plaintext in tfvars.

### Helm / EKS

The variables are defined in `charts/registry/values.yaml` and mapped in the deployment template and secret:

```yaml
# charts/registry/values.yaml
registrationWebhook:
  url: ""
  authHeader: "Authorization"
  authToken: ""
  timeoutSeconds: 10
```

Sensitive values (auth tokens) are stored in the Kubernetes secret (`charts/registry/templates/secret.yaml`) and injected via `secretKeyRef`.

## Logging and Observability

The webhook service logs at three levels:

| Level | Condition | Example Message |
|-------|-----------|-----------------|
| INFO | Webhook sent successfully | `Registration webhook sent: event=registration, type=agent, status=200, url=https://...` |
| WARNING | Timeout or connection failure | `Registration webhook timed out after 10s: event=registration, type=agent, url=https://...` |
| WARNING | HTTP (not HTTPS) URL configured | `Registration webhook URL uses HTTP (not HTTPS). Credential data may be transmitted insecurely.` |
| ERROR | Invalid URL scheme | `Invalid webhook URL scheme: ftp://...` |

In ECS deployments, these log messages appear in the registry task's CloudWatch Log Group.

## Building a Webhook Receiver

A minimal webhook receiver only needs to accept a POST with a JSON body and return a 2xx status code. Here is a Python example:

```python
from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/webhook")
async def handle_webhook(request: Request):
    payload = await request.json()
    event = payload.get("event_type")
    asset_type = payload.get("registration_type")
    card = payload.get("card", {})
    name = card.get("name") or card.get("display_name", "unknown")

    print(f"Received {event} event for {asset_type}: {name}")

    # Your custom logic here:
    # - Send a Slack notification
    # - Update a CMDB
    # - Trigger a CI/CD pipeline
    # - Sync with an external inventory

    return {"status": "ok"}
```

Run with: `uvicorn receiver:app --host 0.0.0.0 --port 6789`

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| No webhook logs at all | `REGISTRATION_WEBHOOK_URL` is empty or not set | Set the variable in the correct service |
| Webhook env vars set but no calls | Variables on the wrong ECS service | Ensure they are on the **registry** service, not the auth server |
| Timeout warnings | Receiver too slow or unreachable | Increase `REGISTRATION_WEBHOOK_TIMEOUT_SECONDS` or check network connectivity |
| HTTP warning in logs | URL uses `http://` instead of `https://` | Switch to HTTPS for production |

---

## Registration Gate (Coming Soon)

Issue [#809](https://github.com/agentic-community/mcp-gateway-registry/issues/809) adds a **registration gate** (admission control webhook) that is called **before** a registration is persisted. Unlike the notification webhook described above (which fires after the fact and cannot block the operation), the registration gate can **approve or deny** a registration based on custom business logic.

### How It Differs from the Notification Webhook

| Aspect | Notification Webhook (this page) | Registration Gate (#809) |
|--------|----------------------------------|--------------------------|
| Timing | After the registration is persisted | Before the registration is persisted |
| Can block registration | No (fire-and-forget) | Yes (approve/deny) |
| Failure behavior | Logged, never blocks caller | Fail-closed: blocks registration if gate is unavailable |
| Retries | None | Configurable with exponential backoff |
| Applies to | Registration and deletion events | Registration and update events |
| Credential handling | Full card data sent | Credentials stripped from payload |

### Planned Capabilities

- Approve or deny registrations based on naming conventions, compliance rules, or approval workflows
- Applies to all asset types: servers, agents, and skills (both register and update operations)
- Configurable authentication: none, API key, or Bearer token
- Fail-closed design: if the gate is unreachable, registrations are blocked
- Custom denial messages returned to the caller
- Sensitive fields (credentials, tokens, passwords) are never sent to the gate endpoint

### Planned Configuration

| Variable | Description |
|----------|-------------|
| `REGISTRATION_GATE_ENABLED` | Enable/disable the gate (default: false) |
| `REGISTRATION_GATE_URL` | URL of the gate endpoint |
| `REGISTRATION_GATE_AUTH_TYPE` | Auth type: none, api_key, or bearer |
| `REGISTRATION_GATE_AUTH_CREDENTIAL` | API key or Bearer token |
| `REGISTRATION_GATE_AUTH_HEADER_NAME` | Header name for api_key auth (default: X-Api-Key) |
| `REGISTRATION_GATE_TIMEOUT_SECONDS` | HTTP timeout per attempt (default: 5) |
| `REGISTRATION_GATE_MAX_RETRIES` | Max retry attempts (default: 2) |

See [issue #809](https://github.com/agentic-community/mcp-gateway-registry/issues/809) for the full design specification.
