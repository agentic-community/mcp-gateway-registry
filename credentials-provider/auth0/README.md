# Auth0 Credentials Provider

Get M2M (Machine-to-Machine) JWT tokens from Auth0 using OAuth2 client credentials flow.

## Prerequisites

1. **Auth0 M2M Application**: Create a Machine-to-Machine application in Auth0
2. **Management API Authorization**: Authorize the M2M app for Auth0 Management API
3. **Required Scopes**: Grant appropriate scopes (e.g., `read:users`, `create:users`, `read:roles`, etc.)

## Installation

No additional dependencies required beyond the main project dependencies (`requests`, `PyJWT`).

## Usage

### Method 1: Environment Variables (Recommended)

```bash
export AUTH0_DOMAIN=dev-abc123.us.auth0.com
export AUTH0_M2M_CLIENT_ID=your_m2m_client_id
export AUTH0_M2M_CLIENT_SECRET=your_m2m_client_secret

uv run python -m credentials-provider.auth0.get_m2m_token
```

### Method 2: Command-Line Arguments

```bash
uv run python -m credentials-provider.auth0.get_m2m_token \
    --auth0-domain dev-abc123.us.auth0.com \
    --client-id your_m2m_client_id \
    --client-secret your_m2m_client_secret
```

### Custom API Audience

By default, the script requests tokens for the Auth0 Management API (`https://{domain}/api/v2/`). To request tokens for a custom API:

```bash
uv run python -m credentials-provider.auth0.get_m2m_token \
    --audience https://my-custom-api.example.com
```

### Options

- `--auth0-domain`: Auth0 domain (e.g., `dev-abc123.us.auth0.com`)
- `--client-id`: OAuth2 M2M client ID
- `--client-secret`: OAuth2 M2M client secret
- `--audience`: API audience (default: Management API)
- `--show-token`: Display decoded token claims (default: true)
- `--no-show-token`: Skip displaying token claims
- `--debug`: Enable debug logging

## Output

The script:
1. Requests an M2M token from Auth0
2. Displays decoded token claims (unless `--no-show-token` is specified)
3. Saves the token to a temporary file in `/tmp/`
4. Prints the file path to stdout

Example output:

```
Token saved to: /tmp/auth0_m2m_token_abc123.json
```

The token file contains:

```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIs...",
  "token_type": "Bearer",
  "expires_in": 86400
}
```

## Token Lifetime

Auth0 M2M tokens typically have a 24-hour lifetime (86400 seconds). The token expiration is displayed when `--show-token` is enabled.

## Security

- Token files are created with restrictive permissions (0600)
- Tokens are stored in `/tmp/` which is cleared on system restart
- Never commit tokens or credentials to version control
- Use environment variables or secure secret management for credentials

## Troubleshooting

### Error: "Auth0 domain must be provided"

Set the `AUTH0_DOMAIN` environment variable or use the `--auth0-domain` flag.

### Error: "Client ID must be provided"

Set the `AUTH0_M2M_CLIENT_ID` environment variable or use the `--client-id` flag.

### Error: "Client secret must be provided"

Set the `AUTH0_M2M_CLIENT_SECRET` environment variable or use the `--client-secret` flag.

### Error: "M2M token request failed"

Check that:
1. Your M2M application is authorized for the target API
2. The client ID and secret are correct
3. The audience matches your API identifier
4. Network connectivity to Auth0 is available

## Integration with MCP Gateway

The MCP Gateway Registry uses these credentials to manage users and roles via the Auth0 Management API. The credentials are configured in:

- `.env` file: `AUTH0_DOMAIN`, `AUTH0_M2M_CLIENT_ID`, `AUTH0_M2M_CLIENT_SECRET`
- Terraform: `terraform/aws-ecs/variables.tf` and `terraform.tfvars`
- Docker Compose: `docker-compose.yml`
- Helm: `charts/*/values.yaml`

## Related Files

- `registry/utils/auth0_manager.py`: Auth0 Management API integration
- `registry/utils/iam_manager.py`: IAM manager factory including Auth0
