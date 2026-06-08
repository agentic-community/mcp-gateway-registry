# How do I bulk-register all AgentCore Gateways and Runtimes from my AWS account?

The AgentCore scanner CLI discovers all READY gateways and agent runtimes in your AWS account and registers them in the MCP Gateway Registry in a single command. This FAQ walks through the complete procedure step by step.

## Step 0: IAM Permissions

Before running the scanner, you need AWS credentials with permissions to call the Amazon Bedrock AgentCore control-plane APIs. The CLI uses the standard boto3 credential chain, so any of the usual methods work (environment variables, AWS CLI profile, instance profile).

### Required IAM Policy

Attach this policy to the IAM user or role that will run the CLI:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AgentCoreDiscovery",
      "Effect": "Allow",
      "Action": [
        "bedrock-agentcore:ListGateways",
        "bedrock-agentcore:GetGateway",
        "bedrock-agentcore:ListGatewayTargets",
        "bedrock-agentcore:GetGatewayTarget",
        "bedrock-agentcore:ListAgentRuntimes",
        "bedrock-agentcore:GetAgentRuntime",
        "sts:GetCallerIdentity"
      ],
      "Resource": "*"
    }
  ]
}
```

What each permission does:

| Permission | Purpose |
|---|---|
| `bedrock-agentcore:ListGateways` | Discover all gateways in the account |
| `bedrock-agentcore:GetGateway` | Read gateway details (URL, authorizer config) |
| `bedrock-agentcore:ListGatewayTargets` | List targets behind each gateway |
| `bedrock-agentcore:GetGatewayTarget` | Read target details (URL, type) |
| `bedrock-agentcore:ListAgentRuntimes` | Discover all agent runtimes in the account |
| `bedrock-agentcore:GetAgentRuntime` | Read runtime details (protocol, invocation URL) |
| `sts:GetCallerIdentity` | Verify credentials are valid |

### Configure Credentials

Use one of:

```bash
# Option A: Environment variables
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-east-1

# Option B: Named profile
aws configure --profile agentcore-sync
export AWS_PROFILE=agentcore-sync
export AWS_REGION=us-east-1

# Option C: Instance profile (no config needed if running on EC2/ECS with the right role)
```

## Step 1: Preview with Dry-Run

Before registering anything, run the scanner in dry-run mode to see what it will discover and what auth type each resource uses:

```bash
uv run python -m cli.agentcore sync \
    --registry-url http://localhost \
    --token-file .token \
    --dry-run
```

This scans your AWS account for all READY gateways and runtimes, but does not register them. It produces a summary table like this:

```
================================================================================
AGENTCORE SYNC SUMMARY
================================================================================
MODE: DRY-RUN (no changes made)
Would register: 13

DETAILS:
-----------------------------------------------------------------------------------------------
Type       Name                           Path                      Auth           Status
-----------------------------------------------------------------------------------------------
gateway    customersupport-gw             /customersupport-gw       CUSTOM_JWT     dry_run
gateway    geo-mcp                        /geo-mcp                  CUSTOM_JWT     dry_run
gateway    SRE-Gateway                    /sre-gateway              CUSTOM_JWT     dry_run
gateway    TestGWforLambda                /testgwforlambda          CUSTOM_JWT     dry_run
gateway    weather-time-observability-gat /weather-time-observabili CUSTOM_JWT     dry_run
runtime    weather_time_observability_age /weather-time-observabili IAM            dry_run
runtime    sre_agent_simple               /sre-agent-simple         IAM            dry_run
runtime    sre_agent                      /sre-agent                IAM            dry_run
runtime    simple_strands_cognito_agent   /simple-strands-cognito-a IAM            dry_run
runtime    simple_strands_agent           /simple-strands-agent     IAM            dry_run
runtime    simple_a2a_strands_agent       /simple-a2a-strands-agent IAM            dry_run
runtime    my_simple_agent                /my-simple-agent          IAM            dry_run
runtime    my_custom_sre_agent            /my-custom-sre-agent      IAM            dry_run
===============================================================================================
```

### Understanding the Auth Column

| Auth Type | Meaning | Action Required |
|---|---|---|
| `CUSTOM_JWT` | Gateway expects an OAuth2 JWT token from an external IdP (Cognito, Auth0, Entra, Okta, Keycloak) | You will need to configure a client secret for the token refresher (Step 3) |
| `IAM` | Runtime uses AWS IAM (SigV4) for authentication | No additional configuration needed; the registry proxies with IAM credentials |
| `NONE` | No authentication required | No configuration needed |

The dry-run also reports how many manifest entries it would write. The manifest is used by the token refresher to know which gateways need periodic JWT refresh.

## Step 2: Run the Actual Sync

Once you are satisfied with the dry-run output, remove the `--dry-run` flag to register all discovered resources:

```bash
uv run python -m cli.agentcore sync \
    --registry-url http://localhost \
    --token-file .token
```

This will:
1. Register each gateway as an MCP Server (tagged `#agentcore`, `#gateway`, `#auto-registered`)
2. Register each runtime as an Agent (tagged `#agentcore`, `#runtime`, `#auto-registered`)
3. Write `token_refresh_manifest.json` listing all CUSTOM_JWT gateways that need token refresh

If resources were previously registered, the CLI skips them. Use `--overwrite` to update existing registrations:

```bash
uv run python -m cli.agentcore sync \
    --registry-url http://localhost \
    --token-file .token \
    --overwrite
```

## Step 3: Configure Client Secrets (for CUSTOM_JWT Gateways)

If the dry-run showed any gateways with `CUSTOM_JWT` auth, the token refresher needs a client secret to obtain JWTs on behalf of the registry. Whether you need to configure anything depends on which IdP your gateways use.

### Do I Need to Configure a Secret?

| IdP Vendor | Configuration Required? | How |
|---|---|---|
| **Cognito** | No | Auto-retrieved via AWS API (`describe_user_pool_client`). Just ensure your IAM credentials have `cognito-idp:DescribeUserPoolClient` permission. |
| **Entra** | Yes | Set `ENTRA_CLIENT_SECRET` in your `.env` file |
| **Auth0** | Yes | Set `AUTH0_CLIENT_SECRET` in your `.env` file |
| **Okta** | Yes | Set `OKTA_CLIENT_SECRET` in your `.env` file |
| **Keycloak** | Yes | Set `KEYCLOAK_CLIENT_SECRET` in your `.env` file |

If all your gateways are Cognito (which is the most common case for AgentCore), you can skip the rest of this step and go directly to Step 4.

### What is the Client ID?

Each AgentCore gateway with CUSTOM_JWT auth has an `allowedClients` list configured in its authorizer. These are OAuth2 application client IDs registered in the identity provider. The client ID identifies which application is authorized to call the gateway.

The token refresher performs the OAuth2 `client_credentials` grant (machine-to-machine flow) using:
- The **client_id** from `allowedClients` (discovered automatically by the scanner)
- The **client_secret** you provide via environment variables (or auto-retrieved for Cognito)
- The **token endpoint** (auto-discovered from the gateway's OIDC discovery URL)

### How to Find the Client ID

The scanner writes the client IDs into `token_refresh_manifest.json`. After running sync, inspect the manifest:

```bash
cat token_refresh_manifest.json | python3 -m json.tool
```

Example output (multiple gateways, all using Cognito):

```json
[
    {
        "server_path": "/customersupport-gw",
        "gateway_arn": "arn:aws:bedrock-agentcore:us-east-1:123456789012:gateway/customersupport-gw-abc123def",
        "discovery_url": "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_YourPoolId1/.well-known/openid-configuration",
        "allowed_clients": [
            "yourAlphaNumericClientId1abc"
        ],
        "allowed_audience": [],
        "idp_vendor": "cognito"
    },
    {
        "server_path": "/geo-mcp",
        "gateway_arn": "arn:aws:bedrock-agentcore:us-east-1:123456789012:gateway/geo-mcp-xyz789uvw",
        "discovery_url": "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_YourPoolId2/.well-known/openid-configuration",
        "allowed_clients": [
            "yourAlphaNumericClientId2def"
        ],
        "allowed_audience": [],
        "idp_vendor": "cognito"
    },
    {
        "server_path": "/sre-gateway",
        "gateway_arn": "arn:aws:bedrock-agentcore:us-east-1:123456789012:gateway/sre-gateway-qrs456tuv",
        "discovery_url": "https://cognito-idp.us-west-2.amazonaws.com/us-west-2_YourPoolId3/.well-known/openid-configuration",
        "allowed_clients": [
            "yourAlphaNumericClientId3ghi"
        ],
        "allowed_audience": [],
        "idp_vendor": "cognito"
    },
    {
        "server_path": "/weather-time-observability-gateway",
        "gateway_arn": "arn:aws:bedrock-agentcore:us-east-1:123456789012:gateway/weather-time-observability-gateway-lmn012opq",
        "discovery_url": "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_YourPoolId4/.well-known/openid-configuration",
        "allowed_clients": [
            "yourAlphaNumericClientId4jkl"
        ],
        "allowed_audience": [],
        "idp_vendor": "cognito"
    }
]
```

Each `allowed_clients` entry is an alphanumeric client ID (typically 26 characters for Cognito, a UUID for Entra). This is the identifier the token refresher uses to authenticate with the IdP.

### Configure Secrets (Non-Cognito IdPs)

For Entra, Auth0, Okta, or Keycloak gateways, add the appropriate secret to your `.env` file.

#### Scenario A: All your gateways use the same IdP app registration

If you registered all your AgentCore gateways with the **same** app registration in your IdP (one client ID, one secret), set a single vendor-level variable:

```bash
# All Entra gateways share one app registration -> one secret covers them all
# Get this from: Azure Portal -> App registrations -> your app -> Certificates & secrets
ENTRA_CLIENT_SECRET=your-entra-client-secret-value
```

The token refresher uses this secret for every gateway where `idp_vendor` is `entra`.

Same pattern for other IdPs:

```bash
AUTH0_CLIENT_SECRET=your-auth0-client-secret-value
OKTA_CLIENT_SECRET=your-okta-client-secret-value
KEYCLOAK_CLIENT_SECRET=your-keycloak-client-secret-value
```

#### Scenario B: Your gateways use different app registrations (different secrets)

If each gateway has its **own** app registration in the IdP (each with a unique client ID and its own secret), use the per-client form. The client ID comes from the `allowed_clients` field in `token_refresh_manifest.json`.

Example: you have 3 Entra gateways, each with a different app registration:

```bash
# Gateway /customersupport-gw uses Entra app "cs-app" with client_id = a1b2c3d4-...
OAUTH_CLIENT_SECRET_a1b2c3d4-e5f6-7890-abcd-ef1234567890=secret-for-cs-app

# Gateway /sre-gateway uses Entra app "sre-app" with client_id = f9e8d7c6-...
OAUTH_CLIENT_SECRET_f9e8d7c6-b5a4-3210-fedc-ba0987654321=secret-for-sre-app

# Gateway /geo-mcp uses Entra app "geo-app" with client_id = 11223344-...
OAUTH_CLIENT_SECRET_11223344-5566-7788-99aa-bbccddeeff00=secret-for-geo-app
```

#### Scenario C: Mix of both

You can combine the two approaches. The per-client secret takes priority over the vendor-level secret:

```bash
# Default secret for all Entra gateways (most gateways use the same app)
ENTRA_CLIENT_SECRET=shared-entra-secret

# Override for one specific gateway that has its own app registration
OAUTH_CLIENT_SECRET_a1b2c3d4-e5f6-7890-abcd-ef1234567890=special-secret-for-this-one
```

### Secret Resolution Priority

The token refresher resolves secrets in this order (first match wins):

1. **Per-client**: `OAUTH_CLIENT_SECRET_<client_id>` (specific to one gateway)
2. **Cognito auto-retrieval**: via AWS API, no config needed (Cognito only)
3. **Vendor-level**: `ENTRA_CLIENT_SECRET`, `AUTH0_CLIENT_SECRET`, `OKTA_CLIENT_SECRET`, or `KEYCLOAK_CLIENT_SECRET` (shared across all gateways for that IdP)

## Step 4: Run the Token Refresher

The token refresher reads the manifest, resolves secrets, fetches OAuth2 tokens, and PATCHes them into the registry:

```bash
# One-time refresh
uv run python -m cli.agentcore.token_refresher \
    --manifest token_refresh_manifest.json \
    --registry-url http://localhost \
    --token-file .token

# Continuous mode (refreshes every 45 minutes)
uv run python -m cli.agentcore.token_refresher \
    --manifest token_refresh_manifest.json \
    --registry-url http://localhost \
    --token-file .token \
    --loop --interval 2700
```

After the token refresher runs, the gateways will have valid egress tokens. However, they are still **disabled** by default.

## Step 5: Enable the Registered Servers and Agents

Newly registered assets start in a disabled state. You need to enable each one before it appears in search results and health checks begin.

### Enable all CUSTOM_JWT gateways from the manifest

Use the `token_refresh_manifest.json` to loop through all gateway paths and enable them in one shot:

```bash
TOKEN=$(cat .token | python3 -c "import sys,json; print(json.load(sys.stdin)['tokens']['access_token'])")

for path in $(python3 -c "import json; [print(e['server_path']) for e in json.load(open('token_refresh_manifest.json'))]"); do
    echo "Enabling: $path"
    curl -s -X POST \
        -H "Authorization: Bearer $TOKEN" \
        -F "path=$path" \
        -F "new_state=true" \
        "http://localhost/api/servers/toggle"
    echo
done
```

Expected output:

```
Enabling: /customersupport-gw
{"message":"Toggle request for /customersupport-gw processed.","service_path":"/customersupport-gw","new_enabled_state":true,"status":"healthy",...}
Enabling: /geo-mcp
{"message":"Toggle request for /geo-mcp processed.","service_path":"/geo-mcp","new_enabled_state":true,"status":"healthy",...}
...
```

### Enable individual servers or agents

```bash
# Enable a single server
curl -s -X POST \
    -H "Authorization: Bearer $TOKEN" \
    -F "path=/geo-mcp" \
    -F "new_state=true" \
    "http://localhost/api/servers/toggle"

# Enable a single agent
curl -s -X POST \
    -H "Authorization: Bearer $TOKEN" \
    -F "path=/my-simple-agent" \
    -F "new_state=true" \
    "http://localhost/api/agents/toggle"
```

You can also enable them from the registry UI by clicking the toggle switch on each asset's card.

Once enabled, the registry begins health-checking the assets. After a few seconds they should transition to "Healthy" (assuming the token refresher has already run for CUSTOM_JWT gateways).

## Step 6: Verify Registration

Check that the servers and agents are registered and healthy:

```bash
# List registered servers
uv run python api/registry_management.py \
    --registry-url http://localhost \
    --token-file .token \
    list

# Check a specific server
uv run python api/registry_management.py \
    --registry-url http://localhost \
    --token-file .token \
    server-get --path /geo-mcp
```

Or verify in the registry UI: registered assets will show with `#agentcore`, `#gateway` (or `#runtime`), and `#auto-registered` tags.

## Deregistering All Auto-Registered Assets

To remove all previously auto-registered assets (e.g., before a clean re-registration):

```bash
# Remove a server
uv run python api/registry_management.py \
    --registry-url http://localhost \
    --token-file .token \
    remove --path /geo-mcp --force

# Remove an agent
uv run python api/registry_management.py \
    --registry-url http://localhost \
    --token-file .token \
    agent-delete --path /my-simple-agent --force
```

Replace the paths with your actual registered paths. You can find all auto-registered assets by filtering on the `#auto-registered` tag in the UI or API.

## Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| `AccessDeniedException` during discovery | Missing IAM permissions | Attach the IAM policy from Step 0 |
| "Already registered - skipping" | Resource already exists | Use `--overwrite` flag |
| Token refresher returns HTTP 500 | Registry auth token expired | Regenerate with `python credentials-provider/oauth/ingress_oauth.py` |
| Gateway shows unhealthy after sync | Missing egress token | Run the token refresher (Step 4) |
| No CUSTOM_JWT entries in manifest | All gateways use IAM or NONE auth | No token refresh needed |

## Related Documentation

- [AgentCore Full Reference](../agentcore.md)
- [Auto-Registration Prerequisites](../agentcore-auto-registration-prerequisites.md)
