# AgentCore Auto-Registration Prerequisites

This guide covers the setup required before using the AgentCore auto-registration CLI (`python -m cli.agentcore sync`). The prerequisites depend on the **authorizer type** configured on each AgentCore Gateway.

| Authorizer Type | What You Need |
|-----------------|---------------|
| `CUSTOM_JWT` | OAuth2 M2M client credentials from your identity provider (Cognito, Auth0, Okta, etc.) |
| `AWS_IAM` | AWS credentials with appropriate IAM permissions |
| `NONE` | No setup required |

> The auto-registration CLI discovers the authorizer type from each gateway automatically. You only need to prepare credentials for the authorizer types your gateways use.

---

## IAM Permissions for Discovery

Regardless of gateway authorizer type, the CLI needs AWS credentials with permissions to call the Bedrock AgentCore control-plane APIs for resource discovery.

### Required IAM Policy

Attach the following policy to the IAM user or role running the CLI:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AgentCoreDiscovery",
      "Effect": "Allow",
      "Action": [
        "bedrock-agent:ListAgentGateways",
        "bedrock-agent:GetAgentGateway",
        "bedrock-agent:ListAgentRuntimes",
        "bedrock-agent:GetAgentRuntime",
        "bedrock-agent:ListTargets",
        "sts:GetCallerIdentity"
      ],
      "Resource": "*"
    }
  ]
}
```

- `bedrock-agent:ListAgentGateways` / `GetAgentGateway` — discover gateways and their details
- `bedrock-agent:ListAgentRuntimes` / `GetAgentRuntime` — discover runtimes and their protocol configuration
- `bedrock-agent:ListTargets` — enumerate targets behind each gateway
- `sts:GetCallerIdentity` — verify AWS credentials are valid (also used for `AWS_IAM` authorizer verification)

### AWS Credential Setup

The CLI uses the standard boto3 credential chain. Configure credentials using any of these methods:

**Option A: Environment variables**

```bash
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-east-1
```

**Option B: AWS CLI profile**

```bash
aws configure --profile agentcore-sync
export AWS_PROFILE=agentcore-sync
export AWS_REGION=us-east-1
```

**Option C: IAM role (EC2 / ECS / Lambda)**

If running on an AWS compute resource, attach the IAM policy above to the instance role or task role. No explicit credential configuration is needed.

---

## CUSTOM_JWT Authorizer — OAuth2 M2M Client Setup

Gateways with `CUSTOM_JWT` authorizer require OAuth2 machine-to-machine (M2M) client credentials. The CLI uses these credentials to generate egress tokens for authenticating with the gateway.

You need to create an M2M client in your OAuth2 provider and note the **Client ID**, **Client Secret**, and **OAuth2 domain URL**.

### Amazon Cognito

1. Open the [Amazon Cognito console](https://console.aws.amazon.com/cognito/) and select the User Pool associated with your AgentCore Gateway.

2. Navigate to **App integration** → **App clients** and create a new app client:
   - App type: **Confidential client**
   - App client name: e.g., `agentcore-sync-m2m`
   - Generate a client secret: **Yes**
   - Authentication flows: **Client credentials** (`ALLOW_CUSTOM_AUTH` is not needed)

3. Under **Hosted UI**, configure the allowed OAuth scopes for the client. Use the scope defined by your AgentCore Gateway's resource server (e.g., `default-m2m-resource-server-XXXXXXXX/read`).

4. Note the following values:
   - **Client ID**: shown on the app client page
   - **Client Secret**: click "Show client secret"
   - **OAuth2 domain**: `https://<your-domain>.auth.<region>.amazoncognito.com`

5. Set the environment variable:
   ```bash
   export OAUTH_DOMAIN="https://<your-domain>.auth.<region>.amazoncognito.com"
   ```

### Auth0

1. Log in to the [Auth0 Dashboard](https://manage.auth0.com/) and navigate to **Applications** → **Applications**.

2. Click **Create Application**:
   - Name: e.g., `agentcore-sync-m2m`
   - Application type: **Machine to Machine**

3. Authorize the application for the API (audience) that your AgentCore Gateway uses. Select the required scopes.

4. Note the following values from the **Settings** tab:
   - **Client ID**
   - **Client Secret**
   - **Domain**: e.g., `your-tenant.auth0.com`

5. Set the environment variable:
   ```bash
   export OAUTH_DOMAIN="https://your-tenant.auth0.com"
   ```

### Okta

1. Log in to the [Okta Admin Console](https://developer.okta.com/) and navigate to **Applications** → **Applications**.

2. Click **Create App Integration**:
   - Sign-in method: **API Services** (client credentials)
   - App integration name: e.g., `agentcore-sync-m2m`

3. On the app's **General** tab, note:
   - **Client ID**
   - **Client Secret**

4. Under **Okta API Scopes**, grant the scopes required by your AgentCore Gateway.

5. Set the environment variable using your Okta domain:
   ```bash
   export OAUTH_DOMAIN="https://your-org.okta.com"
   ```

### Providing Credentials to the CLI

You can provide OAuth2 credentials in two ways:

**Option A: Environment variables (recommended for CI/CD)**

```bash
# Gateway 1: CUSTOM_JWT (requires OAuth2 credentials)
export AGENTCORE_CLIENT_ID_1="your-client-id"
export AGENTCORE_CLIENT_SECRET_1="your-client-secret"
export AGENTCORE_GATEWAY_ARN_1="arn:aws:bedrock:us-east-1:123456789012:gateway/gw-abc123"
export AGENTCORE_SERVER_NAME_1="my-oauth-gateway"
export AGENTCORE_AUTHORIZER_TYPE_1="CUSTOM_JWT"

# Gateway 2: AWS_IAM (no OAuth2 credentials needed)
export AGENTCORE_GATEWAY_ARN_2="arn:aws:bedrock:us-east-1:123456789012:gateway/gw-def456"
export AGENTCORE_SERVER_NAME_2="my-iam-gateway"
export AGENTCORE_AUTHORIZER_TYPE_2="AWS_IAM"

# Gateway 3: NONE (no credentials needed)
export AGENTCORE_GATEWAY_ARN_3="arn:aws:bedrock:us-east-1:123456789012:gateway/gw-ghi789"
export AGENTCORE_SERVER_NAME_3="my-public-gateway"
export AGENTCORE_AUTHORIZER_TYPE_3="NONE"
```

> The `AGENTCORE_AUTHORIZER_TYPE_{N}` variable is optional — the CLI auto-detects the authorizer type from the gateway. Set it explicitly only if you want to override the detected type.

**Option B: Interactive prompt**

If no environment variables are set, the CLI will prompt for credentials during `sync`:

```
OAuth2 credentials needed for gateway: arn:aws:bedrock:us-east-1:123456789012:gateway/gw-abc123
(Press Enter to skip)
  Client ID: <your-client-id>
  Client Secret: <hidden input>
```

The Client Secret is entered securely (not echoed to the terminal).

---

## AWS_IAM Authorizer

Gateways with `AWS_IAM` authorizer use the standard AWS credential chain for authentication (SigV4 signing). No OAuth2 client setup is needed.

### What You Need

1. AWS credentials configured (see [AWS Credential Setup](#aws-credential-setup) above).
2. The `sts:GetCallerIdentity` permission (included in the discovery policy above).

The CLI verifies your AWS credentials by calling `sts:GetCallerIdentity` during the sync process. If verification succeeds, the gateway is registered without any OAuth2 credential collection or token generation.

---

## NONE Authorizer

Gateways with `NONE` authorizer require **no setup**. The CLI registers these gateways without collecting credentials or generating tokens.

---

## Cross-Account Scanning

To scan AgentCore resources in other AWS accounts, you need an IAM role in each target account that the CLI can assume.

### Target Account Role Setup

In each target account, create an IAM role (default name: `AgentCoreSyncRole`) with:

1. **Trust policy** — allows the caller's account to assume the role:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::CALLER_ACCOUNT_ID:root"
      },
      "Action": "sts:AssumeRole",
      "Condition": {}
    }
  ]
}
```

Replace `CALLER_ACCOUNT_ID` with the AWS account ID where the CLI runs. You can restrict the principal to a specific IAM user or role instead of `root` for tighter security.

2. **Permissions policy** — the same AgentCore discovery policy from [IAM Permissions for Discovery](#iam-permissions-for-discovery):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AgentCoreDiscovery",
      "Effect": "Allow",
      "Action": [
        "bedrock-agent:ListAgentGateways",
        "bedrock-agent:GetAgentGateway",
        "bedrock-agent:ListAgentRuntimes",
        "bedrock-agent:GetAgentRuntime",
        "bedrock-agent:ListTargets",
        "sts:GetCallerIdentity"
      ],
      "Resource": "*"
    }
  ]
}
```

### Caller Account Permissions

The IAM user or role running the CLI also needs permission to assume the role in each target account:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AssumeAgentCoreSyncRole",
      "Effect": "Allow",
      "Action": "sts:AssumeRole",
      "Resource": [
        "arn:aws:iam::111111111111:role/AgentCoreSyncRole",
        "arn:aws:iam::222222222222:role/AgentCoreSyncRole"
      ]
    }
  ]
}
```

Replace the account IDs and role name with your actual values.

### Quick Setup (AWS CLI)

```bash
# In each target account, create the role:
aws iam create-role \
  --role-name AgentCoreSyncRole \
  --assume-role-policy-document file://trust-policy.json

aws iam put-role-policy \
  --role-name AgentCoreSyncRole \
  --policy-name AgentCoreDiscovery \
  --policy-document file://discovery-policy.json
```

---

## Verification Checklist

Before running `python -m cli.agentcore sync`, verify:

- [ ] AWS credentials are configured and can call `sts:GetCallerIdentity`
- [ ] The IAM policy includes all required `bedrock-agent:*` permissions
- [ ] For `CUSTOM_JWT` gateways: OAuth2 M2M client is created and `OAUTH_DOMAIN` is set
- [ ] For `AWS_IAM` gateways: AWS credentials are available in the environment
- [ ] The MCP Gateway Registry is running and accessible at the configured `REGISTRY_URL`
- [ ] A valid registry auth token exists at the configured `--token-file` path (default: `.oauth-tokens/ingress.json`). Generate it with: `python credentials-provider/oauth/ingress_oauth.py`
- [ ] For cross-account scanning: `AgentCoreSyncRole` (or custom role) exists in each target account
- [ ] For cross-account scanning: The caller has `sts:AssumeRole` permission for each target role

## Next Steps

- [Auto-Registration CLI Usage](agentcore.md#auto-registration) — CLI commands, environment variables, and troubleshooting
- [AgentCore Gateway Integration Guide](agentcore.md) — Manual gateway registration walkthrough
