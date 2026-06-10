# Amazon Cognito Setup Guide

This guide walks through configuring Amazon Cognito as the identity provider for the MCP Gateway Registry across all three deployment surfaces.

Cognito is an AWS-managed service, so unlike the bundled PingFederate dev container there is no mode where the registry runs Cognito for you. In every mode you bring your own Cognito User Pool and App Client and supply their values via configuration. The application code is identical across all three modes; only the deployment surface differs.

This guide covers the deployment wiring. For the step-by-step Cognito console walkthrough (creating the User Pool, App Client, groups, and users, including the agent-identity M2M setup), see [docs/cognito.md](../cognito.md).

## Prerequisites

- An AWS account with permission to create and manage a Cognito User Pool.
- A Cognito User Pool with an App Client configured for the OAuth Authorization Code flow. See [Console Configuration](#console-configuration) below; this part is the same regardless of deployment mode.
- The App Client must have a client secret (the registry uses a confidential client).

## Variable reference

The same set of variables is set across all three deployment modes; only the file you edit and the variable name casing differ. Every variable maps 1:1 across surfaces.

| What it is | `.env` (Docker Compose) | Terraform variable | Helm value |
|---|---|---|---|
| Active provider switch | `AUTH_PROVIDER=cognito` | `cognito_enabled = true` (no `auth_provider` variable; see note) | `global.authProvider.type: cognito` |
| Show login button | `COGNITO_ENABLED=true` | `cognito_enabled = true` | `cognito.enabled` is implied by `authProvider.type` |
| User Pool ID | `COGNITO_USER_POOL_ID` | `cognito_user_pool_id` | `cognito.userPoolId` |
| App Client (web login) ID | `COGNITO_CLIENT_ID` | `cognito_client_id` | `cognito.clientId` |
| App Client secret (web login, **secret**) | `COGNITO_CLIENT_SECRET` | `cognito_client_secret` | `cognito.clientSecret` |
| Hosted UI domain (optional) | `COGNITO_DOMAIN` | `cognito_domain` | `cognito.domain` |
| AWS region of the User Pool | `AWS_REGION` | `AWS_REGION` (provider/region, injected automatically) | `cognito.region` |

Note on the provider switch: Docker (`AUTH_PROVIDER`) and Helm (`authProvider.type`) take a provider-name string. The Terraform module has no `auth_provider` variable; you enable exactly one provider by setting its boolean `*_enabled` flag to true (leave the others false), and the module derives the `AUTH_PROVIDER` value for the containers. If you enable none, Keycloak is the default. So for Terraform, `cognito_enabled = true` is the only switch needed.

Note on the domain: `COGNITO_DOMAIN` is optional. When left empty, the auth-server derives the hosted UI domain from the User Pool ID, as `https://<pool-id-without-underscore>.auth.<region>.amazoncognito.com`. Set it only if you configured a custom hosted UI domain prefix.

Note on the region: Cognito needs the AWS region to build the issuer and JWKS URLs. `AWS_REGION` is already injected into the registry and auth-server containers on the Terraform surface, so there is no separate Cognito region variable there.

The "secret" row must be sourced from a secrets store in production (AWS Secrets Manager for Terraform, a Kubernetes Secret for Helm). Don't paste the client secret into `terraform.tfvars` or `values.yaml` checked into git.

For full cross-surface parameter reference, see [docs/unified-parameter-reference.md](../unified-parameter-reference.md).

## Mode 1: Docker Compose (BYO Cognito)

There is no bundled Cognito container. Create the User Pool and App Client in the AWS console first (see [Console Configuration](#console-configuration)), then point the registry at them.

### Step 1: Set Cognito config in `.env`

```bash
# Tell the registry to use Cognito
AUTH_PROVIDER=cognito
COGNITO_ENABLED=true

# User Pool and App Client created in the Cognito console
COGNITO_USER_POOL_ID=us-east-1_XXXXXXXXX
COGNITO_CLIENT_ID=your_cognito_client_id_here
COGNITO_CLIENT_SECRET=your_cognito_client_secret_here

# Region of the User Pool
AWS_REGION=us-east-1

# Optional: only if you set a custom hosted UI domain prefix
# COGNITO_DOMAIN=your-custom-domain
```

### Step 2: Configure the App Client redirect URI

In the Cognito App Client, add the registry's callback URL as an allowed callback URL:

- Local Docker Compose: `http://localhost:8888/oauth2/callback/cognito`
- Production: `https://your-gateway.example.com/oauth2/callback/cognito`

### Step 3: Start the stack

```bash
docker compose up -d
```

Log in to the registry and you will be redirected to the Cognito hosted UI. After login, groups come from the `cognito:groups` claim in the ID token.

## Mode 2: Terraform / AWS ECS (BYO Cognito)

The Terraform module does NOT create the Cognito User Pool. You create the User Pool and App Client (in the console, or via separate Terraform/IaC you manage), and the module wires those values into the registry and auth-server task definitions, with the client secret stored in AWS Secrets Manager.

### Step 1: Create the User Pool and App Client

Follow [Console Configuration](#console-configuration) below. Note the User Pool ID, App Client ID, App Client secret, and the region.

### Step 2: Set the variables in `terraform.tfvars`

Use a separate non-committed `*.auto.tfvars` file (e.g. `secrets.auto.tfvars`) for the secret value, OR populate the AWS Secrets Manager secret value out-of-band via `aws secretsmanager update-secret` after `terraform apply` — the secret resource has `lifecycle { ignore_changes = [secret_string] }` so future plans won't drift.

Note: the Terraform module has no `auth_provider` variable (that name is the Docker `.env` / Helm `values.yaml` switch). On this surface the provider is selected by the boolean flag below; setting `cognito_enabled = true` is what makes the module emit `AUTH_PROVIDER=cognito` to the containers.

```hcl
# terraform.tfvars
cognito_enabled       = true

# User Pool and App Client
cognito_user_pool_id  = "us-east-1_XXXXXXXXX"
cognito_client_id     = "your_cognito_client_id_here"
cognito_client_secret = "<set in secrets.auto.tfvars or via aws secretsmanager>"

# Optional: only if you set a custom hosted UI domain prefix
# cognito_domain      = "your-custom-domain"
```

The User Pool's region is taken from the module's AWS region, which is already injected into both containers as `AWS_REGION`, so there is no separate Cognito region variable.

### Step 3: Apply

```bash
terraform plan
terraform apply
```

One Secrets Manager entry is created (`cognito_client_secret`) and the registry/auth-server tasks read it via `valueFrom` at boot. The plain (non-secret) values are wired as regular environment variables.

### Step 4: Register the callback and sign-out URLs on the App Client (post-deployment)

Two separate URL lists on the App Client must be configured, and Cognito rejects any value not present in the matching list:

- **Allowed callback URLs** — the auth-server sends `redirect_uri = <registry-external-url>/oauth2/callback/cognito` during login.
- **Allowed sign-out URLs** — the auth-server sends `logout_uri = <registry-external-url>/login` during logout. This is easy to miss; if only the callback is registered, login works but logout fails with a Cognito error page (`Required String parameter 'redirect_uri' is not present` or a sign-out URL mismatch).

The registry does NOT configure Cognito for you (it is bring-your-own), so this is a manual step.

When you use a custom domain (`enable_route53_dns = true`), you know the registry URL up front and can register both URLs before deploying. When you use CloudFront-only mode (`enable_cloudfront = true`, no custom domain), the registry domain is the CloudFront distribution domain, which is not known until after `terraform apply` — so this step must happen AFTER the first apply.

1. Get the registry's external URL from the Terraform output:

   ```bash
   terraform output cloudfront_mcp_gateway_url
   # e.g. https://d2xl2zfuhgc4l0.cloudfront.net
   ```

   For a custom-domain deployment it is `https://<your-registry-domain>` instead.

2. On the App Client, set:
   - **Allowed callback URLs:** `<that URL>/oauth2/callback/cognito`
   - **Allowed sign-out URLs:** `<that URL>/login`

   **From the AWS console:**
   - Open the Cognito console, select your User Pool.
   - Go to **App integration** -> **App clients** -> select your app client.
   - Under **Hosted UI** (or **Login pages**), click **Edit**.
   - Add the callback URL to **Allowed callback URLs** and the `/login` URL to **Allowed sign-out URLs** (keep any existing entries, e.g. the localhost ones for local testing), then **Save changes**.

   **From the CLI** (note: `update-user-pool-client` replaces the full lists, so include every URL you want to keep):

   ```bash
   aws cognito-idp update-user-pool-client \
     --user-pool-id "<your-user-pool-id>" \
     --client-id "<your-app-client-id>" \
     --region "<your-region>" \
     --callback-urls \
       "https://<your-registry-domain>/oauth2/callback/cognito" \
       "http://localhost:8888/oauth2/callback/cognito" \
     --logout-urls \
       "https://<your-registry-domain>/login" \
       "http://localhost:8888/login" \
     --allowed-o-auth-flows code \
     --allowed-o-auth-scopes openid email profile aws.cognito.signin.user.admin \
     --allowed-o-auth-flows-user-pool-client \
     --supported-identity-providers COGNITO
   ```

   Updating these URLs is a Cognito-side change only; it takes effect immediately and does NOT require redeploying the stack.

> **CloudFront domain changes on recreate.** The callback and sign-out URLs you register here are fixed strings stored on the App Client; nothing in Terraform manages them. If a later `terraform apply` recreates the CloudFront distribution, its domain changes, the auth-server starts sending the new domain in `redirect_uri` / `logout_uri`, and login or logout fails until you re-register the new URLs (repeat this step). For a stable configuration that never drifts, use a custom domain (`enable_route53_dns = true`) instead of CloudFront-only mode.

## Mode 3: Helm / Kubernetes (BYO Cognito)

The Helm chart does NOT create a Cognito User Pool. You bring your own User Pool and App Client and supply their endpoint and credentials.

### Step 1: Create the User Pool and App Client

Same as the Terraform mode — follow [Console Configuration](#console-configuration) once in the AWS console.

### Step 2: Create the Kubernetes secret (recommended for production)

```bash
kubectl create secret generic cognito-credentials \
  --from-literal=COGNITO_CLIENT_SECRET='<your client secret>'
```

### Step 3: Configure values

```yaml
# values.yaml override
global:
  authProvider:
    type: cognito

cognito:
  userPoolId: "us-east-1_XXXXXXXXX"
  clientId: "your_cognito_client_id_here"
  clientSecret: "your_cognito_client_secret_here"  # or reference an existing secret
  region: "us-east-1"
  # domain: "your-custom-domain"   # optional; derived from userPoolId if unset
```

For the umbrella stack chart (`charts/mcp-gateway-registry-stack`), the same keys live under both `registry:` and `auth-server:` stanzas, since both services read the Cognito configuration.

### Step 4: Install

```bash
helm install mcp-gateway-registry charts/mcp-gateway-registry-stack \
  --namespace mcp-gateway --create-namespace \
  -f values.yaml
```

### Step 5: Register the registry callback URL on the App Client

Add `https://<your-registry-domain>/oauth2/callback/cognito` to the App Client's allowed callback URLs (console or `aws cognito-idp update-user-pool-client`, exactly as in [Mode 2, Step 4](#step-4-register-the-registry-callback-url-on-the-app-client-post-deployment)). On Helm the registry domain is your ingress host, so it is known up front; register it before or right after install. It must match what the auth-server sends (`<registry-external-url>/oauth2/callback/cognito`).

## Console Configuration

These are the one-time steps you must perform in the AWS Cognito console. They are the same across all three deployment modes. For the full walkthrough with screenshots and the agent-identity (M2M) setup, see [docs/cognito.md](../cognito.md).

### 1. Create the User Pool

1. Open the [Amazon Cognito console](https://console.aws.amazon.com/cognito/) and select your region.
2. Create a User Pool with email (and optionally username) as the sign-in identifier.
3. Note the **User Pool ID** (e.g. `us-east-1_XXXXXXXXX`).

### 2. Create the App Client

1. Under the User Pool, create an App Client of type "Traditional Web App" (confidential client).
2. Enable **Authorization code grant** and the `openid`, `email`, and `profile` scopes.
3. Generate a **client secret** and save it into your secret store.
4. Add the registry callback URL as an allowed callback URL:
   - Local Docker Compose: `http://localhost:8888/oauth2/callback/cognito`
   - Production: `https://your-gateway.example.com/oauth2/callback/cognito`
5. Note the **App Client ID** and **client secret**.

### 3. Create groups (not scopes)

You do NOT configure OAuth scopes or a resource server in Cognito for this. The only thing Cognito contributes is group membership; the actual permissions are defined in the registry. See [How groups map to access](#how-groups-map-to-access) below for the full model.

1. In the User Pool, create the **groups** your users will belong to. Use the same names the registry expects, e.g. `registry-admins` (admin) and a non-admin group such as `public-mcp-users`.
2. Assign users to those groups. Cognito places group memberships in the `cognito:groups` claim of the ID token.
3. Make sure a matching group mapping exists in the registry (next section). The group name in Cognito is the contract: it must match a group mapping in the registry exactly.

**From the AWS console:** User Pool -> **Groups** -> **Create group** for each name; then User Pool -> **Users** -> select a user -> **Add to group**.

**From the CLI** (replace `<pool-id>` and `<region>`):

```bash
POOL=<pool-id>
REGION=<region>

# Create the groups
aws cognito-idp create-group --group-name registry-admins \
  --user-pool-id "$POOL" --region "$REGION" \
  --description "Registry admins"
aws cognito-idp create-group --group-name public-mcp-users \
  --user-pool-id "$POOL" --region "$REGION" \
  --description "General read-only users"

# Create an admin user and a test user (email as username; no invite email)
aws cognito-idp admin-create-user --user-pool-id "$POOL" --region "$REGION" \
  --username admin@example.com \
  --user-attributes Name=email,Value=admin@example.com Name=email_verified,Value=true \
  --message-action SUPPRESS
aws cognito-idp admin-create-user --user-pool-id "$POOL" --region "$REGION" \
  --username testuser@example.com \
  --user-attributes Name=email,Value=testuser@example.com Name=email_verified,Value=true \
  --message-action SUPPRESS

# Set permanent passwords so the users can log in immediately
# (requires the cognito-idp:AdminSetUserPassword permission)
aws cognito-idp admin-set-user-password --user-pool-id "$POOL" --region "$REGION" \
  --username admin@example.com --password '<admin-password>' --permanent
aws cognito-idp admin-set-user-password --user-pool-id "$POOL" --region "$REGION" \
  --username testuser@example.com --password '<test-password>' --permanent

# Assign each user to its group
aws cognito-idp admin-add-user-to-group --user-pool-id "$POOL" --region "$REGION" \
  --username admin@example.com --group-name registry-admins
aws cognito-idp admin-add-user-to-group --user-pool-id "$POOL" --region "$REGION" \
  --username testuser@example.com --group-name public-mcp-users
```

> A freshly created user without `admin-set-user-password --permanent` stays in `FORCE_CHANGE_PASSWORD` status and cannot sign in through the hosted UI sign-in form. If your principal lacks `cognito-idp:AdminSetUserPassword`, set the password from the console instead (Users -> select user -> **Actions -> Set password**, mark as Permanent), or have the user complete the forgot-password flow.

### 4. (Optional) Custom hosted UI domain

If you want a friendly hosted UI domain, configure a domain prefix on the User Pool and set `COGNITO_DOMAIN` (or `cognito_domain` / `cognito.domain`). If you skip this, the registry derives the default `amazoncognito.com` domain from the User Pool ID automatically.

## How groups map to access

There is nothing to configure on the Cognito side beyond groups. Cognito does not own scopes or permissions for the registry; it only owns which groups a user belongs to. The split is:

- **Cognito owns** the group-to-user assignment. At login, Cognito emits the user's groups in the `cognito:groups` claim of the ID token.
- **The registry owns** the group-to-scope mapping, stored in **DocumentDB** (managed through the registry's **IAM > Groups / Scopes** UI). There is no scopes file to edit on this path: the auth-server's `map_groups_to_scopes()` queries DocumentDB directly for each group's scopes.

The flow at login:

1. Cognito returns the user's groups in `cognito:groups` (configured as `groups_claim: "cognito:groups"` in `auth_server/oauth2_providers.yml`).
2. The auth-server looks up each group name in DocumentDB and collects the mapped scopes.
3. Those scopes drive what the user can see and do (including agent visibility filtering and MCP server/tool access).

This is the same groups-to-scopes path used by every other IdP (Keycloak, Entra, Okta, Auth0, PingFederate); only the claim name differs (`cognito:groups` instead of `groups`). No Cognito-specific mapping logic is involved.

So to set up, for example, an admin group and a general-user group:

1. In Cognito, create groups `registry-admins` and `public-mcp-users`, and assign users (see the CLI/console commands in [Create groups](#3-create-groups-not-scopes) above).
2. In the registry's IAM > Groups / Scopes UI (backed by DocumentDB), make sure a group mapping exists for each of those exact names with the scopes you want. `registry-admins` maps to the admin scope; `public-mcp-users` (or any name you choose) is whatever read-only/limited scope set you define.
3. Nothing else is configured in Cognito: no resource server, no custom OAuth scopes, no scope-to-group rules.

If a Cognito group name has no matching mapping in DocumentDB, that group simply contributes no scopes (the user may end up with no access). The group names must match exactly.

For agents that authenticate with their own identity (M2M / client credentials), see the agent-identity section in [docs/cognito.md](../cognito.md).

## Seeding the group-to-scope mappings in DocumentDB

The registry reads group-to-scope mappings from DocumentDB at validation time; it does NOT auto-seed them from `scopes.yml` on startup. So a fresh DocumentDB has no mappings, with one exception: the deployment's database-init step seeds a single bootstrap admin group named **`registry-admins`** (mapped to full admin: all servers, all tools, all agent actions). This is what lets the first admin log in and then manage everything else from the UI.

How the bootstrap admin group gets seeded per surface:

- **Terraform / ECS:** run the DocumentDB init task once after the stack is up. It runs `init-documentdb-indexes.py` inside an ECS task (which has VPC access to the cluster) and loads the default `registry-admins` admin scope:

  ```bash
  ./terraform/aws-ecs/scripts/run-documentdb-init.sh
  ```

- **Helm:** the `mongodb-configure` Job seeds `registry-admins` automatically on install.
- **Docker Compose:** the database-init step seeds `registry-admins` as part of `build_and_run.sh`.

This is why the admin group in the examples above is named `registry-admins`: it matches the seeded DocumentDB mapping. Point your Cognito admin group at that exact name and the first admin gets full access with no extra DocumentDB work.

Every OTHER group (for example `public-mcp-users`) is NOT seeded. After the admin logs in, create those mappings through the registry's **IAM > Groups / Scopes** UI, using the same names as the Cognito groups. There is no separate init script to write — the bootstrap task plus the UI cover it. The model is: seed `registry-admins` via the init step, then create all other groups in the UI.

> **Cognito IAM management in the UI is read-only.** The registry's IAM > Groups / Users pages can *list* Cognito groups and users, but cannot create or delete Cognito groups/users from the UI yet (the registry task role is granted only `cognito-idp:ListGroups`, `ListUsers`, and `AdminListGroupsForUser`). Create the Cognito groups and assign users via the AWS console or `aws cognito-idp ...` (see [Create groups](#3-create-groups-not-scopes)). The registry IAM UI is still where you define each group's **scopes/permissions** in DocumentDB — that part works normally. In short: define the group's membership in Cognito, define the group's scopes in the registry UI.

## Troubleshooting

### Redirect URI mismatch

**Symptom:** Cognito returns `redirect_mismatch` or an error page after login.

**Fix:** The callback URL registered on the App Client must exactly match `<protocol>://<auth-server-host>/oauth2/callback/cognito`. Confirm the protocol (http vs https), the host, and the path. On Terraform/ECS and Helm the host is your registry domain; on local Docker Compose it is `http://localhost:8888`.

A common cause on Terraform/ECS CloudFront-only deployments: a `terraform apply` recreated the CloudFront distribution, so the registry domain changed but the App Client's callback URL still points at the old domain. Re-run `terraform output cloudfront_mcp_gateway_url` and re-register the new callback URL on the App Client (see [Mode 2, Step 4](#step-4-register-the-callback-and-sign-out-urls-on-the-app-client-post-deployment)).

### Logout fails with a Cognito error page

**Symptom:** Login works, but clicking logout lands on a Cognito error page (e.g. `Required String parameter 'redirect_uri' is not present`), with a URL like `.../error?...&logout_uri=https%3A%2F%2F<domain>%2Flogin`.

**Fix:** The App Client's **Allowed sign-out URLs** must include `<registry-external-url>/login` — this is a separate list from the callback URLs, and is easy to miss. The auth-server sends `logout_uri = <registry-external-url>/login` on logout; Cognito rejects it if it is not registered. Add it (see [Mode 2, Step 4](#step-4-register-the-callback-and-sign-out-urls-on-the-app-client-post-deployment)). The `logout_uri` echoed in the error page URL is the value Cognito rejected, not a separate problem.

### Empty groups after login

**Symptom:** The user logs in but has no permissions.

**Fix:** Confirm the user is assigned to at least one Cognito group, and that a group mapping with that exact name exists in the registry's DocumentDB (IAM > Groups / Scopes UI). Cognito only emits `cognito:groups` for groups the user actually belongs to, and a group with no DocumentDB mapping contributes no scopes.

If even your admin user has no permissions, the bootstrap admin mapping was likely never seeded into DocumentDB. Run the database-init step for your surface (see [Seeding the group-to-scope mappings in DocumentDB](#seeding-the-group-to-scope-mappings-in-documentdb)) — on Terraform/ECS that is `./terraform/aws-ecs/scripts/run-documentdb-init.sh` — and make sure your Cognito admin group is named `registry-admins` to match the seeded mapping.

### "Unable to list IAM groups" on the IAM pages

**Symptom:** Login works, but the **Settings -> IAM -> Groups** (or Users) page shows `Unable to list IAM groups` (HTTP 502).

**Fix:** This is a server-side failure listing groups from Cognito. Check two things:

1. **Registry version** must include the Cognito IAM manager. Older builds had no Cognito case in the IAM manager factory and silently fell back to the Keycloak manager, which fails against a Cognito deployment. Confirm the registry image is recent enough to support `AUTH_PROVIDER=cognito` for IAM listing.
2. **Task role permissions.** The registry task role must allow `cognito-idp:ListGroups`, `cognito-idp:ListUsers`, and `cognito-idp:AdminListGroupsForUser` on the User Pool. On Terraform/ECS this is wired automatically when `cognito_enabled = true`; if you deployed before that wiring existed, re-apply so the `cognito_iam_read` policy is attached. Check the registry logs for an `AccessDenied` from `cognito-idp` to confirm.

### Wrong region / token validation failure

**Symptom:** Auth-server logs show JWKS fetch failures or issuer mismatch.

**Fix:** `AWS_REGION` must match the region of the User Pool. The issuer and JWKS URLs are built from the region and User Pool ID; a region mismatch makes token validation fail.

### Hosted UI domain not found

**Symptom:** The login redirect lands on a non-existent `amazoncognito.com` domain.

**Fix:** Either configure a hosted UI domain on the User Pool and set `COGNITO_DOMAIN`, or leave `COGNITO_DOMAIN` empty so the registry derives the default domain from the User Pool ID. A custom domain prefix that was never created in Cognito will 404.
