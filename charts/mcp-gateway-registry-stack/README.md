# MCP Gateway Registry Stack Charts

This collection of charts deploys everything needed to install the MCP Gateway Registry using Helm or ArgoCD.

## Prerequisites

### Amazon EKS Cluster

For production deployments, we recommend using the [AWS AI/ML on Amazon EKS](https://github.com/awslabs/ai-on-eks)
blueprints to provision an EKS cluster:

```bash
# Clone AI on EKS repository
git clone https://github.com/awslabs/ai-on-eks.git
cd ai-on-eks

cd infra/solutions/agents-on-eks

# Edit the terraform/blueprint.tfvars to set your domain

./install.sh
```

The ai-on-eks blueprints provide:

- GPU support for AI/ML workloads
- Karpenter for efficient auto-scaling
- EKS-optimized configurations
- Security best practices
- Observability with Prometheus/Grafana
- Well-documented infrastructure patterns

### Additional Requirements

- `helm` CLI installed (v3.0+)
- `kubectl` configured to access your EKS cluster
- AWS Load Balancer Controller for EKS
- ExternalDNS (optional, for automatic DNS management)
- Domain name with DNS access
- TLS certificates (AWS Certificate Manager or Let's Encrypt)

## Setup

```
git clone https://github.com/agentic-community/mcp-gateway-registry
cd mcp-gateway-registry/charts/mcp-gateway-registry-stack
```

## Values file

The `values.yaml` file needs to be updated for your setup, specifically:

- `DOMAIN`: there are placeholders for `DOMAIN` that should be updated with your full domain. For example, if you intend
  to use `example.com`, replace `DOMAIN` with `example.com`. If you intend to use a subdomain like
  `subdomain.example.com`, `DOMAIN` should be replaced with `subdomain.example.com`
- `secretKey`: the registry and auth-server both have a placeholder for `secretKey`, this should be updated to the same
  random, secure key that is used in both locations
- `routingMode`: choose between `subdomain` (default) or `path` based routing (see Routing Modes section below)

Choose `keycloak.postgres.bootstrap.mode` before installing the bundled Keycloak.
Use `fresh-install` only for a new database. Existing deployments must follow
[Keycloak database cutover](#keycloak-database-cutover).

### Authentication Provider Selection

This chart supports five authentication providers: Keycloak (default), Microsoft Entra ID, Okta, Auth0, and AWS Cognito.

When another provider is active, the chart skips bundled Keycloak resources, the realm configuration Job, and Keycloak-specific ingress paths, even if their creation switches remain enabled. Path-mode ingress keeps the registry prefix and `/oauth2` on the primary and additional hostnames. Selecting Keycloak with `keycloak.create=false` retains its login routes for an external Keycloak server.

#### Option 1: Keycloak (Default)

**Deploy Keycloak in the stack:**

```yaml
global:
  authProvider:
    type: keycloak

keycloak:
  create: true  # Deploy Keycloak as part of this stack
  postgres:
    bootstrap:
      mode: fresh-install  # Only for an intentionally empty database

keycloak-configure:
  enabled: true  # Run Keycloak configuration job
```

**Use an external Keycloak instance:**

```yaml
global:
  authProvider:
    type: keycloak
    keycloak:
      externalUrl: https://your-keycloak.example.com
      realm: mcp-gateway

keycloak:
  create: false  # Don't deploy Keycloak

keycloak-configure:
  enabled: false  # Keep a preconfigured external realm unchanged
```

For external Keycloak, provide `KEYCLOAK_URL` with its reachable backend URL and
`KEYCLOAK_EXTERNAL_URL` with its canonical browser URL through
`global.existingOauthProviderSecret`. Set `global.authProvider.keycloak.adminExistingSecret`
to its administrator password Secret. To run realm setup, enable `keycloak-configure`
and set `keycloak-configure.keycloak.existingSecret` to a configuration Secret containing
`KEYCLOAK_ADMIN`, `KEYCLOAK_ADMIN_URL`, `KEYCLOAK_URL`, `REALM`,
`AUTH_SERVER_EXTERNAL_URL`, and `REGISTRY_URL`. Also set
`keycloak-configure.keycloak.realm` to that Secret's `REALM` value.

**Optional: Keycloak M2M authentication:**

```yaml
auth-server:
  keycloak:
    m2mClientId: "mcp-gateway-m2m"
    m2mClientSecret: "your-m2m-client-secret"
```

#### Option 2: Microsoft Entra ID

```yaml
global:
  authProvider:
    type: entra
    entra:
      adminGroupId: "your-admin-group-uuid"  # Optional: maps Entra group to admin role

auth-server:
  entra:
    clientId: "your-entra-client-id"
    clientSecret: "your-entra-client-secret"
    tenantId: "your-entra-tenant-id"
    loginBaseUrl: ""  # Optional: override for sovereign clouds (e.g., https://login.microsoftonline.us)
```

See the [Entra ID documentation](../../docs/entra.md) for details on setting up your Entra ID app registration.

#### Option 3: Okta

```yaml
global:
  authProvider:
    type: okta

auth-server:
  okta:
    domain: "dev-123456.okta.com"
    clientId: "your-client-id"
    clientSecret: "your-client-secret"
    m2mClientId: ""       # Optional: for machine-to-machine auth
    m2mClientSecret: ""   # Optional: for machine-to-machine auth
    apiToken: ""          # Optional: for IAM operations
    authServerId: ""      # Optional: custom authorization server
```

#### Option 4: Auth0

```yaml
global:
  authProvider:
    type: auth0

auth-server:
  auth0:
    domain: "your-tenant.us.auth0.com"
    clientId: "your-client-id"
    clientSecret: "your-client-secret"
    audience: ""                              # Optional: API audience for M2M tokens
    groupsClaim: "https://mcp-gateway/groups" # Custom claim for group memberships
    m2mClientId: ""                           # Required for IAM management
    m2mClientSecret: ""                       # Required for IAM management
    managementApiToken: ""                    # Optional: alternative to M2M credentials (expires 24h)
```

#### Option 5: AWS Cognito

```yaml
global:
  authProvider:
    type: cognito

auth-server:
  cognito:
    userPoolId: "us-east-1_xxxxxxxxx"
    clientId: "your-client-id"
    clientSecret: "your-client-secret"
    domain: ""              # Optional: custom Cognito domain
    region: "us-east-1"     # AWS region for the User Pool
```

### Routing Modes

The stack supports two routing modes for accessing services:

#### Subdomain-Based Routing (Default)

Services are accessed via subdomains:

- `keycloak.{domain}` - Keycloak authentication server
- `auth-server.{domain}` - MCP Gateway auth server
- `mcpregistry.{domain}` - MCP server registry

**Configuration:**

```yaml
global:
  domain: "yourdomain.com"
  ingress:
    routingMode: subdomain
    registrySubdomain: mcpregistry
    additionalHostnames: []
```

`registrySubdomain` controls the registry hostname prefix used by the Ingress
and generated public URLs (OAuth callbacks, auth-server, and gateway URLs). It
defaults to `mcpregistry`, preserving the default `mcpregistry.{domain}` hostname.
`additionalHostnames` optionally adds full hostnames to the same registry
Ingress and backend. Each hostname must also have DNS and TLS coverage.
Each alias must be within the effective session-cookie domain (by default,
`.<global.domain>`), or browser login cannot complete on that alias. If using a
custom cookie domain, configure the same parent domain in both
`registry.app.sessionCookieDomain` and `auth-server.app.sessionCookieDomain`.
Unrelated domains cannot share a browser cookie; supporting those requires a
host-only/per-host session design rather than a broader cookie domain.

**DNS Requirements:** Configure A/CNAME records for each subdomain pointing to your ingress load balancer.

#### Path-Based Routing

Services are accessed via paths on a single domain:

- `{domain}/keycloak` - Keycloak authentication server (default, configurable)
- `{domain}/auth-server` - MCP Gateway auth server (default, configurable)
- `{domain}/registry` - MCP server registry (default, configurable)
- `{domain}/` - MCP server registry (root path)

**Configuration:**

```yaml
global:
  domain: "yourdomain.com"
  ingress:
    routingMode: path
    paths:
      registry: /registry   # Customize as needed (e.g., /api)
    additionalHostnames: []
```

Only the registry service exposes a public ingress. Its nginx proxy serves the configured
Keycloak realm and static resources, and blocks master-realm and admin routes.
Keep `keycloak.httpRelativePath=/` for the bundled server. Set a canonical public URL
or prefix with `global.authProvider.keycloak.externalUrl`; the server and gateway clients
use the same URL. HTTP ingress requires `global.ingress.allowPlaintext=true`.

Use a private route for the Keycloak admin console. A private reverse proxy must retain
the canonical hostname and scheme while allowing `/admin` and `/realms/master` only
to administrators. Do not reopen those paths on the public registry ingress.

**DNS Requirements:** Configure a single A/CNAME record for your domain pointing to your ingress load balancer.

### UI Branding (optional)

You can reskin the registry UI (colors, logos, favicon, extra themes, fonts) from a ConfigMap, without rebuilding the image. Put the branding files in a folder, then create the ConfigMap in the release namespace:

```bash
kubectl create configmap acme-branding -n MYNAMESPACE --from-file=./branding/
```

```yaml
registry:
  branding:
    enabled: true
    existingConfigMap: acme-branding
```

The registry mounts the ConfigMap read-only and serves its files at `{registry path}/branding/`. That path sits under the registry's own prefix, so the existing registry ingress covers it in both routing modes and you add no ingress rule. Edits to the ConfigMap reach running pods within a minute or two, with no restart. See [UI branding](../../docs/ui-branding.md) for what goes in the folder.

## Install

Once the `values.yaml` file is updated and saved, run (substitute MYNAMESPACE for the namespace in which this should be
installed):

```bash
helm dependency build && helm dependency update
helm install mcp-gateway-registry -n MYNAMESPACE --create-namespace .
```

This will deploy the necessary resources for a Kubernetes deployment of the MCP Gateway Registry

**Note:** You can add `--set global.chartVersion=$(git rev-parse HEAD)` to your helm install command, which will create
a configmap that has the version of the repository as the value. This can aid in debugging by making it much faster to
identify which version was used to deploy the charts.

## Deploy Process

### With Keycloak:

- postgres, keycloak, registry, and auth-server will be deployed as the core components
- A `keycloak-configure` job will also be created
- Postgres will need to be running first before Keycloak will run
- Keycloak needs to be available before the `keycloak-configure` job will run
- auth-server will not start until the `keycloak-configure` job has succeeded and generated a secret that is needed for
  the auth-server.
- The registry will start as soon as the image is pulled

### With Entra ID, Okta, Auth0, or Cognito:

- MongoDB, registry, and auth-server will be deployed as the core components
- Keycloak and keycloak-configure are skipped
- auth-server will use the configured IdP credentials from your values file
- The registry will start as soon as the image is pulled

## Deployment Examples (all run from charts/mcp-gateway-registry-stack)

### Subdomain with Keycloak

Creates a self-contained deployment. This is the simplest deployment.

```bash
helm install mcp-gateway-registry -n mcp-gateway-registry --create-namespace . \
 --set global.domain=agents.domain.example
```

### Subdomain with Entra and Inbound IP Allowlisting

Creates a deployment using Entra. Please follow the [instructions](../../docs/entra-id-setup.md) to set up Entra.

```bash
helm install mcp-gateway-registry -n mcp-gateway-registry --create-namespace . \
--set global.domain=agents.domain.example \
--set global.ingress.routingMode=subdomain \
--set global.authProvider.type=entra \
--set auth-server.entra.clientId=ENTRA_CLIENT_UUID \
--set auth-server.entra.clientSecret=ENTRA_CLIENT_SECRET \
--set auth-server.entra.tenantId=ENTRA_TENANT_ID  \
--set global.authProvider.entra.adminGroupId=ENTRA_ADMIN_GROUP_UUID \
--set keycloak-configure.enabled=false \
--set keycloak.create=false \
--set global.ingress.inboundCidrs='my.public.ip.address/32'
```

### Subdomain with Okta and Inbound IP Allowlisting

Creates a deployment using Okta.

```bash
helm install mcp-gateway-registry -n mcp-gateway-registry --create-namespace . \
--set global.domain=agents.domain.example \
--set global.ingress.routingMode=subdomain \
--set global.authProvider.type=okta \
--set auth-server.okta.domain=OKTA_DOMAIN \
--set auth-server.okta.clientId=OKTA_CLIENT_ID \
--set auth-server.okta.clientSecret=OKTA_CLIENT_SECRET  \
--set keycloak-configure.enabled=false \
--set keycloak.create=false \
--set global.ingress.inboundCidrs='my.public.ip.address/32'
```


### Subdomain with Auth0

Creates a deployment using Auth0.

```bash
helm install mcp-gateway-registry -n mcp-gateway-registry --create-namespace . \
--set global.domain=agents.domain.example \
--set global.authProvider.type=auth0 \
--set auth-server.auth0.domain=YOUR_TENANT.us.auth0.com \
--set auth-server.auth0.clientId=AUTH0_CLIENT_ID \
--set auth-server.auth0.clientSecret=AUTH0_CLIENT_SECRET \
--set keycloak-configure.enabled=false \
--set keycloak.create=false
```

### Subdomain with AWS Cognito

Creates a deployment using AWS Cognito.

```bash
helm install mcp-gateway-registry -n mcp-gateway-registry --create-namespace . \
--set global.domain=agents.domain.example \
--set global.authProvider.type=cognito \
--set auth-server.cognito.userPoolId=us-east-1_XXXXXXXXX \
--set auth-server.cognito.clientId=COGNITO_CLIENT_ID \
--set auth-server.cognito.clientSecret=COGNITO_CLIENT_SECRET \
--set keycloak-configure.enabled=false \
--set keycloak.create=false
```

### Path with Keycloak and git hash retention for debugging

Will create a configmap in the `mcp-gateway-registry` namespace called `chart-version` with the git hash of the current
repo (if cloned) to aid in debugging.

```bash
helm install mcp-gateway-registry -n mcp-gateway-registry --create-namespace . \
--set global.domain=agents.domain.example \
--set global.ingress.routingMode=path \
--set global.chartVersion=$(git rev-parse --short HEAD)
```

### Federation with Keycloak on path

Will enable registry federation for this deployment. Creates a static token in the `shared-secret` in the
`mcp-gateway-registry` namespace that needs to be shared with the connecting registry. If used with `inboundCidr` allow
listing, the connecting registry IP needs to be part of the allowed CIDR range.

```bash
helm install mcp-gateway-registry -n mcp-gateway-registry --create-namespace . \
 --set global.domain=agents.domain.example \
 --set global.ingress.routingMode=path \
 --set global.federation.staticTokenAuthEnabled=true
```

**Federation with OAuth2 for outbound peer connections:**

```bash
helm install mcp-gateway-registry -n mcp-gateway-registry --create-namespace . \
 --set global.domain=agents.domain.example \
 --set global.federation.staticTokenAuthEnabled=true \
 --set registry.app.federationTokenEndpoint=https://idp.example.com/oauth2/token \
 --set registry.app.federationClientId=federation-client \
 --set registry.app.federationClientSecret=federation-secret
```

### ASOR (Workday) Integration

ASOR integration is independent of peer federation and can be enabled alongside any auth provider:

```bash
helm install mcp-gateway-registry -n mcp-gateway-registry --create-namespace . \
 --set global.domain=agents.domain.example \
 --set registry.app.asorAccessToken=your-asor-access-token \
 --set registry.app.workdayTokenUrl=https://services.wd101.myworkday.com/ccx/oauth2/instance/token
```

### Auth Server Advanced Configuration

**Static token authentication** (use a static API key instead of IdP JWT for Registry API):

```bash
helm install mcp-gateway-registry -n mcp-gateway-registry --create-namespace . \
 --set global.domain=agents.domain.example \
 --set auth-server.app.registryStaticTokenAuthEnabled=true \
 --set auth-server.app.registryApiToken=your-secure-api-token
```

**Custom JWT configuration** (override internal service-to-service token claims):

```yaml
auth-server:
  app:
    jwtIssuer: "custom-issuer"      # Default: mcp-auth-server
    jwtAudience: "custom-audience"  # Default: mcp-registry
    maxTokensPerUserPerHour: "50"   # Default: 100
```

## Use

Navigate to the registry based on your routing mode:

**Subdomain mode:** https://mcpregistry.DOMAIN

**Path mode:** https://DOMAIN/registry or https://DOMAIN/

### With Keycloak

The configuration Job stores the initial login accounts in the `registry-login-credentials` Kubernetes Secret in the release namespace; it does not print passwords in Job logs. Retrieve only the credentials you need from a trusted terminal (replace `MYNAMESPACE` with the release namespace). Run each command separately so you can see lookup errors; these commands do not put passwords in your shell history:

```bash
set -o pipefail
# Initial administrator account
kubectl get secret registry-login-credentials -n MYNAMESPACE -o jsonpath='{.data.REGISTRY_ADMIN_NAME}' | base64 -d
kubectl get secret registry-login-credentials -n MYNAMESPACE -o jsonpath='{.data.REGISTRY_ADMIN_PASSWORD}' | base64 -d

# Initial test user account, if needed
kubectl get secret registry-login-credentials -n MYNAMESPACE -o jsonpath='{.data.REGISTRY_USER_NAME}' | base64 -d
kubectl get secret registry-login-credentials -n MYNAMESPACE -o jsonpath='{.data.REGISTRY_USER_PASSWORD}' | base64 -d
```

The Secret keys are `REGISTRY_ADMIN_NAME`, `REGISTRY_ADMIN_PASSWORD`, `REGISTRY_USER_NAME`, and `REGISTRY_USER_PASSWORD`. The commands display decoded values on your terminal; avoid running them in shared logs, recorded sessions, or CI output. Do not use `kubectl logs` to look for credentials.

### With Entra ID:

Navigate to https://mcpregistry.DOMAIN to log in. Users will authenticate using their Microsoft Entra ID credentials.
Ensure that:

1. Your Entra ID app registration has the correct redirect URIs configured
2. Users are assigned to the appropriate Entra ID groups
3. Group mappings are configured in your scopes.yml or MongoDB

See the [Entra ID documentation](../../docs/entra.md) for complete setup instructions.

### With Okta, Auth0, or Cognito:

Navigate to https://mcpregistry.DOMAIN to log in. Users will authenticate through your configured identity provider.
Ensure that your IdP application has the correct redirect URIs configured:

- Callback URL: `https://auth-server.DOMAIN/callback` (subdomain) or `https://DOMAIN/auth-server/callback` (path)
- Logout URL: `https://mcpregistry.DOMAIN` (subdomain) or `https://DOMAIN/registry` (path)

## Scaling and Redundancy

### Replica Configuration

Both the auth-server and registry deployments support configuring the number of replicas via `values.yaml`:

```yaml
auth-server:
  replicaCount: 2

registry:
  replicaCount: 2
```

For production environments, we recommend running at least 2 replicas of each service for redundancy.

### Topology Spread Constraints

By default, neither the auth-server nor registry deployments include `topologySpreadConstraints`. This is intentional
for several reasons:

1. **Routing Complexity**: Routing is complex and handled differently between deployments
2. **Development Flexibility**: Single-node or small clusters (common in dev/test) would fail to schedule pods with
   strict spread constraints
3. **Custom Requirements**: Organizations often have specific topology requirements that vary by environment

For production deployments on multi-AZ clusters, we recommend adding topology spread constraints to both deployments to
distribute pods across availability zones and nodes. This improves fault tolerance and ensures service availability
during zone or node failures.

#### Adding Topology Spread Constraints

To add topology spread constraints, patch the deployments after installation:

```bash
# Patch auth-server deployment
kubectl patch deployment auth-server -n MYNAMESPACE --type='json' -p='[
  {
    "op": "add",
    "path": "/spec/template/spec/topologySpreadConstraints",
    "value": [
      {
        "maxSkew": 1,
        "topologyKey": "topology.kubernetes.io/zone",
        "whenUnsatisfiable": "ScheduleAnyway",
        "labelSelector": {
          "matchLabels": {
            "app.kubernetes.io/name": "auth-server",
            "app.kubernetes.io/component": "auth-server"
          }
        }
      },
      {
        "maxSkew": 1,
        "topologyKey": "kubernetes.io/hostname",
        "whenUnsatisfiable": "ScheduleAnyway",
        "labelSelector": {
          "matchLabels": {
            "app.kubernetes.io/name": "auth-server",
            "app.kubernetes.io/component": "auth-server"
          }
        }
      }
    ]
  }
]'

# Patch registry deployment
kubectl patch deployment registry -n MYNAMESPACE --type='json' -p='[
  {
    "op": "add",
    "path": "/spec/template/spec/topologySpreadConstraints",
    "value": [
      {
        "maxSkew": 1,
        "topologyKey": "topology.kubernetes.io/zone",
        "whenUnsatisfiable": "ScheduleAnyway",
        "labelSelector": {
          "matchLabels": {
            "app.kubernetes.io/name": "registry",
            "app.kubernetes.io/component": "registry"
          }
        }
      },
      {
        "maxSkew": 1,
        "topologyKey": "kubernetes.io/hostname",
        "whenUnsatisfiable": "ScheduleAnyway",
        "labelSelector": {
          "matchLabels": {
            "app.kubernetes.io/name": "registry",
            "app.kubernetes.io/component": "registry"
          }
        }
      }
    ]
  }
]'
```

#### Constraint Explanation

- **`topology.kubernetes.io/zone`**: Spreads pods across availability zones for zone-level fault tolerance
- **`kubernetes.io/hostname`**: Spreads pods across different nodes within each zone for node-level fault tolerance
- **`maxSkew: 1`**: Ensures pods are distributed as evenly as possible (difference between zones/nodes is at most 1)
- **`whenUnsatisfiable: ScheduleAnyway`**: Uses soft constraints that prefer even distribution but won't block
  scheduling if perfect distribution isn't possible. Change to `DoNotSchedule` for strict enforcement

## Using Existing Secrets

By default, the stack chart creates and manages Kubernetes Secrets for all components. For production environments
using external secret management (e.g., AWS Secrets Manager with External Secrets Operator, HashiCorp Vault), you
can reference pre-existing secrets instead.

### Stack-Level Existing Secrets

| Value | Default Secret Name | Description |
|-------|---------------------|-------------|
| `global.existingSharedSecret` | `shared-secret` | SECRET_KEY and federation tokens shared by auth-server and registry |
| `global.existingOauthProviderSecret` | `oauth-provider-secret` | Auth provider credentials (Keycloak/Entra/Okta/Auth0/Cognito) |
| `global.existingMongoCredentialsSecret` | `mongo-credentials` | MongoDB connection credentials used by auth-server and registry. When set, `mongodb.connectionString` has no effect — deployment pods read their connection values directly from this existing secret. |
| `global.existingKeycloakClientSecret` | `keycloak-client-secret` | OIDC client credentials (`KEYCLOAK_CLIENT_ID`, `KEYCLOAK_CLIENT_SECRET`, `KEYCLOAK_M2M_CLIENT_ID`, `KEYCLOAK_M2M_CLIENT_SECRET`) for the realm's `mcp-gateway-web` / `mcp-gateway-m2m` clients. Generated by the chart when unset, and published to Keycloak by the `setup-keycloak` Job. Set it — or `global.authProvider.keycloak.clientSecret` / `.m2mClientSecret` — under ArgoCD; see [ArgoCD and other GitOps tools](#argocd-and-other-gitops-tools). |
| `mongodb.existingPasswordSecret` | _(unset)_ | MongoDB operator user password (BYO). When set, the operator's `passwordSecretRef` points at this Secret with key `password`. When unset, the operator reads the password from `mongo-credentials.DOCUMENTDB_PASSWORD` (managed by the chart). |

### Per-Service Existing Secrets

When deploying individual charts (not the stack), each chart supports its own existing secret:

| Chart | Value | Default Secret Name |
|-------|-------|---------------------|
| auth-server | `app.existingSecret` | `auth-server-secret` |
| registry | `app.existingSecret` | `registry-secret` |
| mcpgw | `app.existingSecret` | `mcpgw-secret` |
| keycloak-configure | `keycloak.existingSecret` | `keycloak-configure-secret` |
| mongodb-configure | `mongodb.existingSecret` | `mongo-credentials` |

### Per-Key Existing Secrets

For finer-grained control, individual sensitive values can be sourced from separate existing secrets. Each sensitive field supports two companion values: `{field}ExistingSecret` (secret name) and `{field}ExistingSecretKey` (key within that secret, defaults to the env var name).

**auth-server and registry:**

| Field | ExistingSecret value | ExistingSecretKey default |
|-------|---------------------|--------------------------|
| `entra.clientSecret` | `entra.clientSecretExistingSecret` | `ENTRA_CLIENT_SECRET` |
| `okta.clientSecret` | `okta.clientSecretExistingSecret` | `OKTA_CLIENT_SECRET` |
| `okta.m2mClientSecret` | `okta.m2mClientSecretExistingSecret` | `OKTA_M2M_CLIENT_SECRET` |
| `okta.apiToken` | `okta.apiTokenExistingSecret` | `OKTA_API_TOKEN` |
| `auth0.clientSecret` | `auth0.clientSecretExistingSecret` | `AUTH0_CLIENT_SECRET` |
| `auth0.m2mClientSecret` | `auth0.m2mClientSecretExistingSecret` | `AUTH0_M2M_CLIENT_SECRET` |
| `auth0.managementApiToken` | `auth0.managementApiTokenExistingSecret` | `AUTH0_MANAGEMENT_API_TOKEN` |

**registry only:**

| Field | ExistingSecret value | ExistingSecretKey default |
|-------|---------------------|--------------------------|
| `ans.apiKey` | `ans.apiKeyExistingSecret` | `ANS_API_KEY` |
| `ans.apiSecret` | `ans.apiSecretExistingSecret` | `ANS_API_SECRET` |

When a per-key existing secret is set, the chart skips writing that key into its managed secret and instead injects the value via `env.valueFrom.secretKeyRef`. The key name within the existing secret can be customized using the corresponding `ExistingSecretKey` value.

### Example: Using External Secrets

```bash
# Deploy stack using pre-existing secrets
helm install mcp-gateway-registry -n mcp-gateway-registry --create-namespace . \
  --set global.domain=agents.domain.example \
  --set global.existingSharedSecret=my-shared-secret \
  --set global.existingOauthProviderSecret=my-oauth-secret \
  --set global.existingMongoCredentialsSecret=my-mongo-creds \
  --set mongodb.existingPasswordSecret=my-mongo-password
```

```bash
# Deploy auth-server with Okta client secret from a separate existing secret
helm install mcp-gateway-registry -n mcp-gateway-registry --create-namespace . \
  --set global.domain=agents.domain.example \
  --set global.authProvider.type=okta \
  --set auth-server.okta.domain=dev-123456.okta.com \
  --set auth-server.okta.clientId=MY_CLIENT_ID \
  --set auth-server.okta.clientSecretExistingSecret=my-okta-secret \
  --set auth-server.okta.clientSecretExistingSecretKey=clientSecret
```

When an existing secret is specified:

1. The chart skips creating the corresponding managed Secret resource (or skips that key for per-key references)
2. Deployments and jobs reference the specified secret name instead
3. The existing secret must contain the expected key (defaulting to the env var name)

## ArgoCD and other GitOps tools

The stack installs under ArgoCD with no extra annotations, but two properties of
ArgoCD's Helm support change how secrets must be supplied.

### Turn off secret generation: `global.generateSecrets=false`

ArgoCD renders manifests with `helm template` and no cluster access, so Helm's
`lookup` function always returns empty. Every chart-generated random secret
therefore gets a **new** value on each sync instead of being preserved, which:

- replaces `shared-secret` / `keycloak-client-secret` under running pods — a
  changed `envFrom` Secret does not restart them, so pods hold one value while
  the Secret holds another,
- rotates `mongo-credentials.DOCUMENTDB_PASSWORD`, desyncing it from the SCRAM
  credential the MongoDB Community Operator already provisioned, and
- re-randomizes the Keycloak PostgreSQL password against a retained PVC, which
  is the `bn_keycloak` authentication failure.

Set `global.generateSecrets=false` for any GitOps deployment. The chart then
refuses to invent a value it cannot resolve and fails the render naming the
value to set, instead of rotating a secret on the next sync:

```
Error: execution error at (.../shared-secret.yaml): global.generateSecrets is
false and SECRET_KEY has no value, so the chart will not generate one. Set
global.secretKey explicitly, or set global.existingSharedSecret to a Secret that
already holds it.
```

Supply every value (ArgoCD `Application`: `spec.source.helm.parameters` or
`valuesObject`):

| Value | Secret it lands in |
|-------|--------------------|
| `global.secretKey` | `shared-secret.SECRET_KEY` |
| `registry.egressAuth.markerSecret` | `shared-secret.AUTH_SERVER_NGINX_MARKER_SECRET` |
| `global.federation.staticToken`, `global.federation.encryptionKey` | `shared-secret` (only when `global.federation.staticTokenAuthEnabled=true`) |
| `global.authProvider.keycloak.clientSecret`, `.m2mClientSecret` | `keycloak-client-secret` |
| `keycloak.auth.adminPassword` | `<release>-keycloak.admin-password` |
| `keycloak.postgresql.auth.password`, `.postgresPassword` | `<release>-keycloak-postgresql` |
| `mongodb.password` | `mongo-credentials.DOCUMENTDB_PASSWORD` |

```bash
--set global.generateSecrets=false \
--set global.secretKey=$SECRET_KEY \
--set registry.egressAuth.markerSecret=$MARKER_SECRET \
--set global.authProvider.keycloak.clientSecret=$WEB_CLIENT_SECRET \
--set global.authProvider.keycloak.m2mClientSecret=$M2M_CLIENT_SECRET \
--set keycloak.auth.adminPassword=$KEYCLOAK_ADMIN_PASSWORD \
--set keycloak.postgresql.auth.password=$KEYCLOAK_PG_PASSWORD \
--set keycloak.postgresql.auth.postgresPassword=$KEYCLOAK_PG_SUPERUSER_PASSWORD \
--set mongodb-configure.mongodb.password=$MONGO_PASSWORD
```

Or bring your own Secrets, managed by External Secrets Operator /
sealed-secrets — an existing Secret satisfies the gate with no explicit value:

```bash
--set global.generateSecrets=false \
--set global.existingSharedSecret=my-shared-secret \
--set global.existingKeycloakClientSecret=my-keycloak-client \
--set global.existingMongoCredentialsSecret=my-mongo-creds \
--set global.authProvider.keycloak.adminExistingSecret=my-keycloak-admin \
--set keycloak.postgres.existingSecret=my-keycloak-postgres \
--set mongodb-configure.mongodb.existingSecret=my-mongo-creds
```

Notes:

- `mongodb.password` reaches the `mongodb-configure` subchart through a YAML
  anchor, which only applies to values **files**. With `--set` (and with
  ArgoCD's `parameters:`) use `mongodb-configure.mongodb.password`.
- `keycloak.auth.adminPassword` and `keycloak.postgresql.auth.*` remain stack Secret
  inputs. The official subchart reads the admin and database Secrets through
  `global.authProvider.keycloak.adminExistingSecret` and `keycloak.postgres.existingSecret`.
- The shared admin Secret defaults to key `admin-password`. Set
  `global.authProvider.keycloak.adminExistingSecretPasswordKey` for a custom key.
  The server, initialization Job, realm configuration Job and registry use that same key.
- The PostgreSQL Secret requires `password` and `postgres-password`. Custom keys require
  an existing Secret and `keycloak.postgres.existingSecretPasswordKey` plus
  `keycloak.postgres.existingSecretAdminPasswordKey`.
- Set database passwords on first install only. PostgreSQL initializes its roles once;
  changing a Secret while retaining the volume breaks authentication.
- Changing any of these values on a live install rewrites the Secret but does
  not restart pods. Follow up with
  `kubectl rollout restart deploy/registry deploy/auth-server`.

### Why `keycloak-client-secret` is chart-rendered

`registry` and `auth-server` mount `keycloak-client-secret` with
`envFrom.secretRef`, so their pods cannot start until it exists. The chart
renders it in the install/Sync phase and the `setup-keycloak` Job publishes those
values into the realm's `mcp-gateway-web` / `mcp-gateway-m2m` clients.

It used to work the other way around — the Job created the Secret from
Keycloak-generated values. That deadlocked under ArgoCD: the Job
carries `helm.sh/hook: post-install,post-upgrade`, which ArgoCD maps to its
PostSync phase, and PostSync only runs once every Sync-phase resource reports
Healthy. The Deployments could never become Healthy without the Secret that only
the PostSync Job would create.

On a fresh install, expect a short window where pods are Healthy but logins fail
with `invalid_client`: the realm does not exist until the PostSync Job finishes.
It is self-clearing and requires no pod restart.

### Upgrading a cluster installed before this change

Verify first — the ownership check below is reported by a dry run, before
anything is applied:

```bash
helm upgrade <release> . -n <ns> -f <your-values.yaml> --dry-run=server
```

Use `--dry-run=server`, **not** `--dry-run`. Client-side dry run disables Helm's
`lookup`, so every chart-generated secret renders as a fresh random value and the
preview shows `shared-secret`, `mongo-credentials`, the Keycloak admin/PostgreSQL
passwords and `keycloak-client-secret` all rotating. That is an artifact of the
preview, not of the upgrade (the same thing happens on the previous chart
version). The `helm diff` plugin has the same caveat.


Existing clusters have a `keycloak-client-secret` that was written by the Job
with `kubectl`, so it carries no Helm ownership metadata. The chart detects
this and stops the upgrade before anything is applied, printing the commands
to run with your release name and namespace filled in:

```
Error: UPGRADE FAILED: execution error at (.../keycloak-client-secret.yaml):
Secret keycloak-client-secret already exists in namespace <ns> but is not owned
by a Helm release. ... Adopt it into release <release> (this keeps the current
client secrets, so nobody has to log in again), then re-run the upgrade:
  kubectl -n <ns> label secret keycloak-client-secret app.kubernetes.io/managed-by=Helm --overwrite
  kubectl -n <ns> annotate secret keycloak-client-secret meta.helm.sh/release-name=<release> meta.helm.sh/release-namespace=<ns> --overwrite
```

The running deployment is untouched while the upgrade is refused. Adopting the
Secret keeps the current client secrets, so there is no re-login and no realm
change:

```bash
NS=mcp-gateway-registry
RELEASE=mcp-gateway-registry
kubectl -n "$NS" label secret keycloak-client-secret app.kubernetes.io/managed-by=Helm --overwrite
kubectl -n "$NS" annotate secret keycloak-client-secret \
  meta.helm.sh/release-name="$RELEASE" meta.helm.sh/release-namespace="$NS" --overwrite
```

With Helm (which can read the cluster), the chart then preserves the adopted
values on every subsequent upgrade. Under ArgoCD, also set the explicit values
from the section above, otherwise the next sync replaces them with new random
ones. Deleting the Secret instead of adopting it also works: the chart generates
fresh credentials and the Job pushes them to Keycloak, at the cost of a brief
`invalid_client` window until both the Job and a pod rollout complete.

### Rollback

`keycloak-client-secret` is part of the release manifest now, so `helm rollback`
restores the previous revision's client secrets. The `setup-keycloak` Job
therefore also runs on `post-rollback`, re-pointing the realm's clients at the
restored values. Logins fail with `invalid_client` between the rollback and the
Job completing.

## Keycloak database cutover

This procedure backs up the current Keycloak database, restores it with `keycloak.postgres.bootstrap.mode=restore`, then switches the verified deployment to `existing`. It applies to a Bitnami source or an older official-image deployment. It does not back up MongoDB, OpenBao, or other gateway data stores.

**This is a maintenance operation with a login outage.** Helm runs the restore and initialization hooks; the commands below stop the source writers and prepare the snapshot. Keep the source data and credentials until the upgraded deployment passes verification and your rollback retention period expires.

| Mode | When to use it |
|------|----------------|
| `fresh-install` | An intentionally empty installation. Do not use it for this upgrade. |
| `restore` | The target must restore the verified archive prepared below. |
| `existing` | The target already contains a verified official-image database with a non-superuser Keycloak role. Use it after completing the restore. |

The chart uses Keycloak 26.8.2 and PostgreSQL 17.12. PostgreSQL restores through a local socket and commits a receipt before accepting network connections. Keycloak then starts and applies its supported schema updates. The administrator-initialization hook runs at weight 5; realm configuration follows at weight 10. A Helm rollback does not undo database schema changes or restore data.

### 1. Identify the source and save the release configuration

Run from the repository root in Bash. Install `helm`, `kubectl`, `jq`, and `envsubst` locally. Use the same shell throughout so the variables remain available. All names below are examples; verify them against your namespace before proceeding.

```bash
set -euo pipefail
set +x
umask 077

export CONTEXT=YOUR_KUBERNETES_CONTEXT
export NS=YOUR_NAMESPACE
export RELEASE=YOUR_HELM_RELEASE
export CHART="$PWD/charts/mcp-gateway-registry-stack"
export RUN_ID="$(date -u +%Y%m%d%H%M%S)"
export BACKUP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/keycloak-cutover.XXXXXXXX")"
chmod 0700 "$BACKUP_DIR"
k() { kubectl --context "$CONTEXT" -n "$NS" "$@"; }
h() { helm --kube-context "$CONTEXT" -n "$NS" "$@"; }

h status "$RELEASE"
k get deployments,statefulsets,pods,services,pvc,hpa
h get values "$RELEASE" -o yaml > "$BACKUP_DIR/release-values.yaml"
h get manifest "$RELEASE" > "$BACKUP_DIR/release-manifest.yaml"
h get hooks "$RELEASE" > "$BACKUP_DIR/release-hooks.yaml"
h history "$RELEASE" -o json > "$BACKUP_DIR/release-history.json"
k get deployments,statefulsets,services,pvc,hpa -o yaml > "$BACKUP_DIR/workloads-storage.yaml"
```

The backup directory contains sensitive configuration. Copy it to your approved encrypted backup store before ending the maintenance session; do not commit it or serve it over HTTP. Save the application image digests from the running pods as well, because a mutable `latest` tag cannot reproduce the old deployment.

Set the source identifiers from the workload and Secret references you inspected. On older Bitnami installs the database might be `bn_keycloak`; on older official-image installs it is often `keycloak`. The source database user must be able to dump every table and inspect `pg_stat_activity`.

```bash
export OLD_KEYCLOAK_WORKLOAD=deployment/YOUR_OLD_KEYCLOAK_DEPLOYMENT
export OLD_KEYCLOAK_SELECTOR='YOUR_KEYCLOAK_POD_LABEL_SELECTOR'
export OLD_PG_WORKLOAD=statefulset/YOUR_OLD_POSTGRES_STATEFULSET
export OLD_PG_POD=YOUR_OLD_POSTGRES_POD
export SOURCE_DATA_PVC=YOUR_SOURCE_DATABASE_PVC
export PGHOST=YOUR_SOURCE_POSTGRES_SERVICE
export PGPORT=5432
export PGDATABASE=YOUR_SOURCE_DATABASE
export PGUSER=YOUR_SOURCE_BACKUP_USER
export SOURCE_PASSWORD_SECRET=YOUR_SOURCE_PASSWORD_SECRET
export SOURCE_PASSWORD_KEY=YOUR_SOURCE_PASSWORD_KEY
export PGSSLMODE=require
export SNAPSHOT_CLASS=YOUR_STORAGE_CLASS
export SNAPSHOT_SIZE=2Gi
export SNAPSHOT_PVC="keycloak-snapshot-$RUN_ID"
export BACKUP_POD="keycloak-backup-$RUN_ID"
export TARGET_PG_STS="${RELEASE}-keycloak-postgres"
export TARGET_PG_POD="${TARGET_PG_STS}-0"
export TARGET_DATA_PVC="data-${TARGET_PG_STS}-0"
export ADMIN_SECRET="${RELEASE}-keycloak"
export TARGET_PASSWORD_SECRET="${RELEASE}-keycloak-postgresql"
export OLD_REVISION="$(h history "$RELEASE" -o json | jq -r 'map(select(.status == "deployed")) | last | .revision')"
export OLD_KEYCLOAK_REPLICAS="$(k get "$OLD_KEYCLOAK_WORKLOAD" -o jsonpath='{.spec.replicas}')"
export SOURCE_PV="$(k get pvc "$SOURCE_DATA_PVC" -o jsonpath='{.spec.volumeName}')"
```

Use a PostgreSQL client compatible with the source version. These commands use 17.12 and must not run against a newer PostgreSQL major version. Keep `PGSSLMODE=require` for a TLS-enabled source; use `verify-full` with your mounted CA for certificate verification. If the source has no TLS, explicitly set `PGSSLMODE=disable` only for its trusted private cluster connection. Confirm that the chosen storage class supports the backup pod's UID/GID 999 and can provision enough space for the archive. A zonal RWO snapshot PVC can constrain where the new PostgreSQL pod runs; ensure its target data volume can attach in the same zone.

Preserve credentials without decoding them into terminal output:

```bash
k get secret "$ADMIN_SECRET" "$TARGET_PASSWORD_SECRET" \
  shared-secret keycloak-client-secret mongo-credentials registry-login-credentials \
  -o yaml > "$BACKUP_DIR/credentials.yaml"
k get "$OLD_PG_WORKLOAD" -o json > "$BACKUP_DIR/source-postgres.json"
k get "$OLD_KEYCLOAK_WORKLOAD" -o json > "$BACKUP_DIR/source-keycloak.json"
k get pods -o json | jq '[.items[] | {name: .metadata.name, images: [.status.containerStatuses[]? | {name, image, imageID}]}]' \
  > "$BACKUP_DIR/image-identities.json"
```

If a listed Secret has a different name, replace it with the reference used by the running workload. Do not generate replacement credentials during this operation. The target PostgreSQL Secret needs `password` for the dedicated Keycloak role and `postgres-password` for its separate bootstrap administrator. Restore preserves the realm's existing administrator password; `KC_BOOTSTRAP_ADMIN_PASSWORD` does not reset it.

### 2. Stop writers and create the snapshot pod

Pause GitOps reconciliation and disable or suspend any HPA that can restart the old Keycloak workload. Stop other applications or maintenance Jobs that write to this database. Keep the old PostgreSQL server running until the dump finishes.

Protect the source volume before any Helm operation. If the source StatefulSet has a PVC deletion policy, change it to retain claims on deletion and scale-down:

```bash
k patch pv "$SOURCE_PV" --type=merge \
  -p '{"spec":{"persistentVolumeReclaimPolicy":"Retain"}}'
k patch "$OLD_PG_WORKLOAD" --type=merge \
  -p '{"spec":{"persistentVolumeClaimRetentionPolicy":{"whenDeleted":"Retain","whenScaled":"Retain"}}}'
k scale "$OLD_KEYCLOAK_WORKLOAD" --replicas=0
k wait --for=delete pod -l "$OLD_KEYCLOAK_SELECTOR" --timeout=5m
```

The PV is cluster-scoped: this command changes only the PV bound to the verified source PVC. Also prevent GitOps from pruning the source or snapshot PVC. Inspect existing NetworkPolicies and permit the backup pod below to reach only the source PostgreSQL service on its database port. NetworkPolicy allow rules are additive; adding one narrow policy does not cancel a broader existing policy.

Create a snapshot PVC and a temporary client pod. The password comes from the existing Secret through an environment reference, never a command argument.

```bash
cat > "$BACKUP_DIR/backup-pod.template.yaml" <<'YAML'
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${SNAPSHOT_PVC}
  namespace: ${NS}
spec:
  accessModes: [ReadWriteOnce]
  storageClassName: ${SNAPSHOT_CLASS}
  resources:
    requests:
      storage: ${SNAPSHOT_SIZE}
---
apiVersion: v1
kind: Pod
metadata:
  name: ${BACKUP_POD}
  namespace: ${NS}
  labels:
    app.kubernetes.io/name: keycloak-cutover-backup
spec:
  restartPolicy: Never
  automountServiceAccountToken: false
  securityContext:
    runAsNonRoot: true
    runAsUser: 999
    runAsGroup: 999
    fsGroup: 999
    seccompProfile:
      type: RuntimeDefault
  containers:
    - name: postgres-client
      image: postgres:17.12
      command: ["sleep", "86400"]
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        capabilities:
          drop: [ALL]
      env:
        - name: PGHOST
          value: "${PGHOST}"
        - name: PGPORT
          value: "${PGPORT}"
        - name: PGDATABASE
          value: "${PGDATABASE}"
        - name: PGUSER
          value: "${PGUSER}"
        - name: PGSSLMODE
          value: "${PGSSLMODE}"
        - name: PGPASSWORD
          valueFrom:
            secretKeyRef:
              name: ${SOURCE_PASSWORD_SECRET}
              key: ${SOURCE_PASSWORD_KEY}
      volumeMounts:
        - name: snapshot
          mountPath: /snapshot
        - name: tmp
          mountPath: /tmp
  volumes:
    - name: snapshot
      persistentVolumeClaim:
        claimName: ${SNAPSHOT_PVC}
    - name: tmp
      emptyDir: {}
YAML
envsubst < "$BACKUP_DIR/backup-pod.template.yaml" > "$BACKUP_DIR/backup-pod.yaml"
k apply -f "$BACKUP_DIR/backup-pod.yaml"
k wait --for=condition=Ready "pod/$BACKUP_POD" --timeout=5m
export SNAPSHOT_PV="$(k get pvc "$SNAPSHOT_PVC" -o jsonpath='{.spec.volumeName}')"
k patch pv "$SNAPSHOT_PV" --type=merge \
  -p '{"spec":{"persistentVolumeReclaimPolicy":"Retain"}}'
```

Check connections before the dump. The first query must show no other client backends, including idle connections. If it lists any, identify and stop their owners rather than terminating arbitrary database sessions.

```bash
k exec "$BACKUP_POD" -- psql -X -v ON_ERROR_STOP=1 -c \
  "SELECT pid, usename, application_name, client_addr, state FROM pg_stat_activity WHERE datname = current_database() AND backend_type = 'client backend' AND pid <> pg_backend_pid();"
k exec "$BACKUP_POD" -- sh -ec '
  test "$(psql -X -At -v ON_ERROR_STOP=1 -c "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() AND backend_type = '\''client backend'\'' AND pid <> pg_backend_pid()")" = 0
'
```

Capture counts and create the definitive dump while the writers remain stopped. Excluding `_mcp_bootstrap` also makes this command suitable for a source that already uses the new chart: the target creates its own receipt.

```bash
cat > "$BACKUP_DIR/counts.sql" <<'SQL'
SELECT 'realm', count(*) FROM public.realm
UNION ALL SELECT 'user_entity', count(*) FROM public.user_entity
UNION ALL SELECT 'client', count(*) FROM public.client
ORDER BY 1;
SQL
k exec -i "$BACKUP_POD" -- psql -X -At -v ON_ERROR_STOP=1 \
  < "$BACKUP_DIR/counts.sql" > "$BACKUP_DIR/source-counts.txt"
k exec "$BACKUP_POD" -- sh -ec '
  umask 077
  test ! -e /snapshot/restore.dump
  test ! -e /snapshot/restore.dump.partial
  pg_dump --format=custom --no-owner --no-acl --exclude-schema=_mcp_bootstrap \
    --file=/snapshot/restore.dump.partial
  pg_restore --list /snapshot/restore.dump.partial >/snapshot/archive-list.txt
  mv /snapshot/restore.dump.partial /snapshot/restore.dump
  chmod 0400 /snapshot/restore.dump
  sha256sum /snapshot/restore.dump
' > "$BACKUP_DIR/snapshot.sha256"
export SNAPSHOT_SHA="$(cut -d ' ' -f 1 "$BACKUP_DIR/snapshot.sha256")"
test "${#SNAPSHOT_SHA}" -eq 64
k exec "$BACKUP_POD" -- cat /snapshot/restore.dump > "$BACKUP_DIR/restore.dump"
LOCAL_SHA="$(shasum -a 256 "$BACKUP_DIR/restore.dump" | cut -d ' ' -f 1)"
test "$LOCAL_SHA" = "$SNAPSHOT_SHA"
k delete pod "$BACKUP_POD" --wait=true --timeout=5m
```

A successful archive listing proves format, not source identity. Check the recorded counts and source names, and copy the local archive to your encrypted backup store. Keep the snapshot PVC and archive unchanged. Deleting the client pod releases its RWO attachment for the target PostgreSQL pod.

### 3. Prepare and validate the Helm restore values

Create `upgrade-values.yaml` in the private backup directory from your deployment's reviewed desired values. Use `release-values.yaml` as an input, not as an unquestioned replacement for current chart defaults. Remove obsolete Bitnami keys such as `extraEnvVars`, `replicaCount`, `externalDatabase`, `lifecycleHooks`, and automatic migration options. Preserve domains, provider settings, Secret references, and unrelated services' image versions and configuration. For the target, set `keycloak.image.repository=quay.io/keycloak/keycloak`, `keycloak.image.tag="26.8.2"`, `keycloak.postgres.image.repository=postgres`, and `keycloak.postgres.image.tag="17.12"`, or use an explicitly verified compatible derived image. Keep these target image settings in `upgrade-values.yaml` for both restore and existing mode; do not carry forward a Bitnami repository or an unsupported old Keycloak tag. Keep `keycloak.create=true` and `global.authProvider.type=keycloak`.

For a normal Helm upgrade, leave existing chart-managed Secret references managed by the chart so `lookup` preserves their values. Set `global.generateSecrets=false` to refuse accidental generation. Do not change a chart-managed Secret to `existingSecret` with the same name without reviewing Helm ownership: omitting it from the new manifest can make Helm delete it. Follow the existing-Secret adoption guidance above when migrating an older Job-created client Secret.

In the reviewed values, keep the existing Keycloak administrator username and password. Set `keycloak.postgres.username=keycloak` and `keycloak.postgres.database=keycloak` for the restored target. The bootstrap creates that role without superuser privileges, even when the source used a superuser named `keycloak`. The source's database owner does not need to match the target owner because the archive excludes ownership and ACLs.

Create the restore overlay and inspect a server-side dry run. The dry run includes Secret manifests, so save it privately.

```bash
export VALUES="$BACKUP_DIR/upgrade-values.yaml"
test -f "$VALUES"
cat > "$BACKUP_DIR/restore-values.yaml" <<YAML
global:
  generateSecrets: false
keycloak:
  postgres:
    username: keycloak
    database: keycloak
    bootstrap:
      mode: restore
      existingClaim: "$SNAPSHOT_PVC"
      sha256: "$SNAPSHOT_SHA"
      sourceQuiesced: true
      existingVerified: false
YAML
helm dependency build "$CHART"
h upgrade "$RELEASE" "$CHART" -f "$VALUES" -f "$BACKUP_DIR/restore-values.yaml" \
  --dry-run=server > "$BACKUP_DIR/restore-dry-run.txt"
```

Choose a timeout long enough for your archive and Keycloak schema update. The PostgreSQL startup-probe allowance defaults to 24 hours; a shorter Helm timeout can report failure while the pod continues working. Inspect progress before retrying. Do not add `--atomic`: automatic manifest rollback after a database migration can start an old Keycloak image against a changed schema. Keep rollback under operator control.

The restore preserves realm data, then the `setup-keycloak` hook reconciles the configured clients, groups and users. Review those intended changes before upgrading. An older experimental realm with full-path group mappers also needs an application image that understands those claims; restoring the database does not convert its mapper format.

### 4. Prepare target storage without destroying the source

The chart uses StatefulSet `${RELEASE}-keycloak-postgres` and PVC `data-${RELEASE}-keycloak-postgres-0`. Inspect that exact claim:

```bash
k get pvc "$TARGET_DATA_PVC" --ignore-not-found
k scale "$OLD_PG_WORKLOAD" --replicas=0
k wait --for=delete "pod/$OLD_PG_POD" --timeout=5m
```

If the target PVC does not exist, the new StatefulSet creates it. Keep the old source PVC and PV retained. If a target PVC already exists but is not `SOURCE_DATA_PVC`, stop here and identify its contents; do not assume it is empty or delete it.

If `TARGET_DATA_PVC` equals `SOURCE_DATA_PVC`, as with an older official-image chart, the new restore cannot start while `/var/lib/postgresql/data/pgdata` already exists. Preserve that directory under a unique name on the same volume. This requires free space for the retained source, restored database, staging files and the temporary SQL restore file. Check `df` and `du` below; expand the PVC first through your storage provider if necessary. Do not reduce the source database merely to make it fit.

Run this block only for that verified same-PVC case. It assumes the old official PostgreSQL server used UID/GID 999 and `PGDATA=/var/lib/postgresql/data/pgdata`; confirm both in the saved StatefulSet. Other layouts need a storage-specific plan.

```bash
test "$SOURCE_DATA_PVC" = "$TARGET_DATA_PVC"
export MAINTENANCE_POD="keycloak-volume-$RUN_ID"
export RETAINED_DIR="pgdata.pre-cutover-$RUN_ID"
cat > "$BACKUP_DIR/maintenance-pod.template.yaml" <<'YAML'
apiVersion: v1
kind: Pod
metadata:
  name: ${MAINTENANCE_POD}
  namespace: ${NS}
spec:
  restartPolicy: Never
  automountServiceAccountToken: false
  securityContext:
    runAsNonRoot: true
    runAsUser: 999
    runAsGroup: 999
    seccompProfile:
      type: RuntimeDefault
  containers:
    - name: postgres-client
      image: postgres:17.12
      command: ["sleep", "86400"]
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        capabilities:
          drop: [ALL]
      volumeMounts:
        - name: data
          mountPath: /data
  volumes:
    - name: data
      persistentVolumeClaim:
        claimName: ${TARGET_DATA_PVC}
YAML
envsubst < "$BACKUP_DIR/maintenance-pod.template.yaml" > "$BACKUP_DIR/maintenance-pod.yaml"
k apply -f "$BACKUP_DIR/maintenance-pod.yaml"
k wait --for=condition=Ready "pod/$MAINTENANCE_POD" --timeout=5m
k exec "$MAINTENANCE_POD" -- df -h /data
k exec "$MAINTENANCE_POD" -- du -sh /data/pgdata
# Proceed only after confirming sufficient free capacity.
k exec "$MAINTENANCE_POD" -- sh -ec '
  test -s /data/pgdata/PG_VERSION
  test ! -e /data/pgdata/postmaster.pid
  test ! -e /data/.keycloak-bootstrap-pending
  test ! -e /data/pgdata.bootstrap
  test ! -e /data/pgdata.initializing
  test ! -e "/data/$1"
  mv /data/pgdata "/data/$1"
  sync
' sh "$RETAINED_DIR"
printf '%s\n' "$RETAINED_DIR" > "$BACKUP_DIR/retained-directory.txt"
k delete pod "$MAINTENANCE_POD" --wait=true --timeout=5m
```

Keep the old source workload stopped. For the same-StatefulSet case, recreate only its controller before the Helm upgrade so changes to immutable claim-template metadata cannot block the update. The pod must already be gone and its claim-retention policy must be `Retain` as set above. This command does not delete the PVC:

```bash
if [ "$OLD_PG_WORKLOAD" = "statefulset/$TARGET_PG_STS" ]; then
  k delete "$OLD_PG_WORKLOAD" --cascade=orphan --wait=true
fi
```

Do not use `helm upgrade --force` to replace storage resources. Never delete a pending or completion marker to make a failed migration look new.

### 5. Apply the restore upgrade

After completing the storage preparation and reviewing the server-side dry run, run the upgrade with the same values and snapshot:

```bash
h upgrade "$RELEASE" "$CHART" -f "$VALUES" -f "$BACKUP_DIR/restore-values.yaml" \
  --wait --timeout 30m > "$BACKUP_DIR/restore-upgrade.log" 2>&1
```

### 6. Verify data and access before reopening the deployment

Check the PostgreSQL pod, Keycloak rollout and both initialization Jobs. The counts below are taken after the hooks finish; Keycloak startup and realm setup may have added or reconciled records.

```bash
k rollout status "statefulset/$TARGET_PG_STS" --timeout=10m
k rollout status "deployment/${RELEASE}-keycloak" --timeout=10m
k wait --for=condition=complete "job/${RELEASE}-keycloak-admin-init" --timeout=10m
k wait --for=condition=complete job/setup-keycloak --timeout=10m
k logs "$TARGET_PG_POD" --tail=100
k logs "job/${RELEASE}-keycloak-admin-init" --tail=50
k logs job/setup-keycloak --tail=100
k exec -i "$TARGET_PG_POD" -- psql -X -U postgres -d keycloak -At -v ON_ERROR_STOP=1 \
  < "$BACKUP_DIR/counts.sql" > "$BACKUP_DIR/restored-counts.txt"
diff -u "$BACKUP_DIR/source-counts.txt" "$BACKUP_DIR/restored-counts.txt"
k exec "$TARGET_PG_POD" -- psql -X -U postgres -d keycloak -At -v ON_ERROR_STOP=1 -c \
  "SELECT rolname, rolsuper, rolcreatedb, rolcreaterole, rolreplication FROM pg_roles WHERE rolname = 'keycloak'; SELECT intent FROM _mcp_bootstrap.receipt;"
k exec "$TARGET_PG_POD" -- sh -ec '
  test ! -e /var/lib/postgresql/data/.keycloak-bootstrap-pending
  cat "$PGDATA/.keycloak-bootstrap-complete"
'
```

All four role flags must be `f`. The database and filesystem receipts must contain `restore:` followed by `SNAPSHOT_SHA`. Investigate every count difference using the saved baseline and setup logs; expected additions are not proof that all original records survived. Check important realm, user and client identities, group memberships and existing client secrets.

Sign out and log in through the public gateway with an existing administrator. Check `/api/auth/me` for the expected groups and `is_admin=true`, and verify that expected resources are visible. Use an existing workload's unchanged credential to obtain a fresh token and call an allowed operation. Confirm a restricted user still receives a denial for an ungranted operation. Compare the running deployment environment with the rendered values; changing a Secret alone does not restart its consumers.

Keep the old source stopped. If any of these checks fail, do not reopen login or workload traffic and do not remove the backup.

### 7. Transition from restore to existing

After data and authentication checks pass, replace the restore overlay with this one. Supply the same reviewed base values and change all four restore-specific settings together. `existing` verifies the retained target; it does not restore the archive again or change its data.

```bash
cat > "$BACKUP_DIR/existing-values.yaml" <<'YAML'
global:
  generateSecrets: false
keycloak:
  postgres:
    username: keycloak
    database: keycloak
    bootstrap:
      mode: existing
      existingVerified: true
      existingClaim: ""
      sha256: ""
      sourceQuiesced: false
YAML
h upgrade "$RELEASE" "$CHART" -f "$VALUES" -f "$BACKUP_DIR/existing-values.yaml" \
  --dry-run=server > "$BACKUP_DIR/existing-dry-run.txt"
h upgrade "$RELEASE" "$CHART" -f "$VALUES" -f "$BACKUP_DIR/existing-values.yaml" \
  --wait --timeout 30m > "$BACKUP_DIR/existing-upgrade.log" 2>&1
k rollout status "statefulset/$TARGET_PG_STS" --timeout=10m
k get "statefulset/$TARGET_PG_STS" -o json | jq -e \
  'all(.spec.template.spec.volumes[]; .name != "migration")'
k exec "$TARGET_PG_POD" -- sh -ec '
  test ! -e /var/lib/postgresql/data/.keycloak-bootstrap-pending
  test "$(cat "$PGDATA/.keycloak-bootstrap-complete")" = "existing:"
'
k exec -i "$TARGET_PG_POD" -- psql -X -U postgres -d keycloak -At -v ON_ERROR_STOP=1 \
  < "$BACKUP_DIR/counts.sql" > "$BACKUP_DIR/existing-counts.txt"
diff -u "$BACKUP_DIR/restored-counts.txt" "$BACKUP_DIR/existing-counts.txt"
```

The filesystem receipt changes to `existing:`. The database's `_mcp_bootstrap.receipt` still records the original restore intent; that is expected. Confirm browser login and workload access again after the hooks finish. Merge `existing-values.yaml` into the deployment's maintained values so the next upgrade cannot accidentally return to the old restore intent. Keep `existingVerified=true` for normal upgrades of this verified target.

Only after the StatefulSet no longer references the snapshot, access checks pass, and backup retention requirements are met may you retire the snapshot PVC. Removing it while `restore` mode is configured breaks later pod starts.

```bash
# Optional after retention approval; this does not delete the separately saved archive.
k delete pvc "$SNAPSHOT_PVC"
```

The snapshot PV was marked `Retain`, so deleting its PVC leaves the backing volume for separate, deliberate disposal through your storage process. Keep the original source PVC or retained directory until its rollback retention period also expires. Do not include source-data deletion in the upgrade command sequence.

### Failure and rollback

If restore fails, collect the PostgreSQL and hook logs. Keep both source and target data, the snapshot, checksum and original mode. Fix the reported cause and rerun the same restore command; a completed transaction is not replayed. Do not switch to `fresh-install`, change snapshots, or remove receipt/pending files to bypass a failed restore.

Before accepting new application writes, rollback can use the old image/configuration with the retained source database. First stop the new Keycloak workload and PostgreSQL pod and verify that both have terminated. For separate source and target PVCs, restore the old workload configuration and resume the retained source. For the same-PVC procedure, recreate the maintenance pod after PostgreSQL stops. Move every artifact of the failed target into one new diagnostic directory before moving `RETAINED_DIR` back to `pgdata`: this includes `pgdata` if promoted, `pgdata.bootstrap`, `pgdata.initializing`, `.keycloak-bootstrap-pending`, `.keycloak-bootstrap-pending.tmp`, and `.keycloak-bootstrap.sql` when present. A failed restore may never have created `pgdata`. Preserve these artifacts together instead of deleting them or leaving a pending intent beside the recovered source. The source directory includes its own original completion receipt, if it had one. Never mount the volume in two PostgreSQL servers at once.

Inspect the old release's hooks before using `helm rollback`: rollback runs hooks and can change realm configuration. If rollback needs to suppress those hooks, use `--no-hooks` and reconcile only the required settings after the old database is running. If the PostgreSQL StatefulSet claim template changed, remove only the stopped controller with `--cascade=orphan` before recreating it from the retained configuration. Restore the original replica counts and credentials, then verify login and data again.

After users or workloads write to the new deployment, switching back to the old snapshot loses those changes. Stop and prepare a reconciliation or forward-recovery plan instead. Helm release history is not a database backup.
