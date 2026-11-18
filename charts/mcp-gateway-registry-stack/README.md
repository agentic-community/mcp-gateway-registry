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

When another provider is active, the chart skips bundled Keycloak resources and the
realm configuration Job, even if their creation switches remain enabled.

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

The official Keycloak/PostgreSQL chart requires an explicit maintenance cutover from Bitnami. Helm and GitOps render the
same restore path. The chart issues no source or snapshot deletion commands; Helm and GitOps can still prune old
resources unless you retain them as described below.

| `keycloak.postgres.bootstrap.mode` | Use                                                                                                                                                   |
|------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| `fresh-install`                    | Create an intentionally empty database. Keep this setting through ordinary upgrades of that installation.                                             |
| `restore`                          | Restore `restore.dump` from an operator-owned PVC after stopping every source writer. Requires `existingClaim`, `sha256`, and `sourceQuiesced: true`. |
| `existing`                         | Adopt a previously verified official PostgreSQL volume. Requires `existingVerified: true`; the runtime rejects an empty or incomplete database.       |

The chart pins Keycloak 26.8.2 and PostgreSQL 17.12. A separate post-install, post-upgrade or
post-rollback Job initializes the admin realm at hook weight 5. Realm configuration follows at
weight 10. Initialization failures stop the Helm operation without restarting the server.
Correct the administrator Secret to match the realm before retrying.

For multiple Keycloak replicas, set `keycloak.cache=ispn`; the chart uses JDBC discovery over the shared database. The chart refuses multiple replicas with the single-process `local` cache.

### Freeze and snapshot the source

Schedule a login outage. Save the deployed values and preserve the existing Keycloak/PostgreSQL credentials in your
secret manager. Stop GitOps reconciliation and autoscaling for the old Keycloak workload before scaling it to zero. Stop
any other process that can write to the source database. Keep those writers stopped until the new deployment passes
verification.

Inspect the deployed workload names rather than assuming the release prefix:

```bash
kubectl -n "$NS" get statefulsets,deployments,pods,pvc
kubectl -n "$NS" scale "$OLD_KEYCLOAK_WORKLOAD" --replicas=0
```

Verify that every old Keycloak pod has terminated and that `pg_stat_activity` shows no remaining application sessions on
the source database. Restrict its network access to the snapshot process. If you cannot establish this state, stop the
cutover before taking the definitive dump.

Retain the old database PVC and its PV for rollback. For ArgoCD, prevent pruning/deletion of the source resources or
move them outside the application's ownership before syncing the new chart. Set the PV reclaim policy to `Retain`. A
Helm rollback alone cannot recover an old database that the cluster has deleted.

Create a dedicated PVC for the snapshot. In a temporary PostgreSQL 17.12 client pod, mount that PVC at `/snapshot` and
the source password through a Kubernetes Secret as `PGPASSWORD` or `PGPASSFILE`. Set `PGHOST`, `PGPORT`, `PGUSER`, and
`PGDATABASE` to the deployed source values. Permit this pod through the source's NetworkPolicy. Do not put the password
on the command line.

Run these commands inside that pod as UID/GID `999:999`:

```bash
set -euo pipefail
umask 077
pg_dump --format=custom --no-owner --no-acl --file=/snapshot/restore.dump.partial
pg_restore --list /snapshot/restore.dump.partial
mv /snapshot/restore.dump.partial /snapshot/restore.dump
chmod 0400 /snapshot/restore.dump
sha256sum /snapshot/restore.dump
```

Record the source's realm, user and client counts separately for post-restore comparison. A successful archive listing
checks its format, not whether you selected the right source database. Delete the temporary client pod after the dump so
an RWO PVC can attach to the new PostgreSQL pod. Keep the PVC and its archive unchanged.

For an archive from an existing official-chart installation, add `--exclude-schema=_mcp_bootstrap` to `pg_dump` so the
new bootstrap can create its own completion receipt.

### Restore the new database

Remove obsolete Bitnami values before rendering. The chart rejects `externalDatabase`, `extraEnvVars`, `replicaCount`,
`lifecycleHooks`, and the former automatic migration fields rather than ignoring them. Use `extraEnv` only for names
outside the chart-managed reserved list; use the named chart values for database, hostname and security settings.

Merge these values into the deployment configuration:

```yaml
keycloak:
  postgres:
    bootstrap:
      mode: restore
      existingClaim: keycloak-cutover-snapshot
      sha256: YOUR_RESTORE_DUMP_SHA256
      sourceQuiesced: true
```

Replace the SHA placeholder with the lowercase 64-character digest from `sha256sum`. The snapshot PVC must expose
`restore.dump` at its root, readable by UID/GID 999. The PostgreSQL pod mounts it read-only at
`/migration/restore.dump`.

Render the chart and inspect the new resources before applying. GitOps deployments must also use the explicit
existing-secret configuration described above; do not generate new database or OAuth secrets during migration. Apply
only after confirming that the old source remains stopped and retained.

The bootstrap wrapper verifies the archive digest, restores through a local-only PostgreSQL socket, and records
completion before accepting network connections. An incomplete restore cannot become a healthy database merely because
`PG_VERSION` exists. Keep the same restore mode, snapshot and digest when retrying an interrupted restore; changing
those values does not bypass its pending marker.

### Verify before reopening access

Compare realm, user and client counts with the source. Check browser login, then use an existing
client to fetch a fresh token through the gateway. Verify group-based access and a denied
operation before reopening access. Keep the old source stopped so the databases cannot diverge.

If restore fails, retain both volumes and inspect the PostgreSQL logs. Correct the reported cause and retry with the
same snapshot. If you must abandon the attempt, preserve the failed target for diagnosis and use a new empty target
volume with the original verified snapshot. Never delete a completion or pending marker to force startup.

Before the new deployment accepts writes, rollback can restore the previous manifests and resume
the retained source. After writes begin, rollback needs a plan to preserve those changes.
Apply your backup retention and access policy to the snapshot.

### Retire the restore snapshot

Verify the restored data and confirm that no bootstrap operation remains pending. Change the
mode to `existing`, set `existingVerified=true`, and remove all restore-only fields together:

```yaml
keycloak:
  postgres:
    bootstrap:
      mode: existing
      existingVerified: true
      existingClaim: ""
      sha256: ""
      sourceQuiesced: false
```

Roll out this change. Verify database readiness and an authenticated gateway request before
retiring the snapshot PVC. A deployment left in `restore` mode mounts that PVC on every pod
start. Never remove it first or delete a pending marker to force a mode change.

