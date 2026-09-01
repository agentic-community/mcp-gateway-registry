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

### Authentication Provider Selection

This chart supports five authentication providers: Keycloak (default), Microsoft Entra ID, Okta, Auth0, and AWS Cognito.

When using any provider other than Keycloak, disable the Keycloak components:

```yaml
keycloak:
  create: false
keycloak-configure:
  enabled: false
```

#### Option 1: Keycloak (Default)

**Deploy Keycloak in the stack:**

```yaml
global:
  authProvider:
    type: keycloak

keycloak:
  create: true  # Deploy Keycloak as part of this stack

keycloak-configure:
  enabled: true  # Run Keycloak configuration job
```

**Use an external Keycloak instance:**

```yaml
global:
  authProvider:
    type: keycloak
    keycloak:
      realm: mcp-gateway
      # REQUIRED when keycloak.create=false: base URL of your Keycloak,
      # reachable from inside the cluster. Without it the chart fails to
      # render — there is no in-cluster Keycloak Service to default to.
      internalUrl: https://your-keycloak.example.com

keycloak:
  create: false  # Don't deploy Keycloak

keycloak-configure:
  enabled: true  # Still configure the external Keycloak
```

The configure Job reads the Keycloak admin password from the Secret `<release>-keycloak` (key `admin-password`).
The bundled Keycloak generates that Secret; with an external instance, create it yourself with the admin password for the account in `keycloak-configure.keycloak.adminUser` — or set `keycloak-configure.enabled=false` and configure the realm manually.

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
```

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
```

Only the registry service exposes a public ingress. auth-server,
Keycloak, and mcpgw stay on their ClusterIP Services and are fronted by
the registry pod's in-cluster nginx reverse proxy — mcpgw's MCP tools
are reachable at `/airegistry-tools/*`, seeded into the registry at
startup. The Keycloak path (`/keycloak`) is pinned by the nginx config
— do NOT set `keycloak.httpRelativePath`; leave it at the Bitnami
default (`/`).

**DNS Requirements:** Configure a single A/CNAME record for your domain pointing to your ingress load balancer.

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

The username/password are displayed in the output of the `keycloak-configure job`

```bash
kubectl get pods -l job-name=setup-keycloak -n MYNAMESPACE
```

The output will look similar to:

```
NAME                   READY   STATUS      RESTARTS   AGE
setup-keycloak-d6g2r   0/1     Completed   0          29m
setup-keycloak-nnqgj   0/1     Error       0          31m
```

Use the pod name that completed successfully:

```
kubectl logs -n MYNAMESPACE setup-keycloak-d6g2r --tail 20
```

You will see the credentials in the output

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
