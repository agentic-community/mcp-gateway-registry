# How do I configure MongoDB Atlas instead of MongoDB CE?

The registry storage layer accepts a full MongoDB connection string via the `MONGODB_CONNECTION_STRING` environment variable. When set, this URI takes precedence over the discrete `DOCUMENTDB_HOST` / `DOCUMENTDB_PORT` / `DOCUMENTDB_USERNAME` / `DOCUMENTDB_PASSWORD` / `DOCUMENTDB_DATABASE` variables, so you can point the registry at MongoDB Atlas, a self-managed replica set, or any other MongoDB-compatible service without changing code.

When the variable is empty or unset, the registry falls back to the `DOCUMENTDB_*` variables, so existing deployments keep working unchanged.

## Prerequisites

- A MongoDB Atlas cluster (free M0 tier is fine for dev) or any MongoDB-compatible service reachable from your deployment.
- A database user with read/write access to the target database (default: `mcp_registry`).
- The cluster's SRV connection string from the Atlas **Connect** dialog, which looks like:
  ```
  mongodb+srv://<username>:<password>@<cluster>.mongodb.net/mcp_registry?retryWrites=true&w=majority
  ```
- For Atlas: your deployment's egress IP(s) or `0.0.0.0/0` added to the Atlas Network Access list.

> **Security tip:** Atlas URIs embed the database user password. For any deployment type that persists configuration (Terraform state, Helm values files, committed `.env`), prefer the secret-reference variant described below over the plain URI.

## How the URI is built

The same value works across every deployment type. Build it once and reuse it.

| Part | Example | Notes |
|------|---------|-------|
| Scheme | `mongodb+srv://` | Atlas uses `mongodb+srv` (SRV + TLS). Self-managed replica sets may use `mongodb://` with explicit host list. |
| Credentials | `user:password@` | URL-encode special characters in the password (`@` → `%40`, `:` → `%3A`, etc.). |
| Host | `cluster0.abc123.mongodb.net` | Atlas gives you this in the Connect dialog. |
| Database | `/mcp_registry` | Must match `DOCUMENTDB_DATABASE` (or leave blank and rely on `defaultauthdb` / the URI). |
| Options | `?retryWrites=true&w=majority` | URI-owned. When `MONGODB_CONNECTION_STRING` is set, the registry does not layer any additional client options on top. |

When `MONGODB_CONNECTION_STRING` is set, the `DOCUMENTDB_*` variables are ignored for the connection (including `DOCUMENTDB_USE_TLS`, `DOCUMENTDB_REPLICA_SET`, `DOCUMENTDB_DIRECT_CONNECTION`). TLS, retryWrites, replicaSet, etc. must all be expressed in the URI itself — which is exactly why this knob exists.

---

## Deployment Type 1: Docker Compose (local or single-host)

Edit your `.env` file (copy from [`.env.example`](../../.env.example) if you haven't already):

```bash
# Keep STORAGE_BACKEND=mongodb-ce (the URI backend is the same code path).
STORAGE_BACKEND=mongodb-ce

# Leave DOCUMENTDB_* set for ops scripts that still read them individually,
# but they will be ignored for the live connection when the URI is set.
DOCUMENTDB_HOST=mongodb
DOCUMENTDB_PORT=27017
DOCUMENTDB_DATABASE=mcp_registry
DOCUMENTDB_USERNAME=admin
DOCUMENTDB_PASSWORD=admin
DOCUMENTDB_USE_TLS=false
DOCUMENTDB_NAMESPACE=default

# Your Atlas connection string — this takes precedence.
MONGODB_CONNECTION_STRING=mongodb+srv://mcp_user:<urlencoded-password>@cluster0.abc123.mongodb.net/mcp_registry?retryWrites=true&w=majority
```

Then restart the registry container so it picks up the new env value:

```bash
docker compose up -d registry
```

> **Note:** `docker compose restart` does NOT re-read `.env`. You must use `up -d` (or `down` + `up -d`) to rebuild the container with fresh environment variables.

Verify the override is active:

```bash
docker logs mcp-gateway-registry-registry-1 --since 30s | grep -E 'connection string override|Connected to'
```

You should see:

```
Waiting for MongoDB via connection string override...
Connecting to mongodb-ce via connection string override
Connected to DocumentDB/MongoDB 8.x.x
```

You can also delete or not deploy the local `mongodb` service in [`docker-compose.yml`](../../docker-compose.yml) if you only want Atlas — the `docker-compose.yml` bundles a local MongoDB container for convenience, but the registry code doesn't care where the connection points.

---

## Deployment Type 2: AWS ECS via Terraform

The ECS Terraform module accepts two new variables (see [`terraform/aws-ecs/variables.tf`](../../terraform/aws-ecs/variables.tf)):

| Variable | Type | Purpose |
|----------|------|---------|
| `mongodb_connection_string` | string (sensitive) | Plain-text URI. Fine for non-sensitive URIs or dev. The value will be stored in Terraform state. |
| `mongodb_connection_string_secret_arn` | string | ARN of a Secrets Manager secret whose value is the full URI. **Preferred** when the URI contains credentials — the value never lands in Terraform state, and ECS pulls it at container start. |

### Option A: Secrets Manager (recommended for Atlas)

Create the secret first:

```bash
aws secretsmanager create-secret \
  --name mcp-registry/mongodb-atlas-uri \
  --secret-string 'mongodb+srv://mcp_user:<urlencoded-password>@cluster0.abc123.mongodb.net/mcp_registry?retryWrites=true&w=majority' \
  --region us-east-1
```

Then in [`terraform/aws-ecs/terraform.tfvars`](../../terraform/aws-ecs/terraform.tfvars):

```hcl
mongodb_connection_string_secret_arn = "arn:aws:secretsmanager:us-east-1:123456789012:secret:mcp-registry/mongodb-atlas-uri-abc123"
```

Leave `mongodb_connection_string = ""` (the default). The task definition will inject the URI as the `MONGODB_CONNECTION_STRING` env var pulled from Secrets Manager at container start.

Apply:

```bash
cd terraform/aws-ecs
terraform apply
```

### Option B: Plain variable (dev / non-sensitive)

```hcl
mongodb_connection_string = "mongodb+srv://mcp_user:<urlencoded-password>@cluster0.abc123.mongodb.net/mcp_registry?retryWrites=true&w=majority"
```

Same effect but the URI lives in Terraform state. Only do this for dev stacks or URIs without embedded credentials.

### IAM + network notes for Atlas

- The ECS task role does **not** need any extra IAM permissions to reach Atlas — Atlas is an external HTTPS endpoint, not an AWS service.
- If using Secrets Manager, the task execution role already has `secretsmanager:GetSecretValue` granted by the module — no policy change needed.
- Add your VPC's NAT Gateway EIPs (or the VPC CIDR if using PrivateLink / VPC Peering) to the Atlas Network Access list.
- You can delete or skip the `aws_docdb_cluster` resources if you are exclusively using Atlas — but leaving them doesn't hurt because the registry won't connect to the DocDB cluster when the URI override is set.

### Verifying on ECS

After `terraform apply` finishes and the task restarts, inspect the container's env vars via the AWS console (ECS > Task > Containers > Environment variables, or Secrets) to confirm `MONGODB_CONNECTION_STRING` is present. Then tail CloudWatch logs for the registry service:

```
Waiting for MongoDB via connection string override...
Connecting to mongodb-ce via connection string override
Connected to DocumentDB/MongoDB 8.x.x
```

---

## Deployment Type 3: Kubernetes / EKS via Helm

The [Helm chart](../../charts/mcp-gateway-registry-stack/) exposes the connection string through its values file. Create (or append to) a values override file, for example `values-atlas.yaml`:

```yaml
registry:
  env:
    # Your Atlas URI here. For production, load it from a Kubernetes Secret
    # via envFrom (see below) instead of putting it directly in values.
    MONGODB_CONNECTION_STRING: "mongodb+srv://mcp_user:<urlencoded-password>@cluster0.abc123.mongodb.net/mcp_registry?retryWrites=true&w=majority"
```

### Recommended: pull the URI from a Kubernetes Secret

Plain values files get committed to git. For any URI that contains credentials, store the URI in a Kubernetes Secret and reference it from the pod's env:

```bash
kubectl create secret generic mcp-registry-mongodb \
  --from-literal=MONGODB_CONNECTION_STRING='mongodb+srv://mcp_user:<urlencoded-password>@cluster0.abc123.mongodb.net/mcp_registry?retryWrites=true&w=majority' \
  -n mcp-registry
```

Then in your `values-atlas.yaml`:

```yaml
registry:
  envFrom:
    - secretRef:
        name: mcp-registry-mongodb
```

(The exact key under `registry:` depends on the chart version — check the chart's default `values.yaml` for the `env` / `envFrom` keys. If your chart version predates the envFrom wiring, use `env:` with a `valueFrom.secretKeyRef`.)

### Upgrade the release

```bash
helm upgrade mcp-registry charts/mcp-gateway-registry-stack/ \
  -n mcp-registry \
  -f values-atlas.yaml
```

### Verifying on Kubernetes

Wait for the rollout, then grep the pod logs:

```bash
kubectl -n mcp-registry logs deployment/mcp-registry --since 1m | grep -E 'connection string override|Connected to'
```

Same three lines as the Docker Compose and ECS cases.

### Optional: skip the in-cluster MongoDB

The default chart provisions an in-cluster MongoDB StatefulSet for dev. If you are using Atlas exclusively, set:

```yaml
mongodb:
  enabled: false
```

to avoid running an unused pod and PVC.

---

## Troubleshooting

### The override isn't being picked up

**Most common cause:** the container wasn't recreated after you edited the env source.

- Docker Compose: `docker compose restart` does NOT re-read `.env`. Use `docker compose up -d registry` (or `down` + `up -d`).
- ECS: `terraform apply` should roll the task. If it didn't, force a new deployment: `aws ecs update-service --cluster ... --service ... --force-new-deployment`.
- Kubernetes: a Helm upgrade should trigger a rollout, but a ConfigMap/Secret value change without a pod spec change won't. Restart with `kubectl -n mcp-registry rollout restart deployment/mcp-registry`.

Confirm the env var is actually inside the container:

```bash
# Compose
docker exec mcp-gateway-registry-registry-1 env | grep MONGODB_CONNECTION_STRING

# ECS (via exec, if enabled)
aws ecs execute-command --cluster ... --task ... --container registry --interactive --command "env | grep MONGODB"

# Kubernetes
kubectl -n mcp-registry exec deployment/mcp-registry -- env | grep MONGODB_CONNECTION_STRING
```

### Connection fails with "authentication failed"

- URL-encode special characters in the password. `@`, `:`, `/`, `?`, `#`, `[`, `]` all need encoding.
- Confirm the database user exists and has access to the database named in the URI.
- In Atlas, confirm the user's built-in role includes `readWrite` on `mcp_registry` (or whichever database you target).

### Connection fails with "connection timed out" or "no servers found"

- For Atlas: add the deployment's egress IP (or VPC CIDR) to the Atlas Network Access list. From ECS, that's the NAT Gateway EIPs; from EKS, the worker node egress IPs.
- For replica sets: make sure the URI is a `mongodb+srv://` SRV record OR an explicit comma-separated host list, and the replica set name matches what the server advertises.

### TLS errors

When `MONGODB_CONNECTION_STRING` is set, `DOCUMENTDB_USE_TLS` and `DOCUMENTDB_TLS_CA_FILE` are ignored. Express TLS settings in the URI instead:

- Atlas (`mongodb+srv://`) implies TLS automatically; no `tls=true` needed.
- Self-managed with a custom CA: `mongodb://...&tls=true&tlsCAFile=/path/inside/container/ca.pem` (and mount the CA file into the container; the path must exist inside the container).

## Related

- [`.env.example`](../../.env.example) — full variable reference with an inline example.
- [`terraform/aws-ecs/terraform.tfvars.example`](../../terraform/aws-ecs/terraform.tfvars.example) — ECS variable examples.
- [Configuration Guide](../configuration.md) — all registry configuration parameters.
