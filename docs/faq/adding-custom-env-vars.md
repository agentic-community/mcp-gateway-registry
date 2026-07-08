# How do I add custom environment variables to the registry, auth-server, or mcpgw services?

## Question

I need to pass a custom environment variable (a feature flag, an external API key, a per-deployment override) into one of the three gateway services without forking the Helm chart, the Terraform module, or the Docker Compose files. How do I do this for each of the three deployment surfaces?

## Answer

Each of the three deployment surfaces (Docker Compose, Terraform / AWS ECS, Helm / Kubernetes) exposes a first-class extension point for injecting custom environment variables. All three share the same reserved-name lists at `charts/<subchart>/reserved-env-names.txt`, so reserved variables that the chart manages via its own canonical settings cannot be shadowed accidentally.

The user-facing shape of each surface follows its native convention: `.env`-style files for Docker, HCL lists for Terraform, YAML lists for Helm.

## Surface 1: Docker Compose

### Where to put the variables

Create `extra_env/` at the repo root (next to `.env`) with one file per service:

```bash
mkdir -p extra_env

cat > extra_env/registry.env << 'EOF'
MY_FEATURE_FLAG=true
CUSTOM_TIMEOUT=30
EOF

cat > extra_env/auth-server.env << 'EOF'
CUSTOM_AUDIT_DESTINATION=s3://my-bucket/audit
EOF

cat > extra_env/mcpgw.env << 'EOF'
UPSTREAM_TIMEOUT_MS=5000
EOF
```

Files are `KEY=VALUE` per line; blank lines and `#`-prefixed comments are allowed. Any service whose file is missing is skipped (`env_file: { required: false }`), so you only need to create the files for services you actually want to customize.

### Override the default location

Both the preflight validator and the Compose files resolve the directory from `MCP_EXTRA_ENV_DIR`, falling back to `./extra_env` when unset:

```bash
# Point both the validator and compose at a shared-host location
export MCP_EXTRA_ENV_DIR=/etc/mcp-gateway/extra_env
./build_and_run.sh
```

### Start the stack

```bash
./build_and_run.sh
```

`build_and_run.sh` runs a preflight validator (`scripts/validate-extra-env.sh`) before starting containers. The validator:

- Rejects any reserved name (chart-managed) with the service, filename, and line number.
- Warns on malformed lines (missing `=` or empty key) without failing.
- Compares names case-insensitively so `secret_key=foo` is still caught.
- Collects every collision across all three services before exiting, so you fix them in one pass.
- Logs a summary line with the per-service custom-variable count.

### Verify inside the container

```bash
docker exec mcp-gateway-registry-registry-1 env | grep MY_FEATURE_FLAG
# MY_FEATURE_FLAG=true
```

### Requirements

- **Docker Compose v2.24+** (Docker Desktop 4.27+) or **Podman Compose v1.0.7+**. Earlier versions silently ignore `required: false` and error out when the file is missing. Check with `docker compose version`.

## Surface 2: Terraform / AWS ECS

### Where to put the variables

Edit `terraform/aws-ecs/terraform.tfvars` and add one list per service:

```hcl
registry_extra_env = [
  { name = "MY_FEATURE_FLAG", value = "true" },
  { name = "CUSTOM_TIMEOUT_SECONDS", value = "30" },
]

auth_server_extra_env = [
  { name = "CUSTOM_AUDIT_DESTINATION", value = "s3://my-bucket/audit" },
]

mcpgw_extra_env = [
  { name = "UPSTREAM_TIMEOUT_MS", value = "5000" },
]
```

Every entry is an HCL object with `name` and `value` string fields. All three variables default to `[]`, so you only set the ones you need.

### Sensitive values

The three variables are marked `sensitive = true`. Terraform will redact the values from `plan`/`apply` output and from the rendered `container_definitions` JSON of the affected task definitions. This prevents accidental leakage into CI logs. However, the values still live in plaintext in `terraform.tfvars` and in Terraform state — for production secrets, prefer the AWS Secrets Manager ARN pattern used by `mongodb_connection_string_secret_arn` over plaintext `*_extra_env`.

### Collision check

At `terraform plan` time, each `*_extra_env` variable runs a validation block that reads `charts/<subchart>/reserved-env-names.txt` via `file()` and rejects reserved names via `contains()`. A collision looks like:

```text
Error: Invalid value for variable
  on terraform.tfvars line 1:
   1: registry_extra_env = [
   2:   { name = "SECRET_KEY", value = "attacker-value" }
   3: ]

registry_extra_env contains one or more reserved environment variable names
that are managed by the chart. See charts/registry/reserved-env-names.txt
for the full list. Configure reserved variables via their canonical
Terraform variable or Helm value instead.
```

### Apply and verify

```bash
cd terraform/aws-ecs
terraform plan
terraform apply

# After apply:
aws ecs describe-task-definition \
  --task-definition <your-registry-task-def> \
  --query 'taskDefinition.containerDefinitions[0].environment'
```

You can also read the values back via `terraform console`:

```bash
echo 'nonsensitive(var.registry_extra_env)' | terraform console
```

## Surface 3: Helm / Kubernetes

### Where to put the variables

In the values file you pass to `helm install` / `helm upgrade` (typically a customer-specific `values-<env>.yaml`), set `extraEnv` on each subchart:

```yaml
registry:
  extraEnv:
    - name: MY_FEATURE_FLAG
      value: "true"
    - name: CUSTOM_TIMEOUT
      value: "30"
  # extraEnvFrom for pulling from an existing Secret or ConfigMap
  extraEnvFrom:
    - secretRef:
        name: my-external-secrets

auth-server:
  extraEnv:
    - name: CUSTOM_AUDIT_DESTINATION
      value: "s3://my-bucket/audit"

mcpgw:
  extraEnv:
    - name: UPSTREAM_TIMEOUT_MS
      value: "5000"
```

`extraEnv` takes the standard Kubernetes env var shape, so `valueFrom` is supported for referencing Secrets/ConfigMaps:

```yaml
registry:
  extraEnv:
    - name: MY_EXTERNAL_API_KEY
      valueFrom:
        secretKeyRef:
          name: my-secret
          key: api-key
```

### Collision check

At `helm template`/`install`/`upgrade` time, each subchart's `validateExtraEnv` helper in `_helpers.tpl` fails rendering if `extraEnv` contains a reserved name, a duplicate name, or an entry missing the required `name` field:

```text
Error: template: registry/templates/deployment.yaml:11:6: executing
"registry/templates/deployment.yaml" at <include "registry.validateExtraEnv" .>:
error calling include: template: registry/templates/_helpers.tpl:N:N:
executing "registry.validateExtraEnv" at <fail ...>: error calling fail:
registry.extraEnv[0]: "SECRET_KEY" is a reserved variable managed by the
chart (via env: or envFrom from the chart's secrets/configmaps). Remove it
from extraEnv. If a values.yaml field controls it (e.g. app.showSkillsTab
for SHOW_SKILLS_TAB), set that instead; otherwise the value is managed by
the chart's internal secrets and must not be overridden via extraEnv.
```

### Install and verify

```bash
helm upgrade --install mcp-gateway charts/mcp-gateway-registry-stack \
  -f values-prod.yaml

# After install:
kubectl describe pod -l app.kubernetes.io/component=registry | grep MY_FEATURE_FLAG
```

## What is a "reserved" name?

Reserved names are the environment variables the chart itself sets on the container, either directly in the Deployment's `env:` block or indirectly via `envFrom` from a chart-managed ConfigMap or Secret. Overriding one of these via `extraEnv` (or its Docker / Terraform equivalent) would either:

1. Shadow a chart-managed secret with a plaintext user-supplied value, or
2. Bypass a chart-managed feature flag whose canonical setting is a values.yaml / tfvars field.

The three reserved-name lists are the shared source of truth across all three surfaces:

- [`charts/registry/reserved-env-names.txt`](../../charts/registry/reserved-env-names.txt)
- [`charts/auth-server/reserved-env-names.txt`](../../charts/auth-server/reserved-env-names.txt)
- [`charts/mcpgw/reserved-env-names.txt`](../../charts/mcpgw/reserved-env-names.txt)

CI (`.github/workflows/helm-test.yml` → `reserved-list-sync`) renders the stack chart and fails the build if any name the chart actually injects is missing from the corresponding reserved list, so the lists cannot silently drift out of sync with the templates.

## When should I NOT use extra_env?

`extra_env` is for customizations the chart does not expose yet. If there is already a dedicated values field, `.env` variable, or Terraform variable for the setting you want to change, use that instead — those are the documented, supported extension points and they survive chart upgrades. `extra_env` should be the escape hatch, not the primary configuration mechanism.

## Related documentation

- [Unified Parameter Reference](../unified-parameter-reference.md) — surface-by-surface parameter mapping (see Group 29 for extra_env).
- [`CLAUDE.md`](../../CLAUDE.md) "Deployment Surface Customization" section — detailed notes for maintainers.
- [Issue #1000](https://github.com/agentic-community/mcp-gateway-registry/issues/1000) — design rationale.
