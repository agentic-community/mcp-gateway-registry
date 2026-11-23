# MCP Gateway Registry CDK Infrastructure

The CDK app under `infra/` deploys the registry to AWS as 6 stacks (TypeScript).
Mirrors the Terraform deployment in `terraform/aws-ecs/`.

> **Status: not actively maintained.** The supported AWS deployment is the Terraform stack in [`terraform/aws-ecs/`](../../terraform/aws-ecs/). This CDK stack lags behind it in features and configuration settings, and those gaps are not planned work. See the [known gaps](../README.md#known-gaps).

## Stacks

| Stack | File | Purpose |
| --- | --- | --- |
| `Registry-Network` | [registry-network-stack.ts](../lib/registry/registry-network-stack.ts) | VPC, subnets, NAT, VPC endpoints |
| `Registry-Data` | [registry-data-stack.ts](../lib/registry/registry-data-stack.ts) | DocumentDB + Aurora MySQL + RDS Proxy |
| `Registry-Auth` | [registry-auth-stack.ts](../lib/registry/registry-auth-stack.ts) | Keycloak ECS service + ALB + DNS |
| `Registry-Service` | [registry-service-stack.ts](../lib/registry/registry-service-stack.ts) | Registry/auth-server ECS, ALB, EFS, secrets, optional MCP servers/A2A agents, observability |
| `Registry-Ops` | [registry-ops-stack.ts](../lib/registry/registry-ops-stack.ts) | DocumentDB rotation Lambda + AWS-hosted Aurora rotation |
| `Registry-Cdn` | [registry-cdn-stack.ts](../lib/registry/registry-cdn-stack.ts) | CloudFront distributions + WAFv2 (no-ops when disabled) |
| `Registry-Build` | [registry-build-stack.ts](../lib/registry/registry-build-stack.ts) | ECR repos + CodeBuild (no-op when disabled) |

Dependencies: `Network ← Data ← Auth ← Service`; `Network,Data ← Ops`; `Service,Auth ← Cdn`; `Build` standalone.

## L3 constructs

`lib/registry/constructs/`:

- `documentdb-cluster.ts`, `keycloak-database.ts`: stateful data stores
- `keycloak-service.ts`: Keycloak ECS + ALB + Route53/ACM (full stack)
- `registry-network.ts`, `registry-alb.ts`, `registry-efs.ts`, `registry-secrets.ts`: service-stack building blocks
- `registry-ecs-service.ts`, `mcp-server-service.ts`: generic ECS Fargate L3s
- `observability-pipeline.ts`: AMP + Grafana + ADOT
- `cloudfront-distribution.ts`, `waf-rules.ts`, `codebuild-pipeline.ts`
- `secret-rotation.ts`: DocumentDB rotation Lambda (Aurora uses `secretsmanager.HostedRotation`)
- `registry-alarms.ts`: CloudWatch alarms (ECS CPU/memory, ALB unhealthy/5xx/latency, DocumentDB audit failures). No-op when `monitoring.enabled=false` or no `alarmEmail`/`alarmSnsTopicArn` set.
- `_lib.ts`: `putSecureSsmParam` shared helper

## Configuration

Edit [`infra/config.yaml`](../config.yaml). Defaults live in
`DEFAULT_REGISTRY_CONFIG` in [registry-config.ts](../lib/registry/registry-config.ts).

### Two ways a secret reaches a container

**By ARN, preferred.** The operator creates the secret, and config carries only
its ARN (`cognito.clientSecretArn`, `pingfederate.clientSecretArn`,
`pingfederate.m2mClientSecretArn`, `egress.credentialEncryptionKeyArn`). CDK
imports it, so no secret value passes through synthesis. Use this for anything
new.

**By value, legacy.** The remaining secrets come from `CDK_*` environment variables (see `SECRET_ENV_PATHS`). CDK creates a Secrets Manager secret from each value and writes the value into the template's `SecretString`. That means the value sits in plaintext in `cdk.out` and in the deployed CloudFormation stack, readable by anyone with `cloudformation:GetTemplate`. It affects the Keycloak admin password, the embeddings API key, all three Okta secrets, both Auth0 secrets, the Entra client secret, the Grafana admin password, the OTLP exporter headers, and the eight container credentials listed below. Treat `cdk.out` as secret-bearing until those move to ARN references. Terraform has the same exposure through `terraform.tfstate`.

Every secret reaches its container through the task definition's `secrets` block, so the task definition carries only the ARN. That includes `AUTH0_MANAGEMENT_API_TOKEN`, `REGISTRY_API_TOKEN`, `FEDERATION_STATIC_TOKEN`, `FEDERATION_ENCRYPTION_KEY`, `ANS_API_KEY` and `ANS_API_SECRET` (registry and auth-server), and `GITHUB_PAT` and `GITHUB_APP_PRIVATE_KEY` (registry only). Those eight were plain environment values until [#2063](https://github.com/agentic-community/mcp-gateway-registry/issues/2063), which stored them in the task definition. A credential left unset creates no secret and injects nothing.

When you upgrade a stack deployed before that change, the new task definition revision holds only ARNs, but the earlier revisions still hold the values and stay readable through `ecs:DescribeTaskDefinition`. Deregister and delete the earlier registry and auth-server revisions, then rotate any of the eight credentials you had set.

If a secret you reference by ARN is encrypted with a customer-managed KMS key,
grant the ECS task execution role `kms:Decrypt` on that key. CDK cannot
discover it from the secret ARN, so the task otherwise fails to start.

`CDK_GRAFANA_ADMIN_PASSWORD` is required whenever `enableObservability` is
true, which the shipped `config.yaml` sets. The secret is created from that
value and CloudFormation rejects a `SecretString` shorter than one character,
so an unset password used to synthesize cleanly and then fail the deploy. Synth
now refuses it alongside the other required secrets.

Each execution role is granted only the secrets its own containers receive.
mcpgw gets no secrets grant at all, and the metrics service and Grafana get
their own secrets rather than the whole bundle, so a role cannot read a
credential it is never handed.

### Ports

The container port is not always the port callers use. Service Connect injects
a proxy into every task in the namespace and binds each client alias on
loopback, so a service whose application listened on its own alias port would
collide with that proxy.

| Service | Container port | Callers use | Notes |
| --- | --- | --- | --- |
| registry | 8080 | `registry:8080` | nginx. The only port the ALB reaches. |
| registry | 8443, 7860 | task-internal | nginx TLS and the loopback-bound app process. |
| registry | 8091 | `registry:8091` | Internal egress-token vend. Never ALB-fronted; only the auth-server security group may reach it. |
| auth-server | 18888 | `auth-server:8888` | `AUTH_LISTEN_PORT=18888`. The security group opens 18888, since the proxy's ingress listener terminates on the container port. |
| mcpgw | 8003 | `mcpgw-server:8003` | Reached by the registry and by the auth-server's mcp-proxy hop. |

## Testing

```bash
cd infra
npm install
npm test        # jest + aws-cdk-lib/assertions, no AWS credentials needed
npm run build   # tsc, compiles to dist/
```

The suites in `test/` synthesize the real stack graph and assert on the
resulting CloudFormation template, so a value a config branch silently dropped
shows up as a failure. They cover the invariants that otherwise only surface as
a crash-looping task: the registry and auth-server reading the same nginx
marker secret, every injected secret being readable by the execution role, no
environment value rendering as the literal `undefined`, and the port table
above.

`test/registry-config.test.ts` covers the synth-time validation. Two groups
matter most:

**"values that reach an IAM policy Resource"** runs the same dangerous inputs
(`*`, `?`, `:`) against every config value that ends up inside an ARN in an IAM
policy. One wildcard in any of those positions widens the grant silently, and
the list in `registry-config.ts` above `_requireIamResourceSegment` names the
fields. Extend both together.

**The rest** pins the application's contract: an incomplete Cognito or
PingFederate block, PingFederate without the admin password its manager raises
without, an egress config on a non-Mongo storage backend, an unsupported
secret-store backend, a Cognito domain passed as a hostname where the
auth-server wants a prefix, and a comma-separated OBO audience list the
application would read as one value.

Deliberately accepted, because the application and Terraform both support
them: an empty Cognito domain, which the auth-server derives from the user pool
id, and an empty OBO audience allowlist, which falls back to a shape rule
admitting only an internal server's own IdP audience.

## Deploying

```bash
cd infra
./scripts/deploy.sh           # bootstrap + synth + deploy --all + post-deploy
./scripts/deploy.sh --help    # usage, required env vars
./scripts/deploy.sh --diff    # cdk diff
./scripts/deploy.sh --status  # CFN stack status
./scripts/deploy.sh --destroy # destroy --all
```

`scripts/deploy.sh` handles: prerequisite checks (aws/node/jq), AWS service-linked
roles (ECS + ELB), CDK bootstrap, npm install, synth, deploy, then runs
`scripts/post-deploy.sh` (Keycloak realm/clients/groups/users init via
ECS Exec + Admin API).

The `buildspec.yml` at the repo root is consumed by CodeBuild, so keep it next
to the Dockerfiles it builds.

## Adding a new service

1. Add image URI to `RegistryConfig.images` in `registry-config.ts`.
2. Instantiate `McpServerService` in `registry-service-stack.ts`.
3. Add ingress rule from registry SG (cross-stack pattern uses `CfnSecurityGroupIngress`).

## Parity with Terraform

The Terraform deployment under `terraform/aws-ecs/` is the reference implementation;
when adding to one, mirror the change in the other. The `infra-sync` skill
in `.claude/skills/infra-sync/` describes the diff process.
