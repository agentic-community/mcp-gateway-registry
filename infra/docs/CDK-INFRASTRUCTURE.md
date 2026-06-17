# MCP Gateway Registry — CDK Infrastructure

The CDK app under `infra/` deploys the registry to AWS as 7 stacks (TypeScript).
Mirrors the Terraform deployment in `terraform/aws-ecs/`.

## Stacks

| Stack | File | Purpose |
|---|---|---|
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

- `documentdb-cluster.ts`, `keycloak-database.ts` — stateful data stores
- `keycloak-service.ts` — Keycloak ECS + ALB + Route53/ACM (full stack)
- `registry-network.ts`, `registry-alb.ts`, `registry-efs.ts`, `registry-secrets.ts` — service-stack building blocks
- `registry-ecs-service.ts`, `mcp-server-service.ts` — generic ECS Fargate L3s
- `observability-pipeline.ts` — AMP + Grafana + ADOT
- `cloudfront-distribution.ts`, `waf-rules.ts`, `codebuild-pipeline.ts`
- `secret-rotation.ts` — DocumentDB rotation Lambda (Aurora uses `secretsmanager.HostedRotation`)
- `scopes-loader.ts` — Lambda + custom resource that upserts UI-Scope group docs into DocumentDB and copies `auth_server/scopes.yml` to EFS on every deploy. Source in [`infra/lambda/scopes-loader/`](../lambda/scopes-loader/). Bridges a gap with the upstream registry image, which only seeds `registry-admins`.
- `registry-alarms.ts` — CloudWatch alarms (ECS CPU/memory, ALB unhealthy/5xx/latency, DocumentDB audit failures). No-op when `monitoring.enabled=false` or no `alarmEmail`/`alarmSnsTopicArn` set.
- `_lib.ts` — `putSecureSsmParam` shared helper

## Configuration

Edit [`infra/config.yaml`](../config.yaml). Sensitive values come from env vars
prefixed `CDK_*` (see `SECRET_ENV_PATHS` in [registry-config.ts](../lib/registry/registry-config.ts)).
Defaults live in `DEFAULT_REGISTRY_CONFIG` in the same file.

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

The `buildspec.yml` at the repo root is consumed by CodeBuild — keep it next
to the Dockerfiles it builds.

## Adding a new service

1. Add image URI to `RegistryConfig.images` in `registry-config.ts`.
2. Instantiate `McpServerService` in `registry-service-stack.ts`.
3. Add ingress rule from registry SG (cross-stack pattern uses `CfnSecurityGroupIngress`).

## Parity with Terraform

The Terraform deployment under `terraform/aws-ecs/` is the reference implementation;
when adding to one, mirror the change in the other. The `infra-sync` skill
in `.claude/skills/infra-sync/` describes the diff process.
