# MCP Gateway Registry — Deployment Guide

End-to-end deployment, credentials, and troubleshooting for the CDK deployment.

> **Status: not actively maintained.** The supported AWS deployment is the Terraform stack in [`terraform/aws-ecs/`](../../terraform/aws-ecs/). This CDK stack lags behind it in features and configuration settings, and those gaps are not planned work. See the [known gaps](../README.md#known-gaps).

## Quickstart

```bash
cd infra
export CDK_KEYCLOAK_ADMIN_PASSWORD="<...>"
export CDK_KEYCLOAK_DATABASE_PASSWORD="<...>"
export CDK_DOCUMENTDB_ADMIN_PASSWORD="<...>"
export CDK_GRAFANA_ADMIN_PASSWORD="<...>"   # required while enableObservability is true
./scripts/deploy.sh   # bootstrap + synth + deploy --all + post-deploy
```

`deploy.sh --help` for full options. Architecture: see [CDK-INFRASTRUCTURE.md](CDK-INFRASTRUCTURE.md).

## Prerequisites

- AWS CLI v2 + Admin-equivalent IAM
- Node.js 18+, npm, `jq`
- Container images pushed to ECR (configured in [`config.yaml`](../config.yaml) under `images:`)

Keycloak defaults to the official `quay.io/keycloak/keycloak:26.8.2` image and
retains Aurora MySQL. Set `images.keycloak` to an explicit fixed tag or digest for
a custom image. The ECS task supplies database and hostname settings at startup.
It uses `KC_BOOTSTRAP_ADMIN_USERNAME` and `KC_BOOTSTRAP_ADMIN_PASSWORD` for initial
administrator creation, and `KC_LOG_LEVEL` for logging.

CDK selects PingFederate, then Auth0, Okta, Entra, Cognito and Keycloak from the existing provider settings, matching Terraform's precedence.
When another provider wins, CDK omits Keycloak ECS, ALB listeners and rules, CloudFront, DNS and certificates.
Registry and auth-server receive `KEYCLOAK_ENABLED=false`, blank Keycloak URLs and no Keycloak secrets.
Inactive Keycloak HTTPS and admin-CIDR settings do not block synth.

Provider changes keep Aurora, its proxy and rotation, bootstrap SSM parameters,
and application credentials at their existing resource addresses. Keep supplying
the Keycloak admin and database passwords because those retained resources still
use them. Post-deploy setup skips Keycloak for other providers; `validate.sh`
checks infrastructure and skips Keycloak login and authenticated CRUD checks.

For an existing Keycloak deployment, remove cross-stack consumers before their producers. Under the new provider configuration, deploy `Registry-Service` with `--exclusively`, then `Registry-Auth` with `--exclusively`, then all stacks. The first step releases the service stack's imports from Auth; the second releases Auth's NAT-IP imports from Network. Inspect the synthesized diff first. A single dependency-ordered deployment can fail because CloudFormation cannot delete an export that a deployed consumer still imports.

## Service URLs

After deploy, run `./scripts/deploy.sh --endpoints` (reads `cdk-outputs.json`).
Standard mapping:

| Service | URL |
|---|---|
| Registry UI / API / Health | `<registry-alb>/`, `/api/v1`, `/health` |
| Keycloak / Admin Console | `<keycloak-alb>/`, `/admin` |
| Grafana, Gradio | `<registry-alb>/grafana/`, `/gradio/` |

### Keycloak ALB path routing

The Keycloak ALB forwards `/realms/mcp-gateway`, `/realms/mcp-gateway/*`,
`/resources/*` and `/` for any source its security groups admit. In direct-ALB
mode, admin and master-realm paths also require an approved source IP. Other
paths return 404.

`keycloakAdminCidrBlocks` in [`config.yaml`](../config.yaml) defaults to `null`,
which uses `ingressCidrBlocks`. Set it to your operator networks to restrict
`/admin`, `/admin/*`, `/realms/master` and `/realms/master/*`. Include the machine
that runs setup and user-management scripts. CDK adds the VPC NAT gateway egress
IPs for the registry's admin calls. The HTTPS security group admits the union of
ordinary ingress and admin CIDRs.

CloudFront hides the client IP from the ALB. In CloudFront mode, admin paths have
no source-IP condition, and synth rejects an admin list that differs from
`ingressCidrBlocks`. Each admin rule accepts up to three CIDRs. CDK rejects lists
that would exceed the default 100-rule listener quota.

Use the dedicated Keycloak URL for administration. Do not expose admin or
master-realm routes through the registry gateway.

## Credentials

| Where | Source |
|---|---|
| Keycloak admin | SSM `/keycloak/admin_password` (= `CDK_KEYCLOAK_ADMIN_PASSWORD`) |
| Registry users (`admin`, `lob1-user`, `lob2-user`) | Created by `keycloak/setup/init-keycloak.sh`. All passwords are required env vars (no defaults) and are set temporary (reset forced on first login): admin = `INITIAL_ADMIN_PASSWORD`, LOB users = `LOB1_USER_PASSWORD`/`LOB2_USER_PASSWORD` |
| Grafana admin | env `CDK_GRAFANA_ADMIN_PASSWORD` (default `admin`) |
| DocumentDB | Secrets Manager `mcp-gateway/documentdb/credentials` |
| Aurora MySQL | Secrets Manager `keycloak/database` |
| OAuth client secrets | Secrets Manager `mcp-gateway-keycloak-client-secret`, `mcp-gateway-keycloak-m2m-client-secret` |

Retrieve a secret:
```bash
aws secretsmanager get-secret-value --secret-id <id> --region "$AWS_REGION" --query SecretString --output text
```

## Environment Variables

**Required:** `CDK_KEYCLOAK_ADMIN_PASSWORD`, `CDK_KEYCLOAK_DATABASE_PASSWORD`, `CDK_DOCUMENTDB_ADMIN_PASSWORD`.

**Optional:** `AWS_REGION` (default `us-east-1`), `CDK_GRAFANA_ADMIN_PASSWORD`,
`CDK_EMBEDDINGS_API_KEY`, `CDK_{ENTRA,OKTA,AUTH0}_CLIENT_SECRET`,
`CDK_{OKTA,AUTH0}_M2M_CLIENT_SECRET`, `CDK_GITHUB_PAT`, `CDK_OTEL_EXPORTER_OTLP_HEADERS`.

**Password rules:** ≥ 8 printable-ASCII chars; no `/`, `@`, `"`, or spaces (DocumentDB/RDS restrictions).

Full list: see `SECRET_ENV_PATHS` in [`registry-config.ts`](../lib/registry/registry-config.ts).

## Common ops

```bash
./scripts/deploy.sh --status        # CFN stack status
./scripts/deploy.sh --diff          # cdk diff
./scripts/deploy.sh --endpoints     # service URLs
./scripts/deploy.sh --validate      # end-to-end smoke test (CRUD an agent)
./scripts/deploy.sh --destroy       # destroy --all (interactive confirm)
./scripts/deploy.sh --stack <Name>  # deploy / destroy a single stack

# Force-restart all ECS services (e.g. after secret update)
for svc in $(aws ecs list-services --cluster mcp-gateway-ecs-cluster --query 'serviceArns[]' --output text); do
  aws ecs update-service --cluster mcp-gateway-ecs-cluster --service "$svc" --force-new-deployment >/dev/null
done
```

## Agent Registration

Get a token and register:

```bash
TOKEN=$(curl -s -X POST "$KEYCLOAK_URL/realms/mcp-gateway/protocol/openid-connect/token" \
  -d "client_id=mcp-gateway-m2m&client_secret=$M2M_SECRET&grant_type=client_credentials&scope=openid" \
  | jq -r .access_token)

curl -X POST "$REGISTRY_URL/api/agents/register" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"name":"my-agent","url":"http://example.com","supportedProtocol":"a2a","visibility":"public"}'
```

Required fields: `name`, `url`, `supportedProtocol` (`a2a` | `other`),
`visibility` (`public` | `private` | `group-restricted`).
Token TTL: 5 min. Registered agents start **disabled** by design — enable via
`POST /api/agents/<path>/toggle?enabled=true` (needs `toggle_service` permission,
i.e. group `mcp-registry-admin`).

## Troubleshooting

### Agent registration returns 500
Nginx wraps non-200 from auth-server/registry as HTML 500. Likely causes (in order):
1. Token expired (5-min TTL)
2. Missing `supportedProtocol` or invalid `visibility`
3. Token user not in `mcp-registry-admin` Keycloak group (no `publish_agent`)
4. Scopes not loaded — re-run `post-deploy.sh`

Debug: `aws logs tail /ecs/mcp-gateway-auth-server --since 5m` and `/ecs/mcp-gateway-registry`.

### Empty scopes / "permission denied" after login
Auth-server log shows `Final mapped scopes: []`. `storageBackend` in `config.yaml`
must be `documentdb`, and the DocumentDB `mcp_scopes_*` collection must be seeded
by the mongodb-configure init job.

### `oauth2_callback_failed` on login
Auth-server still has the placeholder Keycloak client secret. Run `post-deploy.sh`,
or check:
```bash
aws secretsmanager get-secret-value --secret-id mcp-gateway-keycloak-client-secret --query SecretString --output text
```
If it shows `placeholder-will-be-updated-by-init-script`, the post-deploy step
to update Secrets Manager didn't run. Re-run, then force-restart services (see Common ops).

### Keycloak 404 on `/realms/mcp-gateway`
Realm not created — re-run `post-deploy.sh` (which calls `keycloak/setup/init-keycloak.sh`).

### Keycloak ECS tasks restart in a loop
Check `aws logs tail /ecs/keycloak --since 30m`. Usual suspects:
- DB connectivity: SG ingress from Keycloak ECS to Aurora on 3306
- For an HTTP ALB upstream, keep `KC_HTTP_ENABLED=true` and the public Keycloak URL on HTTPS. CDK pins `KC_HOSTNAME` and sets `KC_HOSTNAME_BACKCHANNEL_DYNAMIC=false`, so tokens and discovery use the canonical URL. Registry, auth-server and A2A agents receive it as `KEYCLOAK_EXTERNAL_URL`.
- Admin password contains forbidden chars

### DocumentDB / RDS password validation failure
`CDK_DOCUMENTDB_ADMIN_PASSWORD` (and Keycloak DB password) must be ≥8 printable-ASCII
chars, no `/`, `@`, `"`, or spaces. Re-export and redeploy.

### Viewing ECS task logs
```bash
aws logs tail /ecs/<service> --since 30m --region "$AWS_REGION"
# services: mcp-gateway-registry, mcp-gateway-auth-server, keycloak,
#           mcp-gateway-grafana, mcp-gateway-metrics-service
```
