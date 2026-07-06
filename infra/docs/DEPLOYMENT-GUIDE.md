# MCP Gateway Registry — Deployment Guide

End-to-end deployment, credentials, and troubleshooting for the CDK deployment.

## Quickstart

```bash
cd infra
export CDK_KEYCLOAK_ADMIN_PASSWORD="<...>"
export CDK_KEYCLOAK_DATABASE_PASSWORD="<...>"
export CDK_DOCUMENTDB_ADMIN_PASSWORD="<...>"
./scripts/deploy.sh   # bootstrap + synth + deploy --all + post-deploy
```

`deploy.sh --help` for full options. Architecture: see [CDK-INFRASTRUCTURE.md](CDK-INFRASTRUCTURE.md).

## Prerequisites

- AWS CLI v2 + Admin-equivalent IAM
- Node.js 18+, npm, `jq`
- Container images pushed to ECR (configured in [`config.yaml`](../config.yaml) under `images:`)

## Service URLs

After deploy, run `./scripts/deploy.sh --endpoints` (reads `cdk-outputs.json`).
Standard mapping:

| Service | URL |
|---|---|
| Registry UI / API / Health | `<registry-alb>/`, `/api/v1`, `/health` |
| Keycloak / Admin Console | `<keycloak-alb>/`, `/admin` |
| Grafana, Gradio | `<registry-alb>/grafana/`, `/gradio/` |

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
Auth-server log shows `Final mapped scopes: []`. Either `storageBackend` ≠ `documentdb`
in `config.yaml`, or `load-scopes.py` hasn't run. Re-run `post-deploy.sh`.

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
- HTTPS strict mode with HTTP ALB: ensure `KC_HOSTNAME_STRICT_HTTPS=false`, `KC_HTTP_ENABLED=true`
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
