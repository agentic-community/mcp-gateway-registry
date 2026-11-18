# MCP Gateway Registry: AWS CDK Stack

> **Status: not actively maintained.** The supported AWS deployment is the Terraform stack in [`terraform/aws-ecs/`](../terraform/aws-ecs/). This CDK stack lags behind it in features and configuration settings, and those gaps are not planned work.

This directory holds an AWS CDK app (TypeScript) that deploys the registry to Amazon ECS on Fargate as six stacks: `Registry-Network`, `Registry-Data`, `Registry-Auth`, `Registry-Ops`, `Registry-Service` and `Registry-Build`.

## Prerequisites

- AWS CLI v2 with administrator-equivalent IAM permissions
- Node.js 18 or later (CI uses 22), npm and `jq`
- Container images pushed to Amazon ECR, configured under `images:` in [`config.yaml`](config.yaml)

## Deploy

```bash
cd infra
export CDK_KEYCLOAK_ADMIN_PASSWORD="<...>"
export CDK_KEYCLOAK_DATABASE_PASSWORD="<...>"
export CDK_DOCUMENTDB_ADMIN_PASSWORD="<...>"
export CDK_GRAFANA_ADMIN_PASSWORD="<...>"   # required while enableObservability is true
./scripts/deploy.sh                          # bootstrap, synth, deploy all stacks, post-deploy
```

Settings live in [`config.yaml`](config.yaml). Run `./scripts/deploy.sh --help` for the remaining options.

## Test

```bash
cd infra
npm install
npm test     # synthesizes the stacks and asserts on the templates; no AWS credentials needed
```

## Known gaps

Compared with the Terraform stack, this one cannot set roughly 245 settings that Terraform passes to the services, so the application defaults apply. Several secrets are also written into the synthesized template by value, so treat `cdk.out` as secret-bearing ([#2063](https://github.com/agentic-community/mcp-gateway-registry/issues/2063)).

## More documentation

- [Architecture and configuration](docs/CDK-INFRASTRUCTURE.md)
- [Deployment guide and troubleshooting](docs/DEPLOYMENT-GUIDE.md)
- [WAF and MCP request flow](docs/WAF-AND-MCP-FLOW.md)
