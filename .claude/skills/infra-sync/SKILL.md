---
name: infra-sync
description: "Keep Terraform and CDK infrastructure in sync. Compares terraform/aws-ecs/ and infra/ to identify discrepancies in ECS services, networking, storage, secrets, IAM, and configuration. Reports parity gaps and generates fixes for the target IaC tool. Use when changes are made to either Terraform or CDK infrastructure."
license: Apache-2.0
metadata:
  author: mcp-gateway-registry
  version: "1.0"
---

# Infrastructure Sync Skill

Use this skill when changes are made to either the Terraform (`terraform/aws-ecs/`) or CDK (`infra/`) infrastructure and the other needs to be updated to maintain parity.

## When to Use

- After modifying Terraform files in `terraform/aws-ecs/`
- After modifying CDK files in `infra/lib/registry/`
- When asked to check if Terraform and CDK are in sync
- Before deploying to verify both IaC tools produce equivalent infrastructure

## Input Modes

1. **Full Sync Check** - Compare all resources across both IaC tools
2. **Targeted Sync** - Compare a specific area (e.g., "sync ECS services" or "sync networking")
3. **Propagate Change** - User made changes to one tool and wants the other updated

## Workflow

### Step 1: Identify the Source of Truth

Ask the user (or determine from context):
- Which IaC tool was just modified? (Terraform or CDK)
- Is this a full sync check or targeted to a specific area?

### Step 2: Read Both Configurations

Read the relevant files from both Terraform and CDK based on the comparison area.

#### File Mapping (Terraform to CDK)

| Comparison Area | Terraform Files | CDK Files |
|----------------|-----------------|-----------|
| **ECS Services** | `terraform/aws-ecs/modules/mcp-gateway/ecs-services.tf` | `infra/lib/registry/registry-service-stack.ts` |
| **Networking** | `terraform/aws-ecs/modules/mcp-gateway/networking.tf`, `terraform/aws-ecs/vpc.tf` | `infra/lib/registry/registry-network-stack.ts`, `infra/lib/registry/registry-service-stack.ts` (ALB section) |
| **Storage (EFS)** | `terraform/aws-ecs/modules/mcp-gateway/storage.tf` | `infra/lib/registry/registry-service-stack.ts` (EFS section) |
| **Storage (DocumentDB)** | `terraform/aws-ecs/documentdb.tf` | `infra/lib/registry/registry-data-stack.ts`, `infra/lib/registry/constructs/documentdb-cluster.ts` |
| **Secrets** | `terraform/aws-ecs/modules/mcp-gateway/secrets.tf` | `infra/lib/registry/registry-service-stack.ts` (secrets section) |
| **IAM** | `terraform/aws-ecs/modules/mcp-gateway/iam.tf` | `infra/lib/registry/constructs/registry-ecs-service.ts` (task roles) |
| **Keycloak** | `terraform/aws-ecs/keycloak-ecs.tf`, `terraform/aws-ecs/keycloak-*.tf` | `infra/lib/registry/registry-auth-stack.ts`, `infra/lib/registry/constructs/keycloak-service.ts` |
| **Observability** | `terraform/aws-ecs/modules/mcp-gateway/observability.tf` | `infra/lib/registry/constructs/observability-pipeline.ts` |
| **CloudFront/WAF** | `terraform/aws-ecs/cloudfront.tf`, `terraform/aws-ecs/waf.tf` | `infra/lib/registry/registry-cdn-stack.ts`, `infra/lib/registry/constructs/cloudfront-distribution.ts`, `infra/lib/registry/constructs/waf-rules.ts` |
| **Secret Rotation** | `terraform/aws-ecs/secret-rotation.tf` | `infra/lib/registry/registry-ops-stack.ts`, `infra/lib/registry/constructs/secret-rotation.ts` |
| **CI/CD** | `terraform/aws-ecs/codebuild.tf` | `infra/lib/registry/registry-build-stack.ts`, `infra/lib/registry/constructs/codebuild-pipeline.ts` |
| **Variables/Config** | `terraform/aws-ecs/variables.tf`, `terraform/aws-ecs/modules/mcp-gateway/variables.tf` | `infra/lib/registry/registry-config.ts`, `infra/config.yaml` |
| **Monitoring** | `terraform/aws-ecs/cloudwatch-alarms.tf`, `terraform/aws-ecs/modules/mcp-gateway/monitoring.tf` | `infra/lib/registry/registry-service-stack.ts` (monitoring section) |

### Step 3: Compare Resource by Resource

For each comparison area, check the following attributes:

#### ECS Services Comparison Checklist

For each service (registry, auth-server, MCP servers, A2A agents):

- [ ] **Container image** - Same ECR repository and tag
- [ ] **CPU/Memory** - Same task definition resources
- [ ] **Port mappings** - Same container ports
- [ ] **Environment variables** - Same env vars passed to container
  - Check every env var key and value source
  - Flag any env var present in one but not the other
- [ ] **Secrets** - Same Secrets Manager references
  - Check JSON key extraction syntax (`:username::`, `:password::`)
- [ ] **EFS mounts** - Same volume mounts (or both empty)
- [ ] **Health check** - Same endpoint, interval, timeout
- [ ] **Service Connect** - Same discovery name and port
- [ ] **Auto-scaling** - Same min/max/target values
- [ ] **Circuit breaker** - Document if only in one (CDK has it, Terraform may not)
- [ ] **Replicas** - Same desired count

#### Networking Comparison Checklist

- [ ] **VPC** - Same CIDR, AZ count, subnet layout
- [ ] **ALB** - Same scheme, listeners, target groups
- [ ] **Security groups** - Same ingress/egress rules
- [ ] **Service Discovery** - Same namespace name, service names
- [ ] **VPC Endpoints** - Same endpoint types

#### Storage Comparison Checklist

- [ ] **EFS access points** - Same paths, POSIX user/group, permissions
- [ ] **EFS mount targets** - Same subnet placement
- [ ] **DocumentDB** - Same engine, instance class, encryption, auth
- [ ] **Aurora MySQL** - Same engine, scaling, encryption

#### Secrets Comparison Checklist

- [ ] **Secret names** - Same set of secrets created
- [ ] **KMS encryption** - Same key configuration
- [ ] **Conditional creation** - Same conditions for optional secrets
- [ ] **Rotation** - Same rotation schedule and Lambda config

### Step 4: Generate Parity Report

Present findings in this format:

```markdown
## Infrastructure Parity Report

**Date:** {date}
**Source of Truth:** {Terraform|CDK}
**Scope:** {Full|Targeted area}

### Summary

| Area | Status | Discrepancies |
|------|--------|---------------|
| ECS Services | {In Sync / Diverged} | {count} |
| Networking | {In Sync / Diverged} | {count} |
| Storage | {In Sync / Diverged} | {count} |
| Secrets | {In Sync / Diverged} | {count} |
| IAM | {In Sync / Diverged} | {count} |
| Observability | {In Sync / Diverged} | {count} |

### Discrepancies

#### {Area}: {Description}

| Attribute | Terraform | CDK | Action Required |
|-----------|-----------|-----|-----------------|
| {attr} | {tf value} | {cdk value} | {Fix in TF / Fix in CDK / Intentional} |

### Intentional Differences

These differences are by design and should NOT be synced:

| Feature | CDK | Terraform | Rationale |
|---------|-----|-----------|-----------|
| Circuit Breaker | Enabled | Not configured | Faster failure detection (~4 min vs ~3 hours) |
| HOME=/tmp | Set on registry | Not set | Fixes Path.home() PermissionError in container |
| DocumentDB credentials | Always passed when cluster exists | Gated on storage_backend | Skills repository always uses DocumentDB |
| Keycloak URL fallback | Falls back to http://{alb-dns} | Always https:// | Deployment without Route53/certs |

### Recommendations

1. **Must Fix:** {Critical parity gaps}
2. **Should Fix:** {Important but non-breaking gaps}
3. **Consider:** {Nice-to-have improvements}
```

### Step 5: Apply Fixes

If the user wants fixes applied:

1. Determine the target (Terraform or CDK)
2. Make the edits to bring the target in sync
3. For CDK changes: run `cd infra && npx tsc --noEmit` to verify TypeScript compiles
4. For Terraform changes: run `cd terraform/aws-ecs && terraform validate` if available
5. Present a summary of all changes made

## Comparison Details by Area

### ECS Services - Environment Variables Deep Comparison

This is the most common source of drift. For each ECS service, extract and compare every environment variable:

**Terraform location:** Look for `environment = [{ name = "...", value = "..." }]` blocks in `ecs-services.tf`

**CDK location:** Look for `environment: { KEY: "value" }` objects in `registry-service-stack.ts`

Compare line by line. Pay special attention to:
- Variables that reference other resources (ALB DNS, Keycloak URL, DocumentDB endpoint)
- Variables with conditional logic (only set when a feature is enabled)
- Variables that use Secrets Manager references vs plain environment variables

### ECS Services - Secrets Deep Comparison

**Terraform:** Look for `secrets = [{ name = "...", valueFrom = "..." }]` blocks
**CDK:** Look for `secrets: { KEY: ecs.Secret.fromSecretsManager(...) }` objects

Compare the JSON key extraction syntax carefully:
- Terraform: `"${aws_secretsmanager_secret.x.arn}:username::"`
- CDK: `ecs.Secret.fromSecretsManager(secret, 'username')`

### EFS Mounts Deep Comparison

**Terraform:** Check `mount_points` and `volume` blocks in each service definition
**CDK:** Check `efsVolumes` property in `RegistryEcsService` constructor calls

Important: The registry service should have NO EFS mounts in both tools. Only auth-server and specific MCP servers mount EFS.

### ALB Listeners and Target Groups

**Terraform:** Check `aws_lb_listener` and `aws_lb_target_group` resources in `networking.tf`
**CDK:** Check listener and target group creation in `registry-service-stack.ts`

Compare:
- Listener ports and protocols
- Target group health check paths, intervals, and thresholds
- Deregistration delay
- Stickiness settings

## Known Intentional Differences

Maintain this list of differences that should NOT trigger sync warnings:

1. **Circuit Breaker** - CDK enables `circuitBreaker: { enable: true, rollback: true }` on ECS services. Terraform does not. This is intentional for faster failure detection.

2. **HOME=/tmp** - CDK sets `HOME=/tmp` on the registry container. Terraform does not. This fixes a `Path.home()` PermissionError that occurs when the container runs as a non-root user without a home directory.

3. **DocumentDB Credential Passing** - CDK passes DocumentDB credentials whenever `dataStack.documentDbSecretArn` exists. Terraform gates on `var.storage_backend == "documentdb"`. CDK's approach is correct because the skills repository always uses DocumentDB regardless of the storage backend setting.

4. **Keycloak URL Fallback** - CDK falls back to `http://{alb-dns-name}` when Route53 is disabled. Terraform always uses `https://`. CDK's approach allows deployment without DNS/certificates.

## Tips

- When comparing environment variables, sort them alphabetically for easier visual comparison
- Use `grep -n "ENV_VAR_NAME"` to quickly find where a variable is defined in both codebases
- After making changes to CDK, always run `cd infra && npx tsc --noEmit` to verify compilation
- After making changes to Terraform, run `terraform fmt` and `terraform validate`
- Check `infra/docs/CDK-INFRASTRUCTURE.md` for the latest CDK documentation and parity notes
