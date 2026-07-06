# Release v1.0.12 - Multi-Provider IAM, DocumentDB Storage & Well-Known Health Fix

**January 2026**

---

## Major Features

### Multi-Provider IAM Support for Keycloak and Microsoft Entra ID

Full IAM support enabling both Keycloak AND Microsoft Entra ID through a unified API:

- **Harmonized API**: Same user and group management experience regardless of IdP
- **Self-Signed JWT Tokens**: Human users can generate tokens for CLI tools and AI coding assistants
- **M2M Service Accounts**: AI agent identity with OAuth2 Client Credentials flow
- **Fine-Grained Access Control**: Scopes define exactly which MCP servers, methods, tools, and agents each user can access

[PR #378](https://github.com/agentic-community/mcp-gateway-registry/pull/378)

### CloudFront HTTPS Support for AWS ECS

Production-ready AWS deployment with CloudFront for HTTPS termination:

- **CDN Caching**: Global edge distribution for improved latency
- **Three Deployment Modes**: Flexible configurations to match requirements
- **SSL/TLS Termination**: Secure connections without managing certificates on ECS

[PR #363](https://github.com/agentic-community/mcp-gateway-registry/pull/363) | [Issue #293](https://github.com/agentic-community/mcp-gateway-registry/issues/293)

### Well-Known Discovery Health Status Fix

The `/.well-known/mcp-servers` endpoint now returns actual health status instead of hardcoded "healthy":

- **Accurate Status Reporting**: Servers show real health status (healthy, unhealthy, disabled, unknown)
- **Status Normalization**: Detailed messages like "unhealthy: timeout" normalized to "unhealthy" for client consumption
- **Comprehensive Tests**: 457 lines of new tests for the well-known routes

[PR #384](https://github.com/agentic-community/mcp-gateway-registry/pull/384) | [Issue #375](https://github.com/agentic-community/mcp-gateway-registry/issues/375)

---

## What's New

### Authentication & IAM
- Multi-Provider IAM Support for Keycloak and Microsoft Entra ID (#378)
- JWT token scopes improvements (#383)

### Infrastructure & Docker
- Add lightweight Dockerfile for simple MCP servers (Dockerfile.mcp-server-light)
- Synchronize docker-compose files for consistency
- Reference official mongo8:2 image in docker-compose.prebuilt.yml (#364)
- Update images and add scope.yml to mongo setup job (#360)

### AWS ECS Deployment
- CloudFront HTTPS support for AWS ECS deployment (#363)
- Deployment mode fixes and security group rules limit (#374)
- AWS ECS deployment improvements and script hardening (#365)

### Bug Fixes
- Fix well-known endpoint returning hardcoded health status (#384)
- Quick Start docs, MongoDB auth, and JWT token scopes fixes (#383)

### Documentation
- Mark #232 and #297 as completed in roadmap
- Add HuggingFace CLI explanation and installation link (#371)
- Mark MCP server description as required (#362)

---

## Configuration Changes

### New Dockerfile for Simple MCP Servers

A lightweight Dockerfile (`docker/Dockerfile.mcp-server-light`) is now available for simple MCP servers that don't need PyTorch or the registry module:

```yaml
# docker-compose.yml example
currenttime-server:
  build:
    context: .
    dockerfile: docker/Dockerfile.mcp-server-light
    args:
      SERVER_DIR: servers/currenttime
```

Benefits:
- Smaller image size (no PyTorch dependencies)
- Faster builds
- Suitable for: currenttime, fininfo, realserverfaketools servers

---

## Upgrade Instructions

### For Docker Compose Deployments

1. **Pull the latest changes:**
```bash
cd mcp-gateway-registry
git pull origin main
git checkout v1.0.12
```

2. **Rebuild and restart:**
```bash
./build_and_run.sh
```

### For AWS ECS Deployment

1. **Update Terraform variables** for CloudFront configuration if desired
2. **Apply Terraform changes:**
```bash
cd terraform/aws-ecs
terraform plan
terraform apply
```

---

## Bug Fixes

- Fix `/.well-known/mcp-servers` endpoint returning hardcoded "healthy" status for all servers (#384)
- Fix Quick Start documentation and MongoDB authentication issues (#383)
- Fix JWT token scopes handling (#383)
- Fix deployment mode issues and security group rules limit in AWS ECS (#374)
- Synchronize docker-compose files for consistency

---

## Pull Requests Included

| PR | Title |
|----|-------|
| #384 | fix: retrieve actual health status in well-known discovery endpoint |
| #383 | fix: Quick Start docs, MongoDB auth, and JWT token scopes |
| #378 | feat: Multi-Provider IAM Support for Keycloak and Microsoft Entra ID |
| #374 | fix: deployment mode fixes, security group rules limit, and documentation improvements |
| #371 | docs: add HuggingFace CLI explanation and installation link |
| #365 | fix: AWS ECS deployment improvements and script hardening for v1.0.10 |
| #364 | Changed line 365 of docker-compose.prebuilt.yml to reference the official mongo8:2 image |
| #363 | feat: Add CloudFront HTTPS support for AWS ECS deployment |
| #362 | mark mcp server description as required |
| #360 | update images and add scope.yml to mongo setup job |

---

## Issues Closed

- [#375](https://github.com/agentic-community/mcp-gateway-registry/issues/375) - Bug: /.well-known/mcp-servers endpoint returns hardcoded "healthy" status
- [#293](https://github.com/agentic-community/mcp-gateway-registry/issues/293) - Add CloudFront HTTPS support for AWS ECS deployment

---

## Contributors

Thank you to our amazing contributors for this release:

- Omri Shiv
- Viviana Luccioli
- Andreas Feldmann
- Wallace Printz
- Gaurav Rele
- cxhello
- Gabriel Rojas

---

## Resources

### Documentation
- [Storage Architecture](docs/design/storage-architecture-mongodb-documentdb.md) - MongoDB/DocumentDB storage design
- [IdP Provider Support](docs/design/idp-provider-support.md) - Multi-provider IAM documentation
- [Authentication Design](docs/design/authentication-design.md) - Authentication architecture

---

## Support

- [GitHub Issues](https://github.com/agentic-community/mcp-gateway-registry/issues)
- [GitHub Discussions](https://github.com/agentic-community/mcp-gateway-registry/discussions)
- [Documentation](https://github.com/agentic-community/mcp-gateway-registry/tree/main/docs)

---

**Full Changelog:** [v1.0.10...v1.0.12](https://github.com/agentic-community/mcp-gateway-registry/compare/v1.0.10...v1.0.12)
