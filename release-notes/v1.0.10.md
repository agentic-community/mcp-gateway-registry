# Release v1.0.10 - A2A Discovery, OAuth2 Providers & Enhanced Search

**January 2026**

---

## Major Features

### A2A Agent Discovery and Invocation Pattern

Complete implementation of Agent-to-Agent (A2A) communication workflow enabling agents to discover and collaborate with other agents:

- **Registry Discovery Client**: Semantic search and skill-based agent discovery
- **Remote Agent Client**: A2A protocol communication with discovered agents
- **Travel Assistant + Flight Booking Agents**: Reference implementation demonstrating the pattern
- **Agent Caching**: Efficient caching of discovered agents for reuse

[PR #344](https://github.com/agentic-community/mcp-gateway-registry/pull/344) | [Issue #198](https://github.com/agentic-community/mcp-gateway-registry/issues/198)

### OAuth2 Provider Configuration

Flexible OAuth2 provider enablement through environment variables:

- **Dynamic Provider Selection**: Enable/disable Keycloak, Cognito, Entra ID, GitHub, and Google via environment variables
- **Nginx Route Configuration**: Added OAuth2 callback routes for Entra, GitHub, and Google providers
- **Backward Compatibility**: Keycloak enabled by default to match previous behavior

[PR #353](https://github.com/agentic-community/mcp-gateway-registry/pull/353) | [PR #354](https://github.com/agentic-community/mcp-gateway-registry/pull/354)

### Enhanced Semantic Search

Improved search capabilities with hybrid keyword matching:

- **Hybrid Search**: Combines semantic similarity with keyword matching
- **Tool Discovery**: Better discovery of MCP server tools and agent skills
- **Query Tokenization**: Improved handling of multi-word search queries

[PR #352](https://github.com/agentic-community/mcp-gateway-registry/pull/352)

---

## What's New

### Authentication & OAuth2
- Make OAuth2 provider enablement configurable via environment variables (#353)
- Add OAuth2 nginx routes for Entra, GitHub, and Google providers (#354)
- Fix outdated placeholder checks in registry-entrypoint.sh (#355)
- Fix boolean conversion in `substitute_env_vars` for OAuth2 provider enablement

### A2A Agent Registry
- Complete A2A agent discovery and invocation pattern (#344)
- Web-based UI for A2A agent management (#349)
- Closes issue #198: Agent-to-Agent Communication Workflow

### Search & Discovery
- Improve semantic search with hybrid keyword matching (#352)
- Better agent tool discovery through enhanced search

### Security & UI
- Add security scan results popup to ServerCard and AgentCard (#341)
- Support `server_name` field in JSON upload for server registration

### MongoDB/DocumentDB
- MongoDB deployment and configuration improvements (#343)
- Update index creation for mongodb-ce (#342)
- Update MongoDB init to support SCRAM-SHA-256 and custom replicaset (#337)
- Fix MongoDB authentication compatibility for DocumentDB (#335)

### Frontend
- Frontend performance optimizations with webpack-dev-server v5 fix (#339)
- Bump react-router and react-router-dom dependencies (#345)

### Documentation
- Clarify AUTH_SERVER_EXTERNAL_URL configuration in macOS setup guide (#338)

---

## Configuration Changes

### New Environment Variables

```bash
# OAuth2 Provider Enablement (all default to false except KEYCLOAK_ENABLED)
KEYCLOAK_ENABLED=true    # Default: true (for backward compatibility)
COGNITO_ENABLED=false
ENTRA_ENABLED=false
GITHUB_ENABLED=false
GOOGLE_ENABLED=false
```

### OAuth2 Provider Setup

To enable additional OAuth2 providers:

1. **Entra ID (Microsoft)**:
   ```bash
   ENTRA_ENABLED=true
   ENTRA_TENANT_ID=your-tenant-id
   ENTRA_CLIENT_ID=your-client-id
   ENTRA_CLIENT_SECRET=your-client-secret
   ```

2. **GitHub**:
   ```bash
   GITHUB_ENABLED=true
   GITHUB_CLIENT_ID=your-github-client-id
   GITHUB_CLIENT_SECRET=your-github-client-secret
   ```

3. **Google**:
   ```bash
   GOOGLE_ENABLED=true
   GOOGLE_CLIENT_ID=your-google-client-id
   GOOGLE_CLIENT_SECRET=your-google-client-secret
   ```

---

## Upgrade Instructions

### For Docker Compose Deployments

1. **Pull the latest changes:**
```bash
cd mcp-gateway-registry
git pull origin main
git checkout v1.0.10
```

2. **Update environment configuration** (if enabling OAuth2 providers):
```bash
# Add OAuth2 provider configuration to .env
# See Configuration Changes section above
```

3. **Rebuild and restart:**
```bash
./build_and_run.sh
```

### For AWS ECS Deployment

1. **Update Terraform variables** for any new OAuth2 providers
2. **Apply Terraform changes:**
```bash
cd terraform/aws-ecs
terraform plan
terraform apply
```

---

## Bug Fixes

- Fix boolean conversion in `substitute_env_vars` for OAuth2 provider enablement (environment variables return strings, not booleans)
- Fix outdated placeholder checks in registry-entrypoint.sh (#355)
- Fix MongoDB authentication compatibility for DocumentDB (#335)
- Frontend performance optimizations with webpack-dev-server v5 fix (#339)

---

## Pull Requests Included

| PR | Title |
|----|-------|
| #353 | Registry-Auth: Make OAuth2 provider enablement configurable via environment variables |
| #354 | Add OAuth2 nginx routes for Entra, GitHub, and Google providers |
| #355 | Fix outdated placeholder checks in registry-entrypoint.sh |
| #352 | Improve semantic search with hybrid keyword matching and agent tool discovery |
| #349 | Web-based UI for A2A agent management |
| #345 | chore(deps): bump react-router and react-router-dom in /frontend |
| #344 | A2A Agent Discovery and Invocation Pattern |
| #343 | MongoDB deployment and configuration |
| #342 | Update index creation for mongodb-ce |
| #341 | Add security scan results popup to ServerCard and AgentCard |
| #339 | Frontend Performance Optimizations with webpack-dev-server v5 Fix |
| #338 | docs: clarify AUTH_SERVER_EXTERNAL_URL config in macOS setup guide |
| #337 | Update mongodb init to support SCRAM-SHA-256, support auth and custom replicaset |
| #335 | Fix MongoDB authentication compatibility for DocumentDB |

---

## Issues Closed

- [#198](https://github.com/agentic-community/mcp-gateway-registry/issues/198) - Implement Agent-to-Agent Communication Workflow

---

## Resources

### Documentation
- [A2A Agents README](agents/a2a/README.md) - A2A agent setup and usage
- [OAuth2 Configuration](.env.example) - Environment variable reference
- [macOS Setup Guide](docs/podman-setup.md) - Updated with AUTH_SERVER_EXTERNAL_URL clarification

---

## Support

- [GitHub Issues](https://github.com/agentic-community/mcp-gateway-registry/issues)
- [GitHub Discussions](https://github.com/agentic-community/mcp-gateway-registry/discussions)
- [Documentation](https://github.com/agentic-community/mcp-gateway-registry/tree/main/docs)

---

**Full Changelog:** [v1.0.9...v1.0.10](https://github.com/agentic-community/mcp-gateway-registry/compare/v1.0.9...v1.0.10)
