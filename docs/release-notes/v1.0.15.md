# Release v1.0.15 - Virtual MCP Servers, IAM Settings, Skill Security Scanning & Internal JWT Auth

**February 2026**

---

## Upgrading from v1.0.13

This section covers everything you need to know to upgrade from v1.0.13 to v1.0.15.

### Breaking Changes

**Helm Chart Dependency Removal (EKS/Helm users only)**

The `bitnami/common` chart dependency has been **removed** from both the `registry` and `auth-server` sub-charts. If you are upgrading Helm charts from v1.0.13, you **must** rebuild dependencies before upgrading:

```bash
# Required before helm upgrade
cd charts/mcp-gateway-registry-stack
helm dependency build
helm dependency update
```

Without this step, `helm upgrade` will fail because the old Chart.lock references a dependency that no longer exists.

**Internal Service-to-Service Auth Changed to JWT (#533)**

Internal communication between the registry and auth-server now uses self-signed JWTs instead of Basic Auth. This change is transparent -- no configuration is needed -- but the `SECRET_KEY` environment variable is now used for both JWT token signing and internal service authentication. Ensure your `SECRET_KEY` is set consistently across registry and auth-server containers.

### New Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEPLOYMENT_MODE` | `with-gateway` | `with-gateway` or `registry-only` |
| `REGISTRY_MODE` | `full` | `full`, `skills-only`, `mcp-servers-only`, `agents-only` |
| `OAUTH_STORE_TOKENS_IN_SESSION` | `false` | Store OAuth tokens in session cookie (disable for Entra ID) |
| `SKILL_SECURITY_SCAN_ENABLED` | `true` | Enable skill security scanning on registration |
| `SKILL_SECURITY_ANALYZERS` | `yara,spec,heuristic` | Comma-separated list of skill analyzers |

### Upgrade Instructions

#### Docker Compose

```bash
cd mcp-gateway-registry
git pull origin main
git checkout v1.0.15

# Review new env vars in .env.example and update your .env if needed
# Then rebuild and restart:
./build_and_run.sh
```

#### Kubernetes / Helm (EKS)

```bash
cd mcp-gateway-registry
git pull origin main
git checkout v1.0.15

# REQUIRED: Rebuild dependencies (bitnami/common was removed)
cd charts/mcp-gateway-registry-stack
helm dependency build
helm dependency update

# Update values.yaml if needed for new features (deployment mode, node selectors, etc.)
# Then upgrade:
helm upgrade mcp-gateway . -f your-values.yaml
```

#### Terraform / ECS

```bash
cd mcp-gateway-registry
git pull origin main
git checkout v1.0.15

# Update your .tfvars with any new variables you want to configure
# New Terraform variables available: deployment_mode, registry_mode, oauth_store_tokens_in_session
cd terraform/aws-ecs
terraform plan
terraform apply
```

#### DockerHub Images

Pre-built images are available:

```bash
docker pull mcpgateway/registry:v1.0.15
docker pull mcpgateway/auth-server:v1.0.15
docker pull mcpgateway/currenttime-server:v1.0.15
docker pull mcpgateway/realserverfaketools-server:v1.0.15
docker pull mcpgateway/mcpgw-server:v1.0.15
docker pull mcpgateway/fininfo-server:v1.0.15
docker pull mcpgateway/metrics-service:v1.0.15
```

---

## Major Features

### Virtual MCP Servers

Aggregate tools from multiple backend MCP servers into a single virtual endpoint:

- **Virtual Server Management**: Create virtual servers that combine tools from multiple real backend servers into one unified endpoint
- **Lua-Based Router**: High-performance nginx Lua router handles MCP protocol routing to backend servers
- **Streamable HTTP Compliance**: Full MCP Streamable HTTP protocol support for virtual server endpoints
- **Semantic Search Integration**: Virtual server tools are indexed and searchable via semantic search
- **Scope-Based Access Control**: Virtual servers integrate with the existing IAM/scopes system
- **Rating Support**: Virtual servers support the same 5-star rating system as regular servers
- **CLI Commands**: Register, list, and manage virtual servers from the command line
- **E2E Test Coverage**: Playwright E2E tests, MCP protocol compliance tests, and stress tests included

[PR #501](https://github.com/agentic-community/mcp-gateway-registry/pull/501)

### IAM Settings UI

Full IAM management interface in the Settings page:

- **Groups Management**: Create, edit, and delete IAM groups with server/tool/agent access dropdowns
- **Users Management**: View and manage user accounts and group assignments
- **M2M Client Management**: Manage machine-to-machine OAuth clients
- **Searchable Selectors**: Scalable searchable dropdowns for servers and agents (handles large lists)
- **Virtual Server Permissions**: Auto-populate `list_virtual_server` permission for virtual servers
- **Agent Access Control**: Full CRUD operations for IAM groups with agent access

[PR #494](https://github.com/agentic-community/mcp-gateway-registry/pull/494)

### Skill Security Scanning

Integrate Cisco AI Defense Skill Scanner for automated skill security analysis:

- **Automated Scanning**: Skills are scanned on registration using configurable analyzers (YARA, spec, heuristic, LLM, endpoint)
- **Security Scan API**: New `/api/skills/{path}/security-scan` endpoints for triggering and viewing scan results
- **Frontend Integration**: Security scan results displayed on SkillCard component
- **CLI Commands**: `skill-scan` and `skill-scan-result` CLI commands for scripting
- **Property-Based Tests**: Comprehensive test coverage including property-based tests for schemas and repository layer

[PR #510](https://github.com/agentic-community/mcp-gateway-registry/pull/510) | [Issue #495](https://github.com/agentic-community/mcp-gateway-registry/issues/495)

### System Configuration Viewer

Admin configuration viewer in the Settings page:

- **Configuration Groups**: 11 groups covering deployment, storage, auth, embeddings, health checks, websockets, security scanning, audit, federation, and discovery
- **Export Formats**: Export configuration as `.env`, JSON, Terraform `.tfvars`, or YAML
- **Sensitive Value Masking**: Passwords, tokens, and API keys are automatically masked
- **Search and Filter**: Search across all configuration parameters
- **Rate Limited**: 10 requests per minute per user

[PR #508](https://github.com/agentic-community/mcp-gateway-registry/pull/508) | [Issue #492](https://github.com/agentic-community/mcp-gateway-registry/issues/492)

### Internal JWT Authentication (#533)

Service-to-service communication now uses self-signed JWTs instead of Basic Auth:

- **JWT-Based Auth**: Registry signs JWTs with `SECRET_KEY` when calling auth-server internal endpoints
- **Configurable Auth Server URL**: `AUTH_SERVER_URL` setting replaces hardcoded `http://auth-server:8888` for EKS compatibility
- **Single Source of Truth**: JWT issuer and audience constants defined once in `registry/auth/internal.py`

[PR #533](https://github.com/agentic-community/mcp-gateway-registry/pull/533) | [Issue #515](https://github.com/agentic-community/mcp-gateway-registry/issues/515)

---

## What's New

### Deployment Modes
- Registry-only deployment mode without nginx integration (#485, #486)
- Skills-only registry mode for lightweight deployment (#493)
- Deployment and registry mode added to Helm stack chart (#497)
- Auto-correction for invalid mode combinations (e.g., `with-gateway` + `skills-only`)

### Helm Chart Improvements
- Node selector support for all pods including Keycloak, Postgres, and MongoDB (#514)
- Option to disable Keycloak ingress patch for service-mesh environments (#516)
- Keycloak auth for registry API endpoints in Helm (#517)
- Federation environment variables added to charts (#474)
- Disable Keycloak when using Entra ID (#482)
- Git hash/tag pushed to images for version tracking (#480, #481)
- Removed unneeded `bitnami/common` chart dependency (#483)
- Helm install examples added to README (#484)

### Security Fixes
- Strip newlines from X-Body header to prevent scope validation bypass (#529)
- Normalize leading slashes in scope server name matching (#529)
- Recognize `registry-admins` group in `can_modify_servers` check
- Move security-scan routes before catch-all path route

### Audit Logging Fixes
- Audit composite key index fix for concurrent MCPServerAccessRecord and RegistryApiAccessRecord events (#530)
- Handle duplicate audit event inserts gracefully (#513)
- Case-insensitive regex for username filtering in audit logs
- Stream-aware filters for audit queries

### OAuth and Authentication
- Default `OAUTH_STORE_TOKENS_IN_SESSION` to `false` to prevent cookie size issues with Entra ID (#528)
- OAuth token storage control surfaced in system config panel (#528)
- Terraform support for `OAUTH_STORE_TOKENS_IN_SESSION` variable (#528)

### Infrastructure
- Docker build optimizations for faster image builds (#473)
- Preserve client IP address in logs/audit (#476)
- `REGISTRY_ROOT_PATH` support for path-based API hosting (#472)
- ECR-based container image references (#479)
- Consistent Keycloak fallback behavior (#482)
- CI parallel test execution with `-n auto` (#501)

### Frontend Improvements
- Roo Code IDE option with streamable-http format and copy feedback
- Auto-populate JWT token in MCP configuration modal
- Virtual MCP Servers tab with rating support
- Searchable select component for IAM server/agent dropdowns
- Skill security scan display on SkillCard component

### Documentation
- Virtual MCP server design document and operations guide
- IAM Settings UI documentation
- Registry deployment modes documentation
- System Configuration Viewer documentation
- Claude Code skills for development workflow

---

## Bug Fixes

- Nginx config failed to load on startup due to excessive variables in the file (#512)
- Audit composite key allowing only one event type per request (#530)
- Duplicate audit event insert errors on concurrent writes (#513)
- X-Body header newline injection in scope validation (#529)
- Leading slash normalization in scope server name matching (#529)
- `registry-admins` group not recognized in `can_modify_servers` check
- Skill toggle sending query parameter instead of JSON body
- Path mismatch in skill toggle causing UI not to update
- Disabled skills excluded from API requests
- Dashboard sections not rendering when feature enabled
- Admin toggle for servers/agents/skills requiring explicit UI permission
- Semantic search results not filtered by registry mode
- MongoDB `nodeSelector` config not wrapping statefulset spec correctly (#514)
- Security-scan routes shadowed by catch-all path route
- Virtual server search returning incorrect tool results
- JWT token extraction from API response

---

## Pull Requests Included

| PR | Title |
|----|-------|
| #533 | Sign SECRET_KEY as JWT for internal communication |
| #530 | fix: audit composite key index and stream-aware filters |
| #529 | fix: strip newlines from X-Body header to prevent scope validation bypass |
| #528 | feat: OAuth token session storage control with Terraform and config panel support |
| #517 | Bug: Keycloak auth required for Registry API endpoints missing from Helm Chart |
| #516 | Feature: Allow Helm Chart to not deploy the keycloak-ingress-patch |
| #514 | Feature: Allow setting nodeSelector for pods in Helm charts |
| #513 | bug: Handle duplicate inserts into the audit events DB |
| #512 | fix: Nginx config would not load on startup due to variable count |
| #510 | feat: Skill scanner integration (Issue #495) |
| #509 | chore(deps): bump fast-xml-parser and @aws-sdk/xml-builder in /cli |
| #508 | feat: Add System Configuration Viewer with documentation |
| #507 | Add Claude Code skills for development workflow |
| #506 | docs: Remove remaining production-grade instances |
| #505 | docs: Use more precise language in documentation |
| #503 | chore(deps): bump ajv from 8.14.0 to 8.18.0 in /frontend |
| #501 | feat: Virtual MCP Server - Aggregate tools from multiple backend servers |
| #497 | Add registry/deployment modes to mcp gateway registry stack chart |
| #494 | feat: IAM Settings UI with Groups, Users, and M2M Management |
| #493 | feat: Add skills-only registry mode for lightweight deployment |
| #486 | fix: Registry-only mode nginx config and terraform updates |
| #485 | feat: Add Registry-Only Deployment Mode (#478) |
| #484 | Add helm install examples |
| #483 | Remove unneeded chart dependency |
| #482 | Disable keycloak if Entra ID |
| #481 | Add git hash to helm deployment |
| #480 | Set git hash/tag to BUILD_VERSION in images |
| #479 | Update images to use ECR |
| #478 | feat: add registry-only deployment mode |
| #477 | chore(deps): bump jsonpath from 1.2.0 to 1.2.1 in /frontend |
| #476 | Preserve client IP in logs |
| #474 | Add federation env vars to charts |
| #473 | Docker build optimizations |
| #472 | Add REGISTRY_ROOT_PATH for path-based API hosting |
| #471 | Fix 1.0.13 chart image tags |
| #468 | chore(deps): bump langchain-core from 1.2.5 to 1.2.11 |
| #467 | chore(deps): bump cryptography from 46.0.3 to 46.0.5 |
| #466 | chore(deps): bump cryptography from 46.0.3 to 46.0.5 in /agents/a2a |
| #462 | chore(deps): bump axios from 1.13.2 to 1.13.5 in /frontend |

---

## Security Dependency Updates

| Package | Previous | Updated | Scope |
|---------|----------|---------|-------|
| cryptography | 46.0.3 | 46.0.5 | registry, agents/a2a |
| axios | 1.13.2 | 1.13.5 | frontend |
| ajv | 8.14.0 | 8.18.0 | frontend |
| langchain-core | 1.2.5 | 1.2.11 | registry |
| jsonpath | 1.2.0 | 1.2.1 | frontend |
| fast-xml-parser | - | latest | cli |

---

## Contributors

Thank you to all contributors for this release:

- **Amit Arora** ([@aarora79](https://github.com/aarora79))
- **Omri Shiv** ([@omrishiv](https://github.com/omrishiv))
- **Geoffrey Norman** ([@gknorman](https://github.com/gknorman))
- **Dheeraj Oruganty** ([@dheerajoruganty](https://github.com/dheerajoruganty))
- **snorlaX-sleeps** ([@snorlaX-sleeps](https://github.com/snorlaX-sleeps))
- **Abhishek Singh** ([@abkrsinh](https://github.com/abkrsinh))
- **Andreas Feldmann** ([@ndrsfel](https://github.com/ndrsfel))

---

## Support

- [GitHub Issues](https://github.com/agentic-community/mcp-gateway-registry/issues)
- [GitHub Discussions](https://github.com/agentic-community/mcp-gateway-registry/discussions)
- [Documentation](https://github.com/agentic-community/mcp-gateway-registry/tree/main/docs)

---

**Full Changelog:** [v1.0.13...v1.0.15](https://github.com/agentic-community/mcp-gateway-registry/compare/v1.0.13...v1.0.15)
