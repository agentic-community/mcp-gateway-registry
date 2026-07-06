# Release v1.0.14 - Registry-Only Deployment Mode

**February 2026**

---

## Major Features

### Registry-Only Deployment Mode

Deploy the registry as a standalone catalog/discovery service without gateway integration:

- **New `DEPLOYMENT_MODE` parameter**: Choose between `with-gateway` (default) or `registry-only`
- **Works with `REGISTRY_MODE`**: Combine deployment mode with registry mode (`full`, `skills-only`, `mcp-servers-only`, `agents-only`)
- **Lightweight deployments**: Run registry without nginx dynamic location block generation for catalog-only use cases
- **Auto-correction**: Invalid combinations (e.g., `with-gateway` + `skills-only`) are automatically corrected with warnings
- **Prometheus metrics**: Monitor deployment mode via `registry_deployment_mode_info` gauge and track skipped nginx updates via `registry_nginx_updates_skipped_total` counter
- **Health check enhancement**: `/health` endpoint now includes `deployment_mode`, `registry_mode`, and `nginx_updates_enabled` fields
- **Configuration API**: New `/api/config` endpoint exposes deployment mode and feature flags to the frontend
- **Frontend awareness**: UI adapts to deployment mode — shows direct server URLs and hides gateway auth instructions in registry-only mode

[PR #478](https://github.com/agentic-community/mcp-gateway-registry/pull/478)

---

## Configuration

### Environment Variables

```bash
# Deployment mode: with-gateway (default) or registry-only
DEPLOYMENT_MODE=with-gateway

# Registry mode: full (default), skills-only, mcp-servers-only, agents-only
REGISTRY_MODE=full
```

### Example: Skills-Only Registry

```bash
DEPLOYMENT_MODE=registry-only
REGISTRY_MODE=skills-only
```

### Auto-Correction Behavior

If an invalid combination is detected at startup, the registry auto-corrects and logs a warning banner:

| Configuration | Auto-Corrected To |
|---|---|
| `with-gateway` + `skills-only` | `registry-only` + `skills-only` |

All other combinations are valid and pass through unchanged.

### Helm Chart

```yaml
registry:
  deployment_mode: with-gateway
  registry_mode: full
```

### Terraform

```hcl
deployment_mode = "with-gateway"
registry_mode   = "full"
```

---

## What's Changed

- New `DeploymentMode` and `RegistryMode` enums in configuration
- `nginx_updates_enabled` property on Settings controls nginx behavior
- Nginx service methods (`generate_config`, `generate_config_async`, `reload_nginx`) return early in registry-only mode
- Prometheus metrics for deployment mode info and skipped nginx operations
- `/api/config` endpoint for frontend deployment mode awareness
- `/health` endpoint enhanced with deployment mode fields
- Docker entrypoint logs deployment mode at container startup
- `useRegistryConfig` React hook for frontend components
- `DeploymentModeIndicator` badge component for registry-only mode
- `ServerConfigModal` uses `proxy_pass_url` in registry-only mode instead of constructed gateway URL
- Helm chart, Terraform, and `.env.example` updated with new variables

---

## Upgrade Instructions

### For Docker Compose Deployments

1. **Pull the latest changes:**
```bash
cd mcp-gateway-registry
git pull origin main
git checkout v1.0.14
```

2. **Rebuild and restart:**
```bash
./build_and_run.sh
```

No configuration changes required — defaults to `with-gateway` + `full` for backward compatibility.

### For Kubernetes/Helm Deployments

1. **Update chart values** if you want registry-only mode:
```yaml
registry:
  deployment_mode: registry-only
  registry_mode: skills-only  # or full, mcp-servers-only, agents-only
```

2. **Apply changes:**
```bash
helm upgrade mcp-gateway ./charts/mcp-gateway -f your-values.yaml
```

---

## Support

- [GitHub Issues](https://github.com/agentic-community/mcp-gateway-registry/issues)
- [GitHub Discussions](https://github.com/agentic-community/mcp-gateway-registry/discussions)
- [Documentation](https://github.com/agentic-community/mcp-gateway-registry/tree/main/docs)

---

**Full Changelog:** [v1.0.13...v1.0.14](https://github.com/agentic-community/mcp-gateway-registry/compare/v1.0.13...v1.0.14)
