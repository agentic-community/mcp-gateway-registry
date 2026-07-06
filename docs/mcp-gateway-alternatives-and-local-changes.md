# MCP Gateway Alternatives and Local Changes

This document has two parts:

1. A comparison of MCP gateway / registry projects with explicit open-source license and commercial-use clarity.
2. A changelog of the local changes made while bringing this deployment up (Docker Compose based, macOS / Docker Desktop).

## Part 1: MCP Gateway / Registry Alternatives

The focus is on how each option handles upstream (egress) authentication to MCP servers -- in particular per-user OAuth, which is what a hosted OAuth-only server such as Notion requires -- and whether the license permits commercial use.

Licenses below were verified against each project's GitHub repository.

### License and commercial-use summary

| Project | Type | License | Commercial use | Notes |
|---|---|---|---|---|
| agentic-community/mcp-gateway-registry | Gateway + registry (this project) | Apache-2.0 | Allowed | Permissive with patent grant. Inbound OIDC via Keycloak/Entra/Auth0/Okta/GitHub/Google; egress supports static bearer/api_key. Per-user egress OAuth vault is designed but its registration write-path is not fully implemented yet. |
| IBM/mcp-context-forge (ContextForge) | Gateway + registry + federation | Apache-2.0 | Allowed | Closest full-featured peer. RFC 8693 token exchange (on-behalf-of), zero-trust secret injection, RBAC, guardrails. v1.0 approaching GA. Per-user OAuth passthrough to upstream not explicitly documented. Operationally heavier. |
| agentgateway/agentgateway | Agent/MCP data-plane proxy | Apache-2.0 | Allowed | Strongest fit for per-user egress OAuth: the gateway acquires, stores, and injects upstream credentials transparently, plus an authorization portal for MCP URL elicitation. Rust. |
| kgateway-dev/kgateway | Kubernetes API gateway (integrates agentgateway) | Apache-2.0 | Allowed | Formerly Gloo Gateway (Envoy-based). Hosts agentgateway as its AI-native data plane for MCP at scale. |
| docker/mcp-gateway | Gateway + catalog | MIT | Allowed | Container-isolation model: each MCP server runs sandboxed. Strong for local/dev and supply-chain isolation; less focused on per-user egress OAuth. |
| lasso-security/mcp-gateway | Security-first gateway | MIT | Allowed | Plugin-based: real-time scanning, token masking, guardrails, threat detection. |
| TheLunarCompany/lunar (MCPX) | Gateway | MIT | Allowed | Open-source gateway from Lunar.dev. |
| mcpjungle/MCPJungle | Self-hosted registry + gateway | MPL-2.0 | Allowed (weak copyleft) | MPL is file-level copyleft: you may use it commercially and combine with proprietary code, but modifications to MPL-licensed files must be published. |
| obot-platform/obot | Gateway / platform | MIT | Allowed | |
| atrawog/mcp-oauth-gateway | OAuth 2.1 adapter | Apache-2.0 | Allowed | Adds OAuth 2.1 to any MCP server without code changes. Not a registry. |
| modelcontextprotocol/registry | Official registry (catalog) | See repo (MIT-family; GitHub detects NOASSERTION) | Allowed | A server directory, not a gateway. |
| TrueFoundry / Portkey / MintMCP / Composio / Cloudflare One MCP portals | Managed / commercial | Proprietary SaaS (paid) | Per their terms | These explicitly support per-user egress OAuth (users authorize each upstream with their own account; the gateway stores and injects the per-user token). Least operational effort; not open source. Note some are built on open-source cores (e.g. agentgateway/kgateway). |

### License meaning for commercial use

- Apache-2.0 and MIT: permissive. Commercial use, modification, and redistribution are allowed; Apache-2.0 additionally grants patent rights. No obligation to open-source your own code.
- MPL-2.0: weak (file-level) copyleft. Commercial use is allowed and it can be combined with proprietary code, but if you modify MPL-licensed files you must publish those file changes under MPL.
- Proprietary SaaS: governed by the vendor's commercial terms.

All the open-source options above permit commercial use.

### Recommendation by need

- Per-user egress OAuth for OAuth-only upstreams, open source: agentgateway (agentgateway.dev), optionally fronted by kgateway (kgateway.dev) on Kubernetes.
- Full-featured open-source peer to this project (registry + RBAC + guardrails + federation): IBM ContextForge.
- Lowest operational effort with first-class per-user upstream OAuth: a managed gateway (TrueFoundry, Cloudflare One MCP portals, Composio).
- Security scanning as the priority: Lasso.

### Context

Per the November 2025 MCP specification revision, remote HTTP MCP servers exposed on the internet are expected to implement OAuth 2.1 with PKCE. OAuth-only upstreams will therefore become more common, so how well a gateway handles per-user egress OAuth is a primary selection criterion.

## Part 2: Local Changes Made in This Deployment

Bringing this stack up on macOS / Docker Desktop required several fixes. Only the Docker Compose change is a tracked-file change; the rest are local runtime or gitignored-config changes and are documented here for reproducibility.

### Tracked change (in this commit)

- docker-compose.yml -- fixed two false-negative container healthchecks:
  - keycloak: added `KC_HEALTH_ENABLED: 'true'` and replaced the `curl`-based healthcheck (the image ships no curl) with a bash `/dev/tcp` probe against the management port (9000) `/health/ready` endpoint.
  - otel-collector: removed the `wget`-based healthcheck. The upstream image is distroless (no curl/wget/shell), so no in-container probe is possible; nothing depends on this service's health. The health_check extension on port 13133 remains for external monitoring.

### Local / runtime changes (not committed; documented for reproducibility)

- MongoDB unhealthy: the local `.env` had multiple `DOCUMENTDB_PASSWORD` entries. Docker Compose uses the last value, but the `mongodb-data` volume was initialized with an earlier one, so authentication failed. Fixed by deduplicating `.env` to the password the volume was actually initialized with. (`.env` is gitignored.)
- Registry crash-loop: the bind-mounted log directory `/var/log/containers/ai-registry` was owned root:root inside the container, so the non-root app user (uid 1000) could not write nginx logs; nginx failed to start and the container restarted repeatedly. Fixed by chowning the directory to 1000:1000. On macOS the host-side chown from `scripts/prepare-log-dirs.sh` does not propagate to the Docker Desktop bind inode, so this must be applied at the mounted path.
- Keycloak realm missing (login showed "Page not found"): the `mcp-gateway` realm is created by `keycloak/setup/init-keycloak.sh`, which `build_and_run.sh` does not invoke. Ran `init-keycloak.sh` and `get-all-client-credentials.sh`, then set the real `KEYCLOAK_CLIENT_SECRET` and `KEYCLOAK_M2M_CLIENT_SECRET` in `.env`.
- Notion MCP server showed "security-pending": the hosted endpoint `https://mcp.notion.com/mcp` is OAuth-only, and this gateway supports only `none`/`bearer`/`api_key` for upstream auth, so the security scan could not authenticate. Resolved by running the official `@notionhq/notion-mcp-server` locally in HTTP transport mode (127.0.0.1:3100) with a Notion internal integration token, repointing the gateway server to `http://host.docker.internal:3100/mcp` with a bearer token for the HTTP transport, allowlisting the host bridge for the SSRF guard via `extra_env/registry.env` (`SSRF_ALLOWED_HOSTS=host.docker.internal`), then rescanning (passed) and clearing the `security-pending` tag. A launcher and launchd agent under `~/.notion-mcp/` provide persistence.

### Notes

- `extra_env/registry.env` is gitignored by design, so the SSRF allowlist change is documented here rather than committed. To reproduce, create `extra_env/registry.env` with `SSRF_ALLOWED_HOSTS=host.docker.internal` and recreate the registry container.
- No secrets (`.env`, `.env.*` backups, `.oauth-tokens/`, `extra_env/*.env`) are committed; all are gitignored.

## Sources

- AWS Open Source Blog -- Governing AI Assets at Scale with MCP Gateway and Registry: https://aws.amazon.com/blogs/opensource/governing-ai-assets-at-scale-with-mcp-gateway-and-registry/
- agentic-community/mcp-gateway-registry: https://github.com/agentic-community/mcp-gateway-registry
- IBM/mcp-context-forge: https://github.com/IBM/mcp-context-forge
- ContextForge -- Selecting an MCP Gateway: https://ibm.github.io/mcp-context-forge/best-practices/selecting-an-mcp-gateway/
- agentgateway: https://agentgateway.dev/
- kgateway: https://kgateway.dev/
- Solo.io -- MCP Authorization Patterns for Upstream API Calls: https://www.solo.io/blog/mcp-authorization-patterns-for-upstream-api-calls
- TrueFoundry -- MCP Gateway Auth and Security: https://www.truefoundry.com/docs/ai-gateway/mcp/mcp-gateway-auth-security
- Cloudflare One -- MCP server portals: https://developers.cloudflare.com/cloudflare-one/access-controls/ai-controls/mcp-portals/
- Lunar.dev -- Best Open Source MCP Gateways 2026: https://www.lunar.dev/post/the-best-open-source-mcp-gateways-in-2026
