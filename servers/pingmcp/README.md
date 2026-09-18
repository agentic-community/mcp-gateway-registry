# pingmcp (vendored)

A tiny, fast Go streamable-http MCP server with a single `echo` tool. Used here as a
**fast upstream** for end-to-end load testing of the gateway's per-request auth check
(`/validate`), so the measurement is bounded by the gateway, not the upstream (issue #1652).

Canonical, standalone source: **https://github.com/aarora79/pingmcp**
This copy is vendored so the `pingmcp-server` docker-compose service can build locally.

## Run in the stack (opt-in benchmark profile)

```bash
docker compose --profile benchmark up -d pingmcp-server
# add pingmcp-server to SSRF_ALLOWED_HOSTS in .env so the registry health check allows it
uv run python api/registry_management.py --registry-url http://localhost --token-file .token \
    register --config cli/examples/pingmcp.json --overwrite
```
