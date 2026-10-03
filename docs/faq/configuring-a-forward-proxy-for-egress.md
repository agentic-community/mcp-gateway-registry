# How do I configure an external proxy server when my corporate network has no direct internet access?

Set `EGRESS_FORWARD_PROXY_ENABLED=true` and point `HTTPS_PROXY` at your corporate proxy. Both go on the registry **and** the auth-server. Full detail, including the security trade-off, is in [Forward-Proxy Egress](../forward-proxy-egress.md).

## The symptom this fixes

Every external MCP server you register goes `unhealthy`, and calls to it return 405.

The registry probes each registered server to decide whether to route to it. Those probes are made by a hardened HTTP client that pins each connection to a validated IP address, and pinning requires a custom transport. `httpx` only reads `HTTP_PROXY` and `HTTPS_PROXY` for its own default transport, so a custom one never sees them. Where the pod has no route to the internet, the probe times out, the server is marked unhealthy, and the generated nginx config comments out its `location` block. A `POST` then falls through to the catch-all and returns 405.

You can confirm it is an egress problem rather than a broken upstream by running both from inside the pod:

```bash
curl -s -o /dev/null -w '%{http_code}\n' https://your-mcp-server.example/mcp          # honours the proxy
curl -s -o /dev/null -w '%{http_code}\n' --noproxy '*' https://your-mcp-server.example/mcp  # direct
```

A status code from the first and a timeout from the second means the upstream is reachable only through the proxy.

## Configuration

The flag is chart-managed, so it lives with your other settings. The proxy variables are standard environment variables and ride the arbitrary-environment pass-through each deployment surface already has. They are deliberately absent from `charts/*/reserved-env-names.txt`, because reserving them would make the charts reject this configuration.

### Docker Compose

```bash
# .env
EGRESS_FORWARD_PROXY_ENABLED=true
```

```bash
# extra_env/registry.env and extra_env/auth-server.env
HTTP_PROXY=http://corp-proxy.internal:3128
HTTPS_PROXY=http://corp-proxy.internal:3128
NO_PROXY=localhost,127.0.0.1,169.254.169.254,169.254.170.2,169.254.170.23,registry,auth-server,mcpgw-server,keycloak,mongodb
```

### Helm

```yaml
registry:
  app:
    egressForwardProxyEnabled: true
  extraEnv:
    - name: HTTPS_PROXY
      value: http://corp-proxy.internal:3128
    - name: HTTP_PROXY
      value: http://corp-proxy.internal:3128
    - name: NO_PROXY
      value: localhost,127.0.0.1,169.254.169.254,169.254.170.2,169.254.170.23,.svc.cluster.local,mcpgw-server,auth-server,keycloak

auth-server:
  app:
    egressForwardProxyEnabled: true
  extraEnv:
    - name: HTTPS_PROXY
      value: http://corp-proxy.internal:3128
    - name: HTTP_PROXY
      value: http://corp-proxy.internal:3128
    - name: NO_PROXY
      value: localhost,127.0.0.1,169.254.169.254,169.254.170.2,169.254.170.23,.svc.cluster.local,mcpgw-server,registry,keycloak
```

### Terraform and ECS

```hcl
egress_forward_proxy_enabled = true

registry_extra_env = [
  { name = "HTTPS_PROXY", value = "http://corp-proxy.internal:3128" },
  { name = "HTTP_PROXY", value = "http://corp-proxy.internal:3128" },
  { name = "NO_PROXY", value = "localhost,127.0.0.1,169.254.169.254,169.254.170.2,169.254.170.23,mcp-gateway-v2.local" },
]
auth_server_extra_env = [
  { name = "HTTPS_PROXY", value = "http://corp-proxy.internal:3128" },
  { name = "HTTP_PROXY", value = "http://corp-proxy.internal:3128" },
  { name = "NO_PROXY", value = "localhost,127.0.0.1,169.254.169.254,169.254.170.2,169.254.170.23,mcp-gateway-v2.local" },
]
```

## Set it on both services

The registry runs the health probes and tool discovery. The auth-server runs the runtime tool call and the OAuth exchange that mints an injected egress credential. Configure one and not the other and you get a confusing half-failure: servers go healthy and routes appear, then tool calls fail.

## What gets proxied, and what does not

A target goes through the proxy only when every address it resolves to is on the public internet. An internal target stays direct and keeps its IP pin with no extra configuration, so in-cluster MCP servers are unaffected and you do not have to list them anywhere.

`NO_PROXY` is an override rather than the routing mechanism. Use it for public addresses this host reaches directly, which saves a pointless tunnel. Listing your in-cluster hostnames as well costs nothing and is good hygiene.

Keep the cloud metadata addresses in `NO_PROXY`. The AWS SDK reads `HTTP_PROXY` on its own, so once you set it the SDK's instance-metadata credential provider starts sending IAM credential requests to your proxy. Helm and Terraform set `AWS_EC2_METADATA_DISABLED=true` to close that at the source; on Docker Compose, where an EC2 instance role may be the credential source for Amazon Bedrock, these `NO_PROXY` entries are what stop it.

A CIDR entry in `NO_PROXY` does not work. Neither curl nor httpx expands one, so `10.0.0.0/8` fails to match `10.1.2.3`. Name the host, or use a domain suffix such as `.svc.cluster.local`. The guard logs a warning if it sees a CIDR-shaped entry.

## Two prerequisites that cause confusing failures

**The pod must resolve public DNS.** Classification runs locally before the tunnel is opened, so the lookup has to succeed even though the connection goes through the proxy. Where a split-horizon resolver cannot answer public queries, the guard fails closed with `DNS resolution failed` instead of timing out. Give the resolver a forwarder that can answer public queries.

**Your proxy must allow `CONNECT` to port 443.** Most do. If it restricts the port list and your upstream listens elsewhere, you get a 403 from the proxy that reads like a registry fault.

## If your proxy intercepts TLS

Most corporate proxies terminate TLS and re-sign with an internal CA, which makes every proxied certificate fail against public roots. Point `EGRESS_FORWARD_PROXY_CA_BUNDLE` at a PEM bundle holding that CA, and mount the file into both containers. The bundle is trusted in addition to the default roots.

Do not use `SSL_CERT_FILE` for this. httpx reads it, so it appears to work, but it **replaces** the trust store rather than adding to it, and a bundle holding only your corporate CA then breaks TLS to every direct-path target.

## Verifying it

Each service logs one line at startup, with the proxy host credential-stripped:

```
SSRF guard: forward proxy enabled (http=http://corp-proxy.internal:3128
https=http://corp-proxy.internal:3128 no_proxy_entries=6).
IP-rebind pinning is relaxed for targets that resolve exclusively public.
```

No line means the flag never reached the process. A warning naming `HTTP_PROXY` means the flag is on but no proxy is configured, in which case egress stays direct and the deployment behaves as though the flag were off.

The counter `mcpgw_egress_forward_proxy_requests_total{profile,route,outcome}` on the Prometheus endpoint shows the split between `route="proxied"` and `route="direct"`. On a deployment you expect to be fully proxied, `direct` traffic is the signal to check `NO_PROXY` first.

## Security trade-off to be aware of before enabling

For a proxied target, DNS-rebinding protection is given up. A `CONNECT` tunnel takes its TLS hostname from the request URL and ignores the SNI override that pinning sets, so a pinned request through a tunnel would verify the certificate against an IP address and fail.

Everything else is preserved. The guard still resolves the host and classifies every answer before opening the tunnel, and cloud metadata, workload credential, link-local, reserved and multicast addresses stay denied. The relaxation applies to public destinations only, so internal targets keep their pin. Requests carrying a secret accept only `https` through a proxy.

A rebind that slips through is dialled from the proxy's network position rather than the registry's, so configure your proxy to deny `CONNECT` to RFC-1918, loopback and link-local destinations and the residual gap closes.

The proxy also sees every destination you reach, and for an `http` target it sees the full URL, headers and body.

## Related

- [Forward-Proxy Egress](../forward-proxy-egress.md) for the full operator guide and troubleshooting.
- [How do I monitor the health of MCP servers?](monitoring-server-health.md) for reading health state.
- [Unified Parameter Reference](../unified-parameter-reference.md) for both parameters across all three surfaces.
