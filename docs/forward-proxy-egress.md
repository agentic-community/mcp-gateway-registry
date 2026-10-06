# Forward-Proxy Egress

The registry fetches external MCP servers directly. Where a network gives the registry no route to the internet and requires all outbound traffic to pass through a corporate forward proxy, every external server stays permanently unreachable: the health probe times out, the server is marked unhealthy, and the generated nginx config comments out its `location` block, so requests to it return 405.

Setting `EGRESS_FORWARD_PROXY_ENABLED=true` lets the SSRF-guarded egress clients route through that proxy. The flag is off by default and changes nothing until you set it.

## When you need it

Turn it on when all three of these describe your deployment:

- The registry and auth-server pods or tasks have no NAT gateway, no internet route, or an egress firewall that drops direct outbound connections.
- You already set `HTTP_PROXY` and `HTTPS_PROXY` for other software in the same network.
- You register MCP servers, agents, or skills hosted outside your network.

You do not need it for a registry that only proxies to in-cluster servers. Those resolve to internal addresses, which stay on the direct path whether or not this flag is set.

## What it does

When the flag is on, the guard sends a request through the proxy only when all of these hold:

1. A proxy is configured for the request's scheme, from `HTTPS_PROXY` for an `https` target or `HTTP_PROXY` for an `http` one.
2. `NO_PROXY` does not name the target host.
3. Every address the host resolves to is on the public internet.

Anything else takes the direct path it takes today, pinned to a validated IP address. An `https` target through the proxy becomes a `CONNECT` tunnel, so the proxy sees only `host:port` and the request body stays inside TLS.

Routing comes from the DNS classification rather than from a list you maintain. Register an internal MCP server tomorrow and its probe still goes direct, with no edit to `NO_PROXY`.

The flag applies to every guarded egress path, so one setting covers the health probe, tool discovery, the runtime tool call that carries an injected egress credential, the OAuth token exchange that mints that credential, and peer federation. Set it on the registry and on the auth-server. The registry runs the probe and tool discovery; the auth-server runs the runtime hop and the token exchange. Setting it on one service fixes half the paths and produces a partial failure that is hard to read.

## The security trade-off

The guard normally defeats DNS rebinding by pinning: it resolves the hostname, classifies every answer, and then connects to a validated IP address, leaving no window for the name to rebind to a private address between the check and the connect.

A `CONNECT` tunnel cannot carry that pin. httpcore builds the tunnel's TLS handshake with `server_hostname` taken from the request URL and never reads the SNI override that pinning sets, so a pinned request through a tunnel would verify the upstream certificate against an IP address and fail. For a proxied target, the guard therefore connects by hostname and the proxy resolves that name itself.

Everything else survives. Before any `CONNECT`, the guard still resolves the host and classifies every answer, and it still denies:

- cloud and workload credential endpoints, including EC2 IMDS, ECS task credentials, EKS Pod Identity, and the Alibaba metadata address,
- link-local, unspecified, reserved, and multicast addresses,
- private, loopback, and CGNAT addresses, unless your allowlist admits them,
- obfuscated IPv4 spellings and IPv6 forms that embed an IPv4 address.

One blocked answer denies the whole request. Redirects are re-checked on every hop, so a `302` pointing at `169.254.169.254` is denied at the second hop before any connection.

What remains is that a hostname passing classification here could rebind before the proxy dials it. Such a connection is made from the proxy's network position, not the registry's, so an attacker reaches whatever the proxy can reach. Configure your forward proxy to deny `CONNECT` to RFC-1918, loopback, and link-local destinations, and that gap closes.

Two further consequences of routing traffic through a proxy:

- The proxy sees every destination you reach. For an `https` target it sees `host:port`; for an `http` target it sees the full URL, every header, and the body.
- A `CONNECT` to an `http://` proxy is itself cleartext, so anyone on the path between the pod and the proxy sees the destination `host:port`. The payload stays inside TLS, so this discloses metadata only.

Requests that carry a secret accept only `https` through a proxy. The egress OAuth token exchange, the injected egress credential, and peer federation all refuse an `http` target when a proxy applies, because httpcore's plain forward path would hand the full URL and the `Authorization` header to the proxy in cleartext. If you hit that error for an in-cluster upstream, add the host to `NO_PROXY` so it is dialed directly.

## Prerequisite: the pod must resolve public DNS

Classification runs locally, before the `CONNECT`, so the pod needs a resolver that answers queries for public names. Where a split-horizon resolver cannot, the guard fails closed with `DNS resolution failed` rather than timing out. Point the resolver at a forwarder that can answer public queries.

## Turning it on

### Docker Compose

Set the flag in `.env`:

```bash
EGRESS_FORWARD_PROXY_ENABLED=true
```

Put the proxy variables in `extra_env/registry.env` and `extra_env/auth-server.env`, which both services already read:

```bash
# extra_env/registry.env and extra_env/auth-server.env
HTTP_PROXY=http://corp-proxy.internal:3128
HTTPS_PROXY=http://corp-proxy.internal:3128
NO_PROXY=localhost,127.0.0.1,169.254.169.254,169.254.170.2,169.254.170.23,registry,auth-server,mcpgw-server,keycloak,mongodb
```

### Helm and EKS

```yaml
registry:
  app:
    egressForwardProxyEnabled: true
  extraEnv:
    - name: HTTP_PROXY
      value: http://corp-proxy.internal:3128
    - name: HTTPS_PROXY
      value: http://corp-proxy.internal:3128
    - name: NO_PROXY
      value: localhost,127.0.0.1,169.254.169.254,169.254.170.2,169.254.170.23,.svc.cluster.local,.cluster.local,mcpgw-server,auth-server,keycloak

auth-server:
  app:
    egressForwardProxyEnabled: true
  extraEnv:
    - name: HTTP_PROXY
      value: http://corp-proxy.internal:3128
    - name: HTTPS_PROXY
      value: http://corp-proxy.internal:3128
    - name: NO_PROXY
      value: localhost,127.0.0.1,169.254.169.254,169.254.170.2,169.254.170.23,.svc.cluster.local,.cluster.local,mcpgw-server,registry,keycloak
```

`HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY` are deliberately absent from `charts/*/reserved-env-names.txt`. Reserving them would make the chart reject the configuration this feature needs.

### Terraform and ECS

```hcl
egress_forward_proxy_enabled = true

registry_extra_env = [
  { name = "HTTP_PROXY", value = "http://corp-proxy.internal:3128" },
  { name = "HTTPS_PROXY", value = "http://corp-proxy.internal:3128" },
  { name = "NO_PROXY", value = "localhost,127.0.0.1,169.254.169.254,169.254.170.2,169.254.170.23,mcp-gateway-v2.local" },
]

auth_server_extra_env = [
  { name = "HTTP_PROXY", value = "http://corp-proxy.internal:3128" },
  { name = "HTTPS_PROXY", value = "http://corp-proxy.internal:3128" },
  { name = "NO_PROXY", value = "localhost,127.0.0.1,169.254.169.254,169.254.170.2,169.254.170.23,mcp-gateway-v2.local" },
]
```

The ECS task definitions pick up the flag for both services from the single `egress_forward_proxy_enabled` variable.

## NO_PROXY

`NO_PROXY` is an override, not the routing mechanism. Internal targets are already detected by classification, so a missing entry no longer breaks an in-cluster hop. Use it for two things: public addresses this host reaches directly, which saves a pointless tunnel, and in-cluster hostnames you want to keep explicit.

The guard reads the forms curl reads. An entry matches an exact host (`mcpgw-server`), a `host:port` pair (`acme.example:443`), a domain suffix (`svc.cluster.local` matches `a.svc.cluster.local`, with or without a leading dot), or everything (`*`, which behaves the same as leaving the flag off). Upper case wins over lower case when both spellings are set.

`NO_PROXY` **must** include the three cloud credential endpoints, `169.254.169.254`, `169.254.170.2` and `169.254.170.23`. This is enforced: with the flag on and any of them missing, the process refuses to start and names the remedy. The reason is not cosmetic. The AWS SDK reads `HTTP_PROXY` itself, independently of this feature, so once the variable is set its instance-metadata credential provider sends IAM credential requests to your proxy. Observed on a test deployment: the proxy established connections to `169.254.169.254`, the SDK read the instance role name from `/latest/meta-data/iam/security-credentials/`, and then used the resulting credentials. A proxy operator should never see a credential request, and a proxy that can reach a metadata endpoint itself would be answering from its own network position rather than yours.

`AWS_EC2_METADATA_DISABLED=true` is not accepted as a substitute, and the startup error says so. botocore applies that flag in `IMDSFetcher` only, so it covers `169.254.169.254`; `ContainerProvider` is gated purely on `AWS_CONTAINER_CREDENTIALS_RELATIVE_URI`, which leaves the ECS and EKS task-credential endpoints exposed. `NO_PROXY` is the only control covering all three.

A CIDR entry does not work. Neither curl nor httpx expands one, so `NO_PROXY=10.0.0.0/8` fails to match `10.1.2.3`. The guard logs a warning for a CIDR-shaped entry. Name the host or use a domain suffix instead.

Include your Keycloak hostname if you self-host it, along with the registry, auth-server, mcpgw-server, and database hostnames.

## TLS-intercepting proxies

Most corporate proxies terminate TLS and re-sign it with an internal CA. Through such a proxy, the certificate the registry sees comes from that CA rather than a public root, so verification fails until the registry trusts it. Point `EGRESS_FORWARD_PROXY_CA_BUNDLE` at a PEM bundle holding the CA:

```bash
EGRESS_FORWARD_PROXY_CA_BUNDLE=/etc/ssl/mcp-gateway/ca-bundle.pem
```

The bundle loads on top of the default roots, so public certificates keep verifying on the direct path. A path that is missing, unreadable, or not valid PEM fails the process at startup rather than surfacing as a certificate error on the first proxied request.

Do not use `SSL_CERT_FILE` for this. httpx reads it, so it appears to work, but `ssl.create_default_context(cafile=...)` replaces the trust store instead of adding to it. A bundle holding only your corporate CA then breaks TLS to every direct-path target, including `NO_PROXY` hosts with public certificates, and you get a second outage with an unrelated cause.

The file has to exist inside the container, so get it in there as well as naming its path:

- Compose: bind-mount it read-only. `docker-compose.yml` carries a commented example beside the variable.
- Helm: create a ConfigMap or Secret holding the PEM and set the `caBundle` block. The chart derives the env var from `caBundle.mountPath` and `caBundle.key`, so you never write the path twice.

```yaml
registry:
  caBundle:
    enabled: true
    existingConfigMap: corporate-ca   # or existingSecret
    key: ca-bundle.pem
    mountPath: /etc/ssl/mcp-gateway
auth-server:
  caBundle:
    enabled: true
    existingConfigMap: corporate-ca
    key: ca-bundle.pem
    mountPath: /etc/ssl/mcp-gateway
```

- ECS: bake the bundle into the image or mount it from the EFS volume the stack already provisions, then set `egress_forward_proxy_ca_bundle` to that path.

If your proxy endpoint is itself `https://` and signed by the internal CA, the same bundle verifies the connection to the proxy.

## Settings to revisit

`HEALTH_CHECK_TIMEOUT_SECONDS` defaults to 2 and may need raising. It is a flat timeout, so the connect budget has to cover the TCP connection to the proxy, the `CONNECT` round trip, and the TLS handshake. Two seconds for all three can produce the same timeout you are trying to fix, with a different cause.

No deployment surface exposes it as a first-class setting, so set it through the same pass-through as the proxy variables: `extra_env/registry.env` on Compose, `registry.extraEnv` on Helm, `registry_extra_env` on Terraform. Measure before changing it. Against a proxy on the same host, 2 seconds was enough in testing; a proxy across a corporate WAN is the case that needs more.

Check `EGRESS_HTTP_POOL_MAX_CONNECTIONS`, which defaults to 100. Every proxied connection now terminates at the proxy, so this becomes the number of simultaneous tunnels the proxy has to tolerate. Health checks run in batches of 10 with a pause between them, which limits the burst, but a proxy that rate-limits `CONNECT` will drop probes in waves and flap servers between healthy and unhealthy. For a few hundred registered servers against one proxy, 25 is a safer starting point.

Leave `EGRESS_HTTP_POOL_MAX_KEEPALIVE` at 20 or raise it. Keep-alive amortizes the extra `CONNECT` round trip, and a low value makes every probe pay tunnel setup again.

Internal targets still need an entry in `SSRF_ALLOWED_HOSTS` or `SSRF_ALLOWED_CIDRS` to pass classification. That is unchanged by this feature.

## Verifying it works

The registry and auth-server each log one line at startup when the flag resolves. The proxy host appears with credentials stripped:

```
SSRF guard: forward proxy enabled (http=http://corp-proxy.internal:3128 https=http://corp-proxy.internal:3128 no_proxy_entries=6). IP-rebind pinning is relaxed for targets that resolve exclusively public.
```

A flag set with no proxy configured logs a warning and behaves as if the flag were off:

```
SSRF guard: EGRESS_FORWARD_PROXY_ENABLED is true but neither HTTP_PROXY nor HTTPS_PROXY is set; egress stays direct and pinned
```

The System Config page shows the flag and the CA bundle path under Egress Credential Vault. The proxy URL is not shown, because it can embed basic-auth credentials.

Per-request routing decisions are logged at DEBUG:

```
SSRF guard[proxy]: acme.example resolves public ['93.184.216.34'], routing via forward proxy
SSRF guard[proxy]: internal.svc.cluster.local resolves internal, staying direct and pinned
```

The counter `mcpgw_egress_forward_proxy_requests_total` carries `profile`, `route`, and `outcome` labels on the Prometheus endpoint at `:9464/metrics`:

```
mcpgw_egress_forward_proxy_requests_total{profile="proxy",route="proxied",outcome="ok"} 536
mcpgw_egress_forward_proxy_requests_total{profile="federation",route="proxied",outcome="ok"} 4
```

Nothing is recorded while the flag is off, so the counter costs an unconfigured deployment nothing. With the flag on, every guarded request lands in one of two routes. `route="proxied"` went through the proxy. `route="direct"` was kept direct and pinned, either because `NO_PROXY` named the host, because no proxy is set for that scheme, or because the target resolved to an internal address. On a deployment you expect to be fully proxied, `route="direct"` traffic is the signal to check `NO_PROXY` first.

The `profile` label shows which trust surface the request came from, so a deployment using egress credential injection should show `egress-upstream` and `credentialed-oauth` alongside `proxy`.

End to end, an external server that was stuck unhealthy should go healthy within one health-check cycle, and its nginx `location` block should appear live rather than commented out:

```bash
docker compose exec registry sh -c 'grep -n -A3 "location /your-server/" /etc/nginx/conf.d/*.conf'
```

## Troubleshooting

An external server stays unhealthy after you set the flag. Check the startup log for the INFO line. No line means the flag did not reach the process, and a warning about missing proxy variables means they did not either. Confirm you set both on both services.

Every proxied request fails with a certificate error. Your proxy intercepts TLS. Set `EGRESS_FORWARD_PROXY_CA_BUNDLE` and mount the PEM.

Every proxied request fails with `The proxy_ssl_context argument is not allowed for the http scheme`, and sign-in returns `?error=oauth2_callback_failed`. This affects 1.32.0 only, whenever `EGRESS_FORWARD_PROXY_CA_BUNDLE` is set and the proxy URL is `http://`. Upgrade to 1.32.1, or leave the bundle empty until you do: a proxy that does not intercept TLS needs no bundle (#1849).

Probes fail with `DNS resolution failed` rather than a timeout. The pod cannot resolve public names. Classification needs that lookup before the `CONNECT`.

An in-cluster hop breaks after you set `NO_PROXY`. A CIDR entry does not match. Replace it with hostnames or a domain suffix.

A credential-bearing request is refused with a message about an `http` target. The upstream resolves to a public address over plain HTTP, so the credential would reach the proxy in cleartext. Move the upstream to `https`, or add it to `NO_PROXY` if it is reachable directly.

Probes time out in waves and servers flap. The proxy is rate-limiting or capping concurrent tunnels. Lower `EGRESS_HTTP_POOL_MAX_CONNECTIONS`.

## Known limitation: virtual-server tool calls

A tool call to an ordinary MCP server path goes through the auth-server, whose egress this feature covers. A tool call fanned out by a **virtual server** does not. The generated nginx config gives each virtual-server member its own internal location:

```nginx
location /_vs_backend_cloudflare_docs {
    internal;
    proxy_pass https://docs.mcp.cloudflare.com/mcp;
```

That location is reached by a Lua subrequest, so nginx performs the upstream fetch itself, and nginx has no environment-proxy support for `proxy_pass`. Measured on a test deployment: the same tool through `/cloudflare-docs` opened four new tunnels through the proxy, while the same tool through `/virtual/dev-essentials` opened none.

In a proxy-only network, virtual-server tool calls therefore fail even with this feature enabled. Use the per-server paths there. Closing the gap needs `ngx_http_proxy_connect_module` or an equivalent, which is a separate change.

## Not supported

SOCKS proxies need the `socksio` extra and are not implemented. Only `http://` and `https://` proxy endpoints work; anything else raises at the first request that would have used it rather than falling back to a direct dial.

There is no way to force an internal target through the proxy. `NO_PROXY` moves a target off the proxy, and nothing moves one onto it against the classification.

## Related

- [Unified Parameter Reference](unified-parameter-reference.md) for both variables across all three deployment surfaces.
- [Security Guidelines](SECURITY_GUIDELINES.md) for the SSRF invariants this feature works within.
- [Configuration Reference](configuration.md) for the rest of the environment.
