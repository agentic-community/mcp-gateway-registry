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

## If you are not using a proxy, nothing changes

The feature is off unless you set `EGRESS_FORWARD_PROXY_ENABLED=true`. With it unset or `false`:

- Both egress clients resolve, classify and pin exactly as before, and dial direct.
- Nothing is logged, and no metric series is created.
- The mandatory `NO_PROXY` check does not run, so an existing `NO_PROXY` of any shape is accepted.

Exporting `HTTP_PROXY` or `HTTPS_PROXY` on its own changes nothing either, and does not trip the startup check. The flag is the only switch. That is deliberate: enabling the proxy route relaxes DNS-rebinding protection for public destinations, so it has to be an explicit decision rather than a side effect of a variable being present in the environment.

Two unrelated fixes do apply regardless of the flag, both of them repairs to in-cluster hops that should never have been proxy-aware. The `/_egress_internal/generic-upstream-headers` vend now uses the pooled plain client like its sibling, which means connection reuse and one transparent retry on a keep-alive reset rather than a fresh connection per call. The IMDSv2 gateway-name probe now sets `trust_env=False`, matching the telemetry probe; with no proxy configured that makes no difference, and with one configured it stops a link-local request being handed to the proxy.

## What you need to set

Four parameters, and they travel through two different channels.

| Parameter | Value | Channel | Required |
|-----------|-------|---------|----------|
| `EGRESS_FORWARD_PROXY_ENABLED` | `true` | chart-managed | Yes |
| `HTTPS_PROXY` | `http://your-proxy:3128` | pass-through | Yes |
| `HTTP_PROXY` | `http://your-proxy:3128` | pass-through | Yes |
| `NO_PROXY` | must contain the three credential endpoints | pass-through | Yes |
| `EGRESS_FORWARD_PROXY_CA_BUNDLE` | path to a PEM inside the container | chart-managed | Only if the proxy intercepts TLS |

The channels differ on purpose. `EGRESS_FORWARD_PROXY_*` are chart-managed and reserved, so the preflight validator rejects them in a pass-through file. `HTTP_PROXY`, `HTTPS_PROXY` and `NO_PROXY` are ecosystem-standard variables and deliberately **not** reserved, so they ride the arbitrary-environment pass-through each surface already has.

**Set all four on both the registry and the auth-server.** The registry runs the health probes and tool discovery; the auth-server runs the runtime tool call and the login token exchange. Configure one and not the other and you get a confusing half-failure: servers go healthy and routes appear, then tool calls or sign-in fail.

### NO_PROXY has a mandatory minimum

The process **refuses to start** unless `NO_PROXY` excludes all three cloud credential endpoints:

```
169.254.169.254,169.254.170.2,169.254.170.23
```

That is not a style preference. Setting `HTTP_PROXY` re-points every proxy-aware SDK in the process, and the AWS SDK is one: its credential providers would send IAM credential requests to your proxy. Those endpoints speak plain HTTP, so there is no `CONNECT` tunnel and the proxy would see the returned `AccessKeyId`, `SecretAccessKey` and `Token` in cleartext. The addresses are link-local, so a proxy dialing one answers from its own host rather than yours.

`AWS_EC2_METADATA_DISABLED=true` does **not** substitute for these entries. botocore applies that flag in `IMDSFetcher` only, so it covers `169.254.169.254`; `ContainerProvider` is gated purely on `AWS_CONTAINER_CREDENTIALS_RELATIVE_URI`, leaving the ECS and EKS task-credential endpoints exposed.

Add your own internal hostnames after the three. Those are not strictly required, since internal targets stay direct by classification, but they also keep other proxy-aware libraries in the process off the proxy.

### Docker Compose

The flag goes in `.env`. The proxy variables **must** go in `extra_env/`, because the compose files never pass `HTTP_PROXY` from `.env`.

```bash
# .env
EGRESS_FORWARD_PROXY_ENABLED=true
```

```bash
# extra_env/registry.env
HTTP_PROXY=http://corp-proxy.internal:3128
HTTPS_PROXY=http://corp-proxy.internal:3128
NO_PROXY=localhost,127.0.0.1,169.254.169.254,169.254.170.2,169.254.170.23,registry,auth-server,mcpgw-server,keycloak,mongodb,openbao
```

```bash
# extra_env/auth-server.env
HTTP_PROXY=http://corp-proxy.internal:3128
HTTPS_PROXY=http://corp-proxy.internal:3128
NO_PROXY=localhost,127.0.0.1,169.254.169.254,169.254.170.2,169.254.170.23,registry,auth-server,mcpgw-server,keycloak,mongodb,openbao
```

Then start the stack as usual. `scripts/validate-extra-env.sh` runs first and rejects a reserved name if you put the flag in the wrong file.

For a TLS-intercepting proxy, add the CA path to `.env` and bind-mount the PEM in `docker-compose.yml`:

```bash
# .env
EGRESS_FORWARD_PROXY_CA_BUNDLE=/etc/ssl/mcp-gateway/ca-bundle.pem
```
```yaml
# docker-compose.yml, under both the registry and auth-server services
volumes:
  - ./certs/corporate-ca.pem:/etc/ssl/mcp-gateway/ca-bundle.pem:ro
```

### Helm and EKS

```yaml
registry:
  app:
    egressForwardProxyEnabled: true
  extraEnv:
    - name: HTTPS_PROXY
      value: "http://corp-proxy.internal:3128"
    - name: HTTP_PROXY
      value: "http://corp-proxy.internal:3128"
    - name: NO_PROXY
      value: "localhost,127.0.0.1,169.254.169.254,169.254.170.2,169.254.170.23,.svc.cluster.local,.cluster.local,mcpgw-server,auth-server,keycloak"

auth-server:
  app:
    egressForwardProxyEnabled: true
  extraEnv:
    - name: HTTPS_PROXY
      value: "http://corp-proxy.internal:3128"
    - name: HTTP_PROXY
      value: "http://corp-proxy.internal:3128"
    - name: NO_PROXY
      value: "localhost,127.0.0.1,169.254.169.254,169.254.170.2,169.254.170.23,.svc.cluster.local,.cluster.local,mcpgw-server,registry,keycloak"
```

For a TLS-intercepting proxy, create a ConfigMap or Secret holding the PEM and enable the `caBundle` block on both subcharts. The chart derives the env var from `mountPath` and `key`, so the path is never written twice:

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

### Terraform and ECS

One variable covers the flag for both services. The proxy variables go in each service's `*_extra_env` list.

```hcl
egress_forward_proxy_enabled = true

registry_extra_env = [
  { name = "HTTPS_PROXY", value = "http://corp-proxy.internal:3128" },
  { name = "HTTP_PROXY",  value = "http://corp-proxy.internal:3128" },
  { name = "NO_PROXY",    value = "localhost,127.0.0.1,169.254.169.254,169.254.170.2,169.254.170.23,mcp-gateway-v2.local" },
]

auth_server_extra_env = [
  { name = "HTTPS_PROXY", value = "http://corp-proxy.internal:3128" },
  { name = "HTTP_PROXY",  value = "http://corp-proxy.internal:3128" },
  { name = "NO_PROXY",    value = "localhost,127.0.0.1,169.254.169.254,169.254.170.2,169.254.170.23,mcp-gateway-v2.local" },
]
```

Pushing a new image is not enough on ECS. These variables live in the task definition, so `terraform apply` has to run to produce a new revision carrying them. Set `egress_forward_proxy_enabled = true` without applying and the container never sees it: the flag stays absent, the feature stays off, and nothing in the logs explains why. Confirm with:

```bash
aws ecs describe-task-definition --task-definition <arn> \
  --query "taskDefinition.containerDefinitions[].environment[?starts_with(name,'EGRESS_FORWARD_PROXY')]"
```

For a TLS-intercepting proxy, bake the bundle into the image or mount it from the EFS volume the stack already provisions, then name the path:

```hcl
egress_forward_proxy_ca_bundle = "/etc/ssl/mcp-gateway/ca-bundle.pem"
```

### Settings worth reviewing, on any surface

Neither has a first-class field, so set them through the same pass-through as the proxy variables.

- `HEALTH_CHECK_TIMEOUT_SECONDS` defaults to 2. It is a flat timeout, so the connect budget covers the TCP connection to the proxy, the `CONNECT` round trip and the TLS handshake. Measure before changing it: 2 seconds was enough against a proxy on the same host in testing, and a proxy across a corporate WAN is the case that needs more.
- `EGRESS_HTTP_POOL_MAX_CONNECTIONS` defaults to 100 and becomes the simultaneous-tunnel count your proxy must tolerate. Lower it if the proxy rate-limits `CONNECT`.

Every parameter above, with its name on all three surfaces, is in the [Unified Parameter Reference](../unified-parameter-reference.md).

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
