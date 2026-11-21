# Service Mesh Deployment (Helm / EKS)

The Helm stack chart can run inside an Istio service mesh. A mesh is highly encouraged but not required: with mesh
support off (the default), the chart renders as it did before mesh support. The one exception is Keycloak: the first
upgrade to this chart version restarts it once, because a pod annotation added for mesh support changes its pod
template (see Keycloak's public hostname). There are no application code changes for the
mesh, and no code path depends on it.

This page covers what the chart does when you turn the mesh on, what you have to provide, the rollout order that avoids
an outage, and the limits of the first version.

## What the chart does, and does not do

When you enable the mesh, the chart:

1. Labels its own pods for sidecar injection with `sidecar.istio.io/inject: "true"`: the registry, auth-server and mcpgw
   Deployments, the three bootstrap Jobs, and the MongoDB replica-set pods (through the `MongoDBCommunity` resource, so
   the operator rolls them).
2. Renders a namespaced `PeerAuthentication` for the mTLS posture.
3. Renders one ALLOW `AuthorizationPolicy` per destination workload.

The chart does not install Istio. The control plane (istiod and the injector webhook) is a cluster prerequisite, usually
owned by the platform team. Install it with `istioctl`, the Istio Helm charts, or your cluster's add-on mechanism before
you enable mesh support here.

## Prerequisites

- An Istio control plane in the cluster. The chart was validated against Istio 1.30 and 1.32. Install it from
  `https://blob.istio.io/istio-release/charts`.
- Kubernetes 1.29 or newer, with Istio native sidecars enabled (`ENABLE_NATIVE_SIDECARS=true` on istiod). A native
  sidecar starts before the app container and stops after it. The bootstrap Jobs depend on this: with a classic sidecar,
  a Job pod never reaches Completed and `helm upgrade --wait` hangs.
- The namespace labeled for injection: `istio-injection=enabled` for a control plane installed without a revision, or
  `istio.io/rev=<revision>` (for example `1-32-0`) for a revisioned one. The chart's labels opt in its own pods, but the
  subchart pods it does not template (Keycloak, PostgreSQL, OpenBao and the MongoDB operator) join the mesh only through
  the namespace label.

To move to a new Istio version, use a revision-based (canary) upgrade, which can skip one minor version: install the new
`istiod` with `revision=<new>`, relabel the namespace `istio.io/rev=<new>`, restart the workloads in the order of the
rollout below (MongoDB members one at a time, the registry last), confirm every proxy reports the new version, then
uninstall the old `istiod` and set `istio-base` `defaultRevision=<new>`.

## Enabling it

```yaml
global:
  serviceMesh:
    enabled: true
    type: istio          # only istio is implemented today
    mtls:
      mode: STRICT        # STRICT (default) or PERMISSIVE
    istio:
      holdApplicationUntilProxyStarts: true
    authorizationPolicies:
      enabled: true
      trustDomain: cluster.local
```

`global.serviceMesh.type` is a discriminator so other meshes can be added later. A value other than `istio` renders no
Istio objects or labels.

## Rollout order

The defaults are the hardened end state: STRICT mTLS and the AuthorizationPolicies on. Both deny plaintext arriving at a
pod with a sidecar, so they cannot be switched on while some pods still have none. Pods without a sidecar cannot reach
pods that have one, and MongoDB replication breaks while the operator rolls the replica set. Enable the mesh for the
first time in two stages:

1. Label the namespace, then upgrade with the strict layers relaxed:

   ```yaml
   global:
     serviceMesh:
       enabled: true
       mtls:
         mode: PERMISSIVE
       authorizationPolicies:
         enabled: false
   ```

   The app Deployments roll with sidecars and the operator rolls the MongoDB members.
2. Restart the pods the upgrade does not recreate: `kubectl rollout restart` the PostgreSQL and Keycloak StatefulSets
   and the MongoDB operator Deployment, and delete the OpenBao pod (its StatefulSet uses the `OnDelete` strategy).
   OpenBao boots sealed and its unseal sidecar unseals it. Then restart the registry Deployment so nginx picks up
   Keycloak's new address.
3. Confirm every pod in the namespace has an `istio-proxy` container and nothing but the load balancer still arrives in
   plaintext (see the query below), then upgrade again without the two overrides. The bootstrap Jobs run again as
   post-upgrade hooks, which exercises the Job policies.

An install that enables the mesh at first install, before any pod exists, can skip the staging: every pod starts with a
sidecar.

## mTLS posture: STRICT and PERMISSIVE

The default is `STRICT` (`global.serviceMesh.mtls.mode`): every pod in the namespace rejects plaintext. It covers
inbound traffic only: the registry and auth-server can still reach MCP backends elsewhere, meshed or not. Two callers
cannot present a mesh identity, so the chart keeps their ports `PERMISSIVE` with workload-scoped policies:

| Exception                | Caller                    | Ports                                                   | Value                                          |
|--------------------------|---------------------------|---------------------------------------------------------|------------------------------------------------|
| Registry entry point     | the load balancer         | 8080, 8443 on the registry pods                         | `mtls.strictExceptions.registryEntrypoint`     |
| MongoDB operator webhook | the Kubernetes API server | `mongodbOperatorWebhookPort` (1993) on the operator pod | `mtls.strictExceptions.mongodbOperatorWebhook` |

The AuthorizationPolicies still apply on those ports. With `ingressGateway.routeIngress` on, the registry exception is
dropped automatically: only the gateway reaches the registry then, so the registry is fully STRICT.

### Switching to PERMISSIVE

Set `global.serviceMesh.mtls.mode: PERMISSIVE` to accept plaintext as well as mTLS, then `helm upgrade`. Use it for the
first rollout, to bring in a caller that cannot be meshed yet, or as the rollback if STRICT blocks something. It does
not exempt a caller from the AuthorizationPolicies; set `authorizationPolicies.enabled: false` too if that caller must
reach a policy-protected workload.

Before switching back to STRICT, check that nothing else still arrives in plaintext:

```promql
sum by (source_workload, destination_workload) (
  increase(istio_requests_total{reporter="destination", destination_workload_namespace="<namespace>", connection_security_policy!="mutual_tls"}[1h]))
```

Expect only `unknown -> registry` (the load balancer). Keep any override in the values file you deploy with, so a later
`helm upgrade` does not revert it.

## The AuthorizationPolicy set

Each policy selects one destination workload and lists the source identities and ports allowed to reach it. Ports are
container ports, not Service ports.

| Destination                       | Allowed sources                                                                   | Container ports                                         |
|-----------------------------------|-----------------------------------------------------------------------------------|---------------------------------------------------------|
| registry                          | any source (public entry point, reached by the load balancer)                     | 8080, 8443                                              |
| registry                          | auth-server                                                                       | 8091 (internal egress-token vend)                       |
| auth-server                       | registry, mcpgw                                                                   | 8888                                                    |
| mcpgw                             | registry, auth-server                                                             | `mcpgw.app.port` (8000; the Service exposes it as 8003) |
| keycloak (if `keycloak.create`)   | registry, auth-server, mcpgw, keycloak-configure Job                              | 8080                                                    |
| keycloak (if `keycloak.create`)   | other Keycloak members                                                            | 7800                                                    |
| postgresql (if `keycloak.create`) | keycloak                                                                          | 5432                                                    |
| openbao (if `openbao.enabled`)    | registry, auth-server, openbao-init Job                                           | 8200                                                    |
| mongodb (in-cluster only)         | registry, auth-server, setup-mongodb Job, other replica-set members, the operator | 27017                                                   |
| registry, auth-server, mcpgw      | any source (Prometheus)                                                           | 9464                                                    |

Sources are Istio service-account principals of the form `<trustDomain>/ns/<namespace>/sa/<serviceAccount>`. Set
`authorizationPolicies.trustDomain` if your mesh uses a trust domain other than `cluster.local`. mcpgw and the
setup-mongodb Job run as their own ServiceAccounts so they can be named here.

Two consequences of the deny-by-default behavior:

- A caller with no sidecar is admitted only by a rule with no source restriction. Only the registry entry ports and the
  metrics port have such rules.
- The policies may be the only network-level control. On a cluster where the CNI does not enforce Kubernetes
  NetworkPolicies, the chart's NetworkPolicies render but do nothing, and these policies are the boundary.

No egress policy is placed on any workload. The registry and auth-server proxy to MCP backends registered at runtime,
which a fixed egress allowlist would break.

Set `authorizationPolicies.enabled` to false to run the mesh for mTLS without the identity layer.

## Where the sidecars do not parse HTTP

A sidecar that handles a port as HTTP parses each request, enforces its own limits and adds headers. Three hops break
under that, so the chart declares their ports as TCP with `Sidecar` resources:

| Hop                                            | Problem as HTTP                                                                                                                                                                        | Declared TCP on                                                                             |
|------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| registry nginx to the auth-server and Keycloak | routing by the client's public `Host` loses the mTLS identity (`RBAC: access denied`)                                                                                                  | registry egress                                                                             |
| registry nginx to the auth-server `/validate`  | nginx sends the MCP request body base64-encoded in an `X-Body` header (bodies up to 1 MiB); Envoy caps request headers at 60 KiB and answers 431, so tool calls over about 45 KiB fail | auth-server inbound 8888 and 9464                                                           |
| registry and auth-server to OpenBao            | OpenBao's list API uses the HTTP method `LIST`, which Envoy rejects with 400 before any policy runs; Kiali shows the caller as `unknown`                                               | OpenBao inbound 8200, registry egress, auth-server egress (when its OBO cache uses OpenBao) |

The AuthorizationPolicies check only identity and port, so they apply to these TCP connections unchanged. The cost is
HTTP-level telemetry: Kiali shows these hops as TCP.

## Headers the sidecars add

Where a sidecar still handles HTTP (the registry's inbound from the load balancer, mcpgw, Keycloak):

- `X-Forwarded-For` is not changed. Sidecars run with `use_remote_address: false`, so the application-level
  `TRUSTED_PROXY_HOPS` attribution is the same as without the mesh.
- `X-Forwarded-Proto` is added when absent: `https` on a connection that arrives over mTLS as TCP, `http` otherwise.
  This is what moved Keycloak's issuer; see the next sections.
- `X-Forwarded-Client-Cert` is appended (`APPEND_FORWARD`), so a client-supplied value passes through with the caller's
  SPIFFE identity added. Nothing in the registry, auth-server or mcpgw reads it; do not start trusting it without
  switching the mode to `SANITIZE_SET`.
- `X-Request-Id` is generated when absent. The registry and auth-server treat it as a client correlation value only;
  audit records keep their own server-generated identifier.
- The request path is normalized (`.` and `..` segments) before nginx sees it, and request headers are capped at 60 KiB
  in total.

A sidecar in its default interception mode hands every connection to the application from `127.0.0.6`, its local
address, instead of the caller's. The registry can avoid this with TPROXY or the ingress gateway (see Client IP in the
audit log). The auth-server and mcpgw
still see `127.0.0.6` as their peer, but they take the caller's address from the `X-Real-IP` header the registry's nginx
sets, so their logs are unaffected. The auth-server's uvicorn trusts forwarded headers only from `127.0.0.1` and `::1`,
which never included the registry's pod IP, so its trust decision is unchanged.

## Ingress gateway

The chart can put an Istio ingress gateway between the load balancer and the registry. It renders it from plain
resources, so it needs no Gateway API CRDs and no separate Helm install: istiod turns a placeholder Deployment into the
gateway Envoy through its `gateway` injection template. Two switches let you check the gateway before traffic moves to
it:

| Value                                            | Effect                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|--------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `global.serviceMesh.ingressGateway.enabled`      | Renders the gateway: ServiceAccount, Deployment, ClusterIP Service, an HTTPS `Gateway`, a `VirtualService` to the registry, a TLS Secret, and an AuthorizationPolicy admitting the load balancer on 8443. Traffic still goes to the registry.                                                                                                                                                                                                                                                                                                                                                               |
| `global.serviceMesh.ingressGateway.routeIngress` | Points the ALB Ingress at the gateway over HTTPS (`backend-protocol` and `healthcheck-protocol: HTTPS`), admits only the load balancer's CIDRs to the gateway, admits only the gateway and mcpgw identities to the registry's HTTP port, closes the registry's TLS port 8443, and drops the registry STRICT exception. nginx does not trust the gateway's forwarded headers, so all clients share one rate-limit bucket (see Client IP in the audit log). Requires `enabled`, `authorizationPolicies.enabled: true`, `mtls.mode: STRICT` and `registry.app.trustedRealIpCidrs`; the render fails otherwise. |

With both on, every hop is encrypted: client to ALB (TLS, the ACM certificate), ALB to gateway (TLS, the gateway's
certificate), gateway to registry (mTLS).

The gateway's certificate is a Kubernetes TLS Secret. With `ingressGateway.tls.existingSecret` empty, the chart
generates a self-signed certificate for the public hostname on first install and reuses it on upgrade; with
`global.generateSecrets: false` (ArgoCD) the render fails until you name a Secret you manage. The ALB encrypts to the
target but does not validate its certificate, so a self-signed certificate protects the hop on the wire without
authenticating the gateway to the ALB.

istiod serves the gateway its certificate only if the gateway's ServiceAccount may `list` Secrets in the namespace, so
the chart creates that Role and RoleBinding. The gateway pod mounts no Kubernetes API token, so the grant is only used
by istiod's check.

The gateway trusts one upstream proxy, so only the load balancer should reach it. In routed mode its AuthorizationPolicy
admits only the load balancer's CIDRs, but a CIDR identifies a network range, not the load balancer. Set
`ingressGateway.securityGroupIds` to a security group that admits 8443 only from the load balancer's security group: the
chart then renders a `SecurityGroupPolicy` (EKS Security Groups for Pods) for the gateway pods, and the CIDR rule stays
as a second layer. This requires the VPC CNI with `ENABLE_POD_ENI=true`; without it the object has no effect.

The gateway parses HTTP, so Envoy's limits (60 KiB of request headers in total) apply before a request reaches nginx.

Roll out in order: enable the gateway and check it from inside the cluster (health, login, an MCP call), then set
`routeIngress` together with `registry.app.trustedRealIpCidrs`. Until `routeIngress` is on, the gateway admits any
source so it can be tested in-cluster. The load balancer re-registers its targets on the gateway pods. To roll back, set
`routeIngress: false`.

## Client IP in the audit log

The audit log's `client_ip` is the address nginx sees as its peer. Under the mesh that peer is the registry's own
sidecar, so what `client_ip` shows depends on `global.serviceMesh.istio.preserveClientIp`.

### Default: `preserveClientIp: false`

The sidecar runs in Istio's default REDIRECT mode with no extra privileges, and every audited request shows
`client_ip: 127.0.0.6`. The end user's address is still recorded in `request.forwarded_for`, the raw `X-Forwarded-For`
chain. For a request through the load balancer it looks like this:

```text
"forwarded_for": "75.80.225.106, 127.0.0.6"
```

The load balancer adds the address it received the connection from, and nginx then appends its peer (`127.0.0.6`). So
the real client is the entry immediately to the left of the final `127.0.0.6`. Read it from the right, not the left: the
load balancer appends to any `X-Forwarded-For` the client sent, so the leftmost entries can be anything a client chose.

This entry is trustworthy only for requests that came through the load balancer. Without an ingress gateway,
the registry's entry ports accept any source, so a pod in the cluster can send its own `X-Forwarded-For` directly to the
registry, and the entry before `127.0.0.6` is then whatever it wrote. Treat `forwarded_for` as an investigative aid, not
as attribution. `registry.app.trustedRealIpCidrs` has no effect in this mode: nginx never sees the load balancer's
address, and trusting `127.0.0.6` instead would trust every caller that reaches the sidecar.

### `preserveClientIp: true`

The registry pods' sidecar uses TPROXY interception, which keeps the caller's source address, so nginx sees the load
balancer, or a pod's real address for in-cluster callers. Set `registry.app.trustedRealIpCidrs` to the load balancer's
subnets, and nginx recovers the end user from the load balancer's `X-Forwarded-For` only on connections from those
subnets. `client_ip` is then the real client, and a forged header from any other caller is ignored.

The cost is pod privilege: TPROXY needs the `IP_TRANSPARENT` socket option, so the registry's long-running `istio-proxy`
container runs as root with the `NET_ADMIN` capability instead of as an unprivileged user. If Envoy in a registry pod
were compromised, it could change that pod's iptables rules and stop its traffic going through the sidecar. The
`baseline` and `restricted` Pod Security profiles reject this configuration.

### Behind the ingress gateway: `routeIngress: true`

nginx does not trust forwarded headers from the gateway. Behind the gateway, nginx sees every request arriving from its
own sidecar (`127.0.0.6`), the same address an in-mesh caller such as mcpgw arrives from, so it cannot tell the
gateway's `X-Forwarded-For` from one a caller made up. As a result:

- `client_ip` in the audit log is `127.0.0.6` for every request. The end user is the entry left of the load balancer's
  address in `request.forwarded_for`: the gateway keeps the load balancer's `X-Forwarded-For` (`numTrustedProxies: 1`)
  and appends the load balancer's address. Use it for investigation, not attribution.
- nginx's per-IP rate limits key on `127.0.0.6`, so all clients share one bucket, and one busy client can throttle the
  others.
- `preserveClientIp: true` makes nginx see the gateway pod's address instead, which still does not separate clients.

This mode trades per-client limits for an encrypted load balancer hop and a registry reachable only by the gateway and
mcpgw. Per-client attribution and limits behind the gateway need nginx to trust forwarded headers only on a listener the
gateway alone can reach, which requires an nginx change and is planned as a follow-up. If per-client limits matter more
than encrypting the hop, leave `routeIngress` off and use `preserveClientIp: true` with `trustedRealIpCidrs`.

## Keycloak chart compatibility

The policies work with both Keycloak charts: the Bitnami chart, and the official-image chart from #2070. The chart
detects which one is installed from its values (`keycloak.postgres` for the official chart, `keycloak.postgresql` for
Bitnami). Both keep the Service names `<release>-keycloak` and `<release>-keycloak-headless`, but they differ in pod
labels and identities:

|                        | Bitnami               | Official-image chart                              |
|------------------------|-----------------------|---------------------------------------------------|
| Keycloak pods          | `component: keycloak` | `component: server`                               |
| PostgreSQL pods        | `name: postgresql`    | `name: keycloak`, `component: postgres`           |
| Keycloak cluster ports | 7800                  | 7800, 57800                                       |
| Admin-init Job         | none                  | its own ServiceAccount, allowed to reach Keycloak |

Both charts run Keycloak as `keycloak.serviceAccount.name`, or `<release>-keycloak` when they create it, and the
policies name that identity for Keycloak's cluster ports and for PostgreSQL. The official chart's admin-init Job runs as
`keycloak.sslInit.serviceAccount.name`, or `<release>-keycloak-admin-init`. No policy admits the namespace `default`
ServiceAccount: setting `keycloak.serviceAccount.create: false` without a name fails the render.

## Keycloak's public hostname

The Bitnami Keycloak runs with `KC_PROXY_HEADERS=xforwarded` and no fixed hostname, so it builds its issuer from each
request. A sidecar adds `X-Forwarded-Proto` to every inbound request, and Keycloak then drops the port and can switch
the scheme: an internal discovery fetch returns
`https://<release>-keycloak-headless.<namespace>.svc.cluster.local/realms/...`. The auth-server only rewrites internal
URLs that start with its exact internal URL, so the internal name reached the OAuth protected-resource metadata, and MCP
clients such as Claude Code failed with "SDK auth failed".

With the mesh on, the chart pins the Bitnami Keycloak's hostname through a ConfigMap (
`<release>-keycloak-mesh-hostname`, wired with `keycloak.extraEnvVarsCM`): `KC_HOSTNAME` is the public URL (the same
value as `KEYCLOAK_EXTERNAL_URL`), `KC_HOSTNAME_STRICT=true`, and `KC_HOSTNAME_BACKCHANNEL_DYNAMIC=true` so token, JWKS
and userinfo calls from inside the cluster stay internal. The official-image Keycloak chart sets the same three values
itself; when moving to it, remove `keycloak.extraEnvVarsCM` and `keycloak.podAnnotations` together with the other
Bitnami-only
`keycloak.extraEnvVars`.

Keycloak reads the ConfigMap only at startup, so the chart also sets a `keycloak.podAnnotations` entry that carries the
pinned hostname (or `off`). Turning the mesh on or off changes the annotation, and the StatefulSet rolls Keycloak with
the new settings. Adding that annotation changes Keycloak's pod template, so the first upgrade to this chart version
restarts Keycloak once, with the mesh on or off; restart the registry afterwards so nginx picks up Keycloak's new
address.

## Restarting Keycloak

nginx in the registry resolves the Keycloak headless Service to a pod address once, when it starts. After Keycloak
restarts on a new address, the registry's Keycloak routes return 502 until the registry restarts: run
`kubectl rollout restart deployment/registry`. This happens with or without the mesh.

## Verifying it

Check that every pod has a sidecar:

```bash
kubectl get pods -n <namespace> -o custom-columns='NAME:.metadata.name,INIT:.spec.initContainers[*].name'
```

Count requests each sidecar denied. A count that keeps rising points at a missing rule:

```bash
kubectl exec -n <namespace> <pod> -c istio-proxy -- pilot-agent request GET stats | grep rbac.denied
```

A denied HTTP request returns `403 RBAC: access denied`. A denied TCP connection (MongoDB, PostgreSQL) is closed without
a reply.

## Limits of this version

**The mesh is not the external-traffic boundary.** Traffic to external identity providers, to SaaS MCP servers, and to
an external MongoDB or DocumentDB crosses the mesh edge. Without `ingressGateway.routeIngress`, the load balancer's hop
to the registry is plain HTTP inside the VPC. Under PERMISSIVE mTLS and Istio's
default outbound policy, these work with no extra configuration. Locking them down with `ServiceEntry` objects is a
later step.

**Injection outside the chart.** If you label the namespace for injection but leave `global.serviceMesh.enabled` false,
every pod is injected but the chart renders no policies, and the bootstrap Jobs depend on native sidecars to complete.

## Parameters

See the service mesh group in the [unified configuration parameter reference](unified-parameter-reference.md) for every
`global.serviceMesh.*` value.
