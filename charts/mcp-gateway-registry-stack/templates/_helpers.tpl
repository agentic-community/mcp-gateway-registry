{{/*
Expand the name of the chart.
*/}}
{{- define "mcp-gateway-registry-stack.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "mcp-gateway-registry-stack.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
OpenBao resource name (matches the openbao subchart's fullname logic).

Honors openbao.fullnameOverride if set; otherwise derives "<release>-openbao"
(or just the release name if it already contains "openbao"), exactly as the
subchart's "openbao.fullname" does. Used to build the in-cluster service DNS so
the egress OPENBAO_ADDR + init/unseal Job always target the right Service even
though the name is release-scoped (which keeps the cluster-scoped
"<name>-server-binding" ClusterRoleBinding unique per release).
*/}}
{{- define "mcp-gateway-registry-stack.openbaoName" -}}
{{- $ob := .Values.openbao | default dict -}}
{{- if $ob.fullnameOverride }}
{{- $ob.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else if contains "openbao" .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-openbao" .Release.Name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{/*
OpenBao in-cluster service host: "<openbaoName>.<namespace>.svc".
*/}}
{{- define "mcp-gateway-registry-stack.openbaoServiceHost" -}}
{{- printf "%s.%s.svc" (include "mcp-gateway-registry-stack.openbaoName" .) .Release.Namespace }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "mcp-gateway-registry-stack.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "mcp-gateway-registry-stack.labels" -}}
helm.sh/chart: {{ include "mcp-gateway-registry-stack.chart" . }}
{{ include "mcp-gateway-registry-stack.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "mcp-gateway-registry-stack.selectorLabels" -}}
app.kubernetes.io/name: {{ include "mcp-gateway-registry-stack.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Validate generic-proxy cross-chart parity and egress prerequisites. The stack
owns both consumers, so it can reject a split-brain switch configuration before
subcharts render.
*/}}
{{- define "mcp-gateway-registry-stack.validateGenericProxy" -}}
{{- $registryEnabled := dig "app" "gatewayGenericProxyEnabled" false .Values.registry -}}
{{- $authValues := index .Values "auth-server" -}}
{{- $authEnabled := dig "app" "gatewayGenericProxyEnabled" false $authValues -}}
{{- if ne $registryEnabled $authEnabled -}}
{{- fail "generic proxy switch mismatch: registry.app.gatewayGenericProxyEnabled and auth-server.app.gatewayGenericProxyEnabled must be identical" -}}
{{- end -}}
{{- $managedPolicyEnabled := dig "egress" "networkPolicy" "enabled" false $authValues -}}
{{- $externalPolicyAcknowledged := dig "egress" "externalEgressPolicyAcknowledged" false $authValues -}}
{{- if and $authEnabled (not $managedPolicyEnabled) (not $externalPolicyAcknowledged) -}}
{{- fail "generic proxy is enabled but no egress policy is declared: enable auth-server.egress.networkPolicy or explicitly acknowledge an equivalent external policy with auth-server.egress.externalEgressPolicyAcknowledged=true" -}}
{{- end -}}
{{- end -}}

{{/*
A value as a string for comparison. A whole number read from a values file
arrives as a float, so 28800 from a file, 28800 from --set and "28800" all
compare equal, and a large one is not written with an exponent.
*/}}
{{- define "mcp-gateway-registry-stack.valueString" -}}
{{- if and (kindIs "float64" .) (eq . (float64 (int64 .))) -}}
{{- int64 . -}}
{{- else -}}
{{- toString . -}}
{{- end -}}
{{- end -}}

{{/*
Refuse a subchart-level session setting that differs from its global. The
registry and the auth-server must agree on every one: the auth-server stores
the session and sets the cookie's name, Domain, Max-Age and Secure attribute,
and the registry reads, checks and expires the same cookie. A split prefix lets
/validate find the session while CSRF, consent and logout do not; a split
domain leaves the browser two cookies, which both services refuse, so the user
is signed out on every request. Both subcharts render each setting from
global.<setting>, so the stack requires the globals, and a differing override
is ignored by the subcharts and would read as if it took effect, so name it
instead. An override is compared with the value the subcharts render: for the
cookie domain that is global.sessionCookieDomain, ".<global.domain>" when that
global is empty, and no Domain at all with global.sessionCookieHostPrefix, so
an override equal to the rendered domain is accepted. An empty override is a
subchart default (app.sessionCookieDomain ships empty) and is ignored. The
registry has no app key for the Secure flag.
*/}}
{{- define "mcp-gateway-registry-stack.validateSessionSettings" -}}
{{- $checks := list (list "registry" "sessionMaxAgeSeconds" "SESSION_MAX_AGE_SECONDS") (list "auth-server" "sessionMaxAgeSeconds" "SESSION_MAX_AGE_SECONDS") (list "auth-server" "sessionCookieSecure" "SESSION_COOKIE_SECURE") (list "registry" "sessionCookieHostPrefix" "SESSION_COOKIE_HOST_PREFIX") (list "auth-server" "sessionCookieHostPrefix" "SESSION_COOKIE_HOST_PREFIX") (list "registry" "sessionCookieDomain" "SESSION_COOKIE_DOMAIN") (list "auth-server" "sessionCookieDomain" "SESSION_COOKIE_DOMAIN") -}}
{{- range $check := $checks -}}
{{- $chart := index $check 0 -}}
{{- $key := index $check 1 -}}
{{- $envName := index $check 2 -}}
{{- $global := index $.Values.global $key -}}
{{- if kindIs "invalid" $global -}}
{{- fail (printf "global.%s is not set. The stack sets %s for the registry and the auth-server from global.%s, so the two cannot disagree. Set global.%s." $key $envName $key $key) -}}
{{- end -}}
{{- $globalText := include "mcp-gateway-registry-stack.valueString" $global -}}
{{- $renderedText := $globalText -}}
{{- $rendered := printf "global.%s is %s" $key ($globalText | default "empty") -}}
{{- if eq $key "sessionCookieDomain" -}}
{{- if eq (toString $.Values.global.sessionCookieHostPrefix) "true" -}}
{{- $renderedText = "" -}}
{{- $rendered = "global.sessionCookieHostPrefix is true, so the stack renders no Domain" -}}
{{- else if eq $globalText "" -}}
{{- $renderedText = printf ".%s" ($.Values.global.domain | default "localhost") -}}
{{- $rendered = printf "global.sessionCookieDomain is empty, so the stack renders %s from global.domain" $renderedText -}}
{{- end -}}
{{- end -}}
{{- $override := dig "app" $key nil (index $.Values $chart | default dict) -}}
{{- if not (kindIs "invalid" $override) -}}
{{- $overrideText := include "mcp-gateway-registry-stack.valueString" $override -}}
{{- if and (ne $overrideText "") (ne $overrideText $renderedText) -}}
{{- fail (printf "%s.app.%s is %s but %s. The stack sets %s for the registry and the auth-server from the globals only, so the two cannot disagree. Remove %s.app.%s and set global.%s." $chart $key $overrideText $rendered $envName $chart $key $key) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Guard chart-generated random secret value.

`global.generateSecrets: true` (the default) generates a random value,
`lookup` preserves it across `helm upgrade`.

`global.generateSecrets: false` is the GitOps mode. ArgoCD and every other
`helm template`-based tool renders without cluster access, so `lookup` always
returns empty and a generated value is a NEW value on every sync — which then
silently disagrees with the value the running pods already hold (a changed
`envFrom` Secret does not restart pods), rotates the MongoDB user password out
from under the operator-provisioned SCRAM credential, and re-randomizes the
Keycloak PostgreSQL password against a retained PVC. With generation off, a
value the chart cannot resolve is a render-time error naming the value to set
instead of a silent rotation.

Usage:
  {{- include "mcp-gateway-registry-stack.requireGeneratedSecret" (dict "ctx" . "key" "SECRET_KEY" "value" "global.secretKey" "byo" "global.existingSharedSecret") }}
*/}}
{{- define "mcp-gateway-registry-stack.requireGeneratedSecret" -}}
{{- if not (dig "generateSecrets" true .ctx.Values.global) -}}
{{- fail (printf "global.generateSecrets is false and %s has no value, so the chart will not generate one. Set %s explicitly, or set %s to a Secret that already holds it. Generation is unsafe under ArgoCD and other `helm template` based tools: they render without cluster access, so the chart cannot read the previous value back and every sync would produce a different secret." .key .value .byo) -}}
{{- end -}}
{{- end -}}

{{/*
True when an Istio service mesh is enabled for this release, i.e.
serviceMesh.enabled AND serviceMesh.type == "istio". Returns a non-empty string
(truthy) or empty. Use as the single gate for every Istio object and injection
annotation so the condition cannot drift across templates:

  {{- if include "mcp-gateway-registry-stack.istioEnabled" . }}
*/}}
{{- define "mcp-gateway-registry-stack.istioEnabled" -}}
{{- $sm := .Values.global.serviceMesh | default dict -}}
{{- if and $sm.enabled (eq ($sm.type | default "istio") "istio") -}}
true
{{- end -}}
{{- end -}}

{{/*
Which Keycloak chart backs this release, for service-mesh policy selectors.
"official" when the values use the local official-image chart (keycloak.postgres,
PR #2070); "bitnami" when they use the Bitnami chart (keycloak.postgresql). Both
charts keep the Service names <release>-keycloak and <release>-keycloak-headless,
but label their pods and run them as ServiceAccounts differently.
*/}}
{{- define "mcp-gateway-registry-stack.keycloakFlavor" -}}
{{- if hasKey (.Values.keycloak | default dict) "postgres" -}}official{{- else -}}bitnami{{- end -}}
{{- end -}}

{{/*
Pod selector labels (YAML map) for the Keycloak server pods.
*/}}
{{- define "mcp-gateway-registry-stack.keycloakServerSelector" -}}
app.kubernetes.io/name: keycloak
app.kubernetes.io/instance: {{ .Release.Name | quote }}
app.kubernetes.io/component: {{ ternary "server" "keycloak" (eq (include "mcp-gateway-registry-stack.keycloakFlavor" .) "official") }}
{{- end -}}

{{/*
Pod selector labels (YAML map) for Keycloak's PostgreSQL pods.
*/}}
{{- define "mcp-gateway-registry-stack.keycloakPostgresSelector" -}}
{{- if eq (include "mcp-gateway-registry-stack.keycloakFlavor" .) "official" -}}
app.kubernetes.io/name: keycloak
app.kubernetes.io/instance: {{ .Release.Name | quote }}
app.kubernetes.io/component: postgres
{{- else -}}
app.kubernetes.io/name: postgresql
app.kubernetes.io/instance: {{ .Release.Name | quote }}
{{- end -}}
{{- end -}}

{{/*
ServiceAccount the Keycloak pods run as, used as their identity in the Keycloak
and PostgreSQL AuthorizationPolicies. Read from keycloak.serviceAccount, which
both the Bitnami chart and the official-image chart (#2070) use: an explicit
name wins, otherwise "<release>-keycloak", the name both charts create. A
configuration that leaves Keycloak on the namespace "default" ServiceAccount
(create: false with no name) fails the render instead of admitting "default",
which every pod without its own ServiceAccount shares.
*/}}
{{- define "mcp-gateway-registry-stack.keycloakServiceAccount" -}}
{{- $sa := (.Values.keycloak | default dict).serviceAccount | default dict -}}
{{- if $sa.name -}}{{ $sa.name }}
{{- else if eq (toString (dig "create" true $sa)) "false" -}}
{{- fail "global.serviceMesh.authorizationPolicies: keycloak.serviceAccount.create is false and keycloak.serviceAccount.name is empty, so Keycloak would run as the namespace default ServiceAccount, which the mesh policies do not admit. Set keycloak.serviceAccount.name to the ServiceAccount Keycloak uses." -}}
{{- else -}}{{ printf "%s-keycloak" .Release.Name }}
{{- end -}}
{{- end -}}

{{/*
Public Keycloak URL: the host Keycloak must put in `iss` and in discovery
documents. Same derivation as KEYCLOAK_EXTERNAL_URL in oauth-provider-secret.yaml;
tests/service_mesh_keycloak_hostname_test.yaml asserts the two agree.
*/}}
{{- define "mcp-gateway-registry-stack.keycloakExternalUrl" -}}
{{- $domain := .Values.global.domain | default "localhost" -}}
{{- $protocol := ternary "https" "http" .Values.global.ingress.tls -}}
{{- if eq (.Values.global.ingress.routingMode | default "subdomain") "path" -}}
{{- printf "%s://%s" $protocol $domain -}}
{{- else -}}
{{- printf "%s://%s.%s" $protocol (.Values.global.ingress.registrySubdomain | default "mcpregistry") $domain -}}
{{- end -}}
{{- end -}}

{{/*
ServiceAccount of the official-image Keycloak chart's admin-init Job (#2070):
keycloak.sslInit.serviceAccount.name, otherwise "<release>-keycloak-admin-init",
the name that chart creates. The chart never runs the Job as "default".
*/}}
{{- define "mcp-gateway-registry-stack.keycloakAdminInitServiceAccount" -}}
{{- dig "sslInit" "serviceAccount" "name" "" (.Values.keycloak | default dict) | default (printf "%s-keycloak-admin-init" .Release.Name) -}}
{{- end -}}

{{/*
app.name of an application subchart (registry, auth-server, mcpgw). The subchart
labels its pods app.kubernetes.io/name=<app.name> and names its Service after
it, so mesh selectors and hosts must follow this value, not a literal.
Usage: include "mcp-gateway-registry-stack.appName" (dict "ctx" . "chart" "registry" "default" "registry")
*/}}
{{- define "mcp-gateway-registry-stack.appName" -}}
{{- dig "app" "name" .default (index .ctx.Values .chart | default dict) -}}
{{- end -}}

{{/*
ServiceAccount an application subchart's pods run as: serviceAccount.name, else
app.name (the subcharts' own serviceAccountName rule).
*/}}
{{- define "mcp-gateway-registry-stack.appServiceAccount" -}}
{{- $sub := index .ctx.Values .chart | default dict -}}
{{- dig "serviceAccount" "name" "" $sub | default (dig "app" "name" .default $sub) -}}
{{- end -}}

{{/*
Trusted load balancer CIDRs (registry.app.trustedRealIpCidrs) as a list.
*/}}
{{- define "mcp-gateway-registry-stack.trustedProxyCidrs" -}}
{{- $out := list -}}
{{- range (splitList "," (dig "app" "trustedRealIpCidrs" "" (.Values.registry | default dict))) -}}
{{- if trim . }}{{- $out = append $out (trim .) }}{{- end -}}
{{- end -}}
{{- toYaml $out -}}
{{- end -}}

{{/*
True when the ALB is routed through the Istio ingress gateway.
*/}}
{{- define "mcp-gateway-registry-stack.ingressRouted" -}}
{{- $gw := .Values.global.serviceMesh.ingressGateway | default dict -}}
{{- if and (include "mcp-gateway-registry-stack.istioEnabled" .) $gw.enabled $gw.routeIngress -}}true{{- end -}}
{{- end -}}

{{/*
Fail closed unless the conditions routed mode depends on hold. In routed mode
the registry's entry port admits only the gateway, auth-server and mcpgw, and
the gateway admits only the load balancer's CIDRs; both are AuthorizationPolicy
rules, enforced only with the policies on and with STRICT mTLS (a plaintext
caller has no identity to check). The gateway trusts one upstream proxy, so its
allow list (registry.app.trustedRealIpCidrs) must be set.
*/}}
{{- define "mcp-gateway-registry-stack.validateIngressRoute" -}}
{{- if include "mcp-gateway-registry-stack.ingressRouted" . -}}
{{- if not .Values.global.serviceMesh.authorizationPolicies.enabled -}}
{{- fail "global.serviceMesh.ingressGateway.routeIngress requires global.serviceMesh.authorizationPolicies.enabled: true. Routed mode relies on them to restrict who reaches the registry and the gateway." -}}
{{- end -}}
{{- if ne (.Values.global.serviceMesh.mtls.mode | default "STRICT") "STRICT" -}}
{{- fail "global.serviceMesh.ingressGateway.routeIngress requires global.serviceMesh.mtls.mode: STRICT. Under PERMISSIVE a caller without a mesh identity can reach the registry's sidecar." -}}
{{- end -}}
{{- if not (fromYamlArray (include "mcp-gateway-registry-stack.trustedProxyCidrs" .)) -}}
{{- fail "global.serviceMesh.ingressGateway.routeIngress requires registry.app.trustedRealIpCidrs (the load balancer's subnets). The gateway trusts one upstream proxy (numTrustedProxies: 1) and admits only those addresses." -}}
{{- end -}}
{{- end -}}
{{- end -}}
