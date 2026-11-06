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
instead. An empty override is a subchart default (app.sessionCookieDomain
ships empty) and is ignored. The registry has no app key for the Secure flag.
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
{{- $override := dig "app" $key nil (index $.Values $chart | default dict) -}}
{{- if not (kindIs "invalid" $override) -}}
{{- $overrideText := include "mcp-gateway-registry-stack.valueString" $override -}}
{{- if and (ne $overrideText "") (ne $overrideText $globalText) -}}
{{- fail (printf "%s.app.%s is %s but global.%s is %s. The stack sets %s for the registry and the auth-server from global.%s only, so the two cannot disagree. Remove %s.app.%s and set global.%s." $chart $key $overrideText $key ($globalText | default "empty") $envName $key $chart $key $key) -}}
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
