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
