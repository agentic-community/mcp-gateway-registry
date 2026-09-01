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
Name of the stack-owned internal Keycloak Service
(templates/keycloak-internal-service.yaml). A release-scoped constant:
deliberately NOT derived from bitnami common.names.fullname, so
keycloak.nameOverride / keycloak.fullnameOverride cannot move the
endpoint out from under the URL. Shared by the Service's metadata.name
and keycloakInternalUrl below so the two cannot drift.

No truncation: Helm caps release names at 53 chars, so with the 7-char
"-kc-int" suffix the name never exceeds the 63-char DNS label limit.
Keeping the full release name keeps the Service name unique per
release — any truncation would map two long releases sharing a prefix
onto the same name. The subchart helpers replicate this printf; the
long-release-name tests in each chart's keycloak_service_test.yaml pin
both properties.
*/}}
{{- define "mcp-gateway-registry-stack.keycloakInternalHost" -}}
{{- printf "%s-kc-int" .Release.Name -}}
{{- end -}}

{{/*
Fail-closed guards for the stack-owned keycloak-internal Service. The
Service finds Keycloak pods with a fixed label selector in the release
namespace; two supported Bitnami overrides would silently leave it with
zero endpoints behind healthy-looking manifests, so both are rejected
at render time: overriding a selector label via keycloak.podLabels /
keycloak.commonLabels, and moving the pods with keycloak.namespaceOverride.
*/}}
{{- define "mcp-gateway-registry-stack.validateKeycloakInternal" -}}
{{- $kc := index .Values "keycloak" | default dict -}}
{{- if and $kc.namespaceOverride (ne $kc.namespaceOverride .Release.Namespace) -}}
{{- fail (printf "keycloak.namespaceOverride=%q: the keycloak-internal Service lives in the release namespace %q and cannot select pods in another namespace" $kc.namespaceOverride .Release.Namespace) -}}
{{- end -}}
{{- range $key := list "app.kubernetes.io/instance" "app.kubernetes.io/component" "app.kubernetes.io/part-of" -}}
{{- range $src := list "podLabels" "commonLabels" -}}
{{- if hasKey (index $kc $src | default dict) $key -}}
{{- fail (printf "keycloak.%s[%q]: reserved label — the keycloak-internal Service selects Keycloak pods by it, and overriding it would leave the Service with no endpoints" $src $key) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
In-cluster Keycloak HTTP URL for the shared oauth-provider Secret.

See registry.keycloakInternalUrl for the full rationale (nginx pins
proxy_pass IPs at start, so the URL must be a stable ClusterIP name;
port 8080 must stay explicit so Keycloak keeps deriving issuers with a
port). The hostname is short (no namespace / cluster-domain suffix):
every consumer runs in the release namespace and the resolver search
path expands it, which keeps the URL independent of namespace and
cluster-domain settings.

Resolution order:
  1. global.authProvider.keycloak.internalUrl — the one canonical
     override, seen identically by this chart and by the subcharts'
     copies of this helper. REQUIRED when keycloak.create=false: with
     an external Keycloak the keycloak-internal Service is not
     rendered, so the default URL would point at nothing; fail closed
     rather than write a dead URL into the shared Secret. (The Secret
     template only resolves this helper when the auth provider is
     keycloak, so Entra/Okta/... installs with keycloak.create=false
     are unaffected.)
  2. The stack-owned Service above (keycloak.create=true).
keycloak_service_test.yaml in each chart pins the four helper copies
to the same output.
*/}}
{{- define "mcp-gateway-registry-stack.keycloakInternalUrl" -}}
{{- $override := dig "authProvider" "keycloak" "internalUrl" "" (.Values.global | default dict) -}}
{{- if $override -}}
{{- $override -}}
{{- else if .Values.keycloak.create -}}
{{- printf "http://%s:8080" (include "mcp-gateway-registry-stack.keycloakInternalHost" .) -}}
{{- else -}}
{{- fail "keycloak.create=false (external Keycloak) requires global.authProvider.keycloak.internalUrl: the stack-owned keycloak-internal Service only exists for the bundled Keycloak, so the default in-cluster URL would point at nothing" -}}
{{- end -}}
{{- end -}}
