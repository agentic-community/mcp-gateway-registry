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
