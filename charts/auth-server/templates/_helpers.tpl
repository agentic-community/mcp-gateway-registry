{{/*
Reserved env var names for the auth-server chart.

Users must not supply these via .Values.extraEnv. The list is the union of:
  - the superset of names the chart may render into `env:` (including
    every conditional branch), and
  - every key the chart sources via `envFrom` from stack-level or
    per-chart secrets/configmaps.

To update: edit charts/auth-server/reserved-env-names.txt and run
`helm dep update` on any parent chart that depends on this subchart.

Sections (in order below):
  1. env: block — IdP secrets via valueFrom (conditional)
  2. auth-server-app-log-config configmap
  3. auth-server per-chart secret
  4. keycloak-client-secret (runtime-created by keycloak-configure Job)
  5. mongo-credentials secret
  6. shared-secret (stack-level)

Over-rejection is preferred to under-rejection: a user attempting to
inject one of these via extraEnv gets a clear template-render error.
*/}}
{{- define "auth-server.reservedEnvNames" -}}
{{- $content := .Files.Get "reserved-env-names.txt" -}}
{{- compact (splitList "\n" $content) | toYaml -}}
{{- end -}}

{{/*
Validate .Values.extraEnv for the auth-server chart.

Fails helm template render if any entry:
  - is missing the required `name` field,
  - shares a name with another entry in extraEnv (would silently shadow
    under Kubernetes merge rules), or
  - collides with a chart-reserved name.

Call as: {{- include "auth-server.validateExtraEnv" . -}}
*/}}
{{- define "auth-server.validateExtraEnv" -}}
{{- $reserved := fromYamlArray (include "auth-server.reservedEnvNames" .) -}}
{{- $seen := dict -}}
{{- range $i, $e := .Values.extraEnv -}}
  {{- if not $e.name -}}
    {{- fail (printf "auth-server.extraEnv[%d]: missing required 'name' field" $i) -}}
  {{- end -}}
  {{- if has $e.name $reserved -}}
    {{- fail (printf "auth-server.extraEnv[%d]: %q is a reserved variable managed by the chart (via env: or envFrom from the chart's secrets/configmaps). Remove it from extraEnv. If a values.yaml field controls it (e.g. app.jwtIssuer for JWT_ISSUER), set that instead; otherwise the value is managed by the chart's internal secrets and must not be overridden via extraEnv." $i $e.name) -}}
  {{- end -}}
  {{- if hasKey $seen $e.name -}}
    {{- fail (printf "auth-server.extraEnv[%d]: duplicate name %q (first seen at index %v)" $i $e.name (index $seen $e.name)) -}}
  {{- end -}}
  {{- $_ := set $seen $e.name $i -}}
{{- end -}}
{{- end -}}
