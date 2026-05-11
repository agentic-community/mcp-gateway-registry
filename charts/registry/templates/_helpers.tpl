{{/*
Reserved env var names for the registry chart.

Users must not supply these via .Values.extraEnv. The list is the union of:
  - the superset of names the chart may render into `env:` (including
    every conditional branch), and
  - every key the chart sources via `envFrom` from stack-level or
    per-chart secrets/configmaps.

To update: edit charts/registry/reserved-env-names.txt and run
`helm dep update` on any parent chart that depends on this subchart.

Sections (in order below):
  1. env: block — feature flags and IdP secrets via valueFrom
  2. registry-app-log-config configmap
  3. registry-otel-config configmap
  4. registry per-chart secret
  5. keycloak-client-secret (runtime-created by keycloak-configure Job)
  6. mongo-credentials secret
  7. shared-secret (stack-level)

Over-rejection is preferred to under-rejection: a user attempting to
inject one of these via extraEnv gets a clear template-render error.
*/}}
{{- define "registry.reservedEnvNames" -}}
{{- $content := .Files.Get "reserved-env-names.txt" -}}
{{- compact (splitList "\n" $content) | toYaml -}}
{{- end -}}

{{/*
Validate .Values.extraEnv for the registry chart.

Fails helm template render if any entry:
  - is missing the required `name` field,
  - shares a name with another entry in extraEnv (would silently shadow
    under Kubernetes merge rules), or
  - collides with a chart-reserved name.

Call as: {{- include "registry.validateExtraEnv" . -}}
*/}}
{{- define "registry.validateExtraEnv" -}}
{{- $reserved := fromYamlArray (include "registry.reservedEnvNames" .) -}}
{{- $seen := dict -}}
{{- range $i, $e := .Values.extraEnv -}}
  {{- if not $e.name -}}
    {{- fail (printf "registry.extraEnv[%d]: missing required 'name' field" $i) -}}
  {{- end -}}
  {{- if has $e.name $reserved -}}
    {{- fail (printf "registry.extraEnv[%d]: %q is a reserved variable managed by the chart (via env: or envFrom from the chart's secrets/configmaps). Remove it from extraEnv. If a values.yaml field controls it (e.g. app.showSkillsTab for SHOW_SKILLS_TAB), set that instead; otherwise the value is managed by the chart's internal secrets and must not be overridden via extraEnv." $i $e.name) -}}
  {{- end -}}
  {{- if hasKey $seen $e.name -}}
    {{- fail (printf "registry.extraEnv[%d]: duplicate name %q (first seen at index %v)" $i $e.name (index $seen $e.name)) -}}
  {{- end -}}
  {{- $_ := set $seen $e.name $i -}}
{{- end -}}
{{- end -}}
