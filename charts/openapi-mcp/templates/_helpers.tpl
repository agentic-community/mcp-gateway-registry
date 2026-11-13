{{/*
Reserved env var names for the openapi-mcp chart.

Users must not supply these via .Values.extraEnv. The list is the superset of
names the chart renders into `env:`, plus names the guard reads that belong to
the registry. The chart uses no envFrom sources of its own.

To update: edit charts/openapi-mcp/reserved-env-names.txt and run
`helm dep update` on any parent chart that depends on this subchart.
Lines starting with '#' are comments.
*/}}
{{- define "openapi-mcp.reservedEnvNames" -}}
{{- $names := list -}}
{{- range (splitList "\n" (.Files.Get "reserved-env-names.txt")) -}}
  {{- $line := trim . -}}
  {{- if and $line (not (hasPrefix "#" $line)) -}}
    {{- $names = append $names $line -}}
  {{- end -}}
{{- end -}}
{{- $names | toYaml -}}
{{- end -}}

{{/*
Validate .Values.extraEnv. Fails the render if an entry has no name, repeats
another entry's name, or uses a reserved name. Reserved names compare trimmed
and upper-cased, because pydantic-settings reads them case-insensitively.

Call as: {{- include "openapi-mcp.validateExtraEnv" . -}}
*/}}
{{- define "openapi-mcp.validateExtraEnv" -}}
{{- $reserved := list -}}
{{- range (fromYamlArray (include "openapi-mcp.reservedEnvNames" .)) -}}
  {{- $reserved = append $reserved (. | trim | upper) -}}
{{- end -}}
{{- $seen := dict -}}
{{- range $i, $e := .Values.extraEnv -}}
  {{- if not $e.name -}}
    {{- fail (printf "openapi-mcp.extraEnv[%d]: missing required 'name' field" $i) -}}
  {{- end -}}
  {{- if has ($e.name | toString | trim | upper) $reserved -}}
    {{- fail (printf "openapi-mcp.extraEnv[%d]: %q is a reserved variable managed by the chart. Remove it from extraEnv and set the matching values.yaml field instead." $i $e.name) -}}
  {{- end -}}
  {{- if hasKey $seen $e.name -}}
    {{- fail (printf "openapi-mcp.extraEnv[%d]: duplicate name %q (first seen at index %v)" $i $e.name (index $seen $e.name)) -}}
  {{- end -}}
  {{- $_ := set $seen $e.name $i -}}
{{- end -}}
{{- end -}}

{{/*
Name of the Secret that holds OPENAPI_MCP_SYNC_SECRET. The stack-level
global.openapiMcp.existingSyncSecret wins, then syncSecret.existingSecret, then
the chart-created Secret. The registry chart resolves the same name.
*/}}
{{- define "openapi-mcp.syncSecretName" -}}
{{- $existing := dig "openapiMcp" "existingSyncSecret" "" .Values.global -}}
{{- $existing | default .Values.syncSecret.existingSecret | default .Values.syncSecret.name -}}
{{- end -}}

{{/*
True (non-empty) when the operator supplies the sync Secret, so the chart must
not create one.
*/}}
{{- define "openapi-mcp.usesExistingSyncSecret" -}}
{{- $existing := dig "openapiMcp" "existingSyncSecret" "" .Values.global -}}
{{- if or $existing .Values.syncSecret.existingSecret -}}true{{- end -}}
{{- end -}}

{{/*
Fail the render unless NO_PROXY in extraEnv excludes the cloud credential
endpoints while the forward proxy is on. Mirrors the startup check in
registry/utils/url_guard.py and the registry chart. Exact per-entry comparison:
"169.254.170.2" is a substring of "169.254.170.23".
*/}}
{{- define "openapi-mcp.validateForwardProxy" -}}
{{- if .Values.app.egressForwardProxyEnabled -}}
  {{- $noProxy := "" -}}
  {{- range .Values.extraEnv -}}
    {{- if eq .name "NO_PROXY" }}{{- $noProxy = .value | toString }}{{- end -}}
  {{- end -}}
  {{- $patterns := list -}}
  {{- range (splitList "," $noProxy) }}{{- $patterns = append $patterns (trim .) }}{{- end -}}
  {{- if not (has "*" $patterns) -}}
    {{- $missing := list -}}
    {{- range $ip := list "169.254.169.254" "169.254.170.2" "169.254.170.23" -}}
      {{- if not (has $ip $patterns) }}{{- $missing = append $missing $ip }}{{- end -}}
    {{- end -}}
    {{- if $missing -}}
      {{- fail (printf "openapi-mcp.app.egressForwardProxyEnabled is true but extraEnv NO_PROXY does not exclude the cloud credential endpoint(s) %s. Add '169.254.169.254,169.254.170.2,169.254.170.23' to the NO_PROXY entry in openapiMcp.extraEnv. See docs/forward-proxy-egress.md." (join ", " $missing)) -}}
    {{- end -}}
  {{- end -}}
{{- end -}}
{{- end -}}

{{/*
caBundle needs exactly one source, a key and a mount path.
*/}}
{{- define "openapi-mcp.validateCaBundle" -}}
{{- if .Values.caBundle.enabled -}}
  {{- if and .Values.caBundle.existingConfigMap .Values.caBundle.existingSecret -}}
    {{- fail "openapi-mcp.caBundle: set exactly ONE of existingConfigMap or existingSecret, not both" -}}
  {{- end -}}
  {{- if not (or .Values.caBundle.existingConfigMap .Values.caBundle.existingSecret) -}}
    {{- fail "openapi-mcp.caBundle.enabled is true but neither existingConfigMap nor existingSecret is set" -}}
  {{- end -}}
  {{- if not .Values.caBundle.key -}}
    {{- fail "openapi-mcp.caBundle.key must not be empty" -}}
  {{- end -}}
  {{- if not .Values.caBundle.mountPath -}}
    {{- fail "openapi-mcp.caBundle.mountPath must not be empty" -}}
  {{- end -}}
{{- end -}}
{{- end -}}
