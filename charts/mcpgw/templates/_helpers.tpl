{{/*
Name of the ServiceAccount the mcpgw pod runs as. Explicit serviceAccount.name
wins; otherwise defaults to the app name ("mcpgw-server"). A stable first-class
identity so mesh AuthorizationPolicy and future RBAC/IRSA can key on it; without
it the pod would run as the namespace "default" SA, which cannot be used as an
authz principal.
*/}}
{{- define "mcpgw.serviceAccountName" -}}
{{- if .Values.serviceAccount.name -}}
{{- .Values.serviceAccount.name -}}
{{- else -}}
{{- .Values.app.name -}}
{{- end -}}
{{- end -}}

{{/*
Reserved env var names for the mcpgw chart.

Users must not supply these via .Values.extraEnv. The list is the union of:
  - the superset of names the chart may render into `env:` (including
    every conditional branch), and
  - every key the chart sources via `envFrom` from stack-level or
    per-chart secrets/configmaps.

To update: edit charts/mcpgw/reserved-env-names.txt and run
`helm dep update` on any parent chart that depends on this subchart.

Sections (in order below):
  1. env: block (HOST, GITHUB_*)
  2. mcpgw per-chart secret
  3. shared-secret (stack-level)
  4. OIDC / OAuth proxy vars read by server.py — chart does not wire
     OIDC today (see issue #895). OIDC_ENABLED is reserved so an
     operator cannot half-enable OAuthProxy by setting the flag
     in via extraEnv without the rest of the required config.

Over-rejection is preferred to under-rejection: a user attempting to
inject one of these via extraEnv gets a clear template-render error.
*/}}
{{- define "mcpgw.reservedEnvNames" -}}
{{- $content := .Files.Get "reserved-env-names.txt" -}}
{{- compact (splitList "\n" $content) | toYaml -}}
{{- end -}}

{{/*
Normalize an environment variable name for comparison: trimmed and upper-cased.

The reserved check compares every extraEnv name through this helper, against
a reserved list normalized the same way. Kubernetes env names are
case-sensitive, but the services read some of them case-insensitively
(pydantic-settings Settings uses case_sensitive=False), so
`ssrf_allowed_cidrs` would otherwise slip past the reserved check for
SSRF_ALLOWED_CIDRS and override the chart-managed value. The duplicate check
does NOT use it. scripts/validate-extra-env.sh is stricter: it also removes
whitespace inside a name.

Call as: {{ include "mcpgw.envNameKey" $name }}
*/}}
{{- define "mcpgw.envNameKey" -}}
{{- . | toString | trim | upper -}}
{{- end -}}

{{/*
Validate .Values.extraEnv for the mcpgw chart.

Fails helm template render if any entry:
  - is missing the required `name` field,
  - shares a name with another entry in extraEnv (would silently shadow
    under Kubernetes merge rules), or
  - collides with a chart-reserved name.

Call as: {{- include "mcpgw.validateExtraEnv" . -}}
*/}}
{{- define "mcpgw.validateExtraEnv" -}}
{{- $reserved := list -}}
{{- range (fromYamlArray (include "mcpgw.reservedEnvNames" .)) -}}
  {{- $reserved = append $reserved (include "mcpgw.envNameKey" .) -}}
{{- end -}}
{{- $seen := dict -}}
{{- range $i, $e := .Values.extraEnv -}}
  {{- if not $e.name -}}
    {{- fail (printf "mcpgw.extraEnv[%d]: missing required 'name' field" $i) -}}
  {{- end -}}
  {{- $key := include "mcpgw.envNameKey" $e.name -}}
  {{- if has $key $reserved -}}
    {{- if eq $key "OIDC_ENABLED" -}}
      {{- fail (printf "mcpgw.extraEnv[%d]: %q is reserved by the chart. OIDC is not currently configurable through chart values (see issue #895); setting just this flag starts a misconfigured OAuthProxy. To enable OIDC, supply the full OIDC env set (OIDC_ENABLED, OIDC_CLIENT_ID/SECRET, Keycloak URLs, M2M creds, MCPGW_BASE_URL) via extraEnvFrom from a secret you manage." $i $e.name) -}}
    {{- else -}}
      {{- fail (printf "mcpgw.extraEnv[%d]: %q is a reserved variable managed by the chart (via env: or envFrom from the chart's secrets/configmaps). Remove it from extraEnv. If a values.yaml field controls it (e.g. app.githubAppId for GITHUB_APP_ID), set that instead; otherwise the value is managed by the chart's internal secrets and must not be overridden via extraEnv." $i $e.name) -}}
    {{- end -}}
  {{- end -}}
  {{- /* Duplicates compare the exact name: Kubernetes treats HTTPS_PROXY and
         https_proxy as two variables, and proxy setups legitimately set both. */ -}}
  {{- if hasKey $seen $e.name -}}
    {{- fail (printf "mcpgw.extraEnv[%d]: duplicate name %q (first seen at index %v)" $i $e.name (index $seen $e.name)) -}}
  {{- end -}}
  {{- $_ := set $seen $e.name $i -}}
{{- end -}}
{{- end -}}

{{/*
Render a value that may legitimately be zero, falling back only when it is unset.

Sprig's `default` treats 0 as empty, so `x | default "20"` silently replaces a
numeric 0 (`--set app.httpPoolMaxKeepalive=0`, or an unquoted 0 in a values file)
with the default -- which disables the EGRESS_HTTP_POOL_* runtime rollback lever.
Only nil and "" fall back here.

Call as: {{ include "mcpgw.valueOrDefault" (list .Values.app.httpPoolMaxKeepalive "20") }}
*/}}
{{- define "mcpgw.valueOrDefault" -}}
{{- $value := index . 0 -}}
{{- if or (kindIs "invalid" $value) (eq (toString $value) "") -}}
{{- index . 1 -}}
{{- else -}}
{{- toString $value -}}
{{- end -}}
{{- end -}}

{{/*
Pod annotations for this workload, including optional service-mesh sidecar
injection. Mesh-agnostic: the only mesh wired today is Istio, gated on the
stack's global.serviceMesh passthrough (global.serviceMesh.enabled AND
global.serviceMesh.type == "istio"). When active, this long-running workload is
annotated for sidecar injection and (optionally)
holdApplicationUntilProxyStarts. Operator-supplied .Values.podAnnotations merge
on top. Additive: with the mesh off, only .Values.podAnnotations (default {})
renders, so a non-mesh install is byte-for-byte unchanged.

Call as: {{- include "mcpgw.podAnnotations" . | nindent 8 }}
(renders nothing when there are no annotations).
*/}}
{{- define "mcpgw.podAnnotations" -}}
{{- $ann := dict -}}
{{- $sm := .Values.global.serviceMesh | default dict -}}
{{- if and $sm.enabled (eq ($sm.type | default "istio") "istio") -}}
{{- $istio := $sm.istio | default dict -}}
{{- if $istio.holdApplicationUntilProxyStarts -}}
{{- $_ := set $ann "proxy.istio.io/config" "{\"holdApplicationUntilProxyStarts\": true}" -}}
{{- end -}}
{{- end -}}
{{- with .Values.podAnnotations -}}
{{- $ann = merge (deepCopy .) $ann -}}
{{- end -}}
{{- if $ann -}}
{{- toYaml $ann -}}
{{- end -}}
{{- end -}}

{{/*
Pod labels for service-mesh sidecar injection. Istio's injector webhook selects
pods by the LABEL sidecar.istio.io/inject (its objectSelector reads labels, not
annotations), so the opt-in is a label. Emitted only when global.serviceMesh is
enabled with type "istio"; renders nothing otherwise, so a non-mesh install is
unchanged.

Call as: {{- include "mcpgw.meshPodLabels" . | nindent 8 }}
*/}}
{{- define "mcpgw.meshPodLabels" -}}
{{- $sm := .Values.global.serviceMesh | default dict -}}
{{- if and $sm.enabled (eq ($sm.type | default "istio") "istio") -}}
sidecar.istio.io/inject: "true"
{{- end -}}
{{- end -}}
