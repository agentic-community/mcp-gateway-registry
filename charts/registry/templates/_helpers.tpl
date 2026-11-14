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
  1. env: block — feature flags and IdP secrets via valueFrom. Includes
     PingFederate plain env vars (BASE_URL, EXTERNAL_URL, CLIENT_ID,
     M2M_CLIENT_ID, APPLICATION_ID_URI, GROUPS_CLAIM, ENABLED) and
     valueFrom-sourced secrets (CLIENT_SECRET, M2M_CLIENT_SECRET). Also
     RUM_SNIPPET_B64 and RUM_ALLOWED_HOSTS (feature #1471), rendered from
     .Values.rumSnippetB64 / .Values.rumAllowedHosts only when non-empty;
     token-bearing snippets come via extraEnvFrom. Also UI_BRANDING_DIR,
     rendered from .Values.branding.mountPath only when branding.enabled.
  2. registry-app-log-config configmap
  3. registry-otel-config configmap
  4. registry-batch-config configmap
  5. registry per-chart secret
  6. keycloak-client-secret (runtime-created by keycloak-configure Job)
  7. mongo-credentials secret
  8. shared-secret (stack-level)
  9. registry-egress-config configmap (egress vault, when egressAuth.enabled)
     + AUTH_SERVER_NGINX_MARKER_SECRET (env/shared secret)

Over-rejection is preferred to under-rejection: a user attempting to
inject one of these via extraEnv gets a clear template-render error.
*/}}
{{- define "registry.reservedEnvNames" -}}
{{- $content := .Files.Get "reserved-env-names.txt" -}}
{{- compact (splitList "\n" $content) | toYaml -}}
{{- end -}}

{{/*
Name of the ServiceAccount the registry pod runs as. Explicit
serviceAccount.name wins; otherwise defaults to the app name ("registry").
OpenBao's kubernetes-auth `registry` role is bound to this SA, and other
roles (RBAC, IRSA, etc.) can be attached to the same SA over time.
*/}}
{{- define "registry.serviceAccountName" -}}
{{- if .Values.serviceAccount.name -}}
{{- .Values.serviceAccount.name -}}
{{- else -}}
{{- .Values.app.name -}}
{{- end -}}
{{- end -}}

{{/*
Validate additional registry hostnames and ensure browser session cookies can
be shared with them. The auth-server and registry emit their own cookie-domain
settings, so validate against the registry chart's effective setting here.
*/}}
{{- define "registry.additionalHostnames" -}}
{{- $raw := .Values.global.ingress.additionalHostnames | default (list) -}}
{{- if not (kindIs "slice" $raw) -}}
  {{- fail (printf "global.ingress.additionalHostnames must be a list, got %s %q (with --set use: 'global.ingress.additionalHostnames={a.example.com}')" (kindOf $raw) (toString $raw)) -}}
{{- end -}}
{{- $domain := .Values.global.domain | default "localhost" -}}
{{- $configuredDomain := include "registry.sessionSetting" (list . "sessionCookieDomain" (list "app" "sessionCookieDomain")) | fromJson -}}
{{- $cookieDomain := $configuredDomain.value | default (printf ".%s" $domain) -}}
{{- $apex := trimPrefix "." $cookieDomain | lower -}}
{{- $hosts := list -}}
{{- range $i, $hostname := $raw -}}
  {{- $hostname = toString $hostname -}}
  {{- if or (gt (len $hostname) 253) (not (regexMatch "^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?(\\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)+$" $hostname)) -}}
    {{- fail (printf "global.ingress.additionalHostnames[%d]: %q is not a bare lowercase DNS hostname (no scheme, port, path, wildcard, or hostname over 253 characters)" $i $hostname) -}}
  {{- end -}}
  {{- if not (or (eq $hostname $apex) (hasSuffix (printf ".%s" $apex) $hostname)) -}}
    {{- fail (printf "global.ingress.additionalHostnames[%d]: %q is outside the effective session cookie domain %q; browser login cannot complete on it. Keep the hostname under that cookie domain, or set %s to a parent domain covering every hostname." $i $hostname $cookieDomain $configuredDomain.key) -}}
  {{- end -}}
  {{- $hosts = append $hosts $hostname -}}
{{- end -}}
{{- toYaml $hosts -}}
{{- end -}}

{{/*
Whether the dedicated internal vend listener (nginx :8091, egressAuth.internalPort)
must be exposed. Two INDEPENDENT features share this listener:
  - the per-user egress-token vault (egressAuth.enabled), and
  - the generic-proxy static upstream-header vend
    (app.gatewayGenericProxyEnabled, /_egress_internal/generic-upstream-headers).
Either one requires the Service port and its NetworkPolicy, so both the
service.yaml port block and networkpolicy-egress-internal.yaml gate on this to
stay in sync. Emits the string "true" when enabled, empty otherwise (use with
`if`).
*/}}
{{- define "registry.internalVendListenerEnabled" -}}
{{- if or .Values.egressAuth.enabled .Values.app.gatewayGenericProxyEnabled -}}
true
{{- end -}}
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

Call as: {{ include "registry.envNameKey" $name }}
*/}}
{{- define "registry.envNameKey" -}}
{{- . | toString | trim | upper -}}
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
{{- $reserved := list -}}
{{- range (fromYamlArray (include "registry.reservedEnvNames" .)) -}}
  {{- $reserved = append $reserved (include "registry.envNameKey" .) -}}
{{- end -}}
{{- $seen := dict -}}
{{- range $i, $e := .Values.extraEnv -}}
  {{- if not $e.name -}}
    {{- fail (printf "registry.extraEnv[%d]: missing required 'name' field" $i) -}}
  {{- end -}}
  {{- $key := include "registry.envNameKey" $e.name -}}
  {{- if has $key $reserved -}}
    {{- fail (printf "registry.extraEnv[%d]: %q is a reserved variable managed by the chart (via env: or envFrom from the chart's secrets/configmaps). Remove it from extraEnv. If a values.yaml field controls it (e.g. app.showSkillsTab for SHOW_SKILLS_TAB), set that instead; otherwise the value is managed by the chart's internal secrets and must not be overridden via extraEnv." $i $e.name) -}}
  {{- end -}}
  {{- /* Duplicates compare the exact name: Kubernetes treats HTTPS_PROXY and
         https_proxy as two variables, and proxy setups legitimately set both. */ -}}
  {{- if hasKey $seen $e.name -}}
    {{- fail (printf "registry.extraEnv[%d]: duplicate name %q (first seen at index %v)" $i $e.name (index $seen $e.name)) -}}
  {{- end -}}
  {{- $_ := set $seen $e.name $i -}}
{{- end -}}
{{- end -}}

{{/*
Render a value that may legitimately be zero, falling back only when it is unset.

Sprig's `default` treats 0 as empty, so `x | default "20"` silently replaces a
numeric 0 (`--set egressAuth.httpPoolMaxKeepalive=0`, or an unquoted 0 in a
values file) with the default -- which disables the EGRESS_HTTP_POOL_* runtime
rollback lever. Only nil and "" fall back here.

Call as: {{ include "registry.valueOrDefault" (list .Values.egressAuth.httpPoolMaxKeepalive "20") }}
*/}}
{{- define "registry.valueOrDefault" -}}
{{- $value := index . 0 -}}
{{- if or (kindIs "invalid" $value) (eq (toString $value) "") -}}
{{- index . 1 -}}
{{- else -}}
{{- toString $value -}}
{{- end -}}
{{- end -}}

{{/*
The effective value of a session setting and the value path it came from, as
JSON {"value": ..., "key": ...}. The registry and the auth-server must agree on
every session setting, so the stack chart sets each once as global.<name> and
both charts render from that. The fallback path applies only when the global
is unset, which is a standalone install of this chart. "value" is null when
neither is set; "key" names the path for error messages.

Call as: {{ include "registry.sessionSetting" (list . "sessionMaxAgeSeconds" (list "app" "sessionMaxAgeSeconds")) | fromJson }}
*/}}
{{- define "registry.sessionSetting" -}}
{{- $values := (index . 0).Values -}}
{{- $name := index . 1 -}}
{{- $result := dict "value" (index $values.global $name) "key" (printf "global.%s" $name) -}}
{{- if kindIs "invalid" $result.value -}}
{{- $fallback := index . 2 -}}
{{- $node := $values -}}
{{- range $part := $fallback -}}
{{- $node = index ($node | default dict) $part -}}
{{- end -}}
{{- $result = dict "value" $node "key" (join "." $fallback) -}}
{{- end -}}
{{- toJson $result -}}
{{- end -}}

{{/*
Name of the Secret holding OPENAPI_MCP_SYNC_SECRET (issue #2022). The stack's
global.openapiMcp.existingSyncSecret wins, then app.openapiMcpSyncSecretName.
The openapi-mcp chart resolves the same name, so both pods read one Secret.
*/}}
{{- define "registry.openapiMcpSyncSecretName" -}}
{{- $existing := dig "openapiMcp" "existingSyncSecret" "" .Values.global -}}
{{- $existing | default .Values.app.openapiMcpSyncSecretName -}}
{{- end -}}
