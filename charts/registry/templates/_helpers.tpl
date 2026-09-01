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
     token-bearing snippets come via extraEnvFrom.
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

{{/*
In-cluster Keycloak HTTP URL.

Nginx proxy_pass interpolates KEYCLOAK_URL as a literal hostname and
resolves it once at start. The Bitnami headless Service's A record is
the pod IP, so a Keycloak roll 503s /realms/ until the registry
restarts. The URL therefore targets the stack-owned ClusterIP Service
"<release>-kc-int" (stack chart,
templates/keycloak-internal-service.yaml): a stable VIP whose name and
8080 port are fixed by the stack chart itself, unlike the Bitnami
Services whose name follows nameOverride/fullnameOverride and whose
port list is replaceable user values.

The short in-namespace hostname is deliberate: every consumer runs in
the release namespace, the resolver search path expands it, and it
keeps the URL independent of namespace and cluster-domain settings.
No truncation: Helm caps release names at 53 chars, so with the
7-char "-kc-int" suffix the name never exceeds the 63-char DNS label
limit, and keeping the full release name keeps it unique per release.

Port 8080 MUST stay explicit: HTTP clients omit default ports from the
Host header, Keycloak derives issuer/discovery URLs from Host, and a
portless issuer breaks auth-server's exact-match issuer allowlist and
startswith() discovery rewrite. Note the hostname does differ from the
old headless URL, so tokens minted against the previous internal
issuer are invalidated when this fix lands; deployments that pin the
issuer via KC_HOSTNAME are unaffected.

global.authProvider.keycloak.internalUrl is the one canonical
override, shared with the stack chart (Helm propagates globals to
subcharts): the stack requires it for external Keycloak
(keycloak.create=false), and standalone installs of this chart use it
to point at their own Keycloak. The default assumes the stack
topology (same release name and namespace as the stack that owns the
Service). The stack and sibling charts render the same URL from their
own copies of this helper; each chart's keycloak_service_test.yaml
pins the string.
*/}}
{{- define "registry.keycloakInternalUrl" -}}
{{- $override := dig "authProvider" "keycloak" "internalUrl" "" (.Values.global | default dict) -}}
{{- if $override -}}
{{- $override -}}
{{- else -}}
{{- printf "http://%s-kc-int:8080" .Release.Name -}}
{{- end -}}
{{- end -}}
