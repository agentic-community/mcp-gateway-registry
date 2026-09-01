{{/*
In-cluster Keycloak HTTP URL for KEYCLOAK_URL / KEYCLOAK_ADMIN_URL,
targeting the stack-owned ClusterIP Service "<release>-kc-int" (stack
chart, templates/keycloak-internal-service.yaml). See
registry.keycloakInternalUrl for the full rationale: nginx pins
proxy_pass IPs at start, so the URL must be a stable ClusterIP name;
port 8080 must stay explicit so Keycloak keeps deriving issuers with a
port; the short hostname resolves through the pod's namespace search
path; no truncation is needed because Helm caps release names at 53
chars (53 + 7-char suffix < 63).
global.authProvider.keycloak.internalUrl is the one canonical
override, for external Keycloak and standalone installs. The default
must render the same string as the registry, auth-server, and stack
copies of this helper.
*/}}
{{- define "keycloak-configure.keycloakInternalUrl" -}}
{{- $override := dig "authProvider" "keycloak" "internalUrl" "" (.Values.global | default dict) -}}
{{- if $override -}}
{{- $override -}}
{{- else -}}
{{- printf "http://%s-kc-int:8080" .Release.Name -}}
{{- end -}}
{{- end -}}
