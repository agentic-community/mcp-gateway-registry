{{/*
"true" when the gateway authenticates with Keycloak
(global.authProvider.type, default "keycloak"). Every template in this chart
renders only then: with any other provider there is no realm to configure.
*/}}
{{- define "keycloak-configure.enabled" -}}
{{- if eq (dig "authProvider" "type" "keycloak" (.Values.global | default dict)) "keycloak" -}}
true
{{- end -}}
{{- end }}

{{- define "keycloak-configure.realm" -}}
{{- $globalKeycloak := dig "authProvider" "keycloak" dict .Values.global -}}
{{- if and .Values.keycloak.existingSecret (not .Values.keycloak.realm) -}}
{{- fail "keycloak.existingSecret requires an explicit keycloak.realm matching the Secret REALM; the chart will not infer or overwrite an unknown realm" -}}
{{- end -}}
{{- $realm := $globalKeycloak.realm | default (.Values.keycloak.realm | default "mcp-gateway") -}}
{{- if and $globalKeycloak.realm .Values.keycloak.realm (ne $globalKeycloak.realm .Values.keycloak.realm) -}}
{{- fail "keycloak.realm conflicts with global.authProvider.keycloak.realm; use the global realm or leave the local value empty" -}}
{{- end -}}
{{- if or (eq (lower $realm) "master") (not (regexMatch "^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$" $realm)) -}}
{{- fail "Keycloak gateway realm must be 1-64 letters, digits, underscores or hyphens and must not be master" -}}
{{- end -}}
{{- $realm -}}
{{- end -}}
