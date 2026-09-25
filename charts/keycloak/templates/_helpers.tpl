{{/*
Names. These are load-bearing: other components resolve Keycloak at
"{{ .Release.Name }}-keycloak-headless:8080", so the headless service name and
port are a hard contract. Keycloak connects to postgres at a NEW service name
(avoids colliding with the Bitnami postgresql service during upgrade).
*/}}
{{- define "keycloak.fullname" -}}
{{- printf "%s-keycloak" .Release.Name -}}
{{- end -}}

{{- define "keycloak.headlessName" -}}
{{- printf "%s-keycloak-headless" .Release.Name -}}
{{- end -}}

{{- define "keycloak.postgresName" -}}
{{- printf "%s-keycloak-postgres" .Release.Name -}}
{{- end -}}

{{- define "keycloak.postgresHeadlessName" -}}
{{- printf "%s-keycloak-postgres-headless" .Release.Name -}}
{{- end -}}

{{- define "keycloak.migrationPvcName" -}}
{{- printf "%s-keycloak-pg-migration" .Release.Name -}}
{{- end -}}

{{- define "keycloak.adminSecretName" -}}
{{- .Values.auth.existingSecret | default (printf "%s-keycloak" .Release.Name) -}}
{{- end -}}

{{- define "keycloak.pgSecretName" -}}
{{- .Values.postgres.existingSecret | default (printf "%s-keycloak-postgresql" .Release.Name) -}}
{{- end -}}

{{- define "keycloak.bitnamiPgService" -}}
{{- .Values.postgres.source.serviceName | default (printf "%s-postgresql" .Release.Name) -}}
{{- end -}}

{{/* Labels */}}
{{- define "keycloak.commonLabels" -}}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{- end -}}

{{- define "keycloak.serverSelectorLabels" -}}
app.kubernetes.io/name: keycloak
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: server
{{- end -}}

{{- define "keycloak.postgresSelectorLabels" -}}
app.kubernetes.io/name: keycloak
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: postgres
{{- end -}}
