{{/*
Name of the ServiceAccount the setup-mongodb Job runs as. Explicit
serviceAccount.name wins; otherwise "setup-mongodb". A dedicated identity, not
the namespace "default" ServiceAccount, so a service-mesh AuthorizationPolicy on
MongoDB can allow exactly this Job. The stack chart reads the same value to build
that policy's principal.
*/}}
{{- define "mongodb-configure.serviceAccountName" -}}
{{- .Values.serviceAccount.name | default "setup-mongodb" -}}
{{- end -}}
