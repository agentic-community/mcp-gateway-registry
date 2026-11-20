{{/*
Names. These are load-bearing: other components resolve Keycloak at
"{{ .Release.Name }}-keycloak-headless:8080", so the headless service name and
port are a hard contract. Keycloak connects to postgres at a NEW service name
(avoids colliding with the Bitnami postgresql service during upgrade).
*/}}
{{- define "keycloak.fullname" -}}
{{- printf "%s-keycloak" .Release.Name -}}
{{- end -}}

{{- define "keycloak.serviceAccountName" -}}
{{- .Values.serviceAccount.name | default (ternary (include "keycloak.fullname" .) "default" .Values.serviceAccount.create) -}}
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


{{- define "keycloak.adminSecretName" -}}
{{- dig "authProvider" "keycloak" "adminExistingSecret" "" (.Values.global | default dict)
  | default .Values.auth.existingSecret
  | default (printf "%s-keycloak" .Release.Name) -}}
{{- end -}}

{{- define "keycloak.adminSecretKey" -}}
{{- dig "authProvider" "keycloak" "adminExistingSecretPasswordKey" ""
  (.Values.global | default dict) | default .Values.auth.existingSecretPasswordKey -}}
{{- end -}}

{{- define "keycloak.adminUsername" -}}
{{- dig "authProvider" "keycloak" "adminUsername" "" (.Values.global | default dict)
  | default .Values.auth.adminUser -}}
{{- end -}}

{{- define "keycloak.pgSecretName" -}}
{{- .Values.postgres.existingSecret | default (printf "%s-keycloak-postgresql" .Release.Name) -}}
{{- end -}}

{{- define "keycloak.postgresInitName" -}}
{{- printf "%s-keycloak-postgres-init" .Release.Name -}}
{{- end -}}


{{/*
"true" when this chart deploys Keycloak: `create` is on AND the gateway's auth
provider (global.authProvider.type, default "keycloak") is Keycloak. With any
other provider nothing renders, so a stack set to Entra/Okta/etc. never runs
an unused Keycloak just because `create` kept its default.
*/}}
{{- define "keycloak.enabled" -}}
{{- $authType := dig "authProvider" "type" "keycloak" (.Values.global | default dict) -}}
{{- if and .Values.create (eq $authType "keycloak") -}}
true
{{- end -}}
{{- end -}}

{{/* Explicit bootstrap intent is independent of Helm/GitOps release detection. */}}
{{- define "keycloak.validate" -}}
{{- if include "keycloak.enabled" . -}}
{{- $legacy := dict
  "extraEnvVars" "extraEnv"
  "replicaCount" "replicas"
  "production" "the production start command (always enabled)"
  "proxy" "proxyHeaders"
  "externalDatabase" "an external Keycloak (create=false); bundled external PostgreSQL is unsupported"
  "lifecycleHooks" "sslInit"
  "extraEnvVarsCM" "extraEnv"
  "extraEnvVarsSecret" "extraEnv"
  "hostnameStrict" "the pinned canonical hostname (strict validation is always enabled)"
  "extraVolumes" "a derived official-layout image; extra mounts are unsupported"
  "extraVolumeMounts" "a derived official-layout image; extra mounts are unsupported"
  "initdbScripts" "postgres.bootstrap"
  "initdbScriptsConfigMap" "postgres.bootstrap"
  "spi" "a derived official-layout image; legacy SPI options are unsupported"
-}}
{{- range $key, $replacement := $legacy -}}
{{- if hasKey $.Values $key -}}
{{- fail (printf "keycloak.%s is a legacy Bitnami value and is not supported; remove it and configure %s" $key $replacement) -}}
{{- end -}}
{{- end -}}
{{- range $key := list "initContainers" "sidecars" "customCaExistingSecret" "ingress" -}}
{{- if hasKey $.Values $key -}}
{{- fail (printf (print "keycloak.%s is a legacy Bitnami value and is unsupported; "
  "remove it and use an official-layout derived image or an external Keycloak") $key) -}}
{{- end -}}
{{- end -}}
{{- range $key := list "registry" "debug" -}}
{{- if hasKey $.Values.image $key -}}
{{- fail (printf (print "keycloak.image.%s is unsupported; use image.repository with "
  "the complete registry/repository and an official-compatible image layout") $key) -}}
{{- end -}}
{{- end -}}
{{- range $key := list "passwordSecretKey" "secretKeys" "existingSecretPerPassword" -}}
{{- if hasKey $.Values.auth $key -}}
{{- fail (printf "keycloak.auth.%s is unsupported; use auth.existingSecretPasswordKey" $key) -}}
{{- end -}}
{{- end -}}
{{- range $key, $value := (.Values.postgresql | default dict) -}}
{{- if ne $key "auth" -}}
{{- fail (printf "keycloak.postgresql.%s is a legacy Bitnami value; remove it and configure keycloak.postgres (externalDatabase is unsupported)" $key) -}}
{{- end -}}
{{- range $authKey, $authValue := $value -}}
{{- if not (has $authKey (list "password" "postgresPassword")) -}}
{{- fail (printf "keycloak.postgresql.auth.%s is a legacy Bitnami value; use keycloak.postgres.%s instead" $authKey $authKey) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- range $key := list "migrateFromBitnami" "source" "migrationPvcSize" -}}
{{- if hasKey $.Values.postgres $key -}}
{{- fail (printf "keycloak.postgres.%s is no longer supported; choose postgres.bootstrap.mode and prepare a verified, quiescent snapshot for restore" $key) -}}
{{- end -}}
{{- end -}}
{{- range $name := list (include "keycloak.adminSecretName" .)
  .Values.auth.existingSecret .Values.postgres.existingSecret -}}
{{- if contains "{{" $name -}}
{{- fail "Keycloak existingSecret names are literal, not templates; replace {{ .Release.Name }} with the actual release name" -}}
{{- end -}}
{{- end -}}
{{/* Reserved names include every conditional keycloak.yaml env entry and direct database URL overrides. Edit reserved-env-names.txt and rebuild the parent dependency. */}}
{{- $reserved := compact (splitList "\n" (.Files.Get "reserved-env-names.txt")) -}}
{{- $seen := dict -}}
{{- range .Values.extraEnv -}}
{{- if has .name $reserved -}}
{{- fail (printf "keycloak.extraEnv cannot override chart-managed %s; use the corresponding keycloak value" .name) -}}
{{- end -}}
{{- if hasKey $seen .name -}}
{{- fail (printf "keycloak.extraEnv has duplicate variable %s" .name) -}}
{{- end -}}
{{- $_ := set $seen .name true -}}
{{- end -}}
{{- $bootstrap := .Values.postgres.bootstrap -}}
{{- range $field := list "username" "database" -}}
{{- $name := index $.Values.postgres $field -}}
{{- if not (kindIs "string" $name) -}}
{{- fail (printf "keycloak.postgres.%s must be a string" $field) -}}
{{- end -}}
{{- if not (regexMatch "^[A-Za-z_][A-Za-z0-9_-]{0,62}$" $name) -}}
{{- fail (printf "keycloak.postgres.%s must be 1-63 ASCII letters, digits, underscores or hyphens, starting with a letter or underscore" $field) -}}
{{- end -}}
{{- end -}}
{{- if or (has (lower .Values.postgres.username) (list "postgres" "public")) (hasPrefix "pg_" (lower .Values.postgres.username)) -}}
{{- fail "keycloak.postgres.username must be a dedicated non-superuser role, not postgres, public or a pg_-prefixed role" -}}
{{- end -}}
{{- if has (lower .Values.postgres.database) (list "postgres" "template0" "template1") -}}
{{- fail "keycloak.postgres.database must be a dedicated database, not postgres, template0 or template1" -}}
{{- end -}}
{{- if semverCompare "<26.8.2" .Values.image.tag -}}
{{- fail "the bundled Keycloak chart requires version 26.8.2 or newer" -}}
{{- end -}}
{{- if not (has .Values.cache (list "local" "ispn")) -}}
{{- fail "keycloak.cache must be local or ispn" -}}
{{- end -}}
{{- if and (gt (int .Values.replicas) 1) (ne .Values.cache "ispn") -}}
{{- fail "keycloak.replicas greater than one requires cache=ispn; local cache cannot share login state" -}}
{{- end -}}
{{- if not (has .Values.proxyHeaders (list "xforwarded" "forwarded")) -}}
{{- fail "keycloak.proxyHeaders must be xforwarded or forwarded for the bundled reverse-proxy topology" -}}
{{- end -}}
{{- if not (has $bootstrap.mode (list "fresh-install" "restore" "existing")) -}}
{{- fail "keycloak.postgres.bootstrap.mode is required: fresh-install for an intentionally empty database, restore for a verified quiescent snapshot, or existing for an externally verified official PostgreSQL volume; Helm/GitOps cannot infer absence of legacy data" -}}
{{- end -}}
{{- if eq $bootstrap.mode "restore" -}}
{{- if or (not $bootstrap.existingClaim) (not (regexMatch "^[a-f0-9]{64}$" $bootstrap.sha256)) (not $bootstrap.sourceQuiesced) -}}
{{- fail "restore requires keycloak.postgres.bootstrap.existingClaim containing restore.dump, its lowercase sha256, and sourceQuiesced=true; stop ALL legacy writers before the definitive pg_dump and keep them stopped through verified cutover" -}}
{{- end -}}
{{- else if or $bootstrap.existingClaim $bootstrap.sha256 $bootstrap.sourceQuiesced -}}
{{- fail "snapshot settings are only valid with keycloak.postgres.bootstrap.mode=restore; do not bypass an unfinished restore by changing modes" -}}
{{- end -}}
{{- if and (eq $bootstrap.mode "existing") (not $bootstrap.existingVerified) -}}
{{- fail "keycloak.postgres.bootstrap.mode=existing requires existingVerified=true after verifying the official PostgreSQL volume and Keycloak data; this mode refuses an empty or incomplete volume" -}}
{{- end -}}
{{- $_ := include "keycloak.hostname" . -}}
{{- end -}}
{{- end -}}

{{- define "keycloak.hostname" -}}
{{- $global := .Values.global | default dict -}}
{{- $external := dig "authProvider" "keycloak" "externalUrl" "" $global -}}
{{- if and $global.domain .Values.hostname (not $external) -}}
{{- fail "keycloak.hostname is standalone-only with global ingress; set global.authProvider.keycloak.externalUrl so Keycloak and its clients share one canonical URL" -}}
{{- end -}}
{{- if and $external .Values.hostname (ne $external .Values.hostname) -}}
{{- fail "keycloak.hostname conflicts with global.authProvider.keycloak.externalUrl; use one canonical external URL" -}}
{{- end -}}
{{- if or $external .Values.hostname -}}
{{- $url := $external | default .Values.hostname -}}
{{- if not (regexMatch "^https?://[^/?#@]+(/[^?#]*)?$" $url) -}}
{{- fail "Keycloak hostname must be a full http(s) URL without userinfo, query, or fragment" -}}
{{- end -}}
{{- if $global.domain -}}
{{- $ingress := $global.ingress | default dict -}}
{{- $protocol := ternary "https" "http" ($ingress.tls | default false) -}}
{{- $port := ternary "443" "80" ($ingress.tls | default false) -}}
{{- $primary := $global.domain -}}
{{- if ne ($ingress.routingMode | default "subdomain") "path" -}}
{{- $primary = printf "%s.%s" ($ingress.registrySubdomain | default "mcpregistry") $global.domain -}}
{{- end -}}
{{- $allowed := list $primary -}}
{{- range ($ingress.additionalHostnames | default list) -}}
{{- $allowed = append $allowed . -}}
{{- end -}}
{{- $parsed := urlParse $url -}}
{{- $host := trimSuffix (printf ":%s" $port) $parsed.host -}}
{{- if or (ne $parsed.scheme $protocol) (not (has $host $allowed)) -}}
{{- fail (printf "global.authProvider.keycloak.externalUrl is not served by the bundled gateway ingress; use %s and a primary/additional gateway hostname with its default port, or deploy an external Keycloak with keycloak.create=false" $protocol) -}}
{{- end -}}
{{- end -}}
{{- trimSuffix "/" $url -}}
{{- else if $global.domain -}}
{{- $ingress := $global.ingress | default dict -}}
{{- $protocol := ternary "https" "http" ($ingress.tls | default false) -}}
{{- if eq ($ingress.routingMode | default "subdomain") "path" -}}
{{- printf "%s://%s" $protocol $global.domain -}}
{{- else -}}
{{- printf "%s://%s.%s" $protocol ($ingress.registrySubdomain | default "mcpregistry") $global.domain -}}
{{- end -}}
{{- else -}}
{{- fail "keycloak.hostname is required for the standalone chart; set the canonical external http(s) URL (the stack derives it from global ingress)" -}}
{{- end -}}
{{- end -}}

{{/*
Container hardening shared by Keycloak, its wait-for-postgres init container,
and PostgreSQL. The root filesystem is
read-only; each pod mounts emptyDirs for the paths its image writes to.
*/}}
{{- define "keycloak.containerSecurityContext" -}}
allowPrivilegeEscalation: false
privileged: false
readOnlyRootFilesystem: true
capabilities:
  drop:
    - ALL
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
