{{/*
Reserved env var names for the registry chart.

Users must not supply these via .Values.extraEnv. The list is the union of:
  - the superset of names the chart may render into `env:` (including
    every conditional branch), and
  - every key the chart sources via `envFrom` from stack-level or
    per-chart secrets/configmaps.

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
- DEPLOYMENT_MODE
- REGISTRY_MODE
- SHOW_SERVERS_TAB
- SHOW_VIRTUAL_SERVERS_TAB
- SHOW_SKILLS_TAB
- SHOW_AGENTS_TAB
- AWS_REGISTRY_FEDERATION_ENABLED
- KEYCLOAK_ADMIN_PASSWORD
- ENTRA_CLIENT_SECRET
- OKTA_CLIENT_SECRET
- OKTA_M2M_CLIENT_SECRET
- OKTA_API_TOKEN
- AUTH0_CLIENT_SECRET
- AUTH0_M2M_CLIENT_SECRET
- AUTH0_MANAGEMENT_API_TOKEN
- REGISTRY_API_KEYS
- ANS_API_KEY
- ANS_API_SECRET
- APP_LOG_BACKUP_COUNT
- APP_LOG_CENTRALIZED_ENABLED
- APP_LOG_CENTRALIZED_TTL_DAYS
- APP_LOG_EXCLUDED_LOGGERS
- APP_LOG_LEVEL
- APP_LOG_MAX_BYTES
- APP_LOG_MONGODB_BUFFER_SIZE
- APP_LOG_MONGODB_FLUSH_INTERVAL_SECONDS
- DISABLE_AI_REGISTRY_TOOLS_SERVER
- MCP_TELEMETRY_DEBUG
- MCP_TELEMETRY_DISABLED
- MCP_TELEMETRY_HEARTBEAT_INTERVAL_MINUTES
- MCP_TELEMETRY_IMDS_PROBE_DISABLED
- MCP_TELEMETRY_OPT_OUT
- OTEL_EXPORTER_OTLP_HEADERS
- OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE
- OTEL_OTLP_ENDPOINT
- OTEL_OTLP_EXPORT_INTERVAL_MS
- ANS_API_ENDPOINT
- ANS_API_TIMEOUT_SECONDS
- ANS_INTEGRATION_ENABLED
- ANS_SYNC_INTERVAL_HOURS
- ANS_VERIFICATION_CACHE_TTL_SECONDS
- AUTH_PROVIDER
- AUTH_SERVER_EXTERNAL_URL
- AUTH_SERVER_URL
- AWS_REGION
- COGNITO_CLIENT_ID
- COGNITO_CLIENT_SECRET
- COGNITO_DOMAIN
- COGNITO_ENABLED
- COGNITO_USER_POOL_ID
- ENTRA_CLIENT_ID
- ENTRA_ENABLED
- ENTRA_TENANT_ID
- GATEWAY_ADDITIONAL_SERVER_NAMES
- IDP_GROUP_FILTER_PREFIX
- KEYCLOAK_ADMIN
- KEYCLOAK_ENABLED
- KEYCLOAK_REALM
- KEYCLOAK_URL
- OKTA_AUTH_SERVER_ID
- OKTA_CLIENT_ID
- OKTA_DOMAIN
- OKTA_ENABLED
- REGISTRATION_GATE_AUTH_CREDENTIAL
- REGISTRATION_GATE_AUTH_HEADER_NAME
- REGISTRATION_GATE_AUTH_TYPE
- REGISTRATION_GATE_ENABLED
- REGISTRATION_GATE_MAX_RETRIES
- REGISTRATION_GATE_TIMEOUT_SECONDS
- REGISTRATION_GATE_URL
- REGISTRATION_GATE_OAUTH2_TOKEN_URL
- REGISTRATION_GATE_OAUTH2_CLIENT_ID
- REGISTRATION_GATE_OAUTH2_CLIENT_SECRET
- REGISTRATION_GATE_OAUTH2_SCOPE
- M2M_DIRECT_REGISTRATION_ENABLED
- REGISTRATION_WEBHOOK_AUTH_HEADER
- REGISTRATION_WEBHOOK_AUTH_TOKEN
- REGISTRATION_WEBHOOK_TIMEOUT_SECONDS
- REGISTRATION_WEBHOOK_URL
- REGISTRY_API_TOKEN
- REGISTRY_CONTACT_EMAIL
- REGISTRY_CONTACT_URL
- REGISTRY_DESCRIPTION
- REGISTRY_ID
- REGISTRY_NAME
- REGISTRY_ORGANIZATION_NAME
- REGISTRY_URL
- ROOT_PATH
- SECRET_KEY
- SESSION_COOKIE_DOMAIN
- SESSION_COOKIE_SECURE
- SKILL_SECURITY_ANALYZERS
- SKILL_SECURITY_SCAN_ENABLED
- KEYCLOAK_CLIENT_ID
- KEYCLOAK_CLIENT_SECRET
- KEYCLOAK_M2M_CLIENT_ID
- KEYCLOAK_M2M_CLIENT_SECRET
- DOCUMENTDB_DATABASE
- DOCUMENTDB_HOST
- DOCUMENTDB_NAMESPACE
- DOCUMENTDB_PASSWORD
- DOCUMENTDB_PORT
- DOCUMENTDB_REPLICA_SET
- DOCUMENTDB_USERNAME
- DOCUMENTDB_USE_TLS
- MONGODB_CONNECTION_STRING
- STORAGE_BACKEND
- ASOR_ACCESS_TOKEN
- FEDERATION_CLIENT_ID
- FEDERATION_CLIENT_SECRET
- FEDERATION_ENCRYPTION_KEY
- FEDERATION_STATIC_TOKEN
- FEDERATION_STATIC_TOKEN_AUTH_ENABLED
- FEDERATION_TOKEN_ENDPOINT
- WORKDAY_TOKEN_URL
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
