variable "name" {
  description = "Name of the deployment"
  type        = string
  default     = "mcp-gateway"
}

variable "aws_region" {
  description = "AWS region for deployment. Can be set via TF_VAR_aws_region environment variable or terraform.tfvars"
  type        = string
  default     = "us-west-2"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "ingress_cidr_blocks" {
  description = "List of CIDR blocks allowed to access the ALB (main ALB + auth server + registry)"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "enable_monitoring" {
  description = "Whether to enable CloudWatch monitoring and alarms"
  type        = bool
  default     = true
}

variable "alarm_email" {
  description = "Email address for CloudWatch alarm notifications"
  type        = string
  default     = ""
}

variable "alarm_sns_topic_arn" {
  description = "SNS topic ARN for CloudWatch alarm notifications. Leave empty to disable SNS notifications."
  type        = string
  default     = ""
}

#
# Keycloak Configuration Variables
#

variable "use_regional_domains" {
  description = "Use region-based domains (e.g., kc.us-west-2.mycorp.click). If false, uses keycloak_domain and root_domain directly"
  type        = bool
  default     = true
}

variable "base_domain" {
  description = "Base domain for regional domains (e.g., mycorp.click). Used when use_regional_domains is true"
  type        = string
  default     = "mycorp.click"
}

variable "certificate_arn" {
  description = "ARN of ACM certificate for HTTPS. Leave empty to disable HTTPS"
  type        = string
  default     = ""
}

variable "keycloak_domain" {
  description = "Full domain for Keycloak (e.g., kc.example.com). Used when use_regional_domains is false"
  type        = string
  default     = ""
}

variable "root_domain" {
  description = "Root domain with Route53 hosted zone. Used when use_regional_domains is false"
  type        = string
  default     = ""
}

variable "keycloak_admin" {
  description = "Keycloak admin username"
  type        = string
  sensitive   = true
  default     = "admin"
}

variable "keycloak_admin_password" {
  description = "Keycloak admin password"
  type        = string
  sensitive   = true
}

variable "keycloak_database_username" {
  description = "Keycloak database username"
  type        = string
  sensitive   = true
  default     = "keycloak"
}

variable "keycloak_database_password" {
  description = "Keycloak database password"
  type        = string
  sensitive   = true
}

variable "keycloak_database_min_acu" {
  description = "Minimum Aurora Capacity Units"
  type        = number
  default     = 0.5
}

variable "keycloak_database_max_acu" {
  description = "Maximum Aurora Capacity Units"
  type        = number
  default     = 2
}

variable "keycloak_log_level" {
  description = "Keycloak log level"
  type        = string
  default     = "INFO"
}

#
# MCP Gateway Services - Container Images
#

variable "registry_image_uri" {
  description = "Container image URI for registry service"
  type        = string
  default     = ""
}

variable "auth_server_image_uri" {
  description = "Container image URI for auth server service"
  type        = string
  default     = "mcpgateway/auth-server:latest"
}

variable "currenttime_image_uri" {
  description = "Container image URI for currenttime MCP server"
  type        = string
  default     = ""
}

variable "mcpgw_image_uri" {
  description = "Container image URI for mcpgw MCP server"
  type        = string
  default     = ""
}

variable "realserverfaketools_image_uri" {
  description = "Container image URI for realserverfaketools MCP server"
  type        = string
  default     = ""
}

variable "flight_booking_agent_image_uri" {
  description = "Container image URI for flight booking A2A agent"
  type        = string
  default     = ""
}

variable "travel_assistant_agent_image_uri" {
  description = "Container image URI for travel assistant A2A agent"
  type        = string
  default     = ""
}

#
# MCP Gateway Services - Replica Counts
#

variable "currenttime_replicas" {
  description = "Number of replicas for CurrentTime MCP server"
  type        = number
  default     = 1
}

variable "mcpgw_replicas" {
  description = "Number of replicas for MCPGW MCP server"
  type        = number
  default     = 1
}

variable "realserverfaketools_replicas" {
  description = "Number of replicas for RealServerFakeTools MCP server"
  type        = number
  default     = 1
}

variable "flight_booking_agent_replicas" {
  description = "Number of replicas for Flight Booking A2A agent"
  type        = number
  default     = 1
}

variable "travel_assistant_agent_replicas" {
  description = "Number of replicas for Travel Assistant A2A agent"
  type        = number
  default     = 1
}


#
# Embeddings Configuration
#

variable "embeddings_provider" {
  description = "Embeddings provider: 'sentence-transformers' for local models or 'litellm' for API-based models"
  type        = string
  default     = "sentence-transformers"
}

variable "embeddings_model_name" {
  description = "Name of the embeddings model to use (e.g., 'all-MiniLM-L6-v2' for sentence-transformers, 'openai/text-embedding-ada-002' for litellm)"
  type        = string
  default     = "all-MiniLM-L6-v2"
}

variable "embeddings_model_dimensions" {
  description = "Dimension of the embeddings model (e.g., 384 for MiniLM, 1536 for OpenAI/Titan)"
  type        = number
  default     = 384
}

variable "embeddings_aws_region" {
  description = "AWS region for Bedrock embeddings (only used when embeddings_provider is 'litellm' with Bedrock)"
  type        = string
  default     = "us-east-1"
}

variable "embeddings_api_key" {
  description = "API key for embeddings provider (OpenAI, Anthropic, etc.). Only used when embeddings_provider is 'litellm'. Leave empty for Bedrock (uses IAM)."
  type        = string
  default     = ""
  sensitive   = true
}


# =============================================================================
# SESSION COOKIE SECURITY CONFIGURATION
# =============================================================================

variable "session_cookie_secure" {
  description = "Enable secure flag on session cookies (HTTPS-only transmission). Set to true in production with HTTPS."
  type        = bool
  default     = true
}

variable "session_cookie_domain" {
  description = "Domain for session cookies (e.g., '.example.com' for cross-subdomain sharing). Leave empty for single-domain deployments (cookie scoped to exact host only)."
  type        = string
  default     = ""
}

# =============================================================================
# DOCUMENTDB CONFIGURATION (from upstream v1.0.9)
# =============================================================================

variable "documentdb_admin_username" {
  description = "DocumentDB Elastic Cluster admin username"
  type        = string
  sensitive   = true
  default     = "docdbadmin"
}

variable "documentdb_admin_password" {
  description = "DocumentDB Elastic Cluster admin password (minimum 8 characters). Only required when storage_backend is 'documentdb'."
  type        = string
  sensitive   = true
  default     = "" # Not required when using file storage backend
}

variable "documentdb_shard_capacity" {
  description = "vCPU capacity per shard (2, 4, 8, 16, 32, or 64)"
  type        = number
  default     = 2

  validation {
    condition     = contains([2, 4, 8, 16, 32, 64], var.documentdb_shard_capacity)
    error_message = "Shard capacity must be one of: 2, 4, 8, 16, 32, 64"
  }
}

variable "documentdb_shard_count" {
  description = "Number of shards (1-32). Start with 1, scale as needed."
  type        = number
  default     = 1

  validation {
    condition     = var.documentdb_shard_count >= 1 && var.documentdb_shard_count <= 32
    error_message = "Shard count must be between 1 and 32"
  }
}

variable "documentdb_instance_class" {
  description = "Instance class for DocumentDB cluster instances (e.g., db.t3.medium, db.r5.large)"
  type        = string
  default     = "db.t3.medium"

  validation {
    condition     = can(regex("^db\\.(t3|t4g|r5|r6g)\\.(medium|large|xlarge|2xlarge|4xlarge|8xlarge|12xlarge|16xlarge)$", var.documentdb_instance_class))
    error_message = "Instance class must be a valid DocumentDB instance type (e.g., db.t3.medium, db.r5.large)"
  }
}

variable "documentdb_replica_count" {
  description = "Number of read replica instances (0-15). Start with 0, add replicas for HA."
  type        = number
  default     = 0

  validation {
    condition     = var.documentdb_replica_count >= 0 && var.documentdb_replica_count <= 15
    error_message = "Replica count must be between 0 and 15"
  }
}


# Storage Backend Configuration
variable "storage_backend" {
  description = "Storage backend to use: 'file' or 'documentdb'"
  type        = string
  default     = "file"

  validation {
    condition     = contains(["file", "documentdb"], var.storage_backend)
    error_message = "Storage backend must be either 'file' or 'documentdb'."
  }
}

variable "documentdb_database" {
  description = "DocumentDB database name"
  type        = string
  default     = "mcp_registry"
}

variable "documentdb_namespace" {
  description = "DocumentDB namespace for collections"
  type        = string
  default     = "default"
}

variable "documentdb_use_tls" {
  description = "Use TLS for DocumentDB connections"
  type        = bool
  default     = true
}

variable "documentdb_use_iam" {
  description = "Use IAM authentication for DocumentDB"
  type        = bool
  default     = false
}

# =============================================================================
# CLOUDFRONT CONFIGURATION (CloudFront HTTPS Support feature)
# =============================================================================

variable "enable_cloudfront" {
  description = "Enable CloudFront distributions for HTTPS without custom domain. Uses default *.cloudfront.net certificates."
  type        = bool
  default     = false
}

variable "cloudfront_prefix_list_name" {
  description = "Name of the managed prefix list for ALB ingress (e.g., CloudFront origin-facing IPs). Leave empty to disable prefix list rule. Default is AWS CloudFront prefix list."
  type        = string
  default     = "" # Set to "com.amazonaws.global.cloudfront.origin-facing" when enable_cloudfront=true
}

variable "enable_route53_dns" {
  description = "Enable Route53 DNS records and ACM certificates for custom domain. Set to false when using CloudFront-only deployment."
  type        = bool
  default     = true
}

# =============================================================================
# SECURITY SCANNING CONFIGURATION
# =============================================================================

variable "security_scan_enabled" {
  description = "Enable security scanning for MCP servers"
  type        = bool
  default     = false
}

variable "security_scan_on_registration" {
  description = "Automatically scan servers when they are registered"
  type        = bool
  default     = false
}

variable "security_block_unsafe_servers" {
  description = "Block (disable) servers that fail security scans"
  type        = bool
  default     = false
}

variable "security_analyzers" {
  description = "Analyzers to use for security scanning (comma-separated: yara, llm, api)"
  type        = string
  default     = "yara"
}

variable "security_scan_timeout" {
  description = "Security scan timeout in seconds"
  type        = number
  default     = 60
}

variable "security_add_pending_tag" {
  description = "Add 'security-pending' tag to servers that fail security scan"
  type        = bool
  default     = false
}

# =============================================================================
# MICROSOFT ENTRA ID CONFIGURATION
# =============================================================================

variable "entra_enabled" {
  description = "Enable Microsoft Entra ID as authentication provider"
  type        = bool
  default     = false
}

variable "entra_tenant_id" {
  description = "Azure AD Tenant ID (Directory/tenant ID from Azure Portal)"
  type        = string
  default     = ""
}

variable "entra_client_id" {
  description = "Entra ID Application (client) ID"
  type        = string
  default     = ""
}

variable "entra_client_secret" {
  description = "Entra ID Client Secret (Application secret value)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "idp_group_filter_prefix" {
  description = "Comma-separated list of prefixes to filter IdP groups in IAM > Groups page (e.g., 'mcp-,registry-'). Applies to all identity providers."
  type        = string
  default     = ""
}

# =============================================================================
# OKTA CONFIGURATION
# =============================================================================

variable "okta_enabled" {
  description = "Enable Okta as authentication provider"
  type        = bool
  default     = false
}

variable "okta_domain" {
  description = "Okta domain (e.g., dev-12345678.okta.com or your-org.okta.com)"
  type        = string
  default     = ""
}

variable "okta_client_id" {
  description = "Okta Web Application (client) ID"
  type        = string
  default     = ""
}

variable "okta_client_secret" {
  description = "Okta Client Secret"
  type        = string
  default     = ""
  sensitive   = true
}

variable "okta_m2m_client_id" {
  description = "Okta M2M Client ID (for service account operations)"
  type        = string
  default     = ""
}

variable "okta_m2m_client_secret" {
  description = "Okta M2M Client Secret"
  type        = string
  default     = ""
  sensitive   = true
}

variable "okta_api_token" {
  description = "Okta API Token (for IAM management operations)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "okta_auth_server_id" {
  description = "Okta Custom Authorization Server ID (optional - for M2M tokens)"
  type        = string
  default     = ""
}

# =============================================================================
# AUTH0 CONFIGURATION
# =============================================================================

variable "auth0_enabled" {
  description = "Enable Auth0 as authentication provider"
  type        = bool
  default     = false
}

variable "auth0_domain" {
  description = "Auth0 domain (e.g., your-tenant.us.auth0.com)"
  type        = string
  default     = ""
}

variable "auth0_client_id" {
  description = "Auth0 Web Application (client) ID"
  type        = string
  default     = ""
}

variable "auth0_client_secret" {
  description = "Auth0 Client Secret"
  type        = string
  default     = ""
  sensitive   = true
}

variable "auth0_audience" {
  description = "Auth0 API Audience (optional - for API access tokens)"
  type        = string
  default     = ""
}

variable "auth0_groups_claim" {
  description = "Auth0 custom claim for group memberships (must be namespaced URI)"
  type        = string
  default     = "https://mcp-gateway/groups"
}

variable "auth0_m2m_client_id" {
  description = "Auth0 M2M Client ID (for IAM Management - user/role administration)"
  type        = string
  default     = ""
}

variable "auth0_m2m_client_secret" {
  description = "Auth0 M2M Client Secret"
  type        = string
  default     = ""
  sensitive   = true
}

variable "auth0_management_api_token" {
  description = "Auth0 Management API Token (alternative to M2M credentials)"
  type        = string
  default     = ""
  sensitive   = true
}

# =============================================================================
# OAUTH TOKEN STORAGE CONFIGURATION
# =============================================================================

variable "oauth_store_tokens_in_session" {
  description = "Store OAuth provider tokens in session cookies. Set to false to avoid cookie size limits with large tokens (e.g., Entra ID). Tokens are not used functionally."
  type        = bool
  default     = false
}

# =============================================================================
# REGISTRY STATIC TOKEN AUTH (IdP-independent API access)
# =============================================================================

variable "registry_static_token_auth_enabled" {
  description = "Enable static token auth for Registry API endpoints (/api/*, /v0.1/*). MCP Gateway endpoints still require full IdP authentication."
  type        = bool
  default     = false
}

variable "registry_api_token" {
  description = "Static API key for Registry API. Clients send: Authorization: Bearer <token>. Generate with: python3 -c \"import secrets; print(secrets.token_urlsafe(32))\""
  type        = string
  default     = ""
  sensitive   = true
}

variable "registry_api_keys" {
  description = "JSON string configuring multiple static API keys with per-key group assignments. Example: '{\"monitoring\":{\"key\":\"<token>\",\"groups\":[\"mcp-readonly\"]}}'"
  type        = string
  default     = ""
  sensitive   = true
}

variable "max_tokens_per_user_per_hour" {
  description = "Maximum JWT tokens that can be vended per user per hour."
  type        = number
  default     = 100
}

# =============================================================================
# REGISTRATION WEBHOOK (Issue #742)
# =============================================================================

variable "registration_webhook_url" {
  description = "Webhook URL to POST to on successful registration or deletion. Disabled if empty."
  type        = string
  default     = ""
}

variable "registration_webhook_auth_header" {
  description = "Auth header name for webhook requests (e.g. Authorization, X-API-Key). If Authorization, Bearer is auto-prepended."
  type        = string
  default     = "Authorization"
}

variable "registration_webhook_auth_token" {
  description = "Auth token for webhook requests. Leave empty for unauthenticated webhooks."
  type        = string
  default     = ""
  sensitive   = true
}

variable "registration_webhook_timeout_seconds" {
  description = "Timeout for webhook HTTP calls in seconds."
  type        = number
  default     = 10
}

# =============================================================================
# M2M DIRECT CLIENT REGISTRATION (Issue #851)
# =============================================================================

variable "m2m_direct_registration_enabled" {
  description = "Enable the admin API at /api/iam/m2m-clients that writes M2M client_ids and groups directly to the idp_m2m_clients collection without an IdP Admin API token. Default: true."
  type        = bool
  default     = true
}

# =============================================================================
# FEDERATION CONFIGURATION (Peer-to-Peer Registry Sync)
# =============================================================================

variable "registry_id" {
  description = "Unique identifier for this registry instance in federation. Used to identify the source of synced items."
  type        = string
  default     = ""
}

variable "federation_static_token_auth_enabled" {
  description = "Enable static token auth for Federation API endpoints (/api/federation/*, /api/peers/*). When enabled, peer registries can authenticate using FEDERATION_STATIC_TOKEN."
  type        = bool
  default     = false
}

variable "federation_static_token" {
  description = "Static token for Federation API access. Peer registries use this as Bearer token. Generate with: python3 -c \"import secrets; print(secrets.token_urlsafe(32))\""
  type        = string
  default     = ""
  sensitive   = true
}

variable "federation_encryption_key" {
  description = "Fernet encryption key for storing federation tokens in MongoDB. Required on importing registry. Generate with: python3 -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
  type        = string
  default     = ""
  sensitive   = true
}

# =============================================================================
# AWS AGENT REGISTRY FEDERATION CONFIGURATION
# =============================================================================

variable "aws_registry_federation_enabled" {
  description = "Enable AWS Agent Registry federation."
  type        = bool
  default     = false
}

# =============================================================================
# ANS (AGENT NAMING SERVICE) CONFIGURATION
# =============================================================================

variable "ans_integration_enabled" {
  description = "Enable ANS integration for agent identity verification."
  type        = bool
  default     = false
}

variable "ans_api_endpoint" {
  description = "ANS API endpoint URL."
  type        = string
  default     = "https://api.godaddy.com"
}

variable "ans_api_key" {
  description = "ANS API key for authentication."
  type        = string
  default     = ""
  sensitive   = true
}

variable "ans_api_secret" {
  description = "ANS API secret for authentication."
  type        = string
  default     = ""
  sensitive   = true
}

variable "ans_api_timeout_seconds" {
  description = "ANS API request timeout in seconds."
  type        = number
  default     = 30
}

variable "ans_sync_interval_hours" {
  description = "How often to re-sync ANS verification status (in hours)."
  type        = number
  default     = 6
}

variable "ans_verification_cache_ttl_seconds" {
  description = "Cache TTL for ANS verification results (in seconds)."
  type        = number
  default     = 3600
}

# =============================================================================
# AUDIT LOGGING CONFIGURATION
# =============================================================================

variable "audit_log_enabled" {
  description = "Enable audit logging for all API and MCP requests. Logs are stored in DocumentDB with automatic TTL-based retention."
  type        = bool
  default     = true
}

variable "audit_log_ttl_days" {
  description = "Audit log retention period in days. Logs older than this are automatically deleted via DocumentDB TTL index. Common values: 7 (dev), 30 (standard), 90 (compliance)."
  type        = number
  default     = 7

  validation {
    condition     = var.audit_log_ttl_days >= 1 && var.audit_log_ttl_days <= 365
    error_message = "Audit log TTL must be between 1 and 365 days"
  }
}

# =============================================================================
# REGISTRY CARD CONFIGURATION (Federation Metadata)
# =============================================================================

variable "registry_name" {
  description = "Human-readable registry name for federation and discovery. If not set, a random Docker-style name will be generated."
  type        = string
  default     = ""
}

variable "registry_organization_name" {
  description = "Organization that operates this registry. Defaults to 'ACME Inc.' if not set."
  type        = string
  default     = ""
}

variable "registry_description" {
  description = "Registry description for federation discovery."
  type        = string
  default     = ""
}

variable "registry_contact_email" {
  description = "Contact email for registry administrators. Leave empty if not publicly shared."
  type        = string
  default     = ""
}

variable "registry_contact_url" {
  description = "Documentation or support URL for this registry. Leave empty if not available."
  type        = string
  default     = ""
}

# =============================================================================
# DEPLOYMENT MODE CONFIGURATION
# =============================================================================

variable "deployment_mode" {
  description = <<-EOT
    Controls how the registry integrates with the gateway/nginx.
    - "with-gateway" (default): Full integration with nginx reverse proxy.
      Nginx config is regenerated when servers are registered/deleted.
      Frontend shows gateway authentication instructions.
    - "registry-only": Registry operates as catalog/discovery service only.
      Nginx config is NOT updated on server changes.
      Frontend shows direct connection mode (proxy_pass_url).
      Use when registry is separate from gateway infrastructure.
  EOT
  type        = string
  default     = "with-gateway"

  validation {
    condition     = contains(["with-gateway", "registry-only"], var.deployment_mode)
    error_message = "deployment_mode must be either 'with-gateway' or 'registry-only'"
  }
}

variable "registry_mode" {
  description = <<-EOT
    Controls which features are enabled (informational - for UI feature flags).
    This setting affects the /api/config response which the frontend can use
    to show/hide navigation elements. Currently informational only - all APIs remain active.
    - "full" (default): All features enabled (mcp_servers, agents, skills, federation)
    - "skills-only": Only skills feature flag enabled
    - "mcp-servers-only": Only MCP server feature flag enabled
    - "agents-only": Only A2A agent feature flag enabled
    Note: with-gateway + skills-only is invalid and auto-corrects to registry-only + skills-only
  EOT
  type        = string
  default     = "full"

  validation {
    condition     = contains(["full", "skills-only", "mcp-servers-only", "agents-only"], var.registry_mode)
    error_message = "registry_mode must be one of: 'full', 'skills-only', 'mcp-servers-only', 'agents-only'"
  }
}

variable "show_servers_tab" {
  description = "Show the MCP Servers tab in the UI. AND-ed with registry_mode."
  type        = bool
  default     = true
}

variable "show_virtual_servers_tab" {
  description = "Show the Virtual MCP Servers tab in the UI."
  type        = bool
  default     = true
}

variable "show_skills_tab" {
  description = "Show the Skills tab in the UI. AND-ed with registry_mode."
  type        = bool
  default     = true
}

variable "show_agents_tab" {
  description = "Show the Agents tab in the UI. AND-ed with registry_mode."
  type        = bool
  default     = true
}

# =============================================================================
# OBSERVABILITY CONFIGURATION (Metrics Pipeline)
# =============================================================================

variable "enable_observability" {
  description = "Enable full observability pipeline (AMP, metrics-service, ADOT collector, Grafana). When false, no observability resources are created."
  type        = bool
  default     = true
}

variable "metrics_service_image_uri" {
  description = "Container image URI for metrics-service. Required when enable_observability is true."
  type        = string
  default     = ""
}

variable "grafana_image_uri" {
  description = "Container image URI for Grafana OSS (custom image with baked-in provisioning). Required when enable_observability is true."
  type        = string
  default     = ""
}

variable "grafana_admin_password" {
  description = "Admin password for Grafana. Must be set when enable_observability is true."
  type        = string
  sensitive   = true
  default     = ""
}

variable "otel_otlp_endpoint" {
  description = "OTLP endpoint for pushing metrics to an external platform (e.g., Datadog). Leave empty to disable."
  type        = string
  default     = ""
}

variable "otel_exporter_otlp_headers" {
  description = "Headers for OTLP exporter (e.g., 'dd-api-key=YOUR_KEY' for Datadog). Stored in Secrets Manager. Leave empty if not needed."
  type        = string
  sensitive   = true
  default     = ""
}

variable "otel_otlp_export_interval_ms" {
  description = "OTLP export interval in milliseconds. Default 30000 (30 seconds)."
  type        = number
  default     = 30000
}

variable "otel_exporter_otlp_metrics_temporality_preference" {
  description = "OTLP metrics temporality preference. Datadog requires delta. Default cumulative."
  type        = string
  default     = "cumulative"
}

# =============================================================================
# TELEMETRY CONFIGURATION (Issue #559)
# =============================================================================

variable "mcp_telemetry_disabled" {
  description = "Disable anonymous startup telemetry. Set to '1' to opt out."
  type        = string
  default     = ""
}

variable "mcp_telemetry_opt_out" {
  description = "Disable daily heartbeat telemetry only. Set to '1' to opt out (startup ping still sent)."
  type        = string
  default     = ""
}

variable "mcp_telemetry_heartbeat_interval_minutes" {
  description = "Heartbeat telemetry interval in minutes. Default: 1440 (24 hours)."
  type        = string
  default     = "1440"
}

variable "telemetry_debug" {
  description = "Enable telemetry debug mode (logs payload instead of sending). Set to 'true' to enable."
  type        = string
  default     = "false"
}

variable "disable_ai_registry_tools_server" {
  description = "Disable auto-registration of the built-in airegistry-tools server on startup. Set to 'true' for GitOps/production deployments."
  type        = string
  default     = "false"
}

# =============================================================================
# GITHUB PRIVATE REPO AUTH (Issue #814)
# =============================================================================

variable "github_pat" {
  description = "GitHub Personal Access Token for private repo SKILL.md access. Generate at https://github.com/settings/tokens with 'repo' scope."
  type        = string
  default     = ""
  sensitive   = true
}

variable "github_app_id" {
  description = "GitHub App ID for installation-based auth."
  type        = string
  default     = ""
}

variable "github_app_installation_id" {
  description = "GitHub App Installation ID."
  type        = string
  default     = ""
}

variable "github_app_private_key" {
  description = "GitHub App private key (PEM format). Newlines should be encoded as literal \\n."
  type        = string
  default     = ""
  sensitive   = true
}

variable "github_extra_hosts" {
  description = "Comma-separated extra GitHub hosts for enterprise instances (e.g. github.mycompany.com,raw.github.mycompany.com)."
  type        = string
  default     = ""
}

variable "github_api_base_url" {
  description = "GitHub API base URL. For GitHub Enterprise Server use https://<hostname>/api/v3."
  type        = string
  default     = "https://api.github.com"
}

# =============================================================================
# WAF CONFIGURATION (Issue #603 Security Hardening)
# =============================================================================

variable "enable_waf" {
  description = "Enable WAFv2 Web ACLs for ALBs. Requires wafv2:* IAM permissions. Set to false if IAM permissions are not available."
  type        = bool
  default     = false
}
