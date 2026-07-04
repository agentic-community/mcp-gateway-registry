# MCP Gateway Registry Module Variables

# Required Variables - Shared Resources
variable "name" {
  description = "Name prefix for MCP Gateway Registry resources"
  type        = string
}

variable "vpc_id" {
  description = "ID of the VPC where resources will be created"
  type        = string
}

variable "private_subnet_ids" {
  description = "List of private subnet IDs for ECS services"
  type        = list(string)
}

variable "public_subnet_ids" {
  description = "List of public subnet IDs for ALB"
  type        = list(string)
}

variable "ecs_cluster_arn" {
  description = "ARN of the existing ECS cluster"
  type        = string
}

variable "ecs_cluster_name" {
  description = "Name of the existing ECS cluster"
  type        = string
}

variable "task_execution_role_arn" {
  description = "ARN of the task execution IAM role (DEPRECATED: Module now creates its own task execution roles)"
  type        = string
  default     = ""
}

# Container Image URIs (pre-built images from public ECR)
variable "registry_image_uri" {
  description = "Container image URI for registry service (defaults to pre-built image from public ECR)"
  type        = string
  default     = "public.ecr.aws/p3v1o3c6/registry:1.25.0"
}

variable "auth_server_image_uri" {
  description = "Container image URI for auth server service (defaults to pre-built image from public ECR)"
  type        = string
  default     = "public.ecr.aws/p3v1o3c6/auth-server:1.25.0"
}

variable "mcpgw_image_uri" {
  description = "Container image URI for mcpgw service (defaults to pre-built image from public ECR)"
  type        = string
  default     = "public.ecr.aws/p3v1o3c6/mcpgw:1.25.0"
}

variable "enable_demo_servers" {
  description = "Deploy demo MCP servers and A2A agents (currenttime, realserverfaketools, flight-booking-agent, travel-assistant-agent). Requires setting the corresponding image URIs."
  type        = bool
  default     = false
}

variable "currenttime_image_uri" {
  description = "Container image URI for currenttime MCP server (only used when enable_demo_servers is true)"
  type        = string
  default     = ""
}

variable "realserverfaketools_image_uri" {
  description = "Container image URI for realserverfaketools MCP server (only used when enable_demo_servers is true)"
  type        = string
  default     = ""
}

variable "flight_booking_agent_image_uri" {
  description = "Container image URI for flight booking A2A agent (only used when enable_demo_servers is true)"
  type        = string
  default     = ""
}

variable "travel_assistant_agent_image_uri" {
  description = "Container image URI for travel assistant A2A agent (only used when enable_demo_servers is true)"
  type        = string
  default     = ""
}

variable "dockerhub_org" {
  description = "DEPRECATED: Docker Hub organization. No longer used; images default to public ECR."
  type        = string
  default     = "mcpgateway"
}


# Resource Configuration
variable "cpu" {
  description = "CPU allocation for MCP Gateway Registry containers (in vCPU units: 256, 512, 1024, 2048, 4096)"
  type        = string
  default     = "1024"
  validation {
    condition     = contains(["256", "512", "1024", "2048", "4096"], var.cpu)
    error_message = "CPU must be one of: 256, 512, 1024, 2048, 4096"
  }
}

variable "memory" {
  description = "Memory allocation for MCP Gateway Registry containers (in MB, must be compatible with CPU)"
  type        = string
  default     = "2048"
}

variable "registry_replicas" {
  description = "Number of replicas for MCP Gateway Registry main service"
  type        = number
  default     = 1
  validation {
    condition     = var.registry_replicas > 0
    error_message = "Registry replicas must be greater than 0."
  }
}

variable "auth_replicas" {
  description = "Number of replicas for MCP Gateway Auth service"
  type        = number
  default     = 1
  validation {
    condition     = var.auth_replicas > 0
    error_message = "Auth replicas must be greater than 0."
  }
}

variable "currenttime_replicas" {
  description = "Number of replicas for CurrentTime MCP server (only used when enable_demo_servers is true)"
  type        = number
  default     = 1
  validation {
    condition     = var.currenttime_replicas >= 0
    error_message = "CurrentTime replicas must be 0 or greater."
  }
}

variable "mcpgw_replicas" {
  description = "Number of replicas for MCPGW MCP server"
  type        = number
  default     = 1
  validation {
    condition     = var.mcpgw_replicas > 0
    error_message = "MCPGW replicas must be greater than 0."
  }
}

variable "realserverfaketools_replicas" {
  description = "Number of replicas for RealServerFakeTools MCP server (only used when enable_demo_servers is true)"
  type        = number
  default     = 1
  validation {
    condition     = var.realserverfaketools_replicas >= 0
    error_message = "RealServerFakeTools replicas must be 0 or greater."
  }
}

variable "flight_booking_agent_replicas" {
  description = "Number of replicas for Flight Booking A2A agent (only used when enable_demo_servers is true)"
  type        = number
  default     = 1
  validation {
    condition     = var.flight_booking_agent_replicas >= 0
    error_message = "Flight Booking agent replicas must be 0 or greater."
  }
}

variable "travel_assistant_agent_replicas" {
  description = "Number of replicas for Travel Assistant A2A agent (only used when enable_demo_servers is true)"
  type        = number
  default     = 1
  validation {
    condition     = var.travel_assistant_agent_replicas >= 0
    error_message = "Travel Assistant agent replicas must be 0 or greater."
  }
}

# ALB Configuration
variable "alb_scheme" {
  description = "Scheme for the ALB (internal or internet-facing)"
  type        = string
  default     = "internet-facing"
  validation {
    condition     = contains(["internal", "internet-facing"], var.alb_scheme)
    error_message = "ALB scheme must be either 'internal' or 'internet-facing'."
  }
}

variable "alb_logs_bucket" {
  description = "S3 bucket for ALB access logs"
  type        = string
}

variable "ingress_cidr_blocks" {
  description = "List of CIDR blocks allowed to access the ALB (main ALB + auth server + registry)"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "certificate_arn" {
  description = "ARN of ACM certificate for HTTPS (optional)"
  type        = string
  default     = ""
}

variable "keycloak_domain" {
  description = "Domain name for Keycloak (e.g., kc.mycorp.click)"
  type        = string
  default     = ""
}

variable "enable_autoscaling" {
  description = "Whether to enable auto-scaling for ECS services"
  type        = bool
  default     = true
}

variable "autoscaling_min_capacity" {
  description = "Minimum number of tasks for auto-scaling"
  type        = number
  default     = 2
}

variable "autoscaling_max_capacity" {
  description = "Maximum number of tasks for auto-scaling"
  type        = number
  default     = 4
}

variable "autoscaling_target_cpu" {
  description = "Target CPU utilization percentage for auto-scaling"
  type        = number
  default     = 70
}

variable "autoscaling_target_memory" {
  description = "Target memory utilization percentage for auto-scaling"
  type        = number
  default     = 80
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

# EFS Configuration
variable "efs_throughput_mode" {
  description = "Throughput mode for EFS (bursting or provisioned)"
  type        = string
  default     = "bursting"
  validation {
    condition     = contains(["bursting", "provisioned"], var.efs_throughput_mode)
    error_message = "EFS throughput mode must be either 'bursting' or 'provisioned'."
  }
}

variable "efs_provisioned_throughput" {
  description = "Provisioned throughput in MiB/s for EFS (only used if throughput_mode is provisioned)"
  type        = number
  default     = 100
}

variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}


# Domain Configuration (Optional)
variable "domain_name" {
  description = "Domain name for the MCP Gateway Registry (optional)"
  type        = string
  default     = ""
}

variable "auth_server_url" {
  description = "Internal URL the registry/nginx use to reach the auth-server. Set to a Cloud Map / Service Connect FQDN (e.g. http://auth-server.<namespace>.local:8888) for deployments where only FQDNs resolve. Defaults to the Compose-style service name for backward compatibility."
  type        = string
  default     = "http://auth-server:8888"
}

variable "create_route53_record" {
  description = "Whether to create Route53 DNS record for the domain"
  type        = bool
  default     = false
}

variable "route53_zone_id" {
  description = "Route53 hosted zone ID (required if create_route53_record is true)"
  type        = string
  default     = ""
}


# Embeddings Configuration
variable "embeddings_provider" {
  description = "Embeddings provider: 'sentence-transformers' for local models or 'litellm' for API-based models"
  type        = string
  default     = "sentence-transformers"
  validation {
    condition     = contains(["sentence-transformers", "litellm"], var.embeddings_provider)
    error_message = "Embeddings provider must be either 'sentence-transformers' or 'litellm'."
  }
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
  validation {
    condition     = var.embeddings_model_dimensions > 0
    error_message = "Embeddings model dimensions must be greater than 0."
  }
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


# Registration Deduplication. Advisory only; reuses the embeddings
# model above. The /api/<entity>/check-duplicates endpoints are always
# available; the hint flag only governs whether the registration UI
# pre-flights the check. The check never blocks registration.
variable "dedup_registration_hint_enabled" {
  description = "When true, registration UI pre-flights /check-duplicates and shows a hint modal. Endpoints remain available regardless."
  type        = bool
  default     = true
}

variable "dedup_score_threshold" {
  description = "Minimum similarity score (0.0..1.0) for an advisory match. Default 0.7."
  type        = number
  default     = 0.7
  validation {
    condition     = var.dedup_score_threshold >= 0.0 && var.dedup_score_threshold <= 1.0
    error_message = "dedup_score_threshold must be between 0.0 and 1.0."
  }
}

variable "dedup_max_suggestions" {
  description = "Cap on duplicate suggestions returned per request. Default 3."
  type        = number
  default     = 3
  validation {
    condition     = var.dedup_max_suggestions >= 1 && var.dedup_max_suggestions <= 10
    error_message = "dedup_max_suggestions must be between 1 and 10."
  }
}


# Keycloak Admin Credentials (for Management API)
variable "keycloak_admin_password" {
  description = "Keycloak admin password for Management API user/group operations"
  type        = string
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

variable "cors_allowed_origins" {
  description = "Comma-separated exact browser origins allowed to make credentialed cross-origin requests to the registry API (e.g. 'https://app.example.com,https://admin.example.com'). The registry's own origin is always trusted. Empty means same-origin only; there is no wildcard fallback."
  type        = string
  default     = ""
}

variable "bind_host" {
  description = "Network bind address for registry and gateway services. Default '0.0.0.0' (IPv4) works on all hosts. Set to '::' only for IPv6-only deployments (requires net.ipv6.bindv6only=0 on the host)."
  type        = string
  default     = "0.0.0.0"
}

# Security Scanning Configuration
variable "security_scan_enabled" {
  description = "Enable/disable security scanning for MCP servers during registration"
  type        = bool
  default     = true
}

variable "security_scan_on_registration" {
  description = "Automatically scan servers when they are registered"
  type        = bool
  default     = true
}

variable "security_block_unsafe_servers" {
  description = "Block (disable) servers that fail security scans"
  type        = bool
  default     = true
}

variable "security_analyzers" {
  description = "Comma-separated list of analyzers to use for security scanning (available: yara, llm, api)"
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
  default     = true
}

# =============================================================================
# DOCUMENTDB CONFIGURATION (from upstream v1.0.9)
# =============================================================================

variable "storage_backend" {
  description = <<-DESC
    Storage backend. Accepted values (mirrors root variables.tf and
    registry/core/config.py ALLOWED_STORAGE_BACKENDS): file, documentdb,
    mongodb-ce, mongodb, mongodb-atlas. mongodb and mongodb-atlas are
    aliases for mongodb-ce at the Python repository layer.
  DESC
  type        = string
  default     = "file"
  validation {
    condition = contains(
      ["file", "documentdb", "mongodb-ce", "mongodb", "mongodb-atlas"],
      var.storage_backend,
    )
    error_message = "Storage backend must be one of: file, documentdb, mongodb-ce, mongodb, mongodb-atlas."
  }
}

variable "documentdb_endpoint" {
  description = "DocumentDB cluster endpoint (required when storage_backend is 'documentdb')"
  type        = string
  default     = ""
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

variable "documentdb_credentials_secret_arn" {
  description = "ARN of the Secrets Manager secret containing DocumentDB credentials"
  type        = string
  default     = ""
}

# PR #947: Optional full MongoDB connection string override. When set, takes
# precedence over the documentdb_* variables above. Use for MongoDB Atlas
# (mongodb+srv://), replica sets, or URI-level tuning not expressible via the
# discrete variables. Prefer mongodb_connection_string_secret_arn when the
# URI contains credentials to avoid storing secrets in Terraform state.
variable "mongodb_connection_string" {
  description = "Optional full MongoDB connection string override (plain text). Takes precedence over documentdb_* variables. Leave empty to use documentdb_* variables."
  type        = string
  default     = ""
  sensitive   = true
}

variable "mongodb_connection_string_secret_arn" {
  description = "Optional Secrets Manager ARN for the full MongoDB connection string. Preferred over mongodb_connection_string when the URI contains credentials."
  type        = string
  default     = ""
}

# =============================================================================
# CLOUDFRONT CONFIGURATION (CloudFront HTTPS Support feature)
# =============================================================================

variable "enable_cloudfront" {
  description = "Whether CloudFront is enabled (adds CloudFront prefix list to ALB security group)"
  type        = bool
  default     = false
}

variable "cloudfront_prefix_list_name" {
  description = "Name of the managed prefix list for CloudFront origin-facing IPs"
  type        = string
  default     = "com.amazonaws.global.cloudfront.origin-facing"
}

variable "additional_server_names" {
  description = "Additional server names for nginx (space-separated). Used in dual-mode to accept both CloudFront and custom domain requests."
  type        = string
  default     = ""
}


# HTTPS Configuration
variable "enable_https" {
  description = "Whether to enable HTTPS listener on ALB. Set to true when certificate_arn is provided."
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

variable "entra_login_base_url" {
  description = "Entra ID login base URL. Override only for sovereign clouds. Empty default uses public cloud."
  type        = string
  default     = ""
}

variable "entra_graph_base_url" {
  description = "Microsoft Graph base URL override. Leave empty on standard deployments — auto-inferred from entra_login_base_url."
  type        = string
  default     = ""
}

variable "idp_group_filter_prefix" {
  description = "Comma-separated list of prefixes to filter IdP groups in IAM > Groups page (e.g., 'mcp-,registry-'). Applies to all identity providers."
  type        = string
  default     = ""
}

variable "allowed_idp_groups" {
  description = "Comma-separated EXACT IdP group names/IDs to keep in a user's session at login. Empty means auto-derive from scope mappings (recommended). Applies to all identity providers."
  type        = string
  default     = ""
}

variable "idp_user_group_fallback_enabled_providers" {
  description = "Comma-separated list of IdP providers (e.g. pingfederate) for which the registry's local idp_user_groups collection is consulted to populate empty JWT groups claims. Empty list disables the fallback for all providers. Default: pingfederate."
  type        = string
  default     = "pingfederate"
}

# =============================================================================
# AMAZON COGNITO CONFIGURATION
# =============================================================================

variable "cognito_enabled" {
  description = "Enable Amazon Cognito as the authentication provider"
  type        = bool
  default     = false
}

variable "cognito_user_pool_id" {
  description = "Cognito User Pool ID (e.g. us-east-1_XXXXXXXXX)"
  type        = string
  default     = ""
}

variable "cognito_client_id" {
  description = "Cognito App Client ID for web login"
  type        = string
  default     = ""
}

variable "cognito_client_secret" {
  description = "Cognito App Client secret for web login"
  type        = string
  default     = ""
  sensitive   = true
}

variable "cognito_domain" {
  description = "Optional Cognito hosted UI domain prefix or custom domain. Leave empty to derive it from the User Pool ID."
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
  description = "Okta domain (e.g., your-org.okta.com)"
  type        = string
  default     = ""
}

variable "okta_client_id" {
  description = "Okta Application (client) ID"
  type        = string
  default     = ""
}

variable "okta_client_secret" {
  description = "Okta Client Secret (Application secret value)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "okta_m2m_client_id" {
  description = "Okta M2M client ID for service account operations"
  type        = string
  default     = ""
}

variable "okta_m2m_client_secret" {
  description = "Okta M2M client secret for service account operations"
  type        = string
  default     = ""
  sensitive   = true
}

variable "okta_api_token" {
  description = "Okta API token for management operations"
  type        = string
  default     = ""
  sensitive   = true
}

variable "okta_auth_server_id" {
  description = "Okta Custom Authorization Server ID (for M2M tokens). Leave empty to use default Org Authorization Server."
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
  description = "Auth0 Application (client) ID"
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
  description = "Auth0 API audience for M2M tokens"
  type        = string
  default     = ""
}

variable "auth0_groups_claim" {
  description = "Custom namespaced claim for groups in Auth0 tokens"
  type        = string
  default     = "https://mcp-gateway/groups"
}

variable "auth0_m2m_client_id" {
  description = "Auth0 M2M client ID for IAM Management operations"
  type        = string
  default     = ""
}

variable "auth0_m2m_client_secret" {
  description = "Auth0 M2M client secret for IAM Management operations"
  type        = string
  default     = ""
  sensitive   = true
}

variable "auth0_management_api_token" {
  description = "Auth0 Management API token (alternative to M2M credentials, expires after 24h)"
  type        = string
  default     = ""
  sensitive   = true
}

# =============================================================================
# PINGFEDERATE CONFIGURATION
# =============================================================================

variable "pingfederate_enabled" {
  description = "Enable PingFederate as authentication provider"
  type        = bool
  default     = false
}

variable "pingfederate_base_url" {
  description = "PingFederate runtime base URL (internal, server-to-server), e.g. https://pf.example.com:9031"
  type        = string
  default     = ""
}

variable "pingfederate_external_url" {
  description = "PingFederate external URL (browser-facing, for auth redirects)"
  type        = string
  default     = ""
}

variable "pingfederate_client_id" {
  description = "PingFederate OAuth client ID for the gateway web app"
  type        = string
  default     = ""
}

variable "pingfederate_client_secret" {
  description = "PingFederate OAuth client secret"
  type        = string
  default     = ""
  sensitive   = true
}

variable "pingfederate_m2m_client_id" {
  description = "PingFederate M2M client ID (defaults to web client if empty)"
  type        = string
  default     = ""
}

variable "pingfederate_m2m_client_secret" {
  description = "PingFederate M2M client secret (defaults to web client secret if empty)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "pingfederate_application_id_uri" {
  description = "Optional resource-server identifier accepted as the JWT aud claim"
  type        = string
  default     = ""
}

variable "pingfederate_groups_claim" {
  description = "JWT claim name carrying group memberships (default: groups)"
  type        = string
  default     = "groups"
}

# -----------------------------------------------------------------------------
# PINGFEDERATE ADMIN API (registry only)
# -----------------------------------------------------------------------------

variable "pf_admin_url" {
  description = "PingFederate admin API URL (used by registry to create OAuth clients and PCV users)"
  type        = string
  default     = "https://pingfederate:9999"
}

variable "pf_admin_user" {
  description = "PingFederate admin API username"
  type        = string
  default     = "administrator"
}

variable "pf_admin_pass" {
  description = "PingFederate admin API password (sensitive). Wired through AWS Secrets Manager in production. No default: supply a strong value when pingfederate_enabled is true."
  type        = string
  default     = ""
  sensitive   = true

  validation {
    condition     = var.pf_admin_pass != "2FederateM0re"
    error_message = "pf_admin_pass must not be the well-known development default. Set a strong, unique PingFederate admin password."
  }
}

variable "registry_static_token_auth_enabled" {
  description = "Enable static token auth for Registry API (IdP-independent access using REGISTRY_API_TOKEN)"
  type        = bool
  default     = false
}

variable "registry_api_token" {
  description = "Static API key for network-trusted mode. Must match the Bearer token value sent by clients."
  type        = string
  default     = ""
  sensitive   = true
}

variable "registry_api_keys" {
  description = "JSON string configuring multiple static API keys with per-key group assignments."
  type        = string
  default     = ""
  sensitive   = true
}

variable "max_tokens_per_user_per_hour" {
  description = "Maximum JWT tokens that can be vended per user per hour."
  type        = number
  default     = 100
}

# Registration webhook (issue #742)
variable "registration_webhook_url" {
  description = "Webhook URL to POST to on successful registration or deletion. Disabled if empty."
  type        = string
  default     = ""
}

variable "registration_webhook_auth_header" {
  description = "Auth header name for webhook requests."
  type        = string
  default     = "Authorization"
}

variable "registration_webhook_auth_token" {
  description = "Auth token for webhook requests."
  type        = string
  default     = ""
  sensitive   = true
}

variable "registration_webhook_timeout_seconds" {
  description = "Timeout for webhook HTTP calls in seconds."
  type        = number
  default     = 10
}

variable "registration_webhook_signing_secret" {
  description = "Shared secret for HMAC-SHA256 signing of outbound webhook payloads."
  type        = string
  default     = ""
  sensitive   = true
}

variable "registration_enforced_status" {
  description = "Mandated initial lifecycle status for new registrations (e.g. 'draft'). Empty = default 'active'."
  type        = string
  default     = ""
}

# Agent batch API (issue #956)
variable "batch_worker_enabled" {
  description = "Enable the in-process agent batch worker loop. v1 single-worker constraint."
  type        = bool
  default     = true
}

variable "batch_max_operations_per_job" {
  description = "Maximum number of items allowed in a single agent batch submission."
  type        = number
  default     = 1000
}

variable "batch_max_concurrent_jobs_per_user" {
  description = "Maximum number of active batch jobs per submitter."
  type        = number
  default     = 3
}

variable "batch_job_retention_days" {
  description = "Retention window for agent batch jobs in MongoDB (TTL on updated_at)."
  type        = number
  default     = 7
}

variable "batch_worker_poll_interval_seconds" {
  description = "How often the batch worker polls MongoDB for queued jobs."
  type        = number
  default     = 1.0
}

variable "batch_max_request_bytes" {
  description = "Maximum request body size (bytes) accepted by POST /api/agents/batch."
  type        = number
  default     = 4194304
}

variable "batch_worker_lease_ttl_seconds" {
  description = "How long a claimed batch job stays owned before its lease expires and another worker may reclaim it."
  type        = number
  default     = 60
}

variable "batch_worker_lease_heartbeat_seconds" {
  description = "Interval at which a worker renews the lease on its in-flight job. Should be below batch_worker_lease_ttl_seconds."
  type        = number
  default     = 15
}

# Registration gate / admission control (issue #809)
variable "registration_gate_enabled" {
  description = "Enable registration gate (admission control). Default: false."
  type        = bool
  default     = false
}

variable "registration_gate_url" {
  description = "URL of the registration gate endpoint."
  type        = string
  default     = ""
}

variable "registration_gate_auth_type" {
  description = "Auth type for gate: none, api_key, or bearer."
  type        = string
  default     = "none"
}

variable "registration_gate_auth_credential" {
  description = "Auth credential for the gate endpoint."
  type        = string
  default     = ""
  sensitive   = true
}

variable "registration_gate_auth_header_name" {
  description = "Header name when auth_type=api_key."
  type        = string
  default     = "X-Api-Key"
}

variable "registration_gate_timeout_seconds" {
  description = "HTTP timeout per gate attempt in seconds."
  type        = number
  default     = 5
}

variable "registration_gate_max_retries" {
  description = "Retries after first gate attempt."
  type        = number
  default     = 2
}

variable "registration_gate_oauth2_token_url" {
  description = "OAuth2 token endpoint URL for gate client credentials flow."
  type        = string
  default     = ""
}

variable "registration_gate_oauth2_client_id" {
  description = "OAuth2 client ID for gate client credentials flow."
  type        = string
  default     = ""
}

variable "registration_gate_oauth2_client_secret" {
  description = "OAuth2 client secret for gate client credentials flow."
  type        = string
  default     = ""
  sensitive   = true
}

variable "registration_gate_oauth2_scope" {
  description = "OAuth2 scope parameter for gate client credentials flow."
  type        = string
  default     = ""
}

variable "m2m_direct_registration_enabled" {
  description = "Enable the admin API at /api/iam/m2m-clients for direct M2M client registration (issue #851). Default: true."
  type        = bool
  default     = true
}

# =============================================================================
# FEDERATION CONFIGURATION (Peer-to-Peer Registry Sync)
# =============================================================================

variable "registry_id" {
  description = "Unique identifier for this registry instance in federation."
  type        = string
  default     = ""
}

variable "federation_static_token_auth_enabled" {
  description = "Enable static token auth for Federation API endpoints."
  type        = bool
  default     = false
}

variable "federation_static_token" {
  description = "Static token for Federation API access."
  type        = string
  default     = ""
  sensitive   = true
}

variable "federation_encryption_key" {
  description = "Fernet encryption key for storing federation tokens in MongoDB."
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
# AUDIT LOGGING CONFIGURATION
# =============================================================================

variable "audit_log_enabled" {
  description = "Enable audit logging for all API and MCP requests."
  type        = bool
  default     = true
}

variable "audit_log_ttl_days" {
  description = "Audit log retention period in days."
  type        = number
  default     = 7
}

# =============================================================================
# APPLICATION LOG CONFIGURATION
# =============================================================================

variable "app_log_centralized_enabled" {
  description = "Write application logs to a centralized store for cross-pod retrieval."
  type        = bool
  default     = true
}

variable "app_log_centralized_ttl_days" {
  description = "Days to retain centralized application logs (TTL index)."
  type        = number
  default     = 1
}

variable "app_log_level" {
  description = "Application log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)."
  type        = string
  default     = "INFO"
}

variable "app_log_excluded_loggers" {
  description = "Comma-separated logger names to exclude from MongoDB log writes."
  type        = string
  default     = "uvicorn.access,httpx,pymongo,motor"
}

variable "app_log_dir" {
  description = "Directory where service log files are written (issue #987). Empty string means use the backend default (/var/log/containers/ai-registry)."
  type        = string
  default     = ""
}

variable "app_log_file_format" {
  description = "On-disk format for service .log files: 'json' (default) or 'text' (legacy). Console format is unaffected (issue #987)."
  type        = string
  default     = "json"
}

variable "app_log_console_format" {
  description = "STDOUT/console format: 'json' (default, JSONL stdout for log-agent scraping) or 'text' (human-readable)."
  type        = string
  default     = "json"
}

# =============================================================================
# TOOL-LEVEL ACCESS CONTROL (Issue #1026)
# =============================================================================

variable "mcp_tools_list_filter_enabled" {
  description = "Enable filtering of MCP tools/list JSON-RPC responses against the per-user tool allowlist. Set to false to revert to pre-fix behavior on the MCP protocol path only. REST endpoints always filter regardless of this flag."
  type        = bool
  default     = true
}

variable "mcp_proxy_max_body_bytes" {
  description = "Upper bound on a tools/list upstream response body (in bytes) that the auth-server proxy hop will buffer for filtering. Responses exceeding this return HTTP 413. Default 2097152 (2 MiB)."
  type        = number
  default     = 2097152
}

variable "mcp_proxy_timeout" {
  description = "Timeout (seconds) for the auth-server proxy hop's upstream MCP request. Raise for servers with long-running tools. Minimum 1. Default 30. Values above 60 also require raising proxy_read_timeout on the generated /mcp-proxy/ nginx blocks (they inherit nginx's 60s default)."
  type        = number
  default     = 30
}

variable "tool_filter_audit_log_level" {
  description = "Log level for tool-pruning audit lines during the launch window. Valid values: DEBUG, INFO, WARNING."
  type        = string
  default     = "INFO"
}

variable "internal_token_ttl_seconds" {
  description = "Lifetime (seconds) of the /validate-minted /mcp-proxy internal token; the replay-window cap. Minimum 5. Default 30."
  type        = number
  default     = 30
}

variable "internal_token_leeway_seconds" {
  description = "Clock-skew leeway (seconds) on the /mcp-proxy internal token exp/iat checks. Default 5."
  type        = number
  default     = 5
}

variable "custom_entity_types_enabled" {
  description = "Main switch for the custom-entity-types feature (dynamic tabs + endpoints). Off by default = no behavior change for existing deployments."
  type        = bool
  default     = false
}

variable "custom_type_cache_ttl_seconds" {
  description = "TTL (seconds) for the in-process custom-type descriptor cache used by the config tab list and default search scope."
  type        = number
  default     = 60
}

variable "max_custom_records_per_type" {
  description = "Soft cap on records per custom type (0 = unlimited). When non-zero, record creation is rejected with HTTP 409 once a type reaches the cap. Best-effort (concurrent creates may overshoot slightly)."
  type        = number
  default     = 1000
}

variable "max_custom_types" {
  description = "Cap on the number of custom entity types an admin can define (0 = unlimited). When non-zero, type creation is rejected with HTTP 409 once the limit is reached."
  type        = number
  default     = 50
}

variable "update_check_enabled" {
  description = "Enable background polling of the GitHub Releases API to surface newer registry versions in an admin-only banner. Fail-silent and air-gap safe. Set false for air-gapped deployments or to silence the banner."
  type        = bool
  default     = true
}

variable "update_check_interval_hours" {
  description = "Polling interval in hours for the update-check background task."
  type        = number
  default     = 24
}

variable "mcp_advertised_scopes" {
  description = <<-EOT
    Space-separated override for the `scopes_supported` array in the gateway's
    /.well-known/oauth-protected-resource document. Required when the IdP's
    RFC 7591 DCR rejects scopes that don't exist as client-scope objects in
    the realm. Default ("profile email offline_access") is the safe set of
    OIDC scopes that all major IdPs ship with by default. Set to "" to fall
    back to scope names from the registry's scopes config (which advertises
    DocumentDB group names — Keycloak / Auth0 / Okta will reject those
    during DCR unless they are also defined as client-scopes in the realm).
  EOT
  type        = string
  default     = "profile email offline_access"
}

variable "ide_oauth_client_id" {
  description = <<-EOT
    Pre-registered PUBLIC OAuth client_id that IDEs (Cursor, Claude Code, Codex)
    use to start the gateway login flow. When set, a server's Connect config
    advertises this client_id and omits the static gateway token, so the IDE
    shows a login button and runs the OAuth/PKCE flow. Use when anonymous
    Dynamic Client Registration is disabled and a fixed public client is
    registered instead. Empty (default) keeps the static-token Connect config.
    This is a public client identifier, NOT a secret.
  EOT
  type        = string
  default     = ""
}

variable "ide_oauth_callback_port" {
  description = <<-EOT
    Fixed loopback callback port the IDE uses for the OAuth login redirect
    (http://localhost:<port>/callback). Needed for IdPs that match the
    redirect_uri literally including the port (Okta, Entra, Cognito): register
    http://localhost:<port>/callback on the public client and set the same value
    here so the Connect dialog emits --callback-port. 0 (default) lets the IDE
    pick a port, which is correct for Keycloak (wildcard loopback redirect).
  EOT
  type        = number
  default     = 0
}

# =============================================================================
# DEPLOYMENT MODE CONFIGURATION
# =============================================================================

variable "deployment_mode" {
  description = "Controls how the registry integrates with the gateway/nginx. 'with-gateway' for full integration, 'registry-only' for catalog-only mode."
  type        = string
  default     = "with-gateway"
}

variable "registry_mode" {
  description = "Controls which features are enabled (informational - for UI feature flags). Options: 'full', 'skills-only', 'mcp-servers-only', 'agents-only'."
  type        = string
  default     = "full"
}

variable "internal_only_deployment" {
  description = "Marks an internal/workshop deployment (telemetry label; issue #1216). Does not change access control."
  type        = bool
  default     = false
}

variable "internal_deployment_type" {
  description = "Internal deployment classification: 'none', 'dev', 'workshop', or 'other' (telemetry label; issue #1216)."
  type        = string
  default     = "none"
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

variable "ui_title" {
  description = "Override for the UI title. Empty string defers to the deployment-mode default ('AI Gateway & Registry' for with-gateway, 'AI Registry' for registry-only)."
  type        = string
  default     = ""
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

# Telemetry configuration
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

variable "mcp_telemetry_imds_probe_disabled" {
  description = "Disable IMDS probing in cloud detection (issue #986). Set to '1' to opt out. Env-var, DMI, ECS-metadata, and k8s heuristics still run."
  type        = string
  default     = ""
}

variable "mcp_cloud_provider" {
  description = "Override the cloud auto-detection cascade (issue #1120). Allowed: aws, azure, gcp, on_premises, other. Leave empty to let the cascade run."
  type        = string
  default     = ""
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
  description = "GitHub Personal Access Token for private repo SKILL.md access."
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
  description = "GitHub App private key (PEM format)."
  type        = string
  default     = ""
  sensitive   = true
}

variable "github_extra_hosts" {
  description = "Comma-separated extra GitHub hosts for enterprise instances."
  type        = string
  default     = ""
}

variable "github_api_base_url" {
  description = "GitHub API base URL. For GitHub Enterprise Server use https://<hostname>/api/v3."
  type        = string
  default     = "https://api.github.com"
}

# =============================================================================
# EXTRA ENVIRONMENT VARIABLES (Issue #1000)
# =============================================================================

variable "registry_extra_env" {
  description = "Extra environment variables for registry service. List of objects with 'name' and 'value' string fields. Reserved-name validation is performed at the root module (see terraform/aws-ecs/variables.tf)."
  type        = list(object({ name = string, value = string }))
  default     = []
  sensitive   = true
}

variable "auth_server_extra_env" {
  description = "Extra environment variables for auth-server service. List of objects with 'name' and 'value' string fields. Reserved-name validation is performed at the root module."
  type        = list(object({ name = string, value = string }))
  default     = []
  sensitive   = true
}

variable "mcpgw_extra_env" {
  description = "Extra environment variables for mcpgw service. List of objects with 'name' and 'value' string fields. Reserved-name validation is performed at the root module."
  type        = list(object({ name = string, value = string }))
  default     = []
  sensitive   = true
}

# ---------------------------------------------------------------------------
# Per-user egress credential vault (third-party OBO support).
# On ECS the natural secret store is AWS Secrets Manager (OpenBao is the EKS
# path), so only the secrets-manager vars are wired here. iam.tf grants the
# task role secretsmanager + kms access (scoped to the path prefix) when
# egress_auth_enabled.
# ---------------------------------------------------------------------------

variable "egress_auth_enabled" {
  description = "Enable the per-user egress credential vault. Default: false."
  type        = bool
  default     = false
}

variable "egress_secret_store_backend" {
  description = "Egress secret store backend: secrets-manager (openbao is the EKS/Helm path)."
  type        = string
  default     = "secrets-manager"
}

variable "egress_oauth_callback_base_url" {
  description = "Public base URL for the egress OAuth callback ({base}/oauth2/egress/callback)."
  type        = string
  default     = ""
}

variable "egress_token_refresh_skew_seconds" {
  description = "Refresh a vaulted token this many seconds before expiry."
  type        = number
  default     = 300
}

variable "egress_state_ttl_seconds" {
  description = "TTL for the AEAD-encrypted egress OAuth state blob."
  type        = number
  default     = 600
}

variable "egress_registry_internal_url" {
  description = "URL the auth-server uses to reach the registry internal vend endpoint."
  type        = string
  default     = "http://registry:8080"
}

variable "egress_nginx_marker_secret" {
  description = "Optional override for the nginx marker secret shared by registry + auth-server. Empty auto-generates a strong value (stored in Secrets Manager). The marker is required unconditionally -- both services refuse to start without it."
  type        = string
  default     = ""
  sensitive   = true
}

variable "egress_secrets_manager_kms_key_id" {
  description = "Optional KMS CMK id/ARN for the egress Secrets Manager secrets. Empty uses the AWS-managed key."
  type        = string
  default     = ""
}

variable "egress_secrets_manager_path_prefix" {
  description = "Secrets Manager name prefix for the egress vault (also scopes the task IAM grant)."
  type        = string
  default     = "mcp/egress"
}
