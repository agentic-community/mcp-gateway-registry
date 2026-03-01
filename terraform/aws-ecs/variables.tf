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

variable "max_tokens_per_user_per_hour" {
  description = "Maximum JWT tokens that can be vended per user per hour."
  type        = number
  default     = 100
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
