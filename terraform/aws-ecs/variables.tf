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
  default     = ""  # Not required when using file storage backend
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
  default     = ""  # Set to "com.amazonaws.global.cloudfront.origin-facing" when enable_cloudfront=true
}

variable "enable_route53_dns" {
  description = "Enable Route53 DNS records and ACM certificates for custom domain. Set to false when using CloudFront-only deployment."
  type        = bool
  default     = true
}
