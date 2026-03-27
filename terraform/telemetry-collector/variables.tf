variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "deployment_stage" {
  description = "Deployment stage: testing or production"
  type        = string
  default     = "testing"

  validation {
    condition     = contains(["testing", "production"], var.deployment_stage)
    error_message = "deployment_stage must be either 'testing' or 'production'"
  }
}

variable "documentdb_instance_class" {
  description = "DocumentDB instance class (db.t3.medium for testing, db.r5.large for production)"
  type        = string
  default     = "db.t3.medium"
}

variable "documentdb_master_username" {
  description = "DocumentDB master username"
  type        = string
  default     = "telemetry_admin"
}

variable "documentdb_database_name" {
  description = "DocumentDB database name"
  type        = string
  default     = "telemetry"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

variable "custom_domain" {
  description = "Optional custom domain for API Gateway (e.g., telemetry.mcpgateway.io)"
  type        = string
  default     = ""
}

variable "route53_zone_id" {
  description = "Optional Route53 hosted zone ID for custom domain"
  type        = string
  default     = ""
}

variable "alarm_email" {
  description = "Optional email address for CloudWatch alarms"
  type        = string
  default     = ""
}

variable "rate_limit_max_requests" {
  description = "Maximum requests per minute per IP"
  type        = number
  default     = 10
}

variable "rate_limit_window_seconds" {
  description = "Rate limit time window in seconds"
  type        = number
  default     = 60
}

variable "cors_allowed_origins" {
  description = "Origins allowed to submit telemetry (restrict to known registry domains)"
  type        = list(string)
  default     = ["https://mcpgateway.io", "https://app.mcpgateway.io"]
}

variable "lambda_package_path" {
  description = "Path to the Lambda deployment package zip file"
  type        = string
  default     = "lambda_function.zip"
}

# Bastion variables
variable "bastion_enabled" {
  description = "Whether to create a bastion host for DocumentDB access"
  type        = bool
  default     = false
}

variable "bastion_public_key" {
  description = "SSH public key for bastion host access"
  type        = string
  default     = ""
}

variable "bastion_allowed_cidrs" {
  description = "CIDR blocks allowed to SSH to the bastion host"
  type        = list(string)
  default     = []
}
