# MCP Gateway Registry - AWS ECS Deployment
# This Terraform configuration deploys the MCP Gateway to AWS ECS Fargate

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# MCP Gateway Module
module "mcp_gateway" {
  source = "./modules/mcp-gateway"

  # Basic configuration
  name = "${var.name}-v2"

  # Network configuration
  vpc_id              = module.vpc.vpc_id
  private_subnet_ids  = module.vpc.private_subnets
  public_subnet_ids   = module.vpc.public_subnets
  ingress_cidr_blocks = var.ingress_cidr_blocks

  # ECS configuration
  ecs_cluster_arn         = module.ecs_cluster.arn
  ecs_cluster_name        = module.ecs_cluster.name
  task_execution_role_arn = module.ecs_cluster.task_exec_iam_role_arn

  # HTTPS configuration - only use certificate when Route53 DNS is enabled (without CloudFront)
  # When CloudFront is enabled, HTTPS termination happens at CloudFront, not ALB
  enable_https    = var.enable_route53_dns && !var.enable_cloudfront
  certificate_arn = var.enable_route53_dns && !var.enable_cloudfront ? aws_acm_certificate.registry[0].arn : ""

  # Domain name for the registry - determines REGISTRY_URL and OAuth redirect URIs
  # Simplified to 3 modes (no dual-access):
  #   Mode 1: CloudFront-only - use CloudFront domain
  #   Mode 2: Custom Domain → ALB - use custom domain
  #   Mode 3: Custom Domain → CloudFront - use custom domain (traffic flows through CloudFront)
  domain_name = var.enable_route53_dns ? "registry.${local.root_domain}" : (
    var.enable_cloudfront ? aws_cloudfront_distribution.mcp_gateway[0].domain_name : ""
  )

  # Additional server names for nginx - no longer needed with simplified modes
  # Each deployment has a single entry point (either custom domain or CloudFront domain)
  additional_server_names = ""

  # Keycloak configuration
  # Mode 1: CloudFront-only - use CloudFront domain
  # Mode 2 & 3: Custom domain (Route53 enabled) - use custom domain
  keycloak_domain = var.enable_route53_dns ? local.keycloak_domain : (
    var.enable_cloudfront ? aws_cloudfront_distribution.keycloak[0].domain_name : local.keycloak_domain
  )

  # CloudFront configuration - allows CloudFront IPs to reach ALB
  enable_cloudfront           = var.enable_cloudfront
  cloudfront_prefix_list_name = local.cloudfront_prefix_list_name

  # Container images
  registry_image_uri               = var.registry_image_uri
  auth_server_image_uri            = var.auth_server_image_uri
  currenttime_image_uri            = var.currenttime_image_uri
  mcpgw_image_uri                  = var.mcpgw_image_uri
  realserverfaketools_image_uri    = var.realserverfaketools_image_uri
  flight_booking_agent_image_uri   = var.flight_booking_agent_image_uri
  travel_assistant_agent_image_uri = var.travel_assistant_agent_image_uri

  # Service replicas
  currenttime_replicas            = var.currenttime_replicas
  mcpgw_replicas                  = var.mcpgw_replicas
  realserverfaketools_replicas    = var.realserverfaketools_replicas
  flight_booking_agent_replicas   = var.flight_booking_agent_replicas
  travel_assistant_agent_replicas = var.travel_assistant_agent_replicas

  # Auto-scaling configuration
  enable_autoscaling        = true
  autoscaling_min_capacity  = 2
  autoscaling_max_capacity  = 4
  autoscaling_target_cpu    = 70
  autoscaling_target_memory = 80

  # Monitoring configuration
  enable_monitoring = var.enable_monitoring
  alarm_email       = var.alarm_email

  # Embeddings configuration
  embeddings_provider         = var.embeddings_provider
  embeddings_model_name       = var.embeddings_model_name
  embeddings_model_dimensions = var.embeddings_model_dimensions
  embeddings_aws_region       = var.embeddings_aws_region
  embeddings_api_key          = var.embeddings_api_key

  # Keycloak admin credentials (for Management API)
  keycloak_admin_password = var.keycloak_admin_password

  # Session cookie security configuration
  session_cookie_secure = var.session_cookie_secure
  session_cookie_domain = var.session_cookie_domain

  # DocumentDB configuration
  storage_backend                   = var.storage_backend
  documentdb_endpoint               = aws_docdb_cluster.registry.endpoint
  documentdb_database               = var.documentdb_database
  documentdb_namespace              = var.documentdb_namespace
  documentdb_use_tls                = var.documentdb_use_tls
  documentdb_use_iam                = var.documentdb_use_iam
  documentdb_credentials_secret_arn = var.storage_backend == "documentdb" ? aws_secretsmanager_secret.documentdb_credentials.arn : ""

  # Security scanning configuration
  security_scan_enabled         = var.security_scan_enabled
  security_scan_on_registration = var.security_scan_on_registration
  security_block_unsafe_servers = var.security_block_unsafe_servers
  security_analyzers            = var.security_analyzers
  security_scan_timeout         = var.security_scan_timeout
  security_add_pending_tag      = var.security_add_pending_tag

  # Microsoft Entra ID configuration
  entra_enabled       = var.entra_enabled
  entra_tenant_id     = var.entra_tenant_id
  entra_client_id     = var.entra_client_id
  entra_client_secret = var.entra_client_secret

  # OAuth token storage
  oauth_store_tokens_in_session = var.oauth_store_tokens_in_session

  # Registry static token auth
  registry_static_token_auth_enabled = var.registry_static_token_auth_enabled
  registry_api_token                 = var.registry_api_token
  max_tokens_per_user_per_hour       = var.max_tokens_per_user_per_hour

  # Federation configuration (peer-to-peer registry sync)
  registry_id                          = var.registry_id
  federation_static_token_auth_enabled = var.federation_static_token_auth_enabled
  federation_static_token              = var.federation_static_token
  federation_encryption_key            = var.federation_encryption_key

  # Audit logging configuration
  audit_log_enabled  = var.audit_log_enabled
  audit_log_ttl_days = var.audit_log_ttl_days

  # Deployment mode configuration
  deployment_mode = var.deployment_mode
  registry_mode   = var.registry_mode

  # Observability configuration
  enable_observability      = var.enable_observability
  metrics_service_image_uri = var.metrics_service_image_uri
  grafana_image_uri         = var.grafana_image_uri
  grafana_admin_password    = var.grafana_admin_password
}

# =============================================================================
# CloudFront Configuration Warnings
# =============================================================================

# Warning for dual ingress configuration (both CloudFront and custom domain)
resource "null_resource" "dual_ingress_warning" {
  count = var.enable_cloudfront && var.enable_route53_dns ? 1 : 0

  triggers = {
    always_run = timestamp()
  }

  provisioner "local-exec" {
    command = <<-EOT
      echo ""
      echo "============================================================"
      echo "INFO: Custom Domain → CloudFront Configuration (Mode 3)"
      echo "============================================================"
      echo "Both CloudFront (enable_cloudfront=true) and Route53 DNS"
      echo "(enable_route53_dns=true) are enabled."
      echo ""
      echo "Traffic flow: Custom Domain → CloudFront → ALB → ECS"
      echo ""
      echo "Access URL: https://registry.${local.root_domain}"
      echo ""
      echo "Benefits:"
      echo "  - Custom branded domain"
      echo "  - CloudFront edge caching and DDoS protection"
      echo "  - Single entry point (no dual-access confusion)"
      echo "============================================================"
      echo ""
    EOT
  }
}
