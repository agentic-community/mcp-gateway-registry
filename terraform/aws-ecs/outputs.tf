# Root Module Outputs

# VPC Outputs
output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "vpc_cidr" {
  description = "VPC CIDR block"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = module.vpc.private_subnets
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = module.vpc.public_subnets
}

# ECS Cluster Outputs
output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = module.ecs_cluster.name
}

output "ecs_cluster_arn" {
  description = "ECS cluster ARN"
  value       = module.ecs_cluster.arn
}

# MCP Gateway Outputs
output "mcp_gateway_url" {
  description = "MCP Gateway main URL"
  value       = module.mcp_gateway.service_urls.registry
}

output "mcp_gateway_auth_url" {
  description = "MCP Gateway auth server URL"
  value       = module.mcp_gateway.service_urls.auth
}


output "mcp_gateway_alb_dns" {
  description = "MCP Gateway ALB DNS name"
  value       = module.mcp_gateway.alb_dns_name
}

output "mcp_gateway_https_enabled" {
  description = "Whether HTTPS is enabled for MCP Gateway"
  value       = module.mcp_gateway.https_enabled
}

output "mcp_gateway_autoscaling_enabled" {
  description = "Whether auto-scaling is enabled for MCP Gateway"
  value       = module.mcp_gateway.autoscaling_enabled
}

output "mcp_gateway_monitoring_enabled" {
  description = "Whether monitoring is enabled for MCP Gateway"
  value       = module.mcp_gateway.monitoring_enabled
}

# EFS Outputs
output "mcp_gateway_efs_id" {
  description = "MCP Gateway EFS file system ID"
  value       = module.mcp_gateway.efs_id
}

output "mcp_gateway_efs_arn" {
  description = "MCP Gateway EFS file system ARN"
  value       = module.mcp_gateway.efs_arn
}

output "mcp_gateway_efs_access_points" {
  description = "MCP Gateway EFS access point IDs"
  value       = module.mcp_gateway.efs_access_points
}

# Monitoring Outputs
output "monitoring_sns_topic" {
  description = "SNS topic ARN for CloudWatch alarms"
  value       = var.enable_monitoring ? module.mcp_gateway.sns_topic_arn : null
}

# Summary Output
output "deployment_summary" {
  description = "Summary of deployed components"
  value = {
    mcp_gateway_deployed = true
    https_enabled        = var.enable_route53_dns || var.enable_cloudfront
    monitoring_enabled   = var.enable_monitoring
    multi_az_nat         = true
    autoscaling_enabled  = true
    deployment_mode      = var.enable_cloudfront && !var.enable_route53_dns ? "cloudfront" : (var.enable_route53_dns ? "custom-domain" : "development")
  }
}

#
# Keycloak Outputs
#

output "keycloak_url" {
  description = "Keycloak URL"
  value       = var.enable_route53_dns ? "https://${local.keycloak_domain}" : (
    var.enable_cloudfront ? "https://${aws_cloudfront_distribution.keycloak[0].domain_name}" : "http://${aws_lb.keycloak.dns_name}"
  )
}

output "keycloak_admin_console" {
  description = "Keycloak admin console URL"
  value       = var.enable_route53_dns ? "https://${local.keycloak_domain}/admin" : (
    var.enable_cloudfront ? "https://${aws_cloudfront_distribution.keycloak[0].domain_name}/admin" : "http://${aws_lb.keycloak.dns_name}/admin"
  )
}

output "keycloak_alb_dns" {
  description = "Keycloak ALB DNS name"
  value       = aws_lb.keycloak.dns_name
}

output "keycloak_ecr_repository" {
  description = "Keycloak ECR repository URL"
  value       = aws_ecr_repository.keycloak.repository_url
}

#
# Registry DNS and Certificate Outputs
#

output "registry_url" {
  description = "Registry URL with custom domain"
  value       = var.enable_route53_dns ? "https://registry.${local.root_domain}" : null
}

output "registry_certificate_arn" {
  description = "ACM certificate ARN for registry subdomain"
  value       = var.enable_route53_dns ? aws_acm_certificate.registry[0].arn : null
}

output "registry_dns_record" {
  description = "Registry DNS A record details"
  value = var.enable_route53_dns ? {
    name    = aws_route53_record.registry[0].name
    type    = aws_route53_record.registry[0].type
    zone_id = aws_route53_record.registry[0].zone_id
  } : null
}


#
# CloudFront Outputs (when enabled)
#

output "cloudfront_mcp_gateway_url" {
  description = "CloudFront URL for MCP Gateway (when CloudFront is enabled)"
  value       = var.enable_cloudfront ? "https://${aws_cloudfront_distribution.mcp_gateway[0].domain_name}" : null
}

output "cloudfront_keycloak_url" {
  description = "CloudFront URL for Keycloak (when CloudFront is enabled)"
  value       = var.enable_cloudfront ? "https://${aws_cloudfront_distribution.keycloak[0].domain_name}" : null
}

output "deployment_mode" {
  description = "Current deployment mode based on configuration"
  value = var.enable_cloudfront && !var.enable_route53_dns ? "cloudfront" : (
    var.enable_route53_dns ? "custom-domain" : "development"
  )
}

#
# DocumentDB Cluster Outputs
#

output "documentdb_cluster_endpoint" {
  description = "DocumentDB Cluster endpoint"
  value       = aws_docdb_cluster.registry.endpoint
}

output "documentdb_cluster_arn" {
  description = "DocumentDB Cluster ARN"
  value       = aws_docdb_cluster.registry.arn
}

output "documentdb_reader_endpoint" {
  description = "DocumentDB Cluster reader endpoint"
  value       = aws_docdb_cluster.registry.reader_endpoint
}

output "documentdb_security_group_id" {
  description = "DocumentDB security group ID"
  value       = aws_security_group.documentdb.id
}

output "documentdb_kms_key_id" {
  description = "KMS key ID for DocumentDB encryption"
  value       = aws_kms_key.documentdb.id
}

output "documentdb_kms_key_arn" {
  description = "KMS key ARN for DocumentDB encryption"
  value       = aws_kms_key.documentdb.arn
}

output "documentdb_secrets_manager_secret_arn" {
  description = "Secrets Manager secret ARN for DocumentDB credentials"
  value       = aws_secretsmanager_secret.documentdb_credentials.arn
  sensitive   = true
}
