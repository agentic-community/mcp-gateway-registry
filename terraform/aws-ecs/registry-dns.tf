#
# Registry DNS and SSL Certificate Configuration
#
# Provides DNS and HTTPS support for the main MCP Gateway Registry ALB
# Domain: registry.mycorp.click (configured via local.root_domain)
#
# These resources are only created when enable_route53_dns = true
#

# Use existing hosted zone for the root domain
data "aws_route53_zone" "registry_root" {
  count        = var.enable_route53_dns ? 1 : 0
  name         = local.hosted_zone_domain
  private_zone = false
}

# Create SSL certificate for registry subdomain
resource "aws_acm_certificate" "registry" {
  count             = var.enable_route53_dns ? 1 : 0
  domain_name       = "registry.${local.root_domain}"
  validation_method = "DNS"

  tags = merge(
    local.common_tags,
    {
      Name      = "registry-cert"
      Component = "registry"
    }
  )

  lifecycle {
    create_before_destroy = true
  }
}

# Create DNS validation records for ACM certificate
resource "aws_route53_record" "registry_certificate_validation" {
  for_each = var.enable_route53_dns ? {
    for dvo in aws_acm_certificate.registry[0].domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  } : {}

  allow_overwrite = true
  name            = each.value.name
  records         = [each.value.record]
  ttl             = 60
  type            = each.value.type
  zone_id         = data.aws_route53_zone.registry_root[0].zone_id
}

# Wait for certificate validation to complete
resource "aws_acm_certificate_validation" "registry" {
  count           = var.enable_route53_dns ? 1 : 0
  certificate_arn = aws_acm_certificate.registry[0].arn

  timeouts {
    create = "5m"
  }

  validation_record_fqdns = [for record in aws_route53_record.registry_certificate_validation : record.fqdn]
}

# Create A record for registry subdomain
# Points to CloudFront when both CloudFront and Route53 are enabled (Mode 3)
# Points to ALB when only Route53 is enabled (Mode 2)
resource "aws_route53_record" "registry" {
  count   = var.enable_route53_dns ? 1 : 0
  zone_id = data.aws_route53_zone.registry_root[0].zone_id
  name    = "registry.${local.root_domain}"
  type    = "A"

  alias {
    # Mode 3: Route53 → CloudFront (when both enabled)
    # Mode 2: Route53 → ALB (when only Route53 enabled)
    name                   = var.enable_cloudfront ? aws_cloudfront_distribution.mcp_gateway[0].domain_name : module.mcp_gateway.alb_dns_name
    zone_id                = var.enable_cloudfront ? aws_cloudfront_distribution.mcp_gateway[0].hosted_zone_id : module.mcp_gateway.alb_zone_id
    evaluate_target_health = true
  }
}
