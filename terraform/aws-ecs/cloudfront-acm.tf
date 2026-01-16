#
# ACM Certificates in us-east-1 for CloudFront Custom Domains
#
# CloudFront requires certificates to be in us-east-1 regardless of where
# the origin resources are deployed. These certificates are only created
# when both CloudFront AND Route53 DNS are enabled (Mode 3: Custom Domain → CloudFront)
#

# Provider alias for us-east-1 (required for CloudFront certificates)
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"
}

# ACM Certificate for Registry custom domain on CloudFront
resource "aws_acm_certificate" "registry_cloudfront" {
  count    = var.enable_cloudfront && var.enable_route53_dns ? 1 : 0
  provider = aws.us_east_1

  domain_name       = "registry.${local.root_domain}"
  validation_method = "DNS"

  tags = merge(
    local.common_tags,
    {
      Name      = "${var.name}-registry-cloudfront-cert"
      Component = "registry"
      Purpose   = "CloudFront custom domain"
    }
  )

  lifecycle {
    create_before_destroy = true
  }
}

# DNS validation records for Registry CloudFront certificate
resource "aws_route53_record" "registry_cloudfront_cert_validation" {
  for_each = var.enable_cloudfront && var.enable_route53_dns ? {
    for dvo in aws_acm_certificate.registry_cloudfront[0].domain_validation_options : dvo.domain_name => {
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

# Wait for Registry CloudFront certificate validation
resource "aws_acm_certificate_validation" "registry_cloudfront" {
  count    = var.enable_cloudfront && var.enable_route53_dns ? 1 : 0
  provider = aws.us_east_1

  certificate_arn = aws_acm_certificate.registry_cloudfront[0].arn

  timeouts {
    create = "10m"
  }

  validation_record_fqdns = [for record in aws_route53_record.registry_cloudfront_cert_validation : record.fqdn]
}

# ACM Certificate for Keycloak custom domain on CloudFront
resource "aws_acm_certificate" "keycloak_cloudfront" {
  count    = var.enable_cloudfront && var.enable_route53_dns ? 1 : 0
  provider = aws.us_east_1

  domain_name       = local.keycloak_domain
  validation_method = "DNS"

  tags = merge(
    local.common_tags,
    {
      Name      = "${var.name}-keycloak-cloudfront-cert"
      Component = "keycloak"
      Purpose   = "CloudFront custom domain"
    }
  )

  lifecycle {
    create_before_destroy = true
  }
}

# DNS validation records for Keycloak CloudFront certificate
resource "aws_route53_record" "keycloak_cloudfront_cert_validation" {
  for_each = var.enable_cloudfront && var.enable_route53_dns ? {
    for dvo in aws_acm_certificate.keycloak_cloudfront[0].domain_validation_options : dvo.domain_name => {
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
  zone_id         = data.aws_route53_zone.root[0].zone_id
}

# Wait for Keycloak CloudFront certificate validation
resource "aws_acm_certificate_validation" "keycloak_cloudfront" {
  count    = var.enable_cloudfront && var.enable_route53_dns ? 1 : 0
  provider = aws.us_east_1

  certificate_arn = aws_acm_certificate.keycloak_cloudfront[0].arn

  timeouts {
    create = "10m"
  }

  validation_record_fqdns = [for record in aws_route53_record.keycloak_cloudfront_cert_validation : record.fqdn]
}
