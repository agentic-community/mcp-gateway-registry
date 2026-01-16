#
# CloudFront Distributions for HTTPS
#
# Supports three deployment modes:
#   1. CloudFront-only: Use *.cloudfront.net URLs directly (no custom domain)
#   2. Custom Domain → ALB: Traditional setup with ACM certificates (CloudFront disabled)
#   3. Custom Domain → CloudFront: Route53 points to CloudFront (best of both)
#
# When enable_cloudfront=true AND enable_route53_dns=true (Mode 3), CloudFront
# is configured with custom domain aliases and ACM certificates from us-east-1.
# Route53 points to CloudFront instead of ALBs.
#

# Data sources for managed CloudFront policies
data "aws_cloudfront_cache_policy" "caching_disabled" {
  name = "Managed-CachingDisabled"
}

data "aws_cloudfront_origin_request_policy" "all_viewer" {
  name = "Managed-AllViewer"
}

# CloudFront distribution for MCP Gateway ALB
resource "aws_cloudfront_distribution" "mcp_gateway" {
  count = var.enable_cloudfront ? 1 : 0

  enabled             = true
  comment             = "${var.name} MCP Gateway Registry CloudFront Distribution"
  default_root_object = ""
  price_class         = "PriceClass_100"

  # Custom domain alias when Route53 is also enabled (Mode 3)
  aliases = var.enable_route53_dns ? ["registry.${local.root_domain}"] : []

  origin {
    domain_name = module.mcp_gateway.alb_dns_name
    origin_id   = "mcp-gateway-alb"

    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "http-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }

    # Custom header to tell backend the original protocol was HTTPS
    # Note: We use X-Forwarded-Proto directly - ALB won't overwrite origin custom headers
    custom_header {
      name  = "X-Forwarded-Proto"
      value = "https"
    }

    # Custom header to indicate this request came through CloudFront
    # The auth server uses this for reliable HTTPS detection
    custom_header {
      name  = "X-Cloudfront-Forwarded-Proto"
      value = "https"
    }
  }

  default_cache_behavior {
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "mcp-gateway-alb"

    # Disable caching for dynamic content
    cache_policy_id          = data.aws_cloudfront_cache_policy.caching_disabled.id
    # Forward all headers to origin
    origin_request_policy_id = data.aws_cloudfront_origin_request_policy.all_viewer.id

    viewer_protocol_policy = "redirect-to-https"
    compress               = true
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  # Use ACM certificate from us-east-1 when custom domain is configured (Mode 3)
  # Otherwise use default CloudFront certificate (Mode 1)
  viewer_certificate {
    cloudfront_default_certificate = var.enable_route53_dns ? false : true
    acm_certificate_arn            = var.enable_route53_dns ? aws_acm_certificate.registry_cloudfront[0].arn : null
    ssl_support_method             = var.enable_route53_dns ? "sni-only" : null
    minimum_protocol_version       = var.enable_route53_dns ? "TLSv1.2_2021" : null
  }

  # Ensure certificate is validated before CloudFront uses it
  depends_on = [aws_acm_certificate_validation.registry_cloudfront]

  tags = merge(
    local.common_tags,
    {
      Name      = "${var.name}-mcp-gateway-cloudfront"
      Component = "mcp-gateway"
    }
  )
}

# CloudFront distribution for Keycloak ALB
resource "aws_cloudfront_distribution" "keycloak" {
  count = var.enable_cloudfront ? 1 : 0

  enabled     = true
  comment     = "${var.name} Keycloak CloudFront Distribution"
  price_class = "PriceClass_100"

  # Custom domain alias when Route53 is also enabled (Mode 3)
  aliases = var.enable_route53_dns ? [local.keycloak_domain] : []

  origin {
    domain_name = aws_lb.keycloak.dns_name
    origin_id   = "keycloak-alb"

    custom_origin_config {
      http_port              = 80
      https_port             = 443
      # Always use HTTP to ALB - the ALB HTTP listener is configured to forward
      # (not redirect) when CloudFront is enabled. Using HTTPS would fail because
      # the ALB cert is for the custom domain, not the ALB DNS name.
      origin_protocol_policy = "http-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }

    # Custom header to tell Keycloak the original protocol was HTTPS
    custom_header {
      name  = "X-Forwarded-Proto"
      value = "https"
    }
  }

  default_cache_behavior {
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "keycloak-alb"

    cache_policy_id          = data.aws_cloudfront_cache_policy.caching_disabled.id
    origin_request_policy_id = data.aws_cloudfront_origin_request_policy.all_viewer.id

    viewer_protocol_policy = "redirect-to-https"
    compress               = true
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  # Use ACM certificate from us-east-1 when custom domain is configured (Mode 3)
  # Otherwise use default CloudFront certificate (Mode 1)
  viewer_certificate {
    cloudfront_default_certificate = var.enable_route53_dns ? false : true
    acm_certificate_arn            = var.enable_route53_dns ? aws_acm_certificate.keycloak_cloudfront[0].arn : null
    ssl_support_method             = var.enable_route53_dns ? "sni-only" : null
    minimum_protocol_version       = var.enable_route53_dns ? "TLSv1.2_2021" : null
  }

  # Ensure certificate is validated before CloudFront uses it
  depends_on = [aws_acm_certificate_validation.keycloak_cloudfront]

  tags = merge(
    local.common_tags,
    {
      Name      = "${var.name}-keycloak-cloudfront"
      Component = "keycloak"
    }
  )
}
