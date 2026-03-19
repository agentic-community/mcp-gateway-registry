# ACM certificate for custom domain (production only)
resource "aws_acm_certificate" "telemetry" {
  count = var.custom_domain != "" ? 1 : 0

  domain_name       = var.custom_domain
  validation_method = "DNS"

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name = "telemetry-collector-cert"
  }
}

# Route53 record for ACM certificate validation
resource "aws_route53_record" "cert_validation" {
  count = var.custom_domain != "" && var.route53_zone_id != "" ? 1 : 0

  zone_id = var.route53_zone_id
  name    = tolist(aws_acm_certificate.telemetry[0].domain_validation_options)[0].resource_record_name
  type    = tolist(aws_acm_certificate.telemetry[0].domain_validation_options)[0].resource_record_type
  records = [tolist(aws_acm_certificate.telemetry[0].domain_validation_options)[0].resource_record_value]
  ttl     = 60
}

# ACM certificate validation
resource "aws_acm_certificate_validation" "telemetry" {
  count = var.custom_domain != "" && var.route53_zone_id != "" ? 1 : 0

  certificate_arn         = aws_acm_certificate.telemetry[0].arn
  validation_record_fqdns = [aws_route53_record.cert_validation[0].fqdn]
}

# API Gateway custom domain name
resource "aws_apigatewayv2_domain_name" "telemetry" {
  count = var.custom_domain != "" ? 1 : 0

  domain_name = var.custom_domain

  domain_name_configuration {
    certificate_arn = aws_acm_certificate.telemetry[0].arn
    endpoint_type   = "REGIONAL"
    security_policy = "TLS_1_2"
  }

  depends_on = [aws_acm_certificate_validation.telemetry]
}

# API Gateway domain mapping
resource "aws_apigatewayv2_api_mapping" "telemetry" {
  count = var.custom_domain != "" ? 1 : 0

  api_id      = aws_apigatewayv2_api.telemetry.id
  domain_name = aws_apigatewayv2_domain_name.telemetry[0].id
  stage       = aws_apigatewayv2_stage.telemetry.id
}

# Route53 A record for custom domain
resource "aws_route53_record" "telemetry" {
  count = var.custom_domain != "" && var.route53_zone_id != "" ? 1 : 0

  zone_id = var.route53_zone_id
  name    = var.custom_domain
  type    = "A"

  alias {
    name                   = aws_apigatewayv2_domain_name.telemetry[0].domain_name_configuration[0].target_domain_name
    zone_id                = aws_apigatewayv2_domain_name.telemetry[0].domain_name_configuration[0].hosted_zone_id
    evaluate_target_health = false
  }
}
