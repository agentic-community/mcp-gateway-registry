#
# WAFv2 Web ACL Configuration for MCP Gateway and Keycloak ALBs
# Set enable_waf = false in terraform.tfvars if you don't have wafv2:* IAM permissions
#
# Both ACLs share the same 5-rule chain:
#   1. AWSManagedRulesCommonRuleSet — full Block, everywhere EXCEPT the Keycloak
#      anonymous DCR endpoint (scope-down excludes that path).
#   2. AWSManagedRulesCommonRuleSet — DCR endpoint only, with count-mode
#      overrides for EC2MetaDataSSRF_BODY and GenericRFI_BODY (both fire on
#      loopback callback URIs per RFC 8252; documented false positives).
#   3. AWSManagedRulesKnownBadInputsRuleSet — full Block.
#   4. IP rate limit — 100 req/5min per IP.
#   5. Global rate limit — 2000 req/5min total (scope-down = URI CONTAINS "/").

locals {
  # Keycloak anonymous Dynamic Client Registration endpoint. Requests here
  # legitimately carry http://localhost:<port>/ callback URIs (RFC 8252).
  keycloak_dcr_path_prefix = "/realms/mcp-gateway/clients-registrations/"
}


# WAFv2 Web ACL for MCP Gateway ALB
resource "aws_wafv2_web_acl" "mcp_gateway" {
  count = var.enable_waf ? 1 : 0

  name  = "${var.name}-mcp-gateway-waf"
  scope = "REGIONAL"

  default_action {
    allow {}
  }

  # Rule 1: CommonRuleSet EVERYWHERE EXCEPT Keycloak DCR path.
  # AWS requires managed_rule_group_statement at the rule's top level (cannot
  # be nested under and_statement/not_statement). Path exclusion goes via the
  # managed group's own scope_down_statement.
  rule {
    name     = "CommonRuleSet"
    priority = 1

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"

        scope_down_statement {
          not_statement {
            statement {
              byte_match_statement {
                positional_constraint = "STARTS_WITH"
                search_string         = local.keycloak_dcr_path_prefix
                field_to_match {
                  uri_path {}
                }
                text_transformation {
                  priority = 0
                  type     = "NONE"
                }
              }
            }
          }
        }
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "CommonRuleSetMetric"
      sampled_requests_enabled   = true
    }
  }

  # Rule 2: CommonRuleSet on Keycloak DCR endpoint ONLY, with count-mode
  # overrides for the sub-rules that false-positive on RFC 8252 loopback URIs.
  rule {
    name     = "CommonRuleSetForDcr"
    priority = 2

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"

        # 'localhost' / '127.0.0.1' / '169.254.*' in the body — legit for
        # native-client DCR redirect_uris.
        rule_action_override {
          name = "EC2MetaDataSSRF_BODY"
          action_to_use {
            count {}
          }
        }
        # Shadowed by _BODY until _BODY was downgraded; now it becomes the
        # next terminating match on the same body content.
        rule_action_override {
          name = "GenericRFI_BODY"
          action_to_use {
            count {}
          }
        }

        scope_down_statement {
          byte_match_statement {
            positional_constraint = "STARTS_WITH"
            search_string         = local.keycloak_dcr_path_prefix
            field_to_match {
              uri_path {}
            }
            text_transformation {
              priority = 0
              type     = "NONE"
            }
          }
        }
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "CommonRuleSetForDcrMetric"
      sampled_requests_enabled   = true
    }
  }

  # Rule 3: AWS Managed Rules - Known Bad Inputs
  rule {
    name     = "AWSManagedRulesKnownBadInputsRuleSet"
    priority = 3

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesKnownBadInputsRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "AWSManagedRulesKnownBadInputsRuleSetMetric"
      sampled_requests_enabled   = true
    }
  }

  # Rule 4: IP-based rate limiting (100 req/5min per IP)
  rule {
    name     = "IPRateLimitRule"
    priority = 4

    action {
      block {}
    }

    statement {
      rate_based_statement {
        limit              = 100
        aggregate_key_type = "IP"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "IPRateLimitRuleMetric"
      sampled_requests_enabled   = true
    }
  }

  # Rule 5: Global rate limiting (2000 req/5min globally).
  # WAF requires a scope_down_statement when aggregate_key_type=CONSTANT. Match-all
  # achieved via byte_match on URI path containing "/" — every HTTP request has a
  # URI, so all traffic counts toward the global bucket.
  rule {
    name     = "GlobalRateLimitRule"
    priority = 5

    action {
      block {}
    }

    statement {
      rate_based_statement {
        limit              = 2000
        aggregate_key_type = "CONSTANT"

        scope_down_statement {
          byte_match_statement {
            field_to_match {
              uri_path {}
            }
            positional_constraint = "CONTAINS"
            search_string         = "/"
            text_transformation {
              priority = 0
              type     = "NONE"
            }
          }
        }
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "GlobalRateLimitRuleMetric"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${var.name}-mcp-gateway-waf"
    sampled_requests_enabled   = true
  }

  tags = merge(
    local.common_tags,
    {
      Purpose   = "WAF protection for MCP Gateway ALB"
      Component = "security"
    }
  )
}


# Associate WAF with MCP Gateway ALB
resource "aws_wafv2_web_acl_association" "mcp_gateway" {
  count = var.enable_waf ? 1 : 0

  resource_arn = module.mcp_gateway.alb_arn
  web_acl_arn  = aws_wafv2_web_acl.mcp_gateway[0].arn
}


# CloudWatch Log Group for WAF logs
resource "aws_cloudwatch_log_group" "waf_mcp_gateway" {
  count = var.enable_waf ? 1 : 0

  name              = "/aws/wafv2/${var.name}-mcp-gateway"
  retention_in_days = 30

  tags = merge(
    local.common_tags,
    {
      Purpose   = "WAF logs for MCP Gateway"
      Component = "security"
    }
  )
}


# WAF Logging Configuration
resource "aws_wafv2_web_acl_logging_configuration" "mcp_gateway" {
  count = var.enable_waf ? 1 : 0

  resource_arn            = aws_wafv2_web_acl.mcp_gateway[0].arn
  log_destination_configs = [aws_cloudwatch_log_group.waf_mcp_gateway[0].arn]

  # Redact sensitive headers from logs
  redacted_fields {
    single_header {
      name = "authorization"
    }
  }
}


# WAFv2 Web ACL for Keycloak ALB
resource "aws_wafv2_web_acl" "keycloak" {
  count = var.enable_waf ? 1 : 0

  name  = "${var.name}-keycloak-waf"
  scope = "REGIONAL"

  default_action {
    allow {}
  }

  # Rule 1: CommonRuleSet EVERYWHERE EXCEPT Keycloak DCR path.
  # AWS requires managed_rule_group_statement at the rule's top level (cannot
  # be nested under and_statement/not_statement). Path exclusion goes via the
  # managed group's own scope_down_statement.
  rule {
    name     = "CommonRuleSet"
    priority = 1

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"

        scope_down_statement {
          not_statement {
            statement {
              byte_match_statement {
                positional_constraint = "STARTS_WITH"
                search_string         = local.keycloak_dcr_path_prefix
                field_to_match {
                  uri_path {}
                }
                text_transformation {
                  priority = 0
                  type     = "NONE"
                }
              }
            }
          }
        }
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "CommonRuleSetMetric"
      sampled_requests_enabled   = true
    }
  }

  # Rule 2: CommonRuleSet on Keycloak DCR endpoint ONLY, with count-mode
  # overrides for the sub-rules that false-positive on RFC 8252 loopback URIs.
  rule {
    name     = "CommonRuleSetForDcr"
    priority = 2

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"

        rule_action_override {
          name = "EC2MetaDataSSRF_BODY"
          action_to_use {
            count {}
          }
        }
        rule_action_override {
          name = "GenericRFI_BODY"
          action_to_use {
            count {}
          }
        }

        scope_down_statement {
          byte_match_statement {
            positional_constraint = "STARTS_WITH"
            search_string         = local.keycloak_dcr_path_prefix
            field_to_match {
              uri_path {}
            }
            text_transformation {
              priority = 0
              type     = "NONE"
            }
          }
        }
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "CommonRuleSetForDcrMetric"
      sampled_requests_enabled   = true
    }
  }

  # Rule 3: AWS Managed Rules - Known Bad Inputs
  rule {
    name     = "AWSManagedRulesKnownBadInputsRuleSet"
    priority = 3

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesKnownBadInputsRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "AWSManagedRulesKnownBadInputsRuleSetMetric"
      sampled_requests_enabled   = true
    }
  }

  # Rule 4: IP-based rate limiting (100 req/5min per IP)
  rule {
    name     = "IPRateLimitRule"
    priority = 4

    action {
      block {}
    }

    statement {
      rate_based_statement {
        limit              = 100
        aggregate_key_type = "IP"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "IPRateLimitRuleMetric"
      sampled_requests_enabled   = true
    }
  }

  # Rule 5: Global rate limiting (2000 req/5min globally).
  # WAF requires a scope_down_statement when aggregate_key_type=CONSTANT. Match-all
  # achieved via byte_match on URI path containing "/" — every HTTP request has a
  # URI, so all traffic counts toward the global bucket.
  rule {
    name     = "GlobalRateLimitRule"
    priority = 5

    action {
      block {}
    }

    statement {
      rate_based_statement {
        limit              = 2000
        aggregate_key_type = "CONSTANT"

        scope_down_statement {
          byte_match_statement {
            field_to_match {
              uri_path {}
            }
            positional_constraint = "CONTAINS"
            search_string         = "/"
            text_transformation {
              priority = 0
              type     = "NONE"
            }
          }
        }
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "GlobalRateLimitRuleMetric"
      sampled_requests_enabled   = true
    }
  }


  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${var.name}-keycloak-waf"
    sampled_requests_enabled   = true
  }

  tags = merge(
    local.common_tags,
    {
      Purpose   = "WAF protection for Keycloak ALB"
      Component = "security"
    }
  )
}


# Associate WAF with Keycloak ALB
resource "aws_wafv2_web_acl_association" "keycloak" {
  count = var.enable_waf ? 1 : 0

  resource_arn = aws_lb.keycloak.arn
  web_acl_arn  = aws_wafv2_web_acl.keycloak[0].arn
}


# CloudWatch Log Group for Keycloak WAF logs
resource "aws_cloudwatch_log_group" "waf_keycloak" {
  count = var.enable_waf ? 1 : 0

  name              = "/aws/wafv2/${var.name}-keycloak"
  retention_in_days = 30

  tags = merge(
    local.common_tags,
    {
      Purpose   = "WAF logs for Keycloak"
      Component = "security"
    }
  )
}


# WAF Logging Configuration for Keycloak
resource "aws_wafv2_web_acl_logging_configuration" "keycloak" {
  count = var.enable_waf ? 1 : 0

  resource_arn            = aws_wafv2_web_acl.keycloak[0].arn
  log_destination_configs = [aws_cloudwatch_log_group.waf_keycloak[0].arn]

  # Redact sensitive headers from logs
  redacted_fields {
    single_header {
      name = "authorization"
    }
  }
}


# Outputs
output "mcp_gateway_waf_arn" {
  description = "ARN of WAF Web ACL for MCP Gateway"
  value       = var.enable_waf ? aws_wafv2_web_acl.mcp_gateway[0].arn : ""
}


output "keycloak_waf_arn" {
  description = "ARN of WAF Web ACL for Keycloak"
  value       = var.enable_waf ? aws_wafv2_web_acl.keycloak[0].arn : ""
}
