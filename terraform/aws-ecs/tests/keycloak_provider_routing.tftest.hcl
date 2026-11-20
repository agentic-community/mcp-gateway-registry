# Mocked plans only: no AWS credentials, API calls, or infrastructure changes.
mock_provider "aws" {
  override_during = plan
  mock_data "aws_availability_zones" {
    defaults = { names = ["us-west-2a", "us-west-2b", "us-west-2c"] }
  }
  mock_data "aws_partition" {
    defaults = { partition = "aws", dns_suffix = "amazonaws.com" }
  }
  mock_data "aws_region" {
    defaults = { name = "us-west-2", region = "us-west-2" }
  }
  mock_data "aws_caller_identity" {
    defaults = { account_id = "123456789012" }
  }
  mock_data "aws_iam_policy_document" {
    defaults = { json = "{\"Version\":\"2012-10-17\",\"Statement\":[]}" }
  }
  mock_data "aws_vpc" {
    defaults = { cidr_block = "10.0.0.0/16" }
  }
  mock_resource "aws_security_group" {
    defaults = { id = "sg-0123456789abcdef0" }
  }
  mock_resource "aws_lb" {
    defaults = {
      arn      = "arn:aws:elasticloadbalancing:us-west-2:123456789012:loadbalancer/app/example/0123456789abcdef"
      dns_name = "example.us-west-2.elb.amazonaws.com"
      zone_id  = "Z1H1FL5HABSF5"
    }
  }
  mock_resource "aws_lb_target_group" {
    defaults = { arn = "arn:aws:elasticloadbalancing:us-west-2:123456789012:targetgroup/example/0123456789abcdef" }
  }
  mock_resource "aws_lb_listener" {
    defaults = { arn = "arn:aws:elasticloadbalancing:us-west-2:123456789012:listener/app/example/0123456789abcdef/0123456789abcdef" }
  }
  mock_resource "aws_kms_key" {
    defaults = {
      arn    = "arn:aws:kms:us-west-2:123456789012:key/12345678-1234-1234-1234-123456789012"
      key_id = "12345678-1234-1234-1234-123456789012"
    }
  }
  mock_resource "aws_iam_role" {
    defaults = { arn = "arn:aws:iam::123456789012:role/example" }
  }
  mock_resource "aws_secretsmanager_secret" {
    defaults = { arn = "arn:aws:secretsmanager:us-west-2:123456789012:secret:example-AbCdEf" }
  }
  mock_resource "aws_ecs_cluster" {
    defaults = { arn = "arn:aws:ecs:us-west-2:123456789012:cluster/example" }
  }
  mock_resource "aws_cloudfront_distribution" {
    defaults = { domain_name = "example.cloudfront.net", hosted_zone_id = "Z2FDTNDATAQYW2" }
  }
}
mock_provider "aws" {
  alias = "us_east_1"
  override_during = plan
}
mock_provider "random" {
  override_during = plan
}
mock_provider "time" {
  override_during = plan
}
mock_provider "null" {}
mock_provider "archive" {
  override_during = plan
}

variables {
  name                       = "provider-routing-test"
  aws_region                 = "us-west-2"
  use_existing_vpc            = true
  existing_vpc_id             = "vpc-0123456789abcdef0"
  existing_private_subnet_ids = ["subnet-0123456789abcdef0", "subnet-0123456789abcdef1"]
  existing_public_subnet_ids  = ["subnet-0123456789abcdef2", "subnet-0123456789abcdef3"]
  existing_nat_public_ips    = ["198.51.100.2"]
  create_vpc_endpoints       = false
  enable_observability      = false
  enable_monitoring         = false
  enable_cloudfront         = true
  enable_route53_dns        = true
  enable_waf                = true
  use_regional_domains      = false
  keycloak_domain           = "kc.example.com"
  root_domain               = "example.com"
  keycloak_admin_password    = "EXAMPLE-admin-password-12345" # pragma: allowlist secret
  keycloak_database_password = "EXAMPLE-database-password-12345" # pragma: allowlist secret
  documentdb_admin_password  = "EXAMPLE-documentdb-password-12345" # pragma: allowlist secret
}

# Certificate validation records need known keys during a mocked plan.
override_resource {
  target          = aws_acm_certificate.registry[0]
  override_during = plan
  values = {
    arn = "arn:aws:acm:us-west-2:123456789012:certificate/12345678-1234-1234-1234-123456789001"
    domain_validation_options = [{
      domain_name = "registry.example.com", resource_record_name = "_registry.example.com",
      resource_record_type = "CNAME", resource_record_value = "_registry.acm-validations.aws."
    }]
  }
}
override_resource {
  target          = aws_acm_certificate.registry_cloudfront[0]
  override_during = plan
  values = {
    arn = "arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789002"
    domain_validation_options = [{
      domain_name = "registry.example.com", resource_record_name = "_registry-cf.example.com",
      resource_record_type = "CNAME", resource_record_value = "_registry-cf.acm-validations.aws."
    }]
  }
}
override_resource {
  target          = aws_acm_certificate.keycloak[0]
  override_during = plan
  values = {
    arn = "arn:aws:acm:us-west-2:123456789012:certificate/12345678-1234-1234-1234-123456789003"
    domain_validation_options = [{
      domain_name = "kc.example.com", resource_record_name = "_kc.example.com",
      resource_record_type = "CNAME", resource_record_value = "_kc.acm-validations.aws."
    }]
  }
}
override_resource {
  target          = aws_acm_certificate.keycloak_cloudfront[0]
  override_during = plan
  values = {
    arn = "arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789004"
    domain_validation_options = [{
      domain_name = "kc.example.com", resource_record_name = "_kc-cf.example.com",
      resource_record_type = "CNAME", resource_record_value = "_kc-cf.acm-validations.aws."
    }]
  }
}

run "auth0_removes_keycloak_edge_with_all_edge_features_enabled" {
  command = plan
  variables {
    auth0_enabled             = true
    okta_enabled              = true
    entra_enabled             = true
    cognito_enabled           = true
    keycloak_admin_cidr_blocks = ["unused-invalid-cidr"]
  }
  assert {
    condition = alltrue([
      local.auth_provider == "auth0",
      length(aws_lb.keycloak) == 0,
      length(aws_lb_target_group.keycloak) == 0,
      length(aws_lb_listener.keycloak_http) == 0,
      length(aws_lb_listener.keycloak_https) == 0,
      length(aws_lb_listener_rule.keycloak_routing) == 0,
      length(aws_security_group.keycloak_lb) == 0,
      length(aws_security_group.keycloak_lb_cloudfront) == 0,
      length(aws_cloudfront_distribution.keycloak) == 0,
      length(aws_route53_record.keycloak) == 0,
      length(aws_acm_certificate.keycloak) == 0,
      length(aws_acm_certificate.keycloak_cloudfront) == 0,
      length(aws_route53_record.keycloak_certificate_validation) == 0,
      length(aws_route53_record.keycloak_cloudfront_cert_validation) == 0,
      length(aws_wafv2_web_acl.keycloak) == 0,
      length(aws_wafv2_web_acl_association.keycloak) == 0,
      length(aws_wafv2_web_acl_logging_configuration.keycloak) == 0,
      length(aws_ecs_service.keycloak) == 0,
      length(aws_ecs_task_definition.keycloak) == 0,
      length(aws_appautoscaling_target.keycloak) == 0,
      length(terraform_data.keycloak_admin_cidr_settings) == 0,
      output.keycloak_url == null,
      output.keycloak_admin_console == null,
      output.keycloak_alb_dns == null,
      output.cloudfront_keycloak_url == null,
      output.keycloak_waf_arn == "",
    ])
    error_message = "Auth0 must remove Keycloak compute and every public Keycloak route, even with stale Keycloak settings."
  }
  assert {
    condition = (
      aws_rds_cluster.keycloak.database_name == "keycloak" &&
      aws_secretsmanager_secret.keycloak_db_secret.name == "keycloak/database" &&
      aws_ssm_parameter.keycloak_admin_password.name == "/keycloak/admin_password" &&
      length(aws_cloudfront_distribution.mcp_gateway) == 1 &&
      length(aws_route53_record.registry) == 1 &&
      length(aws_wafv2_web_acl_association.mcp_gateway) == 1
    )
    error_message = "Provider selection must preserve the Keycloak database/credentials and the registry public edge."
  }
}

run "pingfederate_takes_precedence_over_other_providers" {
  command = plan
  variables {
    pingfederate_enabled = true
    auth0_enabled       = true
    okta_enabled        = true
    entra_enabled       = true
    cognito_enabled     = true
  }
  assert {
    condition     = local.auth_provider == "pingfederate" && length(aws_lb.keycloak) == 0 && length(aws_lb_listener_rule.keycloak_routing) == 0
    error_message = "PingFederate must win provider precedence and remove Keycloak routing."
  }
}

run "okta_removes_keycloak_routes" {
  command = plan
  variables {
    okta_enabled    = true
    entra_enabled   = true
    cognito_enabled = true
  }
  assert {
    condition     = local.auth_provider == "okta" && length(aws_lb.keycloak) == 0 && length(aws_cloudfront_distribution.keycloak) == 0 && length(aws_lb_listener_rule.keycloak_routing) == 0
    error_message = "Okta must win over Entra/Cognito and remove Keycloak routing."
  }
}

run "entra_removes_keycloak_routes" {
  command = plan
  variables {
    entra_enabled   = true
    cognito_enabled = true
  }
  assert {
    condition     = local.auth_provider == "entra" && length(aws_lb.keycloak) == 0 && length(aws_route53_record.keycloak) == 0 && length(aws_lb_listener_rule.keycloak_routing) == 0
    error_message = "Entra must win over Cognito and remove Keycloak routing."
  }
}

run "cognito_removes_keycloak_routes" {
  command = plan
  variables {
    cognito_enabled    = true
    enable_cloudfront  = false
    enable_route53_dns = false
  }
  assert {
    condition     = local.auth_provider == "cognito" && length(aws_lb.keycloak) == 0 && length(aws_lb_listener.keycloak_http) == 0 && length(aws_ecs_service.keycloak) == 0
    error_message = "Cognito must remove Keycloak in direct HTTP mode too."
  }
}

run "no_domain_uses_default_without_keycloak" {
  command = plan
  variables {
    keycloak_domain    = ""
    enable_cloudfront  = false
    enable_route53_dns = false
  }
  assert {
    condition     = local.auth_provider == "default" && length(aws_lb.keycloak) == 0 && length(aws_lb_listener_rule.keycloak_routing) == 0
    error_message = "The existing default-provider fallback must not provision Keycloak routes."
  }
}

run "keycloak_preserves_cloudfront_custom_domain_edge" {
  command = plan
  assert {
    condition = (
      length(aws_lb.keycloak) == 1 &&
      length(aws_lb_listener.keycloak_http) == 1 &&
      length(aws_lb_listener.keycloak_https) == 1 &&
      length(aws_lb_listener_rule.keycloak_routing) == 8 &&
      length(aws_cloudfront_distribution.keycloak) == 1 &&
      length(aws_route53_record.keycloak) == 1 &&
      length(aws_acm_certificate.keycloak_cloudfront) == 1 &&
      length(aws_wafv2_web_acl_association.keycloak) == 1 &&
      length(aws_ecs_service.keycloak) == 1 &&
      aws_lb_listener.keycloak_https[0].default_action[0].fixed_response[0].status_code == "404" &&
      output.keycloak_url == "https://kc.example.com"
    )
    error_message = "Keycloak must retain its custom-domain CloudFront edge and deny unmatched paths."
  }
  assert {
    condition = (
      length([for entry in local.keycloak_container_env : entry.name
        if entry.name == "KC_SPI_CLIENT_REGISTRATION__DEFAULT__ENABLED" && entry.value == "false"
      ]) == 1 &&
      length([for entry in local.keycloak_container_env : entry.name
        if entry.name == "KC_SPI_CLIENT_REGISTRATION__OPENID_CONNECT__ENABLED" && entry.value == "false"
      ]) == 0
    )
    error_message = "Keycloak environment must disable native registration and retain OIDC registration."
  }
}

run "keycloak_direct_dns_redirects_http_and_restricts_admin" {
  command = plan
  variables {
    enable_cloudfront         = false
    keycloak_admin_cidr_blocks = ["203.0.113.10/32"]
  }
  assert {
    condition = (
      aws_lb_listener.keycloak_http[0].default_action[0].type == "redirect" &&
      length(aws_lb_listener_rule.keycloak_routing) == 6 &&
      length(aws_cloudfront_distribution.keycloak) == 0 &&
      alltrue([for key, rule in aws_lb_listener_rule.keycloak_routing :
        startswith(key, "https-") && (strcontains(key, "admin-") ? anytrue([
          for condition in rule.condition : length(condition.source_ip) > 0
        ]) : true)
      ])
    )
    error_message = "Direct-DNS Keycloak must retain HTTPS-only forwarding and source-restricted admin routes."
  }
}

run "keycloak_cloudfront_only_supplies_domain" {
  command = plan
  variables {
    enable_route53_dns = false
    keycloak_domain    = ""
  }
  assert {
    condition = (
      local.auth_provider == "keycloak" &&
      length(aws_cloudfront_distribution.keycloak) == 1 &&
      length(aws_lb_listener.keycloak_https) == 0 &&
      length(aws_lb_listener_rule.keycloak_routing) == 4 &&
      aws_lb_listener.keycloak_http[0].default_action[0].type == "fixed-response"
    )
    error_message = "CloudFront-only mode must still select Keycloak and forward its public routes."
  }
}

run "active_keycloak_rejects_unenforceable_admin_cidrs" {
  command = plan
  variables {
    keycloak_admin_cidr_blocks = ["203.0.113.10/32"]
  }
  expect_failures = [terraform_data.keycloak_admin_cidr_settings]
}
