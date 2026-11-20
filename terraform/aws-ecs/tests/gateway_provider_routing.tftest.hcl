# Render the gateway module's task definitions without AWS API calls.
mock_provider "aws" {
  override_during = plan
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
}
mock_provider "random" {
  override_during = plan
}
mock_provider "time" {
  override_during = plan
}

variables {
  keycloak_admin_password = "EXAMPLE-admin-password-12345" # pragma: allowlist secret
  alb_logs_bucket         = "example-provider-routing-alb-logs"
}

run "gateway_auth0_ignores_stale_keycloak_domain" {
  command = plan
  module {
    source = "./modules/mcp-gateway"
  }
  variables {
    name                 = "provider-routing-module-test"
    vpc_id               = "vpc-0123456789abcdef0"
    private_subnet_ids   = ["subnet-0123456789abcdef0", "subnet-0123456789abcdef1"]
    public_subnet_ids    = ["subnet-0123456789abcdef2", "subnet-0123456789abcdef3"]
    ecs_cluster_arn      = "arn:aws:ecs:us-west-2:123456789012:cluster/example"
    ecs_cluster_name     = "example"
    auth0_enabled        = true
    keycloak_domain      = "unused-keycloak.example.com"
    enable_observability = false
  }
  assert {
    condition = alltrue([
      for container in [
        module.ecs_service_auth.container_definitions["auth-server"].container_definition,
        module.ecs_service_registry.container_definitions["registry"].container_definition,
      ] : (
        one([for entry in container.environment : entry.value if entry.name == "AUTH_PROVIDER"]) == "auth0" &&
        alltrue([for entry in container.environment : entry.value == "" if contains(["KEYCLOAK_URL", "KEYCLOAK_EXTERNAL_URL"], entry.name)]) &&
        alltrue([for entry in container.secrets : !startswith(entry.name, "KEYCLOAK_")])
      )
    ]) && one([
      for entry in module.ecs_service_registry.container_definitions["registry"].container_definition.environment :
      entry.value if entry.name == "KEYCLOAK_ENABLED"
    ]) == "false"
    error_message = "Auth0 tasks must disable Keycloak URLs, routing and credential injection despite a stale domain."
  }
}

run "gateway_keycloak_preserves_external_domain" {
  command = plan
  module {
    source = "./modules/mcp-gateway"
  }
  variables {
    name                 = "provider-routing-module-test"
    vpc_id               = "vpc-0123456789abcdef0"
    private_subnet_ids   = ["subnet-0123456789abcdef0", "subnet-0123456789abcdef1"]
    public_subnet_ids    = ["subnet-0123456789abcdef2", "subnet-0123456789abcdef3"]
    ecs_cluster_arn      = "arn:aws:ecs:us-west-2:123456789012:cluster/example"
    ecs_cluster_name     = "example"
    keycloak_domain      = "external-keycloak.example.com"
    enable_observability = false
  }
  assert {
    condition = alltrue([
      for container in [
        module.ecs_service_auth.container_definitions["auth-server"].container_definition,
        module.ecs_service_registry.container_definitions["registry"].container_definition,
      ] : (
        one([for entry in container.environment : entry.value if entry.name == "AUTH_PROVIDER"]) == "keycloak" &&
        alltrue([for entry in container.environment : entry.value == "https://external-keycloak.example.com" if contains(["KEYCLOAK_URL", "KEYCLOAK_EXTERNAL_URL"], entry.name)]) &&
        contains([for entry in container.secrets : entry.name], "KEYCLOAK_CLIENT_SECRET")
      )
    ]) && one([
      for entry in module.ecs_service_registry.container_definitions["registry"].container_definition.environment :
      entry.value if entry.name == "KEYCLOAK_ENABLED"
    ]) == "true"
    error_message = "Keycloak tasks must preserve external Keycloak routing and client credentials."
  }
}
