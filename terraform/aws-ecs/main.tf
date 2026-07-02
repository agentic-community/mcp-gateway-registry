# MCP Gateway Registry - AWS ECS Deployment
# This Terraform configuration deploys the MCP Gateway to AWS ECS Fargate

terraform {
  # 1.2 required for `precondition` blocks on modules (used in the
  # mcp_gateway module to validate mongodb_connection_string* presence
  # for non-documentdb MongoDB backends, issue #955).
  required_version = ">= 1.2"

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
  vpc_id              = local.selected_vpc_id
  private_subnet_ids  = local.selected_private_subnet_ids
  public_subnet_ids   = local.selected_public_subnet_ids
  ingress_cidr_blocks = var.ingress_cidr_blocks

  # Internal auth-server URL (override for Cloud Map / Service Connect FQDNs)
  auth_server_url = var.auth_server_url

  # ALB logging
  alb_logs_bucket = aws_s3_bucket.alb_logs.id

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

  # Container images (core services default to public ECR)
  registry_image_uri    = var.registry_image_uri
  auth_server_image_uri = var.auth_server_image_uri
  mcpgw_image_uri       = var.mcpgw_image_uri

  # Demo servers (disabled by default)
  enable_demo_servers              = var.enable_demo_servers
  currenttime_image_uri            = var.currenttime_image_uri
  realserverfaketools_image_uri    = var.realserverfaketools_image_uri
  flight_booking_agent_image_uri   = var.flight_booking_agent_image_uri
  travel_assistant_agent_image_uri = var.travel_assistant_agent_image_uri

  # Service replicas
  mcpgw_replicas                  = var.mcpgw_replicas
  currenttime_replicas            = var.currenttime_replicas
  realserverfaketools_replicas    = var.realserverfaketools_replicas
  flight_booking_agent_replicas   = var.flight_booking_agent_replicas
  travel_assistant_agent_replicas = var.travel_assistant_agent_replicas

  # Auto-scaling configuration
  enable_autoscaling        = true
  autoscaling_min_capacity  = var.autoscaling_min_capacity
  autoscaling_max_capacity  = var.autoscaling_max_capacity
  autoscaling_target_cpu    = var.autoscaling_target_cpu
  autoscaling_target_memory = var.autoscaling_target_memory

  # Monitoring configuration
  enable_monitoring = var.enable_monitoring
  alarm_email       = var.alarm_email

  # Embeddings configuration
  embeddings_provider         = var.embeddings_provider
  embeddings_model_name       = var.embeddings_model_name
  embeddings_model_dimensions = var.embeddings_model_dimensions
  embeddings_aws_region       = var.embeddings_aws_region
  embeddings_api_key          = var.embeddings_api_key

  # Registration deduplication
  dedup_registration_hint_enabled = var.dedup_registration_hint_enabled
  dedup_score_threshold           = var.dedup_score_threshold
  dedup_max_suggestions           = var.dedup_max_suggestions

  # Keycloak admin credentials (for Management API)
  keycloak_admin_password = var.keycloak_admin_password

  # Session cookie security configuration
  session_cookie_secure = var.session_cookie_secure
  session_cookie_domain = var.session_cookie_domain
  bind_host             = var.bind_host

  # DocumentDB configuration
  storage_backend = var.storage_backend
  # Cluster endpoint + credentials secret are gated on is_aws_documentdb so
  # that external-MongoDB (Atlas / self-managed) deployments do not require
  # the AWS DocumentDB resources to exist (issue #955).
  documentdb_endpoint               = local.is_aws_documentdb ? aws_docdb_cluster.registry[0].endpoint : ""
  documentdb_database               = var.documentdb_database
  documentdb_namespace              = var.documentdb_namespace
  documentdb_use_tls                = var.documentdb_use_tls
  documentdb_use_iam                = var.documentdb_use_iam
  documentdb_credentials_secret_arn = local.is_aws_documentdb ? aws_secretsmanager_secret.documentdb_credentials[0].arn : ""

  # Optional full MongoDB connection string override (PR #947). See variable
  # docs in variables.tf. Leave both empty to use the DOCUMENTDB_* block above.
  mongodb_connection_string            = var.mongodb_connection_string
  mongodb_connection_string_secret_arn = var.mongodb_connection_string_secret_arn

  # Security scanning configuration
  security_scan_enabled         = var.security_scan_enabled
  security_scan_on_registration = var.security_scan_on_registration
  security_block_unsafe_servers = var.security_block_unsafe_servers
  security_analyzers            = var.security_analyzers
  security_scan_timeout         = var.security_scan_timeout
  security_add_pending_tag      = var.security_add_pending_tag

  # Microsoft Entra ID configuration
  entra_enabled                             = var.entra_enabled
  entra_tenant_id                           = var.entra_tenant_id
  entra_client_id                           = var.entra_client_id
  entra_client_secret                       = var.entra_client_secret
  entra_login_base_url                      = var.entra_login_base_url
  entra_graph_base_url                      = var.entra_graph_base_url
  idp_group_filter_prefix                   = var.idp_group_filter_prefix
  allowed_idp_groups                        = var.allowed_idp_groups
  idp_user_group_fallback_enabled_providers = var.idp_user_group_fallback_enabled_providers

  # Amazon Cognito configuration
  cognito_enabled       = var.cognito_enabled
  cognito_user_pool_id  = var.cognito_user_pool_id
  cognito_client_id     = var.cognito_client_id
  cognito_client_secret = var.cognito_client_secret
  cognito_domain        = var.cognito_domain

  # Okta configuration
  okta_enabled           = var.okta_enabled
  okta_domain            = var.okta_domain
  okta_client_id         = var.okta_client_id
  okta_client_secret     = var.okta_client_secret
  okta_m2m_client_id     = var.okta_m2m_client_id
  okta_m2m_client_secret = var.okta_m2m_client_secret
  okta_api_token         = var.okta_api_token
  okta_auth_server_id    = var.okta_auth_server_id

  # Auth0 configuration
  auth0_enabled              = var.auth0_enabled
  auth0_domain               = var.auth0_domain
  auth0_client_id            = var.auth0_client_id
  auth0_client_secret        = var.auth0_client_secret
  auth0_audience             = var.auth0_audience
  auth0_groups_claim         = var.auth0_groups_claim
  auth0_m2m_client_id        = var.auth0_m2m_client_id
  auth0_m2m_client_secret    = var.auth0_m2m_client_secret
  auth0_management_api_token = var.auth0_management_api_token

  # PingFederate configuration
  pingfederate_enabled            = var.pingfederate_enabled
  pingfederate_base_url           = var.pingfederate_base_url
  pingfederate_external_url       = var.pingfederate_external_url
  pingfederate_client_id          = var.pingfederate_client_id
  pingfederate_client_secret      = var.pingfederate_client_secret
  pingfederate_m2m_client_id      = var.pingfederate_m2m_client_id
  pingfederate_m2m_client_secret  = var.pingfederate_m2m_client_secret
  pingfederate_application_id_uri = var.pingfederate_application_id_uri
  pingfederate_groups_claim       = var.pingfederate_groups_claim

  # PingFederate Admin API (registry only)
  pf_admin_url  = var.pf_admin_url
  pf_admin_user = var.pf_admin_user
  pf_admin_pass = var.pf_admin_pass

  # Registry static token auth
  registry_static_token_auth_enabled = var.registry_static_token_auth_enabled
  registry_api_token                 = var.registry_api_token
  registry_api_keys                  = var.registry_api_keys
  max_tokens_per_user_per_hour       = var.max_tokens_per_user_per_hour

  # Registration webhook (issue #742)
  registration_webhook_url             = var.registration_webhook_url
  registration_webhook_auth_header     = var.registration_webhook_auth_header
  registration_webhook_auth_token      = var.registration_webhook_auth_token
  registration_webhook_timeout_seconds = var.registration_webhook_timeout_seconds
  registration_webhook_signing_secret  = var.registration_webhook_signing_secret
  registration_enforced_status         = var.registration_enforced_status

  # Agent batch API (issue #956)
  batch_worker_enabled                 = var.batch_worker_enabled
  batch_max_operations_per_job         = var.batch_max_operations_per_job
  batch_max_concurrent_jobs_per_user   = var.batch_max_concurrent_jobs_per_user
  batch_job_retention_days             = var.batch_job_retention_days
  batch_worker_poll_interval_seconds   = var.batch_worker_poll_interval_seconds
  batch_max_request_bytes              = var.batch_max_request_bytes
  batch_worker_lease_ttl_seconds       = var.batch_worker_lease_ttl_seconds
  batch_worker_lease_heartbeat_seconds = var.batch_worker_lease_heartbeat_seconds

  # Registration gate / admission control (issue #809)
  registration_gate_enabled              = var.registration_gate_enabled
  registration_gate_url                  = var.registration_gate_url
  registration_gate_auth_type            = var.registration_gate_auth_type
  registration_gate_auth_credential      = var.registration_gate_auth_credential
  registration_gate_auth_header_name     = var.registration_gate_auth_header_name
  registration_gate_timeout_seconds      = var.registration_gate_timeout_seconds
  registration_gate_max_retries          = var.registration_gate_max_retries
  registration_gate_oauth2_token_url     = var.registration_gate_oauth2_token_url
  registration_gate_oauth2_client_id     = var.registration_gate_oauth2_client_id
  registration_gate_oauth2_client_secret = var.registration_gate_oauth2_client_secret
  registration_gate_oauth2_scope         = var.registration_gate_oauth2_scope

  # M2M direct client registration (issue #851)
  m2m_direct_registration_enabled = var.m2m_direct_registration_enabled

  # Federation configuration (peer-to-peer registry sync)
  registry_id                          = var.registry_id
  federation_static_token_auth_enabled = var.federation_static_token_auth_enabled
  federation_static_token              = var.federation_static_token
  federation_encryption_key            = var.federation_encryption_key

  # AWS Agent Registry federation configuration
  aws_registry_federation_enabled = var.aws_registry_federation_enabled

  # ANS (Agent Name Service) configuration
  ans_integration_enabled            = var.ans_integration_enabled
  ans_api_endpoint                   = var.ans_api_endpoint
  ans_api_key                        = var.ans_api_key
  ans_api_secret                     = var.ans_api_secret
  ans_api_timeout_seconds            = var.ans_api_timeout_seconds
  ans_sync_interval_hours            = var.ans_sync_interval_hours
  ans_verification_cache_ttl_seconds = var.ans_verification_cache_ttl_seconds

  # Registry card configuration (federation metadata)
  registry_name              = var.registry_name
  registry_organization_name = var.registry_organization_name
  registry_description       = var.registry_description
  registry_contact_email     = var.registry_contact_email
  registry_contact_url       = var.registry_contact_url

  # Audit logging configuration
  audit_log_enabled  = var.audit_log_enabled
  audit_log_ttl_days = var.audit_log_ttl_days

  # Application log configuration
  app_log_centralized_enabled  = var.app_log_centralized_enabled
  app_log_centralized_ttl_days = var.app_log_centralized_ttl_days
  app_log_level                = var.app_log_level
  app_log_excluded_loggers     = var.app_log_excluded_loggers
  app_log_dir                  = var.app_log_dir
  app_log_file_format          = var.app_log_file_format
  app_log_console_format       = var.app_log_console_format

  # Tool-level access control (issue #1026)
  mcp_tools_list_filter_enabled = var.mcp_tools_list_filter_enabled
  mcp_proxy_max_body_bytes      = var.mcp_proxy_max_body_bytes
  mcp_proxy_timeout             = var.mcp_proxy_timeout
  tool_filter_audit_log_level   = var.tool_filter_audit_log_level

  internal_token_ttl_seconds    = var.internal_token_ttl_seconds
  internal_token_leeway_seconds = var.internal_token_leeway_seconds

  # Custom entity types (admin-defined, schema-driven catalog types)
  custom_entity_types_enabled   = var.custom_entity_types_enabled
  custom_type_cache_ttl_seconds = var.custom_type_cache_ttl_seconds
  max_custom_records_per_type   = var.max_custom_records_per_type
  max_custom_types              = var.max_custom_types

  # Update check (admin "newer release available" banner)
  update_check_enabled        = var.update_check_enabled
  update_check_interval_hours = var.update_check_interval_hours

  # Deployment mode configuration
  deployment_mode = var.deployment_mode
  registry_mode   = var.registry_mode

  # Internal/workshop deployment classification (telemetry labels; issue #1216)
  internal_only_deployment = var.internal_only_deployment
  internal_deployment_type = var.internal_deployment_type

  # Tab visibility overrides
  show_servers_tab         = var.show_servers_tab
  show_virtual_servers_tab = var.show_virtual_servers_tab
  show_skills_tab          = var.show_skills_tab
  show_agents_tab          = var.show_agents_tab

  # UI title override
  ui_title = var.ui_title

  # Observability configuration
  enable_observability      = var.enable_observability
  metrics_service_image_uri = var.metrics_service_image_uri
  grafana_image_uri         = var.grafana_image_uri
  grafana_admin_password    = var.grafana_admin_password

  otel_otlp_endpoint                                = var.otel_otlp_endpoint
  otel_exporter_otlp_headers                        = var.otel_exporter_otlp_headers
  otel_otlp_export_interval_ms                      = var.otel_otlp_export_interval_ms
  otel_exporter_otlp_metrics_temporality_preference = var.otel_exporter_otlp_metrics_temporality_preference

  # Telemetry configuration
  mcp_telemetry_disabled                   = var.mcp_telemetry_disabled
  mcp_telemetry_opt_out                    = var.mcp_telemetry_opt_out
  mcp_telemetry_heartbeat_interval_minutes = var.mcp_telemetry_heartbeat_interval_minutes
  telemetry_debug                          = var.telemetry_debug
  mcp_telemetry_imds_probe_disabled        = var.mcp_telemetry_imds_probe_disabled
  mcp_cloud_provider                       = var.mcp_cloud_provider

  # Demo server configuration
  disable_ai_registry_tools_server = var.disable_ai_registry_tools_server

  # GitHub private repo auth
  github_pat                 = var.github_pat
  github_app_id              = var.github_app_id
  github_app_installation_id = var.github_app_installation_id
  github_app_private_key     = var.github_app_private_key
  github_extra_hosts         = var.github_extra_hosts
  github_api_base_url        = var.github_api_base_url

  # MCP OAuth discovery / IDE login (PR #1224)
  mcp_advertised_scopes   = var.mcp_advertised_scopes
  ide_oauth_client_id     = var.ide_oauth_client_id
  ide_oauth_callback_port = var.ide_oauth_callback_port

  # Extra environment variables for custom configuration (Issue #1000)
  registry_extra_env    = var.registry_extra_env
  auth_server_extra_env = var.auth_server_extra_env
  mcpgw_extra_env       = var.mcpgw_extra_env

  # Per-user egress credential vault (third-party OBO). secrets-manager backend
  # on ECS; IAM grants are added in the module when enabled.
  egress_auth_enabled                = var.egress_auth_enabled
  egress_secret_store_backend        = var.egress_secret_store_backend
  egress_oauth_callback_base_url     = var.egress_oauth_callback_base_url
  egress_token_refresh_skew_seconds  = var.egress_token_refresh_skew_seconds
  egress_state_ttl_seconds           = var.egress_state_ttl_seconds
  egress_registry_internal_url       = var.egress_registry_internal_url
  egress_nginx_marker_secret         = var.egress_nginx_marker_secret
  egress_secrets_manager_kms_key_id  = var.egress_secrets_manager_kms_key_id
  egress_secrets_manager_path_prefix = var.egress_secrets_manager_path_prefix

  # Wait for S3 bucket policy to propagate (30s delay)
  # This prevents "Access Denied" errors when ALB tests write permissions
  depends_on = [time_sleep.wait_for_bucket_policy]
}

# When storage_backend is a non-documentdb MongoDB variant (mongodb-ce,
# mongodb, mongodb-atlas), Terraform does not provision AWS DocumentDB, so
# the registry MUST be given a MongoDB URI via mongodb_connection_string or
# mongodb_connection_string_secret_arn. Enforce at plan time rather than at
# apply/runtime. Implemented as a precondition on a terraform_data resource
# because Terraform 1.11 does not yet support `lifecycle` blocks directly on
# `module` references. Issue #955.
resource "terraform_data" "mongodb_backend_uri_required" {
  input = var.storage_backend

  lifecycle {
    precondition {
      condition = !local.uses_external_mongodb || (
        var.mongodb_connection_string != "" ||
        var.mongodb_connection_string_secret_arn != ""
      )
      error_message = join(" ", [
        "When storage_backend is one of mongodb-ce / mongodb / mongodb-atlas,",
        "you must set either mongodb_connection_string or",
        "mongodb_connection_string_secret_arn (prefer the _secret_arn form",
        "for credentials). See terraform.tfvars.example and",
        "docs/faq/configuring-mongodb-atlas-backend.md.",
      ])
    }
  }
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
