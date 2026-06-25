# ECS Services for MCP Gateway Registry

# ECS Service: Auth Server
#checkov:skip=CKV_TF_1:Module version is pinned via version constraint
module "ecs_service_auth" {
  source  = "terraform-aws-modules/ecs/aws//modules/service"
  version = "~> 6.0"

  name                     = "${local.name_prefix}-auth"
  cluster_arn              = var.ecs_cluster_arn
  cpu                      = tonumber(var.cpu)
  memory                   = tonumber(var.memory)
  desired_count            = var.enable_autoscaling ? var.autoscaling_min_capacity : var.auth_replicas
  enable_autoscaling       = var.enable_autoscaling
  autoscaling_min_capacity = var.autoscaling_min_capacity
  autoscaling_max_capacity = var.autoscaling_max_capacity
  autoscaling_policies = var.enable_autoscaling ? {
    cpu = {
      policy_type = "TargetTrackingScaling"
      target_tracking_scaling_policy_configuration = {
        predefined_metric_specification = {
          predefined_metric_type = "ECSServiceAverageCPUUtilization"
        }
        target_value = var.autoscaling_target_cpu
      }
    }
    memory = {
      policy_type = "TargetTrackingScaling"
      target_tracking_scaling_policy_configuration = {
        predefined_metric_specification = {
          predefined_metric_type = "ECSServiceAverageMemoryUtilization"
        }
        target_value = var.autoscaling_target_memory
      }
    }
  } : {}

  enable_execute_command = true

  requires_compatibilities = ["FARGATE", "EC2"]
  capacity_provider_strategy = {
    FARGATE = {
      capacity_provider = "FARGATE"
      weight            = 100
      base              = 1
    }
  }

  # Task roles
  create_task_exec_iam_role = true
  task_exec_iam_role_policies = {
    SecretsManagerAccess = aws_iam_policy.ecs_secrets_access.arn
    EcsExecTaskExecution = aws_iam_policy.ecs_exec_task_execution.arn
  }
  create_tasks_iam_role = true
  tasks_iam_role_policies = merge(
    {
      SecretsManagerAccess = aws_iam_policy.ecs_secrets_access.arn
      EcsExecTask          = aws_iam_policy.ecs_exec_task.arn
    },
    var.enable_observability ? {
      AMPRemoteWrite = aws_iam_policy.adot_amp_write[0].arn
    } : {}
  )

  # Enable Service Connect
  service_connect_configuration = {
    namespace = aws_service_discovery_private_dns_namespace.mcp.arn
    service = [{
      client_alias = {
        port     = 8888
        dns_name = "auth-server"
      }
      port_name      = "auth-server"
      discovery_name = "auth-server"
    }]
  }

  # Container definitions
  container_definitions = merge({
    auth-server = {
      cpu                    = var.enable_observability ? tonumber(var.cpu) - local.adot_sidecar_cpu : tonumber(var.cpu)
      memory                 = var.enable_observability ? tonumber(var.memory) - local.adot_sidecar_memory : tonumber(var.memory)
      essential              = true
      image                  = var.auth_server_image_uri
      versionConsistency     = "disabled"
      readonlyRootFilesystem = false

      portMappings = [
        {
          name          = "auth-server"
          containerPort = 18888
          protocol      = "tcp"
        }
      ]

      environment = concat([
        {
          name  = "REGISTRY_URL"
          value = "https://${var.domain_name}"
        },
        {
          name  = "BIND_HOST"
          value = var.bind_host
        },
        {
          name  = "AUTH_LISTEN_PORT"
          value = "18888"
        },
        {
          name  = "AUTH_SERVER_URL"
          value = var.auth_server_url
        },
        {
          name  = "AUTH_SERVER_EXTERNAL_URL"
          value = "https://${var.domain_name}"
        },
        {
          name  = "AWS_REGION"
          value = data.aws_region.current.id
        },
        {
          name  = "AUTH_PROVIDER"
          value = var.pingfederate_enabled ? "pingfederate" : (var.auth0_enabled ? "auth0" : (var.okta_enabled ? "okta" : (var.entra_enabled ? "entra" : (var.cognito_enabled ? "cognito" : (var.keycloak_domain != "" ? "keycloak" : "default")))))
        },
        # Amazon Cognito configuration (COGNITO_CLIENT_SECRET via Secrets Manager below)
        {
          name  = "COGNITO_ENABLED"
          value = tostring(var.cognito_enabled)
        },
        {
          name  = "COGNITO_USER_POOL_ID"
          value = var.cognito_user_pool_id
        },
        {
          name  = "COGNITO_CLIENT_ID"
          value = var.cognito_client_id
        },
        {
          name  = "COGNITO_DOMAIN"
          value = var.cognito_domain
        },
        # IDE OAuth login public client_id (PR #1224). The auth-server accepts
        # access tokens from this client (e.g. Cognito allowlist). Not a secret.
        {
          name  = "IDE_OAUTH_CLIENT_ID"
          value = var.ide_oauth_client_id
        },
        {
          name  = "KEYCLOAK_URL"
          value = var.keycloak_domain != "" ? "https://${var.keycloak_domain}" : ""
        },
        {
          name  = "KEYCLOAK_EXTERNAL_URL"
          value = var.keycloak_domain != "" ? "https://${var.keycloak_domain}" : ""
        },
        {
          name  = "KEYCLOAK_REALM"
          value = "mcp-gateway"
        },
        {
          name  = "KEYCLOAK_CLIENT_ID"
          value = "mcp-gateway-web"
        },
        {
          name  = "KEYCLOAK_M2M_CLIENT_ID"
          value = "mcp-gateway-m2m"
        },
        {
          name  = "ENTRA_ENABLED"
          value = tostring(var.entra_enabled)
        },
        {
          name  = "ENTRA_TENANT_ID"
          value = var.entra_tenant_id
        },
        {
          name  = "ENTRA_CLIENT_ID"
          value = var.entra_client_id
        },
        {
          name  = "ENTRA_LOGIN_BASE_URL"
          value = var.entra_login_base_url
        },
        {
          name  = "ENTRA_GRAPH_BASE_URL"
          value = var.entra_graph_base_url
        },
        {
          name  = "IDP_GROUP_FILTER_PREFIX"
          value = var.idp_group_filter_prefix
        },
        {
          name  = "ALLOWED_IDP_GROUPS"
          value = var.allowed_idp_groups
        },
        {
          name  = "IDP_USER_GROUP_FALLBACK_ENABLED_PROVIDERS"
          value = var.idp_user_group_fallback_enabled_providers
        },
        # Okta configuration
        {
          name  = "OKTA_ENABLED"
          value = tostring(var.okta_enabled)
        },
        {
          name  = "OKTA_DOMAIN"
          value = var.okta_domain
        },
        {
          name  = "OKTA_CLIENT_ID"
          value = var.okta_client_id
        },
        {
          name  = "OKTA_M2M_CLIENT_ID"
          value = var.okta_m2m_client_id
        },
        {
          name  = "OKTA_AUTH_SERVER_ID"
          value = var.okta_auth_server_id
        },
        {
          name  = "AUTH0_DOMAIN"
          value = var.auth0_domain
        },
        {
          name  = "AUTH0_CLIENT_ID"
          value = var.auth0_client_id
        },
        {
          name  = "AUTH0_AUDIENCE"
          value = var.auth0_audience
        },
        {
          name  = "AUTH0_GROUPS_CLAIM"
          value = var.auth0_groups_claim
        },
        {
          name  = "AUTH0_M2M_CLIENT_ID"
          value = var.auth0_m2m_client_id
        },
        {
          name  = "AUTH0_MANAGEMENT_API_TOKEN"
          value = var.auth0_management_api_token
        },
        {
          name  = "AUTH0_ENABLED"
          value = tostring(var.auth0_enabled)
        },
        # PingFederate configuration
        {
          name  = "PINGFEDERATE_ENABLED"
          value = tostring(var.pingfederate_enabled)
        },
        {
          name  = "PINGFEDERATE_BASE_URL"
          value = var.pingfederate_base_url
        },
        {
          name  = "PINGFEDERATE_EXTERNAL_URL"
          value = var.pingfederate_external_url
        },
        {
          name  = "PINGFEDERATE_CLIENT_ID"
          value = var.pingfederate_client_id
        },
        {
          name  = "PINGFEDERATE_M2M_CLIENT_ID"
          value = var.pingfederate_m2m_client_id
        },
        {
          name  = "PINGFEDERATE_APPLICATION_ID_URI"
          value = var.pingfederate_application_id_uri
        },
        {
          name  = "PINGFEDERATE_GROUPS_CLAIM"
          value = var.pingfederate_groups_claim
        },
        {
          name  = "SESSION_COOKIE_SECURE"
          value = tostring(var.session_cookie_secure)
        },
        {
          name  = "SESSION_COOKIE_DOMAIN"
          value = var.session_cookie_domain
        },
        {
          name  = "REGISTRY_STATIC_TOKEN_AUTH_ENABLED"
          value = tostring(var.registry_static_token_auth_enabled)
        },
        {
          name  = "REGISTRY_API_TOKEN"
          value = var.registry_api_token
        },
        {
          name  = "REGISTRY_API_KEYS"
          value = var.registry_api_keys
        },
        # M2M direct client registration (issue #851)
        {
          name  = "M2M_DIRECT_REGISTRATION_ENABLED"
          value = tostring(var.m2m_direct_registration_enabled)
        },
        # Federation configuration (peer-to-peer registry sync)
        {
          name  = "REGISTRY_ID"
          value = var.registry_id
        },
        {
          name  = "FEDERATION_STATIC_TOKEN_AUTH_ENABLED"
          value = tostring(var.federation_static_token_auth_enabled)
        },
        {
          name  = "FEDERATION_STATIC_TOKEN"
          value = var.federation_static_token
        },
        {
          name  = "FEDERATION_ENCRYPTION_KEY"
          value = var.federation_encryption_key
        },
        {
          name  = "ANS_INTEGRATION_ENABLED"
          value = tostring(var.ans_integration_enabled)
        },
        {
          name  = "ANS_API_ENDPOINT"
          value = var.ans_api_endpoint
        },
        {
          name  = "ANS_API_KEY"
          value = var.ans_api_key
        },
        {
          name  = "ANS_API_SECRET"
          value = var.ans_api_secret
        },
        {
          name  = "ANS_API_TIMEOUT_SECONDS"
          value = tostring(var.ans_api_timeout_seconds)
        },
        {
          name  = "ANS_SYNC_INTERVAL_HOURS"
          value = tostring(var.ans_sync_interval_hours)
        },
        {
          name  = "ANS_VERIFICATION_CACHE_TTL_SECONDS"
          value = tostring(var.ans_verification_cache_ttl_seconds)
        },
        {
          name  = "STORAGE_BACKEND"
          value = var.storage_backend
        },
        {
          name  = "DOCUMENTDB_HOST"
          value = var.documentdb_endpoint
        },
        {
          name  = "DOCUMENTDB_PORT"
          value = "27017"
        },
        {
          name  = "DOCUMENTDB_DATABASE"
          value = var.documentdb_database
        },
        {
          name  = "DOCUMENTDB_NAMESPACE"
          value = var.documentdb_namespace
        },
        {
          name  = "DOCUMENTDB_USE_TLS"
          value = tostring(var.documentdb_use_tls)
        },
        {
          name  = "DOCUMENTDB_USE_IAM"
          value = tostring(var.documentdb_use_iam)
        },
        {
          name  = "DOCUMENTDB_TLS_CA_FILE"
          value = "/app/certs/global-bundle.pem"
        },
        {
          name  = "AUDIT_LOG_ENABLED"
          value = tostring(var.audit_log_enabled)
        },
        {
          name  = "AUDIT_LOG_MONGODB_TTL_DAYS"
          value = tostring(var.audit_log_ttl_days)
        },
        {
          name  = "APP_LOG_CENTRALIZED_ENABLED"
          value = tostring(var.app_log_centralized_enabled)
        },
        {
          name  = "APP_LOG_CENTRALIZED_TTL_DAYS"
          value = tostring(var.app_log_centralized_ttl_days)
        },
        {
          name  = "APP_LOG_LEVEL"
          value = var.app_log_level
        },
        {
          name  = "APP_LOG_EXCLUDED_LOGGERS"
          value = var.app_log_excluded_loggers
        },
        {
          name  = "APP_LOG_DIR"
          value = var.app_log_dir
        },
        {
          name  = "APP_LOG_FILE_FORMAT"
          value = var.app_log_file_format
        },
        {
          name  = "APP_LOG_CONSOLE_FORMAT"
          value = var.app_log_console_format
        },
        # Tool-level access control (issue #1026)
        {
          name  = "MCP_TOOLS_LIST_FILTER_ENABLED"
          value = tostring(var.mcp_tools_list_filter_enabled)
        },
        {
          name  = "MCP_PROXY_MAX_BODY_BYTES"
          value = tostring(var.mcp_proxy_max_body_bytes)
        },
        {
          name  = "TOOL_FILTER_AUDIT_LOG_LEVEL"
          value = var.tool_filter_audit_log_level
        },
        {
          name  = "INTERNAL_TOKEN_TTL_SECONDS"
          value = tostring(var.internal_token_ttl_seconds)
        },
        {
          name  = "INTERNAL_TOKEN_LEEWAY_SECONDS"
          value = tostring(var.internal_token_leeway_seconds)
        },
        {
          name  = "METRICS_LEGACY_HTTP_POST"
          value = "false"
        },
        {
          name  = "OTEL_METRIC_EXPORT_INTERVAL_MS"
          value = "15000"
        },
        {
          name  = "OTEL_EXPORTER_PROMETHEUS_HOST"
          value = "0.0.0.0"
        },
        {
          name  = "OTEL_EXPORTER_PROMETHEUS_PORT"
          value = "9464"
        },
        {
          name  = "OTEL_EXPORTER_OTLP_ENDPOINT"
          value = var.enable_observability ? "http://localhost:4317" : ""
        },
        {
          name  = "OTEL_EXPORTER_OTLP_PROTOCOL"
          value = "grpc"
        }
        ],
        # PR #947: MongoDB connection string override (plain-text variant).
        # Only emitted when var.mongodb_connection_string is non-empty and a
        # Secrets Manager ARN was not provided. When empty, the registry
        # falls back to the DOCUMENTDB_* env vars above.
        var.mongodb_connection_string != "" && var.mongodb_connection_string_secret_arn == "" ? [
          {
            name  = "MONGODB_CONNECTION_STRING"
            value = var.mongodb_connection_string
          }
        ] : [],
        # Extra environment variables from user (Issue #1000)
        var.auth_server_extra_env
      )

      secrets = concat(
        [
          {
            name      = "SECRET_KEY"
            valueFrom = aws_secretsmanager_secret.secret_key.arn
          },
          {
            name      = "KEYCLOAK_CLIENT_SECRET"
            valueFrom = "${aws_secretsmanager_secret.keycloak_client_secret.arn}:client_secret::"
          },
          {
            name      = "KEYCLOAK_M2M_CLIENT_SECRET"
            valueFrom = "${aws_secretsmanager_secret.keycloak_m2m_client_secret.arn}:client_secret::"
          },
          {
            name      = "DOCUMENTDB_USERNAME"
            valueFrom = "${var.documentdb_credentials_secret_arn}:username::"
          },
          {
            name      = "DOCUMENTDB_PASSWORD"
            valueFrom = "${var.documentdb_credentials_secret_arn}:password::"
          }
        ],
        # PR #947: MongoDB connection string override (Secrets Manager variant).
        # Preferred when the URI contains credentials (avoids plain text in state).
        var.mongodb_connection_string_secret_arn != "" ? [
          {
            name      = "MONGODB_CONNECTION_STRING"
            valueFrom = var.mongodb_connection_string_secret_arn
          }
        ] : [],
        var.entra_enabled ? [
          {
            name      = "ENTRA_CLIENT_SECRET"
            valueFrom = aws_secretsmanager_secret.entra_client_secret[0].arn
          }
        ] : [],
        var.cognito_enabled ? [
          {
            name      = "COGNITO_CLIENT_SECRET"
            valueFrom = aws_secretsmanager_secret.cognito_client_secret[0].arn
          }
        ] : [],
        var.okta_enabled ? [
          {
            name      = "OKTA_CLIENT_SECRET"
            valueFrom = aws_secretsmanager_secret.okta_client_secret[0].arn
          },
          {
            name      = "OKTA_M2M_CLIENT_SECRET"
            valueFrom = aws_secretsmanager_secret.okta_m2m_client_secret[0].arn
          },
          {
            name      = "OKTA_API_TOKEN"
            valueFrom = aws_secretsmanager_secret.okta_api_token[0].arn
          }
        ] : [],
        var.auth0_enabled ? [
          {
            name      = "AUTH0_CLIENT_SECRET"
            valueFrom = aws_secretsmanager_secret.auth0_client_secret[0].arn
          },
          {
            name      = "AUTH0_M2M_CLIENT_SECRET"
            valueFrom = aws_secretsmanager_secret.auth0_m2m_client_secret[0].arn
          }
        ] : [],
        var.pingfederate_enabled ? [
          {
            name      = "PINGFEDERATE_CLIENT_SECRET"
            valueFrom = aws_secretsmanager_secret.pingfederate_client_secret[0].arn
          },
          {
            name      = "PINGFEDERATE_M2M_CLIENT_SECRET"
            valueFrom = aws_secretsmanager_secret.pingfederate_m2m_client_secret[0].arn
          },
        ] : [],
        var.enable_observability ? [
          {
            name      = "METRICS_API_KEY"
            valueFrom = aws_secretsmanager_secret.metrics_api_key[0].arn
          }
        ] : []
      )

      mountPoints = [
        {
          sourceVolume  = "mcp-logs"
          containerPath = "/app/logs"
          readOnly      = false
        },
        {
          sourceVolume  = "auth-config"
          containerPath = "/efs/auth_config"
          readOnly      = false
        }
      ]

      enable_cloudwatch_logging              = true
      cloudwatch_log_group_name              = "/ecs/${local.name_prefix}-auth-server"
      cloudwatch_log_group_retention_in_days = 30

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:18888/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
    },
    var.enable_observability ? {
      adot-collector = {
        cpu                    = local.adot_sidecar_cpu
        memory                 = local.adot_sidecar_memory
        essential              = false
        image                  = "public.ecr.aws/aws-observability/aws-otel-collector:latest"
        versionConsistency     = "disabled"
        readonlyRootFilesystem = false

        command = ["--config=env:AOT_CONFIG_CONTENT"]

        environment = [
          {
            name  = "AOT_CONFIG_CONTENT"
            value = local.adot_otlp_to_amp_config
          },
          {
            name  = "AWS_REGION"
            value = data.aws_region.current.id
          }
        ]

        enable_cloudwatch_logging              = true
        cloudwatch_log_group_name              = "/ecs/${local.name_prefix}-auth-adot"
        cloudwatch_log_group_retention_in_days = 30

        dependencies = [{
          containerName = "auth-server"
          condition     = "START"
        }]
      }
    } : {}
  )

  volume = {
    mcp-logs = {
      efs_volume_configuration = {
        file_system_id     = module.efs.id
        access_point_id    = module.efs.access_points["logs"].id
        transit_encryption = "ENABLED"
      }
    }
    auth-config = {
      efs_volume_configuration = {
        file_system_id     = module.efs.id
        access_point_id    = module.efs.access_points["auth_config"].id
        transit_encryption = "ENABLED"
      }
    }
  }

  # No ALB attachment: the public auth (:8888) listener/target-group was removed
  # (registry internal-auth hardening). The auth-server is reached only via
  # Service Connect (auth-server:8888 -> container 18888); external OAuth URLs
  # resolve to the nginx-fronted :443/:80.

  subnet_ids = var.private_subnet_ids
  # No ALB ingress rule needed: Service Connect proxy-to-proxy traffic is allowed
  # by the registry->auth rule defined separately below.
  security_group_egress_rules = {
    all = {
      ip_protocol = "-1"
      cidr_ipv4   = "0.0.0.0/0"
    }
  }

  tags = local.common_tags
}

# ECS Service: Registry (Main service with nginx, SSL, FAISS, models)
#checkov:skip=CKV_TF_1:Module version is pinned via version constraint
module "ecs_service_registry" {
  source  = "terraform-aws-modules/ecs/aws//modules/service"
  version = "~> 6.0"

  name                              = "${local.name_prefix}-registry"
  cluster_arn                       = var.ecs_cluster_arn
  cpu                               = tonumber(var.cpu)
  memory                            = tonumber(var.memory)
  desired_count                     = var.enable_autoscaling ? var.autoscaling_min_capacity : var.registry_replicas
  health_check_grace_period_seconds = 900
  enable_autoscaling                = var.enable_autoscaling
  autoscaling_min_capacity          = var.autoscaling_min_capacity
  autoscaling_max_capacity          = var.autoscaling_max_capacity
  autoscaling_policies = var.enable_autoscaling ? {
    cpu = {
      policy_type = "TargetTrackingScaling"
      target_tracking_scaling_policy_configuration = {
        predefined_metric_specification = {
          predefined_metric_type = "ECSServiceAverageCPUUtilization"
        }
        target_value = var.autoscaling_target_cpu
      }
    }
    memory = {
      policy_type = "TargetTrackingScaling"
      target_tracking_scaling_policy_configuration = {
        predefined_metric_specification = {
          predefined_metric_type = "ECSServiceAverageMemoryUtilization"
        }
        target_value = var.autoscaling_target_memory
      }
    }
  } : {}

  enable_execute_command = true

  requires_compatibilities = ["FARGATE", "EC2"]
  capacity_provider_strategy = {
    FARGATE = {
      capacity_provider = "FARGATE"
      weight            = 100
      base              = 1
    }
  }

  # Task roles
  create_task_exec_iam_role = true
  task_exec_iam_role_policies = {
    SecretsManagerAccess = aws_iam_policy.ecs_secrets_access.arn
    EcsExecTaskExecution = aws_iam_policy.ecs_exec_task_execution.arn
  }
  create_tasks_iam_role = true
  tasks_iam_role_policies = merge(
    {
      SecretsManagerAccess = aws_iam_policy.ecs_secrets_access.arn
      EcsExecTask          = aws_iam_policy.ecs_exec_task.arn
    },
    var.aws_registry_federation_enabled ? {
      BedrockAgentCoreAccess = aws_iam_policy.bedrock_agentcore_access[0].arn
    } : {},
    # Cognito read-only access for the registry IAM management UI (list groups/users)
    var.cognito_enabled ? {
      CognitoIamRead = aws_iam_policy.cognito_iam_read[0].arn
    } : {},
    # Issue #1122: per-task ADOT sidecar needs AMP remote-write permission
    var.enable_observability ? {
      AMPRemoteWrite = aws_iam_policy.adot_amp_write[0].arn
    } : {}
  )

  # Enable Service Connect
  service_connect_configuration = {
    namespace = aws_service_discovery_private_dns_namespace.mcp.arn
    service = [{
      client_alias = {
        port     = 8080 # Non-root nginx listens on 8080
        dns_name = "registry"
      }
      port_name      = "http"
      discovery_name = "registry"
    }]
  }

  # Container definitions
  container_definitions = merge({
    registry = {
      # Issue #1122: leave room for the ADOT sidecar so per-container cpu/memory
      # sums stay within the task-level limit when observability is enabled.
      cpu                    = var.enable_observability ? tonumber(var.cpu) - local.adot_sidecar_cpu : tonumber(var.cpu)
      memory                 = var.enable_observability ? tonumber(var.memory) - local.adot_sidecar_memory : tonumber(var.memory)
      essential              = true
      image                  = var.registry_image_uri
      versionConsistency     = "disabled"
      readonlyRootFilesystem = false

      portMappings = [
        {
          name          = "http"
          containerPort = 8080 # Non-root nginx listens on 8080
          protocol      = "tcp"
        },
        {
          name          = "https"
          containerPort = 8443 # Non-root nginx listens on 8443
          protocol      = "tcp"
        },
        {
          name          = "registry"
          containerPort = 7860
          protocol      = "tcp"
        }
      ]

      environment = concat([
        {
          name  = "REGISTRY_URL"
          value = var.domain_name != "" ? "https://${var.domain_name}" : "http://${module.alb.dns_name}"
        },
        {
          name  = "BIND_HOST"
          value = var.bind_host
        },
        {
          name  = "GATEWAY_ADDITIONAL_SERVER_NAMES"
          value = join(" ", compact([var.domain_name, var.additional_server_names]))
        },
        {
          name  = "EC2_PUBLIC_DNS"
          value = var.domain_name != "" ? var.domain_name : module.alb.dns_name
        },
        {
          name  = "AUTH_SERVER_URL"
          value = var.auth_server_url
        },
        {
          name  = "AUTH_SERVER_EXTERNAL_URL"
          value = var.domain_name != "" ? "https://${var.domain_name}" : "http://${module.alb.dns_name}"
        },
        {
          name  = "KEYCLOAK_URL"
          value = var.keycloak_domain != "" ? "https://${var.keycloak_domain}" : ""
        },
        {
          name  = "KEYCLOAK_ENABLED"
          value = var.keycloak_domain != "" ? "true" : "false"
        },
        {
          name  = "KEYCLOAK_REALM"
          value = "mcp-gateway"
        },
        {
          name  = "KEYCLOAK_CLIENT_ID"
          value = "mcp-gateway-web"
        },
        {
          name  = "AUTH_PROVIDER"
          value = var.pingfederate_enabled ? "pingfederate" : (var.auth0_enabled ? "auth0" : (var.okta_enabled ? "okta" : (var.entra_enabled ? "entra" : (var.cognito_enabled ? "cognito" : (var.keycloak_domain != "" ? "keycloak" : "default")))))
        },
        # Amazon Cognito configuration (COGNITO_CLIENT_SECRET via Secrets Manager below)
        {
          name  = "COGNITO_ENABLED"
          value = tostring(var.cognito_enabled)
        },
        {
          name  = "COGNITO_USER_POOL_ID"
          value = var.cognito_user_pool_id
        },
        {
          name  = "COGNITO_CLIENT_ID"
          value = var.cognito_client_id
        },
        {
          name  = "COGNITO_DOMAIN"
          value = var.cognito_domain
        },
        {
          name  = "ENTRA_ENABLED"
          value = tostring(var.entra_enabled)
        },
        {
          name  = "ENTRA_TENANT_ID"
          value = var.entra_tenant_id
        },
        {
          name  = "ENTRA_CLIENT_ID"
          value = var.entra_client_id
        },
        {
          name  = "ENTRA_LOGIN_BASE_URL"
          value = var.entra_login_base_url
        },
        {
          name  = "ENTRA_GRAPH_BASE_URL"
          value = var.entra_graph_base_url
        },
        {
          name  = "IDP_GROUP_FILTER_PREFIX"
          value = var.idp_group_filter_prefix
        },
        {
          name  = "ALLOWED_IDP_GROUPS"
          value = var.allowed_idp_groups
        },
        {
          name  = "IDP_USER_GROUP_FALLBACK_ENABLED_PROVIDERS"
          value = var.idp_user_group_fallback_enabled_providers
        },
        # Okta configuration
        {
          name  = "OKTA_ENABLED"
          value = tostring(var.okta_enabled)
        },
        {
          name  = "OKTA_DOMAIN"
          value = var.okta_domain
        },
        {
          name  = "OKTA_CLIENT_ID"
          value = var.okta_client_id
        },
        {
          name  = "OKTA_M2M_CLIENT_ID"
          value = var.okta_m2m_client_id
        },
        {
          name  = "OKTA_AUTH_SERVER_ID"
          value = var.okta_auth_server_id
        },
        {
          name  = "AUTH0_ENABLED"
          value = tostring(var.auth0_enabled)
        },
        {
          name  = "AUTH0_DOMAIN"
          value = var.auth0_domain
        },
        {
          name  = "AUTH0_CLIENT_ID"
          value = var.auth0_client_id
        },
        {
          name  = "AUTH0_AUDIENCE"
          value = var.auth0_audience
        },
        {
          name  = "AUTH0_GROUPS_CLAIM"
          value = var.auth0_groups_claim
        },
        {
          name  = "AUTH0_M2M_CLIENT_ID"
          value = var.auth0_m2m_client_id
        },
        {
          name  = "AUTH0_MANAGEMENT_API_TOKEN"
          value = var.auth0_management_api_token
        },
        # PingFederate configuration
        {
          name  = "PINGFEDERATE_ENABLED"
          value = tostring(var.pingfederate_enabled)
        },
        {
          name  = "PINGFEDERATE_BASE_URL"
          value = var.pingfederate_base_url
        },
        {
          name  = "PINGFEDERATE_EXTERNAL_URL"
          value = var.pingfederate_external_url
        },
        {
          name  = "PINGFEDERATE_CLIENT_ID"
          value = var.pingfederate_client_id
        },
        {
          name  = "PINGFEDERATE_M2M_CLIENT_ID"
          value = var.pingfederate_m2m_client_id
        },
        {
          name  = "PINGFEDERATE_APPLICATION_ID_URI"
          value = var.pingfederate_application_id_uri
        },
        {
          name  = "PINGFEDERATE_GROUPS_CLAIM"
          value = var.pingfederate_groups_claim
        },
        # PingFederate Admin API (registry only; PF_ADMIN_PASS via Secrets Manager below)
        {
          name  = "PF_ADMIN_URL"
          value = var.pf_admin_url
        },
        {
          name  = "PF_ADMIN_USER"
          value = var.pf_admin_user
        },
        {
          name  = "AWS_REGION"
          value = data.aws_region.current.id
        },
        {
          name  = "EMBEDDINGS_PROVIDER"
          value = var.embeddings_provider
        },
        {
          name  = "EMBEDDINGS_MODEL_NAME"
          value = var.embeddings_model_name
        },
        {
          name  = "EMBEDDINGS_MODEL_DIMENSIONS"
          value = tostring(var.embeddings_model_dimensions)
        },
        {
          name  = "EMBEDDINGS_AWS_REGION"
          value = var.embeddings_aws_region
        },
        # Registration deduplication. Advisory check; never blocks
        # registration. Reuses the embeddings model above.
        {
          name  = "DEDUP_REGISTRATION_HINT_ENABLED"
          value = tostring(var.dedup_registration_hint_enabled)
        },
        {
          name  = "DEDUP_SCORE_THRESHOLD"
          value = tostring(var.dedup_score_threshold)
        },
        {
          name  = "DEDUP_MAX_SUGGESTIONS"
          value = tostring(var.dedup_max_suggestions)
        },
        {
          name  = "SESSION_COOKIE_SECURE"
          value = tostring(var.session_cookie_secure)
        },
        {
          name  = "SESSION_COOKIE_DOMAIN"
          value = var.session_cookie_domain
        },
        {
          name  = "SECURITY_SCAN_ENABLED"
          value = tostring(var.security_scan_enabled)
        },
        {
          name  = "SECURITY_SCAN_ON_REGISTRATION"
          value = tostring(var.security_scan_on_registration)
        },
        {
          name  = "SECURITY_BLOCK_UNSAFE_SERVERS"
          value = tostring(var.security_block_unsafe_servers)
        },
        {
          name  = "SECURITY_ANALYZERS"
          value = var.security_analyzers
        },
        {
          name  = "SECURITY_SCAN_TIMEOUT"
          value = tostring(var.security_scan_timeout)
        },
        {
          name  = "SECURITY_ADD_PENDING_TAG"
          value = tostring(var.security_add_pending_tag)
        },
        {
          name  = "KEYCLOAK_ADMIN"
          value = "admin"
        },
        {
          name  = "STORAGE_BACKEND"
          value = var.storage_backend
        },
        {
          name  = "DOCUMENTDB_HOST"
          value = var.documentdb_endpoint
        },
        {
          name  = "DOCUMENTDB_PORT"
          value = "27017"
        },
        {
          name  = "DOCUMENTDB_DATABASE"
          value = var.documentdb_database
        },
        {
          name  = "DOCUMENTDB_NAMESPACE"
          value = var.documentdb_namespace
        },
        {
          name  = "DOCUMENTDB_USE_TLS"
          value = tostring(var.documentdb_use_tls)
        },
        {
          name  = "DOCUMENTDB_USE_IAM"
          value = tostring(var.documentdb_use_iam)
        },
        {
          name  = "DOCUMENTDB_TLS_CA_FILE"
          value = "/app/certs/global-bundle.pem"
        },
        {
          name  = "REGISTRY_ID"
          value = var.registry_id
        },
        {
          name  = "REGISTRY_NAME"
          value = var.registry_name
        },
        {
          name  = "REGISTRY_ORGANIZATION_NAME"
          value = var.registry_organization_name
        },
        {
          name  = "REGISTRY_DESCRIPTION"
          value = var.registry_description
        },
        {
          name  = "REGISTRY_CONTACT_EMAIL"
          value = var.registry_contact_email
        },
        {
          name  = "REGISTRY_CONTACT_URL"
          value = var.registry_contact_url
        },
        {
          name  = "FEDERATION_STATIC_TOKEN_AUTH_ENABLED"
          value = tostring(var.federation_static_token_auth_enabled)
        },
        {
          name  = "FEDERATION_STATIC_TOKEN"
          value = var.federation_static_token
        },
        {
          name  = "FEDERATION_ENCRYPTION_KEY"
          value = var.federation_encryption_key
        },
        # AWS Agent Registry federation configuration
        {
          name  = "AWS_REGISTRY_FEDERATION_ENABLED"
          value = tostring(var.aws_registry_federation_enabled)
        },
        {
          name  = "ANS_INTEGRATION_ENABLED"
          value = tostring(var.ans_integration_enabled)
        },
        {
          name  = "ANS_API_ENDPOINT"
          value = var.ans_api_endpoint
        },
        {
          name  = "ANS_API_KEY"
          value = var.ans_api_key
        },
        {
          name  = "ANS_API_SECRET"
          value = var.ans_api_secret
        },
        {
          name  = "ANS_API_TIMEOUT_SECONDS"
          value = tostring(var.ans_api_timeout_seconds)
        },
        {
          name  = "ANS_SYNC_INTERVAL_HOURS"
          value = tostring(var.ans_sync_interval_hours)
        },
        {
          name  = "ANS_VERIFICATION_CACHE_TTL_SECONDS"
          value = tostring(var.ans_verification_cache_ttl_seconds)
        },
        {
          name  = "AUDIT_LOG_ENABLED"
          value = tostring(var.audit_log_enabled)
        },
        {
          name  = "AUDIT_LOG_MONGODB_TTL_DAYS"
          value = tostring(var.audit_log_ttl_days)
        },
        {
          name  = "APP_LOG_CENTRALIZED_ENABLED"
          value = tostring(var.app_log_centralized_enabled)
        },
        {
          name  = "APP_LOG_CENTRALIZED_TTL_DAYS"
          value = tostring(var.app_log_centralized_ttl_days)
        },
        {
          name  = "APP_LOG_LEVEL"
          value = var.app_log_level
        },
        {
          name  = "APP_LOG_EXCLUDED_LOGGERS"
          value = var.app_log_excluded_loggers
        },
        {
          name  = "APP_LOG_DIR"
          value = var.app_log_dir
        },
        {
          name  = "APP_LOG_FILE_FORMAT"
          value = var.app_log_file_format
        },
        {
          name  = "APP_LOG_CONSOLE_FORMAT"
          value = var.app_log_console_format
        },
        # Tool-level access control (issue #1026)
        {
          name  = "MCP_TOOLS_LIST_FILTER_ENABLED"
          value = tostring(var.mcp_tools_list_filter_enabled)
        },
        {
          name  = "TOOL_FILTER_AUDIT_LOG_LEVEL"
          value = var.tool_filter_audit_log_level
        },
        # Registry-UI internal token verification (the /api/ hop; minted by auth-server)
        {
          name  = "INTERNAL_TOKEN_TTL_SECONDS"
          value = tostring(var.internal_token_ttl_seconds)
        },
        {
          name  = "INTERNAL_TOKEN_LEEWAY_SECONDS"
          value = tostring(var.internal_token_leeway_seconds)
        },
        # Custom entity types (admin-defined, schema-driven catalog types)
        {
          name  = "CUSTOM_ENTITY_TYPES_ENABLED"
          value = tostring(var.custom_entity_types_enabled)
        },
        {
          name  = "CUSTOM_TYPE_CACHE_TTL_SECONDS"
          value = tostring(var.custom_type_cache_ttl_seconds)
        },
        {
          name  = "MAX_CUSTOM_RECORDS_PER_TYPE"
          value = tostring(var.max_custom_records_per_type)
        },
        {
          name  = "MAX_CUSTOM_TYPES"
          value = tostring(var.max_custom_types)
        },
        # Update check (admin "newer release available" banner)
        {
          name  = "UPDATE_CHECK_ENABLED"
          value = tostring(var.update_check_enabled)
        },
        {
          name  = "UPDATE_CHECK_INTERVAL_HOURS"
          value = tostring(var.update_check_interval_hours)
        },
        # Override the scopes_supported array advertised in the gateway's
        # /.well-known/oauth-protected-resource document. Required when the
        # IdP's RFC 7591 DCR rejects scopes that don't exist as client-scope
        # objects in the realm — e.g. Keycloak rejects 'mcp-registry-admin'
        # because that name is a logical group in DocumentDB, not a Keycloak
        # client-scope. Defaults to standard OIDC scopes that always exist.
        {
          name  = "MCP_ADVERTISED_SCOPES"
          value = var.mcp_advertised_scopes
        },
        # Public OAuth client_id advertised in Connect configs for IDE login
        # (PR #1224). Public, not a secret; empty = static-token Connect config.
        {
          name  = "IDE_OAUTH_CLIENT_ID"
          value = var.ide_oauth_client_id
        },
        # Fixed OAuth callback port for IdPs that match redirect_uri literally
        # (Okta/Entra/Cognito). 0 = let the IDE pick (Keycloak/DCR).
        {
          name  = "IDE_OAUTH_CALLBACK_PORT"
          value = tostring(var.ide_oauth_callback_port)
        },
        {
          name  = "DEPLOYMENT_MODE"
          value = var.deployment_mode
        },
        {
          name  = "REGISTRY_MODE"
          value = var.registry_mode
        },
        # Internal/workshop deployment classification (telemetry labels; issue #1216)
        {
          name  = "INTERNAL_ONLY_DEPLOYMENT"
          value = tostring(var.internal_only_deployment)
        },
        {
          name  = "INTERNAL_DEPLOYMENT_TYPE"
          value = var.internal_deployment_type
        },
        {
          name  = "SHOW_SERVERS_TAB"
          value = tostring(var.show_servers_tab)
        },
        {
          name  = "SHOW_VIRTUAL_SERVERS_TAB"
          value = tostring(var.show_virtual_servers_tab)
        },
        {
          name  = "SHOW_SKILLS_TAB"
          value = tostring(var.show_skills_tab)
        },
        {
          name  = "SHOW_AGENTS_TAB"
          value = tostring(var.show_agents_tab)
        },
        {
          name  = "UI_TITLE"
          value = var.ui_title
        },
        {
          name  = "REGISTRY_STATIC_TOKEN_AUTH_ENABLED"
          value = tostring(var.registry_static_token_auth_enabled)
        },
        {
          name  = "REGISTRY_API_TOKEN"
          value = var.registry_api_token
        },
        {
          name  = "REGISTRY_API_KEYS"
          value = var.registry_api_keys
        },
        {
          name  = "MAX_TOKENS_PER_USER_PER_HOUR"
          value = tostring(var.max_tokens_per_user_per_hour)
        },
        # M2M direct client registration (issue #851)
        {
          name  = "M2M_DIRECT_REGISTRATION_ENABLED"
          value = tostring(var.m2m_direct_registration_enabled)
        },
        # Registration webhook (issue #742)
        {
          name  = "REGISTRATION_WEBHOOK_URL"
          value = var.registration_webhook_url
        },
        {
          name  = "REGISTRATION_WEBHOOK_AUTH_HEADER"
          value = var.registration_webhook_auth_header
        },
        {
          name  = "REGISTRATION_WEBHOOK_AUTH_TOKEN"
          value = var.registration_webhook_auth_token
        },
        {
          name  = "REGISTRATION_WEBHOOK_TIMEOUT_SECONDS"
          value = tostring(var.registration_webhook_timeout_seconds)
        },
        # Lifecycle workflow webhooks (issue #1330)
        {
          name  = "REGISTRATION_WEBHOOK_SIGNING_SECRET"
          value = var.registration_webhook_signing_secret
        },
        {
          name  = "REGISTRATION_ENFORCED_STATUS"
          value = var.registration_enforced_status
        },
        # Agent batch API (issue #956)
        {
          name  = "BATCH_WORKER_ENABLED"
          value = tostring(var.batch_worker_enabled)
        },
        {
          name  = "BATCH_MAX_OPERATIONS_PER_JOB"
          value = tostring(var.batch_max_operations_per_job)
        },
        {
          name  = "BATCH_MAX_CONCURRENT_JOBS_PER_USER"
          value = tostring(var.batch_max_concurrent_jobs_per_user)
        },
        {
          name  = "BATCH_JOB_RETENTION_DAYS"
          value = tostring(var.batch_job_retention_days)
        },
        {
          name  = "BATCH_WORKER_POLL_INTERVAL_SECONDS"
          value = tostring(var.batch_worker_poll_interval_seconds)
        },
        {
          name  = "BATCH_MAX_REQUEST_BYTES"
          value = tostring(var.batch_max_request_bytes)
        },
        {
          name  = "BATCH_WORKER_LEASE_TTL_SECONDS"
          value = tostring(var.batch_worker_lease_ttl_seconds)
        },
        {
          name  = "BATCH_WORKER_LEASE_HEARTBEAT_SECONDS"
          value = tostring(var.batch_worker_lease_heartbeat_seconds)
        },
        # Registration gate / admission control (issue #809)
        {
          name  = "REGISTRATION_GATE_ENABLED"
          value = tostring(var.registration_gate_enabled)
        },
        {
          name  = "REGISTRATION_GATE_URL"
          value = var.registration_gate_url
        },
        {
          name  = "REGISTRATION_GATE_AUTH_TYPE"
          value = var.registration_gate_auth_type
        },
        {
          name  = "REGISTRATION_GATE_AUTH_CREDENTIAL"
          value = var.registration_gate_auth_credential
        },
        {
          name  = "REGISTRATION_GATE_AUTH_HEADER_NAME"
          value = var.registration_gate_auth_header_name
        },
        {
          name  = "REGISTRATION_GATE_TIMEOUT_SECONDS"
          value = tostring(var.registration_gate_timeout_seconds)
        },
        {
          name  = "REGISTRATION_GATE_MAX_RETRIES"
          value = tostring(var.registration_gate_max_retries)
        },
        {
          name  = "REGISTRATION_GATE_OAUTH2_TOKEN_URL"
          value = var.registration_gate_oauth2_token_url
        },
        {
          name  = "REGISTRATION_GATE_OAUTH2_CLIENT_ID"
          value = var.registration_gate_oauth2_client_id
        },
        {
          name  = "REGISTRATION_GATE_OAUTH2_CLIENT_SECRET"
          value = var.registration_gate_oauth2_client_secret
        },
        {
          name  = "REGISTRATION_GATE_OAUTH2_SCOPE"
          value = var.registration_gate_oauth2_scope
        },
        # Telemetry configuration
        {
          name  = "MCP_TELEMETRY_DISABLED"
          value = var.mcp_telemetry_disabled
        },
        {
          name  = "MCP_TELEMETRY_OPT_OUT"
          value = var.mcp_telemetry_opt_out
        },
        {
          name  = "MCP_TELEMETRY_HEARTBEAT_INTERVAL_MINUTES"
          value = var.mcp_telemetry_heartbeat_interval_minutes
        },
        {
          name  = "TELEMETRY_DEBUG"
          value = var.telemetry_debug
        },
        {
          name  = "MCP_TELEMETRY_IMDS_PROBE_DISABLED"
          value = var.mcp_telemetry_imds_probe_disabled
        },
        {
          name  = "MCP_CLOUD_PROVIDER"
          value = var.mcp_cloud_provider
        },
        # Demo server configuration
        {
          name  = "DISABLE_AI_REGISTRY_TOOLS_SERVER"
          value = var.disable_ai_registry_tools_server
        },
        # Metrics pipeline (only wired when observability is enabled)
        {
          name  = "METRICS_SERVICE_URL"
          value = var.enable_observability ? "http://metrics-service:8890" : ""
        },
        # OTel-native metrics emission (Issue #1122)
        {
          name  = "METRICS_LEGACY_HTTP_POST"
          value = "false"
        },
        {
          name  = "OTEL_METRIC_EXPORT_INTERVAL_MS"
          value = "15000"
        },
        {
          name  = "OTEL_EXPORTER_OTLP_ENDPOINT"
          value = var.enable_observability ? "http://localhost:4317" : ""
        },
        {
          name  = "OTEL_EXPORTER_OTLP_PROTOCOL"
          value = "grpc"
        },
        {
          # Without an explicit service.name, the OTLP push path lands in AMP as
          # job="unknown_service" and the Grafana panels/job filters miss it
          # (issue #1326). Match the name used on Compose/Helm so all surfaces
          # agree.
          name  = "OTEL_SERVICE_NAME"
          value = "mcp-gateway-registry"
        },
        {
          # The ADOT sidecar pipeline is metrics-only, so exported traces are
          # rejected with UNIMPLEMENTED (hundreds of errors/run). Disable the
          # traces exporter; metrics still flow. Issue #1326.
          name  = "OTEL_TRACES_EXPORTER"
          value = "none"
        },
        # Service Connect namespace for FQDN alias injection in entrypoint.
        # Enables Python health checker to resolve both short names and FQDNs.
        {
          name  = "SERVICE_CONNECT_NAMESPACE"
          value = aws_service_discovery_private_dns_namespace.mcp.name
        },
        # GitHub private repo auth (SKILL.md fetching)
        {
          name  = "GITHUB_PAT"
          value = var.github_pat
        },
        {
          name  = "GITHUB_APP_ID"
          value = var.github_app_id
        },
        {
          name  = "GITHUB_APP_INSTALLATION_ID"
          value = var.github_app_installation_id
        },
        {
          name  = "GITHUB_APP_PRIVATE_KEY"
          value = var.github_app_private_key
        },
        {
          name  = "GITHUB_EXTRA_HOSTS"
          value = var.github_extra_hosts
        },
        {
          name  = "GITHUB_API_BASE_URL"
          value = var.github_api_base_url
        },
        ],
        # PR #947: MongoDB connection string override (plain-text variant).
        # Only emitted when var.mongodb_connection_string is non-empty and a
        # Secrets Manager ARN was not provided. When empty, the registry
        # falls back to the DOCUMENTDB_* env vars above.
        var.mongodb_connection_string != "" && var.mongodb_connection_string_secret_arn == "" ? [
          {
            name  = "MONGODB_CONNECTION_STRING"
            value = var.mongodb_connection_string
          }
        ] : [],
        # Extra environment variables from user (Issue #1000)
        var.registry_extra_env
      )

      secrets = concat(
        [
          {
            name      = "SECRET_KEY"
            valueFrom = aws_secretsmanager_secret.secret_key.arn
          },
          {
            name      = "KEYCLOAK_CLIENT_SECRET"
            valueFrom = "${aws_secretsmanager_secret.keycloak_client_secret.arn}:client_secret::"
          },
          {
            name      = "KEYCLOAK_M2M_CLIENT_SECRET"
            valueFrom = "${aws_secretsmanager_secret.keycloak_m2m_client_secret.arn}:client_secret::"
          },
          {
            name      = "KEYCLOAK_ADMIN_PASSWORD"
            valueFrom = aws_secretsmanager_secret.keycloak_admin_password.arn
          },
          {
            name      = "EMBEDDINGS_API_KEY"
            valueFrom = aws_secretsmanager_secret.embeddings_api_key.arn
          }
        ],
        # PR #947: MongoDB connection string override (Secrets Manager variant).
        # Preferred when the URI contains credentials (avoids plain text in state).
        var.mongodb_connection_string_secret_arn != "" ? [
          {
            name      = "MONGODB_CONNECTION_STRING"
            valueFrom = var.mongodb_connection_string_secret_arn
          }
        ] : [],
        var.storage_backend == "documentdb" ? [
          {
            name      = "DOCUMENTDB_USERNAME"
            valueFrom = "${var.documentdb_credentials_secret_arn}:username::"
          },
          {
            name      = "DOCUMENTDB_PASSWORD"
            valueFrom = "${var.documentdb_credentials_secret_arn}:password::"
          }
        ] : [],
        var.entra_enabled ? [
          {
            name      = "ENTRA_CLIENT_SECRET"
            valueFrom = aws_secretsmanager_secret.entra_client_secret[0].arn
          }
        ] : [],
        var.cognito_enabled ? [
          {
            name      = "COGNITO_CLIENT_SECRET"
            valueFrom = aws_secretsmanager_secret.cognito_client_secret[0].arn
          }
        ] : [],
        var.okta_enabled ? [
          {
            name      = "OKTA_CLIENT_SECRET"
            valueFrom = aws_secretsmanager_secret.okta_client_secret[0].arn
          },
          {
            name      = "OKTA_M2M_CLIENT_SECRET"
            valueFrom = aws_secretsmanager_secret.okta_m2m_client_secret[0].arn
          },
          {
            name      = "OKTA_API_TOKEN"
            valueFrom = aws_secretsmanager_secret.okta_api_token[0].arn
          }
        ] : [],
        var.auth0_enabled ? [
          {
            name      = "AUTH0_CLIENT_SECRET"
            valueFrom = aws_secretsmanager_secret.auth0_client_secret[0].arn
          },
          {
            name      = "AUTH0_M2M_CLIENT_SECRET"
            valueFrom = aws_secretsmanager_secret.auth0_m2m_client_secret[0].arn
          }
        ] : [],
        var.pingfederate_enabled ? [
          {
            name      = "PINGFEDERATE_CLIENT_SECRET"
            valueFrom = aws_secretsmanager_secret.pingfederate_client_secret[0].arn
          },
          {
            name      = "PINGFEDERATE_M2M_CLIENT_SECRET"
            valueFrom = aws_secretsmanager_secret.pingfederate_m2m_client_secret[0].arn
          },
          {
            name      = "PF_ADMIN_PASS"
            valueFrom = aws_secretsmanager_secret.pf_admin_pass[0].arn
          },
        ] : [],
        var.enable_observability ? [
          {
            name      = "METRICS_API_KEY"
            valueFrom = aws_secretsmanager_secret.metrics_api_key[0].arn
          }
        ] : []
      )

      # EFS volumes removed - registry now uses ephemeral storage and DocumentDB for persistence
      # Logs go to CloudWatch only
      mountPoints = []

      enable_cloudwatch_logging              = true
      cloudwatch_log_group_name              = "/ecs/${local.name_prefix}-registry"
      cloudwatch_log_group_retention_in_days = 30

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:7860/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
    },
    # Issue #1122: per-task ADOT collector sidecar for the registry service.
    var.enable_observability ? {
      adot-collector = {
        cpu                    = local.adot_sidecar_cpu
        memory                 = local.adot_sidecar_memory
        essential              = false
        image                  = "public.ecr.aws/aws-observability/aws-otel-collector:latest"
        versionConsistency     = "disabled"
        readonlyRootFilesystem = false

        command = ["--config=env:AOT_CONFIG_CONTENT"]

        environment = [
          {
            name  = "AOT_CONFIG_CONTENT"
            value = local.adot_otlp_to_amp_config
          },
          {
            name  = "AWS_REGION"
            value = data.aws_region.current.id
          }
        ]

        enable_cloudwatch_logging              = true
        cloudwatch_log_group_name              = "/ecs/${local.name_prefix}-registry-adot"
        cloudwatch_log_group_retention_in_days = 30

        dependencies = [{
          containerName = "registry"
          condition     = "START"
        }]
      }
    } : {}
  )

  # EFS volumes removed - registry uses ephemeral storage and DocumentDB for persistence
  volume = {}

  load_balancer = {
    http = {
      target_group_arn = module.alb.target_groups["registry"].arn
      container_name   = "registry"
      container_port   = 8080 # Non-root nginx listens on 8080
    }
    # The "gradio" (:7860) raw-app ALB attachment was removed (registry
    # internal-auth hardening). nginx (:8080) is the only ALB-fronted port; the
    # registry app is loopback-bound and reached over loopback inside the pod.
  }

  subnet_ids = var.private_subnet_ids
  security_group_ingress_rules = {
    alb_8080 = {
      description                  = "HTTP port (non-root nginx)"
      from_port                    = 8080
      to_port                      = 8080
      ip_protocol                  = "tcp"
      referenced_security_group_id = module.alb.security_group_id
    }
    alb_8443 = {
      description                  = "HTTPS port (non-root nginx)"
      from_port                    = 8443
      to_port                      = 8443
      ip_protocol                  = "tcp"
      referenced_security_group_id = module.alb.security_group_id
    }
    # The ALB->registry:7860 (gradio) ingress rule was removed with its listener.
    mcpgw_internal = {
      description                  = "HTTP from mcpgw for internal API calls (non-root nginx)"
      from_port                    = 8080
      to_port                      = 8080
      ip_protocol                  = "tcp"
      referenced_security_group_id = module.ecs_service_mcpgw.security_group_id
    }
  }
  security_group_egress_rules = {
    all = {
      ip_protocol = "-1"
      cidr_ipv4   = "0.0.0.0/0"
    }
  }

  tags = local.common_tags

}


# NOTE: the mcpgw->registry:7860 rule was removed (registry internal-auth
# hardening). mcpgw reaches the registry via nginx at registry:8080 (see the
# registry SG's mcpgw_internal rule); the raw app port 7860 is loopback-bound
# and no longer reachable cross-container.


# Allow registry to communicate with auth server on port 18888
# Service Connect proxy-to-proxy traffic uses containerPort (18888),
# not client_alias.port (8888).
resource "aws_vpc_security_group_ingress_rule" "registry_to_auth" {
  security_group_id            = module.ecs_service_auth.security_group_id
  referenced_security_group_id = module.ecs_service_registry.security_group_id
  from_port                    = 18888
  to_port                      = 18888
  ip_protocol                  = "tcp"
  description                  = "Allow registry to access auth server"

  tags = local.common_tags
}


# Allow auth server to communicate with mcpgw on port 8003
# Required for the mcp-proxy hop (PR #1026): auth server intercepts MCP
# requests to filter tools/list responses, then forwards to the upstream.
resource "aws_vpc_security_group_ingress_rule" "auth_to_mcpgw" {
  security_group_id            = module.ecs_service_mcpgw.security_group_id
  referenced_security_group_id = module.ecs_service_auth.security_group_id
  from_port                    = 8003
  to_port                      = 8003
  ip_protocol                  = "tcp"
  description                  = "Allow auth server mcp-proxy to reach mcpgw"

  tags = local.common_tags
}


# ECS Service: CurrentTime MCP Server (demo, opt-in via enable_demo_servers)
#checkov:skip=CKV_TF_1:Module version is pinned via version constraint
module "ecs_service_currenttime" {
  source  = "terraform-aws-modules/ecs/aws//modules/service"
  version = "~> 6.0"
  count   = var.enable_demo_servers ? 1 : 0

  name                     = "${local.name_prefix}-currenttime"
  cluster_arn              = var.ecs_cluster_arn
  cpu                      = "512"
  memory                   = "1024"
  desired_count            = var.enable_autoscaling ? var.autoscaling_min_capacity : var.currenttime_replicas
  enable_autoscaling       = var.enable_autoscaling
  autoscaling_min_capacity = var.autoscaling_min_capacity
  autoscaling_max_capacity = var.autoscaling_max_capacity
  autoscaling_policies = var.enable_autoscaling ? {
    cpu = {
      policy_type = "TargetTrackingScaling"
      target_tracking_scaling_policy_configuration = {
        predefined_metric_specification = {
          predefined_metric_type = "ECSServiceAverageCPUUtilization"
        }
        target_value = var.autoscaling_target_cpu
      }
    }
  } : {}

  enable_execute_command = true

  requires_compatibilities = ["FARGATE", "EC2"]
  capacity_provider_strategy = {
    FARGATE = {
      capacity_provider = "FARGATE"
      weight            = 100
      base              = 1
    }
  }

  create_task_exec_iam_role = true
  task_exec_iam_role_policies = {
    EcsExecTaskExecution = aws_iam_policy.ecs_exec_task_execution.arn
  }
  create_tasks_iam_role = true
  tasks_iam_role_policies = {
    EcsExecTask = aws_iam_policy.ecs_exec_task.arn
  }

  service_connect_configuration = {
    namespace = aws_service_discovery_private_dns_namespace.mcp.arn
    service = [{
      client_alias = {
        port     = 8000
        dns_name = "currenttime-server"
      }
      port_name      = "currenttime"
      discovery_name = "currenttime-server"
    }]
  }

  container_definitions = {
    currenttime-server = {
      cpu                    = 512
      memory                 = 1024
      essential              = true
      image                  = var.currenttime_image_uri
      versionConsistency     = "disabled"
      readonlyRootFilesystem = false

      portMappings = [
        {
          name          = "currenttime"
          containerPort = 8000
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "PORT"
          value = "8000"
        },
        {
          name  = "HOST"
          value = var.bind_host
        },
        {
          name  = "MCP_TRANSPORT"
          value = "streamable-http"
        }
      ]

      enable_cloudwatch_logging              = true
      cloudwatch_log_group_name              = "/ecs/${local.name_prefix}-currenttime"
      cloudwatch_log_group_retention_in_days = 30

      healthCheck = {
        command     = ["CMD-SHELL", "nc -z localhost 8000 || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 30
      }
    }
  }

  subnet_ids = var.private_subnet_ids
  security_group_ingress_rules = {
    service_connect = {
      description                  = "Service Connect from registry"
      from_port                    = 8000
      to_port                      = 8000
      ip_protocol                  = "tcp"
      referenced_security_group_id = module.ecs_service_registry.security_group_id
    }
  }
  security_group_egress_rules = {
    all = {
      ip_protocol = "-1"
      cidr_ipv4   = "0.0.0.0/0"
    }
  }

  tags = local.common_tags

}


# ECS Service: MCPGW MCP Server
#checkov:skip=CKV_TF_1:Module version is pinned via version constraint
module "ecs_service_mcpgw" {
  source  = "terraform-aws-modules/ecs/aws//modules/service"
  version = "~> 6.0"

  name                     = "${local.name_prefix}-mcpgw"
  cluster_arn              = var.ecs_cluster_arn
  cpu                      = "512"
  memory                   = "1024"
  desired_count            = var.mcpgw_replicas
  enable_autoscaling       = var.enable_autoscaling
  autoscaling_min_capacity = var.autoscaling_min_capacity
  autoscaling_max_capacity = var.autoscaling_max_capacity
  autoscaling_policies = var.enable_autoscaling ? {
    cpu = {
      policy_type = "TargetTrackingScaling"
      target_tracking_scaling_policy_configuration = {
        predefined_metric_specification = {
          predefined_metric_type = "ECSServiceAverageCPUUtilization"
        }
        target_value = var.autoscaling_target_cpu
      }
    }
  } : {}

  enable_execute_command = true

  requires_compatibilities = ["FARGATE", "EC2"]
  capacity_provider_strategy = {
    FARGATE = {
      capacity_provider = "FARGATE"
      weight            = 100
      base              = 1
    }
  }

  create_task_exec_iam_role = true
  task_exec_iam_role_policies = {
    SecretsManagerAccess = aws_iam_policy.ecs_secrets_access.arn
    EcsExecTaskExecution = aws_iam_policy.ecs_exec_task_execution.arn
  }
  create_tasks_iam_role = true
  tasks_iam_role_policies = merge(
    {
      SecretsManagerAccess = aws_iam_policy.ecs_secrets_access.arn
      EcsExecTask          = aws_iam_policy.ecs_exec_task.arn
    },
    # Issue #1122: per-task ADOT sidecar needs AMP remote-write permission
    var.enable_observability ? {
      AMPRemoteWrite = aws_iam_policy.adot_amp_write[0].arn
    } : {}
  )

  service_connect_configuration = {
    namespace = aws_service_discovery_private_dns_namespace.mcp.arn
    service = [{
      client_alias = {
        port     = 8003
        dns_name = "mcpgw-server"
      }
      port_name      = "mcpgw"
      discovery_name = "mcpgw-server"
    }]
  }

  container_definitions = merge({
    mcpgw-server = {
      # Issue #1122: leave room for the ADOT sidecar (cpu/memory sums must
      # not exceed the task-level limit of 512/1024).
      cpu                    = var.enable_observability ? 512 - local.adot_sidecar_cpu : 512
      memory                 = var.enable_observability ? 1024 - local.adot_sidecar_memory : 1024
      essential              = true
      image                  = var.mcpgw_image_uri
      versionConsistency     = "disabled"
      readonlyRootFilesystem = false

      portMappings = [
        {
          name          = "mcpgw"
          containerPort = 8003
          protocol      = "tcp"
        }
      ]

      environment = concat([
        {
          name  = "PORT"
          value = "8003"
        },
        {
          name  = "HOST"
          value = var.bind_host
        },
        {
          name  = "REGISTRY_BASE_URL"
          value = "http://registry:8080"
        },
        {
          name  = "REGISTRY_EXTERNAL_URL"
          value = "https://${var.domain_name}"
        },
        {
          name  = "REGISTRY_USERNAME"
          value = "admin"
        },
        # Application log configuration (issue #987). Matches the Docker
        # Compose wiring for the mcpgw container; keeps behavior consistent
        # across deployment surfaces. Values are consumed by
        # servers/mcpgw/logging_setup.py.
        {
          name  = "APP_LOG_DIR"
          value = var.app_log_dir
        },
        {
          name  = "APP_LOG_FILE_FORMAT"
          value = var.app_log_file_format
        },
        {
          name  = "APP_LOG_CONSOLE_FORMAT"
          value = var.app_log_console_format
        },
        {
          name  = "APP_LOG_LEVEL"
          value = var.app_log_level
        },
        # OTel-native metrics emission (Issue #1122)
        {
          name  = "METRICS_LEGACY_HTTP_POST"
          value = "false"
        },
        {
          name  = "OTEL_METRIC_EXPORT_INTERVAL_MS"
          value = "15000"
        },
        {
          name  = "OTEL_EXPORTER_OTLP_ENDPOINT"
          value = var.enable_observability ? "http://localhost:4317" : ""
        },
        {
          name  = "OTEL_EXPORTER_OTLP_PROTOCOL"
          value = "grpc"
        },
        {
          # Without an explicit service.name, the OTLP push path lands in AMP as
          # job="unknown_service" (issue #1326). Match the name used on
          # Compose/Helm so all surfaces agree.
          name  = "OTEL_SERVICE_NAME"
          value = "mcp-mcpgw"
        },
        {
          # The ADOT sidecar pipeline is metrics-only; exported traces are
          # rejected with UNIMPLEMENTED. Disable the traces exporter. Issue #1326.
          name  = "OTEL_TRACES_EXPORTER"
          value = "none"
        }
        ],
        # Extra environment variables from user (Issue #1000)
        [
          for env in var.mcpgw_extra_env : {
            name  = env.name
            value = env.value
          }
        ]
      )

      secrets = []

      mountPoints = [
        {
          sourceVolume  = "mcpgw-data"
          containerPath = "/app/data"
          readOnly      = false
        }
      ]

      enable_cloudwatch_logging              = true
      cloudwatch_log_group_name              = "/ecs/${local.name_prefix}-mcpgw"
      cloudwatch_log_group_retention_in_days = 30

      healthCheck = {
        command     = ["CMD-SHELL", "nc -z localhost 8003 || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 30
      }
    }
    },
    # Issue #1122: per-task ADOT collector sidecar for the mcpgw service.
    var.enable_observability ? {
      adot-collector = {
        cpu                    = local.adot_sidecar_cpu
        memory                 = local.adot_sidecar_memory
        essential              = false
        image                  = "public.ecr.aws/aws-observability/aws-otel-collector:latest"
        versionConsistency     = "disabled"
        readonlyRootFilesystem = false

        command = ["--config=env:AOT_CONFIG_CONTENT"]

        environment = [
          {
            name  = "AOT_CONFIG_CONTENT"
            value = local.adot_otlp_to_amp_config
          },
          {
            name  = "AWS_REGION"
            value = data.aws_region.current.id
          }
        ]

        enable_cloudwatch_logging              = true
        cloudwatch_log_group_name              = "/ecs/${local.name_prefix}-mcpgw-adot"
        cloudwatch_log_group_retention_in_days = 30

        dependencies = [{
          containerName = "mcpgw-server"
          condition     = "START"
        }]
      }
    } : {}
  )

  volume = {
    mcpgw-data = {
      efs_volume_configuration = {
        file_system_id     = module.efs.id
        access_point_id    = module.efs.access_points["mcpgw_data"].id
        transit_encryption = "ENABLED"
      }
    }
  }

  subnet_ids = var.private_subnet_ids
  security_group_ingress_rules = {
    service_connect = {
      description                  = "Service Connect from registry"
      from_port                    = 8003
      to_port                      = 8003
      ip_protocol                  = "tcp"
      referenced_security_group_id = module.ecs_service_registry.security_group_id
    }
  }
  security_group_egress_rules = {
    all = {
      ip_protocol = "-1"
      cidr_ipv4   = "0.0.0.0/0"
    }
  }

  tags = local.common_tags

}


# ECS Service: RealServerFakeTools MCP Server (demo, opt-in via enable_demo_servers)
#checkov:skip=CKV_TF_1:Module version is pinned via version constraint
module "ecs_service_realserverfaketools" {
  source  = "terraform-aws-modules/ecs/aws//modules/service"
  version = "~> 6.0"
  count   = var.enable_demo_servers ? 1 : 0

  name                     = "${local.name_prefix}-realserverfaketools"
  cluster_arn              = var.ecs_cluster_arn
  cpu                      = "512"
  memory                   = "1024"
  desired_count            = var.enable_autoscaling ? var.autoscaling_min_capacity : var.realserverfaketools_replicas
  enable_autoscaling       = var.enable_autoscaling
  autoscaling_min_capacity = var.autoscaling_min_capacity
  autoscaling_max_capacity = var.autoscaling_max_capacity
  autoscaling_policies = var.enable_autoscaling ? {
    cpu = {
      policy_type = "TargetTrackingScaling"
      target_tracking_scaling_policy_configuration = {
        predefined_metric_specification = {
          predefined_metric_type = "ECSServiceAverageCPUUtilization"
        }
        target_value = var.autoscaling_target_cpu
      }
    }
  } : {}

  enable_execute_command = true

  requires_compatibilities = ["FARGATE", "EC2"]
  capacity_provider_strategy = {
    FARGATE = {
      capacity_provider = "FARGATE"
      weight            = 100
      base              = 1
    }
  }

  create_task_exec_iam_role = true
  task_exec_iam_role_policies = {
    EcsExecTaskExecution = aws_iam_policy.ecs_exec_task_execution.arn
  }
  create_tasks_iam_role = true
  tasks_iam_role_policies = {
    EcsExecTask = aws_iam_policy.ecs_exec_task.arn
  }

  service_connect_configuration = {
    namespace = aws_service_discovery_private_dns_namespace.mcp.arn
    service = [{
      client_alias = {
        port     = 8002
        dns_name = "realserverfaketools-server"
      }
      port_name      = "realserverfaketools"
      discovery_name = "realserverfaketools-server"
    }]
  }

  container_definitions = {
    realserverfaketools-server = {
      cpu                    = 512
      memory                 = 1024
      essential              = true
      image                  = var.realserverfaketools_image_uri
      versionConsistency     = "disabled"
      readonlyRootFilesystem = false

      portMappings = [
        {
          name          = "realserverfaketools"
          containerPort = 8002
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "PORT"
          value = "8002"
        },
        {
          name  = "MCP_TRANSPORT"
          value = "streamable-http"
        }
      ]

      enable_cloudwatch_logging              = true
      cloudwatch_log_group_name              = "/ecs/${local.name_prefix}-realserverfaketools"
      cloudwatch_log_group_retention_in_days = 30

      healthCheck = {
        command     = ["CMD-SHELL", "nc -z localhost 8002 || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 30
      }
    }
  }

  subnet_ids = var.private_subnet_ids
  security_group_ingress_rules = {
    service_connect = {
      description                  = "Service Connect from registry"
      from_port                    = 8002
      to_port                      = 8002
      ip_protocol                  = "tcp"
      referenced_security_group_id = module.ecs_service_registry.security_group_id
    }
  }
  security_group_egress_rules = {
    all = {
      ip_protocol = "-1"
      cidr_ipv4   = "0.0.0.0/0"
    }
  }

  tags = local.common_tags

}


# ECS Service: Flight Booking A2A Agent (demo, opt-in via enable_demo_servers)
#checkov:skip=CKV_TF_1:Module version is pinned via version constraint
module "ecs_service_flight_booking_agent" {
  source  = "terraform-aws-modules/ecs/aws//modules/service"
  version = "~> 6.0"
  count   = var.enable_demo_servers ? 1 : 0

  name                     = "${local.name_prefix}-flight-booking-agent"
  cluster_arn              = var.ecs_cluster_arn
  cpu                      = "512"
  memory                   = "1024"
  desired_count            = var.enable_autoscaling ? var.autoscaling_min_capacity : var.flight_booking_agent_replicas
  enable_autoscaling       = var.enable_autoscaling
  autoscaling_min_capacity = var.autoscaling_min_capacity
  autoscaling_max_capacity = var.autoscaling_max_capacity
  autoscaling_policies = var.enable_autoscaling ? {
    cpu = {
      policy_type = "TargetTrackingScaling"
      target_tracking_scaling_policy_configuration = {
        predefined_metric_specification = {
          predefined_metric_type = "ECSServiceAverageCPUUtilization"
        }
        target_value = var.autoscaling_target_cpu
      }
    }
  } : {}

  enable_execute_command = true

  requires_compatibilities = ["FARGATE", "EC2"]
  capacity_provider_strategy = {
    FARGATE = {
      capacity_provider = "FARGATE"
      weight            = 100
      base              = 1
    }
  }

  create_task_exec_iam_role = true
  task_exec_iam_role_policies = {
    EcsExecTaskExecution = aws_iam_policy.ecs_exec_task_execution.arn
  }
  create_tasks_iam_role = true
  tasks_iam_role_policies = {
    EcsExecTask = aws_iam_policy.ecs_exec_task.arn
  }

  service_connect_configuration = {
    namespace = aws_service_discovery_private_dns_namespace.mcp.arn
    service = [{
      client_alias = {
        port     = 9000
        dns_name = "flight-booking-agent"
      }
      port_name      = "flight-booking"
      discovery_name = "flight-booking-agent"
    }]
  }

  container_definitions = {
    flight-booking-agent = {
      cpu                    = 512
      memory                 = 1024
      essential              = true
      image                  = var.flight_booking_agent_image_uri
      versionConsistency     = "disabled"
      readonlyRootFilesystem = false

      portMappings = [
        {
          name          = "flight-booking"
          containerPort = 9000
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "AWS_REGION"
          value = data.aws_region.current.id
        },
        {
          name  = "AWS_DEFAULT_REGION"
          value = data.aws_region.current.id
        }
      ]

      enable_cloudwatch_logging              = true
      cloudwatch_log_group_name              = "/ecs/${local.name_prefix}-flight-booking-agent"
      cloudwatch_log_group_retention_in_days = 30

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:9000/ping || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  }

  subnet_ids = var.private_subnet_ids
  security_group_ingress_rules = {
    service_connect = {
      description = "Service Connect - A2A protocol"
      from_port   = 9000
      to_port     = 9000
      ip_protocol = "tcp"
      cidr_ipv4   = data.aws_vpc.vpc.cidr_block
    }
  }
  security_group_egress_rules = {
    all = {
      ip_protocol = "-1"
      cidr_ipv4   = "0.0.0.0/0"
    }
  }

  tags = local.common_tags

}


# ECS Service: Travel Assistant A2A Agent (demo, opt-in via enable_demo_servers)
#checkov:skip=CKV_TF_1:Module version is pinned via version constraint
module "ecs_service_travel_assistant_agent" {
  source  = "terraform-aws-modules/ecs/aws//modules/service"
  version = "~> 6.0"
  count   = var.enable_demo_servers ? 1 : 0

  name                     = "${local.name_prefix}-travel-assistant-agent"
  cluster_arn              = var.ecs_cluster_arn
  cpu                      = "512"
  memory                   = "1024"
  desired_count            = var.enable_autoscaling ? var.autoscaling_min_capacity : var.travel_assistant_agent_replicas
  enable_autoscaling       = var.enable_autoscaling
  autoscaling_min_capacity = var.autoscaling_min_capacity
  autoscaling_max_capacity = var.autoscaling_max_capacity
  autoscaling_policies = var.enable_autoscaling ? {
    cpu = {
      policy_type = "TargetTrackingScaling"
      target_tracking_scaling_policy_configuration = {
        predefined_metric_specification = {
          predefined_metric_type = "ECSServiceAverageCPUUtilization"
        }
        target_value = var.autoscaling_target_cpu
      }
    }
  } : {}

  enable_execute_command = true

  requires_compatibilities = ["FARGATE", "EC2"]
  capacity_provider_strategy = {
    FARGATE = {
      capacity_provider = "FARGATE"
      weight            = 100
      base              = 1
    }
  }

  create_task_exec_iam_role = true
  task_exec_iam_role_policies = {
    EcsExecTaskExecution = aws_iam_policy.ecs_exec_task_execution.arn
  }
  create_tasks_iam_role = true
  tasks_iam_role_policies = {
    EcsExecTask = aws_iam_policy.ecs_exec_task.arn
  }

  service_connect_configuration = {
    namespace = aws_service_discovery_private_dns_namespace.mcp.arn
    service = [{
      client_alias = {
        port     = 9000
        dns_name = "travel-assistant-agent"
      }
      port_name      = "travel-assistant"
      discovery_name = "travel-assistant-agent"
    }]
  }

  container_definitions = {
    travel-assistant-agent = {
      cpu                    = 512
      memory                 = 1024
      essential              = true
      image                  = var.travel_assistant_agent_image_uri
      versionConsistency     = "disabled"
      readonlyRootFilesystem = false

      portMappings = [
        {
          name          = "travel-assistant"
          containerPort = 9000
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "AWS_REGION"
          value = data.aws_region.current.id
        },
        {
          name  = "AWS_DEFAULT_REGION"
          value = data.aws_region.current.id
        }
      ]

      enable_cloudwatch_logging              = true
      cloudwatch_log_group_name              = "/ecs/${local.name_prefix}-travel-assistant-agent"
      cloudwatch_log_group_retention_in_days = 30

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:9000/ping || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  }

  subnet_ids = var.private_subnet_ids
  security_group_ingress_rules = {
    service_connect = {
      description = "Service Connect - A2A protocol"
      from_port   = 9000
      to_port     = 9000
      ip_protocol = "tcp"
      cidr_ipv4   = data.aws_vpc.vpc.cidr_block
    }
  }
  security_group_egress_rules = {
    all = {
      ip_protocol = "-1"
      cidr_ipv4   = "0.0.0.0/0"
    }
  }

  tags = local.common_tags

}
