# Observability Pipeline for MCP Gateway Registry
# Creates: AMP workspace, metrics-service (with ADOT sidecar), Grafana OSS
# All resources gated by var.enable_observability

# =============================================================================
# AMAZON MANAGED PROMETHEUS (AMP)
# =============================================================================

resource "aws_prometheus_workspace" "mcp" {
  count = var.enable_observability ? 1 : 0
  alias = "${local.name_prefix}-prometheus"
  tags  = local.common_tags
}

locals {
  amp_remote_write_endpoint = var.enable_observability ? "${aws_prometheus_workspace.mcp[0].prometheus_endpoint}api/v1/remote_write" : ""
  amp_query_endpoint        = var.enable_observability ? aws_prometheus_workspace.mcp[0].prometheus_endpoint : ""

  # ADOT collector configuration (embedded YAML)
  # ADOT runs as a sidecar in the metrics-service task, scrapes localhost:9465
  adot_config = var.enable_observability ? yamlencode({
    receivers = {
      prometheus = {
        config = {
          global = {
            scrape_interval = "15s"
          }
          scrape_configs = [
            {
              job_name        = "mcp-metrics-service"
              scrape_interval = "15s"
              metrics_path    = "/metrics"
              static_configs = [
                {
                  targets = ["localhost:9465"]
                }
              ]
            }
          ]
        }
      }
    }
    exporters = {
      prometheusremotewrite = {
        endpoint = local.amp_remote_write_endpoint
        auth = {
          authenticator = "sigv4auth"
        }
      }
    }
    extensions = {
      sigv4auth = {
        region = data.aws_region.current.id
      }
      health_check = {
        endpoint = "0.0.0.0:13133"
      }
    }
    service = {
      extensions = ["sigv4auth", "health_check"]
      pipelines = {
        metrics = {
          receivers = ["prometheus"]
          exporters = ["prometheusremotewrite"]
        }
      }
    }
  }) : ""
}


# =============================================================================
# METRICS-SERVICE ECS SERVICE
# =============================================================================

module "ecs_service_metrics" {
  count   = var.enable_observability ? 1 : 0
  source  = "terraform-aws-modules/ecs/aws//modules/service"
  version = "~> 6.0"

  name        = "${local.name_prefix}-metrics-service"
  cluster_arn = var.ecs_cluster_arn
  cpu         = 512
  memory      = 1024

  desired_count      = 1
  enable_autoscaling = false

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
  tasks_iam_role_policies = {
    SecretsManagerAccess = aws_iam_policy.ecs_secrets_access.arn
    EcsExecTask          = aws_iam_policy.ecs_exec_task.arn
    AMPRemoteWrite       = aws_iam_policy.adot_amp_write[0].arn
  }

  service_connect_configuration = {
    namespace = aws_service_discovery_private_dns_namespace.mcp.arn
    service = [
      {
        client_alias = {
          port     = 8890
          dns_name = "metrics-service"
        }
        port_name      = "metrics-api"
        discovery_name = "metrics-service"
      }
    ]
  }

  container_definitions = {
    metrics-service = {
      cpu                    = 256
      memory                 = 512
      essential              = true
      image                  = var.metrics_service_image_uri
      versionConsistency     = "disabled"
      readonlyRootFilesystem = false

      portMappings = [
        {
          name          = "metrics-api"
          containerPort = 8890
          protocol      = "tcp"
        },
        {
          name          = "prometheus-exporter"
          containerPort = 9465
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "METRICS_SERVICE_HOST"
          value = "0.0.0.0"
        },
        {
          name  = "METRICS_SERVICE_PORT"
          value = "8890"
        },
        {
          name  = "OTEL_SERVICE_NAME"
          value = "mcp-metrics-service"
        },
        {
          name  = "OTEL_PROMETHEUS_ENABLED"
          value = "true"
        },
        {
          name  = "OTEL_PROMETHEUS_PORT"
          value = "9465"
        },
        {
          name  = "METRICS_RATE_LIMIT"
          value = "1000"
        },
        {
          name  = "HISTOGRAM_BUCKET_BOUNDARIES"
          value = "0.005,0.01,0.025,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0,30.0,60.0,120.0,300.0"
        },
        {
          name  = "SQLITE_DB_PATH"
          value = "/tmp/metrics.db"
        },
        {
          name  = "METRICS_RETENTION_DAYS"
          value = "7"
        }
      ]

      secrets = [
        {
          name      = "METRICS_API_KEY_REGISTRY"
          valueFrom = aws_secretsmanager_secret.metrics_api_key[0].arn
        },
        {
          name      = "METRICS_API_KEY_AUTH"
          valueFrom = aws_secretsmanager_secret.metrics_api_key[0].arn
        },
        {
          name      = "METRICS_API_KEY_MCPGW"
          valueFrom = aws_secretsmanager_secret.metrics_api_key[0].arn
        }
      ]

      enable_cloudwatch_logging              = true
      cloudwatch_log_group_name              = "/ecs/${local.name_prefix}-metrics-service"
      cloudwatch_log_group_retention_in_days = 30

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8890/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 30
      }
    }

    # ADOT collector sidecar — scrapes metrics-service on localhost:9465
    # and remote-writes to AMP. Co-located to avoid Service Connect DNS
    # resolution issues (HTTP-type Cloud Map services have no Route53 records).
    adot-collector = {
      cpu                    = 256
      memory                 = 512
      essential              = false
      image                  = "public.ecr.aws/aws-observability/aws-otel-collector:latest"
      versionConsistency     = "disabled"
      readonlyRootFilesystem = false

      command = ["--config=env:AOT_CONFIG_CONTENT"]

      environment = [
        {
          name  = "AOT_CONFIG_CONTENT"
          value = local.adot_config
        },
        {
          name  = "AWS_REGION"
          value = data.aws_region.current.id
        }
      ]

      enable_cloudwatch_logging              = true
      cloudwatch_log_group_name              = "/ecs/${local.name_prefix}-adot-collector"
      cloudwatch_log_group_retention_in_days = 30

      dependencies = [{
        containerName = "metrics-service"
        condition     = "HEALTHY"
      }]
    }
  }

  subnet_ids = var.private_subnet_ids
  security_group_ingress_rules = {
    auth_8890 = {
      description                  = "Metrics API from auth-server"
      from_port                    = 8890
      to_port                      = 8890
      ip_protocol                  = "tcp"
      referenced_security_group_id = module.ecs_service_auth.security_group_id
    }
    registry_8890 = {
      description                  = "Metrics API from registry"
      from_port                    = 8890
      to_port                      = 8890
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


# =============================================================================
# ADOT COLLECTOR — IAM POLICY FOR AMP REMOTE WRITE
# =============================================================================
# ADOT runs as a sidecar in the metrics-service task (above).
# This policy is attached to the metrics-service task role.

resource "aws_iam_policy" "adot_amp_write" {
  count       = var.enable_observability ? 1 : 0
  name_prefix = "${local.name_prefix}-adot-amp-write-"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "aps:RemoteWrite",
          "aps:GetSeries",
          "aps:GetLabels",
          "aps:GetMetricMetadata"
        ]
        Resource = aws_prometheus_workspace.mcp[0].arn
      }
    ]
  })

  tags = local.common_tags
}


# =============================================================================
# GRAFANA OSS ECS SERVICE
# =============================================================================

# IAM policy for Grafana to query AMP
resource "aws_iam_policy" "grafana_amp_query" {
  count       = var.enable_observability ? 1 : 0
  name_prefix = "${local.name_prefix}-grafana-amp-query-"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "aps:QueryMetrics",
          "aps:GetSeries",
          "aps:GetLabels",
          "aps:GetMetricMetadata"
        ]
        Resource = aws_prometheus_workspace.mcp[0].arn
      }
    ]
  })

  tags = local.common_tags
}

# ALB target group for Grafana
resource "aws_lb_target_group" "grafana" {
  count       = var.enable_observability ? 1 : 0
  name_prefix = "graf-"
  port        = 3000
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  deregistration_delay = 5

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/api/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }

  tags = local.common_tags
}

# ALB listener rule for Grafana (path-based routing on /grafana/*)
resource "aws_lb_listener_rule" "grafana_http" {
  count        = var.enable_observability ? 1 : 0
  listener_arn = module.alb.listeners["http"].arn
  priority     = 15

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.grafana[0].arn
  }

  condition {
    path_pattern {
      values = ["/grafana", "/grafana/*"]
    }
  }

  tags = local.common_tags
}

resource "aws_lb_listener_rule" "grafana_https" {
  count        = var.enable_observability && var.enable_https ? 1 : 0
  listener_arn = module.alb.listeners["https"].arn
  priority     = 15

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.grafana[0].arn
  }

  condition {
    path_pattern {
      values = ["/grafana", "/grafana/*"]
    }
  }

  tags = local.common_tags
}

module "ecs_service_grafana" {
  count   = var.enable_observability ? 1 : 0
  source  = "terraform-aws-modules/ecs/aws//modules/service"
  version = "~> 6.0"

  name        = "${local.name_prefix}-grafana"
  cluster_arn = var.ecs_cluster_arn
  cpu         = 512
  memory      = 1024

  desired_count      = 1
  enable_autoscaling = false

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
    EcsExecTask      = aws_iam_policy.ecs_exec_task.arn
    GrafanaAMPAccess = aws_iam_policy.grafana_amp_query[0].arn
  }

  service_connect_configuration = {
    namespace = aws_service_discovery_private_dns_namespace.mcp.arn
    service = [{
      client_alias = {
        port     = 3000
        dns_name = "grafana"
      }
      port_name      = "grafana-http"
      discovery_name = "grafana"
    }]
  }

  container_definitions = {
    grafana = {
      cpu                    = 512
      memory                 = 1024
      essential              = true
      image                  = var.grafana_image_uri
      versionConsistency     = "disabled"
      readonlyRootFilesystem = false

      portMappings = [
        {
          name          = "grafana-http"
          containerPort = 3000
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "AWS_REGION"
          value = data.aws_region.current.id
        },
        {
          name  = "GF_AUTH_SIGV4_AUTH_ENABLED"
          value = "true"
        },
        {
          name  = "GF_AWS_ALLOWED_AUTH_PROVIDERS"
          value = "default,ec2_iam_role"
        },
        {
          name  = "AMP_ENDPOINT"
          value = local.amp_query_endpoint
        },
        {
          name  = "GF_SERVER_ROOT_URL"
          value = "%(protocol)s://%(domain)s/grafana/"
        },
        {
          name  = "GF_SERVER_SERVE_FROM_SUB_PATH"
          value = "true"
        },
        {
          name  = "GF_AUTH_ANONYMOUS_ENABLED"
          value = "false"
        },
        {
          name  = "GF_AUTH_ANONYMOUS_ORG_ROLE"
          value = "Viewer"
        },
        {
          name  = "GF_AUTH_DISABLE_LOGIN_FORM"
          value = "false"
        },
        {
          name  = "GF_SECURITY_ADMIN_PASSWORD"
          value = var.grafana_admin_password
        },
        {
          name  = "GF_LOG_MODE"
          value = "console"
        },
        {
          name  = "GF_LOG_LEVEL"
          value = "info"
        },
        {
          name  = "GF_DASHBOARDS_MIN_REFRESH_INTERVAL"
          value = "10s"
        }
      ]

      enable_cloudwatch_logging              = true
      cloudwatch_log_group_name              = "/ecs/${local.name_prefix}-grafana"
      cloudwatch_log_group_retention_in_days = 30

      healthCheck = {
        command     = ["CMD-SHELL", "wget -q --spider http://localhost:3000/api/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 30
      }
    }
  }

  load_balancer = {
    grafana = {
      target_group_arn = aws_lb_target_group.grafana[0].arn
      container_name   = "grafana"
      container_port   = 3000
    }
  }

  subnet_ids = var.private_subnet_ids
  security_group_ingress_rules = {
    alb_3000 = {
      description                  = "Grafana HTTP from ALB"
      from_port                    = 3000
      to_port                      = 3000
      ip_protocol                  = "tcp"
      referenced_security_group_id = module.alb.security_group_id
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
