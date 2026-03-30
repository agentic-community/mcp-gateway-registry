# Secrets Manager resources for MCP Gateway Registry

#
# KMS Key for Application Secrets Encryption
#
resource "aws_kms_key" "secrets" {
  description             = "KMS key for MCP Gateway application secrets encryption"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow ECS Task Execution Role to Decrypt"
        Effect = "Allow"
        Principal = {
          AWS = "*"
        }
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "aws:PrincipalAccount" = data.aws_caller_identity.current.account_id
          }
          StringLike = {
            "aws:PrincipalArn" = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/*task-exec*"
          }
        }
      },
      {
        Sid    = "Allow CloudWatch Logs"
        Effect = "Allow"
        Principal = {
          Service = "logs.${data.aws_region.current.name}.amazonaws.com"
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:CreateGrant",
          "kms:DescribeKey"
        ]
        Resource = "*"
        Condition = {
          ArnLike = {
            "kms:EncryptionContext:aws:logs:arn" = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:*"
          }
        }
      }
    ]
  })

  tags = merge(
    local.common_tags,
    {
      Name      = "${local.name_prefix}-secrets-key"
      Component = "secrets"
    }
  )
}

resource "aws_kms_alias" "secrets" {
  name          = "alias/${local.name_prefix}-secrets"
  target_key_id = aws_kms_key.secrets.key_id
}

# Random passwords for application secrets

resource "random_password" "secret_key" {
  length  = 64
  special = true
}

# Core application secrets

resource "aws_secretsmanager_secret" "secret_key" {
  name_prefix             = "${local.name_prefix}-secret-key-"
  description             = "Secret key for MCP Gateway Registry"
  recovery_window_in_days = 0
  kms_key_id              = aws_kms_key.secrets.id
  tags                    = local.common_tags
}

resource "aws_secretsmanager_secret_version" "secret_key" {
  secret_id     = aws_secretsmanager_secret.secret_key.id
  secret_string = random_password.secret_key.result
}

# Keycloak client secrets (created with placeholder, updated by init-keycloak.sh)
resource "aws_secretsmanager_secret" "keycloak_client_secret" {
  name                    = "mcp-gateway-keycloak-client-secret"
  description             = "Keycloak web client secret (updated by init-keycloak.sh after deployment)"
  recovery_window_in_days = 0
  kms_key_id              = aws_kms_key.secrets.id
  tags                    = local.common_tags
}

resource "aws_secretsmanager_secret_version" "keycloak_client_secret" {
  secret_id = aws_secretsmanager_secret.keycloak_client_secret.id
  secret_string = jsonencode({
    client_secret = "placeholder-will-be-updated-by-init-script"
  })

  lifecycle {
    ignore_changes = [secret_string]
  }
}

resource "aws_secretsmanager_secret" "keycloak_m2m_client_secret" {
  name                    = "mcp-gateway-keycloak-m2m-client-secret"
  description             = "Keycloak M2M client secret (updated by init-keycloak.sh after deployment)"
  recovery_window_in_days = 0
  kms_key_id              = aws_kms_key.secrets.id
  tags                    = local.common_tags
}

resource "aws_secretsmanager_secret_version" "keycloak_m2m_client_secret" {
  secret_id = aws_secretsmanager_secret.keycloak_m2m_client_secret.id
  secret_string = jsonencode({
    client_secret = "placeholder-will-be-updated-by-init-script"
  })

  lifecycle {
    ignore_changes = [secret_string]
  }
}


# Keycloak admin password secret (for Management API operations)
resource "aws_secretsmanager_secret" "keycloak_admin_password" {
  name_prefix             = "${local.name_prefix}-keycloak-admin-password-"
  description             = "Keycloak admin password for Management API user/group operations"
  recovery_window_in_days = 0
  kms_key_id              = aws_kms_key.secrets.id
  tags                    = local.common_tags
}

resource "aws_secretsmanager_secret_version" "keycloak_admin_password" {
  secret_id     = aws_secretsmanager_secret.keycloak_admin_password.id
  secret_string = var.keycloak_admin_password
}


# Embeddings API key secret (optional - only needed for LiteLLM provider)
resource "aws_secretsmanager_secret" "embeddings_api_key" {
  name_prefix             = "${local.name_prefix}-embeddings-api-key-"
  description             = "API key for embeddings provider (OpenAI, Anthropic, etc.)"
  recovery_window_in_days = 0
  kms_key_id              = aws_kms_key.secrets.id
  tags                    = local.common_tags
}

resource "aws_secretsmanager_secret_version" "embeddings_api_key" {
  secret_id     = aws_secretsmanager_secret.embeddings_api_key.id
  secret_string = var.embeddings_api_key != "" ? var.embeddings_api_key : "not-configured"

  lifecycle {
    ignore_changes = [secret_string]
  }
}


# Microsoft Entra ID client secret (for OAuth and IAM operations)
resource "aws_secretsmanager_secret" "entra_client_secret" {
  count = var.entra_enabled ? 1 : 0

  name_prefix             = "${local.name_prefix}-entra-client-secret-"
  description             = "Microsoft Entra ID client secret for OAuth authentication and IAM operations"
  recovery_window_in_days = 0
  kms_key_id              = aws_kms_key.secrets.id
  tags                    = local.common_tags
}

resource "aws_secretsmanager_secret_version" "entra_client_secret" {
  count = var.entra_enabled ? 1 : 0

  secret_id     = aws_secretsmanager_secret.entra_client_secret[0].id
  secret_string = var.entra_client_secret

  lifecycle {
    ignore_changes = [secret_string]
  }
}


# Okta client secret (for OAuth authentication)
resource "aws_secretsmanager_secret" "okta_client_secret" {
  count = var.okta_enabled ? 1 : 0

  name_prefix             = "${local.name_prefix}-okta-client-secret-"
  description             = "Okta client secret for OAuth authentication"
  recovery_window_in_days = 0
  kms_key_id              = aws_kms_key.secrets.id
  tags                    = local.common_tags
}

resource "aws_secretsmanager_secret_version" "okta_client_secret" {
  count = var.okta_enabled ? 1 : 0

  secret_id     = aws_secretsmanager_secret.okta_client_secret[0].id
  secret_string = var.okta_client_secret

  lifecycle {
    ignore_changes = [secret_string]
  }
}


# Okta M2M client secret (for service account operations)
resource "aws_secretsmanager_secret" "okta_m2m_client_secret" {
  count = var.okta_enabled ? 1 : 0

  name_prefix             = "${local.name_prefix}-okta-m2m-client-secret-"
  description             = "Okta M2M client secret for service account operations"
  recovery_window_in_days = 0
  kms_key_id              = aws_kms_key.secrets.id
  tags                    = local.common_tags
}

resource "aws_secretsmanager_secret_version" "okta_m2m_client_secret" {
  count = var.okta_enabled ? 1 : 0

  secret_id     = aws_secretsmanager_secret.okta_m2m_client_secret[0].id
  secret_string = var.okta_m2m_client_secret

  lifecycle {
    ignore_changes = [secret_string]
  }
}


# Okta API token (for management operations)
resource "aws_secretsmanager_secret" "okta_api_token" {
  count = var.okta_enabled ? 1 : 0

  name_prefix             = "${local.name_prefix}-okta-api-token-"
  description             = "Okta API token for IAM management operations"
  recovery_window_in_days = 0
  kms_key_id              = aws_kms_key.secrets.id
  tags                    = local.common_tags
}

resource "aws_secretsmanager_secret_version" "okta_api_token" {
  count = var.okta_enabled ? 1 : 0

  secret_id     = aws_secretsmanager_secret.okta_api_token[0].id
  secret_string = var.okta_api_token

  lifecycle {
    ignore_changes = [secret_string]
  }
}


# =============================================================================
# AUTH0 SECRETS
# =============================================================================

# Auth0 client secret (for OAuth authentication)
# checkov:skip=CKV_AWS_149: Rotation managed externally in Auth0 dashboard, not applicable for IdP client secrets
resource "aws_secretsmanager_secret" "auth0_client_secret" {
  count = var.auth0_enabled ? 1 : 0

  name_prefix             = "${local.name_prefix}-auth0-client-secret-"
  description             = "Auth0 client secret for OAuth authentication"
  recovery_window_in_days = 0
  kms_key_id              = aws_kms_key.secrets.id
  tags                    = local.common_tags
}

resource "aws_secretsmanager_secret_version" "auth0_client_secret" {
  count = var.auth0_enabled ? 1 : 0

  secret_id     = aws_secretsmanager_secret.auth0_client_secret[0].id
  secret_string = var.auth0_client_secret

  lifecycle {
    ignore_changes = [secret_string]
  }
}


# Auth0 M2M client secret (for IAM Management operations)
# checkov:skip=CKV_AWS_149: Rotation managed externally in Auth0 dashboard, not applicable for IdP client secrets
resource "aws_secretsmanager_secret" "auth0_m2m_client_secret" {
  count = var.auth0_enabled ? 1 : 0

  name_prefix             = "${local.name_prefix}-auth0-m2m-client-secret-"
  description             = "Auth0 M2M client secret for IAM Management operations"
  recovery_window_in_days = 0
  kms_key_id              = aws_kms_key.secrets.id
  tags                    = local.common_tags
}

resource "aws_secretsmanager_secret_version" "auth0_m2m_client_secret" {
  count = var.auth0_enabled ? 1 : 0

  secret_id     = aws_secretsmanager_secret.auth0_m2m_client_secret[0].id
  secret_string = var.auth0_m2m_client_secret

  lifecycle {
    ignore_changes = [secret_string]
  }
}


# Metrics API key (for metrics-service authentication)
resource "random_password" "metrics_api_key" {
  count   = var.enable_observability ? 1 : 0
  length  = 48
  special = false
}

resource "aws_secretsmanager_secret" "metrics_api_key" {
  count = var.enable_observability ? 1 : 0

  name_prefix             = "${local.name_prefix}-metrics-api-key-"
  description             = "API key for metrics-service (shared by auth-server and registry)"
  recovery_window_in_days = 0
  kms_key_id              = aws_kms_key.secrets.id
  tags                    = local.common_tags
}

resource "aws_secretsmanager_secret_version" "metrics_api_key" {
  count = var.enable_observability ? 1 : 0

  secret_id     = aws_secretsmanager_secret.metrics_api_key[0].id
  secret_string = random_password.metrics_api_key[0].result
}


# OTLP exporter headers (e.g., dd-api-key=xxx for Datadog)
# Only created when observability is enabled AND an OTLP endpoint is configured
resource "aws_secretsmanager_secret" "otlp_exporter_headers" {
  count = var.enable_observability && var.otel_otlp_endpoint != "" ? 1 : 0

  name_prefix             = "${local.name_prefix}-otlp-exporter-headers-"
  description             = "OTLP exporter authentication headers (e.g., Datadog API key)"
  recovery_window_in_days = 0
  kms_key_id              = aws_kms_key.secrets.id
  tags                    = local.common_tags
}

resource "aws_secretsmanager_secret_version" "otlp_exporter_headers" {
  count = var.enable_observability && var.otel_otlp_endpoint != "" ? 1 : 0

  secret_id     = aws_secretsmanager_secret.otlp_exporter_headers[0].id
  secret_string = var.otel_exporter_otlp_headers
}
