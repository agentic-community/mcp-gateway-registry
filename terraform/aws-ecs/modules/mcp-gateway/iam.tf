# IAM resources for MCP Gateway Registry ECS services

# IAM policy for ECS tasks to access Secrets Manager
resource "aws_iam_policy" "ecs_secrets_access" {
  name_prefix = "${local.name_prefix}-ecs-secrets-"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = concat(
          [
            aws_secretsmanager_secret.secret_key.arn,
            aws_secretsmanager_secret.nginx_marker_secret.arn,
            aws_secretsmanager_secret.keycloak_client_secret.arn,
            aws_secretsmanager_secret.keycloak_m2m_client_secret.arn,
            aws_secretsmanager_secret.embeddings_api_key.arn,
            aws_secretsmanager_secret.embeddings_idp_client_secret.arn,
            aws_secretsmanager_secret.keycloak_admin_password.arn
          ],
          var.documentdb_credentials_secret_arn != "" ? [var.documentdb_credentials_secret_arn] : [],
          var.entra_enabled ? [aws_secretsmanager_secret.entra_client_secret[0].arn] : [],
          var.cognito_enabled ? [aws_secretsmanager_secret.cognito_client_secret[0].arn] : [],
          var.okta_enabled ? [
            aws_secretsmanager_secret.okta_client_secret[0].arn,
            aws_secretsmanager_secret.okta_m2m_client_secret[0].arn,
            aws_secretsmanager_secret.okta_api_token[0].arn
          ] : [],
          var.auth0_enabled ? [
            aws_secretsmanager_secret.auth0_client_secret[0].arn,
            aws_secretsmanager_secret.auth0_m2m_client_secret[0].arn
          ] : [],
          var.pingfederate_enabled ? [
            aws_secretsmanager_secret.pingfederate_client_secret[0].arn,
            aws_secretsmanager_secret.pingfederate_m2m_client_secret[0].arn,
            aws_secretsmanager_secret.pf_admin_pass[0].arn,
          ] : [],
          var.enable_observability ? [
            aws_secretsmanager_secret.metrics_api_key[0].arn,
            aws_secretsmanager_secret.metrics_key_pepper[0].arn,
            aws_secretsmanager_secret.metrics_admin_api_key[0].arn,
            aws_secretsmanager_secret.grafana_admin_password[0].arn
          ] : [],
          var.enable_observability && var.otel_otlp_endpoint != "" ? [aws_secretsmanager_secret.otlp_exporter_headers[0].arn] : [],
          # ES256 signing key (delivered to auth-server via the signing-key-init
          # container) + optional scoped keys (registry + shared). All are
          # operator-provided ARNs, wired only when set.
          var.internal_signing_key_secret_arn != "" ? [var.internal_signing_key_secret_arn] : [],
          var.csrf_signing_key_secret_arn != "" ? [var.csrf_signing_key_secret_arn] : [],
          var.credential_encryption_key_secret_arn != "" ? [var.credential_encryption_key_secret_arn] : [],
          var.session_token_enc_key_secret_arn != "" ? [var.session_token_enc_key_secret_arn] : []
        )
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey"
        ]
        Resource = [
          aws_kms_key.secrets.arn
        ]
      }
    ]
  })

  tags = local.common_tags
}

# Per-user egress credential vault: the registry task creates/reads/updates/
# deletes per-user secrets at RUNTIME under the configured path prefix (these
# are not pre-provisioned ARNs), so it needs a broader verb set scoped to the
# prefix, plus KMS for the CMK encrypting them. Only created when the egress
# vault is enabled with the secrets-manager backend.
resource "aws_iam_policy" "ecs_egress_vault_access" {
  count       = var.egress_auth_enabled && var.egress_secret_store_backend == "secrets-manager" ? 1 : 0
  name_prefix = "${local.name_prefix}-egress-vault-"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = concat(
      [
        {
          Effect = "Allow"
          Action = [
            "secretsmanager:CreateSecret",
            "secretsmanager:GetSecretValue",
            "secretsmanager:PutSecretValue",
            "secretsmanager:DeleteSecret",
            "secretsmanager:DescribeSecret",
            "secretsmanager:ListSecretVersionIds"
          ]
          # Scope to the egress path prefix only. Secrets Manager ARNs append a
          # random 6-char suffix, so the wildcard covers "<prefix>/*-??????".
          Resource = "arn:aws:secretsmanager:*:*:secret:${var.egress_secrets_manager_path_prefix}/*"
        }
      ],
      # KMS only when a customer-managed CMK is configured; the AWS-managed key
      # needs no explicit grant.
      var.egress_secrets_manager_kms_key_id != "" ? [
        {
          Effect = "Allow"
          Action = [
            "kms:Decrypt",
            "kms:GenerateDataKey"
          ]
          Resource = [var.egress_secrets_manager_kms_key_id]
        }
      ] : []
    )
  })

  tags = local.common_tags
}

# IAM policy for ECS Exec - task execution role
resource "aws_iam_policy" "ecs_exec_task_execution" {
  name_prefix = "${local.name_prefix}-ecs-exec-task-exec-"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ssmmessages:CreateControlChannel",
          "ssmmessages:CreateDataChannel",
          "ssmmessages:OpenControlChannel",
          "ssmmessages:OpenDataChannel"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })

  tags = local.common_tags
}

# IAM policy for Amazon Bedrock AgentCore access (registry federation)
#
# Least-privilege: the registry federation client is READ-ONLY against the
# bedrock-agentcore-control plane -- it only lists registries, lists records,
# and fetches record details (see registry/services/federation/agentcore_client.py).
# It never creates/updates/deletes AgentCore resources, so the action set is
# restricted to those three read operations.
#
# The read grant is split into two statements because the actions differ in
# their IAM resource-level support (per the AWS Service Authorization Reference):
#   - ListRegistries has NO resource type, so IAM only accepts it on
#     Resource = "*". Scoping it to a registry ARN silently makes it a no-op
#     (the action never matches) and boto3 gets AccessDenied at runtime.
#   - ListRegistryRecords (resource type "registry") and GetRegistryRecord
#     (resource type "registry-record") DO support resource-level permissions,
#     so they are scoped to registries in the deploying account. Region is
#     wildcarded so per-registry region overrides keep working; the record ARN
#     (registry/<id>/record/<id>) is a child of the registry/* prefix.
#
# Cross-account federation uses sts:AssumeRole into caller-supplied role ARNs.
# That grant is only emitted when specific role ARNs are configured
# (var.aws_registry_federation_assume_role_arns); an empty list -> no
# sts:AssumeRole statement at all (fail closed, no wildcard cross-account trust).
resource "aws_iam_policy" "bedrock_agentcore_access" {
  count       = var.aws_registry_federation_enabled ? 1 : 0
  name_prefix = "${local.name_prefix}-bedrock-agentcore-"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = concat(
      [
        {
          Sid    = "BedrockAgentCoreListRegistries"
          Effect = "Allow"
          Action = [
            "bedrock-agentcore:ListRegistries"
          ]
          # ListRegistries has no IAM resource type; it must be granted on "*".
          # This is not a privilege-creep wildcard -- it is the only Resource
          # value AWS accepts for this single read/list action.
          Resource = "*"
        },
        {
          Sid    = "BedrockAgentCoreReadRecords"
          Effect = "Allow"
          Action = [
            "bedrock-agentcore:ListRegistryRecords",
            "bedrock-agentcore:GetRegistryRecord"
          ]
          # Scope to registries (and their child records) in the deploying
          # account. registry/* also covers registry/<id>/record/<id>.
          Resource = "arn:${data.aws_partition.current.partition}:bedrock-agentcore:*:${data.aws_caller_identity.current.account_id}:registry/*"
        }
      ],
      length(var.aws_registry_federation_assume_role_arns) > 0 ? [
        {
          Sid    = "StsAssumeRoleForCrossAccount"
          Effect = "Allow"
          Action = [
            "sts:AssumeRole"
          ]
          # Only the explicitly configured cross-account federation roles.
          Resource = var.aws_registry_federation_assume_role_arns
          # Defense-in-depth: the target role must also carry the federation tag.
          Condition = {
            StringLike = {
              "iam:ResourceTag/Purpose" = "agentcore-federation"
            }
          }
        }
      ] : []
    )
  })

  tags = local.common_tags
}


# Cognito read-only access for the registry IAM management UI.
# The registry's CognitoIAMManager lists groups and users from the User Pool to
# populate the IAM > Groups / Users pages. Read-only: no create/delete actions.
resource "aws_iam_policy" "cognito_iam_read" {
  count       = var.cognito_enabled ? 1 : 0
  name_prefix = "${local.name_prefix}-cognito-iam-read-"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "CognitoIdpReadOnly"
        Effect = "Allow"
        Action = [
          "cognito-idp:ListGroups",
          "cognito-idp:ListUsers",
          "cognito-idp:AdminListGroupsForUser"
        ]
        Resource = "arn:aws:cognito-idp:*:*:userpool/${var.cognito_user_pool_id}"
      }
    ]
  })

  tags = local.common_tags
}


# IAM policy for ECS Exec - task role
resource "aws_iam_policy" "ecs_exec_task" {
  name_prefix = "${local.name_prefix}-ecs-exec-task-"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ssmmessages:CreateControlChannel",
          "ssmmessages:CreateDataChannel",
          "ssmmessages:OpenControlChannel",
          "ssmmessages:OpenDataChannel"
        ]
        Resource = "*"
      }
    ]
  })

  tags = local.common_tags
}
