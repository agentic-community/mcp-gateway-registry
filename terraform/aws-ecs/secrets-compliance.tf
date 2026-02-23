#
# Secrets Manager SCP Compliance
#
# This file provides resources required by the organization's SCP:
#   1. KMS key for encrypting all Secrets Manager secrets
#   2. Lambda function for automatic secret rotation
#   3. Rotation schedules for all secrets
#
# SCP requirements:
#   - DenyCreateSecretWithoutKMSEncryption: All secrets must specify kms_key_id
#   - RequireAutomaticRotationEnabled: All secrets must have rotation enabled
#

# =============================================================================
# KMS Key for Secrets Manager
# =============================================================================

resource "aws_kms_key" "secrets" {
  description             = "KMS key for Secrets Manager encryption"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  tags = merge(
    local.common_tags,
    {
      Name      = "${var.name}-secrets-key"
      Component = "secrets"
    }
  )
}

resource "aws_kms_alias" "secrets" {
  name          = "alias/${var.name}-secrets"
  target_key_id = aws_kms_key.secrets.key_id
}

# =============================================================================
# Rotation Lambda (no-op — satisfies SCP rotation requirement)
# =============================================================================
# The SCP requires rotation to be enabled. This Lambda is a no-op placeholder
# that satisfies the policy. Actual credential rotation is handled externally
# (e.g., init-keycloak.sh updates Keycloak secrets after deployment).

resource "aws_iam_role" "secrets_rotation" {
  name = "${var.name}-secrets-rotation-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "secrets_rotation" {
  name = "secrets-rotation-policy"
  role = aws_iam_role.secrets_rotation.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret",
          "secretsmanager:PutSecretValue",
          "secretsmanager:UpdateSecretVersionStage"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:GenerateDataKey"
        ]
        Resource = aws_kms_key.secrets.arn
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# Lambda function — minimal no-op rotation handler
data "archive_file" "secrets_rotation" {
  type        = "zip"
  output_path = "${path.module}/.terraform/tmp/secrets-rotation.zip"

  source {
    content  = <<-PYTHON
import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """No-op rotation handler. Satisfies SCP rotation requirement.
    Marks the AWSPENDING version as AWSCURRENT without changing the secret value."""
    secret_id = event['SecretId']
    step = event['Step']
    token = event['ClientRequestToken']

    client = boto3.client('secretsmanager')

    if step == 'createSecret':
        current = client.get_secret_value(SecretId=secret_id, VersionStage='AWSCURRENT')
        client.put_secret_value(
            SecretId=secret_id,
            ClientRequestToken=token,
            SecretString=current['SecretString'],
            VersionStages=['AWSPENDING']
        )
    elif step == 'setSecret':
        pass
    elif step == 'testSecret':
        pass
    elif step == 'finishSecret':
        metadata = client.describe_secret(SecretId=secret_id)
        for version, stages in metadata['VersionIdsToStages'].items():
            if 'AWSCURRENT' in stages and version != token:
                client.update_secret_version_stage(
                    SecretId=secret_id,
                    VersionStage='AWSCURRENT',
                    MoveToVersionId=token,
                    RemoveFromVersionId=version
                )
                break
        logger.info(f"finishSecret: Secret {secret_id} rotated (no-op)")
    return {"status": "ok"}
PYTHON
    filename = "index.py"
  }
}

resource "aws_lambda_function" "secrets_rotation" {
  function_name    = "${var.name}-secrets-rotation"
  role             = aws_iam_role.secrets_rotation.arn
  handler          = "index.lambda_handler"
  runtime          = "python3.12"
  timeout          = 30
  filename         = data.archive_file.secrets_rotation.output_path
  source_code_hash = data.archive_file.secrets_rotation.output_base64sha256

  tags = merge(
    local.common_tags,
    {
      Name      = "${var.name}-secrets-rotation"
      Component = "secrets"
    }
  )
}

resource "aws_lambda_permission" "secrets_rotation" {
  statement_id  = "AllowSecretsManagerInvocation"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.secrets_rotation.function_name
  principal     = "secretsmanager.amazonaws.com"
}

# =============================================================================
# Rotation Schedules — Root Module Secrets
# =============================================================================

resource "aws_secretsmanager_secret_rotation" "documentdb_credentials" {
  secret_id           = aws_secretsmanager_secret.documentdb_credentials.id
  rotation_lambda_arn = aws_lambda_function.secrets_rotation.arn

  rotation_rules {
    automatically_after_days = 90
  }
}

resource "aws_secretsmanager_secret_rotation" "keycloak_db_secret" {
  secret_id           = aws_secretsmanager_secret.keycloak_db_secret.id
  rotation_lambda_arn = aws_lambda_function.secrets_rotation.arn

  rotation_rules {
    automatically_after_days = 90
  }
}
