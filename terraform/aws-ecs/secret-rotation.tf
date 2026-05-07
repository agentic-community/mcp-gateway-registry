#
# AWS Secrets Manager Rotation with Lambda Functions
#
# This configuration implements automatic password rotation for DocumentDB and RDS
# using AWS Lambda functions. Secrets are rotated every 30 days automatically.
#
# Architecture:
# - Lambda functions are deployed in VPC private subnets for database access
# - IAM roles grant permissions to read/update secrets and modify databases
# - CloudWatch Logs capture rotation execution logs for troubleshooting
# - Lambda functions implement the 4-step AWS rotation process:
#   1. createSecret: Generate new random password
#   2. setSecret: Update database with new password
#   3. testSecret: Verify new password works
#   4. finishSecret: Promote new version to AWSCURRENT
#

#
# IAM Role for Lambda Rotation Functions
#
resource "aws_iam_role" "rotation_lambda" {
  name = "${var.name}-secret-rotation-lambda"

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

#
# IAM Policy for Lambda to Rotate Secrets
# The IAM role is shared across the DocumentDB rotation Lambda (gated on
# is_aws_documentdb) and the RDS/Keycloak rotation Lambda (always created).
# DocumentDB-specific resources are conditionally included so the policy
# does not reference non-existent resources when DocumentDB is gated out.
# Issue #955.
#
#checkov:skip=CKV_AWS_290:GetRandomPassword and EC2 network interface actions require wildcard resource per AWS API design
#checkov:skip=CKV_AWS_355:GetRandomPassword and EC2 network interface actions require wildcard resource per AWS API design
resource "aws_iam_role_policy" "rotation_lambda" {
  name = "${var.name}-secret-rotation-policy"
  role = aws_iam_role.rotation_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = concat(
      [
        {
          Sid    = "SecretsManagerAccess"
          Effect = "Allow"
          Action = [
            "secretsmanager:DescribeSecret",
            "secretsmanager:GetSecretValue",
            "secretsmanager:PutSecretValue",
            "secretsmanager:UpdateSecretVersionStage"
          ]
          Resource = concat(
            local.is_aws_documentdb ? [aws_secretsmanager_secret.documentdb_credentials[0].arn] : [],
            [aws_secretsmanager_secret.keycloak_db_secret.arn],
          )
        },
        {
          Sid    = "GenerateRandomPassword"
          Effect = "Allow"
          Action = [
            "secretsmanager:GetRandomPassword"
          ]
          Resource = "*"
        },
        {
          Sid    = "KMSAccess"
          Effect = "Allow"
          Action = [
            "kms:Decrypt",
            "kms:DescribeKey",
            "kms:GenerateDataKey"
          ]
          Resource = concat(
            local.is_aws_documentdb ? [aws_kms_key.documentdb[0].arn] : [],
            [aws_kms_key.rds.arn],
          )
        },
        {
          Sid    = "RDSAccess"
          Effect = "Allow"
          Action = [
            "rds:DescribeDBInstances",
            "rds:DescribeDBClusters",
            "rds:ModifyDBCluster"
          ]
          Resource = aws_rds_cluster.keycloak.arn
        },
        {
          Sid    = "VPCNetworkInterface"
          Effect = "Allow"
          Action = [
            "ec2:CreateNetworkInterface",
            "ec2:DescribeNetworkInterfaces",
            "ec2:DeleteNetworkInterface",
            "ec2:AssignPrivateIpAddresses",
            "ec2:UnassignPrivateIpAddresses"
          ]
          Resource = "*"
        },
      ],
      # DocumentDBAccess statement only when the DocumentDB cluster exists.
      local.is_aws_documentdb ? [{
        Sid    = "DocumentDBAccess"
        Effect = "Allow"
        Action = [
          "docdb:DescribeDBClusters",
          "docdb:ModifyDBCluster"
        ]
        Resource = aws_docdb_cluster.registry[0].arn
      }] : [],
    )
  })
}

#
# Attach Lambda VPC Execution Policy
#
resource "aws_iam_role_policy_attachment" "lambda_vpc_execution" {
  role       = aws_iam_role.rotation_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
}

#
# Security Group for Lambda Functions
#
resource "aws_security_group" "rotation_lambda" {
  name        = "${var.name}-rotation-lambda-sg"
  description = "Security group for secret rotation Lambda functions"
  vpc_id      = module.vpc.vpc_id

  tags = merge(
    local.common_tags,
    {
      Name      = "${var.name}-rotation-lambda-sg"
      Component = "secrets-rotation"
    }
  )
}

#
# Lambda -> DocumentDB (gated on is_aws_documentdb - issue #955)
#
resource "aws_vpc_security_group_egress_rule" "lambda_to_documentdb" {
  count = local.is_aws_documentdb ? 1 : 0

  security_group_id = aws_security_group.rotation_lambda.id

  referenced_security_group_id = aws_security_group.documentdb[0].id
  from_port                    = 27017
  to_port                      = 27017
  ip_protocol                  = "tcp"
  description                  = "Allow Lambda to connect to DocumentDB for rotation"

  tags = merge(
    local.common_tags,
    {
      Name = "lambda-to-documentdb"
    }
  )
}

#
# DocumentDB <- Lambda (gated on is_aws_documentdb - issue #955)
#
resource "aws_vpc_security_group_ingress_rule" "documentdb_from_lambda" {
  count = local.is_aws_documentdb ? 1 : 0

  security_group_id = aws_security_group.documentdb[0].id

  referenced_security_group_id = aws_security_group.rotation_lambda.id
  from_port                    = 27017
  to_port                      = 27017
  ip_protocol                  = "tcp"
  description                  = "Allow Lambda rotation function to connect to DocumentDB"

  tags = merge(
    local.common_tags,
    {
      Name = "documentdb-from-lambda"
    }
  )
}

#
# Lambda -> RDS
#
resource "aws_vpc_security_group_egress_rule" "lambda_to_rds" {
  security_group_id = aws_security_group.rotation_lambda.id

  referenced_security_group_id = aws_security_group.keycloak_db.id
  from_port                    = 3306
  to_port                      = 3306
  ip_protocol                  = "tcp"
  description                  = "Allow Lambda to connect to RDS for rotation"

  tags = merge(
    local.common_tags,
    {
      Name = "lambda-to-rds"
    }
  )
}

#
# RDS <- Lambda
#
resource "aws_vpc_security_group_ingress_rule" "rds_from_lambda" {
  security_group_id = aws_security_group.keycloak_db.id

  referenced_security_group_id = aws_security_group.rotation_lambda.id
  from_port                    = 3306
  to_port                      = 3306
  ip_protocol                  = "tcp"
  description                  = "Allow Lambda rotation function to connect to RDS"

  tags = merge(
    local.common_tags,
    {
      Name = "rds-from-lambda"
    }
  )
}

#
# Lambda -> HTTPS (for Secrets Manager API)
#
resource "aws_vpc_security_group_egress_rule" "lambda_to_https" {
  security_group_id = aws_security_group.rotation_lambda.id

  cidr_ipv4   = "0.0.0.0/0"
  from_port   = 443
  to_port     = 443
  ip_protocol = "tcp"
  description = "Allow Lambda to call AWS APIs (Secrets Manager, KMS)"

  tags = merge(
    local.common_tags,
    {
      Name = "lambda-to-https"
    }
  )
}

#
# CloudWatch Log Groups for Lambda Functions
#
#checkov:skip=CKV_AWS_158:KMS encryption for CloudWatch logs not required in this deployment
resource "aws_cloudwatch_log_group" "documentdb_rotation" {
  count = local.is_aws_documentdb ? 1 : 0

  name              = "/aws/lambda/${var.name}-rotate-documentdb"
  retention_in_days = 30

  tags = merge(
    local.common_tags,
    {
      Component = "secrets-rotation"
    }
  )
}

#checkov:skip=CKV_AWS_158:KMS encryption for CloudWatch logs not required in this deployment
resource "aws_cloudwatch_log_group" "rds_rotation" {
  name              = "/aws/lambda/${var.name}-rotate-rds"
  retention_in_days = 30

  tags = merge(
    local.common_tags,
    {
      Component = "secrets-rotation"
    }
  )
}

#
# Lambda Function Package - DocumentDB Rotation
# Gated via `count` on the data source so the archive is not built when
# DocumentDB is not provisioned. Issue #955.
#
data "archive_file" "documentdb_rotation" {
  count = local.is_aws_documentdb ? 1 : 0

  type        = "zip"
  source_dir  = "${path.module}/lambda/rotate-documentdb"
  output_path = "${path.module}/.terraform/lambda/rotate-documentdb.zip"
}

#
# Lambda Function Package - RDS Rotation
#
data "archive_file" "rds_rotation" {
  type        = "zip"
  source_dir  = "${path.module}/lambda/rotate-rds"
  output_path = "${path.module}/.terraform/lambda/rotate-rds.zip"
}

#
# Lambda Function - DocumentDB Rotation (gated on is_aws_documentdb)
#
#checkov:skip=CKV_AWS_115:Reserved concurrency not needed for secret rotation Lambda
#checkov:skip=CKV_AWS_116:DLQ not needed for synchronous secret rotation Lambda
#checkov:skip=CKV_AWS_173:Lambda environment variables use default encryption
#checkov:skip=CKV_AWS_272:Code signing not configured for internal rotation Lambdas
resource "aws_lambda_function" "documentdb_rotation" {
  count = local.is_aws_documentdb ? 1 : 0

  filename         = data.archive_file.documentdb_rotation[0].output_path
  function_name    = "${var.name}-rotate-documentdb"
  role             = aws_iam_role.rotation_lambda.arn
  handler          = "index.lambda_handler"
  source_code_hash = data.archive_file.documentdb_rotation[0].output_base64sha256
  runtime          = "python3.13"
  timeout          = 300
  memory_size      = 256

  vpc_config {
    subnet_ids         = module.vpc.private_subnets
    security_group_ids = [aws_security_group.rotation_lambda.id]
  }

  environment {
    variables = {
      SECRETS_MANAGER_ENDPOINT = "https://secretsmanager.${var.aws_region}.amazonaws.com"
      EXCLUDE_CHARACTERS       = "/@\"'\\"
    }
  }

  tracing_config {
    mode = "Active"
  }

  tags = merge(
    local.common_tags,
    {
      Component = "secrets-rotation"
    }
  )

  depends_on = [
    aws_cloudwatch_log_group.documentdb_rotation,
    aws_iam_role_policy.rotation_lambda,
    aws_iam_role_policy_attachment.lambda_vpc_execution
  ]
}

#
# Lambda Function - RDS Rotation
#
#checkov:skip=CKV_AWS_115:Reserved concurrency not needed for secret rotation Lambda
#checkov:skip=CKV_AWS_116:DLQ not needed for synchronous secret rotation Lambda
#checkov:skip=CKV_AWS_173:Lambda environment variables use default encryption
#checkov:skip=CKV_AWS_272:Code signing not configured for internal rotation Lambdas
resource "aws_lambda_function" "rds_rotation" {
  filename         = data.archive_file.rds_rotation.output_path
  function_name    = "${var.name}-rotate-rds"
  role             = aws_iam_role.rotation_lambda.arn
  handler          = "index.lambda_handler"
  source_code_hash = data.archive_file.rds_rotation.output_base64sha256
  runtime          = "python3.13"
  timeout          = 300
  memory_size      = 256

  vpc_config {
    subnet_ids         = module.vpc.private_subnets
    security_group_ids = [aws_security_group.rotation_lambda.id]
  }

  environment {
    variables = {
      SECRETS_MANAGER_ENDPOINT = "https://secretsmanager.${var.aws_region}.amazonaws.com"
      EXCLUDE_CHARACTERS       = "/@\"'\\"
    }
  }

  tracing_config {
    mode = "Active"
  }

  tags = merge(
    local.common_tags,
    {
      Component = "secrets-rotation"
    }
  )

  depends_on = [
    aws_cloudwatch_log_group.rds_rotation,
    aws_iam_role_policy.rotation_lambda,
    aws_iam_role_policy_attachment.lambda_vpc_execution
  ]
}

#
# Lambda Permission for Secrets Manager - DocumentDB (gated on is_aws_documentdb)
#
#checkov:skip=CKV_AWS_364:Lambda resource-based policy does not use IAM policy document version field
resource "aws_lambda_permission" "documentdb_rotation" {
  count = local.is_aws_documentdb ? 1 : 0

  statement_id  = "AllowExecutionFromSecretsManager"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.documentdb_rotation[0].function_name
  principal     = "secretsmanager.amazonaws.com"
}

#
# Lambda Permission for Secrets Manager - RDS
#
#checkov:skip=CKV_AWS_364:Lambda resource-based policy does not use IAM policy document version field
resource "aws_lambda_permission" "rds_rotation" {
  statement_id  = "AllowExecutionFromSecretsManager"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.rds_rotation.function_name
  principal     = "secretsmanager.amazonaws.com"
}
