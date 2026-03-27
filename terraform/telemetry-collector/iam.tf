# IAM role for Lambda function
resource "aws_iam_role" "lambda_execution" {
  name = "telemetry-collector-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "telemetry-collector-lambda-role"
  }
}

# CloudWatch Logs policy
resource "aws_iam_role_policy" "lambda_cloudwatch" {
  name = "telemetry-collector-cloudwatch-policy"
  role = aws_iam_role.lambda_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "${aws_cloudwatch_log_group.telemetry_collector.arn}:*"
      }
    ]
  })
}

# VPC network interface policy (required for VPC-enabled Lambda)
resource "aws_iam_role_policy" "lambda_vpc" {
  name = "telemetry-collector-vpc-policy"
  role = aws_iam_role.lambda_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ec2:CreateNetworkInterface",
          "ec2:DescribeNetworkInterfaces",
          "ec2:DeleteNetworkInterface",
          "ec2:AssignPrivateIpAddresses",
          "ec2:UnassignPrivateIpAddresses"
        ]
        # AWS requires Resource = "*" for EC2 network interface operations
        # (CreateNetworkInterface, DescribeNetworkInterfaces, etc.)
        Resource = "*"
      }
    ]
  })
}

# DynamoDB policy (rate limiting table)
resource "aws_iam_role_policy" "lambda_dynamodb" {
  name = "telemetry-collector-dynamodb-policy"
  role = aws_iam_role.lambda_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:UpdateItem"
        ]
        Resource = aws_dynamodb_table.rate_limit.arn
      }
    ]
  })
}

# Secrets Manager policy (DocumentDB credentials)
resource "aws_iam_role_policy" "lambda_secrets" {
  name = "telemetry-collector-secrets-policy"
  role = aws_iam_role.lambda_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = aws_secretsmanager_secret.documentdb_credentials.arn
      }
    ]
  })
}
