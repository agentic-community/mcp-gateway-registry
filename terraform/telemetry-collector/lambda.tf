# Lambda function
resource "aws_lambda_function" "telemetry_collector" {
  filename         = var.lambda_package_path
  function_name    = "telemetry-collector"
  role             = aws_iam_role.lambda_execution.arn
  handler          = "index.lambda_handler"
  source_code_hash = filebase64sha256(var.lambda_package_path)
  runtime          = "python3.12"
  timeout          = 30
  memory_size      = 256

  vpc_config {
    subnet_ids         = aws_subnet.private[*].id
    security_group_ids = [aws_security_group.lambda.id]
  }

  environment {
    variables = {
      RATE_LIMIT_TABLE      = aws_dynamodb_table.rate_limit.name
      DOCUMENTDB_SECRET_ARN = aws_secretsmanager_secret.documentdb_credentials.arn
      DOCUMENTDB_ENDPOINT   = aws_docdb_cluster.telemetry.endpoint
    }
  }

  depends_on = [
    aws_cloudwatch_log_group.telemetry_collector,
    aws_iam_role_policy.lambda_cloudwatch,
    aws_iam_role_policy.lambda_vpc,
    aws_iam_role_policy.lambda_dynamodb,
    aws_iam_role_policy.lambda_secrets
  ]

  tags = {
    Name = "telemetry-collector"
  }
}

# API Gateway HTTP API
resource "aws_apigatewayv2_api" "telemetry" {
  name          = "telemetry-collector-api"
  protocol_type = "HTTP"
  description   = "Privacy-first telemetry collector API for MCP Gateway Registry"

  cors_configuration {
    allow_origins = var.cors_allowed_origins
    allow_methods = ["POST"]
    allow_headers = ["content-type"]
    max_age       = 300
  }

  tags = {
    Name = "telemetry-collector-api"
  }
}

# API Gateway integration with Lambda
resource "aws_apigatewayv2_integration" "lambda" {
  api_id           = aws_apigatewayv2_api.telemetry.id
  integration_type = "AWS_PROXY"
  integration_uri  = aws_lambda_function.telemetry_collector.invoke_arn
  payload_format_version = "2.0"
}

# API Gateway route for POST /v1/collect
resource "aws_apigatewayv2_route" "collect" {
  api_id    = aws_apigatewayv2_api.telemetry.id
  route_key = "POST /v1/collect"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

# API Gateway stage (default stage)
resource "aws_apigatewayv2_stage" "telemetry" {
  api_id      = aws_apigatewayv2_api.telemetry.id
  name        = "$default"
  auto_deploy = true

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gateway.arn
    format = jsonencode({
      requestId      = "$context.requestId"
      requestTime    = "$context.requestTime"
      httpMethod     = "$context.httpMethod"
      routeKey       = "$context.routeKey"
      status         = "$context.status"
      protocol       = "$context.protocol"
      responseLength = "$context.responseLength"
    })
  }

  default_route_settings {
    throttling_burst_limit = 100
    throttling_rate_limit  = 50
  }

  tags = {
    Name = "telemetry-collector-stage"
  }
}

# Lambda permission for API Gateway to invoke function
resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.telemetry_collector.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.telemetry.execution_arn}/*/*"
}
