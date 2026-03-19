# CloudWatch log group for Lambda function
resource "aws_cloudwatch_log_group" "telemetry_collector" {
  name              = "/aws/lambda/telemetry-collector"
  retention_in_days = var.log_retention_days

  tags = {
    Name = "telemetry-collector-logs"
  }
}

# CloudWatch log group for API Gateway
resource "aws_cloudwatch_log_group" "api_gateway" {
  name              = "/aws/apigateway/telemetry-collector"
  retention_in_days = var.log_retention_days

  tags = {
    Name = "telemetry-collector-api-logs"
  }
}

# SNS topic for alarms (if email provided)
resource "aws_sns_topic" "alarms" {
  count = var.alarm_email != "" ? 1 : 0

  name = "telemetry-collector-alarms"

  tags = {
    Name = "telemetry-collector-alarms"
  }
}

# SNS topic subscription
resource "aws_sns_topic_subscription" "alarm_email" {
  count = var.alarm_email != "" ? 1 : 0

  topic_arn = aws_sns_topic.alarms[0].arn
  protocol  = "email"
  endpoint  = var.alarm_email
}

# CloudWatch alarm for Lambda errors
resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  count = var.deployment_stage == "production" && var.alarm_email != "" ? 1 : 0

  alarm_name          = "telemetry-collector-lambda-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 10
  alarm_description   = "This metric monitors Lambda function errors"
  alarm_actions       = [aws_sns_topic.alarms[0].arn]

  dimensions = {
    FunctionName = aws_lambda_function.telemetry_collector.function_name
  }
}

# CloudWatch alarm for Lambda throttles
resource "aws_cloudwatch_metric_alarm" "lambda_throttles" {
  count = var.deployment_stage == "production" && var.alarm_email != "" ? 1 : 0

  alarm_name          = "telemetry-collector-lambda-throttles"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "Throttles"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 5
  alarm_description   = "This metric monitors Lambda function throttles"
  alarm_actions       = [aws_sns_topic.alarms[0].arn]

  dimensions = {
    FunctionName = aws_lambda_function.telemetry_collector.function_name
  }
}

# CloudWatch alarm for Lambda duration (high latency)
resource "aws_cloudwatch_metric_alarm" "lambda_duration" {
  count = var.deployment_stage == "production" && var.alarm_email != "" ? 1 : 0

  alarm_name          = "telemetry-collector-lambda-duration"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "Duration"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Average"
  threshold           = 10000  # 10 seconds
  alarm_description   = "This metric monitors Lambda function execution time"
  alarm_actions       = [aws_sns_topic.alarms[0].arn]

  dimensions = {
    FunctionName = aws_lambda_function.telemetry_collector.function_name
  }
}

# CloudWatch alarm for API Gateway 5xx errors
resource "aws_cloudwatch_metric_alarm" "api_gateway_5xx" {
  count = var.deployment_stage == "production" && var.alarm_email != "" ? 1 : 0

  alarm_name          = "telemetry-collector-api-5xx-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "5XXError"
  namespace           = "AWS/ApiGateway"
  period              = 300
  statistic           = "Sum"
  threshold           = 10
  alarm_description   = "This metric monitors API Gateway 5xx errors"
  alarm_actions       = [aws_sns_topic.alarms[0].arn]

  dimensions = {
    ApiId = aws_apigatewayv2_api.telemetry.id
  }
}
