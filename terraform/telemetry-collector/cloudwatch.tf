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

# =============================================================================
# Cloud-detection quality alarms (issue #986)
# =============================================================================
# The registry emits a cloud_detection_method label with every startup event.
# A high share of method=unknown indicates either: a new cloud popular with
# users we do not yet detect, a regression in the probe logic, or operators
# hitting the EKS IMDS hop-limit gotcha documented in docs/TELEMETRY.md.
#
# We extract two metric filters from the collector's own log group:
#   1. CloudDetectionUnknown - count of startup events with method=unknown
#   2. CloudDetectionTotal   - count of all startup events
# The alarm below computes unknown/total as a percentage over a 24h window.

resource "aws_cloudwatch_log_metric_filter" "cloud_detection_unknown" {
  name           = "telemetry-cloud-detection-unknown"
  log_group_name = aws_cloudwatch_log_group.telemetry_collector.name
  pattern        = "{ $.cloud_detection_method = \"unknown\" && $.event = \"startup\" }"

  metric_transformation {
    name          = "CloudDetectionUnknown"
    namespace     = "MCPRegistry/Telemetry"
    value         = "1"
    default_value = "0"
  }
}

resource "aws_cloudwatch_log_metric_filter" "cloud_detection_total" {
  name           = "telemetry-cloud-detection-total"
  log_group_name = aws_cloudwatch_log_group.telemetry_collector.name
  pattern        = "{ $.event = \"startup\" }"

  metric_transformation {
    name          = "CloudDetectionTotal"
    namespace     = "MCPRegistry/Telemetry"
    value         = "1"
    default_value = "0"
  }
}

resource "aws_cloudwatch_metric_alarm" "cloud_detection_unknown_ratio" {
  count = var.deployment_stage == "production" && var.alarm_email != "" ? 1 : 0

  alarm_name          = "telemetry-cloud-detection-unknown-ratio-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  threshold           = 10  # Percentage of startup events with method=unknown over the window
  alarm_description   = "Telemetry cloud_detection_method=unknown exceeds 10% of startup events over a 24h window (issue #986)"
  alarm_actions       = [aws_sns_topic.alarms[0].arn]

  metric_query {
    id          = "ratio"
    expression  = "100 * unknown / total"
    label       = "Unknown detection ratio (%)"
    return_data = true
  }

  metric_query {
    id = "unknown"
    metric {
      namespace   = "MCPRegistry/Telemetry"
      metric_name = "CloudDetectionUnknown"
      period      = 86400  # 24 hours
      stat        = "Sum"
    }
  }

  metric_query {
    id = "total"
    metric {
      namespace   = "MCPRegistry/Telemetry"
      metric_name = "CloudDetectionTotal"
      period      = 86400
      stat        = "Sum"
    }
  }
}
