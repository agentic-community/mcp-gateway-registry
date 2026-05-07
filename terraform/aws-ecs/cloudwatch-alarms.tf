#
# CloudWatch Alarms for Infrastructure Monitoring
#
# This file contains CloudWatch alarms for monitoring security components:
# - WAF blocked requests and rate limiting
# - KMS API throttling
# - DocumentDB audit log failures
# - S3 bucket size monitoring
#

#
# WAF Monitoring Alarms
#

# CloudWatch Alarm: WAF Blocked Requests High (MCP Gateway)
resource "aws_cloudwatch_metric_alarm" "waf_blocked_requests_high_mcp_gateway" {
  count = var.enable_waf ? 1 : 0

  alarm_name          = "${var.name}-waf-blocked-requests-high-mcp-gateway"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "BlockedRequests"
  namespace           = "AWS/WAFV2"
  period              = 300 # 5 minutes
  statistic           = "Sum"
  threshold           = 100
  alarm_description   = "WAF blocking >100 requests in 5 minutes - potential attack on MCP Gateway"
  treat_missing_data  = "notBreaching"

  dimensions = {
    WebACL = aws_wafv2_web_acl.mcp_gateway[0].name
    Region = var.aws_region
    Rule   = "ALL"
  }

  alarm_actions = var.alarm_sns_topic_arn != "" ? [var.alarm_sns_topic_arn] : []

  tags = merge(
    local.common_tags,
    {
      Purpose   = "WAF attack detection"
      Component = "monitoring"
      Service   = "mcp-gateway"
    }
  )
}

# CloudWatch Alarm: WAF Blocked Requests High (Keycloak)
resource "aws_cloudwatch_metric_alarm" "waf_blocked_requests_high_keycloak" {
  count = var.enable_waf ? 1 : 0

  alarm_name          = "${var.name}-waf-blocked-requests-high-keycloak"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "BlockedRequests"
  namespace           = "AWS/WAFV2"
  period              = 300 # 5 minutes
  statistic           = "Sum"
  threshold           = 100
  alarm_description   = "WAF blocking >100 requests in 5 minutes - potential attack on Keycloak"
  treat_missing_data  = "notBreaching"

  dimensions = {
    WebACL = aws_wafv2_web_acl.keycloak[0].name
    Region = var.aws_region
    Rule   = "ALL"
  }

  alarm_actions = var.alarm_sns_topic_arn != "" ? [var.alarm_sns_topic_arn] : []

  tags = merge(
    local.common_tags,
    {
      Purpose   = "WAF attack detection"
      Component = "monitoring"
      Service   = "keycloak"
    }
  )
}

# CloudWatch Alarm: WAF Rate Limit Triggered (MCP Gateway)
resource "aws_cloudwatch_metric_alarm" "waf_rate_limit_triggered_mcp_gateway" {
  count = var.enable_waf ? 1 : 0

  alarm_name          = "${var.name}-waf-rate-limit-triggered-mcp-gateway"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "BlockedRequests"
  namespace           = "AWS/WAFV2"
  period              = 60 # 1 minute
  statistic           = "Sum"
  threshold           = 50
  alarm_description   = "WAF rate limit triggered for MCP Gateway - potential DDoS"
  treat_missing_data  = "notBreaching"

  dimensions = {
    WebACL = aws_wafv2_web_acl.mcp_gateway[0].name
    Region = var.aws_region
    Rule   = "RateLimitRule"
  }

  alarm_actions = var.alarm_sns_topic_arn != "" ? [var.alarm_sns_topic_arn] : []

  tags = merge(
    local.common_tags,
    {
      Purpose   = "Rate limit monitoring"
      Component = "monitoring"
      Service   = "mcp-gateway"
    }
  )
}

# CloudWatch Alarm: WAF Rate Limit Triggered (Keycloak)
resource "aws_cloudwatch_metric_alarm" "waf_rate_limit_triggered_keycloak" {
  count = var.enable_waf ? 1 : 0

  alarm_name          = "${var.name}-waf-rate-limit-triggered-keycloak"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "BlockedRequests"
  namespace           = "AWS/WAFV2"
  period              = 60 # 1 minute
  statistic           = "Sum"
  threshold           = 50
  alarm_description   = "WAF rate limit triggered for Keycloak - potential DDoS"
  treat_missing_data  = "notBreaching"

  dimensions = {
    WebACL = aws_wafv2_web_acl.keycloak[0].name
    Region = var.aws_region
    Rule   = "RateLimitRule"
  }

  alarm_actions = var.alarm_sns_topic_arn != "" ? [var.alarm_sns_topic_arn] : []

  tags = merge(
    local.common_tags,
    {
      Purpose   = "Rate limit monitoring"
      Component = "monitoring"
      Service   = "keycloak"
    }
  )
}

#
# KMS Monitoring Alarms
#

# CloudWatch Alarm: KMS Throttling (DocumentDB Key)
# Gated on is_aws_documentdb: no KMS key exists for external MongoDB
# backends, so there's nothing to alarm on. Issue #955.
resource "aws_cloudwatch_metric_alarm" "kms_throttling_documentdb" {
  count = local.is_aws_documentdb ? 1 : 0

  alarm_name          = "${var.name}-kms-throttling-documentdb"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "UserErrorCount"
  namespace           = "AWS/KMS"
  period              = 60
  statistic           = "Sum"
  threshold           = 10
  alarm_description   = "KMS API throttling detected for DocumentDB key - secrets may be inaccessible"
  treat_missing_data  = "notBreaching"

  dimensions = {
    KeyId = aws_kms_key.documentdb[0].id
  }

  alarm_actions = var.alarm_sns_topic_arn != "" ? [var.alarm_sns_topic_arn] : []

  tags = merge(
    local.common_tags,
    {
      Purpose   = "KMS availability monitoring"
      Component = "monitoring"
      Service   = "documentdb"
    }
  )
}

# CloudWatch Alarm: KMS Throttling (RDS Key)
resource "aws_cloudwatch_metric_alarm" "kms_throttling_rds" {
  alarm_name          = "${var.name}-kms-throttling-rds"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "UserErrorCount"
  namespace           = "AWS/KMS"
  period              = 60
  statistic           = "Sum"
  threshold           = 10
  alarm_description   = "KMS API throttling detected for RDS key - secrets may be inaccessible"
  treat_missing_data  = "notBreaching"

  dimensions = {
    KeyId = aws_kms_key.rds.id
  }

  alarm_actions = var.alarm_sns_topic_arn != "" ? [var.alarm_sns_topic_arn] : []

  tags = merge(
    local.common_tags,
    {
      Purpose   = "KMS availability monitoring"
      Component = "monitoring"
      Service   = "keycloak"
    }
  )
}

#
# DocumentDB Audit Log Monitoring
#

# CloudWatch Alarm: DocumentDB Audit Log Failures
# Gated on is_aws_documentdb. Issue #955.
resource "aws_cloudwatch_metric_alarm" "documentdb_audit_log_failures" {
  count = local.is_aws_documentdb ? 1 : 0

  alarm_name          = "${var.name}-documentdb-audit-log-failures"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "AuditLogFailures"
  namespace           = "AWS/DocDB"
  period              = 300 # 5 minutes
  statistic           = "Sum"
  threshold           = 10
  alarm_description   = "DocumentDB audit logging failures - compliance gap"
  treat_missing_data  = "notBreaching"

  dimensions = {
    DBClusterIdentifier = aws_docdb_cluster.registry[0].id
  }

  alarm_actions = var.alarm_sns_topic_arn != "" ? [var.alarm_sns_topic_arn] : []

  tags = merge(
    local.common_tags,
    {
      Purpose   = "Audit log reliability"
      Component = "monitoring"
      Service   = "documentdb"
    }
  )
}

#
# S3 Bucket Monitoring
#

# CloudWatch Alarm: S3 ALB Logs Bucket Size High
resource "aws_cloudwatch_metric_alarm" "s3_alb_logs_size_high" {
  alarm_name          = "${var.name}-s3-alb-logs-size-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "BucketSizeBytes"
  namespace           = "AWS/S3"
  period              = 86400 # 1 day
  statistic           = "Average"
  threshold           = 107374182400 # 100 GB
  alarm_description   = "ALB logs bucket exceeds 100GB - check lifecycle policy"
  treat_missing_data  = "notBreaching"

  dimensions = {
    BucketName  = aws_s3_bucket.alb_logs.id
    StorageType = "StandardStorage"
  }

  alarm_actions = var.alarm_sns_topic_arn != "" ? [var.alarm_sns_topic_arn] : []

  tags = merge(
    local.common_tags,
    {
      Purpose   = "Cost control"
      Component = "monitoring"
      Service   = "alb-logging"
    }
  )
}

# CloudWatch Alarm: S3 CloudFront Logs Bucket Size High
resource "aws_cloudwatch_metric_alarm" "s3_cloudfront_logs_size_high" {
  alarm_name          = "${var.name}-s3-cloudfront-logs-size-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "BucketSizeBytes"
  namespace           = "AWS/S3"
  period              = 86400 # 1 day
  statistic           = "Average"
  threshold           = 107374182400 # 100 GB
  alarm_description   = "CloudFront logs bucket exceeds 100GB - check lifecycle policy"
  treat_missing_data  = "notBreaching"

  dimensions = {
    BucketName  = aws_s3_bucket.cloudfront_logs.id
    StorageType = "StandardStorage"
  }

  alarm_actions = var.alarm_sns_topic_arn != "" ? [var.alarm_sns_topic_arn] : []

  tags = merge(
    local.common_tags,
    {
      Purpose   = "Cost control"
      Component = "monitoring"
      Service   = "cloudfront-logging"
    }
  )
}
