#
# ALB Access Logging with S3 Security Hardening
#

# S3 bucket for ALB access logs
resource "aws_s3_bucket" "alb_logs" {
  #checkov:skip=CKV_AWS_18:This is a logging destination bucket - enabling access logging would create recursion
  #checkov:skip=CKV_AWS_144:Cross-region replication not required for logging bucket
  #checkov:skip=CKV_AWS_145:SSE-S3 encryption is sufficient for logging bucket
  #checkov:skip=CKV2_AWS_62:Event notifications not required for logging bucket
  bucket = "${var.name}-${var.aws_region}-${data.aws_caller_identity.current.account_id}-alb-logs"

  tags = merge(
    local.common_tags,
    {
      Purpose   = "ALB access logs"
      Component = "logging"
    }
  )
}


# Block public access
resource "aws_s3_bucket_public_access_block" "alb_logs" {
  bucket = aws_s3_bucket.alb_logs.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}


# Enable versioning
resource "aws_s3_bucket_versioning" "alb_logs" {
  bucket = aws_s3_bucket.alb_logs.id

  versioning_configuration {
    status = "Enabled"
  }
}


# Server-side encryption with SSE-S3 (AES256)
# Using SSE-S3 instead of KMS for ALB logs per AWS best practices
# KMS encryption for ALB logs requires complex permission setup and can cause access issues
# SSE-S3 provides strong encryption (AES-256) without the permission complexity
resource "aws_s3_bucket_server_side_encryption_configuration" "alb_logs" {
  bucket = aws_s3_bucket.alb_logs.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}


# Lifecycle policy - delete old logs after 90 days
resource "aws_s3_bucket_lifecycle_configuration" "alb_logs" {
  bucket = aws_s3_bucket.alb_logs.id

  rule {
    id     = "delete-old-logs"
    status = "Enabled"

    expiration {
      days = 90
    }
  }

  rule {
    id     = "abort-incomplete-multipart-uploads"
    status = "Enabled"

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}


# Bucket policy for ALB logging with TLS enforcement
# Using modern service principal approach (recommended by AWS)
# https://docs.aws.amazon.com/elasticloadbalancing/latest/application/enable-access-logging.html
resource "aws_s3_bucket_policy" "alb_logs" {
  bucket = aws_s3_bucket.alb_logs.id

  # Ensure all bucket configurations are applied before the policy
  # This includes encryption, versioning, and public access blocks
  depends_on = [
    aws_s3_bucket_public_access_block.alb_logs,
    aws_s3_bucket_server_side_encryption_configuration.alb_logs,
    aws_s3_bucket_versioning.alb_logs
  ]

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "EnforceTLS"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource = [
          aws_s3_bucket.alb_logs.arn,
          "${aws_s3_bucket.alb_logs.arn}/*"
        ]
        Condition = {
          Bool = {
            "aws:SecureTransport" = "false"
          }
        }
      },
      {
        Sid    = "AWSLogDeliveryWrite"
        Effect = "Allow"
        Principal = {
          Service = "logdelivery.elasticloadbalancing.amazonaws.com"
        }
        Action   = "s3:PutObject"
        Resource = "${aws_s3_bucket.alb_logs.arn}/*"
        Condition = {
          StringEquals = {
            "s3:x-amz-acl" = "bucket-owner-full-control"
          }
        }
      },
      {
        Sid    = "AWSLogDeliveryAclCheck"
        Effect = "Allow"
        Principal = {
          Service = "logdelivery.elasticloadbalancing.amazonaws.com"
        }
        Action   = "s3:GetBucketAcl"
        Resource = aws_s3_bucket.alb_logs.arn
      }
    ]
  })
}


# Wait for S3 bucket policy propagation before enabling ALB logging
# AWS S3 bucket policies can take up to 15-30 seconds to propagate
# Without this delay, ALBs may fail to enable logging due to permission check failures
resource "time_sleep" "wait_for_bucket_policy" {
  depends_on = [aws_s3_bucket_policy.alb_logs]

  create_duration = "30s"
}


# Output for reference
output "alb_logs_bucket" {
  description = "S3 bucket for ALB access logs"
  value       = aws_s3_bucket.alb_logs.id
}


output "alb_logs_bucket_arn" {
  description = "ARN of S3 bucket for ALB access logs"
  value       = aws_s3_bucket.alb_logs.arn
}
