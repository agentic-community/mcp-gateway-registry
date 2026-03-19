# DynamoDB table for rate limiting
resource "aws_dynamodb_table" "rate_limit" {
  name         = "telemetry-collector-rate-limit"
  billing_mode = "PAY_PER_REQUEST"  # On-demand pricing
  hash_key     = "ip_hash"

  attribute {
    name = "ip_hash"
    type = "S"
  }

  ttl {
    attribute_name = "expiry_time"
    enabled        = true
  }

  point_in_time_recovery {
    enabled = var.deployment_stage == "production"
  }

  tags = {
    Name = "telemetry-collector-rate-limit"
  }
}
