output "collector_url" {
  description = "Telemetry collector API endpoint URL"
  value       = "${trimsuffix(aws_apigatewayv2_stage.telemetry.invoke_url, "/")}/v1/collect"
}

output "api_gateway_id" {
  description = "API Gateway HTTP API ID"
  value       = aws_apigatewayv2_api.telemetry.id
}

output "lambda_function_name" {
  description = "Lambda function name"
  value       = aws_lambda_function.telemetry_collector.function_name
}

output "lambda_function_arn" {
  description = "Lambda function ARN"
  value       = aws_lambda_function.telemetry_collector.arn
}

output "documentdb_endpoint" {
  description = "DocumentDB cluster endpoint"
  value       = aws_docdb_cluster.telemetry.endpoint
}

output "documentdb_secret_arn" {
  description = "Secrets Manager ARN for DocumentDB credentials"
  value       = aws_secretsmanager_secret.documentdb_credentials.arn
}

output "rate_limit_table_name" {
  description = "DynamoDB rate limiting table name"
  value       = aws_dynamodb_table.rate_limit.name
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group for Lambda function"
  value       = aws_cloudwatch_log_group.telemetry_collector.name
}

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.telemetry.id
}

output "custom_domain_url" {
  description = "Custom domain URL (if configured)"
  value       = var.custom_domain != "" ? "https://${var.custom_domain}/v1/collect" : "Not configured"
}

output "bastion_public_ip" {
  description = "Public IP of the bastion host (if enabled)"
  value       = var.bastion_enabled ? aws_instance.bastion[0].public_ip : "Bastion not enabled"
}

output "bastion_ssh_command" {
  description = "SSH command to connect to the bastion host"
  value       = var.bastion_enabled ? "ssh -i <your-key.pem> ec2-user@${aws_instance.bastion[0].public_ip}" : "Bastion not enabled"
}

output "aws_region" {
  description = "AWS region of deployment"
  value       = var.aws_region
}
