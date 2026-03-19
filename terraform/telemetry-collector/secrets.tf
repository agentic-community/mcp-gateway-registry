# Secrets Manager secret for DocumentDB credentials
resource "aws_secretsmanager_secret" "documentdb_credentials" {
  name        = "telemetry-collector-docdb"
  description = "DocumentDB credentials for telemetry collector"

  tags = {
    Name = "telemetry-collector-documentdb-credentials"
  }
}

# Store DocumentDB credentials in Secrets Manager
resource "aws_secretsmanager_secret_version" "documentdb_credentials" {
  secret_id = aws_secretsmanager_secret.documentdb_credentials.id

  secret_string = jsonencode({
    username = aws_docdb_cluster.telemetry.master_username
    password = random_password.documentdb_master.result
    endpoint = aws_docdb_cluster.telemetry.endpoint
    port     = aws_docdb_cluster.telemetry.port
    database = var.documentdb_database_name
  })
}
