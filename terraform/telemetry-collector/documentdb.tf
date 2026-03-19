# DocumentDB subnet group (requires at least 2 subnets in different AZs)
resource "aws_docdb_subnet_group" "telemetry" {
  name       = "telemetry-collector-docdb-subnet-group"
  subnet_ids = aws_subnet.private[*].id

  tags = {
    Name = "telemetry-collector-docdb-subnet-group"
  }
}

# DocumentDB cluster parameter group (customize settings)
resource "aws_docdb_cluster_parameter_group" "telemetry" {
  family      = "docdb5.0"
  name        = "telemetry-collector-docdb-params"
  description = "Custom parameter group for telemetry collector DocumentDB cluster"

  parameter {
    name  = "tls"
    value = "enabled"
  }

  parameter {
    name  = "ttl_monitor"
    value = "enabled"
  }

  tags = {
    Name = "telemetry-collector-docdb-params"
  }
}

# DocumentDB cluster
resource "aws_docdb_cluster" "telemetry" {
  cluster_identifier              = "telemetry-collector"
  engine                          = "docdb"
  master_username                 = var.documentdb_master_username
  master_password                 = random_password.documentdb_master.result
  backup_retention_period         = 7
  preferred_backup_window         = "03:00-04:00"  # 3-4 AM UTC
  preferred_maintenance_window    = "sun:04:00-sun:05:00"  # Sunday 4-5 AM UTC
  db_subnet_group_name            = aws_docdb_subnet_group.telemetry.name
  db_cluster_parameter_group_name = aws_docdb_cluster_parameter_group.telemetry.name
  vpc_security_group_ids          = [aws_security_group.documentdb.id]
  skip_final_snapshot             = var.deployment_stage == "testing"
  final_snapshot_identifier       = var.deployment_stage == "production" ? "telemetry-collector-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}" : null
  enabled_cloudwatch_logs_exports = ["audit", "profiler"]
  storage_encrypted               = true

  tags = {
    Name = "telemetry-collector"
  }
}

# DocumentDB cluster instance (single instance for testing, can add more for production)
resource "aws_docdb_cluster_instance" "telemetry" {
  identifier         = "telemetry-collector-instance-1"
  cluster_identifier = aws_docdb_cluster.telemetry.id
  instance_class     = var.documentdb_instance_class

  tags = {
    Name = "telemetry-collector-instance-1"
  }
}

# Random password for DocumentDB master user
resource "random_password" "documentdb_master" {
  length  = 32
  special = true
  # Exclude problematic characters for connection strings
  override_special = "!#$%&*()-_=+[]{}<>:?"
}
