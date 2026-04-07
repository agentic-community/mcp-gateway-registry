#
# Amazon DocumentDB (Regular) Cluster Infrastructure for MCP Gateway Registry
#
# This configuration creates a regular DocumentDB Cluster (instance-based) with VPC access
# for the MCP Gateway Registry backend storage and vector search.
#
# This replaces DocumentDB Elastic to enable vector search support with HNSW indexes.
#

#
# Security Group for DocumentDB
#
resource "aws_security_group" "documentdb" {
  name        = "${var.name}-v2-documentdb-sg"
  description = "Security group for DocumentDB Elastic Cluster" # Keep original description to avoid recreation
  vpc_id      = module.vpc.vpc_id

  tags = merge(
    local.common_tags,
    {
      Name      = "${var.name}-v2-documentdb-sg"
      Component = "documentdb"
    }
  )

  lifecycle {
    ignore_changes = [description] # Ignore description changes to avoid forcing recreation
  }
}

# Ingress from Registry service
resource "aws_vpc_security_group_ingress_rule" "documentdb_from_registry" {
  security_group_id = aws_security_group.documentdb.id

  referenced_security_group_id = module.mcp_gateway.ecs_security_group_ids.registry
  from_port                    = 27017
  to_port                      = 27017
  ip_protocol                  = "tcp"
  description                  = "Allow MongoDB protocol from Registry ECS service to DocumentDB"

  tags = merge(
    local.common_tags,
    {
      Name = "documentdb-from-registry"
    }
  )
}

# Ingress from Auth service
resource "aws_vpc_security_group_ingress_rule" "documentdb_from_auth" {
  security_group_id = aws_security_group.documentdb.id

  referenced_security_group_id = module.mcp_gateway.ecs_security_group_ids.auth
  from_port                    = 27017
  to_port                      = 27017
  ip_protocol                  = "tcp"
  description                  = "Allow MongoDB protocol from Auth ECS service to DocumentDB"

  tags = merge(
    local.common_tags,
    {
      Name = "documentdb-from-auth"
    }
  )
}

# Egress
resource "aws_vpc_security_group_egress_rule" "documentdb_egress" {
  security_group_id = aws_security_group.documentdb.id

  cidr_ipv4   = "0.0.0.0/0"
  ip_protocol = "-1"
  description = "Allow all outbound traffic"

  tags = merge(
    local.common_tags,
    {
      Name = "documentdb-egress-all"
    }
  )
}

# Registry -> DocumentDB
resource "aws_vpc_security_group_egress_rule" "registry_to_documentdb" {
  security_group_id = module.mcp_gateway.ecs_security_group_ids.registry

  referenced_security_group_id = aws_security_group.documentdb.id
  from_port                    = 27017
  to_port                      = 27017
  ip_protocol                  = "tcp"
  description                  = "Allow Registry service to connect to DocumentDB"

  tags = merge(
    local.common_tags,
    {
      Name = "registry-to-documentdb"
    }
  )
}

# Auth -> DocumentDB
resource "aws_vpc_security_group_egress_rule" "auth_to_documentdb" {
  security_group_id = module.mcp_gateway.ecs_security_group_ids.auth

  referenced_security_group_id = aws_security_group.documentdb.id
  from_port                    = 27017
  to_port                      = 27017
  ip_protocol                  = "tcp"
  description                  = "Allow Auth service to connect to DocumentDB"

  tags = merge(
    local.common_tags,
    {
      Name = "auth-to-documentdb"
    }
  )
}

#
# KMS Key for DocumentDB Encryption
#
resource "aws_kms_key" "documentdb" {
  description             = "KMS key for DocumentDB Cluster and secrets encryption"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow ECS Task Execution Role to Decrypt"
        Effect = "Allow"
        Principal = {
          AWS = "*"
        }
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "aws:PrincipalAccount" = data.aws_caller_identity.current.account_id
          }
          StringLike = {
            "aws:PrincipalArn" = [
              # ECS task execution roles (e.g., mcp-gateway-task-exec-role)
              "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/*task-exec*",
              # ECS task roles for v2 deployments (e.g., mcp-gateway-v2-registry-task-role)
              "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/mcp-gateway-v2-*",
            ]
          }
        }
      },
      {
        Sid    = "Allow CloudWatch Logs"
        Effect = "Allow"
        Principal = {
          Service = "logs.${data.aws_region.current.name}.amazonaws.com"
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:CreateGrant",
          "kms:DescribeKey"
        ]
        Resource = "*"
        Condition = {
          ArnLike = {
            "kms:EncryptionContext:aws:logs:arn" = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:*"
          }
        }
      }
    ]
  })

  tags = merge(
    local.common_tags,
    {
      Name      = "${var.name}-documentdb-key"
      Component = "documentdb"
    }
  )
}

resource "aws_kms_alias" "documentdb" {
  name          = "alias/${var.name}-documentdb"
  target_key_id = aws_kms_key.documentdb.key_id
}

#
# Secrets Manager Secret for DocumentDB Credentials
#
resource "aws_secretsmanager_secret" "documentdb_credentials" {
  name                    = "${var.name}/documentdb/credentials"
  description             = "DocumentDB Cluster admin credentials"
  recovery_window_in_days = 0
  kms_key_id              = aws_kms_key.documentdb.id

  tags = merge(
    local.common_tags,
    {
      Component = "documentdb"
    }
  )
}

resource "aws_secretsmanager_secret_version" "documentdb_credentials" {
  secret_id = aws_secretsmanager_secret.documentdb_credentials.id
  secret_string = jsonencode({
    username = var.documentdb_admin_username
    password = var.documentdb_admin_password
    engine   = "docdb"
  })
}

#
# DocumentDB Subnet Group
#
resource "aws_docdb_subnet_group" "registry" {
  name       = "${var.name}-registry-subnet-group"
  subnet_ids = module.vpc.private_subnets

  tags = merge(
    local.common_tags,
    {
      Name      = "${var.name}-registry-subnet-group"
      Component = "documentdb"
    }
  )
}

#
# DocumentDB Cluster Parameter Group
#
resource "aws_docdb_cluster_parameter_group" "registry" {
  family      = "docdb5.0"
  name        = "${var.name}-registry-params"
  description = "DocumentDB cluster parameter group for MCP Gateway Registry"

  # Enable TLS
  parameter {
    name  = "tls"
    value = "enabled"
  }

  # Audit logs - enabled for compliance and security monitoring
  parameter {
    name  = "audit_logs"
    value = "enabled"
  }

  # TTL monitor (for automatic document expiration)
  parameter {
    name  = "ttl_monitor"
    value = "enabled"
  }

  tags = merge(
    local.common_tags,
    {
      Name      = "${var.name}-registry-params"
      Component = "documentdb"
    }
  )
}

#
# DocumentDB Cluster
#
resource "aws_docdb_cluster" "registry" {
  cluster_identifier = "${var.name}-registry"

  # Engine
  engine         = "docdb"
  engine_version = "5.0.0"

  # Authentication
  master_username = var.documentdb_admin_username
  master_password = var.documentdb_admin_password

  # Network configuration
  db_subnet_group_name   = aws_docdb_subnet_group.registry.name
  vpc_security_group_ids = [aws_security_group.documentdb.id]
  port                   = 27017

  # Backup configuration
  backup_retention_period      = 7
  preferred_backup_window      = "02:00-04:00"
  preferred_maintenance_window = "sun:04:00-sun:05:00"
  skip_final_snapshot          = false
  final_snapshot_identifier    = "${var.name}-registry-final-snapshot"

  # Encryption
  storage_encrypted = true
  kms_key_id        = aws_kms_key.documentdb.arn

  # Parameter group
  db_cluster_parameter_group_name = aws_docdb_cluster_parameter_group.registry.name

  # Deletion protection (enable for production)
  deletion_protection = false

  # Enable CloudWatch logs
  enabled_cloudwatch_logs_exports = ["audit", "profiler"]

  tags = merge(
    local.common_tags,
    {
      Name        = "${var.name}-registry-docdb"
      Component   = "documentdb"
      Environment = "production"
      Service     = "mcp-gateway-registry"
    }
  )
}

#
# DocumentDB Cluster Instances
#
# Primary instance
resource "aws_docdb_cluster_instance" "registry_primary" {
  identifier         = "${var.name}-registry-primary"
  cluster_identifier = aws_docdb_cluster.registry.id

  # Instance class (can be adjusted based on needs)
  # db.t3.medium = 2 vCPU, 4 GB RAM - good starting point
  # db.r5.large = 2 vCPU, 16 GB RAM - for larger workloads
  instance_class = var.documentdb_instance_class

  # Monitoring
  auto_minor_version_upgrade  = true
  enable_performance_insights = false # Not available for DocumentDB yet
  promotion_tier              = 0

  tags = merge(
    local.common_tags,
    {
      Name      = "${var.name}-registry-primary"
      Component = "documentdb"
      Role      = "primary"
    }
  )
}

# Read replica instance (optional, for high availability)
# Uncomment to enable a read replica
# resource "aws_docdb_cluster_instance" "registry_replica" {
#   count              = var.documentdb_replica_count
#   identifier         = "${var.name}-registry-replica-${count.index + 1}"
#   cluster_identifier = aws_docdb_cluster.registry.id
#
#   instance_class = var.documentdb_instance_class
#
#   auto_minor_version_upgrade = true
#   promotion_tier            = count.index + 1
#
#   tags = merge(
#     local.common_tags,
#     {
#       Name      = "${var.name}-registry-replica-${count.index + 1}"
#       Component = "documentdb"
#       Role      = "replica"
#     }
#   )
# }

#
# Update SSM Parameters with new cluster endpoints
#
resource "aws_ssm_parameter" "documentdb_endpoint" {
  name        = "/${var.name}/documentdb/endpoint"
  description = "DocumentDB Cluster endpoint"
  type        = "String"
  value       = aws_docdb_cluster.registry.endpoint
  overwrite   = true

  tags = merge(
    local.common_tags,
    {
      Component = "documentdb"
    }
  )
}

resource "aws_ssm_parameter" "documentdb_reader_endpoint" {
  name        = "/${var.name}/documentdb/reader_endpoint"
  description = "DocumentDB Cluster reader endpoint"
  type        = "String"
  value       = aws_docdb_cluster.registry.reader_endpoint

  tags = merge(
    local.common_tags,
    {
      Component = "documentdb"
    }
  )
}

resource "aws_ssm_parameter" "documentdb_connection_string" {
  name        = "/${var.name}/documentdb/connection_string"
  description = "DocumentDB Cluster connection string"
  type        = "SecureString"
  key_id      = aws_kms_key.documentdb.id
  # AWS DocumentDB only supports SCRAM-SHA-1 (not SCRAM-SHA-256 as of v5.0)
  # TODO: Update to SCRAM-SHA-256 when AWS DocumentDB adds support
  value = format(
    "mongodb://%s:%s@%s:27017/?authMechanism=SCRAM-SHA-1&authSource=admin&tls=true&tlsCAFile=global-bundle.pem&replicaSet=rs0&readPreference=secondaryPreferred&retryWrites=false",
    var.documentdb_admin_username,
    var.documentdb_admin_password,
    aws_docdb_cluster.registry.endpoint
  )
  overwrite = true

  tags = merge(
    local.common_tags,
    {
      Component = "documentdb"
    }
  )
}
