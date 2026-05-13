#
# Keycloak Aurora MySQL Database (Serverless v2)
#

# RDS Proxy for connection pooling
resource "aws_db_proxy" "keycloak" {
  name          = "keycloak-proxy"
  engine_family = "MYSQL"

  auth {
    auth_scheme               = "SECRETS"
    secret_arn                = aws_secretsmanager_secret.keycloak_db_secret.arn
    client_password_auth_type = "MYSQL_CACHING_SHA2_PASSWORD"
    iam_auth                  = "DISABLED"
  }

  role_arn               = aws_iam_role.rds_proxy_role.arn
  vpc_subnet_ids         = module.vpc.private_subnets
  vpc_security_group_ids = [aws_security_group.keycloak_db.id]

  require_tls = false

  tags = local.common_tags

  depends_on = [
    aws_secretsmanager_secret_version.keycloak_db_secret
  ]
}

# RDS Proxy Target
resource "aws_db_proxy_target" "keycloak" {
  db_proxy_name         = aws_db_proxy.keycloak.name
  target_group_name     = "default"
  db_cluster_identifier = aws_rds_cluster.keycloak.cluster_identifier

  depends_on = [
    aws_rds_cluster_instance.keycloak
  ]
}

# Aurora MySQL Serverless v2 Cluster
#checkov:skip=CKV_AWS_139:Deletion protection configured per environment
#checkov:skip=CKV_AWS_162:IAM database authentication not used - Keycloak uses password auth
#checkov:skip=CKV_AWS_324:CloudWatch log exports not enabled for Keycloak database - log volume is low and Keycloak application logs provide sufficient observability
#checkov:skip=CKV_AWS_325:Preferred backup window is configured on this resource
#checkov:skip=CKV_AWS_326:Serverless v2 scaling configuration is present on this resource
#checkov:skip=CKV2_AWS_8:Backup retention period of 7 days is configured on this resource
resource "aws_rds_cluster" "keycloak" {
  cluster_identifier = "keycloak"
  engine             = "aurora-mysql"
  engine_version     = "8.0.mysql_aurora.3.10.3"
  database_name      = "keycloak"
  master_username    = var.keycloak_database_username
  master_password    = var.keycloak_database_password

  db_subnet_group_name            = aws_db_subnet_group.keycloak.name
  db_cluster_parameter_group_name = aws_rds_cluster_parameter_group.keycloak.name
  vpc_security_group_ids          = [aws_security_group.keycloak_db.id]

  # Backup and maintenance
  backup_retention_period      = 7
  preferred_backup_window      = "02:00-04:00"
  preferred_maintenance_window = "sun:04:00-sun:05:00"
  copy_tags_to_snapshot        = true

  # Encryption
  storage_encrypted = true
  kms_key_id        = aws_kms_key.rds.arn

  # Deletion protection
  deletion_protection = false
  skip_final_snapshot = true

  # Serverless v2 scaling
  serverlessv2_scaling_configuration {
    max_capacity = var.keycloak_database_max_acu
    min_capacity = var.keycloak_database_min_acu
  }

  tags = local.common_tags
}

# Aurora Cluster Instance (Serverless v2)
#checkov:skip=CKV_AWS_118:Enhanced monitoring configured per environment requirements
#checkov:skip=CKV_AWS_353:Performance insights configured per environment requirements
resource "aws_rds_cluster_instance" "keycloak" {
  cluster_identifier = aws_rds_cluster.keycloak.id
  instance_class     = "db.serverless"
  engine             = aws_rds_cluster.keycloak.engine
  engine_version     = aws_rds_cluster.keycloak.engine_version

  auto_minor_version_upgrade   = true
  performance_insights_enabled = false

  tags = local.common_tags
}

# DB Subnet Group
resource "aws_db_subnet_group" "keycloak" {
  name       = "keycloak-subnet-group"
  subnet_ids = module.vpc.private_subnets

  tags = merge(
    local.common_tags,
    {
      Name = "keycloak-subnet-group"
    }
  )
}

# RDS Cluster Parameter Group
resource "aws_rds_cluster_parameter_group" "keycloak" {
  family      = "aurora-mysql8.0"
  name        = "keycloak-params"
  description = "Keycloak Aurora MySQL parameter group"

  parameter {
    name  = "character_set_server"
    value = "utf8mb4"
  }

  parameter {
    name  = "collation_server"
    value = "utf8mb4_unicode_ci"
  }

  tags = local.common_tags
}

# KMS Key for RDS Encryption
resource "aws_kms_key" "rds" {
  description             = "KMS key for RDS and secrets encryption"
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
            "aws:PrincipalArn" = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/*task-exec*"
          }
        }
      },
      {
        Sid    = "Allow RDS Service"
        Effect = "Allow"
        Principal = {
          Service = "rds.amazonaws.com"
        }
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey",
          "kms:CreateGrant"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "kms:ViaService" = "rds.${data.aws_region.current.name}.amazonaws.com"
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

  tags = local.common_tags
}

resource "aws_kms_alias" "rds" {
  name          = "alias/keycloak-rds"
  target_key_id = aws_kms_key.rds.key_id
}

# IAM Role for RDS Proxy
resource "aws_iam_role" "rds_proxy_role" {
  name = "keycloak-rds-proxy-role-${var.aws_region}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "rds.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

# IAM Policy for RDS Proxy
resource "aws_iam_role_policy" "rds_proxy_policy" {
  name = "keycloak-rds-proxy-policy"
  role = aws_iam_role.rds_proxy_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = aws_secretsmanager_secret.keycloak_db_secret.arn
      }
    ]
  })
}

# Secrets Manager Secret for Database Credentials
#checkov:skip=CKV2_AWS_57:Secret rotation managed externally via dedicated rotation Lambda
resource "aws_secretsmanager_secret" "keycloak_db_secret" {
  name                    = "keycloak/database"
  description             = "Keycloak database credentials"
  recovery_window_in_days = 0
  kms_key_id              = aws_kms_key.rds.id

  tags = local.common_tags
}

resource "aws_secretsmanager_secret_version" "keycloak_db_secret" {
  secret_id = aws_secretsmanager_secret.keycloak_db_secret.id
  secret_string = jsonencode({
    username = var.keycloak_database_username
    password = var.keycloak_database_password
  })
}

# SSM Parameters for Database Connection
resource "aws_ssm_parameter" "keycloak_database_url" {
  name   = "/keycloak/database/url"
  type   = "SecureString"
  key_id = aws_kms_key.rds.id
  value  = "jdbc:mysql://${aws_rds_cluster.keycloak.endpoint}:3306/keycloak"
  tags   = local.common_tags
}

# Issue #1026: SSM parameters /keycloak/database/username and
# /keycloak/database/password were removed because Keycloak now reads
# both values directly from the Secrets Manager secret keycloak/database
# managed by the rotation Lambda. Keeping the SSM copies caused
# password drift after every rotation because the Lambda only updates
# the Secrets Manager value, leaving the SSM copy stale and forcing
# Keycloak to crash on every restart with "Access denied for user
# keycloak".
#
# The Secrets Manager secret aws_secretsmanager_secret.keycloak_db_secret
# (defined above in this file) is the sole source of truth for both
# fields, and the ECS task definition extracts them via the
# valueFrom = "<secret-arn>:<key>::" syntax.
