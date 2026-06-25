#
# Secret Rotation Configuration
#
# This file adds automatic rotation to existing secrets defined in:
# - documentdb.tf: aws_secretsmanager_secret.documentdb_credentials
# - keycloak-database.tf: aws_secretsmanager_secret.keycloak_db_secret
#
# Secrets are rotated every 30 days automatically by Lambda functions.
#

#
# Enable Rotation for DocumentDB Credentials
# Gated on is_aws_documentdb. The rotation Lambda and its permissions are
# also gated in secret-rotation.tf. Issue #955.
#
resource "aws_secretsmanager_secret_rotation" "documentdb_credentials" {
  count = local.is_aws_documentdb ? 1 : 0

  secret_id           = aws_secretsmanager_secret.documentdb_credentials[0].id
  rotation_lambda_arn = aws_lambda_function.documentdb_rotation[0].arn

  # Do NOT rotate on creation. Enabling rotation triggers an immediate rotation
  # by default, which races with cluster provisioning during the initial apply:
  # the Lambda advances the secret before the database is reachable, leaving the
  # stored secret out of sync with the actual DB password. Defer the first
  # rotation to the 30-day schedule, once the stack is stable.
  rotate_immediately = false

  rotation_rules {
    automatically_after_days = 30
  }

  depends_on = [
    aws_lambda_permission.documentdb_rotation,
    aws_secretsmanager_secret_version.documentdb_credentials
  ]
}

#
# Enable Rotation for Keycloak Database Credentials
#
resource "aws_secretsmanager_secret_rotation" "keycloak_db_secret" {
  secret_id           = aws_secretsmanager_secret.keycloak_db_secret.id
  rotation_lambda_arn = aws_lambda_function.rds_rotation.arn

  # Do NOT rotate on creation. The default immediate rotation fires while the
  # Aurora cluster is still being created during the initial apply, so the
  # Lambda updates the Secrets Manager value without applying the new password
  # to Aurora. Keycloak then reads the rotated secret and crash-loops with
  # "Access denied for user 'keycloak'". Deferring to the 30-day schedule lets
  # the first rotation run against a healthy, reachable database.
  rotate_immediately = false

  rotation_rules {
    automatically_after_days = 30
  }

  depends_on = [
    aws_lambda_permission.rds_rotation,
    aws_secretsmanager_secret_version.keycloak_db_secret
  ]
}
