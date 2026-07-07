# AWS Lambda Functions for Secret Rotation

This directory contains Lambda functions that implement AWS Secrets Manager rotation protocol for database credentials.

## Overview

The Lambda functions implement the standard AWS Secrets Manager rotation process:

1. **createSecret**: Generate a new random password and create an AWSPENDING version
2. **setSecret**: Update the database with the new password
3. **testSecret**: Verify the new password works
4. **finishSecret**: Promote AWSPENDING to AWSCURRENT

## Functions

### rotate-documentdb/

Rotates DocumentDB cluster master password.

**Files:**
- `index.py`: Main Lambda handler implementing 4-step rotation
- `requirements.txt`: Python dependencies (boto3)

**Environment Variables:**
- `SECRETS_MANAGER_ENDPOINT`: Secrets Manager API endpoint
- `EXCLUDE_CHARACTERS`: Characters to exclude from generated passwords (default: `` /@"'+:?#&!=%  ``)

**IAM Permissions Required:**
- `secretsmanager:DescribeSecret`
- `secretsmanager:GetSecretValue`
- `secretsmanager:PutSecretValue`
- `secretsmanager:UpdateSecretVersionStage`
- `secretsmanager:GetRandomPassword`
- `kms:Decrypt`
- `kms:GenerateDataKey`
- `docdb:DescribeDBClusters`
- `docdb:ModifyDBCluster`
- VPC networking permissions for private subnet access

**Network Configuration:**
- Deployed in VPC private subnets
- Security group allows egress to DocumentDB on port 27017
- Security group allows egress to HTTPS (443) for AWS API calls

### rotate-rds/

Rotates RDS Aurora MySQL cluster master password (Keycloak database).

**Files:**
- `index.py`: Main Lambda handler implementing 4-step rotation
- `requirements.txt`: Python dependencies (boto3)

**Environment Variables:**
- `SECRETS_MANAGER_ENDPOINT`: Secrets Manager API endpoint
- `EXCLUDE_CHARACTERS`: Characters to exclude from generated passwords (default: `` /@"'+:?#&!=%  ``)

**IAM Permissions Required:**
- `secretsmanager:DescribeSecret`
- `secretsmanager:GetSecretValue`
- `secretsmanager:PutSecretValue`
- `secretsmanager:UpdateSecretVersionStage`
- `secretsmanager:GetRandomPassword`
- `kms:Decrypt`
- `kms:GenerateDataKey`
- `rds:DescribeDBClusters`
- `rds:ModifyDBCluster`
- VPC networking permissions for private subnet access

**Network Configuration:**
- Deployed in VPC private subnets
- Security group allows egress to RDS on port 3306
- Security group allows egress to HTTPS (443) for AWS API calls

## Secret Format

### DocumentDB Secret
```json
{
  "username": "admin",
  "password": "randomly-generated-32-char-password",
  "engine": "docdb",
  "cluster_id": "mcp-gateway-registry"
}
```

### RDS Secret
```json
{
  "username": "keycloak",
  "password": "randomly-generated-32-char-password",
  "cluster_id": "keycloak"
}
```

## Deployment

The Lambda functions are automatically deployed via Terraform:

```hcl
# Deploy from terraform/aws-ecs/
terraform init
terraform plan
terraform apply
```

The deployment process:
1. Creates ZIP archives from Lambda source code
2. Uploads Lambda functions to AWS
3. Configures IAM roles and policies
4. Sets up VPC networking and security groups
5. Enables rotation on secrets with 30-day interval

## Rotation Schedule

Secrets are automatically rotated every 30 days. You can also trigger manual rotation:

```bash
aws secretsmanager rotate-secret --secret-id <secret-name>
```

## Monitoring

Lambda execution logs are sent to CloudWatch Logs:
- `/aws/lambda/mcp-gateway-rotate-documentdb`
- `/aws/lambda/mcp-gateway-rotate-rds`

Log retention: 30 days

## Testing

To test rotation without waiting 30 days:

```bash
# Rotate DocumentDB secret
aws secretsmanager rotate-secret --secret-id mcp-gateway/documentdb/credentials

# Rotate RDS secret
aws secretsmanager rotate-secret --secret-id keycloak/database
```

Monitor the rotation:
```bash
# Check secret status
aws secretsmanager describe-secret --secret-id <secret-name>

# View Lambda logs
aws logs tail /aws/lambda/mcp-gateway-rotate-documentdb --follow
aws logs tail /aws/lambda/mcp-gateway-rotate-rds --follow
```

## Security Considerations

1. **Password Complexity**: 32 characters, alphanumeric + special characters
2. **Excluded Characters**: `/@"'\` to avoid shell/SQL escaping issues
3. **Encryption**: Secrets encrypted with KMS customer-managed keys
4. **Network Isolation**: Lambda functions run in private subnets only
5. **Least Privilege**: IAM roles grant only required permissions
6. **Audit Trail**: All rotations logged to CloudWatch

## Troubleshooting

### Rotation Fails at setSecret Step

Check:
- Lambda has network access to database (security groups)
- Database cluster is in `available` state
- IAM role has `ModifyDBCluster` permission

### Rotation Fails at testSecret Step

Check:
- Database cluster status after password change
- CloudWatch Logs for detailed error messages

### Lambda Timeout

Default timeout: 300 seconds (5 minutes)

If rotation takes longer:
1. Check database cluster is not under heavy load
2. Verify network latency between Lambda and database
3. Review CloudWatch Logs for bottlenecks

## References

- [AWS Secrets Manager Rotation](https://docs.aws.amazon.com/secretsmanager/latest/userguide/rotating-secrets.html)
- [DocumentDB Security](https://docs.aws.amazon.com/documentdb/latest/developerguide/security.html)
- [RDS Aurora Security](https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/UsingWithRDS.html)
- [Lambda VPC Configuration](https://docs.aws.amazon.com/lambda/latest/dg/configuration-vpc.html)
