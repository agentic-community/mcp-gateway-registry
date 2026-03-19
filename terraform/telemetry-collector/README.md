# Telemetry Collector Infrastructure

Server-side telemetry collector for MCP Gateway Registry (Issue #559).

## Overview

Privacy-first serverless telemetry collector that receives anonymous usage data from registry instances worldwide.

**Architecture:**
- **API Gateway HTTP API** - HTTPS endpoint for telemetry events
- **Lambda Function** - VPC-enabled, validates and stores events
- **DynamoDB** - Privacy-preserving rate limiting (IP hashing)
- **DocumentDB** - MongoDB-compatible storage with 365-day TTL
- **Secrets Manager** - Secure credential storage

**Key Features:**
- ✅ Always returns 204 (no information leakage)
- ✅ Hash-based rate limiting (no IP storage)
- ✅ VPC-secured DocumentDB
- ✅ Fail-silent design (never blocks clients)
- ✅ TLS encryption everywhere

## Prerequisites

### AWS Requirements
- AWS CLI v2 configured with credentials
- Terraform >= 1.0
- AWS account with permissions for:
  - VPC, EC2 (NAT Gateway)
  - Lambda, API Gateway
  - DocumentDB
  - DynamoDB
  - Secrets Manager
  - CloudWatch Logs

### Local Requirements
```bash
# Install Terraform
brew install terraform  # macOS
# or
sudo apt-get install terraform  # Linux

# Verify AWS CLI
aws sts get-caller-identity

# Verify Terraform
terraform version
```

## Cost Estimates

### Testing Account (~$85-90/month)
- Lambda: Free tier (1M invocations/month)
- API Gateway HTTP API: ~$1/month
- DynamoDB: Free tier (25GB storage)
- Secrets Manager: $0.40/month
- CloudWatch Logs: ~$0.50/month
- DocumentDB (db.t3.medium): ~$50/month
- NAT Gateway (2 AZs): ~$32/month
- Data transfer: ~$1/month

### Production Account (~$195-200/month)
- Lambda: ~$0.20/month
- API Gateway HTTP API: ~$0.10/month
- DynamoDB: ~$1.25/month
- Secrets Manager: $0.40/month
- CloudWatch Logs: ~$0.50/month
- DocumentDB (db.r5.large): ~$160/month
- NAT Gateway (2 AZs): ~$32/month
- Data transfer: ~$1/month

**Note:** DocumentDB costs are significantly higher than MongoDB Atlas M0 (free), but provide AWS-native integration and VPC security.

## Deployment

### Step 1: Clone Repository

```bash
git clone https://github.com/agentic-community/mcp-gateway-registry.git
cd mcp-gateway-registry/terraform/telemetry-collector
```

### Step 2: Configure Variables

```bash
# Copy example configuration
cp terraform.tfvars.example terraform.tfvars

# Edit configuration
vi terraform.tfvars
```

**Required variables:**
```hcl
aws_region = "us-east-1"
deployment_stage = "testing"  # or "production"
documentdb_instance_class = "db.t3.medium"  # or "db.r5.large"
```

**Optional variables (production):**
```hcl
custom_domain = "telemetry.mcpgateway.io"
route53_zone_id = "Z1234567890ABC"
alarm_email = "alerts@example.com"
```

### Step 3: Deploy Infrastructure

```bash
# Initialize Terraform
terraform init

# Preview changes
terraform plan

# Deploy (takes ~15-20 minutes due to DocumentDB cluster creation)
terraform apply
```

**Expected output:**
```
Apply complete! Resources: 35 added, 0 changed, 0 destroyed.

Outputs:
collector_url = "https://abc123.execute-api.us-east-1.amazonaws.com/v1/collect"
documentdb_endpoint = "telemetry-collector.cluster-abc123.us-east-1.docdb.amazonaws.com:27017"
lambda_function_name = "telemetry-collector"
```

### Step 4: Configure DocumentDB Indexes

**Download DocumentDB CA bundle:**
```bash
wget https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem
```

**Get DocumentDB password:**
```bash
aws secretsmanager get-secret-value \
  --secret-id telemetry-collector-docdb \
  --query SecretString \
  --output text | jq -r '.password'
```

**Connect to DocumentDB:**
```bash
DOCDB_ENDPOINT=$(terraform output -raw documentdb_endpoint)

mongosh --host $DOCDB_ENDPOINT \
  --username telemetry_admin \
  --tls \
  --tlsCAFile global-bundle.pem
```

**Create indexes:**
```javascript
use telemetry;

// TTL indexes (auto-delete after 365 days)
db.startup_events.createIndex(
  { "received_at": 1 },
  { expireAfterSeconds: 31536000 }
);

db.heartbeat_events.createIndex(
  { "received_at": 1 },
  { expireAfterSeconds: 31536000 }
);

// Query indexes
db.startup_events.createIndex({ "instance_id": 1 });
db.startup_events.createIndex({ "v": 1, "received_at": -1 });
db.heartbeat_events.createIndex({ "instance_id": 1 });

// Verify indexes
db.startup_events.getIndexes();
db.heartbeat_events.getIndexes();
```

## Testing

### Manual Testing with curl

```bash
# Get collector URL
COLLECTOR_URL=$(terraform output -raw collector_url)

# Test startup event
curl -X POST $COLLECTOR_URL \
  -H "Content-Type: application/json" \
  -d '{
    "event": "startup",
    "schema_version": "1",
    "instance_id": "00000000-0000-0000-0000-000000000001",
    "v": "1.0.16",
    "py": "3.12",
    "os": "linux",
    "arch": "x86_64",
    "mode": "with-gateway",
    "registry_mode": "full",
    "storage": "file",
    "auth": "keycloak",
    "federation": false,
    "ts": "2026-03-18T10:00:00Z"
  }'

# Expected: HTTP 204 (no response body)
```

### Integration Testing with Registry

```bash
# In registry repository
export MCP_TELEMETRY_ENDPOINT=$COLLECTOR_URL
uv run python -m registry

# Check CloudWatch Logs
aws logs tail /aws/lambda/telemetry-collector --follow

# Verify DocumentDB storage
mongosh --host $DOCDB_ENDPOINT \
  --username telemetry_admin \
  --tls \
  --tlsCAFile global-bundle.pem

use telemetry;
db.startup_events.find().pretty();
```

### Unit Tests

```bash
# In repository root
uv run pytest tests/unit/lambda/test_collector.py -v
```

## Monitoring

### CloudWatch Logs

```bash
# Lambda function logs
aws logs tail /aws/lambda/telemetry-collector --follow

# API Gateway logs
aws logs tail /aws/apigateway/telemetry-collector --follow
```

### CloudWatch Metrics

- **Lambda Invocations**: `AWS/Lambda > Invocations`
- **Lambda Errors**: `AWS/Lambda > Errors`
- **API Gateway Requests**: `AWS/ApiGateway > Count`
- **DynamoDB Operations**: `AWS/DynamoDB > ConsumedReadCapacityUnits`

### CloudWatch Alarms (Production Only)

Alarms are automatically created when `deployment_stage = "production"` and `alarm_email` is set:
- Lambda errors (> 10 in 5 minutes)
- Lambda throttles (> 5 in 5 minutes)
- Lambda duration (> 10 seconds average)
- API Gateway 5xx errors (> 10 in 5 minutes)

## Troubleshooting

### Issue: Lambda cannot connect to DocumentDB

**Symptoms:**
- CloudWatch logs show "Failed to connect to DocumentDB"
- Timeout errors

**Solution:**
1. Verify Lambda is in correct VPC and subnets:
   ```bash
   aws lambda get-function-configuration --function-name telemetry-collector | jq '.VpcConfig'
   ```

2. Verify security groups allow traffic:
   ```bash
   # DocumentDB security group should allow port 27017 from Lambda SG
   aws ec2 describe-security-groups --filters Name=group-name,Values=telemetry-collector-*
   ```

3. Verify DocumentDB is running:
   ```bash
   aws docdb describe-db-clusters --db-cluster-identifier telemetry-collector
   ```

### Issue: Rate limiting not working

**Symptoms:**
- More than 10 requests per minute from same IP are accepted

**Solution:**
1. Check DynamoDB table exists:
   ```bash
   aws dynamodb describe-table --table-name telemetry-collector-rate-limit
   ```

2. Check TTL is enabled:
   ```bash
   aws dynamodb describe-time-to-live --table-name telemetry-collector-rate-limit
   ```

3. Check IAM permissions for Lambda to access DynamoDB

### Issue: Always returns 204 even for valid events

**This is expected behavior!** The collector always returns 204 for privacy (no information leakage).

To verify events are being stored:
1. Check CloudWatch logs for "Stored startup event"
2. Query DocumentDB directly to verify documents are inserted

### Issue: High costs

**DocumentDB is expensive!** If cost is a concern:

1. **Testing:** Use smallest instance (db.t3.medium)
2. **Production:** Consider:
   - MongoDB Atlas M0 (free) instead of DocumentDB
   - Reduce NAT Gateway count to 1 (less redundancy)
   - Reduce CloudWatch log retention (default: 30 days)

## Updating the Collector

### Update Lambda Function Code

```bash
# Make code changes in lambda/collector/
cd terraform/telemetry-collector

# Terraform will detect changes and redeploy
terraform apply
```

### Update Infrastructure

```bash
# Edit Terraform files
# Apply changes
terraform apply
```

## Production Deployment

### Custom Domain Setup

1. **Update variables:**
   ```hcl
   custom_domain = "telemetry.mcpgateway.io"
   route53_zone_id = "Z1234567890ABC"
   ```

2. **Deploy:**
   ```bash
   terraform apply
   ```

3. **Wait for certificate validation** (~5-10 minutes)

4. **Verify DNS:**
   ```bash
   dig telemetry.mcpgateway.io
   curl -X POST https://telemetry.mcpgateway.io/v1/collect -d '{}'
   ```

### Enable Alarms

```hcl
alarm_email = "alerts@example.com"
deployment_stage = "production"
```

**Note:** You'll receive an SNS subscription confirmation email. Click the link to activate alarms.

## Cleanup

```bash
# Destroy all resources (irreversible!)
terraform destroy

# Note: DocumentDB snapshots are retained for production deployments
```

## Security Considerations

1. **No IP Logging:** Source IPs are hashed (SHA-256) for rate limiting only
2. **VPC Isolation:** DocumentDB is not internet-accessible
3. **TLS Everywhere:** All connections use TLS encryption
4. **Secrets Manager:** Credentials are encrypted at rest
5. **IAM Least Privilege:** Lambda has minimal required permissions
6. **Always 204:** No error messages leak system information

## Support

- **GitHub Issues:** https://github.com/agentic-community/mcp-gateway-registry/issues
- **Client Code:** Issue #558 (client-side telemetry)
- **Server Code:** Issue #559 (this infrastructure)

## License

Same as parent repository (MCP Gateway Registry).
