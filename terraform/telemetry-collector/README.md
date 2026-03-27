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
- Always returns 204 (no information leakage)
- Hash-based rate limiting (no IP storage)
- VPC-secured DocumentDB
- Fail-silent design (never blocks clients)
- TLS encryption everywhere

## Architecture

```
                          Registry Instances
                          (worldwide deployments)
                                  |
                                  | HTTPS POST /v1/collect
                                  | (startup + heartbeat events)
                                  v
                    +----------------------------+
                    |    API Gateway HTTP API     |
                    |  (throttle: 50 req/s burst) |
                    |  (CORS: restricted origins) |
                    +----------------------------+
                                  |
                                  | AWS_PROXY integration
                                  v
  +----------------------------------------------------------------+
  |                         AWS VPC (10.0.0.0/16)                  |
  |                                                                |
  |  +------------------+     +------------------+                 |
  |  | Public Subnet    |     | Public Subnet    |   (2 AZs)      |
  |  | (10.0.0.0/24)    |     | (10.0.1.0/24)    |                |
  |  |                  |     |                  |                 |
  |  | +- NAT Gateway --+     +-- NAT Gateway -+ |                |
  |  +--|---------------+     +----------------|--+                |
  |     |                                      |                   |
  |     v                                      v                   |
  |  +------------------+     +------------------+                 |
  |  | Private Subnet   |     | Private Subnet   |   (2 AZs)      |
  |  | (10.0.10.0/24)   |     | (10.0.11.0/24)   |                |
  |  |                  |     |                  |                 |
  |  |  +------------+  |     |  +------------+  |                 |
  |  |  |   Lambda   |  |     |  | DocumentDB |  |                |
  |  |  |  Function  |--+-----+->|  Cluster   |  |                |
  |  |  +------------+  |     |  | (TLS only) |  |                |
  |  |       |          |     |  +------------+  |                 |
  |  +-------|----------+     +------------------+                 |
  |          |                                                     |
  +----------|-----------------------------------------------------+
             |
             | (via NAT Gateway)
             v
  +---------------------+    +---------------------+
  |      DynamoDB       |    |   Secrets Manager    |
  |  (rate limiting)    |    |  (DocumentDB creds)  |
  |                     |    |                      |
  |  ip_hash -> counter |    |  username / password |
  |  TTL auto-cleanup   |    |  database name       |
  +---------------------+    +---------------------+

  Request Flow:
  ---------------------------------------------------------------
  1. Registry sends HTTPS POST to API Gateway
  2. API Gateway invokes Lambda (AWS_PROXY)
  3. Lambda hashes source IP (SHA-256, never stored)
  4. Lambda checks DynamoDB rate limit (10 req/min per IP)
  5. Lambda validates payload (Pydantic schema)
  6. Lambda fetches DocumentDB creds from Secrets Manager
  7. Lambda stores event in DocumentDB (TLS connection)
  8. Lambda returns 204 (always, regardless of outcome)

  Optional Bastion Host:
  ---------------------------------------------------------------
  When bastion_enabled = true, a t2.micro EC2 instance is
  created in a public subnet with SSH access for direct
  DocumentDB queries via mongosh.
```

## Prerequisites

- AWS CLI v2 configured with credentials
- Terraform >= 1.0
- Python 3.12+ and pip (for Lambda packaging)
- mongosh (optional, for DocumentDB index setup)

```bash
# Verify prerequisites
aws sts get-caller-identity
terraform version
python3 --version
```

## Quick Start (Automated)

The `deploy.sh` script handles everything end-to-end:

```bash
cd terraform/telemetry-collector
./deploy.sh testing    # ~$85-90/month
# or
./deploy.sh production # ~$195-200/month
```

**What it does:**
1. Checks prerequisites (AWS CLI, Terraform)
2. Creates `terraform.tfvars` from template
3. Builds Lambda deployment package
4. Deploys all infrastructure (~15-20 minutes)
5. Configures DocumentDB indexes automatically
6. Tests with curl
7. Saves deployment info to `deployment-info.txt`

After deployment, integrate with the registry:

```bash
export MCP_TELEMETRY_ENDPOINT=https://[your-id].execute-api.us-east-1.amazonaws.com/v1/collect
cd ../..
uv run python -m registry
```

## Manual Deployment (Step by Step)

### Step 1: Configure Variables

```bash
cd terraform/telemetry-collector

# Copy example configuration and edit
cp terraform.tfvars.example terraform.tfvars
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
cors_allowed_origins = ["https://mcpgateway.io", "https://app.mcpgateway.io"]
custom_domain = "telemetry.mcpgateway.io"
route53_zone_id = "Z1234567890ABC"
alarm_email = "alerts@example.com"
```

**Bastion host setup (optional, for direct DocumentDB access):**

To enable the bastion host, you need an SSH key pair and your public IP:

```bash
# Generate an SSH key pair (if you don't have one)
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "bastion-telemetry"

# Get your public key
cat ~/.ssh/id_ed25519.pub

# Get your public IP
curl -s ifconfig.me
```

Then set these in `terraform.tfvars`:
```hcl
bastion_enabled       = true
bastion_public_key    = "ssh-ed25519 AAAA... your-key-here"
bastion_allowed_cidrs = ["YOUR_PUBLIC_IP/32"]  # e.g. ["203.0.113.42/32"]
```

After deployment, set up the bastion helper scripts:
```bash
# Run the setup script (copies connect.sh, query.sh, and config to bastion)
./bastion-scripts/setup-bastion.sh

# Or do it manually with SCP:
BASTION_IP=$(terraform output -raw bastion_public_ip)
DOCDB_ENDPOINT=$(terraform output -raw documentdb_endpoint)
SECRET_ARN=$(terraform output -raw documentdb_secret_arn)

# Create config file with terraform output values
cat > /tmp/bastion.env <<EOF
DOCDB_ENDPOINT="$DOCDB_ENDPOINT"
SECRET_ARN="$SECRET_ARN"
AWS_REGION="$(terraform output -raw aws_region)"
EOF

# Copy scripts and config to bastion
scp -i ~/.ssh/id_ed25519 \
    /tmp/bastion.env \
    bastion-scripts/connect.sh \
    bastion-scripts/query.sh \
    ec2-user@$BASTION_IP:~/

# Make executable
ssh -i ~/.ssh/id_ed25519 ec2-user@$BASTION_IP 'chmod +x ~/connect.sh ~/query.sh'
```

Then SSH in and use the helper scripts:
```bash
ssh -i ~/.ssh/id_ed25519 ec2-user@$BASTION_IP

# On the bastion:
./connect.sh   # Interactive mongosh session to DocumentDB
./query.sh     # Quick summary of telemetry data
```

Common DocumentDB queries (run inside `./connect.sh`):
```javascript
// Switch to telemetry database
use telemetry;

// Count all events
db.startup_events.countDocuments();
db.heartbeat_events.countDocuments();

// View recent startup events
db.startup_events.find().sort({_id: -1}).limit(10).pretty();

// View recent heartbeat events
db.heartbeat_events.find().sort({_id: -1}).limit(10).pretty();

// Breakdown by registry version
db.startup_events.aggregate([
  {$group: {_id: "$v", count: {$sum: 1}}},
  {$sort: {count: -1}}
]);

// Breakdown by storage backend
db.startup_events.aggregate([
  {$group: {_id: "$storage", count: {$sum: 1}}},
  {$sort: {count: -1}}
]);

// Breakdown by OS
db.startup_events.aggregate([
  {$group: {_id: "$os", count: {$sum: 1}}},
  {$sort: {count: -1}}
]);

// Find events from a specific instance
db.startup_events.find({instance_id: "YOUR_INSTANCE_ID"}).pretty();

// Events received in the last 24 hours
db.startup_events.find({
  received_at: {$gte: new Date(Date.now() - 24*60*60*1000)}
}).pretty();

// Heartbeat stats: top instances by uptime
db.heartbeat_events.find({}, {
  instance_id: 1, uptime_hours: 1, servers_count: 1, agents_count: 1, _id: 0
}).sort({uptime_hours: -1}).limit(10);
```

### Step 2: Build Lambda Deployment Package

The Lambda function requires `pymongo` and `pydantic` bundled into a zip:

```bash
# Install dependencies into a temp directory
pip install -t /tmp/lambda-package pymongo pydantic boto3

# Copy Lambda source code
cp lambda/collector/index.py /tmp/lambda-package/
cp lambda/collector/schemas.py /tmp/lambda-package/

# Create the zip
cd /tmp/lambda-package
zip -r /path/to/terraform/telemetry-collector/lambda_function.zip .

# Return to terraform directory
cd /path/to/terraform/telemetry-collector
```

**Note:** `boto3` is already available in the Lambda runtime but included for local testing. The zip must be named `lambda_function.zip` (or match `lambda_package_path` in your tfvars).

### Step 3: Deploy Infrastructure

```bash
terraform init
terraform plan
terraform apply
```

Deployment takes ~15-20 minutes (mostly DocumentDB cluster creation).

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

## Testing vs Production

| Aspect | Testing | Production |
|--------|---------|------------|
| DocumentDB instance | db.t3.medium (~$50/mo) | db.r5.large (~$160/mo) |
| DocumentDB snapshot on destroy | Skipped | Final snapshot created |
| DynamoDB point-in-time recovery | Off | On |
| CloudWatch alarms | Not created | 4 alarms (errors, throttles, duration, 5xx) |
| Estimated total cost | ~$85-90/month | ~$195-200/month |

**Cost breakdown:**
- DocumentDB: ~$50-160/month (largest cost)
- NAT Gateway (2 AZs): ~$32/month
- Lambda, API Gateway, DynamoDB, Secrets Manager, CloudWatch: ~$3/month

## Testing

### Manual Testing with curl

```bash
COLLECTOR_URL=$(terraform output -raw collector_url)

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

# Recent events only
aws logs tail /aws/lambda/telemetry-collector --since 5m
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

### Query Telemetry Data

```bash
DOCDB_ENDPOINT=$(terraform output -raw documentdb_endpoint)

# Get password
aws secretsmanager get-secret-value \
  --secret-id telemetry-collector-docdb \
  --query SecretString --output text | jq -r '.password'

# Connect and query
mongosh --host $DOCDB_ENDPOINT \
  --username telemetry_admin \
  --tls \
  --tlsCAFile global-bundle.pem

use telemetry;
db.startup_events.find().count();
db.startup_events.find({"v": "1.0.16"});
db.heartbeat_events.find({"search_backend": "documentdb"});
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

## Updating the Collector

### Update Lambda Function Code

When you change files in `lambda/collector/`, you must rebuild the zip, run terraform
apply, AND force Lambda to pick up the new code. Terraform may not detect zip content
changes if the file path and size are similar.

```bash
cd terraform/telemetry-collector

# Step 1: Rebuild the zip package (see Step 2 in Deployment above)
cd lambda/collector && pip install -r requirements.txt -t . && cd ../..
zip -r lambda_function.zip lambda/collector/

# Step 2: Apply terraform (updates infrastructure and zip hash)
terraform apply -auto-approve

# Step 3: Force Lambda to use the new code
# Terraform may cache the old zip hash — this ensures the update takes effect
aws lambda update-function-code \
  --function-name telemetry-collector \
  --zip-file fileb://lambda_function.zip \
  --region $(terraform output -raw aws_region)

# Step 4: Verify the update
aws logs tail /aws/lambda/telemetry-collector --since 1m --region $(terraform output -raw aws_region)
```

**Why Step 3 is needed:** Terraform tracks the zip file by its `filebase64sha256` hash.
If the hash in the state file matches the new zip (e.g., due to caching), Terraform
skips the Lambda update. The AWS CLI command forces the code update regardless.

### Update Infrastructure

```bash
# Edit Terraform files (.tf)
terraform apply
```

## Troubleshooting

### Lambda cannot connect to DocumentDB

**Symptoms:** CloudWatch logs show "Failed to connect to DocumentDB" or timeout errors.

**Solution:**
1. Verify Lambda is in correct VPC and subnets:
   ```bash
   aws lambda get-function-configuration --function-name telemetry-collector | jq '.VpcConfig'
   ```
2. Verify security groups allow traffic:
   ```bash
   aws ec2 describe-security-groups --filters Name=group-name,Values=telemetry-collector-*
   ```
3. Verify DocumentDB is running:
   ```bash
   aws docdb describe-db-clusters --db-cluster-identifier telemetry-collector
   ```

### Rate limiting not working

**Symptoms:** More than 10 requests per minute from same IP are accepted.

**Solution:**
1. Check DynamoDB table exists:
   ```bash
   aws dynamodb describe-table --table-name telemetry-collector-rate-limit
   ```
2. Check TTL is enabled:
   ```bash
   aws dynamodb describe-time-to-live --table-name telemetry-collector-rate-limit
   ```

### Always returns 204 even for valid events

**This is expected behavior.** The collector always returns 204 for privacy (no information leakage).

To verify events are being stored:
1. Check CloudWatch logs for "Stored startup event"
2. Query DocumentDB directly to verify documents are inserted

### Script fails at prerequisites check

- Install AWS CLI: `brew install awscli` (macOS) or `sudo apt-get install awscli` (Linux)
- Configure AWS: `aws configure`
- Install Terraform: `brew install terraform` (macOS) or see https://developer.hashicorp.com/terraform/install

### High costs

DocumentDB is the largest cost item. To minimize:
- Use smallest instance (`db.t3.medium`) for testing
- Destroy when not actively using: `./destroy.sh`
- Consider MongoDB Atlas M0 (free) as an alternative for non-production use

## Files Reference

**Source files:**
- `lambda/collector/index.py` - Lambda handler code
- `lambda/collector/schemas.py` - Pydantic validation schemas
- `lambda/collector/requirements.txt` - Python dependencies

**Terraform files:**
- `*.tf` - Infrastructure definitions
- `variables.tf` - All configurable variables
- `terraform.tfvars.example` - Example configuration (copy to `terraform.tfvars`)

**Generated files (not committed):**
- `lambda_function.zip` - Lambda deployment package
- `terraform.tfvars` - Your deployment configuration
- `terraform.tfstate` - Terraform state (DO NOT DELETE)
- `deployment-info.txt` - Collector URL, endpoints, test commands
- `global-bundle.pem` - DocumentDB CA certificate

## Cleanup

```bash
cd terraform/telemetry-collector
./destroy.sh
```

**Warning:** This deletes ALL telemetry data. Cannot be undone. Production deployments retain a final DocumentDB snapshot.

## Security Considerations

1. **No IP Logging:** Source IPs are hashed (SHA-256) for rate limiting only
2. **VPC Isolation:** DocumentDB is not internet-accessible
3. **TLS Everywhere:** All connections use TLS encryption
4. **Secrets Manager:** Credentials are encrypted at rest
5. **IAM Least Privilege:** Lambda has minimal required permissions
6. **Always 204:** No error messages leak system information
7. **CORS Restricted:** Only configured origins can submit telemetry via browser

## Support

- **GitHub Issues:** https://github.com/agentic-community/mcp-gateway-registry/issues
- **Client Code:** Issue #558 (client-side telemetry)
- **Server Code:** Issue #559 (this infrastructure)

## License

Same as parent repository (MCP Gateway Registry).
