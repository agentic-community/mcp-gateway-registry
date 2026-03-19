# Telemetry Collector Quick Start

One-command deployment and testing for the telemetry collector.

## Prerequisites

- AWS CLI configured with credentials
- Terraform >= 1.0 installed
- mongosh installed (optional, for index setup)

## Deploy to AWS (Testing)

```bash
cd terraform/telemetry-collector
./deploy.sh testing
```

**What it does:**
1. ✅ Checks prerequisites (AWS CLI, Terraform)
2. ✅ Creates `terraform.tfvars` from template
3. ✅ Deploys all infrastructure (~15-20 minutes)
4. ✅ Configures DocumentDB indexes automatically
5. ✅ Tests with curl
6. ✅ Saves deployment info to `deployment-info.txt`

**Cost:** ~$85-90/month for testing

## Deploy to Production

```bash
cd terraform/telemetry-collector
./deploy.sh production
```

**Cost:** ~$195-200/month for production

## Use with Registry

After deployment, the script will display the collector URL. Use it with the registry:

```bash
# Export the collector URL
export MCP_TELEMETRY_ENDPOINT=https://[your-id].execute-api.us-east-1.amazonaws.com/v1/collect

# Run registry
cd ../..
uv run python -m registry
```

## Monitor Logs

```bash
# Follow Lambda logs
aws logs tail /aws/lambda/telemetry-collector --follow

# Check recent events
aws logs tail /aws/lambda/telemetry-collector --since 5m
```

## Query Telemetry Data

```bash
# Get DocumentDB endpoint
cd terraform/telemetry-collector
DOCDB_ENDPOINT=$(terraform output -raw documentdb_endpoint)

# Get password
aws secretsmanager get-secret-value \
  --secret-id telemetry-collector-docdb \
  --query SecretString --output text | jq -r '.password'

# Connect
mongosh --host $DOCDB_ENDPOINT \
  --username telemetry_admin \
  --tls \
  --tlsCAFile global-bundle.pem

# Query in mongosh
use telemetry;
db.startup_events.find().count();
db.startup_events.find({"v": "1.0.16"});
db.heartbeat_events.find({"search_backend": "documentdb"});
```

## Destroy Infrastructure

```bash
cd terraform/telemetry-collector
./destroy.sh
```

**Warning:** This deletes ALL telemetry data. Cannot be undone!

## Troubleshooting

### Issue: Script fails at prerequisites check

**Solution:**
- Install AWS CLI: `brew install awscli` (macOS)
- Configure AWS: `aws configure`
- Install Terraform: `brew install terraform` (macOS)

### Issue: Terraform apply fails

**Solution:**
1. Check AWS credentials: `aws sts get-caller-identity`
2. Check Terraform logs in output
3. Retry: `terraform apply`

### Issue: DocumentDB indexes not created

**Solution:**
1. Install mongosh: `brew install mongosh`
2. Run script again or create indexes manually (see deployment-info.txt)

### Issue: Test returns non-204 status

**Solution:**
1. Check Lambda logs: `aws logs tail /aws/lambda/telemetry-collector --follow`
2. Check API Gateway logs: `aws logs tail /aws/apigateway/telemetry-collector`
3. Verify VPC/security groups allow Lambda → DocumentDB

## Files Created by Script

- `deployment-info.txt` - Collector URL, endpoints, test commands
- `global-bundle.pem` - DocumentDB CA certificate
- `terraform.tfvars` - Deployment configuration
- `terraform.tfstate` - Terraform state (DO NOT DELETE!)

## Manual Deployment (Without Script)

If you prefer manual control:

```bash
cd terraform/telemetry-collector

# 1. Configure
cp terraform.tfvars.example terraform.tfvars
vi terraform.tfvars

# 2. Deploy
terraform init
terraform plan
terraform apply

# 3. Setup indexes (see README.md)

# 4. Test (see README.md)
```

## Cost Management

**To minimize costs:**
- Use `db.t3.medium` for testing (vs `db.r5.large`)
- Deploy to single region only
- Destroy when not actively using: `./destroy.sh`

**Cost breakdown:**
- DocumentDB: ~$50-160/month (largest cost)
- NAT Gateway: ~$32/month
- Everything else: ~$3/month

## Support

- Full documentation: `README.md`
- GitHub Issues: https://github.com/agentic-community/mcp-gateway-registry/issues
- Client telemetry: Issue #558
- Server collector: Issue #559
