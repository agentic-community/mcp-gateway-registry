#!/bin/bash

# Deployment script for telemetry collector infrastructure
# Usage: ./deploy.sh [testing|production]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default to testing if no argument provided
DEPLOYMENT_STAGE="${1:-testing}"

if [[ "$DEPLOYMENT_STAGE" != "testing" && "$DEPLOYMENT_STAGE" != "production" ]]; then
    echo -e "${RED}Error: Deployment stage must be 'testing' or 'production'${NC}"
    echo "Usage: ./deploy.sh [testing|production]"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Telemetry Collector Deployment Script${NC}"
echo -e "${BLUE}Stage: $DEPLOYMENT_STAGE${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"

    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        echo -e "${RED}Error: AWS CLI not found. Please install it first.${NC}"
        exit 1
    fi

    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        echo -e "${RED}Error: AWS credentials not configured. Run 'aws configure' first.${NC}"
        exit 1
    fi

    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    echo -e "${GREEN}✓ AWS CLI configured (Account: $AWS_ACCOUNT_ID)${NC}"

    # Check Terraform
    if ! command -v terraform &> /dev/null; then
        echo -e "${RED}Error: Terraform not found. Please install it first.${NC}"
        exit 1
    fi

    TERRAFORM_VERSION=$(terraform version -json | grep -o '"terraform_version":"[^"]*' | cut -d'"' -f4)
    echo -e "${GREEN}✓ Terraform installed (Version: $TERRAFORM_VERSION)${NC}"

    # Check if mongosh is available (for post-deployment index setup)
    if command -v mongosh &> /dev/null; then
        echo -e "${GREEN}✓ mongosh installed${NC}"
    else
        echo -e "${YELLOW}⚠ mongosh not found (needed for DocumentDB index setup)${NC}"
        echo -e "${YELLOW}  Install: brew install mongosh (macOS) or download from MongoDB${NC}"
    fi

    echo ""
}

# Function to configure terraform.tfvars
configure_variables() {
    echo -e "${YELLOW}Configuring deployment variables...${NC}"

    if [[ ! -f "$SCRIPT_DIR/terraform.tfvars" ]]; then
        echo -e "${BLUE}Creating terraform.tfvars from template...${NC}"
        cp "$SCRIPT_DIR/terraform.tfvars.example" "$SCRIPT_DIR/terraform.tfvars"

        # Update deployment_stage
        if [[ "$DEPLOYMENT_STAGE" == "testing" ]]; then
            sed -i.bak 's/deployment_stage = "testing"/deployment_stage = "testing"/' "$SCRIPT_DIR/terraform.tfvars"
            sed -i.bak 's/documentdb_instance_class = "db.t3.medium"/documentdb_instance_class = "db.t3.medium"/' "$SCRIPT_DIR/terraform.tfvars"
        else
            sed -i.bak 's/deployment_stage = "testing"/deployment_stage = "production"/' "$SCRIPT_DIR/terraform.tfvars"
            sed -i.bak 's/documentdb_instance_class = "db.t3.medium"/documentdb_instance_class = "db.r5.large"/' "$SCRIPT_DIR/terraform.tfvars"
        fi
        rm "$SCRIPT_DIR/terraform.tfvars.bak"

        echo -e "${GREEN}✓ Created terraform.tfvars${NC}"
        echo -e "${YELLOW}  Please review and edit if needed: $SCRIPT_DIR/terraform.tfvars${NC}"
        echo ""

        # Ask user if they want to continue
        read -p "Continue with deployment? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}Deployment cancelled. Edit terraform.tfvars and run again.${NC}"
            exit 0
        fi
    else
        echo -e "${GREEN}✓ Using existing terraform.tfvars${NC}"
    fi

    echo ""
}

# Function to deploy infrastructure
deploy_infrastructure() {
    echo -e "${YELLOW}Deploying infrastructure with Terraform...${NC}"

    cd "$SCRIPT_DIR"

    # Initialize Terraform
    echo -e "${BLUE}Running terraform init...${NC}"
    terraform init
    echo ""

    # Plan deployment
    echo -e "${BLUE}Running terraform plan...${NC}"
    terraform plan -out=tfplan
    echo ""

    # Estimate cost
    if [[ "$DEPLOYMENT_STAGE" == "testing" ]]; then
        ESTIMATED_COST="~\$85-90/month"
    else
        ESTIMATED_COST="~\$195-200/month"
    fi

    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Estimated monthly cost: $ESTIMATED_COST${NC}"
    echo -e "${YELLOW}Resources to create:${NC}"
    echo -e "${YELLOW}  - VPC with NAT Gateways (2 AZs)${NC}"
    echo -e "${YELLOW}  - DocumentDB cluster (1 instance)${NC}"
    echo -e "${YELLOW}  - Lambda function${NC}"
    echo -e "${YELLOW}  - API Gateway HTTP API${NC}"
    echo -e "${YELLOW}  - DynamoDB table${NC}"
    echo -e "${YELLOW}  - Secrets Manager secret${NC}"
    echo -e "${YELLOW}  - CloudWatch log groups${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""

    read -p "Apply Terraform plan? This will create AWS resources. (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Deployment cancelled.${NC}"
        exit 0
    fi

    # Apply deployment
    echo -e "${BLUE}Running terraform apply (this takes ~15-20 minutes)...${NC}"
    terraform apply tfplan

    echo -e "${GREEN}✓ Infrastructure deployed successfully!${NC}"
    echo ""
}

# Function to save outputs
save_outputs() {
    echo -e "${YELLOW}Saving deployment outputs...${NC}"

    cd "$SCRIPT_DIR"

    COLLECTOR_URL=$(terraform output -raw collector_url)
    DOCDB_ENDPOINT=$(terraform output -raw documentdb_endpoint)
    SECRET_ARN=$(terraform output -raw documentdb_secret_arn)

    # Save to file
    cat > "$SCRIPT_DIR/deployment-info.txt" <<EOF
Telemetry Collector Deployment Information
==========================================
Deployment Stage: $DEPLOYMENT_STAGE
Deployed At: $(date)

Endpoints:
----------
Collector URL: $COLLECTOR_URL
DocumentDB Endpoint: $DOCDB_ENDPOINT

Secrets:
--------
DocumentDB Secret ARN: $SECRET_ARN

Next Steps:
-----------
1. Configure DocumentDB indexes (see below)
2. Test with curl (see below)
3. Integrate with registry

DocumentDB Index Setup:
-----------------------
# Get credentials
aws secretsmanager get-secret-value --secret-id telemetry-collector-docdb --query SecretString --output text | jq -r '.password'

# Download CA bundle
wget https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem

# Connect to DocumentDB
mongosh --host $DOCDB_ENDPOINT --username telemetry_admin --tls --tlsCAFile global-bundle.pem

# Create indexes (paste in mongosh)
use telemetry;
db.startup_events.createIndex({"received_at": 1}, {expireAfterSeconds: 31536000});
db.heartbeat_events.createIndex({"received_at": 1}, {expireAfterSeconds: 31536000});
db.startup_events.createIndex({"instance_id": 1});
db.startup_events.createIndex({"v": 1, "received_at": -1});
db.heartbeat_events.createIndex({"instance_id": 1});

Test with curl:
---------------
curl -X POST $COLLECTOR_URL \\
  -H "Content-Type: application/json" \\
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
    "ts": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  }'

# Expected: HTTP 204 (no response body)

Integrate with Registry:
------------------------
export MCP_TELEMETRY_ENDPOINT=$COLLECTOR_URL
cd ../../
uv run python -m registry

Monitor Logs:
-------------
aws logs tail /aws/lambda/telemetry-collector --follow

EOF

    echo -e "${GREEN}✓ Deployment info saved to: $SCRIPT_DIR/deployment-info.txt${NC}"
    echo ""

    # Display key information
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Deployment Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${BLUE}Collector URL:${NC}"
    echo -e "  $COLLECTOR_URL"
    echo ""
    echo -e "${BLUE}DocumentDB Endpoint:${NC}"
    echo -e "  $DOCDB_ENDPOINT"
    echo ""
}

# Function to setup DocumentDB indexes
setup_documentdb_indexes() {
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}DocumentDB Index Setup${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""

    if ! command -v mongosh &> /dev/null; then
        echo -e "${YELLOW}⚠ mongosh not installed. Skipping automatic index setup.${NC}"
        echo -e "${YELLOW}  Please install mongosh and run index setup manually.${NC}"
        echo -e "${YELLOW}  Instructions saved in: $SCRIPT_DIR/deployment-info.txt${NC}"
        return
    fi

    cd "$SCRIPT_DIR"
    DOCDB_ENDPOINT=$(terraform output -raw documentdb_endpoint)

    echo -e "${BLUE}Retrieving DocumentDB password from Secrets Manager...${NC}"
    DOCDB_PASSWORD=$(aws secretsmanager get-secret-value \
        --secret-id telemetry-collector-docdb \
        --query SecretString \
        --output text | jq -r '.password')

    if [[ -z "$DOCDB_PASSWORD" ]]; then
        echo -e "${RED}Error: Failed to retrieve DocumentDB password${NC}"
        echo -e "${YELLOW}  Run index setup manually using instructions in deployment-info.txt${NC}"
        return
    fi

    echo -e "${BLUE}Downloading DocumentDB CA bundle...${NC}"
    if [[ ! -f "$SCRIPT_DIR/global-bundle.pem" ]]; then
        wget -q https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem -O "$SCRIPT_DIR/global-bundle.pem"
    fi

    echo -e "${BLUE}Creating DocumentDB indexes...${NC}"

    # Create index commands
    cat > "$SCRIPT_DIR/create-indexes.js" <<'EOF'
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
print("Startup Events Indexes:");
printjson(db.startup_events.getIndexes());
print("\nHeartbeat Events Indexes:");
printjson(db.heartbeat_events.getIndexes());
EOF

    # Run index creation
    mongosh "mongodb://telemetry_admin:$DOCDB_PASSWORD@$DOCDB_ENDPOINT/telemetry?authSource=admin&tls=true&tlsCAFile=$SCRIPT_DIR/global-bundle.pem&retryWrites=false" \
        --file "$SCRIPT_DIR/create-indexes.js"

    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✓ DocumentDB indexes created successfully!${NC}"
    else
        echo -e "${YELLOW}⚠ Index creation failed. Run manually using instructions in deployment-info.txt${NC}"
    fi

    # Cleanup
    rm -f "$SCRIPT_DIR/create-indexes.js"

    echo ""
}

# Function to test deployment
test_deployment() {
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Testing Deployment${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""

    cd "$SCRIPT_DIR"
    COLLECTOR_URL=$(terraform output -raw collector_url)

    echo -e "${BLUE}Sending test startup event...${NC}"

    RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$COLLECTOR_URL" \
        -H "Content-Type: application/json" \
        -d "{
            \"event\": \"startup\",
            \"schema_version\": \"1\",
            \"instance_id\": \"00000000-0000-0000-0000-000000000001\",
            \"v\": \"1.0.16\",
            \"py\": \"3.12\",
            \"os\": \"linux\",
            \"arch\": \"x86_64\",
            \"mode\": \"with-gateway\",
            \"registry_mode\": \"full\",
            \"storage\": \"file\",
            \"auth\": \"keycloak\",
            \"federation\": false,
            \"ts\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
        }")

    HTTP_CODE=$(echo "$RESPONSE" | tail -n 1)

    if [[ "$HTTP_CODE" == "204" ]]; then
        echo -e "${GREEN}✓ Test successful! Received HTTP 204${NC}"
        echo ""
        echo -e "${BLUE}Checking CloudWatch logs...${NC}"
        sleep 3  # Wait for logs to appear
        aws logs tail /aws/lambda/telemetry-collector --since 1m | grep "Stored startup event" || true
    else
        echo -e "${RED}✗ Test failed. Expected HTTP 204, got: $HTTP_CODE${NC}"
        echo -e "${YELLOW}  Check Lambda logs: aws logs tail /aws/lambda/telemetry-collector --follow${NC}"
    fi

    echo ""
}

# Main execution
main() {
    check_prerequisites
    configure_variables
    deploy_infrastructure
    save_outputs
    setup_documentdb_indexes
    test_deployment

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Deployment Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo -e "  1. Review deployment info: ${YELLOW}$SCRIPT_DIR/deployment-info.txt${NC}"
    echo -e "  2. Monitor logs: ${YELLOW}aws logs tail /aws/lambda/telemetry-collector --follow${NC}"
    echo -e "  3. Integrate with registry: ${YELLOW}export MCP_TELEMETRY_ENDPOINT=$COLLECTOR_URL${NC}"
    echo ""
    echo -e "${BLUE}To destroy infrastructure later:${NC}"
    echo -e "  ${YELLOW}cd $SCRIPT_DIR && terraform destroy${NC}"
    echo ""
}

# Run main function
main
