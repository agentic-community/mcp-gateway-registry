#!/bin/bash

# Cleanup script for telemetry collector infrastructure
# Usage: ./destroy.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${RED}========================================${NC}"
echo -e "${RED}Telemetry Collector Cleanup Script${NC}"
echo -e "${RED}========================================${NC}"
echo ""

# Warning banner
echo -e "${YELLOW}⚠⚠⚠  WARNING  ⚠⚠⚠${NC}"
echo -e "${YELLOW}This will DESTROY all telemetry collector infrastructure:${NC}"
echo -e "${YELLOW}  - VPC and NAT Gateways${NC}"
echo -e "${YELLOW}  - DocumentDB cluster (including all data)${NC}"
echo -e "${YELLOW}  - Lambda function${NC}"
echo -e "${YELLOW}  - API Gateway${NC}"
echo -e "${YELLOW}  - DynamoDB table${NC}"
echo -e "${YELLOW}  - Secrets Manager secrets${NC}"
echo -e "${YELLOW}  - CloudWatch logs${NC}"
echo ""
echo -e "${RED}THIS ACTION CANNOT BE UNDONE!${NC}"
echo ""

# Check if deployment exists
cd "$SCRIPT_DIR"

if [[ ! -f "terraform.tfstate" ]]; then
    echo -e "${YELLOW}No terraform state found. Nothing to destroy.${NC}"
    exit 0
fi

# Show current deployment info
if command -v terraform &> /dev/null; then
    echo -e "${BLUE}Current deployment:${NC}"
    terraform show -json | jq -r '.values.root_module.resources[] | select(.type == "aws_apigatewayv2_api") | .values.name' 2>/dev/null || echo "  telemetry-collector-api"
    echo ""
fi

# Confirmation 1
read -p "Type 'yes' to confirm destruction: " CONFIRM1
if [[ "$CONFIRM1" != "yes" ]]; then
    echo -e "${GREEN}Destruction cancelled.${NC}"
    exit 0
fi

# Confirmation 2 (double check)
echo ""
echo -e "${RED}Final confirmation: This will delete ALL telemetry data.${NC}"
read -p "Type 'DESTROY' in all caps to proceed: " CONFIRM2
if [[ "$CONFIRM2" != "DESTROY" ]]; then
    echo -e "${GREEN}Destruction cancelled.${NC}"
    exit 0
fi

echo ""
echo -e "${BLUE}Destroying infrastructure...${NC}"
echo ""

# Run terraform destroy
terraform destroy -auto-approve

if [[ $? -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Infrastructure Destroyed Successfully${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""

    # Cleanup local files
    echo -e "${BLUE}Cleaning up local files...${NC}"
    rm -f "$SCRIPT_DIR/deployment-info.txt"
    rm -f "$SCRIPT_DIR/global-bundle.pem"
    rm -f "$SCRIPT_DIR/tfplan"
    rm -f "$SCRIPT_DIR/.terraform.lock.hcl"
    rm -rf "$SCRIPT_DIR/.terraform"

    echo -e "${GREEN}✓ Cleanup complete${NC}"
else
    echo ""
    echo -e "${RED}Error during destruction. Check terraform state.${NC}"
    exit 1
fi
