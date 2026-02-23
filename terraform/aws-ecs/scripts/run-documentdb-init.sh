#!/bin/bash
# Run DocumentDB initialization via ECS task
#
# This script runs the init-documentdb-indexes.py script inside an ECS task
# with proper network access to the DocumentDB cluster in the VPC.
#
# Usage:
#   ./terraform/aws-ecs/scripts/run-documentdb-init.sh
#   ./terraform/aws-ecs/scripts/run-documentdb-init.sh --entra-group-id "your-guid"

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TERRAFORM_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENTRA_GROUP_ID=""

# Show help function
show_help() {
    cat << EOF
DocumentDB Initialization Script

Usage: $0 [options]

This script runs the DocumentDB index initialization inside an ECS task
with proper network access to the DocumentDB cluster.

Options:
  -h, --help                     Show this help message
  --entra-group-id <GUID>        Entra ID Group Object ID for admin group
                                 (required when entra_enabled=true in terraform.tfvars)

Environment Variables:
  DOCUMENTDB_HOST                Override DocumentDB endpoint (optional)
  AWS_REGION                     AWS region (default: us-west-2)
  ENTRA_ADMIN_GROUP_ID           Entra ID Group Object ID (alternative to --entra-group-id)

The script automatically reads the DocumentDB endpoint from SSM Parameter Store
if available, otherwise falls back to DOCUMENTDB_HOST environment variable.

For Microsoft Entra ID deployments:
  1. Create a "registry-admins" group in Azure Portal
  2. Get the Group Object ID: Azure Portal -> Groups -> [group name] -> Object Id
  3. Pass it via --entra-group-id or ENTRA_ADMIN_GROUP_ID env var
EOF
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        --entra-group-id)
            ENTRA_GROUP_ID="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            ;;
    esac
done

# Check for env var if not provided via CLI
if [ -z "$ENTRA_GROUP_ID" ] && [ -n "$ENTRA_ADMIN_GROUP_ID" ]; then
    ENTRA_GROUP_ID="$ENTRA_ADMIN_GROUP_ID"
fi

# Get AWS account and region
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION="${AWS_REGION:-us-west-2}"

# Check if Entra ID is enabled in terraform.tfvars
TFVARS_FILE="$TERRAFORM_DIR/terraform.tfvars"
ENTRA_ENABLED="false"
if [ -f "$TFVARS_FILE" ]; then
    # Extract entra_enabled value from terraform.tfvars
    ENTRA_ENABLED=$(grep -E "^entra_enabled\s*=" "$TFVARS_FILE" | sed 's/.*=\s*//' | tr -d ' "' || echo "false")
fi

# If Entra is enabled, require the group ID
if [ "$ENTRA_ENABLED" = "true" ]; then
    if [ -z "$ENTRA_GROUP_ID" ]; then
        echo -e "${RED}Error: Microsoft Entra ID is enabled (entra_enabled=true in terraform.tfvars)${NC}"
        echo ""
        echo "You must provide the Entra ID Group Object ID for the admin group:"
        echo ""
        echo "  $0 --entra-group-id \"your-group-object-id\""
        echo ""
        echo "To get the Group Object ID:"
        echo "  1. Go to Azure Portal -> Microsoft Entra ID -> Groups"
        echo "  2. Find or create your 'registry-admins' group"
        echo "  3. Copy the 'Object Id' value"
        echo ""
        exit 1
    fi
    echo -e "${GREEN}Entra ID enabled - using Group Object ID: $ENTRA_GROUP_ID${NC}"
else
    echo -e "${BLUE}Keycloak mode - no Entra Group ID required${NC}"
fi

# ECS configuration
CLUSTER_NAME="mcp-gateway-ecs-cluster"
TASK_FAMILY="mcp-gateway-v2-registry"
CONTAINER_NAME="registry"

# Terraform outputs file location
OUTPUTS_FILE="$SCRIPT_DIR/terraform-outputs.json"

# Get DocumentDB host - check sources in order of priority:
# 1. Environment variable (explicit override)
# 2. Terraform outputs file
# 3. SSM Parameter Store
if [ -z "$DOCUMENTDB_HOST" ]; then
    # Try terraform outputs first
    if [ -f "$OUTPUTS_FILE" ]; then
        echo -e "${YELLOW}Checking terraform outputs for DocumentDB endpoint...${NC}"
        DOCUMENTDB_HOST=$(jq -r '.documentdb_cluster_endpoint.value // empty' "$OUTPUTS_FILE" 2>/dev/null || echo "")
        if [ -n "$DOCUMENTDB_HOST" ] && [ "$DOCUMENTDB_HOST" != "null" ]; then
            echo -e "${GREEN}Found DocumentDB endpoint in terraform outputs${NC}"
        else
            DOCUMENTDB_HOST=""
        fi
    fi

    # Fall back to SSM Parameter Store
    if [ -z "$DOCUMENTDB_HOST" ]; then
        echo -e "${YELLOW}Fetching DocumentDB endpoint from SSM Parameter Store...${NC}"
        DOCUMENTDB_HOST=$(aws ssm get-parameter \
            --name "/mcp-gateway/documentdb/endpoint" \
            --query 'Parameter.Value' \
            --output text \
            --region "$AWS_REGION" 2>/dev/null || echo "")

        if [ -n "$DOCUMENTDB_HOST" ] && [ "$DOCUMENTDB_HOST" != "None" ]; then
            echo -e "${GREEN}Found DocumentDB endpoint in SSM${NC}"
        else
            DOCUMENTDB_HOST=""
        fi
    fi
fi

# Validate DocumentDB host
if [ -z "$DOCUMENTDB_HOST" ]; then
    echo -e "${RED}Error: DocumentDB endpoint not found${NC}"
    echo ""
    echo "Checked the following sources:"
    echo "  1. DOCUMENTDB_HOST environment variable"
    echo "  2. Terraform outputs file: $OUTPUTS_FILE"
    echo "  3. SSM Parameter Store: /mcp-gateway/documentdb/endpoint"
    echo ""
    echo "Make sure you have run 'terraform apply' and saved outputs,"
    echo "or set DOCUMENTDB_HOST environment variable."
    exit 1
fi

# Get credentials from Secrets Manager
echo -e "${YELLOW}Fetching DocumentDB credentials from Secrets Manager...${NC}"
SECRET_ARN=$(aws secretsmanager list-secrets \
    --filters Key=name,Values=mcp-gateway/documentdb/credentials \
    --query 'SecretList[0].ARN' \
    --output text \
    --region "$AWS_REGION" 2>/dev/null || echo "")

DOCUMENTDB_USERNAME=""
DOCUMENTDB_PASSWORD=""

if [ -n "$SECRET_ARN" ] && [ "$SECRET_ARN" != "None" ]; then
    SECRET_JSON=$(aws secretsmanager get-secret-value \
        --secret-id "$SECRET_ARN" \
        --query 'SecretString' \
        --output text \
        --region "$AWS_REGION" 2>/dev/null || echo "")

    if [ -n "$SECRET_JSON" ]; then
        DOCUMENTDB_USERNAME=$(echo "$SECRET_JSON" | jq -r '.username // ""')
        DOCUMENTDB_PASSWORD=$(echo "$SECRET_JSON" | jq -r '.password // ""')
        echo -e "${GREEN}Found credentials in Secrets Manager${NC}"
    fi
fi

# Get VPC configuration from registry service
echo -e "${YELLOW}Getting VPC configuration from registry service...${NC}"
VPC_CONFIG=$(aws ecs describe-services \
    --cluster "$CLUSTER_NAME" \
    --services mcp-gateway-v2-registry \
    --region "$AWS_REGION" \
    --query 'services[0].networkConfiguration.awsvpcConfiguration' \
    --output json)

SUBNETS=$(echo "$VPC_CONFIG" | jq -r '.subnets | join(",")')
SECURITY_GROUPS=$(echo "$VPC_CONFIG" | jq -r '.securityGroups | join(",")')

echo -e "${BLUE}Configuration:${NC}"
echo "  Cluster: $CLUSTER_NAME"
echo "  Task: $TASK_FAMILY"
echo "  DocumentDB Host: $DOCUMENTDB_HOST"
echo "  DocumentDB Username: ${DOCUMENTDB_USERNAME:-<not set>}"
echo "  Entra Enabled: $ENTRA_ENABLED"
echo "  Entra Group ID: ${ENTRA_GROUP_ID:-<not set>}"
echo ""

# Create simple command to run Python initialization
# NOTE: init-documentdb-indexes.py creates indexes and loads the admin scope from registry-admins.json.
# load-scopes.py then loads all scope definitions (UI-Scopes, group_mappings, MCP server scopes)
# from scopes.yml into DocumentDB, which the auth server uses at runtime to resolve permissions.
echo -e "${YELLOW}Preparing initialization command...${NC}"

# Build the init command, adding --entra-group-id if provided
# After creating indexes, load all scope definitions from scopes.yml into DocumentDB.
# This ensures group_mappings (including mcp-servers-unrestricted for M2M service accounts)
# and UI-Scopes are available for the auth server to resolve permissions.
if [ -n "$ENTRA_GROUP_ID" ]; then
    INIT_COMMAND="source /app/.venv/bin/activate && cd /app/scripts && python init-documentdb-indexes.py --entra-group-id '$ENTRA_GROUP_ID' && python load-scopes.py --scopes-file /app/config/scopes.yml --clear-existing"
else
    INIT_COMMAND="source /app/.venv/bin/activate && cd /app/scripts && python init-documentdb-indexes.py && python load-scopes.py --scopes-file /app/config/scopes.yml --clear-existing"
fi

# Check if task definition exists
TASK_DEF_ARN=$(aws ecs describe-task-definition \
    --task-definition "$TASK_FAMILY" \
    --region "$AWS_REGION" \
    --query 'taskDefinition.taskDefinitionArn' \
    --output text 2>/dev/null || echo "")

if [ -z "$TASK_DEF_ARN" ] || [ "$TASK_DEF_ARN" = "None" ]; then
    echo -e "${RED}Error: Task definition '$TASK_FAMILY' not found${NC}"
    echo ""
    echo "You need to create the task definition first."
    echo "Run: cd terraform/aws-ecs && terraform apply"
    exit 1
fi

echo -e "${GREEN}Task definition found: $TASK_DEF_ARN${NC}"
echo ""

# Run the ECS task
echo -e "${YELLOW}Starting ECS task...${NC}"
TASK_ARN=$(aws ecs run-task \
    --cluster "$CLUSTER_NAME" \
    --task-definition "$TASK_FAMILY" \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SECURITY_GROUPS],assignPublicIp=DISABLED}" \
    --overrides "$(jq -n \
        --arg container "$CONTAINER_NAME" \
        --arg cmd "$INIT_COMMAND" \
        --arg host "$DOCUMENTDB_HOST" \
        --arg user "$DOCUMENTDB_USERNAME" \
        --arg pass "$DOCUMENTDB_PASSWORD" \
        '{
            "containerOverrides": [{
                "name": $container,
                "command": ["/bin/bash", "-c", $cmd],
                "environment": [
                    {"name": "RUN_INIT_SCRIPTS", "value": "true"},
                    {"name": "DOCUMENTDB_HOST", "value": $host},
                    {"name": "DOCUMENTDB_PORT", "value": "27017"},
                    {"name": "DOCUMENTDB_USERNAME", "value": $user},
                    {"name": "DOCUMENTDB_PASSWORD", "value": $pass},
                    {"name": "DOCUMENTDB_DATABASE", "value": "mcp_registry"},
                    {"name": "DOCUMENTDB_NAMESPACE", "value": "default"},
                    {"name": "DOCUMENTDB_USE_TLS", "value": "true"},
                    {"name": "DOCUMENTDB_USE_IAM", "value": "false"},
                    {"name": "DOCUMENTDB_TLS_CA_FILE", "value": "/app/global-bundle.pem"}
                ]
            }]
        }')" \
    --region "$AWS_REGION" \
    --query 'tasks[0].taskArn' \
    --output text)

if [ -z "$TASK_ARN" ] || [ "$TASK_ARN" = "None" ]; then
    echo -e "${RED}Failed to start ECS task${NC}"
    exit 1
fi

TASK_ID=$(basename "$TASK_ARN")
echo -e "${GREEN}Task started: $TASK_ID${NC}"
echo ""

# Wait for task to complete
echo -e "${YELLOW}Waiting for task to complete (this may take 2-3 minutes)...${NC}"
for i in {1..90}; do
    sleep 2

    STATUS=$(aws ecs describe-tasks \
        --cluster "$CLUSTER_NAME" \
        --tasks "$TASK_ARN" \
        --region "$AWS_REGION" \
        --query 'tasks[0].lastStatus' \
        --output text)

    if [ "$STATUS" = "STOPPED" ]; then
        echo -e "${GREEN}Task completed${NC}"
        break
    fi

    echo "  [$i] Status: $STATUS"
done

# Get exit code
EXIT_CODE=$(aws ecs describe-tasks \
    --cluster "$CLUSTER_NAME" \
    --tasks "$TASK_ARN" \
    --region "$AWS_REGION" \
    --query 'tasks[0].containers[0].exitCode' \
    --output text)

echo ""
echo -e "${BLUE}Task exit code: $EXIT_CODE${NC}"

# Get logs (wait a bit for logs to be available)
echo ""
echo -e "${YELLOW}Retrieving task logs...${NC}"
sleep 3

# Get the actual log stream name
LOG_STREAM_NAME="ecs/registry/$TASK_ID"

echo ""
printf '=%.0s' {1..100}
echo ""

# Try to get logs
LOGS=$(aws logs get-log-events \
    --log-group-name "/ecs/mcp-gateway-v2-registry" \
    --log-stream-name "$LOG_STREAM_NAME" \
    --region "$AWS_REGION" \
    --query 'events[*].message' \
    --output json 2>/dev/null)

if [ $? -eq 0 ] && [ -n "$LOGS" ] && [ "$LOGS" != "[]" ]; then
    # Parse JSON array and print each message on a new line
    echo "$LOGS" | jq -r '.[]' 2>/dev/null || echo "$LOGS"
else
    echo "No logs found in stream: $LOG_STREAM_NAME"
    echo ""
    echo "Available log streams:"
    aws logs describe-log-streams \
        --log-group-name "/ecs/mcp-gateway-v2-registry" \
        --order-by LastEventTime \
        --descending \
        --max-items 5 \
        --region "$AWS_REGION" \
        --query 'logStreams[*].logStreamName' \
        --output text 2>/dev/null || echo "Could not retrieve log streams"
fi

echo ""
printf '=%.0s' {1..100}
echo ""

# Exit with same code as task
if [ "$EXIT_CODE" = "0" ]; then
    echo -e "${GREEN}SUCCESS: DocumentDB initialization completed${NC}"
else
    echo -e "${RED}ERROR: DocumentDB initialization failed${NC}"
fi

exit "${EXIT_CODE:-1}"
