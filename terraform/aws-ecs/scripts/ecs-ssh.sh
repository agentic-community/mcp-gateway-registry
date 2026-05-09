#!/bin/bash

# ECS SSH Script - Dynamically finds and connects to ECS task
# Usage: ./ecs-ssh.sh [service-type] [cluster-name] [region]
#
# Supported service types:
#   registry      - MCP Gateway Registry
#   auth-server   - Auth Server
#   mcpgw         - MCP Gateway Server (ai-registry-tools)
#   keycloak      - Keycloak (if available)
#
# Examples:
#   ./ecs-ssh.sh registry
#   ./ecs-ssh.sh auth-server
#   ./ecs-ssh.sh mcpgw
#   ./ecs-ssh.sh auth-server mcp-gateway-ecs-cluster us-west-2

set -e

# Service type mapping: service_type -> service_name:container_name
declare -A SERVICE_MAP=(
  [registry]="mcp-gateway-v2-registry:registry"
  [auth-server]="mcp-gateway-v2-auth:auth-server"
  [mcpgw]="mcp-gateway-v2-mcpgw:mcpgw-server"
  [keycloak]="keycloak:keycloak"
)

_print_usage() {
  cat <<EOF
Usage: $(basename "$0") [service-type] [cluster-name] [region]

Supported service types: ${!SERVICE_MAP[@]}

Defaults:
  cluster-name  mcp-gateway-ecs-cluster
  region        us-east-1

Examples:
  $(basename "$0") registry
  $(basename "$0") mcpgw
  $(basename "$0") auth-server mcp-gateway-ecs-cluster us-west-2
EOF
}

# Handle help flags before treating the first arg as a service type.
case "${1:-}" in
  -h|--help|help)
    _print_usage
    exit 0
    ;;
esac

# Parameters
SERVICE_TYPE="${1:-registry}"
CLUSTER="${2:-mcp-gateway-ecs-cluster}"
REGION="${3:-us-east-1}"

# Get service name and container name from map
if [[ -z "${SERVICE_MAP[$SERVICE_TYPE]}" ]]; then
  echo "Error: Unknown service type '$SERVICE_TYPE'"
  echo "Supported types: ${!SERVICE_MAP[@]}"
  echo ""
  _print_usage
  exit 1
fi

IFS=':' read -r SERVICE CONTAINER <<< "${SERVICE_MAP[$SERVICE_TYPE]}"

echo "Connecting to ECS task..."
echo "  Service Type: $SERVICE_TYPE"
echo "  Cluster: $CLUSTER"
echo "  Service: $SERVICE"
echo "  Container: $CONTAINER"
echo "  Region: $REGION"
echo ""

# Get the first running task ARN
TASK_ARN=$(aws ecs list-tasks \
  --cluster "$CLUSTER" \
  --service-name "$SERVICE" \
  --region "$REGION" \
  --query 'taskArns[0]' \
  --output text)

if [ -z "$TASK_ARN" ] || [ "$TASK_ARN" = "None" ]; then
  echo "Error: No running tasks found for service '$SERVICE' in cluster '$CLUSTER'"
  exit 1
fi

echo "Task ARN: $TASK_ARN"
echo ""

# Connect to the task
aws ecs execute-command \
  --cluster "$CLUSTER" \
  --task "$TASK_ARN" \
  --container "$CONTAINER" \
  --interactive \
  --command "/bin/bash" \
  --region "$REGION"
