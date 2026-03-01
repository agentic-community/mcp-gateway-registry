#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "Building and pushing all images to ECR..."
echo "  Account:  ${ACCOUNT_ID}"
echo "  Region:   ${REGION}"
echo "  Registry: ${ECR_REGISTRY}"
echo ""

# Login to ECR
echo "Logging into ECR..."
aws ecr get-login-password --region "${REGION}" | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# Image definitions: name|dockerfile|context (relative to repo root)
# Using | as delimiter since Dockerfile paths contain no pipes
IMAGES=(
  "mcp-gateway-registry|docker/Dockerfile.registry|."
  "mcp-gateway-auth-server|docker/Dockerfile.auth|."
  "mcp-gateway-currenttime|docker/Dockerfile.mcp-server|servers/currenttime"
  "mcp-gateway-mcpgw|docker/Dockerfile.mcp-server|servers/mcpgw"
  "mcp-gateway-realserverfaketools|docker/Dockerfile.mcp-server|servers/realserverfaketools"
  "mcp-gateway-flight-booking-agent|agents/a2a/src/flight-booking-agent/Dockerfile|agents/a2a/src/flight-booking-agent"
  "mcp-gateway-travel-assistant-agent|agents/a2a/src/travel-assistant-agent/Dockerfile|agents/a2a/src/travel-assistant-agent"
  "mcp-gateway-metrics-service|metrics-service/Dockerfile|metrics-service"
  "mcp-gateway-grafana|terraform/aws-ecs/grafana/Dockerfile|terraform/aws-ecs/grafana"
)

cd "${REPO_ROOT}"

FAILED=()

for IMAGE_INFO in "${IMAGES[@]}"; do
  IFS='|' read -r REPO_NAME DOCKERFILE CONTEXT <<< "${IMAGE_INFO}"

  echo ""
  echo "========================================="
  echo "Building: ${REPO_NAME}"
  echo "  Dockerfile: ${DOCKERFILE}"
  echo "  Context:    ${CONTEXT}"
  echo "========================================="

  # Create ECR repository if it doesn't exist
  aws ecr create-repository --repository-name "${REPO_NAME}" --region "${REGION}" 2>/dev/null || true

  # Build, tag, and push
  if docker build --platform linux/amd64 -f "${DOCKERFILE}" -t "${REPO_NAME}:latest" "${CONTEXT}"; then
    docker tag "${REPO_NAME}:latest" "${ECR_REGISTRY}/${REPO_NAME}:latest"
    docker push "${ECR_REGISTRY}/${REPO_NAME}:latest"
    echo "Done: ${REPO_NAME}"
  else
    echo "FAILED: ${REPO_NAME}"
    FAILED+=("${REPO_NAME}")
  fi
done

echo ""
echo "========================================="
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All images built and pushed to ECR!"
else
  echo "WARNING: ${#FAILED[@]} image(s) failed to build:"
  for name in "${FAILED[@]}"; do
    echo "  - ${name}"
  done
fi
echo "========================================="
echo ""
echo "Now run: cd terraform/aws-ecs && terraform apply"
