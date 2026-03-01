#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "Building minimal images for testing..."
echo "  Account:  ${ACCOUNT_ID}"
echo "  Region:   ${REGION}"
echo "  Registry: ${ECR_REGISTRY}"
echo ""

# Login to ECR
echo "Logging into ECR..."
aws ecr get-login-password --region "${REGION}" | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# Only build essential images
IMAGES=(
  "mcp-gateway-registry|docker/Dockerfile.registry|."
  "mcp-gateway-currenttime|docker/Dockerfile.mcp-server|servers/currenttime"
)

cd "${REPO_ROOT}"

for IMAGE_INFO in "${IMAGES[@]}"; do
  IFS='|' read -r REPO_NAME DOCKERFILE CONTEXT <<< "${IMAGE_INFO}"

  echo ""
  echo "Building: ${REPO_NAME}"

  aws ecr create-repository --repository-name "${REPO_NAME}" --region "${REGION}" 2>/dev/null || true

  docker build --platform linux/amd64 -f "${DOCKERFILE}" -t "${REPO_NAME}:latest" "${CONTEXT}"
  docker tag "${REPO_NAME}:latest" "${ECR_REGISTRY}/${REPO_NAME}:latest"
  docker push "${ECR_REGISTRY}/${REPO_NAME}:latest"

  echo "Done: ${REPO_NAME}"
done

echo ""
echo "Essential images ready!"
echo ""
echo "Now run: cd terraform/aws-ecs && terraform apply"
