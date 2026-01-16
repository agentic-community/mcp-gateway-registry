#
# CodeBuild Project for Building Container Images from Upstream
# Set create_codebuild = true in terraform.tfvars to enable
#

variable "create_codebuild" {
  description = "Whether to create CodeBuild resources for building container images"
  type        = bool
  default     = false
}

# S3 bucket for CodeBuild artifacts and buildspecs
resource "aws_s3_bucket" "codebuild" {
  count  = var.create_codebuild ? 1 : 0
  bucket = "mcp-gateway-terraform-${data.aws_caller_identity.current.account_id}"

  tags = merge(
    local.common_tags,
    {
      Name = "mcp-gateway-codebuild"
    }
  )
}

resource "aws_s3_bucket_versioning" "codebuild" {
  count  = var.create_codebuild ? 1 : 0
  bucket = aws_s3_bucket.codebuild[0].id
  versioning_configuration {
    status = "Enabled"
  }
}

# Upload buildspec to S3
resource "aws_s3_object" "upstream_buildspec" {
  count  = var.create_codebuild ? 1 : 0
  bucket = aws_s3_bucket.codebuild[0].id
  key    = "buildspecs/upstream-buildspec.yaml"
  content = <<-EOF
version: 0.2

env:
  variables:
    DOCKER_BUILDKIT: "1"

phases:
  pre_build:
    commands:
      - echo "=== Building from upstream agentic-community/mcp-gateway-registry ==="
      - echo "Source version - $CODEBUILD_RESOLVED_SOURCE_VERSION"
      - export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
      - export ECR_REGISTRY="$${AWS_ACCOUNT_ID}.dkr.ecr.$${AWS_DEFAULT_REGION}.amazonaws.com"
      - echo "ECR Registry - $ECR_REGISTRY"
      - aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY
      - echo "Pulling base images from ECR public..."
      - docker pull public.ecr.aws/docker/library/python:3.12-slim || true
      - docker tag public.ecr.aws/docker/library/python:3.12-slim python:3.12-slim
      - docker pull quay.io/keycloak/keycloak:latest || true
      - echo "Pulling existing images for cache..."
      - docker pull $ECR_REGISTRY/mcp-gateway-registry:latest || true
      - docker pull $ECR_REGISTRY/mcp-gateway-auth-server:latest || true
      - docker pull $ECR_REGISTRY/keycloak:latest || true
      - docker pull $ECR_REGISTRY/mcp-gateway-currenttime:latest || true
      - docker pull $ECR_REGISTRY/mcp-gateway-mcpgw:latest || true
      - docker pull $ECR_REGISTRY/mcp-gateway-realserverfaketools:latest || true
      - docker pull $ECR_REGISTRY/mcp-gateway-flight-booking-agent:latest || true
      - docker pull $ECR_REGISTRY/mcp-gateway-travel-assistant-agent:latest || true
      - docker pull $ECR_REGISTRY/mcp-gateway-scopes-init:latest || true
      - mkdir -p agents/a2a/src/flight-booking-agent/.tmp agents/a2a/src/travel-assistant-agent/.tmp
      - cp agents/a2a/pyproject.toml agents/a2a/uv.lock agents/a2a/src/flight-booking-agent/.tmp/ 2>/dev/null || true
      - cp agents/a2a/pyproject.toml agents/a2a/uv.lock agents/a2a/src/travel-assistant-agent/.tmp/ 2>/dev/null || true

  build:
    commands:
      - echo "=== Building all container images with cache ==="
      - echo "Building registry (CPU-only)..." && docker build --cache-from $ECR_REGISTRY/mcp-gateway-registry:latest -t $ECR_REGISTRY/mcp-gateway-registry:latest -f docker/Dockerfile.registry-cpu . && docker push $ECR_REGISTRY/mcp-gateway-registry:latest
      - echo "Building auth_server..." && docker build --cache-from $ECR_REGISTRY/mcp-gateway-auth-server:latest -t $ECR_REGISTRY/mcp-gateway-auth-server:latest -f docker/Dockerfile.auth . && docker push $ECR_REGISTRY/mcp-gateway-auth-server:latest
      - echo "Building keycloak..." && docker build --cache-from $ECR_REGISTRY/keycloak:latest -t $ECR_REGISTRY/keycloak:latest -f docker/keycloak/Dockerfile docker/keycloak && docker push $ECR_REGISTRY/keycloak:latest
      - echo "Building currenttime..." && docker build --cache-from $ECR_REGISTRY/mcp-gateway-currenttime:latest -t $ECR_REGISTRY/mcp-gateway-currenttime:latest -f docker/Dockerfile.mcp-server servers/currenttime && docker push $ECR_REGISTRY/mcp-gateway-currenttime:latest
      - echo "Building mcpgw (CPU-only)..." && docker build --build-arg SERVER_DIR=servers/mcpgw --cache-from $ECR_REGISTRY/mcp-gateway-mcpgw:latest -t $ECR_REGISTRY/mcp-gateway-mcpgw:latest -f docker/Dockerfile.mcp-server-cpu . && docker push $ECR_REGISTRY/mcp-gateway-mcpgw:latest
      - echo "Building realserverfaketools..." && docker build --cache-from $ECR_REGISTRY/mcp-gateway-realserverfaketools:latest -t $ECR_REGISTRY/mcp-gateway-realserverfaketools:latest -f docker/Dockerfile.mcp-server servers/realserverfaketools && docker push $ECR_REGISTRY/mcp-gateway-realserverfaketools:latest
      - echo "Building flight_booking_agent..." && docker build --cache-from $ECR_REGISTRY/mcp-gateway-flight-booking-agent:latest -t $ECR_REGISTRY/mcp-gateway-flight-booking-agent:latest -f agents/a2a/src/flight-booking-agent/Dockerfile agents/a2a/src/flight-booking-agent && docker push $ECR_REGISTRY/mcp-gateway-flight-booking-agent:latest
      - echo "Building travel_assistant_agent..." && docker build --cache-from $ECR_REGISTRY/mcp-gateway-travel-assistant-agent:latest -t $ECR_REGISTRY/mcp-gateway-travel-assistant-agent:latest -f agents/a2a/src/travel-assistant-agent/Dockerfile agents/a2a/src/travel-assistant-agent && docker push $ECR_REGISTRY/mcp-gateway-travel-assistant-agent:latest
      - echo "Building scopes-init..." && docker build --cache-from $ECR_REGISTRY/mcp-gateway-scopes-init:latest -t $ECR_REGISTRY/mcp-gateway-scopes-init:latest -f docker/Dockerfile.scopes-init . && docker push $ECR_REGISTRY/mcp-gateway-scopes-init:latest

  post_build:
    commands:
      - echo "Build completed on $(date)"
      - echo "All images pushed to $ECR_REGISTRY"
EOF

  tags = local.common_tags
}

# IAM Role for CodeBuild
resource "aws_iam_role" "codebuild" {
  count = var.create_codebuild ? 1 : 0
  name  = "mcp-gateway-tf-codebuild-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "codebuild.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "codebuild" {
  count = var.create_codebuild ? 1 : 0
  name  = "mcp-gateway-tf-codebuild-policy"
  role  = aws_iam_role.codebuild[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload"
        ]
        Resource = "arn:aws:ecr:${var.aws_region}:${data.aws_caller_identity.current.account_id}:repository/*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion"
        ]
        Resource = "${aws_s3_bucket.codebuild[0].arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "sts:GetCallerIdentity"
        ]
        Resource = "*"
      }
    ]
  })
}

# CodeBuild Project - Upstream Source
resource "aws_codebuild_project" "upstream" {
  count         = var.create_codebuild ? 1 : 0
  name          = "mcp-gateway-upstream-build-tf"
  description   = "Build containers from upstream agentic-community/mcp-gateway-registry"
  build_timeout = 60
  service_role  = aws_iam_role.codebuild[0].arn

  artifacts {
    type = "NO_ARTIFACTS"
  }

  environment {
    compute_type                = "BUILD_GENERAL1_LARGE"
    image                       = "aws/codebuild/amazonlinux2-x86_64-standard:5.0"
    type                        = "LINUX_CONTAINER"
    privileged_mode             = true
    image_pull_credentials_type = "CODEBUILD"
  }

  source {
    type            = "GITHUB"
    location        = "https://github.com/WPrintz/mcp-gateway-registry.git"
    buildspec       = aws_s3_object.upstream_buildspec[0].content
    git_clone_depth = 1

    git_submodules_config {
      fetch_submodules = false
    }
  }

  source_version = "feature/issue-293-cloudfront-v1.0.9-patch1"

  # Enable Docker layer caching for faster builds
  cache {
    type  = "LOCAL"
    modes = ["LOCAL_DOCKER_LAYER_CACHE", "LOCAL_SOURCE_CACHE"]
  }

  tags = local.common_tags
}

# Output the project name for easy reference
output "codebuild_project_upstream" {
  description = "CodeBuild project for building from upstream"
  value       = var.create_codebuild ? aws_codebuild_project.upstream[0].name : null
}

output "codebuild_s3_bucket" {
  description = "S3 bucket for CodeBuild artifacts"
  value       = var.create_codebuild ? aws_s3_bucket.codebuild[0].id : null
}
